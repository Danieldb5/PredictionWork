"""
sim.py - event-driven paper exchange + strategy, shared by live and backtest.

Everything is driven by two event types, in time order:

  on_book(token, book, ts)   a new order-book state for the token we quote
  on_trade(row)              a real trade print (any outcome of the market),
                             converted to the quoted token's terms

so a live websocket session and a replay of recorded data run through the
exact same code and produce the same output files.

Paper exchange (per market), YES-token terms
--------------------------------------------
Our resting bid / ask are virtual orders. They become live `latency` seconds
after we place them. A virtual order fills when:

  trade-through   a taker trade prints at a price through ours (taker sold
                  below our bid / bought above our ask): we would have been
                  hit first, fill = min(trade size, our size)
  at-level        a taker trade prints exactly at our price: the queue that
                  was ahead of us when we joined trades first
  cross           the book shows resting orders on the other side at or
                  through our price (someone posted inside the spread where
                  our virtual order sits). Their order would have matched
                  ours; each unit of crossing liquidity fills us only once.

Queue ahead of us shrinks with trades at our level and, with
--queue prorata (default), with cancellations at our level in proportion
(cancels are spread evenly over the queue ahead of and behind us).
'conservative' ignores cancellations, 'optimistic' assumes they were all
ahead of us.

Strategy
--------
On each book update (throttled by --requote seconds) the FairMV engine
values the token from the book + recent trades, and MarketMaker.quote()
turns fair value, inventory and the market's best bid/ask into our bid/ask
price and size. --observe records fair values only (no orders), which is
what the accuracy report needs.
"""
from __future__ import annotations

import json
import math
import os
import sys
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import FetchData as FD                                    # noqa: E402
import Scalper as SC                                      # noqa: E402
from FairMV import DEFAULT_USER_DIR, FairValueEngine, OrderBook, PolymarketProvider  # noqa: E402
from MarketMaker import MMParams, quote                   # noqa: E402

log = FD.log
EPS = 1e-9


# =============================================================================
# Trades
# =============================================================================
def token_terms(token: str, trade_token: str, side: str, price: float) -> tuple[int, float]:
    """Taker side (+1 bought) and price in the terms of `token` (binary complement)."""
    sgn = 1 if str(side).upper() == "BUY" else -1
    return (sgn, price) if str(trade_token) == str(token) else (-sgn, 1.0 - price)


def trade_row(m: "Market", trade_token: str, side: str, price: float, size: float,
              ts: pd.Timestamp, tx: str, source: str) -> dict:
    s, p = token_terms(m.token, trade_token, side, float(price))
    return dict(ts=ts, token_id=m.token, slug=m.slug, condition_id=m.condition,
                trade_token=str(trade_token), raw_side=str(side).upper(), raw_price=float(price),
                side=s, price=p, size=float(size), transaction_hash=str(tx or ""), source=source)


def trade_key(r: dict) -> tuple:
    return (r["transaction_hash"], r["trade_token"], round(r["raw_price"], 6), round(r["size"], 6),
            r["raw_side"], int(pd.Timestamp(r["ts"]).timestamp()))


def api_trade_rows(m: "Market", df: pd.DataFrame, source: str = "api") -> list[dict]:
    """Scalper.fetch_trades frame -> trade rows in m.token terms."""
    out = []
    for t in df.itertuples():
        out.append(trade_row(m, t.token_id, t.side, t.price, t.size, t.ts, t.transaction_hash, source))
    return out


def calib_frame(rows) -> pd.DataFrame:
    """Rows -> the column names Scalper.calibrate expects (yes_* = quoted-token terms)."""
    df = pd.DataFrame(list(rows))
    if df.empty:
        return pd.DataFrame(columns=["ts", "yes_side", "yes_price", "size"])
    return df.rename(columns={"side": "yes_side", "price": "yes_price"}).sort_values("ts")


# =============================================================================
# Paper exchange
# =============================================================================
def level_size(levels, price) -> float:
    return float(sum(s for p, s in levels if abs(p - price) < EPS))


def crossing_size(book: OrderBook, side: str, price: float) -> float:
    """Liquidity on the other side at or through our price."""
    if side == "bid":
        return float(sum(s for p, s in book.asks if p <= price + EPS))
    return float(sum(s for p, s in book.bids if p >= price - EPS))


@dataclass
class Order:
    side: str                 # 'bid' | 'ask'
    price: float
    size: float               # remaining
    queue: float              # shares ahead of us at our price
    placed: pd.Timestamp
    live: pd.Timestamp        # placed + latency
    level_seen: float         # level size at our price last time we looked (excl. us)
    cross_used: float         # crossing liquidity already counted as filling us


class PaperExchange:
    def __init__(self, a, m: "Market"):
        self.a, self.m = a, m
        self.latency = pd.Timedelta(seconds=a.latency)
        self.bid: Order | None = None
        self.ask: Order | None = None
        self.q = self.cash = self.fees = 0.0
        self.n_fills = 0
        self.volume = 0.0

    # ---- orders -------------------------------------------------------------
    def set_quote(self, ts, side: str, price, size: float, book: OrderBook) -> None:
        cur: Order | None = getattr(self, side)
        if price is None or size < self.a.min_size - EPS:
            setattr(self, side, None)
            return
        if cur is not None and abs(cur.price - price) < EPS:
            cur.size = size                        # same price: keep queue priority
            return
        lv = level_size(book.bids if side == "bid" else book.asks, price)
        setattr(self, side, Order(side, price, size, lv, ts, ts + self.latency, lv,
                                  crossing_size(book, side, price)))

    def cancel_all(self) -> None:
        self.bid = self.ask = None

    # ---- fills ----------------------------------------------------------------
    def _fill(self, o: Order, take: float, ts, kind: str, fills: list, fv: float, trade=None) -> None:
        sgn = 1 if o.side == "bid" else -1         # +1 we bought
        fee = take * o.price * self.a.fee_bps / 1e4
        self.q += sgn * take
        self.cash -= sgn * take * o.price + fee
        self.fees += fee
        self.n_fills += 1
        self.volume += take
        o.size -= take
        fills.append(dict(ts=ts, token_id=self.m.token, slug=self.m.slug, side="buy" if sgn > 0 else "sell",
                          price=o.price, size=take, kind=kind, fv=fv, inventory=self.q, cash=self.cash,
                          queue_left=o.queue, placed=o.placed,
                          trade_price=trade["price"] if trade else np.nan,
                          trade_size=trade["size"] if trade else np.nan,
                          mkt_bid=self.m.book.best_bid if self.m.book else np.nan,
                          mkt_ask=self.m.book.best_ask if self.m.book else np.nan))
        if o.size < EPS:
            setattr(self, o.side, None)

    def on_trade(self, t: dict, fills: list, fv: float) -> None:
        """t in token terms: side +1 = taker bought (hits our ask), -1 = taker sold (hits our bid)."""
        o = self.ask if t["side"] > 0 else self.bid
        if o is None or t["ts"] < o.live:
            return
        px, sz = t["price"], t["size"]
        through = px > o.price + EPS if o.side == "ask" else px < o.price - EPS
        at = abs(px - o.price) < EPS
        if through:
            self._fill(o, min(sz, o.size), t["ts"], "through", fills, fv, t)
        elif at:
            take = min(max(sz - o.queue, 0.0), o.size)
            o.queue = max(o.queue - sz, 0.0)
            o.level_seen = max(o.level_seen - sz, 0.0)   # the trade, not a cancel, shrank the level
            if take > 0:
                self._fill(o, take, t["ts"], "at_level", fills, fv, t)

    def on_book(self, book: OrderBook, ts, fills: list, fv: float) -> None:
        for o in (self.bid, self.ask):
            if o is None:
                continue
            lv = level_size(book.bids if o.side == "bid" else book.asks, o.price)
            if lv < o.level_seen - EPS:               # cancellations at our level
                mode = self.a.queue
                if mode == "prorata" and o.level_seen > 0:
                    o.queue *= lv / o.level_seen
                elif mode == "optimistic":
                    o.queue = min(o.queue, lv)
            o.queue = min(o.queue, lv) if self.a.queue != "conservative" else o.queue
            o.level_seen = lv
            cross = crossing_size(book, o.side, o.price)
            if cross < o.cross_used:                  # crossing orders left: forget them
                o.cross_used = cross
            if ts >= o.live and cross > o.cross_used + EPS:
                take = min(cross - o.cross_used, o.size)
                o.cross_used += take
                self._fill(o, take, ts, "cross", fills, fv)


# =============================================================================
# Market + engine
# =============================================================================
@dataclass
class Market:
    token: str                     # the token we quote (usually YES = outcome 0)
    condition: str
    slug: str
    other_token: str | None = None
    end_date: str | None = None
    meta: dict = field(default_factory=dict)
    tick: float = 0.01
    P: MMParams | None = None
    calib: object = None
    book: OrderBook | None = None
    recent: deque = field(default_factory=lambda: deque(maxlen=300))
    ex: PaperExchange | None = None
    fv: float = float("nan")
    fv_sd: float = float("nan")
    last_eval: pd.Timestamp | None = None
    last_trade_ts: pd.Timestamp | None = None

    @property
    def pnl(self) -> float:
        mark = self.fv if not math.isnan(self.fv) else (self.book.mid if self.book and self.book.mid else 0.0)
        return self.ex.cash + self.ex.q * mark

    def trades_frame(self) -> pd.DataFrame:
        if not self.recent:
            return pd.DataFrame(columns=["ts", "price", "side", "size"])
        return pd.DataFrame(list(self.recent))[["ts", "price", "side", "size"]]


def make_engine(a, live: bool) -> FairValueEngine:
    prov = PolymarketProvider(http_get=lambda url, **kw: FD.http("GET", url, **kw),
                              http_post=lambda url, **kw: FD.http("POST", url, **kw)) if live else None
    eng = FairValueEngine.from_user_dir(a.fv_user_dir or DEFAULT_USER_DIR, provider=prov)
    # links/feeds pull *current* data from the API: in a backtest that is look-ahead
    eng.cfg.use_links = live and not a.no_links
    eng.cfg.use_feeds = live and not a.no_feeds
    return eng


def sigma2_at(m: Market, p: float, now: pd.Timestamp) -> float:
    """Scalper.sigma2_for, with an explicit clock (backtests run in the past)."""
    end = pd.to_datetime(m.end_date, utc=True, errors="coerce")
    T = max((end - now).total_seconds() / 3600, 24.0) if not pd.isna(end) else 24 * 30
    return max(m.calib.sigma2, m.P.sigma2_jumps(), p * (1 - p) / T, 1e-6)


def calibrate_market(a, m: Market, history: list[dict]) -> bool:
    """Flow / informed-trading parameters from trades *before* the run starts."""
    if m.book is None or m.book.spread is None or m.book.spread <= 0:
        return False
    c = SC.calibrate(calib_frame(history), m.book.spread / 2, m.tick, a.window, a.mu_floor)
    if c is None or c.n_trades < a.min_trades:
        log(f"  {m.slug[:55]}: only {0 if c is None else c.n_trades} trades in window -> observe only")
        return False
    m.calib, m.P = c, SC.mm_params(c, m.book.spread / 2, m.tick, a)
    log(f"  {m.slug[:55]:<55} trades/h {c.R_buy + c.R_sell:5.2f}  mu {c.mu:.2f}  J {c.J:.3f}  m {c.m:.0f}")
    return True


# =============================================================================
# Runner: consumes events, keeps state, writes the run directory
# =============================================================================
class Runner:
    def __init__(self, a, markets: list[Market], out: str, mode: str, live: bool):
        self.a, self.out, self.mode, self.live = a, out, mode, live
        self.mk = {m.token: m for m in markets}
        self.by_any = {}                           # any outcome token -> market
        for m in markets:
            self.by_any[m.token] = m
            if m.other_token:
                self.by_any[m.other_token] = m
        self.by_cond = {m.condition: m for m in markets}
        for m in markets:
            m.ex = PaperExchange(a, m)
        self.engine = make_engine(a, live)
        self.seen: set = {trade_key(r) for m in markets for r in m.recent}   # warm-up history
        self.fills, self.quotes, self.preds, self.trades = [], [], [], []
        self.snaps, self.levels, self.last_hash = [], [], {}
        self.sid = 0
        self.clock: pd.Timestamp | None = None
        self.n_events = 0
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "config.json"), "w") as f:
            json.dump({"mode": mode, **{k: v for k, v in vars(a).items() if k != "func"},
                       "markets": [{"slug": m.slug, "token": m.token, "condition": m.condition,
                                    "end_date": m.end_date, "trading": m.P is not None} for m in markets]},
                      f, indent=1, default=str)

    # ---- events -----------------------------------------------------------------
    def on_trade(self, r: dict) -> None:
        k = trade_key(r)
        if k in self.seen:
            return
        self.seen.add(k)
        m = self.mk.get(r["token_id"])
        if m is None:
            return
        late = self.clock is not None and r["ts"] < self.clock - pd.Timedelta(seconds=self.a.late_s)
        r = {**r, "late": late}
        self.trades.append(r)
        m.recent.append(r)
        m.last_trade_ts = r["ts"]
        if not late:                               # a backfilled trade can't fill quotes that moved since
            self.clock = max(self.clock, r["ts"]) if self.clock is not None else r["ts"]
            m.ex.on_trade(r, self.fills, m.fv)
        self.n_events += 1

    def on_book(self, token: str, book: OrderBook, ts: pd.Timestamp, raw: dict | None = None) -> None:
        m = self.mk.get(token)
        if m is None:
            return
        self.clock = max(self.clock, ts) if self.clock is not None else ts
        m.book = book
        if book.tick_size:
            m.tick = book.tick_size
        m.ex.on_book(book, ts, self.fills, m.fv)
        self.n_events += 1
        if m.last_eval is None or (ts - m.last_eval).total_seconds() >= self.a.requote:
            self.requote(m, ts)

    # ---- strategy -----------------------------------------------------------------
    def requote(self, m: Market, ts: pd.Timestamp) -> None:
        m.last_eval = ts
        b = m.book
        if b is None or b.best_bid is None or b.best_ask is None:
            m.ex.cancel_all()
            return
        try:
            res = self.engine.evaluate(b, trades=m.trades_frame(), meta=m.meta, now=ts,
                                       smooth=True, key=m.token)
        except Exception as e:  # noqa: BLE001
            log(f"! fv {m.slug[:40]}: {type(e).__name__}: {e}")
            m.ex.cancel_all()
            return
        fv = res.fair_value if res.fair_value is not None else 0.5 * (b.best_bid + b.best_ask)
        m.fv, m.fv_sd = fv, (res.sd if res.sd is not None else float("nan"))
        last = m.recent[-1]["price"] if m.recent else np.nan
        bs1, as1 = b.bids[0][1], b.asks[0][1]
        pred = dict(ts=ts, token_id=m.token, slug=m.slug, fv=fv, fv_raw=res.fair_value_raw, fv_sd=m.fv_sd,
                    confidence=res.confidence, bid=b.best_bid, ask=b.best_ask,
                    mid=0.5 * (b.best_bid + b.best_ask),
                    microprice=(b.best_bid * as1 + b.best_ask * bs1) / (bs1 + as1),
                    last_trade=last, n_recent=len(m.recent))
        for k, v in res.components.items():
            if isinstance(v, (int, float)) and v is not None and not math.isnan(v):
                pred[f"c_{k}"] = float(v)
        self.preds.append(pred)

        if self.a.observe or m.P is None:
            return
        s2 = SC.sigma2_with_fv(sigma2_at(m, fv, ts), res.sd, m.P.tau_h)
        Q = quote(fv, m.ex.q, m.P, s2, b.best_bid, b.best_ask, join_or_improve=not self.a.allow_behind)
        m.ex.set_quote(ts, "bid", Q.bid.price if Q.bid else None, Q.bid_size, b)
        m.ex.set_quote(ts, "ask", Q.ask.price if Q.ask else None, Q.ask_size, b)
        self.quotes.append(dict(
            ts=ts, token_id=m.token, slug=m.slug, fv=fv, fv_sd=m.fv_sd, inventory=m.ex.q,
            mkt_bid=b.best_bid, mkt_ask=b.best_ask,
            our_bid=Q.bid.price if Q.bid else np.nan, bid_size=Q.bid_size,
            our_ask=Q.ask.price if Q.ask else np.nan, ask_size=Q.ask_size,
            bid_queue=m.ex.bid.queue if m.ex.bid else np.nan, ask_queue=m.ex.ask.queue if m.ex.ask else np.nan,
            exp_fills_h=(Q.bid.fills_per_h if Q.bid else 0) + (Q.ask.fills_per_h if Q.ask else 0),
            exp_pnl_h=(Q.bid.pnl_per_h if Q.bid else 0) + (Q.ask.pnl_per_h if Q.ask else 0),
            cash=m.ex.cash, pnl=m.pnl))

    # ---- recording (live: so the session can be replayed as a backtest) ----------
    def record_books(self, ts: pd.Timestamp) -> None:
        for m in self.mk.values():
            b = m.book
            if b is None:
                continue
            raw = {"asset_id": m.token, "market": m.condition, "tick_size": b.tick_size,
                   "min_order_size": b.min_order_size, "neg_risk": m.meta.get("neg_risk"),
                   "last_trade_price": m.recent[-1]["price"] if m.recent else None,
                   "timestamp": int(ts.timestamp() * 1000),
                   "hash": str(hash((tuple(b.bids), tuple(b.asks)))),
                   "bids": [{"price": p, "size": s} for p, s in b.bids],
                   "asks": [{"price": p, "size": s} for p, s in b.asks]}
            r, lv = FD.process_book(raw, self.sid, ts, 0.0, {"slug": m.slug, "outcome": "token"}, 10,
                                    self.last_hash)
            self.snaps.append(r)
            self.levels.extend(lv)
        self.sid += 1

    # ---- output ---------------------------------------------------------------------
    def flush(self) -> None:
        if self.snaps or self.levels:
            FD.flush(self.out, self.snaps, self.levels)
        for name, rows in (("fills", self.fills), ("quotes", self.quotes),
                           ("predictions", self.preds), ("trades", self.trades)):
            if not rows:
                continue
            df = pd.DataFrame(rows)
            path = os.path.join(self.out, f"{name}.parquet")
            if os.path.exists(path):
                df = pd.concat([pd.read_parquet(path), df], ignore_index=True)
            for c in df.columns:
                if df[c].dtype == object and c not in ("ts", "placed"):
                    df[c] = df[c].astype("string")
            df.to_parquet(path, index=False)
            rows.clear()

    def status(self) -> str:
        tot = sum(m.pnl for m in self.mk.values())
        nf = sum(m.ex.n_fills for m in self.mk.values())
        inv = ", ".join(f"{m.slug[:16]}:{m.ex.q:+.0f}" for m in self.mk.values() if abs(m.ex.q) > EPS)
        return f"PnL ${tot:+.2f}  fills {nf}  events {self.n_events:,}" + (f"  inv {inv}" if inv else "")

    def finish(self) -> dict:
        self.flush()
        rows = []
        for m in self.mk.values():
            rows.append(dict(slug=m.slug, token_id=m.token, trading=m.P is not None, inventory=m.ex.q,
                             cash=m.ex.cash, fees=m.ex.fees, fv=m.fv, pnl=m.pnl, fills=m.ex.n_fills,
                             volume=m.ex.volume))
        summ = {"mode": self.mode, "end": str(self.clock), "total_pnl": float(sum(r["pnl"] for r in rows)),
                "markets": rows}
        with open(os.path.join(self.out, "summary.json"), "w") as f:
            json.dump(summ, f, indent=1, default=str)
        return summ
