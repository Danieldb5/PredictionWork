"""
Scalper.py - find low-volume, wide-spread Polymarket markets, calibrate the
market maker to each one from its real trade history, download data, and
paper-trade (scalp) the spread.

PAPER TRADING ONLY: no orders are sent. Fills are simulated from real trades
printed on Polymarket while our virtual quotes were resting.

-------------------------------------------------------------------------------
PIPELINE
-------------------------------------------------------------------------------
  1. scan      Gamma API -> live markets with 24h volume in [min, max], spread
               >= min_spread, price in [0.05, 0.95], resolution >= N hours away.
               Verified against the live CLOB book.
  2. calibrate Trade history (Data API v2) -> MarketMaker parameters per market
               (formulas below). Then rank by the MarketMaker's expected $/h.
  3. download  Full trade history + order book snapshots to Parquet (data/).
  4. scalp     Every interval: fetch books + new trades, simulate fills on our
               resting quotes, update inventory, re-quote with MarketMaker,
               log everything.

-------------------------------------------------------------------------------
CALIBRATION  (all in YES terms: buying NO at x == selling YES at 1-x)
-------------------------------------------------------------------------------
Over the last T hours of trades, with current half-spread s:
  R_buy, R_sell   taker buys / sells per hour (observed at distance ~ s)
  m               mean trade size (shares)
  impact I        permanent price impact of a trade: compare each trade with
                  the next same-side trade at least 1h later, then
                  I = mean( side * (p_later - p_now) ). Comparing same-side
                  prices removes bid-ask bounce.
  J               mean absolute size of those 1h same-side moves (>= 1 tick)
  mu              informed share: in the model E[v | fill] - p = mu (s + J),
                  so mu = clip(I / (s + J), 0, 0.9)
Then map observed rates at distance s to model parameters:
  A_buy  = (1 - mu) R_buy  e^{k s}        (noise flow if we quoted at fair value)
  lam_I  = mu (R_buy + R_sell) e^{s / J}   (informed traders/h, both sides)
k (how much more flow you get by quoting tighter) CANNOT be identified from
trades alone. It is the key assumption (--k, default 30: flow x2 per 2.3c
tighter). Exponential extrapolation over a 30c half-spread is not credible, so
the total gain is capped: k_eff = min(k, ln(flow_cap)/s), default flow_cap 3x.
Check both against your paper fills.
Guards: mu >= --mu-floor (0.10: quiet markets still have informed traders);
markets with < --min-trades in the window are not ranked; sigma^2 = max of
measured 1h variance, 2 lam_I J^2 and p(1-p)/T.

Fair value: FairMV.FairValueEngine (see FairMV/README.md). It combines, by
inverse variance in log-odds:
  book     (dust-filtered, variance = uniform over the spread: weak when wide)
  trades   (side-corrected, staleness- and size-weighted)
  links    (correlated markets, neg-risk event sums: automatic)
  feeds    (your views / scraped tables / plugins in FairMV/user/)
then applies a Kalman tracker per market and clips to [bid, ask].
FV uncertainty sd^2 is added to the MarketMaker's inventory variance
(sigma^2 += sd^2 / tau), so it sizes down when it does not know the value.
Quotes are kept at or inside the market's best bid/ask (join or improve),
because the flow model assumes we are the best price.

-------------------------------------------------------------------------------
FILL SIMULATION (conservative)
-------------------------------------------------------------------------------
A taker BUY at price >= our ask fills us if:
  * our ask was strictly better than the book's best ask (we are first), or
  * our ask equalled the best ask, after the queue ahead of us at that level
    has been consumed (queue_ahead = size there when we quoted).
Same for sells against our bid. Fill size = min(trade size, our remaining size).
PnL is marked to the current FairMV fair value.

-------------------------------------------------------------------------------
USAGE
-------------------------------------------------------------------------------
  py Scalper.py scan                         rank candidates (writes data/scan_*.parquet)
       [--min-vol 50 --max-vol 5000 --min-spread 0.04 --min-hours 48 --max-pages 150 --top 20]
  py Scalper.py download --n 10              trade history + 1 book snapshot of top-N from last scan
  py Scalper.py scalp --n 5 [--interval 15 --duration 3600 --bankroll 1000]
       paper-trade the top-N from the last scan (or --slug s1 s2 ...)
  py Scalper.py report                       PnL / fills summary of paper trading
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import signal
import sys
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import FetchData as FD                                   # noqa: E402
from FairMV import (DEFAULT_USER_DIR, FairValueEngine, OrderBook,  # noqa: E402
                    PolymarketProvider)
from MarketMaker import MMParams, quote                  # noqa: E402

GAMMA = "https://gamma-api.polymarket.com"
TRADES = "https://data-api.polymarket.com/v2/trades"
log = FD.log
ENGINE: FairValueEngine | None = None


def get_engine(a) -> FairValueEngine:
    """One engine per run: holds the Kalman trackers, link bases and caches."""
    global ENGINE
    if ENGINE is None:
        prov = PolymarketProvider(http_get=lambda url, **kw: FD.http("GET", url, **kw),
                                  http_post=lambda url, **kw: FD.http("POST", url, **kw))
        ENGINE = FairValueEngine.from_user_dir(getattr(a, "fv_user_dir", DEFAULT_USER_DIR), provider=prov)
        if getattr(a, "no_fv_links", False):
            ENGINE.cfg.use_links = False
        if getattr(a, "no_fv_feeds", False):
            ENGINE.cfg.use_feeds = False
    return ENGINE


def clean_row(r) -> dict:
    """pandas row -> dict with NaN as None (bool(NaN) is True, which would fake neg_risk)."""
    d = r.to_dict() if hasattr(r, "to_dict") else dict(r)
    return {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in d.items()}


def fv_meta(m: dict) -> dict:
    """Engine meta from a Gamma market dict or a scan row (YES token)."""
    ev = (m.get("events") or [{}])[0] if isinstance(m.get("events"), list) else {}
    return {"slug": m.get("slug"), "question": m.get("question"),
            "condition_id": m.get("condition_id") or m.get("conditionId"),
            "token_id": m.get("token_id"), "outcome_index": 0,
            "event_slug": m.get("event_slug") or ev.get("slug") or "",
            "neg_risk": bool(m.get("neg_risk", m.get("negRisk", ev.get("negRisk", False)))),
            "end_date": m.get("end_date") or m.get("endDate")}


# =============================================================================
# 1. Scan
# =============================================================================
def scan_gamma(a) -> list[dict]:
    """Walk live markets by 24h volume (desc) and keep low-volume, wide-spread ones."""
    out, cursor, pages = [], None, 0
    min_end = pd.Timestamp.now(tz="UTC") + pd.Timedelta(hours=a.min_hours)
    while pages < a.max_pages:
        p = {"active": "true", "closed": "false", "limit": 100, "order": "volume24hr", "ascending": "false"}
        if cursor:
            p["after_cursor"] = cursor
        d = FD.http("GET", f"{GAMMA}/markets/keyset", params=p)
        ms, pages = d.get("markets", []), pages + 1
        for m in ms:
            v = m.get("volume24hr") or 0
            bb, ba = m.get("bestBid"), m.get("bestAsk")
            if not (a.min_vol <= v <= a.max_vol and m.get("acceptingOrders") and m.get("enableOrderBook")):
                continue
            if bb is None or ba is None or ba - bb < a.min_spread or bb < 0.05 or ba > 0.95:
                continue
            end = pd.to_datetime(m.get("endDate"), utc=True, errors="coerce")
            if pd.isna(end) or end < min_end:
                continue
            out.append(m)
        if pages % 25 == 0:
            log(f"  scanned {pages} pages, {len(out)} candidates, now at 24h vol ${ms[-1].get('volume24hr', 0):,.0f}")
        cursor = d.get("next_cursor")
        if not ms or not cursor or (ms[-1].get("volume24hr") or 0) < a.min_vol:
            break
    return out


def fetch_trades(condition_id: str, max_trades=2000, since_ts: int | None = None) -> pd.DataFrame:
    """Trade history for a market (newest first, cursor-paginated), converted to YES terms."""
    rows, cursor = [], None
    while len(rows) < max_trades:
        p = {"condition": condition_id, "limit": 500}
        if cursor:
            p["cursor"] = cursor
        d = FD.http("GET", TRADES, params=p)
        data = d.get("data", [])
        rows.extend(data)
        cursor = (d.get("pagination") or {}).get("next_cursor")
        if not data or not cursor or not (d.get("pagination") or {}).get("has_more"):
            break
        if since_ts and data[-1]["timestamp"] < since_ts:
            break
    if not rows:
        return pd.DataFrame(columns=["ts", "side", "price", "size", "yes_side", "yes_price"])
    df = pd.DataFrame(rows)
    keep = ["timestamp", "side", "price", "size", "outcome", "outcome_index", "token_id",
            "condition_id", "transaction_hash", "proxy_wallet", "slug"]
    df = df[[c for c in keep if c in df.columns]].copy()
    df["ts"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    is_yes = df["outcome_index"] == 0
    sgn = np.where(df["side"].str.upper() == "BUY", 1, -1)
    df["yes_side"] = np.where(is_yes, sgn, -sgn)               # +1 taker bought YES exposure
    df["yes_price"] = np.where(is_yes, df["price"], 1 - df["price"])
    if since_ts:
        df = df[df["timestamp"] >= since_ts]
    return df.sort_values("ts").reset_index(drop=True)


# =============================================================================
# 2. Calibrate
# =============================================================================
@dataclass
class Calib:
    hours: float
    n_trades: int
    R_buy: float
    R_sell: float
    m: float
    impact: float
    J: float
    mu: float
    sigma2: float      # price variance per hour from 1h same-side changes
    vwap: float        # recent volume-weighted YES price (last 20 trades)


def trade_vwap(tr: pd.DataFrame, n: int = 20) -> float:
    t = tr.tail(n)
    return float((t["yes_price"] * t["size"]).sum() / t["size"].sum()) if len(t) and t["size"].sum() > 0 else float("nan")


def fair_value(a, ob: OrderBook, trades: pd.DataFrame | None, meta: dict, smooth: bool):
    """FairMV engine fair value (YES terms). Falls back to the logit mid if nothing usable."""
    res = get_engine(a).evaluate(ob, trades=trades, meta=meta, smooth=smooth, key=meta.get("token_id"))
    if res.fair_value is None and ob.best_bid is not None and ob.best_ask is not None:
        res.fair_value = 0.5 * (ob.best_bid + ob.best_ask)
    return res


def sigma2_with_fv(s2: float, fv_sd: float | None, tau_h: float) -> float:
    """Inventory variance per hour + fair-value uncertainty spread over the holding time."""
    return s2 + ((fv_sd or 0.0) ** 2) / max(tau_h, 1e-6)


def calibrate(tr: pd.DataFrame, half_spread: float, tick: float, window_h: float,
              mu_floor: float = 0.0) -> Calib | None:
    if len(tr) < 5:
        return None
    end = tr["ts"].max()
    vwap = trade_vwap(tr)
    tr = tr[tr["ts"] >= end - pd.Timedelta(hours=window_h)]
    if len(tr) < 5:
        return None
    hours = max((end - tr["ts"].min()).total_seconds() / 3600, 1.0)
    buys, sells = (tr["yes_side"] > 0).sum(), (tr["yes_side"] < 0).sum()
    m = float(tr["size"].mean())

    moves, imps = [], []
    for side in (1, -1):
        s = tr[tr["yes_side"] == side][["ts", "yes_price"]].reset_index(drop=True)
        if len(s) < 3:
            continue
        later = pd.merge_asof(s.assign(t2=s["ts"] + pd.Timedelta(hours=1)).sort_values("t2"),
                              s.rename(columns={"ts": "t_later", "yes_price": "p_later"}),
                              left_on="t2", right_on="t_later", direction="forward").dropna()
        dp = later["p_later"] - later["yes_price"]
        imps.extend((side * dp).tolist())
        moves.extend(dp.abs().tolist())
    impact = float(np.mean(imps)) if imps else 0.0
    J = max(float(np.mean(moves)) if moves else tick, tick)
    mu = float(np.clip(impact / (half_spread + J), mu_floor, 0.9))
    sigma2 = float(np.mean(np.square(moves))) if moves else J * J   # per 1h
    return Calib(hours, len(tr), buys / hours, sells / hours, m, impact, J, mu, sigma2, vwap)


def mm_params(c: Calib, half_spread: float, tick: float, a) -> MMParams:
    """Map observed flow at the market's quotes to MarketMaker parameters.
    Flow gained by quoting tighter is A e^{-k d}; we cap the total gain from
    quoting at fair value instead of at the market's quotes at `flow_cap` x,
    i.e. k_eff = min(k, ln(flow_cap) / s). Exponential extrapolation over
    tens of cents is not credible."""
    k_eff = min(a.k, math.log(a.flow_cap) / half_spread) if half_spread > 0 else a.k
    gain = math.exp(k_eff * half_spread)
    return MMParams(
        A_buy=max((1 - c.mu) * c.R_buy * gain, 1e-6),
        A_sell=max((1 - c.mu) * c.R_sell * gain, 1e-6),
        k=k_eff, noise_size=max(c.m, 1.0),
        lam_informed=c.mu * (c.R_buy + c.R_sell) * math.exp(half_spread / c.J),
        J=c.J, bankroll=a.bankroll, kelly_frac=a.kelly, tau_h=a.tau,
        max_position=a.max_position, tick=tick)


def sigma2_for(c: Calib, P: MMParams, p: float, end_date) -> float:
    """max of: measured 1h variance, jump model 2 lam_I J^2, binary floor p(1-p)/T."""
    end = pd.to_datetime(end_date, utc=True, errors="coerce")
    T = max((end - pd.Timestamp.now(tz="UTC")).total_seconds() / 3600, 24.0) if not pd.isna(end) else 24 * 30
    return max(c.sigma2, P.sigma2_jumps(), p * (1 - p) / T, 1e-6)


def evaluate_candidates(mkts: list[dict], a) -> pd.DataFrame:
    """Verify books, calibrate from trades, compute model quotes and expected $/h."""
    tok_of = {}
    for m in mkts:
        toks = FD._jl(m.get("clobTokenIds"))
        if toks:
            tok_of[toks[0]] = m
    books = {str(b.get("asset_id")): b for b, _, _ in FD.fetch_books(list(tok_of))}
    rows = []
    for i, (tok, m) in enumerate(tok_of.items()):
        b = books.get(tok)
        if not b:
            continue
        ob = OrderBook.from_clob_json(b)
        if ob.spread is None or ob.spread < a.min_spread:
            continue
        s = ob.spread / 2
        try:
            time.sleep(0.25)   # Data API throttles bursts (429/503)
            tr = fetch_trades(m["conditionId"], max_trades=a.max_trades)
        except Exception as e:  # noqa: BLE001 - network hiccup: skip this market, keep scanning
            log(f"  ! skip {m['slug'][:50]}: trades fetch failed ({type(e).__name__})")
            continue
        c = calibrate(tr, s, ob.tick_size, a.window, a.mu_floor)
        meta = fv_meta({**m, "token_id": tok})
        res = fair_value(a, ob, tr, meta, smooth=False)
        p = res.fair_value
        row = {"slug": m["slug"], "question": m.get("question"), "condition_id": m["conditionId"],
               "token_id": tok, "end_date": m.get("endDate"), "event_slug": meta["event_slug"],
               "neg_risk": meta["neg_risk"], "volume24hr": m.get("volume24hr"),
               "liquidity": m.get("liquidityNum"), "best_bid": ob.best_bid, "best_ask": ob.best_ask,
               "spread": ob.spread, "tick": ob.tick_size, "book_fv": res.components.get("book", np.nan),
               "trade_vwap": c.vwap if c else np.nan, "fair_value": p, "fv_raw": res.fair_value_raw,
               "fv_sd": res.sd, "fv_conf": res.confidence,
               "fv_signals": ", ".join(f"{s_['name']}:{s_['weight']:.0%}" for s_ in
                                       sorted(res.signals, key=lambda z: -z["weight"])),
               "depth_bid": res.diagnostics.get("depth_bid_shares"),
               "depth_ask": res.diagnostics.get("depth_ask_shares"),
               "n_trades": c.n_trades if c else len(tr)}
        if c and c.n_trades >= a.min_trades:
            P = mm_params(c, s, ob.tick_size, a)
            Q = quote(p, 0.0, P, sigma2_with_fv(sigma2_for(c, P, p, m.get("endDate")), res.sd, P.tau_h),
                      ob.best_bid, ob.best_ask, join_or_improve=not a.allow_behind)
            row.update(trades_per_h=(c.R_buy + c.R_sell), mean_size=c.m, impact=c.impact, J=c.J, mu=c.mu,
                       our_bid=Q.bid.price if Q.bid else np.nan, our_bid_size=Q.bid_size,
                       our_ask=Q.ask.price if Q.ask else np.nan, our_ask_size=Q.ask_size,
                       exp_pnl_h=(Q.bid.pnl_per_h if Q.bid else 0) + (Q.ask.pnl_per_h if Q.ask else 0),
                       exp_fills_h=(Q.bid.fills_per_h if Q.bid else 0) + (Q.ask.fills_per_h if Q.ask else 0))
        rows.append(row)
        if (i + 1) % 10 == 0:
            log(f"  calibrated {i + 1}/{len(tok_of)}")
    df = pd.DataFrame(rows)
    if "exp_pnl_h" in df:
        df = df.sort_values("exp_pnl_h", ascending=False, na_position="last")
    return df.reset_index(drop=True)


def cmd_scan(a) -> None:
    log(f"scanning: 24h vol ${a.min_vol:g}-{a.max_vol:g}, spread >= {a.min_spread}, "
        f">= {a.min_hours}h to resolution")
    mkts = scan_gamma(a)
    log(f"{len(mkts)} candidates from Gamma; keeping {min(len(mkts), a.candidates)} widest for calibration")
    mkts = sorted(mkts, key=lambda m: -(m["bestAsk"] - m["bestBid"]))[: a.candidates]
    df = evaluate_candidates(mkts, a)
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"scan_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}.parquet")
    df.to_parquet(path, index=False)
    show = df.head(a.top)
    cols = ["slug", "best_bid", "best_ask", "fair_value", "fv_sd", "volume24hr", "n_trades", "trades_per_h", "mu", "J",
            "our_bid", "our_bid_size", "our_ask", "our_ask_size", "exp_fills_h", "exp_pnl_h"]
    with pd.option_context("display.width", 250, "display.max_colwidth", 45,
                           "display.float_format", "{:.3f}".format):
        print(show[[c for c in cols if c in show]].to_string())
    log(f"saved {len(df)} rows -> {path}")


def last_scan(a) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(a.out, "scan_*.parquet")))
    if not files:
        raise SystemExit("no scan found: run `py Scalper.py scan` first")
    return pd.read_parquet(files[-1])


def pick_targets(a) -> pd.DataFrame:
    if a.slug:
        mkts = []
        for s in a.slug:
            mkts.extend(FD.http("GET", f"{GAMMA}/markets", params={"slug": s}))
        return evaluate_candidates(mkts, replace_ns(a, min_spread=0.0))
    df = last_scan(a)
    df = df[df.get("exp_pnl_h", pd.Series(0, index=df.index)) > 0] if not a.include_unprofitable else df
    return df.head(a.n).reset_index(drop=True)


def replace_ns(ns, **kw):
    d = vars(ns).copy()
    d.update(kw)
    return argparse.Namespace(**d)


# =============================================================================
# 3. Download
# =============================================================================
def save_trades(root: str, df: pd.DataFrame) -> None:
    if df.empty:
        return
    path = os.path.join(root, "trades.parquet")
    if os.path.exists(path):
        df = pd.concat([pd.read_parquet(path), df], ignore_index=True)
    df = df.drop_duplicates(["transaction_hash", "token_id", "price", "size", "side", "timestamp"])
    df.sort_values("ts").to_parquet(path, index=False)


def cmd_download(a) -> None:
    tg = pick_targets(a)
    log(f"downloading {len(tg)} markets")
    for _, r in tg.iterrows():
        tr = fetch_trades(r["condition_id"], max_trades=a.history)
        save_trades(a.out, tr)
        log(f"  {r['slug'][:60]}: {len(tr):,} trades "
            f"({tr['ts'].min() if len(tr) else '-'} -> {tr['ts'].max() if len(tr) else '-'})")
    # one book snapshot + metadata via FetchData
    FD.run(replace_ns(a, token=list(tg["token_id"]), slug=None, event=None, top=None, all=False,
                      once=True, interval=1, duration=0, flush=1e9, levels=20,
                      refresh_markets=1e9, max_buffer=10**9, min_volume=0, min_liquidity=0))
    log(f"done -> {os.path.abspath(a.out)}  (record continuously with: "
        f"py FetchData.py --token <ids> --out {a.out})")


# =============================================================================
# 4. Paper scalping
# =============================================================================
@dataclass
class Book:
    """Paper position + resting quotes for one market (YES token)."""
    slug: str
    token: str
    condition: str
    P: MMParams
    calib: Calib
    end_date: str | None
    recent: pd.DataFrame
    meta: dict = field(default_factory=dict)   # FairMV engine meta (slug, event, neg_risk, ...)
    q: float = 0.0
    cash: float = 0.0
    fees: float = 0.0
    bid: tuple | None = None    # (price, size_left, queue_ahead)
    ask: tuple | None = None
    fv: float = float("nan")
    fv_sd: float = float("nan")
    seen: set = field(default_factory=set)
    last_ts: int = 0
    n_fills: int = 0

    @property
    def pnl(self) -> float:
        return self.cash + self.q * self.fv if not math.isnan(self.fv) else self.cash


def match_trades(bk: Book, tr: pd.DataFrame, fills: list, now) -> None:
    """Simulate fills of our resting quotes against new real trades (YES terms)."""
    for _, t in tr.iterrows():
        key = (t["transaction_hash"], t["token_id"], t["price"], t["size"], t["side"], t["timestamp"])
        if key in bk.seen:
            continue
        bk.seen.add(key)
        px, sz, side = float(t["yes_price"]), float(t["size"]), int(t["yes_side"])
        if side > 0 and bk.ask and px >= bk.ask[0] - 1e-9:
            price, left, queue = bk.ask
            if px <= price + 1e-9:                       # trade at our level: queue first
                take = min(max(sz - queue, 0.0), left); queue = max(queue - sz, 0.0)
            else:                                        # trade through our level: we are hit first
                take = min(sz, left)
            if take > 0:
                bk.q -= take; bk.cash += take * price; bk.n_fills += 1
                fills.append(dict(ts=now, trade_ts=t["ts"], slug=bk.slug, token_id=bk.token, side="sell",
                                  price=price, size=take, fv=bk.fv, trade_price=px, trade_size=sz))
            bk.ask = (price, left - take, queue) if left - take > 0 else None
        elif side < 0 and bk.bid and px <= bk.bid[0] + 1e-9:
            price, left, queue = bk.bid
            if px >= price - 1e-9:
                take = min(max(sz - queue, 0.0), left); queue = max(queue - sz, 0.0)
            else:
                take = min(sz, left)
            if take > 0:
                bk.q += take; bk.cash -= take * price; bk.n_fills += 1
                fills.append(dict(ts=now, trade_ts=t["ts"], slug=bk.slug, token_id=bk.token, side="buy",
                                  price=price, size=take, fv=bk.fv, trade_price=px, trade_size=sz))
            bk.bid = (price, left - take, queue) if left - take > 0 else None


def level_size(levels, price) -> float:
    return sum(s for p, s in levels if abs(p - price) < 1e-9)


def cmd_scalp(a) -> None:
    tg = pick_targets(a)
    if tg.empty:
        raise SystemExit("no targets with positive expected PnL (use --include-unprofitable to force)")
    start_ts = int(time.time())
    books: dict[str, Book] = {}
    for _, r in tg.iterrows():
        ob = OrderBook.from_clob_json(FD.http("GET", f"{FD.CLOB_URL}/book", params={"token_id": r["token_id"]}))
        tr = fetch_trades(r["condition_id"], max_trades=a.max_trades)
        c = calibrate(tr, ob.spread / 2, ob.tick_size, a.window, a.mu_floor) if ob.spread else None
        if c is None:
            log(f"  skip {r['slug']}: not enough trades to calibrate")
            continue
        bk = Book(r["slug"], r["token_id"], r["condition_id"], mm_params(c, ob.spread / 2, ob.tick_size, a),
                  c, r.get("end_date"), tr.tail(200), meta=fv_meta(clean_row(r)))
        bk.seen = {(t.transaction_hash, t.token_id, t.price, t.size, t.side, t.timestamp)
                   for t in tr.itertuples()}                     # ignore history: only new trades fill us
        books[bk.token] = bk
        log(f"  {bk.slug[:55]:<55} trades/h {c.R_buy + c.R_sell:5.2f}  mu {c.mu:.2f}  J {c.J:.3f}  m {c.m:.0f}")
    if not books:
        raise SystemExit("nothing to scalp")

    stop = {"f": False}
    signal.signal(signal.SIGINT, lambda *_: stop.update(f=True))
    fills, quotes, snaps, levels, last_hash = [], [], [], [], {}
    sid, t0, last_flush = FD.next_snapshot_id(a.out), time.time(), time.time()
    log(f"paper scalping {len(books)} markets every {a.interval:g}s  (Ctrl+C to stop)")

    while not stop["f"]:
        tick_start = time.time()
        now = pd.Timestamp.now(tz="UTC")
        got = FD.fetch_books(list(books))
        for raw, fts, lat in got:
            bk = books.get(str(raw.get("asset_id")))
            if bk is None:
                continue
            # record the book (same schema as FetchData)
            r, lv = FD.process_book(raw, sid, fts, lat, {"slug": bk.slug, "outcome": "Yes"}, 20, last_hash)
            snaps.append(r); levels.extend(lv)
            ob = OrderBook.from_clob_json(raw)
            # 1) fills from trades printed since last poll, against the quotes that were resting
            try:
                tr = fetch_trades(bk.condition, max_trades=500, since_ts=start_ts)
                save_trades(a.out, tr) if a.save_trades and len(tr) else None
                match_trades(bk, tr, fills, now)
                if len(tr):
                    bk.recent = (pd.concat([bk.recent, tr]).drop_duplicates(
                        ["transaction_hash", "token_id", "price", "size", "side", "timestamp"])
                        .sort_values("ts").tail(200))
            except Exception as e:  # noqa: BLE001
                log(f"! trades {bk.slug[:40]}: {e}")
            # 2) re-quote
            if ob.best_bid is None or ob.best_ask is None:
                bk.bid = bk.ask = None
                continue
            res = fair_value(a, ob, bk.recent, bk.meta, smooth=True)
            if res.fair_value is None:
                bk.bid = bk.ask = None
                continue
            bk.fv, bk.fv_sd = res.fair_value, (res.sd if res.sd is not None else float("nan"))
            s2 = sigma2_with_fv(sigma2_for(bk.calib, bk.P, bk.fv, bk.end_date), res.sd, bk.P.tau_h)
            Q = quote(bk.fv, bk.q, bk.P, s2, ob.best_bid, ob.best_ask, join_or_improve=not a.allow_behind)
            for w in res.warnings:
                if "dust" not in w:
                    log(f"  fv {bk.slug[:30]}: {w}")
            bk.bid = bk.ask = None
            if Q.bid:
                ahead = level_size(ob.bids, Q.bid.price) if Q.bid.price >= ob.best_bid - 1e-9 else 0.0
                bk.bid = (Q.bid.price, Q.bid_size, ahead)
            if Q.ask:
                ahead = level_size(ob.asks, Q.ask.price) if Q.ask.price <= ob.best_ask + 1e-9 else 0.0
                bk.ask = (Q.ask.price, Q.ask_size, ahead)
            quotes.append(dict(ts=now, slug=bk.slug, token_id=bk.token, fv=bk.fv, fv_sd=bk.fv_sd,
                               fv_raw=res.fair_value_raw, fv_conf=res.confidence, inventory=bk.q,
                               mkt_bid=ob.best_bid, mkt_ask=ob.best_ask,
                               our_bid=Q.bid.price if Q.bid else np.nan, bid_size=Q.bid_size,
                               our_ask=Q.ask.price if Q.ask else np.nan, ask_size=Q.ask_size,
                               exp_pnl_h=(Q.bid.pnl_per_h if Q.bid else 0) + (Q.ask.pnl_per_h if Q.ask else 0),
                               cash=bk.cash, pnl=bk.pnl))
        sid += 1
        tot = sum(b.pnl for b in books.values())
        nf = sum(b.n_fills for b in books.values())
        log(f"round {sid}: PnL ${tot:+.2f}  fills {nf}  inventory " +
            ", ".join(f"{b.slug[:18]}:{b.q:+.0f}" for b in books.values() if b.q))

        done = a.duration and time.time() - t0 >= a.duration
        if done or stop["f"] or time.time() - last_flush >= a.flush:
            FD.flush(a.out, snaps, levels)
            for name, rows in (("paper_fills", fills), ("paper_quotes", quotes)):
                if rows:
                    df = pd.DataFrame(rows)
                    path = os.path.join(a.out, f"{name}.parquet")
                    if os.path.exists(path):
                        df = pd.concat([pd.read_parquet(path), df], ignore_index=True)
                    df.to_parquet(path, index=False)
                    rows.clear()
            last_flush = time.time()
        if done:
            break
        while not stop["f"] and time.time() - tick_start < a.interval:
            time.sleep(0.2)

    print("\nFinal paper positions (marked to fair value):")
    for b in books.values():
        print(f"  {b.slug[:60]:<60} q {b.q:+8.0f}  cash {b.cash:+9.2f}  fv {b.fv:.3f}  "
              f"PnL {b.pnl:+8.2f}  fills {b.n_fills}")
    print(f"  TOTAL PnL ${sum(b.pnl for b in books.values()):+.2f}")


def cmd_report(a) -> None:
    fp, qp = os.path.join(a.out, "paper_fills.parquet"), os.path.join(a.out, "paper_quotes.parquet")
    if not os.path.exists(qp):
        raise SystemExit("no paper trading data yet")
    q = pd.read_parquet(qp)
    last = q.sort_values("ts").groupby("slug").tail(1)
    print(f"Quotes logged: {len(q):,} from {q.ts.min()} to {q.ts.max()}")
    if os.path.exists(fp):
        f = pd.read_parquet(fp)
        f["edge_vs_fv"] = np.where(f.side == "sell", f.price - f.fv, f.fv - f.price) * f["size"]
        print(f"\nFills: {len(f)}   volume {f['size'].sum():,.0f} shares   "
              f"edge vs FV at fill ${f.edge_vs_fv.sum():+.2f}")
        print(f.groupby("slug").agg(fills=("size", "size"), shares=("size", "sum"),
                                    edge=("edge_vs_fv", "sum")).to_string())
    print("\nLatest mark-to-fair-value PnL per market:")
    print(last[["slug", "inventory", "cash", "fv", "pnl"]].to_string(index=False))
    print(f"TOTAL ${last.pnl.sum():+.2f}")
    print("\nNote: per-run PnL; positions restart flat each scalp run.")


# =============================================================================
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Wide-spread Polymarket scanner + paper scalper")
    ap.add_argument("cmd", choices=["scan", "download", "scalp", "report"])
    # scan
    ap.add_argument("--min-vol", type=float, default=50, help="min 24h volume $")
    ap.add_argument("--max-vol", type=float, default=5000, help="max 24h volume $")
    ap.add_argument("--min-spread", type=float, default=0.04)
    ap.add_argument("--min-hours", type=float, default=48, help="min hours until resolution")
    ap.add_argument("--max-pages", type=int, default=150, help="Gamma pages to scan (100 markets each)")
    ap.add_argument("--candidates", type=int, default=60, help="max markets to calibrate")
    ap.add_argument("--top", type=int, default=20, help="rows to print")
    # calibration / model
    ap.add_argument("--window", type=float, default=24 * 7, help="trade history window, hours")
    ap.add_argument("--max-trades", type=int, default=2000)
    ap.add_argument("--k", type=float, default=30.0, help="flow sensitivity to price (unidentified, see doc)")
    ap.add_argument("--flow-cap", type=float, default=3.0,
                    help="max flow multiple from quoting at fair value vs at market quotes")
    ap.add_argument("--mu-floor", type=float, default=0.10, help="minimum informed share")
    ap.add_argument("--min-trades", type=int, default=20, help="min trades in window to rank a market")
    ap.add_argument("--allow-behind", action="store_true",
                    help="allow quotes behind the market's best (default: join or improve only)")
    ap.add_argument("--bankroll", type=float, default=1000.0)
    ap.add_argument("--kelly", type=float, default=0.25)
    ap.add_argument("--tau", type=float, default=4.0, help="inventory holding time, hours")
    ap.add_argument("--max-position", type=float, default=500.0)
    # fair value (FairMV engine)
    ap.add_argument("--fv-user-dir", default=DEFAULT_USER_DIR,
                    help="folder with your views.json / links.json / feeds/ / tables/ / config.json")
    ap.add_argument("--no-fv-links", action="store_true", help="disable correlated-market / event-sum links")
    ap.add_argument("--no-fv-feeds", action="store_true", help="disable your views and feeds")
    # targets / run
    ap.add_argument("--n", type=int, default=5, help="top-N markets from last scan")
    ap.add_argument("--slug", nargs="+", help="explicit markets instead of last scan")
    ap.add_argument("--include-unprofitable", action="store_true")
    ap.add_argument("--history", type=int, default=20000, help="max trades to download per market")
    ap.add_argument("--interval", type=float, default=15)
    ap.add_argument("--duration", type=float, default=0, help="seconds (0 = until Ctrl+C)")
    ap.add_argument("--flush", type=float, default=120)
    ap.add_argument("--save-trades", action="store_true", help="also append live trades to trades.parquet")
    ap.add_argument("--out", default="data")
    a = ap.parse_args(argv)
    {"scan": cmd_scan, "download": cmd_download, "scalp": cmd_scalp, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
