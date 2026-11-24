"""
sources.py - where events come from.

  live_ws     Polymarket market websocket: full book on subscribe, price_change
              diffs, last_trade_price prints (with tx hash). Real time.
  live_poll   REST fallback: POST /books + data-api trades every --interval s.
  replay      recorded books (FetchData format: <root>/levels + snapshots, which
              both FetchData.py and live runs here write) + trades (local files
              and/or the data-api), merged in time order.
"""
from __future__ import annotations

import asyncio
import glob
import json
import os
import signal
import time

import pandas as pd

from sim import (EPS, FD, ROOT, SC, Market, OrderBook, Runner, api_trade_rows, calibrate_market, log,
                 trade_key, trade_row)

WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
GAMMA = "https://gamma-api.polymarket.com"


# =============================================================================
# Market setup
# =============================================================================
def gamma_market(slug: str | None = None, token: str | None = None, condition: str | None = None) -> dict | None:
    p = {"slug": slug} if slug else {"clob_token_ids": token} if token else {"condition_ids": condition}
    try:
        res = FD.http("GET", f"{GAMMA}/markets", params=p)
        return res[0] if res else None
    except Exception as e:  # noqa: BLE001
        log(f"! gamma lookup {slug or token or condition}: {e}")
        return None


def market_from_gamma(g: dict, outcome: int = 0) -> Market:
    toks = FD._jl(g.get("clobTokenIds")) or []
    ev = (g.get("events") or [{}])[0] if isinstance(g.get("events"), list) else {}
    tok = str(toks[outcome])
    meta = SC.fv_meta({**g, "token_id": tok})
    meta["outcome_index"] = outcome
    return Market(token=tok, condition=g["conditionId"], slug=g["slug"],
                  other_token=str(toks[1 - outcome]) if len(toks) == 2 else None,
                  end_date=g.get("endDate"), meta=meta,
                  tick=float(g.get("orderPriceMinTickSize") or 0.01))


def resolve_targets(a) -> list[Market]:
    out = []
    for s in a.slug or []:
        outcome = 1 if s.lower().endswith(":no") else 0
        g = gamma_market(slug=s.split(":")[0])
        if g:
            out.append(market_from_gamma(g, outcome))
    for t in a.token or []:
        g = gamma_market(token=t)
        if g:
            toks = [str(x) for x in FD._jl(g.get("clobTokenIds"))]
            out.append(market_from_gamma(g, toks.index(str(t)) if str(t) in toks else 0))
    if a.from_scan:
        sc = SC.last_scan(argparse_ns(out=a.scan_dir))
        sc = sc[sc.get("exp_pnl_h", 0) > 0] if "exp_pnl_h" in sc else sc
        for r in sc.head(a.from_scan).itertuples():
            g = gamma_market(slug=r.slug)
            if g:
                out.append(market_from_gamma(g, 0))
    seen, uniq = set(), []
    for m in out:
        if m.token not in seen:
            seen.add(m.token)
            uniq.append(m)
    if not uniq:
        raise SystemExit("no markets: pass --slug, --token or --from-scan N")
    return uniq


def argparse_ns(**kw):
    import argparse
    return argparse.Namespace(**kw)


def rest_book(token: str) -> OrderBook | None:
    try:
        return OrderBook.from_clob_json(FD.http("GET", f"{FD.CLOB_URL}/book", params={"token_id": token}))
    except Exception as e:  # noqa: BLE001
        log(f"! book {token[:12]}..: {e}")
        return None


def prepare_live(a, markets: list[Market]) -> None:
    """Initial book + trade history (calibration and FV warm-up) for each market."""
    for m in markets:
        m.book = rest_book(m.token)
        if m.book and m.book.tick_size:
            m.tick = m.book.tick_size
        try:
            hist = api_trade_rows(m, SC.fetch_trades(m.condition, max_trades=a.max_trades), "history")
        except Exception as e:  # noqa: BLE001
            log(f"! history {m.slug[:40]}: {e}")
            hist = []
        for r in hist[-m.recent.maxlen:]:
            m.recent.append({**r, "late": True})
        if not a.observe:
            calibrate_market(a, m, hist)


# =============================================================================
# Live: websocket
# =============================================================================
class WSBooks:
    """Level dicts per token, maintained from 'book' snapshots + 'price_change' diffs."""

    def __init__(self):
        self.bids: dict[str, dict] = {}
        self.asks: dict[str, dict] = {}

    def snapshot(self, tok, bids, asks):
        self.bids[tok] = {float(x["price"]): float(x["size"]) for x in bids}
        self.asks[tok] = {float(x["price"]): float(x["size"]) for x in asks}

    def change(self, tok, price, size, side):
        d = (self.bids if str(side).upper() == "BUY" else self.asks).setdefault(tok, {})
        p, s = float(price), float(size)
        if s <= EPS:
            d.pop(p, None)
        else:
            d[p] = s

    def book(self, tok, m: Market) -> OrderBook:
        return OrderBook.from_levels(self.bids.get(tok, {}).items(), self.asks.get(tok, {}).items(),
                                     tick_size=m.tick, token_id=tok,
                                     min_order_size=m.book.min_order_size if m.book else 5.0)


def _ts_ms(v) -> pd.Timestamp:
    try:
        return pd.Timestamp(int(v), unit="ms", tz="UTC")
    except (TypeError, ValueError):
        return pd.Timestamp.now(tz="UTC")


def handle_ws(R: Runner, W: WSBooks, msg: dict) -> None:
    et = msg.get("event_type")
    if et == "book":
        tok = str(msg.get("asset_id"))
        if tok in R.mk:
            W.snapshot(tok, msg.get("bids", []), msg.get("asks", []))
            R.on_book(tok, W.book(tok, R.mk[tok]), _ts_ms(msg.get("timestamp")), msg)
    elif et == "price_change":
        ts, touched = _ts_ms(msg.get("timestamp")), set()
        for c in msg.get("price_changes", []):
            tok = str(c.get("asset_id"))
            if tok in R.mk:
                W.change(tok, c["price"], c["size"], c["side"])
                touched.add(tok)
        for tok in touched:
            R.on_book(tok, W.book(tok, R.mk[tok]), ts)
    elif et == "last_trade_price":
        m = R.by_any.get(str(msg.get("asset_id")))
        if m is not None:
            R.on_trade(trade_row(m, msg["asset_id"], msg["side"], msg["price"], msg["size"],
                                 _ts_ms(msg.get("timestamp")), msg.get("transaction_hash"), "ws"))
    elif et == "tick_size_change":
        m = R.mk.get(str(msg.get("asset_id")))
        if m is not None and msg.get("new_tick_size"):
            m.tick = float(msg["new_tick_size"])


class Clock:
    """Shared periodic work for live loops: status, recording, flushing, stop."""

    def __init__(self, a, R: Runner):
        self.a, self.R = a, R
        self.t0 = self.last_status = self.last_flush = self.last_rec = time.time()
        self.stop = False
        signal.signal(signal.SIGINT, lambda *_: setattr(self, "stop", True))

    def tick(self) -> bool:
        now = time.time()
        if now - self.last_rec >= self.a.record_every:
            self.R.record_books(pd.Timestamp.now(tz="UTC"))
            self.last_rec = now
        if now - self.last_status >= self.a.status_every:
            log(self.R.status())
            self.last_status = now
        if now - self.last_flush >= self.a.flush:
            self.R.flush()
            self.last_flush = now
        if self.a.duration and now - self.t0 >= self.a.duration:
            self.stop = True
        return not self.stop


def backfill(R: Runner, since: pd.Timestamp) -> None:
    """After a reconnect: trades we missed are recorded (late) but never fill us."""
    for m in R.mk.values():
        try:
            df = SC.fetch_trades(m.condition, max_trades=500, since_ts=int(since.timestamp()))
            for r in api_trade_rows(m, df, "backfill"):
                R.on_trade(r)
        except Exception as e:  # noqa: BLE001
            log(f"! backfill {m.slug[:40]}: {e}")


def live_ws(a, R: Runner) -> None:
    import websockets
    C, W = Clock(a, R), WSBooks()
    assets = sorted(set(R.by_any))

    async def run():
        down_since, backoff = None, 1
        while C.tick():
            try:
                async with websockets.connect(WS_URL, ping_interval=None, max_size=2 ** 24,
                                              open_timeout=20) as ws:
                    await ws.send(json.dumps({"assets_ids": assets, "type": "market"}))
                    log(f"websocket subscribed: {len(R.mk)} markets ({len(assets)} tokens)")
                    if down_since is not None:
                        backfill(R, down_since)
                    down_since, backoff, last_ping = None, 1, time.time()
                    while C.tick():
                        try:
                            raw = await asyncio.wait_for(ws.recv(), 1.0)
                        except asyncio.TimeoutError:
                            raw = None
                        if time.time() - last_ping >= 10:
                            await ws.send("PING")
                            last_ping = time.time()
                        if not raw or raw == "PONG":
                            continue
                        try:
                            msg = json.loads(raw)
                        except ValueError:
                            continue
                        for x in (msg if isinstance(msg, list) else [msg]):
                            handle_ws(R, W, x)
            except Exception as e:  # noqa: BLE001 - network: reconnect with backoff
                down_since = down_since or pd.Timestamp.now(tz="UTC")
                log(f"! websocket: {type(e).__name__}: {e} - reconnect in {backoff}s")
                t = time.time()
                while time.time() - t < backoff and C.tick():
                    await asyncio.sleep(0.2)
                backoff = min(backoff * 2, 60)

    asyncio.run(run())


# =============================================================================
# Live: REST polling
# =============================================================================
def live_poll(a, R: Runner) -> None:
    C = Clock(a, R)
    since = int(time.time())
    while C.tick():
        t0 = time.time()
        # trades first: they printed while the previous quotes were resting
        for m in R.mk.values():
            try:
                df = SC.fetch_trades(m.condition, max_trades=500, since_ts=since)
                for r in sorted(api_trade_rows(m, df, "poll"), key=lambda r: r["ts"]):
                    R.on_trade(r)
            except Exception as e:  # noqa: BLE001
                log(f"! trades {m.slug[:40]}: {e}")
        for raw, fts, _lat in FD.fetch_books(list(R.mk)):
            tok = str(raw.get("asset_id"))
            if tok in R.mk:
                R.on_book(tok, OrderBook.from_clob_json(raw), fts, raw)
        while C.tick() and time.time() - t0 < a.interval:
            time.sleep(0.2)


# =============================================================================
# Replay (backtest)
# =============================================================================
def _read_parquet_dir(path: str, tokens: list | None, columns=None) -> pd.DataFrame:
    if not os.path.isdir(path):
        return pd.DataFrame()
    filt = [("token_id", "in", tokens)] if tokens else None
    try:
        return pd.read_parquet(path, columns=columns, filters=filt)
    except Exception:  # noqa: BLE001 - schema drift between part files: read one by one
        parts = []
        for f in glob.glob(os.path.join(path, "**", "*.parquet"), recursive=True):
            d = pd.read_parquet(f, columns=columns)
            parts.append(d[d["token_id"].isin(tokens)] if tokens else d)
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def load_recorded_books(roots: list[str], tokens: list | None, start, end) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(snapshots, levels) across roots, restricted to tokens / time window."""
    scols = ["snapshot_id", "fetch_ts", "token_id", "condition_id", "slug", "tick_size",
             "min_order_size", "neg_risk", "best_bid", "best_ask"]
    S, L = [], []
    for i, root in enumerate(roots):
        s = _read_parquet_dir(os.path.join(root, "snapshots"), tokens, None)
        lv = _read_parquet_dir(os.path.join(root, "levels"), tokens,
                               ["snapshot_id", "fetch_ts", "token_id", "side", "price", "size"])
        if s.empty:
            continue
        s = s[[c for c in scols if c in s.columns]].copy()
        s["root"], lv["root"] = i, i
        S.append(s)
        L.append(lv)
    if not S:
        raise SystemExit(f"no recorded books under {roots} (run FetchData.py or `paper.py live` first)")
    S, L = pd.concat(S, ignore_index=True), pd.concat(L, ignore_index=True)
    for d in (S, L):
        d["token_id"] = d["token_id"].astype(str)
        d["fetch_ts"] = pd.to_datetime(d["fetch_ts"], utc=True)
    if start is not None:
        S, L = S[S.fetch_ts >= start], L[L.fetch_ts >= start]
    if end is not None:
        S, L = S[S.fetch_ts <= end], L[L.fetch_ts <= end]
    return S.sort_values("fetch_ts"), L


def local_trades(paths: list[str]) -> pd.DataFrame:
    """Any trades files we have: Scalper's data/trades.parquet, previous runs' trades.parquet."""
    out = []
    for p in paths:
        for f in glob.glob(p):
            try:
                d = pd.read_parquet(f)
            except Exception:  # noqa: BLE001
                continue
            if {"trade_token", "raw_side", "raw_price"} <= set(d.columns):       # a PaperTrades run
                d = d.rename(columns={"trade_token": "tok", "raw_side": "rside", "raw_price": "rprice"})
            elif {"token_id", "side", "price"} <= set(d.columns):                 # data-api / Scalper
                d = d.rename(columns={"token_id": "tok", "side": "rside", "price": "rprice"})
            else:
                continue
            need = ["ts", "tok", "rside", "rprice", "size", "transaction_hash", "condition_id"]
            if not set(need) <= set(d.columns):
                continue
            out.append(d[need])
    if not out:
        return pd.DataFrame(columns=["ts", "tok", "rside", "rprice", "size", "transaction_hash", "condition_id"])
    d = pd.concat(out, ignore_index=True)
    d["ts"] = pd.to_datetime(d["ts"], utc=True)
    d["tok"] = d["tok"].astype(str)
    return d


def replay(a, out: str) -> Runner:
    roots = a.data or [os.path.join(ROOT, "data")]
    start = pd.Timestamp(a.start, tz="UTC") if a.start and pd.Timestamp(a.start).tzinfo is None \
        else (pd.Timestamp(a.start) if a.start else None)
    end = pd.Timestamp(a.end, tz="UTC") if a.end and pd.Timestamp(a.end).tzinfo is None \
        else (pd.Timestamp(a.end) if a.end else None)
    want = None
    if a.slug or a.token:
        want = [m.token for m in resolve_targets(a)]
    S, L = load_recorded_books(roots, want, start, end)
    toks = list(dict.fromkeys(S["token_id"]))
    t_lo, t_hi = S.fetch_ts.min(), S.fetch_ts.max()
    log(f"replaying {len(S):,} book snapshots for {len(toks)} tokens, {t_lo} -> {t_hi}")

    # markets: Gamma for metadata (end date, complement token), recorded columns as fallback
    markets = []
    for tok in toks:
        g = gamma_market(token=tok) if not a.offline else None
        first = S[S.token_id == tok].iloc[0]
        if g:
            gt = [str(x) for x in FD._jl(g.get("clobTokenIds"))]
            m = market_from_gamma(g, gt.index(tok) if tok in gt else 0)
        else:
            m = Market(token=tok, condition=str(first.get("condition_id")), slug=str(first.get("slug")),
                       meta={"slug": str(first.get("slug")), "token_id": tok,
                             "condition_id": str(first.get("condition_id")), "outcome_index": 0})
        m.tick = float(first["tick_size"]) if pd.notna(first.get("tick_size")) and first["tick_size"] > 0 else m.tick
        markets.append(m)

    # trades: local files, plus the data-api (it keeps the full history per market)
    tl = local_trades(a.trades_files)
    hist: dict[str, list] = {}
    for m in markets:
        rows = []
        lt = tl[tl.condition_id.astype(str) == m.condition] if len(tl) else tl
        for t in lt.itertuples():
            rows.append(trade_row(m, t.tok, t.rside, t.rprice, t.size, t.ts, t.transaction_hash, "local"))
        if not a.offline:
            try:
                df = SC.fetch_trades(m.condition, max_trades=a.max_trades)
                df = df[df.ts <= t_hi + pd.Timedelta(minutes=5)]
                rows += api_trade_rows(m, df, "api")
            except Exception as e:  # noqa: BLE001
                log(f"! trades {m.slug[:40]}: {e}")
        uniq, seen = [], set()
        for r in sorted(rows, key=lambda r: r["ts"]):
            k = trade_key(r)
            if k not in seen:
                seen.add(k)
                uniq.append(r)
        hist[m.token] = uniq
        before = [r for r in uniq if r["ts"] < t_lo]
        during = [r for r in uniq if t_lo <= r["ts"] <= t_hi]
        log(f"  {m.slug[:55]:<55} trades: {len(before):,} before, {len(during):,} during")
        if before and before[0]["ts"] > t_lo - pd.Timedelta(hours=a.window) and len(before) >= a.max_trades - 5:
            log("    (history truncated by --max-trades; calibration window shorter than --window)")

    # first book per token -> calibration on pre-window trades only (no look-ahead)
    L = L.sort_values(["root", "snapshot_id", "token_id", "side", "price"])
    groups = {k: g for k, g in L.groupby(["root", "snapshot_id", "token_id"], sort=False)}

    def book_of(s) -> OrderBook:
        g = groups.get((s.root, s.snapshot_id, s.token_id))
        bids = [] if g is None else list(zip(g.price[g.side == "bid"], g["size"][g.side == "bid"]))
        asks = [] if g is None else list(zip(g.price[g.side == "ask"], g["size"][g.side == "ask"]))
        mos = s.min_order_size if "min_order_size" in S.columns and pd.notna(s.min_order_size) else 5.0
        return OrderBook.from_levels(bids, asks, tick_size=float(s.tick_size) if pd.notna(s.tick_size) and s.tick_size > 0 else 0.01, token_id=s.token_id,
                                     min_order_size=float(mos))

    by_tok = {m.token: m for m in markets}
    for m in markets:
        s0 = S[S.token_id == m.token].iloc[0]
        m.book = book_of(s0)
        before = [r for r in hist[m.token] if r["ts"] < t_lo]
        for r in before[-m.recent.maxlen:]:
            m.recent.append({**r, "late": True})
        if not a.observe:
            calibrate_market(a, m, before)
        m.book = None
    R = Runner(a, markets, out, "backtest", live=False)

    # merged event stream: trades sort before a book taken in the same instant
    ev = [(r["ts"], 0, i, r) for tok in by_tok for i, r in enumerate(hist[tok]) if t_lo <= r["ts"] <= t_hi]
    ev += [(s.fetch_ts, 1, i, s) for i, s in enumerate(S.itertuples())]
    ev.sort(key=lambda e: (e[0], e[1], e[2]))
    for k, (ts, kind, _, x) in enumerate(ev):
        if kind == 0:
            R.on_trade(x)
        else:
            R.on_book(x.token_id, book_of(x), ts)
        if k and k % 20000 == 0:
            log(f"  {k:,}/{len(ev):,} events  {R.status()}")
    return R
