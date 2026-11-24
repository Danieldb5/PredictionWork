"""
FetchData.py - Polymarket order book recorder.

Polls the CLOB order book of one, several, or all live Polymarket markets at a
fixed interval. Every fetch records the full book (every price level) plus
market metadata to Parquet, which loads directly into pandas, polars, or any
ML pipeline.

-------------------------------------------------------------------------------
OUTPUT LAYOUT  (default ./data, change with --out)
-------------------------------------------------------------------------------
  data/
    markets.parquet                               one row per outcome token:
                                                  every scalar Gamma field +
                                                  event info, upserted on refresh
    snapshots/date=YYYY-MM-DD/part-*.parquet      one row per token per fetch
    levels/date=YYYY-MM-DD/part-*.parquet         one row per price level per
                                                  token per fetch (full depth)

snapshots (wide, ML-ready, fixed schema)
  ids/time   : snapshot_id, fetch_ts (UTC), server_ts (UTC), latency_ms,
               token_id, condition_id, slug, outcome, book_hash, changed
  book meta  : tick_size, min_order_size, neg_risk, last_trade_price
  top of book: best_bid, best_ask, bid_size_1, ask_size_1, mid, logit_mid,
               spread, spread_ticks, microprice, imbalance_1
  depth      : n_bid_levels, n_ask_levels, bid/ask_depth_{total,1c,5c,10c}
               (shares), bid/ask_notional_total ($), imbalance_{5c,10c}
  vwap       : bid/ask_vwap_{100,1000} (cost to trade N shares)
  levels     : bid_px_1..K, bid_sz_1..K, ask_px_1..K, ask_sz_1..K
               (K = --levels, NaN-padded)
  fair value : fv, fv_confidence, gm_mu   (FairMV.FairValueModel: book-only, if present)

levels (long, lossless)
  snapshot_id, fetch_ts, token_id, side ('bid'/'ask'), level (1 = best),
  price, size, cum_size, notional

-------------------------------------------------------------------------------
USAGE
-------------------------------------------------------------------------------
  py FetchData.py --slug <market-slug> [<slug> ...]       specific markets
  py FetchData.py --event <event-slug>                    every market in event(s)
  py FetchData.py --token <token-id> [...]                specific outcome tokens
  py FetchData.py --top 50                                top 50 live markets by 24h volume
  py FetchData.py --all --min-volume 10000                every live market with a book
                                                          (filtered; there are ~140k live
                                                          markets, so unfiltered --all
                                                          takes minutes per round)

  Options: --interval 5 (s between fetches)   --duration 3600 (s, default forever)
           --min-volume / --min-liquidity ($ filters for --top / --all)
           --flush 300 (s between disk writes) --levels 20 (wide columns)
           --out data                          --once (single fetch then exit)
           --compact                           merge part files per day, then exit

  Stop with Ctrl+C. Buffered rows are flushed before exit.

-------------------------------------------------------------------------------
LOADING
-------------------------------------------------------------------------------
  from FetchData import load
  snaps, levels, markets = load("data")
  # or: pd.read_parquet("data/snapshots", filters=[("date", "=", "2026-11-23")])
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

CLOB_URL = "https://clob.polymarket.com"
GAMMA_URL = "https://gamma-api.polymarket.com"
BOOKS_BATCH = 100       # token ids per POST /books request
GAMMA_PAGE = 100        # Gamma caps page size at 100
WORKERS = 4             # parallel /books requests

# Optional fair value model
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # FairMV package lives here
    from FairMV import FairValueModel, OrderBook as FVBook  # type: ignore
    _FV_MODEL = FairValueModel()
except Exception:  # noqa: BLE001
    _FV_MODEL = None

SESSION = requests.Session()
SESSION.headers["User-Agent"] = "polymarket-book-recorder/1.0"


# =============================================================================
# HTTP with retry / backoff
# =============================================================================
def http(method: str, url: str, retries: int = 5, **kw):
    delay = 1.0
    for attempt in range(retries):
        try:
            r = SESSION.request(method, url, timeout=30, **kw)
            if r.status_code == 429 or r.status_code >= 500:
                raise requests.HTTPError(f"{r.status_code}", response=r)
            r.raise_for_status()
            return r.json()
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as e:
            resp = getattr(e, "response", None)
            if resp is not None and 400 <= resp.status_code < 500 and resp.status_code != 429:
                raise
            if attempt == retries - 1:
                raise
            wait = delay
            if resp is not None and resp.headers.get("Retry-After"):
                try:
                    wait = float(resp.headers["Retry-After"])
                except ValueError:
                    pass
            log(f"  http {method} retry {attempt + 1} in {wait:.0f}s ({e})")
            time.sleep(wait)
            delay = min(delay * 2, 30)


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# =============================================================================
# Market discovery / metadata  (Gamma API)
# =============================================================================
def _jl(v):
    return json.loads(v) if isinstance(v, str) else (v or [])


def market_rows(m: dict) -> list[dict]:
    """Explode one Gamma market into one metadata row per outcome token (all scalar fields)."""
    tokens, outcomes = _jl(m.get("clobTokenIds")), _jl(m.get("outcomes"))
    prices = _jl(m.get("outcomePrices"))
    base = {k: v for k, v in m.items() if not isinstance(v, (list, dict))}
    base.pop("clobTokenIds", None), base.pop("outcomes", None), base.pop("outcomePrices", None)
    ev = (m.get("events") or [{}])[0]
    base.update({f"event_{k}": ev.get(k) for k in ("id", "slug", "title", "ticker", "endDate")})
    rows = []
    for i, tok in enumerate(tokens):
        r = dict(base)
        r.update(token_id=str(tok), outcome=outcomes[i] if i < len(outcomes) else None,
                 outcome_index=i,
                 outcome_price=float(prices[i]) if i < len(prices) and prices[i] not in (None, "") else None,
                 complement_token_id=str(tokens[1 - i]) if len(tokens) == 2 else None,
                 meta_fetched_at=pd.Timestamp.now(tz="UTC"))
        rows.append(r)
    return rows


def _live(m: dict) -> bool:
    return bool(m.get("enableOrderBook") and m.get("acceptingOrders") and m.get("clobTokenIds"))


def live_markets_by_volume(limit: int | None, min_volume: float, min_liquidity: float) -> list[dict]:
    """Live markets sorted by 24h volume (desc), via Gamma keyset pagination
    (the offset API is capped at ~2000). Stops early once `limit` is reached or
    volume falls below `min_volume`."""
    out, cursor, pages = [], None, 0
    while True:
        params = {"active": "true", "closed": "false", "limit": GAMMA_PAGE,
                  "order": "volume24hr", "ascending": "false"}
        if min_liquidity > 0:
            params["liquidity_num_min"] = min_liquidity
        if cursor:
            params["after_cursor"] = cursor
        d = http("GET", f"{GAMMA_URL}/markets/keyset", params=params)
        page = d.get("markets", [])
        pages += 1
        for m in page:
            if min_volume > 0 and _f(m.get("volume24hr")) < min_volume:
                return out
            if _live(m) and _f(m.get("liquidityNum") or m.get("liquidity") or 0) >= min_liquidity:
                out.append(m)
                if limit and len(out) >= limit:
                    return out
        if pages % 50 == 0:
            log(f"  discovering markets... {len(out):,} live so far")
        nxt = d.get("next_cursor")
        if not page or not nxt or nxt == cursor:
            return out
        cursor = nxt
        time.sleep(0.1)  # be polite: a full scan is ~1,400 pages


def discover(args) -> list[dict]:
    """Return Gamma market dicts for the requested targets."""
    mkts: list[dict] = []
    for s in args.slug or []:
        found = http("GET", f"{GAMMA_URL}/markets", params={"slug": s})
        if not found:
            log(f"! no market for slug '{s}' (if it is an event slug use --event)")
        mkts.extend(found)
    for s in args.event or []:
        ev = http("GET", f"{GAMMA_URL}/events", params={"slug": s})
        if not ev:
            log(f"! no event for slug '{s}'")
            continue
        for m in ev[0].get("markets", []):
            m.setdefault("events", [{k: ev[0].get(k) for k in ("id", "slug", "title", "ticker", "endDate")}])
            mkts.append(m)
    for t in args.token or []:
        found = http("GET", f"{GAMMA_URL}/markets", params={"clob_token_ids": t})
        if found:
            mkts.extend(found)
        else:  # unknown to Gamma: record it bare
            mkts.append({"clobTokenIds": json.dumps([t]), "outcomes": json.dumps(["?"]),
                         "enableOrderBook": True, "acceptingOrders": True})
    if args.top or args.all:
        mkts.extend(live_markets_by_volume(None if args.all else args.top,
                                           args.min_volume, args.min_liquidity))
    # de-duplicate by condition id / slug
    seen, uniq = set(), []
    for m in mkts:
        key = m.get("conditionId") or m.get("slug") or m.get("clobTokenIds")
        if key not in seen:
            seen.add(key)
            uniq.append(m)
    return uniq


# =============================================================================
# Book fetching (CLOB API)
# =============================================================================
def fetch_books(token_ids: list[str]) -> list[tuple[dict, pd.Timestamp, float]]:
    """Batch-fetch books -> [(book, fetch_ts, latency_ms)]. fetch_ts is per request batch,
    so it stays accurate when one round spans many requests. Closed tokens are omitted by the API."""
    chunks = [token_ids[i:i + BOOKS_BATCH] for i in range(0, len(token_ids), BOOKS_BATCH)]

    def one(chunk):
        ts, t0 = pd.Timestamp.now(tz="UTC"), time.time()
        try:
            res = http("POST", f"{CLOB_URL}/books", json=[{"token_id": t} for t in chunk]) or []
        except Exception as e:  # noqa: BLE001
            log(f"! /books chunk failed ({len(chunk)} tokens): {e}")
            return []
        lat = (time.time() - t0) * 1000
        return [(b, ts, lat) for b in res]

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        return [x for res in ex.map(one, chunks) for x in res]


# =============================================================================
# Feature extraction
# =============================================================================
def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def _logit(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def _vwap(levels, n):
    rem, cost = n, 0.0
    for p, s in levels:
        t = min(s, rem)
        cost += t * p
        rem -= t
        if rem <= 0:
            return cost / n
    return np.nan  # not enough depth


def process_book(book: dict, sid: int, fetch_ts, latency, meta: dict, K: int,
                 last_hash: dict) -> tuple[dict, list[dict]]:
    tok = str(book.get("asset_id"))
    bids = sorted(((_f(x["price"]), _f(x["size"])) for x in book.get("bids", [])), key=lambda x: -x[0])
    asks = sorted(((_f(x["price"]), _f(x["size"])) for x in book.get("asks", [])), key=lambda x: x[0])
    bids = [(p, s) for p, s in bids if s > 0]
    asks = [(p, s) for p, s in asks if s > 0]
    tick = _f(book.get("tick_size"))
    ts_ms = book.get("timestamp")
    h = book.get("hash")

    bb = bids[0][0] if bids else np.nan
    ba = asks[0][0] if asks else np.nan
    bs1 = bids[0][1] if bids else np.nan
    as1 = asks[0][1] if asks else np.nan
    two = bool(bids and asks)

    def depth(levels, ref, w, sign):
        return float(sum(s for p, s in levels if sign * (ref - p) <= w + 1e-12)) if levels else 0.0

    row = {
        "snapshot_id": sid,
        "fetch_ts": fetch_ts,
        "server_ts": pd.to_datetime(int(ts_ms), unit="ms", utc=True) if ts_ms else pd.NaT,
        "latency_ms": latency,
        "token_id": tok,
        "condition_id": book.get("market") or meta.get("conditionId"),
        "slug": meta.get("slug"),
        "outcome": meta.get("outcome"),
        "book_hash": h,
        "changed": last_hash.get(tok) != h,
        "tick_size": tick,
        "min_order_size": _f(book.get("min_order_size")),
        "neg_risk": bool(book.get("neg_risk")),
        "last_trade_price": _f(book.get("last_trade_price")),
        "best_bid": bb, "best_ask": ba, "bid_size_1": bs1, "ask_size_1": as1,
        "mid": (bb + ba) / 2 if two else np.nan,
        "logit_mid": 1 / (1 + math.exp(-(_logit(bb) + _logit(ba)) / 2)) if two else np.nan,
        "spread": ba - bb if two else np.nan,
        "spread_ticks": round((ba - bb) / tick) if two and tick > 0 else np.nan,
        "microprice": (bb * as1 + ba * bs1) / (bs1 + as1) if two else np.nan,
        "imbalance_1": bs1 / (bs1 + as1) if two else np.nan,
        "n_bid_levels": len(bids), "n_ask_levels": len(asks),
        "bid_depth_total": float(sum(s for _, s in bids)),
        "ask_depth_total": float(sum(s for _, s in asks)),
        "bid_notional_total": float(sum(p * s for p, s in bids)),
        "ask_notional_total": float(sum(p * s for p, s in asks)),
    }
    for c, w in (("1c", 0.01), ("5c", 0.05), ("10c", 0.10)):
        row[f"bid_depth_{c}"] = depth(bids, bb, w, +1) if bids else 0.0
        row[f"ask_depth_{c}"] = depth(asks, ba, w, -1) if asks else 0.0
    for c in ("5c", "10c"):
        tot = row[f"bid_depth_{c}"] + row[f"ask_depth_{c}"]
        row[f"imbalance_{c}"] = row[f"bid_depth_{c}"] / tot if tot > 0 else np.nan
    for n in (100, 1000):
        row[f"bid_vwap_{n}"] = _vwap(bids, n)
        row[f"ask_vwap_{n}"] = _vwap(asks, n)
    for side, lv in (("bid", bids), ("ask", asks)):
        for i in range(K):
            row[f"{side}_px_{i + 1}"] = lv[i][0] if i < len(lv) else np.nan
            row[f"{side}_sz_{i + 1}"] = lv[i][1] if i < len(lv) else np.nan

    # fair value
    row["fv"], row["fv_confidence"], row["gm_mu"] = np.nan, None, np.nan
    if _FV_MODEL is not None:
        try:
            ltp = row["last_trade_price"]
            res = _FV_MODEL.evaluate(FVBook.from_levels(
                bids, asks, tick_size=tick if tick > 0 else 0.01,
                last_trade=None if np.isnan(ltp) else ltp))
            row["fv"] = np.nan if res.fair_value is None else res.fair_value
            row["fv_confidence"] = res.confidence
            row["gm_mu"] = res.diagnostics.get("informed_mu (GM)", np.nan)
        except Exception:  # noqa: BLE001
            pass
    last_hash[tok] = h

    lv_rows = []
    for side, lv in (("bid", bids), ("ask", asks)):
        cum = 0.0
        for i, (p, s) in enumerate(lv):
            cum += s
            lv_rows.append({"snapshot_id": sid, "fetch_ts": fetch_ts, "token_id": tok,
                            "side": side, "level": i + 1, "price": p, "size": s,
                            "cum_size": cum, "notional": p * s})
    return row, lv_rows


# =============================================================================
# Storage
# =============================================================================
LEVEL_DTYPES = {"snapshot_id": "int64", "token_id": "string", "side": "category",
                "level": "int32", "price": "float64", "size": "float64",
                "cum_size": "float64", "notional": "float64"}


def _write_parts(df: pd.DataFrame, root: str, name: str) -> None:
    if df.empty:
        return
    df = df.copy()
    df["date"] = df["fetch_ts"].dt.strftime("%Y-%m-%d")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    for d, part in df.groupby("date"):
        folder = os.path.join(root, name, f"date={d}")
        os.makedirs(folder, exist_ok=True)
        part.drop(columns="date").to_parquet(
            os.path.join(folder, f"part-{stamp}.parquet"), index=False, compression="zstd")


def flush(root: str, snaps: list, levels: list) -> tuple[int, int]:
    ns, nl = len(snaps), len(levels)
    if snaps:
        s = pd.DataFrame(snaps)
        s["token_id"] = s["token_id"].astype("string")
        for c in ("condition_id", "slug", "outcome", "book_hash", "fv_confidence"):
            s[c] = s[c].astype("string")
        _write_parts(s, root, "snapshots")
    if levels:
        _write_parts(pd.DataFrame(levels).astype(LEVEL_DTYPES), root, "levels")
    snaps.clear()
    levels.clear()
    return ns, nl


def save_markets(root: str, rows: list[dict]) -> None:
    if not rows:
        return
    path = os.path.join(root, "markets.parquet")
    new = pd.DataFrame(rows)
    if os.path.exists(path):
        new = pd.concat([pd.read_parquet(path), new], ignore_index=True)
    new = new.drop_duplicates("token_id", keep="last")
    # mixed-type object columns -> string, so Parquet accepts them
    for c in new.columns:
        if new[c].dtype == object:
            new[c] = new[c].map(lambda v: None if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)).astype("string")
    os.makedirs(root, exist_ok=True)
    new.to_parquet(path, index=False)


def compact(root: str) -> None:
    """Merge each day's part files into one file (run while not recording)."""
    for name in ("snapshots", "levels"):
        base = os.path.join(root, name)
        if not os.path.isdir(base):
            continue
        for d in sorted(os.listdir(base)):
            folder = os.path.join(base, d)
            parts = [f for f in os.listdir(folder) if f.endswith(".parquet")]
            if len(parts) <= 1:
                continue
            df = pd.concat([pd.read_parquet(os.path.join(folder, f)) for f in parts], ignore_index=True)
            df = df.sort_values(["snapshot_id", "token_id"] + (["side", "level"] if name == "levels" else []))
            tmp = os.path.join(folder, "compact.tmp")
            df.to_parquet(tmp, index=False, compression="zstd")
            for f in parts:
                os.remove(os.path.join(folder, f))
            os.replace(tmp, os.path.join(folder, "part-compacted.parquet"))
            log(f"compacted {name}/{d}: {len(parts)} files -> 1 ({len(df):,} rows)")


def load(root: str = "data", date: str | None = None):
    """Load (snapshots, levels, markets) DataFrames. `date`='YYYY-MM-DD' to load one day."""
    filt = [("date", "=", date)] if date else None
    rd = lambda n: (pd.read_parquet(os.path.join(root, n), filters=filt)
                    if os.path.isdir(os.path.join(root, n)) else pd.DataFrame())
    mp = os.path.join(root, "markets.parquet")
    return rd("snapshots"), rd("levels"), (pd.read_parquet(mp) if os.path.exists(mp) else pd.DataFrame())


# =============================================================================
# Main loop
# =============================================================================
def next_snapshot_id(root: str) -> int:
    """Continue snapshot_id numbering across runs."""
    base = os.path.join(root, "snapshots")
    if not os.path.isdir(base):
        return 0
    try:
        s = pd.read_parquet(base, columns=["snapshot_id"])
        return int(s["snapshot_id"].max()) + 1 if len(s) else 0
    except Exception:  # noqa: BLE001
        return int(time.time())


def run(args) -> None:
    root = args.out
    stop = {"flag": False}
    signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))

    def refresh_targets():
        mkts = discover(args)
        rows = [r for m in mkts if _live(m) or args.slug or args.token or args.event
                for r in market_rows(m)]
        save_markets(root, rows)
        meta = {r["token_id"]: r for r in rows}
        log(f"tracking {len(mkts)} markets / {len(meta)} tokens")
        return meta

    meta = refresh_targets()
    if not meta:
        log("nothing to record")
        return
    last_meta_refresh = time.time()
    sid = next_snapshot_id(root)
    snaps, levels, last_hash = [], [], {}
    t_start = last_flush = time.time()
    tot_s = tot_l = 0

    while not stop["flag"]:
        t0 = time.time()
        if (args.top or args.all) and t0 - last_meta_refresh > args.refresh_markets:
            try:
                meta = refresh_targets() or meta
            except Exception as e:  # noqa: BLE001
                log(f"! market refresh failed: {e}")
            last_meta_refresh = t0

        books = fetch_books(list(meta))
        n_changed = 0
        for b, fetch_ts, lat in books:
            r, lv = process_book(b, sid, fetch_ts, lat, meta.get(str(b.get("asset_id")), {}),
                                 args.levels, last_hash)
            snaps.append(r)
            levels.extend(lv)
            n_changed += r["changed"]
        log(f"snap {sid}: {len(books)}/{len(meta)} books, {n_changed} changed, "
            f"round {time.time() - t0:.1f}s, buffer {len(snaps):,} snaps / {len(levels):,} levels")
        sid += 1

        if args.once or (args.duration and time.time() - t_start >= args.duration):
            break
        if time.time() - last_flush >= args.flush or len(levels) >= args.max_buffer:
            a, b_ = flush(root, snaps, levels)
            tot_s, tot_l = tot_s + a, tot_l + b_
            last_flush = time.time()
            log(f"flushed {a:,} snapshot rows, {b_:,} level rows -> {root}")
        # sleep in small steps so Ctrl+C is responsive
        while not stop["flag"] and time.time() - t0 < args.interval:
            time.sleep(min(0.2, args.interval))

    a, b_ = flush(root, snaps, levels)
    tot_s, tot_l = tot_s + a, tot_l + b_
    log(f"done. wrote {tot_s:,} snapshot rows and {tot_l:,} level rows to {os.path.abspath(root)}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Record Polymarket order books to Parquet")
    ap.add_argument("--slug", nargs="+", help="market slug(s)")
    ap.add_argument("--event", nargs="+", help="event slug(s): records all their markets")
    ap.add_argument("--token", nargs="+", help="CLOB token id(s)")
    ap.add_argument("--top", type=int, help="top N live markets by 24h volume")
    ap.add_argument("--all", action="store_true",
                    help="every live market with an order book (~140k markets; combine with filters)")
    ap.add_argument("--min-volume", type=float, default=0,
                    help="for --top/--all: only markets with 24h volume >= this ($)")
    ap.add_argument("--min-liquidity", type=float, default=0,
                    help="for --top/--all: only markets with liquidity >= this ($)")
    ap.add_argument("--max-buffer", type=int, default=2_000_000,
                    help="flush early once this many level rows are buffered (default 2M)")
    ap.add_argument("--interval", type=float, default=5.0, help="seconds between fetches (default 5)")
    ap.add_argument("--duration", type=float, default=0, help="stop after N seconds (default: run forever)")
    ap.add_argument("--flush", type=float, default=300, help="seconds between disk writes (default 300)")
    ap.add_argument("--levels", type=int, default=20, help="price levels per side as wide columns (default 20)")
    ap.add_argument("--refresh-markets", type=float, default=900,
                    help="re-discover markets every N s for --top/--all (default 900)")
    ap.add_argument("--out", default="data", help="output directory (default ./data)")
    ap.add_argument("--once", action="store_true", help="fetch one snapshot and exit")
    ap.add_argument("--compact", action="store_true", help="merge part files per day and exit")
    args = ap.parse_args(argv)

    if args.compact:
        return compact(args.out)
    if not (args.slug or args.event or args.token or args.top or args.all):
        ap.error("choose targets: --slug, --event, --token, --top N or --all")
    if _FV_MODEL is None:
        log("FairMV not found - fair value columns will be NaN")
    run(args)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
