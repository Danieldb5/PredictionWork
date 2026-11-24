"""
FetchHistory.py - bulk download of Polymarket history from the public APIs.

FetchData.py records books going forward; this pulls what already exists:

  markets   every live market + the N biggest resolved ones (Gamma), one row per token.
            Resolved rows carry outcome_price = 1/0 -> calibration labels.
  books     one full order-book snapshot of every live token (CLOB POST /books).
  prices    price history (CLOB /prices-history) for the YES token of every market:
              hourly   interval=max, fidelity 60 min (720 for old markets that refuse 60)
              minute   last 14 days at 1-minute fidelity for a volume-stratified sample
  trades    up to --trades-n recent trades per market (data-api v2), live and resolved.

Output (default data/history/), all parquet, resumable (done-lists per stage):
  markets_live.parquet  markets_closed.parquet
  books/summary_<ts>.parquet   books/levels_<ts>.parquet
  prices_hourly/part-*.parquet  (token_id, ts, p)
  prices_minute/part-*.parquet  (token_id, ts, p)
  trades/part-*.parquet         (raw data-api columns + condition_id)

Usage:
  py FetchHistory.py all                      # everything, defaults below
  py FetchHistory.py markets|books|prices|trades
  options: --out data/history --closed 5000 --minute 2000 --trades-n 1000
           --workers 8 --limit N (cap live markets, for a quick test)

Loading:
  import pandas as pd
  ph = pd.read_parquet("data/history/prices_hourly")
  mk = pd.read_parquet("data/history/markets_closed.parquet")
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from FetchData import CLOB_URL, GAMMA_URL, GAMMA_PAGE, http, log, market_rows  # noqa: E402

TRADES_URL = "https://data-api.polymarket.com/v2/trades"
PART_SIZE = 500          # tokens / markets per part file


# =============================================================================
# helpers
# =============================================================================
def _f(x, d=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return d


class Parts:
    """Append-only part files + a done-list so a stage can be resumed after Ctrl+C."""

    def __init__(self, root: str, name: str):
        self.dir = os.path.join(root, name)
        os.makedirs(self.dir, exist_ok=True)
        self.done_path = os.path.join(self.dir, "_done.txt")
        self.done = set()
        if os.path.exists(self.done_path):
            with open(self.done_path) as f:
                self.done = {ln.strip() for ln in f if ln.strip()}
        self.buf, self.keys, self.lock = [], [], threading.Lock()
        self.n_rows = 0

    def add(self, key: str, rows: list[dict]):
        with self.lock:
            self.buf.extend(rows)
            self.keys.append(key)
            if len(self.keys) >= PART_SIZE:
                self._flush()

    def _flush(self):
        if not self.keys:
            return
        if self.buf:
            df = pd.DataFrame(self.buf)
            for c in df.columns:                      # mixed-type object columns break parquet
                if df[c].dtype == object:
                    df[c] = df[c].map(lambda v: v if v is None or isinstance(v, str)
                                      else json.dumps(v) if isinstance(v, (list, dict)) else str(v))
            fn = os.path.join(self.dir, f"part-{int(time.time() * 1000)}.parquet")
            df.to_parquet(fn, index=False)
            self.n_rows += len(df)
        with open(self.done_path, "a") as f:
            f.write("\n".join(self.keys) + "\n")
        self.done.update(self.keys)
        self.buf, self.keys = [], []

    def close(self):
        with self.lock:
            self._flush()


def run_pool(items, fn, workers, label):
    t0, n = time.time(), 0
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(fn, it) for it in items]
        for fu in as_completed(futs):
            n += 1
            try:
                fu.result()
            except Exception as e:  # noqa: BLE001
                log(f"  ! {label}: {e}")
            if n % 500 == 0 or n == len(items):
                rate = n / max(time.time() - t0, 1e-9)
                log(f"  {label}: {n:,}/{len(items):,}  ({rate:.1f}/s, eta {(len(items) - n) / max(rate, 1e-9) / 60:.1f} min)")


# =============================================================================
# markets
# =============================================================================
def scan_gamma(params: dict, limit: int | None) -> list[dict]:
    out, cursor, pages = [], None, 0
    while True:
        p = dict(params, limit=GAMMA_PAGE)
        if cursor:
            p["after_cursor"] = cursor
        d = http("GET", f"{GAMMA_URL}/markets/keyset", params=p)
        page = d.get("markets", [])
        out.extend(m for m in page if m.get("clobTokenIds"))
        pages += 1
        if pages % 50 == 0:
            log(f"  gamma {params.get('closed') == 'true' and 'closed' or 'live'}: {len(out):,} markets")
        if limit and len(out) >= limit:
            return out[:limit]
        nxt = d.get("next_cursor")
        if not page or not nxt or nxt == cursor:
            return out
        cursor = nxt
        time.sleep(0.05)


def stage_markets(a):
    log("markets: live")
    live = scan_gamma({"active": "true", "closed": "false", "order": "volume24hr", "ascending": "false"}, a.limit)
    live = [m for m in live if m.get("enableOrderBook") and m.get("acceptingOrders")]
    df = pd.DataFrame([r for m in live for r in market_rows(m)])
    _save(df, os.path.join(a.out, "markets_live.parquet"))
    log(f"  {len(live):,} live markets, {len(df):,} tokens")
    if a.closed:
        log(f"markets: {a.closed:,} biggest resolved")
        closed = scan_gamma({"closed": "true", "order": "volumeNum", "ascending": "false"}, a.closed)
        dc = pd.DataFrame([r for m in closed for r in market_rows(m)])
        _save(dc, os.path.join(a.out, "markets_closed.parquet"))
        log(f"  {len(closed):,} resolved markets, {len(dc):,} tokens")


def _save(df, path):
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].map(lambda v: v if v is None or isinstance(v, str) else str(v))
    df.to_parquet(path, index=False)


def load_markets(a, which=("live", "closed")) -> pd.DataFrame:
    fs = [os.path.join(a.out, f"markets_{w}.parquet") for w in which]
    fs = [f for f in fs if os.path.exists(f)]
    if not fs:
        raise SystemExit("run `py FetchHistory.py markets` first")
    return pd.concat([pd.read_parquet(f).assign(_set=os.path.basename(f)[8:-8]) for f in fs], ignore_index=True)


# =============================================================================
# books
# =============================================================================
def stage_books(a):
    mk = load_markets(a, ("live",))
    toks = mk["token_id"].astype(str).unique().tolist()
    log(f"books: {len(toks):,} tokens")
    ts = pd.Timestamp.now(tz="UTC")
    summ, levels, lock = [], [], threading.Lock()

    def one(chunk):
        res = http("POST", f"{CLOB_URL}/books", json=[{"token_id": t} for t in chunk]) or []
        s, l = [], []
        for b in res:
            tok = str(b.get("asset_id"))
            bids = sorted(((_f(x["price"]), _f(x["size"])) for x in b.get("bids", [])), reverse=True)
            asks = sorted((_f(x["price"]), _f(x["size"])) for x in b.get("asks", []))
            s.append({"token_id": tok, "market": b.get("market"), "ts": ts,
                      "book_ts": b.get("timestamp"), "hash": b.get("hash"),
                      "best_bid": bids[0][0] if bids else None, "best_ask": asks[0][0] if asks else None,
                      "n_bids": len(bids), "n_asks": len(asks),
                      "bid_shares": sum(q for _, q in bids), "ask_shares": sum(q for _, q in asks),
                      "tick_size": _f(b.get("tick_size"), None), "min_order_size": _f(b.get("min_order_size"), None),
                      "neg_risk": b.get("neg_risk"), "last_trade_price": _f(b.get("last_trade_price"), None)})
            for side, lv in (("bid", bids), ("ask", asks)):
                for i, (p, q) in enumerate(lv):
                    l.append({"token_id": tok, "side": side, "level": i, "price": p, "size": q})
        with lock:
            summ.extend(s)
            levels.extend(l)

    chunks = [toks[i:i + 100] for i in range(0, len(toks), 100)]
    run_pool(chunks, one, min(a.workers, 4), "books")
    d = os.path.join(a.out, "books")
    os.makedirs(d, exist_ok=True)
    tag = ts.strftime("%Y%m%dT%H%M%SZ")
    sm = pd.DataFrame(summ)
    sm["spread"] = sm["best_ask"] - sm["best_bid"]
    sm.to_parquet(os.path.join(d, f"summary_{tag}.parquet"), index=False)
    pd.DataFrame(levels).assign(ts=ts).to_parquet(os.path.join(d, f"levels_{tag}.parquet"), index=False)
    log(f"  {len(sm):,} books, {len(levels):,} levels -> books/*_{tag}.parquet")


# =============================================================================
# prices
# =============================================================================
def _history(params) -> list:
    d = http("GET", f"{CLOB_URL}/prices-history", params=params)
    return (d or {}).get("history", []) if isinstance(d, dict) else []


def yes_tokens(mk: pd.DataFrame) -> pd.DataFrame:
    return mk[mk["outcome_index"] == 0].drop_duplicates("token_id")


def stage_prices(a):
    mk = yes_tokens(load_markets(a))
    parts = Parts(a.out, "prices_hourly")
    todo = [t for t in mk["token_id"].astype(str) if t not in parts.done]
    log(f"prices hourly: {len(todo):,} tokens to do ({len(parts.done):,} done)")

    def hourly(tok):
        fid = 60
        h = _history({"market": tok, "interval": "max", "fidelity": 60})
        if not h:                                  # old/resolved markets only answer coarse fidelity
            fid = 720
            h = _history({"market": tok, "interval": "max", "fidelity": 720})
        parts.add(tok, [{"token_id": tok, "ts": int(x["t"]), "p": float(x["p"]), "fidelity_min": fid} for x in h])

    run_pool(todo, hourly, a.workers, "prices hourly")
    parts.close()

    if not a.minute:
        return
    live = mk[mk["_set"] == "live"].copy()
    live["v24"] = live["volume24hr"].map(_f)
    live = live[live["v24"] > 0].sort_values("v24", ascending=False)
    step = max(len(live) // a.minute, 1)            # stratified across the volume distribution
    sample = live.iloc[::step].head(a.minute)["token_id"].astype(str).tolist()
    parts = Parts(a.out, "prices_minute")
    todo = [t for t in sample if t not in parts.done]
    end = int(time.time())
    log(f"prices minute (14d @1m): {len(todo):,} tokens, stratified every {step}th by 24h volume")

    def minute(tok):
        h = _history({"market": tok, "startTs": end - 14 * 86400, "endTs": end, "fidelity": 1})
        parts.add(tok, [{"token_id": tok, "ts": int(x["t"]), "p": float(x["p"])} for x in h])

    run_pool(todo, minute, a.workers, "prices minute")
    parts.close()


# =============================================================================
# trades
# =============================================================================
def stage_trades(a):
    mk = yes_tokens(load_markets(a))
    conds = mk.drop_duplicates("conditionId")[["conditionId", "_set"]]
    parts = Parts(a.out, "trades")
    todo = [c for c in conds["conditionId"].astype(str) if c not in parts.done]
    log(f"trades: {len(todo):,} markets to do (<= {a.trades_n} each)")

    def one(cond):
        rows, cursor = [], None
        while len(rows) < a.trades_n:
            p = {"condition": cond, "limit": min(500, a.trades_n - len(rows))}
            if cursor:
                p["cursor"] = cursor
            d = http("GET", TRADES_URL, params=p)
            data = d.get("data", []) if isinstance(d, dict) else (d or [])
            rows.extend(data)
            pag = (d.get("pagination") or {}) if isinstance(d, dict) else {}
            cursor = pag.get("next_cursor")
            if not data or not cursor or not pag.get("has_more"):
                break
        drop = ("title", "icon", "name", "pseudonym", "bio", "profile_image", "profile_image_optimized")
        for r in rows:
            r["condition_id"] = cond
            for k in drop:
                r.pop(k, None)
        parts.add(cond, rows)

    run_pool(todo, one, a.workers, "trades")
    parts.close()


# =============================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["all", "markets", "books", "prices", "trades"])
    ap.add_argument("--out", default=os.path.join("data", "history"))
    ap.add_argument("--closed", type=int, default=5000, help="biggest resolved markets to include")
    ap.add_argument("--minute", type=int, default=2000, help="live tokens to fetch 14d of 1-min prices for")
    ap.add_argument("--trades-n", type=int, default=1000, help="max trades per market")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="cap live markets (testing)")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    stages = ["markets", "books", "prices", "trades"] if a.stage == "all" else [a.stage]
    for s in stages:
        t0 = time.time()
        globals()[f"stage_{s}"](a)
        log(f"== {s} done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
