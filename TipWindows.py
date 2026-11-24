"""
TipWindows.py - cut Polymarket order-book / price data into "tipping point" windows.

A window is PRE + TIP + POST consecutive grid points (default 100 + 10 + 10 = 120,
10 s grid) holding F = 50 features per point, so the output tensor is
    X.shape == (n_tips, 120, 50)

------------------------------------------------------------------------------
Tip definition (the maths)
------------------------------------------------------------------------------
Probabilities near 0/1 move less in absolute terms than near 0.5 (a Bernoulli
p has std sqrt(p(1-p))). The variance-stabilising (arcsine) transform

        y(p) = 2 * arcsin( sqrt(p) )          dy/dp = 1 / sqrt(p (1-p))

puts every move on one scale.  Examples:
        50% -> 60%   dy = 0.201
        98% -> 95%   dy = 0.167
        96% -> 99%   dy = 0.202
so one threshold THETA (default 0.15) catches all of them.

Let ~y be y of the price after a rolling median of 3 points (kills single-print
outliers).  A tip segment starts at grid index s and lasts L = TIP points:

        D(s) = ~y[s+L-1] - ~y[s-1]                 (move across the segment)

s is a tip when
   1. |D(s)| >= THETA
   2. |D(s)| is a local max (non-max suppression, peaks >= L+POST apart)
   3. persistence: mean(~y[s+L : s+L+POST]) - ~y[s-1] has the sign of D(s)
      and magnitude >= RHO * THETA        (RHO default 0.5: the move sticks)
   4. TIP and POST points all lie inside the market's observed data.
PRE points before the market's first observation are NaN-padded (valid = 0).

------------------------------------------------------------------------------
Data source (data/external/hf_crypto_5m15m)
------------------------------------------------------------------------------
* trades/      -> price. Trades in both outcome tokens are merged onto one
                  canonical token (other token's price -> 1 - price), bucketed
                  to the grid: VWAP, buy/sell volume, count; price forward-filled.
* orderbooks/  -> 10 bid + 10 ask levels (as-of joined to the grid).
                  NOTE: this dataset stores the 10 *farthest* levels of the
                  book (bids 0.01-0.10, asks 0.90-0.99), not top-of-book, so its
                  best_bid / best_ask / mid are NOT the traded price. They are
                  kept as features (deep-tail liquidity) but tips use trades.
* markets/     -> end_time (for secs_to_end).

------------------------------------------------------------------------------
Features (F = 50), in order - also saved as FEATURES in the parquet metadata
------------------------------------------------------------------------------
 0 valid        1 if the point is inside the market's data, else 0 (padding)
 1 price        trade VWAP of canonical token, forward-filled
 2 y            2*asin(sqrt(price))
 3 buy_vol      shares bought of canonical token in the bin (other token's
                sells count as buys)
 4 sell_vol
 5 n_trades
 6 book_best_bid  7 book_best_ask  8 book_spread  9 secs_to_end
10..19  bid_px_1..10  (level 1 = highest bid)
20..29  bid_sz_1..10
30..39  ask_px_1..10  (level 1 = lowest ask)
40..49  ask_sz_1..10

------------------------------------------------------------------------------
Output
------------------------------------------------------------------------------
<out>.parquet : one row per tip: metadata columns + `window`
                (FixedSizeList[PRE+TIP+POST] of FixedSizeList[50] float32).
<out>.npy     : the same tensor as a plain float32 array (n, T, F).
Load with   X, meta, feats = load_tips("data/ml/tips.parquet")

Usage
    py TipWindows.py                          # defaults
    py TipWindows.py --theta 0.2 --min-pre 50 --exclude-final-secs 60
    py TipWindows.py --max-tips 1000 --out data/ml/tips_1k
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "data", "external", "hf_crypto_5m15m")

N_LEVELS = 10
FEATURES = (
    ["valid", "price", "y", "buy_vol", "sell_vol", "n_trades",
     "book_best_bid", "book_best_ask", "book_spread", "secs_to_end"]
    + [f"bid_px_{i}" for i in range(1, N_LEVELS + 1)]
    + [f"bid_sz_{i}" for i in range(1, N_LEVELS + 1)]
    + [f"ask_px_{i}" for i in range(1, N_LEVELS + 1)]
    + [f"ask_sz_{i}" for i in range(1, N_LEVELS + 1)]
)
F = len(FEATURES)  # 50
FI = {name: i for i, name in enumerate(FEATURES)}


# ---------------------------------------------------------------------------
# Tip maths
# ---------------------------------------------------------------------------
def arcsine(p: np.ndarray) -> np.ndarray:
    """Variance-stabilising transform for a probability: 2*asin(sqrt(p))."""
    return 2.0 * np.arcsin(np.sqrt(np.clip(p, 0.0, 1.0)))


def detect_tips(p: np.ndarray, tip: int, post: int, theta: float, rho: float,
                first_ok: int = 1) -> list[tuple[int, float]]:
    """Return [(s, D)] tip-segment starts on a price series p (NaN = no data).

    s is the first index of the TIP segment; p[s-1] is the pre-tip level.
    """
    n = len(p)
    if n < tip + post + 1:
        return []
    ys = pd.Series(arcsine(p)).rolling(3, center=True, min_periods=1).median().to_numpy()

    last = n - tip - post            # s must satisfy s + tip + post <= n
    s_idx = np.arange(max(1, first_ok), last + 1)
    if len(s_idx) == 0:
        return []
    base = ys[s_idx - 1]
    D = ys[s_idx + tip - 1] - base
    csum = np.concatenate([[0.0], np.cumsum(np.nan_to_num(ys))])
    post_mean = (csum[s_idx + tip + post] - csum[s_idx + tip]) / post
    P = post_mean - base

    ok = (np.abs(D) >= theta) & (np.sign(P) == np.sign(D)) & (np.abs(P) >= rho * theta)
    ok &= np.isfinite(D) & np.isfinite(P)
    cand = np.flatnonzero(ok)
    if len(cand) == 0:
        return []

    # greedy non-maximum suppression: biggest |D| first, keep peaks >= sep apart
    sep = tip + post
    order = cand[np.argsort(-np.abs(D[cand]))]
    taken: list[int] = []
    for c in order:
        if all(abs(c - t) >= sep for t in taken):
            taken.append(c)
    return sorted((int(s_idx[c]), float(D[c])) for c in taken)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_book_index(files: list[str]) -> pd.DataFrame:
    """Light book columns for all days + (file, row) pointer for later level parsing."""
    cols = ["timestamp", "condition_id", "token_id", "best_bid", "best_ask", "spread"]
    parts = []
    for fi, f in enumerate(files):
        d = pd.read_parquet(f, columns=cols)
        d["file"] = fi
        d["row"] = np.arange(len(d), dtype=np.int64)
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def canonical_tokens(book: pd.DataFrame) -> pd.Series:
    """One token per condition (first listed in the book feed) -> avoids mirrored duplicate tips."""
    return book.drop_duplicates("condition_id").set_index("condition_id")["token_id"]


def load_trade_bins(files: list[str], canon: pd.Series, grid_s: int) -> pd.DataFrame:
    """Aggregate all trades onto (condition_id, bin) in canonical-token terms."""
    canon_set = pd.DataFrame({"condition_id": canon.index, "canon": canon.values})
    out = []
    for f in files:
        t = pd.read_parquet(f, columns=["timestamp", "condition_id", "token_id", "side", "price", "size"])
        t = t.merge(canon_set, on="condition_id", how="inner")
        same = (t["token_id"] == t["canon"]).to_numpy()
        price = np.where(same, t["price"], 1.0 - t["price"])
        buy = (t["side"] == "BUY").to_numpy() == same          # buy of other token = sell of canon
        size = t["size"].to_numpy()
        g = pd.DataFrame({
            "condition_id": t["condition_id"].to_numpy(),
            "bin": t["timestamp"].dt.floor(f"{grid_s}s").to_numpy(),
            "pv": price * size, "size": size,
            "buy_vol": np.where(buy, size, 0.0), "sell_vol": np.where(buy, 0.0, size),
            "n": 1,
        }).groupby(["condition_id", "bin"], sort=False).sum()
        out.append(g)
        print(f"  trades {os.path.basename(f)}: {len(t):,} -> {len(g):,} bins", flush=True)
    agg = pd.concat(out).groupby(level=[0, 1]).sum()          # merge bins split across files
    agg["vwap"] = agg["pv"] / agg["size"].where(agg["size"] > 0)
    return agg.drop(columns=["pv", "size"]).reset_index()


def parse_levels(js: str, side: str) -> tuple[np.ndarray, np.ndarray]:
    px = np.full(N_LEVELS, np.nan, np.float32)
    sz = np.full(N_LEVELS, np.nan, np.float32)
    if not js:
        return px, sz
    lv = json.loads(js)
    lv.sort(key=lambda x: -float(x["price"]) if side == "bid" else float(x["price"]))  # best first
    for i, x in enumerate(lv[:N_LEVELS]):
        px[i] = float(x["price"])
        sz[i] = float(x["size"])
    return px, sz


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def build(args) -> None:
    t0 = time.time()
    ob_files = sorted(glob.glob(os.path.join(SRC, "orderbooks", "*.parquet")))
    tr_files = sorted(glob.glob(os.path.join(SRC, "trades", "*.parquet")))
    T = args.pre + args.tip + args.post
    g = args.grid_s

    print("1/4 loading book index ...", flush=True)
    book = load_book_index(ob_files)
    canon = canonical_tokens(book)
    book = book[book["condition_id"].map(canon).eq(book["token_id"])]
    book = book.sort_values(["condition_id", "timestamp"]).reset_index(drop=True)

    mk = pd.read_parquet(os.path.join(SRC, "markets", "all.parquet"),
                         columns=["condition_id", "market_id", "asset", "end_time"]).drop_duplicates("condition_id")
    mk = mk.set_index("condition_id")

    print("2/4 aggregating trades ...", flush=True)
    tb = load_trade_bins(tr_files, canon, g)
    tb_groups = dict(tuple(tb.groupby("condition_id", sort=False)))

    print("3/4 detecting tips ...", flush=True)
    tips = []   # dicts with metadata + grid + book rows
    for cid, bk in book.groupby("condition_id", sort=False):
        tr = tb_groups.get(cid)
        if tr is None or len(bk) < 2:
            continue
        start = bk["timestamp"].iloc[0].floor(f"{g}s")
        stop = bk["timestamp"].iloc[-1].floor(f"{g}s")
        grid = pd.date_range(start, stop, freq=f"{g}s")
        if len(grid) < args.tip + args.post + 1:
            continue
        tr = tr.set_index("bin").reindex(grid)
        price = tr["vwap"].ffill().to_numpy()
        first_ok = int(np.argmax(np.isfinite(price))) + 1 if np.isfinite(price).any() else len(grid)
        first_ok = max(first_ok, args.min_pre)

        end_time = mk["end_time"].get(cid, pd.NaT) if cid in mk.index else pd.NaT
        if pd.isna(end_time):
            end_time = bk["timestamp"].iloc[-1]
        secs_to_end = (end_time - grid).total_seconds().to_numpy()

        for s, D in detect_tips(price, args.tip, args.post, args.theta, args.rho, first_ok):
            if secs_to_end[s] < args.exclude_final_secs:
                continue
            tips.append(dict(cid=cid, s=s, D=D, grid=grid, price=price, tr=tr,
                             secs_to_end=secs_to_end, bk=bk))
    print(f"   found {len(tips):,} tips", flush=True)
    if args.max_tips and len(tips) > args.max_tips:
        rng = np.random.default_rng(0)
        tips = [tips[i] for i in sorted(rng.choice(len(tips), args.max_tips, replace=False))]
        print(f"   sampled {len(tips):,}", flush=True)
    if not tips:
        print("no tips - try a lower --theta"); return

    print("4/4 assembling windows ...", flush=True)
    X = np.full((len(tips), T, F), np.nan, np.float32)
    X[:, :, FI["valid"]] = 0.0
    need = []   # (tip_i, t_in_window, file, row)
    meta = []
    for i, tp in enumerate(tips):
        s, grid, price, tr, bk = tp["s"], tp["grid"], tp["price"], tp["tr"], tp["bk"]
        lo = s - args.pre
        src = np.arange(lo, s + args.tip + args.post)
        inside = (src >= 0) & (src < len(grid))
        w = np.flatnonzero(inside)          # window positions with data
        si = src[inside]                    # grid indices
        X[i, w, FI["valid"]] = 1.0
        X[i, w, FI["price"]] = price[si]
        X[i, w, FI["y"]] = arcsine(price[si])
        X[i, w, FI["buy_vol"]] = np.nan_to_num(tr["buy_vol"].to_numpy()[si])
        X[i, w, FI["sell_vol"]] = np.nan_to_num(tr["sell_vol"].to_numpy()[si])
        X[i, w, FI["n_trades"]] = np.nan_to_num(tr["n"].to_numpy()[si])
        X[i, w, FI["secs_to_end"]] = tp["secs_to_end"][si]

        # as-of (backward) book snapshot for each grid point; bins end at grid + g
        bts = bk["timestamp"].to_numpy()
        pos = np.searchsorted(bts, (grid[si] + pd.Timedelta(seconds=g)).to_numpy(), side="left") - 1
        has = pos >= 0
        stale = np.zeros_like(has)
        stale[has] = (grid[si][has] + pd.Timedelta(seconds=g) - bts[pos[has]]) > pd.Timedelta(seconds=args.book_tol_s)
        has &= ~stale
        wh, ph = w[has], pos[has]
        X[i, wh, FI["book_best_bid"]] = bk["best_bid"].to_numpy()[ph]
        X[i, wh, FI["book_best_ask"]] = bk["best_ask"].to_numpy()[ph]
        X[i, wh, FI["book_spread"]] = bk["spread"].to_numpy()[ph]
        fcol, rcol = bk["file"].to_numpy()[ph], bk["row"].to_numpy()[ph]
        need.extend(zip(np.full(len(wh), i), wh, fcol, rcol))

        p_before = price[s - 1]
        p_after = price[min(s + args.tip - 1, len(price) - 1)]
        m = mk.loc[tp["cid"]] if tp["cid"] in mk.index else None
        meta.append(dict(
            tip_id=i, condition_id=tp["cid"], token_id=bk["token_id"].iloc[0],
            market_id=None if m is None else m["market_id"],
            asset=None if m is None else m["asset"],
            tip_start_ts=grid[s], direction=int(np.sign(tp["D"])), delta_y=tp["D"],
            p_before=float(p_before), p_after=float(p_after),
            secs_to_end=float(tp["secs_to_end"][s]),
            n_pre_valid=int(((src[:args.pre] >= 0)).sum()),
        ))

    need = pd.DataFrame(need, columns=["i", "w", "file", "row"])
    for fi, grp in need.groupby("file"):
        lv = pd.read_parquet(ob_files[fi], columns=["bid_levels", "ask_levels"])
        rows = grp["row"].to_numpy()
        bids = lv["bid_levels"].to_numpy()[rows]
        asks = lv["ask_levels"].to_numpy()[rows]
        for (ti, wp), bj, aj in zip(grp[["i", "w"]].to_numpy(), bids, asks):
            bp, bs = parse_levels(bj, "bid")
            ap, as_ = parse_levels(aj, "ask")
            X[ti, wp, FI["bid_px_1"]:FI["bid_px_1"] + N_LEVELS] = bp
            X[ti, wp, FI["bid_sz_1"]:FI["bid_sz_1"] + N_LEVELS] = bs
            X[ti, wp, FI["ask_px_1"]:FI["ask_px_1"] + N_LEVELS] = ap
            X[ti, wp, FI["ask_sz_1"]:FI["ask_sz_1"] + N_LEVELS] = as_
        print(f"  levels {os.path.basename(ob_files[fi])}: {len(grp):,} rows", flush=True)

    save(args.out, X, pd.DataFrame(meta), args)
    print(f"done: X{X.shape} in {time.time() - t0:.0f}s -> {args.out}.parquet / .npy")


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------
def save(out: str, X: np.ndarray, meta: pd.DataFrame, args) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    n, T, Fn = X.shape
    inner = pa.FixedSizeListArray.from_arrays(pa.array(X.reshape(-1), pa.float32()), Fn)
    window = pa.FixedSizeListArray.from_arrays(inner, T)
    table = pa.Table.from_pandas(meta, preserve_index=False).append_column("window", window)
    info = dict(FEATURES=FEATURES, shape=[n, T, Fn], pre=args.pre, tip=args.tip, post=args.post,
                grid_s=args.grid_s, theta=args.theta, rho=args.rho,
                tip_function="y=2*asin(sqrt(p)); |y[s+tip-1]-y[s-1]|>=theta, persists over post")
    table = table.replace_schema_metadata({**(table.schema.metadata or {}),
                                           b"tipwindows": json.dumps(info).encode()})
    pq.write_table(table, out + ".parquet", compression="zstd")
    np.save(out + ".npy", X)


def load_tips(path: str):
    """-> (X float32 [n, T, F], meta DataFrame, feature name list)."""
    table = pq.read_table(path)
    info = json.loads(table.schema.metadata[b"tipwindows"])
    n, T, Fn = info["shape"]
    flat = table.column("window").combine_chunks().flatten().flatten().to_numpy(zero_copy_only=False)
    X = flat.reshape(n, T, Fn)
    meta = table.drop(["window"]).to_pandas()
    return X, meta, info["FEATURES"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pre", type=int, default=100, help="points before the tip")
    ap.add_argument("--tip", type=int, default=10, help="points in the tip segment")
    ap.add_argument("--post", type=int, default=10, help="points after the tip")
    ap.add_argument("--grid-s", type=int, default=10, help="grid spacing in seconds (book is ~10 s)")
    ap.add_argument("--theta", type=float, default=0.15, help="min |delta arcsine| across the tip")
    ap.add_argument("--rho", type=float, default=0.5, help="post-window persistence as fraction of theta")
    ap.add_argument("--min-pre", type=int, default=0, help="require this many real (unpadded) pre points")
    ap.add_argument("--exclude-final-secs", type=float, default=0,
                    help="drop tips starting within N s of market end (resolution snaps)")
    ap.add_argument("--book-tol-s", type=float, default=30, help="max staleness of as-of book snapshot")
    ap.add_argument("--max-tips", type=int, default=0, help="random subsample (0 = keep all)")
    ap.add_argument("--out", default=os.path.join(HERE, "data", "ml", "tips"), help="output path (no extension)")
    build(ap.parse_args())


if __name__ == "__main__":
    main()
