"""
BuildML.py - turn the raw downloads into clean, typed, ML-ready parquet under data/ml/.

Inputs (raw, never modified):
  data/history/                 FetchHistory.py output (Polymarket APIs)
  data/external/hf_crypto_5m15m HuggingFace BrockMisner/polymarket-crypto-5m-15m (MIT)

Outputs: see data/README.md. Rerun any time (e.g. after FetchHistory finishes or is
resumed); every output is rebuilt from the raw files.

  py BuildML.py            # everything
  py BuildML.py markets prices trades books resolved crypto   # any subset
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import requests

RAW = os.path.join("data", "history")
HF = os.path.join("data", "external", "hf_crypto_5m15m")
OUT = os.path.join("data", "ml")
GAMMA = "https://gamma-api.polymarket.com/markets"


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def fresh_dir(path):
    """Empty a folder (files only: OneDrive locks folders it is syncing)."""
    os.makedirs(path, exist_ok=True)
    for f in glob.glob(os.path.join(path, "**", "*"), recursive=True):
        if os.path.isfile(f):
            os.remove(f)


def utc(s, unit=None):
    if unit:
        return pd.to_datetime(s, unit=unit, utc=True, errors="coerce")
    return pd.to_datetime(s, utc=True, errors="coerce", format="mixed")


def num(s, dtype="float64"):
    return pd.to_numeric(s, errors="coerce").astype(dtype)


def boolcol(s):
    return s.map(lambda v: None if v is None or (isinstance(v, float) and np.isnan(v))
                 else str(v).lower() in ("true", "1")).astype("boolean")


# =============================================================================
# markets
# =============================================================================
MARKET_COLS = {  # raw gamma name -> clean name
    "token_id": "token_id", "conditionId": "condition_id", "id": "market_id", "slug": "slug",
    "question": "question", "description": "description", "outcome": "outcome",
    "outcome_index": "outcome_index", "complement_token_id": "complement_token_id",
    "event_id": "event_id", "event_slug": "event_slug", "event_title": "event_title",
    "groupItemTitle": "group_item_title", "sportsMarketType": "sports_market_type", "line": "line",
    "negRisk": "neg_risk", "negRiskOther": "neg_risk_other", "negRiskMarketID": "neg_risk_market_id",
    "createdAt": "created_at", "startDate": "start_date", "endDate": "end_date",
    "closedTime": "closed_time", "gameStartTime": "game_start_time", "event_endDate": "event_end_date",
    "orderPriceMinTickSize": "tick_size", "orderMinSize": "min_order_size",
    "volumeNum": "volume", "volume24hr": "volume_24h", "volume1wk": "volume_1wk",
    "volume1mo": "volume_1mo", "liquidityNum": "liquidity",
    "bestBid": "best_bid", "bestAsk": "best_ask", "lastTradePrice": "last_trade_price",
    "spread": "spread", "oneDayPriceChange": "price_change_1d", "oneWeekPriceChange": "price_change_1w",
    "feesEnabled": "fees_enabled", "takerBaseFee": "taker_base_fee", "makerBaseFee": "maker_base_fee",
    "rewardsMinSize": "rewards_min_size", "rewardsMaxSpread": "rewards_max_spread",
    "umaResolutionStatus": "uma_status", "resolutionSource": "resolution_source",
    "outcome_price": "outcome_price", "meta_fetched_at": "fetched_at",
}
DATE_COLS = ["created_at", "start_date", "end_date", "closed_time", "game_start_time", "event_end_date", "fetched_at"]
NUM_COLS = ["line", "tick_size", "min_order_size", "volume", "volume_24h", "volume_1wk", "volume_1mo",
            "liquidity", "best_bid", "best_ask", "last_trade_price", "spread", "price_change_1d",
            "price_change_1w", "taker_base_fee", "maker_base_fee", "rewards_min_size",
            "rewards_max_spread", "outcome_price"]


def clean_markets(df: pd.DataFrame, which: str) -> pd.DataFrame:
    d = pd.DataFrame({new: df[old] if old in df else None for old, new in MARKET_COLS.items()})
    for c in DATE_COLS:
        d[c] = utc(d[c])
    for c in NUM_COLS:
        d[c] = num(d[c])
    for c in ("neg_risk", "neg_risk_other", "fees_enabled"):
        d[c] = boolcol(d[c])
    d["outcome_index"] = num(d["outcome_index"], "int8")
    for c in ("token_id", "condition_id", "market_id", "complement_token_id", "event_id"):
        d[c] = d[c].astype("string")
    d["set"] = which
    # Bid/ask on a NO token from Gamma are the YES token's: flip so every row is in its own terms.
    no = d["outcome_index"] == 1
    bb, ba = d.loc[no, "best_bid"], d.loc[no, "best_ask"]
    d.loc[no, "best_bid"], d.loc[no, "best_ask"] = 1 - ba, 1 - bb
    d.loc[no, "last_trade_price"] = np.nan          # ambiguous side; use trades instead
    d.loc[no, ["price_change_1d", "price_change_1w"]] *= -1
    # Label: resolved markets pay 1 or 0. Anything else (0.5 = 50/50 or unresolved) -> NaN.
    if which == "closed":
        r = d["outcome_price"].round(6)
        d["resolved"] = r.isin([0.0, 1.0]) & d["closed_time"].notna()
        d["label"] = r.where(d["resolved"]).astype("Float64")
    else:
        d["resolved"] = False
        d["label"] = pd.Series(pd.NA, index=d.index, dtype="Float64")
    d = d.drop_duplicates("token_id")
    return d


def build_markets():
    out = []
    for w in ("live", "closed"):
        f = os.path.join(RAW, f"markets_{w}.parquet")
        if os.path.exists(f):
            out.append(clean_markets(pd.read_parquet(f), w))
    m = pd.concat(out, ignore_index=True).drop_duplicates("token_id", keep="last")
    m.to_parquet(os.path.join(OUT, "markets.parquet"), index=False)
    log(f"markets: {len(m):,} tokens ({(m.set == 'live').sum():,} live, {m.resolved.sum():,} resolved w/ label)")
    return m


def load_markets():
    return pd.read_parquet(os.path.join(OUT, "markets.parquet"))


# =============================================================================
# prices
# =============================================================================
def build_prices():
    for name in ("prices_hourly", "prices_minute"):
        src = os.path.join(RAW, name)
        if not os.path.isdir(src):
            continue
        dst = os.path.join(OUT, name)
        fresh_dir(dst)
        n_in = n_out = 0
        for i, f in enumerate(sorted(glob.glob(os.path.join(src, "part-*.parquet")))):
            p = pd.read_parquet(f)
            n_in += len(p)
            p = pd.DataFrame({"token_id": p["token_id"].astype("string"), "ts": utc(p["ts"], "s"),
                              "p": p["p"].astype("float32"),
                              **({"fidelity_min": p["fidelity_min"].astype("int16")} if "fidelity_min" in p else {})})
            p = p[(p["p"] > 0) & (p["p"] < 1) & p["ts"].notna()]
            p = p.drop_duplicates(["token_id", "ts"], keep="last").sort_values(["token_id", "ts"])
            n_out += len(p)
            p.to_parquet(os.path.join(dst, f"part-{i:05d}.parquet"), index=False)
        log(f"{name}: {n_in:,} raw -> {n_out:,} clean rows")


# =============================================================================
# trades
# =============================================================================
def clean_trades(t: pd.DataFrame) -> pd.DataFrame:
    d = pd.DataFrame({
        "ts": utc(num(t["timestamp"]), "s"),
        "condition_id": t["condition_id"].astype("string"),
        "token_id": t["token_id"].astype("string"),
        "outcome": t["outcome"].astype("string"),
        "outcome_index": num(t["outcome_index"], "int8"),
        "side": np.where(t["side"].astype(str).str.upper() == "BUY", 1, -1).astype("int8"),
        "price": num(t["price"]), "size": num(t["size"]),
        "wallet": t.get("proxy_wallet", pd.Series(index=t.index, dtype="string")).astype("string"),
        "tx_hash": t["transaction_hash"].astype("string"),
    })
    d = d[(d["price"] > 0) & (d["price"] < 1) & (d["size"] > 0) & d["ts"].notna()]
    d = d.drop_duplicates(["tx_hash", "token_id", "side", "price", "size", "wallet"])
    d["notional"] = d["price"] * d["size"]
    yes = d["outcome_index"] == 0
    d["yes_price"] = np.where(yes, d["price"], 1 - d["price"])     # every trade in YES terms
    d["yes_side"] = np.where(yes, d["side"], -d["side"]).astype("int8")
    return d.sort_values(["condition_id", "ts"])


def build_trades():
    src = os.path.join(RAW, "trades")
    if not os.path.isdir(src):
        log("trades: not downloaded yet")
        return
    dst = os.path.join(OUT, "trades")
    fresh_dir(dst)
    n = 0
    for i, f in enumerate(sorted(glob.glob(os.path.join(src, "part-*.parquet")))):
        t = clean_trades(pd.read_parquet(f))
        n += len(t)
        t.to_parquet(os.path.join(dst, f"part-{i:05d}.parquet"), index=False)
    log(f"trades: {n:,} clean rows")


# =============================================================================
# books (snapshot): summary features + wide top-10 levels
# =============================================================================
def build_books(K=10):
    fs = sorted(glob.glob(os.path.join(RAW, "books", "levels_*.parquet")))
    if not fs:
        return
    out = []
    for f in fs:
        tag = os.path.basename(f)[7:-8]
        lv = pd.read_parquet(f)
        sm = pd.read_parquet(os.path.join(RAW, "books", f"summary_{tag}.parquet"))
        lv["token_id"] = lv["token_id"].astype(str)
        lv = lv[(lv["size"] > 0) & (lv["price"] > 0) & (lv["price"] < 1)]
        lv = lv.sort_values(["token_id", "side", "level"])
        top = lv[lv["level"] < K]
        wide = top.pivot_table(index="token_id", columns=["side", "level"], values=["price", "size"])
        wide.columns = [f"{s}_{'px' if v == 'price' else 'sz'}_{l + 1}" for v, s, l in wide.columns]
        wide = wide[sorted(wide.columns, key=lambda c: (c[:3], int(c.split('_')[-1]), c))]
        # depth features: shares and $ committed within 2c / 5c / 10c of the touch
        bb = lv[lv.side == "bid"].groupby("token_id")["price"].max()
        ba = lv[lv.side == "ask"].groupby("token_id")["price"].min()
        lv = lv.join(bb.rename("bb"), on="token_id").join(ba.rename("ba"), on="token_id")
        lv["dist"] = np.where(lv.side == "bid", lv.bb - lv.price, lv.price - lv.ba)
        lv["usd"] = np.where(lv.side == "bid", lv.price, 1 - lv.price) * lv["size"]
        feats = {}
        for c in (0.02, 0.05, 0.10):
            w = lv[lv["dist"] <= c + 1e-9]
            g = w.groupby(["token_id", "side"])["usd"].sum().unstack()
            feats[f"bid_usd_{int(c * 100)}c"] = g.get("bid")
            feats[f"ask_usd_{int(c * 100)}c"] = g.get("ask")
        feats = pd.DataFrame(feats)
        s = sm.assign(token_id=sm["token_id"].astype(str)).set_index("token_id")
        s = s[["ts", "market", "best_bid", "best_ask", "spread", "n_bids", "n_asks", "bid_shares", "ask_shares",
               "tick_size", "min_order_size", "neg_risk", "last_trade_price"]]
        s = s.rename(columns={"market": "condition_id"})
        s["mid"] = (s.best_bid + s.best_ask) / 2
        b1, a1 = wide.get("bid_sz_1"), wide.get("ask_sz_1")
        s["microprice"] = (s.best_bid * a1 + s.best_ask * b1) / (a1 + b1)
        s["imbalance_1"] = (b1 - a1) / (b1 + a1)
        s["two_sided"] = s.best_bid.notna() & s.best_ask.notna()
        out.append(s.join(feats).join(wide).reset_index())
    b = pd.concat(out, ignore_index=True)
    b["ts"] = utc(b["ts"])
    b.to_parquet(os.path.join(OUT, "book_snapshots.parquet"), index=False)
    log(f"book_snapshots: {len(b):,} token books ({b.two_sided.mean():.0%} two-sided), {len(fs)} snapshot(s)")


# =============================================================================
# resolved panel: hourly prices of resolved markets + label (calibration / forecasting)
# =============================================================================
def build_resolved():
    m = load_markets()
    m = m[m.resolved & (m.outcome_index == 0)][["token_id", "condition_id", "slug", "event_slug", "neg_risk",
                                                 "volume", "created_at", "end_date", "closed_time", "label"]]
    d = os.path.join(OUT, "prices_hourly")
    if not os.path.isdir(d) or m.empty:
        return
    p = pd.read_parquet(d, filters=[("token_id", "in", m.token_id.tolist())])
    p = p.merge(m, on="token_id")
    # leakage guard: only prices strictly before the market closed
    p = p[p.ts < p.closed_time]
    p = p.sort_values(["token_id", "ts"])
    p["hours_to_close"] = ((p.closed_time - p.ts).dt.total_seconds() / 3600).astype("float32")
    p["hours_since_created"] = ((p.ts - p.created_at).dt.total_seconds() / 3600).astype("float32")
    pc = p.p.clip(0.001, 0.999).astype("float64")
    p["logit_p"] = np.log(pc / (1 - pc)).astype("float32")
    g = p.groupby("token_id")["logit_p"]
    step_h = (p.groupby("token_id")["ts"].diff().dt.total_seconds() / 3600)
    p["step_h"] = step_h.astype("float32")
    p["dlogit_1"] = g.diff().astype("float32")
    p["dlogit_4"] = g.diff(4).astype("float32")
    p["rv_24"] = (g.diff().pow(2).groupby(p.token_id).rolling(24, min_periods=4).mean()
                  .reset_index(level=0, drop=True)).pow(0.5).astype("float32")
    p["label"] = p["label"].astype("int8")
    p = p.drop(columns=["created_at"])
    p.to_parquet(os.path.join(OUT, "resolved_panel.parquet"), index=False)
    brier = float(((p.p - p.label) ** 2).mean())
    log(f"resolved_panel: {len(p):,} rows, {p.token_id.nunique():,} markets, market Brier {brier:.4f}")


# =============================================================================
# crypto 5m/15m (HuggingFace) -> trades, spot, price history, labelled panel
# =============================================================================
def crypto_token_map(cids: list[str]) -> pd.DataFrame:
    """condition_id -> Up/Down token ids + resolution, from Gamma (cached)."""
    cache = os.path.join(HF, "_gamma_tokens.parquet")
    have = pd.read_parquet(cache) if os.path.exists(cache) else pd.DataFrame(columns=["condition_id"])
    todo = sorted(set(cids) - set(have.condition_id))
    rows, s = [], requests.Session()
    for i in range(0, len(todo), 50):
        chunk = todo[i:i + 50]
        for attempt in range(5):
            r = s.get(GAMMA, params=[("condition_ids", c) for c in chunk] + [("closed", "true"), ("limit", 100)], timeout=30)
            if r.status_code == 200:
                break
            time.sleep(2 ** attempt)
        for x in r.json() if r.ok else []:
            toks, outs = json.loads(x.get("clobTokenIds") or "[]"), json.loads(x.get("outcomes") or "[]")
            px = [float(v) for v in json.loads(x.get("outcomePrices") or "[]")]
            if len(toks) != 2 or "Up" not in outs:
                continue
            u = outs.index("Up")
            rows.append({"condition_id": x["conditionId"], "slug": x["slug"], "up_token": str(toks[u]),
                         "down_token": str(toks[1 - u]), "up_final": px[u] if px else None,
                         "closed": bool(x.get("closed"))})
        if i and i % 2000 == 0:
            log(f"  gamma tokens {i:,}/{len(todo):,}")
    if rows:
        have = pd.concat([have, pd.DataFrame(rows)], ignore_index=True).drop_duplicates("condition_id", keep="last")
        have.to_parquet(cache, index=False)
    return have


def build_crypto():
    if not os.path.isdir(HF):
        return
    dst = os.path.join(OUT, "crypto")
    fresh_dir(dst)
    mk = pd.read_parquet(os.path.join(HF, "markets", "all.parquet"))
    tr_files = sorted(glob.glob(os.path.join(HF, "trades", "*.parquet")))
    cids = set(mk.condition_id.dropna())
    for f in tr_files:
        cids |= set(pd.read_parquet(f, columns=["condition_id"]).condition_id.unique())
    cids = sorted(c for c in cids if isinstance(c, str) and c.startswith("0x"))
    log(f"crypto: mapping {len(cids):,} markets to Up/Down tokens via Gamma")
    tm = crypto_token_map(cids)
    tm = tm[tm.closed]
    tm["label_up"] = tm.up_final.round(6).where(tm.up_final.round(6).isin([0.0, 1.0]))
    mk = mk.drop(columns=["condition_id"], errors="ignore").merge(
        tm[["condition_id", "slug", "up_token", "down_token", "label_up"]], on="slug", how="inner")
    mk["start_time"] = utc(mk.start_time)
    mk["end_time"] = utc(mk.end_time)
    mk = mk.rename(columns={"market_id": "slug_id"})[
        ["condition_id", "slug", "asset", "market_type", "start_time", "end_time", "up_token", "down_token", "label_up"]]
    mk = mk.drop_duplicates("condition_id")
    mk.to_parquet(os.path.join(dst, "markets.parquet"), index=False)
    log(f"  crypto markets: {len(mk):,} ({mk.label_up.notna().sum():,} labelled, up rate {mk.label_up.mean():.3f})")
    up = dict(zip(mk.up_token, [True] * len(mk))) | dict(zip(mk.down_token, [False] * len(mk)))

    # trades: typed, deduped, Up-terms columns
    os.makedirs(os.path.join(dst, "trades"), exist_ok=True)
    n = 0
    bins = []
    for f in tr_files:
        t = pd.read_parquet(f)
        t = pd.DataFrame({"ts": utc(t.timestamp), "asset": t.asset.astype("string"),
                          "condition_id": t.condition_id.astype("string"), "token_id": t.token_id.astype("string"),
                          "side": np.where(t.side.str.upper() == "BUY", 1, -1).astype("int8"),
                          "price": t.price.astype("float64"), "size": t["size"].astype("float64"),
                          "maker": t.maker.astype("string"), "tx_hash": t.tx_hash.astype("string")})
        t = t[(t.price > 0) & (t.price < 1) & (t["size"] > 0)]
        t = t.drop_duplicates(["tx_hash", "token_id", "side", "price", "size", "maker"])
        isup = t.token_id.map(up)
        t = t[isup.notna()]
        isup = isup[isup.notna()].astype(bool)
        t["is_up_token"] = isup
        t["up_price"] = np.where(isup, t.price, 1 - t.price)
        t["up_side"] = np.where(isup, t.side, -t.side).astype("int8")
        t["notional"] = t.price * t["size"]
        t = t.drop(columns="tx_hash").sort_values("ts")
        n += len(t)
        t.to_parquet(os.path.join(dst, "trades", os.path.basename(f)), index=False, compression="zstd")
        t = t[["condition_id", "ts", "size", "up_price", "up_side"]].copy()
        # 1-minute trade bars per market (Up terms) for the panel
        t["minute"] = t.ts.dt.floor("min")
        t["upv"] = t.up_price * t["size"]
        t["buy_sz"] = t["size"] * (t.up_side > 0)
        g = t.groupby(["condition_id", "minute"])
        bins.append(pd.DataFrame({"n_trades": g.size(), "volume": g["size"].sum(),
                                  "vwap_up": g["upv"].sum() / g["size"].sum(),
                                  "buy_up_share": g["buy_sz"].sum() / g["size"].sum()}).reset_index())
    log(f"  crypto trades: {n:,} rows")

    spot = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(HF, "crypto_prices", "*.parquet")))])
    spot["ts"] = utc(spot.timestamp)
    spot = spot.drop(columns="timestamp").drop_duplicates(["asset", "ts"]).sort_values(["asset", "ts"])
    spot.to_parquet(os.path.join(dst, "spot_1m.parquet"), index=False)

    ph = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(HF, "price_history", "*.parquet")))])
    ph = ph[ph.token_id.isin(mk.up_token)].rename(columns={"price": "p_up", "token_id": "up_token"})
    ph["ts"] = utc(ph.timestamp)
    ph = ph.drop(columns=["timestamp", "asset", "market_id"]).drop_duplicates(["up_token", "ts"])
    ph = ph.merge(mk, on="up_token")
    ph = ph[(ph.ts >= ph.start_time) & (ph.ts < ph.end_time)]       # live window only, no leakage
    ph.drop(columns=["down_token"]).to_parquet(os.path.join(dst, "price_history.parquet"), index=False)

    # Labelled panel: one row per market per minute of its life. The row's timestamp `ts` is the
    # END of the minute; every feature uses only information available at ts (no leakage).
    bars = pd.concat(bins, ignore_index=True)
    bars["ts"] = bars.pop("minute") + pd.Timedelta("1min")
    lab = mk[mk.label_up.notna()]
    win = (lab.end_time - lab.start_time).dt.total_seconds().div(60).round().astype(int)
    grid = lab.loc[lab.index.repeat(win)].copy()
    grid["ts"] = grid.start_time + pd.to_timedelta(grid.groupby(level=0).cumcount() + 1, unit="min")
    grid = grid.reset_index(drop=True).drop(columns=["down_token"])
    pn = grid.merge(bars, on=["condition_id", "ts"], how="left")
    pn[["n_trades", "volume"]] = pn[["n_trades", "volume"]].fillna(0)
    pn = pn.sort_values(["condition_id", "ts"])
    pn["last_vwap_up"] = pn.groupby("condition_id")["vwap_up"].ffill()
    pn["cum_trades"] = pn.groupby("condition_id")["n_trades"].cumsum()
    pn["cum_volume"] = pn.groupby("condition_id")["volume"].cumsum()
    # CLOB price history (only 2026-03-06..13): latest point at or before ts
    hp = ph[["condition_id", "ts", "p_up"]].rename(columns={"ts": "ph_ts"}).sort_values("ph_ts")
    pn = pd.merge_asof(pn.sort_values("ts"), hp, left_on="ts", right_on="ph_ts", by="condition_id",
                       direction="backward", tolerance=pd.Timedelta("2min"))
    pn["p_up_best"] = pn["p_up"].fillna(pn["last_vwap_up"])
    # spot: open of the window's first 1m bar, and close of the last bar finished by ts
    sp = spot[["asset", "ts", "open", "close"]]
    pn = pn.merge(sp[["asset", "ts", "open"]].rename(columns={"ts": "start_time", "open": "spot_open"}),
                  on=["asset", "start_time"], how="left")
    sc = sp[["asset", "ts", "close"]].rename(columns={"ts": "spot_bar_open", "close": "spot"})
    pn["spot_bar_open"] = pn.ts - pd.Timedelta("1min")
    pn = pd.merge_asof(pn.sort_values("spot_bar_open"), sc.sort_values("spot_bar_open"), on="spot_bar_open",
                       by="asset", direction="backward", tolerance=pd.Timedelta("1min"))
    pn["spot_ret_bps"] = (1e4 * (pn.spot / pn.spot_open - 1)).astype("float32")
    pn["secs_left"] = (pn.end_time - pn.ts).dt.total_seconds().astype("float32")
    pn["secs_elapsed"] = (pn.ts - pn.start_time).dt.total_seconds().astype("float32")
    pn["label_up"] = pn.label_up.astype("int8")
    pn = pn.drop(columns=["spot_bar_open", "ph_ts"]).sort_values(["condition_id", "ts"]).reset_index(drop=True)
    pn.to_parquet(os.path.join(dst, "panel_1m.parquet"), index=False, compression="zstd")
    ok = pn.p_up_best.notna()
    log(f"  crypto panel_1m: {len(pn):,} rows, {pn.condition_id.nunique():,} markets, "
        f"price {ok.mean():.0%}, spot {pn.spot_ret_bps.notna().mean():.0%}, "
        f"Brier(p_up_best) {((pn.p_up_best[ok] - pn.label_up[ok]) ** 2).mean():.4f}")


# =============================================================================
# validation report
# =============================================================================
def report():
    lines = []
    for f in sorted(glob.glob(os.path.join(OUT, "**", "*.parquet"), recursive=True)):
        if "part-" in f and not f.endswith("part-00000.parquet"):
            continue
    for name in ["markets.parquet", "prices_hourly", "prices_minute", "trades", "book_snapshots.parquet",
                 "resolved_panel.parquet", "crypto/markets.parquet", "crypto/trades", "crypto/spot_1m.parquet",
                 "crypto/price_history.parquet", "crypto/panel_1m.parquet"]:
        p = os.path.join(OUT, name)
        if not os.path.exists(p):
            continue
        import pyarrow.dataset as ds
        dset = ds.dataset(p, format="parquet")
        size = sum(os.path.getsize(x) for x in (glob.glob(os.path.join(p, "*.parquet")) if os.path.isdir(p) else [p]))
        lines.append(f"{name:32s} {dset.count_rows():>13,} rows  {size / 1e6:9.1f} MB  {len(dset.schema)} cols")
    log("report:\n  " + "\n  ".join(lines))


STAGES = {"markets": build_markets, "prices": build_prices, "trades": build_trades, "books": build_books,
          "resolved": build_resolved, "crypto": build_crypto}

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    todo = sys.argv[1:] or list(STAGES)
    for s in todo:
        t0 = time.time()
        STAGES[s]()
        log(f"== {s} {time.time() - t0:.0f}s")
    report()
