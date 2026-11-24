"""
analysis.py - what did the run teach us? (`paper.py report [RUN]`)

1. PnL         per market, marked to the final fair value.
2. Fills       by kind (through / at_level / cross) and side, with MARKOUTS:
               markout_h = side * (mid at t+h - fill price) * size.
               Positive = the price moved our way after we traded; negative =
               we were picked off (adverse selection). This is the honest test
               of the informed-trader parameters (mu, J).
3. Fill rate   realised fills/h vs the model's expected fills/h while quoting.
               ratio < 1: the flow model (--k / --flow-cap) is too optimistic.
4. Which estimate was right?
               For every fair-value prediction (one per market per --every s),
               the target is what traders actually paid next: the VWAP of real
               trades in (t, t+h]. Each estimator (FairMV fv, mid, microprice,
               last trade, and every FairMV component) is scored on:
                 MAE, RMSE, bias       vs that target
                 skill                 1 - MSE / MSE(mid): > 0 beats the mid
                 direction             when the estimate disagrees with the mid,
                                       how often the next trades land on its side
               plus a non-negative least-squares blend: the weights that would
               have predicted the trades best (a suggestion, fit in-sample).
5. FV sd       coverage of the 80% interval fv +- 1.28 sd (should be ~80%).
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd

HORIZONS = {"1m": 60, "5m": 300, "30m": 1800, "2h": 7200}


def _read(run: str, name: str) -> pd.DataFrame:
    p = os.path.join(run, f"{name}.parquet")
    if not os.path.exists(p):
        return pd.DataFrame()
    d = pd.read_parquet(p)
    for c in ("ts", "placed"):
        if c in d:
            d[c] = pd.to_datetime(d[c], utc=True)
    if "token_id" in d:
        d["token_id"] = d["token_id"].astype(str)
    return d


def latest_run(base: str) -> str:
    runs = sorted(glob.glob(os.path.join(base, "*", "config.json")), key=os.path.getmtime)
    if not runs:
        raise SystemExit(f"no runs under {base}")
    return os.path.dirname(runs[-1])


# =============================================================================
# Markouts and fill rate
# =============================================================================
def markouts(fills: pd.DataFrame, preds: pd.DataFrame) -> pd.DataFrame:
    if fills.empty or preds.empty:
        return fills
    f = fills.sort_values("ts").copy()
    sgn = np.where(f["side"] == "buy", 1.0, -1.0)
    for name, sec in HORIZONS.items():
        f[f"t_{name}"] = f["ts"] + pd.Timedelta(seconds=sec)
        cols = []
        for tok, g in f.groupby("token_id"):
            p = preds[preds.token_id == tok][["ts", "mid"]].sort_values("ts")
            if p.empty:
                continue
            m = pd.merge_asof(g.sort_values(f"t_{name}"), p.rename(columns={"ts": "pt", "mid": f"mid_{name}"}),
                              left_on=f"t_{name}", right_on="pt", direction="backward")
            # only valid if the run actually lasted that long after the fill
            m.loc[m["pt"] < m[f"t_{name}"] - pd.Timedelta(seconds=max(sec * 0.5, 60)), f"mid_{name}"] = np.nan
            cols.append(m.set_index(g.sort_values(f"t_{name}").index)[f"mid_{name}"])
        f[f"mid_{name}"] = pd.concat(cols) if cols else np.nan
        f[f"mo_{name}"] = sgn * (f[f"mid_{name}"] - f["price"]) * f["size"]
        f = f.drop(columns=[f"t_{name}"])
    return f


def fill_rate(quotes: pd.DataFrame, fills: pd.DataFrame) -> pd.DataFrame:
    if quotes.empty:
        return pd.DataFrame()
    rows = []
    for tok, q in quotes.sort_values("ts").groupby("token_id"):
        dt = q["ts"].diff().shift(-1).dt.total_seconds().fillna(0).clip(upper=600) / 3600
        quoting = (q["bid_size"] + q["ask_size"]) > 0
        hours = float(dt[quoting].sum())
        exp = float((q["exp_fills_h"] * dt).sum())
        n = int((fills.token_id == tok).sum()) if len(fills) else 0
        rows.append(dict(slug=q["slug"].iloc[0][:50], hours_quoting=hours, exp_fills=exp, fills=n,
                         ratio=n / exp if exp > 0 else np.nan,
                         exp_pnl=float((q["exp_pnl_h"] * dt).sum())))
    return pd.DataFrame(rows)


# =============================================================================
# Which estimate was right?
# =============================================================================
def future_vwap(preds: pd.DataFrame, trades: pd.DataFrame, sec: int) -> pd.Series:
    """VWAP (token terms) of real trades in (t, t+sec] for each prediction row."""
    out = pd.Series(np.nan, index=preds.index)
    if trades.empty:
        return out
    for tok, p in preds.groupby("token_id"):
        t = trades[trades.token_id == tok].sort_values("ts")
        if t.empty:
            continue
        ts = t["ts"].astype("int64").to_numpy()
        pv = np.concatenate([[0.0], np.cumsum((t["price"] * t["size"]).to_numpy())])
        v = np.concatenate([[0.0], np.cumsum(t["size"].to_numpy())])
        t0 = p["ts"].astype("int64").to_numpy()
        lo = np.searchsorted(ts, t0, side="right")
        hi = np.searchsorted(ts, t0 + sec * 10 ** 9, side="right")
        vol = v[hi] - v[lo]
        with np.errstate(invalid="ignore", divide="ignore"):
            out.loc[p.index] = np.where(vol > 0, (pv[hi] - pv[lo]) / vol, np.nan)
    return out


def score(preds: pd.DataFrame, trades: pd.DataFrame, every: float, horizon: str) -> tuple[pd.DataFrame, dict, int]:
    if preds.empty or trades.empty:
        return pd.DataFrame(), {}, 0
    p = preds.sort_values("ts").copy()
    p["bucket"] = p["ts"].dt.floor(f"{int(max(every, 1))}s")
    p = p.groupby(["token_id", "bucket"]).tail(1)                 # thin overlapping predictions
    p["target"] = future_vwap(p, trades, HORIZONS[horizon])
    p = p[p["target"].notna()]
    if p.empty:
        return pd.DataFrame(), {}, 0
    est = [c for c in ["fv", "fv_raw", "mid", "microprice", "last_trade"] if c in p] + \
          sorted(c for c in p.columns if c.startswith("c_"))
    rows = []
    mse_mid = float(np.mean((p["mid"] - p["target"]) ** 2))
    for c in est:
        e = p[c].astype(float)
        ok = e.notna()
        if ok.sum() < 5:
            continue
        err = e[ok] - p.loc[ok, "target"]
        d_est = e[ok] - p.loc[ok, "mid"]
        d_tgt = p.loc[ok, "target"] - p.loc[ok, "mid"]
        disagree = d_est.abs() > 0.005
        hit = float((np.sign(d_est[disagree]) == np.sign(d_tgt[disagree])).mean()) if disagree.sum() else np.nan
        mse_mid_ok = float(np.mean((p.loc[ok, "mid"] - p.loc[ok, "target"]) ** 2))
        rows.append(dict(estimator=c.removeprefix("c_"), n=int(ok.sum()), mae=float(err.abs().mean()),
                         rmse=float(np.sqrt((err ** 2).mean())), bias=float(err.mean()),
                         skill_vs_mid=1 - float((err ** 2).mean()) / mse_mid_ok if mse_mid_ok > 0 else np.nan,
                         direction_hit=hit, n_disagree=int(disagree.sum())))
    tab = pd.DataFrame(rows).sort_values("rmse") if rows else pd.DataFrame()

    blend = {}
    cand = [c for c in est if c not in ("fv_raw",) and p[c].notna().mean() > 0.9]
    if len(cand) >= 2:
        q = p[cand + ["target"]].dropna()
        if len(q) >= 10:
            try:
                from scipy.optimize import nnls
                w, _ = nnls(q[cand].to_numpy(float), q["target"].to_numpy(float))
            except Exception:  # noqa: BLE001
                w = np.clip(np.linalg.lstsq(q[cand].to_numpy(float), q["target"].to_numpy(float), rcond=None)[0], 0, None)
            if w.sum() > 0:
                w = w / w.sum()
                pred = q[cand].to_numpy(float) @ w
                blend = {"weights": {c.removeprefix("c_"): round(float(x), 3) for c, x in zip(cand, w) if x > 1e-3},
                         "rmse": float(np.sqrt(np.mean((pred - q["target"]) ** 2))), "n": len(q),
                         "rmse_mid": float(np.sqrt(mse_mid))}
    return tab, blend, len(p)


def coverage(preds: pd.DataFrame, trades: pd.DataFrame) -> float:
    p = preds.dropna(subset=["fv", "fv_sd"]).copy()
    if p.empty:
        return np.nan
    p["target"] = future_vwap(p, trades, HORIZONS["30m"])
    p = p.dropna(subset=["target"])
    if p.empty:
        return np.nan
    return float(((p["target"] - p["fv"]).abs() <= 1.2816 * p["fv_sd"]).mean())


# =============================================================================
# Report
# =============================================================================
def report(run: str, every: float = 60.0, horizon: str = "30m") -> str:
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    fills, quotes, preds, trades = (_read(run, n) for n in ("fills", "quotes", "predictions", "trades"))
    cfg = json.load(open(os.path.join(run, "config.json"))) if os.path.exists(os.path.join(run, "config.json")) else {}
    summ = json.load(open(os.path.join(run, "summary.json"))) if os.path.exists(os.path.join(run, "summary.json")) else {}
    L = [f"Run: {run}   mode: {cfg.get('mode')}   observe: {cfg.get('observe')}"]
    if len(preds):
        L.append(f"Span: {preds.ts.min()} -> {preds.ts.max()}  "
                 f"({(preds.ts.max() - preds.ts.min()).total_seconds() / 3600:.2f} h)  "
                 f"predictions {len(preds):,}  real trades {len(trades):,}")

    if summ.get("markets"):
        s = pd.DataFrame(summ["markets"])
        s["slug"] = s["slug"].str[:55]
        L += ["", "== PnL (marked to final fair value) ==",
              s[["slug", "trading", "fills", "volume", "inventory", "cash", "fv", "pnl"]].round(3).to_string(index=False),
              f"TOTAL ${summ.get('total_pnl', 0):+.2f}"]

    if len(fills):
        f = markouts(fills, preds)
        L += ["", "== Fills ==", f.groupby(["kind", "side"]).agg(n=("size", "size"), shares=("size", "sum"),
                                                                 avg_px=("price", "mean")).round(3).to_string()]
        mo = {k: f[f"mo_{k}"].sum() for k in HORIZONS if f"mo_{k}" in f and f[f"mo_{k}"].notna().any()}
        if mo:
            L += ["", "Markouts vs later mid ($, + = price moved our way; - = picked off):",
                  "  " + "   ".join(f"{k}: {v:+.2f} (n={int(f[f'mo_{k}'].notna().sum())})" for k, v in mo.items())]
            by = f.groupby("kind")[[f"mo_{k}" for k in mo]].sum().round(2)
            L.append(by.to_string())
    else:
        L += ["", "== Fills ==", "none"]

    fr = fill_rate(quotes, fills)
    if len(fr):
        L += ["", "== Fill rate: model vs reality ==", fr.round(2).to_string(index=False)]
        tot_e, tot_f = fr["exp_fills"].sum(), fr["fills"].sum()
        if tot_e > 0:
            r = tot_f / tot_e
            L.append(f"overall realised/expected = {r:.2f}  "
                     + ("-> flow model too optimistic: lower --flow-cap / --k" if r < 0.5 else
                        "-> flow model too pessimistic: raise --flow-cap" if r > 2 else "-> roughly consistent"))

    for h in ([horizon] if horizon != "all" else list(HORIZONS)):
        tab, blend, n = score(preds, trades, every, h)
        L += ["", f"== Which estimate was right? target = VWAP of real trades in next {h} "
                  f"({n:,} predictions with a trade after them) =="]
        if tab.empty:
            L.append("no predictions were followed by trades in this horizon (try a longer run or --horizon 2h)")
            continue
        L.append(tab.round(4).to_string(index=False))
        if blend:
            L.append(f"best non-negative blend (in-sample): {blend['weights']}  "
                     f"rmse {blend['rmse']:.4f} vs mid {blend['rmse_mid']:.4f}")
    cov = coverage(preds, trades)
    if not np.isnan(cov):
        L += ["", f"FV 80% interval coverage of next-30m trade VWAP: {cov:.0%} "
                  + ("(sd too small: overconfident)" if cov < 0.65 else "(sd too large: underconfident)" if cov > 0.92 else "(ok)")]
    out = "\n".join(L)
    with open(os.path.join(run, "report.txt"), "w", encoding="utf-8") as fh:
        fh.write(out)
    return out
