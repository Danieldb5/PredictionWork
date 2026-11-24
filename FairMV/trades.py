"""
trades.py - fair value from recent trades, and volatility, for thin markets.

In a wide-spread market the trades carry most of the information: each one
is someone actually putting money down. Three problems with a plain VWAP:

1. Bid-ask bounce. Takers buy at the ask and sell at the bid, so a run of
   buys overstates value and a run of sells understates it. Each trade is
   corrected toward the inside by a fraction `trade_side_adjust` (alpha) of
   the half-spread h (log-odds):
       L_i = logit(price_i) - side_i * alpha * h
   alpha = 0 is pure Glosten-Milgrom (the trade price IS the post-trade
   expected value); alpha = 1 treats takers as uninformed noise (value = mid).
   The default 0.5 sits between, and the uncertainty about alpha goes into
   the variance: the adjustment is uniform on [0, h], so its sd is h/sqrt(12).
   h is the smaller of the effective book half-spread and the REALISED
   half-spread from trades, (median L of buys - median L of sells) / 2,
   capped at trade_h_cap. A book whose maker went away can be 50c wide while
   trades print 3c apart, and the current dust spread says nothing about
   what yesterday's takers paid.

2. Staleness. The value follows a random walk with variance sigma^2 per
   hour (log-odds), so a trade a_i hours old has error variance
       v_i = v0 + sigma^2 a_i,      v0 = (h/sqrt(12))^2 + trade_idio_sd^2
   Trades share the random-walk path, so the errors are correlated:
   Cov(e_i, e_j) = sigma^2 min(a_i, a_j). The variance of the weighted mean
   is computed exactly with that covariance. Ten trades from yesterday are
   therefore NOT ten times as good as one: they all share yesterday-to-now
   drift.

3. Whales and dust. Weight w_i = sqrt(min(capital_i, cap)) / v_i: size
   counts, but with diminishing returns and a cap.

4. Disagreeing prints. If the prints scatter more than the model allows
   (0.04 and 0.29 on the same day), the variance is floored at their
   weighted dispersion / (n_eff - 1). The model then admits it doesn't know.

Volatility sigma^2 (log-odds variance per hour) uses SAME-SIDE prints, which
removes bounce, sampled at least sigma2_min_dt_h (1h) apart: bursts of prints
seconds apart measure noise, not volatility. Fills in the same second on the
same side (one taker sweeping levels) are first collapsed into one VWAP print.
The print noise 2*sigma2_noise_sd^2 is subtracted from each pair, then
    sigma^2 = (sum dL^2 + n0 * prior) / (sum dt + n0)
i.e. the prior counts as n0 hours of data, which keeps very quiet markets
from reporting zero volatility.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

from .core import Estimate, FVConfig, logit


# =============================================================================
# Normalisation
# =============================================================================
def normalize_trades(df: Optional[pd.DataFrame], outcome_index: Optional[int] = None) -> pd.DataFrame:
    """Return columns ts (UTC Timestamp), price, side (+1 taker bought this token), size,
    in the terms of the token being valued.

    Accepts:
      * Scalper.fetch_trades output (yes_price / yes_side), valued as YES unless outcome_index=1
      * raw Data API rows (price, side 'BUY'/'SELL', size, timestamp, outcome_index)
      * generic frames with price, side (+1/-1 or BUY/SELL), size and ts or timestamp
    """
    cols = ["ts", "price", "side", "size"]
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=cols)
    d = pd.DataFrame(index=df.index)
    if "ts" in df:
        d["ts"] = pd.to_datetime(df["ts"], utc=True)
    else:
        d["ts"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    if "yes_price" in df and "yes_side" in df:
        flip = outcome_index == 1
        d["price"] = 1 - df["yes_price"].astype(float) if flip else df["yes_price"].astype(float)
        d["side"] = -df["yes_side"].astype(int) if flip else df["yes_side"].astype(int)
    else:
        side = df["side"]
        if not pd.api.types.is_numeric_dtype(side):
            side = np.where(side.astype(str).str.upper() == "BUY", 1, -1)
        d["price"] = df["price"].astype(float)
        d["side"] = np.asarray(side).astype(int)
        if outcome_index is not None and "outcome_index" in df:
            other = (df["outcome_index"].astype(int) != outcome_index).to_numpy()
            d.loc[other, "price"] = 1 - d.loc[other, "price"]
            d.loc[other, "side"] = -d.loc[other, "side"]
    d["size"] = df["size"].astype(float)
    d = d[(d["price"] > 0) & (d["price"] < 1) & (d["size"] > 0)]
    return d.sort_values("ts").reset_index(drop=True)[cols]


def collapse_sweeps(tr: pd.DataFrame) -> pd.DataFrame:
    """Merge fills with the same timestamp and side (one taker walking the book) into a VWAP print."""
    if tr.empty:
        return tr
    g = tr.assign(notional=tr["price"] * tr["size"]).groupby(["ts", "side"], as_index=False)
    out = g.agg(notional=("notional", "sum"), size=("size", "sum"))
    out["price"] = out["notional"] / out["size"]
    return out.sort_values("ts").reset_index(drop=True)[["ts", "price", "side", "size"]]


# =============================================================================
# Volatility
# =============================================================================
def estimate_sigma2(tr: pd.DataFrame, cfg: FVConfig) -> tuple[float, dict]:
    """Log-odds variance per hour from same-side prints, sampled >= sigma2_min_dt_h apart
    (bursts of prints seconds apart measure noise, not volatility), with the print noise
    2*sigma2_noise_sd^2 per pair removed, then shrunk to the prior."""
    tr = collapse_sweeps(tr)
    ss, st, n = 0.0, 0.0, 0
    for side in (1, -1):
        s = tr[tr["side"] == side]
        last_t = last_L = None
        for t, p in zip(s["ts"], s["price"]):
            L = logit(p)
            if last_t is None:
                last_t, last_L = t, L
                continue
            dt = (t - last_t).total_seconds() / 3600.0
            if dt >= cfg.sigma2_min_dt_h:
                ss += (L - last_L) ** 2
                st += dt
                n += 1
                last_t, last_L = t, L
    ss_clean = max(ss - n * 2 * cfg.sigma2_noise_sd ** 2, 0.0)
    n0 = cfg.sigma2_prior_n
    s2 = (ss_clean + n0 * cfg.sigma2_prior) / (st + n0)
    s2 = min(max(s2, cfg.sigma2_min), cfg.sigma2_max)
    return s2, {"sigma2_pairs": n, "sigma2_hours": round(st, 1),
                "sigma2_raw": round(ss_clean / st, 5) if st > 0 else None}


# =============================================================================
# Estimators
# =============================================================================
def realised_half_spread(tr: pd.DataFrame, k: int = 10) -> Optional[float]:
    """(median logit of recent buys - median logit of recent sells) / 2, or None."""
    b = tr[tr["side"] > 0].tail(k)["price"]
    s = tr[tr["side"] < 0].tail(k)["price"]
    if len(b) < 2 or len(s) < 2:
        return None
    return max(0.5 * (logit(float(b.median())) - logit(float(s.median()))), 0.0)


def trade_estimate(tr: pd.DataFrame, now: pd.Timestamp, sigma2: float, h_book: Optional[float],
                   cfg: FVConfig) -> Optional[Estimate]:
    """Side-corrected, staleness- and size-weighted trade estimate (see module doc)."""
    if tr is None or tr.empty:
        return None
    tr = tr[tr["ts"] >= now - pd.Timedelta(hours=cfg.trade_max_age_h)]
    tr = collapse_sweeps(tr).tail(cfg.trade_max_n)
    if tr.empty:
        return None
    h_tr = realised_half_spread(tr)
    hs = [h for h in (h_book, h_tr) if h is not None]
    # Cap: the CURRENT book (possibly dust 10c/93c) says little about the spread those takers
    # actually crossed, and an uncapped h would shove every one-sided print far from its price.
    h = min(min(hs) if hs else cfg.trade_h_cap, cfg.trade_h_cap)
    alpha = cfg.trade_side_adjust

    L = np.array([logit(p) for p in tr["price"]]) - tr["side"].to_numpy() * alpha * h
    age = np.maximum((now - tr["ts"]).dt.total_seconds().to_numpy() / 3600.0, 0.0)
    cap = np.minimum(np.where(tr["side"] > 0, tr["price"], 1 - tr["price"]) * tr["size"], cfg.trade_size_cap)
    v0 = (h / math.sqrt(12)) ** 2 + cfg.trade_idio_sd ** 2
    v = v0 + sigma2 * age
    w = np.sqrt(np.maximum(cap, 0.01)) / v
    w = w / w.sum()

    x = float(np.dot(w, L))
    common = sigma2 * np.minimum.outer(age, age)           # shared random-walk drift
    var_model = float(w @ common @ w) + float(np.sum(w ** 2) * v0)
    n_eff = 1.0 / float(np.sum(w ** 2))
    # Empirical floor: if the prints disagree more than the model allows (0.04 and 0.29 in the
    # same day), the estimate is only as good as their actual dispersion says.
    disp = float(np.dot(w, (L - x) ** 2)) / max(n_eff - 1.0, 1.0)
    var = max(var_model, disp)
    info = {"n": len(tr), "n_eff": round(n_eff, 1), "newest_age_h": round(float(age.min()), 2),
            "wavg_age_h": round(float(np.dot(w, age)), 2), "h_used": round(h, 3),
            "sd_model": round(math.sqrt(var_model), 3), "sd_dispersion": round(math.sqrt(disp), 3),
            "h_book": None if h_book is None else round(h_book, 3),
            "h_trades": None if h_tr is None else round(h_tr, 3),
            "buys": int((tr["side"] > 0).sum()), "sells": int((tr["side"] < 0).sum())}
    return Estimate("trades", x, var, "trades", info)


def last_trade_estimate(ltp: Optional[float], bid: Optional[float], ask: Optional[float],
                        sigma2: float, h_book: Optional[float], cfg: FVConfig) -> Optional[Estimate]:
    """Fallback when no trade history is given: the book's last_trade_price, with unknown
    side (uniform +-h) and an assumed age. Polymarket sometimes reports the complement
    token's fill price (one book, two tokens): flip it if 1-ltp fits the spread better."""
    if ltp is None or not (0 < ltp < 1):
        return None
    if bid is not None and ask is not None:
        dist = lambda z: max(bid - z, 0.0, z - ask)
        if dist(1 - ltp) < dist(ltp):
            ltp = 1 - ltp
    h = h_book if h_book is not None else 0.25
    var = h * h / 3.0 + cfg.trade_idio_sd ** 2 + sigma2 * cfg.ltp_assumed_age_h
    return Estimate("last_trade", logit(ltp), var, "trades", {"assumed_age_h": cfg.ltp_assumed_age_h})
