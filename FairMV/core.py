"""
core.py - shared maths, the Estimate type and the configuration.

Everything inside the package works in LOG-ODDS (logit) space:

    x = logit(p) = ln(p / (1 - p))

Reasons:
  * a binary price lives in (0, 1) but log-odds live on the whole real line,
    so Gaussian errors and inverse-variance averaging make sense there;
  * the Glosten-Milgrom spread is symmetric in log-odds (see book.py);
  * one cent means much more near 0.02 than near 0.50, and log-odds account
    for that automatically.

Each signal (book, trades, a correlated market, one of your feeds) produces
an `Estimate`: a mean `x` and a variance `var`, both in log-odds. The
variance is what makes the model robust: a 60c-wide spread gives the book a
huge variance, so trades, linked markets and your views take over.

Convert a logit sd to a probability sd with the delta method:
    sd_p ~= p (1 - p) sd_x
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from typing import Optional

EPS = 1e-6          # keeps logit finite at the boundaries
P_MIN, P_MAX = 0.001, 0.999   # Polymarket's tradable range (tick 0.001 markets)


# =============================================================================
# Maths
# =============================================================================
def logit(p: float) -> float:
    p = min(max(float(p), EPS), 1.0 - EPS)
    return math.log(p / (1.0 - p))


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def clip_p(p: float, lo: float = P_MIN, hi: float = P_MAX) -> float:
    return min(max(p, lo), hi)


def sd_prob(x: float, var: float) -> float:
    """Probability-space sd of a logit-normal estimate (delta method)."""
    p = sigmoid(x)
    return p * (1 - p) * math.sqrt(max(var, 0.0))


def var_logit_from_sd_prob(p: float, sd_p: float) -> float:
    """Inverse of sd_prob: logit variance equivalent to a probability sd at p."""
    p = clip_p(p, 0.005, 0.995)
    return (sd_p / (p * (1 - p))) ** 2


def shift_logit_from_prob(p: float, dp: float) -> float:
    """Logit shift that moves probability p by dp (exact, not linearised)."""
    return logit(clip_p(p + dp)) - logit(clip_p(p))


# =============================================================================
# Estimate
# =============================================================================
@dataclass
class Estimate:
    """One signal about fair value: mean x and variance var, both in log-odds."""
    name: str
    x: float
    var: float
    kind: str = "other"          # book / trades / link / feed / prior
    info: dict = field(default_factory=dict)

    @property
    def p(self) -> float:
        return sigmoid(self.x)

    @property
    def sd_p(self) -> float:
        return sd_prob(self.x, self.var)

    @classmethod
    def from_prob(cls, name: str, p: float, sd_p: float, kind: str = "other", **info) -> "Estimate":
        p = clip_p(p)
        return cls(name, logit(p), var_logit_from_sd_prob(p, sd_p), kind, dict(info))

    def __repr__(self) -> str:
        return f"Estimate({self.name}: p={self.p:.4f} sd_p={self.sd_p:.4f} [{self.kind}])"


@dataclass
class Bound:
    """A hard constraint lo <= fair value <= hi (probabilities), from a link or a view."""
    lo: float = 0.0
    hi: float = 1.0
    source: str = ""


# =============================================================================
# Configuration
# =============================================================================
@dataclass
class FVConfig:
    """All tunable parameters. Every field can be overridden from user/config.json."""

    # --- book (book.py) ------------------------------------------------------
    dust_notional: float = 10.0      # $: orders in front of this much notional are dust
    depth_window: float = 0.10       # price distance from the effective touch to use
    depth_decay: float = 0.03        # exp decay (price units) for deeper levels
    impact_notional: float = 50.0    # $: size for the impact (VWAP) mid
    belief_ref_notional: float = 25.0  # $: notional at which an order counts as "one belief"
    belief_h_frac: float = 0.15      # belief-MLE logistic scale as a fraction of the logit half-gap
    imbalance_full_ticks: float = 2.0  # microprice imbalance is used fully up to this spread
    w_gm: float = 0.35               # book-centre blend weights (logit space)
    w_depth: float = 0.20
    w_belief: float = 0.30
    w_impact: float = 0.15
    book_var_floor: float = 0.0004   # logit var floor (sd 0.02 = 0.5c at p=0.5)
    thin_side_var_mult: float = 2.0  # var multiplier when one side is all dust

    # --- trades (trades.py) --------------------------------------------------
    trade_max_age_h: float = 24 * 7  # ignore trades older than this
    trade_max_n: int = 60            # use the most recent N trades
    trade_side_adjust: float = 0.5   # fraction of half-spread removed from each trade
                                     # (0 = Glosten-Milgrom: trade price IS the new value,
                                     #  1 = noise taker: value is the mid)
    trade_h_cap: float = 0.25        # max log-odds half-spread used for the side correction
    trade_size_cap: float = 500.0    # $: notional cap for a trade's weight (no whale dominance)
    trade_idio_sd: float = 0.20      # logit sd of one trade around value besides the spread
                                     # (5c at p=0.5: thin-market prints are noisy)
    ltp_assumed_age_h: float = 12.0  # last_trade_price from the book has no timestamp
    sigma2_prior: float = 0.004      # logit var / hour prior (sd 0.063/sqrt(h) = 1.6c/sqrt(h) at 0.5)
    sigma2_prior_n: float = 10.0     # hours of pseudo-data behind the prior
    sigma2_min_dt_h: float = 1.0     # sample same-side prints at least this far apart
    sigma2_noise_sd: float = 0.08    # log-odds print noise removed from each sampled pair
    sigma2_min: float = 0.0005
    sigma2_max: float = 0.10

    # --- fusion (combine.py) -------------------------------------------------
    fusion_corr: float = 0.20        # assumed error correlation between signals
    robust_nu: float = 4.0           # Student-t dof for outlier down-weighting (0 = off)
    robust_iters: int = 8

    # --- tracker (tracker.py) ------------------------------------------------
    use_tracker: bool = True
    tracker_floor: float = 0.50      # posterior var >= floor * measurement var (repeat looks at a stale book are not new info)
    tracker_jump_z: float = 4.0      # innovation z above which we reset (news)

    # --- links (links.py) ----------------------------------------------------
    use_links: bool = True
    auto_event_links: bool = True    # neg-risk events: legs sum to 1
    event_residual_sd: float = 0.02  # prob sd for unlisted / "other" outcomes
    move_basis_q: float = 0.002      # logit var / hour drift of a "move" link's basis
    link_cache_ttl: float = 20.0     # s: cache other markets' estimates

    # --- feeds (feeds.py) ----------------------------------------------------
    use_feeds: bool = True
    feed_default_sd: float = 0.05    # prob sd when a view gives none

    # --- output --------------------------------------------------------------
    clamp_to_book: bool = True       # no-arbitrage: fair value inside [best bid, best ask]
    conf_levels: tuple = (0.01, 0.025, 0.05, 0.10)   # prob sd for high/med/low/very low

    @classmethod
    def from_dict(cls, d: dict) -> "FVConfig":
        names = {f.name for f in fields(cls)}
        unknown = set(d) - names - {"_doc"}
        if unknown:
            raise ValueError(f"unknown FairMV config keys: {sorted(unknown)}")
        kw = {k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items() if k in names}
        return cls(**kw)

    def confidence(self, sd_p: float) -> str:
        h, m, l, vl = self.conf_levels
        if sd_p <= h:
            return "high"
        if sd_p <= m:
            return "medium"
        if sd_p <= l:
            return "low"
        if sd_p <= vl:
            return "very low"
        return "none"


def opt_float(v) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f
