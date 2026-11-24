"""
combine.py - robust fusion of Estimates in log-odds space.

Precision weighting
-------------------
With estimates x_i of variance v_i, the minimum-variance unbiased combination
weights each by its precision:
    w_i = (1/v_i) / sum_j (1/v_j),     x = sum w_i x_i
A 60c-wide book (v large) therefore gets almost no say when there are fresh
trades or a tight linked market, and full say when it is the only signal.

Correlated errors
-----------------
Signals are not independent (trades happened against this book; your feed
may read the same news as the market). The reported variance uses an
equicorrelation rho (`fusion_corr`):
    Var(x) = sum_ij w_i w_j c_ij sqrt(v_i v_j),   c_ii = 1, c_ij = rho
With rho = 0 this is 1/sum(1/v_i). With rho > 0, piling up similar signals
stops shrinking the uncertainty, so the model does not become overconfident.

Outliers (Student-t reweighting)
--------------------------------
A stale trade, a broken feed, or a link whose relationship has stopped
holding should not drag the answer. Each estimate is compared with the
fusion of all the OTHERS:
    z_i^2 = (x_i - m_{-i})^2 / (v_i + V_{-i})
and its precision is scaled by u_i = min(1, (nu+1)/(nu+z_i^2)) (the EM weight
of a Student-t with nu degrees of freedom), iterated to convergence. An
estimate 4 sd away from the rest keeps about 25% of its weight at nu=4.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from .core import FVConfig, sigmoid


@dataclass
class Fusion:
    x: float
    var: float
    weights: dict = field(default_factory=dict)     # name -> final weight share
    robust: dict = field(default_factory=dict)      # name -> u_i multiplier

    @property
    def p(self) -> float:
        return sigmoid(self.x)


def _combine(xs, vs, rho):
    prec = [1.0 / v for v in vs]
    tot = sum(prec)
    w = [p / tot for p in prec]
    x = sum(wi * xi for wi, xi in zip(w, xs))
    sd = [math.sqrt(v) for v in vs]
    var = 0.0
    for i in range(len(xs)):
        for j in range(len(xs)):
            c = 1.0 if i == j else rho
            var += w[i] * w[j] * c * sd[i] * sd[j]
    return x, var, w


def fuse(estimates: list, cfg: FVConfig) -> Optional[Fusion]:
    ests = [e for e in estimates if e is not None and e.var > 0 and math.isfinite(e.x) and math.isfinite(e.var)]
    if not ests:
        return None
    if len(ests) == 1:
        e = ests[0]
        return Fusion(e.x, e.var, {e.name: 1.0}, {e.name: 1.0})

    xs = [e.x for e in ests]
    base_v = [e.var for e in ests]
    u = [1.0] * len(ests)
    if cfg.robust_nu > 0:
        nu = cfg.robust_nu
        for _ in range(cfg.robust_iters):
            vs = [v / ui for v, ui in zip(base_v, u)]
            new_u = []
            for i in range(len(ests)):
                ox = [x for j, x in enumerate(xs) if j != i]
                ov = [v for j, v in enumerate(vs) if j != i]
                m, V, _ = _combine(ox, ov, 0.0)
                z2 = (xs[i] - m) ** 2 / (base_v[i] + V)
                new_u.append(min(1.0, (nu + 1) / (nu + z2)))
            if max(abs(a - b) for a, b in zip(new_u, u)) < 1e-6:
                u = new_u
                break
            u = new_u

    vs = [v / ui for v, ui in zip(base_v, u)]
    x, var, w = _combine(xs, vs, cfg.fusion_corr)
    return Fusion(x, var, {e.name: wi for e, wi in zip(ests, w)}, {e.name: ui for e, ui in zip(ests, u)})


def apply_bounds(p: float, bounds: list) -> tuple[float, list]:
    """Clip p into every Bound; returns (p, names of bounds that were binding).
    If bounds contradict each other, the later one wins (logged by the caller)."""
    active = []
    for b in bounds:
        if p < b.lo:
            p = b.lo
            active.append(f"{b.source} (>= {b.lo:.3f})")
        elif p > b.hi:
            p = b.hi
            active.append(f"{b.source} (<= {b.hi:.3f})")
    return p, active
