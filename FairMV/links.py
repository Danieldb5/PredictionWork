"""
links.py - fair value from correlated / related markets.

A thin market is often priced much better by a liquid relative than by its own
book. Every link turns the other market's local fair value (its own book +
trades, no links, so links never recurse) into an Estimate or a Bound for
this market. Uncertainty is carried through, so a link to another
wide-spread market is automatically weak.

Relations (x = log-odds; o = the other market's estimate, variance v_o)
-----------------------------------------------------------------------
  same        x = x_o                         same event, different market or wording
  complement  x = -x_o                        "A wins" vs "A loses" in another market
  linear      x = alpha + beta x_o            a stable mapping you believe in
  move        x = basis + beta x_o            "moves with": the LEVEL may differ but CHANGES
                                              carry over. basis is learnt online
                                              (local-level Kalman, drift move_basis_q per hour)
                                              from this market's own local estimate,
                                              so the link starts uninformative and earns
                                              trust as the basis settles.
  le / ge     p <= p_o + margin (or >=)       logical bounds: "wins by 10+" <= "wins";
                                              "by March" <= "by June"
  Variance of an estimate link: beta^2 v_o + noise_sd^2 (+ basis variance for move).

Event sums (mutually exclusive outcomes)
----------------------------------------
In a neg-risk event (exactly one leg resolves YES) the legs sum to 1, so each
leg has an independent estimate from all the others:
    p_i = T - sum_{j != i} p_j,   Var = sum_j sd_j^2 + residual_sd^2
T = 1 for neg-risk events. Use T = K for "top K qualify"-style events (or
'auto': T = round(sum of all legs)). residual_sd covers outcomes that are not
listed or not trading ("Other", placeholders). This applies automatically to
neg-risk events (cfg.auto_event_links). In a 2-3 leg event it is often the
best signal there is. In a 20-leg event the variances add up and it fades.
"""

from __future__ import annotations

import fnmatch
import math
from dataclasses import dataclass, field
from typing import Optional, Union

from .core import Bound, Estimate, FVConfig, clip_p, logit, var_logit_from_sd_prob


# =============================================================================
# Specs (what you write in user/links.json)
# =============================================================================
@dataclass
class LinkSpec:
    target: str                       # market this link applies to: slug, slug:Outcome, token id or slug glob
    other: str                        # related market: slug, slug:Outcome or token id
    relation: str = "same"            # same | complement | linear | move | le | ge
    alpha: float = 0.0
    beta: float = 1.0
    noise_sd: float = 0.15            # log-odds sd of the relationship itself (how much you trust it)
    margin: float = 0.0               # prob slack for le/ge
    name: str = ""

    def __post_init__(self):
        if self.relation not in ("same", "complement", "linear", "move", "le", "ge"):
            raise ValueError(f"unknown link relation '{self.relation}'")
        self.name = self.name or f"{self.relation}:{self.other.split(':')[0][:40]}"


@dataclass
class EventSumSpec:
    target: str                       # slug / glob / token id (or '*' with event='...' to cover all legs)
    event: Optional[str] = None       # event slug; default = the target's own event
    total: Union[float, str] = 1.0    # 1 for exclusive outcomes, K for "K of N", or "auto"
    residual_sd: Optional[float] = None
    name: str = ""

    def __post_init__(self):
        self.name = self.name or f"event_sum:{(self.event or 'own')[:40]}"


def target_matches(target: str, meta) -> tuple[bool, bool]:
    """(matches, flip). flip=True means the spec is about the OTHER outcome of this market
    (e.g. spec 'slug' = YES, valuing the NO token), so estimates must be mirrored."""
    if target == meta.token_id:
        return True, False
    if target == meta.condition_id:
        return True, meta.outcome_index == 1
    slug, outcome = (target.rsplit(":", 1) + [None])[:2] if ":" in target else (target, None)
    if not (slug == meta.slug or fnmatch.fnmatch((meta.slug or "").lower(), slug.lower())):
        return False, False
    if outcome is None:
        return True, meta.outcome_index == 1                       # specs default to the first outcome
    return True, (str(meta.outcome).lower() != outcome.lower())


# =============================================================================
# Evaluation
# =============================================================================
@dataclass
class _Basis:
    b: float
    P: float
    t: float


@dataclass
class LinkResult:
    estimates: list = field(default_factory=list)
    bounds: list = field(default_factory=list)
    info: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)


class LinkBook:
    """Holds link specs and the state of 'move' links (their learnt basis)."""

    def __init__(self, links: list = (), events: list = ()):
        self.links: list = list(links)
        self.events: list = list(events)
        self._basis: dict = {}

    def add(self, spec) -> None:
        (self.events if isinstance(spec, EventSumSpec) else self.links).append(spec)

    # -------------------------------------------------------------------------
    def evaluate(self, meta, local: Optional[Estimate], resolve, cfg: FVConfig, now_s: float,
                 provider=None, local_many=None) -> LinkResult:
        """resolve(ref) -> (Estimate in the ref's own token terms, meta) or (None, None).
        local = this market's own (book+trades) estimate, used to learn 'move' bases."""
        out = LinkResult()
        for spec in self.links:
            ok, flip = target_matches(spec.target, meta)
            if not ok:
                continue
            try:
                self._one(spec, flip, meta, local, resolve, cfg, now_s, out)
            except Exception as e:  # noqa: BLE001 - a broken link must never break pricing
                out.warnings.append(f"link {spec.name}: {type(e).__name__}: {e}")
        specs = [s for s in self.events if target_matches(s.target, meta)[0]]
        if cfg.auto_event_links and meta.neg_risk and meta.event_slug and not any(
                (s.event or meta.event_slug) == meta.event_slug for s in specs):
            specs.append(EventSumSpec(target=meta.token_id, event=meta.event_slug, total=1.0,
                                      name=f"event_sum:{meta.event_slug[:40]}"))
        for spec in specs:
            try:
                self._event(spec, meta, cfg, provider, local_many, local, out)
            except Exception as e:  # noqa: BLE001
                out.warnings.append(f"{spec.name}: {type(e).__name__}: {e}")
        return out

    def _one(self, spec: LinkSpec, flip: bool, meta, local, resolve, cfg, now_s, out: LinkResult):
        o, _ = resolve(spec.other)
        if o is None:
            out.warnings.append(f"link {spec.name}: no estimate for '{spec.other}'")
            return
        sgn = -1.0 if flip else 1.0         # spec written for the other outcome: mirror the result
        rel = spec.relation
        if rel in ("le", "ge"):
            po = o.p
            lo, hi = (0.0, po + spec.margin) if rel == "le" else (po - spec.margin, 1.0)   # spec terms
            if flip:                        # valuing the other outcome: p -> 1-p turns <= into >=
                lo, hi = 1.0 - hi, 1.0 - lo
            out.bounds.append(Bound(clip_p(lo, 0.0, 1.0), clip_p(hi, 0.0, 1.0), spec.name))
            out.info[spec.name] = {"other_p": round(po, 4)}
            return
        if rel == "same":
            x, var = o.x, o.var + spec.noise_sd ** 2
        elif rel == "complement":
            x, var = -o.x, o.var + spec.noise_sd ** 2
        elif rel == "linear":
            x, var = spec.alpha + spec.beta * o.x, spec.beta ** 2 * o.var + spec.noise_sd ** 2
        else:  # move
            key = (spec.name, meta.token_id)
            st = self._basis.get(key)
            x_self = None if local is None else sgn * local.x       # basis lives in spec terms
            R = None if local is None else local.var + spec.beta ** 2 * o.var
            if st is None:
                if x_self is None:
                    out.warnings.append(f"link {spec.name}: waiting for a local estimate to learn the basis")
                    return
                st = _Basis(x_self - spec.beta * o.x, R, now_s)
                self._basis[key] = st
                out.info[spec.name] = {"basis": round(st.b, 4), "note": "basis initialised (no information yet)"}
                return   # first observation: the link knows nothing beyond our own estimate
            P = st.P + cfg.move_basis_q * max(now_s - st.t, 0.0) / 3600.0
            x, var = st.b + spec.beta * o.x, P + spec.beta ** 2 * o.var + spec.noise_sd ** 2
            if x_self is not None:                                   # learn the basis AFTER predicting
                K = P / (P + R)
                st.b += K * ((x_self - spec.beta * o.x) - st.b)
                st.P = (1 - K) * P
            else:
                st.P = P
            st.t = now_s
        e = Estimate(f"link:{spec.name}", sgn * x, var, "link",
                     {"other_p": round(o.p, 4), "relation": rel})
        out.estimates.append(e)

    def _event(self, spec: EventSumSpec, meta, cfg, provider, local_many, local, out: LinkResult):
        if provider is None or local_many is None:
            return
        ev = spec.event or meta.event_slug
        if not ev:
            return
        legs = [m for m in provider.event_markets(ev) if m.active and m.tokens]
        mine = [m for m in legs if m.condition_id == meta.condition_id]
        sibs = [m for m in legs if m.condition_id != meta.condition_id]
        if not mine or not sibs:
            return
        ests = local_many([m.tokens[0] for m in sibs])            # YES tokens of the other legs
        missing = [m.slug for m in sibs if ests.get(m.tokens[0]) is None]
        if missing:
            out.warnings.append(f"{spec.name}: no estimate for {len(missing)} leg(s), e.g. {missing[0]}; skipped")
            return
        ps = [ests[m.tokens[0]].p for m in sibs]
        sds = [ests[m.tokens[0]].sd_p for m in sibs]
        total = spec.total
        if total == "auto":
            own = local.p if local is not None else 0.5
            if meta.outcome_index == 1:
                own = 1 - own
            total = max(1.0, float(round(sum(ps) + own)))
        res = cfg.event_residual_sd if spec.residual_sd is None else spec.residual_sd
        p_yes = float(total) - sum(ps)
        var_p = sum(s * s for s in sds) + res * res
        p_c = clip_p(p_yes, 0.002, 0.998)
        var = min(var_logit_from_sd_prob(p_c, math.sqrt(var_p)), 25.0)
        x = logit(p_c)
        if meta.outcome_index == 1:
            x = -x
        out.estimates.append(Estimate(f"link:{spec.name}", x, var, "link",
                                      {"legs": len(sibs) + 1, "total": total, "sum_others": round(sum(ps), 4),
                                       "raw_p": round(p_yes, 4)}))
        out.info[spec.name] = {"legs": len(sibs) + 1, "sum_others": round(sum(ps), 4), "total": total}


def specs_from_json(d: dict) -> tuple[list, list]:
    links = [LinkSpec(**{k: v for k, v in x.items() if not k.startswith("_")}) for x in d.get("links", [])]
    events = [EventSumSpec(**{k: v for k, v in x.items() if not k.startswith("_")}) for x in d.get("event_sums", [])]
    return links, events


__all__ = ["LinkSpec", "EventSumSpec", "LinkBook", "target_matches", "specs_from_json"]
