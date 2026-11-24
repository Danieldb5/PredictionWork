"""
book.py - order book and book-based fair value, built for WIDE, THIN books.

Why the plain midpoint fails on low-volume Polymarket markets
-------------------------------------------------------------
Typical scanned book (real example):
    bids 0.11 x 5, 0.04 x 20, ...           asks 0.927 x 26, 0.928 x 28, ...
Mid = 0.52, but neither touch is a real commitment: 5 shares at 0.11 is
$0.55, and the 0.927 ask risks $1.90. The touch is dust, the spread is 80c,
and the "fair value" just reflects who forgot an order where.

What this module does instead
-----------------------------
1. Dust filter (committed capital). A bid at p for s shares risks p*s; an
   ask at p risks (1-p)*s, because selling YES at p is buying NO at 1-p. This
   is symmetric under YES<->NO. The EFFECTIVE touch is the first level where
   cumulative committed capital reaches `dust_notional` ($10 by default).

2. Four centre estimators on the effective book, blended in log-odds:
   a. Glosten-Milgrom logit mid. With a share mu of informed traders,
      logit(a) = logit(p) + L and logit(b) = logit(p) - L, so
      p* = sigmoid((logit a + logit b)/2) and mu* = tanh((logit a - logit b)/4).
   b. Depth microprice. Distance-weighted capital imbalance I, SHRUNK toward
      1/2 as the spread widens. The microprice (Stoikov 2018) is a
      one-tick-spread result: in a 50-tick spread, a 3000-share wall at the
      bid says little about where inside the gap the value is. Shrink factor
      min(1, imbalance_full_ticks / spread_ticks).
   c. Belief MLE (new). Treat every resting order as a noisy, one-sided
      statement of belief. A bid at L_j says "value > L_j", an ask says
      "value < L_j". Logistic likelihood:
          ell(x) = sum_bids w_j ln s((x - L_j)/h) + sum_asks w_j ln s((L_j - x)/h)
      with weight w_j = ln(1 + capital_j / ref) * exp(-dist_j / decay). The
      log grows slowly, so a whale wall cannot dominate; the decay downweights
      far-away orders. ell is concave, so Newton finds the unique maximum. It
      uses every order on both sides, not just the two touches. The scale h is
      a fraction (belief_h_frac) of the logit half-gap. In that regime the
      optimum is approximately
          x ~ logit-mid + (h/2) ln(W_bid / W_ask)
      so lopsided capital moves the estimate only LOGARITHMICALLY and by a
      bounded amount: a $700 bid wall against $10 of asks moves it ~1/4 of
      the way to the ask, not all the way.
   d. Impact mid. Logit mid of the VWAPs to trade `impact_notional` dollars of
      capital on each side (Kyle depth).

3. Uncertainty. Inside the spread the book alone cannot locate the value, so
   the book's variance is that of a UNIFORM over the effective spread in
   log-odds, (L_ask - L_bid)^2 / 12, plus disagreement between the four
   centres. A one-tick book gets a small variance; a 60c book gets a huge one
   and the other signals (trades, linked markets, your views) take over in
   the fusion step.

Hard bound: the ACTUAL best bid/ask (dust included, since you can trade
against it) is the no-arbitrage bound applied at the very end (engine.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional

from .core import Estimate, FVConfig, logit, sigmoid


# =============================================================================
# Order book
# =============================================================================
@dataclass
class OrderBook:
    """Bids sorted best (highest) first; asks sorted best (lowest) first."""

    bids: list
    asks: list
    tick_size: float = 0.01
    last_trade: Optional[float] = None
    token_id: str = ""
    timestamp: Optional[float] = None
    min_order_size: float = 5.0

    @classmethod
    def from_levels(cls, bids: Iterable, asks: Iterable, **kw) -> "OrderBook":
        b = sorted(((float(p), float(s)) for p, s in bids if float(s) > 0), key=lambda x: -x[0])
        a = sorted(((float(p), float(s)) for p, s in asks if float(s) > 0), key=lambda x: x[0])
        return cls(bids=b, asks=a, **kw)

    @classmethod
    def from_clob_json(cls, d: dict) -> "OrderBook":
        ltp, ts, mos = d.get("last_trade_price"), d.get("timestamp"), d.get("min_order_size")
        return cls.from_levels(
            bids=[(x["price"], x["size"]) for x in d.get("bids", [])],
            asks=[(x["price"], x["size"]) for x in d.get("asks", [])],
            tick_size=float(d.get("tick_size") or 0.01),
            last_trade=float(ltp) if ltp not in (None, "") else None,
            token_id=str(d.get("asset_id", "")),
            timestamp=float(ts) / 1000 if ts else None,
            min_order_size=float(mos) if mos not in (None, "") else 5.0,
        )

    def complement(self) -> "OrderBook":
        """The mirrored book of the other outcome (YES <-> NO)."""
        return OrderBook.from_levels(
            bids=[(1 - p, s) for p, s in self.asks],
            asks=[(1 - p, s) for p, s in self.bids],
            tick_size=self.tick_size,
            last_trade=None if self.last_trade is None else 1 - self.last_trade,
            timestamp=self.timestamp, min_order_size=self.min_order_size,
        )

    @property
    def best_bid(self) -> Optional[float]:
        return self.bids[0][0] if self.bids else None

    @property
    def best_ask(self) -> Optional[float]:
        return self.asks[0][0] if self.asks else None

    @property
    def spread(self) -> Optional[float]:
        if self.bids and self.asks:
            return self.best_ask - self.best_bid
        return None

    @property
    def mid(self) -> Optional[float]:
        return None if self.spread is None else 0.5 * (self.best_bid + self.best_ask)


# =============================================================================
# Primitives
# =============================================================================
def capital(price: float, size: float, side: str) -> float:
    """$ at risk: a bid risks p*s, an ask risks (1-p)*s (selling YES = buying NO)."""
    return size * (price if side == "bid" else 1.0 - price)


def effective_touch(levels: list, side: str, dust_notional: float) -> tuple[Optional[float], bool]:
    """Price of the first level where cumulative committed capital >= dust_notional.
    Returns (price, thin) where thin=True means the whole side is below the threshold
    (then the raw touch is returned: it is all the information there is)."""
    if not levels:
        return None, True
    cum = 0.0
    for p, s in levels:
        cum += capital(p, s, side)
        if cum >= dust_notional - 1e-12:
            return p, False
    return levels[0][0], True


def glosten_milgrom(bid: float, ask: float) -> tuple[float, float]:
    """Invert symmetric Glosten-Milgrom quotes -> (fair prob p*, informed share mu*)."""
    lb, la = logit(bid), logit(ask)
    return sigmoid(0.5 * (la + lb)), math.tanh(0.25 * max(la - lb, 0.0))


def depth_imbalance(book: OrderBook, eb: float, ea: float, window: float, decay: float) -> float:
    """Capital imbalance within `window` of the effective touch, exp-decayed with distance.
    Orders in front of the effective touch (dust) count at full weight but are tiny anyway."""
    wb = sum(capital(p, s, "bid") * math.exp(-max(eb - p, 0.0) / decay)
             for p, s in book.bids if eb - p <= window + 1e-12)
    wa = sum(capital(p, s, "ask") * math.exp(-max(p - ea, 0.0) / decay)
             for p, s in book.asks if p - ea <= window + 1e-12)
    return wb / (wb + wa) if wb + wa > 0 else 0.5


def belief_mle(book: OrderBook, eb: float, ea: float, cfg: FVConfig) -> Optional[float]:
    """Maximum-likelihood value from all resting orders as one-sided beliefs (see module doc).
    Returns a logit, or None if a side is empty."""
    terms = []   # (L_j, w_j, +1 bid / -1 ask)
    for side, levels, ref, sgn in (("bid", book.bids, eb, +1), ("ask", book.asks, ea, -1)):
        for p, s in levels:
            dist = (ref - p) if sgn > 0 else (p - ref)
            if dist > cfg.depth_window + 1e-12:
                break
            w = math.log1p(capital(p, s, side) / cfg.belief_ref_notional) * math.exp(-max(dist, 0.0) / cfg.depth_decay)
            if w > 0:
                terms.append((logit(p), w, sgn))
    if not any(t[2] > 0 for t in terms) or not any(t[2] < 0 for t in terms):
        return None
    lb, la = logit(eb), logit(ea)
    # Small scale h: inside the gap the optimum is x ~ mid + (h/2) ln(W_bid / W_ask), so the pull of
    # a lopsided book is LOGARITHMIC in the capital ratio and bounded by h. A large h would let a
    # whale wall drag the estimate past the opposite quote.
    h = max(cfg.belief_h_frac * 0.5 * (la - lb), 0.02)
    x = 0.5 * (la + lb)
    lo, hi = logit(book.best_bid) - 3, logit(book.best_ask) + 3
    for _ in range(50):
        g = hss = 0.0
        for L, w, sgn in terms:
            z = sgn * (x - L) / h            # bid: (x-L)/h, ask: (L-x)/h
            s = sigmoid(z)
            g += w * sgn * (1 - s) / h       # d/dx ln sigmoid(z) = sgn (1-s)/h
            hss -= w * s * (1 - s) / (h * h)
        if hss >= -1e-12:
            break
        step = -g / hss
        step = max(min(step, 1.0), -1.0)      # damped Newton
        x = min(max(x + step, lo), hi)
        if abs(step) < 1e-9:
            break
    return x


def vwap_capital(levels: list, side: str, notional: float) -> tuple[Optional[float], bool]:
    """VWAP of walking `levels` until `notional` $ of committed capital is taken."""
    rem, cost, shares = notional, 0.0, 0.0
    for p, s in levels:
        per_share = p if side == "bid" else 1.0 - p
        if per_share <= 0:
            continue
        take = min(s, rem / per_share)
        cost += take * p
        shares += take
        rem -= take * per_share
        if rem <= 1e-9:
            return cost / shares, True
    return (cost / shares if shares > 0 else None), False


# =============================================================================
# Book estimate
# =============================================================================
@dataclass
class BookAnalysis:
    estimate: Optional[Estimate]
    eff_bid: Optional[float]
    eff_ask: Optional[float]
    components: dict
    diagnostics: dict
    warnings: list


def analyse_book(book: OrderBook, cfg: FVConfig) -> BookAnalysis:
    b, a = book.best_bid, book.best_ask
    warnings: list = []
    if b is None or a is None:
        warnings.append("one-sided or empty book: no book estimate")
        return BookAnalysis(None, b, a, {}, {"one_sided": True}, warnings)
    if a <= b:
        warnings.append("crossed/locked book: book estimate = mid with large variance")
        m = 0.5 * (a + b)
        return BookAnalysis(Estimate("book", logit(m), 0.25, "book"), b, a, {"mid": m}, {}, warnings)

    eb, thin_b = effective_touch(book.bids, "bid", cfg.dust_notional)
    ea, thin_a = effective_touch(book.asks, "ask", cfg.dust_notional)
    if ea <= eb:   # can only happen with odd inputs; fall back to the touch
        eb, ea = b, a
    lb, la = logit(eb), logit(ea)
    spread_ticks = max((ea - eb) / book.tick_size, 1.0)

    p_gm, mu = glosten_milgrom(eb, ea)
    imb = depth_imbalance(book, eb, ea, cfg.depth_window, cfg.depth_decay)
    shrink = min(1.0, cfg.imbalance_full_ticks / spread_ticks)
    imb_eff = 0.5 + (imb - 0.5) * shrink
    x_depth = imb_eff * la + (1 - imb_eff) * lb
    x_belief = belief_mle(book, eb, ea, cfg)
    # impact VWAPs only within depth_window of the effective touch: on a thin side, walking $50
    # could reach 60c deep, which measures emptiness, not value
    vb, fb = vwap_capital([(p, s) for p, s in book.bids if eb - p <= cfg.depth_window + 1e-12],
                          "bid", cfg.impact_notional)
    va, fa = vwap_capital([(p, s) for p, s in book.asks if p - ea <= cfg.depth_window + 1e-12],
                          "ask", cfg.impact_notional)
    x_imp = 0.5 * (logit(vb) + logit(va)) if vb is not None and va is not None else None

    parts = [("glosten_milgrom", cfg.w_gm, logit(p_gm)), ("microprice_depth", cfg.w_depth, x_depth),
             ("belief_mle", cfg.w_belief, x_belief), ("impact_mid", cfg.w_impact, x_imp)]
    parts = [(n, w, x) for n, w, x in parts if x is not None and w > 0]
    ws = sum(w for _, w, _ in parts)
    x = sum(w * v for _, w, v in parts) / ws
    disagreement = sum(w * (v - x) ** 2 for _, w, v in parts) / ws

    var = (la - lb) ** 2 / 12.0 + disagreement
    if thin_b or thin_a:
        var *= cfg.thin_side_var_mult
        warnings.append("a book side has < $%.0f committed: treated as dust" % cfg.dust_notional)
    var = max(var, cfg.book_var_floor)
    # clamp the centre into the effective spread (and therefore the real one)
    x = min(max(x, lb), la)

    comps = {"arithmetic_mid": 0.5 * (a + b), "logit_mid_touch": glosten_milgrom(b, a)[0]}
    comps.update({n: sigmoid(v) for n, _, v in parts})
    diags = {
        "eff_bid": eb, "eff_ask": ea, "dust_bid": eb != b, "dust_ask": ea != a,
        "thin_bid": thin_b, "thin_ask": thin_a,
        "informed_mu (GM)": mu, "imbalance_depth": imb, "imbalance_used": imb_eff,
        "spread_ticks": round((a - b) / book.tick_size),
        "eff_spread": ea - eb,
        "impact_filled": bool(fb and fa),
        "depth_bid_shares": round(sum(s for p, s in book.bids if eb - p <= cfg.depth_window + 1e-12), 1),
        "depth_ask_shares": round(sum(s for p, s in book.asks if p - ea <= cfg.depth_window + 1e-12), 1),
        "bid_capital": round(sum(capital(p, s, "bid") for p, s in book.bids), 2),
        "ask_capital": round(sum(capital(p, s, "ask") for p, s in book.asks), 2),
    }
    est = Estimate("book", x, var, "book", {"eff_bid": eb, "eff_ask": ea})
    return BookAnalysis(est, eb, ea, comps, diags, warnings)
