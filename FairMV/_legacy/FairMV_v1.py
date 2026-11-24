"""
FairMV.py - Fair market value model for Polymarket binary outcome tokens,
built from order book data and the bid-ask spread.

-------------------------------------------------------------------------------
THEORY
-------------------------------------------------------------------------------
A Polymarket outcome token pays $1 if the outcome happens and $0 otherwise, so
its fair value V* is a probability in (0, 1). The order book gives us:

  b, a          best bid / best ask
  Q_b, Q_a      size resting at the best bid / ask
  deeper levels (price, size) on both sides

We combine four estimators, each based on a standard market microstructure
model:

1. Glosten-Milgrom (1985) logit midpoint, the spread as adverse selection
   -----------------------------------------------------------------------
   A competitive market maker trades with a fraction mu of informed traders
   (who know V in {0,1}) and 1-mu noise traders (who buy/sell with prob 1/2).
   Zero-profit quotes are conditional expectations:

       a = P(V=1 | buy)  = p(1+mu) / [p(1+mu) + (1-p)(1-mu)]
       b = P(V=1 | sell) = p(1-mu) / [p(1-mu) + (1-p)(1+mu)]

   In log-odds space this is exactly symmetric:

       logit(a) = logit(p) + L,   logit(b) = logit(p) - L,
       L = ln((1+mu)/(1-mu))

   Inverting gives closed forms for the prior fair value and the informed
   trading intensity implied by the spread:

       p*  = sigmoid( (logit(a) + logit(b)) / 2 )      <- "logit mid"
       mu* = tanh( (logit(a) - logit(b)) / 4 )

   So in a binary market the arithmetic mid is biased. Near 0 or 1 the
   theoretically correct centre of the spread is the logit midpoint, which
   sits toward the nearer boundary.
   Example: b=0.90, a=0.96 gives mid=0.930 but logit mid=0.936 (mu*=0.24).

2. Microprice (Gatheral & Oomen 2010; Stoikov 2018), queue imbalance
   ------------------------------------------------------------------
   The next mid-price move is predicted by top-of-book imbalance
   I = Q_b / (Q_b + Q_a). A thin ask queue is likely to be depleted, so price
   is likely to tick up. The weighted mid

       P_micro = I * a + (1 - I) * b

   is a martingale-corrected estimate of where the mid is heading. We apply it
   in logit space so it stays consistent with (1).

3. Depth-weighted imbalance (multi-level microprice)
   -------------------------------------------------
   Top-of-book size on Polymarket is noisy: dust orders and 5-share minimums
   are common. We compute imbalance from all levels within `depth_window` of
   the touch, discounting each level by exp(-d / depth_scale), where d is its
   distance from the touch (Cao, Hansch & Wang 2009 show that deeper levels
   carry information).

4. Impact midpoint (VWAP to trade a reference size)
   -------------------------------------------------
   The average execution price to buy / sell `impact_size` shares is the
   "effective" ask / bid a real trader faces (Kyle 1985: price impact as
   depth). Its logit midpoint is robust to tiny orders sitting at the touch.

Combination and bounds
----------------------
The estimators are blended in logit space with configurable weights. The
result is then clamped to [b, a]: if the fair value were outside the
spread, hitting the quote would have positive expected value, so the
quote would not survive. This is a no-arbitrage bound.

Confidence is reported from:
  * spread width (Polymarket itself stops showing the mid when the spread
    exceeds $0.10 and shows the last trade instead)
  * mu* (high implied informed-trading intensity => quotes are defensive and
    the FV is less precise)
  * depth on each side

YES/NO consistency: Polymarket runs one book for both outcomes (a NO bid at x
is a YES ask at 1-x), so FV(YES) + FV(NO) should equal 1. The CLI checks this.

-------------------------------------------------------------------------------
USAGE
-------------------------------------------------------------------------------
  py FairMV.py --slug <market-slug>          # both outcomes of a market
  py FairMV.py --token <clob-token-id>       # one outcome token
  py FairMV.py --slug <slug> --watch 5       # refresh every 5 s
  py FairMV.py --top 10                      # the 10 highest-volume markets
  py FairMV.py --demo                        # offline synthetic example

As a library:
  from FairMV import OrderBook, FairValueModel
  book = OrderBook.from_levels(bids=[(0.48, 900)], asks=[(0.50, 300)])
  print(FairValueModel().evaluate(book))
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field
from typing import Iterable, Optional

try:
    import requests
except ImportError:  # only needed for live data
    requests = None

CLOB_URL = "https://clob.polymarket.com"
GAMMA_URL = "https://gamma-api.polymarket.com"

EPS = 1e-6  # keeps logit finite at the 0 / 1 boundaries


# =============================================================================
# Helpers
# =============================================================================
def logit(p: float) -> float:
    p = min(max(p, EPS), 1.0 - EPS)
    return math.log(p / (1.0 - p))


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


# =============================================================================
# Order book
# =============================================================================
@dataclass
class OrderBook:
    """Bids sorted best (highest) first; asks sorted best (lowest) first."""

    bids: list[tuple[float, float]]
    asks: list[tuple[float, float]]
    tick_size: float = 0.01
    last_trade: Optional[float] = None
    token_id: str = ""
    timestamp: Optional[float] = None

    @classmethod
    def from_levels(cls, bids: Iterable, asks: Iterable, **kw) -> "OrderBook":
        b = sorted(((float(p), float(s)) for p, s in bids if float(s) > 0),
                   key=lambda x: -x[0])
        a = sorted(((float(p), float(s)) for p, s in asks if float(s) > 0),
                   key=lambda x: x[0])
        return cls(bids=b, asks=a, **kw)

    @classmethod
    def from_clob_json(cls, d: dict) -> "OrderBook":
        ltp = d.get("last_trade_price")
        ts = d.get("timestamp")
        return cls.from_levels(
            bids=[(x["price"], x["size"]) for x in d.get("bids", [])],
            asks=[(x["price"], x["size"]) for x in d.get("asks", [])],
            tick_size=float(d.get("tick_size") or 0.01),
            last_trade=float(ltp) if ltp not in (None, "") else None,
            token_id=d.get("asset_id", ""),
            timestamp=float(ts) / 1000 if ts else None,
        )

    def complement(self) -> "OrderBook":
        """The mirrored book of the other outcome (YES <-> NO)."""
        return OrderBook.from_levels(
            bids=[(1 - p, s) for p, s in self.asks],
            asks=[(1 - p, s) for p, s in self.bids],
            tick_size=self.tick_size,
            last_trade=None if self.last_trade is None else 1 - self.last_trade,
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


# =============================================================================
# Estimators (each returns a probability in (0,1) or None)
# =============================================================================
def arithmetic_mid(book: OrderBook) -> Optional[float]:
    if book.spread is None:
        return None
    return 0.5 * (book.best_bid + book.best_ask)


def glosten_milgrom(bid: float, ask: float) -> tuple[float, float]:
    """Invert symmetric Glosten-Milgrom quotes -> (fair prob p*, informed share mu*)."""
    lb, la = logit(bid), logit(ask)
    p = sigmoid(0.5 * (la + lb))
    mu = math.tanh(0.25 * max(la - lb, 0.0))
    return p, mu


def imbalance_top(book: OrderBook) -> Optional[float]:
    if not (book.bids and book.asks):
        return None
    qb, qa = book.bids[0][1], book.asks[0][1]
    return qb / (qb + qa)


def imbalance_depth(book: OrderBook, depth_window: float, depth_scale: float) -> Optional[float]:
    """Exponentially distance-weighted multi-level queue imbalance."""
    if not (book.bids and book.asks):
        return None
    b0, a0 = book.best_bid, book.best_ask
    wb = sum(s * math.exp(-(b0 - p) / depth_scale)
             for p, s in book.bids if b0 - p <= depth_window + 1e-12)
    wa = sum(s * math.exp(-(p - a0) / depth_scale)
             for p, s in book.asks if p - a0 <= depth_window + 1e-12)
    return wb / (wb + wa) if wb + wa > 0 else 0.5


def logit_microprice(bid: float, ask: float, imb: float) -> float:
    """Microprice I*a + (1-I)*b evaluated in log-odds space."""
    return sigmoid(imb * logit(ask) + (1 - imb) * logit(bid))


def vwap_fill(levels: list[tuple[float, float]], size: float) -> tuple[Optional[float], float]:
    """Average price for walking `size` shares through `levels`. Returns (vwap, filled)."""
    remaining, cost = size, 0.0
    for p, s in levels:
        take = min(s, remaining)
        cost += take * p
        remaining -= take
        if remaining <= 1e-12:
            break
    filled = size - remaining
    return (cost / filled if filled > 0 else None), filled


def impact_mid(book: OrderBook, size: float) -> tuple[Optional[float], bool]:
    """Logit-mid of the VWAP bid and VWAP ask for `size` shares. Returns (fv, fully_filled)."""
    vb, fb = vwap_fill(book.bids, size)
    va, fa = vwap_fill(book.asks, size)
    if vb is None or va is None:
        return None, False
    return sigmoid(0.5 * (logit(vb) + logit(va))), (fb >= size and fa >= size)


# =============================================================================
# Combined model
# =============================================================================
@dataclass
class FairValueResult:
    fair_value: Optional[float]
    bid: Optional[float]
    ask: Optional[float]
    spread: Optional[float]
    confidence: str
    components: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)

    def __str__(self) -> str:
        f = lambda x, n=4: "  n/a " if x is None else f"{x:.{n}f}"
        lines = [
            f"  Fair value : {f(self.fair_value)}   (confidence: {self.confidence})",
            f"  Bid / Ask  : {f(self.bid, 3)} / {f(self.ask, 3)}   spread {f(self.spread, 3)}",
            "  Components :",
        ]
        for k, v in self.components.items():
            lines.append(f"     {k:<20s} {f(v)}")
        lines.append("  Diagnostics:")
        for k, v in self.diagnostics.items():
            vs = f"{v:.4f}" if isinstance(v, float) else str(v)
            lines.append(f"     {k:<20s} {vs}")
        for w in self.warnings:
            lines.append(f"  ! {w}")
        return "\n".join(lines)


@dataclass
class FairValueModel:
    # Blend weights (applied in logit space; renormalised over available estimators)
    w_gm: float = 0.20          # Glosten-Milgrom logit mid
    w_micro: float = 0.25       # top-of-book microprice
    w_depth: float = 0.30       # depth-weighted microprice
    w_impact: float = 0.25      # impact (VWAP) mid
    # Parameters
    depth_window: float = 0.05  # price distance from touch to include (5c)
    depth_scale: float = 0.02   # exp decay length for deeper levels (2c)
    impact_size: float = 200.0  # shares used for the impact mid
    wide_spread: float = 0.10   # Polymarket's own "spread too wide" threshold

    def evaluate(self, book: OrderBook) -> FairValueResult:
        b, a = book.best_bid, book.best_ask
        res = FairValueResult(None, b, a, book.spread, "none")

        if b is None or a is None:
            # One-sided / empty book: nothing structural to say; fall back to last trade
            res.warnings.append("one-sided or empty book; falling back to last trade")
            res.fair_value = book.last_trade
            res.confidence = "very low"
            if b is not None and book.last_trade is not None:
                res.fair_value = max(book.last_trade, b)
            if a is not None and book.last_trade is not None:
                res.fair_value = min(book.last_trade, a)
            return res

        if a <= b:
            res.warnings.append("crossed/locked book (a <= b); using arithmetic mid")
            res.fair_value = 0.5 * (a + b)
            res.confidence = "low"
            return res

        # Polymarket's last_trade_price is sometimes the complementary token's
        # fill price (one book, two tokens). Flip it if 1-ltp fits the spread better.
        ltp = book.last_trade
        if ltp is not None:
            dist = lambda x: max(b - x, 0.0, x - a)
            if dist(1 - ltp) < dist(ltp):
                ltp = 1 - ltp

        # --- component estimators -------------------------------------------
        p_gm, mu = glosten_milgrom(b, a)
        I_top = imbalance_top(book)
        I_dep = imbalance_depth(book, self.depth_window, self.depth_scale)
        p_micro = logit_microprice(b, a, I_top)
        p_depth = logit_microprice(b, a, I_dep)
        p_imp, imp_full = impact_mid(book, self.impact_size)

        res.components = {
            "arithmetic_mid": arithmetic_mid(book),
            "glosten_milgrom": p_gm,
            "microprice_top": p_micro,
            "microprice_depth": p_depth,
            "impact_mid": p_imp,
            "last_trade": ltp,
        }

        # --- blend in logit space -------------------------------------------
        parts = [(self.w_gm, p_gm), (self.w_micro, p_micro),
                 (self.w_depth, p_depth), (self.w_impact, p_imp)]
        parts = [(w, p) for w, p in parts if p is not None and w > 0]
        wsum = sum(w for w, _ in parts)
        fv = sigmoid(sum(w * logit(p) for w, p in parts) / wsum)

        # --- no-arbitrage bound ----------------------------------------------
        fv = min(max(fv, b), a)
        res.fair_value = fv

        # --- diagnostics -----------------------------------------------------
        depth_b = sum(s for p, s in book.bids if b - p <= self.depth_window)
        depth_a = sum(s for p, s in book.asks if p - a <= self.depth_window)
        spread = a - b
        res.diagnostics = {
            "informed_mu (GM)": mu,
            "imbalance_top": I_top,
            "imbalance_depth": I_dep,
            "spread_ticks": round(spread / book.tick_size),
            "logit_half_spread": 0.5 * (logit(a) - logit(b)),
            "depth_bid_shares": round(depth_b, 1),
            "depth_ask_shares": round(depth_a, 1),
            "fv_minus_mid": fv - 0.5 * (a + b),
        }

        # --- confidence --------------------------------------------------------
        if spread > self.wide_spread:
            res.confidence = "low"
            res.warnings.append(
                f"spread {spread:.3f} > {self.wide_spread:.2f}: Polymarket would show last trade")
        elif spread <= 2 * book.tick_size + 1e-9 and mu < 0.5 and imp_full:
            res.confidence = "high"
        else:
            res.confidence = "medium"
        if not imp_full:
            res.warnings.append(
                f"book too thin to fill {self.impact_size:.0f} shares on at least one side")
        return res


# =============================================================================
# Polymarket data access
# =============================================================================
def _get(url: str, params: dict) -> object:
    if requests is None:
        raise RuntimeError("`requests` not installed: pip install requests")
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    return r.json()


def fetch_book(token_id: str) -> OrderBook:
    try:
        return OrderBook.from_clob_json(_get(f"{CLOB_URL}/book", {"token_id": token_id}))
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            raise RuntimeError("no order book (market closed/resolved or token id wrong)") from None
        raise


def _parse_market(m: dict) -> dict:
    load = lambda v: json.loads(v) if isinstance(v, str) else (v or [])
    return {"question": m.get("question", ""), "slug": m.get("slug", ""),
            "outcomes": load(m.get("outcomes")), "tokens": load(m.get("clobTokenIds"))}


def fetch_market(slug: str) -> dict:
    data = _get(f"{GAMMA_URL}/markets", {"slug": slug})
    if not data:
        # maybe it is an event slug containing several markets
        ev = _get(f"{GAMMA_URL}/events", {"slug": slug})
        if ev and ev[0].get("markets"):
            names = [m["slug"] for m in ev[0]["markets"]]
            raise SystemExit(f"'{slug}' is an event; pick a market slug:\n  " + "\n  ".join(names))
        raise SystemExit(f"No market found for slug '{slug}'")
    return _parse_market(data[0])


def fetch_top_markets(n: int) -> list[dict]:
    """Top-volume markets that are still live (accepting orders, two-sided, not ~resolved)."""
    data = _get(f"{GAMMA_URL}/markets", {"active": "true", "closed": "false", "limit": 20 * n,
                                         "order": "volume24hr", "ascending": "false"})
    live = [m for m in data
            if m.get("acceptingOrders") and m.get("enableOrderBook")
            and m.get("bestBid") is not None and m.get("bestAsk") is not None
            and 0.02 < float(m["bestBid"]) and float(m["bestAsk"]) < 0.98]
    return [_parse_market(m) for m in live[:n]]


# =============================================================================
# CLI
# =============================================================================
def report_market(model: FairValueModel, mkt: dict) -> None:
    print(f"\n=== {mkt['question']}  [{mkt['slug']}]")
    fvs = []
    for name, tok in zip(mkt["outcomes"], mkt["tokens"]):
        try:
            res = model.evaluate(fetch_book(tok))
        except Exception as e:  # noqa: BLE001 - keep going for other outcomes
            print(f"\n [{name}] error: {e}")
            continue
        print(f"\n [{name}]")
        print(res)
        fvs.append(res.fair_value)
    if len(fvs) == 2 and None not in fvs:
        print(f"\n  YES+NO fair value sum = {sum(fvs):.4f} (should be ~1.0000)")


def demo(model: FairValueModel) -> None:
    book = OrderBook.from_levels(
        bids=[(0.90, 1500), (0.89, 4000), (0.88, 2500), (0.85, 10000)],
        asks=[(0.93, 120), (0.94, 800), (0.95, 3000), (0.97, 6000)],
        tick_size=0.01, last_trade=0.92,
    )
    print("Synthetic book: bids 0.90x1500 / 0.89x4000 / 0.88x2500 / 0.85x10000")
    print("                asks 0.93x120  / 0.94x800  / 0.95x3000 / 0.97x6000\n")
    print("[YES]")
    print(model.evaluate(book))
    print("\n[NO] (mirrored book)")
    print(model.evaluate(book.complement()))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Polymarket order-book fair value model")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--slug", help="Polymarket market slug")
    g.add_argument("--token", help="CLOB token id of one outcome")
    g.add_argument("--top", type=int, metavar="N", help="evaluate the N highest-volume markets")
    g.add_argument("--demo", action="store_true", help="run on a synthetic book (offline)")
    ap.add_argument("--watch", type=float, metavar="SEC", help="refresh every SEC seconds")
    ap.add_argument("--impact-size", type=float, default=200.0)
    ap.add_argument("--depth-window", type=float, default=0.05)
    ap.add_argument("--depth-scale", type=float, default=0.02)
    args = ap.parse_args(argv)

    model = FairValueModel(impact_size=args.impact_size,
                           depth_window=args.depth_window, depth_scale=args.depth_scale)
    if args.demo:
        return demo(model)

    def run_once():
        if args.token:
            print(f"\n=== token {args.token[:16]}...")
            print(model.evaluate(fetch_book(args.token)))
        elif args.slug:
            report_market(model, fetch_market(args.slug))
        else:
            for m in fetch_top_markets(args.top):
                report_market(model, m)

    if args.watch:
        try:
            while True:
                print(f"\n----- {time.strftime('%H:%M:%S')} -----")
                run_once()
                time.sleep(args.watch)
        except KeyboardInterrupt:
            pass
    else:
        run_once()


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
