"""
MarketMaker.py - market making model for Polymarket binary tokens.

Uses the fair price (FairMV), the expected number of buyers and sellers, and
the information content of informed flow to choose bid/ask prices and sizes.

No orders are placed. It prints recommended quotes and has a Monte Carlo
simulator to check that the maths makes money.

The theory is the standard market making literature (also how trading firms
such as Jane Street describe the job publicly: edge vs adverse selection,
inventory risk, sizing):
  Glosten & Milgrom (1985)    adverse selection: spread pays for information
  Avellaneda & Stoikov (2008) arrival rates, optimal distance, inventory skew
  Kelly (1956) / Merton       growth-optimal sizing

-------------------------------------------------------------------------------
THE MATHS  (prices in $ per share; a share pays $1 if YES)
-------------------------------------------------------------------------------
Inputs
  p       fair value (FairMV)
  q       inventory in YES shares (negative = short YES = long NO)
  A, k    noise (uninformed) traders: lam_U(d) = A e^{-k d} per hour when we
          quote d away from p. "Expected buyers / sellers".
  m       mean noise trade size (shares); sizes X ~ Exponential(mean m)
  lam_I   informed traders per hour
  J       information size: an informed trader knows the price is about to
          move up or down (half each) by M ~ Exponential(mean J). They trade
          with us only if M > d, which has probability e^{-d/J}, and then take
          ALL of our size. Given that, the loss per share is
          E[M - d | M > d] = J (the exponential is memoryless).
  W, c_K  bankroll and fractional-Kelly multiplier (default 0.25)

1. Expected profit per hour of quoting N shares, d above fair value
   Noise volume is lam_U(d) E[min(N, X)] = lam_U(d) m (1 - e^{-N/m});
   informed volume is lam_I(d) N with lam_I(d) = (lam_I/2) e^{-d/J}. So

       Pi(d, N) = lam_U(d) m (1 - e^{-N/m}) d  -  lam_I(d) N J
                  profit from noise buyers        loss to informed buyers

   (mirror image for the bid). This is the Glosten-Milgrom adverse-selection
   trade-off with Avellaneda-Stoikov arrival rates.

2. Optimal size from flow: dPi/dN = 0
       N_flow(d) = m * ln( lam_U(d) d / (lam_I(d) J) )      (0 if the log is negative)
   Size is limited by information: noise traders are small, but informed
   traders sweep everything you show. More noise flow, a wider quote, or
   less information means size can grow.

3. Optimal price: evaluate Pi(d, N_flow(d)) on the tick grid and take the
   best. With no information the optimum is d* = 1/k; information pushes it
   wider. If max Pi <= 0 the side is not quoted.
   After a fill: E[value | ask lifted] = p + mu_vol (d + J), where mu_vol is
   the informed share of expected filled volume (winner's curse). Edge per
   share e = d - mu_vol (d + J).

4. Risk cap: continuous Kelly (Merton fraction)
   Holding N shares for tau hours earns N*e with variance N^2 sigma^2 tau,
   so the growth-optimal position is
       N_kelly = c_K * W * e / (sigma^2 * tau)
   with sigma^2 = lam_I E[M^2] = 2 lam_I J^2 (jump model) or measured from
   FetchData recordings. Final size = min(N_flow, N_kelly -/+ q, max_position).

5. Inventory (Avellaneda-Stoikov reservation price)
       r = p - q * gamma * sigma^2 * tau,     gamma = 1 / (c_K W)
   This is the same risk aversion as the Kelly step: log utility on bankroll
   W. Inventory value iv = (p - r) per share is added to every share sold
   (subtracted for the bid), i.e. d -> d + iv in steps 1-3, and the Kelly
   target is a position, so orders fill the gap:
       bid <= N_kelly - q,   ask <= N_kelly + q
   When long, the ask gets bigger and lower and the bid smaller and lower.

Live mode infers lam_I from the market's own spread. Competitive zero-profit
market makers set half-spread s with edge 0, i.e. s = mu (s + J), so
    mu = s / (s + J),   lam_I = 2 mu lam_U(s) e^{s/J} / (1 - mu).

-------------------------------------------------------------------------------
USAGE
-------------------------------------------------------------------------------
  py MarketMaker.py --demo                      example quotes at 3 inventory levels
  py MarketMaker.py --simulate                  Monte Carlo: this model vs naive 1-tick quoting
  py MarketMaker.py --slug <market-slug>        live recommended quotes (first outcome token)
     [--inventory 0] [--bankroll 1000] [--A-buy 60] [--A-sell 60] [--k 150]
     [--informed 6] [--J 0.03] [--fv 0.55] [--data data]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, replace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "FairMV"))


# =============================================================================
# Parameters and output
# =============================================================================
@dataclass
class MMParams:
    A_buy: float = 60.0          # noise buyers per hour if our ask were at fair value
    A_sell: float = 60.0         # noise sellers per hour if our bid were at fair value
    k: float = 150.0             # noise flow decay per $ (halves every ln2/k = 0.46c)
    noise_size: float = 50.0     # m, mean noise trade size (shares)
    lam_informed: float = 6.0    # informed traders per hour
    J: float = 0.03              # mean price move informed traders know about ($)
    bankroll: float = 1000.0     # W
    kelly_frac: float = 0.25     # c_K
    tau_h: float = 1.0           # expected inventory holding time, hours
    max_position: float = 2000.0 # |q| cap, shares
    min_size: float = 5.0        # Polymarket minimum order
    tick: float = 0.01

    @property
    def gamma(self) -> float:    # log-utility risk aversion, shared with Kelly
        return 1.0 / (self.kelly_frac * self.bankroll)

    def sigma2_jumps(self) -> float:
        return 2 * self.lam_informed * self.J ** 2   # E[M^2] = 2 J^2 for exponential M

    def informed_rate(self, d: float) -> float:
        """Informed traders per hour hitting a quote d away: (lam_I/2) P(M > d)."""
        return self.lam_informed / 2 * math.exp(-max(d, 0.0) / self.J) if self.J > 0 else 0.0


@dataclass
class Side:
    price: float
    size: float
    pnl_per_h: float     # expected $/h (incl. inventory value)
    fills_per_h: float
    mu: float            # informed share of expected filled volume
    post_fill: float     # E[value | filled]
    edge: float          # $/share expected profit on a fill (excl. inventory)
    limited_by: str      # which constraint set the size: flow / kelly / position


@dataclass
class Quote:
    bid: Side | None
    ask: Side | None
    reservation: float
    sigma2: float

    @property
    def bid_size(self) -> float:
        return self.bid.size if self.bid else 0.0

    @property
    def ask_size(self) -> float:
        return self.ask.size if self.ask else 0.0

    def __str__(self) -> str:
        def col(s: Side | None, name: str) -> str:
            if s is None:
                return f"  {name}: no quote (expected profit <= 0)"
            return (f"  {name}: {s.size:>6.0f} @ {s.price:.3f}  (size set by {s.limited_by})   "
                    f"fills/h {s.fills_per_h:5.2f}   informed vol {s.mu:5.1%}   "
                    f"E[v|fill] {s.post_fill:.4f}   edge {s.edge:+.4f}/sh   {s.pnl_per_h:+.3f} $/h")
        return (col(self.bid, "BID") + "\n" + col(self.ask, "ASK") +
                f"\n  reservation price {self.reservation:.4f}   sigma^2 {self.sigma2:.2e}/h")


# =============================================================================
# Core maths
# =============================================================================
def flow_size(d: float, iv: float, lu: float, li: float, P: MMParams) -> float:
    """Step 2: N_flow = m ln(lam_U(d)(d+iv) / (lam_I(d)(J - iv))). iv = inventory value per share."""
    gain = d + iv                      # per noise share
    loss = P.J - iv                    # per informed share (inventory value offsets it)
    if gain <= 0:
        return 0.0
    if li <= 0 or loss <= 0:
        return math.inf
    x = lu * gain / (li * loss)
    return P.noise_size * math.log(x) if x > 1 else 0.0


def eval_side(d: float, iv: float, A: float, P: MMParams, kelly_room, pos_room: float):
    """For a quote d from fair value: (size, $/h, fills/h, informed vol share, edge/share, limiter)."""
    m = P.noise_size
    lu = A * math.exp(-P.k * max(d, 0.0))
    li = P.informed_rate(d)
    n_flow = flow_size(d, iv, lu, li, P)

    def mix(N):  # informed share of expected volume and resulting edge per share
        vu, vi = lu * m * (1 - math.exp(-N / m)), li * N
        mu = vi / (vu + vi) if vu + vi > 0 else 0.0
        return vu, vi, mu, d - mu * (d + P.J)

    # Kelly needs the edge, which depends on N through mu: one fixed-point pass
    _, _, _, edge = mix(max(min(n_flow, pos_room, 10 * m), 1.0))
    N, by = min((n_flow, "flow"), (max(kelly_room(edge), 0.0), "kelly"), (pos_room, "position"))
    if N < P.min_size:
        return None
    N = float(math.floor(N))
    vu, vi, mu, edge = mix(N)
    pnl = vu * (d + iv) + vi * (-P.J + iv)
    return N, pnl, lu + li, mu, edge, by


def best_side(p: float, iv: float, A: float, P: MMParams, sign: int, limit: float,
              kelly_room, pos_room: float, passive: float | None = None) -> Side | None:
    """sign=+1 ask, -1 bid. Grid search over price.
    limit   = most aggressive allowed price (post-only: never cross the book)
    passive = least aggressive allowed price (join-or-improve: never sit behind the market)"""
    t, best = P.tick, None
    lo, hi = max(t, p - 0.2), min(1 - t, p + 0.2)
    if passive is not None:   # make sure the join/improve range is searched even if far from p
        lo, hi = (min(lo, limit), max(hi, passive)) if sign > 0 else (min(lo, passive), max(hi, limit))
        lo, hi = max(t, lo), min(1 - t, hi)
    for i in range(math.ceil(lo / t - 1e-9), math.floor(hi / t + 1e-9) + 1):
        price = round(i * t, 6)
        if (sign > 0 and price < limit - 1e-9) or (sign < 0 and price > limit + 1e-9):
            continue
        if passive is not None and ((sign > 0 and price > passive + 1e-9) or
                                    (sign < 0 and price < passive - 1e-9)):
            continue
        d = sign * (price - p)
        r = eval_side(d, iv, A, P, kelly_room, pos_room)
        if r and (best is None or r[1] > best.pnl_per_h):
            N, pnl, F, mu, edge, by = r
            best = Side(price, N, pnl, F, mu, p + sign * mu * (d + P.J), edge, by)
    return best if best and best.pnl_per_h > 0 else None


def quote(p: float, q: float, P: MMParams, sigma2: float | None = None,
          mkt_bid: float | None = None, mkt_ask: float | None = None,
          join_or_improve: bool = False) -> Quote:
    """join_or_improve=True keeps our quotes at or inside the market's best bid/ask
    (the flow model assumes we ARE the best quote; behind the market we would not be filled)."""
    t = P.tick
    p = min(max(p, t), 1 - t)
    s2 = max(sigma2 if sigma2 is not None else P.sigma2_jumps(), 1e-8)   # avoid /0 in Kelly
    r = p - q * P.gamma * s2 * P.tau_h
    kelly_pos = lambda e: P.kelly_frac * P.bankroll * e / (s2 * P.tau_h)   # target |position|

    ask = best_side(p, p - r, P.A_buy, P, +1, (mkt_bid + t) if mkt_bid is not None else t,
                    lambda e: kelly_pos(e) + q, P.max_position + q,
                    mkt_ask if join_or_improve and mkt_ask is not None else None)
    bid = best_side(p, r - p, P.A_sell, P, -1, (mkt_ask - t) if mkt_ask is not None else 1 - t,
                    lambda e: kelly_pos(e) - q, P.max_position - q,
                    mkt_bid if join_or_improve and mkt_bid is not None else None)
    if ask and bid and bid.price >= ask.price:  # extreme skew: keep only the unwinding side
        bid, ask = (None, ask) if q > 0 else (bid, None)
    return Quote(bid, ask, r, s2)


# =============================================================================
# Calibration from FetchData recordings
# =============================================================================
def sigma2_from_data(data_dir: str, token_id: str) -> float | None:
    """Price variance per hour from recorded mids (FetchData snapshots)."""
    try:
        import pandas as pd
        s = pd.read_parquet(os.path.join(data_dir, "snapshots"),
                            columns=["token_id", "fetch_ts", "mid"],
                            filters=[("token_id", "=", token_id)])
    except Exception:  # noqa: BLE001
        return None
    s = s.dropna().sort_values("fetch_ts")
    if len(s) < 30:
        return None
    dm = s["mid"].diff().iloc[1:]
    dt_h = s["fetch_ts"].diff().dt.total_seconds().iloc[1:] / 3600
    ok = dt_h > 0
    return float((dm[ok] ** 2).sum() / dt_h[ok].sum())


# =============================================================================
# Monte Carlo check
# =============================================================================
def simulate(P: MMParams, strategy: str, n_paths=300, hours=8.0, dt_min=0.5, seed=0) -> np.ndarray:
    """
    Public value v starts at 0.5. Each step:
      * noise buyers/sellers arrive at A e^{-k d} per hour, size ~ Exp(mean m)
      * informed traders arrive at lam_I per hour. They know v is about to jump
        up or down by M ~ Exp(mean J), trade (taking our full size) if that
        beats our quote, then the jump happens and becomes public.
    The MM quotes around v. PnL = cash + inventory marked at final v.
    Both strategies see the same random arrivals (common random numbers).
    """
    rng = np.random.default_rng(seed)
    dt, t = dt_min / 60, P.tick
    pnl = np.empty(n_paths)
    for i in range(n_paths):
        v, q, cash = 0.5, 0.0, 0.0
        for _ in range(int(hours / dt)):
            if strategy == "model":
                Q = quote(v, q, P)
                bid, ask = (Q.bid.price if Q.bid else None), (Q.ask.price if Q.ask else None)
                bsz, asz = Q.bid_size, Q.ask_size
            else:  # naive: one tick each side, 100 shares, no skew
                bid = round(math.floor(v / t - 1e-9) * t, 6)
                ask = round(math.ceil(v / t + 1e-9) * t, 6)
                bsz = 100.0 if q < P.max_position else 0.0
                asz = 100.0 if q > -P.max_position else 0.0
            u = rng.random(4)
            sz = rng.exponential(P.noise_size, 2)
            if ask is not None and asz > 0 and u[0] < P.A_buy * math.exp(-P.k * (ask - v)) * dt:
                n = min(asz, sz[0]); q -= n; cash += n * ask; asz -= n
            if bid is not None and bsz > 0 and u[1] < P.A_sell * math.exp(-P.k * (v - bid)) * dt:
                n = min(bsz, sz[1]); q += n; cash -= n * bid; bsz -= n
            if u[2] < P.lam_informed * dt:
                up, move = u[3] < 0.5, rng.exponential(P.J)
                if up and ask is not None and asz > 0 and v + move > ask:
                    q -= asz; cash += asz * ask
                elif not up and bid is not None and bsz > 0 and v - move < bid:
                    q += bsz; cash -= bsz * bid
                v = min(max(v + (move if up else -move), 0.02), 0.98)
        pnl[i] = cash + q * v
    return pnl


def run_simulation(P: MMParams, paths: int) -> None:
    print(f"Simulated market (8h): noise {P.A_buy:.0f}+{P.A_sell:.0f}/h at fair value, k={P.k:.0f}; "
          f"informed {P.lam_informed:g}/h knowing +/-{P.J:.2f} moves; bankroll ${P.bankroll:.0f}\n")
    print(f"  {'strategy':<9}{'mean PnL':>10}{'std':>9}{'5% tail':>10}{'P(loss)':>9}{'Sharpe':>8}")
    for name in ("model", "naive"):
        x = simulate(P, name, n_paths=paths)
        print(f"  {name:<9}{x.mean():>10.2f}{x.std():>9.2f}{np.percentile(x, 5):>10.2f}"
              f"{(x < 0).mean():>9.1%}{x.mean() / (x.std() + 1e-12):>8.2f}")


# =============================================================================
# Live recommendation
# =============================================================================
def live(args, P: MMParams) -> None:
    import requests
    from FairMV import FairValueModel, fetch_book  # noqa: E402

    m = requests.get("https://gamma-api.polymarket.com/markets",
                     params={"slug": args.slug}, timeout=15).json()
    if not m:
        raise SystemExit(f"no market '{args.slug}'")
    m = m[0]
    tok = json.loads(m["clobTokenIds"])[0]
    outcome = json.loads(m["outcomes"])[0]

    book = fetch_book(tok)
    res = FairValueModel().evaluate(book)
    if res.fair_value is None or book.best_bid is None or book.best_ask is None:
        print("Book is one-sided/empty: no quotes recommended.")
        return
    p = args.fv if args.fv is not None else res.fair_value
    P = replace(P, tick=book.tick_size)

    if args.informed is None:   # infer informed rate from the market's spread
        s = 0.5 * (book.best_ask - book.best_bid)
        mu = s / (s + P.J)
        lam_u = 0.5 * (P.A_buy + P.A_sell) * math.exp(-P.k * s)
        P = replace(P, lam_informed=2 * mu * lam_u * math.exp(s / P.J) / (1 - mu))
        src_i = f"implied by spread: mu = s/(s+J) = {mu:.2f}"
    else:
        src_i = "user"

    s2 = sigma2_from_data(args.data, tok) if args.data else None
    src_s = "recorded data" if s2 is not None else "2 lam_I J^2 (jump model)"

    print(f"\n=== {m.get('question')}  [{outcome} token]")
    print(f"  market {book.best_bid:.3f} / {book.best_ask:.3f}   fair value {p:.4f} ({res.confidence})")
    print(f"  informed {P.lam_informed:.2f}/h, J={P.J:.3f} ({src_i})   noise {P.A_buy:.0f}/{P.A_sell:.0f} per h")
    print(f"  sigma^2 from {src_s}   inventory {args.inventory:+.0f}   bankroll ${P.bankroll:.0f}\n")
    print(quote(p, args.inventory, P, s2, book.best_bid, book.best_ask))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Polymarket market making quotes and sizes")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--slug", help="market slug: live recommended quotes")
    g.add_argument("--demo", action="store_true")
    g.add_argument("--simulate", action="store_true")
    ap.add_argument("--paths", type=int, default=300, help="Monte Carlo paths")
    ap.add_argument("--inventory", type=float, default=0.0, help="current YES shares (negative = short)")
    ap.add_argument("--bankroll", type=float, default=1000.0)
    ap.add_argument("--kelly", type=float, default=0.25, help="fractional Kelly multiplier c_K")
    ap.add_argument("--tau", type=float, default=1.0, help="inventory holding time, hours")
    ap.add_argument("--k", type=float, default=150.0, help="noise flow decay per $ of distance")
    ap.add_argument("--A-buy", type=float, default=60.0, help="noise buyers/h at fair value")
    ap.add_argument("--A-sell", type=float, default=60.0, help="noise sellers/h at fair value")
    ap.add_argument("--noise-size", type=float, default=50.0, help="mean noise trade size (shares)")
    ap.add_argument("--informed", type=float, default=None,
                    help="informed traders/h (default: sim 6, live implied by spread)")
    ap.add_argument("--J", type=float, default=0.03, help="price move informed traders know about")
    ap.add_argument("--fv", type=float, default=None, help="override fair value with your own view")
    ap.add_argument("--max-position", type=float, default=2000.0)
    ap.add_argument("--data", help="FetchData output dir to calibrate sigma^2")
    a = ap.parse_args(argv)

    P = MMParams(A_buy=a.A_buy, A_sell=a.A_sell, k=a.k, noise_size=a.noise_size,
                 lam_informed=a.informed if a.informed is not None else 6.0, J=a.J,
                 bankroll=a.bankroll, kelly_frac=a.kelly, tau_h=a.tau, max_position=a.max_position)
    if a.simulate:
        return run_simulation(P, a.paths)
    if a.demo:
        for inv in (0, 300, -300):
            print(f"\nfair 0.600, informed {P.lam_informed:g}/h with J={P.J}, inventory {inv:+d}")
            print(quote(0.60, inv, P))
        return
    live(a, P)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
