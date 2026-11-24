"""
engine.py - FairValueEngine: book + trades + correlated markets + your feeds.

Pipeline for one token (all in log-odds; see core.py):

  1. LOCAL   book estimate (book.py)   +  trade estimate (trades.py)
             or last-trade fallback        sigma^2 from trades (cached per token)
             -> fused "local" estimate. This is what other markets see via links.
  2. LINKS   correlated markets, event sums, logical bounds (links.py)
  3. FEEDS   your views / scraped tables / plugin feeds (feeds.py)
  4. FUSE    robust correlated precision weighting (combine.py)
  5. SHIFT   your biases (feed shifts), in log-odds
  6. SMOOTH  Kalman tracker across calls (tracker.py), optional
  7. BOUND   link / feed hard bounds, then the no-arbitrage clamp to
             [best bid, best ask]. `fair_value_raw` keeps the unclamped
             number: if it lies outside the book, taking the quote has
             positive expected value under the model.

Usage
-----
    from FairMV import FairValueEngine
    eng = FairValueEngine.from_user_dir()            # loads FairMV/user/*
    fv = eng.evaluate(book, trades=df, meta={"slug": ..., "condition_id": ...})
    fv.fair_value, fv.sd, fv.confidence, fv.ci80
    print(fv)                                        # full breakdown

    eng.evaluate_token("some-market-slug")           # fetches book + trades itself
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from .book import BookAnalysis, OrderBook, analyse_book
from .combine import apply_bounds, fuse
from .core import Bound, Estimate, FVConfig, logit, sd_prob, shift_logit_from_prob, sigmoid
from .data import MarketMeta, PolymarketProvider
from .feeds import FeedContext, PluginFeed, TableFeed, ViewsFile, collect
from .links import LinkBook, specs_from_json
from .tracker import Tracker
from .trades import estimate_sigma2, last_trade_estimate, normalize_trades, trade_estimate

PKG_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_USER_DIR = os.path.join(PKG_DIR, "user")
Z80 = 1.2815515655446004


# =============================================================================
# Result
# =============================================================================
@dataclass
class FairValue:
    fair_value: Optional[float]
    fair_value_raw: Optional[float] = None
    sd: Optional[float] = None                 # probability sd
    ci80: tuple = (None, None)
    bid: Optional[float] = None
    ask: Optional[float] = None
    spread: Optional[float] = None
    confidence: str = "none"
    token_id: str = ""
    slug: str = ""
    signals: list = field(default_factory=list)     # per-signal dicts
    components: dict = field(default_factory=dict)  # name -> p (book sub-estimators + signals)
    diagnostics: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    x: Optional[float] = None
    var: Optional[float] = None

    def to_dict(self) -> dict:
        return {"token_id": self.token_id, "slug": self.slug, "fair_value": self.fair_value,
                "fair_value_raw": self.fair_value_raw, "fv_sd": self.sd,
                "ci80_lo": self.ci80[0], "ci80_hi": self.ci80[1], "bid": self.bid, "ask": self.ask,
                "confidence": self.confidence,
                **{f"w_{s['name']}": s["weight"] for s in self.signals}}

    def __str__(self) -> str:
        f = lambda v, n=4: "  n/a " if v is None else f"{v:.{n}f}"
        lo, hi = self.ci80
        lines = [f"  Fair value : {f(self.fair_value)}  sd {f(self.sd)}  80% [{f(lo, 3)}, {f(hi, 3)}]"
                 f"   (confidence: {self.confidence})"]
        if self.fair_value_raw is not None and self.fair_value is not None and abs(self.fair_value_raw - self.fair_value) > 1e-9:
            lines.append(f"  Unclamped  : {f(self.fair_value_raw)}   (outside the book: taking would be +EV)")
        lines.append(f"  Bid / Ask  : {f(self.bid, 3)} / {f(self.ask, 3)}   spread {f(self.spread, 3)}")
        if self.signals:
            lines.append(f"  {'signal':<34}{'p':>8}{'sd':>8}{'weight':>8}{'trust':>7}")
            for s in sorted(self.signals, key=lambda s: -s["weight"]):
                lines.append(f"  {s['name'][:34]:<34}{s['p']:>8.4f}{s['sd_p']:>8.4f}{s['weight']:>8.1%}"
                             f"{s['robust']:>7.2f}")
        book_parts = {k: v for k, v in self.components.items() if not k.startswith(("trades", "last_trade", "link:", "feed:"))}
        if book_parts:
            lines.append("  book parts : " + "  ".join(f"{k}={v:.3f}" for k, v in book_parts.items() if v is not None))
        keys = ("eff_bid", "eff_ask", "sigma2", "informed_mu (GM)", "imbalance_used", "shift", "bounds_active",
                "tracker", "edge_buy_at_ask", "edge_sell_at_bid")
        diag = [f"{k}={self.diagnostics[k]:.4f}" if isinstance(self.diagnostics[k], float) else f"{k}={self.diagnostics[k]}"
                for k in keys if self.diagnostics.get(k) not in (None, [], "")]
        if diag:
            lines.append("  diag       : " + "  ".join(diag))
        for w in self.warnings:
            lines.append(f"  ! {w}")
        return "\n".join(lines)


# Back-compatible name
FairValueResult = FairValue


def _meta_from(meta, book: Optional[OrderBook]) -> MarketMeta:
    if isinstance(meta, MarketMeta):
        return meta
    d = dict(meta or {})
    tok = str(d.get("token_id") or (book.token_id if book is not None else "") or "")
    toks = d.get("tokens") or d.get("clobTokenIds") or []
    if isinstance(toks, str):
        toks = json.loads(toks)
    toks = [str(t) for t in toks]
    oi = d.get("outcome_index")
    if oi is None:
        oi = toks.index(tok) if tok in toks else 0
    ev = d.get("event_slug") or ((d.get("events") or [{}])[0].get("slug") if d.get("events") else "")
    return MarketMeta(token_id=tok, slug=d.get("slug", "") or "", question=d.get("question", "") or "",
                      condition_id=d.get("condition_id") or d.get("conditionId") or "",
                      outcome=d.get("outcome", "") or "", outcome_index=int(oi), tokens=toks,
                      outcomes=d.get("outcomes") or [], event_slug=ev or "",
                      neg_risk=bool(d.get("neg_risk", d.get("negRisk", False))),
                      end_date=d.get("end_date") or d.get("endDate"))


# =============================================================================
# Engine
# =============================================================================
class FairValueEngine:
    def __init__(self, cfg: Optional[FVConfig] = None, feeds: Optional[list] = None,
                 links: Optional[LinkBook] = None, provider=None, auto_meta: bool = True):
        self.cfg = cfg or FVConfig()
        self.feeds = list(feeds or [])
        self.links = links or LinkBook()
        self.provider = provider
        self.auto_meta = auto_meta            # look up slug/event via provider when meta is incomplete
        self.tracker = Tracker(self.cfg.tracker_floor, self.cfg.tracker_jump_z)
        self._sigma2: dict = {}               # token -> (sigma2, info)
        self._local_cache: dict = {}          # token -> (t, Estimate)
        self._meta_cache: dict = {}

    # ---------------------------------------------------------------- loading
    @classmethod
    def from_user_dir(cls, user_dir: str = DEFAULT_USER_DIR, cfg: Optional[FVConfig] = None,
                      provider="default", **kw) -> "FairValueEngine":
        """Load user_dir/{config.json, views.json, links.json, feeds/*.py, tables/*}."""
        cp = os.path.join(user_dir, "config.json")
        if cfg is None:
            cfg = FVConfig()
            if os.path.exists(cp):
                with open(cp, encoding="utf-8") as f:
                    over = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
                cfg = FVConfig.from_dict({**cfg.__dict__, **over})
        feeds = [ViewsFile(os.path.join(user_dir, "views.json")),
                 TableFeed(os.path.join(user_dir, "tables"), name="tables"),
                 PluginFeed(os.path.join(user_dir, "feeds"))]
        lb = LinkBook()
        lp = os.path.join(user_dir, "links.json")
        if os.path.exists(lp):
            with open(lp, encoding="utf-8") as f:
                links, events = specs_from_json(json.load(f))
            lb = LinkBook(links, events)
        if provider == "default":
            provider = PolymarketProvider()
        return cls(cfg, feeds, lb, provider, **kw)

    # ---------------------------------------------------------------- pieces
    def _sigma2_for(self, token: str, tr: Optional[pd.DataFrame]) -> tuple[float, dict]:
        if tr is not None and len(tr) >= 3:
            s2, info = estimate_sigma2(tr, self.cfg)
            self._sigma2[token] = (s2, info)
            return s2, info
        return self._sigma2.get(token, (self.cfg.sigma2_prior, {"sigma2_source": "prior"}))

    def local(self, book: Optional[OrderBook], trades=None, token: str = "", now=None,
              outcome_index: Optional[int] = None) -> tuple[Optional[Estimate], dict]:
        """Book + trades only. Returns (estimate, parts) where parts holds the pieces."""
        cfg = self.cfg
        now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
        ba = analyse_book(book, cfg) if book is not None else BookAnalysis(None, None, None, {}, {}, ["no book"])
        h_book = None
        if ba.eff_bid is not None and ba.eff_ask is not None and ba.eff_ask > ba.eff_bid:
            h_book = 0.5 * (logit(ba.eff_ask) - logit(ba.eff_bid))
        tr = None
        if trades is not None:
            tr = trades if {"ts", "price", "side", "size"} <= set(trades.columns) and "yes_price" not in trades \
                else normalize_trades(trades, outcome_index)
        sigma2, s2info = self._sigma2_for(token, tr)
        if tr is not None:
            te = trade_estimate(tr, now, sigma2, h_book, cfg)
        else:
            te = last_trade_estimate(book.last_trade if book else None, book.best_bid if book else None,
                                     book.best_ask if book else None, sigma2, h_book, cfg)
        ests = [e for e in (ba.estimate, te) if e is not None]
        fz = fuse(ests, cfg)
        est = None if fz is None else Estimate("local", fz.x, fz.var, "local")
        return est, {"book": ba, "trades": te, "sigma2": sigma2, "sigma2_info": s2info, "ests": ests}

    def _local_many(self, tokens: list) -> dict:
        """Book-only (+last trade) local estimates for many tokens, batched and cached."""
        now = time.time()
        out, need = {}, []
        for t in tokens:
            hit = self._local_cache.get(("b", t))
            if hit and now - hit[0] < self.cfg.link_cache_ttl:
                out[t] = hit[1]
            else:
                need.append(t)
        if need:
            books = self.provider.books(need)
            for t in need:
                est, _ = self.local(books.get(t), None, t) if books.get(t) is not None else (None, None)
                self._local_cache[("b", t)] = (now, est)
                out[t] = est
        return out

    def _resolve(self, ref: str):
        """Estimate of another market (book + trades) for explicit links."""
        now = time.time()
        hit = self._local_cache.get(("r", ref))
        if hit and now - hit[0] < self.cfg.link_cache_ttl:
            return hit[1]
        meta = self.provider.market(ref)
        if meta is None:
            res = (None, None)
        else:
            book = self.provider.book(meta.token_id)
            tr = None
            try:
                tr = self.provider.trades(meta, 300)
            except Exception:  # noqa: BLE001 - book alone is fine
                pass
            est, _ = self.local(book, tr, meta.token_id) if (book is not None or tr is not None) else (None, None)
            res = (est, meta)
        self._local_cache[("r", ref)] = (now, res)
        return res

    def _complete_meta(self, meta: MarketMeta) -> MarketMeta:
        if not (self.auto_meta and self.provider is not None and meta.token_id):
            return meta
        if meta.slug and meta.condition_id and (meta.event_slug or not self.cfg.use_links):
            return meta
        if meta.token_id not in self._meta_cache:
            try:
                self._meta_cache[meta.token_id] = self.provider.market(meta.token_id)
            except Exception:  # noqa: BLE001
                self._meta_cache[meta.token_id] = None
        full = self._meta_cache[meta.token_id]
        return full if full is not None else meta

    # ---------------------------------------------------------------- main
    def evaluate(self, book: Optional[OrderBook], trades=None, meta=None, now=None,
                 smooth: Optional[bool] = None, key: Optional[str] = None,
                 use_links: Optional[bool] = None, use_feeds: Optional[bool] = None) -> FairValue:
        cfg = self.cfg
        now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
        if now.tzinfo is None:
            now = now.tz_localize("UTC")
        meta = self._complete_meta(_meta_from(meta, book))
        tok = meta.token_id or (book.token_id if book else "")
        b = book.best_bid if book else None
        a = book.best_ask if book else None
        res = FairValue(None, bid=b, ask=a, spread=(a - b) if a is not None and b is not None else None,
                        token_id=tok, slug=meta.slug)

        local, parts = self.local(book, trades, tok, now, meta.outcome_index)
        ba: BookAnalysis = parts["book"]
        sigma2 = parts["sigma2"]
        res.warnings.extend(ba.warnings)
        res.components.update(ba.components)
        res.diagnostics.update(ba.diagnostics)
        res.diagnostics.update(sigma2=sigma2, **parts["sigma2_info"])
        ests = list(parts["ests"])
        if parts["trades"] is not None:
            res.diagnostics.update({f"trades_{k}": v for k, v in parts["trades"].info.items()})
        bounds: list = []

        # links ---------------------------------------------------------------
        if (cfg.use_links if use_links is None else use_links) and self.provider is not None:
            try:
                lr = self.links.evaluate(meta, local, self._resolve, cfg, now.timestamp(),
                                         self.provider, self._local_many)
                ests += lr.estimates
                bounds += lr.bounds
                res.warnings += lr.warnings
                if lr.info:
                    res.diagnostics["links"] = lr.info
            except Exception as e:  # noqa: BLE001
                res.warnings.append(f"links failed: {type(e).__name__}: {e}")

        # feeds ---------------------------------------------------------------
        shift_l = shift_p = 0.0
        if (cfg.use_feeds if use_feeds is None else use_feeds) and self.feeds:
            ctx = FeedContext(token_id=tok, slug=meta.slug, question=meta.question,
                              condition_id=meta.condition_id, outcome=meta.outcome,
                              outcome_index=meta.outcome_index, event_slug=meta.event_slug, now=now,
                              p_market=None if local is None else local.p, book=book)
            fr = collect(self.feeds, ctx, sigma2, cfg)
            ests += fr.estimates
            bounds += fr.bounds
            shift_l, shift_p = fr.shift_logit, fr.shift_prob
            res.warnings += fr.warnings
            if fr.info:
                res.diagnostics["feeds"] = fr.info

        # fuse ----------------------------------------------------------------
        fz = fuse(ests, cfg)
        if fz is None:
            res.warnings.append("no usable signal (empty book, no trades, no links/feeds)")
            return res
        x, var = fz.x, fz.var
        for e in ests:
            res.signals.append({"name": e.name, "kind": e.kind, "p": e.p, "sd_p": e.sd_p,
                                "weight": fz.weights.get(e.name, 0.0), "robust": fz.robust.get(e.name, 1.0),
                                "info": e.info})
            res.components[e.name] = e.p
        if any(fz.robust.get(e.name, 1.0) < 0.5 for e in ests):
            res.warnings.append("outlier signal(s) down-weighted: " + ", ".join(
                f"{n} (x{u:.2f})" for n, u in fz.robust.items() if u < 0.5))

        # shift (your biases) ---------------------------------------------------
        if shift_l or shift_p:
            dx = shift_l + (shift_logit_from_prob(sigmoid(x + shift_l), shift_p) if shift_p else 0.0)
            x += dx
            res.diagnostics["shift"] = dx

        # smooth ---------------------------------------------------------------
        if cfg.use_tracker if smooth is None else smooth:
            x, var, tinfo = self.tracker.update(key or tok or id(book), x, var, now.timestamp(), sigma2)
            res.diagnostics["tracker"] = tinfo.get("tracker")

        # bounds ---------------------------------------------------------------
        p_raw = sigmoid(x)
        p, active = apply_bounds(p_raw, bounds)
        if active:
            res.diagnostics["bounds_active"] = active
        res.fair_value_raw = p
        if cfg.clamp_to_book:
            lo = b if b is not None else 0.0
            hi = a if a is not None else 1.0
            if lo <= hi:
                p = min(max(p, lo), hi)
        if a is not None:
            res.diagnostics["edge_buy_at_ask"] = res.fair_value_raw - a
        if b is not None:
            res.diagnostics["edge_sell_at_bid"] = b - res.fair_value_raw

        res.fair_value = p
        res.x, res.var = x, var
        res.sd = sd_prob(x, var)
        res.ci80 = (sigmoid(x - Z80 * math.sqrt(var)), sigmoid(x + Z80 * math.sqrt(var)))
        res.confidence = cfg.confidence(res.sd)
        return res

    def evaluate_token(self, ref: str, with_trades: bool = True, **kw) -> FairValue:
        """Fetch everything for a slug / 'slug:No' / token id and evaluate."""
        if self.provider is None:
            raise RuntimeError("evaluate_token needs a provider")
        meta = self.provider.market(ref)
        if meta is None:
            raise LookupError(f"no market for '{ref}'")
        book = self.provider.book(meta.token_id)
        tr = self.provider.trades(meta, 1000) if with_trades else None
        return self.evaluate(book, tr, meta, **kw)
