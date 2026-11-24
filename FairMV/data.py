"""
data.py - Polymarket API access for FairMV, with retries and a small TTL cache.

The engine only needs four things from the outside world. Swap in your own
Provider (e.g. one reading from FetchData recordings, or a fake in tests) by
implementing the same methods:

    market(ref)             -> MarketMeta | None   (ref: slug, 'slug:No', token id)
    event_markets(slug)     -> list[MarketMeta]    (active, order-book markets in an event)
    books(token_ids)        -> {token_id: OrderBook}
    trades(meta, max_n)     -> DataFrame (normalised, in the meta's token terms)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from .book import OrderBook
from .trades import normalize_trades

try:
    import requests
except ImportError:   # only needed for live data
    requests = None

CLOB_URL = "https://clob.polymarket.com"
GAMMA_URL = "https://gamma-api.polymarket.com"
TRADES_URL = "https://data-api.polymarket.com/v2/trades"


def _jl(v):
    return json.loads(v) if isinstance(v, str) else (v or [])


@dataclass
class MarketMeta:
    token_id: str
    slug: str = ""
    question: str = ""
    condition_id: str = ""
    outcome: str = ""
    outcome_index: int = 0
    tokens: list = field(default_factory=list)
    outcomes: list = field(default_factory=list)
    event_slug: str = ""
    neg_risk: bool = False
    end_date: Optional[str] = None
    active: bool = True

    @classmethod
    def from_gamma(cls, m: dict, outcome_index: int = 0) -> "MarketMeta":
        toks, outs = [str(t) for t in _jl(m.get("clobTokenIds"))], _jl(m.get("outcomes"))
        ev = (m.get("events") or [{}])[0]
        oi = outcome_index if outcome_index < len(toks) else 0
        return cls(
            token_id=toks[oi] if toks else "", slug=m.get("slug", ""), question=m.get("question", ""),
            condition_id=m.get("conditionId", ""), outcome=outs[oi] if oi < len(outs) else "",
            outcome_index=oi, tokens=toks, outcomes=outs, event_slug=ev.get("slug", "") or "",
            neg_risk=bool(m.get("negRisk") or ev.get("negRisk")), end_date=m.get("endDate"),
            active=bool(m.get("active", True) and m.get("acceptingOrders", True) and m.get("enableOrderBook", True)
                        and not m.get("closed", False)),
        )

    def other(self) -> "MarketMeta":
        """Meta for the complementary outcome token of the same market."""
        oi = 1 - self.outcome_index
        d = dict(self.__dict__)
        d.update(token_id=self.tokens[oi] if len(self.tokens) > oi else "",
                 outcome=self.outcomes[oi] if len(self.outcomes) > oi else "", outcome_index=oi)
        return MarketMeta(**d)


def is_token_id(ref: str) -> bool:
    return ref.isdigit() and len(ref) > 20


def split_ref(ref: str) -> tuple[str, Optional[str]]:
    """'slug:No' -> ('slug', 'No');  'slug' -> ('slug', None)."""
    if ":" in ref and not ref.startswith("0x"):
        s, o = ref.rsplit(":", 1)
        return s, o
    return ref, None


class PolymarketProvider:
    def __init__(self, ttl_book: float = 5.0, ttl_trades: float = 60.0, ttl_meta: float = 3600.0,
                 http_get=None, http_post=None):
        self.ttl_book, self.ttl_trades, self.ttl_meta = ttl_book, ttl_trades, ttl_meta
        self._cache: dict = {}
        self._session = requests.Session() if requests is not None else None
        if self._session is not None:
            self._session.headers["User-Agent"] = "fairmv/2.0"
        self._get, self._post = http_get, http_post   # allow injecting e.g. FetchData.http

    # --- http ---------------------------------------------------------------
    def _http(self, method: str, url: str, **kw):
        if method == "GET" and self._get:
            return self._get(url, **kw)
        if method == "POST" and self._post:
            return self._post(url, **kw)
        if self._session is None:
            raise RuntimeError("`requests` not installed: pip install requests")
        delay = 1.0
        for attempt in range(4):
            try:
                r = self._session.request(method, url, timeout=20, **kw)
                if r.status_code == 429 or r.status_code >= 500:
                    raise requests.HTTPError(str(r.status_code), response=r)
                r.raise_for_status()
                return r.json()
            except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as e:
                resp = getattr(e, "response", None)
                if resp is not None and 400 <= resp.status_code < 500 and resp.status_code != 429:
                    raise
                if attempt == 3:
                    raise
                time.sleep(delay)
                delay *= 2

    def _cached(self, key, ttl, fn):
        hit = self._cache.get(key)
        if hit and time.time() - hit[0] < ttl:
            return hit[1]
        val = fn()
        self._cache[key] = (time.time(), val)
        return val

    # --- API ----------------------------------------------------------------
    def market(self, ref: str) -> Optional[MarketMeta]:
        def load():
            if is_token_id(ref):
                ms = self._http("GET", f"{GAMMA_URL}/markets", params={"clob_token_ids": ref})
                if not ms:
                    return None
                toks = [str(t) for t in _jl(ms[0].get("clobTokenIds"))]
                return MarketMeta.from_gamma(ms[0], toks.index(ref) if ref in toks else 0)
            slug, outcome = split_ref(ref)
            ms = self._http("GET", f"{GAMMA_URL}/markets", params={"slug": slug})
            if not ms:
                return None
            outs = [str(o).lower() for o in _jl(ms[0].get("outcomes"))]
            oi = outs.index(outcome.lower()) if outcome and outcome.lower() in outs else 0
            return MarketMeta.from_gamma(ms[0], oi)
        return self._cached(("market", ref), self.ttl_meta, load)

    def event_markets(self, event_slug: str) -> list:
        def load():
            ev = self._http("GET", f"{GAMMA_URL}/events", params={"slug": event_slug})
            if not ev:
                return []
            out = []
            for m in ev[0].get("markets", []):
                m.setdefault("events", [{"slug": event_slug, "negRisk": ev[0].get("negRisk")}])
                out.append(MarketMeta.from_gamma(m, 0))
            return out
        return self._cached(("event", event_slug), self.ttl_meta, load)

    def books(self, token_ids: list) -> dict:
        now, out, missing = time.time(), {}, []
        for t in token_ids:
            hit = self._cache.get(("book", t))
            if hit and now - hit[0] < self.ttl_book:
                out[t] = hit[1]
            else:
                missing.append(t)
        for i in range(0, len(missing), 100):
            chunk = missing[i:i + 100]
            res = self._http("POST", f"{CLOB_URL}/books", json=[{"token_id": t} for t in chunk]) or []
            for raw in res:
                ob = OrderBook.from_clob_json(raw)
                self._cache[("book", ob.token_id)] = (time.time(), ob)
                out[ob.token_id] = ob
        return out

    def book(self, token_id: str) -> Optional[OrderBook]:
        return self.books([token_id]).get(token_id)

    def trades(self, meta: MarketMeta, max_n: int = 500) -> pd.DataFrame:
        def load():
            rows, cursor = [], None
            while len(rows) < max_n:
                p = {"condition": meta.condition_id, "limit": min(500, max_n)}
                if cursor:
                    p["cursor"] = cursor
                d = self._http("GET", TRADES_URL, params=p)
                data = d.get("data", []) if isinstance(d, dict) else (d or [])
                rows.extend(data)
                pag = (d.get("pagination") or {}) if isinstance(d, dict) else {}
                cursor = pag.get("next_cursor")
                if not data or not cursor or not pag.get("has_more"):
                    break
            if not rows:
                return normalize_trades(None)
            return normalize_trades(pd.DataFrame(rows), outcome_index=meta.outcome_index)
        return self._cached(("trades", meta.condition_id, meta.outcome_index), self.ttl_trades, load)
