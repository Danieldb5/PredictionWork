"""
model.py - FairValueModel: the old interface, now book (+ last trade) only and
wide-spread aware. Stateless: no links, no feeds, no smoothing. Used by
FetchData.py recordings and anything that only has a book.

    FairValueModel().evaluate(book) -> FairValue   (.fair_value, .confidence, .diagnostics, ...)

For trades, correlated markets and your own views use FairValueEngine.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Optional

from .book import OrderBook
from .core import FVConfig
from .engine import FairValue, FairValueEngine


class FairValueModel:
    def __init__(self, cfg: Optional[FVConfig] = None, **overrides):
        cfg = replace(cfg or FVConfig(), use_tracker=False, use_links=False, use_feeds=False, **overrides)
        self.engine = FairValueEngine(cfg, provider=None, auto_meta=False)

    def evaluate(self, book: OrderBook, trades=None) -> FairValue:
        return self.engine.evaluate(book, trades)
