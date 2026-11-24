"""
FairMV - fair value for Polymarket binary tokens that survives thin, wide-spread books.

    from FairMV import FairValueEngine, OrderBook
    eng = FairValueEngine.from_user_dir()          # your views / links / feeds in FairMV/user/
    fv = eng.evaluate(book, trades=df, meta={...})
    print(fv)

Modules
    core.py      log-odds maths, Estimate, FVConfig (all parameters)
    book.py      order book; dust filter, belief MLE, shrunk microprice, spread variance
    trades.py    side-corrected, staleness-weighted trade estimate; volatility
    links.py     correlated markets: same / complement / linear / move / le / ge, event sums
    feeds.py     your views, scraped tables, plugin feeds, biases (shifts), bounds
    combine.py   robust correlated precision-weighted fusion
    tracker.py   Kalman smoothing across polls with news-jump reset
    data.py      Polymarket API provider (cached)
    engine.py    FairValueEngine: the pipeline
    model.py     FairValueModel: book-only, stateless (old interface)
    cli.py       py -m FairMV ...
See README.md for the theory and how to plug in your own information.
"""

from .book import OrderBook, analyse_book
from .core import Bound, Estimate, FVConfig, logit, sigmoid
from .data import MarketMeta, PolymarketProvider
from .engine import DEFAULT_USER_DIR, FairValue, FairValueEngine, FairValueResult
from .feeds import FeedContext, FunctionFeed, PluginFeed, Reading, TableFeed, ViewsFile
from .links import EventSumSpec, LinkBook, LinkSpec
from .model import FairValueModel
from .tracker import Tracker


def fetch_book(token_id: str) -> OrderBook:
    """Convenience: one live order book (old FairMV.fetch_book)."""
    ob = PolymarketProvider(ttl_book=0).book(token_id)
    if ob is None:
        raise RuntimeError("no order book (market closed/resolved or token id wrong)")
    return ob


__all__ = ["OrderBook", "analyse_book", "Bound", "Estimate", "FVConfig", "logit", "sigmoid",
           "MarketMeta", "PolymarketProvider", "DEFAULT_USER_DIR", "FairValue", "FairValueEngine",
           "FairValueResult", "FeedContext", "FunctionFeed", "PluginFeed", "Reading", "TableFeed",
           "ViewsFile", "EventSumSpec", "LinkBook", "LinkSpec", "FairValueModel", "Tracker", "fetch_book"]
