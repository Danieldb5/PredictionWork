"""Offline tests for FairMV.  Run from PolyM:  py -m pytest FairMV/tests -q"""

from __future__ import annotations

import json
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from FairMV import (EventSumSpec, FairValueEngine, FairValueModel, FunctionFeed, FVConfig, LinkBook,  # noqa: E402
                    LinkSpec, MarketMeta, OrderBook, PluginFeed, TableFeed, ViewsFile)
from FairMV.combine import fuse  # noqa: E402
from FairMV.core import Estimate, logit  # noqa: E402
from FairMV.tracker import Tracker  # noqa: E402
from FairMV.trades import estimate_sigma2, normalize_trades  # noqa: E402

NOW = pd.Timestamp("2026-11-24T18:00:00Z")
CFG = FVConfig(use_tracker=False)


def wide_dust_book(**kw):
    return OrderBook.from_levels([(0.11, 5), (0.04, 20)], [(0.927, 26), (0.928, 28), (0.93, 5)],
                                 tick_size=0.001, **kw)


def trades_at(prices, sides, hours_ago, sizes=None):
    return pd.DataFrame({"ts": [NOW - pd.Timedelta(hours=h) for h in hours_ago], "price": prices,
                         "side": sides, "size": sizes or [50] * len(prices)})


# ----------------------------------------------------------------------------- book
def test_tight_book_near_mid_high_confidence():
    ob = OrderBook.from_levels([(0.54, 800), (0.53, 1500)], [(0.55, 700), (0.56, 1200)])
    fv = FairValueEngine(CFG).evaluate(ob, now=NOW)
    assert 0.54 <= fv.fair_value <= 0.55
    assert abs(fv.fair_value - 0.545) < 0.003
    assert fv.confidence == "high"


def test_wide_book_is_uncertain():
    fv = FairValueEngine(CFG).evaluate(wide_dust_book(), now=NOW)
    assert fv.sd > 0.2 and fv.confidence == "none"
    assert fv.bid <= fv.fair_value <= fv.ask


def test_dust_filtered_from_touch():
    # 2 shares at 0.40 in front of a real 0.30 bid
    ob = OrderBook.from_levels([(0.40, 2), (0.30, 500)], [(0.60, 500)])
    fv = FairValueModel(dust_notional=10).evaluate(ob)
    assert fv.diagnostics["eff_bid"] == 0.30 and fv.diagnostics["dust_bid"]


def test_whale_wall_does_not_drag_to_ask():
    ob = OrderBook.from_levels([(0.24, 33), (0.23, 3000)], [(0.89, 72), (0.90, 21)])
    fv = FairValueModel().evaluate(ob)
    assert fv.fair_value < 0.72        # old model said 0.82


def test_yes_no_consistency():
    ob = OrderBook.from_levels([(0.31, 40), (0.30, 300), (0.25, 90)], [(0.45, 25), (0.47, 400)], last_trade=0.4)
    m = FairValueModel()
    assert abs(m.evaluate(ob).fair_value + m.evaluate(ob.complement()).fair_value - 1) < 1e-6


def test_one_sided_and_empty_books():
    m = FairValueModel()
    assert m.evaluate(OrderBook.from_levels([], [])).fair_value is None
    fv = m.evaluate(OrderBook.from_levels([(0.2, 100)], [], last_trade=0.35))
    assert fv.fair_value is not None and fv.fair_value >= 0.2


# ----------------------------------------------------------------------------- trades
def test_trades_dominate_wide_book_and_stay_in_spread():
    tr = trades_at([0.60, 0.57, 0.62, 0.58], [1, -1, 1, -1], [9, 5, 2, 0.5])
    fv = FairValueEngine(CFG).evaluate(wide_dust_book(), tr, now=NOW)
    assert 0.54 < fv.fair_value < 0.63
    w = {s["name"]: s["weight"] for s in fv.signals}
    assert w["trades"] > 0.9


def test_stale_trades_lose_weight():
    eng = FairValueEngine(CFG)
    ob = OrderBook.from_levels([(0.40, 300)], [(0.50, 300)])
    fresh = eng.evaluate(ob, trades_at([0.48, 0.47], [1, -1], [0.2, 0.1]), now=NOW)
    stale = eng.evaluate(ob, trades_at([0.48, 0.47], [1, -1], [140, 139]), now=NOW)
    wf = {s["name"]: s["weight"] for s in fresh.signals}["trades"]
    ws = {s["name"]: s["weight"] for s in stale.signals}["trades"]
    assert wf > ws


def test_one_sided_buys_are_side_corrected():
    eng = FairValueEngine(CFG)
    ob = OrderBook.from_levels([(0.40, 300)], [(0.60, 300)])
    fv = eng.evaluate(ob, trades_at([0.60] * 4, [1] * 4, [3, 2, 1, 0.5]), now=NOW)
    t = next(s for s in fv.signals if s["name"] == "trades")
    assert t["p"] < 0.60                     # buys at the ask overstate value


def test_scalper_yes_columns_and_no_token():
    df = pd.DataFrame({"ts": [NOW - pd.Timedelta(hours=1)], "yes_price": [0.7], "yes_side": [1], "size": [10]})
    y = normalize_trades(df, 0)
    n = normalize_trades(df, 1)
    assert y.price[0] == 0.7 and y.side[0] == 1
    assert abs(n.price[0] - 0.3) < 1e-12 and n.side[0] == -1


def test_raw_api_rows_converted_to_token_terms():
    raw = pd.DataFrame({"timestamp": [1795540570, 1795540580], "price": [0.3, 0.72], "side": ["BUY", "SELL"],
                        "size": [10, 5], "outcome_index": [1, 0]})
    t = normalize_trades(raw, 0)
    assert abs(t.price[0] - 0.7) < 1e-12 and t.side[0] == -1        # NO buy = YES sell at 0.7
    assert t.price[1] == 0.72 and t.side[1] == -1


def test_dust_spread_does_not_blow_up_side_correction():
    # live case: book 0.11 / 0.926 dust, only buys -> must not shove prints far below their price
    tr = trades_at([0.039, 0.039, 0.288], [1, 1, 1], [7.2, 7.1, 3.7], sizes=[28, 50, 5])
    fv = FairValueEngine(CFG).evaluate(wide_dust_book(), tr, now=NOW)
    t = next(s for s in fv.signals if s["name"] == "trades")
    assert t["p"] > 0.03
    assert t["info"]["h_used"] <= CFG.trade_h_cap
    # prints at 0.04 and 0.29 disagree: estimate must be uncertain (dispersion floor)
    assert t["info"]["sd_dispersion"] > 0.3


def test_sigma2_ignores_bursts_of_prints():
    # 40 buys seconds apart alternating 0.74/0.76: noise, not volatility
    ts = [NOW - pd.Timedelta(seconds=10 * i) for i in range(40)]
    tr = pd.DataFrame({"ts": ts, "price": [0.74, 0.76] * 20, "side": [1] * 40, "size": [10] * 40})
    s2, info = estimate_sigma2(normalize_trades(tr), CFG)
    assert info["sigma2_pairs"] == 0 and s2 < 0.01


def test_impact_ignores_far_levels():
    ob = OrderBook.from_levels([(0.23, 3000)], [(0.29, 30), (0.89, 500)])
    fv = FairValueModel().evaluate(ob)
    assert fv.components["impact_mid"] < 0.35


def test_sigma2_shrinks_to_prior_without_data():
    s2, _ = estimate_sigma2(normalize_trades(None), CFG)
    assert abs(s2 - CFG.sigma2_prior) < 1e-12


# ----------------------------------------------------------------------------- fusion / tracker
def test_robust_fusion_downweights_outlier():
    es = [Estimate("a", logit(0.50), 0.01), Estimate("b", logit(0.52), 0.01), Estimate("bad", logit(0.95), 0.01)]
    fz = fuse(es, CFG)
    assert fz.robust["bad"] < 0.3 and fz.p < 0.62


def test_correlation_prevents_overconfidence():
    es = [Estimate(f"e{i}", 0.0, 0.04) for i in range(10)]
    indep = fuse(es, FVConfig(fusion_corr=0.0, robust_nu=0)).var
    corr = fuse(es, FVConfig(fusion_corr=0.2, robust_nu=0)).var
    assert abs(indep - 0.004) < 1e-9 and corr > 2 * indep


def test_tracker_smooths_flicker_and_resets_on_jump():
    tk = Tracker(floor=0.35, jump_z=4)
    x0, _, _ = tk.update("t", 0.0, 0.04, 0, 0.004)
    x1, _, _ = tk.update("t", 0.2, 0.04, 15, 0.004)          # 1 sd flicker 15 s later
    assert 0 < x1 < 0.2
    x2, _, info = tk.update("t", 3.0, 0.04, 30, 0.004)       # news
    assert x2 == 3.0 and info["tracker"] == "jump-reset"


# ----------------------------------------------------------------------------- feeds
def test_views_file_estimate_shift_bounds_and_mirror(tmp_path):
    p = tmp_path / "views.json"
    p.write_text(json.dumps({"views": [
        {"match": "my-mkt", "p": 0.70, "sd": 0.02},
        {"match": "*mkt*", "shift": 0.01},
        {"match": "q:some question", "hi": 0.66},
    ]}))
    eng = FairValueEngine(CFG, feeds=[ViewsFile(str(p))])
    meta = {"slug": "my-mkt", "question": "Some question?", "token_id": "1", "tokens": ["1", "2"]}
    fv = eng.evaluate(wide_dust_book(), meta=meta, now=NOW)
    assert abs(fv.fair_value - 0.66) < 1e-9 and fv.diagnostics["bounds_active"]
    assert fv.fair_value_raw == fv.fair_value
    # NO token: everything mirrored -> p ~ 0.3 - shift, bound becomes lo = 0.34
    fvn = eng.evaluate(wide_dust_book().complement(), meta={**meta, "token_id": "2"}, now=NOW)
    assert abs(fvn.fair_value - 0.34) < 1e-9


def test_expired_view_ignored(tmp_path):
    p = tmp_path / "views.json"
    p.write_text(json.dumps({"views": [{"match": "my-mkt", "p": 0.9, "sd": 0.01, "expires": "2020-01-01"}]}))
    fv = FairValueEngine(CFG, feeds=[ViewsFile(str(p))]).evaluate(wide_dust_book(), meta={"slug": "my-mkt"}, now=NOW)
    assert not any(s["kind"] == "feed" for s in fv.signals)


def test_table_feed_latest_row_wins(tmp_path):
    (tmp_path / "scrape.csv").write_text("key,p,sd,ts\nmy-mkt,0.30,0.02,2026-11-24T10:00:00Z\n"
                                        "my-mkt,0.75,0.02,2026-11-24T17:00:00Z\n")
    fv = FairValueEngine(CFG, feeds=[TableFeed(str(tmp_path))]).evaluate(
        wide_dust_book(), meta={"slug": "my-mkt"}, now=NOW)
    f = next(s for s in fv.signals if s["kind"] == "feed")
    assert abs(f["p"] - 0.75) < 1e-9
    assert fv.fair_value > 0.65


def test_plugin_feed_and_broken_plugin(tmp_path):
    (tmp_path / "good.py").write_text("NAME='good'\ndef read(ctx):\n    return {'p': 0.2, 'sd': 0.02} if 'mkt' in ctx.slug else None\n")
    (tmp_path / "bad.py").write_text("def read(ctx):\n    raise ValueError('boom')\n")
    (tmp_path / "_skip.py").write_text("def read(ctx):\n    return 0.99\n")
    fv = FairValueEngine(CFG, feeds=[PluginFeed(str(tmp_path))]).evaluate(
        wide_dust_book(), meta={"slug": "my-mkt"}, now=NOW)
    assert any(s["name"] == "feed:good" for s in fv.signals)
    assert fv.fair_value < 0.3
    assert any("boom" in w for w in fv.warnings)


def test_function_feed_outside_book_reports_raw():
    ob = OrderBook.from_levels([(0.40, 500)], [(0.42, 500)])
    eng = FairValueEngine(CFG, feeds=[FunctionFeed(lambda ctx: {"p": 0.6, "sd": 0.005}, "mine")])
    fv = eng.evaluate(ob, meta={"slug": "x"}, now=NOW)
    assert fv.fair_value == 0.42 and fv.fair_value_raw > 0.42 and fv.diagnostics["edge_buy_at_ask"] > 0


# ----------------------------------------------------------------------------- links
class FakeProvider:
    def __init__(self, metas: dict, books: dict, events: dict = None):
        self.metas, self._books, self.events = metas, books, events or {}

    def market(self, ref):
        return self.metas.get(ref)

    def event_markets(self, slug):
        return self.events.get(slug, [])

    def books(self, toks):
        return {t: self._books[t] for t in toks if t in self._books}

    def book(self, t):
        return self._books.get(t)

    def trades(self, meta, n):
        return None


def M(tok, slug, cond, **kw):
    return MarketMeta(token_id=tok, slug=slug, condition_id=cond, tokens=[tok, tok + "n"], outcomes=["Yes", "No"], **kw)


def test_same_link_pulls_thin_market():
    liquid = OrderBook.from_levels([(0.69, 2000)], [(0.70, 2000)])
    prov = FakeProvider({"liquid": M("L", "liquid", "c2")}, {"L": liquid})
    eng = FairValueEngine(CFG, links=LinkBook([LinkSpec("thin", "liquid", "same", noise_sd=0.05)]), provider=prov)
    fv = eng.evaluate(wide_dust_book(), meta=M("T", "thin", "c1"), now=NOW)
    assert abs(fv.fair_value - 0.695) < 0.02


def test_le_bound_link():
    other = OrderBook.from_levels([(0.29, 2000)], [(0.30, 2000)])
    prov = FakeProvider({"later": M("O", "later", "c2")}, {"O": other})
    eng = FairValueEngine(CFG, links=LinkBook([LinkSpec("earlier", "later", "le")]), provider=prov)
    fv = eng.evaluate(wide_dust_book(), trades_at([0.6, 0.6], [1, -1], [1, 0.5]),
                      meta=M("T", "earlier", "c1"), now=NOW)
    assert fv.fair_value <= 0.296


def test_move_link_learns_basis_then_tracks_moves():
    prov = FakeProvider({"other": M("O", "other", "c2")}, {})
    lb = LinkBook([LinkSpec("thin", "other", "move", beta=1.0, noise_sd=0.05)])
    eng = FairValueEngine(FVConfig(use_tracker=False, move_basis_q=0.0), links=lb, provider=prov)
    own = OrderBook.from_levels([(0.29, 3000)], [(0.31, 3000)])     # thin market sits at 0.30
    meta = M("T", "thin", "c1")
    t = NOW
    for _ in range(5):                                              # other at 0.50 while we sit at 0.30
        prov._books["O"] = OrderBook.from_levels([(0.495, 3000)], [(0.505, 3000)])
        eng._local_cache.clear()
        eng.evaluate(own, meta=meta, now=t)
        t += pd.Timedelta(minutes=1)
    # the other market jumps to 0.70; our book is now wide/stale
    prov._books["O"] = OrderBook.from_levels([(0.695, 3000)], [(0.705, 3000)])
    eng._local_cache.clear()
    fv = eng.evaluate(OrderBook.from_levels([(0.05, 3000)], [(0.95, 3000)]), meta=meta, now=t)
    link = next(s for s in fv.signals if s["kind"] == "link")
    assert 0.45 < link["p"] < 0.55          # 0.30 basis + the other's move, in log-odds
    assert 0.40 < fv.fair_value < 0.60


def test_neg_risk_event_sum():
    legs = [M("A", "a", "ca", neg_risk=True, event_slug="ev"), M("B", "b", "cb", neg_risk=True, event_slug="ev"),
            M("C", "c", "cc", neg_risk=True, event_slug="ev")]
    books = {"B": OrderBook.from_levels([(0.19, 2000)], [(0.21, 2000)]),
             "C": OrderBook.from_levels([(0.09, 2000)], [(0.11, 2000)])}
    prov = FakeProvider({}, books, {"ev": legs})
    eng = FairValueEngine(CFG, provider=prov)
    fv = eng.evaluate(wide_dust_book(), meta=legs[0], now=NOW)
    link = next(s for s in fv.signals if s["kind"] == "link")
    assert abs(link["p"] - 0.70) < 0.01
    assert abs(fv.fair_value - 0.70) < 0.03
    # valuing the NO token of leg A mirrors it
    fvn = eng.evaluate(wide_dust_book().complement(), meta=legs[0].other(), now=NOW)
    assert abs(fvn.fair_value - 0.30) < 0.03


def test_event_sum_total_k():
    legs = [M(t, t.lower(), "c" + t, event_slug="q8") for t in "ABCD"]
    books = {t: OrderBook.from_levels([(0.59, 2000)], [(0.61, 2000)]) for t in "BCD"}
    prov = FakeProvider({}, books, {"q8": legs})
    lb = LinkBook(events=[EventSumSpec(target="*", event="q8", total=2)])
    fv = FairValueEngine(CFG, links=lb, provider=prov).evaluate(wide_dust_book(), meta=legs[0], now=NOW)
    link = next(s for s in fv.signals if s["kind"] == "link")
    assert abs(link["p"] - 0.20) < 0.01


def test_broken_link_never_raises():
    class Boom(FakeProvider):
        def market(self, ref):
            raise ConnectionError("down")
    eng = FairValueEngine(CFG, links=LinkBook([LinkSpec("thin", "x")]), provider=Boom({}, {}))
    fv = eng.evaluate(wide_dust_book(), meta=M("T", "thin", "c1"), now=NOW)
    assert fv.fair_value is not None and any("down" in w for w in fv.warnings)


def test_user_dir_examples_load():
    from FairMV.engine import DEFAULT_USER_DIR
    eng = FairValueEngine.from_user_dir(DEFAULT_USER_DIR, provider=None)
    fv = eng.evaluate(wide_dust_book(), meta={"slug": "anything"}, now=NOW)
    assert fv.fair_value is not None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
