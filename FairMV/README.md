# FairMV: fair value for thin, wide-spread Polymarket books

The old model blended four book estimators and clamped the result to the spread. On a
market whose book is `0.11 x 5 / 0.927 x 26`, every estimator only reshuffles two dust
orders, and the result looked like a precise number. This version treats **every signal
as an estimate with an uncertainty**, in log-odds, and combines them by precision. A 60c
spread now means the book has almost no say, and trades, related markets and your own
information take over.

```
            ┌ book    dust-filtered; centre = GM logit-mid / shrunk microprice / belief MLE / impact
 LOCAL ─────┤         var = uniform over the effective spread  (wide spread → weak)
            └ trades  side-corrected, staleness- & size-weighted; var from vol × age + dispersion
 LINKS        correlated markets (same / complement / linear / move / le / ge), event sums
 FEEDS        your views, scraped tables, plugin feeds: estimates, biases (shift), bounds
   │
   ▼  robust (Student-t) correlated inverse-variance fusion in log-odds
   ▼  + your shifts → Kalman tracker (smooths flicker, resets on news)
   ▼  → link/feed bounds → clamp to [best bid, best ask]
 FairValue: fair_value, fair_value_raw (unclamped = +EV take signal), sd, ci80, confidence,
            per-signal weights
```

## Files

| file | what |
|---|---|
| `core.py` | log-odds maths, `Estimate`, **`FVConfig` (every parameter, commented)** |
| `book.py` | `OrderBook`, dust filter, book estimators, book variance |
| `trades.py` | trade estimate, volatility σ², trade normalisation |
| `links.py` | correlated-market links and event sums |
| `feeds.py` | your views / tables / plugins |
| `combine.py` | robust fusion |
| `tracker.py` | Kalman smoothing |
| `data.py` | Polymarket API provider (cached); swap for your own |
| `engine.py` | `FairValueEngine`: the pipeline and the `FairValue` result |
| `model.py` | `FairValueModel`: book-only, stateless (old interface, used by FetchData) |
| `cli.py` | `py -m FairMV ...` |
| `user/` | **your stuff**: `views.json`, `links.json`, `config.json`, `feeds/*.py`, `tables/*` |
| `tests/` | `py -m pytest FairMV/tests -q` (offline) |
| `_legacy/` | the previous FairMV.py and pre-change copies of Scalper / MarketMaker / FetchData |

## Why each piece exists (the low-volume failure modes)

| failure on thin books | fix |
|---|---|
| a 5-share order at the touch sets the "mid" | **dust filter**: effective touch = first level with ≥ `dust_notional` ($10) of committed capital (bid p·s, ask (1−p)·s: YES/NO symmetric) |
| wide spread, but the mid is reported as if precise | book variance = (Δlogit spread)² / 12, the variance of a uniform over the spread |
| a 3000-share wall drags microprice to the ask (old: 0.82 on a 0.23/0.89 book) | imbalance shrunk by `imbalance_full_ticks / spread_ticks`; **belief MLE** moves only by (h/2)·ln(W_bid/W_ask), which is logarithmic and bounded |
| impact VWAP walks 60c into an empty side | impact only within `depth_window` of the effective touch |
| trades at the ask overstate value | per-trade side correction α·h, h capped (`trade_h_cap`) so a dust spread can't shove prints |
| yesterday's prints treated like fresh ones | trade variance + σ²·age, with exact random-walk covariance between trades |
| prints at 0.04 and 0.29 averaged into false precision | variance floored at the prints' empirical dispersion |
| bursts of prints seconds apart look like volatility | σ² from same-side prints ≥1h apart, print noise subtracted, shrunk to a prior |
| one stale/broken signal hijacks the answer | Student-t leave-one-out reweighting (`trust` column) |
| 10 similar signals → fake certainty | equicorrelated fusion variance (`fusion_corr`) |
| re-quoting on every flicker | Kalman tracker; jump > 4σ resets immediately (news) |
| thin leg of a multi-outcome event | automatic **event sum** for neg-risk events: p = 1 − Σ others |

## Adding your own information

Everything lives in `FairMV/user/` and is reloaded when the file changes, so you can
edit it while the scalper is running.

**A view or bias** (CLI writes `user/views.json`):
```
py -m FairMV view will-hexed-score-at-least-65-on-the-rotten-tomatoes-tomatometer-20261130 --p 0.62 --sd 0.04 --expires 48h --note "reviews strong"
py -m FairMV view "*pgl-major*" --shift -0.02          # esports YES legs trade rich: -2c after fusion
py -m FairMV view "q:acquired before 2028" --hi 0.6     # hard bound by question text
py -m FairMV view <match> --remove
```
`sd` is your confidence. 0.02 overrides most thin books, 0.15 is a nudge. A view fades
with age (variance + σ²·age), so refresh it or let it expire.

**Scraped feeds (tables)**: have your scraper append rows to any CSV / JSONL / Parquet
file in `user/tables/`:
```
key,p,sd,shift,ts,source,note
will-hexed-...-20261130,0.66,0.04,,2026-11-24T19:00:00Z,rt_scraper,20 reviews 85% fresh
```
The latest row per (key, source) wins. Files starting with `_` are ignored (see
`_example.csv`).

**Plugin feeds (live code)**: drop a `.py` into `user/feeds/` with
`def read(ctx): return {"p": .., "sd": ..} | {"shift": ..} | {"lo": .., "hi": ..} | None`.
See `user/feeds/example_feed.py`. A plugin that crashes produces a warning, never a crash.

From code: `FairValueEngine(feeds=[FunctionFeed(my_fn)])`.

**Correlated markets** (CLI writes `user/links.json`):
```
py -m FairMV link thin-market liquid-twin --relation same --noise-sd 0.1
py -m FairMV link will-periodic-labs-be-acquired-before-2028 will-ami-labs-be-acquired-before-2028 --relation move --beta 0.6 --noise-sd 0.3
py -m FairMV link us-diplomat-expelled-by-december-31-2026 us-diplomat-expelled-by-march-31-2027 --relation le
py -m FairMV event "*" --event pgl-major-singapore-2026-qualify-to-stage-3 --total 8
```
- `same`, `complement`, `linear` (x = α + βx_other): fixed mappings; `noise_sd` = how much you trust them.
- `move`: the level may differ, but changes carry over. The basis is **learnt online**,
  so the link adds no information on the first call and gains weight as the basis settles.
- `le` / `ge`: logical bounds (earlier deadline ≤ later deadline; "by 10+" ≤ "wins").
- event sums: neg-risk events (exactly one winner) are automatic. Use `--total K` for
  "K of N qualify" events, or `auto`.

Matching for views / tables / link targets: token id, condition id, exact slug, slug glob
(`*bitcoin*`), `slug:No` for a specific outcome, or `q:<question text>`. Anything written
without `:Outcome` refers to the first outcome (YES) and is **mirrored automatically**
when the NO token is valued.

## Using it

```
py -m FairMV show <slug> [<slug:No> ...] [--watch 5] [--both] [--no-links] [--no-feeds]
py -m FairMV demo                       # offline: plain mid vs this model on thin books
py Scalper.py scan / scalp ...          # uses the engine; --fv-user-dir, --no-fv-links, --no-fv-feeds
py MarketMaker.py --slug <slug>         # live quotes now use the full engine
```
```python
from FairMV import FairValueEngine
eng = FairValueEngine.from_user_dir()
fv = eng.evaluate(book, trades=df, meta={"slug": s, "condition_id": c, "token_id": t})
fv.fair_value, fv.sd, fv.ci80, fv.confidence, fv.fair_value_raw, fv.signals
```
`trades` can be Scalper's frame (yes_price / yes_side), raw Data-API rows, or
`ts, price, side(+1 buy), size`.

Scalper integration: the scan writes `fv_sd`, `fv_conf`, `fv_raw` and `fv_signals` (weight per
signal) per market. The paper quote log writes `fv_sd`, `fv_raw` and `fv_conf`. FV uncertainty is
also added to the MarketMaker's inventory variance (σ² += sd²/τ), which shrinks the Kelly size
when the fair value is unclear.

## Tuning

All knobs are in `FVConfig` (`core.py`). Override them in `user/config.json`, e.g.
`{"trade_side_adjust": 0.3, "dust_notional": 20}`. The ones that matter most:
- `trade_side_adjust`: 0 means takers are informed (trade price = new value); 1 means takers
  are noise (value = mid). Calibrate it with paper fills: if fills keep going against you
  right after a taker trade, lower it.
- `dust_notional`: raise it in markets where people leave $10–20 junk orders.
- `fusion_corr`, `tracker_floor`: how hard the model resists becoming overconfident.
- `event_residual_sd`: raise it for events whose listed outcomes may not be exhaustive.

What is still assumption rather than estimate: `trade_side_adjust`, `fusion_corr`, the σ²
prior, and the `noise_sd` of links you write. Check them against paper-trading fills
(`py Scalper.py report`: edge vs FV at fill).
