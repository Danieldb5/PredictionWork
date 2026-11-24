# predictionwork

Research tooling for Polymarket prediction markets: data collection, fair-value
estimation, market-making maths, and **paper** trading of thin, wide-spread markets.

> **No real orders are ever placed.** Every trading component here is a simulation:
> fills are inferred from real trades printed on Polymarket while virtual quotes were resting.

## Layout

| path | what |
|---|---|
| [`FetchData.py`](FetchData.py) | Live order-book recorder. Polls the CLOB book of chosen/top/all markets at a fixed interval and writes full-depth snapshots + per-level rows to Parquet. |
| [`FetchHistory.py`](FetchHistory.py) | Bulk download of what already exists: live + resolved markets (Gamma), one book snapshot per token, hourly/minute price history, recent trades. Resumable. |
| [`BuildML.py`](BuildML.py) | Turns the raw downloads (plus the HuggingFace `polymarket-crypto-5m-15m` dataset) into clean, typed, ML-ready Parquet under `data/ml/`. |
| [`TipWindows.py`](TipWindows.py) | Cuts price/book data into "tipping point" windows (PRE + TIP + POST grid points × 50 features) using an arcsine variance-stabilised move threshold. Output tensor `(n_tips, 120, 50)`. |
| [`FairMV/`](FairMV/README.md) | Fair-value engine for thin books: book, trades, linked markets and user views fused by precision in log-odds, Student-t robust, Kalman-smoothed. See its own README. |
| [`MarketMaker.py`](MarketMaker.py) | Market-making model (Glosten–Milgrom adverse selection, Avellaneda–Stoikov arrival rates and inventory skew, fractional Kelly sizing). Prints recommended quotes; includes a Monte Carlo simulator. |
| [`Scalper.py`](Scalper.py) | Scans for low-volume, wide-spread markets, calibrates the market maker to each from its trade history, downloads data and paper-trades the spread. |
| [`PaperTrades/`](PaperTrades/) | Event-driven paper exchange shared by live (websocket / REST poll) and backtest (recorded-data replay) runs, with queue-position modelling. Work in progress. |

## Setup

Python 3.11+ (developed on 3.13, Windows).

```bash
pip install -r requirements.txt
```

All data comes from Polymarket's public APIs (Gamma, CLOB, Data API); no keys are needed.

## Quick start

```bash
# collect data
py FetchHistory.py all                       # bulk history -> data/history/
py FetchData.py --top 50 --interval 5        # live books -> data/snapshots, data/levels
py BuildML.py                                # -> data/ml/
py TipWindows.py                             # tipping-point windows

# fair value / quoting
py -m FairMV --help
py MarketMaker.py --demo
py MarketMaker.py --simulate

# paper trading
py Scalper.py scan
py Scalper.py download --n 10
py Scalper.py scalp --n 5 --interval 15 --duration 3600 --bankroll 1000
py Scalper.py report

# tests
py -m pytest FairMV/tests -q
```

Each script's module docstring documents its full options, output schema and maths.

## Data

`data/` is **not** committed (it is ~1.6 GB of Parquet). Regenerate it with the
`Fetch*` scripts and `BuildML.py`. The external crypto dataset is
[BrockMisner/polymarket-crypto-5m-15m](https://huggingface.co/datasets/BrockMisner/polymarket-crypto-5m-15m)
(MIT), expected at `data/external/hf_crypto_5m15m/`.

## Disclaimer

This is research code for educational purposes. Nothing here is financial advice.
