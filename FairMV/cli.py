"""
cli.py - command line for FairMV. Run from the PolyM folder:

  py -m FairMV show <slug|slug:No|token> [...] [--watch 5] [--both] [--no-trades/--no-links/--no-feeds]
  py -m FairMV top 10                                   highest-volume live markets
  py -m FairMV demo                                     offline: thin wide books, old mid vs new

  Your information (written to FairMV/user/):
  py -m FairMV view <match> --p 0.62 --sd 0.04 [--expires 48h] [--note "..."]
  py -m FairMV view <match> --shift 0.03                bias: +3c after fusion
  py -m FairMV view <match> --lo 0.2 --hi 0.5           hard bounds
  py -m FairMV view <match> --remove
  py -m FairMV views                                    list views
  py -m FairMV link <target> <other> --relation same|complement|linear|move|le|ge
                     [--beta 1 --alpha 0 --noise-sd 0.15 --margin 0]
  py -m FairMV event <target> [--event <event-slug>] --total 1|K|auto [--residual-sd 0.02]
  py -m FairMV links                                    list links
  <match> = slug, slug glob ("*bitcoin*"), slug:Outcome, token id, condition id, or "q:<question text>".
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import pandas as pd

from .book import OrderBook
from .core import FVConfig
from .engine import DEFAULT_USER_DIR, FairValueEngine
from .feeds import _ts


# =============================================================================
# JSON helpers
# =============================================================================
def _read(path: str, default: dict) -> dict:
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return dict(default)


def _write(path: str, d: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2)
    os.replace(tmp, path)


# =============================================================================
# Commands
# =============================================================================
def cmd_show(a) -> None:
    eng = FairValueEngine.from_user_dir(a.user_dir)
    refs = list(a.refs)
    if a.both:
        more = []
        for r in refs:
            m = eng.provider.market(r)
            if m is not None and len(m.tokens) == 2:
                more += [m.tokens[0], m.tokens[1]]
            else:
                more.append(r)
        refs = more

    def once():
        for r in refs:
            try:
                meta = eng.provider.market(r)
                if meta is None:
                    print(f"\n=== {r}: not found")
                    continue
                fv = eng.evaluate_token(r, with_trades=not a.no_trades,
                                        use_links=False if a.no_links else None,
                                        use_feeds=False if a.no_feeds else None)
                print(f"\n=== {meta.question}  [{meta.outcome}]  {meta.slug}")
                print(fv)
            except Exception as e:  # noqa: BLE001
                print(f"\n=== {r}: error {type(e).__name__}: {e}")

    if a.watch:
        try:
            while True:
                print(f"\n----- {time.strftime('%H:%M:%S')} -----")
                once()
                time.sleep(a.watch)
        except KeyboardInterrupt:
            pass
    else:
        once()


def cmd_top(a) -> None:
    eng = FairValueEngine.from_user_dir(a.user_dir)
    d = eng.provider._http("GET", "https://gamma-api.polymarket.com/markets",
                           params={"active": "true", "closed": "false", "limit": 20 * a.n,
                                   "order": "volume24hr", "ascending": "false"})
    live = [m for m in d if m.get("acceptingOrders") and m.get("enableOrderBook")
            and m.get("bestBid") is not None and m.get("bestAsk") is not None
            and 0.02 < float(m["bestBid"]) and float(m["bestAsk"]) < 0.98][: a.n]
    for m in live:
        a.refs = [m["slug"]]
        a.both, a.watch, a.no_trades, a.no_links, a.no_feeds = False, None, False, False, False
        cmd_show(a)


def cmd_demo(_a) -> None:
    eng = FairValueEngine(FVConfig(use_tracker=False))
    now = pd.Timestamp.now(tz="UTC")
    cases = {
        "tight, balanced": OrderBook.from_levels([(0.54, 800), (0.53, 1500)], [(0.55, 700), (0.56, 1200)]),
        "wide + dust at touch": OrderBook.from_levels([(0.11, 5), (0.04, 20)],
                                                     [(0.927, 26), (0.928, 28), (0.93, 5)], tick_size=0.001),
        "wide + whale wall on bid": OrderBook.from_levels([(0.24, 33), (0.23, 3000), (0.02, 28)],
                                                         [(0.89, 72), (0.90, 21), (0.91, 5)]),
    }
    trades = pd.DataFrame({"ts": [now - pd.Timedelta(hours=h) for h in (30, 9, 5, 2, 0.5)],
                           "price": [0.50, 0.60, 0.57, 0.62, 0.58], "side": [1, 1, -1, 1, -1],
                           "size": [100, 40, 60, 25, 80]})
    for name, ob in cases.items():
        print(f"\n### {name}:  bids {ob.bids[:3]}  asks {ob.asks[:3]}")
        print(f"  plain mid {ob.mid:.4f}")
        print("  -- book only --")
        print(eng.evaluate(ob))
        if ob.spread > 0.1:
            print("  -- with 5 recent trades around 0.58 --")
            print(eng.evaluate(ob, trades))


def cmd_view(a) -> None:
    path = os.path.join(a.user_dir, "views.json")
    d = _read(path, {"views": []})
    views = [v for v in d.get("views", []) if v.get("match") != a.match]
    if not a.remove:
        now = pd.Timestamp.now(tz="UTC")
        v = {"match": a.match, "ts": now.isoformat()}
        for k in ("p", "sd", "shift", "lo", "hi"):
            if getattr(a, k) is not None:
                v[k] = getattr(a, k)
        if a.expires:
            v["expires"] = _ts(a.expires, now).isoformat()
        if a.note:
            v["note"] = a.note
        if len(v) <= 2:
            raise SystemExit("give at least one of --p, --shift, --lo, --hi (or --remove)")
        views.append(v)
    d["views"] = views
    _write(path, d)
    print(f"{'removed' if a.remove else 'saved'} view for '{a.match}' -> {path}")


def cmd_views(a) -> None:
    d = _read(os.path.join(a.user_dir, "views.json"), {"views": []})
    for v in d.get("views", []):
        print(" ", json.dumps(v))
    if not d.get("views"):
        print("  (no views)")


def cmd_link(a) -> None:
    path = os.path.join(a.user_dir, "links.json")
    d = _read(path, {"links": [], "event_sums": []})
    spec = {"target": a.target, "other": a.other, "relation": a.relation, "beta": a.beta,
            "alpha": a.alpha, "noise_sd": a.noise_sd, "margin": a.margin}
    if a.name:
        spec["name"] = a.name
    d.setdefault("links", []).append(spec)
    _write(path, d)
    print(f"saved link {a.target} --{a.relation}--> {a.other} -> {path}")


def cmd_event(a) -> None:
    path = os.path.join(a.user_dir, "links.json")
    d = _read(path, {"links": [], "event_sums": []})
    total = a.total if a.total == "auto" else float(a.total)
    spec = {"target": a.target, "event": a.event, "total": total}
    if a.residual_sd is not None:
        spec["residual_sd"] = a.residual_sd
    d.setdefault("event_sums", []).append(spec)
    _write(path, d)
    print(f"saved event sum for {a.target} (total {total}) -> {path}")


def cmd_links(a) -> None:
    d = _read(os.path.join(a.user_dir, "links.json"), {})
    for k in ("links", "event_sums"):
        print(f"{k}:")
        for v in d.get(k, []):
            print("  ", json.dumps(v))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="py -m FairMV", description="Polymarket fair value (thin-market aware)")
    ap.add_argument("--user-dir", default=DEFAULT_USER_DIR)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("show", help="evaluate markets live")
    s.add_argument("refs", nargs="+")
    s.add_argument("--watch", type=float)
    s.add_argument("--both", action="store_true", help="both outcome tokens")
    s.add_argument("--no-trades", action="store_true")
    s.add_argument("--no-links", action="store_true")
    s.add_argument("--no-feeds", action="store_true")
    s = sub.add_parser("top", help="evaluate the N highest-volume live markets")
    s.add_argument("n", type=int)
    sub.add_parser("demo", help="offline synthetic examples")

    s = sub.add_parser("view", help="add / replace / remove one of your views")
    s.add_argument("match")
    s.add_argument("--p", type=float)
    s.add_argument("--sd", type=float, help="prob sd = your confidence (default 0.05)")
    s.add_argument("--shift", type=float, help="bias in probability points, applied after fusion")
    s.add_argument("--lo", type=float)
    s.add_argument("--hi", type=float)
    s.add_argument("--expires", help="ISO time or relative: 48h, 3d")
    s.add_argument("--note")
    s.add_argument("--remove", action="store_true")
    sub.add_parser("views")

    s = sub.add_parser("link", help="add a correlated-market link")
    s.add_argument("target")
    s.add_argument("other")
    s.add_argument("--relation", default="same", choices=["same", "complement", "linear", "move", "le", "ge"])
    s.add_argument("--beta", type=float, default=1.0)
    s.add_argument("--alpha", type=float, default=0.0)
    s.add_argument("--noise-sd", type=float, default=0.15)
    s.add_argument("--margin", type=float, default=0.0)
    s.add_argument("--name")
    s = sub.add_parser("event", help="add an event-sum constraint")
    s.add_argument("target")
    s.add_argument("--event")
    s.add_argument("--total", default="1")
    s.add_argument("--residual-sd", type=float)
    sub.add_parser("links")

    a = ap.parse_args(argv)
    {"show": cmd_show, "top": cmd_top, "demo": cmd_demo, "view": cmd_view, "views": cmd_views,
     "link": cmd_link, "event": cmd_event, "links": cmd_links}[a.cmd](a)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
