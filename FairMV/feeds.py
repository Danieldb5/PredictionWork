"""
feeds.py - your own information: views, biases, scraped feeds and plugins.

Every feed returns Readings. A Reading can carry any of:
    p, sd     your probability and how sure you are (prob sd). Enters the
              fusion like any other signal, so a confident view (sd 0.02)
              dominates a 60c book, while a hunch (sd 0.15) barely moves a
              tight one. Staleness: var grows with sigma^2 * age, exactly as
              for trades, so an old view fades unless you refresh it.
    shift     a BIAS in probability points applied AFTER fusion (e.g. +0.03:
              "this market is always too low"). shift_logit is the same in
              log-odds.
    lo, hi    hard bounds on fair value.
    expires   ignored after this time.

Matching ("key" / "match"), checked in this order:
    exact token id | exact condition id | "q:<text>" (question contains text)
    | slug or slug glob ("*bitcoin*", "will-x-win*"), optionally ":Outcome".
A view without ":Outcome" is about the FIRST outcome (YES). When the NO token
is valued, it is mirrored automatically (p -> 1-p, shift -> -shift, bounds
swapped).

Feed types
----------
  ViewsFile(path)     JSON {"views": [{"match": ..., "p": .., "sd": .., "shift": .., "lo": .., "hi": ..,
                      "ts": iso, "expires": iso or "48h", "note": ..}]}. Hand-edited, or use
                      `py -m FairMV view <slug> --p 0.6 --sd 0.04`. Reloaded when the file changes.
  TableFeed(path)     CSV / JSONL / Parquet file OR a folder of them, for scrapers. Columns:
                      key, p[, sd, shift, lo, hi, ts, expires, source, note]. The latest row
                      per key wins (by ts, else file order). Your scraper just appends rows.
  PluginFeed(folder)  every *.py in the folder (not starting with _) that defines
                      `read(ctx) -> Reading | dict | list | None`. For live scrapers,
                      models, anything. Reloaded when the file changes; exceptions are
                      caught and reported as warnings, never raised into the trader.
  FunctionFeed(fn)    any callable with the same signature, from code.
"""

from __future__ import annotations

import fnmatch
import importlib.util
import json
import math
import os
from dataclasses import dataclass, field
from typing import Callable, Optional

import pandas as pd

from .core import Bound, Estimate, FVConfig, clip_p, logit, opt_float, var_logit_from_sd_prob


# =============================================================================
# Context and readings
# =============================================================================
@dataclass
class FeedContext:
    token_id: str
    slug: str = ""
    question: str = ""
    condition_id: str = ""
    outcome: str = ""
    outcome_index: int = 0
    event_slug: str = ""
    now: pd.Timestamp = field(default_factory=lambda: pd.Timestamp.now(tz="UTC"))
    p_market: Optional[float] = None     # this market's local (book+trades) fair value
    book: object = None                  # OrderBook
    extra: dict = field(default_factory=dict)


def _ts(v, now: pd.Timestamp) -> Optional[pd.Timestamp]:
    """ISO string / epoch seconds / Timestamp / relative '48h', '3d' (from now) -> UTC Timestamp."""
    if v is None or (isinstance(v, float) and math.isnan(v)) or v == "":
        return None
    if isinstance(v, str) and v[-1:] in ("h", "d", "m") and v[:-1].replace(".", "", 1).isdigit():
        mult = {"m": 1 / 60, "h": 1, "d": 24}[v[-1]]
        return now + pd.Timedelta(hours=float(v[:-1]) * mult)
    if isinstance(v, (int, float)):
        return pd.Timestamp(v, unit="s", tz="UTC")
    t = pd.Timestamp(v)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


@dataclass
class Reading:
    p: Optional[float] = None
    sd: Optional[float] = None
    shift: float = 0.0
    shift_logit: float = 0.0
    lo: Optional[float] = None
    hi: Optional[float] = None
    ts: object = None
    expires: object = None
    source: str = ""
    note: str = ""
    outcome: Optional[str] = None       # set by matching; None = first outcome

    @classmethod
    def from_any(cls, r, source: str = "") -> Optional["Reading"]:
        if r is None:
            return None
        if isinstance(r, Reading):
            r.source = r.source or source
            return r
        if isinstance(r, (int, float)):
            return cls(p=float(r), source=source)
        if isinstance(r, dict):
            known = {f for f in cls.__dataclass_fields__}
            d = {k: v for k, v in r.items() if k in known}
            d.setdefault("source", source)
            for k in ("p", "sd", "lo", "hi"):
                if k in d:
                    d[k] = opt_float(d[k])
            for k in ("shift", "shift_logit"):
                d[k] = opt_float(d.get(k)) or 0.0
            return cls(**d)
        raise TypeError(f"feed returned {type(r).__name__}; expected Reading, dict, float or None")

    def mirrored(self) -> "Reading":
        """The same view expressed for the complementary outcome."""
        return Reading(p=None if self.p is None else 1 - self.p, sd=self.sd, shift=-self.shift,
                       shift_logit=-self.shift_logit,
                       lo=None if self.hi is None else 1 - self.hi, hi=None if self.lo is None else 1 - self.lo,
                       ts=self.ts, expires=self.expires, source=self.source, note=self.note)


def key_matches(key: str, ctx: FeedContext) -> tuple[bool, bool]:
    """(matches, mirror). See module doc for the order of checks."""
    key = str(key).strip()
    if not key:
        return False, False
    if key == ctx.token_id:
        return True, False
    if key == ctx.condition_id:
        return True, ctx.outcome_index == 1
    if key.lower().startswith("q:"):
        return key[2:].strip().lower() in (ctx.question or "").lower(), ctx.outcome_index == 1
    slug, outcome = (key.rsplit(":", 1) if ":" in key else (key, None))
    if not fnmatch.fnmatch((ctx.slug or "").lower(), slug.lower()):
        return False, False
    if outcome is None:
        return True, ctx.outcome_index == 1
    return True, str(ctx.outcome).lower() != outcome.lower()


# =============================================================================
# Feeds
# =============================================================================
class Feed:
    name = "feed"

    def read(self, ctx: FeedContext) -> list:
        raise NotImplementedError


class FunctionFeed(Feed):
    def __init__(self, fn: Callable, name: str = ""):
        self.fn, self.name = fn, name or getattr(fn, "__name__", "function")

    def read(self, ctx):
        r = self.fn(ctx)
        rs = r if isinstance(r, list) else [r]
        return [x for x in (Reading.from_any(v, self.name) for v in rs) if x is not None]


def _match_rows(rows: list, ctx: FeedContext, name: str) -> list:
    out = []
    for row in rows:
        key = row.get("match", row.get("key"))
        ok, mirror = key_matches(key, ctx)
        if not ok:
            continue
        r = Reading.from_any(row, name)
        if r is None:
            continue
        r.source = row.get("source") or name
        out.append(r.mirrored() if mirror else r)
    return out


class ViewsFile(Feed):
    def __init__(self, path: str, name: str = "views"):
        self.path, self.name = path, name
        self._mtime, self._rows = None, []

    def _load(self):
        if not os.path.exists(self.path):
            self._rows = []
            return
        m = os.path.getmtime(self.path)
        if m != self._mtime:
            with open(self.path, encoding="utf-8") as f:
                d = json.load(f)
            self._rows = [v for v in d.get("views", []) if not v.get("disabled")]
            self._mtime = m

    def read(self, ctx):
        self._load()
        return _match_rows(self._rows, ctx, self.name)


class TableFeed(Feed):
    """CSV / JSONL / Parquet file or folder; latest row per key wins."""

    def __init__(self, path: str, name: str = ""):
        self.path = path
        self.name = name or os.path.splitext(os.path.basename(path.rstrip("/\\")))[0]
        self._sig, self._rows = None, []

    def _files(self):
        if os.path.isdir(self.path):
            return sorted(os.path.join(self.path, f) for f in os.listdir(self.path)
                          if f.lower().endswith((".csv", ".jsonl", ".json", ".parquet")) and not f.startswith("_"))
        return [self.path] if os.path.exists(self.path) else []

    def _load(self):
        files = self._files()
        sig = tuple((f, os.path.getmtime(f)) for f in files)
        if sig == self._sig:
            return
        frames = []
        for f in files:
            fl = f.lower()
            if fl.endswith(".csv"):
                df = pd.read_csv(f)
            elif fl.endswith((".jsonl", ".json")):
                df = pd.read_json(f, lines=fl.endswith(".jsonl"))
            else:
                df = pd.read_parquet(f)
            df["source"] = df["source"] if "source" in df else os.path.splitext(os.path.basename(f))[0]
            df["_order"] = range(len(df))
            frames.append(df)
        rows = []
        if frames:
            df = pd.concat(frames, ignore_index=True)
            if "key" not in df and "match" in df:
                df["key"] = df["match"]
            if "ts" in df:
                df["_t"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
                df = df.sort_values(["_t", "_order"], na_position="first")
            df = df.drop_duplicates(["key", "source"], keep="last")
            rows = [{k: (None if isinstance(v, float) and math.isnan(v) else v)
                     for k, v in r.items() if not str(k).startswith("_")} for r in df.to_dict("records")]
        self._rows, self._sig = rows, sig

    def read(self, ctx):
        self._load()
        return _match_rows(self._rows, ctx, self.name)


class PluginFeed(Feed):
    """Loads every *.py in a folder that defines read(ctx)."""

    def __init__(self, folder: str):
        self.folder, self.name = folder, "plugins"
        self._mods: dict = {}       # path -> (mtime, module)
        self.errors: list = []

    def _load(self):
        if not os.path.isdir(self.folder):
            return []
        mods = []
        for f in sorted(os.listdir(self.folder)):
            if not f.endswith(".py") or f.startswith("_"):
                continue
            path = os.path.join(self.folder, f)
            m = os.path.getmtime(path)
            hit = self._mods.get(path)
            if hit is None or hit[0] != m:
                try:
                    spec = importlib.util.spec_from_file_location(f"fairmv_feed_{f[:-3]}", path)
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    hit = (m, mod)
                except Exception as e:  # noqa: BLE001
                    self.errors.append(f"plugin {f}: load failed: {type(e).__name__}: {e}")
                    hit = (m, None)
                self._mods[path] = hit
            if hit[1] is not None and callable(getattr(hit[1], "read", None)) and getattr(hit[1], "ENABLED", True):
                mods.append(hit[1])
        return mods

    def read(self, ctx):
        out = []
        for mod in self._load():
            name = getattr(mod, "NAME", mod.__name__.replace("fairmv_feed_", ""))
            try:
                r = mod.read(ctx)
                rs = r if isinstance(r, list) else [r]
                out.extend(x for x in (Reading.from_any(v, name) for v in rs) if x is not None)
            except Exception as e:  # noqa: BLE001
                self.errors.append(f"plugin {name}: {type(e).__name__}: {e}")
        return out


# =============================================================================
# Turning readings into model inputs
# =============================================================================
@dataclass
class FeedResult:
    estimates: list = field(default_factory=list)
    shift_logit: float = 0.0       # total log-odds shift (prob shifts converted at the fused value later)
    shift_prob: float = 0.0
    bounds: list = field(default_factory=list)
    info: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)


def collect(feeds: list, ctx: FeedContext, sigma2: float, cfg: FVConfig) -> FeedResult:
    out = FeedResult()
    for feed in feeds:
        try:
            readings = feed.read(ctx)
        except Exception as e:  # noqa: BLE001
            out.warnings.append(f"feed {feed.name}: {type(e).__name__}: {e}")
            continue
        errs = getattr(feed, "errors", None)
        if errs:
            out.warnings.extend(errs)
            errs.clear()
        for i, r in enumerate(readings):
            exp = _ts(r.expires, ctx.now) if not isinstance(r.expires, str) or r.expires[-1:] not in "hdm" else None
            if exp is not None and exp < ctx.now:
                continue
            tag = r.source or feed.name
            name = f"feed:{tag}" if tag not in out.info else f"feed:{tag}#{i}"
            ts = _ts(r.ts, ctx.now)
            age_h = max((ctx.now - ts).total_seconds() / 3600.0, 0.0) if ts is not None else 0.0
            if r.p is not None and 0 < r.p < 1:
                p = clip_p(r.p)
                sd = r.sd if r.sd and r.sd > 0 else cfg.feed_default_sd
                var = var_logit_from_sd_prob(p, sd) + sigma2 * age_h
                out.estimates.append(Estimate(name, logit(p), var, "feed",
                                              {"age_h": round(age_h, 2), "note": r.note}))
            out.shift_prob += r.shift or 0.0
            out.shift_logit += r.shift_logit or 0.0
            if r.lo is not None or r.hi is not None:
                out.bounds.append(Bound(r.lo if r.lo is not None else 0.0, r.hi if r.hi is not None else 1.0, name))
            out.info[tag] = {k: v for k, v in (("p", r.p), ("sd", r.sd), ("shift", r.shift or None),
                                                ("lo", r.lo), ("hi", r.hi), ("note", r.note or None)) if v is not None}
    return out
