"""Write ``paper/numbers.tex``: one ``\\newcommand`` per number the prose quotes.

    PYTHONPATH=src python paper/numbers.py [--results DIR] [--paper-dir DIR]

Every number in ``sections/*.tex`` is a macro from this file (MASTER_PLAN
section 5); ``check.py`` fails on a hand-typed one. The macros are generated
from ``results/**`` only, in systematic families, so a sentence can quote
any row of a results table without anyone typing a digit:

=================  ==========================================================
family             name pattern (parts are CamelCase; digits become words)
=================  ==========================================================
disk table         ``\\disk[<Corpus>][<Cfg>][<Merge>][BreakEven]<Qty><Policy><Cap>``
                   from table_main.csv / table_by_corpus.csv (per corpus for
                   C1 only, to keep TeX's memory in bounds); Qty is Hit,
                   WeightedHit, VsNone, VsSeq, Precision, Waste, IORatio,
                   MovedRatio, WinsVsNone, LossesVsNone, Traces. Cfg is
                   omitted for C1, Merge for auto, latency for assumed.
                   ``\\diskHitDeepAtEightOneMiB``, ``\\diskFiuVsNoneGuardSixtyFourMiB``
disk deltas        ``\\diskDelta[<Corpus>][<Cfg>]...<Metric><Policy>Vs<Baseline><Cap>``
                   from deltas.csv, with Lo, Hi, CI, Cluster, ClusterCI,
                   Pvalue, Wins, Losses, Ties, N suffixes; a non-default
                   statistic, then the twin treatment, is appended when needed
                   to keep names unique (``delta_names``)
kappa              ``\\diskKappaMedian<Policy><Cap>``, ``...Infinite``, ``...N``,
                   ``\\diskKappaFavoured<Device><Policy><Cap>``; device constants
                   ``\\device<Dev>Kappa``, ``TPos``, ``TXfer``, ``RandFourK``, ...
corpora            ``\\corpusTraces<Corpus>``, ``\\corpusRequests<Corpus>``, ...
learned_depth      ``\\ldVsVcfCap<N>`` (+ CI, Wins, Losses), ``\\ldTracesCap<N>``,
                   ``\\ldSeedsCap<N>``, ``\\ldGapToOracleCap<N>``, ...
realio             ``\\realio<Qty><Dev><Backend><Mode><Think><Trace>``
web                ``\\webHit<Scorer>[<Scope>]Cap<N><Policy>[<Split>]``,
                   ``\\webCmp<Qty><Selection><Scorer>Cap<N>Cost<c>``,
                   ``\\webLever...``, ``\\webCoverage<Index><K>``, ``\\webSeq...``,
                   ``\\webData<Key>`` (dataset_profile.json)
live               ``\\liveSaved<Store><Policy>`` (+ CI), ``\\liveParityMismatches``, ...
=================  ==========================================================

Every estimate ``\\X`` with an interval also has ``\\XLo``, ``\\XHi`` and
``\\XCI`` ("estimate [lo, hi]"). Fractions are printed as percent (levels)
or percentage points (differences); ratios as multipliers.

A short list of **headline** macros (``CURATED``) is always defined, even
when its input is missing, so the prose can use them before the experiments
have run: they then print a bold "??", and the file's header lists what is
missing. The systematic families exist only for rows that exist.

Values are text-mode LaTeX (they hold their own ``$...$``), so use a macro in
running text, never inside math. When the results tree is a fixture
(``FIXTURE`` marker), every value is
wrapped in ``\\fixturemark`` and prints in red with a FIXTURE superscript.
"""

from __future__ import annotations

import os
import sys

# This file is named numbers.py (MASTER_PLAN section 5), which shadows the
# standard library's ``numbers`` module for anything imported while its own
# directory is on sys.path -- and ``statistics``, ``decimal`` and the results
# schemas all import ``numbers``. Run as a script, Python puts this directory
# first on sys.path, so take it off again before importing anything else.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.curdir) != _HERE]

import argparse  # noqa: E402
import importlib.util  # noqa: E402
import math  # noqa: E402
import statistics  # noqa: E402
from collections import defaultdict  # noqa: E402
from pathlib import Path  # noqa: E402


def _genlib():
    """``paper/genlib.py``, loaded by path (never by putting paper/ on sys.path)."""
    if "paper_genlib" not in sys.modules:
        spec = importlib.util.spec_from_file_location("paper_genlib",
                                                      Path(_HERE) / "genlib.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules["paper_genlib"] = module
        spec.loader.exec_module(module)
    return sys.modules["paper_genlib"]


g = _genlib()
camel, disk_capacity_word, fci, fcount = g.camel, g.disk_capacity_word, g.fci, g.fcount
fnum, fpct, fpp, fratio = g.fnum, g.fpct, g.fpp, g.fratio

from prefetch_common import stats, utility  # noqa: E402

OUTPUT = "numbers.tex"
MANIFEST = "numbers.manifest.json"
MISSING = r"\missingnumber"

#: Suffixes of a curated estimate: the interval, and the paired counts.
CI_SET = ("", "Lo", "Hi", "CI")
PAIRED_SET = CI_SET + ("Wins", "Losses", "N")
VALUE_SET = ("",)

#: Headline macros that always exist, with every suffix in their set:
#: (alias, source file, suffixes, description). The family code below fills
#: them via ``Macros.alias``; whatever is still undefined afterwards prints
#: ``\missingnumber``, so the prose can use these names before any result.
CURATED = (
    ("diskAdaptiveVsSeq", "disk/deltas.csv", PAIRED_SET,
     "adaptive - sequential, hit ratio, C1, 1 MiB"),
    ("diskGuardVsSeq", "disk/deltas.csv", PAIRED_SET, "guard - sequential, hit ratio, C1, 1 MiB"),
    ("diskDeepEightVsNone", "disk/deltas.csv", PAIRED_SET, "deep@8 - none, hit ratio, C1, 1 MiB"),
    ("diskLinuxRaVsNone", "disk/deltas.csv", PAIRED_SET,
     "linux_ra@256 - none, hit ratio, C1, 1 MiB"),
    ("diskLinuxRaIORatio", "disk/deltas.csv", PAIRED_SET,
     "linux_ra@256 / none, device I/Os, C1, 1 MiB"),
    ("diskDeepEightIORatio", "disk/deltas.csv", PAIRED_SET,
     "deep@8 / none, device I/Os, C1, 1 MiB"),
    ("diskTraces", "disk/corpus_profile.csv", VALUE_SET, "traces in the disk collection"),
    ("diskCorpora", "disk/corpus_profile.csv", VALUE_SET, "corpora in the disk collection"),
    ("ldVsVcf", "gpu/disk_learned_depth.csv", PAIRED_SET,
     "learned_depth - VCF over traces (seed means), largest action cap"),
    ("webControllerVsBestFixed", "web/comparisons.csv", CI_SET,
     "per-capacity adaptive - best fixed, test, capacity 8, smallest positive cost"),
    ("webFreshnessLever", "web/decomposition.csv", CI_SET, "freshness lever estimate"),
    ("liveParityMismatches", "live/live_parity.csv", VALUE_SET,
     "mismatched sessions, all passes"),
    ("liveInvalidPasses", "live/live_metrics.csv", VALUE_SET, "rows failing the validity gate"),
)
#: The pairs behind the curated disk aliases: alias -> (metric, policy, baseline, capacity).
CURATED_DISK = {
    "diskAdaptiveVsSeq": ("hit_ratio", "adaptive", "sequential", 2048),
    "diskGuardVsSeq": ("hit_ratio", "guard", "sequential", 2048),
    "diskDeepEightVsNone": ("hit_ratio", "deep@8", "none", 2048),
    "diskLinuxRaVsNone": ("hit_ratio", "linux_ra@256", "none", 2048),
    "diskLinuxRaIORatio": ("device_ios", "linux_ra@256", "none", 2048),
    "diskDeepEightIORatio": ("device_ios", "deep@8", "none", 2048),
}


@g.traced_formatter
def fus(value, digits: int = 1) -> str:
    """Microseconds."""
    return fnum(value, digits) + (r"\,$\mu$s" if g.finite(value) else "")


@g.traced_formatter
def fms(value, digits: int = 2) -> str:
    return fnum(value, digits) + (r"\,ms" if g.finite(value) else "")


@g.traced_formatter
def fsignedpct(fraction, digits: int = 1) -> str:
    """A signed relative change as a percentage: -0.2 -> ``$-20.0$\\,\\%``."""
    if not g.finite(fraction):
        return fnum(fraction)
    return fnum(100.0 * fraction, digits, sign=True) + r"\,\%"


@g.traced_formatter
def fpvalue(p) -> str:
    if p is None:
        return g.DASH
    if p < 1e-4:
        return r"$<10^{-4}$"
    return f"${p:.2g}$" if p < 0.01 else f"${p:.3f}$"


#: Sources that are code constants, not results files (the cost model's own
#: parameters). Their macros are ledger rows of kind ``constant``.
CODE_SOURCES = ("prefetch_common.utility", "prefetch_common.stats")


def _jsonable(value):
    if callable(value):
        return {"fn": getattr(value, "__name__", repr(value))}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return value


class UntracedMacro(ValueError):
    """A macro whose value cannot be traced to a results file."""


class Macros:
    """An ordered set of macro definitions, each tagged with its source.

    Every macro also gets a **trace** (``self.traces[name]``) for the claims
    ledger (``paper/claims.py``): the results file, row and column(s) its raw
    value was read from, the raw value(s) and the formatter that printed
    them. The trace is collected automatically while tracing is on (``build``
    switches it on): read rows from ``results.rows``/``json_object``/
    ``provenance`` (they are :class:`genlib.TracedRow`), print through a
    genlib formatter (or one wrapped in ``genlib.traced_formatter``), and pass
    the results-relative file as ``source``. A macro that prints no traced
    value, or names a source that was never read from the results tree,
    raises :class:`UntracedMacro`; pass ``raw=[...]`` to record a value that
    no traced formatter printed.
    """

    def __init__(self, fixture: bool):
        self.fixture = fixture
        self.values: dict[str, str] = {}
        self.sources: dict[str, str] = {}
        self.traces: dict[str, dict] = {}
        self._last_window = (set(), set(), set())

    def add(self, name: str, text: str, source: str, *, raw=None, meta: dict | None = None,
            _copy: dict | None = None) -> None:
        taken = g.TRACE.take() if g.TRACE.active else None
        if not g.is_macro_name(name):
            raise ValueError(f"not a TeX control word: {name!r}")
        if name in self.values:
            if self.values[name] == text:
                return
            raise ValueError(f"macro \\{name} defined twice ({self.sources[name]}, {source})")
        if taken is not None:
            if taken[2]:
                self._last_window = taken[2:]
            elif _copy is None:
                # Values computed once and printed by several macros (an
                # estimate, then its Lo, Hi, CI, N): the rows read are those
                # of the window before.
                taken = taken[:2] + self._last_window
            self.traces[name] = (dict(_copy) if _copy is not None
                                 else self._trace(name, text, source, taken, raw, meta))
        self.values[name] = text
        self.sources[name] = source

    @staticmethod
    def _trace(name, text, source, taken, raw, meta) -> dict:
        calls, recent, rows, columns, files = taken
        record = {"source": source, "calls": [], "raw": [], "where": [], "meta": meta or {}}
        for function, raws, other, kwargs, printed, where in calls:
            record["calls"].append([function, _jsonable(list(other)), _jsonable(kwargs),
                                    printed])
            record["raw"] += list(raws)
            record["where"] += [w or g.TRACE.locate(v, recent) for v, w in zip(raws, where)]
        if raw is not None:
            record["raw"] = list(raw)
            record["where"] = [g.TRACE.locate(v, recent) for v in record["raw"]]
        if text == MISSING:
            record["kind"] = "missing"
            return record
        if source in CODE_SOURCES:
            record["kind"] = "constant"
            return record
        if not record["raw"]:
            raise UntracedMacro(
                f"\\{name}: no traced value. Print it with a genlib formatter (or one "
                "wrapped in genlib.traced_formatter), or pass raw=[...] (paper/README.md, "
                "'The claims ledger')")
        if source != DECISION_SOURCE and source not in g.TRACE.loaded:
            raise UntracedMacro(
                f"\\{name}: source {source!r} is not a results file this build read; "
                "pass the results-relative path the value came from")
        located = record["where"]
        if all(located) and len({(w[0], w[1]) for w in located}) == 1:
            record["kind"] = "value"
            record["file"], record["row"] = located[0][0], located[0][1]
            record["columns"] = [g.trace_column(w) for w in located]
        else:
            used = {w[0] for w in located if w} | set(files)
            if not used and source != DECISION_SOURCE:
                used = {source}
            record["kind"] = "derived"
            record["files"] = sorted(used)
            record["rows"] = len({r for r in rows if r[0] in used})
            record["columns"] = sorted({c for f, c in columns if f in used})
        if source == DECISION_SOURCE:
            record["kind"] = "decision" if meta and meta.get("rule") else "decision_count"
            if record["kind"] == "decision_count" and meta and meta.get("files"):
                record["files"] = list(meta["files"])
                record.pop("row", None)
                record["rows"] = None
        return record

    def add_ci(self, name: str, estimate, lo, hi, formatter, source: str, *,
               meta: dict | None = None, **kwargs) -> None:
        self.add(name, formatter(estimate, **kwargs), source, meta=meta)
        if lo is None or hi is None:
            return
        self.add(name + "Lo", formatter(lo, **kwargs), source, meta=meta)
        self.add(name + "Hi", formatter(hi, **kwargs), source, meta=meta)
        self.add(name + "CI", formatter(estimate, **kwargs) + "~"
                 + fci(lo, hi, formatter, **kwargs), source, meta=meta)

    def alias(self, alias: str, target: str) -> None:
        """Copy ``target`` (and its interval suffixes) to ``alias``."""
        for suffix in PAIRED_SET + ("Ties",):
            if target + suffix in self.values:
                copy = self.traces.get(target + suffix)
                if copy is not None:
                    copy = dict(copy, alias_of=target + suffix)
                self.add(alias + suffix, self.values[target + suffix],
                         self.sources[target + suffix], _copy=copy)

    def render(self, missing: list[str], results_root: str) -> str:
        lines = [
            "% numbers.tex -- GENERATED by paper/numbers.py from " + results_root + ".",
            "% Do not edit: rerun `python reproduce.py paper`. Every number the prose",
            "% quotes is one of these macros (MASTER_PLAN section 5; paper/check.py).",
        ]
        if self.fixture:
            lines.append("% FIXTURE DATA: built from synthetic test fixtures, not results.")
        for item in sorted(set(missing)):
            lines.append(f"% MISSING input: {item}")
        lines += [
            r"\providecommand{\missingnumber}{\textbf{??}}",
            r"\providecommand{\fixturemark}[1]{\textcolor{red}{#1\textsuperscript{FIXTURE}}}",
            r"\providecommand{\paperFixtureData}{" + ("true" if self.fixture else "false") + "}",
        ]
        last_source = None
        for name in sorted(self.values, key=lambda n: (self.sources[n], n)):
            if self.sources[name] != last_source:
                last_source = self.sources[name]
                lines.append(f"% --- {last_source}")
            text = self.values[name]
            if self.fixture and text != MISSING:
                text = r"\fixturemark{" + text + "}"
            lines.append(f"\\newcommand{{\\{name}}}{{{text}}}")
        return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Disk
# ---------------------------------------------------------------------------

def _variant(row) -> str:
    """The configuration part of a disk macro name; empty for the headline."""
    parts = []
    if row["config"] != g.HEADLINE_CONFIG:
        parts.append(camel(row["config"]))
    if row["merge"] != g.HEADLINE_MERGE:
        parts.append(camel(row["merge"]))
    if row["latency"] != g.HEADLINE_LATENCY:
        parts.append(camel(row["latency"]))
    return "".join(parts)


def disk_tables(results: g.Results, out: g.Output, m: Macros) -> None:
    for relative in ("disk/table_main.csv", "disk/table_by_corpus.csv"):
        rows = results.rows(relative, out)
        for row in rows or ():
            if row["corpus"] != "all" and row["config"] != g.HEADLINE_CONFIG:
                continue  # per-corpus macros for the headline table only (TeX memory)
            corpus = "" if row["corpus"] == "all" else camel(row["corpus"])
            pre = "disk" + corpus + _variant(row)
            tail = camel(row["policy"]) + disk_capacity_word(row["capacity"])
            m.add(pre + "Traces" + tail, fcount(row["n_traces"]), relative)
            if row["mean_hit_ratio"] is None:
                continue
            m.add(pre + "Hit" + tail, fpct(row["mean_hit_ratio"]), relative)
            if row["config"] != g.HEADLINE_CONFIG:
                # Sensitivity configurations (C2-C4): the level, the gain and
                # the device work only, which is all the prose quotes of them.
                m.add_ci(pre + "VsNone" + tail, row["delta_vs_none"], row["delta_vs_none_lo"],
                         row["delta_vs_none_hi"], fpp, relative)
                for quantity, key in (("IORatio", "device_io_ratio"),
                                      ("MovedRatio", "blocks_moved_ratio")):
                    if row[key] is not None:
                        m.add(pre + quantity + tail, fratio(row[key]), relative)
                continue
            if corpus:
                # Per corpus: the level and the two paired gains, which is what
                # the prose and Table T2b need; the file holds the rest.
                m.add_ci(pre + "VsNone" + tail, row["delta_vs_none"], row["delta_vs_none_lo"],
                         row["delta_vs_none_hi"], fpp, relative)
                if row["delta_vs_sequential"] is not None:
                    m.add_ci(pre + "VsSeq" + tail, row["delta_vs_sequential"],
                             row["delta_vs_sequential_lo"], row["delta_vs_sequential_hi"], fpp,
                             relative)
                if row["device_io_ratio"] is not None:
                    m.add(pre + "IORatio" + tail, fratio(row["device_io_ratio"]), relative)
                continue
            m.add(pre + "WeightedHit" + tail, fpct(row["weighted_hit_ratio"]), relative)
            m.add_ci(pre + "VsNone" + tail, row["delta_vs_none"], row["delta_vs_none_lo"],
                     row["delta_vs_none_hi"], fpp, relative)
            if row["delta_vs_sequential"] is not None:
                m.add_ci(pre + "VsSeq" + tail, row["delta_vs_sequential"],
                         row["delta_vs_sequential_lo"], row["delta_vs_sequential_hi"], fpp,
                         relative)
            for quantity, key, formatter in (("Precision", "mean_precision", fpct),
                                             ("Waste", "mean_wasted_share", fpct),
                                             ("IORatio", "device_io_ratio", fratio),
                                             ("MovedRatio", "blocks_moved_ratio", fratio)):
                if row[key] is not None:
                    m.add(pre + quantity + tail, formatter(row[key]), relative)
            for quantity, key in (("WinsVsNone", "wins_vs_none"),
                                  ("LossesVsNone", "losses_vs_none")):
                if row[key] is not None:
                    m.add(pre + quantity + tail, fcount(row[key]), relative)


def disk_best_depth(results: g.Results, out: g.Output, m: Macros) -> None:
    """C2's best fixed depth per (corpus, capacity): the argmax of the pooled
    mean hit ratio over depths (disk PREREGISTERED.md section 6, descriptive),
    and how often it is the deepest depth tested."""
    relative = "disk/table_by_corpus.csv"
    rows = [r for r in results.rows(relative, out, required=False) or ()
            if r["config"] == "C2" and r["merge"] == g.HEADLINE_MERGE
            and r["latency"] == g.HEADLINE_LATENCY and r["mean_hit_ratio"] is not None]
    curves = defaultdict(dict)
    for row in rows:
        base, _, depth = row["policy"].partition("@")
        if base == "deep" and depth.isdigit():
            curves[(row["corpus"], row["capacity"])][int(depth)] = row["mean_hit_ratio"]
    if not curves:
        return
    deepest = max(max(c) for c in curves.values())
    m.add("diskCTwoDeepest", fcount(deepest), relative)
    best_is_deepest = 0
    for (corpus, capacity), curve in curves.items():
        best = max(curve, key=lambda d: (curve[d], -d))
        best_is_deepest += best == deepest
        m.add("diskCTwoBestDepth" + camel(corpus) + disk_capacity_word(capacity), fcount(best),
              relative)
    m.add("diskCTwoCurves", fcount(len(curves)), relative)
    m.add("diskCTwoBestIsDeepest", fcount(best_is_deepest), relative)


def delta_name(row) -> str:
    corpus = "" if row["corpus"] == "all" else camel(row["corpus"])
    return ("diskDelta" + corpus + _variant(row) + camel(row["metric"]) + camel(row["policy"])
            + "Vs" + camel(row["baseline"]) + disk_capacity_word(row["capacity"]))


def delta_formatter(row):
    return fratio if row["statistic"] == "ratio" else fpp


#: The statistic each delta metric is read as by default; a row with the
#: other statistic gets it appended to its name.
DEFAULT_STATISTIC = {"hit_ratio": "difference", "precision": "difference",
                     "wasted_share": "difference", "device_ios": "ratio",
                     "blocks_moved": "ratio"}


def delta_names(rows) -> list[str]:
    """One macro stem per deltas.csv row. A row whose statistic is not its
    metric's default carries it in the name; rows that still share a stem
    (the twin treatments of one comparison) carry their ``twins`` value."""
    names = []
    for row in rows:
        name = delta_name(row)
        if row["statistic"] != DEFAULT_STATISTIC.get(row["metric"], row["statistic"]):
            name += camel(row["statistic"])
        names.append(name)
    counts = defaultdict(int)
    for name in names:
        counts[name] += 1
    names = [n + camel(r["twins"]) if counts[n] > 1 else n for n, r in zip(names, rows)]
    if len(set(names)) != len(names):
        raise ValueError("disk/deltas.csv: rows repeat the same comparison")
    return names


#: deltas.csv rows outside the pooled headline (C1, all corpora) get macros
#: only for these policies: the pre-registered levers and the P3/P4 depths.
#: deltas.csv holds about 3,850 rows at 13 macros each, and numbers.tex has
#: to load within TeX's memory on Overleaf; the pooled C1 rows keep every
#: policy, and the generated tables print every row.
DELTA_DETAIL_POLICIES = ("sequential", "adaptive", "guard", "deep@8", "deep@32", "deep@256",
                         "linux_ra@64", "linux_ra@256", "linux_ra@1024", "belady_min")
#: Sensitivity configurations whose pooled rows get macros (C3, C4; C2 only
#: for its merge-rule views of the two headline depth/window policies).
DELTA_DETAIL_CONFIGS = ("C3:skip=1000000", "C3:whole", "C4", "C4:mms=256")


#: Metrics of the detail rows: hit ratio (both baselines) and device I/Os.
DELTA_DETAIL_METRICS = ("hit_ratio", "device_ios")


def delta_wanted(row) -> bool:
    """Whether a deltas.csv row gets macros (see DELTA_DETAIL_POLICIES)."""
    if row["config"] == g.HEADLINE_CONFIG and row["corpus"] == "all":
        return True
    if row["policy"] not in DELTA_DETAIL_POLICIES or row["metric"] not in DELTA_DETAIL_METRICS:
        return False
    if row["config"] == g.HEADLINE_CONFIG:
        return True
    if row["config"] in DELTA_DETAIL_CONFIGS:
        return row["corpus"] == "all"
    return (row["config"].split(":")[0] == "C2" and row["metric"] == "hit_ratio"
            and row["policy"] in ("deep@8", "linux_ra@256"))


def disk_deltas(results: g.Results, out: g.Output, m: Macros) -> None:
    relative = "disk/deltas.csv"
    rows = results.rows(relative, out) or []
    names = delta_names(rows)
    for name, row in zip(names, rows):
        if not delta_wanted(row):
            continue
        pooled = row["config"] == g.HEADLINE_CONFIG and row["corpus"] == "all"
        formatter = delta_formatter(row)
        m.add_ci(name, row["estimate"], row["ci_lo"], row["ci_hi"], formatter, relative)
        if row["cluster_estimate"] is not None:
            m.add_ci(name + "Cluster", row["cluster_estimate"], row["cluster_ci_lo"],
                     row["cluster_ci_hi"], formatter, relative)
        if pooled:
            m.add(name + "Pvalue", fpvalue(row["wilcoxon_p"]), relative)
        for suffix, key in (("Wins", "wins"), ("Losses", "losses"), ("Ties", "ties"),
                            ("N", "n")):
            if pooled or suffix != "Ties":
                m.add(name + suffix, fcount(row[key]), relative)
    for alias, (metric, policy, baseline, capacity) in CURATED_DISK.items():
        match = [name for name, row in zip(names, rows)
                 if (row["config"], row["merge"], row["latency"], row["corpus"], row["metric"],
                     row["statistic"], row["policy"], row["baseline"], row["capacity"])
                 == (g.HEADLINE_CONFIG, g.HEADLINE_MERGE, g.HEADLINE_LATENCY, "all", metric,
                     DEFAULT_STATISTIC[metric], policy, baseline, capacity)]
        if match:
            m.alias(alias, match[0])


def disk_corpora(results: g.Results, out: g.Output, m: Macros) -> None:
    relative = "disk/corpus_profile.csv"
    rows = results.rows(relative, out)
    if not rows:
        return
    by_corpus = defaultdict(list)
    for row in rows:
        by_corpus[row["corpus"]].append(row)
    m.add("diskTraces", fcount(len(rows)), relative)
    m.add("diskCorpora", fcount(len(by_corpus)), relative)
    m.add("corpusTracesAll", fcount(len(rows)), relative)
    m.add("corpusRequestsAll", fcount(sum(r["requests"] for r in rows)), relative)
    m.add("corpusRecordsAll", fcount(sum(r["records"] for r in rows)), relative)
    m.add("corpusPaperSubset", fcount(sum(r["paper_subset"] for r in rows)), relative)
    m.add("corpusTwinTraces", fcount(sum(r["twin_group"] is not None for r in rows)), relative)
    for corpus, items in by_corpus.items():
        word = camel(corpus)
        m.add("corpusTraces" + word, fcount(len(items)), relative)
        m.add("corpusRequests" + word, fcount(sum(r["requests"] for r in items)), relative)
        m.add("corpusRecords" + word, fcount(sum(r["records"] for r in items)), relative)
        m.add("corpusHoursMedian" + word,
              fnum(statistics.median(r["hours_covered"] for r in items), 1), relative)
        m.add("corpusMeanRequestBlocks" + word,
              fnum(statistics.fmean(r["mean_request_blocks"] for r in items), 1), relative)


def devices(results: g.Results, out: g.Output, m: Macros, *, required=True) -> dict[str, float]:
    """Device constants; returns ``{device: kappa}``."""
    found = results.glob("realio/device_*.json", out, required=required)
    kappas = {}
    for relative, row in found.items():
        dev = camel(row["device"])
        kappas[row["device"]] = row["kappa"]
        m.add(f"device{dev}Kappa", fnum(row["kappa"], 0), relative)
        m.add(f"device{dev}TPos", fus(row["t_pos_us"]), relative)
        m.add(f"device{dev}TXfer", fus(row["t_xfer_us_per_block"], 2), relative)
        m.add(f"device{dev}RandFourK", fus(row["rand4k_p50_us"]), relative)
        m.add(f"device{dev}SeqFourK", fus(row["seq4k_p50_us"]), relative)
        m.add(f"device{dev}SeqSixtyFourK", fus(row["seq64k_p50_us"]), relative)
        m.add(f"device{dev}SeqMBps", fnum(row["seq1m_MBps"], 1) + r"\,MB/s", relative)
        if row["model"] is not None:
            m.add(f"device{dev}Model", g.esc(row["model"]), relative)
    return kappas


def disk_kappa(results: g.Results, out: g.Output, m: Macros) -> None:
    relative = "disk/kappa.csv"
    rows = results.rows(relative, out)
    kappas = devices(results, out, m, required=False)
    groups = defaultdict(list)
    for row in rows or ():
        groups[(row["config"], row["merge"], row["policy"], row["capacity"])].append(row)
    for (config, merge, policy, capacity), items in groups.items():
        pre = "diskKappa" + _variant({"config": config, "merge": merge,
                                      "latency": g.HEADLINE_LATENCY})
        tail = camel(policy) + disk_capacity_word(capacity)
        values = [r["kappa_star"] for r in items if r["kappa_star"] is not None]
        m.add(pre + "N" + tail, fcount(len(values)), relative)
        m.add(pre + "Infinite" + tail, fcount(sum(math.isinf(v) for v in values)), relative)
        if values:
            m.add(pre + "Median" + tail, fnum(statistics.median(values), 1), relative)
        for device, kappa in kappas.items():
            m.add(pre + "Favoured" + camel(device) + tail,
                  fcount(sum(v < kappa for v in values)), relative)


def learned_depth(results: g.Results, out: g.Output, m: Macros) -> None:
    relative = "gpu/disk_learned_depth.csv"
    rows = results.rows(relative, out)
    if not rows:
        return
    m.add("ldSkipped", fcount(sum(r["status"] != "ok" for r in rows)), relative)
    caps = sorted({r["action_cap"] for r in rows})
    for cap in caps:
        per_trace = defaultdict(list)
        for row in g.select(rows, action_cap=cap, status="ok"):
            per_trace[row["trace"]].append(row)
        if not per_trace:
            continue
        traces = sorted(per_trace)

        def mean(key, trace):
            values = [r[key] for r in per_trace[trace] if r[key] is not None]
            return statistics.fmean(values) if values else None

        word = "Cap" + camel(cap)
        model = [mean("test_hit_ratio", t) for t in traces]
        seeds = sorted({r["seed"] for t in traces for r in per_trace[t]})
        m.add("ldTraces" + word, fcount(len(traces)), relative)
        m.add("ldSeeds" + word, fcount(len(seeds)), relative)
        for quantity, key in (("VsVcf", "vcf_hit_ratio"), ("VsDeepEight", "deep8_hit_ratio"),
                              ("VsNone", "none_hit_ratio"), ("VsRelSize", "relsize_hit_ratio")):
            pairs = [(a, mean(key, t)) for a, t in zip(model, traces)
                     if a is not None and mean(key, t) is not None]
            if not pairs:
                continue
            a, b = zip(*pairs)
            ci = stats.paired_bootstrap(a, b, reps=10000, seed=42)
            m.add_ci("ld" + quantity + word, ci.estimate, ci.low, ci.high, fpp, relative)
            won, lost, tied = stats.wins(a, b)
            m.add("ld" + quantity + word + "Wins", fcount(won), relative)
            m.add("ld" + quantity + word + "Losses", fcount(lost), relative)
            m.add("ld" + quantity + word + "N", fcount(len(pairs)), relative)
        # E2 records prefetches issued, not device I/Os: the device-work side
        # of the comparison is the ratio of issued prefetches (mean over traces).
        issued = [(mean("test_issued", t), mean("vcf_issued", t)) for t in traces]
        issued = [a / b for a, b in issued if a is not None and b]
        if issued:
            m.add("ldIssuedVsVcf" + word, fratio(statistics.fmean(issued), 2), relative)
        oracle = [(mean("oracle_same_actions_hit_ratio", t), a) for a, t in zip(model, traces)]
        oracle = [(o, a) for o, a in oracle if o is not None and a is not None]
        if oracle:
            o, a = zip(*oracle)
            m.add("ldGapToOracle" + word, fpp(statistics.fmean(o) - statistics.fmean(a)),
                  relative)
        if cap == caps[-1]:
            m.alias("ldVsVcf", "ldVsVcf" + word)


# ---------------------------------------------------------------------------
# Real-device testbed
# ---------------------------------------------------------------------------

def realio(results: g.Results, out: g.Output, m: Macros) -> None:
    relative = "realio/realio_summary.csv"
    rows = results.rows(relative, out)
    devices(results, out, m, required=False)
    if not rows:
        return
    none_lat = {(r["device"], r["backend"], r["think"], r["trace"]): r["lat_mean_us"]
                for r in rows if r["mode"] == "none"}
    for row in rows:
        group = camel(row["device"]) + camel(row["backend"])
        tail = group + camel(row["mode"]) + camel(row["think"]) + camel(row["trace"])
        m.add_ci("realioLat" + tail, row["lat_mean_us"], row["lat_mean_us_lo"],
                 row["lat_mean_us_hi"], fus, relative)
        m.add("realioLatPFifty" + tail, fus(row["lat_p50_us"]), relative)
        m.add("realioLatPNinetyNine" + tail, fus(row["lat_p99_us"]), relative)
        m.add("realioHit" + tail, fpct(row["real_hit_ratio"]), relative)
        m.add("realioFullHit" + tail, fpct(row["full_hit_request_frac"]), relative)
        m.add("realioLateHits" + tail, fnum(row["late_hits"], 0), relative)
        m.add("realioReps" + tail, fcount(row["reps"]), relative)
        # pred_err is a relative error, (pred - measured) / measured, so it is
        # a signed percent, not percentage points.
        for quantity, key, formatter in (("SimHit", "sim_hit_ratio", fpct),
                                         ("PredErr", "pred_err", fsignedpct),
                                         ("KEff", "k_eff", lambda v: fnum(v, 2))):
            if row[key] is not None:
                m.add("realio" + quantity + tail, formatter(row[key]), relative)
        base = none_lat.get((row["device"], row["backend"], row["think"], row["trace"]))
        if base and row["mode"] != "none":
            m.add("realioLatVsNone" + tail, fratio(row["lat_mean_us"] / base), relative)
        scope = group + camel(row["think"]) + camel(row["trace"])
        for quantity, key in (("SpearmanDeviceTime", "spearman_device_time"),
                              ("SpearmanHitRatio", "spearman_hit_ratio")):
            if row[key] is not None:
                m.add("realio" + quantity + scope, fnum(row[key], 2, sign=True), relative)


# ---------------------------------------------------------------------------
# Web
# ---------------------------------------------------------------------------

def _split_word(split: str) -> str:
    return "" if split == "test" else camel(split)


#: policy_metrics.csv rows that get macros: the test split at the capacities
#: of the headline comparisons (W1-W8), for the rankers the prose discusses.
#: The file has about 2,150 rows; the full set is in the generated tables.
WEB_METRIC_SCORERS = ("logistic", "trans_online")
WEB_METRIC_CAPACITIES = (8, 32)


def web_policy_metrics(results: g.Results, out: g.Output, m: Macros) -> None:
    relative = "web/policy_metrics.csv"
    for row in results.rows(relative, out) or ():
        if (row["split"] != "test" or row["scorer"] not in WEB_METRIC_SCORERS
                or row["capacity"] not in WEB_METRIC_CAPACITIES):
            continue
        scope = "" if row["cache_scope"] == "session" else camel(row["cache_scope"])
        tail = (camel(row["scorer"]) + scope + "Cap" + camel(row["capacity"])
                + camel(row["policy"]) + _split_word(row["split"]))
        m.add("webHit" + tail, fpct(row["hit_ratio"]), relative)
        m.add("webIssued" + tail, fnum(row["issued_per_view"], 2), relative)
        m.add("webPrecision" + tail, fpct(row["precision"]), relative)
        m.add("webWasted" + tail, fnum(row["wasted_per_view"], 2), relative)
        if row["c_star"] is not None:
            m.add("webCStar" + tail, fnum(row["c_star"], 4), relative)


#: Controller families of ``webprefetch.benchmark`` (longest prefix first), so
#: a comparisons row can be named by its family without importing pandas.
WEB_CONTROLLERS = ("adaptive_ca", "adaptive", "gate", "feedback")


def web_family(policy: str) -> str:
    """'adaptive_ca_0.01' -> 'adaptive_ca'; 'feedback_0.01_eps0.005' -> 'feedback'."""
    for family in WEB_CONTROLLERS:
        if str(policy).startswith(family + "_"):
            return family
    return str(policy).split("_", 1)[0]


def _comparison_tail(row) -> str:
    """One comparisons.csv row's macro tail. comparisons.csv holds one row per
    controller family at each (selection, scorer, capacity, cost), so the
    family is part of the name; the plain adaptive family is left implicit."""
    family = web_family(row["adaptive_policy"])
    return (camel(row["selection"]) + camel(row["scorer"])
            + ("" if family == "adaptive" else camel(family))
            + "Cap" + camel(row["capacity"]) + "Cost" + camel(repr(float(row["cost"])))
            + _split_word(row["split"]))


def web_comparisons(results: g.Results, out: g.Output, m: Macros) -> None:
    relative = "web/comparisons.csv"
    rows = results.rows(relative, out)
    for row in rows or ():
        if row["selection"] != "per_capacity" or row["scorer"] not in WEB_METRIC_SCORERS:
            continue  # capacity_transfer is never a headline row (TeX memory)
        tail = _comparison_tail(row)
        for quantity, key in (("VsBestFixed", "vs_best_fixed_same_cap"),
                              ("VsFixedZero", "vs_fixed0"),
                              ("VsTransition", "vs_transition_top1")):
            if row[key] is not None:
                m.add_ci("webCmp" + quantity + tail, row[key], row[key + "_lo"],
                         row[key + "_hi"], fpp, relative)
        m.add("webCmpAdaptivePolicy" + tail, g.esc(row["adaptive_policy"]), relative)
        m.add("webCmpBestFixedPolicy" + tail, g.esc(row["best_fixed_policy"]), relative)
        m.add("webCmpLabel" + tail, g.esc(row["label"]), relative)
    headline = [r for r in g.select(rows, split="test", selection="per_capacity", capacity=8)
                if r["cost"] > 0 and web_family(r["adaptive_policy"]) == "adaptive"]
    if headline:
        row = min(headline, key=lambda r: (r["scorer"] != "logistic", r["cost"]))
        m.alias("webControllerVsBestFixed", "webCmpVsBestFixed" + _comparison_tail(row))


def unique_names(rows, field_lists, prefix: str) -> list[str]:
    """The shortest field list (from ``field_lists``) that names every row uniquely."""
    for fields in field_lists:
        names = [prefix + "".join(camel(r[f]) if r[f] is not None else "Inf" for f in fields)
                 for r in rows]
        if len(set(names)) == len(names):
            return names
    raise ValueError(f"{prefix}: rows are not distinguishable by {field_lists[-1]}")


def web_decomposition(results: g.Results, out: g.Output, m: Macros) -> None:
    relative = "web/decomposition.csv"
    rows = results.rows(relative, out)
    if not rows:
        return
    names = unique_names(rows, (
        ("lever",), ("lever", "capacity"), ("lever", "policy", "baseline", "capacity"),
        ("lever", "policy", "baseline", "scorer", "baseline_scorer", "capacity", "cost"),
        ("lever", "policy", "baseline", "scorer", "baseline_scorer", "capacity", "cost",
         "cache_scope", "metric", "split"),
    ), "webLever")
    for name, row in zip(names, rows):
        m.add_ci(name, row["estimate"], row["ci_lo"], row["ci_hi"], fpp, relative)
        m.add(name + "Sessions", fcount(row["sessions"]), relative)
    freshness = [n for n, r in zip(names, rows) if r["lever"] == "freshness"]
    if freshness:
        m.alias("webFreshnessLever", freshness[0])


def web_freshness(results: g.Results, out: g.Output, m: Macros) -> None:
    relative = "web/freshness.csv"
    for row in results.rows(relative, out) or ():
        k = "All" if row["k"] is None else camel(row["k"])
        m.add("webCoverage" + camel(row["index"]) + "K" + k + _split_word(row["split"]),
              fpct(row["coverage"]), relative)


def web_seq(results: g.Results, out: g.Output, m: Macros) -> None:
    relative = "web/seq_metrics.csv"
    rows = results.rows(relative, out)
    groups = defaultdict(list)
    for row in rows or ():
        if row["status"] == "ok":
            groups[(row["model"], row["protocol"], row["capacity"], row["policy"],
                    row["cost"])].append(row)
    multi = len({k[2:] for k in groups}) > 1
    # The prose compares the final protocol's depth-1 and depth-4 rows at the
    # W8 capacity; the full grid (about 1,800 rows) is in the results file.
    groups = {k: v for k, v in groups.items()
              if not multi or (k[1] == "final" and k[2] == 8 and k[3] in ("fixed1", "fixed4")
                               and k[4] == 0.01)}
    for (model, protocol, capacity, policy, cost), items in groups.items():
        pre = "webSeq" + camel(model) + camel(protocol)
        if multi:
            pre += "Cap" + camel(capacity) + camel(policy) + "Cost" + camel(repr(float(cost)))
        m.add(pre + "Seeds", fcount(len({r["seed"] for r in items})), relative)
        for quantity, key, formatter in (("HitAtOne", "hit_at_1", fpct),
                                         ("NovelRecallEight", "novel_recall_at_8", fpct),
                                         ("NovelHitAtOne", "novel_hit_at_1", fpct),
                                         ("CacheHit", "hit_ratio", fpct),
                                         ("TrainSeconds", "train_s", lambda v: fnum(v, 0)),
                                         ("InferUs", "infer_us_per_request", fus)):
            values = [r[key] for r in items if r[key] is not None]
            if values:
                m.add(pre + quantity, formatter(statistics.fmean(values)), relative)
                m.add(pre + quantity + "Min", formatter(min(values)), relative)
                m.add(pre + quantity + "Max", formatter(max(values)), relative)


def web_calibration(results: g.Results, out: g.Output, m: Macros) -> None:
    """Brier skill and expected calibration error per (split, scorer): one
    value per pair, repeated on every reliability bin of calibration.csv."""
    relative = "web/calibration.csv"
    seen = set()
    for row in results.rows(relative, out, required=False) or ():
        key = (row["split"], row["scorer"])
        if key in seen:
            continue
        seen.add(key)
        tail = camel(row["scorer"]) + _split_word(row["split"])
        if row["brier_skill_score"] is not None:
            m.add("webBrierSkill" + tail, fnum(row["brier_skill_score"], 3, sign=True), relative)
        if row["ece"] is not None:
            m.add("webEce" + tail, fnum(row["ece"], 4), relative)


def _flatten(payload, prefix=()):
    for key in payload:
        value = payload[key]  # indexed, so the claims trace sees the read
        if isinstance(value, dict):
            yield from _flatten(value, prefix + (key,))
        else:
            yield prefix + (key,), value


#: The web protocol's chronological splits by index (PREREGISTERED.md).
WEB_SPLITS = {"0": "index", "1": "train", "2": "validation", "3": "test"}


def dataset_key_words(path) -> str:
    """Macro name part of a dataset-profile key path. A split index under a
    ``*_by_split`` key is named by its role ('3' -> 'Test'), so the prose
    reads ``\\webDataViewsBySplitTest`` instead of a bare split number."""
    words = []
    for parent, key in zip((None,) + tuple(path[:-1]), path):
        if parent is not None and str(parent).endswith("_by_split") and str(key) in WEB_SPLITS:
            key = WEB_SPLITS[str(key)]
        words.append(camel(key))
    return "".join(words)


def web_dataset(results: g.Results, out: g.Output, m: Macros) -> None:
    """Every numeric key of the (schema-less) dataset profile, as it is."""
    relative = "web/dataset_profile.json"
    payload = results.json_object(relative, out)
    for path, value in _flatten(payload or {}):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        name = "webData" + dataset_key_words(path)
        # A share may be keyed by its kind under a ``*_share`` parent.
        fraction = any(word in str(part) for part in path for word in ("frac", "share", "ratio"))
        text = (fcount(value) if isinstance(value, int)
                else fpct(value) if fraction and abs(value) <= 1 else fnum(value, 2))
        m.add(name, text, relative)


# ---------------------------------------------------------------------------
# Live HTTP
# ---------------------------------------------------------------------------

def live_groups(rows):
    """Citable rows (valid, store foreign I/O at most 1 %; ``g.live_citable``)
    grouped by (experiment, store, config, link, policy)."""
    groups = defaultdict(list)
    for row in g.live_citable(rows):
        groups[(row["experiment"], row["store"], row["config"], row["link"],
                row["policy"])].append(row)
    return groups


def live(results: g.Results, out: g.Output, m: Macros) -> None:
    relative = "live/live_metrics.csv"
    rows = results.rows(relative, out)
    parity = results.rows("live/live_parity.csv", out)
    if parity is not None:
        m.add("liveParityMismatches", fcount(sum(r["mismatched_sessions"] for r in parity)),
              "live/live_parity.csv")
        m.add("liveParityPasses", fcount(len(parity)), "live/live_parity.csv")
    if not rows:
        return
    m.add("liveInvalidPasses", fcount(sum(not r["valid"] for r in rows)), relative)
    m.add("liveValidPasses", fcount(sum(r["valid"] for r in rows)), relative)
    citable = g.live_citable(rows)
    m.add("liveFlaggedRows", fcount(sum(r["valid"] for r in rows) - len(citable)), relative)
    m.add("liveCitableRows", fcount(len(citable)), relative)
    m.add("liveRows", fcount(len(rows)), relative)
    groups = live_groups(rows)
    configs = defaultdict(set)
    for experiment, store, config, link, _ in groups:
        configs[(experiment, store)].add((config, link))
    for (experiment, store, config, link, policy), items in groups.items():
        pre = "" if experiment == "headline" else camel(experiment)
        tail = pre + camel(store) + camel(policy)
        if len(configs[(experiment, store)]) > 1:
            tail += camel(config) + camel(link)
        # Savings, ideal savings and bytes are relative to a baseline pass, so
        # they use only the passes whose baseline was valid too (g.live_saved).
        saved = g.live_saved(items)
        compared = g.live_comparable(items)
        if saved is not None:
            m.add_ci("liveSaved" + tail, *saved, fms, relative)
            m.add("liveIdealSaved" + tail,
                  fms(-statistics.fmean(r["ideal_delta_vs_base_ms"] for r in compared)),
                  relative)
            m.add("liveBytesX" + tail,
                  fratio(statistics.fmean(r["bytes_vs_base"] for r in compared), 2), relative)
            m.add("liveComparedReps" + tail, fcount(len(compared)), relative)
        m.add("liveHit" + tail, fpct(statistics.fmean(r["hit_ratio"] for r in items)), relative)
        m.add("liveWastedKiB" + tail,
              fnum(statistics.fmean(r["wasted_kb_per_view"] for r in items), 1) + r"\,KiB",
              relative)
        m.add("liveMeanWait" + tail,
              fms(statistics.fmean(r["mean_wait_ms"] for r in items)), relative)
        m.add("liveReps" + tail, fcount(len(items)), relative)

# ---------------------------------------------------------------------------
# The classifier study (tracks CLF, HSR, WRK) and the MSRC light sample (E7),
# written after E1-E6 were known. Verdicts and their intervals are the
# ``\dec...`` macros (paper/decisions.py); these families give the levels.
# ---------------------------------------------------------------------------

def _condition_word(condition: str) -> str:
    return "" if condition == "in_regime" else camel(condition.replace("->", " to "))


@g.traced_formatter
def facc(value) -> str:
    """A synthetic accuracy (a fraction, three decimals)."""
    return fnum(value, 3)


@g.traced_formatter
def fdacc(value) -> str:
    """A difference of synthetic accuracies (signed, three decimals)."""
    return fnum(value, 3, sign=True)


def classifier_synthetic(results: g.Results, out: g.Output, m: Macros) -> None:
    """``\\clfSyn<Acc|Delta><Setting>[<Condition>]<Family>`` (Part 1; synthetic
    test accuracy, mean over seed pairs, and its difference from gnb_online
    with the paired interval as ``...CI``), the protocol's sizes, and the
    contended single-window predict times (never a cost)."""
    relative = "classifier/synthetic_eval.csv"
    rows = results.rows(relative, out, required=False)
    if not rows:
        return
    for row in rows:
        tail = camel(row["setting"]) + _condition_word(row["condition"]) + camel(row["family"])
        m.add("clfSynAcc" + tail, facc(row["accuracy"]), relative)
        if row["delta_accuracy"] is not None:
            m.add("clfSynDelta" + tail, fdacc(row["delta_accuracy"]), relative)
            m.add("clfSynDelta" + tail + "CI", fdacc(row["delta_accuracy"]) + "~"
                  + fci(row["delta_accuracy_lo"], row["delta_accuracy_hi"], fdacc), relative)
        if row["setting"] == "baseline" and row["predict_us"] is not None:
            m.add("clfSynPredictUs" + camel(row["family"]), fus(row["predict_us"], 0), relative)
    m.add("clfSynSeedPairs", fcount(_one((r["n_seed_pairs"] for r in rows), "pairs")), relative)
    m.add("clfSynFamilies", fcount(len({r["family"] for r in rows})), relative)
    m.add("clfSynCells", fcount(len(rows)), relative)
    m.add("clfSynComparisons", fcount(sum(r["delta_accuracy"] is not None for r in rows)),
          relative)
    tuning = results.rows("classifier/synthetic_tuning.csv", out, required=False)
    if tuning:
        source = "classifier/synthetic_tuning.csv"
        per = defaultdict(set)
        for row in tuning:
            per[(row["setting"], row["family"])].add(row["candidate"])
        m.add("clfSynCandidates", fcount(max(len(v) for v in per.values())), source)
        m.add("clfSynTunedFamilies", fcount(len({f for (_, f), v in per.items() if len(v) > 1})),
              source)
        m.add("clfSynFolds", fcount(_one((r["folds"] for r in tuning), "folds")), source)


def classifier_router(results: g.Results, out: g.Output, m: Macros) -> None:
    """``\\clfRouter<Qty><Policy><Cap>`` from router_summary.csv (Part 2,
    simulated): Hit, Precision, IORatio, MovedRatio, Agreement (with GNB);
    and the paired win/loss counts of the hit-ratio comparisons."""
    relative = "classifier/router_summary.csv"
    rows = results.rows(relative, out, required=False)
    for row in rows or ():
        tail = camel(row["policy"]) + disk_capacity_word(row["capacity"])
        for quantity, key, formatter in (("Hit", "mean_hit_ratio", fpct),
                                         ("Precision", "mean_precision", fpct),
                                         ("IORatio", "device_io_ratio", fratio),
                                         ("MovedRatio", "blocks_moved_ratio", fratio),
                                         ("Agreement", "mean_agreement_with_gnb",
                                          lambda v: fnum(v, 2))):
            if row[key] is not None:
                m.add("clfRouter" + quantity + tail, formatter(row[key]), relative)
    if rows:
        m.add("clfRouterTraces", fcount(_one((r["n_traces"] for r in rows), "traces")), relative)
        m.add("clfRouterClassifiers", fcount(len({r["policy"] for r in rows} - {
            "none", "sequential", *_decisions_module().CLF_REFERENCE_POLICIES})), relative)
        agreement = [r["mean_agreement_with_gnb"] for r in rows
                     if r["mean_agreement_with_gnb"] is not None and r["policy"] != "gnb_online"]
        if agreement:
            m.add("clfRouterAgreementMin", fnum(min(agreement), 2), relative)
            m.add("clfRouterAgreementMax", fnum(max(agreement), 2), relative)
        reference = ("none", "sequential") + _decisions_module().CLF_REFERENCE_POLICIES
        routed = [r for r in rows if r["policy"] not in reference]
        for capacity in sorted({r["capacity"] for r in routed}):
            ratios = [r["device_io_ratio"] for r in routed if r["capacity"] == capacity
                      and r["device_io_ratio"] is not None]
            if ratios:
                word = disk_capacity_word(capacity)
                m.add("clfRouterIORatioMin" + word, fratio(min(ratios)), relative)
                m.add("clfRouterIORatioMax" + word, fratio(max(ratios)), relative)
    comparisons = "classifier/router_comparisons.csv"
    for row in results.rows(comparisons, out, required=False) or ():
        if row["metric"] != "hit_ratio":
            continue
        stem = ("clfRouterHit" + camel(row["policy"]) + "Vs" + camel(row["baseline"])
                + disk_capacity_word(row["capacity"]))
        for suffix, key in (("Wins", "wins"), ("Losses", "losses"), ("N", "n")):
            m.add(stem + suffix, fcount(row[key]), comparisons)


def _shares(text) -> dict:
    import json
    if isinstance(text, dict):
        return text
    try:
        return json.loads(text) if text else {}
    except (TypeError, ValueError):
        return {}


def hindsight(results: g.Results, out: g.Output, m: Macros) -> None:
    """``\\hsr<Qty><Policy><Cap>`` from hindsight/summary.csv (simulated):
    Hit, IORatio, MovedRatio, Headroom (descriptive), DeepShare (share of
    windows routed to the deepest action); win/loss counts of the H and S
    comparisons; and the grouped-CV ranking of the families (descriptive)."""
    relative = "hindsight/summary.csv"
    rows = results.rows(relative, out, required=False)
    for row in rows or ():
        tail = camel(row["policy"]) + disk_capacity_word(row["capacity"])
        for quantity, key, formatter in (("Hit", "mean_hit_ratio", fpct),
                                         ("IORatio", "device_io_ratio", fratio),
                                         ("MovedRatio", "blocks_moved_ratio", fratio),
                                         ("Headroom", "headroom_captured", lambda v: fnum(v, 3))):
            if row[key] is not None:
                m.add("hsr" + quantity + tail, formatter(row[key]), relative)
        shares = _shares(row["action_shares"])
        if "deep@256" in shares:
            m.add("hsrDeepShare" + tail, fpct(shares["deep@256"], 1), relative,
                  raw=[shares["deep@256"]])
    if rows:
        m.add("hsrTraces", fcount(_one((r["n_traces"] for r in rows), "traces")), relative)
    comparisons = "hindsight/comparisons.csv"
    for row in results.rows(comparisons, out, required=False) or ():
        if row["comparison"] == "D":
            continue
        stem = ("hsr" + camel(row["comparison"]) + camel(row["policy"]) + "Vs"
                + camel(row["baseline"]) + camel(row["metric"])
                + disk_capacity_word(row["capacity"]))
        for suffix, key in (("Wins", "wins"), ("Losses", "losses"), ("Ties", "ties"),
                            ("N", "n")):
            m.add(stem + suffix, fcount(row[key]), comparisons)
    selection = "hindsight/selection.csv"
    sel = results.rows(selection, out, required=False)
    if sel:
        cells = defaultdict(dict)
        for row in sel:
            if row["family_best"]:
                cells[(row["objective"], row["capacity"])][row["family"]] = row["cv_score"]
        lowest_gnb = 0
        for (objective, capacity), scores in sorted(cells.items()):
            lowest = min(scores, key=lambda f: (scores[f], f))
            lowest_gnb += lowest == "gnb"
            best = max(scores, key=lambda f: (scores[f], f))
            word = camel(objective) + disk_capacity_word(capacity)
            m.add("hsrCvLowest" + word, g.esc(lowest), selection)
            m.add("hsrCvBest" + word, g.esc(best), selection)
        m.add("hsrCvCells", fcount(len(cells)), selection)
        m.add("hsrCvGnbLowest", fcount(lowest_gnb), selection)
        m.add("hsrCvFamilies", fcount(len({r["family"] for r in sel})), selection)
        m.add("hsrCvConfigs", fcount(max(sum(1 for r in sel if r["family"] == f
                                             and r["objective"] == sel[0]["objective"]
                                             and r["capacity"] == sel[0]["capacity"])
                                         for f in {r["family"] for r in sel})), selection)
        m.add("hsrTrainTraces", fcount(_one((r["cv_traces_scored"] for r in sel), "traces")),
              selection)


def web_rankers(results: g.Results, out: g.Output, m: Macros) -> None:
    """``\\wrk<Metric><Split><Scorer>`` ranking levels (HitAtOne, MrrAtEight,
    NovelHitAtOne) from ranking_metrics.csv; the selected trial's tuning MRR@8
    and the trial counts; the contended per-request scoring times (never a
    cost); the bootstrap and session counts."""
    relative = "web_rankers/ranking_metrics.csv"
    for row in results.rows(relative, out, required=False) or ():
        tail = camel(row["split"]) + camel(row["scorer"])
        for quantity, key in (("HitAtOne", "hit_at_1"), ("MrrAtEight", "mrr_at_8"),
                              ("NovelHitAtOne", "novel_hit_at_1")):
            if row[key] is not None:
                m.add("wrk" + quantity + tail, fpct(row[key]), relative)
        if row["split"] == "test" and row["request_us"] is not None:
            m.add("wrkRequestUs" + camel(row["scorer"]), fus(row["request_us"], 0), relative)
    tuning = "web_rankers/tuning.csv"
    rows = results.rows(tuning, out, required=False)
    if rows:
        for family in dict.fromkeys(r["family"] for r in rows):
            items = [r for r in rows if r["family"] == family]
            chosen = [r for r in items if r["selected"]]
            if chosen:
                m.add("wrkTuneMrr" + camel(family), fnum(chosen[0]["mrr_at_8"], 4), tuning)
            searched = [r for r in items if r["trial"] >= 0]
            if searched:
                m.add("wrkTrials" + camel(family), fcount(len(searched)), tuning)
        m.add("wrkTrainDecisions", fcount(max(r["train_decisions"] for r in rows)), tuning)
        m.add("wrkTuneDecisions", fcount(_one((r["tune_decisions"] for r in rows), "tune")),
              tuning)
    comparisons = "web_rankers/comparisons.csv"
    rows = results.rows(comparisons, out, required=False)
    if rows:
        m.add("wrkSessions", fcount(_one((r["sessions"] for r in rows), "sessions")), comparisons)
        m.add("wrkBootstrapReps", fcount(_one((r["reps"] for r in rows), "reps")), comparisons)
        s1 = [r for r in rows if r["comparison"] == "S1"]
        for row in s1:
            m.add("wrkSOneScorerCap" + camel(row["capacity"]), g.esc(row["scorer"]), comparisons)
            m.add("wrkSOnePolicyCap" + camel(row["capacity"]), g.esc(row["policy"]),
                  comparisons)


def msrc_samples(results: g.Results, out: g.Output, m: Macros) -> None:
    """E7, descriptive: the sample's size, how closely the seq flag tracks
    address contiguity, and the majority-label balanced accuracy of each
    protocol and classifier on the pooled, web_1 and web_3 scopes."""
    relative = "msrc_samples/traces.csv"
    rows = results.rows(relative, out, required=False)
    if rows:
        m.add("msrcTraces", fcount(len(rows)), relative)
        match = [r["flag_matches_contiguity"] for r in rows
                 if r["flag_matches_contiguity"] is not None]
        if match:
            m.add("msrcFlagContiguityMin", fnum(min(match), 3), relative)
            m.add("msrcFlagContiguityMax", fnum(max(match), 3), relative)
        m.add("msrcWindows", fcount(sum(r["windows"] for r in rows)), relative)
        m.add("msrcSeqWindows", fcount(sum(r["seq_windows_majority"] for r in rows)), relative)
        m.add("msrcSmallNTraces", fcount(sum(bool(r["small_n_majority"]) for r in rows)),
              relative)
    metrics = "msrc_samples/classifier_metrics.csv"
    for row in results.rows(metrics, out, required=False) or ():
        if (row["label_rule"] != "majority" or row["trace"] not in ("pooled", "web_1", "web_3")
                or row["balanced_accuracy"] is None):
            continue
        m.add("msrcBalAcc" + camel(row["protocol"]) + camel(row["classifier"])
              + camel(row["trace"]), facc(row["balanced_accuracy"]), metrics)


# ---------------------------------------------------------------------------
# Pre-registered decisions (paper/decisions.py)
# ---------------------------------------------------------------------------

def _decisions_module():
    if "paper_decisions" not in sys.modules:
        spec = importlib.util.spec_from_file_location("paper_decisions",
                                                      Path(_HERE) / "decisions.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules["paper_decisions"] = module
        spec.loader.exec_module(module)
    return sys.modules["paper_decisions"]


DECISION_SOURCE = "paper/decisions.py"


@g.traced_formatter
def fabspp(value) -> str:
    """An unsigned interval of |D| in percentage points."""
    return fnum(100.0 * value, 2) + r"\,pp"


@g.traced_formatter
def frho(value) -> str:
    return fnum(value, 2, sign=True)


@g.traced_formatter
def fppfine(value) -> str:
    """A difference in percentage points to three decimals."""
    return fpp(value, 3)


def _decision_formatter(statistic):
    return {"difference": fpp, "difference_fine": fppfine, "ratio": fratio, "us": fus,
            "ms": fms, "absolute": fabspp, "rho": frho}[statistic]


def _decision_meta(d) -> dict:
    """What the ledger records about the verdict a macro belongs to."""
    study = {"learned_depth": "gpu"}.get(d.study, d.study)
    return {"rule": d.rule, "study": d.study, "scope": d.scope, "comparison": d.comparison,
            "verdict": d.verdict, "note": d.note,
            "intervals": [list(i) for i in d.intervals],
            "preregistered": f"results/{study}/PREREGISTERED.md"}


def _verdict_word(verdict: str) -> str:
    """A verdict's class for counting: 'raises mean latency' -> 'Raises'."""
    first = verdict.split(" (")[0].split(";")[0]
    return camel({"lowers mean latency": "lowers", "raises mean latency": "raises",
                  "no difference detected": "no difference",
                  "contended": "contended"}.get(first, first))


#: Every verdict each tallied rule can give (paper/decisions.py wording).
TALLY_VERDICTS = {
    "C3": ("robust", "prefix-dependent"),
    "C4": ("cap-insensitive", "cap-sensitive"),
    "H2": ("lowers the mean wait", "raises the mean wait", "no consistent difference",
           "not established"),
    "H3": ("the instant-prefetch model overstates the benefit",
           "the instant-prefetch model understates the benefit", "consistent with the model"),
    "H7": ("the store changes the delta", "no store difference by the H7 rule"),
    "R7": ("think time changes the effect", "no change by the R7 rule"),
    "R2": ("ranks consistently with measured latency", "ranks against measured latency",
           "ranking not supported", "descriptive only", "contended"),
    # The classifier study and E7 (paper/decisions.py, after E1-E6).
    "CLF0": ("better", "worse", "no difference detected"),
    "CLF1": ("better", "worse", "no difference detected"),
    "CLF2": ("better", "worse", "no difference detected"),
    "CLF3": ("reduces device I/Os", "increases device I/Os", "no difference detected"),
    "CLF4": ("moves fewer blocks", "moves more blocks", "no difference detected"),
    "D": ("reduces device I/Os", "increases device I/Os", "moves fewer blocks",
          "moves more blocks", "no difference detected"),
    "M1": ("better", "worse", "no difference detected", "insufficient n"),
    "M2": ("better", "worse", "no difference detected", "insufficient n"),
}


def _rule_word(rule: str) -> str:
    """A rule's part of a tally name: 'CLF0' -> 'ClfZero', 'H2' -> 'HTwo'."""
    return camel(rule.lower()) if rule.startswith("CLF") else camel(rule)


def _tally_group(d) -> str | None:
    """The group a verdict is counted in for ``\\decTally<Rule><Group><Verdict>``
    (``None``: not tallied): C3/C4, H7 and R7 over all their comparisons, H2/H3
    per store, R2 per simulated quantity."""
    if d.study in _decisions_module().COMPACT_STUDIES:
        return _classifier_tally_group(d)
    if d.rule in ("C3", "C4", "H7", "R7"):
        return ""
    if d.rule in ("H2", "H3"):
        return camel(d.scope.split()[0])
    if d.rule == "R2":
        return "Hit" if "simulated hit," in d.comparison else "DeviceTime"
    return None


def _classifier_tally_group(d) -> str | None:
    """The tally group of a classifier-study or E7 verdict (CLF0 per setting,
    CLF1-CLF4 per capacity over the routed classifiers, HSR's D per capacity
    and quantity, E7's M1/M2 per protocol and scope)."""
    if d.rule == "CLF0":
        setting, condition = d.scope.split(", ")
        return camel(setting) + ("" if condition == "in_regime"
                                 else camel(condition.replace("->", " to ")))
    if d.rule in ("CLF1", "CLF2", "CLF3", "CLF4"):
        # The routed classifiers only; the reference policies are not counted.
        policy = d.comparison.split(" - ")[0].split(" / ")[0]
        reference = _decisions_module().CLF_REFERENCE_POLICIES
        return None if policy in reference else camel(d.scope)
    if d.rule == "D":
        return camel(d.scope) + ("IO" if "device I/Os" in d.comparison else "Moved")
    if d.rule in ("M1", "M2"):
        trace, protocol = d.scope.split(", ")
        return camel(protocol) + camel(trace)
    return None


def decision_macros(results: g.Results, out: g.Output, m: Macros) -> None:
    """``\\dec<Key>Verdict``: the verdict a pre-registered rule gives, as text,
    and ``\\dec<Key><Interval>`` (with Lo, Hi, CI; the unnamed main interval
    is ``\\dec<Key>`` itself) for each interval it rests on;
    plus verdict counts per real-device cell group and the policies P5 finds
    reducing device I/Os."""
    g.TRACE.files_read = set()
    decisions = _decisions_module().decide(results, out)
    read = sorted(g.TRACE.files_read)

    def count_meta(name: str) -> dict:
        """A verdict count: the files the decisions it counts were read from."""
        rest = name[len("dec"):]
        study = ("disk" if rest.startswith(("PFive", "TallyCThree", "TallyCFour")) else
                 "classifier" if rest.startswith("TallyClf") else
                 "hindsight" if rest.startswith("TallyD") else
                 "msrc_samples" if rest.startswith("TallyM") else
                 "live" if rest.startswith(("TallyH", "CountH")) else "realio")
        return {"study": study, "files": [f for f in read if f.startswith(study + "/")],
                "preregistered": f"results/{study}/PREREGISTERED.md"}
    counts = defaultdict(int)
    tallies = defaultdict(int)
    reducers = defaultdict(list)
    fastest = defaultdict(lambda: [0, 0, 0])
    module = _decisions_module()
    for d in decisions:
        stem = "dec" + d.key[:1].upper() + d.key[1:]
        meta = _decision_meta(d)
        m.add(stem + "Verdict", g.esc(d.verdict), DECISION_SOURCE, meta=meta)
        for label, estimate, lo, hi, statistic in d.intervals:
            if d.rule in module.VERDICT_ONLY_RULES:
                break  # numbers.py's own family prints these intervals
            name = stem + camel(label)
            formatter = _decision_formatter(statistic)
            if estimate is not None and d.study in module.COMPACT_STUDIES:
                # The estimate and "estimate [lo, hi]" only (TeX memory).
                m.add(name, formatter(estimate), DECISION_SOURCE, meta=meta)
                if lo is not None and hi is not None:
                    m.add(name + "CI", formatter(estimate) + "~" + fci(lo, hi, formatter),
                          DECISION_SOURCE, meta=meta)
            elif estimate is not None:
                m.add_ci(name, estimate, lo, hi, formatter, DECISION_SOURCE, meta=meta)
            elif lo is not None and hi is not None:
                m.add(name + "Lo", formatter(lo), DECISION_SOURCE, meta=meta)
                m.add(name + "Hi", formatter(hi), DECISION_SOURCE, meta=meta)
                m.add(name + "CI", fci(lo, hi, formatter), DECISION_SOURCE, meta=meta)
        if d.study == "realio" and d.rule in ("R1", "R6"):
            device, backend, think = d.scope.split(", ")[:3]
            group = camel(device) + camel(backend) + camel(think)
            counts["decCount" + camel(d.rule) + group + _verdict_word(d.verdict)] += 1
            counts["decCount" + camel(d.rule) + group] += 1
        group = _tally_group(d)
        if group is not None:
            stem_count = "decTally" + _rule_word(d.rule) + group
            tallies[stem_count] += 1
            tallies[stem_count + camel(d.verdict.split(" (")[0].split(";")[0])] += 1
        if d.rule == "R2" and d.verdict != "contended (not used)" and d.intervals:
            rho = d.intervals[0][1]
            sign = "Hit" if "simulated hit," in d.comparison else "DeviceTime"
            device = camel(d.scope.split(", ")[0])
            if rho is not None:
                tallies["decRhoCells" + sign + device] += 1
                tallies["decRhoPositive" + sign + device] += rho > 0
        if d.study == "disk" and d.rule == "P5" and d.verdict == "reduces device I/Os":
            reducers[d.scope].append(d.comparison.split(" / ")[0])
        if d.study == "realio" and d.rule == "fastest":
            device, backend = d.scope.split(", ")[:2]
            for key in ("", camel(device) + camel(backend)):
                tally = fastest[key]
                tally[0] += 1
                tally[1] += d.verdict == "none"
                tally[2] += d.note == "highest simulated hit ratio: " + d.verdict
    # Every verdict class gets a count in every group, zero included, so the
    # prose can say "lowered it in none" through a macro.
    classes = ("Lowers", "Raises", "NoDifference", "Contended", "NotEstablished")
    for name in [n for n in counts if not n.endswith(classes)]:
        for word in classes:
            counts.setdefault(name + word, 0)
    for name, value in sorted(counts.items()):
        m.add(name, fcount(value), DECISION_SOURCE, meta=count_meta(name))
    # Zero counts too, for every verdict a tallied rule can give.
    for name in [n for n in tallies if n.startswith("decTally")]:
        for rule, verdicts in TALLY_VERDICTS.items():
            prefix = "decTally" + _rule_word(rule)
            if name.startswith(prefix) and not any(
                    name.endswith(camel(v)) and len(name) > len(prefix) for v in verdicts):
                for verdict in verdicts:
                    tallies.setdefault(name + camel(verdict), 0)
    for name, value in sorted(tallies.items()):
        m.add(name, fcount(value), DECISION_SOURCE, meta=count_meta(name))
    for scope, policies in reducers.items():
        m.add("decPFiveReducers" + camel(scope.replace(" ", "")),
              g.esc(", ".join(policies)), DECISION_SOURCE, meta=count_meta("decPFive"))
    for key, (cells, none_fastest, top_fastest) in fastest.items():
        meta = count_meta("decCells")
        m.add("decCells" + key, fcount(cells), DECISION_SOURCE, meta=meta)
        m.add("decCellsNoneFastest" + key, fcount(none_fastest), DECISION_SOURCE, meta=meta)
        m.add("decCellsTopSimFastest" + key, fcount(top_fastest), DECISION_SOURCE, meta=meta)


# ---------------------------------------------------------------------------
# U(c) envelopes (Fig. F8): the best policy at the costs the paper marks
# ---------------------------------------------------------------------------

def envelopes(results: g.Results, out: g.Output, m: Macros) -> None:
    """The policy on top of the U(c) envelope at each marked cost, from the
    same rows Fig. F8 draws (``genlib.disk_utility_rows``; the web test split,
    logistic scorer, capacity 8)."""
    sweep = results.rows("disk/sweep.jsonl", out, required=False)
    if sweep:
        for capacity in g.HEADLINE_CAPACITIES:
            rows, n = g.disk_utility_rows(sweep, capacity)
            if not rows:
                continue
            word = disk_capacity_word(capacity)
            m.add("diskEnvelopeTraces" + word, fcount(n), "disk/sweep.jsonl")
            for name, c in (("Assumed", utility.cost_to_c(5, 100, 50)),
                            ("BreakEven", utility.BREAK_EVEN_C), ("Zero", 0.0)):
                best = utility.envelope(rows, [c])[0]
                m.add("diskEnvelopeBest" + name + word, g.esc(best["policy"]), "disk/sweep.jsonl")
    metrics = results.rows("web/policy_metrics.csv", out, required=False)
    if metrics:
        rows = g.select(metrics, split="test", cache_scope="session", scorer="logistic",
                        capacity=8)
        if rows:
            costs = sorted({r["cost"] for r in results.rows("web/comparisons.csv", out,
                                                              required=False) or ()})
            for c in costs or (0.0,):
                best = utility.envelope(rows, [c])[0]
                m.add("webEnvelopeBestCost" + camel(repr(float(c))), g.esc(best["policy"]),
                      "web/policy_metrics.csv")


# ---------------------------------------------------------------------------
# Protocol facts: slice lengths, repeats, sample sizes, assumed inputs
# ---------------------------------------------------------------------------

def _one(values, what):
    values = sorted({v for v in values if v is not None})
    return values[0] if len(values) == 1 else None


def facts(results: g.Results, out: g.Output, m: Macros) -> None:
    """The protocol's own parameters as the results files record them, so the
    prose never types a sample size, slice length or assumed input."""
    sweep = results.rows("disk/sweep.jsonl", out, required=False)
    if sweep:
        for config in ("C1", "C2", "C3", "C4"):
            rows = [r for r in sweep if r["config"] == config and r["skip_requests"] == 0]
            cap = _one((r["max_requests"] for r in rows), config)
            if cap is not None:
                m.add("diskSlice" + camel(config), fcount(cap), "disk/sweep.jsonl")
            skips = sorted({r["skip_requests"] for r in sweep if r["config"] == config
                            and r["skip_requests"]})
            if skips:
                m.add("diskSkip" + camel(config), fcount(skips[0]), "disk/sweep.jsonl")
    deltas = results.rows("disk/deltas.csv", out, required=False)
    if deltas:
        m.add("diskBootstrapReps", fcount(_one((r["reps"] for r in deltas), "reps")),
              "disk/deltas.csv")
    summary = results.rows("realio/realio_summary.csv", out, required=False)
    if summary:
        m.add("realioRequests", fcount(_one((r["requests"] for r in summary), "requests")),
              "realio/realio_summary.csv")
        m.add("realioReps", fcount(_one((r["reps"] for r in summary), "reps")),
              "realio/realio_summary.csv")
        m.add("realioTraces", fcount(len({r["trace"] for r in summary})),
              "realio/realio_summary.csv")
    raw = results.rows("realio/realio.csv", out, required=False)
    if raw:
        m.add("realioRuns", fcount(len(raw)), "realio/realio.csv")
        m.add("realioOdirectRuns", fcount(sum(r["backend"] == "odirect" for r in raw)),
              "realio/realio.csv")
    live_rows = results.rows("live/live_metrics.csv", out, required=False)
    if live_rows:
        source = "live/live_metrics.csv"
        for name, key, formatter in (("liveSessions", "sessions", fcount),
                                     ("liveViews", "views", fcount),
                                     ("liveRttMs", "rtt_ms", lambda v: fnum(v, 0) + r"\,ms"),
                                     ("liveMbps", "mbps", lambda v: fnum(v, 0) + r"\,Mbit/s"),
                                     ("liveSizeMedianKiB", "size_median_kib",
                                      lambda v: fnum(v, 0) + r"\,KiB"),
                                     ("liveGapCapS", "gap_cap_s", lambda v: fnum(v, 0) + r"\,s"),
                                     ("liveRampS", "ramp_s", lambda v: fnum(v, 0) + r"\,s")):
            value = _one((r[key] for r in live_rows), key)
            if value is not None:
                m.add(name, formatter(value), source)
        headline = [r for r in live_rows if r["experiment"] == "headline"]
        m.add("liveHeadlineReps", fcount(len({r["rep"] for r in headline})), source)
        m.add("livePasses", fcount(len({(r["experiment"], r["store"], r["config"], r["rep"],
                                         r["policy"]) for r in live_rows})), source)
        base = [r for r in headline if r["policy"] == r["baseline"]]
        for store in sorted({r["store"] for r in base}):
            waits = [r["mean_wait_ms"] for r in base if r["store"] == store]
            m.add("liveBaseWait" + camel(store), fms(statistics.fmean(waits)), source)
            ideal = [r["model_ideal_wait_ms"] for r in base if r["store"] == store]
            m.add("liveBaseIdealWait" + camel(store), fms(statistics.fmean(ideal)), source)
    comparisons = results.rows("web/comparisons.csv", out, required=False)
    if comparisons:
        costs = sorted({r["cost"] for r in comparisons})
        m.add("webCosts", ", ".join(fnum(c, 2 if c else 0) for c in costs),
              "web/comparisons.csv")
        m.add("webCapacities", ", ".join(fcount(c) for c in
                                         sorted({r["capacity"] for r in comparisons})),
              "web/comparisons.csv")
    decomposition = results.rows("web/decomposition.csv", out, required=False)
    if decomposition:
        # The cost the pre-registered utility comparisons (W1-W8) use.
        cost = _one((r["cost"] for r in decomposition if r["metric"] != "hit_ratio"), "cost")
        if cost is not None:
            m.add("webHeadlineCost", fnum(cost, 2), "web/decomposition.csv")
        m.add("webSessions", fcount(_one((r["sessions"] for r in decomposition), "sessions")),
              "web/decomposition.csv")
        m.add("webBootstrapReps", fcount(_one((r["reps"] for r in decomposition), "reps")),
              "web/decomposition.csv")
    learned = results.rows("gpu/disk_learned_depth.csv", out, required=False)
    if learned:
        source = "gpu/disk_learned_depth.csv"
        m.add("ldPrefixMax", fcount(max(r["n_requests"] for r in learned)), source)
        # The split points as shares of each trace's prefix (short FIU traces
        # use their whole length, so the request counts differ per trace).
        for name, start, end in (("ldTrainShare", "train_start", "train_end"),
                                 ("ldValShare", "val_start", "val_end"),
                                 ("ldTestShare", "test_start", "test_end")):
            share = _one((round((r[end] - r[start]) / r["n_requests"], 4) for r in learned
                          if r["n_requests"]), name)
            if share is not None:
                m.add(name, fnum(100 * share, 0) + r"\,\%", source)
    lstm = results.rows("gpu/disk_lstm.csv", out, required=False)
    if lstm:
        ok = [r for r in lstm if r["status"] == "ok"]
        m.add("lstmTraces", fcount(len(ok)), "gpu/disk_lstm.csv")
        pairs = [(r["lstm_hit_ratio"], r["linux_ra_hit_ratio"], r["deep8_hit_ratio"],
                  r["none_hit_ratio"]) for r in ok]
        m.add("lstmBelowLinuxRa", fcount(sum(a < b for a, b, _, _ in pairs)), "gpu/disk_lstm.csv")
        m.add("lstmBelowDeepEight", fcount(sum(a < c for a, _, c, _ in pairs)),
              "gpu/disk_lstm.csv")
        m.add("lstmMeanHit", fpct(statistics.fmean(p[0] for p in pairs)), "gpu/disk_lstm.csv")
        m.add("lstmMeanLinuxRaHit", fpct(statistics.fmean(p[1] for p in pairs)),
              "gpu/disk_lstm.csv")
        m.add("lstmMeanDeepEightHit", fpct(statistics.fmean(p[2] for p in pairs)),
              "gpu/disk_lstm.csv")
        m.add("lstmMeanNoneHit", fpct(statistics.fmean(p[3] for p in pairs)),
              "gpu/disk_lstm.csv")


# ---------------------------------------------------------------------------
# Constants and host
# ---------------------------------------------------------------------------

def constants(m: Macros) -> None:
    """Values of the cost models themselves, quoted in the model section."""
    source = "prefetch_common.utility"
    m.add("diskBreakEvenC", fnum(utility.BREAK_EVEN_C, 4), source)
    m.add("diskAssumedC", fnum(utility.cost_to_c(5, 100, 50), 3), source)
    # The interval level every study uses (the bootstraps' default).
    import inspect
    level = inspect.signature(stats.paired_bootstrap).parameters["level"].default
    m.add("ciLevel", fnum(100 * level, 0) + r"\,\%", "prefetch_common.stats")


def host(results: g.Results, out: g.Output, m: Macros) -> None:
    """Host facts from the first provenance that records them (the studies ran
    on one machine), and the GPU from the GPU study's own provenance."""
    for study in ("disk", "realio", "web", "live", "gpu"):
        record = results.provenance(study)
        info = (record or {}).get("host") or {}
        if not info:
            continue
        source = f"{study}/provenance.json"
        if info.get("cpu_count") is not None:
            m.add("hostCpuCount", fcount(info["cpu_count"]), source)
        if info.get("kernel"):
            m.add("hostKernel", g.esc(info["kernel"]), source)
        break
    gpus = (results.provenance("gpu") or {}).get("gpu")
    if isinstance(gpus, list) and gpus and gpus[0].get("name"):
        m.add("hostGpu", g.esc(gpus[0]["name"]), "gpu/provenance.json")


FAMILIES = (disk_tables, disk_best_depth, disk_deltas, disk_corpora, disk_kappa, learned_depth, realio,
            web_policy_metrics, web_comparisons, web_decomposition, web_freshness, web_seq,
            web_dataset, web_calibration, live, classifier_synthetic, classifier_router, hindsight,
            web_rankers, msrc_samples, host, decision_macros, facts, envelopes)


def build(results: g.Results, *, trace: bool = True) -> tuple[Macros, g.Output]:
    """Every macro, with its trace (``Macros.traces``) when ``trace`` is on."""
    out = results.output(OUTPUT)
    m = Macros(results.fixture)
    previous = g.TRACE.active
    if trace:
        g.TRACE.reset()
        g.TRACE.active = True
    try:
        _build(results, out, m)
    finally:
        g.TRACE.active = previous
    return m, out


def _build(results: g.Results, out: g.Output, m: Macros) -> None:
    constants(m)
    for family in FAMILIES:
        family(results, out, m)
    for alias, source, suffixes, _ in CURATED:
        for suffix in suffixes:
            if alias + suffix not in m.values:
                m.add(alias + suffix, MISSING, source)
                gap = source if source not in out.inputs else f"{source}: no row for \\{alias}"
                if gap not in out.missing:
                    out.missing.append(gap)


def generate(results: g.Results, paper_dir: Path = g.PAPER_DIR) -> list[g.Output]:
    macros, out = build(results, trace=False)
    paper_dir.mkdir(parents=True, exist_ok=True)
    (paper_dir / OUTPUT).write_text(macros.render(out.missing, results.display_root),
                                    encoding="utf-8")
    out.note = f"{len(macros.values)} macros"
    g.write_manifest(paper_dir, [out], results, name=MANIFEST)
    return [out]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results", type=Path, default=g.DEFAULT_RESULTS)
    parser.add_argument("--paper-dir", type=Path, default=g.PAPER_DIR,
                        help="where numbers.tex and numbers.manifest.json go")
    args = parser.parse_args(argv)
    results = g.Results(args.results)
    outputs = generate(results, args.paper_dir)
    print(f"numbers.py: {outputs[0].note}; missing {len(outputs[0].missing)} input(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
