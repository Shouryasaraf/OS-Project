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


def fus(value, digits: int = 1) -> str:
    """Microseconds."""
    return fnum(value, digits) + (r"\,$\mu$s" if g.finite(value) else "")


def fms(value, digits: int = 2) -> str:
    return fnum(value, digits) + (r"\,ms" if g.finite(value) else "")


def fsignedpct(fraction, digits: int = 1) -> str:
    """A signed relative change as a percentage: -0.2 -> ``$-20.0$\\,\\%``."""
    if not g.finite(fraction):
        return fnum(fraction)
    return fnum(100.0 * fraction, digits, sign=True) + r"\,\%"


def fpvalue(p) -> str:
    if p is None:
        return g.DASH
    if p < 1e-4:
        return r"$<10^{-4}$"
    return f"${p:.2g}$" if p < 0.01 else f"${p:.3f}$"


class Macros:
    """An ordered set of macro definitions, each tagged with its source file."""

    def __init__(self, fixture: bool):
        self.fixture = fixture
        self.values: dict[str, str] = {}
        self.sources: dict[str, str] = {}

    def add(self, name: str, text: str, source: str) -> None:
        if not g.is_macro_name(name):
            raise ValueError(f"not a TeX control word: {name!r}")
        if name in self.values:
            if self.values[name] == text:
                return
            raise ValueError(f"macro \\{name} defined twice ({self.sources[name]}, {source})")
        self.values[name] = text
        self.sources[name] = source

    def add_ci(self, name: str, estimate, lo, hi, formatter, source: str, **kwargs) -> None:
        self.add(name, formatter(estimate, **kwargs), source)
        if lo is None or hi is None:
            return
        self.add(name + "Lo", formatter(lo, **kwargs), source)
        self.add(name + "Hi", formatter(hi, **kwargs), source)
        self.add(name + "CI", formatter(estimate, **kwargs) + "~"
                 + fci(lo, hi, formatter, **kwargs), source)

    def alias(self, alias: str, target: str) -> None:
        """Copy ``target`` (and its interval suffixes) to ``alias``."""
        for suffix in PAIRED_SET + ("Ties",):
            if target + suffix in self.values:
                self.add(alias + suffix, self.values[target + suffix],
                         self.sources[target + suffix])

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


def disk_deltas(results: g.Results, out: g.Output, m: Macros) -> None:
    relative = "disk/deltas.csv"
    rows = results.rows(relative, out) or []
    names = delta_names(rows)
    for name, row in zip(names, rows):
        formatter = delta_formatter(row)
        m.add_ci(name, row["estimate"], row["ci_lo"], row["ci_hi"], formatter, relative)
        if row["cluster_estimate"] is not None:
            m.add_ci(name + "Cluster", row["cluster_estimate"], row["cluster_ci_lo"],
                     row["cluster_ci_hi"], formatter, relative)
        m.add(name + "Pvalue", fpvalue(row["wilcoxon_p"]), relative)
        for suffix, key in (("Wins", "wins"), ("Losses", "losses"), ("Ties", "ties"),
                            ("N", "n")):
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


def web_policy_metrics(results: g.Results, out: g.Output, m: Macros) -> None:
    relative = "web/policy_metrics.csv"
    for row in results.rows(relative, out) or ():
        scope = "" if row["cache_scope"] == "session" else camel(row["cache_scope"])
        tail = (camel(row["scorer"]) + scope + "Cap" + camel(row["capacity"])
                + camel(row["policy"]) + _split_word(row["split"]))
        m.add("webHit" + tail, fpct(row["hit_ratio"]), relative)
        m.add("webIssued" + tail, fnum(row["issued_per_view"], 2), relative)
        m.add("webPrecision" + tail, fpct(row["precision"]), relative)
        m.add("webWasted" + tail, fnum(row["wasted_per_view"], 2), relative)
        if row["c_star"] is not None:
            m.add("webCStar" + tail, fnum(row["c_star"], 4), relative)


def web_comparisons(results: g.Results, out: g.Output, m: Macros) -> None:
    relative = "web/comparisons.csv"
    rows = results.rows(relative, out)
    for row in rows or ():
        tail = (camel(row["selection"]) + camel(row["scorer"]) + "Cap" + camel(row["capacity"])
                + "Cost" + camel(repr(float(row["cost"]))) + _split_word(row["split"]))
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
                if r["cost"] > 0]
    if headline:
        row = min(headline, key=lambda r: (r["scorer"] != "logistic", r["cost"]))
        tail = (camel(row["selection"]) + camel(row["scorer"]) + "Cap" + camel(row["capacity"])
                + "Cost" + camel(repr(float(row["cost"]))))
        m.alias("webControllerVsBestFixed", "webCmpVsBestFixed" + tail)


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


def _flatten(payload, prefix=()):
    for key, value in payload.items():
        if isinstance(value, dict):
            yield from _flatten(value, prefix + (key,))
        else:
            yield prefix + (key,), value


def web_dataset(results: g.Results, out: g.Output, m: Macros) -> None:
    """Every numeric key of the (schema-less) dataset profile, as it is."""
    relative = "web/dataset_profile.json"
    payload = results.json_object(relative, out)
    for path, value in _flatten(payload or {}):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        name = "webData" + "".join(camel(p) for p in path)
        fraction = any(word in path[-1] for word in ("frac", "share", "ratio"))
        text = (fcount(value) if isinstance(value, int)
                else fpct(value) if fraction and abs(value) <= 1 else fnum(value, 2))
        m.add(name, text, relative)


# ---------------------------------------------------------------------------
# Live HTTP
# ---------------------------------------------------------------------------

def live_groups(rows):
    """Valid rows grouped by (experiment, store, config, link, policy)."""
    groups = defaultdict(list)
    for row in rows:
        if row["valid"]:
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
# Constants and host
# ---------------------------------------------------------------------------

def constants(m: Macros) -> None:
    """Values of the cost models themselves, quoted in the model section."""
    source = "prefetch_common.utility"
    m.add("diskBreakEvenC", fnum(utility.BREAK_EVEN_C, 4), source)
    m.add("diskAssumedC", fnum(utility.cost_to_c(5, 100, 50), 3), source)


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


FAMILIES = (disk_tables, disk_deltas, disk_corpora, disk_kappa, learned_depth, realio,
            web_policy_metrics, web_comparisons, web_decomposition, web_freshness, web_seq,
            web_dataset, live, host)


def build(results: g.Results) -> tuple[Macros, g.Output]:
    out = results.output(OUTPUT)
    m = Macros(results.fixture)
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
    return m, out


def generate(results: g.Results, paper_dir: Path = g.PAPER_DIR) -> list[g.Output]:
    macros, out = build(results)
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
