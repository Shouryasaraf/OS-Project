"""Build and execute ``notebooks/walkthrough.ipynb`` from ``results/``.

    PYTHONPATH=src .venv/bin/python notebooks/build_notebook.py [--out PATH] [--no-execute]
    python reproduce.py notebook                     # the same, with a runtime estimate

The notebook is written as source here (``CELLS``) so it can be reviewed as
code and rebuilt whenever ``results/`` changes. The build executes every
code cell in order in one namespace (cwd = the repository root, matplotlib's
Agg backend), the way a kernel would, and stores the outputs in the
notebook: printed text as a stream, a DataFrame as ``text/html`` plus
``text/plain``, every open matplotlib figure as a PNG. The committed
``.ipynb`` therefore opens pre-rendered in VS Code or on GitHub, and still
re-runs normally under a real Jupyter kernel.

No kernel (ipykernel, nbclient) is needed to build it: only nbformat,
pandas and matplotlib (the ``[web,paper]`` extras). The output is
deterministic: fixed cell ids, no timestamps, fixed figure DPI, PNG
metadata stripped, a fixed pandas float format.

Every number the notebook shows is computed by a cell from ``results/``;
the markdown cells only say where to look.
"""

from __future__ import annotations

import argparse
import ast
import base64
import contextlib
import io
import os
import sys
import traceback
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "notebooks" / "walkthrough.ipynb"
FIGURE_DPI = 90
PANDAS_OPTIONS = ("display.float_format", "display.max_rows", "display.max_columns",
                  "display.width")


def md(text: str) -> tuple[str, str]:
    return ("markdown", text.strip("\n"))


def code(text: str) -> tuple[str, str]:
    return ("code", text.strip("\n"))


# ---------------------------------------------------------------------------
# The cells, in the order of the talk and the paper
# ---------------------------------------------------------------------------

CELLS = [
    md("""
# Depth and freshness, not learned prediction: a walk through the results

This notebook walks through every experiment of the project in the order of
the talk and the paper (`paper/main.tex`): the data, the disk simulation
(E1), the GPU-trained predictors (E2), the real-device testbed (E3), the web
replay (E4, E5), the live HTTP replay (E6), the pre-registered decisions,
and the classifier studies that came after them (E7 and later).

**Every number below is computed by a code cell from the tracked `results/`
tree.** Nothing is typed by hand; the markdown only says which file and
which paper section or table a cell corresponds to. Rebuild it after new
results land with `python reproduce.py notebook`.

How to read it:

- **Simulated or measured.** E1, E2, E4 and E5 are *simulated* (a causal
  cache replay): their hit ratios are not latencies. Latency is *measured*
  only in E3 (`results/realio/`, a real HDD `sdb` and a shared SSD) and E6
  (`results/live/`, loopback with an emulated network and assumed object
  sizes).
- **Hit ratio is not device work.** Every disk hit-ratio difference is shown
  next to the `device_ios` and `blocks_moved` ratios against `none`.
- **Pre-registered words only.** "better", "worse" and "no difference
  detected" come from each study's `results/<study>/PREREGISTERED.md`; the
  verdict columns are computed by `paper/decisions.py`, the same code that
  writes the paper's verdict macros. On disk, the paired and the
  corpus-cluster bootstrap intervals must agree.
- **`belady_min`** is Belady's MIN for demand paging *without* prefetch: a
  reference for replacement, not an upper bound for prefetching policies.
- **Flagged rows are not cited.** SSD rows carry `foreign_io_frac` (other
  tenants' I/O); a row above the 1% limit, a busy-host row or an invalid
  live pass is shown as flagged and kept out of every comparison.
"""),
    md("""
## 0. Setup

The first cell finds the repository root, puts `src/` on `sys.path` (so no
`PYTHONPATH` is needed in the kernel) and loads `paper/genlib.py` and
`paper/decisions.py` **by file path**: `paper/` must never be on `sys.path`,
because `paper/numbers.py` would shadow the standard library's `numbers`.
The kernel needs the `[web,paper]` extras (pandas, numpy, matplotlib, scipy).
"""),
    code('''
import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = Path.cwd().resolve()
while not (ROOT / "reproduce.py").is_file() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
os.chdir(ROOT)
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    display
except NameError:  # outside IPython
    from IPython.display import display

pd.set_option("display.float_format", "{:.4f}".format)
pd.set_option("display.max_rows", 400)
pd.set_option("display.max_columns", 40)
pd.set_option("display.width", 200)
plt.rcParams.update({"figure.dpi": 90, "savefig.dpi": 90, "figure.figsize": (9, 3.6),
                     "axes.grid": True, "grid.alpha": 0.3})


def load_by_path(name, relative):
    """Import a paper/ script by file path (never put paper/ on sys.path)."""
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ROOT / relative)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


genlib = load_by_path("paper_genlib", "paper/genlib.py")
decisions = load_by_path("paper_decisions", "paper/decisions.py")
RESULTS = ROOT / "results"
TREE = genlib.Results(RESULTS)        # schema-checked reader shared with the paper


def csv(relative):
    return pd.read_csv(RESULTS / relative)


def ci(lo, hi, scale=1.0, digits=2):
    """'[lo, hi]' with a fixed format; '--' when either end is missing."""
    if lo is None or hi is None or pd.isna(lo) or pd.isna(hi):
        return "--"
    return f"[{lo * scale:.{digits}f}, {hi * scale:.{digits}f}]"


def fmt_interval(item):
    label, estimate, lo, hi, statistic = item
    scale, digits, unit = {"difference": (100, 2, " pp"), "absolute": (100, 2, " pp"),
                           "ratio": (1, 3, "x"), "us": (1, 0, " us"),
                           "ms": (1, 2, " ms"), "rho": (1, 3, "")}.get(statistic, (1, 4, ""))
    head = f"{label}: " if label else ""
    point = "--" if estimate is None else f"{estimate * scale:.{digits}f}{unit}"
    if lo is None or hi is None:
        return head + point
    return head + f"{point} {ci(lo, hi, scale, digits)}"


DECISIONS = decisions.decide(TREE)


def decision_frame(study=None, rules=None, scope_contains=None):
    """The pre-registered verdicts of paper/decisions.py as a table."""
    rows = []
    for d in DECISIONS:
        if study and d.study != study:
            continue
        if rules and d.rule not in rules:
            continue
        if scope_contains and scope_contains not in d.scope:
            continue
        rows.append({"study": d.study, "rule": d.rule, "scope": d.scope,
                     "comparison": d.comparison, "verdict": d.verdict,
                     "interval(s)": "; ".join(fmt_interval(i) for i in d.intervals),
                     "note": d.note})
    return pd.DataFrame(rows)


print(f"{len(DECISIONS)} pre-registered verdicts computed from results/")
'''),
    md("""
### Provenance of every results directory

Each results directory carries a `provenance.json` with the SHA-256 of every
file it holds; `prefetch_common.provenance.verify_provenance` re-hashes them.
An empty "problems" column means no file was edited after its reproduce
command wrote it.
"""),
    code('''
from prefetch_common import provenance

rows = []
for directory in sorted(p for p in RESULTS.iterdir() if p.is_dir()):
    record_path = directory / "provenance.json"
    if not record_path.is_file():
        rows.append({"directory": directory.name, "files": None, "git_sha": None,
                     "problems": "no provenance.json yet"})
        continue
    record = json.loads(record_path.read_text())
    problems = provenance.verify_provenance(directory)
    rows.append({"directory": directory.name, "files": len(record.get("outputs", {})),
                 "git_sha": (record.get("git_sha") or "")[:10],
                 "problems": "; ".join(problems)})
pd.DataFrame(rows)
'''),
    md("""
## 1. Data and corpora

**Disk.** The libCacheSim `oracleGeneral` block traces from four corpora
(MSR, FIU, Alibaba, Tencent). Writes replay as reads and the traces carry no
service times, so oracle-trace hit ratios are never compared with
MSRC-trace-003 figures. Source: `results/disk/corpus_profile.csv`; paper
Table `t1_datasets` (Sec. Methodology).
"""),
    code('''
profile = csv("disk/corpus_profile.csv")
by_corpus = profile.groupby("corpus").agg(
    traces=("trace", "count"), paper_subset=("paper_subset", "sum"),
    requests=("requests", "sum"), blocks=("blocks", "sum"),
    median_mean_request_blocks=("mean_request_blocks", "median"),
    median_contiguous_fraction=("contiguous_fraction", "median"),
    twin_grouped=("twin_group", lambda s: int(s.notna().sum())))
by_corpus.loc["all"] = [len(profile), int(profile["paper_subset"].sum()),
                        int(profile["requests"].sum()), int(profile["blocks"].sum()),
                        profile["mean_request_blocks"].median(),
                        profile["contiguous_fraction"].median(),
                        int(profile["twin_group"].notna().sum())]
by_corpus
'''),
    md("""
**Web.** RetailRocket product views, sessions broken after 30 minutes and at
every split boundary; split 0 builds the index, split 1 trains the ranker,
split 2 is the only split choices are made on, split 3 is replayed once.
Source: `results/web/dataset_profile.json`; paper Table `t4_web_dataset`.
"""),
    code('''
web_profile = json.loads((RESULTS / "web/dataset_profile.json").read_text())
scalars = {k: v for k, v in web_profile.items() if isinstance(v, (int, float, str, bool))
           and k not in ("source",)}
display(pd.DataFrame({"value": scalars}).astype(str))
splits = pd.DataFrame({key: web_profile[key] for key in
                       ("views_by_split", "eligible_pairs_by_split") if key in web_profile})
splits.index.name = "split"
splits
'''),
    md("""
## 2. E1: disk read-ahead in simulation

**Simulated.** Every trace is replayed through the causal LRU simulator by
every mode; the headline is configuration C1 (canonical merge, the assumed
5/100/50 latency model) at its two capacities. Sources:
`results/disk/table_main.csv` and `results/disk/deltas.csv`; paper
Sec. Disk Results (`sections/disk_results.tex`), Table `t2_disk_main`.

### 2.1 The main table

Mean per-trace hit ratio, its difference from `none` (percentage points,
paired bootstrap interval), and next to it the device work: `device_ios`
and `blocks_moved` as ratios to `none` (above 1 means more device work than
no prefetching). Precision, wasted share and the win count are shown
together, never a win count alone. `belady_min` is a replacement reference
without prefetch, not a prefetching bound.
"""),
    code('''
main = csv("disk/table_main.csv")
headline = main[(main["config"] == "C1") & (main["merge"] == "auto")
                & (main["latency"] == "assumed") & (main["corpus"] == "all")].copy()
headline["capacity"] = headline["capacity"].map(genlib.capacity_label)
for name in ("wins_vs_none", "losses_vs_none"):
    headline[name] = headline[name].astype("Int64")
headline["hit_ratio_%"] = 100 * headline["mean_hit_ratio"]
headline["vs_none_pp"] = 100 * headline["delta_vs_none"]
headline["vs_none_CI_pp"] = [ci(lo, hi, 100) for lo, hi in
                             zip(headline["delta_vs_none_lo"], headline["delta_vs_none_hi"])]
columns = ["capacity", "policy", "n_traces", "hit_ratio_%", "vs_none_pp", "vs_none_CI_pp",
           "device_io_ratio", "blocks_moved_ratio", "mean_precision", "mean_wasted_share",
           "wins_vs_none", "losses_vs_none"]
for capacity, table in headline.groupby("capacity", sort=False):
    print(f"C1, capacity {capacity}, all corpora (simulated)")
    display(table[columns].set_index(["capacity", "policy"]))
'''),
    md("""
### 2.2 Differences with both intervals and the pre-registered verdict

For each mode: the hit-ratio difference against `none` and against
`sequential`, each with the paired and the corpus-cluster bootstrap interval
(twins down-weighted), and the verdict of the disk rule (both intervals must
agree; `decisions.disk_verdict`). The `device_ios` and `blocks_moved` ratios
against `none`, with their paired intervals, stand next to them. Source:
`results/disk/deltas.csv`.
"""),
    code('''
deltas = csv("disk/deltas.csv")
c1 = deltas[(deltas["config"] == "C1") & (deltas["merge"] == "auto")
            & (deltas["latency"] == "assumed") & (deltas["corpus"] == "all")]


def pick(metric, baseline):
    part = c1[(c1["metric"] == metric) & (c1["baseline"] == baseline)]
    return part.set_index(["capacity", "policy"])


hit_none, hit_seq = pick("hit_ratio", "none"), pick("hit_ratio", "sequential")
ios, moved = pick("device_ios", "none"), pick("blocks_moved", "none")
rows = []
for key, h in hit_none.iterrows():
    row = {"capacity": genlib.capacity_label(key[0]), "policy": key[1], "n": h["n"],
           "hit_vs_none_pp": 100 * h["estimate"],
           "paired_CI": ci(h["ci_lo"], h["ci_hi"], 100),
           "cluster_CI": ci(h["cluster_ci_lo"], h["cluster_ci_hi"], 100),
           "verdict_vs_none": decisions.disk_verdict((h["ci_lo"], h["ci_hi"]),
                                                     (h["cluster_ci_lo"], h["cluster_ci_hi"]))}
    if key in hit_seq.index:
        s = hit_seq.loc[key]
        row["hit_vs_sequential_pp"] = 100 * s["estimate"]
        row["verdict_vs_sequential"] = decisions.disk_verdict(
            (s["ci_lo"], s["ci_hi"]), (s["cluster_ci_lo"], s["cluster_ci_hi"]))
    for name, frame in (("device_ios", ios), ("blocks_moved", moved)):
        if key in frame.index:
            r = frame.loc[key]
            row[f"{name}_ratio"] = r["estimate"]
            row[f"{name}_CI"] = ci(r["ci_lo"], r["ci_hi"], 1, 3)
    rows.append(row)
pd.DataFrame(rows).set_index(["capacity", "policy"])
'''),
    md("""
### 2.3 The pre-registered disk decisions (P1-P6, the lever ordering, C3, C4)

Computed by `paper/decisions.py` from the same files; these are the verdict
sentences the paper prints (`\\dec...` macros, Table `t9_decisions`).
"""),
    code('''
decision_frame("disk", rules={"P1", "P2", "P3", "P4", "P5", "P6", "levers", "C3", "C4"})
'''),
    md("""
### 2.4 Hit ratio and device I/O against read-ahead depth

Configuration C2 (four capacities, a depth grid, all corpora). The left
panel is the simulated hit ratio, the right panel the device-I/O ratio to
`none` at the same depth: deeper read-ahead raises the hit ratio *and* the
device work under the per-block, no-merge cost model. Source:
`results/disk/table_main.csv`; paper Fig. `f2_hit_vs_depth` (per corpus) and
`f3_device_io_vs_hit`.
"""),
    code('''
c2 = main[(main["config"] == "C2") & (main["merge"] == "auto") & (main["latency"] == "assumed")
          & (main["corpus"] == "all") & main["policy"].str.startswith("deep@")].copy()
c2["depth"] = c2["policy"].str.split("@").str[1].astype(int)
fig, (left, right) = plt.subplots(1, 2, figsize=(9, 3.6))
for capacity, part in c2.groupby("capacity"):
    part = part.sort_values("depth")
    label = genlib.capacity_label(capacity)
    left.plot(part["depth"], 100 * part["mean_hit_ratio"], marker="o", label=label)
    right.plot(part["depth"], part["device_io_ratio"], marker="o", label=label)
for ax in (left, right):
    ax.set_xscale("symlog", base=2, linthresh=1)
    ax.set_xlabel("read-ahead depth (blocks)")
left.set_ylabel("mean hit ratio (%), simulated")
right.set_ylabel("device I/Os / none, simulated")
right.axhline(1.0, color="grey", linewidth=0.8)
left.legend(title="capacity", fontsize=8)
fig.suptitle(f"E1 C2: deep@N over {int(c2['n_traces'].max())} traces (simulated)")
fig.tight_layout()
c2.pivot_table(index="depth", columns="capacity", values="mean_hit_ratio")
'''),
    md("""
## 3. E2: GPU-trained predictors on disk

**Simulated, scored only on segments the models were not trained or
selected on.** `learned_depth` is compared with the validation-chosen fixed
depth (VCF) on the test segment of each trace, averaged over seeds; the LSTM
predicts address deltas and is compared on cache metrics only (not as a
classifier). GPU timings were taken on a shared GPU (`gpu_contention`) and are
not shown as costs. Sources: `results/gpu/disk_learned_depth.csv`,
`results/gpu/disk_lstm.csv`; paper Table `t2c_learned_depth`.
"""),
    code('''
ld = csv("gpu/disk_learned_depth.csv")
ld = ld[ld["status"] == "ok"]
for cap, part in ld.groupby("action_cap"):
    per_trace = part.groupby(["corpus", "trace"]).agg(
        seeds=("seed", "nunique"), learned_depth=("test_hit_ratio", "mean"),
        vcf=("vcf_hit_ratio", "mean"), vcf_depth=("vcf_depth", "first"),
        none=("none_hit_ratio", "mean"))
    per_trace["learned_minus_vcf_pp"] = 100 * (per_trace["learned_depth"] - per_trace["vcf"])
    below = int((per_trace["learned_minus_vcf_pp"] < 0).sum())
    print(f"action cap {cap}: learned_depth below VCF on {below} of {len(per_trace)} traces "
          f"(test segment, simulated, seed means)")
    display(per_trace)
decision_frame("disk", rules={"E2"})
'''),
    code('''
lstm = csv("gpu/disk_lstm.csv")
lstm = lstm[lstm["status"] == "ok"]
cols = ["trace", "seed", "train_start", "train_end", "test_start", "test_end", "lstm_hit_ratio",
        "linux_ra_hit_ratio", "deep8_hit_ratio", "sequential_hit_ratio", "none_hit_ratio",
        "lstm_precision"]
table = lstm[cols].copy()
table["lstm_minus_linux_ra_pp"] = 100 * (table["lstm_hit_ratio"] - table["linux_ra_hit_ratio"])
print(f"LSTM below linux_ra on {int((table['lstm_minus_linux_ra_pp'] < 0).sum())} of "
      f"{len(table)} traces (test segment, simulated hit ratio)")
table.set_index("trace")
'''),
    md("""
## 4. E3: the real Linux testbed (measured)

The simulator's decisions are replayed on a real device: Backend A
(`pagecache`, `posix_fadvise(WILLNEED)` hints, the kernel's own cache) and
Backend B (`odirect`, an exact simulator-parity replay that asserts the
simulated hit/miss sequence on every request). The HDD `sdb` is the
headline device; the shared SSD rows are secondary and carry
`foreign_io_frac`. Latency figures are per trace, think model and device,
not properties of HDDs or SSDs. Sources: `results/realio/realio_summary.csv`,
`results/realio/device_*.json`; paper Sec. Real-Device Validation
(`sections/realio_results.tex`), Tables `t3_realio`, `t3b_realio_summary`,
Fig. `f5_sim_vs_measured`.
"""),
    code('''
devices = {}
for path in sorted(RESULTS.glob("realio/device_*.json")):
    record = json.loads(path.read_text())
    devices[record["device"]] = {k: record.get(k) for k in (
        "model", "rotational", "read_ahead_kb", "max_sectors_kb", "rand4k_mean_us",
        "seq4k_p50_us", "seq1m_MBps", "kappa", "foreign_io_frac")}
pd.DataFrame(devices).astype(str)
'''),
    md("""
### 4.1 HDD `sdb` (headline): simulated vs measured, per cell

Mean latency per request with its interval over repeats and the difference
from `none`. A row on a busy host is excluded by the realio rule
(`decisions._realio_usable`: load at most 32, foreign I/O at most 1%).
"""),
    code('''
summary = csv("realio/realio_summary.csv")
summary["usable"] = [decisions._realio_usable({"foreign_io_frac_max": f if pd.notna(f) else None,
                                               "load_avg_max": l if pd.notna(l) else None})
                     for f, l in zip(summary["foreign_io_frac_max"], summary["load_avg_max"])]
summary["lat_mean_CI_us"] = [ci(lo, hi, 1, 0) for lo, hi in
                             zip(summary["lat_mean_us_lo"], summary["lat_mean_us_hi"])]
summary["vs_none_CI_us"] = [ci(lo, hi, 1, 0) for lo, hi in
                            zip(summary["delta_vs_none_us_lo"], summary["delta_vs_none_us_hi"])]
show = ["backend", "think", "trace", "mode", "reps", "sim_hit_ratio", "sim_device_ios",
        "real_hit_ratio", "lat_mean_us", "lat_mean_CI_us", "delta_vs_none_us", "vs_none_CI_us",
        "load_avg_max", "usable"]
hdd = summary[summary["device"] == "sdb"].sort_values(["backend", "think", "trace", "lat_mean_us"])
hdd[show].set_index(["backend", "think", "trace", "mode"])
'''),
    md("""
### 4.2 The highest simulated hit ratio was never the fastest

For every cell (device, backend, think model, trace), the mode with the
lowest measured mean latency next to the mode with the highest *simulated*
hit ratio, and the R2 rank correlations between the simulated quantities and
measured latency (exact permutation critical values). From
`paper/decisions.py` rules `fastest` and R2.
"""),
    code('''
fast = decision_frame("realio", rules={"fastest"})
fast["highest_simulated_hit"] = fast["note"].str.replace("highest simulated hit ratio: ", "",
                                                         regex=False)
fast = fast.rename(columns={"verdict": "fastest_measured"})
fast["same_mode"] = fast["fastest_measured"] == fast["highest_simulated_hit"]
print(f"cells where the top simulated mode was also the fastest measured: "
      f"{int(fast['same_mode'].sum())} of {len(fast)}")
display(fast[["scope", "fastest_measured", "interval(s)", "highest_simulated_hit", "same_mode"]])
decision_frame("realio", rules={"R2"})
'''),
    code('''
cells = hdd[(hdd["think"] == "closed") & hdd["usable"]].groupby(["backend", "trace"])
n = len(cells)
cols_n = min(5, n)
rows_n = -(-n // cols_n)
fig, axes = plt.subplots(rows_n, cols_n, figsize=(2.2 * cols_n, 2.2 * rows_n), squeeze=False)
for ax, ((backend, trace), part) in zip(axes.flat, cells):
    ax.scatter(100 * part["sim_hit_ratio"], part["lat_mean_us"], s=14)
    for _, r in part.iterrows():
        if r["mode"] in ("none", "deep@64", "linux_ra"):
            ax.annotate(r["mode"], (100 * r["sim_hit_ratio"], r["lat_mean_us"]), fontsize=6)
    ax.set_title(f"{backend}\\n{trace}", fontsize=7)
    ax.tick_params(labelsize=6)
for ax in list(axes.flat)[n:]:
    ax.axis("off")
fig.supxlabel("simulated hit ratio (%)", fontsize=8)
fig.supylabel("measured mean latency (us), HDD sdb", fontsize=8)
fig.tight_layout()
'''),
    md("""
### 4.3 SSD rows (secondary, flagged)

The SSD carries other tenants' I/O. Every SSD row shows `foreign_io_frac_max`;
a row above the 1% limit is flagged and not used by any decision
(`contended (not used)`).
"""),
    code('''
ssd = summary[summary["device"] != "sdb"].copy()
ssd["flag"] = np.where(ssd["foreign_io_frac_max"] > decisions.FOREIGN_LIMIT,
                       "FLAGGED: foreign I/O > 1%", "")
ssd[["device", "backend", "think", "trace", "mode", "reps", "sim_hit_ratio", "lat_mean_us",
     "lat_mean_CI_us", "foreign_io_frac_max", "load_avg_max", "flag"]].set_index(
    ["device", "backend", "trace", "mode"])
'''),
    code('''
verdicts = decision_frame("realio", rules={"R1", "R6", "R3"})
print("R1/R6 verdict counts (mode - none, measured mean latency):")
display(verdicts[verdicts["rule"] != "R3"].groupby(["rule", "verdict"]).size().rename("cells")
        .to_frame())
verdicts[verdicts["rule"] == "R3"]
'''),
    md("""
## 5. E4 and E5: web prefetching in simulation

**Simulated** per-session browser-cache replay of RetailRocket split 3, every
choice frozen on split 2. Utility `U(c)` charges each issued prefetch an
assumed cost `c` in hit-equivalents; object sizes are equal and prefetches
complete instantly. Every comparison is against the best fixed depth *at the
same capacity*, against `fixed0` (no prefetch) and against transition top-1.
Sources: `results/web/comparisons.csv`, `decomposition.csv`,
`freshness.csv`, `seq_metrics.csv`; paper Sec. Web Results
(`sections/web_results.tex`), Tables `t5_web_decomposition`,
`t6_web_comparisons`, Figs. `f6_freshness`, `f8_utility_envelopes`.

### 5.1 The validated controller against the strongest fixed baseline

Per-capacity selection (W1), the logistic ranker, test split. Differences in
utility per view with session-bootstrap intervals.
"""),
    code('''
comparisons = csv("web/comparisons.csv")
headline_web = comparisons[(comparisons["split"] == "test")
                           & (comparisons["selection"] == "per_capacity")
                           & (comparisons["scorer"] == "logistic")
                           & comparisons["adaptive_policy"].str.match(r"adaptive_\\d")].copy()
for name in ("vs_best_fixed_same_cap", "vs_fixed0", "vs_transition_top1"):
    headline_web[name + "_CI"] = [ci(lo, hi, 1, 5) for lo, hi in
                                  zip(headline_web[name + "_lo"], headline_web[name + "_hi"])]
headline_web[["capacity", "cost", "adaptive_policy", "best_fixed_policy", "adaptive_utility",
              "best_fixed_utility", "vs_best_fixed_same_cap", "vs_best_fixed_same_cap_CI",
              "vs_fixed0_CI", "vs_transition_top1_CI", "adaptive_issued_per_view",
              "best_fixed_issued_per_view", "label"]].set_index(["capacity", "cost"])
'''),
    code('''
decision_frame("web", rules={"W1", "W2", "rule 1"})
'''),
    md("""
### 5.2 Which lever moves the result: the decomposition

Each lever changes one thing against a baseline (ranker, freshness of the
transition counts, depth, the controller, the feedback gate, cache scope, the
GRU sequence model). The verdict column uses the web rule's three wordings
(`decisions.web_verdict`); rule 2 orders the levers by their absolute
hit-ratio intervals and rule 3 states the feedback-gate claim.
"""),
    code('''
decomposition = csv("web/decomposition.csv")
decomposition["CI"] = [ci(lo, hi, 1, 5) for lo, hi in
                       zip(decomposition["ci_lo"], decomposition["ci_hi"])]
decomposition["verdict"] = [decisions.web_verdict(lo, hi) for lo, hi in
                            zip(decomposition["ci_lo"], decomposition["ci_hi"])]
decomposition[["lever", "policy", "scorer", "baseline", "baseline_scorer", "capacity", "metric",
               "cost", "estimate", "CI", "verdict", "sessions"]]
'''),
    code('''
decision_frame("web", rules={"rule 2", "rule 3"})
'''),
    md("""
### 5.3 Freshness: coverage of the next view by the transition index

The share of eligible next-view pairs whose target is in the top-k
transitions of the source, for a frozen index (split 0, splits 0-1 or 0-2)
and the causal online index. Source: `results/web/freshness.csv`; paper
Fig. `f6_freshness`.
"""),
    code('''
freshness = csv("web/freshness.csv")
freshness["k"] = freshness["k"].map(lambda k: "all" if pd.isna(k) else str(int(k)))
display(freshness.pivot_table(index=["split", "index"], columns="k", values="coverage",
                              sort=False)[["1", "2", "4", "8", "20", "all"]])
fig, ax = plt.subplots(figsize=(5.5, 3.4))
test = freshness[(freshness["split"] == "test") & (freshness["k"] != "all")]
for index, part in test.groupby("index", sort=False):
    ax.plot(part["k"].astype(int), 100 * part["coverage"], marker="o", label=index)
ax.set_xscale("log", base=2)
ax.set_xlabel("top-k transitions")
ax.set_ylabel("coverage of next view (%), test split")
ax.legend(title="index", fontsize=8)
fig.tight_layout()
'''),
    md("""
### 5.4 The GRU sequence model

The GRU is scored on the final protocol (split 3) for each seed; it never
ranks an item already in the window, so it is compared on the `novel_*`
columns. W8 compares it with online transition counts at the same depth.
Source: `results/web/seq_metrics.csv`.
"""),
    code('''
seq = csv("web/seq_metrics.csv")
final = seq[(seq["protocol"] == "final") & (seq["status"] == "ok")]
ranking = final.groupby("seed")[["hit_at_1", "mrr_at_8", "novel_hit_at_1",
                                 "novel_recall_at_8"]].first()
display(ranking)
cap8 = final[(final["capacity"] == 8) & (final["cost"] == 0.01)]
display(cap8.groupby("policy")[["hit_ratio", "issued_per_view", "utility"]]
        .agg(["mean", "min", "max"]).round(4))
decision_frame("web", rules={"W8"})
'''),
    md("""
## 6. E6: live HTTP replay (measured)

A real HTTP client replays split-3 sessions against an origin server over
loopback with an **emulated** network (RTT, bandwidth) and **assumed**
object sizes, from a store on the HDD or the shared SSD. A pass is valid only
if every lag p99 is at most 2 ms with no parity mismatch and no connection
error; an SSD pass above the 1% foreign-I/O limit is flagged and not cited
(`genlib.live_citable`). Sources: `results/live/live_metrics.csv`,
`live_parity.csv`; paper Sec. Live HTTP Results (`sections/live_results.tex`),
Tables `t7_live`, `t7b_live_ablation`, Fig. `f7_live_saved_vs_bytes`.
"""),
    code('''
live_rows = TREE.rows("live/live_metrics.csv")
citable_ids = {id(r) for r in genlib.live_citable(live_rows)}
live = pd.DataFrame(live_rows)
live["citable"] = [id(r) in citable_ids for r in live_rows]
print(f"passes: {len(live)}; valid: {int(live['valid'].sum())}; "
      f"citable (valid and not flagged): {int(live['citable'].sum())}")
flagged = live[~live["citable"]]
if len(flagged):
    print("not cited (flagged or invalid):")
    display(flagged[["experiment", "store", "rep", "policy", "valid", "foreign_io_frac"]])
'''),
    code('''
cited = live[live["citable"] & (live["experiment"] == "headline")].copy()
cited["saved_ms"] = -cited["delta_vs_base_ms"]
per_policy = cited.groupby(["store", "policy"]).agg(
    reps=("rep", "nunique"), hit_ratio=("hit_ratio", "mean"),
    mean_wait_ms=("mean_wait_ms", "mean"), saved_ms=("saved_ms", "mean"),
    model_ideal_saved_ms=("ideal_delta_vs_base_ms", lambda s: -s.mean()),
    bytes_vs_base=("bytes_vs_base", "mean"), kb_per_view=("kb_per_view", "mean"))
envelope = {}
for (store, policy), part in cited.groupby(["store", "policy"]):
    saved = genlib.live_saved(part.to_dict("records"))
    envelope[(store, policy)] = "--" if saved is None else ci(saved[1], saved[2], 1, 2)
per_policy["saved_CI_envelope_ms"] = pd.Series(envelope)
per_policy
'''),
    code('''
parity = csv("live/live_parity.csv")
totals = parity.groupby("experiment")[["sessions", "sim_hits", "live_hits", "sim_issued",
                                       "live_issued", "mismatched_sessions"]].sum()
totals["hits_equal"] = totals["sim_hits"] == totals["live_hits"]
totals["issued_equal"] = totals["sim_issued"] == totals["live_issued"]
display(totals)
decision_frame("live")
'''),
    md("""
## 7. Every pre-registered decision

All verdicts computed by `paper/decisions.py` (the paper's Table
`t9_decisions`, appendix "The Pre-Registered Decisions"): first the count of
each verdict per study and rule, then the full list.
"""),
    code('''
everything = decision_frame()
display(everything.groupby(["study", "rule", "verdict"]).size().rename("count").to_frame())
everything
'''),
    md("""
## 8. Classifier study (E7 and later)

These studies were added after E1-E6: the classifier zoo and router
benchmark (`results/classifier`), the hindsight router (`results/hindsight`),
the web rankers (`results/web_rankers`) and the MSRC light-sample transfer
check (`results/msrc_samples`). Each was pre-registered before it ran. The
cell below discovers whatever has landed: for each study it prints the
decision-rule section of its `PREREGISTERED.md`, lists its results files,
checks their provenance and shows each table (any `verdict` column in
full). A study whose results are not there yet prints how to produce them.
Classifier accuracy is on synthetic windows only; real traces have no
workload labels.
"""),
    code('''
import re

LATER_STUDIES = (("classifier", "classifier", "classifier zoo and router benchmark"),
                 ("hindsight", "hindsight", "hindsight router"),
                 ("web_rankers", "web_rankers", "web rankers"),
                 ("msrc_samples", "msrc", "MSRC light-sample transfer check"))


def rule_sections(text):
    """The '## ...' sections of a PREREGISTERED.md that state decision rules."""
    parts = re.split(r"(?m)^(?=## )", text)
    keep = [p for p in parts if p.startswith("## ")
            and re.search(r"decision|comparison|rule", p.splitlines()[0], re.I)]
    return "\\n".join(keep).strip()


for directory, study, title in LATER_STUDIES:
    root = RESULTS / directory
    print("=" * 78)
    print(f"{title}: results/{directory}")
    if not root.is_dir():
        print(f"not yet generated: run reproduce.py {study}")
        continue
    prereg = root / "PREREGISTERED.md"
    if prereg.is_file():
        text = prereg.read_text(encoding="utf-8")
        print(text.splitlines()[0])
        rules = rule_sections(text)
        if rules:
            print(rules)
    tables = sorted(p for p in root.rglob("*") if p.suffix in (".csv", ".json")
                    and p.name != "provenance.json")
    if not tables:
        print(f"not yet generated: run reproduce.py {study}")
        continue
    if (root / "provenance.json").is_file():
        problems = provenance.verify_provenance(root)
        print("provenance: " + ("all files match" if not problems else "; ".join(problems)))
    else:
        print("provenance: no provenance.json (not a complete reproduce run)")
    for path in tables:
        print(f"-- {path.relative_to(RESULTS)}")
        if path.suffix == ".csv":
            frame = pd.read_csv(path)
            verdict_columns = [c for c in frame.columns if "verdict" in c.lower()]
            if verdict_columns:
                display(frame)
            else:
                print(f"{len(frame)} rows; the first 25:")
                display(frame.head(25))
        else:
            payload = json.loads(path.read_text(encoding="utf-8"))
            print(json.dumps(payload, indent=1, sort_keys=True)[:3000])
'''),
    md("""
## Where to go next

- `paper/main.tex` and `paper/sections/*.tex`: the paper, written from these
  verdicts; `paper/numbers.tex` holds every number it quotes.
- `docs/REPRODUCE.md`: the command that produces each results file, with its
  runtime; `docs/METRICS.md`: what each column means and its denominator.
- `results/<study>/PREREGISTERED.md`: the decision rules applied above.
"""),
]


# ---------------------------------------------------------------------------
# Execution (no kernel needed)
# ---------------------------------------------------------------------------

def _text_lines(text: str) -> list[str]:
    return text.splitlines(keepends=True)


def _figure_outputs(nbformat, plt) -> list:
    outputs = []
    for number in plt.get_fignums():
        figure = plt.figure(number)
        buffer = io.BytesIO()
        figure.savefig(buffer, format="png", dpi=FIGURE_DPI, bbox_inches="tight",
                       metadata={"Software": None})
        data = base64.b64encode(buffer.getvalue()).decode("ascii")
        outputs.append(nbformat.v4.new_output(
            "display_data", data={"image/png": data,
                                  "text/plain": f"<Figure {number}>"}))
    plt.close("all")
    return outputs


def _rich(nbformat, value, kind: str, count: int | None = None):
    data = {"text/plain": repr(value)}
    html = getattr(value, "_repr_html_", None)
    if callable(html):
        rendered = html()
        if rendered:
            data["text/html"] = rendered
    if kind == "execute_result":
        return nbformat.v4.new_output("execute_result", data=data, execution_count=count)
    return nbformat.v4.new_output("display_data", data=data)


class CellError(RuntimeError):
    pass


def execute(cells, nbformat) -> list:
    """Run the code cells in one namespace; return nbformat cells with outputs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import pandas as pd

    namespace: dict = {"__name__": "__main__"}
    built = []
    count = 0
    previous = Path.cwd()
    # The cells set matplotlib rcParams and pandas display options; restore
    # both afterwards so a build inside another process (the test suite)
    # leaves no global state behind.
    pandas_options = {key: pd.get_option(key) for key in PANDAS_OPTIONS}
    rc = matplotlib.rc_context()
    rc.__enter__()
    os.chdir(REPO)
    try:
        for index, (kind, source) in enumerate(cells):
            cell_id = f"cell-{index:03d}"
            if kind == "markdown":
                cell = nbformat.v4.new_markdown_cell(source)
                cell["id"] = cell_id
                built.append(cell)
                continue
            count += 1
            outputs: list = []
            stdout = io.StringIO()

            def flush():
                text = stdout.getvalue()
                if text:
                    outputs.append(nbformat.v4.new_output("stream", name="stdout",
                                                          text=text))
                    stdout.seek(0)
                    stdout.truncate()

            def display(*values):
                flush()
                outputs.extend(_figure_outputs(nbformat, plt))
                for value in values:
                    outputs.append(_rich(nbformat, value, "display_data"))

            namespace["display"] = display
            tree = ast.parse(source)
            last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
            result = None
            try:
                with contextlib.redirect_stdout(stdout), warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    exec(compile(tree, f"<cell {count}>", "exec"), namespace)
                    if last is not None:
                        result = eval(compile(ast.Expression(last.value), f"<cell {count}>",
                                              "eval"), namespace)
            except Exception as error:
                raise CellError(f"cell {count} failed:\n{traceback.format_exc()}") from error
            flush()
            outputs.extend(_figure_outputs(nbformat, plt))
            if result is not None:
                outputs.append(_rich(nbformat, result, "execute_result", count))
            cell = nbformat.v4.new_code_cell(source, execution_count=count, outputs=outputs)
            cell["id"] = cell_id
            built.append(cell)
    finally:
        os.chdir(previous)
        plt.close("all")
        rc.__exit__(None, None, None)
        for key, value in pandas_options.items():
            pd.set_option(key, value)
    return built


def build(out: Path = DEFAULT_OUT, *, run: bool = True) -> Path:
    import nbformat

    if str(REPO / "src") not in sys.path:
        sys.path.insert(0, str(REPO / "src"))
    if run:
        cells = execute(CELLS, nbformat)
    else:
        cells = []
        for index, (kind, source) in enumerate(CELLS):
            cell = (nbformat.v4.new_markdown_cell(source) if kind == "markdown"
                    else nbformat.v4.new_code_cell(source))
            cell["id"] = f"cell-{index:03d}"
            cells.append(cell)
    notebook = nbformat.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}})
    notebook["nbformat"], notebook["nbformat_minor"] = 4, 5
    nbformat.validate(notebook)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(nbformat.writes(notebook) + "\n", encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--no-execute", action="store_true",
                        help="write the cells without outputs")
    args = parser.parse_args(argv)
    path = build(args.out, run=not args.no_execute)
    print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
