"""Write the paper's figures, ``paper/figures/*.pdf``, from ``results/**``.

    PYTHONPATH=src python paper/figures/make_figures.py [--results DIR] [--paper-dir DIR]

One function per figure (MASTER_PLAN section 5, F1-F8). matplotlib with the
Agg backend writes vector PDFs at IEEE widths: one column is 3.5 in, the
full text width 7.16 in (``WIDTHS`` records which a figure uses, so the
section knows whether it needs ``figure`` or ``figure*``). Fonts are embedded
as TrueType (``pdf.fonttype 42``), as IEEE PDF checks require.

Colour follows the job it does. Identity (policy, device, index) takes the
categorical slots in fixed order; magnitude (cache size) takes one hue, light
to dark; every series also differs by marker or line style, so no figure
depends on colour alone, in print or under colour-vision deficiency.

A figure whose input is missing is still written, at its size, stamped
"MISSING: <file>". Built from a fixture tree (``FIXTURE`` marker) every
figure carries a "FIXTURE DATA" watermark. The PDFs carry no creation date,
so regenerating from unchanged results gives identical bytes.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator  # noqa: E402

HERE = Path(__file__).resolve().parent


def _genlib():
    """``paper/genlib.py``, loaded by path: putting ``paper/`` on sys.path
    would let ``paper/numbers.py`` shadow the stdlib ``numbers`` module."""
    if "paper_genlib" not in sys.modules:
        spec = importlib.util.spec_from_file_location("paper_genlib", HERE.parent / "genlib.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules["paper_genlib"] = module
        spec.loader.exec_module(module)
    return sys.modules["paper_genlib"]


g = _genlib()
from prefetch_common import utility  # noqa: E402

COLUMN_IN = 3.5   # IEEEtran \columnwidth
TEXT_IN = 7.16    # IEEEtran \textwidth (figure*)

# Categorical slots in their validated order (blue, orange, aqua, yellow,
# magenta, green, violet, red); scatter-type figures use at most three.
CATEGORICAL = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
               "#6250d6", "#e34948")
# One hue, light to dark, for an ordinal magnitude (cache size); the lightest
# step still clears 2:1 on white.
SEQUENTIAL = ("#86b6ef", "#3987e5", "#1c5cab", "#0d366b")
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")
LINESTYLES = ("-", "--", "-.", ":")
INK, INK_2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

RC = {
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "mathtext.fontset": "dejavusans", "font.size": 7, "axes.labelsize": 7,
    "axes.titlesize": 7, "legend.fontsize": 6, "xtick.labelsize": 6, "ytick.labelsize": 6,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK, "axes.linewidth": 0.6,
    "xtick.color": INK_2, "ytick.color": INK_2, "xtick.major.width": 0.5,
    "ytick.major.width": 0.5, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.4,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.2,
    "lines.markersize": 3.5, "legend.frameon": False, "pdf.fonttype": 42,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "text.color": INK, "axes.axisbelow": True,
}
METADATA = {"Creator": "paper/figures/make_figures.py", "CreationDate": None}

#: Figure name -> (width in, height in). Widths are COLUMN_IN or TEXT_IN.
SIZES = {
    "f1_schematic": (COLUMN_IN, 1.9),
    "f2_hit_vs_depth": (COLUMN_IN, 2.9),
    "f3_device_io_vs_hit": (COLUMN_IN, 2.5),
    "f4_kappa_cdf": (COLUMN_IN, 2.3),
    "f5_sim_vs_measured": (TEXT_IN, 2.4),
    "f6_freshness": (COLUMN_IN, 2.2),
    "f7_live_saved_vs_bytes": (COLUMN_IN, 2.4),
    "f8_utility_envelopes": (TEXT_IN, 2.5),
}
WIDTHS = {name: ("column" if size[0] == COLUMN_IN else "text") for name, size in SIZES.items()}


class Missing(Exception):
    """Raised by a figure function when a required input yields nothing to draw."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _figure(name, nrows=1, ncols=1, **kwargs):
    fig, axes = plt.subplots(nrows, ncols, figsize=SIZES[name], layout="constrained", **kwargs)
    return fig, axes


def _watermark(fig) -> None:
    fig.text(0.5, 0.5, g.FIXTURE_TEXT, fontsize=22, color="#e34948", alpha=0.18,
             ha="center", va="center", rotation=25, weight="bold", zorder=100)


def _save(fig, path: Path, out) -> None:
    if out.fixture:
        _watermark(fig)
    fig.savefig(path, format="pdf", metadata=METADATA)
    plt.close(fig)


def placeholder(name: str, path: Path, out) -> None:
    fig = plt.figure(figsize=SIZES[name])
    missing = sorted(set(out.missing)) or ["input"]
    fig.text(0.5, 0.62, f"{g.MISSING_TEXT}:", ha="center", va="center", fontsize=10,
             weight="bold", color=INK)
    fig.text(0.5, 0.42, "\n".join(missing), ha="center", va="center", fontsize=7, color=INK_2)
    fig.text(0.5, 0.12, f"{name} (paper/figures/make_figures.py)", ha="center", fontsize=6,
             color=MUTED)
    _save(fig, path, out)


def _label(row) -> str:
    """Sweep row -> table policy label (deep@8, linux_ra@256)."""
    if row["mode"] == "deep":
        return f"deep@{row['depth']}"
    if row["mode"] == "linux_ra":
        return f"linux_ra@{row['max_blocks']}"
    return row["mode"]


def _headline(rows, **criteria):
    return g.select(rows, config=g.HEADLINE_CONFIG, merge=g.HEADLINE_MERGE, **criteria)


def _plain_ticks(axis, values) -> None:
    """Ticks exactly at ``values``, labelled as plain numbers (no 2^k, no 10^k)."""
    axis.set_major_locator(FixedLocator(sorted(values)))
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_locator(NullLocator())


def _depth_of(policy: str):
    base, _, parameter = policy.partition("@")
    if base == "deep" and parameter.isdigit():
        return int(parameter)
    return None


# ---------------------------------------------------------------------------
# F1: the shared abstraction
# ---------------------------------------------------------------------------

def f1_schematic(results, out, path):
    fig = plt.figure(figsize=SIZES["f1_schematic"])
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.4)
    ax.axis("off")

    def box(x, y, w, h, title, body, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.15",
                                    linewidth=0.8, edgecolor=color, facecolor="white"))
        ax.text(x + w / 2, y + h - 0.32, title, ha="center", va="top", fontsize=6.3,
                weight="bold", color=INK)
        ax.text(x + w / 2, y + h - 0.85, body, ha="center", va="top", fontsize=5.6,
                color=INK_2, linespacing=1.25)

    def arrow(a, b, text=None, offset=(0, 0.18), style="-|>", color=INK_2, rad=0.0):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=7, linewidth=0.8,
                                     color=color, connectionstyle=f"arc3,rad={rad}"))
        if text:
            ax.text((a[0] + b[0]) / 2 + offset[0], (a[1] + b[1]) / 2 + offset[1], text,
                    ha="center", va="bottom", fontsize=5.6, color=INK_2)

    box(0.1, 2.9, 2.35, 2.2, "Demand stream", "disk: 512-B blocks\nof each request\n"
        "web: product views\nof each session", CATEGORICAL[0])
    box(3.825, 2.9, 2.35, 2.2, "Cache", "LRU, capacity C\nhits and misses;\nprefetched entries\n"
        "marked speculative", INK_2)
    box(7.55, 2.9, 2.35, 2.2, "Outcome", "hits, issued,\nuseful, wasted", CATEGORICAL[2])
    ax.text(8.725, 3.35, r"$U(c)=\frac{\mathrm{hits}-c\,\mathrm{issued}}{\mathrm{requests}}$",
            ha="center", va="center", fontsize=6.4, color=INK)
    box(3.825, 0.15, 2.35, 2.1, "Prefetch policy", "how much: depth,\nwindow growth\n"
        "which: predictor\nwhen: feedback gate", CATEGORICAL[1])
    arrow((2.45, 4.0), (3.825, 4.0), "demand")
    arrow((6.175, 4.0), (7.55, 4.0), "count")
    arrow((5.0, 2.25), (5.0, 2.9), "prefetch", offset=(0.55, -0.12))
    arrow((1.275, 2.9), (3.825, 1.2), "history", offset=(-0.8, -0.1), rad=0.25)
    arrow((8.725, 2.9), (6.175, 1.2), "observed precision vs c", offset=(0.9, -0.1), rad=-0.25,
          color=CATEGORICAL[1])
    _save(fig, path, out)


# ---------------------------------------------------------------------------
# F2: hit ratio vs depth per cache size and corpus
# ---------------------------------------------------------------------------

def f2_hit_vs_depth(results, out, path):
    rows = _headline_merge(results.rows("disk/table_by_corpus.csv", out))
    if rows is None:
        raise Missing
    depth_rows = [r for r in rows if r["latency"] == g.HEADLINE_LATENCY
                  and r["mean_hit_ratio"] is not None
                  and (_depth_of(r["policy"]) is not None or r["policy"] == "none")]
    by_config = defaultdict(set)
    for row in depth_rows:
        if _depth_of(row["policy"]) is not None:
            by_config[row["config"]].add(_depth_of(row["policy"]))
    if not by_config:
        out.missing.append("disk/table_by_corpus.csv: no deep@<depth> rows")
        raise Missing
    config = max(sorted(by_config), key=lambda c: len(by_config[c]))
    depth_rows = g.select(depth_rows, config=config)
    corpora = sorted({r["corpus"] for r in depth_rows})
    capacities = sorted({r["capacity"] for r in depth_rows})[:len(SEQUENTIAL)]
    ncols = 2 if len(corpora) > 1 else 1
    nrows = math.ceil(len(corpora) / ncols)
    fig, axes = _figure("f2_hit_vs_depth", nrows, ncols, sharex=True, squeeze=False)
    out.note = f"config {config}, merge {g.HEADLINE_MERGE}, assumed latency"
    shade = SEQUENTIAL if len(capacities) > 2 else (SEQUENTIAL[1], SEQUENTIAL[3])
    for index, corpus in enumerate(corpora):
        ax = axes[index // ncols][index % ncols]
        for k, capacity in enumerate(capacities):
            points = {}
            for row in g.select(depth_rows, corpus=corpus, capacity=capacity):
                depth = _depth_of(row["policy"])
                depth = 0 if depth is None else depth
                if row["policy"] == "none" and 0 in points:
                    continue  # an explicit deep@0 wins over none
                points[depth] = 100 * row["mean_hit_ratio"]
            if not points:
                continue
            xs = sorted(points)
            ax.plot(xs, [points[x] for x in xs], color=shade[k], marker=MARKERS[k],
                    linestyle=LINESTYLES[k % len(LINESTYLES)], label=g.capacity_label(capacity))
        n = max((r["n_traces"] for r in g.select(depth_rows, corpus=corpus)), default=0)
        ax.set_title(f"{corpus} (n = {n})", loc="left", color=INK)
        ax.set_xscale("symlog", base=2, linthresh=1)
        ax.set_xlim(left=0)
        depths = sorted({_depth_of(r["policy"]) or 0 for r in depth_rows})
        _plain_ticks(ax.xaxis, depths[::2] if len(depths) > 6 else depths)
        if index % ncols == 0:
            ax.set_ylabel("mean hit ratio (%)")
        if index // ncols == nrows - 1:
            ax.set_xlabel("read-ahead depth (blocks)")
    for index in range(len(corpora), nrows * ncols):
        axes[index // ncols][index % ncols].axis("off")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncols=len(labels),
               title="cache size", title_fontsize=6)
    _save(fig, path, out)


def _headline_merge(rows):
    if rows is None:
        return None
    return g.select(rows, merge=g.HEADLINE_MERGE)


# ---------------------------------------------------------------------------
# F3: device I/O ratio vs hit gain
# ---------------------------------------------------------------------------

def f3_device_io_vs_hit(results, out, path):
    rows = results.rows("disk/deltas.csv", out)
    if rows is None:
        raise Missing
    rows = _headline(rows, corpus="all", latency=g.HEADLINE_LATENCY, baseline="none")
    # setdefault: the first row wins when a comparison appears per twin treatment.
    hit, ios = {}, {}
    for row in rows:
        if (row["metric"], row["statistic"]) == ("hit_ratio", "difference"):
            hit.setdefault((row["policy"], row["capacity"]), row)
        elif (row["metric"], row["statistic"]) == ("device_ios", "ratio"):
            ios.setdefault((row["policy"], row["capacity"]), row)
    keys = sorted(set(hit) & set(ios), key=lambda k: (k[1], g.policy_key(k[0])))
    if not keys:
        out.missing.append("disk/deltas.csv: no C1 hit_ratio and device_ios rows vs none")
        raise Missing
    fig, ax = _figure("f3_device_io_vs_hit")
    capacities = sorted({k[1] for k in keys})[:3]
    for k, capacity in enumerate(capacities):
        items = [key for key in keys if key[1] == capacity]
        xs = [100 * hit[key]["estimate"] for key in items]
        ys = [ios[key]["estimate"] for key in items]
        xerr = [[100 * (hit[key]["estimate"] - hit[key]["ci_lo"]) for key in items],
                [100 * (hit[key]["ci_hi"] - hit[key]["estimate"]) for key in items]]
        yerr = [[ios[key]["estimate"] - ios[key]["ci_lo"] for key in items],
                [ios[key]["ci_hi"] - ios[key]["estimate"] for key in items]]
        ax.errorbar(xs, ys, xerr=xerr, yerr=yerr, fmt=MARKERS[k], color=CATEGORICAL[k],
                    markersize=4, markeredgecolor="white", markeredgewidth=0.5, elinewidth=0.6,
                    capsize=1.2, label=f"{g.capacity_label(capacity)} cache")
        if k == 0:  # one label per policy; the legend tells the capacities apart
            # Policies within F3_CLUSTER_PP of no gain overlap at this scale,
            # so they share one label; the rest are named where they sit.
            ranked = sorted(zip(xs, ys, items))
            crowd = [key[0] for x, y, key in ranked if abs(x) < F3_CLUSTER_PP]
            spread = [(x, y, key) for x, y, key in ranked if abs(x) >= F3_CLUSTER_PP]
            for rank, (x, y, key) in enumerate(spread):
                ax.annotate(key[0], (x, y), xytext=(3, 3 if rank % 2 == 0 else -7),
                            textcoords="offset points", fontsize=5, color=INK_2)
            if crowd:
                cx = max(x for x, _, key in ranked if key[0] in crowd)
                cy = max(y for _, y, key in ranked if key[0] in crowd)
                ax.annotate("\n".join(_wrap(crowd, 3)), (cx, cy), xytext=(6, 4),
                            textcoords="offset points", fontsize=4.6, color=INK_2,
                            va="bottom")
    ax.axhline(1.0, color=AXIS, linewidth=0.6, zorder=0)
    ax.axvline(0.0, color=AXIS, linewidth=0.6, zorder=0)
    ax.set_yscale("log")
    low, high = ax.get_ylim()
    ticks = [v for v in (0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5, 10, 20, 50) if low <= v <= high]
    _plain_ticks(ax.yaxis, ticks or [1])
    ax.set_xlabel("hit-ratio gain over none (pp)")
    ax.set_ylabel("device I/Os relative to none")
    ax.legend(loc="best")
    _save(fig, path, out)


#: F3: policies whose hit gain is smaller than this many pp share one label.
F3_CLUSTER_PP = 4.0


def _wrap(names, per_line):
    return [", ".join(names[i:i + per_line]) for i in range(0, len(names), per_line)]


# ---------------------------------------------------------------------------
# F4: CDF of kappa*
# ---------------------------------------------------------------------------

F4_POLICIES = ("deep@8", "linux_ra@256", "sequential")


def f4_kappa_cdf(results, out, path):
    rows = results.rows("disk/kappa.csv", out)
    devices = results.glob("realio/device_*.json", out, required=False)
    if rows is None:
        raise Missing
    rows = _headline(rows, capacity=min((r["capacity"] for r in rows), default=None))
    present = sorted({r["policy"] for r in rows}, key=g.policy_key)
    policies = [p for p in F4_POLICIES if p in present] or present[:3]
    if not policies:
        out.missing.append("disk/kappa.csv: no C1 rows")
        raise Missing
    finite = [r["kappa_star"] for r in rows if r["kappa_star"] is not None
              and math.isfinite(r["kappa_star"]) and r["kappa_star"] > 0]
    kappas = [d["kappa"] for d in devices.values() if math.isfinite(d["kappa"]) and d["kappa"] > 0]
    span = finite + kappas or [1.0, 100.0]
    low, high = 10 ** math.floor(math.log10(min(span)) - 0.3), 10 ** math.ceil(
        math.log10(max(span)) + 0.3)
    fig, ax = _figure("f4_kappa_cdf")
    for k, policy in enumerate(policies):
        values = [r["kappa_star"] for r in rows if r["policy"] == policy
                  and r["kappa_star"] is not None]
        n = len(values)
        if not n:
            continue
        infinite = sum(math.isinf(v) for v in values)
        # kappa* = 0 means "wins at every ratio": drawn at the left edge.
        points = sorted(max(v, low) for v in values if not math.isinf(v))
        xs, ys = [low], [0.0]
        for i, v in enumerate(points, start=1):
            xs += [v, v]
            ys += [ys[-1], i / n]
        xs.append(high)
        ys.append(ys[-1])
        ax.plot(xs, ys, color=CATEGORICAL[k], linestyle=LINESTYLES[k],
                label=f"{policy} (n = {n}; {infinite} never)")
    for k, (relative, device) in enumerate(sorted(devices.items())):
        if not (math.isfinite(device["kappa"]) and device["kappa"] > 0):
            continue
        ax.axvline(device["kappa"], color=INK_2, linestyle=":", linewidth=0.8)
        ax.text(device["kappa"], 1.02, f" {device['device']} $\\kappa$={device['kappa']:.0f}",
                fontsize=5.5, color=INK_2, ha="left", va="bottom", rotation=0)
    ax.set_xscale("log")
    ax.set_xlim(low, high)
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    capacity = rows[0]["capacity"] if rows else None
    ax.set_xlabel(r"$\kappa^*$ = $t_{pos}/t_{xfer}$ above which the policy wins (blocks)")
    ax.set_ylabel("fraction of traces")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.08), title=(f"{g.capacity_label(capacity)} cache; never = "
                                         r"$\kappa^*=\infty$") if capacity else None,
              title_fontsize=5.5)
    _save(fig, path, out)


# ---------------------------------------------------------------------------
# F5: simulated hit ratio vs measured latency
# ---------------------------------------------------------------------------

def f5_sim_vs_measured(results, out, path):
    rows = results.rows("realio/realio_summary.csv", out)
    if rows is None:
        raise Missing
    rows = [r for r in rows if r["sim_hit_ratio"] is not None]
    if not rows:
        out.missing.append("realio/realio_summary.csv: no rows with a simulator plan")
        raise Missing
    thinks = g.distinct(rows, "think")
    devices = g.distinct(rows, "device")[:3]
    backends = g.distinct(rows, "backend")
    fig, axes = _figure("f5_sim_vs_measured", 1, len(thinks), sharey=True, squeeze=False)
    out.note = "kernel_ra/kernel_seq rows have no simulator plan and are not drawn"
    for index, think in enumerate(thinks):
        ax = axes[0][index]
        for d, device in enumerate(devices):
            for b, backend in enumerate(backends):
                items = g.select(rows, think=think, device=device, backend=backend)
                if not items:
                    continue
                ax.errorbar([100 * r["sim_hit_ratio"] for r in items],
                            [r["lat_mean_us"] for r in items],
                            yerr=[[r["lat_mean_us"] - r["lat_mean_us_lo"] for r in items],
                                  [r["lat_mean_us_hi"] - r["lat_mean_us"] for r in items]],
                            fmt=MARKERS[b], color=CATEGORICAL[d], markersize=3.5,
                            markerfacecolor=CATEGORICAL[d] if b == 0 else "white",
                            elinewidth=0.5, capsize=1, label=f"{device}, {backend}")
        ax.set_title(f"think: {think}", loc="left")
        ax.set_yscale("log")
        ax.set_xlabel("simulated hit ratio (%)")
        if index == 0:
            ax.set_ylabel(r"measured mean latency ($\mu$s)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside right center")
    _save(fig, path, out)


# ---------------------------------------------------------------------------
# F6: coverage@K vs index freshness
# ---------------------------------------------------------------------------

def f6_freshness(results, out, path):
    rows = results.rows("web/freshness.csv", out)
    if rows is None:
        raise Missing
    splits = g.distinct(rows, "split")
    split = "test" if "test" in splits else splits[0]
    rows = [r for r in g.select(rows, split=split) if r["k"] is not None]
    if not rows:
        out.missing.append("web/freshness.csv: no finite-k rows")
        raise Missing
    fig, ax = _figure("f6_freshness")
    for k, index in enumerate(g.distinct(rows, "index")[:len(CATEGORICAL)]):
        items = sorted(g.select(rows, index=index), key=lambda r: r["k"])
        ax.plot([r["k"] for r in items], [100 * r["coverage"] for r in items],
                color=CATEGORICAL[k], marker=MARKERS[k], linestyle=LINESTYLES[k % 4], label=index)
    ax.set_xscale("log", base=2)
    _plain_ticks(ax.xaxis, {r["k"] for r in rows})
    ax.set_xlabel("candidates K")
    ax.set_ylabel(f"next item in top K (%, {split})")
    ax.legend(title="index", title_fontsize=6, loc="lower right")
    _save(fig, path, out)


# ---------------------------------------------------------------------------
# F7: live latency saved vs bytes wasted
# ---------------------------------------------------------------------------

def f7_live_saved_vs_bytes(results, out, path):
    rows = results.rows("live/live_metrics.csv", out)
    if rows is None:
        raise Missing
    groups = defaultdict(list)
    for row in g.live_citable(rows):  # valid, and store foreign I/O at most 1 %
        if row["experiment"] == "headline":
            groups[(row["store"], row["policy"])].append(row)
    if not any(g.live_saved(items) is not None for items in groups.values()):
        out.missing.append("live/live_metrics.csv: no valid headline rows")
        raise Missing
    fig, ax = _figure("f7_live_saved_vs_bytes")
    out.note = ("points: mean over citable passes (valid, foreign I/O at most 1%); bars: "
                "envelope of the per-pass 95% CIs")
    for k, store in enumerate(sorted({key[0] for key in groups})[:3]):
        keys = sorted((key for key in groups if key[0] == store), key=lambda key: g.policy_key(key[1]))
        keys = [key for key in keys if g.live_saved(groups[key]) is not None]
        xs, ys, lo, hi = [], [], [], []
        for key in keys:
            items = g.live_comparable(groups[key])
            saved, low, high = g.live_saved(items)
            xs.append(statistics.fmean(r["wasted_kb_per_view"] for r in items))
            ys.append(saved)
            lo.append(saved - low)
            hi.append(high - saved)
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt=MARKERS[k], color=CATEGORICAL[k], markersize=4,
                    markeredgecolor="white", markeredgewidth=0.5, elinewidth=0.6, capsize=1.2,
                    label=f"{store} store")
        if k == 0:  # one label per policy; the legend tells the stores apart
            for key, x, y in zip(keys, xs, ys):
                ax.annotate(key[1], (x, y), xytext=(4, 3), textcoords="offset points",
                            fontsize=5, color=INK_2)
    ax.axhline(0.0, color=AXIS, linewidth=0.6, zorder=0)
    ax.set_xlabel("wasted KiB per view")
    ax.set_ylabel("wait saved per view (ms)")
    ax.legend(loc="lower right")
    _save(fig, path, out)


# ---------------------------------------------------------------------------
# F8: U(c) envelopes, disk and web
# ---------------------------------------------------------------------------

C_GRID = tuple(10 ** (-3 + i * (math.log10(1.1) + 3) / 120) for i in range(121))


def _disk_utility_rows(sweep, capacity):
    return g.disk_utility_rows(sweep, capacity)


def _envelope_panel(ax, rows, title, marks, colors):
    curves = {r["policy"]: [utility.row_U(r, c) for c in C_GRID] for r in rows}
    for policy, values in curves.items():
        ax.plot(C_GRID, values, color=GRID, linewidth=0.7, zorder=1)
    envelope = utility.envelope(rows, C_GRID)
    # Segments of the upper envelope, coloured by the winning policy.
    segments = []
    for point in envelope:
        if segments and segments[-1][0] == point["policy"]:
            segments[-1][1].append(point)
        else:
            segments.append((point["policy"], [point]))
    for i, (policy, points) in enumerate(segments):
        if i + 1 < len(segments):
            points = points + [segments[i + 1][1][0]]
        if policy not in colors:  # fixed slot order, by first appearance
            colors[policy] = (CATEGORICAL[len(colors) % len(CATEGORICAL)],
                              LINESTYLES[len(colors) % len(LINESTYLES)])
        color, style = colors[policy]
        ax.plot([p["c"] for p in points], [p["utility"] for p in points], color=color,
                linewidth=1.8, linestyle=style, zorder=3,
                label=policy if policy not in ax.get_legend_handles_labels()[1] else None)
    # Frame the envelope: losing policies may fall far below it at high c.
    values = [p["utility"] for p in envelope] + [v for v in curves.get("none", [])]
    low, high = min(values), max(values)
    pad = 0.15 * (high - low) + 0.01
    ax.set_ylim(low - 2 * pad, high + pad)
    for c, text, style in marks:
        ax.axvline(c, color=INK_2, linestyle=style, linewidth=0.7, zorder=2)
        ax.text(c, 1.0, text, transform=ax.get_xaxis_transform(), fontsize=5.2, color=INK_2,
                ha="right", va="top", rotation=90)
    ax.set_xscale("log")
    ax.set_xlim(C_GRID[0], C_GRID[-1])
    ax.set_xlabel("cost per prefetch c (hits)")
    ax.set_title(title, loc="left")
    entries = len(ax.get_legend_handles_labels()[1])
    ax.legend(loc="lower left", title="best policy", title_fontsize=6,
              ncols=2 if entries > 4 else 1, fontsize=5 if entries > 4 else 6)


def f8_utility_envelopes(results, out, path):
    sweep = results.rows("disk/sweep.jsonl", out)
    metrics = results.rows("web/policy_metrics.csv", out)
    comparisons = results.rows("web/comparisons.csv", out, required=False)
    if sweep is None and metrics is None:
        raise Missing
    fig, axes = _figure("f8_utility_envelopes", 1, 2, squeeze=False)
    notes = []
    disk_ax, web_ax = axes[0]
    if sweep is not None:
        capacities = sorted({r["capacity"] for r in sweep if r["config"] == g.HEADLINE_CONFIG})
        capacity = g.HEADLINE_CAPACITIES[0] if g.HEADLINE_CAPACITIES[0] in capacities else (
            capacities[0] if capacities else None)
        rows, n = _disk_utility_rows(sweep, capacity) if capacity else ([], 0)
    else:
        rows, n = [], 0
    if rows:
        _envelope_panel(disk_ax, rows, f"disk (simulated), {g.capacity_label(capacity)}, n = {n} traces",
                        [(utility.BREAK_EVEN_C, "break-even model ", "--"),
                         (utility.cost_to_c(5, 100, 50), r"assumed 5/100/50 $\mu$s ", ":")], {})
        disk_ax.set_ylabel("U(c) per read block")
        notes.append(f"disk: C1 {g.HEADLINE_MERGE}, mean over {n} paired traces")
    else:
        _missing_panel(disk_ax, "disk/sweep.jsonl")
    web_rows = []
    if metrics is not None:
        test = g.select(metrics, split="test", cache_scope="session")
        scorers = g.distinct(test, "scorer")
        scorer = "logistic" if "logistic" in scorers else (scorers[0] if scorers else None)
        capacities = sorted({r["capacity"] for r in g.select(test, scorer=scorer)})
        capacity_w = 8 if 8 in capacities else (capacities[len(capacities) // 2]
                                                if capacities else None)
        web_rows = g.select(test, scorer=scorer, capacity=capacity_w)
    if web_rows:
        costs = sorted({r["cost"] for r in comparisons or () if r["cost"] > 0})
        marks = [(c, f"assumed c={c:g} ", ":") for c in costs]
        _envelope_panel(web_ax, web_rows, f"web (simulated), {scorer}, capacity {capacity_w}",
                        marks, {})
        web_ax.set_ylabel("U(c) per view")
        notes.append(f"web: test split, scorer {scorer}, capacity {capacity_w}; no measured "
                     "web c in the results schema yet")
    else:
        _missing_panel(web_ax, "web/policy_metrics.csv")
    out.note = "; ".join(notes)
    _save(fig, path, out)


def _missing_panel(ax, name):
    ax.axis("off")
    ax.text(0.5, 0.5, f"{g.MISSING_TEXT}:\n{name}", ha="center", va="center", fontsize=8,
            weight="bold", transform=ax.transAxes)


FIGURES = (f1_schematic, f2_hit_vs_depth, f3_device_io_vs_hit, f4_kappa_cdf,
           f5_sim_vs_measured, f6_freshness, f7_live_saved_vs_bytes, f8_utility_envelopes)


def generate(results, paper_dir: Path = g.PAPER_DIR) -> list:
    directory = paper_dir / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    outputs = []
    with plt.rc_context(RC):
        for function in FIGURES:
            name = function.__name__
            out = results.output(f"figures/{name}.pdf")
            path = directory / f"{name}.pdf"
            try:
                function(results, out, path)
            except Missing:
                plt.close("all")
                placeholder(name, path, out)
            out.note = "; ".join(filter(None, [out.note, f"width: {WIDTHS[name]}"]))
            outputs.append(out)
    g.write_manifest(directory, outputs, results)
    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results", type=Path, default=g.DEFAULT_RESULTS)
    parser.add_argument("--paper-dir", type=Path, default=g.PAPER_DIR)
    args = parser.parse_args(argv)
    outputs = generate(g.Results(args.results), args.paper_dir)
    gaps = [o.name for o in outputs if o.missing]
    print(f"make_figures.py: {len(outputs)} figures; {len(gaps)} with missing input")
    return 0


if __name__ == "__main__":
    sys.exit(main())
