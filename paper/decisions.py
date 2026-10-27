"""The pre-registered decision rules, applied to ``results/**`` (WP14).

Each study committed a ``results/<study>/PREREGISTERED.md`` before its
experiment ran. It fixes the headline comparisons and the rule that turns an
interval into a sentence ("better", "worse", "no difference detected", the
lever ordering). The paper may state a finding only in the words its rule
gives, so this module computes every verdict from the results files instead
of leaving it to the prose:

- disk (E1): P1-P6 at both C1 capacities, the lever ordering, the C3
  robustness check (which needs the C1 comparison restricted to the
  ``PAPER_SUBSET`` traces, computed here from ``sweep.jsonl`` with the same
  bootstrap as ``adaptive_prefetch.analysis``) and the C4 merge-cap check;
  plus learned_depth against VCF (E2), which has no pre-registration of its
  own and is read with the disk rule.
- realio (E3): R1/R6 per cell, R2 with the exact permutation critical values,
  R7, and the Backend B parity count.
- web (E4/E5): W1/W2 under decision rule 1, the W3-W8 levers, rule 2 (lever
  ordering) and rule 3 (feedback gate).
- live (E6): H1-H7. Following the lead's ruling, an SSD-store pass whose
  ``foreign_io_frac`` exceeds 1 % is not cited, so H2-H7 use the passes that
  are valid *and* unflagged (``genlib.live_citable``).

:func:`decide` returns one :class:`Decision` per verdict. ``numbers.py``
turns them into macros and ``tables/make_tables.py`` into the decisions
table, so the prose, the table and the rule can never disagree silently.

Stdlib only (the disk core's analysis code is stdlib too).
"""

from __future__ import annotations

import importlib.util
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _genlib():
    if "paper_genlib" not in sys.modules:
        spec = importlib.util.spec_from_file_location("paper_genlib", HERE / "genlib.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules["paper_genlib"] = module
        spec.loader.exec_module(module)
    return sys.modules["paper_genlib"]


g = _genlib()

from prefetch_common import stats  # noqa: E402

#: The disk levers of PREREGISTERED.md section 5, in its order.
DISK_LEVERS = (("predictor", "adaptive"), ("feedback", "guard"), ("depth", "deep@8"),
               ("window growth", "linux_ra@256"))
#: P1-P4 of PREREGISTERED.md section 3: (id, policy, latency, baseline).
DISK_PRIMARY = (
    ("P1", "adaptive", "assumed", "sequential"),
    ("P2", "guard", "assumed", "sequential"),
    ("P2", "guard", "break_even", "sequential"),
    ("P3", "deep@8", "assumed", "none"),
    ("P3", "deep@32", "assumed", "none"),
    ("P3", "deep@256", "assumed", "none"),
    ("P4", "linux_ra@64", "assumed", "none"),
    ("P4", "linux_ra@256", "assumed", "none"),
    ("P4", "linux_ra@1024", "assumed", "none"),
)
C3_CONFIG = "C3:skip=1000000"
C4_CAPS = (("C4", 1024), ("C4:mms=256", 256))
#: R2: exact two-sided 5 % permutation critical values of |rho| without ties.
SPEARMAN_CRITICAL = {6: 0.886, 7: 0.786, 8: 0.738, 9: 0.700, 10: 0.648}
#: A row with more foreign I/O than this is contended (E3 gate 4, E6 ruling).
FOREIGN_LIMIT = 0.01
LOAD_LIMIT = 32.0

BETTER, WORSE = "better", "worse"
NO_DIFFERENCE = "no difference detected"
PAIRED_ONLY = "paired interval only; not robust to corpus clustering"


@dataclass(frozen=True)
class Decision:
    """One verdict. ``key`` is a letters-only macro stem; ``intervals`` holds
    the interval(s) the verdict rests on, as (label, estimate, lo, hi,
    statistic) with statistic ``difference`` (fractions, printed in pp),
    ``absolute`` (an interval of |D|, unsigned pp),
    ``ratio`` (printed as a multiplier), ``us``, ``ms`` or ``rho``."""
    study: str
    rule: str
    scope: str
    comparison: str
    verdict: str
    key: str
    intervals: tuple = ()
    note: str = ""


# ---------------------------------------------------------------------------
# Interval rules
# ---------------------------------------------------------------------------

def side(lo, hi, *, ratio: bool = False, higher_is_better: bool = True) -> str | None:
    """'better', 'worse' or None for one interval (a ratio is read against 1)."""
    if lo is None or hi is None:
        return None
    zero = 1.0 if ratio else 0.0
    if lo > zero:
        return BETTER if higher_is_better else WORSE
    if hi < zero:
        return WORSE if higher_is_better else BETTER
    return None


def disk_verdict(paired, cluster, *, ratio: bool = False,
                 higher_is_better: bool = True) -> str:
    """Disk PREREGISTERED.md section 5: both intervals must agree."""
    first = side(*paired, ratio=ratio, higher_is_better=higher_is_better)
    second = side(*cluster, ratio=ratio, higher_is_better=higher_is_better)
    if first is not None and first == second:
        return first
    if first is not None:
        return PAIRED_ONLY
    return NO_DIFFERENCE


def abs_interval(lo: float, hi: float) -> tuple[float, float]:
    """The interval of |D| for an interval [lo, hi] of D (both studies' rule)."""
    if lo < 0 < hi:
        return 0.0, max(abs(lo), abs(hi))
    return min(abs(lo), abs(hi)), max(abs(lo), abs(hi))


def web_verdict(lo, hi) -> str:
    """Web PREREGISTERED.md 'Statistics': the only three wordings."""
    if lo > 0:
        return "A better"
    if hi < 0:
        return "A worse"
    return "no detectable difference"


# ---------------------------------------------------------------------------
# Disk (E1, E2)
# ---------------------------------------------------------------------------

def _delta_index(rows):
    return {(r["config"], r["merge"], r["capacity"], r["latency"], r["corpus"], r["metric"],
             r["policy"], r["baseline"]): r for r in rows or ()}


def _cap(capacity: int) -> str:
    return g.capacity_label(capacity)


def _row_intervals(row, label=""):
    stat = row["statistic"]
    return ((f"{label}paired", row["estimate"], row["ci_lo"], row["ci_hi"], stat),
            (f"{label}cluster", row["cluster_estimate"], row["cluster_ci_lo"],
             row["cluster_ci_hi"], stat))


def _row_verdict(row, higher_is_better=True):
    ratio = row["statistic"] == "ratio"
    return disk_verdict((row["ci_lo"], row["ci_hi"]),
                        (row["cluster_ci_lo"], row["cluster_ci_hi"]), ratio=ratio,
                        higher_is_better=higher_is_better)


def _subset_deltas(sweep, capacities, needed):
    """C1 comparisons restricted to the PAPER_SUBSET traces (for the C3 rule),
    with the bootstrap of ``adaptive_prefetch.analysis`` (same reps and seed)."""
    from adaptive_prefetch import analysis, corpora
    subset = set(corpora.PAPER_SUBSET)
    rows = [r for r in sweep if r["config"] == g.HEADLINE_CONFIG and r["merge"] == g.HEADLINE_MERGE
            and r["trace"] in subset and r["capacity"] in capacities
            and r["skip_requests"] == 0 and r["max_requests"] is not None]
    groups = analysis.group_rows(rows)
    out = {}
    for (cell, merge, capacity), group in groups.items():
        if cell != g.HEADLINE_CONFIG:
            continue
        for policy, latency, baseline in needed:
            rows_p = group.get((latency, policy))
            base = analysis._baseline(group, latency, baseline)
            if not rows_p or not base:
                continue
            traces, a, b = analysis._pairs(rows_p, base, "hit_ratio", "all")
            if not traces:
                continue
            task = analysis._delta_task({
                "config": "C1:subset", "merge": merge, "capacity": capacity, "latency": latency,
                "corpus": "all", "metric": "hit_ratio", "statistic": "difference",
                "policy": policy, "baseline": baseline, "reps": 10_000, "seed": 42,
                "_a": a, "_b": b, "_traces": traces,
                "_corpora": [rows_p[t]["corpus"] for t in traces]})
            out[(capacity, latency, policy, baseline)] = task
    return out


def disk_decisions(results, out) -> list[Decision]:
    deltas = results.rows("disk/deltas.csv", out, required=False)
    if not deltas:
        return []
    index = _delta_index(deltas)
    head = (g.HEADLINE_CONFIG, g.HEADLINE_MERGE)
    capacities = sorted({r["capacity"] for r in deltas if (r["config"], r["merge"]) == head})
    found: list[Decision] = []

    def get(config, capacity, latency, metric, policy, baseline, corpus="all"):
        return index.get((config, g.HEADLINE_MERGE, capacity, latency, corpus, metric, policy,
                          baseline))

    # P1-P4, pooled, both capacities.
    for rule, policy, latency, baseline in DISK_PRIMARY:
        for capacity in capacities:
            row = get(g.HEADLINE_CONFIG, capacity, latency, "hit_ratio", policy, baseline)
            if row is None:
                continue
            scope = _cap(capacity) + ("" if latency == "assumed" else ", break-even cost model")
            found.append(Decision(
                "disk", rule, scope, f"{policy} - {baseline}, hit ratio", _row_verdict(row),
                "disk" + g.camel(rule) + g.camel(policy) + g.camel(latency)
                + g.disk_capacity_word(capacity), _row_intervals(row),
                f"n = {row['n']} traces"))
    # P5 / P6: device work of every policy against none.
    for capacity in capacities:
        for metric, rule, word in (("device_ios", "P5", "device I/Os"),
                                   ("blocks_moved", "P6", "blocks moved")):
            for key, row in sorted(index.items(), key=lambda kv: g.policy_key(kv[0][6])):
                if key[:5] != (g.HEADLINE_CONFIG, g.HEADLINE_MERGE, capacity, "assumed", "all") \
                        or key[5] != metric or key[7] != "none":
                    continue
                verdict = _row_verdict(row, higher_is_better=False)
                text = {BETTER: "reduces " + word, WORSE: "increases " + word}.get(verdict, verdict)
                if metric == "blocks_moved":
                    text = {BETTER: "moves fewer blocks", WORSE: "moves more blocks"}.get(
                        verdict, verdict)
                found.append(Decision(
                    "disk", rule, _cap(capacity), f"{row['policy']} / none, {word}", text,
                    "disk" + g.camel(rule) + g.camel(row["policy"]) + g.disk_capacity_word(capacity),
                    _row_intervals(row), f"n = {row['n']} traces"))
    # Lever ordering (section 5).
    for capacity in capacities:
        levers = []
        for name, policy in DISK_LEVERS:
            row = get(g.HEADLINE_CONFIG, capacity, "assumed", "hit_ratio", policy, "sequential")
            if row is not None:
                levers.append((name, policy, row))
        found.extend(_lever_order("disk", _cap(capacity), levers,
                                  "disk" + g.disk_capacity_word(capacity)))
    # C3 robustness and C4 merge-cap sensitivity.
    sweep = results.rows("disk/sweep.jsonl", out, required=False)
    needed = [(p, lat, b) for _, p, lat, b in DISK_PRIMARY]
    subset = _subset_deltas(sweep or [], capacities, needed) if sweep else {}
    for rule, policy, latency, baseline in DISK_PRIMARY:
        verdicts, intervals = [], []
        for capacity in capacities:
            first = subset.get((capacity, latency, policy, baseline))
            second = get(C3_CONFIG, capacity, latency, "hit_ratio", policy, baseline)
            if first is None or second is None:
                verdicts = None
                break
            verdicts.append((_row_verdict(first), _row_verdict(second)))
            intervals += _row_intervals(first, f"{_cap(capacity)} C1 subset ")
            intervals += _row_intervals(second, f"{_cap(capacity)} slice 2 ")
        if verdicts:
            robust = all(a == b for a, b in verdicts)
            found.append(Decision(
                "disk", "C3", "both capacities", f"{policy} - {baseline} ({rule})",
                "robust" if robust else "prefix-dependent",
                "diskCThree" + g.camel(policy) + g.camel(latency), tuple(intervals),
                "; ".join(f"{_cap(c)}: {a} / {b}" for c, (a, b) in zip(capacities, verdicts))))
        verdicts, intervals = [], []
        for capacity in capacities:
            rows = [get(config, capacity, latency, "hit_ratio", policy, baseline)
                    for config, _ in C4_CAPS]
            if None in rows:
                verdicts = None
                break
            verdicts.append(tuple(_row_verdict(r) for r in rows))
            for (config, cap), row in zip(C4_CAPS, rows):
                intervals += _row_intervals(row, f"{_cap(capacity)} merge cap {cap} ")
        if verdicts:
            same = all(a == b for a, b in verdicts)
            found.append(Decision(
                "disk", "C4", "FIU, both capacities", f"{policy} - {baseline} ({rule})",
                "cap-insensitive" if same else "cap-sensitive",
                "diskCFour" + g.camel(policy) + g.camel(latency), tuple(intervals),
                "; ".join(f"{_cap(c)}: {a} / {b}" for c, (a, b) in zip(capacities, verdicts))))
    found.extend(learned_depth_decisions(results, out))
    return found


def _lever_order(study, scope, levers, stem, *, paired_only=False):
    """Pairwise 'smaller' relations between levers, and the overall verdict.

    ``levers`` is [(name, policy, row)] with deltas.csv-like rows (paired
    ``ci_lo``/``ci_hi`` and, unless ``paired_only``, cluster bounds)."""
    found = []
    absolute = {}
    for name, _, row in levers:
        paired = abs_interval(row["ci_lo"], row["ci_hi"])
        cluster = None if paired_only else abs_interval(row["cluster_ci_lo"],
                                                         row["cluster_ci_hi"])
        absolute[name] = (paired, cluster)

    def smaller(a, b):
        pa, ca = absolute[a]
        pb, cb = absolute[b]
        return pa[1] < pb[0] and (paired_only or ca[1] < cb[0])

    names = [name for name, _, _ in levers]
    pairs = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if smaller(a, b):
                pairs.append(f"{a} < {b}")
            elif smaller(b, a):
                pairs.append(f"{b} < {a}")
            else:
                pairs.append(f"{a} ? {b}")
    predictor = names[0]
    smallest = all(smaller(predictor, other) for other in names[1:])
    intervals = []
    for name, _, row in levers:
        (plo, phi), cluster = absolute[name]
        intervals.append((f"|{name}| paired", None, plo, phi, "absolute"))
        if cluster is not None:
            intervals.append((f"|{name}| cluster", None, cluster[0], cluster[1], "absolute"))
    found.append(Decision(
        study, "levers", scope, "which lever moves hit ratio least",
        ("the predictor is the smallest lever" if smallest
         else "the predictor is not the smallest lever"),
        stem + "PredictorSmallest", tuple(intervals), "; ".join(pairs)))
    return found


def learned_depth_decisions(results, out) -> list[Decision]:
    """learned_depth - VCF per action cap (E2): seed means per trace, then
    the disk rule (paired bootstrap over traces and corpus-cluster bootstrap
    with twin down-weighting, 10,000 resamples, seed 42)."""
    rows = results.rows("gpu/disk_learned_depth.csv", out, required=False)
    if not rows:
        return []
    from adaptive_prefetch import corpora
    found = []
    for cap in sorted({r["action_cap"] for r in rows}):
        per_trace = defaultdict(list)
        for row in g.select(rows, action_cap=cap, status="ok"):
            per_trace[(row["trace"], row["corpus"])].append(row)
        traces, a, b, clusters = [], [], [], []
        for (trace, corpus), items in sorted(per_trace.items()):
            model = [r["test_hit_ratio"] for r in items if r["test_hit_ratio"] is not None]
            vcf = [r["vcf_hit_ratio"] for r in items if r["vcf_hit_ratio"] is not None]
            if model and vcf:
                traces.append(trace)
                a.append(statistics.fmean(model))
                b.append(statistics.fmean(vcf))
                clusters.append(corpus)
        if not traces:
            continue
        paired = stats.paired_bootstrap(a, b, reps=10000, seed=42)
        cluster = stats.cluster_bootstrap([x - y for x, y in zip(a, b)], clusters,
                                          stats.twin_weights(traces, corpora.TWIN_GROUPS),
                                          reps=10000, seed=42)
        seeds = len({r["seed"] for r in rows if r["action_cap"] == cap})
        found.append(Decision(
            "disk", "E2", f"action cap {cap}", "learned_depth - VCF, test-segment hit ratio",
            disk_verdict((paired.low, paired.high), (cluster.low, cluster.high)),
            "diskLearnedDepthVsVcfCap" + g.camel(cap),
            (("paired", paired.estimate, paired.low, paired.high, "difference"),
             ("cluster", cluster.estimate, cluster.low, cluster.high, "difference")),
            f"n = {len(traces)} traces, {seeds} seeds; no E2 pre-registration, disk rule"))
    return found


# ---------------------------------------------------------------------------
# Real-device testbed (E3)
# ---------------------------------------------------------------------------

def _realio_usable(row) -> bool:
    foreign = row["foreign_io_frac_max"]
    load = row["load_avg_max"]
    return (foreign is None or foreign <= FOREIGN_LIMIT) and (load is None or load <= LOAD_LIMIT)


def realio_cells(rows):
    cells = defaultdict(list)
    for row in rows or ():
        cells[(row["device"], row["backend"], row["think"], row["trace"])].append(row)
    return cells


def r2_verdict(rho, n, *, sign_consistent: int) -> str:
    """R2. ``sign_consistent`` is the sign of rho that agrees with measurement
    (-1 for hit ratio, +1 for simulated device time)."""
    if rho is None:
        return "not computed"
    critical = SPEARMAN_CRITICAL.get(n)
    if critical is None:
        return f"descriptive only (n = {n})"
    if abs(rho) >= critical:
        return ("ranks consistently with measured latency" if rho * sign_consistent > 0
                else "ranks against measured latency")
    return "ranking not supported"


def realio_decisions(results, out) -> list[Decision]:
    rows = results.rows("realio/realio_summary.csv", out, required=False)
    if not rows:
        return []
    found = []
    for (device, backend, think, trace), items in sorted(realio_cells(rows).items()):
        scope = f"{device}, {backend}, {think}, {trace}"
        stem = g.camel(device) + g.camel(backend) + g.camel(think) + g.camel(trace)
        contended = (device != "sdb" and any(not _realio_usable(r) for r in items))
        for row in sorted(items, key=lambda r: g.policy_key(r["mode"])):
            if row["mode"] == "none":
                continue
            lo, hi = row["delta_vs_none_us_lo"], row["delta_vs_none_us_hi"]
            if contended or not _realio_usable(row):
                verdict = "contended (not used)"
            elif row["reps"] < 2 or lo is None or hi is None:
                verdict = "not established"
            elif hi < 0:
                verdict = "lowers mean latency"
            elif lo > 0:
                verdict = "raises mean latency"
            else:
                verdict = f"no difference detected ({row['reps']} repeats)"
            rule = "R6" if row["mode"].startswith("kernel_") else "R1"
            found.append(Decision(
                "realio", rule, scope, f"{row['mode']} - none, mean latency", verdict,
                "realio" + g.camel(rule) + stem + g.camel(row["mode"]),
                (("", row["delta_vs_none_us"], lo, hi, "us"),)))
        planned = [r for r in items if r["sim_hit_ratio"] is not None]
        n = len(planned)
        first = items[0]
        for quantity, key, sign in (("hit", "spearman_hit_ratio", -1),
                                    ("device time", "spearman_device_time", +1)):
            verdict = ("contended (not used)" if contended
                       else r2_verdict(first[key], n, sign_consistent=sign))
            found.append(Decision(
                "realio", "R2", scope, f"rho(simulated {quantity}, measured latency)", verdict,
                "realioRTwo" + g.camel(quantity) + stem,
                (("", first[key], None, None, "rho"),), f"n = {n} modes"))
        usable = [r for r in items if not contended and _realio_usable(r)]
        if usable:
            fastest = min(usable, key=lambda r: r["lat_mean_us"])
            planned_usable = [r for r in usable if r["sim_hit_ratio"] is not None]
            top = max(planned_usable, key=lambda r: r["sim_hit_ratio"]) if planned_usable else None
            found.append(Decision(
                "realio", "fastest", scope, "lowest measured mean latency", fastest["mode"],
                "realioFastest" + stem,
                (("", fastest["lat_mean_us"], fastest["lat_mean_us_lo"],
                  fastest["lat_mean_us_hi"], "us"),),
                "highest simulated hit ratio: " + (top["mode"] if top else "none planned")))
    # R7: think time on msr_hm_0 (sdb), per backend and mode.
    cells = realio_cells(rows)
    for (device, backend, think, trace), items in sorted(cells.items()):
        if think != "closed":
            continue
        spread = [k for k in cells if k[0] == device and k[1] == backend and k[3] == trace
                  and k[2] != "closed"]
        for other in spread:
            closed_rows = {r["mode"]: r for r in items}
            for row in sorted(cells[other], key=lambda r: g.policy_key(r["mode"])):
                base = closed_rows.get(row["mode"])
                if row["mode"] == "none" or base is None:
                    continue
                below = [r["delta_vs_none_us_hi"] is not None and r["delta_vs_none_us_hi"] < 0
                         for r in (base, row)]
                verdict = ("think time changes the effect" if below[0] != below[1]
                           else "no change by the R7 rule")
                found.append(Decision(
                    "realio", "R7", f"{device}, {backend}, {trace}",
                    f"{row['mode']}: closed vs {other[2]}", verdict,
                    "realioRSeven" + g.camel(device) + g.camel(backend) + g.camel(trace)
                    + g.camel(row["mode"]),
                    (("closed", base["delta_vs_none_us"], base["delta_vs_none_us_lo"],
                      base["delta_vs_none_us_hi"], "us"),
                     (other[2], row["delta_vs_none_us"], row["delta_vs_none_us_lo"],
                      row["delta_vs_none_us_hi"], "us"))))
    raw = results.rows("realio/realio.csv", out, required=False)
    if raw:
        odirect = [r for r in raw if r["backend"] == "odirect"]
        found.append(Decision(
            "realio", "R3", "all devices", "Backend B parity with the simulator plan",
            f"held on all {len(odirect)} odirect runs", "realioRThree",
            note="asserted per request; a divergence raises CacheDivergence and stops the run"))
    return found


# ---------------------------------------------------------------------------
# Web (E4, E5)
# ---------------------------------------------------------------------------

WEB_LEVER_IDS = {"ranker": "W3", "freshness": "W4", "depth": "W5", "feedback": "W6",
                 "cache_scope": "W7", "sequence_model": "W8"}


def web_decisions(results, out) -> list[Decision]:
    found = []
    comparisons = results.rows("web/comparisons.csv", out, required=False)
    for capacity in (8, 32):
        row = next((r for r in g.select(comparisons, split="test", selection="per_capacity",
                                        scorer="logistic", capacity=capacity)
                    if r["cost"] == 0.01 and r["adaptive_policy"].startswith("adaptive_")
                    and not r["adaptive_policy"].startswith("adaptive_ca")), None)
        if row is None:
            continue
        w1 = web_verdict(row["vs_best_fixed_same_cap_lo"], row["vs_best_fixed_same_cap_hi"])
        w2 = web_verdict(row["vs_fixed0_lo"], row["vs_fixed0_hi"])
        scope = f"cap {capacity}, c = 0.01, logistic"
        found.append(Decision(
            "web", "W1", scope, f"{row['adaptive_policy']} - {row['best_fixed_policy']}, U(0.01)",
            w1, "webWOneCap" + g.camel(capacity),
            (("", row["vs_best_fixed_same_cap"],
              row["vs_best_fixed_same_cap_lo"], row["vs_best_fixed_same_cap_hi"], "difference"),)))
        found.append(Decision(
            "web", "W2", scope, f"{row['adaptive_policy']} - fixed0, U(0.01)", w2,
            "webWTwoCap" + g.camel(capacity),
            (("", row["vs_fixed0"], row["vs_fixed0_lo"], row["vs_fixed0_hi"],
              "difference"),)))
        if w2 == "A worse":
            rule1 = "loses to no prefetch"
        elif w1 == "A better":
            rule1 = "the adaptive controller beats fixed prefetching"
        else:
            rule1 = "no claim that it beats fixed prefetching"
        found.append(Decision("web", "rule 1", scope, "controller claim", rule1,
                              "webRuleOneCap" + g.camel(capacity)))
    decomposition = results.rows("web/decomposition.csv", out, required=False) or []
    hit_rows = {}
    for row in decomposition:
        rule = WEB_LEVER_IDS.get(row["lever"])
        if rule is None:
            continue
        metric = "hit ratio" if row["metric"] == "hit_ratio" else f"U({row['cost']:g})"
        cap = "unbounded" if row["capacity"] is None else f"cap {row['capacity']}"
        found.append(Decision(
            "web", rule, cap, f"{row['lever']}: {row['policy']}@{row['scorer']} - "
            f"{row['baseline']}@{row['baseline_scorer']}, {metric}",
            web_verdict(row["ci_lo"], row["ci_hi"]),
            "web" + g.camel(rule) + ("Hit" if row["metric"] == "hit_ratio" else "Utility")
            + g.camel(row["lever"]),
            (("", row["estimate"], row["ci_lo"], row["ci_hi"], "difference"),),
            f"{row['sessions']} sessions"))
        if row["metric"] == "hit_ratio" and row["capacity"] == 8:
            hit_rows.setdefault(row["lever"], row)
    # Rule 2: hi(predictor) < lo(depth) and hi(predictor) < lo(freshness).
    predictors = [lever for lever in ("ranker", "sequence_model") if lever in hit_rows]
    if predictors and "depth" in hit_rows and "freshness" in hit_rows:
        relations, intervals, smallest = [], [], True
        for lever in predictors + ["depth", "freshness"]:
            lo, hi = abs_interval(hit_rows[lever]["ci_lo"], hit_rows[lever]["ci_hi"])
            intervals.append((f"|{lever}|", None, lo, hi, "absolute"))
        for predictor in predictors:
            _, hi_p = abs_interval(hit_rows[predictor]["ci_lo"], hit_rows[predictor]["ci_hi"])
            for other in ("depth", "freshness"):
                lo_o, _ = abs_interval(hit_rows[other]["ci_lo"], hit_rows[other]["ci_hi"])
                if hi_p < lo_o:
                    relations.append(f"{predictor} < {other}")
                else:
                    relations.append(f"{predictor} ? {other} (no ordering claim)")
                    smallest = False
        found.append(Decision(
            "web", "rule 2", "cap 8, hit ratio", "which lever moves hit ratio least",
            ("the predictor is the smallest lever" if smallest
             else "the predictor is not called the smallest lever"),
            "webPredictorSmallest", tuple(intervals), "; ".join(relations)))
    feedback = next((r for r in decomposition if r["lever"] == "feedback"
                     and r["metric"] != "hit_ratio"), None)
    if feedback is not None:
        verdict = web_verdict(feedback["ci_lo"], feedback["ci_hi"])
        found.append(Decision(
            "web", "rule 3", f"cap {feedback['capacity']}, c = {feedback['cost']:g}",
            "feedback gate claim",
            ("improves on the adaptive controller" if verdict == "A better"
             else "not described as improving on the adaptive controller"),
            "webRuleThree"))
    return found


# ---------------------------------------------------------------------------
# Live HTTP (E6)
# ---------------------------------------------------------------------------

def _interval_overlap(a, b) -> bool:
    return not (a[1] < b[0] or b[1] < a[0])


def live_decisions(results, out) -> list[Decision]:
    rows = results.rows("live/live_metrics.csv", out, required=False)
    if not rows:
        return []
    found = []
    parity = results.rows("live/live_parity.csv", out, required=False)
    if parity is not None:
        mismatched = sum(r["mismatched_sessions"] for r in parity)
        found.append(Decision(
            "live", "H1", "all passes", "live counters vs simulator",
            "no mismatch" if mismatched == 0 else f"{mismatched} mismatched sessions (defect)",
            "liveHOne", note=f"{len(parity)} pass-policy rows"))
    citable = g.live_citable(rows)
    headline = defaultdict(list)
    for row in citable:
        if row["experiment"] == "headline" and row["policy"] != row["baseline"]:
            headline[(row["store"], row["policy"])].append(row)
    flagged = defaultdict(int)
    for row in rows:
        if (row["valid"] and row["experiment"] == "headline" and row not in citable):
            flagged[(row["store"], row["policy"])] += 1
    for (store, policy), items in sorted(headline.items(), key=lambda kv: (kv[0][0],
                                                                           g.policy_key(kv[0][1]))):
        items = [r for r in items if g.finite(r["delta_vs_base_ms_hi"])]
        intervals = tuple((f"rep {r['rep']}", r["delta_vs_base_ms"], r["delta_vs_base_ms_lo"],
                           r["delta_vs_base_ms_hi"], "ms") for r in items)
        note = f"{len(items)} citable repetitions"
        if flagged.get((store, policy)):
            note += f"; {flagged[(store, policy)]} flagged (foreign I/O > 1 %) not cited"
        if len(items) < 2:
            h2 = "not established"
        elif all(r["delta_vs_base_ms_hi"] < 0 for r in items):
            h2 = "lowers the mean wait"
        elif all(r["delta_vs_base_ms_lo"] > 0 for r in items):
            h2 = "raises the mean wait"
        else:
            h2 = "no consistent difference"
        stem = g.camel(store) + g.camel(policy)
        found.append(Decision("live", "H2", f"{store} store", f"{policy} - fixed0, mean wait",
                              h2, "liveHTwo" + stem, intervals, note))
        if items:
            if all(r["delta_vs_base_ms_lo"] > r["ideal_delta_vs_base_ms"] for r in items):
                h3 = "the instant-prefetch model overstates the benefit"
            elif all(r["delta_vs_base_ms_hi"] < r["ideal_delta_vs_base_ms"] for r in items):
                h3 = "the instant-prefetch model understates the benefit"
            else:
                h3 = "consistent with the model"
            found.append(Decision(
                "live", "H3", f"{store} store", f"{policy}: measured vs model-ideal delta", h3,
                "liveHThree" + stem,
                intervals + tuple((f"ideal rep {r['rep']}", r["ideal_delta_vs_base_ms"], None,
                                   None, "ms") for r in items[:1]), note))
    first = {}
    for (store, policy), items in headline.items():
        items = sorted((r for r in items if g.finite(r["delta_vs_base_ms_hi"])),
                       key=lambda r: r["rep"])
        if items:
            first[(store, policy)] = items[0]
    for row in citable:
        if row["experiment"] in ("fifo", "hint") and row["policy"] != row["baseline"]:
            if not g.finite(row["delta_vs_base_ms_hi"]):
                continue
            interval = (row["delta_vs_base_ms_lo"], row["delta_vs_base_ms_hi"])
            stem = g.camel(row["store"]) + g.camel(row["policy"])
            if row["experiment"] == "fifo":
                ref = first.get((row["store"], row["policy"]))
                if ref is None:
                    continue
                changed = not _interval_overlap(interval, (ref["delta_vs_base_ms_lo"],
                                                           ref["delta_vs_base_ms_hi"]))
                found.append(Decision(
                    "live", "H5", f"{row['store']} store, 1 repetition",
                    f"{row['policy']}: FIFO vs priority link",
                    "demand priority changes the delta" if changed
                    else "no change detected by the H5 rule", "liveHFive" + stem,
                    (("FIFO", row["delta_vs_base_ms"], *interval, "ms"),
                     (f"priority rep {ref['rep']}", ref["delta_vs_base_ms"],
                      ref["delta_vs_base_ms_lo"], ref["delta_vs_base_ms_hi"], "ms"))))
            else:
                verdict = ("lowers the mean wait in one repetition" if interval[1] < 0
                           else "raises the mean wait in one repetition" if interval[0] > 0
                           else "no difference in one repetition")
                found.append(Decision(
                    "live", "H6", f"{row['store']} store, 1 repetition",
                    f"{row['policy']} - fixed0, mean wait", verdict, "liveHSix" + stem,
                    (("", row["delta_vs_base_ms"], *interval, "ms"),)))
    policies = sorted({p for _, p in first}, key=g.policy_key)
    for policy in policies:
        ssd, hdd = first.get(("ssd", policy)), first.get(("hdd", policy))
        if ssd is None or hdd is None:
            continue
        differ = not _interval_overlap((ssd["delta_vs_base_ms_lo"], ssd["delta_vs_base_ms_hi"]),
                                       (hdd["delta_vs_base_ms_lo"], hdd["delta_vs_base_ms_hi"]))
        found.append(Decision(
            "live", "H7", "first citable repetition per store", f"{policy}: SSD vs HDD store",
            "the store changes the delta" if differ else "no store difference by the H7 rule",
            "liveHSeven" + g.camel(policy),
            ((f"ssd rep {ssd['rep']}", ssd["delta_vs_base_ms"], ssd["delta_vs_base_ms_lo"],
              ssd["delta_vs_base_ms_hi"], "ms"),
             (f"hdd rep {hdd['rep']}", hdd["delta_vs_base_ms"], hdd["delta_vs_base_ms_lo"],
              hdd["delta_vs_base_ms_hi"], "ms"))))
    # H4 (descriptive): policies no other policy beats on both saved wait and bytes.
    for store in sorted({s for s, _ in headline}):
        points = {}
        for (s, policy), items in headline.items():
            items = [r for r in items if g.finite(r["delta_vs_base_ms"])]
            if s == store and items:
                points[policy] = (-statistics.fmean(r["delta_vs_base_ms"] for r in items),
                                  statistics.fmean(r["bytes_vs_base"] for r in items))
        front = [p for p, (saved, byte) in points.items()
                 if not any(o != p and s2 >= saved and b2 <= byte and (s2, b2) != (saved, byte)
                            for o, (s2, b2) in points.items())]
        found.append(Decision(
            "live", "H4", f"{store} store", "not beaten on both saved wait and bytes",
            ", ".join(sorted(front, key=g.policy_key)), "liveHFour" + g.camel(store),
            note="descriptive; per-repetition means over citable passes"))
    return found


def decide(results, out=None) -> list[Decision]:
    """Every verdict, disk then real device, web and live."""
    return (disk_decisions(results, out) + realio_decisions(results, out)
            + web_decisions(results, out) + live_decisions(results, out))
