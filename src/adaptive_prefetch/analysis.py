"""E1 analysis: tables, deltas, winners and kappa* from ``disk/sweep.jsonl``.

    python -m adaptive_prefetch.analysis [--sweep results/disk/sweep.jsonl]
        [--out results/disk] [--reps 10000] [--seed 42] [--workers N]
        [--sweep2 PATH] [--gated-be PATH] [--grid PATH]

Writes ``table_main.csv`` (corpus ``all``), ``table_by_corpus.csv``,
``deltas.csv``, ``winners.csv`` and ``kappa.csv`` through
``prefetch_common.schemas``, and returns the consistency checks (which the
reproduce step records in ``provenance.json``).

How rows are grouped and compared, so a reader can audit every number:

* **Cell.** Rows are compared only within one (cell, merge, capacity). The
  cell is the config name, suffixed when a config holds more than one slice
  or merge cap: ``C3:skip=1000000`` (slice 2), ``C3:whole`` (whole files),
  ``C4:mms=256`` (FIU merge cap 256). The tables have no slice column, so the
  suffix is what keeps two slices from being averaged together.
* **Policy.** ``deep@<depth>``, ``linux_ra@<max_blocks>``, else the mode.
  ``belady_min`` is the demand-only Belady/MIN reference from the ``none``
  rows: a reference, not a policy and not a bound on prefetching.
* **Latency.** Only the feedback-gated modes depend on the cost model, so
  ``break_even`` rows exist only for them; their baselines (``none``,
  ``sequential``) are taken from the ``assumed`` rows, whose counters do not
  depend on latency.
* **Pairing.** A delta pairs a policy and its baseline on the traces where
  both ran (``status == "ok"``, ``read_blocks > 0``). ``device_ios`` and
  ``blocks_moved`` are per-trace ratios to the baseline (traces where the
  baseline is 0 are dropped). ``precision`` and ``wasted_share`` pair only
  traces where both issued a prefetch.
* **Intervals.** ``ci_*``: paired bootstrap over traces, every trace once.
  ``cluster_*``: two-level bootstrap (corpora, then traces) with each twin
  family down-weighted to total weight 1 (``corpora.TWIN_GROUPS``). Both are
  95% percentile intervals, ``reps`` resamples, ``seed`` fixed, so the output
  is deterministic. ``wilcoxon_p`` is two-sided on the paired differences
  (ratios minus 1); ``None`` without scipy.
* **Wins.** ``wins`` counts traces where the policy is *better*: higher for
  hit ratio and precision, lower for device I/Os, blocks moved and waste.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from statistics import mean
from types import SimpleNamespace

from prefetch_common import schemas, stats

from . import corpora
from .multicorpus import (CONFIGS, REPO, policy_label, read_sweep, row_key,
                          single_threaded_math)
from .simulator import GATED_MODES, IOCostModel
from .trace import DEFAULT_MAX_MERGE_SECTORS

DEFAULT_SWEEP = REPO / "results" / "disk" / "sweep.jsonl"
DEFAULT_OUT = REPO / "results" / "disk"
OUTPUTS = ("table_main.csv", "table_by_corpus.csv", "deltas.csv",
           "winners.csv", "kappa.csv")
BELADY = "belady_min"

#: metric -> (statistic, higher is better)
METRICS = {
    "hit_ratio": ("difference", True),
    "device_ios": ("ratio", False),
    "blocks_moved": ("ratio", False),
    "precision": ("difference", True),
    "wasted_share": ("difference", False),
}
VS_NONE = ("hit_ratio", "device_ios", "blocks_moved")
VS_SEQUENTIAL = ("hit_ratio", "precision", "wasted_share")
#: Counter columns that must agree for two rows to count as identical.
IDENTITY_FIELDS = (
    "read_blocks", "hit_ratio", "precision", "total_prefetches",
    "useful_prefetches", "unused", "mean_us", "policy_switches",
    "observed_precision", "precision_margin", "observed_stride", "demand_ios",
    "prefetch_ios", "prefetch_ios_mergeable", "device_ios", "blocks_moved")
#: The plan's hm_0 cross-check (sweep2.jsonl): first 250k, cap 2048, deep@8.
HM0_CHECK = {"trace": "msr_hm_0", "max_requests": 250_000, "capacity": 2048,
             "policy": "deep@8", "hit_ratio": 0.357231, "total_prefetches": 901487}

_CANONICAL_MMS = {"auto": DEFAULT_MAX_MERGE_SECTORS, "pages": DEFAULT_MAX_MERGE_SECTORS,
                  "same_second": None, "none": None}


# ------------------------------------------------------------------ labels

def cell_label(row) -> str:
    """Config name plus a suffix for a non-default slice or merge cap."""
    label = row["config"]
    if row["skip_requests"]:
        label += f":skip={row['skip_requests']}"
    if row["max_requests"] is None:
        label += ":whole"
    if row["max_merge_sectors"] != _CANONICAL_MMS.get(row["merge"]):
        label += f":mms={row['max_merge_sectors']}"
    return label


def row_policy(row) -> str:
    return policy_label(row["mode"], row["depth"], row["max_blocks"])


def _policy_order(label: str):
    if label == BELADY:
        return (2, 0, 0, label)
    mode, _, arg = label.partition("@")
    from .sweep import RANKING_MODES
    index = RANKING_MODES.index(mode) if mode in RANKING_MODES else len(RANKING_MODES)
    return (0 if mode in RANKING_MODES else 1, index, int(arg) if arg.isdigit() else 0, label)


def _config_order(label: str):
    base = label.split(":")[0]
    known = list(CONFIGS)
    return (known.index(base) if base in known else len(known), label)


def _corpus_order(corpus: str):
    return (-1, corpus) if corpus == "all" else (
        corpora.CORPORA.index(corpus) if corpus in corpora.CORPORA else 99, corpus)


def _scored(row) -> bool:
    return row.get("status") == "ok" and (row.get("read_blocks") or 0) > 0


def group_rows(rows) -> dict:
    """``{(cell, merge, capacity): {(latency, policy): {trace: row}}}`` of scored rows."""
    groups: dict = {}
    for row in rows:
        if not _scored(row):
            continue
        group = groups.setdefault((cell_label(row), row["merge"], row["capacity"]), {})
        bucket = group.setdefault((row["latency"], row_policy(row)), {})
        if row["trace"] in bucket:
            raise ValueError(f"two scored rows for one cell: {row_key(row)}")
        bucket[row["trace"]] = row
    return groups


def _baseline(group: dict, latency: str, name: str) -> dict | None:
    # Baselines are not gated, so their counters are the same at any latency.
    return group.get((latency, name)) or group.get(("assumed", name))


def _value(row, metric: str) -> float | None:
    if metric == "wasted_share":
        issued = row["total_prefetches"]
        return row["unused"] / issued if issued else None
    if metric == "precision":
        return row["precision"] if row["total_prefetches"] else None
    if metric == "hit_ratio" and row["mode"] == "none" and row.get("_belady"):
        return row["belady_hit_ratio"]
    return row[metric]


def _pairs(policy_rows: dict, base_rows: dict, metric: str, corpus: str):
    traces, a, b = [], [], []
    statistic = METRICS[metric][0]
    for trace in sorted(policy_rows):
        row = policy_rows[trace]
        if corpus != "all" and row["corpus"] != corpus:
            continue
        base = base_rows.get(trace)
        if base is None:
            continue
        x, y = _value(row, metric), _value(base, metric)
        if x is None or y is None or (statistic == "ratio" and y == 0):
            continue
        traces.append(trace)
        a.append(float(x))
        b.append(float(y))
    return traces, a, b


# ------------------------------------------------------------------ deltas

def _delta_task(task: dict) -> dict:
    """One ``deltas.csv`` row; top-level so a process pool can run it."""
    a, b, traces = task.pop("_a"), task.pop("_b"), task.pop("_traces")
    corpora_of, reps, seed = task.pop("_corpora"), task["reps"], task["seed"]
    statistic, higher = METRICS[task["metric"]]
    paired = stats.paired_bootstrap(a, b, reps, seed, statistic=statistic)
    effects = ([x - y for x, y in zip(a, b)] if statistic == "difference"
               else [x / y for x, y in zip(a, b)])
    weights = stats.twin_weights(traces, corpora.TWIN_GROUPS)
    cluster = stats.cluster_bootstrap(effects, corpora_of, weights, reps, seed)
    if statistic == "difference":
        p = stats.wilcoxon(a, b)
    else:
        p = stats.wilcoxon([e - 1.0 for e in effects])
    won, lost, tied = stats.wins(a, b) if higher else stats.wins(b, a)
    task.update({
        "statistic": statistic, "n": len(a),
        "estimate": paired.estimate, "ci_lo": paired.low, "ci_hi": paired.high,
        "cluster_estimate": cluster.estimate, "cluster_ci_lo": cluster.low,
        "cluster_ci_hi": cluster.high, "twins": "down_weighted",
        "wilcoxon_p": p, "wins": won, "losses": lost, "ties": tied,
    })
    return task


def _belady_rows(group: dict) -> dict:
    none = group.get(("assumed", "none"), {})
    out = {}
    for trace, row in none.items():
        if row.get("belady_hit_ratio") is not None:
            out[trace] = dict(row, _belady=True)
    return out


def delta_tasks(groups: dict, reps: int, seed: int) -> list[dict]:
    tasks = []
    for (cell, merge, capacity), group in groups.items():
        belady = _belady_rows(group)
        policies = dict(group)
        if belady:
            policies[("assumed", BELADY)] = belady
        present = sorted({row["corpus"] for bucket in group.values() for row in bucket.values()},
                         key=_corpus_order)
        for (latency, policy), rows in policies.items():
            for baseline, metrics in (("none", VS_NONE), ("sequential", VS_SEQUENTIAL)):
                if policy == baseline or (policy == BELADY and baseline != "none"):
                    continue
                base = _baseline(group, latency, baseline)
                if not base:
                    continue
                for metric in metrics:
                    if policy == BELADY and metric != "hit_ratio":
                        continue
                    for corpus in ["all", *present]:
                        traces, a, b = _pairs(rows, base, metric, corpus)
                        if not traces:
                            continue
                        tasks.append({
                            "config": cell, "merge": merge, "capacity": capacity,
                            "latency": latency, "corpus": corpus, "metric": metric,
                            "statistic": METRICS[metric][0], "policy": policy,
                            "baseline": baseline, "reps": reps, "seed": seed,
                            "_a": a, "_b": b, "_traces": traces,
                            "_corpora": [rows[t]["corpus"] for t in traces]})
    return tasks


def _run_tasks(tasks: list[dict], workers: int) -> list[dict]:
    if workers <= 1 or len(tasks) < 2:
        return [_delta_task(t) for t in tasks]
    import multiprocessing
    with multiprocessing.Pool(min(workers, len(tasks))) as pool:
        return pool.map(_delta_task, tasks, chunksize=1)


_DELTA_ORDER = ("hit_ratio", "device_ios", "blocks_moved", "precision", "wasted_share")


def _delta_sort(row):
    return (_config_order(row["config"]), row["merge"], row["capacity"], row["latency"],
            _corpus_order(row["corpus"]), row["baseline"] != "none",
            _policy_order(row["policy"]), _DELTA_ORDER.index(row["metric"]))


def compute_deltas(groups: dict, *, reps: int = 10_000, seed: int = 42,
                   workers: int = 1) -> list[dict]:
    rows = _run_tasks(delta_tasks(groups, reps, seed), workers)
    columns = schemas.columns("disk/deltas.csv")
    rows = [{c: row[c] for c in columns} for row in rows]
    return sorted(rows, key=_delta_sort)


# ------------------------------------------------------------------ tables

def _mean(values):
    values = list(values)
    return mean(values) if values else None


def table_rows(groups: dict, deltas: list[dict]) -> tuple[list[dict], list[dict]]:
    """``(table_main rows, table_by_corpus rows)``."""
    index = {(d["config"], d["merge"], d["capacity"], d["latency"], d["corpus"],
              d["metric"], d["policy"], d["baseline"]): d for d in deltas}
    main, by_corpus = [], []
    for (cell, merge, capacity), group in groups.items():
        policies = dict(group)
        belady = _belady_rows(group)
        if belady:
            policies[("assumed", BELADY)] = belady
        present = sorted({row["corpus"] for bucket in group.values() for row in bucket.values()},
                         key=_corpus_order)
        for (latency, policy), rows in policies.items():
            for corpus in ["all", *present]:
                chosen = [r for r in rows.values() if corpus == "all" or r["corpus"] == corpus]
                if not chosen:
                    continue
                key = (cell, merge, capacity, latency, corpus)

                def delta(metric, baseline):
                    return index.get(key + (metric, policy, baseline))

                vs_none = delta("hit_ratio", "none")
                vs_seq = delta("hit_ratio", "sequential")
                ios = delta("device_ios", "none")
                moved = delta("blocks_moved", "none")
                hits = [_value(r, "hit_ratio") for r in chosen]
                weights = [r["read_blocks"] for r in chosen]
                issued = [r for r in chosen if r["total_prefetches"]] if policy != BELADY else []
                row = {
                    "config": cell, "merge": merge, "capacity": capacity,
                    "latency": latency, "corpus": corpus, "policy": policy,
                    "n_traces": len(chosen),
                    "mean_hit_ratio": _mean(hits),
                    "weighted_hit_ratio": sum(h * w for h, w in zip(hits, weights)) / sum(weights),
                    "delta_vs_none": vs_none and vs_none["estimate"],
                    "delta_vs_none_lo": vs_none and vs_none["ci_lo"],
                    "delta_vs_none_hi": vs_none and vs_none["ci_hi"],
                    "delta_vs_sequential": vs_seq and vs_seq["estimate"],
                    "delta_vs_sequential_lo": vs_seq and vs_seq["ci_lo"],
                    "delta_vs_sequential_hi": vs_seq and vs_seq["ci_hi"],
                    "mean_precision": _mean(r["precision"] for r in issued),
                    "mean_wasted_share": _mean(r["unused"] / r["total_prefetches"]
                                               for r in issued),
                    "device_io_ratio": ios and ios["estimate"],
                    "blocks_moved_ratio": moved and moved["estimate"],
                    "wins_vs_none": vs_none and vs_none["wins"],
                    "losses_vs_none": vs_none and vs_none["losses"],
                }
                (main if corpus == "all" else by_corpus).append(row)
    order = lambda r: (_config_order(r["config"]), r["merge"], r["capacity"],
                       r["latency"], _corpus_order(r["corpus"]), _policy_order(r["policy"]))
    return sorted(main, key=order), sorted(by_corpus, key=order)


# ----------------------------------------------------------------- winners

def winners_rows(groups: dict, *, tol: float = 1e-12) -> list[dict]:
    """Per-trace winners by hit ratio (highest) and device I/Os (fewest).

    Only ``assumed``-latency rows compete, so every policy runs under one
    cost model. A trace whose best value is shared counts as a win for every
    tied policy, and those rows say so in ``note``.
    """
    out = []
    for (cell, merge, capacity), group in groups.items():
        policies = {p: rows for (lat, p), rows in group.items() if lat == "assumed"}
        none = policies.get("none", {})
        present = sorted({r["corpus"] for rows in policies.values() for r in rows.values()},
                         key=_corpus_order)
        for corpus in ["all", *present]:
            subset = {p: {t: r for t, r in rows.items()
                          if corpus == "all" or r["corpus"] == corpus}
                      for p, rows in policies.items()}
            subset = {p: rows for p, rows in subset.items() if rows}
            traces = sorted({t for rows in subset.values() for t in rows})
            for metric, higher in (("hit_ratio", True), ("device_ios", False)):
                won = {p: 0 for p in subset}
                tied = {p: False for p in subset}
                for trace in traces:
                    values = {p: rows[trace][metric] for p, rows in subset.items()
                              if trace in rows}
                    best = max(values.values()) if higher else min(values.values())
                    leaders = [p for p, v in values.items() if abs(v - best) <= tol]
                    for p in leaders:
                        won[p] += 1
                        if len(leaders) > 1:
                            tied[p] = True
                means = {}
                for p, rows in subset.items():
                    if metric == "hit_ratio":
                        means[p] = mean(r["hit_ratio"] for r in rows.values())
                    else:
                        ratios = [r["device_ios"] / none[t]["device_ios"]
                                  for t, r in rows.items()
                                  if t in none and none[t]["device_ios"]]
                        means[p] = mean(ratios) if ratios else None
                ranked = sorted((p for p in subset if means[p] is not None),
                                key=lambda p: -means[p] if higher else means[p])
                for p in sorted(subset, key=_policy_order):
                    out.append({
                        "config": cell, "merge": merge, "capacity": capacity,
                        "latency": "assumed", "corpus": corpus, "metric": metric,
                        "policy": p, "traces_won": won[p], "n_traces": len(subset[p]),
                        "mean_value": means[p],
                        "rank_by_mean": ranked.index(p) + 1 if p in ranked else None,
                        "note": "ties counted for every tied policy" if tied[p] else None,
                    })
    return sorted(out, key=lambda r: (_config_order(r["config"]), r["merge"], r["capacity"],
                                      _corpus_order(r["corpus"]), r["metric"],
                                      _policy_order(r["policy"])))


# ------------------------------------------------------------------- kappa

def kappa_rows(groups: dict) -> list[dict]:
    """kappa* per (trace, policy) against ``none`` at the same cell (assumed latency)."""
    out = []
    for (cell, merge, capacity), group in groups.items():
        none = group.get(("assumed", "none"), {})
        for (latency, policy), rows in group.items():
            if latency != "assumed" or policy == "none":
                continue
            for trace, row in rows.items():
                base = none.get(trace)
                if base is None:
                    continue
                k = IOCostModel.kappa_star(
                    SimpleNamespace(device_ios=row["device_ios"], blocks_moved=row["blocks_moved"]),
                    SimpleNamespace(device_ios=base["device_ios"], blocks_moved=base["blocks_moved"]))
                out.append({
                    "config": cell, "merge": merge, "capacity": capacity,
                    "trace": trace, "corpus": row["corpus"], "policy": policy,
                    "ios_policy": row["device_ios"], "ios_none": base["device_ios"],
                    "blocks_policy": row["blocks_moved"], "blocks_none": base["blocks_moved"],
                    "delta_ios": row["device_ios"] - base["device_ios"],
                    "delta_blocks": row["blocks_moved"] - base["blocks_moved"],
                    "kappa_star": k,
                })
    return sorted(out, key=lambda r: (_config_order(r["config"]), r["merge"], r["capacity"],
                                      _corpus_order(r["corpus"]), r["trace"],
                                      _policy_order(r["policy"])))


# ------------------------------------------------------------------ checks

def _same(a: dict, b: dict) -> bool:
    return all(a[f] == b[f] for f in IDENTITY_FIELDS)


def _identity(groups: dict, latency: str, left: str, right: str) -> dict:
    pairs = identical = 0
    mismatches = []
    for (cell, merge, capacity), group in sorted(groups.items(), key=lambda kv: str(kv[0])):
        a, b = group.get((latency, left), {}), group.get((latency, right), {})
        for trace in sorted(set(a) & set(b)):
            pairs += 1
            if _same(a[trace], b[trace]):
                identical += 1
            elif len(mismatches) < 20:
                mismatches.append(f"{cell} {merge} cap={capacity} {trace}")
    return {"pairs": pairs, "identical": identical, "mismatches": mismatches}


def checks(rows: list[dict]) -> dict:
    """Consistency checks over the raw sweep rows (recorded in provenance).

    ``pass`` is the overall verdict (the validity gates of
    ``results/disk/PREREGISTERED.md`` section 2): ``False`` iff a gate that
    had something to compare failed, and ``failures`` names those gates. A
    gate with nothing to compare (no hm_0 cross-check cell, no guard/
    depth_adaptive pair at break-even, no deep@0 cell) is listed in
    ``not_checked`` instead: it is neither a pass nor a failure, so a caller
    that needs every gate checked tests ``pass and not not_checked``.
    """
    keys = [row_key(r) for r in rows]
    statuses: dict[str, int] = {}
    for row in rows:
        status = row["status"] if row["status"] == "ok" else row["status"].split(" (")[0]
        statuses[status] = statuses.get(status, 0) + 1
    groups = group_rows(rows)
    hm0 = []
    for row in rows:
        if (_scored(row) and row["trace"] == HM0_CHECK["trace"]
                and row["skip_requests"] == 0
                and row["max_requests"] == HM0_CHECK["max_requests"]
                and row["capacity"] == HM0_CHECK["capacity"]
                and row_policy(row) == HM0_CHECK["policy"]
                and row["latency"] == "assumed" and row["merge"] in ("auto", "none")):
            hm0.append({"config": row["config"], "merge": row["merge"],
                        "hit_ratio": row["hit_ratio"],
                        "total_prefetches": row["total_prefetches"],
                        "pass": (round(row["hit_ratio"], 6) == HM0_CHECK["hit_ratio"]
                                 and row["total_prefetches"] == HM0_CHECK["total_prefetches"])})
    report = {
        "rows": len(rows), "duplicate_keys": len(keys) - len(set(keys)),
        "status_counts": dict(sorted(statuses.items())),
        "guard_eq_depth_adaptive_at_break_even": _identity(
            groups, "break_even", "guard", "depth_adaptive"),
        "deep0_eq_none": _identity(groups, "assumed", "deep@0", "none"),
        "hm0_deep8_250k": hm0,
        "wilcoxon": None if stats.HAVE_SCIPY else stats.WILCOXON_UNAVAILABLE,
    }
    failures, not_checked = [], []
    if report["duplicate_keys"]:
        failures.append("duplicate_keys")
    for name in ("guard_eq_depth_adaptive_at_break_even", "deep0_eq_none"):
        gate = report[name]
        if not gate["pairs"]:
            not_checked.append(name)
        elif gate["identical"] != gate["pairs"]:
            failures.append(name)
    if not hm0:
        not_checked.append("hm0_deep8_250k")
    elif not all(cell["pass"] for cell in hm0):
        failures.append("hm0_deep8_250k")
    report.update({"pass": not failures, "failures": failures, "not_checked": not_checked})
    return report


# -------------------------------------------------------- prototype cross-check

def _read_jsonl(path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _proto_policy(mode: str, depth: int) -> tuple[str, int, int | None]:
    if mode == "linuxra":
        return "linux_ra", 0, depth
    if mode == "deep":
        return "deep", depth, None
    return mode, 0, None


def crosscheck(rows: list[dict], *, sweep2=None, gated_be=None, grid=None,
               tol: float = 1e-9) -> dict:
    """Compare sweep rows with the scratch prototype's cells (plan §2.3, C2).

    Only cells with the same definition are compared: prototype ``raw`` cells
    (250k records = 250k requests) against ``merge="none"`` rows, or
    ``merge="auto"`` rows on non-FIU traces, where the two coincide. The
    prototype's ``merged`` cells capped 250k *records* before merging, while
    master caps canonical *requests*, so they are counted as not comparable,
    never matched. ``demand_ios`` is not compared: the prototype counted one
    per request with a miss, master one per missed run (D18).
    """
    index: dict = {}
    for row in rows:
        if not _scored(row) or row["skip_requests"] or row["max_requests"] != 250_000:
            continue
        if row["merge"] == "none" or (row["merge"] == "auto" and row["corpus"] != "fiu"):
            key = (row["trace"], row["capacity"], row["mode"], row["depth"],
                   row["max_blocks"], row["latency"])
            index.setdefault(key, []).append(row)
    fields = (("hit", "hit_ratio", True), ("prefetches", "total_prefetches", False),
              ("useful", "useful_prefetches", False), ("read_blocks", "read_blocks", False),
              ("switches", "policy_switches", False), ("prefetch_ios", "prefetch_ios", False))
    report = {"compared": 0, "matched": 0, "missing": 0, "not_comparable": 0,
              "mismatches": []}

    def compare(cells, latency_of):
        for cell in cells:
            if cell.get("variant", "raw") != "raw":
                report["not_comparable"] += 1
                continue
            mode, latency = latency_of(cell["mode"])
            policy = _proto_policy(mode, cell.get("depth", 0))
            ours = index.get((cell["trace"], cell["cap"], *policy, latency))
            if not ours:
                report["missing"] += 1
                continue
            for row in ours:
                report["compared"] += 1
                bad = [name for proto, name, is_float in fields if proto in cell and (
                    abs(cell[proto] - row[name]) > tol if is_float else cell[proto] != row[name])]
                if bad:
                    if len(report["mismatches"]) < 50:
                        report["mismatches"].append(
                            f"{cell['trace']} cap={cell['cap']} {cell['mode']}@{cell.get('depth')} "
                            f"{row['config']}/{row['merge']}: {bad}")
                else:
                    report["matched"] += 1

    plain = lambda mode: (mode, "assumed")
    gated = lambda mode: (mode[:-len("_be1.01")], "break_even") if mode.endswith(
        "_be1.01") else (mode, "assumed")
    if sweep2:
        compare(_read_jsonl(sweep2), plain)
    if gated_be:
        compare(_read_jsonl(gated_be), gated)
    if grid:
        compare(_read_jsonl(grid), plain)
    return report


# -------------------------------------------------------------------- write

def write_analysis(rows: list[dict], out_dir: str | Path, *, reps: int = 10_000,
                   seed: int = 42, workers: int = 1) -> dict[str, Path]:
    """Write the five analysis CSVs; returns ``{file name: path}``."""
    out_dir = Path(out_dir)
    groups = group_rows(rows)
    deltas = compute_deltas(groups, reps=reps, seed=seed, workers=workers)
    main, by_corpus = table_rows(groups, deltas)
    written = {}
    for name, data in (("table_main.csv", main), ("table_by_corpus.csv", by_corpus),
                       ("deltas.csv", deltas), ("winners.csv", winners_rows(groups)),
                       ("kappa.csv", kappa_rows(groups))):
        written[name] = schemas.write_rows(f"disk/{name}", data, out_dir / name)
    return written


def _jsonable(value):
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m adaptive_prefetch.analysis", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sweep", default=str(DEFAULT_SWEEP))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--reps", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--sweep2", default=None,
                        help="prototype sweep2.jsonl to cross-check against")
    parser.add_argument("--gated-be", default=None, help="prototype gated_be.jsonl")
    parser.add_argument("--grid", default=None, help="prototype grid.jsonl")
    parser.add_argument("--checks", default=None,
                        help="also write the consistency checks to this JSON file")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    single_threaded_math()
    rows = read_sweep(args.sweep)
    if not rows:
        print(f"no sweep rows in {args.sweep}")
        return 1
    written = write_analysis(rows, args.out, reps=args.reps, seed=args.seed,
                             workers=args.workers)
    report = checks(rows)
    if args.sweep2 or args.gated_be or args.grid:
        report["prototype_crosscheck"] = crosscheck(
            rows, sweep2=args.sweep2, gated_be=args.gated_be, grid=args.grid)
    text = json.dumps(report, indent=2, default=_jsonable)
    print(text)
    if args.checks:
        Path(args.checks).parent.mkdir(parents=True, exist_ok=True)
        Path(args.checks).write_text(text + "\n", encoding="utf-8")
    for name, path in written.items():
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
