"""Results files of the hindsight-router experiment (``results/hindsight``).

Stdlib only. Reads the driver's cache (records, CV results, proxy scores,
replays) and writes five files through ``prefetch_common.schemas``:

``rows.csv``         one row per (trace, slice, capacity, policy) replay
``selection.csv``    every random-search configuration's grouped-CV score
``families.csv``     every family's best router on slice 2 (trajectory proxy)
``summary.csv``      per (capacity, policy): means, device work vs ``none``
``comparisons.csv``  the pre-registered comparisons with paired and cluster
                     intervals and the verdict word
"""

from __future__ import annotations

import json
from pathlib import Path

from prefetch_common import schemas, stats

from . import corpora, hindsight

REPS_SEED = 42
#: Counters a reproduced sweep row must match exactly.
MATCH = ("read_blocks", "hit_ratio", "total_prefetches", "useful_prefetches",
         "demand_ios", "prefetch_ios", "prefetch_ios_mergeable", "device_ios",
         "blocks_moved")
#: Higher-is-better metrics; the rest are lower-is-better ratios.
HIGHER = ("hit_ratio", "u0.5")
#: The pre-registered comparisons (PREREGISTERED.md section 6). ``None`` in
#: the objective position means "every objective".
COMPARISONS = (
    ("H1", "hit_ratio", "routed_hit", "vcf_hit", True),
    ("H2", "hit_ratio", "routed_hit", "sequential", False),
    ("H3", "u0.5", "routed_u0.5", "vcf_u0.5", False),
    ("H4", "device_ios", "routed_dev", "vcf_dev", False),
    ("H5", "hit_ratio", "routed_hit", "adaptive", False),
    ("H6", "hit_ratio", "routed_hit_loco", "vcf_hit", False),
    ("H7", "hit_ratio", "routed_hit", "gvcf_hit", False),
    ("H7", "u0.5", "routed_u0.5", "gvcf_u0.5", False),
    ("H7", "device_ios", "routed_dev", "gvcf_dev", False),
    ("S1", "hit_ratio", "routed_hit", "deep@8", False),
    ("S1", "hit_ratio", "routed_hit", "linux_ra@256", False),
    ("S1", "hit_ratio", "routed_hit", "adaptive_evidence", False),
    ("S2", "hit_ratio", "routed_hit_gnb", "routed_hit", False),
    ("S2", "hit_ratio", "routed_hit_gnb", "vcf_hit", False),
    ("R", "hit_ratio", "oracle_hit", "vcf_hit", False),
    ("R", "u0.5", "oracle_u0.5", "vcf_u0.5", False),
    ("R", "device_ios", "oracle_dev", "vcf_dev", False),
)
OBJECTIVE_METRIC = {"hit": "hit_ratio", "u0.5": "u0.5", "dev": "device_ios"}


def metric_value(row: dict, metric: str) -> float:
    if metric == "u0.5":
        return (row["read_hits"] - hindsight.UTIL_C * row["total_prefetches"]) / row["read_blocks"]
    return float(row[metric])


def verdict(paired, cluster, higher: bool, ratio: bool) -> str:
    """The pre-registered word for a paired and a cluster interval."""
    pivot = 1.0 if ratio else 0.0

    def side(ci):
        if ci.low > pivot:
            return "above"
        if ci.high < pivot:
            return "below"
        return None

    good = "above" if higher else "below"
    p, c = side(paired), side(cluster)
    if p is not None and p == c:
        return "better" if p == good else "worse"
    if p is not None:
        return "no difference detected (paired interval only; not robust to corpus clustering)"
    return "no difference detected"


def compare(values_a, values_b, names, metric, reps):
    ratio = metric not in HIGHER
    statistic = "ratio" if ratio else "difference"
    paired = stats.paired_bootstrap(values_a, values_b, reps, REPS_SEED, statistic=statistic)
    effects = ([a / b for a, b in zip(values_a, values_b)] if ratio
               else [a - b for a, b in zip(values_a, values_b)])
    weights = stats.twin_weights(names, corpora.TWIN_GROUPS)
    cluster = stats.cluster_bootstrap(effects, [corpora.CORPUS_OF(t) for t in names],
                                      weights, reps, REPS_SEED)
    higher = metric in HIGHER
    better = sum((a > b) if higher else (a < b) for a, b in zip(values_a, values_b))
    worse = sum((a < b) if higher else (a > b) for a, b in zip(values_a, values_b))
    return {"statistic": statistic, "n": paired.n, "estimate": paired.estimate,
            "ci_lo": paired.low, "ci_hi": paired.high,
            "cluster_estimate": cluster.estimate, "cluster_ci_lo": cluster.low,
            "cluster_ci_hi": cluster.high, "wins": better, "losses": worse,
            "ties": len(values_a) - better - worse,
            "verdict": verdict(paired, cluster, higher, ratio)}


def _read_gz(path):
    from .hindsight_experiment import read_gz
    return read_gz(path)


def _sweep_rows(sweep: Path | None, n: int) -> dict:
    out = {}
    if sweep is None or not Path(sweep).exists():
        return out
    for line in Path(sweep).read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["status"] != "ok" or row["merge"] != "auto" or row["latency"] != "assumed":
            continue
        if row["config"] not in ("C1", "C3") or row["max_requests"] != n:
            continue
        label = (f"deep@{row['depth']}" if row["mode"] == "deep" else
                 f"linux_ra@{row['max_blocks']}" if row["mode"] == "linux_ra" else row["mode"])
        out[(row["trace"], row["skip_requests"], row["capacity"], label)] = row
    return out


def _matches(metrics: dict, sweep_row: dict | None) -> bool | None:
    if sweep_row is None:
        return None
    return all(metrics[k] == sweep_row[k] for k in MATCH)


def _row(trace, skip, n, cap, policy, metrics, *, source, family=None, objective=None,
         action=None, shares=None, switches=None, training=None, match=None,
         timing=None):
    m = metrics
    issued = m["total_prefetches"]
    return {
        "trace": trace, "corpus": corpora.CORPUS_OF(trace), "skip_requests": skip,
        "max_requests": n, "capacity": cap, "policy": policy, "family": family,
        "objective": objective, "action": action, "status": "ok", "source": source,
        "read_blocks": m["read_blocks"], "read_hits": m["read_hits"],
        "hit_ratio": m["hit_ratio"], "precision": m["precision"],
        "total_prefetches": issued, "useful_prefetches": m["useful_prefetches"],
        "wasted_share": (issued - m["useful_prefetches"]) / issued if issued else None,
        "u05_per_block": metric_value(m, "u0.5"),
        "demand_ios": m["demand_ios"], "prefetch_ios": m["prefetch_ios"],
        "prefetch_ios_mergeable": m["prefetch_ios_mergeable"],
        "device_ios": m["device_ios"], "blocks_moved": m["blocks_moved"],
        "policy_switches": switches, "action_shares": shares,
        "training_range": training, "sweep_match": match,
        "load_start": (timing or {}).get("load_start"),
        "load_end": (timing or {}).get("load_end"),
        "wall_s": (timing or {}).get("wall_s"),
    }


def _with_hits(row: dict) -> dict:
    row = dict(row)
    row.setdefault("read_hits", round(row["hit_ratio"] * row["read_blocks"]))
    return row


def analyze(cache: Path, out: Path, traces, capacities, n: int, reps: int,
            sweep: Path | None = None) -> dict:
    from .hindsight_experiment import _key
    cache, out = Path(cache), Path(out)
    sweep_rows = _sweep_rows(sweep, n)
    skip1, skip2 = 0, n
    training = f"slice 1 [0, {n}) of {len(traces)} PAPER_SUBSET traces"
    rows: list[dict] = []
    checks: dict = {"fixed_rows_compared": 0, "fixed_rows_matching": 0}
    fixed: dict = {}
    usable = []
    for trace in traces:
        f2 = _read_gz(cache / "features" / f"{_key(trace, skip2)}.json.gz")
        if f2["requests"] >= n:
            usable.append(trace)
        for skip in (skip1, skip2):
            if skip == skip2 and trace not in usable:
                continue
            for cap in capacities:
                for action in hindsight.ACTIONS:
                    rec = _read_gz(cache / "records" / f"{_key(trace, skip, cap, action)}.json.gz")
                    match = _matches(rec["metrics"], sweep_rows.get((trace, skip, cap, action)))
                    if match is not None:
                        checks["fixed_rows_compared"] += 1
                        checks["fixed_rows_matching"] += int(match)
                    fixed[(trace, skip, cap, action)] = rec["metrics"]
                    rows.append(_row(trace, skip, n, cap, action, rec["metrics"],
                                     source="record", action=action, match=match,
                                     timing=rec))
    if sweep_rows and checks["fixed_rows_matching"] != checks["fixed_rows_compared"]:
        raise SystemExit(f"fixed-action replays disagree with sweep.jsonl: {checks}")

    # Validation-chosen fixed actions, from slice 1 only.
    vcf: dict = {}
    for cap in capacities:
        for objective in hindsight.OBJECTIVES:
            means = {}
            for action in hindsight.ACTIONS:
                scores = [hindsight.slice_objective(fixed[(t, skip1, cap, action)], objective,
                                                    fixed[(t, skip1, cap, "none")])
                          for t in traces]
                means[action] = sum(scores) / len(scores)
            gbest = hindsight.ACTIONS[hindsight.best_action_index(
                [means[a] for a in hindsight.ACTIONS])]
            for trace in traces:
                values = [hindsight.slice_objective(fixed[(trace, skip1, cap, a)], objective,
                                                    fixed[(trace, skip1, cap, "none")])
                          for a in hindsight.ACTIONS]
                vcf[(trace, cap, f"vcf_{objective}")] = hindsight.ACTIONS[
                    hindsight.best_action_index(values)]
                vcf[(trace, cap, f"gvcf_{objective}")] = gbest
    for (trace, cap, label), action in sorted(vcf.items()):
        if trace not in usable:
            continue
        objective = label.split("_", 1)[1]
        rows.append(_row(trace, skip2, n, cap, label, fixed[(trace, skip2, cap, action)],
                         source="record", objective=objective, action=action,
                         training=training))

    # Replays of routers, oracles and adaptive_evidence.
    chosen = json.loads((cache / "train" / "chosen.json").read_text(encoding="utf-8"))
    score_dir = cache / "score"
    for path in sorted(score_dir.glob("*.json.gz")):
        rec = _read_gz(path)
        label = rec["label"]
        if label.startswith("check_"):
            continue
        objective = family = None
        if label.startswith(("routed_", "oracle_")):
            objective = label.split("_")[1]
        if label.startswith("routed_"):
            which = "gnb" if label.endswith("_gnb") else "selected"
            family = chosen[f"{rec['capacity']}|{objective}|{which}"]["family"]
        tr = (training if label.startswith("routed_") and not label.endswith("_loco")
              else (f"slice 1 [0, {n}) of the PAPER_SUBSET traces outside "
                    f"{corpora.CORPUS_OF(rec['trace'])}") if label.endswith("_loco")
              else "slice 2 itself (hindsight labels; reference, not a policy)"
              if label.startswith("oracle_") else None)
        rows.append(_row(rec["trace"], rec["skip"], n, rec["capacity"], label,
                         rec["metrics"], source="replay", family=family,
                         objective=objective, shares=rec["action_shares"],
                         switches=rec["metrics"]["policy_switches"], training=tr,
                         timing=rec))
    # Reused sweep rows, after one exact reproduction of each.
    for mode, label in (("adaptive", "adaptive"), ("linux_ra", "linux_ra@256")):
        checks_files = list(score_dir.glob(f"*__check_{mode}.json.gz"))
        if not checks_files:
            continue
        rec = _read_gz(checks_files[0])
        reference = sweep_rows.get((rec["trace"], rec["skip"], rec["capacity"], label))
        match = _matches(rec["metrics"], reference)
        checks[f"reproduced_{label}"] = {"trace": rec["trace"], "capacity": rec["capacity"],
                                         "match": match}
        if match:
            for trace in usable:
                for cap in capacities:
                    ref = sweep_rows.get((trace, skip2, cap, label))
                    if ref is None:
                        continue
                    rows.append(_row(trace, skip2, n, cap, label, _with_hits(ref),
                                     source="sweep", match=True,
                                     timing={"wall_s": ref["wall_s"],
                                             "load_start": ref["load_start"],
                                             "load_end": ref["load_end"]}))
        elif match is False:
            raise SystemExit(f"{label} does not reproduce sweep.jsonl: {rec['metrics']}")
        elif not sweep_rows:
            rows.append(_row(rec["trace"], rec["skip"], n, rec["capacity"], label,
                             rec["metrics"], source="replay", timing=rec))
    schemas.write_rows("hindsight/rows.csv", rows, out / "rows.csv")

    # Selection (every CV configuration).
    cv = json.loads((cache / "train" / "cv.json").read_text(encoding="utf-8"))
    selection = [{
        "capacity": r["capacity"], "objective": r["objective"], "family": r["family"],
        "config": r["config"], "params": r["params"], "cv_score": r["cv_score"],
        "cv_accuracy": r["cv_accuracy"], "cv_fold_scores": r["cv_fold_scores"],
        "cv_traces_scored": r["cv_traces_scored"], "family_best": r["family_best"],
        "selected": r["selected"], "n_train_windows": r["n_train_windows"],
        "wall_s": r["wall_s"], "load": r["load"]} for r in cv]
    schemas.write_rows("hindsight/selection.csv", selection, out / "selection.csv")

    # Families on slice 2, trajectory proxy (descriptive).
    proxies = json.loads((score_dir / "proxy.json").read_text(encoding="utf-8"))
    families = []
    for cap in capacities:
        for objective in hindsight.OBJECTIVES:
            for which in ("selected", *sorted({p["which"] for p in proxies} - {"selected"},
                                               key=lambda w: w)):
                cell = [p for p in proxies if p["capacity"] == cap
                        and p["objective"] == objective and p["which"] == which]
                if not cell:
                    continue
                head = [p["headroom"] for p in cell if p["headroom"] is not None]
                ci = (stats.paired_bootstrap(head, [0.0] * len(head), reps, REPS_SEED)
                      if head else None)
                pick = chosen[f"{cap}|{objective}|{which}"]
                families.append({
                    "capacity": cap, "objective": objective, "which": which,
                    "family": cell[0]["family"], "params": pick["params"],
                    "cv_score": pick["cv_score"], "n_traces": len(cell),
                    "proxy_headroom": ci.estimate if ci else None,
                    "proxy_headroom_lo": ci.low if ci else None,
                    "proxy_headroom_hi": ci.high if ci else None,
                    "label_accuracy": sum(p["accuracy"] for p in cell) / len(cell),
                    "action_shares": {a: sum(p["shares"][a] for p in cell) / len(cell)
                                      for a in hindsight.ACTIONS}})
    schemas.write_rows("hindsight/families.csv", families, out / "families.csv")

    # Summary and comparisons over the test slice.
    table = {(r["trace"], r["capacity"], r["policy"]): r for r in rows
             if r["skip_requests"] == skip2}
    policies = sorted({r["policy"] for r in rows if r["skip_requests"] == skip2})
    summary, comparisons = [], []
    for cap in capacities:
        for policy in policies:
            names = [t for t in usable if (t, cap, policy) in table and (t, cap, "none") in table]
            if not names:
                continue
            got = [table[(t, cap, policy)] for t in names]
            none = [table[(t, cap, "none")] for t in names]
            issuing = [r for r in got if r["total_prefetches"]]
            dev = stats.paired_bootstrap([r["device_ios"] for r in got],
                                         [r["device_ios"] for r in none], reps, REPS_SEED,
                                         statistic="ratio")
            moved = stats.paired_bootstrap([r["blocks_moved"] for r in got],
                                           [r["blocks_moved"] for r in none], reps, REPS_SEED,
                                           statistic="ratio")
            headroom = None
            if policy.startswith("routed_") and not policy.endswith(("_gnb", "_loco")):
                objective = policy.split("_", 1)[1]
                metric = OBJECTIVE_METRIC[objective]
                keys = [t for t in names if (t, cap, f"vcf_{objective}") in table
                        and (t, cap, f"oracle_{objective}") in table]
                sign = 1.0 if metric in HIGHER else -1.0

                def mean(label):
                    # Device I/Os are normalised by none's on the same trace,
                    # so every trace weighs the same (hit ratio and u0.5 are
                    # per block already).
                    def scale(t):
                        return (table[(t, cap, "none")]["device_ios"]
                                if metric == "device_ios" else 1.0)
                    return sum(sign * metric_value(table[(t, cap, label)], metric) / scale(t)
                               for t in keys) / len(keys)
                if keys and mean(f"oracle_{objective}") != mean(f"vcf_{objective}"):
                    headroom = ((mean(policy) - mean(f"vcf_{objective}"))
                                / (mean(f"oracle_{objective}") - mean(f"vcf_{objective}")))
            shares = [r["action_shares"] for r in got if r["action_shares"]]
            summary.append({
                "capacity": cap, "policy": policy, "n_traces": len(names),
                "mean_hit_ratio": sum(r["hit_ratio"] for r in got) / len(got),
                "mean_u05_per_block": sum(r["u05_per_block"] for r in got) / len(got),
                "mean_precision": (sum(r["precision"] for r in issuing) / len(issuing)
                                   if issuing else None),
                "mean_wasted_share": (sum(r["wasted_share"] for r in issuing) / len(issuing)
                                      if issuing else None),
                "device_io_ratio": dev.estimate, "device_io_ratio_lo": dev.low,
                "device_io_ratio_hi": dev.high,
                "blocks_moved_ratio": moved.estimate, "blocks_moved_ratio_lo": moved.low,
                "blocks_moved_ratio_hi": moved.high,
                "headroom_captured": headroom,
                "action_shares": ({a: sum(s[a] for s in shares) / len(shares)
                                   for a in hindsight.ACTIONS} if shares else None)})
        for cid, metric, policy, baseline, primary in COMPARISONS:
            names = [t for t in usable if (t, cap, policy) in table and (t, cap, baseline) in table]
            if not names:
                continue
            a = [metric_value(table[(t, cap, policy)], metric) for t in names]
            b = [metric_value(table[(t, cap, baseline)], metric) for t in names]
            comparisons.append({"capacity": cap, "comparison": cid, "metric": metric,
                                "policy": policy, "baseline": baseline, "primary": primary,
                                **compare(a, b, names, metric, reps)})
        for policy in policies:
            if policy == "none":
                continue
            for metric in ("device_ios", "blocks_moved"):
                names = [t for t in usable if (t, cap, policy) in table
                         and (t, cap, "none") in table and table[(t, cap, "none")][metric]]
                if not names:
                    continue
                a = [metric_value(table[(t, cap, policy)], metric) for t in names]
                b = [metric_value(table[(t, cap, "none")], metric) for t in names]
                comparisons.append({"capacity": cap, "comparison": "D", "metric": metric,
                                    "policy": policy, "baseline": "none", "primary": False,
                                    **compare(a, b, names, metric, reps)})
    schemas.write_rows("hindsight/summary.csv", summary, out / "summary.csv")
    schemas.write_rows("hindsight/comparisons.csv", comparisons, out / "comparisons.csv")
    checks["test_traces"] = usable
    checks["test_traces_na"] = [t for t in traces if t not in usable]
    (cache / "checks.json").write_text(json.dumps(checks, indent=1), encoding="utf-8")
    return checks
