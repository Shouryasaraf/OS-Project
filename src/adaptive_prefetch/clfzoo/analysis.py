"""Turn the zoo's raw records into the ``results/classifier/`` rows.

Every interval is ``prefetch_common.stats`` with 10,000 resamples and seed
42; every verdict is the rule of ``results/classifier/PREREGISTERED.md``
section 4 (:func:`verdict`). Stdlib only.
"""

from __future__ import annotations

import json
from statistics import mean, median

from prefetch_common import stats

from .. import corpora
from ..trace import CLASSES
from .zoo import BASELINE_FAMILY

REPS = 10_000
SEED = 42


def verdict(paired, cluster=None, *, higher_is_better: bool = True,
            null: float = 0.0) -> str:
    """``better`` / ``worse`` / ``no difference detected``.

    With a cluster interval both must lie on the same side of ``null``;
    without one (synthetic seed pairs have no clustering) the paired
    interval decides alone.
    """
    intervals = [paired] + ([cluster] if cluster is not None else [])
    above = all(ci.low > null for ci in intervals)
    below = all(ci.high < null for ci in intervals)
    if above:
        return "better" if higher_is_better else "worse"
    if below:
        return "worse" if higher_is_better else "better"
    return "no difference detected"


def _boot_mean(values, reps: int, seed: int):
    """Percentile bootstrap of a mean over independent units (seed pairs)."""
    return stats.cluster_bootstrap(values, list(range(len(values))), reps=reps, seed=seed)


# ------------------------------------------------------------- Part 1 rows

def tuning_rows(tuning) -> list[dict]:
    return [{"setting": r["setting"], "family": r["family"], "candidate": r["candidate"],
             "params": r["params"], "folds": r["folds"],
             "val_accuracy": r["val_accuracy"], "val_macro_f1": r["val_macro_f1"],
             "selected": bool(r["selected"]), "error": r.get("error"),
             "wall_s": r["wall_s"],
             "load_avg": r["load_avg"]}
            for r in sorted(tuning, key=lambda r: (r["setting"], r["family"], r["candidate"]))]


def eval_rows(records, chosen, families, settings, conditions_of,
              reps: int = REPS, seed: int = SEED) -> list[dict]:
    """One row per (setting, condition, family), paired against ``gnb_online``."""
    by = {}
    for r in records:
        by.setdefault((r["setting"], r["condition"], r["family"]), []).append(r)
    out = []
    for setting in settings:
        for condition in conditions_of(setting):
            base = sorted(by[(setting, condition, BASELINE_FAMILY)], key=lambda r: r["seed_pair"])
            for family in families:
                rows = sorted(by[(setting, condition, family)], key=lambda r: r["seed_pair"])
                acc = [r["accuracy"] for r in rows]
                f1 = [r["macro_f1"] for r in rows]
                acc_ci = _boot_mean(acc, reps, seed)
                f1_ci = _boot_mean(f1, reps, seed)
                confusion = [[sum(r["confusion"][i][j] for r in rows)
                              for j in range(len(CLASSES))] for i in range(len(CLASSES))]
                row = {
                    "setting": setting, "condition": condition, "family": family,
                    "params": chosen[(setting, family)]["params"],
                    "val_accuracy": chosen[(setting, family)]["val_accuracy"],
                    "n_seed_pairs": len(rows), "n_test_per_pair": rows[0]["n_test"],
                    "accuracy": acc_ci.estimate, "accuracy_lo": acc_ci.low,
                    "accuracy_hi": acc_ci.high, "accuracy_sd": _sd(acc),
                    "macro_f1": f1_ci.estimate, "macro_f1_lo": f1_ci.low,
                    "macro_f1_hi": f1_ci.high,
                    "ece": mean(r["ece"] for r in rows),
                    "brier": mean(r["brier"] for r in rows),
                    "confusion": {"rows_true_cols_pred": list(CLASSES), "counts": confusion},
                    "predict_us": median(r["predict_us"] for r in rows),
                    "train_ms": median(r["train_ms"] for r in rows),
                    "load_avg": mean(r["load_avg"] for r in rows if r["load_avg"] is not None),
                }
                if family == BASELINE_FAMILY:
                    row.update({k: None for k in (
                        "delta_accuracy", "delta_accuracy_lo", "delta_accuracy_hi",
                        "delta_macro_f1", "delta_macro_f1_lo", "delta_macro_f1_hi")})
                    row.update({"verdict_vs_gnb": "baseline", "wins": None,
                                "losses": None, "ties": None})
                else:
                    base_acc = [r["accuracy"] for r in base]
                    d = stats.paired_bootstrap(acc, base_acc, reps, seed)
                    df = stats.paired_bootstrap(f1, [r["macro_f1"] for r in base], reps, seed)
                    won, lost, tied = stats.wins(acc, base_acc)
                    row.update({"delta_accuracy": d.estimate, "delta_accuracy_lo": d.low,
                                "delta_accuracy_hi": d.high, "delta_macro_f1": df.estimate,
                                "delta_macro_f1_lo": df.low, "delta_macro_f1_hi": df.high,
                                "verdict_vs_gnb": verdict(d), "wins": won,
                                "losses": lost, "ties": tied})
                out.append(row)
    return out


def _sd(values) -> float:
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return (sum((v - m) ** 2 for v in values) / len(values)) ** 0.5


# ------------------------------------------------------------- Part 2 rows

def comparisons(runs, policies, capacities, reps: int = REPS, seed: int = SEED) -> list[dict]:
    """Paired and cluster comparisons over traces (``router_comparisons.csv``).

    Hit ratio against ``sequential`` and against ``adaptive`` with the
    shipped GNB (``gnb_online``), and device I/Os and blocks moved as ratios
    to ``none``, for every routed policy at each capacity.
    """
    table = {(r["trace"], r["capacity"], r["policy"]): r for r in runs}
    out = []
    plan = []
    for policy in policies:
        plan.append(("hit_ratio", "difference", policy, "sequential", True))
        if policy != BASELINE_FAMILY:
            plan.append(("hit_ratio", "difference", policy, BASELINE_FAMILY, True))
        plan.append(("device_ios", "ratio", policy, "none", False))
        plan.append(("blocks_moved", "ratio", policy, "none", False))
    for capacity in capacities:
        for metric, statistic, policy, baseline, higher in plan:
            traces = sorted({t for (t, c, p) in table if c == capacity and p == policy}
                            & {t for (t, c, p) in table if c == capacity and p == baseline})
            if not traces:
                continue
            a = [table[(t, capacity, policy)][metric] for t in traces]
            b = [table[(t, capacity, baseline)][metric] for t in traces]
            paired = stats.paired_bootstrap(a, b, reps, seed, statistic=statistic)
            effects = ([x - y for x, y in zip(a, b)] if statistic == "difference"
                       else [x / y for x, y in zip(a, b)])
            weights = stats.twin_weights(traces, corpora.TWIN_GROUPS)
            clusters = [corpora.CORPUS_OF(t) for t in traces]
            cluster = stats.cluster_bootstrap(effects, clusters, weights, reps, seed)
            null = 0.0 if statistic == "difference" else 1.0
            won, lost, tied = stats.wins(a, b) if higher else stats.wins(b, a)
            out.append({
                "capacity": capacity, "metric": metric, "statistic": statistic,
                "policy": policy, "baseline": baseline, "n": len(traces),
                "estimate": paired.estimate, "ci_lo": paired.low, "ci_hi": paired.high,
                "cluster_estimate": cluster.estimate, "cluster_ci_lo": cluster.low,
                "cluster_ci_hi": cluster.high,
                "verdict": verdict(paired, cluster, higher_is_better=higher, null=null),
                "wins": won, "losses": lost, "ties": tied, "reps": reps, "seed": seed,
            })
    return out


def summary(runs, policies, capacities) -> list[dict]:
    """Per (capacity, policy): means over traces, with routing agreement."""
    table = {(r["trace"], r["capacity"], r["policy"]): r for r in runs}
    out = []
    for capacity in capacities:
        for policy in policies:
            rows = [r for (t, c, p), r in sorted(table.items()) if c == capacity and p == policy]
            if not rows:
                continue
            none = {r["trace"]: r for (t, c, p), r in table.items()
                    if c == capacity and p == "none"}
            agree = [r["agreement_with_gnb"] for r in rows if r["agreement_with_gnb"] is not None]
            issued = [r for r in rows if r["total_prefetches"]]
            out.append({
                "capacity": capacity, "policy": policy, "n_traces": len(rows),
                "mean_hit_ratio": mean(r["hit_ratio"] for r in rows),
                "mean_precision": (mean(r["precision"] for r in issued) if issued else None),
                "mean_wasted_share": (mean(1 - r["precision"] for r in issued)
                                      if issued else None),
                "device_io_ratio": mean(r["device_ios"] / none[r["trace"]]["device_ios"]
                                        for r in rows),
                "blocks_moved_ratio": mean(r["blocks_moved"] / none[r["trace"]]["blocks_moved"]
                                           for r in rows),
                "mean_agreement_with_gnb": mean(agree) if agree else None,
                "min_agreement_with_gnb": min(agree) if agree else None,
                "traces_identical_to_gnb": (sum(r["label_key"] == r["gnb_label_key"]
                                                for r in rows)
                                            if rows[0]["label_key"] is not None else None),
                "share_sequential": _mean_share(rows, "sequential"),
                "share_strided": _mean_share(rows, "strided"),
                "share_random": _mean_share(rows, "random"),
                "share_mixed": _mean_share(rows, "mixed"),
            })
    return out


def _mean_share(rows, label):
    values = [r[f"share_{label}"] for r in rows if r[f"share_{label}"] is not None]
    return mean(values) if values else None


def dumps(value) -> str:
    return json.dumps(value, sort_keys=True)
