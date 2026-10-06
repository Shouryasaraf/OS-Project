"""Validation-only policy search followed by a frozen held-out replay."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .model import FastScorer, load_artifacts
from .simulator import Cache, Counters, Policy, paired_bootstrap, replay

SCENARIO_COSTS = (0.0, 0.01, 0.05, 0.25)
ADAPTIVE_PENALTIES = (0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2)
GATE_THRESHOLDS = (0.01, 0.02, 0.05, 0.1)


def _score(row: dict, cost: float) -> float:
    return (row["hits"] - cost * row["issued"]) / row["views"]


def _real_session_demo(test: pd.DataFrame, index, model, policy: Policy) -> dict:
    """Show the first held-out session with at least five views, without picking a success."""
    lengths = test.groupby("session_id", sort=False).size()
    eligible_ids = lengths.index[lengths.ge(5)]
    if len(eligible_ids) == 0:
        raise ValueError("test prefix contains no session with five views")
    session_id = int(eligible_ids[0])
    rows = test.loc[test.session_id.eq(session_id)]
    scorer = FastScorer(model)
    cache = Cache(8, Counters())
    steps = []
    for row in rows.itertuples(index=False):
        hit = cache.demand(int(row.itemid))
        ranked = scorer.score(row, index)
        depth = policy.depth(ranked)
        prior_issued = cache.counters.issued
        chosen = [item for item, _ in ranked[:depth]]
        for item in chosen:
            cache.prefetch(item)
        steps.append({
            "timestamp": int(row.timestamp), "viewed_item": int(row.itemid),
            "cache_hit": hit, "selected_depth": depth,
            "issued_now": cache.counters.issued - prior_issued,
            "prefetched_candidates": chosen,
            "top_scored_candidates": [[item, round(prob, 4)] for item, prob in ranked[:5]],
            "actual_next_item_after_decision": int(row.next_item) if bool(row.eligible) else None,
        })
    return {"selection_rule": "first test-prefix session with at least five views",
            "session_id": session_id, "policy": policy.name, "capacity": 8,
            "steps": steps, "observed_hits": cache.counters.hits,
            "prefetches_issued": cache.counters.issued}


def run_benchmark(prepared: str | Path, model_dir: str | Path,
                  output_dir: str | Path, validation_views: int = 50000,
                  test_views: int = 100000) -> dict:
    if validation_views <= 0 or test_views <= 0:
        raise ValueError("validation_views and test_views must be positive")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    views = pd.read_csv(prepared)
    index, model = load_artifacts(model_dir)
    validation = views.loc[views.split.eq(2)]
    test = views.loc[views.split.eq(3)]
    if len(validation) < validation_views or len(test) < test_views:
        raise ValueError("requested replay size exceeds a chronological split")
    policies = ([Policy(f"fixed{k}", "fixed", k) for k in (0, 1, 2, 4, 8)]
                + [Policy(f"adaptive_{penalty:g}", "adaptive", penalty)
                   for penalty in ADAPTIVE_PENALTIES]
                + [Policy(f"gate_{threshold:g}", "gate", threshold)
                   for threshold in GATE_THRESHOLDS])
    validation_rows, validation_extra, _ = replay(
        validation, index, model, policies, capacity=8, max_views=validation_views)
    val_map = {row["policy"]: row for row in validation_rows}
    selection = {}
    selected_names = {"fixed0"}
    for cost in SCENARIO_COSTS:
        best = {}
        for family, prefix in (("fixed", "fixed"), ("adaptive", "adaptive_"), ("gate", "gate_")):
            names = [name for name in val_map if name.startswith(prefix)]
            name = max(names, key=lambda n: (_score(val_map[n], cost), -val_map[n]["issued"], n))
            best[family] = name
            selected_names.add(name)
        selection[str(cost)] = best
    frozen = [policy for policy in policies if policy.name in selected_names]
    test_rows = []
    comparisons = []
    demo = None
    coverages = {}
    for capacity in (4, 8, 16):
        rows, extra, _ = replay(test, index, model, frozen, capacity,
                                     max_views=test_views, with_sessions=True)
        test_rows.extend(rows)
        coverages[str(capacity)] = extra["oracle_candidate_coverage_at_8"]
        if capacity == 8:
            demo_policy_name = selection["0.01"]["adaptive"]
            demo_policy = next(policy for policy in frozen if policy.name == demo_policy_name)
            demo = _real_session_demo(test.iloc[:test_views], index, model, demo_policy)
        by_name = {row["policy"]: row for row in rows}
        for cost in SCENARIO_COSTS:
            names = selection[str(cost)]
            fixed = by_name[names["fixed"]]
            adaptive = by_name[names["adaptive"]]
            gate = by_name[names["gate"]]
            low, high = paired_bootstrap(extra["sessions"], names["adaptive"],
                                         names["fixed"], cost=cost)
            comparisons.append({
                "capacity": capacity, "cost_per_prefetch": cost,
                "fixed_policy": fixed["policy"], "adaptive_policy": adaptive["policy"],
                "gate_policy": gate["policy"],
                "fixed_utility": _score(fixed, cost),
                "adaptive_utility": _score(adaptive, cost),
                "gate_utility": _score(gate, cost),
                "adaptive_minus_fixed_utility": _score(adaptive, cost) - _score(fixed, cost),
                "paired_95pct_low": low, "paired_95pct_high": high,
                "fixed_hit_ratio": fixed["hit_ratio"],
                "adaptive_hit_ratio": adaptive["hit_ratio"],
                "fixed_issued_per_view": fixed["issued_per_view"],
                "adaptive_issued_per_view": adaptive["issued_per_view"],
            })
    pd.DataFrame([{k: v for k, v in row.items() if k != "depth_counts"}
                  for row in validation_rows]).to_csv(output_dir / "validation_metrics.csv", index=False)
    pd.DataFrame([{k: v for k, v in row.items() if k != "depth_counts"}
                  for row in test_rows]).to_csv(output_dir / "test_metrics.csv", index=False)
    pd.DataFrame(comparisons).to_csv(output_dir / "comparisons.csv", index=False)
    (output_dir / "demo.json").write_text(json.dumps(demo, indent=2), encoding="utf-8")
    summary = {
        "validation_views": validation_views, "test_views": test_views,
        "validation_capacity": 8, "test_capacities": [4, 8, 16],
        "cost_scenarios": list(SCENARIO_COSTS),
        "cost_note": "Object cost is an assumed sensitivity parameter, not observed bytes or latency.",
        "validation_candidate_coverage_at_8": validation_extra["oracle_candidate_coverage_at_8"],
        "validation_ranking_hit_at_1": validation_extra["ranking_hit_at_1"],
        "validation_ranking_mrr_at_8": validation_extra["ranking_mrr_at_8"],
        "test_candidate_coverage_at_8": coverages,
        "validation_selection": selection,
        "comparisons": comparisons,
    }
    (output_dir / "benchmark.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
