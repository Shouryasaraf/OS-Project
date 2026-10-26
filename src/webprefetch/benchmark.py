"""Validation-only policy search followed by a frozen held-out replay."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .model import FastScorer, load_artifacts
from .simulator import (Cache, Counters, Policy, TransitionScorer, break_even_waste,
                        complete_views, expected_calibration_error, paired_bootstrap,
                        replay, replay_capacities)

SCENARIO_COSTS = (0.0, 0.01, 0.05, 0.25)
ADAPTIVE_PENALTIES = (0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2)
GATE_THRESHOLDS = (0.01, 0.02, 0.05, 0.1)


def _score(row: dict, cost: float) -> float:
    return (row["hits"] - cost * row["issued"]) / row["views"]


def _reliability_svg(rows: list[dict]) -> str:
    """Dependency-free candidate-level reliability curve."""
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 620 450" role="img" aria-label="Candidate probability reliability curve">',
             '<rect width="620" height="450" fill="#fff"/>',
             '<text x="55" y="28" font-size="19" font-family="sans-serif">Candidate-level reliability</text>',
             '<path d="M65 370 L380 370 M65 370 L65 55" stroke="#333" fill="none" stroke-width="2"/>',
             '<path d="M65 370 L380 55" stroke="#aaa" fill="none" stroke-dasharray="5 5"/>',
             '<text x="145" y="420" font-size="14" font-family="sans-serif">Mean predicted probability</text>',
             '<text transform="translate(20 300) rotate(-90)" font-size="14" font-family="sans-serif">Observed frequency</text>']
    # Plot only bins with observations; markers scale with bin population.
    for split, color in (("validation", "#2463a5"), ("test", "#d06a2c")):
        selected = [row for row in rows if row["split"] == split and row["count"]]
        points = []
        for row in selected:
            x = 65 + 315 * row["mean_probability"]
            y = 370 - 315 * row["observed_frequency"]
            points.append(f"{x:.2f},{y:.2f}")
            radius = min(7, max(2, row["count"] ** .25 / 4))
            parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius:.2f}" fill="{color}"/>')
        if points:
            parts.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2"/>')
        ece = selected[0]["ece"] if selected else 0
        y = 80 if split == "validation" else 105
        parts.append(f'<text x="400" y="{y}" fill="{color}" font-size="13" font-family="sans-serif">{split} ECE {ece:.5f}</text>')
    parts += ['<text x="62" y="390" font-size="12">0</text>',
              '<text x="375" y="390" font-size="12">1</text>',
              '<text x="48" y="60" font-size="12">1</text>', '</svg>']
    return "\n".join(parts)


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
                  output_dir: str | Path, validation_views: int = 0,
                  test_views: int = 0) -> dict:
    if validation_views < 0 or test_views < 0:
        raise ValueError("view limits must be non-negative; zero means full split")
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
        validation, index, model, policies, capacity=8, max_views=validation_views,
        with_sessions=True)
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
    capacities = (2, 4, 8, 16, 32)
    test_replays = replay_capacities(test, index, model, frozen, capacities, test_views)
    break_even = []
    for cost in SCENARIO_COSTS:
        names = selection[str(cost)]
        break_even.append({"scenario_cost": cost, "adaptive_policy": names["adaptive"],
                           "fixed_policy": names["fixed"],
                           **break_even_waste(validation_extra["sessions"], names["adaptive"], names["fixed"])})
    for capacity in capacities:
        rows, extra = test_replays[capacity]
        test_rows.extend(rows)
        coverages[str(capacity)] = extra["oracle_candidate_coverage_at_8"]
        if capacity == 8:
            demo_policy_name = selection["0.01"]["adaptive"]
            demo_policy = next(policy for policy in frozen if policy.name == demo_policy_name)
            demo = _real_session_demo(test if not test_views else test.iloc[:test_views], index, model, demo_policy)
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
                "no_prefetch_hit_ratio": by_name["fixed0"]["hit_ratio"],
                "adaptive_minus_no_prefetch_utility": _score(adaptive, cost) - _score(by_name["fixed0"], cost),
                "fixed_issued_per_view": fixed["issued_per_view"],
                "adaptive_issued_per_view": adaptive["issued_per_view"],
                "candidate_coverage_at_8": extra["oracle_candidate_coverage_at_8"],
            })
    # The historical default prefixes are diagnostics, never used for final selection.
    subset_test, subset_textra, _ = replay(test, index, model, frozen, 8,
                                          min(100000, test_views) if test_views else 100000)
    subset_map = {r["policy"]: r for r in subset_test}
    subset_comparisons = []
    for cost in SCENARIO_COSTS:
        chosen = selection[str(cost)]
        a, b = subset_map[chosen["adaptive"]], subset_map[chosen["fixed"]]
        full = next(r for r in comparisons if r["capacity"] == 8 and r["cost_per_prefetch"] == cost)
        subset_comparisons.append({"cost": cost, "subset_views": subset_textra["replayed_views"],
            "full_views": test_replays[8][1]["replayed_views"],
            "subset_delta": _score(a,cost)-_score(b,cost),
            "full_delta": full["adaptive_minus_fixed_utility"],
            "subset_coverage": subset_textra["oracle_candidate_coverage_at_8"],
            "full_coverage": full["candidate_coverage_at_8"]})
    pd.DataFrame([{k: v for k, v in row.items() if k != "depth_counts"}
                  for row in validation_rows]).to_csv(output_dir / "validation_metrics.csv", index=False)
    pd.DataFrame([{k: v for k, v in row.items() if k != "depth_counts"}
                  for row in test_rows]).to_csv(output_dir / "test_metrics.csv", index=False)
    pd.DataFrame(comparisons).to_csv(output_dir / "comparisons.csv", index=False)
    pd.DataFrame(subset_comparisons).to_csv(output_dir / "subset_vs_full.csv", index=False)
    pd.DataFrame(break_even).to_csv(output_dir / "break_even.csv", index=False)
    curve = []
    for cost in SCENARIO_COSTS:
        names = selection[str(cost)]
        a, b = validation_extra["sessions"][names["adaptive"]], validation_extra["sessions"][names["fixed"]]
        dh = sum(a[k][0]-b[k][0] for k in a)
        dw = sum((a[k][1]-a[k][3])-(b[k][1]-b[k][3]) for k in a)
        for point in range(0, 301, 5):
            curve.append({"scenario_cost":cost,"waste_cost":point/1000,
                          "validation_utility_delta_per_view":(dh-(point/1000)*dw)/validation_extra["replayed_views"]})
    pd.DataFrame(curve).to_csv(output_dir / "break_even_curve.csv", index=False)
    baseline_rows, baseline_extra, _ = replay(test, index, model,
        [Policy("transition_top1", "fixed", 1)], 8, test_views, scorer=TransitionScorer())
    pd.DataFrame(baseline_rows).to_csv(output_dir / "transition_baseline.csv", index=False)
    calibration_rows = []
    for split_name, bins in (("validation", validation_extra["calibration_bins"]),
                             ("test", test_replays[8][1]["calibration_bins"])):
        for i,(count, sum_p, positives) in enumerate(bins):
            calibration_rows.append({"split":split_name,"bin":i,"count":count,
                "mean_probability":sum_p/count if count else 0.0,
                "observed_frequency":positives/count if count else 0.0,
                "ece":expected_calibration_error(bins)})
    pd.DataFrame(calibration_rows).to_csv(output_dir / "calibration.csv", index=False)
    (output_dir / "reliability.svg").write_text(_reliability_svg(calibration_rows), encoding="utf-8")
    cohort_counts = {}
    selected_test = complete_views(test, test_views)
    for row in selected_test.itertuples(index=False):
        if not bool(row.eligible):
            continue
        month = pd.Timestamp(int(row.timestamp), unit="ms", tz="UTC").strftime("%Y-%m")
        source_status = "seen_source" if int(row.itemid) in index.global_count else "cold_source"
        target_status = "seen_target" if int(row.next_item) in index.global_count else "cold_target"
        covered = int(any(item == row.next_item for item, _ in index.candidates(int(row.itemid))[:8]))
        for dimension, cohort in (("month", month), ("source", source_status), ("target", target_status)):
            record = cohort_counts.setdefault((dimension, cohort), [0, 0])
            record[0] += 1
            record[1] += covered
    pd.DataFrame([{"dimension":d,"cohort":c,"eligible_pairs":v[0],
                   "candidate_coverage_at_8":v[1]/v[0]} for (d,c),v in sorted(cohort_counts.items())]
                 ).to_csv(output_dir / "cohorts.csv", index=False)
    from .diagnostics import evaluate_models
    model_diagnostics = evaluate_models(views, index, model, model_dir, output_dir)
    (output_dir / "demo.json").write_text(json.dumps(demo, indent=2), encoding="utf-8")
    summary = {
        "validation_views": validation_extra["replayed_views"], "test_views": test_replays[8][1]["replayed_views"],
        "validation_capacity": 8, "test_capacities": list(capacities),
        "cost_scenarios": list(SCENARIO_COSTS),
        "cost_note": "Object cost is an assumed sensitivity parameter, not observed bytes or latency.",
        "validation_candidate_coverage_at_8": validation_extra["oracle_candidate_coverage_at_8"],
        "validation_ranking_hit_at_1": validation_extra["ranking_hit_at_1"],
        "validation_ranking_mrr_at_8": validation_extra["ranking_mrr_at_8"],
        "test_candidate_coverage_at_8": coverages,
        "validation_ece": expected_calibration_error(validation_extra["calibration_bins"]),
        "test_ece": expected_calibration_error(test_replays[8][1]["calibration_bins"]),
        "subset_vs_full": subset_comparisons, "break_even_waste": break_even,
        "transition_top1": baseline_rows[0],
        "model_diagnostics": model_diagnostics,
        "validation_selection": selection,
        "comparisons": comparisons,
    }
    (output_dir / "benchmark.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_sensitivity(raw: str | Path, processed: str | Path,
                    model_dir: str | Path, output_dir: str | Path) -> None:
    """Isolate segmentation sensitivity with a frozen trained model/policies.

    Uses the historical view budgets for this expensive supplemental analysis;
    full-period results remain in benchmark.json and comparisons.csv.
    """
    from .data import prepare
    output_dir = Path(output_dir)
    summary = json.loads((output_dir / "benchmark.json").read_text())
    index, model = load_artifacts(model_dir)
    rows = []
    for gap in (15, 30, 60):
        if gap == 30:
            views = pd.read_csv(Path(processed) / "views.csv.gz")
        else:
            directory = Path(processed) / f"gap{gap}"
            prepare(raw, directory, gap)
            views = pd.read_csv(directory / "views.csv.gz")
        test = views.loc[views.split.eq(3)]
        names = summary["validation_selection"]["0.01"]
        policies = [Policy(names["fixed"], "fixed", int(names["fixed"].replace("fixed", ""))),
                    Policy(names["adaptive"], "adaptive", float(names["adaptive"].split("_")[1]))]
        result, extra, _ = replay(test, index, model, policies, 8, 100000)
        mapping = {row["policy"]: row for row in result}
        fixed, adaptive = mapping[names["fixed"]], mapping[names["adaptive"]]
        rows.append({"session_gap_minutes":gap,"frozen_from_gap_minutes":30,
                     "approximate_view_budget":100000,"replayed_views":extra["replayed_views"],
                     "eligible_pairs":extra["oracle_eligible_pairs"],
                     "candidate_coverage_at_8":extra["oracle_candidate_coverage_at_8"],
                     "fixed_hit_ratio":fixed["hit_ratio"],"adaptive_hit_ratio":adaptive["hit_ratio"],
                     "adaptive_minus_fixed_utility_at_0.01":_score(adaptive,.01)-_score(fixed,.01)})
    pd.DataFrame(rows).to_csv(output_dir / "session_gap_sensitivity.csv", index=False)
