"""Reproducible commands for the real-data workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .benchmark import run_benchmark
from .data import acquire, eda_markdown, load_views, prepare
from .model import fit_index, save_artifacts, train_ranker

RAW = Path("data/raw/events.csv")
PROCESSED = Path("data/processed")
MODELS = Path("models")
OUTPUTS = Path("outputs")


def _eda() -> None:
    profile = json.loads((PROCESSED / "profile.json").read_text(encoding="utf-8"))
    views = load_views(PROCESSED / "views.csv.gz")
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    (OUTPUTS / "eda.md").write_text(eda_markdown(profile, views), encoding="utf-8")
    print(f"EDA written to {OUTPUTS / 'eda.md'}")


def _train() -> None:
    views = load_views(PROCESSED / "views.csv.gz")
    index = fit_index(views)
    model, summary = train_ranker(views, index)
    save_artifacts(MODELS, index, model, summary)
    print(json.dumps(summary, indent=2))


def _report(summary: dict) -> None:
    profile = json.loads((PROCESSED / "profile.json").read_text(encoding="utf-8"))
    training = json.loads((MODELS / "training_summary.json").read_text(encoding="utf-8"))
    lines = [
        "# Real-data product-page prefetch benchmark", "",
        "RetailRocket Kaggle events, version 2; see the dataset link in the README.",
        f"Raw events: {profile['raw_rows']:,}; view requests: {profile['views']:,}; "
        f"eligible next-view pairs: {profile['eligible_next_view_pairs']:,}.",
        f"Raw CSV SHA-256: `{profile['raw_sha256']}`.",
        f"Ranker trained on {training['training_decisions']:,} real decisions; "
        f"candidate coverage: {training['candidate_coverage']:.3%}; "
        f"selected logistic C: {training['selected_C']}.",
        f"Validation next-view ranking: coverage@8 {training['validation_coverage_at_8']:.3%}, "
        f"hit@1 {training['validation_hit_at_1']:.3%}, MRR@8 {training['validation_mrr_at_8']:.4f}.",
        f"Controller tuning: first {summary['validation_views']:,} validation views; "
        f"final replay: first {summary['test_views']:,} untouched test views.",
        "Each session has an independent browser object cache; capacities are 4, 8, and 16 objects.",
        "", "## Frozen-policy comparisons", "",
        "The `cost` values are hypothetical useful-hit equivalents per issued object; "
        "the trace contains neither page bytes nor latency. A positive delta means the "
        "adaptive policy beats the validation-selected fixed policy under that assumption.", "",
        "| Cache | Cost | Fixed | Adaptive | Hit-rate delta | Prefetches/view (fixed to adaptive) | Utility delta | 95% paired CI |",
        "|---:|---:|---|---|---:|---:|---:|---:|",
    ]
    for row in summary["comparisons"]:
        lines.append(
            f"| {row['capacity']} | {row['cost_per_prefetch']:g} | {row['fixed_policy']} | "
            f"{row['adaptive_policy']} | {row['adaptive_hit_ratio'] - row['fixed_hit_ratio']:+.4%} | "
            f"{row['fixed_issued_per_view']:.3f} to {row['adaptive_issued_per_view']:.3f} | "
            f"{row['adaptive_minus_fixed_utility']:+.5f} | "
            f"[{row['paired_95pct_low']:+.5f}, {row['paired_95pct_high']:+.5f}] |"
        )
    lines += [
        "", "## Interpretation", "",
        "These are object-hit and request-count outcomes, not measured speedups. "
        "Candidate coverage is a strict upper bound on next-page prefetch accuracy; "
        "ineligible requests still participate in cache replay.",
        "No policy was selected using test data. See `validation_metrics.csv`, "
        "`test_metrics.csv`, `comparisons.csv`, and `demo.json` for the complete evidence.",
        "", "## Limitations", "",
        "The log lacks page size, response timing, cache headers, browser behavior, "
        "and visitor-side idle time. Equal-size objects and immediate prefetch completion "
        "are simulation assumptions. Session boundaries are heuristic, and only the "
        "first N time-ordered requests of each holdout split are replayed by default.", "",
    ]
    (OUTPUTS / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("download", "prepare", "eda", "train", "benchmark", "demo", "run"))
    parser.add_argument("--raw", type=Path, default=RAW)
    parser.add_argument("--validation-views", type=int, default=50000)
    parser.add_argument("--test-views", type=int, default=100000)
    args = parser.parse_args()
    if args.command in ("download", "run"):
        print(f"Dataset ready: {acquire(args.raw)}")
    if args.command in ("prepare", "run"):
        print(json.dumps(prepare(args.raw, PROCESSED), indent=2))
    if args.command in ("eda", "run"):
        _eda()
    if args.command in ("train", "run"):
        _train()
    if args.command in ("benchmark", "run"):
        summary = run_benchmark(PROCESSED / "views.csv.gz", MODELS, OUTPUTS,
                                args.validation_views, args.test_views)
        _report(summary)
        print(f"Benchmark report written to {OUTPUTS / 'report.md'}")
    if args.command == "demo":
        demo = json.loads((OUTPUTS / "demo.json").read_text(encoding="utf-8"))
        print(f"Real held-out session {demo['session_id']}; policy {demo['policy']}; "
              f"hits {demo['observed_hits']}; issued {demo['prefetches_issued']}")
        for step in demo["steps"]:
            print(json.dumps(step))


if __name__ == "__main__":
    main()
