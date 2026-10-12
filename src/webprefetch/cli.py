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
    model, summary = train_ranker(views, index, diagnostic_dir=MODELS)
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
        f"Controller tuning: {summary['validation_views']:,} validation views; "
        f"final replay: {summary['test_views']:,} untouched test views.",
        "Each session has an independent browser object cache; capacities are 2, 4, 8, 16, and 32 objects.",
        "", "## Frozen-policy comparisons", "",
        "The `cost` values are hypothetical useful-hit equivalents per issued object; "
        "the trace contains neither page bytes nor latency. A positive delta means the "
        "adaptive policy beats the validation-selected fixed policy under that assumption.", "",
        "| Cache | Cost | Fixed | Adaptive | Hit-rate delta | Coverage@8 ceiling | Prefetches/view (fixed to adaptive) | Utility delta | 95% paired CI |",
        "|---:|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["comparisons"]:
        lines.append(
            f"| {row['capacity']} | {row['cost_per_prefetch']:g} | {row['fixed_policy']} | "
            f"{row['adaptive_policy']} | {row['adaptive_hit_ratio'] - row['fixed_hit_ratio']:+.4%} | "
            f"{row['candidate_coverage_at_8']:.3%} | "
            f"{row['fixed_issued_per_view']:.3f} to {row['adaptive_issued_per_view']:.3f} | "
            f"{row['adaptive_minus_fixed_utility']:+.5f} | "
            f"[{row['paired_95pct_low']:+.5f}, {row['paired_95pct_high']:+.5f}] |"
        )
    lines += [
        "", "## Interpretation", "",
        "These are object-hit and request-count outcomes, not measured speedups. "
        "Candidate coverage bounds immediate next-item retrieval among eligible pairs; "
        "it does not bound ordinary demand-cache revisits.",
        "No policy was selected using test data. See `validation_metrics.csv`, "
        "`test_metrics.csv`, `comparisons.csv`, and `demo.json` for the complete evidence.",
        "", "## Limitations", "",
        "The log lacks page size, response timing, cache headers, browser behavior, "
        "and visitor-side idle time. Equal-size objects and immediate prefetch completion "
        "are simulation assumptions. Session boundaries are heuristic. One retailer dataset "
        "does not establish cross-site generalisation or measured page-load speedups.", "",
    ]
    lines += ["", "## Diagnostics and sensitivity", "",
        f"Candidate coverage@8 on full test: {summary['test_candidate_coverage_at_8']['8']:.3%}; "
        "the ceiling appears beside each result in `comparisons.csv`.",
        f"Candidate-level ECE: validation {summary['validation_ece']:.5f}, "
        f"test {summary['test_ece']:.5f}; reliability bins are in `calibration.csv`. "
        "Candidate scores do not form a normalized next-item distribution. "
        "See `reliability.svg` for the plotted curve.",
        f"History-only top-1 transition cache baseline: {summary['transition_top1']['hit_ratio']:.3%} hits, "
        f"{summary['transition_top1']['issued_per_view']:.3f} issued/view at capacity 8; "
        "details in `transition_baseline.csv`. "
        "Context ablation and a gradient-boosting diagnostic are in `models/training_summary.json`. "
        "Boosting is diagnostic and did not select the controller.",
        "At capacity 8, the history-only transition baseline can outperform the "
        "logistic fixed-1 policy in raw hit rate; compare `transition_baseline.csv` "
        "with `test_metrics.csv`. This is a diagnostic, not a test-selected replacement.",
        "`subset_vs_full.csv` compares old prefix budgets with full-period frozen policies. "
        "`break_even.csv` and `break_even_curve.csv` use cost per unused prefetch on validation. "
        "`cohorts.csv` examines cold-start and chronological retrieval coverage.",
        "", "## Positive, negative, and uncertain results", ""]
    for row in summary["comparisons"]:
        low, high = row["paired_95pct_low"], row["paired_95pct_high"]
        verdict = "wins" if low > 0 else "loses" if high < 0 else "is within paired-bootstrap noise"
        against_none = row.get('adaptive_minus_no_prefetch_utility')
        none_note = (f"; versus no prefetch {against_none:+.5f}" if against_none is not None else "")
        lines.append(f"- Cache {row['capacity']}, cost {row['cost_per_prefetch']:g}: adaptive {verdict}; "
                     f"utility delta {row['adaptive_minus_fixed_utility']:+.5f}, "
                     f"95% CI [{low:+.5f}, {high:+.5f}]{none_note}.")
    (OUTPUTS / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("download", "prepare", "eda", "train", "benchmark", "demo", "run", "sensitivity", "serve", "export-demo"))
    parser.add_argument("--raw", type=Path, default=RAW)
    parser.add_argument("--validation-views", type=int, default=0)
    parser.add_argument("--test-views", type=int, default=0)
    parser.add_argument("--session-gap-minutes", type=int, default=30)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if args.command in ("download", "run"):
        print(f"Dataset ready: {acquire(args.raw)}")
    if args.command in ("prepare", "run"):
        print(json.dumps(prepare(args.raw, PROCESSED, args.session_gap_minutes), indent=2))
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

    if args.command in ("sensitivity", "run"):
        # Supplemental segmentation replay uses a frozen 30-minute model.
        from .benchmark import run_sensitivity
        run_sensitivity(args.raw, PROCESSED, MODELS, OUTPUTS)
    if args.command == "serve":
        from .demo_server import serve
        serve(Path.cwd(), args.host, args.port)
    if args.command == "export-demo":
        from .demo_server import export_demo
        print(f"Bounded static data written to {export_demo(Path.cwd())}")


if __name__ == "__main__":
    main()
