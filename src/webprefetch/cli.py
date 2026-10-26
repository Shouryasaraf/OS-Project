"""Reproducible commands for the real-data workflow."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .benchmark import run_benchmark
from .data import acquire, eda_markdown, load_views, prepare
from .model import fit_index, save_artifacts, train_ranker
from .paths import MODELS, OUTPUTS, PROCESSED, RAW

COMMANDS = ("download", "prepare", "eda", "train", "benchmark", "demo", "run", "sensitivity",
            "serve", "export-demo", "parity", "freshness", "decompose", "learning-curve")
#: ``--quick``: the bounded smoke path (complete sessions within these prefixes).
QUICK_VALIDATION_VIEWS = 1000
QUICK_TEST_VIEWS = 2000
#: Full runs write the paper's tables here; partial runs never do.
RESULTS = Path("results/web")


def _eda(processed: Path, outputs: Path) -> None:
    profile = json.loads((processed / "profile.json").read_text(encoding="utf-8"))
    views = load_views(processed / "views.csv.gz")
    outputs.mkdir(parents=True, exist_ok=True)
    (outputs / "eda.md").write_text(eda_markdown(profile, views), encoding="utf-8")
    print(f"EDA written to {outputs / 'eda.md'}")


def _train(processed: Path, models: Path, novel: bool) -> None:
    views = load_views(processed / "views.csv.gz")
    index = fit_index(views)
    model, summary = train_ranker(views, index, diagnostic_dir=models)
    save_artifacts(models, index, model, summary)
    print(json.dumps(summary, indent=2))
    if novel:
        from .scorers import save_novel_ranker, train_novel_ranker
        novel_model, novel_summary = train_novel_ranker(views, index)
        print(f"Relabelled ranker written to {save_novel_ranker(models, novel_model, novel_summary)}")


def _pct(value) -> str:
    return "n/a" if value is None else f"{value:.3%}"


def _comparison_lines(rows: list[dict]) -> list[str]:
    lines = ["| Cache | Cost | Best fixed | Adaptive | Hit-rate delta | Prefetches/view (fixed to adaptive) "
             "| vs best fixed [95% CI] | vs no prefetch [95% CI] | Verdict |",
             "|---:|---:|---|---|---:|---:|---:|---:|---|"]
    for row in rows:
        lines.append(
            f"| {row['capacity']} | {row['cost_per_prefetch']:g} | {row['fixed_policy']} | "
            f"{row['adaptive_policy']} | {row['adaptive_hit_ratio'] - row['fixed_hit_ratio']:+.4%} | "
            f"{row['fixed_issued_per_view']:.3f} to {row['adaptive_issued_per_view']:.3f} | "
            f"{row['adaptive_minus_fixed_utility']:+.5f} "
            f"[{row['paired_95pct_low']:+.5f}, {row['paired_95pct_high']:+.5f}] | "
            f"{row['adaptive_minus_no_prefetch_utility']:+.5f} "
            f"[{row['no_prefetch_95pct_low']:+.5f}, {row['no_prefetch_95pct_high']:+.5f}] | "
            f"{row['label']} |")
    return lines


def _report(summary: dict, processed: Path, models: Path, outputs: Path) -> None:
    profile = json.loads((processed / "profile.json").read_text(encoding="utf-8"))
    training = json.loads((models / "training_summary.json").read_text(encoding="utf-8"))
    primary = summary["primary_scorer"]
    calibration = summary["calibration"].get(primary, {})
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
        f"final replay: {summary['test_views']:,} untouched test views"
        + (" (a partial run: bounded prefixes of complete sessions)." if summary["partial_run"] else "."),
        "Each session has an independent browser object cache; capacities are 2, 4, 8, 16, and 32 objects.",
        f"Scorer in the tables below: `{primary}`. Every scorer's rows are in `policy_metrics.csv` "
        "and `comparisons.csv` under the results directory.",
        "", "## Frozen-policy comparisons, selected per capacity", "",
        "Every policy was chosen on validation separately at each cache size, then frozen for test. "
        "The `cost` values are hypothetical useful-hit equivalents per issued object; "
        "the trace contains neither page bytes nor latency. Each delta is a paired session "
        "bootstrap; a controller is never called a winner when it loses to no prefetching.", "",
        *_comparison_lines(summary["comparisons"]),
        "", "## Capacity transfer (the pivot's setting, for reference)", "",
        "Here the capacity-8 validation choice is reused at every capacity, as the pivot branch did. "
        "It is not the headline: at other capacities the best fixed depth differs.", "",
        *_comparison_lines(summary["capacity_transfer"]),
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
        "## Calibration", "",
        "Candidate-level reliability for bins with predicted probability at least 0.1 "
        "(almost every candidate sits below 0.1, where any calibration error looks small):", "",
        "| Split | Bin | Candidates | Mean predicted | Observed |", "|---|---|---:|---:|---:|",
    ]
    for row in summary.get("calibration_rows", []):
        lines.append(f"| {row['split']} | [{row['p_low']:.1f}, {row['p_high']:.1f}) | {row['count']:,} | "
                     f"{row['mean_probability']:.4f} | {row['observed_frequency']:.4f} |")
    for split in ("validation", "test"):
        values = calibration.get(split, {})
        if values.get("brier") is not None:
            lines.append("")
            lines.append(f"{split.capitalize()}: Brier {values['brier']:.6f}, base-rate Brier "
                         f"{values['brier_reference']:.6f}, Brier skill score "
                         f"{values['brier_skill_score']:+.4f}; ECE {values['ece']:.5f} (secondary).")
    lines += ["", "## Diagnostics and sensitivity", "",
        f"Candidate coverage@8 on test: {summary['test_candidate_coverage_at_8']['8']:.3%}; "
        "the ceiling appears beside each result in `comparisons.csv`. "
        "See `reliability.svg` for the plotted curve.",
        f"History-only top-1 transition cache baseline: {summary['transition_top1']['hit_ratio']:.3%} hits, "
        f"{summary['transition_top1']['issued_per_view']:.3f} issued/view at capacity 8; "
        "details in `transition_baseline.csv`. "
        "Context ablation and a gradient-boosting diagnostic are in `models/web/training_summary.json`. "
        "Boosting is diagnostic and did not select the controller.",
        "`break_even.csv` and `break_even_curve.csv` use cost per unused prefetch on validation "
        "at capacity 8. `cohorts.csv` examines cold-start and chronological retrieval coverage."
        + (" `subset_vs_full.csv` compares the historical 100k-view prefix with the full replay."
           if summary["subset_vs_full"] else ""), ""]
    (outputs / "report.md").write_text("\n".join(lines), encoding="utf-8")


def _results_dir(args) -> Path:
    if args.results_dir is not None:
        return args.results_dir
    # A bounded run must never overwrite the full run's tracked tables.
    if args.validation_views or args.test_views:
        return args.outputs / "results"
    return RESULTS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("--raw", type=Path, default=RAW)
    parser.add_argument("--processed", type=Path, default=PROCESSED)
    parser.add_argument("--models", type=Path, default=MODELS)
    parser.add_argument("--outputs", type=Path, default=OUTPUTS)
    parser.add_argument("--results-dir", type=Path, default=None,
                        help="schema tables (default results/web for a full run, "
                             "OUTPUTS/results for a bounded one)")
    parser.add_argument("--validation-views", type=int, default=0)
    parser.add_argument("--test-views", type=int, default=0)
    parser.add_argument("--quick", action="store_true",
                        help=f"bounded smoke run: {QUICK_VALIDATION_VIEWS} validation and "
                             f"{QUICK_TEST_VIEWS} test views, no sensitivity replay")
    parser.add_argument("--scorers", default=None,
                        help="comma list from scorers.SCORERS (default logistic)")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--reps", type=int, default=500, help="bootstrap repetitions")
    parser.add_argument("--no-feedback", action="store_true", help="leave the feedback gate out")
    parser.add_argument("--subset-check", action="store_true",
                        help="also replay the pivot's historical 100k-view test prefix")
    parser.add_argument("--seq-model", type=Path, default=None, help="WP11 sequence-model artifact")
    parser.add_argument("--novel-ranker", action="store_true",
                        help="train: also fit the relabelled ranker (P2b)")
    parser.add_argument("--parity-sessions", type=int, default=20)
    parser.add_argument("--decisions", default=None,
                        help="learning-curve training sizes, comma list (default 30000,100000,300000)")
    parser.add_argument("--session-gap-minutes", type=int, default=30)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.quick:
        args.validation_views = args.validation_views or QUICK_VALIDATION_VIEWS
        args.test_views = args.test_views or QUICK_TEST_VIEWS
    prepared = args.processed / "views.csv.gz"
    results = _results_dir(args)
    if args.command in ("download", "run"):
        print(f"Dataset ready: {acquire(args.raw)}")
    if args.command in ("prepare", "run"):
        print(json.dumps(prepare(args.raw, args.processed, args.session_gap_minutes), indent=2))
    if args.command in ("eda", "run"):
        _eda(args.processed, args.outputs)
    if args.command in ("train", "run"):
        _train(args.processed, args.models, args.novel_ranker)
    if args.command in ("benchmark", "run"):
        from .scorers import parse_scorers
        summary = run_benchmark(prepared, args.models, args.outputs,
                                args.validation_views, args.test_views, results_dir=results,
                                scorers=parse_scorers(args.scorers), workers=args.workers,
                                feedback=not args.no_feedback, subset_check=args.subset_check,
                                reps=args.reps, seq_path=args.seq_model)
        _report(summary, args.processed, args.models, args.outputs)
        print(f"Benchmark report written to {args.outputs / 'report.md'}; tables in {results}")
    if args.command == "demo":
        demo = json.loads((args.outputs / "demo.json").read_text(encoding="utf-8"))
        if demo["session_id"] is None:
            raise SystemExit(demo["note"])
        print(f"Real held-out session {demo['session_id']}; policy {demo['policy']}; "
              f"hits {demo['observed_hits']}; issued {demo['prefetches_issued']}")
        for step in demo["steps"]:
            print(json.dumps(step))
    if args.command in ("sensitivity", "run"):
        if args.quick:
            print("Sensitivity replay skipped under --quick (it re-prepares two segmentations).")
        else:
            # Supplemental segmentation replay uses a frozen 30-minute model.
            from .benchmark import run_sensitivity
            run_sensitivity(args.raw, args.processed, args.models, args.outputs)
    if args.command == "parity":
        from .demo_server import run_parity
        record = run_parity(Path.cwd(), results / "session_parity.json", models=args.models,
                            processed=args.processed, outputs=args.outputs,
                            sessions=args.parity_sessions)
        print(f"Session parity: {len(record['cases'])} cases, {record['mismatches']} mismatches; "
              f"written to {results / 'session_parity.json'}")
        if record["mismatches"]:
            raise SystemExit(1)
    if args.command == "freshness":
        from .benchmark import run_freshness
        rows = run_freshness(prepared, results)
        print(f"{len(rows)} freshness rows written to {results / 'freshness.csv'}")
    if args.command == "decompose":
        from .benchmark import run_decomposition
        rows = run_decomposition(prepared, args.models, args.outputs, results,
                                 workers=args.workers, reps=args.reps, seq_path=args.seq_model)
        print(f"{len(rows)} decomposition rows written to {results / 'decomposition.csv'}")
    if args.command == "learning-curve":
        from .benchmark import run_learning_curve
        decisions = tuple(int(x) for x in args.decisions.split(",")) if args.decisions else None
        rows = run_learning_curve(prepared, args.models, results, decisions=decisions,
                                  validation_views=args.validation_views, workers=args.workers)
        print(f"{len(rows)} learning-curve rows written to {results / 'learning_curve.csv'}")
    if args.command == "serve":
        from .demo_server import serve
        serve(Path.cwd(), args.host, args.port, models=args.models, processed=args.processed,
              outputs=args.outputs)
    if args.command == "export-demo":
        from .demo_server import export_demo
        target = export_demo(Path.cwd(), models=args.models, processed=args.processed,
                             outputs=args.outputs)
        print(f"Bounded static data written to {target}")


if __name__ == "__main__":
    main()
