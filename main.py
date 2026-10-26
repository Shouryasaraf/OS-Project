"""End-to-end entry point for the adaptive disk I/O prefetching project.

    python main.py                 # full research run, no prompts
    python main.py --interactive   # guided run with dataset/LSTM selection
    python main.py --quick         # skip LSTM training and the regime sweep

Running it with no arguments executes the whole pipeline and prints the
findings. Raw tables, per-dataset EDA, and every numeric row are written to
``outputs/``; the console deliberately shows conclusions rather than dumping
28-row tables.

The console/files split is the point: a reader should be able to see what the
results *mean* without scrolling, and still recover every raw number.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import mean
from time import perf_counter

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

from adaptive_prefetch.benchmark import (  # noqa: E402
    MODES, aggregate_results, analyze_results, benchmark_dataset, drift_report,
    markdown_table, write_results_csv,
)
from adaptive_prefetch.datasets import (  # noqa: E402
    FORMATS, LSTM_ARTIFACT, LSTM_TRAINING_RECORD, MSR_SAMPLES, MSRC_TRACE_DIR,
    OUTPUT_DIR, default_datasets, is_synthetic, lstm_training_set,
    synthetic_datasets,
)
from adaptive_prefetch.eda import (  # noqa: E402
    domain_shift, label_support, profile_trace, window_features,
)
from adaptive_prefetch.features import FEATURE_NAMES, extract  # noqa: E402
from adaptive_prefetch.model import DEFAULT_CLASSIFIER  # noqa: E402
from adaptive_prefetch.pipeline import (  # noqa: E402
    evaluate_classifier, labelled_dataset,
)
from adaptive_prefetch.report import (  # noqa: E402
    benchmark_section, caveats_section, classifier_section, dataset_overview,
    cost_model_section, domain_shift_section, drift_section, files_section,
    full_report, header, real_world_section,
)
from adaptive_prefetch.simulator import (  # noqa: E402
    DEFAULT_CAPACITY, LatencyModel, confusion_matrix, oracle_reference,
)
from adaptive_prefetch.trace import (  # noqa: E402
    CLASSES, load_csv, synthetic_dataset,
)

# Dataset construction (``default_datasets``, the sample paths and the seeds)
# lives in adaptive_prefetch.datasets, shared with the sweep.


def ask(prompt: str, default: str = "", cast=None):
    """Prompt with an optional default; re-asks until input is valid."""
    while True:
        suffix = f" [{default}]" if default else ""
        try:
            raw = input(f"{prompt}{suffix}: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            raise SystemExit(0)
        raw = raw or default
        if not raw:
            print("  Invalid input; try again.")
            continue
        if cast is None:
            return raw
        try:
            return cast(raw)
        except ValueError:
            print(f"  Expected {getattr(cast, '__name__', 'a value')}; try again.")


def ask_int(prompt: str, default: int, minimum: int | None = None) -> int:
    while True:
        value = ask(prompt, str(default), int)
        if minimum is None or value >= minimum:
            return value
        print(f"  Must be >= {minimum}; try again.")


def ask_yes_no(prompt: str, default: bool = False) -> bool:
    text = "Y/n" if default else "y/N"
    fallback = "y" if default else "n"
    return ask(f"{prompt} ({text})", fallback).lower() in ("y", "yes")


def interactive_datasets(window_size: int,
                         msrc_dir: Path = MSRC_TRACE_DIR) -> list[tuple[str, list]]:
    """Dataset picker. Only reachable with ``--interactive``."""
    print("\n--- Datasets (comma-separated) ---")
    print("  1. Synthetic transition traces (default; labelled, multi-seed)")
    print("  2. Bundled MSR Cambridge samples")
    print("  3. MSRC-trace-003 traces (real, unlabelled; 'all' is very slow)")
    print("  4. Custom trace file")
    wanted = ask("Datasets", "1")
    names = {t.strip() for t in wanted.split(",") if t.strip()}
    datasets: list[tuple[str, list]] = []
    if "1" in names:
        datasets.extend(synthetic_datasets(window_size))
    if "2" in names:
        for key, path in MSR_SAMPLES.items():
            if path.is_file():
                print(f"    {key}. {path.name}")
        key = ask("Sample number (1-3, comma-separated)", "1")
        for token in key.split(","):
            if token.strip() in MSR_SAMPLES:
                path = MSR_SAMPLES[token.strip()]
                datasets.append((path.stem, load_csv(path, "msr")))
    if "3" in names:
        if not msrc_dir.is_dir():
            print(f"  Not found: {msrc_dir}")
        else:
            paths = sorted(msrc_dir.glob("*.revised"))
            for number, path in enumerate(paths, 1):
                print(f"    {number:2}. {path.name}")
            selection = ask("Trace numbers (or 'all')", "1").lower()
            if selection == "all":
                selected = paths
            else:
                selected = []
                for token in selection.split(","):
                    if token.strip().isdigit() and 1 <= int(token) <= len(paths):
                        selected.append(paths[int(token) - 1])
            datasets.extend((f"msrc_{p.stem}", load_csv(p, "revised"))
                            for p in dict.fromkeys(selected))
    if "4" in names:
        raw = ask("Trace file path", str(REPO / "data" / "samples" / "demo.csv"))
        path = Path(raw)
        path = path if path.is_absolute() else REPO / path
        if path.is_file():
            fmt = ask(f"Format {FORMATS}", "auto")
            datasets.append((path.stem, load_csv(path, fmt if fmt in FORMATS else "auto")))
        else:
            print(f"  Not found: {path}")
    return datasets or default_datasets(window_size, msrc_dir=msrc_dir)


# --------------------------------------------------------------------------
# LSTM
# --------------------------------------------------------------------------

def _training_record() -> dict | None:
    try:
        return json.loads(LSTM_TRAINING_RECORD.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def resolve_lstm(datasets: list[tuple[str, list]], interactive: bool,
                 quick: bool, epochs: int,
                 window_size: int = 32) -> tuple[str | None, str]:
    """Ensure a usable LSTM artifact; return (path_or_None, note).

    The LSTM is trained on :func:`lstm_training_set`, generated traces whose
    seeds are disjoint from every benchmark dataset. It used to be trained on
    the benchmark traces themselves and then scored on them. A cached
    artifact is reused only when its training record matches that selection,
    so an artifact of unknown or in-sample provenance is retrained.
    """
    if quick:
        return None, "skipped (--quick)"
    training, record = lstm_training_set(datasets, window_size)
    record["epochs"] = epochs
    seeds = record["seeds"]
    provenance = (f"{len(training)} generated trace(s), seeds {seeds[0]}-{seeds[-1]}, "
                  "disjoint from the benchmark")

    if LSTM_ARTIFACT.is_file():
        from adaptive_prefetch.lstm import load_predictor
        if _training_record() != record:
            print("  cached LSTM artifact has no matching training record "
                  "(unknown or in-sample provenance); retraining")
        else:
            try:
                load_predictor(LSTM_ARTIFACT)
                reuse = not interactive
                if interactive:
                    reuse = not ask_yes_no("Retrain the LSTM from scratch?", False)
                if reuse:
                    return str(LSTM_ARTIFACT), f"reused cached artifact ({provenance})"
            except (ValueError, RuntimeError) as exc:
                print(f"  cached artifact unusable ({exc}); retraining")
    elif interactive:
        if not ask_yes_no("Train the LSTM baseline now (needs PyTorch)?", True):
            return None, "declined"

    from adaptive_prefetch.lstm import train_lstm
    print(f"  training LSTM on {provenance} / {record['requests']:,} requests "
          f"for {epochs} epochs...")
    try:
        LSTM_ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
        result = train_lstm(LSTM_ARTIFACT, traces=[r for _, r in training],
                            epochs=epochs)
    except (RuntimeError, ValueError) as exc:
        return None, f"unavailable: {exc}"
    LSTM_TRAINING_RECORD.write_text(json.dumps(record, indent=2) + "\n",
                                    encoding="utf-8")
    return str(LSTM_ARTIFACT), (f"trained on {provenance}, "
                                f"{result['examples']:,} examples, "
                                f"{result['classes']} classes")


# --------------------------------------------------------------------------
# Cost model
# --------------------------------------------------------------------------

def service_means_us(datasets: list[tuple[str, list]]) -> dict[str, float]:
    """Mean recorded device service time per dataset that records one, in us."""
    means = {}
    for name, requests in datasets:
        values = [r.service_ms * 1000.0 for r in requests
                  if getattr(r, "service_ms", None) is not None]
        if values:
            means[name] = sum(values) / len(values)
    return means


def select_cost_model(datasets: list[tuple[str, list]], cost_model: str,
                      prefetch_multiple: float = 1.0,
                      base: LatencyModel | None = None) -> tuple[LatencyModel, str]:
    """The cost model actually applied, and a label that names *it*.

    The label used to come from whether any trace carried service times,
    so ``--cost-model assumed`` on such traces was reported as measured. It is
    now derived from the model returned: ``measured ...`` only when the model
    was calibrated from recorded service times.
    """
    base = base or LatencyModel()
    kind = "assumed" if base == LatencyModel() else "custom"
    base_label = (f"{kind} {base.hit_us:g}/{base.demand_miss_us:g}/"
                  f"{base.prefetch_us:g} us")
    if cost_model != "measured":
        return base, base_label
    sources = [reqs for _, reqs in datasets
               if any(getattr(r, "service_ms", None) is not None for r in reqs)]
    merged = [r for reqs in sources for r in reqs
              if getattr(r, "service_ms", None) is not None]
    if not merged:
        return base, f"{base_label} (no recorded service times to measure)"
    latency = LatencyModel.measured(trace=merged,
                                    prefetch_multiple=prefetch_multiple)
    return latency, f"measured from {len(sources)} trace(s)"


# --------------------------------------------------------------------------
# Analyses
# --------------------------------------------------------------------------

def classifier_facts(seed: int, train_per_class: int, test_per_class: int):
    """Held-out accuracy, confusion matrix, and mean confidence."""
    from adaptive_prefetch.benchmark import make_model
    from adaptive_prefetch.features import WindowContext, dominant_stride

    model = make_model(seed, train_per_class)
    # Contextual features need a stream, exactly as at replay time.
    context = WindowContext()
    held_out = synthetic_dataset(test_per_class, seed + 100000)
    pairs = []
    for window, label in held_out:
        values = context.observe(window, dominant_stride(window))
        predicted, confidence = model.predict(values)
        pairs.append((label, predicted, confidence))
    correct = sum(t == p for t, p, _ in pairs)
    confident = [c for t, p, c in pairs if t == p]
    return (correct / len(pairs) if pairs else 0.0, correct, len(pairs),
            confusion_matrix([(t, p) for t, p, _ in pairs]),
            mean(confident) if confident else 0.0)


def transfer_probe(seeds: int = 4) -> list[dict]:
    """Block-size transfer: train at 1-block requests, test at larger sizes.

    This is the measurement that decided the default classifier, so the run
    reproduces it rather than asserting it.
    """
    import random

    from adaptive_prefetch.features import WindowContext, dominant_stride
    from adaptive_prefetch.model import make_classifier
    from adaptive_prefetch.pipeline import make_harder_generator

    def build(size, seed, per_class):
        """Contextual vectors laid out as a stream, matching replay."""
        rng = random.Random(seed)
        context = WindowContext()
        rows = []
        for label in CLASSES:
            for _ in range(per_class):
                window = make_harder_generator(label, rng, 32, noise=0.15,
                                               lba_base=10**7, lba_span=10**9,
                                               size=size)
                rows.append((context.observe(window, dominant_stride(window)),
                             label))
        rng.shuffle(rows)
        return rows

    results = []
    for test_size in (1, 8, 32, 128):
        row: dict[str, object] = {"label": f"1 -> {test_size}"}
        for name in ("gnb", "qda"):
            accuracies = []
            for i in range(seeds):
                model = make_classifier(name, len(FEATURE_NAMES))
                for values, label in build(1, 1000 + i, 60):
                    model.update(values, label)
                test = build(test_size, 900000 + i, 30)
                correct = sum(model.predict(v)[0] == label for v, label in test)
                accuracies.append(correct / len(test))
            row[name] = sum(accuracies) / len(accuracies)
        results.append(row)
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--interactive", action="store_true",
                        help="prompt for datasets, cache size and LSTM")
    parser.add_argument("--quick", action="store_true",
                        help="skip LSTM training and the regime sweep")
    parser.add_argument("--cache-blocks", type=int, default=DEFAULT_CAPACITY)
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument("--seeds", type=int, default=6,
                        help="seed pairs per cell in the classifier regime sweep")
    parser.add_argument("--lstm-epochs", type=int, default=60)
    parser.add_argument("--prefetch-multiple", type=float, default=1.0,
                        help="cost of a speculative read relative to a demand "
                             "read; <1.0 models overlapped prefetch")
    parser.add_argument("--cost-model", default="measured",
                        choices=("assumed", "measured"),
                        help="'measured' calibrates the cost model from the "
                             "traces' own recorded device service times")
    parser.add_argument("--no-outputs", action="store_true",
                        help="print findings without writing files")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR),
                        help="where the raw tables are written (default outputs/)")
    parser.add_argument("--msrc-dir", default=str(MSRC_TRACE_DIR),
                        help="directory holding the MSRC-trace-003 *.revised files")
    args = parser.parse_args(argv)

    started = perf_counter()
    window_size = max(8, args.window_size)
    capacity = max(1, args.cache_blocks)

    msrc_dir = Path(args.msrc_dir)
    if args.interactive:
        datasets = interactive_datasets(window_size, msrc_dir)
        base = LatencyModel() if ask_yes_no(
            "Default latency model (5/100/50 us)?", True) else LatencyModel(
            float(ask("Hit us", "5")), float(ask("Miss us", "100")),
            float(ask("Prefetch us", "50")))
    else:
        datasets = default_datasets(window_size, msrc_dir=msrc_dir)
        base = LatencyModel()

    # Calibrate the cost model from the traces' own recorded device service
    # times when asked to and when they exist. The label names the model that
    # was actually applied, never merely the data that was available.
    latency, cost_label = select_cost_model(datasets, args.cost_model,
                                            args.prefetch_multiple, base)
    service_means = service_means_us(datasets)
    if args.cost_model == "measured" and not cost_label.startswith("measured"):
        print("  no recorded service times in the selected traces; "
              "using the assumed cost model")

    print(header("ML-BASED ADAPTIVE DISK I/O PREFETCHING", {
        "datasets": len(datasets),
        "cache capacity": f"{capacity} blocks",
        "window size": f"{window_size} requests",
        "classifier": DEFAULT_CLASSIFIER,
        "cost model": (f"MEASURED {cost_label[len('measured '):]}: "
                       f"hit {latency.hit_us:.1f} / miss {latency.demand_miss_us:.0f} "
                       f"/ prefetch {latency.prefetch_us:.0f} us"
                       if cost_label.startswith("measured") else cost_label),
        "mode": "interactive" if args.interactive else ("quick" if args.quick else "full"),
    }))

    print(cost_model_section(latency, cost_label))

    # ---- datasets -------------------------------------------------------
    cacheable = {}
    for name, requests in datasets:
        reference = oracle_reference(requests, capacity)
        if reference["read_blocks"]:
            cacheable[name] = reference["hit_ratio"]
    datasets_text = dataset_overview(datasets, capacity)
    print(datasets_text)

    # ---- classifier -----------------------------------------------------
    accuracy, correct, total, matrix, confidence = classifier_facts(42, 100, 40)
    regime_rows: list[dict] = []
    if not args.quick:
        for regime_name in ("baseline", "noisy", "realistic",
                            "transition_heavy"):
            for name in ("gnb", "qda"):
                regime_rows.append(evaluate_classifier(
                    name, seeds=args.seeds, windows_per_class=100,
                    test_per_class=40, regime_name=regime_name))
    transfer_rows = transfer_probe(seeds=3)
    classifier_text = classifier_section(accuracy, correct, total, matrix,
                                         confidence, regime_rows,
                                         DEFAULT_CLASSIFIER, transfer_rows)
    print(classifier_text)

    # ---- domain shift ---------------------------------------------------
    reference = [extract(w) for w, _ in labelled_dataset(100, 42, window_size)]
    shifts: dict[str, dict] = {}
    for name, requests in datasets:
        if name.startswith("synthetic"):
            continue
        vectors = window_features(requests, window_size)
        if vectors:
            shifts[name] = domain_shift(reference, vectors)
    if shifts:
        worst_name = max(shifts, key=lambda n: shifts[n]["mean_abs_z"])
        print(domain_shift_section(worst_name, shifts[worst_name]))

    # ---- benchmark ------------------------------------------------------
    lstm_path, lstm_note = resolve_lstm(datasets, args.interactive, args.quick,
                                        args.lstm_epochs, window_size)
    print(f"\n  LSTM: {lstm_note}")

    rows: list[dict] = []
    for name, requests in datasets:
        rows.extend(benchmark_dataset(
            name, requests, seed=42, capacity=capacity, window_size=window_size,
            latency=latency, lstm_model_path=lstm_path))
    rows.extend(pooled_rows(rows))
    print(benchmark_section(rows, capacity, cacheable))
    real = real_world_section(rows, capacity, latency)
    if real:
        print(real)

    drift = drift_report(42, 8, window_size)
    print(drift_section(drift, window_size))
    caveats = caveats_section(cost_label, service_means)

    # ---- outputs --------------------------------------------------------
    written: dict[str, str] = {}
    if not args.no_outputs:
        written = write_outputs(
            Path(args.output_dir), datasets, rows, drift, shifts, regime_rows,
            capacity=capacity, window_size=window_size,
            datasets_text=datasets_text, classifier_text=classifier_text,
            real_text=real, caveats_text=caveats,
            lstm_note=f"{lstm_path or 'none'}: {lstm_note}")
    print(caveats)
    if written:
        print(files_section(written, args.output_dir))
    print(f"Completed in {perf_counter() - started:.1f}s.")


def pooled_rows(rows: list[dict]) -> list[dict]:
    """Request-weighted pooled rows, synthetic and real kept apart.

    One pool over both used to put generated traces into the same mean as the
    recorded ones. A pool is written only when it spans two or more datasets.
    """
    pooled: list[dict] = []
    for name, keep in (("ALL_SYNTHETIC", True), ("ALL_REAL", False)):
        group = [r for r in rows if is_synthetic(r["dataset"]) == keep]
        if len({str(r["dataset"]) for r in group}) > 1:
            pooled.extend(aggregate_results(group, name))
    return pooled


def write_outputs(output_dir: Path, datasets, rows, drift, shifts, regime_rows,
                  *, capacity: int, window_size: int, datasets_text: str,
                  classifier_text: str, real_text: str, caveats_text: str,
                  lstm_note: str) -> dict[str, str]:
    """Write every raw table to ``output_dir``; return a description map.

    The console sections are passed in rather than rebuilt, so the files
    report exactly the capacity, window and figures the console printed.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}

    csv_path = output_dir / "results.csv"
    write_results_csv(csv_path, rows)
    written["results.csv"] = "every benchmark row (all datasets x all modes)"

    md_path = output_dir / "results.md"
    md_path.write_text(
        full_report([
            ("Datasets", datasets_text),
            ("Classifier", classifier_text),
            ("Domain shift", "\n\n".join(
                domain_shift_section(name, shift)
                for name, shift in shifts.items()) or "(none)"),
            ("Policy comparison", markdown_table(rows)),
            ("Real-workload result", real_text or "(none)"),
            ("Result interpretation", analyze_results(rows)),
            ("Drift", drift_section(drift, window_size)),
            ("Caveats", caveats_text),
        ]) + f"\n<!-- capacity {capacity} blocks, window {window_size} requests; "
             f"lstm {lstm_note} -->\n", encoding="utf-8")
    written["results.md"] = (
        f"narrative report with the full {len(MODES)}-mode table")

    eda_path = output_dir / "eda.md"
    eda_path.write_text("\n\n".join(
        profile_trace(requests, name, window_size, capacity)
        for name, requests in datasets), encoding="utf-8")
    written["eda.md"] = "per-trace request stream, locality, features, clusters"

    if regime_rows:
        eval_path = output_dir / "classifier_eval.csv"
        with eval_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=[
                "classifier", "regime", "accuracy", "accuracy_sd", "min", "max",
                "predict_us", "train_ms"])
            writer.writeheader()
            writer.writerows(regime_rows)
        written["classifier_eval.csv"] = "classifier x regime accuracy and cost"

    if shifts:
        shift_path = output_dir / "domain_shift.csv"
        with shift_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=[
                "dataset", "feature", "ref_mean", "ref_sd", "target_mean",
                "shift_sd", "coverage"])
            writer.writeheader()
            for name, shift in shifts.items():
                for feature, stats in shift["features"].items():
                    stats = shift["features"][feature]
                    writer.writerow({"dataset": name, "feature": feature,
                                     "ref_mean": round(stats["ref_mean"], 6),
                                     "ref_sd": round(stats["ref_sd"], 6),
                                     "target_mean": round(stats["target_mean"], 6),
                                     "shift_sd": round(stats["shift_sd"], 4),
                                     "coverage": round(stats["coverage"], 4)})
        written["domain_shift.csv"] = "per-feature shift vs the synthetic region"

    drift_path = output_dir / "drift.csv"
    with drift_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "variant", "phase", "switch_lag_windows", "phase_accuracy"])
        writer.writeheader()
        writer.writerows(drift)
    written["drift.csv"] = "frozen vs online switch lag per phase"

    support = label_support(labelled_dataset(1, 42, window_size))
    labels_path = output_dir / "label_support.md"
    labels_path.write_text(
        "# Class support of the labelled population\n\n"
        f"- present: {support['present']}\n"
        f"- missing: {support['missing']}\n"
        f"- counts: {support['counts']}\n\n"
        "Real traces in `data/` carry no access-pattern label, so the four-class\n"
        "problem can only be evaluated on generated windows.\n", encoding="utf-8")
    written["label_support.md"] = "which classes the labelled data actually covers"
    return written


if __name__ == "__main__":
    main()
