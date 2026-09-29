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
from adaptive_prefetch.eda import (  # noqa: E402
    domain_shift, label_support, locality_profile, profile_trace, window_features,
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
    CLASSES, load_csv, synthetic_dataset, transition_trace,
)

MSR_SAMPLES = {
    "1": REPO / "data" / "samples" / "msr-cambridge1-sample.csv",
    "2": REPO / "data" / "samples" / "msr-cambridge2-sample.csv",
    "3": REPO / "data" / "samples" / "msr-cambridge-sample-merged.csv",
}
MSRC_TRACE_DIR = REPO / "data" / "MSRC-trace-003" / "final-trace"
LSTM_ARTIFACT = REPO / "models" / "lstm_delta.pt"
OUTPUT_DIR = REPO / "outputs"
FORMATS = ("auto", "normalized", "msr", "iotta8", "alibaba", "revised")

#: MSRC traces used for the default run. The full directory holds 32 files up
#: to 2.6 GB each and ``load_csv`` materialises a whole file, so the default
#: run samples a spread of read-heavy volumes rather than loading everything.
#: All of these carry the recorded device service times used to calibrate the
#: measured cost model.
QUICK_MSRC = ("hm_1.revised", "mds_0.revised", "stg_0.revised",
              "proj_3.revised", "ts_0.revised")


# --------------------------------------------------------------------------
# Dataset construction
# --------------------------------------------------------------------------

def default_datasets(window_size: int) -> list[tuple[str, list]]:
    """The dataset set used when running non-interactively."""
    datasets: list[tuple[str, list]] = []
    for offset in range(3):
        requests, _ = transition_trace(42 + 200000 + offset, 8, window_size)
        datasets.append((f"synthetic_seed_{42 + offset}", requests))
    for number in ("1", "2"):
        path = MSR_SAMPLES[number]
        if path.is_file():
            datasets.append((path.stem, load_csv(path, "msr")))
    if MSRC_TRACE_DIR.is_dir():
        for name in QUICK_MSRC:
            path = MSRC_TRACE_DIR / name
            if path.is_file():
                # Cap the sample: these files are large and a full load would
                # dominate the runtime. 40k requests is ample for a stable
                # hit-ratio comparison and keeps the run interactive.
                requests = load_csv(path, "revised")
                if len(requests) > 40000:
                    requests = requests[:40000]
                datasets.append((f"msrc_{path.stem}", requests))
    demo = REPO / "data" / "samples" / "demo.csv"
    if demo.is_file():
        datasets.append(("demo_normalized", load_csv(demo, "auto")))
    return datasets


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


def interactive_datasets(window_size: int) -> list[tuple[str, list]]:
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
        for offset in range(3):
            requests, _ = transition_trace(42 + 200000 + offset, 8, window_size)
            datasets.append((f"synthetic_seed_{42 + offset}", requests))
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
        if not MSRC_TRACE_DIR.is_dir():
            print(f"  Not found: {MSRC_TRACE_DIR}")
        else:
            paths = sorted(MSRC_TRACE_DIR.glob("*.revised"))
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
    return datasets or default_datasets(window_size)


# --------------------------------------------------------------------------
# LSTM
# --------------------------------------------------------------------------

def resolve_lstm(datasets: list[tuple[str, list]], interactive: bool,
                 quick: bool, epochs: int) -> tuple[str | None, str]:
    """Ensure a usable LSTM artifact; return (path_or_None, note)."""
    if quick:
        return None, "skipped (--quick)"
    synthetic = [r for name, r in datasets if name.startswith("synthetic")]
    real = [r for name, r in datasets if not name.startswith("synthetic")]
    if not synthetic and not real:
        return None, "no traces available to train on"
    train_on = real if (real and not synthetic) else synthetic

    if LSTM_ARTIFACT.is_file():
        from adaptive_prefetch.lstm import load_predictor
        try:
            load_predictor(LSTM_ARTIFACT)
            reuse = not interactive
            if interactive:
                reuse = not ask_yes_no("Retrain the LSTM from scratch?", False)
            if reuse:
                return str(LSTM_ARTIFACT), "reused cached artifact"
        except (ValueError, RuntimeError) as exc:
            print(f"  cached artifact unusable ({exc}); retraining")
    elif interactive:
        if not ask_yes_no("Train the LSTM baseline now (needs PyTorch)?", True):
            return None, "declined"

    from adaptive_prefetch.lstm import train_lstm
    total = sum(len(t) for t in train_on)
    print(f"  training LSTM on {len(train_on)} trace(s) / {total:,} requests "
          f"for {epochs} epochs...")
    try:
        LSTM_ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
        result = train_lstm(LSTM_ARTIFACT, traces=train_on, epochs=epochs)
    except (RuntimeError, ValueError) as exc:
        return None, f"unavailable: {exc}"
    return str(LSTM_ARTIFACT), (f"trained on {len(train_on)} trace(s), "
                                f"{result['examples']:,} examples, "
                                f"{result['classes']} classes")


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
    args = parser.parse_args(argv)

    started = perf_counter()
    window_size = max(8, args.window_size)
    capacity = max(1, args.cache_blocks)

    if args.interactive:
        datasets = interactive_datasets(window_size)
        latency = LatencyModel() if ask_yes_no(
            "Default latency model (5/100/50 us)?", True) else LatencyModel(
            float(ask("Hit us", "5")), float(ask("Miss us", "100")),
            float(ask("Prefetch us", "50")))
    else:
        datasets = default_datasets(window_size)
        latency = LatencyModel()

    # Calibrate the cost model from the traces' own recorded device service
    # times when available. The assumed 5/100/50 us model is off by 3-50x
    # against these measurements.
    measured_sources = [name for name, reqs in datasets
                        if any(getattr(r, "service_ms", None) is not None
                               for r in reqs)]
    if args.cost_model == "measured":
        merged = [r for _, reqs in datasets for r in reqs
                  if getattr(r, "service_ms", None) is not None]
        if merged:
            latency = LatencyModel.measured(
                trace=merged, prefetch_multiple=args.prefetch_multiple)
        else:
            print("  no recorded service times in the selected traces; "
                  "using the assumed cost model")

    print(header("ML-BASED ADAPTIVE DISK I/O PREFETCHING", {
        "datasets": len(datasets),
        "cache capacity": f"{capacity} blocks",
        "window size": f"{window_size} requests",
        "classifier": DEFAULT_CLASSIFIER,
        "cost model": (f"MEASURED from {len(measured_sources)} trace(s): "
                       f"hit {latency.hit_us:.1f} / miss {latency.demand_miss_us:.0f} "
                       f"/ prefetch {latency.prefetch_us:.0f} us"
                       if measured_sources else
                       f"assumed {latency.hit_us:g}/{latency.demand_miss_us:g}/"
                       f"{latency.prefetch_us:g} us"),
        "mode": "interactive" if args.interactive else ("quick" if args.quick else "full"),
    }))

    print(cost_model_section(latency,
                            'measured' if measured_sources else 'assumed'))

    # ---- datasets -------------------------------------------------------
    cacheable = {}
    for name, requests in datasets:
        reference = oracle_reference(requests, capacity)
        if reference["read_blocks"]:
            cacheable[name] = reference["hit_ratio"]
    print(dataset_overview(datasets, capacity))

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
    print(classifier_section(accuracy, correct, total, matrix, confidence,
                             regime_rows, DEFAULT_CLASSIFIER,
                             transfer_probe(seeds=3)))

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
                                        args.lstm_epochs)
    if lstm_path:
        print(f"\n  LSTM: {lstm_note}")
    else:
        print(f"\n  LSTM: {lstm_note}")

    rows: list[dict] = []
    for name, requests in datasets:
        rows.extend(benchmark_dataset(
            name, requests, seed=42, capacity=capacity, window_size=window_size,
            latency=latency, lstm_model_path=lstm_path))
    if len(datasets) > 1:
        rows.extend(aggregate_results(rows))
    print(benchmark_section(rows, capacity, cacheable))
    real = real_world_section(rows, capacity)
    if real:
        print(real)

    drift = drift_report(42, 8, window_size)
    print(drift_section(drift, window_size))

    # ---- outputs --------------------------------------------------------
    written: dict[str, str] = {}
    if not args.no_outputs:
        written = write_outputs(datasets, rows, drift, shifts, regime_rows,
                                reference, matrix, accuracy, correct, total,
                                confidence, lstm_path)
    print(caveats_section())
    if written:
        print(files_section(written))
    print(f"Completed in {perf_counter() - started:.1f}s.")


def write_outputs(datasets, rows, drift, shifts, regime_rows, reference,
                  matrix, accuracy, correct, total, confidence,
                  lstm_path) -> dict[str, str]:
    """Write every raw table to ``outputs/``; return a description map."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}

    csv_path = OUTPUT_DIR / "results.csv"
    write_results_csv(csv_path, rows)
    written["results.csv"] = "every benchmark row (all datasets x all modes)"

    md_path = OUTPUT_DIR / "results.md"
    md_path.write_text(
        full_report([
            ("Datasets", dataset_overview(datasets, 128)),
            ("Classifier", classifier_section(
                accuracy, correct, total, matrix, confidence, regime_rows,
                DEFAULT_CLASSIFIER, transfer_probe(seeds=3))),
            ("Domain shift", "\n\n".join(
                domain_shift_section(name, shift)
                for name, shift in shifts.items()) or "(none)"),
            ("Policy comparison", markdown_table(rows)),
            ("Real-workload result", real_world_section(rows, 128) or "(none)"),
            ("Result interpretation", analyze_results(rows)),
            ("Drift", drift_section(drift, 32)),
            ("Caveats", caveats_section()),
        ]) + f"\n<!-- lstm: {lstm_path or 'skipped'} -->\n", encoding="utf-8")
    written["results.md"] = (
        f"narrative report with the full {len(MODES)}-mode table")

    eda_path = OUTPUT_DIR / "eda.md"
    eda_path.write_text("\n\n".join(
        profile_trace(requests, name, 32, 128)
        for name, requests in datasets), encoding="utf-8")
    written["eda.md"] = "per-trace request stream, locality, features, clusters"

    if regime_rows:
        eval_path = OUTPUT_DIR / "classifier_eval.csv"
        with eval_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=[
                "classifier", "regime", "accuracy", "accuracy_sd", "min", "max",
                "predict_us", "train_ms"])
            writer.writeheader()
            writer.writerows(regime_rows)
        written["classifier_eval.csv"] = "classifier x regime accuracy and cost"

    if shifts:
        shift_path = OUTPUT_DIR / "domain_shift.csv"
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

    drift_path = OUTPUT_DIR / "drift.csv"
    with drift_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "variant", "phase", "switch_lag_windows", "phase_accuracy"])
        writer.writeheader()
        writer.writerows(drift)
    written["drift.csv"] = "frozen vs online switch lag per phase"

    support = label_support(labelled_dataset(1, 42, 32))
    labels_path = OUTPUT_DIR / "label_support.md"
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
