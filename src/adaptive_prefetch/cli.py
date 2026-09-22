"""Reproducible Review 2 demonstration and trace replay commands."""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

from .benchmark import (MODES, benchmark_dataset, drift_report, make_model,
                        markdown_table, write_results_csv)
from .artifacts import load_model
from .features import FEATURE_NAMES, extract
from .model import OnlineGaussianNB
from .simulator import LatencyModel, confusion_matrix, replay
from .trace import CLASSES, load_csv, synthetic_dataset, transition_trace, write_csv


def train(seed: int = 42, examples_per_class: int = 100) -> OnlineGaussianNB:
    return make_model(seed, examples_per_class)


def run_demo(args: argparse.Namespace) -> None:
    model = train(args.seed, args.train_per_class)
    # Different seed and independently generated traces prevent shared windows.
    held_out = synthetic_dataset(args.test_per_class, args.seed + 100000)
    start = perf_counter()
    pairs = [(label, model.predict(extract(window))[0]) for window, label in held_out]
    prediction_ms = 1000 * (perf_counter() - start) / len(held_out)
    correct = sum(truth == prediction for truth, prediction in pairs)
    print("REVIEW 2: ONLINE WORKLOAD CLASSIFIER + ADAPTIVE PREFETCH SIMULATOR")
    print(f"Training: {args.train_per_class * len(CLASSES)} labelled synthetic windows")
    print(f"Held-out: {len(held_out)} independently generated synthetic windows")
    print(f"Accuracy: {correct / len(pairs):.3f} ({correct}/{len(pairs)})")
    print(f"Classifier prediction time: {prediction_ms:.4f} ms/window (this machine)")
    matrix = confusion_matrix(pairs)
    print("Confusion matrix, rows = true class, columns = predicted class")
    print("true/pred    " + " ".join(f"{name[:4]:>5}" for name in CLASSES))
    for truth in CLASSES:
        print(f"{truth:12}" + " ".join(f"{matrix.get(truth, {}).get(pred, 0):5d}" for pred in CLASSES))

    requests, labels = transition_trace(args.seed + 200000, args.windows_per_class)
    print("\nTransition trace: " + " -> ".join(CLASSES))
    for mode in ("none", "sequential", "strided", "adaptive"):
        # All modes see the same requests and cache capacity. The adaptive
        # model is frozen during fair cache comparison; online updates are
        # demonstrated separately below using known synthetic labels.
        metrics, predictions = replay(requests, args.cache_blocks,
                                      mode=mode, model=model if mode == "adaptive" else None)
        print(f"{mode:11} hit={metrics.hit_ratio:.3f} "
              f"prefetch_precision={metrics.prefetch_precision:.3f} "
              f"prefetches={metrics.prefetches} unused={metrics.unused_prefetches} "
              f"switches={metrics.policy_switches}")
        if mode == "adaptive":
            print("Window predictions (true -> predicted, confidence):")
            for number, predicted, confidence in predictions:
                print(f"  {number + 1:02d}: {labels[number]:10} -> {predicted:10} {confidence:.2f}")

    online_model = train(args.seed, args.train_per_class)
    replay(requests, args.cache_blocks, model=online_model,
           labels=labels, online_updates=True)
    print(f"Online supervised updates: {len(labels)} labelled windows processed")
    print("Note: real unlabelled traces cannot provide these updates automatically.")


def run_replay(args: argparse.Namespace) -> None:
    requests = load_csv(args.trace, args.format)
    print(f"Loaded {len(requests)} requests from {args.trace}")
    print("Real-trace labels are unknown; classifier updates are disabled.")
    for mode in MODES:
        model = (load_model(args.model_path)[0] if args.model_path else
                 train(args.seed, args.train_per_class)) if mode == "adaptive" else None
        if mode == "lstm" and not args.lstm_model:
            print("lstm        skipped: no trained artifact")
            continue
        try:
            metrics, predictions = replay(requests, args.cache_blocks, args.window_size,
                                          mode, model, lstm_model_path=args.lstm_model)
        except (RuntimeError, FileNotFoundError) as exc:
            if mode != "lstm":
                raise
            print(f"lstm        skipped: {exc}")
            continue
        print(f"{mode:11} hit={metrics.hit_ratio:.3f} "
              f"prefetch_precision={metrics.prefetch_precision:.3f} "
              f"prefetch_recall={metrics.prefetch_recall:.3f} "
              f"mean_modelled_us={metrics.mean_access_latency_us:.2f} "
              f"prefetches={metrics.prefetches} unused={metrics.unused_prefetches}")
        if mode == "adaptive":
            print(f"Classified {len(predictions)} complete windows")


def run_benchmark(args: argparse.Namespace) -> None:
    datasets = []
    if "synthetic" in args.datasets:
        for offset in range(args.seeds):
            requests, _ = transition_trace(args.seed + 200000 + offset,
                                           args.windows_per_class, args.window_size)
            datasets.append((f"synthetic_seed_{args.seed + offset}", requests))
    if "msr" in args.datasets:
        sample = (Path(__file__).resolve().parents[2] / "data" / "samples" /
                  "msr-cambridge1-sample.csv")
        datasets.append(("msr_format_sample", load_csv(sample, "msr")))
    if "iotta" in args.datasets:
        if not args.trace:
            raise ValueError("--datasets iotta requires --trace and its --format")
        datasets.append(("iotta_supplied", load_csv(args.trace, args.format)))
    latency = LatencyModel(args.hit_us, args.miss_us, args.prefetch_us)
    rows = []
    for name, requests in datasets:
        rows.extend(benchmark_dataset(name, requests, seed=args.seed,
                                      capacity=args.cache_blocks,
                                      window_size=args.window_size,
                                      latency=latency,
                                      lstm_model_path=args.lstm_model,
                                      online_real=args.online_real and not name.startswith("synthetic")))
    table = markdown_table(rows)
    print(table)
    print("\nModelled latency includes configured prefetch cost; it is not measured device latency.")
    print("Recall uses measurement-only future-reaccess lookahead, never policy input.")
    print("Skipped LSTM means no PyTorch/model artifact was available; no comparison claim is made.")
    print("\nDrift switch lag on labelled synthetic phases:")
    for item in drift_report(args.seed, args.windows_per_class, args.window_size):
        print(f"  {item['variant']:15} {item['phase']:10} "
              f"lag={item['switch_lag_windows']} windows "
              f"phase_accuracy={item['phase_accuracy']:.3f}")
    if args.output_markdown:
        Path(args.output_markdown).write_text(
            "# Stage 2 benchmark\n\n" + table +
            "\n\nLatency is modelled; synthetic labels are not real-trace ground truth.\n",
            encoding="utf-8")
    if args.output_csv:
        write_results_csv(args.output_csv, rows)


def run_normalize(args: argparse.Namespace) -> None:
    requests = load_csv(args.input, args.format)
    write_csv(args.output, requests)
    print(f"Normalized {len(requests)} requests into {args.output}")


def run_train_lstm(args: argparse.Namespace) -> None:
    from .lstm import train_lstm
    traces = [load_csv(path, args.format) for path in args.trace] if args.trace else None
    result = train_lstm(args.save, traces, args.epochs, args.seed)
    print(f"Saved offline LSTM to {args.save}: {result}")


def run_train_msr_sample(args: argparse.Namespace) -> None:
    from .training import adapt_msr_sample
    result = adapt_msr_sample(args.trace, args.save, args.seed, args.window_size)
    print(f"Saved weakly adapted classifier to {args.save}")
    print(f"Source: {result['requests']} requests; {result['complete_windows']} complete windows")
    print(f"Real random-like windows used: {result['real_windows_updated']}")
    print("Real-data labels: heuristic proxy only; no real-trace accuracy claim")
    print(f"Held-out synthetic accuracy before/after: "
          f"{result['synthetic_held_out_accuracy_before']:.3f} / "
          f"{result['synthetic_held_out_accuracy_after']:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="run reproducible synthetic Review 2 demo")
    demo.add_argument("--seed", type=int, default=42)
    demo.add_argument("--train-per-class", type=int, default=100)
    demo.add_argument("--test-per-class", type=int, default=40)
    demo.add_argument("--windows-per-class", type=int, default=4)
    demo.add_argument("--cache-blocks", type=int, default=128)
    replay_command = commands.add_parser("replay", help="replay a normalized CSV trace")
    replay_command.add_argument("trace")
    replay_command.add_argument("--seed", type=int, default=42)
    replay_command.add_argument("--train-per-class", type=int, default=100)
    replay_command.add_argument("--window-size", type=int, default=32)
    replay_command.add_argument("--cache-blocks", type=int, default=128)
    replay_command.add_argument("--format", default="auto",
                                choices=("auto", "normalized", "msr", "iotta8", "alibaba", "revised"))
    replay_command.add_argument("--lstm-model")
    replay_command.add_argument("--model-path", help="saved Gaussian NB classifier JSON")
    benchmark = commands.add_parser("benchmark", help="compare all available policies")
    benchmark.add_argument("--datasets", nargs="+", default=["synthetic", "msr"],
                           choices=("synthetic", "msr", "iotta"))
    benchmark.add_argument("--trace")
    benchmark.add_argument("--format", default="auto",
                           choices=("auto", "normalized", "msr", "iotta8", "alibaba", "revised"))
    benchmark.add_argument("--seed", type=int, default=42)
    benchmark.add_argument("--seeds", type=int, default=3)
    benchmark.add_argument("--windows-per-class", type=int, default=8)
    benchmark.add_argument("--window-size", type=int, default=32)
    benchmark.add_argument("--cache-blocks", type=int, default=128)
    benchmark.add_argument("--hit-us", type=float, default=5)
    benchmark.add_argument("--miss-us", type=float, default=100)
    benchmark.add_argument("--prefetch-us", type=float, default=50)
    benchmark.add_argument("--lstm-model")
    benchmark.add_argument("--online-real", action="store_true")
    benchmark.add_argument("--output-markdown")
    benchmark.add_argument("--output-csv")
    normalize = commands.add_parser("normalize", help="convert a known trace schema to canonical CSV")
    normalize.add_argument("--input", required=True)
    normalize.add_argument("--output", required=True)
    normalize.add_argument("--format", default="auto",
                           choices=("auto", "normalized", "msr", "iotta8", "alibaba", "revised"))
    lstm = commands.add_parser("train-lstm", help="train optional offline LSTM baseline")
    lstm.add_argument("--save", default="models/lstm_delta.pt")
    lstm.add_argument("--trace", action="append", default=[])
    lstm.add_argument("--format", default="auto",
                      choices=("auto", "normalized", "msr", "iotta8", "alibaba", "revised"))
    lstm.add_argument("--epochs", type=int, default=2)
    lstm.add_argument("--seed", type=int, default=42)
    msr_train = commands.add_parser("train-msr-sample", help="weakly adapt classifier on the unlabelled MSR sample")
    msr_train.add_argument("--trace", default=str(Path(__file__).resolve().parents[2] /
                                                 "data" / "samples" / "msr-cambridge1-sample.csv"))
    msr_train.add_argument("--save", default="models/msr_sample_gnb.json")
    msr_train.add_argument("--seed", type=int, default=42)
    msr_train.add_argument("--window-size", type=int, default=32)
    export = commands.add_parser("export-demo", help="write synthetic transition CSV")
    export.add_argument("path")
    export.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        if args.command == "demo":
            run_demo(args)
        elif args.command == "replay":
            run_replay(args)
        elif args.command == "benchmark":
            run_benchmark(args)
        elif args.command == "normalize":
            run_normalize(args)
        elif args.command == "train-lstm":
            run_train_lstm(args)
        elif args.command == "train-msr-sample":
            run_train_msr_sample(args)
        else:
            requests, _ = transition_trace(args.seed)
            write_csv(args.path, requests)
            print(f"Wrote {len(requests)} requests to {args.path}")
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
