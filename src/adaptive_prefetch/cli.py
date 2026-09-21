"""Reproducible Review 2 demonstration and trace replay commands."""

from __future__ import annotations

import argparse
from time import perf_counter

from .features import FEATURE_NAMES, extract
from .model import OnlineGaussianNB
from .simulator import confusion_matrix, replay
from .trace import CLASSES, load_csv, synthetic_dataset, transition_trace, write_csv


def train(seed: int = 42, examples_per_class: int = 100) -> OnlineGaussianNB:
    model = OnlineGaussianNB(len(FEATURE_NAMES))
    for window, label in synthetic_dataset(examples_per_class, seed):
        model.update(extract(window), label)
    return model


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
    requests = load_csv(args.trace)
    model = train(args.seed, args.train_per_class)
    print(f"Loaded {len(requests)} requests from {args.trace}")
    print("Real-trace labels are unknown; classifier updates are disabled.")
    for mode in ("none", "sequential", "strided", "adaptive"):
        metrics, predictions = replay(requests, args.cache_blocks, args.window_size,
                                      mode, model if mode == "adaptive" else None)
        print(f"{mode:11} hit={metrics.hit_ratio:.3f} "
              f"prefetch_precision={metrics.prefetch_precision:.3f} "
              f"prefetches={metrics.prefetches} unused={metrics.unused_prefetches}")
        if mode == "adaptive":
            print(f"Classified {len(predictions)} complete windows")


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
    export = commands.add_parser("export-demo", help="write synthetic transition CSV")
    export.add_argument("path")
    export.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.command == "demo":
        run_demo(args)
    elif args.command == "replay":
        run_replay(args)
    else:
        requests, _ = transition_trace(args.seed)
        write_csv(args.path, requests)
        print(f"Wrote {len(requests)} requests to {args.path}")


if __name__ == "__main__":
    main()
