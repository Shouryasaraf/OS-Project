"""Reproducible Review 2 demonstration and trace replay commands."""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

from .benchmark import (MODES, benchmark_dataset, drift_report, make_model,
                        markdown_table, write_results_csv)
from .artifacts import load_model
from .features import FEATURE_NAMES, WindowContext, dominant_stride, extract
from .model import CLASSIFIERS, DEFAULT_CLASSIFIER, OnlineGaussianNB
from .simulator import (DEFAULT_CAPACITY, LatencyModel, confusion_matrix,
                         replay)
from .trace import (CLASSES, FORMATS, load_csv, synthetic_dataset,
                    transition_trace, write_csv)

# ``FORMATS`` is re-exported from ``trace`` so the loader and every
# ``--format`` choice come from one tuple. ``oracle`` (libCacheSim
# oracleGeneral block traces) is never auto-detected.


def train(seed: int = 42, examples_per_class: int = 100,
          classifier: str = DEFAULT_CLASSIFIER) -> OnlineGaussianNB:
    return make_model(seed, examples_per_class, classifier=classifier)


def run_demo(args: argparse.Namespace) -> None:
    model = train(args.seed, args.train_per_class, args.classifier)
    # Different seed and independently generated traces prevent shared windows.
    held_out = synthetic_dataset(args.test_per_class, args.seed + 100000)
    # The model takes the 12-feature contextual vector, built over the
    # held-out windows as one stream exactly as replay and main.py build it.
    # ``extract`` alone gives the 8 stream features and is rejected.
    context = WindowContext()
    start = perf_counter()
    pairs = [(label, model.predict(context.observe(window, dominant_stride(window)))[0])
             for window, label in held_out]
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

    online_model = train(args.seed, args.train_per_class, args.classifier)
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
                 train(args.seed, args.train_per_class,
                       args.classifier)) if mode == "adaptive" else None
        if mode == "lstm" and not args.lstm_model:
            print("lstm        skipped: no trained artifact")
            continue
        if mode == "learned_depth":
            reason = _depth_skip_reason(args.depth_model, args.trace, len(requests))
            if reason:
                print(f"learned_depth skipped: {reason}")
                continue
        smoothing = ((args.smoothing, 0.0)
                     if args.smoothing and mode == "adaptive" else None)
        try:
            metrics, predictions = replay(requests, args.cache_blocks, args.window_size,
                                          mode, model, lstm_model_path=args.lstm_model,
                                          smoothing=smoothing,
                                          depth_model=(args.depth_model
                                                       if mode == "learned_depth" else None))
        except (RuntimeError, FileNotFoundError, ValueError) as exc:
            if mode not in ("lstm", "learned_depth"):
                raise
            print(f"{mode:11} skipped: {exc}")
            continue
        print(f"{mode:11} hit={metrics.hit_ratio:.3f} "
              f"prefetch_precision={metrics.prefetch_precision:.3f} "
              f"prefetch_recall={metrics.prefetch_recall:.3f} "
              f"mean_modelled_us={metrics.mean_access_latency_us:.2f} "
              f"prefetches={metrics.prefetches} unused={metrics.unused_prefetches}")
        if mode == "adaptive":
            print(f"Classified {len(predictions)} complete windows")


#: Committed samples that are a prefix of a block trace an artifact may have
#: been trained on. ``msr-cambridge1-sample.csv`` matches the first 1000
#: ``msr_hm_0`` oracle requests row for row (MASTER_PLAN §0.1), so an
#: ``msr_hm_0`` artifact must not be scored on it either.
_TRACE_ALIASES = {"msr-cambridge1-sample.csv": "msr_hm_0"}


def _depth_skip_reason(artifact: str | None, trace: str | None,
                       requests: int) -> str | None:
    """Why ``learned_depth`` may not be scored on requests ``[0, requests)``.

    ``trace`` is the replayed file, or ``None`` for generated requests. An
    artifact without a matching training record is refused, as in ``sweep``:
    its training range is unknown, so no slice can be shown to be clean.
    """
    if not artifact:
        return "no trained learned_depth artifact"
    from .corpora import trace_stem
    from .depth_model import read_training_record, scoring_conflict
    if not Path(artifact).is_file():
        return None  # the replay reports the missing file itself
    record = read_training_record(artifact)
    if record is None:
        return ("learned_depth artifact has no matching training record, so "
                "its training range is unknown")
    if trace is None:
        return None
    stem = trace_stem(trace)
    return scoring_conflict(record, _TRACE_ALIASES.get(stem, stem), 0, requests)


def run_benchmark(args: argparse.Namespace) -> None:
    datasets = []
    if "synthetic" in args.datasets:
        for offset in range(args.seeds):
            requests, _ = transition_trace(args.seed + 200000 + offset,
                                           args.windows_per_class, args.window_size)
            datasets.append((f"synthetic_seed_{args.seed + offset}", requests))
    sources = {}  # dataset name -> replayed file, for the constraint-14 guard
    if "msr" in args.datasets:
        sample = (Path(__file__).resolve().parents[2] / "data" / "samples" /
                  "msr-cambridge1-sample.csv")
        datasets.append(("msr_format_sample", load_csv(sample, "msr")))
        sources["msr_format_sample"] = str(sample)
    if "iotta" in args.datasets:
        if not args.trace:
            raise ValueError("--datasets iotta requires --trace and its --format")
        datasets.append(("iotta_supplied", load_csv(args.trace, args.format)))
        sources["iotta_supplied"] = args.trace
    if args.cost_model == "measured":
        sources = [(name, reqs) for name, reqs in datasets
                   if any(getattr(r, "service_ms", None) is not None for r in reqs)]
        if sources:
            merged = [r for _, reqs in sources for r in reqs]
            latency = LatencyModel.measured(trace=merged)
            print(f"  measured cost model from {len(sources)} trace(s) with "
                  f"recorded service times")
            print(f"    hit {latency.hit_us:.3f} us | demand miss "
                  f"{latency.demand_miss_us:.1f} us | prefetch "
                  f"{latency.prefetch_us:.1f} us")
        else:
            print("  no measured service times in the selected traces; "
                  "using the default assumed model")
            latency = LatencyModel()
    else:
        latency = LatencyModel(args.hit_us, args.miss_us, args.prefetch_us)
    rows = []
    for name, requests in datasets:
        refused = (_depth_skip_reason(args.depth_model, sources.get(name), len(requests))
                   if args.depth_model else None)
        dataset_rows = benchmark_dataset(
            name, requests, seed=args.seed, capacity=args.cache_blocks,
            window_size=args.window_size, latency=latency,
            lstm_model_path=args.lstm_model,
            online_real=args.online_real and not name.startswith("synthetic"),
            classifier=args.classifier,
            depth_model_path=None if refused else args.depth_model)
        if refused:
            for row in dataset_rows:
                if row["mode"] == "learned_depth":
                    row["status"] = f"skipped: {refused}"
        rows.extend(dataset_rows)
    table = markdown_table(rows)
    print(table)
    print("\nModelled latency includes configured prefetch cost; it is not measured device latency.")
    print("Recall uses measurement-only future-reaccess lookahead, never policy input.")
    print("Skipped LSTM means no PyTorch/model artifact was available; no comparison claim is made.")
    print("learned_depth is scored only with --depth-model; pass an artifact that was not "
          "trained or selected on these traces.")
    print("\nDrift switch lag on labelled synthetic phases:")
    for item in drift_report(args.seed, args.windows_per_class, args.window_size,
                             args.classifier):
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


def _progress(message: str) -> None:
    print(f"  {message}", flush=True)


def run_train_lstm(args: argparse.Namespace) -> None:
    from .corpora import trace_stem
    from .lstm import train_lstm, train_lstm_on_trace
    save = args.save or (f"models/lstm_{trace_stem(args.trace[0])}_s{args.seed}.pt"
                         if args.trace else "models/lstm_delta.pt")
    if args.format == "oracle":
        # A real block trace: the chronological 60/20/20 split, recorded.
        if len(args.trace) != 1:
            raise ValueError("--format oracle trains on exactly one --trace "
                             "(its own chronological split)")
        result = train_lstm_on_trace(
            save, args.trace[0], format="oracle", n=args.n, epochs=args.epochs,
            batch_size=args.batch_size, seed=args.seed, device=args.device,
            num_threads=args.num_threads, patience=args.patience,
            progress=_progress)
        print(f"Saved LSTM to {save}; train/val/test = {result['split']}; "
              f"the test range is the only part of {result['trace']} it may be scored on")
    else:
        traces = [load_csv(path, args.format) for path in args.trace] if args.trace else None
        # Every request of a supplied trace was used for training or early
        # stopping, so the record claims the whole trace.
        record = ({"source": args.format,
                   "traces": [{"trace": trace_stem(path), "path": str(path),
                               "format": args.format, "n_requests": len(t),
                               "train": [0, len(t)], "val": None, "test": None}
                              for path, t in zip(args.trace, traces)]}
                  if traces else {"source": "transition_trace",
                                  "seeds": [args.seed + i for i in range(4)],
                                  "windows_per_class": 8})
        result = train_lstm(save, traces, args.epochs, seed=args.seed,
                            device=args.device, num_threads=args.num_threads,
                            val_fraction=args.val_fraction,
                            batch_size=args.batch_size, patience=args.patience,
                            record=record, progress=_progress)
        print(f"Saved offline LSTM to {save}")
    print(f"  {result['examples']:,} training examples, {result['val_examples']:,} "
          f"validation, best epoch {result['best_epoch']}, {result['train_s']:.1f} s "
          f"on {result['device']}")
    if result.get("gpu_contention"):
        print(f"  {result['gpu_contention']}")


def run_train_depth(args: argparse.Namespace) -> None:
    from .corpora import trace_stem
    from .depth_model import (ACTION_CAPS, load_prefix, prepare,
                              train_and_select)
    if args.save and len(args.traces) != 1:
        raise ValueError("--save names one artifact; use --out-dir for several traces")
    caps = tuple(args.action_caps or ACTION_CAPS)
    for path in args.traces:
        stem = trace_stem(path)
        save = args.save or str(Path(args.out_dir) / f"{stem}_s{args.seed}.pt")
        print(f"{stem}: loading the first {args.n:,} canonical requests", flush=True)
        start = perf_counter()
        prefix = load_prefix(path, args.format, args.n)
        prepared = prepare(prefix, args.cache_blocks)
        print(f"  {len(prefix):,} requests, split {prepared.split}, horizon "
              f"{prepared.horizon}, prepared in {perf_counter() - start:.1f} s", flush=True)
        result = train_and_select(
            prepared, save, seed=args.seed, epochs=args.epochs,
            batch_size=args.batch_size, device=args.device, action_caps=caps,
            val_replay=args.val_replay, decide_every=args.decide_every,
            num_threads=args.num_threads, patience=args.patience,
            progress=_progress)
        record = result.record
        print(f"  trained {record['epochs_run']} epoch(s) on {record['device']} in "
              f"{record['train_s']:.1f} s ({record['train_examples']:,} examples, "
              f"{record['params']:,} parameters, {record['oom_retries']} OOM retries)")
        if record.get("gpu_contention"):
            print(f"  {record['gpu_contention']}")
        for cap, chosen in result.thresholds.items():
            print(f"  cap {cap}: validation chose seg>={chosen['seg']} gap="
                  f"{chosen['gap']} (validation hit {chosen['val_hit_ratio']:.4f} on "
                  f"requests {chosen['val_slice']})")
        print(f"  saved {result.artifact} and {result.artifact.with_suffix('.training.json')}; "
              f"score it only on requests {record['traces'][0]['test']} of {stem}")


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


def run_evaluate(args: argparse.Namespace) -> None:
    """Model selection and regime evaluation."""
    from .pipeline import BLENDED_REGIMES, REGIMES, evaluate_classifier, select_classifier

    if args.regime:
        stats = evaluate_classifier(args.classifier, args.seeds,
                                    args.train_per_class, args.test_per_class,
                                    args.regime, args.standardize)
        print(f"classifier={stats['classifier']}  regime={stats['regime']}")
        print(f"  accuracy   {100 * stats['accuracy']:.2f}% +/- "
              f"{100 * stats['accuracy_sd']:.2f} "
              f"(min {100 * stats['min']:.2f}%, max {100 * stats['max']:.2f}%)")
        print(f"  predict    {stats['predict_us']:.2f} us/call")
        print(f"  train      {stats['train_ms']:.2f} ms")
        return
    known = list(REGIMES) + list(BLENDED_REGIMES)
    print(select_classifier(args.seeds, args.train_per_class,
                            args.test_per_class)["report"])


def run_eda(args: argparse.Namespace) -> None:
    """Profile a trace: request stream, locality, window features, clusters."""
    from .eda import profile_trace

    requests = load_csv(args.trace, args.format)
    print(profile_trace(requests, Path(args.trace).name, args.window_size,
                        args.cache_blocks))


def run_shift(args: argparse.Namespace) -> None:
    """Measure how far a real trace sits from the synthetic training region."""
    from .eda import domain_shift, label_support, report_shift, window_features
    from .pipeline import labelled_dataset

    reference = [extract(window) for window, _ in
                 labelled_dataset(args.train_per_class, args.seed, args.window_size)]
    target = window_features(load_csv(args.trace, args.format), args.window_size)
    if not target:
        print("target trace has no complete windows; nothing to compare")
        return
    print(report_shift(reference, target,
                       f"synthetic training windows (seed {args.seed})",
                       Path(args.trace).name))
    support = label_support(labelled_dataset(1, args.seed, args.window_size))
    print(f"synthetic class support: {support['present']}")
    print("real traces carry no access-pattern label, so no supervised accuracy")
    print("can be computed on them; only the distance above is meaningful.")


def build_parser() -> argparse.ArgumentParser:
    """The ``adaptive_prefetch`` argument parser, one sub-command per task."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="run reproducible synthetic Review 2 demo")
    demo.add_argument("--seed", type=int, default=42)
    demo.add_argument("--train-per-class", type=int, default=100)
    demo.add_argument("--test-per-class", type=int, default=40)
    demo.add_argument("--windows-per-class", type=int, default=4)
    demo.add_argument("--cache-blocks", type=int, default=DEFAULT_CAPACITY)
    demo.add_argument("--classifier", default=DEFAULT_CLASSIFIER,
                      choices=tuple(CLASSIFIERS))
    replay_command = commands.add_parser("replay", help="replay a normalized CSV trace")
    replay_command.add_argument("trace")
    replay_command.add_argument("--seed", type=int, default=42)
    replay_command.add_argument("--train-per-class", type=int, default=100)
    replay_command.add_argument("--window-size", type=int, default=32)
    replay_command.add_argument("--cache-blocks", type=int, default=DEFAULT_CAPACITY)
    replay_command.add_argument("--classifier", default=DEFAULT_CLASSIFIER,
                                choices=tuple(CLASSIFIERS))
    replay_command.add_argument("--smoothing", type=int, default=0,
                                help="require N agreeing windows before switching policy")
    replay_command.add_argument("--format", default="auto",
                                choices=FORMATS)
    replay_command.add_argument("--lstm-model")
    replay_command.add_argument("--depth-model",
                                help="learned_depth artifact (train-depth); refused on "
                                     "the requests it was trained or selected on")
    replay_command.add_argument("--model-path", help="saved Gaussian NB classifier JSON")
    benchmark = commands.add_parser("benchmark", help="compare all available policies")
    benchmark.add_argument("--datasets", nargs="+", default=["synthetic", "msr"],
                           choices=("synthetic", "msr", "iotta"))
    benchmark.add_argument("--trace")
    benchmark.add_argument("--format", default="auto",
                           choices=FORMATS)
    benchmark.add_argument("--seed", type=int, default=42)
    benchmark.add_argument("--seeds", type=int, default=3)
    benchmark.add_argument("--windows-per-class", type=int, default=8)
    benchmark.add_argument("--window-size", type=int, default=32)
    benchmark.add_argument("--cache-blocks", type=int, default=DEFAULT_CAPACITY)
    benchmark.add_argument("--hit-us", type=float, default=5)
    benchmark.add_argument("--miss-us", type=float, default=100)
    benchmark.add_argument("--prefetch-us", type=float, default=50)
    benchmark.add_argument("--cost-model", default="assumed",
                           choices=("assumed", "measured"),
                           help="'measured' calibrates from the trace's own "
                                "recorded device service times")
    benchmark.add_argument("--lstm-model")
    benchmark.add_argument("--depth-model", help="learned_depth artifact (train-depth)")
    benchmark.add_argument("--online-real", action="store_true")
    benchmark.add_argument("--classifier", default=DEFAULT_CLASSIFIER,
                           choices=tuple(CLASSIFIERS))
    benchmark.add_argument("--output-markdown")
    benchmark.add_argument("--output-csv")
    normalize = commands.add_parser("normalize", help="convert a known trace schema to canonical CSV")
    normalize.add_argument("--input", required=True)
    normalize.add_argument("--output", required=True)
    normalize.add_argument("--format", default="auto",
                           choices=FORMATS)
    lstm = commands.add_parser("train-lstm", help="train optional offline LSTM baseline")
    lstm.add_argument("--save", default=None,
                      help="default models/lstm_delta.pt, or models/lstm_<trace>_s<seed>.pt "
                           "with --trace")
    lstm.add_argument("--trace", action="append", default=[])
    lstm.add_argument("--format", default="auto",
                      choices=FORMATS)
    lstm.add_argument("--epochs", type=int, default=60)
    lstm.add_argument("--seed", type=int, default=42)
    lstm.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    lstm.add_argument("--num-threads", type=int, default=4,
                      help="torch CPU threads (the host is shared)")
    lstm.add_argument("--n", type=int, default=2_000_000,
                      help="--format oracle: canonical requests in the 60/20/20 prefix")
    lstm.add_argument("--batch-size", type=int, default=128)
    lstm.add_argument("--val-fraction", type=float, default=0.2,
                      help="CSV traces: chronological tail held out for early stopping")
    lstm.add_argument("--patience", type=int, default=None,
                      help="stop after this many epochs without validation improvement")
    depth = commands.add_parser(
        "train-depth", help="train learned_depth per trace (60/20/20, thresholds on validation)")
    depth.add_argument("--traces", nargs="+", required=True)
    depth.add_argument("--format", default="oracle", choices=FORMATS)
    depth.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    depth.add_argument("--seed", type=int, default=0)
    depth.add_argument("--n", type=int, default=2_000_000,
                       help="canonical requests in the prefix that is split 60/20/20")
    depth.add_argument("--epochs", type=int, default=8)
    depth.add_argument("--batch-size", type=int, default=8192)
    depth.add_argument("--patience", type=int, default=2)
    depth.add_argument("--cache-blocks", type=int, default=DEFAULT_CAPACITY)
    depth.add_argument("--val-replay", type=int, default=100_000,
                       help="validation requests replayed per threshold candidate")
    depth.add_argument("--action-caps", type=int, nargs="+", default=None,
                       help="segment-end caps to select thresholds for (default 64 256)")
    depth.add_argument("--decide-every", type=int, default=32)
    depth.add_argument("--num-threads", type=int, default=4)
    depth.add_argument("--out-dir", default="models/depth")
    depth.add_argument("--save", default=None, help="artifact path (one trace only)")
    msr_train = commands.add_parser("train-msr-sample", help="weakly adapt classifier on the unlabelled MSR sample")
    msr_train.add_argument("--trace", default=str(Path(__file__).resolve().parents[2] /
                                                 "data" / "samples" / "msr-cambridge1-sample.csv"))
    msr_train.add_argument("--save", default="models/msr_sample_gnb.json")
    msr_train.add_argument("--seed", type=int, default=42)
    msr_train.add_argument("--window-size", type=int, default=32)
    export = commands.add_parser("export-demo", help="write synthetic transition CSV")
    export.add_argument("path")
    export.add_argument("--seed", type=int, default=42)
    evaluate = commands.add_parser(
        "evaluate", help="compare classifiers across evaluation regimes")
    evaluate.add_argument("--classifier", default=DEFAULT_CLASSIFIER,
                          choices=tuple(CLASSIFIERS))
    evaluate.add_argument("--regime", default=None,
                          help="evaluate one regime only; omit to compare all")
    evaluate.add_argument("--seeds", type=int, default=12)
    evaluate.add_argument("--train-per-class", type=int, default=100)
    evaluate.add_argument("--test-per-class", type=int, default=40)
    evaluate.add_argument("--standardize", action="store_true",
                          help="zero-mean/unit-variance feature scaling")
    eda = commands.add_parser("eda", help="exploratory profile of one trace")
    eda.add_argument("trace")
    eda.add_argument("--format", default="auto",
                     choices=FORMATS)
    eda.add_argument("--window-size", type=int, default=32)
    eda.add_argument("--cache-blocks", type=int, default=DEFAULT_CAPACITY)
    shift = commands.add_parser(
        "shift", help="domain shift between synthetic training and a real trace")
    shift.add_argument("trace")
    shift.add_argument("--format", default="auto",
                       choices=FORMATS)
    shift.add_argument("--window-size", type=int, default=32)
    shift.add_argument("--train-per-class", type=int, default=100)
    shift.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
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
        elif args.command == "train-depth":
            run_train_depth(args)
        elif args.command == "train-msr-sample":
            run_train_msr_sample(args)
        elif args.command == "evaluate":
            run_evaluate(args)
        elif args.command == "eda":
            run_eda(args)
        elif args.command == "shift":
            run_shift(args)
        else:
            requests, _ = transition_trace(args.seed)
            write_csv(args.path, requests)
            print(f"Wrote {len(requests)} requests to {args.path}")
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
