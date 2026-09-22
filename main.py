"""Interactive final entry point (replaces run_review2.ps1).

Asks the user what to run, then executes the project modules and produces the
final result: the 7-policy benchmark matrix + drift report, saved to outputs/.

Usage:  python main.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

from adaptive_prefetch.benchmark import (  # noqa: E402
    analyze_results,
    benchmark_dataset,
    drift_report,
    markdown_table,
    write_results_csv,
)
from adaptive_prefetch.cli import run_demo  # noqa: E402
from adaptive_prefetch.simulator import LatencyModel  # noqa: E402
from adaptive_prefetch.trace import CLASSES, load_csv, transition_trace  # noqa: E402

MSR_SAMPLE = REPO / "data" / "samples" / "msr-cambridge1-sample.csv"
LSTM_ARTIFACT = REPO / "models" / "lstm_delta.pt"
OUTPUT_DIR = REPO / "outputs"

FORMATS = ("auto", "normalized", "msr", "iotta8", "alibaba", "revised")


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


def pick_datasets() -> list[tuple[str, list]]:  # list[tuple[name, requests]]
    print("\n--- Datasets (pick one or more, comma-separated) ---")
    print("  1. Synthetic transition traces (default; labelled, multi-seed)")
    print("  2. Bundled MSR Cambridge sample (real requests, unlabelled)")
    print("  3. Custom trace file (normalized/msr/iotta8/alibaba/revised)")
    wanted = ask("Datasets to benchmark", "1")
    names = [token.strip() for token in wanted.split(",") if token.strip()]
    datasets: list[tuple[str, list]] = []
    if "1" in names:
        for offset in range(3):
            requests, _ = transition_trace(42 + 200000 + offset, 8, 32)
            datasets.append((f"synthetic_seed_{42 + offset}", requests))
    if "2" in names:
        datasets.append(("msr_cambridge1_sample", load_csv(MSR_SAMPLE, "msr")))
    if "3" in names:
        path = ask("Trace file path", str(REPO / "data" / "samples" / "demo.csv"))
        trace_path = REPO / path if not Path(path).is_absolute() else Path(path)
        if not trace_path.is_file():
            print(f"  File not found: {trace_path}")
            return pick_datasets()
        fmt = ask("Format", "auto")
        fmt = fmt if fmt in FORMATS else "auto"
        datasets.append((trace_path.stem, load_csv(trace_path, fmt)))
    if not datasets:
        print("  No valid selection; defaulting to synthetic.")
        return pick_datasets()
    return datasets


def ensure_lstm(real_traces: list[list] | None = None) -> str | None:
    """Include the LSTM baseline; retrain or reuse the cached artifact."""
    print("\n--- LSTM baseline (offline delta-prediction LSTM, optional) ---")
    if not ask_yes_no("Include the LSTM baseline?"):
        return None
    if LSTM_ARTIFACT.is_file():
        if ask_yes_no("Retrain the LSTM from scratch?", True):
            return _train_lstm(real_traces)
        print(f"Using cached artifact: {LSTM_ARTIFACT}")
        return str(LSTM_ARTIFACT)
    print("No trained artifact found; training now (needs PyTorch).")
    return _train_lstm(real_traces)


def _train_lstm(real_traces: list[list] | None = None) -> str | None:
    """Train the offline LSTM; return the artifact path or None on failure."""
    epochs = ask_int("Epochs", 2, 1)
    traces = real_traces or None
    if traces is None:
        print("No real trace picked; training on synthetic traces.")
    else:
        total = sum(len(trace) for trace in traces)
        print(f"{len(traces)} real trace(s) selected, {total:,} requests total.")
        limit = ask_int("Requests per trace to train on (0 = all)", 0, 0)
        if limit:
            traces = [trace[:limit] for trace in traces]
            total = sum(len(trace) for trace in traces)
        print(f"Training on {total:,} requests for {epochs} epoch(s) "
              f"-> may take a while...")
    try:
        from adaptive_prefetch.lstm import train_lstm

        LSTM_ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
        result = train_lstm(LSTM_ARTIFACT, traces=traces, epochs=epochs)
        print(f"Saved {LSTM_ARTIFACT}: {result}")
    except (RuntimeError, ValueError) as exc:
        print(f"LSTM skipped: {exc}")
        return None
    return str(LSTM_ARTIFACT)


def ask_latency() -> LatencyModel:
    if ask_yes_no("Use default latency model (hit 5us / miss 100us / prefetch 50us)?", True):
        return LatencyModel()
    return LatencyModel(
        hit_us=float(ask("Hit latency (us)", "5")),
        demand_miss_us=float(ask("Demand-miss latency (us)", "100")),
        prefetch_us=float(ask("Prefetch cost (us)", "50")),
    )


def main() -> None:
    print("=" * 72)
    print("ML-BASED ADAPTIVE DISK I/O PREFETCHING — FINAL RUN")
    print("Collecting inputs, then running the full project pipeline.")
    print("=" * 72)

    show_demo = ask_yes_no("First run the classifier demo (accuracy + confusion matrix)?", True)
    cache_blocks = ask_int("Cache capacity (blocks)", 128, 1)
    window_size = ask_int("Window size (requests)", 32, 8)
    datasets = pick_datasets()
    real_traces = [requests for name, requests in datasets if not name.startswith("synthetic")]
    lstm_model = ensure_lstm(real_traces)
    latency = ask_latency()
    save_outputs = ask_yes_no("Save results to outputs/results.md + .csv?", True)

    print("\nRunning...\n")

    if show_demo:
        from argparse import Namespace

        run_demo(Namespace(seed=42, train_per_class=100, test_per_class=40,
                           windows_per_class=4, cache_blocks=cache_blocks))
        print()

    rows: list[dict] = []
    for name, requests in datasets:
        rows.extend(benchmark_dataset(
            name, requests, seed=42, capacity=cache_blocks, window_size=window_size,
            latency=latency, lstm_model_path=lstm_model))
    table = markdown_table(rows)

    print("FINAL RESULT — POLICIES COMPARED ON IDENTICAL REQUESTS/CACHE/LRU")
    print(table)
    print("\nNotes: latency is modelled (not measured device latency);")
    print("recall uses measurement-only future-reaccess lookahead, never policy input;")
    print("LSTM row appears only if a trained artifact was available.")

    print("\n" + analyze_results(rows))

    print("\nDrift adaptation (labelled synthetic phases, frozen vs online GNB):")
    print(f"  workload order: {' -> '.join(CLASSES)}")
    for item in drift_report(42, 8, window_size):
        print(f"  {item['variant']:15} {item['phase']:10} "
              f"lag={item['switch_lag_windows']} windows "
              f"phase_accuracy={item['phase_accuracy']:.3f}")

    if save_outputs:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        md_path = OUTPUT_DIR / "results.md"
        csv_path = OUTPUT_DIR / "results.csv"
        md_path.write_text("# Final results\n\n" + table + "\n\n"
                           + analyze_results(rows) + "\n", encoding="utf-8")
        write_results_csv(csv_path, rows)
        print(f"\nSaved: {md_path}\n       {csv_path}")

    print("\n=== WHAT THE FINAL OUTPUT IS ===")
    print("A single comparison table of all prefetch policies on the same streams,")
    print("cache capacity and LRU rules. Each row reports hit ratio, prefetch")
    print("precision, oracle recall, unused prefetches, modelled mean latency, and")
    print("latency speedup vs no prefetch. The adaptive ML policy row shows whether")
    print("classify-then-prefetch beats the stride/Markov/LSTM baselines while")
    print("issuing less wasted I/O. The result-interpretation section then names")
    print("the winning policy per dataset with its quantified gains (speedup and")
    print("hit-ratio improvement vs no prefetch). The drift report shows how")
    print("quickly the online classifier re-learns when the workload changes.")


if __name__ == "__main__":
    main()