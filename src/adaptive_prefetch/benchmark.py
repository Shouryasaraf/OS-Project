"""Fair cache-policy comparison and labelled synthetic drift diagnostics."""

from __future__ import annotations

import csv
from pathlib import Path

from .features import FEATURE_NAMES, extract
from .model import OnlineGaussianNB
from .simulator import LatencyModel, replay
from .trace import CLASSES, Request, synthetic_dataset, transition_trace

MODES = ("none", "sequential", "strided", "stride", "markov", "lstm", "adaptive")
COLUMNS = ("dataset", "mode", "status", "hit_ratio", "precision", "recall",
           "unused", "mean_latency_us", "speedup_vs_none", "inference_us")


def make_model(seed: int = 42, examples_per_class: int = 100) -> OnlineGaussianNB:
    model = OnlineGaussianNB(len(FEATURE_NAMES))
    for window, label in synthetic_dataset(examples_per_class, seed):
        model.update(extract(window), label)
    return model


def benchmark_dataset(name: str, requests: list[Request], *, seed: int = 42,
                      capacity: int = 128, window_size: int = 32,
                      latency: LatencyModel | None = None,
                      lstm_model_path: str | None = None,
                      online_real: bool = False) -> list[dict[str, object]]:
    if not requests:
        raise ValueError("benchmark trace is empty")
    baseline, _ = replay(requests, capacity, window_size, "none",
                         latency_model=latency)
    baseline_us = baseline.mean_access_latency_us
    rows: list[dict[str, object]] = []
    modes = MODES + (("adaptive_pseudo",) if online_real else ())
    for mode in modes:
        actual_mode = "adaptive" if mode == "adaptive_pseudo" else mode
        model = make_model(seed) if actual_mode == "adaptive" else None
        try:
            if mode == "lstm" and lstm_model_path is None:
                raise RuntimeError("no trained LSTM artifact; optional baseline skipped")
            metrics, _ = replay(
                requests, capacity, window_size, actual_mode, model,
                latency_model=latency, lstm_model_path=lstm_model_path,
                pseudo_label_threshold=0.95 if mode == "adaptive_pseudo" else None,
            )
        except (RuntimeError, FileNotFoundError) as exc:
            if mode != "lstm":
                raise
            rows.append({"dataset": name, "mode": mode, "status": f"skipped: {exc}",
                         **{column: None for column in COLUMNS[3:]}})
            continue
        current_us = metrics.mean_access_latency_us
        rows.append({
            "dataset": name, "mode": mode, "status": "ok",
            "hit_ratio": metrics.hit_ratio,
            "precision": metrics.prefetch_precision,
            "recall": metrics.prefetch_recall,
            "unused": metrics.unused_prefetches,
            "mean_latency_us": current_us,
            "speedup_vs_none": baseline_us / current_us if current_us else 0.0,
            "inference_us": metrics.inference_us_per_call,
        })
    return rows


def drift_report(seed: int = 42, windows_per_class: int = 8,
                 window_size: int = 32) -> list[dict[str, object]]:
    requests, labels = transition_trace(seed + 200000, windows_per_class,
                                        window_size)
    rows = []
    for variant, update in (("frozen", False), ("labelled_online", True)):
        model = make_model(seed)
        _, predictions = replay(requests, window_size=window_size,
                                model=model, labels=labels if update else None,
                                online_updates=update)
        for phase_number, label in enumerate(CLASSES):
            start = phase_number * windows_per_class
            phase = predictions[start:start + windows_per_class]
            lag = next((offset for offset, (_, predicted, _) in enumerate(phase)
                        if predicted == label), None)
            rows.append({"variant": variant, "phase": label,
                         "switch_lag_windows": lag,
                         "phase_accuracy": sum(predicted == label
                                               for _, predicted, _ in phase) / len(phase)})
    return rows


def markdown_table(rows: list[dict[str, object]]) -> str:
    names = ("Dataset", "Mode", "Status", "Hit %", "Precision %", "Oracle recall %",
             "Unused", "Mean us", "Speedup", "Inference us")
    lines = ["| " + " | ".join(names) + " |",
             "| " + " | ".join("---" for _ in names) + " |"]
    for row in rows:
        def fraction(key: str) -> str:
            value = row[key]
            return f"{100 * value:.2f}" if isinstance(value, (int, float)) else "-"
        def numeric(key: str, digits: int = 2) -> str:
            value = row[key]
            return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "-"
        cells = [str(row["dataset"]), str(row["mode"]), str(row["status"]),
                 fraction("hit_ratio"), fraction("precision"), fraction("recall"),
                 str(row["unused"]) if row["unused"] is not None else "-",
                 numeric("mean_latency_us"), numeric("speedup_vs_none"),
                 numeric("inference_us", 3)]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_results_csv(path: str | Path, rows: list[dict[str, object]]) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
