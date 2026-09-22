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


def analyze_results(rows: list[dict[str, object]]) -> str:
    """Interpret benchmark rows: best policy, adaptive comparison, and percentage gains.

    Groups rows by dataset, identifies the winning mode (highest
    ``speedup_vs_none`` among completed rows), and calculates relative percentage
    gains for adaptive prefetching against traditional and deep learning baselines.
    """
    by_dataset: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_dataset.setdefault(str(row["dataset"]), []).append(row)

    lines = ["## Result interpretation", ""]
    for name in sorted(by_dataset):
        group = by_dataset[name]
        baseline = next((row for row in group if row["mode"] == "none"
                         and row["status"] == "ok"), None)
        completed = [row for row in group if row["status"] == "ok"
                     and row["mode"] != "none"]

        if baseline is None or not completed:
            skipped = [row["mode"] for row in group if row["status"] != "ok"]
            lines.append(f"- **{name}**: no usable result"
                         + (f" (skipped: {', '.join(map(str, skipped))})" if skipped else ""))
            continue

        winner = max(completed, key=lambda row: float(row["speedup_vs_none"]))
        baseline_hit = float(baseline["hit_ratio"])
        winner_hit = float(winner["hit_ratio"])
        hit_gain = winner_hit - baseline_hit
        speedup = float(winner["speedup_vs_none"])

        if speedup <= 1.005:
            lines.append(
                f"- **{name}**: no policy materially beats no prefetch "
                f"(best speedup {speedup:.2f}x, hit ratio "
                f"{100 * winner_hit:.1f}% up {100 * hit_gain:+.1f} pp)")
        else:
            lines.append(
                f"- **{name}** - best policy: **{winner['mode']}** "
                f"(latency {float(winner['mean_latency_us']):.1f} us, "
                f"**{speedup:.2f}x** vs no prefetch; hit ratio "
                f"{100 * winner_hit:.1f}% up {100 * hit_gain:+.1f} pp, "
                f"precision {100 * float(winner['precision']):.1f}%, "
                f"{int(winner['unused'])} unused prefetches)")

        top_hit = max(completed, key=lambda row: float(row["hit_ratio"]))
        if top_hit is not winner and float(top_hit["hit_ratio"]) - winner_hit > 0.005:
            lines.append(
                f"  - best hit ratio: **{top_hit['mode']}** "
                f"({100 * float(top_hit['hit_ratio']):.1f}% vs "
                f"{100 * winner_hit:.1f}% for the latency winner)")

        # --- RELATIVE PERCENTAGE GAIN COMPARISONS FOR ADAPTIVE ---
        adaptive = next((row for row in completed if row["mode"] in ("adaptive", "adaptive_pseudo")), None)
        other_baselines = [row for row in completed if row["mode"] not in ("adaptive", "adaptive_pseudo")]

        if adaptive and other_baselines:
            adaptive_hit = float(adaptive["hit_ratio"])
            adaptive_unused = int(adaptive["unused"])
            adaptive_inf = float(adaptive["inference_us"])

            # 1. Relative Hit Ratio Gain vs Best Non-Adaptive Baseline
            best_other_hit_row = max(other_baselines, key=lambda r: float(r["hit_ratio"]))
            best_other_hit = float(best_other_hit_row["hit_ratio"])

            if best_other_hit > 0:
                rel_hit_gain = ((adaptive_hit - best_other_hit) / best_other_hit) * 100
                if adaptive["mode"] != winner["mode"]:
                    lines.append(
                        f"  - **Adaptive vs Best Baseline Hit Ratio ({best_other_hit_row['mode']})**: "
                        f"{rel_hit_gain:+.1f}% relative gain ({100 * adaptive_hit:.1f}% vs {100 * best_other_hit:.1f}%)"
                    )

            # 2. Cache Pollution Reduction vs High-Unused Baselines (Sequential / LSTM)
            for mode_name in ("sequential", "lstm"):
                target_row = next((r for r in other_baselines if r["mode"] == mode_name), None)
                if target_row and target_row["unused"] is not None:
                    target_unused = int(target_row["unused"])
                    if target_unused > 0:
                        pollution_reduction = ((target_unused - adaptive_unused) / target_unused) * 100
                        lines.append(
                            f"  - **Adaptive Cache Pollution Reduction vs {mode_name}**: "
                            f"**{pollution_reduction:.1f}% fewer wasted prefetches** ({adaptive_unused} vs {target_unused})"
                        )

            # 3. Inference Time Speedup vs LSTM Baseline
            lstm_row = next((r for r in other_baselines if r["mode"] == "lstm"), None)
            if lstm_row and float(lstm_row["inference_us"]) > 0 and adaptive_inf > 0:
                lstm_inf = float(lstm_row["inference_us"])
                inf_speedup = lstm_inf / adaptive_inf
                lines.append(
                    f"  - **Adaptive CPU Efficiency vs LSTM**: "
                    f"**{inf_speedup:.1f}x faster inference** ({adaptive_inf:.1f} us vs {lstm_inf:.1f} us)"
                )

        skipped = [row["mode"] for row in group if row["status"] != "ok"]
        if skipped:
            lines.append(f"  - skipped: {', '.join(map(str, skipped))}")

    return "\n".join(lines)


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
