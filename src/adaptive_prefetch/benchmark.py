"""Fair cache-policy comparison and labelled synthetic drift diagnostics."""

from __future__ import annotations

import csv
import random
from pathlib import Path

from .features import FEATURE_NAMES, extract
from .model import DEFAULT_CLASSIFIER, make_classifier
from .simulator import DEFAULT_CAPACITY, LatencyModel, replay
from .trace import (CLASSES, Request, synthetic_dataset, synthetic_window,
                    transition_trace)

MODES = ("none", "sequential", "strided", "stride", "markov", "lstm",
         "adaptive", "adaptive_evidence")
COLUMNS = ("dataset", "mode", "status", "hit_ratio", "precision", "recall",
           "unused", "total_prefetches", "useful_prefetches",
           "mean_latency_us", "speedup_vs_none", "inference_us",
           "inference_us_per_request")


def make_model(seed: int = 42, examples_per_class: int = 100,
               classifier: str = DEFAULT_CLASSIFIER, contextual: bool = True):
    """Fit a classifier on labelled synthetic windows.

    With ``contextual=True`` the windows of a class are laid out as one
    continuous stream so ``WindowContext`` accumulates block ages and
    previous-window features exactly as it does at replay time. Training on
    shuffled isolated windows would teach the model that the four contextual
    features are always zero, which is a train/deploy mismatch.
    """
    from .features import WindowContext, dominant_stride

    model = make_classifier(classifier, len(FEATURE_NAMES))
    if not contextual:
        for window, label in synthetic_dataset(examples_per_class, seed):
            model.update(extract(window), label)
        return model

    for label in CLASSES:
        rng = random.Random(seed)
        context = WindowContext()
        # One continuous run so block ages and window-to-window persistence
        # carry real values rather than the cold-start defaults.
        flat: list = []
        time = 0.0
        for _ in range(examples_per_class):
            window = synthetic_window(label, rng, 32, time)
            flat.extend(window)
            time = window[-1].timestamp_ms
        for start in range(0, len(flat), 32):
            window = flat[start:start + 32]
            if len(window) < 32:
                break
            model.update(context.observe(window, dominant_stride(window)), label)
    return model


def benchmark_dataset(name: str, requests: list[Request], *, seed: int = 42,
                      capacity: int = DEFAULT_CAPACITY, window_size: int = 32,
                      latency: LatencyModel | None = None,
                      lstm_model_path: str | None = None,
                      online_real: bool = False,
                      classifier: str = DEFAULT_CLASSIFIER) -> list[dict[str, object]]:
    if not requests:
        raise ValueError("benchmark trace is empty")
    baseline, _ = replay(requests, capacity, window_size, "none",
                         latency_model=latency)
    baseline_us = baseline.mean_access_latency_us
    rows: list[dict[str, object]] = []
    modes = MODES + (("adaptive_pseudo",) if online_real else ())
    for mode in modes:
        actual_mode = "adaptive" if mode == "adaptive_pseudo" else mode
        model = (make_model(seed, classifier=classifier)
                 if actual_mode == "adaptive" else None)
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
        # Adaptive classifies once per window, the per-request predictors run
        # on every request. Comparing inference_us (per call) across those two
        # granularities is a 32x error, so normalise to per request as well.
        per_request = (metrics.inference_ms * 1000.0 / len(requests)
                       if requests else 0.0)
        rows.append({
            "dataset": name, "mode": mode, "status": "ok",
            "hit_ratio": metrics.hit_ratio,
            "precision": metrics.prefetch_precision,
            "recall": metrics.prefetch_recall,
            "unused": metrics.unused_prefetches,
            "total_prefetches": metrics.prefetches,
            "useful_prefetches": metrics.useful_prefetches,
            "mean_latency_us": current_us,
            "speedup_vs_none": (baseline_us / current_us if current_us else None),
            "inference_us": metrics.inference_us_per_call,
            "inference_us_per_request": per_request,
            "_read_blocks": metrics.read_blocks,
            "_prefetchable_misses": metrics.prefetchable_misses,
            "_total_latency_us": current_us * metrics.read_blocks,
            "_inference_calls": metrics.inference_calls,
            "_total_inference_us": metrics.inference_us_per_call * metrics.inference_calls,
            "_requests": len(requests),
        })
    return rows


def _request_count(group: list[dict[str, object]]) -> int:
    """Total request count across a group of per-dataset rows."""
    return sum(int(row.get("_requests") or 0) for row in group)


def aggregate_results(rows: list[dict[str, object]],
                      dataset_name: str = "ALL_TRACES") -> list[dict[str, object]]:
    """Combine completed per-trace rows using request-weighted counters."""
    modes = list(dict.fromkeys(str(row["mode"]) for row in rows))
    aggregates: list[dict[str, object]] = []
    dataset_total = len({str(row["dataset"]) for row in rows})
    for mode in modes:
        group = [row for row in rows if row["mode"] == mode]
        status = "ok"
        if any(row["status"] != "ok" for row in group):
            # Aggregate over the datasets where this mode actually ran, and
            # record the coverage. Dropping the whole mode because one trace
            # skipped it discarded real measurements.
            group = [row for row in group if row["status"] == "ok"]
            if not group:
                aggregates.append({
                    "dataset": dataset_name,
                    "mode": mode,
                    "status": "skipped: no trace produced a result",
                    **{column: None for column in COLUMNS[3:]},
                })
                continue
            status = f"ok (partial: {len(group)}/{dataset_total} traces)"

        read_blocks = sum(int(row["_read_blocks"]) for row in group)
        total_prefetches = sum(int(row["total_prefetches"]) for row in group)
        useful_prefetches = sum(int(row["useful_prefetches"]) for row in group)
        prefetchable_misses = sum(int(row["_prefetchable_misses"]) for row in group)
        total_latency = sum(float(row["_total_latency_us"]) for row in group)
        # The baseline must cover exactly the datasets this mode covered, or
        # the speedup compares different request sets.
        covered = {str(row["dataset"]) for row in group}
        baseline_group = [row for row in rows
                          if row["mode"] == "none" and row["status"] == "ok"
                          and str(row["dataset"]) in covered]
        baseline_latency = sum(float(row["_total_latency_us"])
                               for row in baseline_group)
        inference_calls = sum(int(row["_inference_calls"]) for row in group)
        aggregates.append({
            "dataset": dataset_name,
            "mode": mode,
            "status": status,
            "hit_ratio": (sum(float(row["hit_ratio"]) * int(row["_read_blocks"])
                               for row in group) / read_blocks
                          if read_blocks else 0.0),
            "precision": useful_prefetches / total_prefetches if total_prefetches else 0.0,
            "recall": (useful_prefetches /
                       (useful_prefetches + prefetchable_misses)
                       if useful_prefetches + prefetchable_misses else 0.0),
            "unused": total_prefetches - useful_prefetches,
            "total_prefetches": total_prefetches,
            "useful_prefetches": useful_prefetches,
            "mean_latency_us": total_latency / read_blocks if read_blocks else 0.0,
            "speedup_vs_none": (baseline_latency / total_latency
                                if total_latency else None),
            "inference_us": (
                sum(float(row["_total_inference_us"]) for row in group) /
                inference_calls if inference_calls else 0.0),
            "inference_us_per_request": (
                sum(float(row["_total_inference_us"]) for row in group) /
                _request_count(group) if _request_count(group) else 0.0),
        })
    return aggregates


def drift_report(seed: int = 42, windows_per_class: int = 8,
                 window_size: int = 32,
                 classifier: str = DEFAULT_CLASSIFIER) -> list[dict[str, object]]:
    requests, labels = transition_trace(seed + 200000, windows_per_class,
                                        window_size)
    rows = []
    for variant, update in (("frozen", False), ("labelled_online", True)):
        model = make_model(seed, classifier=classifier)
        _, predictions = replay(requests, window_size=window_size,
                                model=model, labels=labels if update else None,
                                online_updates=update)
        for phase_number, label in enumerate(CLASSES):
            start = phase_number * windows_per_class
            phase = predictions[start:start + windows_per_class]
            if not phase:
                continue
            lag = next((offset for offset, (_, predicted, _) in enumerate(phase)
                        if predicted == label), None)
            rows.append({"variant": variant, "phase": label,
                         "switch_lag_windows": lag,
                         "phase_accuracy": sum(predicted == label
                                               for _, predicted, _ in phase) / len(phase)})
    return rows


def _speedup(row: dict[str, object]) -> float | None:
    """Speedup vs no prefetch, or None when the cost model makes it undefined."""
    value = row.get("speedup_vs_none")
    return None if value is None else float(value)


def _wasted_fraction(row: dict[str, object]) -> float | None:
    """Fraction of issued prefetches that were never used, or None if unknown."""
    total = row.get("total_prefetches")
    if not total:
        return None
    return int(row["unused"]) / int(total)


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

        rankable = [r for r in completed if _speedup(r) is not None]
        if not rankable:
            lines.append(f"- **{name}**: no read requests, so every cache metric is 0/0")
            continue
        winner = max(rankable, key=_speedup)
        baseline_hit = float(baseline["hit_ratio"])
        winner_hit = float(winner["hit_ratio"])
        hit_gain = winner_hit - baseline_hit
        speedup = _speedup(winner)

        if speedup is None or speedup <= 1.005:
            detail = "n/a" if speedup is None else f"{speedup:.2f}x"
            lines.append(
                f"- **{name}**: no policy materially beats no prefetch "
                f"(best speedup {detail}, hit ratio "
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
        # The frozen `adaptive` row is the one to compare. `adaptive_pseudo` is
        # the self-training experiment and must not silently win this lookup:
        # benchmark_dataset appends it after MODES, so `next(...)` used to
        # always resolve to the frozen row and discard the pseudo result.
        adaptive = next((row for row in completed if row["mode"] == "adaptive"), None)
        pseudo = next((row for row in completed if row["mode"] == "adaptive_pseudo"), None)
        other_baselines = [row for row in completed
                           if row["mode"] not in ("adaptive", "adaptive_pseudo")]

        if adaptive and pseudo:
            lines.append(
                f"  - **self-training vs frozen adaptive**: hit ratio "
                f"{100 * float(pseudo['hit_ratio']):.1f}% vs "
                f"{100 * float(adaptive['hit_ratio']):.1f}%, "
                f"{int(pseudo['unused'])} vs {int(adaptive['unused'])} unused. "
                f"Pseudo-labels are the model's own predictions, so this "
                f"measures self-consistency, not accuracy.")

        if adaptive and other_baselines:
            adaptive_hit = float(adaptive["hit_ratio"])

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

            # 2. Wasted-prefetch fraction, volume-normalised. Comparing raw
            # unused counts across modes that issue very different numbers of
            # prefetches is meaningless: a policy that simply prefetches less
            # always "wins". Report the wasted fraction instead, and never
            # print a negative reduction as "fewer".
            for mode_name in ("sequential", "lstm", "markov"):
                target_row = next((r for r in other_baselines if r["mode"] == mode_name), None)
                if target_row is None:
                    continue
                adaptive_waste = _wasted_fraction(adaptive)
                target_waste = _wasted_fraction(target_row)
                if adaptive_waste is None or target_waste is None:
                    continue
                delta = (target_waste - adaptive_waste) * 100
                word = "less" if delta >= 0 else "MORE"
                lines.append(
                    f"  - **Wasted-prefetch fraction vs {mode_name}**: "
                    f"{adaptive_waste * 100:.1f}% vs {target_waste * 100:.1f}% "
                    f"({abs(delta):.1f} pp {word} waste; "
                    f"{int(adaptive['unused'])}/{int(adaptive['total_prefetches'])} "
                    f"vs {int(target_row['unused'])}/{int(target_row['total_prefetches'])} unused)")

            # 3. Inference cost per *request*, not per call. Adaptive runs once
            # per 32-request window; the predictors run once per request, so
            # comparing per-call figures overstated adaptive's advantage ~32x.
            lstm_row = next((r for r in other_baselines if r["mode"] == "lstm"), None)
            if lstm_row is not None:
                adaptive_inf = float(adaptive.get("inference_us_per_request") or 0.0)
                lstm_inf = float(lstm_row.get("inference_us_per_request") or 0.0)
                if lstm_inf > 0 and adaptive_inf > 0:
                    inf_speedup = lstm_inf / adaptive_inf
                    lines.append(
                        f"  - **Adaptive CPU cost vs LSTM (per request)**: "
                        f"**{inf_speedup:.1f}x lower** "
                        f"({adaptive_inf:.3f} us vs {lstm_inf:.3f} us/request)")
                else:
                    lines.append(
                        "  - **Adaptive CPU cost vs LSTM**: not comparable "
                        "(one mode issued no inference calls)")

        skipped = [row["mode"] for row in group if row["status"] != "ok"]
        if skipped:
            lines.append(f"  - skipped: {', '.join(map(str, skipped))}")

    return "\n".join(lines)


def markdown_table(rows: list[dict[str, object]]) -> str:
    names = ("Dataset", "Mode", "Status", "Hit %", "Precision %", "Oracle recall %",
             "Unused", "Total Prefetches", "Useful Prefetches", "Mean us",
             "Speedup", "Inference us", "Inference us/req")
    lines = ["| " + " | ".join(names) + " |",
             "| " + " | ".join("---" for _ in names) + " |"]
    for row in rows:
        def fraction(key: str) -> str:
            value = row[key]
            return f"{100 * value:.2f}" if isinstance(value, (int, float)) else "-"
        def numeric(key: str, digits: int = 2) -> str:
            value = row.get(key)
            return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "-"
        cells = [str(row["dataset"]), str(row["mode"]), str(row["status"]),
                 fraction("hit_ratio"), fraction("precision"), fraction("recall"),
                 str(row["unused"]) if row["unused"] is not None else "-",
                 str(row["total_prefetches"]) if row["total_prefetches"] is not None else "-",
                 str(row["useful_prefetches"]) if row["useful_prefetches"] is not None else "-",
                 numeric("mean_latency_us"), numeric("speedup_vs_none"),
                 numeric("inference_us", 3), numeric("inference_us_per_request", 3)]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_results_csv(path: str | Path, rows: list[dict[str, object]]) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows({column: row.get(column) for column in COLUMNS}
                         for row in rows)
