"""Full-collection mode sweep: every MSRC trace, every mode, one winner table.

The MSRC collection is 11.9 GB across 32 files and a ``Request`` costs ~430
bytes resident, so the whole set cannot be loaded. This module streams each
file in bounded chunks (``trace.iter_csv_chunks``) and replays through
``simulator.replay_stream``, which carries cache, metrics, partial-window
state and block ages across chunk boundaries.

Two limits are always reported, never silent:

* **Requests per trace.** The default is a sample, because replaying ~200M
  requests through eight policies in Python takes hours. ``--full`` removes
  the cap.
* **``prefetch_recall`` is omitted** on streamed runs. It needs whole-stream
  future knowledge; the ranking metrics (hit ratio, precision, wasted I/O,
  modelled cost) are exact.
"""

from __future__ import annotations

import time
from pathlib import Path
from statistics import mean

from .benchmark import MODES
from .datasets import MSRC_TRACE_DIR, LSTM_ARTIFACT
from .simulator import DEFAULT_CAPACITY, LatencyModel, replay_stream
from .trace import iter_csv_chunks

#: Requests replayed per trace by default. 250k * 32 traces = 8M requests,
#: which is ample for a per-trace hit-ratio comparison and finishes in
#: minutes rather than hours.
DEFAULT_REQUESTS_PER_TRACE = 250_000
#: Chunk size for streaming. 250k requests is ~108 MB resident, comfortably
#: inside a small machine even with several other things running.
DEFAULT_CHUNK = 250_000

RANKING_MODES = ("none", "sequential", "strided", "stride", "markov", "lstm",
                 "adaptive", "adaptive_evidence")


def discover_traces(directory: Path | None = None) -> list[Path]:
    """Every ``.revised`` trace, in a stable order."""
    root = directory or MSRC_TRACE_DIR
    if not root.is_dir():
        return []
    return sorted(root.glob("*.revised"))


def trace_size_bytes(path: Path) -> int:
    return path.stat().st_size


def capped_chunks(path: Path, requests_per_trace: int | None,
                  chunk_requests: int):
    """Stream chunks, stopping at ``requests_per_trace`` when set."""
    emitted = 0
    for chunk in iter_csv_chunks(path, "revised", chunk_requests):
        if requests_per_trace is not None:
            if emitted >= requests_per_trace:
                return
            if emitted + len(chunk) > requests_per_trace:
                chunk = chunk[:requests_per_trace - emitted]
        emitted += len(chunk)
        yield chunk


def sweep_collection(trace_dir: Path | None = None,
                     capacity: int = DEFAULT_CAPACITY,
                     window_size: int = 32,
                     lstm_model_path: str | None = None,
                     requests_per_trace: int | None = DEFAULT_REQUESTS_PER_TRACE,
                     chunk_requests: int = DEFAULT_CHUNK,
                     latency: LatencyModel | None = None,
                     modes=RANKING_MODES, progress=None
                     ) -> tuple[list[dict], list[dict]]:
    """Replay every trace under every mode. Returns (rows, per-trace meta).

    ``rows`` are plain dicts so they serialise straight to CSV.
    """
    traces = discover_traces(trace_dir)
    if not traces:
        return [], []
    if lstm_model_path is None and LSTM_ARTIFACT.is_file():
        from .lstm import load_predictor
        try:
            load_predictor(LSTM_ARTIFACT)
            lstm_model_path = str(LSTM_ARTIFACT)
        except (ValueError, RuntimeError):
            lstm_model_path = None
    rows: list[dict] = []
    metas: list[dict] = []
    for path in traces:
        t0 = time.time()
        for mode in modes:
            model = None
            if mode == "adaptive":
                from .benchmark import make_model
                model = make_model(42)
            metrics = replay_stream(
                capped_chunks(path, requests_per_trace, chunk_requests),
                capacity=capacity, window_size=window_size, mode=mode,
                model=model,
                lstm_model_path=lstm_model_path if mode == "lstm" else None,
                latency_model=latency)
            rows.append({
                "trace": path.stem,
                "mode": mode,
                "hit_ratio": metrics.hit_ratio,
                "precision": metrics.prefetch_precision,
                "unused": metrics.unused_prefetches,
                "total_prefetches": metrics.prefetches,
                "mean_us": metrics.mean_access_latency_us,
                "policy_switches": metrics.policy_switches,
                "read_blocks": metrics.read_blocks,
            })
        meta = {"trace": path.stem,
                "size_mb": trace_size_bytes(path) / 1e6,
                "seconds": time.time() - t0}
        metas.append(meta)
        if progress is not None:
            progress(meta, len(rows))
    return rows, metas


def aggregate_by_mode(rows: list[dict]) -> list[dict]:
    """One summary row per mode across every trace.

    Traces with no read blocks are counted but excluded from the averages:
    their metrics are 0/0 and would drag every mean toward zero.
    """
    summary = []
    for mode in dict.fromkeys(row["mode"] for row in rows):
        group = [row for row in rows if row["mode"] == mode]
        scored = [row for row in group if row["read_blocks"] > 0]
        summary.append({
            "mode": mode,
            "traces": len(group),
            "scored": len(scored),
            "mean_hit": mean(r["hit_ratio"] for r in scored) if scored else 0.0,
            "mean_prec": mean(r["precision"] for r in scored) if scored else 0.0,
            "mean_us": mean(r["mean_us"] for r in scored) if scored else 0.0,
            "total_prefetches": sum(r["total_prefetches"] for r in group),
            "total_unused": sum(r["unused"] for r in group),
        })
    return summary


def per_trace_winners(rows: list[dict]) -> dict[str, int]:
    """How many traces each mode wins on hit ratio."""
    counts: dict[str, int] = {}
    for trace in dict.fromkeys(row["trace"] for row in rows):
        group = [r for r in rows if r["trace"] == trace and r["read_blocks"] > 0]
        if not group:
            continue
        best = max(group, key=lambda r: r["hit_ratio"])
        counts[best["mode"]] = counts.get(best["mode"], 0) + 1
    return counts


def winners(rows: list[dict]) -> dict:
    """Overall winner per metric. Traces with no reads are excluded."""
    by_mode: dict[str, list[dict]] = {}
    for row in rows:
        if row["read_blocks"] <= 0:
            continue
        by_mode.setdefault(row["mode"], []).append(row)
    if not by_mode:
        return {}
    modes = list(by_mode)

    def avg(mode: str, key: str) -> float:
        return mean(r[key] for r in by_mode[mode])

    return {
        "hit_ratio": max(modes, key=lambda m: avg(m, "hit_ratio")),
        "precision": max(modes, key=lambda m: avg(m, "precision")),
        "cost": min(modes, key=lambda m: avg(m, "mean_us")),
        "waste": min(modes, key=lambda m: sum(r["unused"] for r in by_mode[m])),
    }
