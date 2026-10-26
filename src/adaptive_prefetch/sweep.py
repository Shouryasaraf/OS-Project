"""Full-collection mode sweep: every trace, every mode, one winner table.

The MSRC collection is 11.9 GB across 32 files and a ``Request`` costs ~430
bytes resident, so the whole set cannot be loaded. This module streams each
file in bounded chunks (``trace.iter_csv_chunks``) and replays through
``simulator.replay_stream``, which carries cache, metrics, partial-window
state and block ages across chunk boundaries.

It sweeps MSRC ``.revised`` text traces and libCacheSim ``.oracleGeneral.zst``
block traces alike. The format is chosen per file from its name, and the
oracle loader is reached only through ``iter_csv_chunks(path, "oracle", n)``.

Limits that are always reported, never silent:

* **Requests per trace.** The default is a sample, because replaying ~200M
  requests through twelve policies in Python takes hours. ``--full`` removes
  the cap.
* **``prefetch_recall`` is omitted** on streamed runs. It needs whole-stream
  future knowledge; the ranking metrics (hit ratio, precision, wasted I/O,
  device I/Os, modelled cost) are exact.
* **Oracle traces carry no service times**, so a sweep over them must be
  given an explicit ``latency_model``. Falling back to the assumed 5/100/50
  model silently moved the gated modes' break-even from 1.0101 to 0.526
  (M19), so every row records the ``break_even`` it ran under.
* **A mode that needs a trained artifact** (``lstm``) and has none produces a
  ``skipped:`` row with ``None`` metrics instead of aborting the sweep.
"""

from __future__ import annotations

import time
from pathlib import Path
from statistics import mean

from .baselines import DEFAULT_LINUX_RA_MAX_BLOCKS
from .datasets import MSRC_TRACE_DIR, LSTM_ARTIFACT
from .features import DEFAULT_AGE_TABLE_BOUND
from .simulator import (DEFAULT_CAPACITY, DEFAULT_READ_AHEAD_DEPTH,
                        LatencyModel, replay_stream)
from .trace import iter_csv_chunks

#: Requests replayed per trace by default. 250k * 32 traces = 8M requests,
#: which is ample for a per-trace hit-ratio comparison and finishes in
#: minutes rather than hours.
DEFAULT_REQUESTS_PER_TRACE = 250_000
#: Chunk size for streaming. 250k requests is ~108 MB resident, comfortably
#: inside a small machine even with several other things running.
DEFAULT_CHUNK = 250_000

RANKING_MODES = ("none", "sequential", "strided", "stride", "markov", "lstm",
                 "adaptive", "adaptive_evidence", "guard", "depth_adaptive",
                 "correlate", "deep", "linux_ra")

#: Modes that need a trained artifact. Without one they are reported as
#: ``skipped:`` rows; every other mode must run or raise.
OPTIONAL_MODES = ("lstm",)

#: File patterns ``discover_traces`` looks for, recursively.
TRACE_PATTERNS = ("*.revised", "*.oracleGeneral.zst")

#: Every column of a sweep row, in CSV order. The first twelve are the
#: original ``msrc_sweep.csv`` columns; ``observed_precision``,
#: ``precision_margin`` and ``observed_stride`` must stay (D12).
ROW_COLUMNS = (
    "trace", "mode", "hit_ratio", "precision", "unused", "total_prefetches",
    "mean_us", "policy_switches", "read_blocks",
    "observed_precision", "precision_margin", "observed_stride",
    "status", "useful_prefetches", "read_hits",
    # Device-I/O accounting (D18): positioning work, not just blocks.
    "demand_ios", "prefetch_ios", "prefetch_ios_mergeable", "device_ios",
    "blocks_moved", "inference_calls", "inference_ms",
    # The configuration the row ran under.
    "capacity", "window_size", "read_ahead_depth", "max_blocks",
    "age_table_bound", "age_table_evictions", "break_even",
    # What the replayed slice was.
    "format", "corpus", "merge", "size_unit", "requests", "blocks",
    "hours_covered", "ops_known", "ts_resolution_s",
)

#: Metric columns that are ``None`` on a skipped row.
_METRIC_COLUMNS = (
    "hit_ratio", "precision", "unused", "total_prefetches", "mean_us",
    "policy_switches", "read_blocks", "observed_precision",
    "precision_margin", "observed_stride", "useful_prefetches", "read_hits",
    "demand_ios", "prefetch_ios", "prefetch_ios_mergeable", "device_ios",
    "blocks_moved", "inference_calls", "inference_ms", "age_table_evictions",
    "requests", "blocks", "hours_covered",
)

_ORACLE_SUFFIXES = (".oracleGeneral.zst", ".oracleGeneral")


def discover_traces(directory: Path | None = None,
                    patterns=TRACE_PATTERNS) -> list[Path]:
    """Every trace under ``directory`` matching ``patterns``, in a stable order.

    The search is recursive because the block-trace collection keeps one
    sub-directory per corpus; the flat MSRC directory is unaffected.
    """
    root = Path(directory) if directory is not None else MSRC_TRACE_DIR
    if not root.is_dir():
        return []
    found: set[Path] = set()
    for pattern in patterns:
        found.update(path for path in root.rglob(pattern) if path.is_file())
    return sorted(found)


def trace_format(path: Path) -> str:
    """The loader profile for ``path``, decided by its name alone."""
    name = Path(path).name
    if name.endswith(".revised"):
        return "revised"
    if name.endswith(_ORACLE_SUFFIXES):
        return "oracle"
    raise ValueError(f"cannot tell the trace format of {name!r}; "
                     "expected .revised or .oracleGeneral[.zst]")


def trace_name(path: Path) -> str:
    """Short trace name: ``msr_hm_0`` for ``msr_hm_0.oracleGeneral.zst``."""
    name = Path(path).name
    for suffix in _ORACLE_SUFFIXES:
        if name.endswith(suffix):
            return name[:-len(suffix)]
    return Path(path).stem


def trace_size_bytes(path: Path) -> int:
    return path.stat().st_size


def _corpus_from_name(name: str) -> str:
    for prefix, corpus in (("msr_", "msr"), ("fiu_", "fiu"),
                           ("alibaba", "alibaba"), ("tencent", "tencent")):
        if name.startswith(prefix):
            return corpus
    return "unknown"


def trace_profile(path: Path, format: str | None = None) -> dict:
    """What a sweep row records about the trace itself.

    For oracle traces the reconstruction (``merge``, ``size_unit``) comes
    from the loader's own ``oracle_profile`` when it is available, so the row
    states the rule actually applied rather than an assumption.
    """
    fmt = format or trace_format(path)
    if fmt == "revised":
        return {"format": fmt, "corpus": "msrc", "merge": "none",
                "size_unit": "sectors", "ops_known": True,
                "ts_resolution_s": None}
    if fmt == "oracle":
        profile: dict = {}
        try:
            from .trace import oracle_profile  # type: ignore[attr-defined]
        except ImportError:
            oracle_profile = None
        if oracle_profile is not None:
            profile = oracle_profile(path)
        return {"format": fmt,
                "corpus": profile.get("corpus")
                or _corpus_from_name(trace_name(path)),
                "merge": profile.get("merge", "auto"),
                "size_unit": profile.get("size_unit", "auto"),
                # oracleGeneral records carry no read/write flag and whole-
                # second timestamps: every request replays as a read.
                "ops_known": False, "ts_resolution_s": 1}
    return {"format": fmt, "corpus": "unknown", "merge": None,
            "size_unit": None, "ops_known": None, "ts_resolution_s": None}


def capped_chunks(path: Path, requests_per_trace: int | None,
                  chunk_requests: int, format: str | None = None):
    """Stream chunks, stopping at ``requests_per_trace`` when set.

    ``format`` defaults to :func:`trace_format`; it used to be hard-coded to
    ``revised``, which made every other trace kind unsweepable.
    """
    fmt = format or trace_format(path)
    emitted = 0
    for chunk in iter_csv_chunks(path, fmt, chunk_requests):
        if requests_per_trace is not None:
            if emitted >= requests_per_trace:
                return
            if emitted + len(chunk) > requests_per_trace:
                chunk = chunk[:requests_per_trace - emitted]
        emitted += len(chunk)
        yield chunk


class _SliceStats:
    """Counts what a replay actually consumed, while it streams past."""

    def __init__(self):
        self.requests = 0
        self.blocks = 0
        self.first_ms: float | None = None
        self.last_ms: float | None = None

    def wrap(self, chunks):
        for chunk in chunks:
            if chunk:
                if self.first_ms is None:
                    self.first_ms = chunk[0].timestamp_ms
                self.last_ms = chunk[-1].timestamp_ms
                self.requests += len(chunk)
                self.blocks += sum(request.size_blocks for request in chunk)
            yield chunk

    @property
    def hours_covered(self) -> float:
        if self.first_ms is None or self.last_ms is None:
            return 0.0
        return (self.last_ms - self.first_ms) / 3_600_000.0


def _skipped_row(base: dict, reason: str) -> dict:
    row = dict(base)
    row["status"] = f"skipped: {reason}"
    for column in _METRIC_COLUMNS:
        row[column] = None
    return row


def sweep_trace(path: Path, mode: str, *,
                capacity: int = DEFAULT_CAPACITY, window_size: int = 32,
                requests_per_trace: int | None = DEFAULT_REQUESTS_PER_TRACE,
                chunk_requests: int = DEFAULT_CHUNK,
                latency_model: LatencyModel | None = None,
                lstm_model_path: str | None = None,
                read_ahead_depth: int = DEFAULT_READ_AHEAD_DEPTH,
                max_blocks: int = DEFAULT_LINUX_RA_MAX_BLOCKS,
                age_table_bound: int | None = DEFAULT_AGE_TABLE_BOUND,
                format: str | None = None, model=None,
                profile: dict | None = None) -> dict:
    """Replay one trace under one mode and return its sweep row.

    ``read_ahead_depth`` is the ``deep`` depth and ``max_blocks`` the
    ``linux_ra`` window ceiling; both are recorded in every row whichever
    mode ran, so a row is self-describing. ``model`` is the classifier for
    ``adaptive`` (``make_model(42)`` when omitted). An optional mode with no
    usable artifact yields a ``skipped:`` row instead of raising.
    """
    path = Path(path)
    fmt = format or trace_format(path)
    if fmt == "oracle" and latency_model is None:
        raise ValueError(
            f"{path.name}: oracle traces carry no service times, so the "
            "sweep needs an explicit latency_model")
    latency = latency_model or LatencyModel()
    base = {
        "trace": trace_name(path), "mode": mode,
        "capacity": capacity, "window_size": window_size,
        "read_ahead_depth": read_ahead_depth, "max_blocks": max_blocks,
        "age_table_bound": age_table_bound,
        "break_even": latency.break_even_precision(),
        **(profile if profile is not None else trace_profile(path, fmt)),
    }
    if mode in OPTIONAL_MODES and not lstm_model_path:
        return _skipped_row(base, "no trained artifact")
    if mode == "adaptive" and model is None:
        from .benchmark import make_model
        model = make_model(42)
    stats = _SliceStats()
    try:
        metrics = replay_stream(
            stats.wrap(capped_chunks(path, requests_per_trace, chunk_requests,
                                     fmt)),
            capacity=capacity, window_size=window_size, mode=mode,
            model=model,
            lstm_model_path=lstm_model_path if mode == "lstm" else None,
            latency_model=latency, read_ahead_depth=read_ahead_depth,
            linux_ra_max_blocks=max_blocks, age_table_bound=age_table_bound)
    except (ValueError, RuntimeError, FileNotFoundError, ImportError) as exc:
        if mode not in OPTIONAL_MODES:
            raise
        return _skipped_row(base, str(exc))
    row = dict(base)
    row.update({
        "status": "ok",
        "hit_ratio": metrics.hit_ratio,
        "precision": metrics.prefetch_precision,
        "unused": metrics.unused_prefetches,
        "total_prefetches": metrics.prefetches,
        "useful_prefetches": metrics.useful_prefetches,
        "read_hits": metrics.read_hits,
        "mean_us": metrics.mean_access_latency_us,
        "policy_switches": metrics.policy_switches,
        # Why a feedback-gated mode throttled what it did. Without
        # these the row says a mode issued 2,077 prefetches and gives
        # no way to tell a deliberate decision from a stuck gate.
        "observed_precision": metrics.observed_precision,
        "precision_margin": metrics.precision_margin,
        "observed_stride": metrics.observed_stride,
        "read_blocks": metrics.read_blocks,
        "demand_ios": metrics.demand_ios,
        "prefetch_ios": metrics.prefetch_ios,
        "prefetch_ios_mergeable": metrics.prefetch_ios_mergeable,
        "device_ios": metrics.device_ios,
        "blocks_moved": metrics.blocks_moved,
        "inference_calls": metrics.inference_calls,
        "inference_ms": metrics.inference_ms,
        "age_table_evictions": metrics.age_table_evictions,
        "requests": stats.requests,
        "blocks": stats.blocks,
        "hours_covered": stats.hours_covered,
    })
    return row


def sweep_collection(trace_dir: Path | None = None,
                     capacity: int = DEFAULT_CAPACITY,
                     window_size: int = 32,
                     lstm_model_path: str | None = None,
                     requests_per_trace: int | None = DEFAULT_REQUESTS_PER_TRACE,
                     chunk_requests: int = DEFAULT_CHUNK,
                     latency_model: LatencyModel | None = None,
                     modes=RANKING_MODES, progress=None, *,
                     read_ahead_depth: int = DEFAULT_READ_AHEAD_DEPTH,
                     max_blocks: int = DEFAULT_LINUX_RA_MAX_BLOCKS,
                     age_table_bound: int | None = DEFAULT_AGE_TABLE_BOUND,
                     traces: list[Path] | None = None,
                     patterns=TRACE_PATTERNS,
                     ) -> tuple[list[dict], list[dict]]:
    """Replay every trace under every mode. Returns (rows, per-trace meta).

    ``rows`` are plain dicts (columns in :data:`ROW_COLUMNS`) so they
    serialise straight to CSV. ``traces`` replaces discovery with an explicit
    list. ``latency_model`` is mandatory when any trace is an oracle trace.
    """
    traces = (list(traces) if traces is not None
              else discover_traces(trace_dir, patterns))
    if not traces:
        return [], []
    if latency_model is None and any(trace_format(p) == "oracle"
                                     for p in traces):
        raise ValueError("oracle traces carry no service times, so the sweep "
                         "needs an explicit latency_model")
    if lstm_model_path is None and "lstm" in modes and LSTM_ARTIFACT.is_file():
        from .lstm import load_predictor
        try:
            load_predictor(LSTM_ARTIFACT)
            lstm_model_path = str(LSTM_ARTIFACT)
        except (ValueError, RuntimeError, ImportError):
            lstm_model_path = None
    rows: list[dict] = []
    metas: list[dict] = []
    for path in traces:
        t0 = time.time()
        fmt = trace_format(path)
        profile = trace_profile(path, fmt)
        for mode in modes:
            rows.append(sweep_trace(
                path, mode, capacity=capacity, window_size=window_size,
                requests_per_trace=requests_per_trace,
                chunk_requests=chunk_requests, latency_model=latency_model,
                lstm_model_path=lstm_model_path,
                read_ahead_depth=read_ahead_depth, max_blocks=max_blocks,
                age_table_bound=age_table_bound, format=fmt, profile=profile))
        meta = {"trace": trace_name(path), "format": fmt,
                "size_mb": trace_size_bytes(path) / 1e6,
                "seconds": time.time() - t0}
        metas.append(meta)
        if progress is not None:
            progress(meta, len(rows))
    return rows, metas


def _ran(row: dict) -> bool:
    """A row that produced metrics (rows without a status predate it)."""
    return (str(row.get("status", "ok")).startswith("ok")
            and row.get("read_blocks") is not None)


def aggregate_by_mode(rows: list[dict]) -> list[dict]:
    """One summary row per mode across every trace, under two weightings.

    ``mean_*`` is the unweighted mean over traces: every trace counts once,
    whatever its size. ``weighted_*`` pools the counters, so a trace counts
    in proportion to its read blocks -- the weighting
    ``benchmark.aggregate_results`` uses. The two can disagree when one large
    trace dominates, which is why both are reported.

    Traces with no read blocks are counted but excluded from the averages:
    their metrics are 0/0 and would drag every mean toward zero. Skipped rows
    are counted in ``skipped`` and contribute nothing else.
    """
    summary = []
    for mode in dict.fromkeys(row["mode"] for row in rows):
        group = [row for row in rows if row["mode"] == mode]
        ran = [row for row in group if _ran(row)]
        scored = [row for row in ran if row["read_blocks"] > 0]
        blocks = sum(r["read_blocks"] for r in scored)
        prefetches = sum(r["total_prefetches"] for r in ran)
        unused = sum(r["unused"] for r in ran)
        summary.append({
            "mode": mode,
            "traces": len(group),
            "skipped": len(group) - len(ran),
            "scored": len(scored),
            "mean_hit": mean(r["hit_ratio"] for r in scored) if scored else 0.0,
            "mean_prec": mean(r["precision"] for r in scored) if scored else 0.0,
            "mean_us": mean(r["mean_us"] for r in scored) if scored else 0.0,
            "weighted_hit": (sum(r["hit_ratio"] * r["read_blocks"]
                                 for r in scored) / blocks if blocks else 0.0),
            "weighted_prec": ((prefetches - unused) / prefetches
                              if prefetches else 0.0),
            "weighted_us": (sum(r["mean_us"] * r["read_blocks"]
                                for r in scored) / blocks if blocks else 0.0),
            "total_prefetches": prefetches,
            "total_unused": unused,
            "total_device_ios": sum(r.get("device_ios") or 0 for r in ran),
            "total_blocks_moved": sum(r.get("blocks_moved") or 0 for r in ran),
        })
    return summary


def per_trace_winners(rows: list[dict]) -> dict[str, int]:
    """How many traces each mode wins on hit ratio."""
    counts: dict[str, int] = {}
    for trace in dict.fromkeys(row["trace"] for row in rows):
        group = [r for r in rows if r["trace"] == trace and _ran(r)
                 and r["read_blocks"] > 0]
        if not group:
            continue
        best = max(group, key=lambda r: r["hit_ratio"])
        counts[best["mode"]] = counts.get(best["mode"], 0) + 1
    return counts


def winners(rows: list[dict]) -> dict:
    """Overall winner per metric. Traces with no reads are excluded.

    The ``cost`` and ``waste`` winners are frequently degenerate; read them
    with :func:`winner_caveats`, never alone.
    """
    by_mode: dict[str, list[dict]] = {}
    for row in rows:
        if not _ran(row) or row["read_blocks"] <= 0:
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


def winner_caveats(rows: list[dict], champ: dict,
                   break_even: float | None) -> dict[str, str]:
    """Why a ``cost`` or ``waste`` winner may be true by construction (M14).

    * cost: with break-even precision above 1.0 no prefetcher can pay for
      itself, so the policy that prefetches least wins by arithmetic (D11).
    * waste: a mode that issues no prefetch wastes none, so it wins "fewest
      wasted prefetches" without having predicted anything.
    """
    notes: dict[str, str] = {}
    if champ.get("cost") and break_even is not None and break_even > 1.0:
        notes["cost"] = (f"degenerate: break-even precision {break_even:.4f} "
                         "> 1, so no prefetcher can pay for itself")
    waste_mode = champ.get("waste")
    if waste_mode:
        issued = sum(r["total_prefetches"] for r in rows
                     if r["mode"] == waste_mode and _ran(r))
        if issued == 0:
            notes["waste"] = ("degenerate: it issues no prefetches, so it "
                              "wastes none by construction")
    return notes
