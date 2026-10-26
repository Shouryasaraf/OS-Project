"""Multi-corpus sweep driver: configurations C1-C4 over the oracleGeneral traces.

    python -m adaptive_prefetch.multicorpus --config C1 [--config C2 ...]
        [--workers N] [--out results/disk/sweep.jsonl] [--traces STEM,...]
        [--trace-root data/block-traces] [--dry-run] [--max-jobs N]
    python -m adaptive_prefetch.multicorpus --profile [--out-profile PATH]

One **job** is one (trace, slice, merge rule, capacity, policy, latency)
replay; its identity is ``prefetch_common.schemas.SWEEP_KEY`` and its row is
the ``disk/sweep.jsonl`` schema. The driver calls ``simulator.replay_stream``
as a library and does not change ``sweep.py``: it needs loader options
(``merge``, ``skip_requests``, ``max_merge_sectors``) that ``sweep_trace``
does not take.

What every row states, because the paper must be able to say it:

* **The cost model is explicit.** Every job runs under ``LatencyModel(5, 100,
  50)``, labelled ``assumed`` (break-even 0.526). The feedback-gated modes also
  run under ``LatencyModel(1, 100, 100)``, labelled ``break_even``: the
  measured shape, break-even 1.0101, at which ``guard`` and
  ``depth_adaptive`` must be identical (D14.1). Oracle traces carry no
  service times, so nothing is calibrated from them.
* **The slice is explicit.** ``skip_requests`` and ``max_requests`` count
  canonical requests (FIU pages merged with a 1024-sector cap, every other
  corpus as recorded; ``trace.iter_oracle_chunks``). A slice that cannot be
  filled (``skip_requests`` past the end, or a second slice shorter than its
  cap) is a ``skipped: n/a`` row with ``None`` metrics, never a short slice.
* **Trained modes are not scored here.** ``lstm`` and ``learned_depth`` are
  ``skipped: evaluated in E2 only``: E2 trains them on a prefix that overlaps
  C1's first million requests, so a C1 score would be in-sample.
* **Writes replay as reads.** oracleGeneral has no read/write flag
  (``ops_known=False``) and whole-second timestamps.

The run is resumable. Rows are appended to the jsonl as jobs finish, one
validated line at a time, by the parent process only; a restarted run skips
every key already present and drops a torn trailing line first. Jobs are
scheduled longest first by an estimated block budget, measured by a cheap
survey pass that is cached under ``--cache-dir``.

``belady_hit_ratio`` (C1, on ``none`` rows) is the demand-only Belady/MIN hit
ratio of the same block stream at the same capacity: a reference that no
demand cache can beat, **not** a bound on prefetching, which can exceed it.
It is computed from the reconstructed block stream (not from the trace's
``next_access_vtime``, which is keyed per request offset, not per block).
"""

from __future__ import annotations

import argparse
import copy
import heapq
import json
import os
import sys
import time
from array import array
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path

from prefetch_common import schemas
from prefetch_common.schemas import SWEEP_KEY

from . import corpora
from .baselines import DEFAULT_LINUX_RA_MAX_BLOCKS
from .features import DEFAULT_AGE_TABLE_BOUND
from .simulator import GATED_MODES, LatencyModel, replay_stream
from .sweep import OPTIONAL_MODES, RANKING_MODES
from .trace import DEFAULT_MAX_MERGE_SECTORS, iter_oracle_chunks, oracle_profile

REPO = Path(__file__).resolve().parents[2]
SWEEP_NAME = "disk/sweep.jsonl"
PROFILE_NAME = "disk/corpus_profile.csv"
DEFAULT_OUT = REPO / "results" / SWEEP_NAME
DEFAULT_CACHE_DIR = REPO / "outputs" / "multicorpus"

#: The two cost models a row can run under, by their ``latency`` label.
LATENCIES = {
    "assumed": LatencyModel(5.0, 100.0, 50.0),
    "break_even": LatencyModel(1.0, 100.0, 100.0),
}
#: Modes trained and scored in E2 only (plan C2: scoring them here would be
#: in-sample). Named rather than read from ``OPTIONAL_MODES`` so a mode that
#: is merely optional is not mislabelled.
E2_ONLY = ("lstm", "learned_depth")
E2_ONLY_REASON = "evaluated in E2 only"

WINDOW_SIZE = 32
MODEL_SEED = 42
CHUNK_REQUESTS = 250_000
CAP_1MIB = 2048
CAP_64MIB = 131072

#: Configurations, selected by ``--config``. Each is plain data:
#:
#: ``traces``      ``"all"``, ``"paper_subset"``, a corpus name, or a tuple of stems
#: ``slices``      ``(skip_requests, max_requests, rule)``; rule ``"full"`` makes a
#:                 short slice ``n/a``, ``("whole_if_at_most", N)`` replays the
#:                 whole file when it has at most N canonical requests
#: ``merges``      ``(merge, max_merge_sectors, corpus or None)``; the corpus limits
#:                 a merge rule to one corpus (``none`` is run only for FIU in C2)
#: ``capacities``  cache sizes in 512-B blocks
#: ``modes``       ``"all"`` (every ``sweep.RANKING_MODES`` entry) or a tuple
#: ``deep_depths`` / ``ra_max``  the ``deep`` depths and ``linux_ra`` ceilings
#: ``belady``      compute the Belady/MIN reference on the ``none`` rows
HEADLINE_MODES = ("none", "sequential", "adaptive", "guard", "deep", "linux_ra")
CONFIGS: dict[str, dict] = {
    "C1": dict(
        description="collection table: every trace, first 1M canonical requests",
        traces="all", slices=((0, 1_000_000, None),),
        merges=(("auto", DEFAULT_MAX_MERGE_SECTORS, None),),
        capacities=(CAP_1MIB, CAP_64MIB), modes="all",
        deep_depths=(8, 32, 256), ra_max=(64, 256, 1024), belady=True),
    "C2": dict(
        description="sensitivity grid: depth x capacity x merge rule at 250k",
        traces="all", slices=((0, 250_000, None),),
        merges=(("auto", DEFAULT_MAX_MERGE_SECTORS, None),
                ("same_second", None, None), ("none", None, "fiu")),
        capacities=(2048, 16384, 131072, 1048576), modes=("none", "deep", "linux_ra"),
        deep_depths=(0, 2, 8, 32, 64, 128, 256, 512), ra_max=(64, 256, 1024),
        belady=False),
    "C3": dict(
        description="robustness: the paper subset on slice 2 and on whole files",
        traces="paper_subset",
        slices=((1_000_000, 1_000_000, "full"),
                (0, None, ("whole_if_at_most", 4_000_000))),
        merges=(("auto", DEFAULT_MAX_MERGE_SECTORS, None),),
        capacities=(CAP_1MIB, CAP_64MIB), modes=HEADLINE_MODES,
        deep_depths=(8, 32, 256), ra_max=(256,), belady=False),
    "C4": dict(
        description="FIU page-merge cap sensitivity: 256 vs 1024 sectors",
        traces="fiu", slices=((0, 1_000_000, None),),
        merges=(("auto", 256, None), ("auto", DEFAULT_MAX_MERGE_SECTORS, None)),
        capacities=(CAP_1MIB, CAP_64MIB), modes=HEADLINE_MODES,
        deep_depths=(8, 32, 256), ra_max=(256,), belady=False),
    # Small configurations for the smoke path and the reproduce.py quick run.
    "smoke": dict(
        description="3 traces x 3 modes x 20k requests",
        traces=("msr_web_2", "fiu_madmax-110108-112108", "tencentBlock_1162"),
        slices=((0, 20_000, None),),
        merges=(("auto", DEFAULT_MAX_MERGE_SECTORS, None),),
        capacities=(CAP_1MIB,), modes=("none", "deep", "linux_ra"),
        deep_depths=(8,), ra_max=(256,), belady=False),
    "quick": dict(
        description="3 traces x every mode x 50k requests",
        traces=("msr_web_2", "fiu_madmax-110108-112108", "tencentBlock_1162"),
        slices=((0, 50_000, None),),
        merges=(("auto", DEFAULT_MAX_MERGE_SECTORS, None),),
        capacities=(CAP_1MIB,), modes="all",
        deep_depths=(8, 32, 256), ra_max=(64, 256, 1024), belady=True),
    # The earlier prototype's cells (scratch main-science sweep2.jsonl "raw"
    # and gated_be.jsonl): raw records, first 250k, default 5/100/50 model.
    # Two prototype conventions are reproduced here and nowhere else: block
    # numbers rebased to the slice's lowest LBA (``linux_ra`` works in 4 KiB
    # pages, so a shift that is not a multiple of 8 changes its windows), and
    # an unbounded age table for ``adaptive`` (the bound changes its features
    # once a slice touches more than 4M blocks, M9).
    "XCHECK": dict(
        description="prototype cross-check: raw records, 250k, two capacities",
        traces="all", slices=((0, 250_000, None),),
        merges=(("none", None, None),), capacities=(CAP_1MIB, CAP_64MIB),
        modes="xcheck", deep_depths=(8, 32, 256), ra_max=(256,), belady=False,
        rebase_lba=True, age_table_bound=None),
}
#: Modes the prototype ran at each capacity, for the ``XCHECK`` config.
_XCHECK_MODES = {
    CAP_1MIB: ("none", "sequential", "strided", "stride", "markov", "adaptive",
               "adaptive_evidence", "guard", "depth_adaptive", "correlate",
               "deep", "linux_ra"),
    CAP_64MIB: ("none", "sequential", "adaptive", "guard", "deep", "linux_ra"),
}
#: The standard configurations of a full E1 run, in order.
E1_CONFIGS = ("C1", "C2", "C3", "C4")

#: Rough relative replay cost per block, for longest-first scheduling only.
_COST_WEIGHT = {"adaptive": 2.5, "markov": 1.5, "linux_ra": 1.5,
                "adaptive_evidence": 1.3, "sequential": 1.2}


@dataclass(frozen=True)
class SliceSpec:
    """One stretch of one trace under one reconstruction rule."""
    config: str
    trace: str
    path: str
    corpus: str
    skip_requests: int
    max_requests: int | None
    merge: str
    max_merge_sectors: int | None
    rule: object = None


@dataclass(frozen=True)
class Job:
    """One sweep job. :meth:`key` is its ``SWEEP_KEY`` identity."""
    config: str
    trace: str
    path: str
    corpus: str
    skip_requests: int
    max_requests: int | None
    merge: str
    max_merge_sectors: int | None
    capacity: int
    mode: str
    depth: int
    max_blocks: int | None
    latency: str
    belady: bool = False
    full_slice: bool = False
    skip_reason: str | None = None
    budget: float = 0.0
    #: Prototype conventions, used only by ``XCHECK``: subtract this from
    #: every LBA, and the ``adaptive`` age-table bound (``None`` = unbounded).
    lba_base: int | None = None
    age_table_bound: int | None = DEFAULT_AGE_TABLE_BOUND
    git_sha: str | None = field(default=None, compare=False)

    def key(self) -> tuple:
        return tuple(getattr(self, name) for name in SWEEP_KEY)


def row_key(row) -> tuple:
    """``SWEEP_KEY`` of a sweep row (a mapping)."""
    return tuple(row[name] for name in SWEEP_KEY)


def policy_label(mode: str, depth: int = 0, max_blocks: int | None = None) -> str:
    """``deep@8``, ``linux_ra@256``, or the bare mode name."""
    if mode == "deep":
        return f"deep@{depth}"
    if mode == "linux_ra":
        return f"linux_ra@{max_blocks}"
    return mode


def config_policies(spec: dict, capacity: int) -> list[tuple[str, int, int | None]]:
    """``(mode, depth, max_blocks)`` for every policy of a configuration.

    ``depth`` is the ``read_ahead_depth`` passed to the replay: the depth for
    ``deep`` and 0 for every other mode (only ``deep`` reads it).
    ``max_blocks`` is ``None`` except for ``linux_ra``.
    """
    modes = spec["modes"]
    if modes == "all":
        modes = RANKING_MODES
    elif modes == "xcheck":
        modes = _XCHECK_MODES[capacity]
    out: list[tuple[str, int, int | None]] = []
    for mode in modes:
        if mode == "deep":
            out.extend(("deep", depth, None) for depth in spec["deep_depths"])
        elif mode == "linux_ra":
            out.extend(("linux_ra", 0, ceiling) for ceiling in spec["ra_max"])
        else:
            out.append((mode, 0, None))
    return out


def latencies_for(mode: str) -> tuple[str, ...]:
    """``assumed`` always; ``break_even`` too for the feedback-gated modes."""
    return ("assumed", "break_even") if mode in GATED_MODES else ("assumed",)


# ------------------------------------------------------------------ planning

def select_traces(spec_traces, root: str | Path,
                  only: tuple[str, ...] | None = None) -> dict[str, Path]:
    """stem -> path for a configuration's trace set, present files only."""
    found = corpora.discover_block_traces(root)
    if spec_traces == "all":
        stems = list(found)
    elif spec_traces == "paper_subset":
        stems = list(corpora.PAPER_SUBSET)
    elif spec_traces in corpora.CORPORA:
        stems = [s for s in found if corpora.CORPUS_OF(s) == spec_traces]
    else:
        stems = list(spec_traces)
    if only is not None:
        stems = [s for s in stems if s in only]
    return {stem: found[stem] for stem in stems if stem in found}


def plan_slices(config: str, root: str | Path = corpora.BLOCK_TRACE_ROOT,
                only: tuple[str, ...] | None = None) -> list[SliceSpec]:
    spec = CONFIGS[config]
    out = []
    for stem, path in select_traces(spec["traces"], root, only).items():
        corpus = corpora.CORPUS_OF(stem)
        for merge, mms, limit_corpus in spec["merges"]:
            if limit_corpus is not None and corpus != limit_corpus:
                continue
            for skip, cap, rule in spec["slices"]:
                out.append(SliceSpec(config, stem, str(path), corpus, skip, cap,
                                     merge, mms, rule))
    return out


def survey_target(spec: SliceSpec) -> tuple[int, int | None]:
    """``(skip, cap)`` the survey must read to schedule and qualify a slice."""
    rule = spec.rule
    if isinstance(rule, tuple) and rule[0] == "whole_if_at_most":
        return spec.skip_requests, rule[1] + 1
    return spec.skip_requests, spec.max_requests


def _file_identity(path: str | Path) -> dict:
    stat = Path(path).stat()
    return {"file_size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def survey_key(spec: SliceSpec) -> tuple:
    skip, cap = survey_target(spec)
    return (spec.trace, spec.merge, spec.max_merge_sectors, skip, cap)


def survey_slice(path: str | Path, merge: str, max_merge_sectors: int | None,
                 skip: int, cap: int | None) -> dict:
    """Canonical requests and blocks in a slice, by streaming it once."""
    requests = blocks = 0
    lowest = None
    for chunk in iter_oracle_chunks(path, CHUNK_REQUESTS, merge=merge,
                                    max_merge_sectors=max_merge_sectors,
                                    skip_requests=skip, max_requests=cap):
        requests += len(chunk)
        blocks += sum(r.size_blocks for r in chunk)
        least = min(r.lba for r in chunk)
        lowest = least if lowest is None else min(lowest, least)
    return {"requests": requests, "blocks": blocks, "min_lba": lowest}


def _survey_job(spec: SliceSpec) -> tuple[tuple, dict]:
    skip, cap = survey_target(spec)
    result = survey_slice(spec.path, spec.merge, spec.max_merge_sectors, skip, cap)
    result.update(_file_identity(spec.path))
    return survey_key(spec), result


def load_survey(cache_dir: str | Path | None) -> dict[tuple, dict]:
    if cache_dir is None:
        return {}
    path = Path(cache_dir) / "survey.jsonl"
    if not path.is_file():
        return {}
    cached = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(line)
        except ValueError:
            continue  # a torn line from an interrupted survey
        cached[tuple(item["key"])] = item["result"]
    return cached


def run_survey(specs: list[SliceSpec], *, workers: int = 1,
               cache_dir: str | Path | None = DEFAULT_CACHE_DIR,
               progress=None) -> dict[tuple, dict]:
    """Survey every distinct slice, reusing cached results for unchanged files."""
    cached = load_survey(cache_dir)
    wanted: dict[tuple, SliceSpec] = {}
    for spec in specs:
        key = survey_key(spec)
        hit = cached.get(key)
        if hit is not None and "min_lba" in hit and all(
                hit.get(k) == v for k, v in _file_identity(spec.path).items()):
            continue
        wanted.setdefault(key, spec)
    results = {k: v for k, v in cached.items()}
    if not wanted:
        return results
    handle = None
    if cache_dir is not None:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        handle = open(Path(cache_dir) / "survey.jsonl", "a", encoding="utf-8")
    try:
        for key, result in _map(_survey_job, list(wanted.values()), workers):
            results[key] = result
            if handle is not None:
                handle.write(json.dumps({"key": list(key), "result": result}) + "\n")
                handle.flush()
            if progress is not None:
                progress(f"survey {key[0]} {key[1]} skip={key[3]}: "
                         f"{result['requests']:,} requests")
    finally:
        if handle is not None:
            handle.close()
    return results


def _budget(blocks: float, mode: str, depth: int, belady: bool) -> float:
    weight = _COST_WEIGHT.get(mode, 1.0)
    if mode == "deep":
        weight = 1.0 + depth / 32.0
    if mode in GATED_MODES:
        weight = 1.4
    return blocks * (weight + (3.0 if belady else 0.0))


def build_jobs(configs, *, root: str | Path = corpora.BLOCK_TRACE_ROOT,
               only: tuple[str, ...] | None = None,
               survey: dict[tuple, dict] | None = None,
               git_sha: str | None = None) -> list[Job]:
    """Every job of ``configs``, in plan order (scheduling sorts later).

    Without a ``survey`` the budget is the request cap, and the slice rules
    that need a length (``full``, ``whole_if_at_most``) cannot be decided, so
    they raise: call :func:`run_survey` first.
    """
    jobs: list[Job] = []
    for config in configs:
        if config not in CONFIGS:
            raise ValueError(f"unknown config {config!r}; known: {sorted(CONFIGS)}")
        spec = CONFIGS[config]
        for sl in plan_slices(config, root, only):
            measured = survey.get(survey_key(sl)) if survey is not None else None
            if sl.rule is not None and measured is None:
                raise ValueError(f"{config}: slice rule {sl.rule!r} needs a survey")
            max_requests, reason, full = sl.max_requests, None, False
            if sl.rule == "full":
                full = True
                if measured["requests"] < sl.max_requests:
                    reason = (f"n/a: no full slice at skip {sl.skip_requests:,} "
                              f"({measured['requests']:,} of {sl.max_requests:,} "
                              "canonical requests)")
            elif isinstance(sl.rule, tuple) and sl.rule[0] == "whole_if_at_most":
                if measured["requests"] > sl.rule[1]:
                    reason = (f"n/a: whole file has more than {sl.rule[1]:,} "
                              "canonical requests")
            if reason is None and measured is not None and measured["requests"] == 0:
                reason = "n/a: slice is empty (skip_requests past the end)"
            blocks = (measured["blocks"] if measured is not None
                      else 8.0 * (sl.max_requests or 1_000_000))
            for capacity in spec["capacities"]:
                for mode, depth, max_blocks in config_policies(spec, capacity):
                    skip = reason
                    if skip is None and mode in OPTIONAL_MODES + E2_ONLY:
                        skip = (E2_ONLY_REASON if mode in E2_ONLY
                                else "no trained artifact")
                    belady = bool(spec.get("belady")) and mode == "none"
                    base = (measured or {}).get("min_lba") if spec.get("rebase_lba") else None
                    for latency in latencies_for(mode):
                        jobs.append(Job(
                            config, sl.trace, sl.path, sl.corpus,
                            sl.skip_requests, max_requests, sl.merge,
                            sl.max_merge_sectors, capacity, mode, depth,
                            max_blocks, latency, belady=belady,
                            full_slice=full, skip_reason=skip,
                            budget=0.0 if skip else _budget(blocks, mode, depth, belady),
                            lba_base=base,
                            age_table_bound=spec.get("age_table_bound",
                                                     DEFAULT_AGE_TABLE_BOUND),
                            git_sha=git_sha))
    return jobs


# ------------------------------------------------------------------- running

def _load1() -> float | None:
    try:
        return round(os.getloadavg()[0], 2)
    except (AttributeError, OSError):
        return None


@lru_cache(maxsize=1)
def _adaptive_model():
    from .benchmark import make_model
    return make_model(MODEL_SEED)


class _Slice:
    """Counts what a replay consumed; optionally keeps (lba, size) for Belady."""

    def __init__(self, keep: bool):
        self.requests = 0
        self.blocks = 0
        self.first_ms: float | None = None
        self.last_ms: float | None = None
        self.lbas = array("q") if keep else None
        self.sizes = array("q") if keep else None

    def wrap(self, chunks):
        for chunk in chunks:
            if chunk:
                if self.first_ms is None:
                    self.first_ms = chunk[0].timestamp_ms
                self.last_ms = chunk[-1].timestamp_ms
                self.requests += len(chunk)
                self.blocks += sum(r.size_blocks for r in chunk)
                if self.lbas is not None:
                    self.lbas.extend(r.lba for r in chunk)
                    self.sizes.extend(r.size_blocks for r in chunk)
            yield chunk

    @property
    def hours(self) -> float:
        if self.first_ms is None:
            return 0.0
        return (self.last_ms - self.first_ms) / 3_600_000.0


#: Sweep-row columns that are ``None`` on a skipped row.
_METRIC_COLUMNS = (
    "requests", "blocks", "hours_covered", "read_blocks", "hit_ratio",
    "precision", "total_prefetches", "useful_prefetches", "unused", "mean_us",
    "policy_switches", "observed_precision", "precision_margin",
    "observed_stride", "demand_ios", "prefetch_ios", "prefetch_ios_mergeable",
    "device_ios", "blocks_moved", "belady_hit_ratio", "age_table_evictions",
)


def _base_row(job: Job) -> dict:
    latency = LATENCIES[job.latency]
    row = {name: getattr(job, name) for name in SWEEP_KEY}
    row.update({
        "corpus": job.corpus,
        "size_unit": oracle_profile(job.path)["size_unit"],
        "hit_us": latency.hit_us, "miss_us": latency.demand_miss_us,
        "prefetch_us": latency.prefetch_us,
        "break_even": latency.break_even_precision(),
        "ops_known": False, "ts_resolution_s": 1.0,
        "age_table_bound": job.age_table_bound if job.mode == "adaptive" else None,
        "model_seed": MODEL_SEED if job.mode == "adaptive" else None,
        "git_sha": job.git_sha,
    })
    return row


def _skipped(row: dict, reason: str) -> dict:
    row["status"] = f"skipped: {reason}"
    for name in _METRIC_COLUMNS:
        row[name] = None
    return row


def _rebased(chunks, base: int):
    for chunk in chunks:
        yield [replace(r, lba=r.lba - base) for r in chunk]


def run_job(job: Job) -> dict:
    """Replay one job and return its validated ``disk/sweep.jsonl`` row."""
    started = time.monotonic()
    load_start = _load1()
    row = _base_row(job)
    if job.skip_reason is not None:
        _skipped(row, job.skip_reason)
    else:
        stats = _Slice(keep=job.belady)
        chunks = iter_oracle_chunks(
            job.path, CHUNK_REQUESTS, merge=job.merge,
            max_merge_sectors=job.max_merge_sectors,
            skip_requests=job.skip_requests, max_requests=job.max_requests)
        if job.lba_base:
            chunks = _rebased(chunks, job.lba_base)
        model = copy.deepcopy(_adaptive_model()) if job.mode == "adaptive" else None
        metrics = replay_stream(
            stats.wrap(chunks), capacity=job.capacity, window_size=WINDOW_SIZE,
            mode=job.mode, model=model, latency_model=LATENCIES[job.latency],
            read_ahead_depth=job.depth,
            linux_ra_max_blocks=job.max_blocks or DEFAULT_LINUX_RA_MAX_BLOCKS,
            age_table_bound=job.age_table_bound)
        if stats.requests == 0:
            _skipped(row, "n/a: slice is empty (skip_requests past the end)")
        elif job.full_slice and stats.requests < (job.max_requests or 0):
            _skipped(row, f"n/a: no full slice at skip {job.skip_requests:,} "
                          f"({stats.requests:,} of {job.max_requests:,} canonical requests)")
        else:
            row.update({
                "status": "ok", "requests": stats.requests,
                "blocks": stats.blocks, "hours_covered": stats.hours,
                "read_blocks": metrics.read_blocks,
                "hit_ratio": metrics.hit_ratio,
                "precision": metrics.prefetch_precision,
                "total_prefetches": metrics.prefetches,
                "useful_prefetches": metrics.useful_prefetches,
                "unused": metrics.unused_prefetches,
                "mean_us": metrics.mean_access_latency_us,
                "policy_switches": metrics.policy_switches,
                "observed_precision": metrics.observed_precision,
                "precision_margin": metrics.precision_margin,
                "observed_stride": metrics.observed_stride,
                "demand_ios": metrics.demand_ios,
                "prefetch_ios": metrics.prefetch_ios,
                "prefetch_ios_mergeable": metrics.prefetch_ios_mergeable,
                "device_ios": metrics.device_ios,
                "blocks_moved": metrics.blocks_moved,
                "belady_hit_ratio": (belady_hit_ratio(stats.lbas, stats.sizes,
                                                      job.capacity)
                                     if job.belady else None),
                "age_table_evictions": (metrics.age_table_evictions
                                        if job.mode == "adaptive" else None),
            })
    row["load_start"] = load_start
    row["load_end"] = _load1()
    row["wall_s"] = round(time.monotonic() - started, 3)
    schemas.validate_row(SWEEP_NAME, row)
    return row


def _map(function, items, workers: int):
    """``imap_unordered`` over a pool, or a plain loop for one worker."""
    if workers <= 1 or len(items) <= 1:
        for item in items:
            yield function(item)
        return
    import multiprocessing
    with multiprocessing.Pool(min(workers, len(items)), maxtasksperchild=50) as pool:
        yield from pool.imap_unordered(function, items, chunksize=1)


def read_sweep(path: str | Path, *, repair: bool = False) -> list[dict]:
    """Rows of a sweep jsonl, each validated against the schema.

    A run killed mid-write can leave a torn last line (no newline, invalid
    JSON). With ``repair=True`` it is cut off the file; otherwise it is
    ignored. A torn line anywhere else is corruption and raises.
    """
    path = Path(path)
    if not path.is_file():
        return []
    data = path.read_bytes()
    lines = data.split(b"\n")
    tail = lines.pop()  # b"" when the file ends with a newline
    rows = [json.loads(line) for line in lines if line.strip()]
    if tail.strip():
        try:
            rows.append(json.loads(tail))
            torn = False  # complete but unterminated: keep it, add the newline
        except ValueError:
            torn = True
        if repair:
            with open(path, "r+b") as handle:
                if torn:
                    handle.truncate(len(data) - len(tail))
                else:
                    handle.seek(0, os.SEEK_END)
                    handle.write(b"\n")
    for row in rows:
        schemas.validate_row(SWEEP_NAME, row)
    return rows


def run_jobs(jobs: list[Job], out: str | Path, *, workers: int = 1,
             max_jobs: int | None = None, progress=None) -> dict:
    """Run every job whose key is not yet in ``out``; append rows as they finish.

    Returns ``{"planned", "already_done", "ran", "duplicates_in_file"}``.
    ``max_jobs`` stops after that many new rows (used to test resumption).
    """
    out = Path(out)
    existing = read_sweep(out, repair=True)
    done = [row_key(row) for row in existing]
    done_set = set(done)
    todo, seen = [], set()
    for job in jobs:
        key = job.key()
        if key in done_set or key in seen:
            continue
        seen.add(key)
        todo.append(job)
    todo.sort(key=lambda j: -j.budget)
    if max_jobs is not None:
        todo = todo[:max_jobs]
    already = sum(1 for job in jobs if job.key() in done_set)
    ran = 0
    for row in _map(run_job, todo, workers):
        schemas.write_rows(SWEEP_NAME, [row], out, append=True)
        ran += 1
        if progress is not None:
            progress(f"[{ran}/{len(todo)}] {row['config']} {row['trace']} "
                     f"cap={row['capacity']} {policy_label(row['mode'], row['depth'], row['max_blocks'])} "
                     f"{row['latency']}: {row['status']} ({row['wall_s']:.1f}s)")
    return {"planned": len(jobs), "already_done": already, "ran": ran,
            "duplicates_in_file": len(done) - len(done_set)}


# -------------------------------------------------------------------- Belady

def _next_use_python(lbas, sizes) -> tuple[list[int], list[int], int]:
    blocks = [lba + i for lba, size in zip(lbas, sizes) for i in range(size)]
    dead = len(blocks) + 1
    next_use = [dead] * len(blocks)
    upcoming: dict[int, int] = {}
    for now in range(len(blocks) - 1, -1, -1):
        block = blocks[now]
        following = upcoming.get(block)
        if following is not None:
            next_use[now] = following
        upcoming[block] = now
    return blocks, next_use, dead


def _next_use_numpy(np, lbas, sizes):
    lba = np.frombuffer(lbas, dtype=np.int64) if isinstance(lbas, array) else np.asarray(lbas, dtype=np.int64)
    size = np.frombuffer(sizes, dtype=np.int64) if isinstance(sizes, array) else np.asarray(sizes, dtype=np.int64)
    total = int(size.sum())
    # Block at global position p of request i is lba_i + (p - offset_i).
    blocks = np.repeat(lba - (np.cumsum(size) - size), size)
    blocks += np.arange(total, dtype=np.int64)
    order = np.argsort(blocks, kind="stable")
    ordered = blocks[order]
    same = ordered[1:] == ordered[:-1]
    del ordered
    dead = total + 1
    next_use = np.full(total, dead, dtype=np.int64)
    next_use[order[:-1][same]] = order[1:][same]
    return blocks, next_use, dead


def belady_hit_ratio(lbas, sizes, capacity: int, *, use_numpy: bool | None = None,
                     chunk: int = 1 << 20) -> float:
    """Belady/MIN demand hit ratio of the block stream of ``(lba, size)`` reads.

    Offline-optimal replacement with bypass, exactly as
    ``eda.belady_demand_hit_ratio`` for an all-read stream (oracle traces have
    no writes): on overflow, evict the resident block, the incoming one
    included, whose next read is farthest away. Two passes: a next-use array
    (numpy argsort when available, a dict otherwise), then a lazy max-heap.

    Two shortcuts keep it fast on large slices without changing the count. A
    block never read again is never inserted (it would be the first victim
    and can never hit), and the heap is rebuilt from the resident set when
    stale entries outnumber live ones. Memory with numpy is about 40 bytes
    per block of the slice: the largest C1 slice (alibabaBlock_6, 143M
    blocks) peaked at 5.7 GB RSS and took 155 s together with its ``none``
    replay. A reference, **not** a bound on prefetching.
    """
    if capacity < 1:
        raise ValueError("cache capacity must be positive")
    np = None
    if use_numpy is not False:
        try:
            import numpy as np  # noqa: F811
        except ImportError:
            if use_numpy:
                raise
            np = None
    if np is not None:
        blocks, next_use, dead = _next_use_numpy(np, lbas, sizes)
    else:
        blocks, next_use, dead = _next_use_python(lbas, sizes)
    total = len(blocks)
    if not total:
        return 0.0
    resident: dict[int, int] = {}
    heap: list[tuple[int, int]] = []
    limit = 4 * capacity + 1024
    hits = 0
    push, pop = heapq.heappush, heapq.heappop
    for start in range(0, total, chunk):
        part_b = blocks[start:start + chunk]
        part_n = next_use[start:start + chunk]
        if np is not None:
            part_b, part_n = part_b.tolist(), part_n.tolist()
        for block, upcoming in zip(part_b, part_n):
            if block in resident:
                hits += 1
                if upcoming == dead:
                    del resident[block]
                else:
                    resident[block] = upcoming
                    push(heap, (-upcoming, block))
                continue
            if upcoming == dead:
                continue
            resident[block] = upcoming
            push(heap, (-upcoming, block))
            if len(resident) > capacity:
                while True:
                    negative, victim = pop(heap)
                    if resident.get(victim) == -negative:
                        del resident[victim]
                        break
            if len(heap) > limit:
                heap = [(-u, b) for b, u in resident.items()]
                heapq.heapify(heap)
    return hits / total


# ------------------------------------------------------------------- profile

def _natural(stem: str):
    head, _, tail = stem.rpartition("_")
    return (head, int(tail)) if tail.isdigit() else (stem, 0)


def twin_label(stem: str) -> str | None:
    group = corpora.twin_group(stem)
    return None if group is None else "+".join(sorted(group, key=_natural))


def count_records(path: str | Path) -> int:
    """Raw oracleGeneral records in a file (decompressed bytes / 24)."""
    from .trace import ORACLE_RECORD, _open_oracle
    total = 0
    with _open_oracle(Path(path)) as stream:
        while True:
            block = stream.read(1 << 24)
            if not block:
                break
            total += len(block)
    if total % ORACLE_RECORD.size:
        raise ValueError(f"{path}: truncated trailing record")
    return total // ORACLE_RECORD.size


def _shown_path(path: Path) -> str:
    """Repo-relative POSIX path when inside the repo (symlinks not resolved)."""
    absolute = Path(os.path.abspath(path))
    try:
        return absolute.relative_to(REPO).as_posix()
    except ValueError:
        return absolute.as_posix()


def profile_trace_file(path: str | Path, *, merge: str = "auto",
                       max_merge_sectors: int | None = DEFAULT_MAX_MERGE_SECTORS) -> dict:
    """One ``disk/corpus_profile.csv`` row: whole-file canonical statistics.

    ``footprint_blocks`` is ``None``: distinct blocks over a whole file (up
    to 4.79 G blocks, msr_prxy_1) do not fit a bounded-memory pass.
    """
    path = Path(path)
    stem = corpora.trace_stem(path)
    profile = oracle_profile(path)
    sizes: dict[int, int] = {}
    requests = blocks = contiguous = 0
    first = last = None
    previous_end = None
    for chunk in iter_oracle_chunks(path, CHUNK_REQUESTS, merge=merge,
                                    max_merge_sectors=max_merge_sectors):
        if first is None:
            first = chunk[0].timestamp_ms
        last = chunk[-1].timestamp_ms
        for r in chunk:
            sizes[r.size_blocks] = sizes.get(r.size_blocks, 0) + 1
            blocks += r.size_blocks
            if r.lba == previous_end:
                contiguous += 1
            previous_end = r.lba + r.size_blocks
        requests += len(chunk)
    return {
        "trace": stem, "corpus": corpora.CORPUS_OF(stem), "file": _shown_path(path),
        "size_unit": profile["size_unit"], "merge": merge,
        "max_merge_sectors": max_merge_sectors, "records": count_records(path),
        "requests": requests, "blocks": blocks, "footprint_blocks": None,
        "mean_request_blocks": blocks / requests if requests else 0.0,
        "median_request_blocks": _median_from_counts(sizes, requests),
        "contiguous_fraction": contiguous / (requests - 1) if requests > 1 else 0.0,
        "hours_covered": ((last - first) / 3_600_000.0) if first is not None else 0.0,
        "tagged": bool(profile["tagged"]), "twin_group": twin_label(stem),
        "paper_subset": stem in corpora.PAPER_SUBSET,
    }


def _median_from_counts(counts: dict[int, int], n: int) -> float:
    if not n:
        return 0.0
    ordered = sorted(counts.items())

    def at(rank: int) -> int:
        seen = 0
        for value, count in ordered:
            seen += count
            if seen > rank:
                return value
        return ordered[-1][0]
    if n % 2:
        return float(at(n // 2))
    return (at(n // 2 - 1) + at(n // 2)) / 2.0


def write_corpus_profile(root: str | Path, out: str | Path, *, workers: int = 1,
                         only: tuple[str, ...] | None = None, progress=None) -> Path:
    """Profile every trace under ``root`` (whole files) into ``corpus_profile.csv``."""
    found = corpora.discover_block_traces(root)
    # Largest file first, so the long whole-file passes start early.
    paths = sorted((str(found[s]) for s in found if only is None or s in only),
                   key=lambda p: -Path(p).stat().st_size)
    rows = []
    for row in _map(profile_trace_file, paths, workers):
        rows.append(row)
        if progress is not None:
            progress(f"profile {row['trace']}: {row['requests']:,} requests")
    rows.sort(key=lambda r: (corpora.CORPORA.index(r["corpus"]), _natural(r["trace"])))
    return schemas.write_rows(PROFILE_NAME, rows, out)


# ----------------------------------------------------------------------- CLI

def git_sha() -> str | None:
    from prefetch_common.provenance import git_state
    return git_state()[0]


def single_threaded_math() -> None:
    """One BLAS/OpenMP thread per process.

    Worker pools here run one job per process; numpy or scipy in each worker
    would otherwise start a thread per core on a shared 128-core host. Set
    before numpy is imported, and inherited by every worker.
    """
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(name, "1")


def default_workers() -> int:
    return max(1, min(100, (os.cpu_count() or 9) - 8))


def run(configs, *, out: str | Path = DEFAULT_OUT,
        root: str | Path = corpora.BLOCK_TRACE_ROOT,
        only: tuple[str, ...] | None = None, workers: int = 1,
        cache_dir: str | Path | None = DEFAULT_CACHE_DIR,
        max_jobs: int | None = None, dry_run: bool = False, progress=None) -> dict:
    """Survey, plan and run ``configs``; returns the job counts."""
    specs = [s for config in configs for s in plan_slices(config, root, only)]
    if not specs:
        raise FileNotFoundError(f"no oracleGeneral traces for {list(configs)} under {root}")
    survey = run_survey(specs, workers=workers, cache_dir=cache_dir, progress=progress)
    jobs = build_jobs(configs, root=root, only=only, survey=survey, git_sha=git_sha())
    if dry_run:
        skipped = sum(1 for j in jobs if j.skip_reason)
        return {"planned": len(jobs), "skipped_rows": skipped,
                "budget_blocks": sum(j.budget for j in jobs)}
    return run_jobs(jobs, out, workers=workers, max_jobs=max_jobs, progress=progress)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m adaptive_prefetch.multicorpus", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", action="append", choices=sorted(CONFIGS),
                        help="configuration to run (repeatable); default C1..C4")
    parser.add_argument("--workers", type=int, default=default_workers())
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="sweep jsonl (appended)")
    parser.add_argument("--trace-root", default=str(REPO / corpora.BLOCK_TRACE_ROOT))
    parser.add_argument("--traces", default=None,
                        help="comma-separated stems to restrict every config to")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR),
                        help="survey cache (slice lengths for scheduling)")
    parser.add_argument("--max-jobs", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="survey and plan, print the job count, run nothing")
    parser.add_argument("--profile", action="store_true",
                        help="write corpus_profile.csv (whole files) instead of sweeping")
    parser.add_argument("--out-profile", default=str(REPO / "results" / PROFILE_NAME))
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    single_threaded_math()
    only = tuple(s for s in args.traces.split(",") if s) if args.traces else None
    say = None if args.quiet else (lambda text: print(text, flush=True))
    if args.profile:
        path = write_corpus_profile(args.trace_root, args.out_profile,
                                    workers=args.workers, only=only, progress=say)
        print(f"wrote {path}")
        return 0
    configs = tuple(args.config or E1_CONFIGS)
    counts = run(configs, out=args.out, root=args.trace_root, only=only,
                 workers=args.workers, cache_dir=args.cache_dir,
                 max_jobs=args.max_jobs, dry_run=args.dry_run, progress=say)
    print(json.dumps(counts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
