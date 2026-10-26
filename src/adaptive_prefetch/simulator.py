"""Causal block-level cache replay with identical rules for all policies."""

from __future__ import annotations

import json
import warnings
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from .baselines import (DEFAULT_LINUX_RA_MAX_BLOCKS, LinuxReadahead,
                        MarkovPrefetcher, StridePrefetcher)
from .features import (DEFAULT_AGE_TABLE_BOUND, STREAM_FEATURES, WindowContext,
                       dominant_stride, extract)
from .guard import GateController, read_ahead
from .model import OnlineGaussianNB
from .pipeline import PolicySmoother
from .trace import Request


@dataclass(frozen=True)
class LatencyModel:
    hit_us: float = 5.0
    demand_miss_us: float = 100.0
    prefetch_us: float = 50.0

    def __post_init__(self) -> None:
        if min(self.hit_us, self.demand_miss_us, self.prefetch_us) < 0:
            raise ValueError("latency model values must be non-negative")

    @classmethod
    def measured(cls, path=None, trace: list | None = None,
                 prefetch_multiple: float = 1.0) -> "LatencyModel":
        """Cost model calibrated from a trace's own measured service times.

        The defaults above (5/100/50 us) are an assumption inherited from the
        original design. Measured ``t1`` values in the MSRC-trace-003
        collection are milliseconds, strongly right-skewed: 65-87% of requests
        complete in under 1 us while the mean is 260-2058 us.

        The **mean** is used, not the median. The median is ~0 for most of
        these traces (the sub-microsecond bulk dominates), which would produce
        a cost model of essentially zero and make every policy look free.

        ``prefetch_multiple`` is what a speculative read costs relative to a
        demand read.

        **1.0 is the default, and under it prefetching can never pay for
        itself.** A speculative read does one device read (cost ``m``); if it
        hits it saves one demand read (``m`` minus the small cache-lookup
        cost). Break-even is therefore ``1/(1 - hit_fraction)`` = 1.0101 at a
        1% hit cost, which no precision can exceed. This is arithmetic, not an
        empirical result, and it means the modelled-cost column of any
        benchmark is degenerate at this setting: "no prefetch" wins it by
        construction. A value **below 1.0** is what models the real benefit of
        overlapped or coalesced prefetch -- the device work is identical, but
        the requester is not blocked by it, so its wall-clock cost is lower
        than a demand read's. That is a modelling assumption, not something
        these traces measure; they record demand-read service time only.

        An earlier version used 1.1 with a comment implying it was neutral.
        It is not, and neither is 1.0: the break-even is above 1 either way.
        :meth:`break_even_precision` is the number to check before reading any
        cost ranking.

        Returns defaults unchanged when the trace carries no measurements.
        ``prefetch_multiple`` is validated before that fallback, so a bad
        value is never silently accepted just because the trace had no
        service times. A measured mean of zero leaves no saving for a hit to
        make, so the break-even is infinite; that is warned about rather than
        returned quietly, because it vetoes every gated prefetch.
        """
        if prefetch_multiple <= 0:
            raise ValueError("prefetch_multiple must be positive")
        values = _measured_service_us(trace) if trace is not None else None
        if not values:
            if path is not None:
                from .trace import load_csv
                try:
                    values = _measured_service_us(load_csv(path, "revised"))
                except (ValueError, OSError):
                    values = None
        if not values:
            return cls()
        mean_service = sum(values) / len(values)
        # A hit is served from cache and costs a small fraction of a device
        # service. 1% is an explicitly-documented approximation, not a
        # measurement: the traces record device time only, never cache time.
        model = cls(
            hit_us=max(0.001, mean_service * 0.01),
            demand_miss_us=mean_service,
            prefetch_us=mean_service * prefetch_multiple,
        )
        if model.break_even_precision() == float("inf"):
            warnings.warn(
                f"measured mean service time is {mean_service:g} us, so a hit "
                "saves nothing and the break-even precision is infinite: every "
                "feedback-gated mode will refuse to prefetch",
                RuntimeWarning, stacklevel=2)
        return model

    def break_even_precision(self) -> float:
        """Prefetch precision needed for a speculative read to pay for itself.

        A prefetch costs ``prefetch_us`` and, when it hits, saves
        ``demand_miss_us - hit_us``. It therefore needs at least
        ``prefetch_us / (demand_miss_us - hit_us)`` precision. Above 1.0 no
        prefetcher can ever be worth running, whatever the workload, so this
        is the number to check before interpreting any cost ranking.
        """
        saving = self.demand_miss_us - self.hit_us
        if saving <= 0:
            return float("inf")
        return self.prefetch_us / saving


def _measured_service_us(requests) -> list[float]:
    """Service times in microseconds for requests that recorded one."""
    return [r.service_ms * 1000.0 for r in requests
            if getattr(r, "service_ms", None) is not None]


@dataclass(frozen=True)
class IOCostModel:
    """Device time as positioning plus transfer: ``T = N_io*t_pos + N_blk*t_xfer``.

    :class:`LatencyModel` charges a whole mean service time per 512-B block,
    for demand and prefetch alike, so a 64-block sequential read costs 64
    seeks. That unit mismatch is what makes its break-even 1.0101 regardless
    of the device (D11, D18). This model charges one positioning cost per
    *device I/O* (``Metrics.device_ios``, a maximal contiguous run after
    block-layer merging) and a transfer cost per block moved
    (``Metrics.blocks_moved``). It sits beside ``LatencyModel`` and replaces
    none of its fields: the legacy per-block columns and the D11 break-even
    are unchanged.

    It is a device-work model, not a latency model: there is no queueing, no
    overlap of prefetch with demand, and no completion timing. The constants
    come from a device calibration (``from_device_json``), never from a trace.
    """

    t_pos_us: float
    t_xfer_us_per_block: float

    def __post_init__(self) -> None:
        if self.t_pos_us < 0 or self.t_xfer_us_per_block < 0:
            raise ValueError("IOCostModel times must be non-negative")

    @classmethod
    def from_device_json(cls, path) -> "IOCostModel":
        """Load ``t_pos_us`` and ``t_xfer_us_per_block`` from a calibration file.

        The file is ``results/realio/device_<dev>.json`` as written by the
        real-device calibration. Missing keys raise rather than defaulting,
        because a guessed device constant is exactly what this model exists
        to replace.
        """
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        try:
            return cls(t_pos_us=float(payload["t_pos_us"]),
                       t_xfer_us_per_block=float(payload["t_xfer_us_per_block"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"{path}: device calibration needs numeric t_pos_us and "
                f"t_xfer_us_per_block ({exc})") from exc

    @property
    def kappa(self) -> float:
        """This device's positioning-to-transfer ratio, in blocks."""
        if self.t_xfer_us_per_block == 0:
            return float("inf")
        return self.t_pos_us / self.t_xfer_us_per_block

    def device_time_us(self, metrics: "Metrics") -> float:
        return (metrics.device_ios * self.t_pos_us
                + metrics.blocks_moved * self.t_xfer_us_per_block)

    @staticmethod
    def kappa_star(policy: "Metrics", none: "Metrics") -> float:
        """Critical ``t_pos/t_xfer`` above which ``policy`` costs less than ``none``.

        ``kappa* = dBlocks / -dIOs``. A policy that issues fewer device I/Os
        than ``none`` wins once positioning is at least kappa* times dearer
        than moving one block; a device with ``kappa > kappa*`` favours it.
        ``inf`` means the policy never wins as positioning comes to dominate:
        it issues at least as many I/Os. ``0.0`` means it wins at every
        ratio (fewer I/Os and no extra blocks).
        """
        d_ios = policy.device_ios - none.device_ios
        d_blocks = policy.blocks_moved - none.blocks_moved
        if d_ios < 0:
            return max(0.0, d_blocks / -d_ios)
        if d_ios == 0 and d_blocks < 0:
            return 0.0
        return float("inf")


@dataclass
class Metrics:
    read_blocks: int = 0
    read_hits: int = 0
    prefetches: int = 0
    useful_prefetches: int = 0
    prefetchable_misses: int = 0
    policy_switches: int = 0
    pseudo_updates: int = 0
    demand_latency_us: float = 0.0
    prefetch_cost_us: float = 0.0
    inference_ms: float = 0.0
    inference_calls: int = 0
    #: Realised prefetch precision at the last decision point, and how far it
    #: sat above the cost model's break-even. Both are observed, not predicted.
    #: ``observed_precision`` is ``None`` when the gate never issued anything,
    #: so "never measured" stays distinguishable from "measured as zero".
    observed_precision: float | None = None
    precision_margin: float = -1.0
    #: Stride confirmed by the correlation detector, when one is running.
    observed_stride: int | None = None
    #: Device-I/O accounting (D18). A demand I/O is one maximal run of
    #: contiguous missed blocks inside one read request -- a resident block in
    #: the middle splits the read, as it does in a page cache. A prefetch I/O
    #: is one maximal run of contiguous blocks newly prefetched after one
    #: request, in issue order. A prefetch run that starts at the request's
    #: end while the request's last block missed would be merged by the block
    #: layer into that demand read, so it is counted as mergeable.
    demand_ios: int = 0
    prefetch_ios: int = 0
    prefetch_ios_mergeable: int = 0
    #: Block-age entries the bounded ``WindowContext`` dropped. Non-zero means
    #: the contextual features differ from an unbounded table (M9).
    age_table_evictions: int = 0

    @property
    def device_ios(self) -> int:
        return self.demand_ios + self.prefetch_ios - self.prefetch_ios_mergeable

    @property
    def blocks_moved(self) -> int:
        """Blocks read from the device: demand-missed plus prefetched."""
        return self.read_blocks - self.read_hits + self.prefetches

    @property
    def hit_ratio(self) -> float:
        return self.read_hits / self.read_blocks if self.read_blocks else 0.0

    @property
    def prefetch_precision(self) -> float:
        return self.useful_prefetches / self.prefetches if self.prefetches else 0.0

    @property
    def prefetch_recall(self) -> float:
        denominator = self.useful_prefetches + self.prefetchable_misses
        return self.useful_prefetches / denominator if denominator else 0.0

    @property
    def unused_prefetches(self) -> int:
        return self.prefetches - self.useful_prefetches

    @property
    def mean_access_latency_us(self) -> float:
        if not self.read_blocks:
            return 0.0
        return (self.demand_latency_us + self.prefetch_cost_us) / self.read_blocks

    @property
    def inference_us_per_call(self) -> float:
        return 1000 * self.inference_ms / self.inference_calls if self.inference_calls else 0.0


class LRUCache:
    def __init__(self, capacity: int, metrics: Metrics):
        if capacity <= 0:
            raise ValueError("cache capacity must be positive")
        self.capacity = capacity
        self.metrics = metrics
        self.blocks: OrderedDict[int, bool] = OrderedDict()

    def _insert(self, block: int, prefetched: bool) -> None:
        if block in self.blocks:
            self.blocks.move_to_end(block)
            return
        if len(self.blocks) == self.capacity:
            self.blocks.popitem(last=False)
        self.blocks[block] = prefetched

    def read(self, block: int) -> bool:
        self.metrics.read_blocks += 1
        if block in self.blocks:
            self.metrics.read_hits += 1
            if self.blocks[block]:
                self.metrics.useful_prefetches += 1
                self.blocks[block] = False
            self.blocks.move_to_end(block)
            return True
        self._insert(block, False)
        return False

    def write(self, block: int) -> None:
        if block in self.blocks:
            self.blocks[block] = False
            self.blocks.move_to_end(block)
        else:
            self._insert(block, False)

    def prefetch(self, block: int) -> bool:
        if block < 0 or block in self.blocks:
            return False
        self._insert(block, True)
        self.metrics.prefetches += 1
        return True


class CacheView:
    """Read-only view of the replay cache for cache-aware predictors.

    A predictor that sets ``cache_aware = True`` is driven through
    ``observe(request, missed_blocks, cache_view)`` instead of
    ``next_candidates``. It may ask whether a block is resident and whether a
    resident block is a still-unused prefetch, but it cannot change the
    cache: every prefetch still goes through the replay loop, under the same
    first-window gate and accounting as every other mode.

    ``prefetch_hits`` holds the blocks of the *current* request that were
    served from a still-unused prefetch. After the demand reads those flags
    are already cleared, so this is the only way a predictor can see "the
    request touched my read-ahead marker". The replay loop sets it before
    each ``observe`` call.
    """

    __slots__ = ("_blocks", "capacity", "prefetch_hits")

    def __init__(self, cache: LRUCache):
        self._blocks = cache.blocks
        self.capacity = cache.capacity
        self.prefetch_hits: list[int] = []

    def __contains__(self, block: int) -> bool:
        return block in self._blocks

    def __len__(self) -> int:
        return len(self._blocks)

    def is_unused_prefetch(self, block: int) -> bool:
        return bool(self._blocks.get(block, False))


def candidates(request: Request, policy: str, stride: int | None) -> list[int]:
    """Window-based adaptive/fixed policy candidate generator."""
    if request.operation != "R":
        return []
    if policy == "sequential":
        return [request.lba + request.size_blocks + offset for offset in range(2)]
    if policy == "strided" and stride is not None:
        return [request.lba + stride]
    if policy == "mixed":
        return [request.lba + request.size_blocks]
    return []


#: Minimum evidence before the adaptive router is allowed to prefetch at all.
#: Calibrated against the measured real-trace window distribution, where
#: ``contiguous_ratio`` has median 0.032 and p90 0.097 -- so a window has to be
#: clearly more contiguous than a typical real window before read-ahead pays.
MIN_CONTIGUITY = 0.10
#: A repeated non-contiguous stride must be this common in the window before
#: we trust it enough to issue a stride prefetch.
MIN_STRIDE_RATIO = 0.10
#: Above this fraction of distant jumps, prefetching is not worth the pollution.
RANDOM_JUMP_CEILING = 0.97


def route_window(window: list[Request], stride: int | None) -> str:
    """Choose a prefetch policy from the window's own measurements.

    This replaces the four-class lookup for real traces. The classifier's
    classes come from the synthetic generator, and measured real windows sit
    ~12 sd outside that region, so a class prediction on a real trace is
    extrapolation. Routing directly on the three features that carry the
    decision (contiguity, repeated stride, jumpiness) needs no training data
    and degrades gracefully on any distribution.

    Returns one of ``sequential``, ``strided``, ``mixed``, ``none``.
    """
    features = extract(window)
    contiguous, stride_ratio, random_jump = features[0], features[1], features[2]
    # Bursty arrivals plus heavy jumping means the next address is not
    # recoverable; polluting the cache costs more than a speculative hit wins.
    if random_jump > RANDOM_JUMP_CEILING and contiguous < MIN_CONTIGUITY * 2:
        return "none"
    if contiguous >= MIN_CONTIGUITY:
        return "sequential"
    if stride_ratio >= MIN_STRIDE_RATIO and stride:
        return "strided"
    if contiguous > 0:
        return "mixed"
    return "none"


#: Modes whose prefetch depth is set by observed precision, not prediction.
GATED_MODES = ("guard", "depth_adaptive", "correlate")

#: Fixed read-ahead depth for the `deep` baseline, in blocks.
#:
#: Depth turned out to matter far more than policy choice: on 8 real traces
#: (first 40k requests each, 2048-block cache) fixed read-ahead at depth 8
#: beats the depth-2 sequential baseline by **+4.10 points of mean hit
#: ratio**, up to +11.05 on src2_2. Depth 16 scores slightly higher still on
#: the mean (+4.66) but costs 4x the prefetches and *loses* 7.2 points on
#: rsrch_2, where a deeper prefetch evicts blocks the workload still wants.
#: 8 is the depth that is best or near-best on every trace, which is why it
#: is the default and not the argmax. Every policy that existed before `deep`
#: prefetched exactly 2 blocks, which is why a perfect policy router could
#: only find +0.4 points of headroom (docs/DECISIONS.md D11, D13).
DEFAULT_READ_AHEAD_DEPTH = 8

#: Default cache capacity, in blocks (512 B each, so 2048 blocks = 1 MiB).
#:
#: This was 128 blocks, which was calibrated for the original generator's
#: 1-3 block requests. Once the generator was corrected to issue realistic
#: 8-128 block requests, a 128-block cache held only ~16 whole requests and
#: starved every policy; measured hit ratios saturated by 2048 blocks. The
#: default now matches the request geometry rather than the old toy.
DEFAULT_CAPACITY = 2048


#: Every mode ``replay()`` accepts. ``_make_predictor`` must agree.
MODES_SUPPORTED = frozenset({
    "adaptive", "adaptive_evidence", "none", "sequential", "strided",
    "stride", "markov", "lstm", "deep", "linux_ra", "learned_depth",
    *GATED_MODES,
})


def _validate_replay(mode: str, window_size: int, policy_interval: int | None,
                     read_ahead_depth: int,
                     linux_ra_max_blocks: int = DEFAULT_LINUX_RA_MAX_BLOCKS,
                     age_table_bound: int | None = DEFAULT_AGE_TABLE_BOUND) -> int:
    """Shared parameter checks for ``replay`` and ``replay_stream``.

    These two paths had drifted: the streamed one accepted a ``window_size``
    below the documented floor and a negative ``read_ahead_depth``, silently
    producing results the in-memory path would have refused. One function so
    they cannot diverge again.

    Returns the decision-interval size to use.
    """
    if mode not in MODES_SUPPORTED:
        raise ValueError(f"unknown mode: {mode!r}")
    if window_size < 8:
        raise ValueError("window_size must be at least 8")
    if policy_interval is not None and policy_interval < 8:
        raise ValueError("policy_interval must be at least 8")
    if read_ahead_depth < 0:
        raise ValueError("read_ahead_depth must be non-negative")
    if linux_ra_max_blocks < 1:
        raise ValueError("linux_ra_max_blocks must be at least 1")
    if age_table_bound is not None and age_table_bound < 1:
        raise ValueError("age_table_bound must be positive or None")
    return policy_interval or window_size


def _publish_gate(gate: "GateController", metrics: "Metrics") -> None:
    """Copy the feedback-gated controller's final state into ``Metrics``.

    One definition, called from both replay paths, so the two can never report
    different things for the same run.
    """
    metrics.policy_switches = gate.switches
    metrics.precision_margin = gate.margin
    # ``None`` means never measured, which is different from measured as zero.
    metrics.observed_precision = gate.precision if gate.has_data else None
    metrics.observed_stride = gate.stride


def _demand_read(cache: LRUCache, metrics: Metrics, latency: LatencyModel,
                 request: Request, missed: list[int] | None = None,
                 reused: list[int] | None = None) -> bool:
    """Serve one request's blocks from the cache; shared by both replay paths.

    Counts one demand device I/O per maximal run of missed blocks. When given,
    ``missed`` collects the blocks that missed and ``reused`` the blocks that
    hit a still-unused prefetch. Returns whether the request's last block
    missed, which is what decides if a read-ahead run starting at its end can
    merge into the demand read.
    """
    start = request.lba
    end = start + request.size_blocks
    if request.operation != "R":
        for block in range(start, end):
            cache.write(block)
        return False
    hit_us = latency.hit_us
    miss_us = latency.demand_miss_us
    in_run = False
    useful = metrics.useful_prefetches
    for block in range(start, end):
        if cache.read(block):
            metrics.demand_latency_us += hit_us
            in_run = False
            if reused is not None and metrics.useful_prefetches != useful:
                reused.append(block)
                useful = metrics.useful_prefetches
        else:
            metrics.demand_latency_us += miss_us
            if not in_run:
                metrics.demand_ios += 1
                in_run = True
            if missed is not None:
                missed.append(block)
    return in_run


def _issue_prefetches(cache: LRUCache, metrics: Metrics, latency: LatencyModel,
                      gate: "GateController | None", proposed, end: int,
                      tail_missed: bool) -> None:
    """Issue one request's prefetch candidates; shared by both replay paths.

    A prefetch I/O is a maximal run of consecutive *newly issued* blocks, so a
    candidate that is already resident splits the run. Only the caller
    decides whether to call this at all (the first-window gate).
    """
    cost = latency.prefetch_us
    last = None
    for block in proposed:
        if cache.prefetch(block):
            metrics.prefetch_cost_us += cost
            if last is None or block != last + 1:
                metrics.prefetch_ios += 1
                if tail_missed and block == end:
                    metrics.prefetch_ios_mergeable += 1
            last = block
            if gate is not None:
                gate.note_prefetch()


def replay(requests: list[Request], capacity: int = DEFAULT_CAPACITY,
           window_size: int = 32,
           mode: str = "adaptive", model: OnlineGaussianNB | None = None,
           labels: list[str] | None = None, online_updates: bool = False,
           latency_model: LatencyModel | None = None,
           lstm_model_path: str | None = None,
           pseudo_label_threshold: float | None = None,
           smoothing: tuple[int, float] | None = None,
           contextual: bool = True,
           policy_interval: int | None = None,
           read_ahead_depth: int = DEFAULT_READ_AHEAD_DEPTH,
           cache_factory=LRUCache,
           linux_ra_max_blocks: int = DEFAULT_LINUX_RA_MAX_BLOCKS,
           age_table_bound: int | None = DEFAULT_AGE_TABLE_BOUND,
           depth_model=None):
    """Replay in request order; a completed window only affects later requests.

    ``labels`` represent controlled ground truth. Optional pseudo-label updates
    are self-training experiments, never accuracy evidence on real traces.
    Every mode observes the first window without issuing prefetches.

    ``policy_interval`` lets the feedback-gated modes re-decide more often than
    once per window. The classifier still scores a whole window (it needs the
    data), but the throttle can act on a shorter cadence, which is what lets it
    react when the workload changes mid-window. Defaults to ``window_size``.

    ``cache_factory(capacity, metrics)`` builds the cache. It must keep
    ``LRUCache``'s semantics and attributes; it exists so an instrumented
    subclass (for example one that records every operation for a real-device
    replay) sees exactly the cache operations this loop performs.
    ``linux_ra_max_blocks`` is the ``linux_ra`` window ceiling, and
    ``age_table_bound`` caps the contextual block-age table (``None`` for
    unbounded). ``depth_model`` is the ``learned_depth`` artifact path, or an
    already-built per-request predictor (one fresh object per replay; see
    :func:`_make_predictor`).
    """
    interval_size = _validate_replay(mode, window_size, policy_interval,
                                     read_ahead_depth, linux_ra_max_blocks,
                                     age_table_bound)
    if mode == "adaptive" and model is None:
        raise ValueError("adaptive replay requires a trained model")
    if smoothing is not None and mode not in ("adaptive", "adaptive_evidence"):
        raise ValueError("policy smoothing only applies to adaptive replay")
    if online_updates and (labels is None or len(labels) < len(requests) // window_size):
        raise ValueError("online updates require a label for each complete window")
    if pseudo_label_threshold is not None:
        if mode != "adaptive" or labels is not None or not 0 < pseudo_label_threshold <= 1:
            raise ValueError("pseudo-label updates need unlabelled adaptive replay and threshold in (0,1]")

    latency = latency_model or LatencyModel()
    metrics = Metrics()
    cache = cache_factory(capacity, metrics)
    smoother = PolicySmoother(*smoothing) if smoothing else None
    remaining_reads = Counter(block for req in requests if req.operation == "R"
                              for block in range(req.lba, req.lba + req.size_blocks))
    predictor = _make_predictor(mode, lstm_model_path, window_size=window_size,
                                linux_ra_max_blocks=linux_ra_max_blocks,
                                depth_model=depth_model)
    observer = predictor if getattr(predictor, "cache_aware", False) else None
    view = CacheView(cache) if observer is not None else None

    history: list[Request] = []
    predictions: list[tuple[int, str, float]] = []
    # Contextual feature state: block ages and the previous window's spatial
    # features. This is stream information a real system has at decision time
    # (block-age tracking), not lookahead.
    context = WindowContext(age_table_bound) if contextual else None
    # A block is a "prefetchable miss" once. Counting it again after eviction
    # inflated the oracle-recall denominator whenever the cache was smaller
    # than the working set, which silently biased recall.
    counted_misses: set[int] = set()
    policy = "none" if mode == "adaptive_evidence" else (
        "random" if mode == "adaptive" else mode)
    stride: int | None = None
    # Feedback-gated state. `monitor` sees the realised precision of our own
    # prefetches (LRUCache already counts useful_prefetches), so the decision
    # is causal and needs no future knowledge and no trained model.
    gated = mode in GATED_MODES
    gate = (GateController(mode, latency.break_even_precision())
            if gated else None)
    interval = 0
    for index, request in enumerate(requests):
        end = request.lba + request.size_blocks
        if request.operation == "R":
            for block in range(request.lba, end):
                remaining_reads[block] -= 1
        missed: list[int] = []
        reused = [] if observer is not None else None
        tail_missed = _demand_read(cache, metrics, latency, request, missed,
                                   reused)
        # Blocks of one request are distinct, so checking the misses after
        # the whole request was decremented equals checking each in turn.
        for block in missed:
            if remaining_reads[block] > 0 and block not in counted_misses:
                counted_misses.add(block)
                metrics.prefetchable_misses += 1

        if observer is not None:
            view.prefetch_hits = reused  # type: ignore[union-attr]
            start = perf_counter()
            proposed = observer.observe(request, missed, view)
            metrics.inference_ms += (perf_counter() - start) * 1000
            metrics.inference_calls += 1
        elif predictor is not None:
            start = perf_counter()
            proposed = predictor.next_candidates(request)
            metrics.inference_ms += (perf_counter() - start) * 1000
            metrics.inference_calls += 1
        elif gate is not None:
            # Depth was decided at the previous decision point.
            proposed = gate.candidates(request)
        elif mode == "deep":
            proposed = read_ahead(request, read_ahead_depth)
        else:
            proposed = candidates(request, policy, stride)
        if index >= window_size:
            _issue_prefetches(cache, metrics, latency, gate, proposed, end,
                              tail_missed)

        if gate is not None:
            gate.observe(request)
            gate.note_useful(metrics.useful_prefetches)
            interval += 1
        history.append(request)
        if gate is not None and interval >= interval_size:
            # The throttle re-decides on its own cadence, independent of the
            # window boundary, so it can react inside a window.
            gate.decide()
            interval = 0

        if len(history) == window_size:
            stride = dominant_stride(history)
            if mode == "adaptive_evidence":
                # Route on the window's own measurements: no training
                # distribution is assumed, so this behaves the same on real
                # traces as on synthetic ones.
                start = perf_counter()
                chosen = route_window(history, stride)
                metrics.inference_ms += (perf_counter() - start) * 1000
                metrics.inference_calls += 1
                window_number = index // window_size
                predictions.append((window_number, chosen, 1.0))
                if smoother is not None:
                    policy = smoother.update(chosen, 1.0)
                else:
                    if chosen != policy:
                        metrics.policy_switches += 1
                    policy = chosen
            elif mode == "adaptive":
                features = (context.observe(history, stride) if context is not None
                            else extract(history))
                start = perf_counter()
                predicted, confidence = model.predict(features)  # type: ignore[union-attr]
                metrics.inference_ms += (perf_counter() - start) * 1000
                metrics.inference_calls += 1
                window_number = index // window_size
                predictions.append((window_number, predicted, confidence))
                # Optional postprocessing: a single misclassified window would
                # otherwise flip the policy for a whole window of requests.
                if smoother is not None:
                    policy = smoother.update(predicted, confidence)
                else:
                    if predicted != policy:
                        metrics.policy_switches += 1
                    policy = predicted
                if online_updates:
                    model.update(features, labels[window_number])  # type: ignore[union-attr,index]
                elif pseudo_label_threshold is not None and confidence >= pseudo_label_threshold:
                    model.update(features, predicted)  # type: ignore[union-attr]
                    metrics.pseudo_updates += 1
            history = []
    if gate is not None:
        # Account the trailing partial interval; otherwise its usefulness is
        # silently dropped and the reported precision is understated. The
        # publish is unconditional: a stream that ends exactly on an interval
        # boundary still has real state to report.
        if interval:
            gate.note_useful(metrics.useful_prefetches)
            gate.decide()
        _publish_gate(gate, metrics)
    if context is not None:
        metrics.age_table_evictions = context.evictions
    return metrics, predictions


def replay_stream(chunks, capacity: int = DEFAULT_CAPACITY,
                  window_size: int = 32, mode: str = "adaptive",
                  model: OnlineGaussianNB | None = None,
                  latency_model: LatencyModel | None = None,
                  lstm_model_path: str | None = None,
                  contextual: bool = True,
                  policy_interval: int | None = None,
                  read_ahead_depth: int = DEFAULT_READ_AHEAD_DEPTH,
                  cache_factory=LRUCache,
                  linux_ra_max_blocks: int = DEFAULT_LINUX_RA_MAX_BLOCKS,
                  age_table_bound: int | None = DEFAULT_AGE_TABLE_BOUND,
                  depth_model=None) -> Metrics:
    """Replay a request stream delivered in bounded chunks.


    Exists because the MSRC collection is 11.9 GB over 32 files and a
    ``Request`` costs ~430 bytes resident, so it cannot be held in memory.
    Cache contents, metrics, partial-window state, block ages, the current
    stride and the active policy all carry across chunk boundaries, so the
    result is identical to calling :func:`replay` on the concatenated stream
    with one caveat:

    ``Metrics.prefetch_recall`` is **not** reported by this function. Recall
    needs to know whether a missed block is read *again later*, which is
    whole-stream knowledge that a bounded-memory pass cannot have without
    holding a block index for the entire trace. Hit ratio, precision, wasted
    I/O, device I/Os, modelled cost, policy switches and inference counters
    are all exact, and those are the metrics that rank policies.

    ``cache_factory``, ``linux_ra_max_blocks``, ``age_table_bound`` and
    ``depth_model`` mean exactly what they mean for :func:`replay`.
    """
    interval_size = _validate_replay(mode, window_size, policy_interval,
                                     read_ahead_depth, linux_ra_max_blocks,
                                     age_table_bound)
    latency = latency_model or LatencyModel()
    metrics = Metrics()
    cache = cache_factory(capacity, metrics)
    context = WindowContext(age_table_bound) if contextual else None
    predictor = _make_predictor(mode, lstm_model_path, window_size=window_size,
                                linux_ra_max_blocks=linux_ra_max_blocks,
                                depth_model=depth_model)
    observer = predictor if getattr(predictor, "cache_aware", False) else None
    view = CacheView(cache) if observer is not None else None
    # Same controller as replay(). These two paths used to carry separate
    # copies of this logic, which is how the gated modes ended up missing
    # from the streamed path entirely.
    gate = (GateController(mode, latency.break_even_precision())
            if mode in GATED_MODES else None)
    policy = "none" if mode == "adaptive_evidence" else (
        "random" if mode == "adaptive" else mode)
    stride: int | None = None
    history: list[Request] = []
    seen = 0
    interval = 0

    for chunk in chunks:
        for request in chunk:
            missed = [] if observer is not None else None
            reused = [] if observer is not None else None
            tail_missed = _demand_read(cache, metrics, latency, request,
                                       missed, reused)
            # Inference is timed and counted exactly as replay() does it;
            # leaving it out here made the streamed rows report zero
            # inference cost for every per-request predictor (M13).
            if observer is not None:
                view.prefetch_hits = reused  # type: ignore[union-attr]
                start = perf_counter()
                proposed = observer.observe(request, missed, view)
                metrics.inference_ms += (perf_counter() - start) * 1000
                metrics.inference_calls += 1
            elif predictor is not None:
                start = perf_counter()
                proposed = predictor.next_candidates(request)
                metrics.inference_ms += (perf_counter() - start) * 1000
                metrics.inference_calls += 1
            elif gate is not None:
                proposed = gate.candidates(request)
            elif mode == "deep":
                proposed = read_ahead(request, read_ahead_depth)
            else:
                proposed = candidates(request, policy, stride)
            # Causality, matching ``replay`` exactly: the first
            # ``window_size`` requests observe without prefetching.
            if seen >= window_size:
                _issue_prefetches(cache, metrics, latency, gate, proposed,
                                  request.lba + request.size_blocks,
                                  tail_missed)
            if gate is not None:
                gate.observe(request)
                gate.note_useful(metrics.useful_prefetches)
                interval += 1
            seen += 1
            history.append(request)

            if gate is not None and interval >= interval_size:
                gate.decide()
                interval = 0
            if len(history) == window_size:
                stride = dominant_stride(history)
                if mode == "adaptive_evidence":
                    start = perf_counter()
                    chosen = route_window(history, stride)
                    metrics.inference_ms += (perf_counter() - start) * 1000
                    metrics.inference_calls += 1
                    # Matches replay(): a switch is a change of chosen
                    # policy, and omitting this made the two paths disagree
                    # by ~230 on web_3 while agreeing on every cache metric.
                    if chosen != policy:
                        metrics.policy_switches += 1
                    policy = chosen
                elif mode == "adaptive":
                    if model is None:
                        raise ValueError("adaptive replay requires a trained model")
                    values = (context.observe(history, stride)
                              if context is not None else extract(history))
                    start = perf_counter()
                    predicted, _ = model.predict(values)  # type: ignore[union-attr]
                    metrics.inference_ms += (perf_counter() - start) * 1000
                    metrics.inference_calls += 1
                    if predicted != policy:
                        metrics.policy_switches += 1
                    policy = predicted
                history = []
    if gate is not None:
        if interval:
            gate.note_useful(metrics.useful_prefetches)
            gate.decide()
        _publish_gate(gate, metrics)
    if context is not None:
        metrics.age_table_evictions = context.evictions
    return metrics


def _make_predictor(mode: str, lstm_model_path: str | None, *,
                    window_size: int = 32,
                    linux_ra_max_blocks: int = DEFAULT_LINUX_RA_MAX_BLOCKS,
                    depth_model=None):
    """The per-request predictor for ``mode``, or ``None`` for window modes.

    Both replay paths build their predictor here, so a mode cannot be wired
    into one path and forgotten in the other.

    ``learned_depth`` takes ``depth_model``: an artifact path, loaded with
    its validation-selected thresholds, or an object with
    ``next_candidates`` that is used as given. The object form exists for
    threshold selection and for baselines replayed through the same loop
    (``depth_model.PrecomputedDepth``, ``RelativeDepth``). A predictor is
    stateful, so each replay needs a fresh one.
    """
    if mode == "stride":
        return StridePrefetcher()
    if mode == "markov":
        return MarkovPrefetcher()
    if mode == "linux_ra":
        # The emulation keeps its window state still through the first
        # window, exactly like the validated prototype, so its first decision
        # sees the same history every other mode's first decision sees.
        return LinuxReadahead(linux_ra_max_blocks, warmup=window_size)
    if mode == "lstm":
        if not lstm_model_path:
            raise ValueError("lstm mode requires --lstm-model path")
        from .lstm import load_predictor
        return load_predictor(lstm_model_path)
    if mode == "learned_depth":
        if depth_model is None:
            raise RuntimeError("no trained learned_depth artifact; "
                               "optional baseline skipped")
        if hasattr(depth_model, "next_candidates"):
            return depth_model
        from .depth_model import load_prefetcher
        return load_prefetcher(depth_model)
    if mode not in MODES_SUPPORTED:
        raise ValueError(f"unknown mode: {mode!r}")
    return None


def oracle_reference(requests: list[Request], capacity: int = DEFAULT_CAPACITY,
                     window_size: int = 32,
                     latency_model: LatencyModel | None = None) -> dict:
    """Perfect-foreknowledge reference: the ceiling for *any* predictor.

    Every read is prefetched one request ahead, so this isolates "can the
    cache and prefetch mechanism turn an access into a hit" from "can a
    predictor know the address". It is a bound, not a policy, and is reported
    separately from the seven benchmark modes.

    Note this is a *prefetch* ceiling. A demand-cache reuse ceiling is a
    different and much smaller number on these traces, because most read
    blocks are never requested twice: a synthetic scan reads each block once,
    so a perfect predictor can still convert the first touch into a hit.
    """
    latency = latency_model or LatencyModel()
    metrics = Metrics()
    cache = LRUCache(capacity, metrics)
    for index, request in enumerate(requests):
        for block in range(request.lba, request.lba + request.size_blocks):
            if request.operation == "R":
                hit = cache.read(block)
                metrics.demand_latency_us += (latency.hit_us if hit
                                              else latency.demand_miss_us)
            else:
                cache.write(block)
        if index >= window_size and index + 1 < len(requests):
            upcoming = requests[index + 1]
            if upcoming.operation == "R":
                for block in range(upcoming.lba, upcoming.lba + upcoming.size_blocks):
                    if cache.prefetch(block):
                        metrics.prefetch_cost_us += latency.prefetch_us
    return {
        "hit_ratio": metrics.hit_ratio,
        "mean_access_latency_us": metrics.mean_access_latency_us,
        "prefetches": metrics.prefetches,
        "read_blocks": metrics.read_blocks,
    }


def confusion_matrix(pairs: list[tuple[str, str]]) -> dict[str, Counter]:
    result: dict[str, Counter] = {}
    for truth, predicted in pairs:
        result.setdefault(truth, Counter())[predicted] += 1
    return result
