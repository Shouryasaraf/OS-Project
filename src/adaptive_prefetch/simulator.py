"""Causal block-level cache replay with identical rules for all policies."""

from __future__ import annotations

from collections import Counter, OrderedDict
from dataclasses import dataclass
from time import perf_counter

from .baselines import MarkovPrefetcher, StridePrefetcher
from .features import STREAM_FEATURES, WindowContext, dominant_stride, extract
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
        """
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
        if prefetch_multiple <= 0:
            raise ValueError("prefetch_multiple must be positive")
        mean_service = sum(values) / len(values)
        # A hit is served from cache and costs a small fraction of a device
        # service. 1% is an explicitly-documented approximation, not a
        # measurement: the traces record device time only, never cache time.
        return cls(
            hit_us=max(0.001, mean_service * 0.01),
            demand_miss_us=mean_service,
            prefetch_us=mean_service * prefetch_multiple,
        )

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
    "stride", "markov", "lstm", "deep", *GATED_MODES,
})


def _validate_replay(mode: str, window_size: int, policy_interval: int | None,
                     read_ahead_depth: int) -> int:
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
           read_ahead_depth: int = DEFAULT_READ_AHEAD_DEPTH):
    """Replay in request order; a completed window only affects later requests.

    ``labels`` represent controlled ground truth. Optional pseudo-label updates
    are self-training experiments, never accuracy evidence on real traces.
    Every mode observes the first window without issuing prefetches.

    ``policy_interval`` lets the feedback-gated modes re-decide more often than
    once per window. The classifier still scores a whole window (it needs the
    data), but the throttle can act on a shorter cadence, which is what lets it
    react when the workload changes mid-window. Defaults to ``window_size``.
    """
    interval_size = _validate_replay(mode, window_size, policy_interval,
                                    read_ahead_depth)
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
    cache = LRUCache(capacity, metrics)
    smoother = PolicySmoother(*smoothing) if smoothing else None
    remaining_reads = Counter(block for req in requests if req.operation == "R"
                              for block in range(req.lba, req.lba + req.size_blocks))
    predictor = None
    if mode == "stride":
        predictor = StridePrefetcher()
    elif mode == "markov":
        predictor = MarkovPrefetcher()
    elif mode == "lstm":
        if not lstm_model_path:
            raise ValueError("lstm mode requires --lstm-model path")
        from .lstm import load_predictor
        predictor = load_predictor(lstm_model_path)

    history: list[Request] = []
    predictions: list[tuple[int, str, float]] = []
    # Contextual feature state: block ages and the previous window's spatial
    # features. This is stream information a real system has at decision time
    # (block-age tracking), not lookahead.
    context = WindowContext() if contextual else None
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
        for block in range(request.lba, request.lba + request.size_blocks):
            if request.operation == "R":
                remaining_reads[block] -= 1
                hit = cache.read(block)
                if (not hit and remaining_reads[block] > 0
                        and block not in counted_misses):
                    counted_misses.add(block)
                    metrics.prefetchable_misses += 1
                metrics.demand_latency_us += latency.hit_us if hit else latency.demand_miss_us
            else:
                cache.write(block)

        if predictor is not None:
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
            for block in proposed:
                if cache.prefetch(block):
                    metrics.prefetch_cost_us += latency.prefetch_us
                    if gate is not None:
                        gate.note_prefetch()

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
    return metrics, predictions


def replay_stream(chunks, capacity: int = DEFAULT_CAPACITY,
                  window_size: int = 32, mode: str = "adaptive",
                  model: OnlineGaussianNB | None = None,
                  latency_model: LatencyModel | None = None,
                  lstm_model_path: str | None = None,
                  contextual: bool = True,
                  policy_interval: int | None = None,
                  read_ahead_depth: int = DEFAULT_READ_AHEAD_DEPTH) -> Metrics:
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
    I/O, modelled cost and policy switches are all exact, and those are the
    metrics that rank policies.
    """
    interval_size = _validate_replay(mode, window_size, policy_interval,
                                    read_ahead_depth)
    latency = latency_model or LatencyModel()
    metrics = Metrics()
    cache = LRUCache(capacity, metrics)
    context = WindowContext() if contextual else None
    predictor = _make_predictor(mode, lstm_model_path)
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
            for block in range(request.lba, request.lba + request.size_blocks):
                if request.operation == "R":
                    hit = cache.read(block)
                    metrics.demand_latency_us += (latency.hit_us if hit
                                                  else latency.demand_miss_us)
                else:
                    cache.write(block)
            if predictor is not None:
                proposed = predictor.next_candidates(request)
            elif gate is not None:
                proposed = gate.candidates(request)
            elif mode == "deep":
                proposed = read_ahead(request, read_ahead_depth)
            else:
                proposed = candidates(request, policy, stride)
            # Causality, matching ``replay`` exactly: the first
            # ``window_size`` requests observe without prefetching.
            if seen >= window_size:
                for block in proposed:
                    if cache.prefetch(block):
                        metrics.prefetch_cost_us += latency.prefetch_us
                        if gate is not None:
                            gate.note_prefetch()
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
                    chosen = route_window(history, stride)
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
    return metrics


def _make_predictor(mode: str, lstm_model_path: str | None):
    if mode == "stride":
        return StridePrefetcher()
    if mode == "markov":
        return MarkovPrefetcher()
    if mode == "lstm":
        if not lstm_model_path:
            raise ValueError("lstm mode requires --lstm-model path")
        from .lstm import load_predictor
        return load_predictor(lstm_model_path)
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
