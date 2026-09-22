"""Causal block-level cache replay with identical rules for all policies."""

from __future__ import annotations

from collections import Counter, OrderedDict
from dataclasses import dataclass
from time import perf_counter

from .baselines import MarkovPrefetcher, StridePrefetcher
from .features import dominant_stride, extract
from .model import OnlineGaussianNB
from .trace import Request


@dataclass(frozen=True)
class LatencyModel:
    hit_us: float = 5.0
    demand_miss_us: float = 100.0
    prefetch_us: float = 50.0

    def __post_init__(self) -> None:
        if min(self.hit_us, self.demand_miss_us, self.prefetch_us) < 0:
            raise ValueError("latency model values must be non-negative")


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


def replay(requests: list[Request], capacity: int = 128, window_size: int = 32,
           mode: str = "adaptive", model: OnlineGaussianNB | None = None,
           labels: list[str] | None = None, online_updates: bool = False,
           latency_model: LatencyModel | None = None,
           lstm_model_path: str | None = None,
           pseudo_label_threshold: float | None = None):
    """Replay in request order; a completed window only affects later requests.

    ``labels`` represent controlled ground truth. Optional pseudo-label updates
    are self-training experiments, never accuracy evidence on real traces.
    Every mode observes the first window without issuing prefetches.
    """
    modes = {"adaptive", "none", "sequential", "strided", "stride", "markov", "lstm"}
    if mode not in modes or window_size < 8:
        raise ValueError("unknown mode or window_size below 8")
    if mode == "adaptive" and model is None:
        raise ValueError("adaptive replay requires a trained model")
    if online_updates and (labels is None or len(labels) < len(requests) // window_size):
        raise ValueError("online updates require a label for each complete window")
    if pseudo_label_threshold is not None:
        if mode != "adaptive" or labels is not None or not 0 < pseudo_label_threshold <= 1:
            raise ValueError("pseudo-label updates need unlabelled adaptive replay and threshold in (0,1]")

    latency = latency_model or LatencyModel()
    metrics = Metrics()
    cache = LRUCache(capacity, metrics)
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
    policy = "random" if mode == "adaptive" else mode
    stride: int | None = None
    for index, request in enumerate(requests):
        for block in range(request.lba, request.lba + request.size_blocks):
            if request.operation == "R":
                remaining_reads[block] -= 1
                hit = cache.read(block)
                if not hit and remaining_reads[block] > 0:
                    metrics.prefetchable_misses += 1
                metrics.demand_latency_us += latency.hit_us if hit else latency.demand_miss_us
            else:
                cache.write(block)

        if predictor is not None:
            start = perf_counter()
            proposed = predictor.next_candidates(request)
            metrics.inference_ms += (perf_counter() - start) * 1000
            metrics.inference_calls += 1
        else:
            proposed = candidates(request, policy, stride)
        if index >= window_size:
            for block in proposed:
                if cache.prefetch(block):
                    metrics.prefetch_cost_us += latency.prefetch_us

        history.append(request)
        if len(history) == window_size:
            stride = dominant_stride(history)
            if mode == "adaptive":
                features = extract(history)
                start = perf_counter()
                predicted, confidence = model.predict(features)  # type: ignore[union-attr]
                metrics.inference_ms += (perf_counter() - start) * 1000
                metrics.inference_calls += 1
                window_number = index // window_size
                predictions.append((window_number, predicted, confidence))
                if predicted != policy:
                    metrics.policy_switches += 1
                policy = predicted
                if online_updates:
                    model.update(features, labels[window_number])  # type: ignore[union-attr,index]
                elif pseudo_label_threshold is not None and confidence >= pseudo_label_threshold:
                    model.update(features, predicted)  # type: ignore[union-attr]
                    metrics.pseudo_updates += 1
            history = []
    return metrics, predictions


def confusion_matrix(pairs: list[tuple[str, str]]) -> dict[str, Counter]:
    result: dict[str, Counter] = {}
    for truth, predicted in pairs:
        result.setdefault(truth, Counter())[predicted] += 1
    return result
