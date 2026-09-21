"""Causal, block-level LRU cache and prefetch-policy replay."""

from __future__ import annotations

from collections import Counter, OrderedDict
from dataclasses import dataclass

from .features import dominant_stride, extract
from .model import OnlineGaussianNB
from .trace import Request


@dataclass
class Metrics:
    read_blocks: int = 0
    read_hits: int = 0
    prefetches: int = 0
    useful_prefetches: int = 0
    policy_switches: int = 0

    @property
    def hit_ratio(self) -> float:
        return self.read_hits / self.read_blocks if self.read_blocks else 0.0

    @property
    def prefetch_precision(self) -> float:
        return self.useful_prefetches / self.prefetches if self.prefetches else 0.0

    @property
    def unused_prefetches(self) -> int:
        return self.prefetches - self.useful_prefetches


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

    def read(self, block: int) -> None:
        self.metrics.read_blocks += 1
        if block in self.blocks:
            self.metrics.read_hits += 1
            if self.blocks[block]:
                self.metrics.useful_prefetches += 1
                self.blocks[block] = False
            self.blocks.move_to_end(block)
        else:
            self._insert(block, False)

    def write(self, block: int) -> None:
        # Write-allocate model; a write does not count as a successful prefetch.
        if block in self.blocks:
            self.blocks[block] = False
            self.blocks.move_to_end(block)
        else:
            self._insert(block, False)

    def prefetch(self, block: int) -> None:
        if block < 0 or block in self.blocks:
            return
        self._insert(block, True)
        self.metrics.prefetches += 1


def candidates(request: Request, policy: str, stride: int | None) -> list[int]:
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
           labels: list[str] | None = None, online_updates: bool = False):
    """Replay requests, applying each completed window's prediction *later*.

    Labels are window labels from controlled synthetic data. They are never
    inferred from prefetch success. Real, unlabelled traces use frozen models.
    """
    if mode not in {"adaptive", "none", "sequential", "strided"}:
        raise ValueError("unknown simulation mode")
    if window_size < 8:
        raise ValueError("window_size must be at least 8")
    if mode == "adaptive" and model is None:
        raise ValueError("adaptive replay requires a trained model")
    if online_updates and labels is None:
        raise ValueError("online updates require trustworthy window labels")
    metrics = Metrics()
    cache = LRUCache(capacity, metrics)
    history: list[Request] = []
    predictions: list[tuple[int, str, float]] = []
    policy = "random" if mode == "adaptive" else mode
    stride: int | None = None
    for index, request in enumerate(requests):
        for block in range(request.lba, request.lba + request.size_blocks):
            if request.operation == "R":
                cache.read(block)
            else:
                cache.write(block)
        for block in candidates(request, policy, stride):
            cache.prefetch(block)
        history.append(request)
        if len(history) == window_size:
            stride = dominant_stride(history)
            if mode == "adaptive":
                predicted, confidence = model.predict(extract(history))  # type: ignore[union-attr]
                predictions.append((index // window_size, predicted, confidence))
                if predicted != policy:
                    metrics.policy_switches += 1
                policy = predicted
                if online_updates:
                    label_index = index // window_size
                    if label_index >= len(labels):  # type: ignore[arg-type]
                        raise ValueError("too few ground-truth labels")
                    model.update(extract(history), labels[label_index])  # type: ignore[union-attr,index]
            history = []
    return metrics, predictions


def confusion_matrix(pairs: list[tuple[str, str]]) -> dict[str, Counter]:
    result: dict[str, Counter] = {}
    for truth, predicted in pairs:
        result.setdefault(truth, Counter())[predicted] += 1
    return result
