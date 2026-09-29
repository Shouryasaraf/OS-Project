"""Optional offline LSTM baseline for next-delta prediction.

This is a different prediction task from four-class workload classification.
Compare end-to-end cache outcomes and inference cost, not class accuracy.

Design notes (the earlier exact-delta-vocabulary version was near-useless):

* **Bucketed magnitude head.** Predicting one class per *exact* delta caps
  addressability at ~58% of requests, because a fixed vocabulary can never
  emit a stride it did not memorise during training. We instead predict
  (sign, magnitude-bucket) and resolve the bucket back to concrete deltas at
  decode time, which recovers 100% coverage of the forward axis.
* **An explicit unknown class, and real abstention.** The old decoder took
  ``topk(10)`` specifically to "bypass unk predictions", which meant the
  predictor emitted candidates on 100% of reads -- turning a calibrated
  "I don't know" into a guaranteed-wrong prefetch. We now gate on the
  unknown probability and return nothing when the model is unsure.
* **log1p input scaling.** Linear ``delta / MAX_DELTA`` squashes the only
  predictable deltas (1..64) into ~1% of the input range.
"""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

from .trace import Request, transition_trace

HISTORY = 16
MAX_DELTA = 4096
ARTIFACT_VERSION = 2

#: Inclusive upper edge of each magnitude bucket. 0,1,2,4,8,...,MAX_DELTA.
BUCKET_EDGES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, MAX_DELTA)
#: Index of the "unknown / out of range" class (always predicted last).
UNK = 2 * len(BUCKET_EDGES) + 1
NUM_CLASSES = UNK + 1
#: Model should abstain when the unknown probability exceeds this.
ABSTAIN_THRESHOLD = 0.6


def bucket_of(delta: int) -> int:
    """Map a delta to a class index, or UNK when it is out of range."""
    magnitude = abs(delta)
    if magnitude > MAX_DELTA:
        return UNK
    for index, edge in enumerate(BUCKET_EDGES):
        if magnitude <= edge:
            break
    return index if delta >= 0 else len(BUCKET_EDGES) + index


def bucket_span(index: int) -> tuple[int, int]:
    """Return the (low, high) magnitude range covered by a class index.

    Returns an empty span for the unknown class, so callers can test the
    result instead of having to special-case UNK before indexing.
    """
    if index < 0 or index >= UNK:
        return (0, -1)
    if index >= len(BUCKET_EDGES):
        index -= len(BUCKET_EDGES)
    if index < 0 or index >= len(BUCKET_EDGES):
        return (0, -1)
    low = 0 if index == 0 else BUCKET_EDGES[index - 1] + 1
    return low, BUCKET_EDGES[index]


def _encode(delta: int) -> float:
    """Signed log1p encoding; spreads small deltas across the input range."""
    clamped = max(-MAX_DELTA, min(MAX_DELTA, delta))
    return math.copysign(math.log1p(abs(clamped)) / math.log1p(MAX_DELTA), clamped)


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("LSTM requires PyTorch: pip install -e .[lstm]") from exc
    return torch


def _network(num_classes: int = NUM_CLASSES, hidden_size: int = 64):
    torch = _torch()

    class DeltaLSTM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = torch.nn.LSTM(1, hidden_size, batch_first=True)
            self.output = torch.nn.Linear(hidden_size, num_classes)

        def forward(self, x):
            sequence, _ = self.lstm(x)
            return self.output(sequence[:, -1, :])

    return DeltaLSTM()


def _read_deltas(requests: list[Request]) -> list[int]:
    reads = [req.lba for req in requests if req.operation == "R"]
    return [b - a for a, b in zip(reads, reads[1:])]


def _build_examples(
    traces: list[list[Request]],
) -> list[tuple[list[float], int]]:
    examples: list[tuple[list[float], int]] = []
    for trace in traces:
        deltas = _read_deltas(trace)
        for i in range(HISTORY, len(deltas)):
            window = deltas[i - HISTORY:i]
            examples.append(([_encode(d) for d in window], bucket_of(deltas[i])))
    return examples


def train_lstm(
    save_path: str | Path,
    traces: list[list[Request]] | None = None,
    epochs: int = 60,
    vocab_size: int = 0,
    seed: int = 42,
) -> dict[str, int]:
    """Train the offline next-delta LSTM and save a versioned artifact.

    ``vocab_size`` is accepted for backward compatibility with the older
    exact-delta head and is ignored: the bucketed head has a fixed size.
    """
    torch = _torch()
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if traces is None:
        traces = [transition_trace(seed + i, windows_per_class=8)[0] for i in range(4)]

    examples = _build_examples(traces)
    if not examples:
        raise ValueError("need at least 18 read requests in training traces")

    torch.manual_seed(seed)
    network = _network()
    optimizer = torch.optim.Adam(network.parameters(), lr=0.003)
    loss_fn = torch.nn.CrossEntropyLoss()

    inputs = torch.tensor([x for x, _ in examples], dtype=torch.float32).unsqueeze(-1)
    labels = torch.tensor([y for _, y in examples], dtype=torch.long)

    network.train()
    generator = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        # Shuffle every epoch: fixed-order minibatches let adjacent deltas
        # stay correlated within a batch and measurably hurt convergence.
        order = torch.randperm(len(examples), generator=generator)
        for start in range(0, len(examples), 128):
            index = order[start:start + 128]
            optimizer.zero_grad()
            loss = loss_fn(network(inputs[index]), labels[index])
            loss.backward()
            optimizer.step()

    destination = Path(save_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "artifact_version": ARTIFACT_VERSION,
            "state_dict": network.state_dict(),
            "num_classes": NUM_CLASSES,
            "bucket_edges": list(BUCKET_EDGES),
            "hidden_size": 64,
            "history": HISTORY,
            "max_delta": MAX_DELTA,
        },
        destination,
    )

    return {
        "examples": len(examples),
        "classes": NUM_CLASSES,
        "epochs": epochs,
    }


class LSTMPrefetcher:
    def __init__(self, network, edges=BUCKET_EDGES,
                 abstain_threshold: float = ABSTAIN_THRESHOLD):
        self.network = network
        self.edges = tuple(edges)
        self.abstain_threshold = abstain_threshold
        self.n_forward = len(self.edges)
        self.unk = 2 * self.n_forward + 1
        self.last_lba: int | None = None
        # Trimmed to HISTORY: only the tail is ever read, and an unbounded list
        # would retain one int per read across a multi-million-request trace.
        self.deltas: list[int] = []
        self.unk_probability = 0.0

    def next_candidates(self, request: Request) -> list[int]:
        if request.operation != "R":
            return []
        if self.last_lba is not None:
            self.deltas.append(request.lba - self.last_lba)
            if len(self.deltas) > HISTORY:
                del self.deltas[0]
        self.last_lba = request.lba
        if len(self.deltas) < HISTORY:
            return []

        torch = _torch()
        values = [_encode(d) for d in self.deltas]
        x = torch.tensor(values, dtype=torch.float32).reshape(1, HISTORY, 1)
        with torch.no_grad():
            scores = self.network(x).squeeze(0)
            probabilities = torch.softmax(scores, dim=0)

        # Abstain rather than emit noise: this is the single most important
        # behavioural change from the previous top-10 "bypass" decoder.
        unk_probability = float(probabilities[self.unk])
        self.unk_probability = unk_probability
        if unk_probability >= self.abstain_threshold:
            return []

        ranked = torch.topk(
            probabilities, min(5, len(probabilities))
        ).indices.tolist()

        end = request.lba + request.size_blocks
        candidates: list[int] = []
        # A learned stride continuation is by far the most likely concrete
        # value inside a bucket, so try it first. bucket_span returns an empty
        # span for out-of-range deltas, which fails the membership test safely.
        if self.deltas and self.deltas[-1] > 0:
            low, high = bucket_span(bucket_of(self.deltas[-1]))
            if low <= self.deltas[-1] <= high and request.lba + self.deltas[-1] >= end:
                candidates.append(request.lba + self.deltas[-1])
        for index in ranked:
            if len(candidates) >= 2:
                break
            if index == self.unk or index >= self.n_forward:
                continue  # unknown or backward-looking: not prefetchable
            low, high = bucket_span(index)
            for delta in (low, high):
                if len(candidates) >= 2:
                    break
                if delta > 0 and request.lba + delta >= end:
                    candidates.append(request.lba + delta)
        # Preserve order while dropping duplicates.
        return list(dict.fromkeys(candidates))


def load_predictor(path: str | Path) -> LSTMPrefetcher:
    torch = _torch()
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"LSTM artifact not found: {source}")

    artifact = torch.load(source, map_location="cpu", weights_only=True)
    if artifact.get("artifact_version") != ARTIFACT_VERSION:
        raise ValueError(
            "LSTM artifact was produced by an older version of the bucketed "
            "delta head; delete it and retrain with train-lstm")
    if (artifact.get("history") != HISTORY
            or artifact.get("max_delta") != MAX_DELTA
            or artifact.get("num_classes") != NUM_CLASSES):
        raise ValueError("unsupported LSTM artifact configuration")

    network = _network(NUM_CLASSES, int(artifact["hidden_size"]))
    try:
        network.load_state_dict(artifact["state_dict"])
    except (RuntimeError, KeyError) as exc:
        raise ValueError(f"corrupt LSTM artifact: {exc}") from exc
    network.eval()
    return LSTMPrefetcher(network, artifact["bucket_edges"])
