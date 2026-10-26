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
import time
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
    """Per-example lists; the reference for :func:`_example_tensors`."""
    examples: list[tuple[list[float], int]] = []
    for trace in traces:
        deltas = _read_deltas(trace)
        for i in range(HISTORY, len(deltas)):
            window = deltas[i - HISTORY:i]
            examples.append(([_encode(d) for d in window], bucket_of(deltas[i])))
    return examples


def _bucket_tensor(deltas):
    """Vectorised :func:`bucket_of` over an int64 tensor."""
    torch = _torch()
    edges = torch.tensor(BUCKET_EDGES, dtype=torch.int64)
    magnitude = deltas.abs()
    index = torch.searchsorted(edges, magnitude, right=False)
    classes = torch.where(deltas >= 0, index, len(BUCKET_EDGES) + index)
    return torch.where(magnitude > MAX_DELTA, torch.full_like(classes, UNK), classes)


def _encode_tensor(deltas):
    """Vectorised :func:`_encode`, computed in float64 then stored as float32."""
    torch = _torch()
    clamped = deltas.clamp(-MAX_DELTA, MAX_DELTA).to(torch.float64)
    return (torch.sign(clamped) * torch.log1p(clamped.abs())
            / math.log1p(MAX_DELTA)).to(torch.float32)


def _example_tensors(sequences):
    """``(inputs (N, HISTORY, 1), labels (N,))`` from read-``lba`` sequences.

    Same examples, in the same order, as :func:`_build_examples`, without a
    Python list per example: a 1.2M-example real-trace split as lists of
    floats needs several GB and minutes to build.
    """
    torch = _torch()
    inputs, labels = [], []
    for lbas in sequences:
        lbas = torch.as_tensor(lbas, dtype=torch.int64)
        if len(lbas) < HISTORY + 2:
            continue
        deltas = lbas[1:] - lbas[:-1]
        encoded = _encode_tensor(deltas)
        windows = encoded.unfold(0, HISTORY, 1)[:len(deltas) - HISTORY]
        inputs.append(windows)
        labels.append(_bucket_tensor(deltas[HISTORY:]))
    if not inputs:
        return (torch.zeros((0, HISTORY, 1), dtype=torch.float32),
                torch.zeros(0, dtype=torch.long))
    return torch.cat(inputs).unsqueeze(-1).contiguous(), torch.cat(labels).long()


def _read_lbas(trace: list[Request]) -> list[int]:
    return [req.lba for req in trace if req.operation == "R"]


def train_lstm(
    save_path: str | Path,
    traces: list[list[Request]] | None = None,
    epochs: int = 60,
    vocab_size: int = 0,
    seed: int = 42,
    *,
    device: str = "auto",
    num_threads: int = 4,
    val_fraction: float = 0.2,
    validation: list[list[Request]] | None = None,
    batch_size: int = 128,
    patience: int | None = None,
    record: dict | None = None,
    progress=None,
    _sequences=None,
) -> dict:
    """Train the offline next-delta LSTM and save a versioned artifact.

    ``vocab_size`` is accepted for backward compatibility with the older
    exact-delta head and is ignored: the bucketed head has a fixed size.

    Validation and early stopping: ``validation`` traces, when given, score
    every epoch and the best epoch's weights are saved. Otherwise the
    chronological tail ``val_fraction`` of each training trace is held out
    for that purpose (``0`` trains on everything and keeps the last epoch,
    the old behaviour). ``patience`` epochs without improvement stop early;
    ``None`` runs every epoch.

    ``device`` is ``auto`` (CUDA when available), ``cuda`` or ``cpu``.
    torch's intra-op pool is capped at ``num_threads`` (M15: the default of
    one thread per core took 859 s instead of 3.7 s on this shared host).
    A CUDA out-of-memory error, which happens on the shared GPU, restarts
    at half the batch size up to four times.

    ``record`` is written as the ``.training.json`` sidecar, bound to the
    artifact by its SHA-256 (the same file ``main.py`` writes for the
    synthetic artifact), so whoever trained an artifact describes it.
    """
    torch = _torch()
    from .depth_model import (GpuSampler, _is_oom, configure_threads,
                              resolve_device, write_training_record)
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if not 0 <= val_fraction < 1:
        raise ValueError("val_fraction must be in [0, 1)")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    configure_threads(num_threads)
    device = resolve_device(device)
    if _sequences is not None:
        train_sequences, val_sequences = _sequences
    else:
        if traces is None:
            traces = [transition_trace(seed + i, windows_per_class=8)[0] for i in range(4)]
        lbas = [_read_lbas(trace) for trace in traces]
        if validation is not None:
            train_sequences = lbas
            val_sequences = [_read_lbas(trace) for trace in validation]
        elif val_fraction > 0:
            cuts = [len(seq) - int(len(seq) * val_fraction) for seq in lbas]
            train_sequences = [seq[:cut] for seq, cut in zip(lbas, cuts)]
            val_sequences = [seq[cut:] for seq, cut in zip(lbas, cuts)]
        else:
            train_sequences, val_sequences = lbas, []
    inputs, labels = _example_tensors(train_sequences)
    val_inputs, val_labels = _example_tensors(val_sequences)
    if not len(labels):
        raise ValueError("need at least 18 read requests in training traces")

    retries = 0
    sampler = GpuSampler()
    with sampler:
        while True:
            try:
                network, info = _fit_lstm(inputs, labels, val_inputs, val_labels,
                                          epochs, batch_size, seed, device,
                                          patience, progress)
                break
            except RuntimeError as exc:
                if device != "cuda" or not _is_oom(exc) or retries >= 4 or batch_size < 2:
                    raise
                retries += 1
                batch_size //= 2
                torch.cuda.empty_cache()
                time.sleep(5)

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
    result = {
        "examples": int(len(labels)),
        "val_examples": int(len(val_labels)),
        "classes": NUM_CLASSES,
        "epochs": epochs,
        "batch_size": batch_size,
        "oom_retries": retries,
        "gpu_contention": sampler.summary() if device == "cuda" else None,
        **info,
    }
    if record is not None:
        write_training_record(destination, {**record, "model": "lstm", **result})
    result["gpu_samples"] = sampler.samples
    return result


def _fit_lstm(inputs, labels, val_inputs, val_labels, epochs, batch_size,
              seed, device, patience, progress):
    torch = _torch()
    torch.manual_seed(seed)
    network = _network().to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=0.003)
    loss_fn = torch.nn.CrossEntropyLoss()
    x, y = inputs.to(device), labels.to(device)
    vx, vy = val_inputs.to(device), val_labels.to(device)
    generator = torch.Generator().manual_seed(seed)
    best = (math.inf, None, -1)
    history: list[float] = []
    stale = 0
    start = time.perf_counter()
    for epoch in range(epochs):
        network.train()
        # Shuffle every epoch: fixed-order minibatches let adjacent deltas
        # stay correlated within a batch and measurably hurt convergence.
        order = torch.randperm(len(y), generator=generator).to(device)
        for offset in range(0, len(y), batch_size):
            index = order[offset:offset + batch_size]
            optimizer.zero_grad()
            loss = loss_fn(network(x[index]), y[index])
            loss.backward()
            optimizer.step()
        if not len(vy):
            continue
        network.eval()
        total = 0.0
        with torch.no_grad():
            for offset in range(0, len(vy), 65536):
                out = network(vx[offset:offset + 65536])
                total += float(loss_fn(out, vy[offset:offset + 65536])) * len(out)
        val_loss = total / len(vy)
        history.append(round(val_loss, 5))
        if progress is not None:
            progress(f"epoch {epoch}: val loss {val_loss:.4f}")
        if val_loss < best[0]:
            best = (val_loss, {k: v.detach().to("cpu").clone()
                               for k, v in network.state_dict().items()}, epoch)
            stale = 0
        else:
            stale += 1
            if patience is not None and stale >= patience:
                break
    if device == "cuda":
        torch.cuda.synchronize()
    train_s = time.perf_counter() - start
    cpu = _network()
    if best[1] is not None:
        cpu.load_state_dict(best[1])
    else:
        cpu.load_state_dict({k: v.to("cpu") for k, v in network.state_dict().items()})
    cpu.eval()
    return cpu, {"device": device, "epochs_run": len(history) or epochs,
                 "best_epoch": best[2] if best[1] is not None else epochs - 1,
                 "val_loss": history, "train_s": round(train_s, 3), "seed": seed}


def train_lstm_on_trace(save_path: str | Path, path: str | Path, *,
                        format: str = "oracle", n: int | None = 2_000_000,
                        epochs: int = 8, batch_size: int = 4096,
                        seed: int = 42, device: str = "auto",
                        num_threads: int = 4, patience: int | None = 2,
                        progress=None, prefix=None) -> dict:
    """Train on one real trace with the chronological 60/20/20 split.

    Train on the first 60% of the first ``n`` canonical requests, stop early
    on the next 20%, and leave the last 20% untouched for scoring. The split
    is recorded in the sidecar, which ``sweep`` and the E2 runner use to
    refuse any scoring slice that overlaps training or validation.
    """
    from .depth_model import chronological_split, load_prefix
    prefix = prefix or load_prefix(path, format, n)
    split = chronological_split(len(prefix))
    (a, b), (c, d), (e, f) = split
    reads = prefix.lba[prefix.read]
    # Read-only positions of each split, so a write never becomes a delta.
    index = prefix.read.cumsum()
    cut1, cut2 = int(index[b - 1]), int(index[d - 1])
    record = {
        "source": format,
        "traces": [{"trace": prefix.trace, "path": prefix.path, "format": format,
                    "n_requests": len(prefix), "train": [a, b], "val": [c, d],
                    "test": [e, f]}],
    }
    result = train_lstm(save_path, epochs=epochs, seed=seed, device=device,
                        num_threads=num_threads, batch_size=batch_size,
                        patience=patience, record=record, progress=progress,
                        _sequences=([reads[:cut1]], [reads[cut1:cut2]]))
    result["split"] = [list(s) for s in split]
    result["trace"] = prefix.trace
    return result


def bucket_accuracy(network, lbas, reference_lbas=None) -> dict:
    """Top-1 bucket accuracy on a read sequence, with two references.

    ``majority`` always predicts the most common target bucket of
    ``reference_lbas`` (pass the training reads), or of ``lbas`` itself when
    none is given, which flatters it; ``repeat`` predicts the bucket of the
    previous delta. This is delta-bucket prediction, not workload
    classification, so it is never a classifier accuracy.
    """
    torch = _torch()
    inputs, labels = _example_tensors([lbas])
    if not len(labels):
        return {"examples": 0, "top1": None, "majority": None, "repeat": None}
    with torch.no_grad():
        predicted = torch.cat([network(inputs[i:i + 65536]).argmax(1)
                               for i in range(0, len(labels), 65536)])
    lbas = torch.as_tensor(lbas, dtype=torch.int64)
    deltas = lbas[1:] - lbas[:-1]
    previous = _bucket_tensor(deltas[HISTORY - 1:HISTORY - 1 + len(labels)])
    reference = (labels if reference_lbas is None
                 else _example_tensors([reference_lbas])[1])
    majority = int(torch.bincount(reference if len(reference) else labels).argmax())
    return {"examples": int(len(labels)),
            "top1": float((predicted == labels).double().mean()),
            "majority": float((labels == majority).double().mean()),
            "repeat": float((previous == labels).double().mean())}


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


def load_predictor(path: str | Path, num_threads: int = 4) -> LSTMPrefetcher:
    # Check the file before importing torch, so a missing artifact reports
    # itself the same way with or without the [lstm] extra installed.
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"LSTM artifact not found: {source}")
    torch = _torch()
    from .depth_model import configure_threads
    # Batch-1 inference on every request: the default one-thread-per-core
    # pool only adds contention on a shared host (M15).
    configure_threads(num_threads)

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
