"""``learned_depth``: a trained read-ahead *depth* controller (D19).

D13 and D17 found that read-ahead depth, not the choice of policy, carries
nearly the whole measured effect, so this model predicts depth instead of the
next address. After request ``i`` it estimates, for each read-ahead segment
past the request's end,

    segment k = blocks [end_i + A_k, end_i + B_k),
    (A, B) in (0,8) (8,16) (16,32) (32,64) (64,128) (128,256)

whether any block of it is read by one of the next ``H`` requests, with
``H = capacity / mean request size`` of the training split. ``H`` is a rough
stand-in for LRU residency and is used **only** to build training labels. A
second head predicts the gap from this request's end to the next request's
start, over the training split's most frequent gaps, for a jump target
read-ahead cannot reach.

Policy (causal; it sees only requests up to and including ``i``):

* every ``decide_every`` requests the model is run on the last ``HISTORY``
  requests' features; depth becomes the end ``B_k`` of the deepest usable
  segment whose probability reaches the threshold, or 0;
* every request in between reads ahead at that depth (``guard.read_ahead``,
  so exactly ``depth`` blocks);
* at a decision request, when a gap threshold is set and the top gap's
  probability reaches it, ``JUMP_BLOCKS`` blocks at ``end + gap`` are added.

``action_cap`` limits the usable segments to those ending at or below it.
The comparison that matters is against a **validation-chosen fixed depth**
(VCF) over the same cap, never against depth 8 (plan §2.5, constraint 7):
the pilot capped the learned segments at 64 and VCF chose 64 on 7 of 8
traces, so both the 64-capped and the 256-capped comparisons are reported.

Thresholds are grid-selected on the validation segment only, through
``simulator.replay``; the test segment is touched once, by the caller.

Requirements: the module imports with the standard library only, so the
simulator can register the mode without torch. Training, labels and
inference need numpy and torch (``pip install -e .[gpu]``).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from .guard import read_ahead
from .trace import Request

#: Read-ahead segments past a request's end, in blocks, as half-open ranges.
SEGMENTS = ((0, 8), (8, 16), (16, 32), (32, 64), (64, 128), (128, 256))
#: The two action spaces the comparison is reported under (plan §2.5).
ACTION_CAPS = (64, 256)
#: Requests of history the network sees per decision.
HISTORY = 32
#: Size of the gap vocabulary; token ``GAP_VOCAB`` is "out of vocabulary".
GAP_VOCAB = 255
#: Features per request: signed-log start delta, signed-log gap from the
#: previous end, log2 size, is_contiguous, log inter-arrival time.
FEATURES = 5
#: Blocks prefetched at a predicted jump target.
JUMP_BLOCKS = 8
#: Requests between model decisions; amortises 0.5-2 ms of batch-1 inference.
DECIDE_EVERY = 32
ARTIFACT_VERSION = 1
#: Chronological split of each trace's prefix: train, validation, test.
SPLIT = (0.6, 0.2, 0.2)
#: "2M" in the plan means canonical (FIU page-merged) requests.
DEFAULT_PREFIX = 2_000_000
#: Bounds on the label horizon ``H``.
HORIZON_BOUNDS = (8, 512)

#: Validation grid: segment threshold x gap threshold (``None`` = no jumps).
#: The pilot stopped at 0.7; a 2M-request smoke run chose 0.7, the edge of
#: that grid, so it now reaches 0.9.
SEG_THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
GAP_THRESHOLDS = (None, 0.3, 0.6)
#: Validation-chosen fixed depth grid (VCF), clipped to the action cap.
VCF_DEPTHS = (0, 2, 4, 8, 16, 32, 64, 128, 256)
#: Request-relative depth ``k * size_blocks`` baseline.
RELSIZE_KS = (0.5, 1.0, 2.0)

_LOG_SCALE = math.log1p(2 ** 30)


def _np():
    try:
        import numpy
    except ImportError as exc:
        raise RuntimeError("learned_depth requires numpy: pip install -e .[gpu]") from exc
    return numpy


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("learned_depth requires PyTorch: pip install -e .[gpu]") from exc
    return torch


def configure_threads(num_threads: int = 4) -> int:
    """Cap torch's intra-op pool; returns the count set.

    torch defaults to one thread per core. On this shared 128-core host that
    made a tiny model take 859 s under load instead of 3.7 s (M15).
    """
    torch = _torch()
    count = max(1, min(int(num_threads), os.cpu_count() or 1))
    torch.set_num_threads(count)
    return count


def resolve_device(device: str = "auto") -> str:
    """``auto`` -> ``cuda`` when available, else ``cpu``; checks the others."""
    torch = _torch()
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available")
    if device not in ("cpu", "cuda"):
        raise ValueError(f"device must be auto, cpu or cuda, not {device!r}")
    return device


# ---------------------------------------------------------------- splits

def chronological_split(n: int, fractions=SPLIT) -> tuple[tuple[int, int], ...]:
    """Half-open ``(train, val, test)`` index ranges over ``n`` requests.

    Chronological, never shuffled: the model is trained on the past and
    scored on the future of the same trace.
    """
    if n < 3:
        raise ValueError("need at least 3 requests to split")
    if len(fractions) != 3 or any(f <= 0 for f in fractions) or \
            abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("fractions must be three positive values summing to 1")
    c1 = int(n * fractions[0])
    c2 = int(n * (fractions[0] + fractions[1]))
    return (0, c1), (c1, c2), (c2, n)


def overlaps(a: tuple[int, int], b: tuple[int, int | None]) -> bool:
    """Whether half-open ranges ``a`` and ``b`` share an index (``None`` = open end)."""
    end = math.inf if b[1] is None else b[1]
    return a[0] < end and b[0] < a[1]


def training_record_path(artifact: str | Path) -> Path:
    """The JSON sidecar next to an artifact: ``x.pt`` -> ``x.training.json``.

    The same convention ``main.py`` uses for the LSTM, so one artifact has
    exactly one record, written by whoever trained it, bound to the
    artifact by its SHA-256.
    """
    return Path(artifact).with_suffix(".training.json")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_training_record(artifact: str | Path, record: dict) -> Path:
    """Write the sidecar with the artifact's current digest; returns its path."""
    target = training_record_path(artifact)
    body = {**record, "artifact_sha256": sha256_file(artifact)}
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, target)
    return target


def read_training_record(artifact: str | Path) -> dict | None:
    """The sidecar, or ``None`` when absent, unreadable or not for this file."""
    try:
        record = json.loads(training_record_path(artifact).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if record.get("artifact_sha256") != sha256_file(artifact):
        return None
    return record


def scoring_conflict(record: dict | None, trace: str, start: int,
                     stop: int | None) -> str | None:
    """Why requests ``[start, stop)`` of ``trace`` may not score this artifact.

    Honesty constraint 14: an artifact is scored only on a segment or trace
    it was neither trained nor selected on. Returns ``None`` when the slice
    is clean, else the reason. A record that names no traces (the synthetic
    LSTM) cannot conflict with a real trace.
    """
    if record is None:
        return None
    for entry in record.get("traces", ()):
        if entry.get("trace") != trace:
            continue
        for part in ("train", "val"):
            span = entry.get(part)
            if span and overlaps((int(span[0]), int(span[1])), (start, stop)):
                return (f"artifact was {'trained' if part == 'train' else 'selected'} "
                        f"on {trace} requests [{span[0]}, {span[1]})")
    return None


# ---------------------------------------------------------------- prefix

@dataclass
class Prefix:
    """One trace's first ``n`` canonical requests as flat arrays."""

    trace: str
    path: str
    format: str
    ts_ms: object   # numpy float64
    lba: object     # numpy int64
    size: object    # numpy int64
    read: object    # numpy bool

    def __len__(self) -> int:
        return len(self.lba)

    def requests(self, start: int, stop: int) -> list[Request]:
        """``Request`` objects for ``[start, stop)``, for a replay."""
        stop = min(stop, len(self))
        ts = self.ts_ms[start:stop].tolist()
        lba = self.lba[start:stop].tolist()
        size = self.size[start:stop].tolist()
        read = self.read[start:stop].tolist()
        return [Request(t, b, s, "R" if r else "W", self.trace)
                for t, b, s, r in zip(ts, lba, size, read)]


def prefix_from_requests(requests: list[Request], trace: str = "in_memory") -> Prefix:
    """A :class:`Prefix` over requests already in memory (tests, small CSVs)."""
    np = _np()
    if not requests:
        raise ValueError("no requests")
    return Prefix(trace, "<memory>", "memory",
                  np.array([r.timestamp_ms for r in requests], dtype=np.float64),
                  np.array([r.lba for r in requests], dtype=np.int64),
                  np.array([r.size_blocks for r in requests], dtype=np.int64),
                  np.array([r.operation == "R" for r in requests], dtype=bool))


def load_prefix(path: str | Path, format: str = "oracle",
                n: int | None = DEFAULT_PREFIX, chunk: int = 250_000,
                **oracle_options) -> Prefix:
    """The first ``n`` canonical requests of a trace, chunk by chunk.

    Oracle traces go through ``trace.iter_oracle_chunks`` (FIU pages merged
    with the 1024-sector cap by default), so the model sees the same
    requests the simulator replays. ``Request`` objects cost ~430 B each, so
    each chunk is converted to arrays and dropped.
    """
    np = _np()
    from .corpora import trace_stem
    from .trace import iter_csv_chunks, load_csv

    path = Path(path)
    parts: list[tuple] = []

    def take(requests):
        parts.append((np.fromiter((r.timestamp_ms for r in requests), np.float64, len(requests)),
                      np.fromiter((r.lba for r in requests), np.int64, len(requests)),
                      np.fromiter((r.size_blocks for r in requests), np.int64, len(requests)),
                      np.fromiter((r.operation == "R" for r in requests), bool, len(requests))))

    if format == "oracle":
        from .trace import iter_oracle_chunks
        for requests in iter_oracle_chunks(path, chunk, max_requests=n, **oracle_options):
            take(requests)
    elif format == "revised":
        got = 0
        for requests in iter_csv_chunks(path, format, chunk):
            if n is not None:
                requests = requests[:n - got]
            take(requests)
            got += len(requests)
            if n is not None and got >= n:
                break
    else:
        requests = load_csv(path, format)
        take(requests[:n] if n is not None else requests)
    if not parts or not sum(len(p[1]) for p in parts):
        raise ValueError(f"{path}: no requests")
    ts, lba, size, read = (np.concatenate([p[i] for p in parts]) for i in range(4))
    return Prefix(trace_stem(path), str(path), format, ts, lba, size, read)


# -------------------------------------------------------------- features

def _slog(value: float) -> float:
    return math.copysign(math.log1p(abs(value)) / _LOG_SCALE, value) if value else 0.0


def request_features(previous: Request | None, request: Request) -> tuple[float, ...]:
    """The five per-request features, computed online from one predecessor.

    Mirrors :func:`step_features` exactly; the first request of a stream has
    zero delta, gap and inter-arrival time (so it counts as contiguous).
    """
    if previous is None:
        delta = gap = 0
        dt_s = 0.0
    else:
        delta = request.lba - previous.lba
        gap = request.lba - (previous.lba + previous.size_blocks)
        dt_s = max(request.timestamp_ms - previous.timestamp_ms, 0.0) / 1000.0
    return (_slog(delta), _slog(gap), math.log2(request.size_blocks) / 12.0,
            1.0 if gap == 0 else 0.0, math.log1p(dt_s) / 10.0)


def step_features(ts_ms, lba, size):
    """``(n, FEATURES)`` float32 features of one contiguous slice (vectorised).

    The slice's first request has no predecessor, as in the online path.
    """
    np = _np()
    lba = np.asarray(lba, dtype=np.int64)
    size = np.asarray(size, dtype=np.int64)
    end = lba + size
    delta = np.zeros_like(lba)
    gap = np.zeros_like(lba)
    dt = np.zeros(len(lba), dtype=np.float64)
    delta[1:] = lba[1:] - lba[:-1]
    gap[1:] = lba[1:] - end[:-1]
    dt[1:] = np.maximum(np.diff(np.asarray(ts_ms, dtype=np.float64)), 0.0) / 1000.0

    def slog(x):
        x = x.astype(np.float64)
        return np.sign(x) * np.log1p(np.abs(x)) / _LOG_SCALE

    columns = (slog(delta), slog(gap), np.log2(size.astype(np.float64)) / 12.0,
               (gap == 0).astype(np.float64), np.log1p(dt) / 10.0)
    return np.stack(columns, axis=1).astype(np.float32)


def label_horizon(size, capacity: int) -> int:
    """``H = capacity / mean request size``, clipped to :data:`HORIZON_BOUNDS`."""
    np = _np()
    mean = float(np.mean(size)) if len(size) else 1.0
    return int(np.clip(int(capacity / max(mean, 1.0)), *HORIZON_BOUNDS))


def seg_labels(lba, size, horizon: int, segments=SEGMENTS):
    """``(n, len(segments))`` float32: is segment k read within ``horizon``?

    Label ``[i, k]`` is 1 when some request ``j`` in ``i+1 .. i+horizon``
    overlaps blocks ``[end_i + A_k, end_i + B_k)``. Exact interval overlap,
    O(n * horizon); requests near the end of the array see a truncated
    future and so lean towards 0. Labels look ahead by construction: they
    are training targets, never policy input.
    """
    np = _np()
    lba = np.asarray(lba, dtype=np.int64)
    end = lba + np.asarray(size, dtype=np.int64)
    n = len(lba)
    labels = np.zeros((n, len(segments)), dtype=bool)
    for h in range(1, min(horizon, n - 1) + 1):
        start_rel = lba[h:] - end[:-h]
        end_rel = end[h:] - end[:-h]
        for k, (a, b) in enumerate(segments):
            labels[:-h, k] |= (start_rel < b) & (end_rel > a)
    return labels.astype(np.float32)


def gap_vocabulary(next_gaps, size: int = GAP_VOCAB):
    """The ``size`` most frequent gaps, ties broken by smaller value."""
    np = _np()
    values, counts = np.unique(np.asarray(next_gaps, dtype=np.int64), return_counts=True)
    order = np.lexsort((values, -counts))
    return values[order[:size]]


def gap_tokens(next_gaps, vocab):
    """Each gap's index in ``vocab``, or :data:`GAP_VOCAB` when it is not in it."""
    np = _np()
    lookup = {int(v): k for k, v in enumerate(vocab)}
    return np.fromiter((lookup.get(int(g), GAP_VOCAB) for g in next_gaps),
                       np.int64, len(next_gaps))


@dataclass
class Prepared:
    """Everything training needs for one trace, built once and reused per seed."""

    prefix: Prefix
    split: tuple[tuple[int, int], ...]
    horizon: int
    capacity: int
    segments: tuple[tuple[int, int], ...]
    features: object       # (n, FEATURES) float32, over the whole prefix
    labels: object         # (n, K) float32
    tokens: object         # (n,) int64, the next request's gap token
    vocab: object          # (<= GAP_VOCAB,) int64
    train_rows: object     # decision rows used for training
    val_rows: object       # decision rows used for early stopping
    prep_s: float = 0.0


def prepare(prefix: Prefix, capacity: int, segments=SEGMENTS,
            fractions=SPLIT) -> Prepared:
    """Split, label and tokenise one prefix.

    Training rows ``i`` need a full history inside the split and labels that
    stay inside it (``i + H < split end``), so no training target ever looks
    into the validation segment and no validation target into the test one.
    ``H`` and the gap vocabulary come from the training split alone.
    """
    np = _np()
    start = time.perf_counter()
    n = len(prefix)
    split = chronological_split(n, fractions)
    (_, c1), (_, c2), _ = split
    horizon = label_horizon(prefix.size[:c1], capacity)
    features = step_features(prefix.ts_ms, prefix.lba, prefix.size)
    labels = seg_labels(prefix.lba, prefix.size, horizon, segments)
    end = prefix.lba + prefix.size
    next_gaps = np.zeros(n, dtype=np.int64)
    next_gaps[:-1] = prefix.lba[1:] - end[:-1]
    vocab = gap_vocabulary(next_gaps[:max(c1 - 1, 0)])
    tokens = gap_tokens(next_gaps, vocab)
    train_rows = np.arange(HISTORY - 1, max(c1 - horizon - 1, HISTORY - 1))
    val_rows = np.arange(c1 + HISTORY - 1, max(c2 - horizon - 1, c1 + HISTORY - 1))
    if not len(train_rows) or not len(val_rows):
        raise ValueError(f"{prefix.trace}: {n} requests is too short for history "
                         f"{HISTORY} and label horizon {horizon}")
    return Prepared(prefix, split, horizon, capacity, tuple(map(tuple, segments)),
                    features, labels, tokens, vocab, train_rows, val_rows,
                    time.perf_counter() - start)


# --------------------------------------------------------------- network

def _network(segments: int, hidden: int = 128, layers: int = 2,
             vocab: int = GAP_VOCAB):
    torch = _torch()

    class DepthNet(torch.nn.Module):
        """Linear(5,h) -> ReLU -> LSTM(h,h,layers) -> segment and gap heads."""

        def __init__(self):
            super().__init__()
            self.inp = torch.nn.Linear(FEATURES, hidden)
            self.rnn = torch.nn.LSTM(hidden, hidden, num_layers=layers, batch_first=True)
            self.seg = torch.nn.Linear(hidden, segments)
            self.gap = torch.nn.Linear(hidden, vocab + 1)

        def forward(self, x):
            state, _ = self.rnn(torch.relu(self.inp(x)))
            last = state[:, -1]
            return self.seg(last), self.gap(last)

    return DepthNet()


def parameter_count(network) -> int:
    return sum(p.numel() for p in network.parameters())


class GpuSampler:
    """Samples ``nvidia-smi`` every ``interval`` seconds while a job runs.

    The GPU is shared with other tenants, so a training time means nothing
    without the memory and utilisation that were in use around it. Samples
    include this process's own use; nvidia-smi cannot separate it.
    """

    def __init__(self, interval: float = 10.0):
        self.interval = interval
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        try:
            from prefetch_common.provenance import gpu_info
        except ImportError:
            return
        info = gpu_info()
        if info is not None:
            self.samples.append({"t": round(time.time(), 1), "gpus": info})

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            self._sample()

    def __enter__(self):
        self._sample()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._sample()
        return False

    def summary(self) -> str | None:
        """One line: memory used and utilisation ranges over the samples."""
        used, util, total = [], [], None
        for sample in self.samples:
            gpus = sample["gpus"]
            if not isinstance(gpus, list) or not gpus:
                continue
            first = gpus[0]
            try:
                used.append(float(first["memory.used"]))
                util.append(float(first["utilization.gpu"]))
                total = first.get("memory.total")
            except (KeyError, ValueError):
                continue
        if not used:
            return None
        return (f"shared GPU over {len(used)} nvidia-smi samples (this job included): "
                f"memory used {min(used):.0f}-{max(used):.0f} of {total} MiB, "
                f"utilisation {min(util):.0f}-{max(util):.0f}%")


def _is_oom(exc: BaseException) -> bool:
    torch = _torch()
    oom = getattr(torch.cuda, "OutOfMemoryError", None)
    return (oom is not None and isinstance(exc, oom)) or "out of memory" in str(exc)


def fit(prepared: Prepared, *, seed: int = 0, epochs: int = 8,
        batch_size: int = 8192, lr: float = 2e-3, weight_decay: float = 1e-4,
        hidden: int = 128, layers: int = 2, device: str = "auto",
        patience: int | None = 2, num_threads: int = 4,
        max_oom_retries: int = 4, progress=None):
    """Train a ``DepthNet`` on ``prepared``; returns ``(network_on_cpu, info)``.

    AdamW, BCE on the segment heads plus cross-entropy on the gap head,
    bf16 autocast on CUDA, early stopping on validation loss (the best
    epoch's weights are kept; ``patience`` epochs without improvement end
    training). A CUDA out-of-memory error -- the GPU is shared -- restarts
    training from scratch at half the batch size, up to ``max_oom_retries``
    times, and is recorded in ``info``.
    """
    torch = _torch()
    configure_threads(num_threads)
    device = resolve_device(device)
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    retries = 0
    while True:
        try:
            network, info = _fit_once(prepared, seed, epochs, batch_size, lr,
                                      weight_decay, hidden, layers, device,
                                      patience, progress)
            break
        except RuntimeError as exc:
            if device != "cuda" or not _is_oom(exc) or retries >= max_oom_retries \
                    or batch_size < 2:
                raise
            retries += 1
            batch_size //= 2
            torch.cuda.empty_cache()
            if progress is not None:
                progress(f"CUDA OOM; retry {retries} at batch {batch_size}")
            time.sleep(5)
    info.update(oom_retries=retries, batch_size=batch_size)
    return network, info


def _fit_once(prepared, seed, epochs, batch_size, lr, weight_decay, hidden,
              layers, device, patience, progress):
    torch = _torch()
    torch.manual_seed(seed)
    network = _network(len(prepared.segments), hidden, layers).to(device)
    optimiser = torch.optim.AdamW(network.parameters(), lr=lr, weight_decay=weight_decay)
    bce = torch.nn.BCEWithLogitsLoss()
    ce = torch.nn.CrossEntropyLoss()
    feats = torch.from_numpy(prepared.features).to(device)
    labels = torch.from_numpy(prepared.labels).to(device)
    tokens = torch.from_numpy(prepared.tokens).to(device)
    train_rows = torch.from_numpy(prepared.train_rows).to(device)
    val_rows = torch.from_numpy(prepared.val_rows).to(device)
    offsets = torch.arange(-HISTORY + 1, 1, device=device)
    generator = torch.Generator(device=device).manual_seed(seed)
    autocast = (torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda"
                else torch.autocast("cpu", enabled=False))

    def loss_of(rows):
        with autocast:
            seg, gap = network(feats[rows[:, None] + offsets[None, :]])
        return bce(seg.float(), labels[rows]) + ce(gap.float(), tokens[rows])

    best = (math.inf, None, -1)
    history: list[float] = []
    stale = 0
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for epoch in range(epochs):
        network.train()
        order = train_rows[torch.randperm(len(train_rows), device=device,
                                          generator=generator)]
        for offset in range(0, len(order), batch_size):
            loss = loss_of(order[offset:offset + batch_size])
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
        network.eval()
        total = 0.0
        with torch.no_grad():
            for offset in range(0, len(val_rows), 16384):
                rows = val_rows[offset:offset + 16384]
                total += float(loss_of(rows)) * len(rows)
        val_loss = total / len(val_rows)
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
    if best[1] is None:
        # Every validation loss was NaN (e.g. a bf16 overflow): there is no
        # epoch to keep, and keeping the last one would hide the failure.
        raise RuntimeError(f"validation loss was never finite: {history}")
    cpu = _network(len(prepared.segments), hidden, layers)
    cpu.load_state_dict(best[1])
    cpu.eval()
    return cpu, {"device": device, "epochs": epochs, "epochs_run": len(history),
                 "best_epoch": best[2], "val_loss": history, "train_s": round(train_s, 3),
                 "train_examples": int(len(prepared.train_rows)),
                 "val_examples": int(len(prepared.val_rows)),
                 "hidden": hidden, "layers": layers, "seed": seed, "lr": lr,
                 "params": parameter_count(cpu)}


def decision_rows(n: int, decide_every: int = DECIDE_EVERY):
    """Segment-relative indices at which the policy runs the model."""
    np = _np()
    return np.arange(HISTORY - 1, n, decide_every)


def decision_probabilities(network, features, rows, batch: int = 4096):
    """Batched CPU inference at ``rows``: ``(seg_p, gap_token, gap_p)`` arrays."""
    np = _np()
    torch = _torch()
    feats = torch.from_numpy(np.ascontiguousarray(features))
    rows = torch.from_numpy(np.asarray(rows, dtype=np.int64))
    offsets = torch.arange(-HISTORY + 1, 1)
    seg_p, token, token_p = [], [], []
    with torch.no_grad():
        for offset in range(0, len(rows), batch):
            index = rows[offset:offset + batch]
            seg, gap = network(feats[index[:, None] + offsets[None, :]])
            probability, choice = torch.softmax(gap, dim=1).max(dim=1)
            seg_p.append(torch.sigmoid(seg))
            token.append(choice)
            token_p.append(probability)
    if not seg_p:
        return (np.zeros((0, network.seg.out_features), np.float32),
                np.zeros(0, np.int64), np.zeros(0, np.float32))
    return (torch.cat(seg_p).numpy(), torch.cat(token).numpy(),
            torch.cat(token_p).numpy())


# ---------------------------------------------------------------- policy

class _DepthPolicy:
    """The decision rule shared by the online and precomputed predictors.

    ``next_candidates`` is called once per request by the replay loop. At
    segment-relative index ``i >= HISTORY - 1`` with ``(i - HISTORY + 1) %
    decide_every == 0`` it asks :meth:`_probabilities` for the model output
    and re-decides; otherwise it keeps the last depth. The replay loop's
    first-window gate still decides whether anything is issued.
    """

    def __init__(self, segments, vocab, threshold: float,
                 gap_threshold: float | None, action_cap: int,
                 decide_every: int = DECIDE_EVERY):
        if decide_every < 1:
            raise ValueError("decide_every must be positive")
        ends = [b for _, b in segments]
        if action_cap not in ends:
            raise ValueError(f"action_cap {action_cap} is not a segment end in {ends}")
        self.segments = tuple(tuple(s) for s in segments)
        self.usable = [k for k, (_, b) in enumerate(self.segments) if b <= action_cap]
        self.vocab = [int(v) for v in vocab]
        self.threshold = float(threshold)
        self.gap_threshold = None if gap_threshold is None else float(gap_threshold)
        self.action_cap = action_cap
        self.decide_every = decide_every
        self.depth = 0
        self.seen = 0
        self.decisions = 0
        self.jumps = 0
        #: Wall time inside the model alone, so a per-decision inference
        #: cost excludes the per-request feature and read-ahead work that
        #: ``Metrics.inference_ms`` also contains.
        self.model_ms = 0.0

    def _observe(self, request: Request, index: int) -> None:
        pass

    def _probabilities(self, index: int):
        raise NotImplementedError

    def decide(self, seg_p, token: int, token_p: float) -> tuple[int, int | None]:
        """``(depth, jump_gap)`` from one model output."""
        depth = 0
        for k in reversed(self.usable):
            if seg_p[k] >= self.threshold:
                depth = self.segments[k][1]
                break
        jump = None
        if (self.gap_threshold is not None and token_p >= self.gap_threshold
                and token < len(self.vocab)):
            gap = self.vocab[token]
            if gap != 0 or depth == 0:
                jump = gap
        return depth, jump

    def next_candidates(self, request: Request) -> list[int]:
        index = self.seen
        self.seen += 1
        self._observe(request, index)
        jump = None
        if index >= HISTORY - 1 and (index - HISTORY + 1) % self.decide_every == 0:
            start = time.perf_counter()
            seg_p, token, token_p = self._probabilities(index)
            self.model_ms += (time.perf_counter() - start) * 1000
            self.decisions += 1
            self.depth, jump = self.decide(seg_p, int(token), float(token_p))
        if request.operation != "R":
            return []
        proposed = read_ahead(request, self.depth)
        if jump is not None:
            start = request.lba + request.size_blocks + jump
            if start >= 0:
                self.jumps += 1
                proposed.extend(range(start, start + JUMP_BLOCKS))
        return proposed


class DepthPrefetcher(_DepthPolicy):
    """The deployable predictor: features online, batch-1 CPU inference."""

    def __init__(self, network, segments, vocab, threshold, gap_threshold,
                 action_cap, decide_every=DECIDE_EVERY):
        super().__init__(segments, vocab, threshold, gap_threshold, action_cap,
                         decide_every)
        self.network = network
        self.window: deque = deque(maxlen=HISTORY)
        self.previous: Request | None = None

    def _observe(self, request, index):
        self.window.append(request_features(self.previous, request))
        self.previous = request

    def _probabilities(self, index):
        torch = _torch()
        x = torch.tensor([list(self.window)], dtype=torch.float32)
        with torch.no_grad():
            seg, gap = self.network(x)
            probability, choice = torch.softmax(gap, dim=1).max(dim=1)
        return torch.sigmoid(seg)[0].tolist(), int(choice[0]), float(probability[0])


class PrecomputedDepth(_DepthPolicy):
    """Same rule, reading model outputs computed in one batch beforehand.

    For grid selection, where the same segment is replayed under many
    thresholds. ``seg_p[r]`` is the output at segment index
    ``HISTORY - 1 + r * decide_every`` (see :func:`decision_rows`). With
    hindsight labels in place of ``seg_p`` it is the oracle for this action
    space.
    """

    def __init__(self, seg_p, tokens, token_p, segments, vocab, threshold,
                 gap_threshold, action_cap, decide_every=DECIDE_EVERY):
        super().__init__(segments, vocab, threshold, gap_threshold, action_cap,
                         decide_every)
        self.seg_p = seg_p
        self.tokens = tokens
        self.token_p = token_p

    def _probabilities(self, index):
        row = (index - HISTORY + 1) // self.decide_every
        return self.seg_p[row], self.tokens[row], self.token_p[row]


class RelativeDepth:
    """Baseline: read ahead ``k * size_blocks`` (rounded, at least 1, capped)."""

    def __init__(self, k: float, action_cap: int = max(ACTION_CAPS)):
        if k <= 0:
            raise ValueError("k must be positive")
        self.k = k
        self.action_cap = action_cap

    def next_candidates(self, request: Request) -> list[int]:
        depth = min(self.action_cap, max(1, int(self.k * request.size_blocks + 0.5)))
        return read_ahead(request, depth)


# ------------------------------------------------------------- artifacts

def save_artifact(path: str | Path, network, prepared: Prepared, info: dict,
                  thresholds: dict | None = None,
                  decide_every: int = DECIDE_EVERY) -> Path:
    """Save weights plus everything the policy needs to run them."""
    torch = _torch()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "artifact_version": ARTIFACT_VERSION,
        "kind": "learned_depth",
        "state_dict": network.state_dict(),
        "segments": [list(s) for s in prepared.segments],
        "hidden": info["hidden"], "layers": info["layers"],
        "history": HISTORY, "features": FEATURES, "gap_vocab": GAP_VOCAB,
        "vocab": [int(v) for v in prepared.vocab],
        "horizon": prepared.horizon, "capacity": prepared.capacity,
        "decide_every": decide_every,
        "thresholds": thresholds or {},
    }
    tmp = destination.with_name(destination.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, destination)
    return destination


def load_artifact(path: str | Path) -> dict:
    """Load and check an artifact; ``payload["network"]`` is ready to run."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"learned_depth artifact not found: {source}")
    torch = _torch()
    payload = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("artifact_version") != ARTIFACT_VERSION \
            or payload.get("kind") != "learned_depth":
        raise ValueError("learned_depth artifact is from another version; "
                         "delete it and retrain with train-depth")
    if payload.get("history") != HISTORY or payload.get("features") != FEATURES \
            or payload.get("gap_vocab") != GAP_VOCAB:
        raise ValueError("unsupported learned_depth artifact configuration; "
                         "retrain with train-depth")
    network = _network(len(payload["segments"]), int(payload["hidden"]),
                       int(payload["layers"]))
    try:
        network.load_state_dict(payload["state_dict"])
    except (RuntimeError, KeyError) as exc:
        raise ValueError(f"corrupt learned_depth artifact: {exc}") from exc
    network.eval()
    payload["network"] = network
    return payload


def load_prefetcher(path: str | Path, action_cap: int | None = None,
                    decide_every: int | None = None,
                    num_threads: int = 4) -> DepthPrefetcher:
    """A fresh :class:`DepthPrefetcher` from an artifact.

    ``action_cap`` defaults to the largest cap the artifact has
    validation-selected thresholds for. An artifact without selected
    thresholds is refused: thresholds are part of the method, and picking
    them anywhere but the validation segment would be test leakage.
    """
    payload = load_artifact(path)
    configure_threads(num_threads)
    chosen = payload.get("thresholds") or {}
    if not chosen:
        raise ValueError("learned_depth artifact has no validation-selected "
                         "thresholds; rerun train-depth")
    cap = action_cap if action_cap is not None else max(int(c) for c in chosen)
    if str(cap) not in chosen:
        raise ValueError(f"artifact has thresholds for caps {sorted(chosen)}, "
                         f"not {cap}")
    entry = chosen[str(cap)]
    return DepthPrefetcher(payload["network"], payload["segments"], payload["vocab"],
                           entry["seg"], entry["gap"], cap,
                           decide_every or int(payload["decide_every"]))


# ------------------------------------------------------- selection, tests

def _replay_row(requests, mode, capacity, **kwargs) -> dict:
    from .simulator import replay
    metrics, _ = replay(requests, capacity, 32, mode, **kwargs)
    return {"hit_ratio": metrics.hit_ratio, "precision": metrics.prefetch_precision,
            "issued": metrics.prefetches, "useful": metrics.useful_prefetches,
            "device_ios": metrics.device_ios, "inference_ms": metrics.inference_ms}


def _best(rows: list[tuple[object, dict]]):
    """Highest hit ratio; ties go to fewer prefetches, then grid order."""
    return max(enumerate(rows),
               key=lambda item: (item[1][1]["hit_ratio"], -item[1][1]["issued"], -item[0])
               )[1]


def validation_slice(prepared: Prepared, replay_cap: int | None) -> tuple[int, int]:
    _, (c1, c2), _ = prepared.split
    return c1, c2 if replay_cap is None else min(c2, c1 + replay_cap)


def scoring_slice(prepared: Prepared, replay_cap: int | None) -> tuple[int, int]:
    _, _, (c2, n) = prepared.split
    return c2, n if replay_cap is None else min(n, c2 + replay_cap)


def select_thresholds(network, prepared: Prepared, action_cap: int, *,
                      val_replay: int | None = 100_000,
                      decide_every: int = DECIDE_EVERY,
                      seg_grid=SEG_THRESHOLDS, gap_grid=GAP_THRESHOLDS):
    """Grid-select ``(seg, gap)`` thresholds on the validation slice only.

    Every candidate is scored by ``simulator.replay`` on requests
    ``[c1, c1 + val_replay)``; nothing at or past the validation split end
    is read. Returns ``(chosen, table)``: the chosen entry and every
    candidate's replay row.
    """
    start, stop = validation_slice(prepared, val_replay)
    prefix = prepared.prefix
    requests = prefix.requests(start, stop)
    features = step_features(prefix.ts_ms[start:stop], prefix.lba[start:stop],
                             prefix.size[start:stop])
    seg_p, tokens, token_p = decision_probabilities(
        network, features, decision_rows(len(requests), decide_every))
    table = []
    for seg in seg_grid:
        for gap in gap_grid:
            policy = PrecomputedDepth(seg_p, tokens, token_p, prepared.segments,
                                      prepared.vocab, seg, gap, action_cap, decide_every)
            row = _replay_row(requests, "learned_depth", prepared.capacity,
                              depth_model=policy)
            table.append(({"seg": seg, "gap": gap}, row))
    chosen, row = _best(table)
    return ({**chosen, "val_slice": [start, stop], "val_hit_ratio": row["hit_ratio"]},
            [{**c, **r} for c, r in table])


def select_fixed_depth(prepared: Prepared, action_cap: int, *,
                       val_replay: int | None = 100_000, depths=VCF_DEPTHS):
    """VCF: the fixed ``deep`` depth with the best validation hit ratio."""
    start, stop = validation_slice(prepared, val_replay)
    requests = prepared.prefix.requests(start, stop)
    table = []
    for depth in depths:
        if depth > action_cap:
            continue
        row = (_replay_row(requests, "deep", prepared.capacity, read_ahead_depth=depth)
               if depth else _replay_row(requests, "none", prepared.capacity))
        table.append((depth, row))
    depth, _ = _best(table)
    return depth, [{"depth": d, **r} for d, r in table]


def select_relative_depth(prepared: Prepared, action_cap: int, *,
                          val_replay: int | None = 100_000, ks=RELSIZE_KS):
    """The request-relative ``k`` with the best validation hit ratio."""
    start, stop = validation_slice(prepared, val_replay)
    requests = prepared.prefix.requests(start, stop)
    table = [(k, _replay_row(requests, "learned_depth", prepared.capacity,
                             depth_model=RelativeDepth(k, action_cap))) for k in ks]
    k, _ = _best(table)
    return k, [{"k": kk, **r} for kk, r in table]


def oracle_policy(prepared: Prepared, start: int, stop: int, action_cap: int,
                  decide_every: int = DECIDE_EVERY) -> PrecomputedDepth:
    """Hindsight labels in place of predictions, same actions and cadence.

    Not a bound on every policy: it is the best this decision rule can do
    with perfect segment labels at horizon ``H``.
    """
    np = _np()
    rows = decision_rows(stop - start, decide_every) + start
    labels = prepared.labels[rows]
    empty = np.zeros(len(rows), dtype=np.int64)
    return PrecomputedDepth(labels, empty, empty.astype(np.float32), prepared.segments,
                            prepared.vocab, 0.5, None, action_cap, decide_every)


def training_record(prepared: Prepared, info: dict, **extra) -> dict:
    """The sidecar body for an artifact trained on ``prepared``."""
    (a, b), (c, d), (e, f) = prepared.split
    return {
        "model": "learned_depth",
        "source": prepared.prefix.format,
        "traces": [{"trace": prepared.prefix.trace, "path": prepared.prefix.path,
                    "format": prepared.prefix.format, "n_requests": len(prepared.prefix),
                    "train": [a, b], "val": [c, d], "test": [e, f]}],
        "horizon": prepared.horizon, "capacity": prepared.capacity,
        "segments": [list(s) for s in prepared.segments],
        **info,
        **extra,
    }


@dataclass
class TrainResult:
    artifact: Path
    record: dict
    thresholds: dict
    validation: dict = field(default_factory=dict)


def train_and_select(prepared: Prepared, save_path: str | Path, *, seed: int = 0,
                     epochs: int = 8, batch_size: int = 8192, device: str = "auto",
                     action_caps=ACTION_CAPS, val_replay: int | None = 100_000,
                     decide_every: int = DECIDE_EVERY, num_threads: int = 4,
                     patience: int | None = 2, progress=None,
                     sampler: GpuSampler | None = None,
                     hidden: int = 128, layers: int = 2) -> TrainResult:
    """Train, select thresholds per action cap on validation, save, record.

    Writes the artifact and its ``.training.json`` sidecar. The test slice
    is not read here.
    """
    sampler = sampler or GpuSampler()
    with sampler:
        network, info = fit(prepared, seed=seed, epochs=epochs, batch_size=batch_size,
                            device=device, patience=patience, num_threads=num_threads,
                            progress=progress, hidden=hidden, layers=layers)
    info["gpu_contention"] = sampler.summary() if info["device"] == "cuda" else None
    thresholds, validation = {}, {}
    for cap in action_caps:
        chosen, table = select_thresholds(network, prepared, cap, val_replay=val_replay,
                                          decide_every=decide_every)
        thresholds[str(cap)] = chosen
        validation[str(cap)] = table
    artifact = save_artifact(save_path, network, prepared, info, thresholds, decide_every)
    record = training_record(prepared, info, thresholds=thresholds,
                             decide_every=decide_every, prep_s=round(prepared.prep_s, 3))
    write_training_record(artifact, record)
    return TrainResult(artifact, record, thresholds, validation)
