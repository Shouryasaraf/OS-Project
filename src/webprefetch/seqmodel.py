"""GPU next-view sequence model (GRU4Rec-style, optional SASRec) for the web study.

The model reads the last ``K`` views of the causal session prefix plus three
context features of the current view, and outputs a full softmax over the
training vocabulary plus an END class ("the session stops here"). Its item
probabilities are therefore a normalised next-view distribution: after the
current item, the session window and the END/pad/OOV classes are zeroed, the
returned probabilities sum to at most 1.

Training protocol (chronological, as plan section 3.2 fixes it):

- ``validation``: train on split 0 plus the first 90% (by view time) of
  split 1, early-stop on MRR@8 over the eligible views of the last 10% of
  split 1, evaluate on split 2 (where policies are selected).
- ``final``: train on splits 0-2 for a fixed number of epochs and evaluate
  once on split 3.

The vocabulary comes from the training splits only (0 = pad, 1 = OOV,
2 = END, items from 3). Replay is CPU-only and exact: :meth:`SequenceScorer.
precompute` scores every view of the requested splits in one batched pass and
returns a :class:`RankTable` keyed by ``(session_id, position)``, which is
itself a ``score(row, index)`` scorer for :mod:`webprefetch.simulator`.

Artifacts hold no pickle: the ``state_dict`` tensors go to ``weights.npz``
(numpy, loaded with ``allow_pickle=False``), the item vocabulary to
``vocab.npy`` and everything else to ``config.json``, which also pins the
SHA-256 of both array files and an ``ARTIFACT_VERSION``.

The GPU is shared with other tenants: CUDA out-of-memory errors are retried
after a pause (:func:`oom_retry`), and ``nvidia-smi`` is sampled during
training (:class:`GpuSampler`) so every timing carries its contention.

Entry point (``webprefetch/cli.py`` belongs to the benchmark)::

    python -m webprefetch.seqmodel train --protocol validation --seed 0
    python -m webprefetch.seqmodel experiment --seeds 0,1,2,3,4

``experiment`` (E5, via ``reproduce.py web --gpu``) writes
``results/web/seq_metrics.csv`` and its provenance to
``results/web/seq/provenance.json`` (``results/web/provenance.json`` is the
E4 benchmark's), with the ``nvidia-smi`` samples in its ``extra``.

torch is imported lazily, so the window, vocabulary and metric code (and an
"is an artifact there" check) work without it.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import sys
import threading
import time
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .paths import MODELS, PROCESSED

K = 10                  # history window; the p99 session has 8 views
TOP_N = 20              # ranked list handed to the policies (they read at most 8)
PAD, OOV, END = 0, 1, 2
FIRST_ITEM = 3
N_CONTEXT = 3
ARTIFACT_VERSION = 1
MODEL_KINDS = ("gru", "sasrec")
PROTOCOLS = ("validation", "final")
# "history" excludes the current item and every item in the K-view session
# window (they are already in the session cache); "current" excludes only
# the current item, as CandidateIndex.candidates does.
MASKS = ("history", "current")
EARLY_STOP_QUANTILE = 0.9
SEQ_ROOT = MODELS / "seq"
EVAL_CAPACITIES = (8, 32)
EVAL_COSTS = (0.0, 0.01, 0.05, 0.25)
FIXED_DEPTHS = (0, 1, 2, 4, 8)
ADAPTIVE_PENALTIES = (0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2)
GATE_THRESHOLDS = (0.01, 0.02, 0.05, 0.1)
VIEW_COLUMNS = ("timestamp", "session_id", "position", "itemid", "next_item",
                "eligible", "split", "gap_ms", "cart_count")


class ArtifactError(RuntimeError):
    """A sequence-model artifact is missing, stale or does not match its config."""


class SeqModelUnavailable(RuntimeError):
    """torch is not installed, so no sequence model can be trained or scored."""


def _torch():
    try:
        import torch
    except ImportError as error:  # the [gpu] extra is optional
        raise SeqModelUnavailable("the sequence model needs torch (pip install -e .[gpu])") from error
    return torch


def _log(message: str) -> None:
    print(f"[seqmodel {time.strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


# ------------------------------------------------------------------ windows

def build_vocab(views: pd.DataFrame, splits) -> np.ndarray:
    """Sorted item ids viewed in ``splits``; item ``vocab[j]`` gets class ``FIRST_ITEM + j``."""
    selected = views.loc[views.split.isin(list(splits)), "itemid"]
    if selected.empty:
        raise ValueError(f"no views in training splits {tuple(splits)}")
    return np.unique(selected.to_numpy().astype(np.int64))


def encode(vocab: np.ndarray, items) -> np.ndarray:
    """Class ids for ``items``; anything outside the vocabulary is OOV."""
    items = np.asarray(items, dtype=np.int64)
    position = np.searchsorted(vocab, items)
    inside = position < len(vocab)
    found = np.zeros(items.shape, dtype=bool)
    found[inside] = vocab[position[inside]] == items[inside]
    return np.where(found, position + FIRST_ITEM, OOV).astype(np.int64)


def decode(vocab: np.ndarray, ids) -> np.ndarray:
    """Item ids for class ids; non-item classes (and -1 padding) become -1."""
    ids = np.asarray(ids, dtype=np.int64)
    ok = ids >= FIRST_ITEM
    out = np.full(ids.shape, -1, dtype=np.int64)
    out[ok] = vocab[ids[ok] - FIRST_ITEM]
    return out


def context_features(gap_ms, cart_count, position) -> np.ndarray:
    """The 3 context inputs of the current view: gap, cart activity, depth."""
    gap = np.clip(np.asarray(gap_ms, dtype=np.float64), 0, None)
    return np.stack([np.log1p(gap / 1000) / 10,
                     (np.asarray(cart_count) > 0).astype(np.float64),
                     np.log1p(np.asarray(position, dtype=np.float64)) / 5], 1).astype(np.float32)


def novel_next(views: pd.DataFrame) -> np.ndarray:
    """Per view (in ``views`` order): the next item was not viewed earlier in the session.

    The simulator's novel metrics use the same definition (a session cache
    could not already hold the next item). False for non-eligible views.
    """
    first = (views.groupby(["session_id", "itemid"], sort=False).position.min()
             .rename("first_position"))
    nxt = pd.DataFrame({"session_id": views.session_id.to_numpy(),
                        "itemid": views.next_item.to_numpy()})
    joined = nxt.join(first, on=["session_id", "itemid"])
    position = views.position.to_numpy()
    # The next item is viewed at position + 1, so it is novel iff that is
    # where it first appears.
    novel = joined.first_position.to_numpy() == position + 1
    return novel & views.eligible.to_numpy().astype(bool)


@dataclass
class Windows:
    """Model inputs for every view, sorted by (session_id, position).

    ``history[:, -1]`` is the current item; earlier columns hold the
    preceding views of the same session (0 = pad). ``target`` is the next
    item's class, END on the last view of a session, OOV when the next item
    is outside the vocabulary.
    """
    frame: pd.DataFrame
    history: np.ndarray
    context: np.ndarray
    target: np.ndarray
    eligible: np.ndarray
    novel: np.ndarray
    k: int

    def __len__(self) -> int:
        return len(self.frame)

    @property
    def current(self) -> np.ndarray:
        return self.history[:, -1]

    @property
    def keys(self) -> np.ndarray:
        return _keys(self.frame.session_id.to_numpy(), self.frame.position.to_numpy())


def _keys(session_id, position) -> np.ndarray:
    position = np.asarray(position, dtype=np.int64)
    if position.size and (position.min() < 0 or position.max() >= 1 << 16):
        raise ValueError("session positions must lie in [0, 65536)")
    return (np.asarray(session_id, dtype=np.int64) << 16) | position


def build_windows(views: pd.DataFrame, vocab: np.ndarray, k: int = K) -> Windows:
    """The last ``k`` views of each causal session prefix plus 3 context features."""
    missing = [c for c in VIEW_COLUMNS if c not in views]
    if missing:
        raise ValueError(f"prepared views lack columns {missing}")
    frame = views.sort_values(["session_id", "position"], kind="mergesort").reset_index(drop=True)
    sessions = frame.session_id.to_numpy()
    ids = encode(vocab, frame.itemid.to_numpy())
    history = np.zeros((len(frame), k), dtype=np.int64)
    for lag in range(k):
        shifted = np.roll(ids, lag)
        same = np.roll(sessions, lag) == sessions
        same[:lag] = False
        history[:, k - 1 - lag] = np.where(same, shifted, PAD)
    eligible = frame.eligible.to_numpy().astype(bool)
    target = np.where(eligible, encode(vocab, np.where(eligible, frame.next_item.to_numpy(), 0)),
                      END).astype(np.int64)
    context = context_features(frame.gap_ms.to_numpy(), frame.cart_count.to_numpy(),
                               frame.position.to_numpy())
    return Windows(frame, history, context, target, eligible, novel_next(frame), k)


def protocol_rows(windows: Windows, protocol: str) -> dict:
    """Training, early-stopping and evaluation rows (indices into ``windows``)."""
    if protocol not in PROTOCOLS:
        raise ValueError(f"protocol must be one of {PROTOCOLS}")
    split = windows.frame.split.to_numpy()
    ts = windows.frame.timestamp.to_numpy()
    learnable = windows.target != OOV
    if protocol == "validation":
        in_one = split == 1
        if not in_one.any():
            raise ValueError("the validation protocol needs views in split 1")
        cut = float(np.quantile(ts[in_one], EARLY_STOP_QUANTILE))
        train = (split == 0) | (in_one & (ts < cut))
        early = in_one & (ts >= cut) & windows.eligible
        evaluate, train_splits, eval_split = split == 2, (0, 1), 2
        description = (f"split 0 + split 1 before ts {int(cut)} "
                       f"(first {EARLY_STOP_QUANTILE:.0%}); early stop on the rest of split 1")
    else:
        train = split <= 2
        early = np.zeros(len(split), dtype=bool)
        evaluate, train_splits, eval_split = split == 3, (0, 1, 2), 3
        description = "splits 0-2; fixed epochs, no early stopping"
    return {"train": np.flatnonzero(train & learnable), "early": np.flatnonzero(early),
            "evaluate": np.flatnonzero(evaluate), "train_splits": train_splits,
            "eval_split": eval_split, "train_range": description}


# ------------------------------------------------------------------ metrics

def ranking_metrics(top_items: np.ndarray, next_item, novel=None, k: int = 8) -> dict:
    """hit@1, MRR@8 and coverage@8 over eligible rows, as the simulator counts them.

    ``top_items`` is (rows, >= k) item ids, best first, padded with -1;
    ``next_item`` the true next item per row. ``novel`` (optional, bool per
    row) adds ``novel_hit_at_1`` and ``novel_recall_at_8`` over novel rows.
    """
    target = np.asarray(next_item, dtype=np.int64)
    top = np.asarray(top_items, dtype=np.int64)
    # reshape(0, -1) is ambiguous, so an empty selection keeps an explicit width.
    top = (top.reshape(len(target), -1) if top.size else np.empty((len(target), 0), np.int64))[:, :k]
    if top.shape[1] < k:
        top = np.pad(top, ((0, 0), (0, k - top.shape[1])), constant_values=-1)
    match = (top == target[:, None]) & (target[:, None] >= 0)
    found = match.any(1)
    rank = match.argmax(1) + 1
    reciprocal = np.where(found, 1.0 / rank, 0.0)
    n = len(target)
    out = {"rows": int(n),
           "hit_at_1": float(match[:, 0].sum() / n) if n else 0.0,
           "mrr_at_8": float(reciprocal.sum() / n) if n else 0.0,
           "coverage_at_8": float(found.sum() / n) if n else 0.0}
    if novel is not None:
        novel = np.asarray(novel, dtype=bool)
        m = int(novel.sum())
        out.update({"novel_rows": m,
                    "novel_hit_at_1": float(match[novel, 0].sum() / m) if m else 0.0,
                    "novel_recall_at_8": float(found[novel].sum() / m) if m else 0.0})
    return out


def ranked_array(lists, k: int = 8) -> np.ndarray:
    """Pad scorer outputs (lists of (item, prob)) into a (rows, k) item array."""
    out = np.full((len(lists), k), -1, dtype=np.int64)
    for n, ranked in enumerate(lists):
        items = [int(item) for item, _ in ranked[:k]]
        out[n, :len(items)] = items
    return out


def pivot_validation_sample(views: pd.DataFrame, n: int = 10000) -> pd.DataFrame:
    """The 10k evenly spaced split-2 views ``model.train_ranker`` validates on.

    ``views`` must be in prepared (timestamp) order, as ``load_views`` returns it.
    """
    validation = views.loc[views.split.eq(2)]
    if len(validation) > n:
        validation = validation.iloc[np.linspace(0, len(validation) - 1, n, dtype=np.int64)]
    return validation


def scorer_metrics(rows: pd.DataFrame, scorer, index, novel=None) -> dict:
    """:func:`ranking_metrics` for any ``score(row, index)`` scorer over ``rows``.

    Non-eligible rows are scored (a stateful scorer sees every view) but not
    counted, exactly as ``train_ranker`` and the replay count them.
    """
    lists, eligible = [], rows.eligible.to_numpy().astype(bool)
    for row in rows.itertuples(index=False):
        lists.append(scorer.score(row, index))
    top = ranked_array(lists)
    novel = None if novel is None else np.asarray(novel, dtype=bool)[eligible]
    return ranking_metrics(top[eligible], rows.next_item.to_numpy()[eligible], novel)


# ------------------------------------------------------------------ models

_CLASSES: dict = {}


def _model_classes() -> dict:
    """GRURec and SASRec, defined on first use so importing needs no torch."""
    if _CLASSES:
        return _CLASSES
    torch = _torch()
    nn = torch.nn

    class GRURec(nn.Module):
        """Embedding -> GRU over the K-view window; last state plus context -> softmax."""

        def __init__(self, vocab_size: int, d: int = 128, h: int = 256, k: int = K):
            super().__init__()
            self.emb = nn.Embedding(vocab_size, d, padding_idx=PAD)
            self.rnn = nn.GRU(d, h, batch_first=True)
            self.ctx = nn.Linear(N_CONTEXT, h)
            self.out = nn.Linear(h, vocab_size)
            self.drop = nn.Dropout(0.2)

        def forward(self, history, context):
            states, _ = self.rnn(self.drop(self.emb(history)))
            return self.out(self.drop(states[:, -1] + self.ctx(context)))

    class SASRec(nn.Module):
        """2-layer causal transformer over the window, tied output embedding."""

        def __init__(self, vocab_size: int, d: int = 128, h: int = 256, k: int = K,
                     layers: int = 2, heads: int = 2):
            super().__init__()
            self.emb = nn.Embedding(vocab_size, d, padding_idx=PAD)
            self.pos = nn.Embedding(k, d)
            layer = nn.TransformerEncoderLayer(d, heads, h, 0.2, batch_first=True,
                                               norm_first=True)
            self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
            self.ctx = nn.Linear(N_CONTEXT, d)
            self.norm = nn.LayerNorm(d)
            self.register_buffer("causal", torch.triu(torch.ones(k, k, dtype=torch.bool), 1))

        def forward(self, history, context):
            # No key-padding mask: with left padding and a causal mask a fully
            # masked pad row yields NaN that leaks through 0 * NaN. Pad slots
            # carry a zero embedding plus position, which the model learns.
            x = self.emb(history) + self.pos.weight[None]
            states = self.enc(x, mask=self.causal)
            return self.norm(states[:, -1] + self.ctx(context)) @ self.emb.weight.T

    _CLASSES.update(gru=GRURec, sasrec=SASRec)
    return _CLASSES


def make_model(kind: str, vocab_size: int, d: int = 128, h: int = 256, k: int = K):
    if kind not in MODEL_KINDS:
        raise ValueError(f"model must be one of {MODEL_KINDS}")
    return _model_classes()[kind](vocab_size, d=d, h=h, k=k)


def resolve_device(name: str = "auto"):
    torch = _torch()
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise SeqModelUnavailable("CUDA was requested but torch.cuda.is_available() is False")
    return torch.device(name)


# ------------------------------------------------------------- shared GPU

def is_oom(error: BaseException) -> bool:
    torch = sys.modules.get("torch")
    oom = getattr(getattr(torch, "cuda", None), "OutOfMemoryError", None) if torch else None
    if oom is not None and isinstance(error, oom):
        return True
    return isinstance(error, RuntimeError) and "out of memory" in str(error).lower()


def oom_retry(fn, *, retries: int = 6, wait_s: float = 30.0, on_oom=None, sleep=time.sleep):
    """Call ``fn()``; on a CUDA OOM, free the cache, wait and call it again.

    Another tenant's allocation spike is the usual cause on the shared GPU,
    so the same work is retried unchanged. ``on_oom(attempt, error)`` runs
    before each pause (it may shrink a batch); the last error is re-raised
    after ``retries`` retries. Any other exception propagates at once.
    """
    for attempt in range(retries + 1):
        try:
            return fn()
        except Exception as error:  # noqa: BLE001 - re-raised unless it is an OOM
            if not is_oom(error) or attempt == retries:
                raise
            torch = sys.modules.get("torch")
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
            if on_oom is not None:
                on_oom(attempt, error)
            _log(f"CUDA out of memory (attempt {attempt + 1}/{retries}); retrying in {wait_s:g} s")
            sleep(wait_s)


class GpuSampler:
    """Samples ``nvidia-smi`` in a thread; a context manager.

    Memory in use includes this process. ``summary()`` is the contention note
    stored next to every GPU timing.
    """

    def __init__(self, interval_s: float = 10.0, query=None):
        if query is None:
            from prefetch_common.provenance import gpu_info as query
        self.interval_s, self.query = interval_s, query
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._thread = None
        self._t0 = time.monotonic()

    def sample(self, label: str = "") -> None:
        info = self.query()
        self.samples.append({"t_s": round(time.monotonic() - self._t0, 1),
                             "label": label, "gpus": info})

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            self.sample()

    def __enter__(self):
        self.sample("start")
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=30)
        self.sample("end")
        return False

    def summary(self, since: int = 0) -> str | None:
        """Utilisation and memory over samples ``[since:]``; None without nvidia-smi."""
        util, used, total = [], [], []
        for record in self.samples[since:]:
            gpus = record["gpus"]
            if not isinstance(gpus, list):
                continue
            for gpu in gpus:
                try:
                    util.append(float(gpu["utilization.gpu"]))
                    used.append(float(gpu["memory.used"]))
                    total.append(float(gpu["memory.total"]))
                except (KeyError, TypeError, ValueError):
                    continue
        if not util:
            return None
        return (f"shared GPU, nvidia-smi n={len(util)}: util mean {sum(util) / len(util):.0f}% "
                f"max {max(util):.0f}%; memory used max {max(used):.0f} of {max(total):.0f} MiB "
                "(includes this process)")


# ------------------------------------------------------------------ ranking

def _rank_batch(net, history, context, mask: str, top_n: int):
    """(class ids, probabilities) of the top ``top_n`` admissible items."""
    torch = _torch()
    probs = torch.softmax(net(history, context).float(), dim=1)
    probs[:, :FIRST_ITEM] = 0.0
    probs.scatter_(1, history[:, -1:], 0.0)
    if mask == "history":
        probs.scatter_(1, history, 0.0)
    values, ids = probs.topk(min(top_n, probs.shape[1]), dim=1)
    return ids, values


def _tidy(ids: np.ndarray, probs: np.ndarray, vocab: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Decode, order each row by (-probability, item id) and pad zero-probability slots with -1.

    The fixed tie order makes a batched pass and a one-row pass agree.
    """
    items = decode(vocab, ids)
    probs = probs.astype(np.float32)
    dead = (probs <= 0) | (items < 0)
    probs = np.where(dead, 0.0, probs).astype(np.float32)
    items = np.where(dead, np.iinfo(np.int64).max, items)
    order = np.lexsort((items, -probs), axis=-1)
    items = np.take_along_axis(items, order, 1)
    probs = np.take_along_axis(probs, order, 1)
    items[probs <= 0] = -1
    return items, probs


def rank_rows(net, windows: Windows, rows, vocab: np.ndarray, *, device, mask: str = "history",
              top_n: int = TOP_N, batch_rows: int = 8192) -> tuple[np.ndarray, np.ndarray]:
    """Item ids and probabilities (rows, top_n) for ``windows`` rows, in fp32."""
    torch = _torch()
    rows = np.asarray(rows, dtype=np.int64)
    out_items = np.full((len(rows), top_n), -1, dtype=np.int64)
    out_probs = np.zeros((len(rows), top_n), dtype=np.float32)
    state = {"batch": int(batch_rows)}

    def shrink(attempt, error):
        state["batch"] = max(256, state["batch"] // 2)

    net.eval()
    start = 0
    with torch.no_grad():
        while start < len(rows):
            def chunk():
                part = rows[start:start + state["batch"]]
                h = torch.from_numpy(windows.history[part]).to(device)
                c = torch.from_numpy(windows.context[part]).to(device)
                ids, values = _rank_batch(net, h, c, mask, top_n)
                return part, ids.cpu().numpy(), values.cpu().numpy()
            part, ids, values = oom_retry(chunk, on_oom=shrink)
            width = ids.shape[1]
            items, probs = _tidy(ids, values, vocab)
            out_items[start:start + len(part), :width] = items
            out_probs[start:start + len(part), :width] = probs
            start += len(part)
    return out_items, out_probs


class RankTable(Mapping):
    """``{(session_id, position): [(item, probability), ...]}``, stored as arrays.

    It is also a scorer: ``score(row, index)`` returns the row's list, so a
    replay reads precomputed GPU rankings on the CPU. A row that was not
    precomputed raises ``KeyError`` instead of silently scoring nothing.
    """

    def __init__(self, keys: np.ndarray, items: np.ndarray, probs: np.ndarray):
        if not (len(keys) == len(items) == len(probs)):
            raise ValueError("keys, items and probs must have the same length")
        self.items, self.probs = items, probs
        self._row = dict(zip(np.asarray(keys, dtype=np.int64).tolist(), range(len(keys))))
        if len(self._row) != len(keys):
            raise ValueError("duplicate (session_id, position) keys")

    def _ranked(self, n: int) -> list[tuple[int, float]]:
        return [(int(i), float(p)) for i, p in zip(self.items[n], self.probs[n]) if i >= 0]

    def __getitem__(self, key) -> list[tuple[int, float]]:
        session_id, position = key
        packed = (int(session_id) << 16) | int(position)
        if packed not in self._row:
            raise KeyError(key)
        return self._ranked(self._row[packed])

    def __iter__(self):
        for packed in self._row:
            yield (packed >> 16, packed & 0xFFFF)

    def __len__(self) -> int:
        return len(self._row)

    def score(self, row, index=None) -> list[tuple[int, float]]:
        try:
            return self[(int(row.session_id), int(row.position))]
        except KeyError:
            raise KeyError(f"view (session {int(row.session_id)}, position {int(row.position)}) "
                           "was not precomputed; precompute its split first") from None


# ------------------------------------------------------------------ training

def train_model(windows: Windows, train_rows, early_rows, *, vocab_size: int, kind: str = "gru",
                seed: int = 0, epochs: int = 20, lr: float = 1e-3, batch: int = 2048,
                device=None, bf16: bool = True, mask: str = "history", d: int = 128,
                h: int = 256, vocab: np.ndarray | None = None, log=_log):
    """Train with AdamW; keep the epoch with the best early-stop MRR@8.

    Without early rows the last epoch is kept. Returns ``(net, info)``.
    bf16 autocast applies on CUDA only; ranking is always fp32.
    """
    torch = _torch()
    device = device if device is not None else resolve_device("auto")
    train_rows = np.asarray(train_rows, dtype=np.int64)
    early_rows = np.asarray(early_rows, dtype=np.int64)
    if len(train_rows) == 0:
        raise ValueError("no training rows")
    if len(early_rows) and vocab is None:
        raise ValueError("early stopping needs the vocabulary to decode rankings")
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    net = oom_retry(lambda: make_model(kind, vocab_size, d=d, h=h, k=windows.k).to(device))
    optimiser = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.0)
    loss_fn = torch.nn.CrossEntropyLoss()
    history = oom_retry(lambda: torch.from_numpy(windows.history).to(device))
    context = oom_retry(lambda: torch.from_numpy(windows.context).to(device))
    target = oom_retry(lambda: torch.from_numpy(windows.target).to(device))
    use_bf16 = bf16 and device.type == "cuda"
    early_next = windows.frame.next_item.to_numpy()[early_rows]
    best = (-1.0, None, -1)
    epoch_log = []
    oom_events = [0]

    def counted(attempt, error):
        oom_events[0] += 1
        optimiser.zero_grad(set_to_none=True)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for epoch in range(epochs):
        net.train()
        order = torch.from_numpy(train_rows)[torch.randperm(len(train_rows), generator=generator)]
        order = order.to(device)
        total, batches = 0.0, 0
        for start in range(0, len(order), batch):
            rows = order[start:start + batch]

            def step():
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
                    logits = net(history[rows], context[rows])
                loss = loss_fn(logits.float(), target[rows])
                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                optimiser.step()
                return float(loss.detach())
            total += oom_retry(step, on_oom=counted)
            batches += 1
        record = {"epoch": epoch, "train_loss": round(total / batches, 4),
                  "elapsed_s": round(time.perf_counter() - started, 1)}
        if len(early_rows):
            items, _ = rank_rows(net, windows, early_rows, vocab, device=device, mask=mask, top_n=8)
            metrics = ranking_metrics(items, early_next)
            record.update({key: round(metrics[key], 5)
                           for key in ("hit_at_1", "mrr_at_8", "coverage_at_8")})
            if metrics["mrr_at_8"] > best[0]:
                best = (metrics["mrr_at_8"],
                        {key: value.detach().cpu().clone() for key, value in net.state_dict().items()},
                        epoch)
        epoch_log.append(record)
        log(f"{kind} epoch {epoch}: {record}")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    train_s = time.perf_counter() - started
    if best[1] is not None:
        net.load_state_dict(best[1])
    net.eval()
    info = {"train_s": round(train_s, 2), "epochs": epochs,
            "best_epoch": best[2] if best[1] is not None else None,
            "best_early_mrr_at_8": round(best[0], 6) if best[1] is not None else None,
            "epoch_log": epoch_log, "train_rows": int(len(train_rows)),
            "early_rows": int(len(early_rows)), "oom_retries": oom_events[0],
            "parameters": int(sum(p.numel() for p in net.parameters())),
            "bf16": use_bf16}
    return net, info


# ------------------------------------------------------------------ artifacts

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_dir(model: str = "gru", protocol: str = "final", seed: int = 0,
                 root: str | Path = SEQ_ROOT) -> Path:
    """Where ``experiment`` keeps the artifact for (model, protocol, seed)."""
    return Path(root) / f"{model}_{protocol}_s{seed}"


def save_artifact(directory: str | Path, net, vocab: np.ndarray, config: dict) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    arrays = {name: tensor.detach().cpu().numpy() for name, tensor in net.state_dict().items()}
    np.savez(directory / "weights.npz", **arrays)
    np.save(directory / "vocab.npy", np.asarray(vocab, dtype=np.int64), allow_pickle=False)
    config = {**config, "artifact_version": ARTIFACT_VERSION,
              "files": {name: _sha256(directory / name) for name in ("weights.npz", "vocab.npy")}}
    tmp = directory / "config.json.tmp"
    tmp.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, directory / "config.json")
    return directory


def read_config(directory: str | Path) -> dict:
    """The artifact's config after the version and file-hash checks."""
    directory = Path(directory)
    path = directory / "config.json"
    if not path.is_file():
        raise ArtifactError(f"no sequence-model artifact at {directory} (missing config.json)")
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("artifact_version") != ARTIFACT_VERSION:
        raise ArtifactError(f"{directory}: artifact version {config.get('artifact_version')} "
                            f"!= {ARTIFACT_VERSION}; retrain with `python -m webprefetch.seqmodel train`")
    for name, digest in config.get("files", {}).items():
        file = directory / name
        if not file.is_file() or _sha256(file) != digest:
            raise ArtifactError(f"{directory}: {name} is missing or does not match config.json")
    if set(config.get("files", {})) != {"weights.npz", "vocab.npy"}:
        raise ArtifactError(f"{directory}: config.json does not pin weights.npz and vocab.npy")
    return config


def load_artifact(directory: str | Path, device="cpu"):
    """``(net, vocab, config)``; nothing is unpickled."""
    torch = _torch()
    directory = Path(directory)
    config = read_config(directory)
    vocab = np.load(directory / "vocab.npy", allow_pickle=False)
    if len(vocab) + FIRST_ITEM != config["vocab_size"]:
        raise ArtifactError(f"{directory}: vocab.npy has {len(vocab)} items, config says "
                            f"{config['vocab_size'] - FIRST_ITEM}")
    device = resolve_device(device) if isinstance(device, str) else device
    net = make_model(config["model"], config["vocab_size"], d=config["d"], h=config["h"],
                     k=config["k"])
    with np.load(directory / "weights.npz", allow_pickle=False) as weights:
        state = {name: torch.from_numpy(weights[name].copy()) for name in weights.files}
    try:
        net.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise ArtifactError(f"{directory}: weights do not fit the configured model: {error}") from None
    return net.to(device).eval(), vocab, config


# ------------------------------------------------------------------ scorer

class SequenceScorer:
    """The trained model as a web-study scorer.

    :meth:`precompute` is the replay path: one batched pass, then a
    :class:`RankTable`. :meth:`score` is live, causal scoring (one view at a
    time on the scorer's device), used to check the table: it needs every
    view of a session, in position order, starting at position 0.
    """

    def __init__(self, net, vocab: np.ndarray, config: dict, device=None):
        torch = _torch()
        self.net = net.eval()
        self.vocab = np.asarray(vocab, dtype=np.int64)
        self.config = dict(config)
        self.k = int(config.get("k", K))
        self.mask = config.get("mask", "history")
        if self.mask not in MASKS:
            raise ArtifactError(f"unknown mask {self.mask!r}")
        self.top_n = int(config.get("top_n", TOP_N))
        self.device = device if device is not None else next(net.parameters()).device
        self.name = f"seq_{config.get('model', 'gru')}"
        self._torch = torch
        self._live: dict[int, tuple[int, deque]] = {}

    @classmethod
    def load(cls, path: str | Path, device="auto") -> "SequenceScorer":
        """Load an artifact; ``device="auto"`` is CUDA when available, else CPU."""
        net, vocab, config = load_artifact(path, device)
        return cls(net, vocab, config)

    def precompute(self, views: pd.DataFrame, splits=None, *, top_n: int | None = None,
                   batch_rows: int = 8192) -> RankTable:
        """Rank every view of ``splits`` (all splits when None) in one batched pass.

        Sessions never cross a split boundary (``partition_sessions`` cuts
        them), so windows built from the selected splits alone are complete.
        """
        selected = views if splits is None else views.loc[views.split.isin(list(splits))]
        windows = build_windows(selected, self.vocab, self.k)
        return self.rank_windows(windows, top_n=top_n, batch_rows=batch_rows)

    def rank_windows(self, windows: Windows, *, top_n: int | None = None,
                     batch_rows: int = 8192) -> RankTable:
        """:meth:`precompute` on windows already built; records the GPU time.

        ``last_rank_s`` is the wall time of the ranking pass alone (window
        building on the CPU is excluded).
        """
        torch, cuda = self._torch, getattr(self.device, "type", "") == "cuda"
        if cuda:
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        items, probs = rank_rows(self.net, windows, np.arange(len(windows)), self.vocab,
                                 device=self.device, mask=self.mask,
                                 top_n=top_n or self.top_n, batch_rows=batch_rows)
        if cuda:
            torch.cuda.synchronize(self.device)
        self.last_rank_s = time.perf_counter() - started
        return RankTable(windows.keys, items, probs)

    def reset(self) -> None:
        self._live.clear()

    def score(self, row, index=None) -> list[tuple[int, float]]:
        torch = self._torch
        session_id, position = int(row.session_id), int(row.position)
        if position == 0:
            seen, window = 0, deque([PAD] * self.k, maxlen=self.k)
        else:
            if session_id not in self._live or self._live[session_id][0] != position:
                raise ValueError(f"live scoring of session {session_id} position {position} "
                                 "needs its earlier views first, in order")
            seen, window = self._live[session_id]
        window.append(int(encode(self.vocab, [int(row.itemid)])[0]))
        if bool(row.eligible):
            self._live[session_id] = (seen + 1, window)
        else:  # the last view of its session
            self._live.pop(session_id, None)
        history = torch.tensor([list(window)], dtype=torch.int64, device=self.device)
        context = torch.from_numpy(context_features([row.gap_ms], [row.cart_count],
                                                    [position])).to(self.device)
        with torch.no_grad():
            ids, values = _rank_batch(self.net, history, context, self.mask, self.top_n)
        items, probs = _tidy(ids.cpu().numpy(), values.cpu().numpy(), self.vocab)
        return [(int(i), float(p)) for i, p in zip(items[0], probs[0]) if i >= 0]


# ------------------------------------------------------------------ commands

def _now_utc() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def train_artifact(views: pd.DataFrame, *, protocol: str, model: str = "gru", seed: int = 0,
                   epochs: int = 20, lr: float = 1e-3, batch: int = 2048, device="auto",
                   bf16: bool = True, mask: str = "history", d: int = 128, h: int = 256,
                   out: str | Path | None = None, views_path: str | Path | None = None,
                   log=_log) -> tuple[SequenceScorer, dict, Windows, dict]:
    """Train one (model, protocol, seed) artifact and save it.

    Returns ``(scorer, config, windows, rows)``; ``rows`` is
    :func:`protocol_rows`' output for the same windows.
    """
    if mask not in MASKS:
        raise ValueError(f"mask must be one of {MASKS}")
    device = resolve_device(device) if isinstance(device, str) else device
    train_splits = (0, 1) if protocol == "validation" else (0, 1, 2)
    vocab = build_vocab(views, train_splits)
    windows = build_windows(views, vocab)
    rows = protocol_rows(windows, protocol)
    log(f"{model} {protocol} seed {seed}: V={len(vocab) + FIRST_ITEM:,} "
        f"train={len(rows['train']):,} early={len(rows['early']):,} on {device}")
    net, info = train_model(windows, rows["train"], rows["early"],
                            vocab_size=len(vocab) + FIRST_ITEM, kind=model, seed=seed,
                            epochs=epochs, lr=lr, batch=batch, device=device, bf16=bf16,
                            mask=mask, d=d, h=h, vocab=vocab, log=log)
    import torch
    config = {"model": model, "protocol": protocol, "seed": seed, "d": d, "h": h, "k": K,
              "n_context": N_CONTEXT, "vocab_size": len(vocab) + FIRST_ITEM, "mask": mask,
              "top_n": TOP_N, "train_splits": list(train_splits),
              "eval_split": rows["eval_split"], "train_range": rows["train_range"],
              "lr": lr, "batch": batch, "weight_decay": 0.0, "optimizer": "AdamW",
              "early_stop_metric": "mrr_at_8" if len(rows["early"]) else None,
              "device": str(device), "torch": torch.__version__, "created_utc": _now_utc(),
              "views_sha256": _sha256(Path(views_path)) if views_path else None,
              **{key: value for key, value in info.items()}}
    if out is not None:
        save_artifact(out, net, vocab, config)
        log(f"artifact written to {out}")
    return SequenceScorer(net, vocab, config, device=device), config, windows, rows


def evaluate_ranking(scorer: SequenceScorer, views: pd.DataFrame, split: int) -> tuple[dict, RankTable, dict]:
    """Ranking metrics over every eligible view of ``split``, the table, and costs.

    The cost dict holds ``infer_us_per_request`` (batched ranking pass per
    view) and ``batch1_us`` (one single-view forward plus top-k, the online
    cost), both on the scorer's device.
    """
    selected = views.loc[views.split.eq(split)]
    windows = build_windows(selected, scorer.vocab, scorer.k)
    table = scorer.rank_windows(windows)
    costs = {"infer_us_per_request": scorer.last_rank_s / max(1, len(windows)) * 1e6,
             "batch1_us": batch1_latency_us(scorer, windows, np.flatnonzero(windows.eligible))}
    eligible = windows.eligible
    top = table.items[eligible][:, :8]
    metrics = ranking_metrics(top, windows.frame.next_item.to_numpy()[eligible],
                              windows.novel[eligible])
    return metrics, table, costs


def batch1_latency_us(scorer: SequenceScorer, windows: Windows, rows, repeats: int = 200) -> float:
    """Mean µs for one single-view forward pass plus top-k (the online cost)."""
    torch = scorer._torch
    if not len(rows):
        return float("nan")
    row = np.asarray(rows[:1], dtype=np.int64)
    h = torch.from_numpy(windows.history[row]).to(scorer.device)
    c = torch.from_numpy(windows.context[row]).to(scorer.device)
    cuda = getattr(scorer.device, "type", "") == "cuda"
    with torch.no_grad():
        for _ in range(10):
            _rank_batch(scorer.net, h, c, scorer.mask, 8)
        if cuda:
            torch.cuda.synchronize(scorer.device)
        started = time.perf_counter()
        for _ in range(repeats):
            ids, _ = _rank_batch(scorer.net, h, c, scorer.mask, 8)
            ids.cpu()
        if cuda:
            torch.cuda.synchronize(scorer.device)
    return (time.perf_counter() - started) / repeats * 1e6


def eval_policies():
    from .cachecore import Policy
    return ([Policy(f"fixed{k}", "fixed", k) for k in FIXED_DEPTHS]
            + [Policy(f"adaptive_{p:g}", "adaptive", p) for p in ADAPTIVE_PENALTIES]
            + [Policy(f"adaptive_ca_{p:g}", "adaptive_ca", p) for p in ADAPTIVE_PENALTIES]
            + [Policy(f"gate_{t:g}", "gate", t) for t in GATE_THRESHOLDS])


def replay_table(views: pd.DataFrame, table: RankTable, split: int, *,
                 capacities=EVAL_CAPACITIES, workers: int = 1, max_views: int = 0):
    """Cache replay of ``split`` on the precomputed table: {capacity: (rows, extra)}."""
    from .parallel import replay_parallel
    selected = views.loc[views.split.eq(split)]
    return replay_parallel(selected, None, table, eval_policies(), tuple(capacities), workers,
                           max_views=max_views, with_sessions=False)


def metric_rows(base: dict, ranking: dict, replayed: dict, costs=EVAL_COSTS) -> list[dict]:
    """``web/seq_metrics.csv`` rows: one per (capacity, policy, cost)."""
    from prefetch_common.utility import U
    out = []
    for capacity, (rows, extra) in sorted(replayed.items()):
        for row in rows:
            for cost in costs:
                out.append({**base, "status": "ok",
                            "hit_at_1": ranking["hit_at_1"], "mrr_at_8": ranking["mrr_at_8"],
                            "novel_hit_at_1": ranking["novel_hit_at_1"],
                            "novel_recall_at_8": ranking["novel_recall_at_8"],
                            "capacity": int(capacity), "policy": row["policy"], "cost": float(cost),
                            "hit_ratio": row["hit_ratio"], "issued_per_view": row["issued_per_view"],
                            "utility": U(row["hits"], row["issued"], row["views"], cost)})
    return out


def skipped_rows(base: dict, reason: str, capacities=EVAL_CAPACITIES, costs=EVAL_COSTS) -> list[dict]:
    """Rows for a run that could not complete: ``status="skipped: ..."``, metrics None."""
    empty = dict.fromkeys(("hit_at_1", "mrr_at_8", "novel_hit_at_1", "novel_recall_at_8",
                           "hit_ratio", "issued_per_view", "utility", "train_s",
                           "infer_us_per_request"))
    return [{**base, **empty, "status": f"skipped: {reason}", "capacity": int(capacity),
             "policy": policy.name, "cost": float(cost)}
            for capacity in capacities for policy in eval_policies() for cost in costs]


def select_on_validation(rows: list[dict]) -> list[dict]:
    """Per (model, seed, capacity, cost): the validation-best policy and its final-test row."""
    ok = [row for row in rows if row["status"] == "ok"]
    out = []
    groups = {}
    for row in ok:
        groups.setdefault((row["model"], row["seed"], row["capacity"], row["cost"]), []).append(row)
    for (model, seed, capacity, cost), group in sorted(groups.items()):
        validation = [row for row in group if row["protocol"] == "validation"]
        if not validation:
            continue
        best = max(validation, key=lambda row: (row["utility"], -row["issued_per_view"]))
        final = {row["policy"]: row for row in group if row["protocol"] == "final"}
        chosen = final.get(best["policy"])
        none = final.get("fixed0")
        out.append({"model": model, "seed": seed, "capacity": capacity, "cost": cost,
                    "selected_policy": best["policy"], "validation_utility": best["utility"],
                    "final_utility": chosen["utility"] if chosen else None,
                    "final_fixed0_utility": none["utility"] if none else None})
    return out


def _load(path: str | Path) -> pd.DataFrame:
    from .data import load_views
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"prepared views missing: {path} (run `python -m webprefetch prepare`)")
    return load_views(path)


def command_train(args) -> dict:
    views = _load(args.views)
    out = args.out or artifact_dir(args.model, args.protocol, args.seed)
    scorer, config, windows, rows = train_artifact(
        views, protocol=args.protocol, model=args.model, seed=args.seed, epochs=args.epochs,
        lr=args.lr, batch=args.batch, device=args.device, bf16=not args.no_bf16,
        mask=args.mask, out=out, views_path=args.views)
    summary = {"artifact": str(out), "model": args.model, "protocol": args.protocol,
               "seed": args.seed, "train_s": config["train_s"], "best_epoch": config["best_epoch"],
               "train_rows": config["train_rows"], "parameters": config["parameters"],
               "oom_retries": config["oom_retries"]}
    if not args.skip_eval:
        metrics, table, costs = evaluate_ranking(scorer, views, rows["eval_split"])
        summary[f"split{rows['eval_split']}_all_eligible"] = metrics
        summary.update({key: round(value, 3) for key, value in costs.items()})
        if rows["eval_split"] == 2:
            sample = pivot_validation_sample(views)
            summary["split2_pivot_10k_sample"] = scorer_metrics(sample, table, None)
    (Path(out) / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def command_experiment(args) -> int:
    """E5: seeds x protocols, ranking metrics and cache replay, into seq_metrics.csv."""
    from prefetch_common import provenance, schemas
    start = provenance.snapshot()
    views = _load(args.views)
    seeds = [int(s) for s in str(args.seeds).split(",") if s != ""]
    protocols = [p for p in args.protocols.split(",") if p]
    unknown = sorted(set(protocols) - set(PROTOCOLS))
    if unknown:
        raise SystemExit(f"unknown protocols {unknown}")
    device = resolve_device(args.device)
    device_name = (_torch().cuda.get_device_name(device) if device.type == "cuda" else "cpu")
    views_sha = _sha256(Path(args.views))
    results, runs = [], []
    with GpuSampler(args.sample_interval_s) as sampler:
        for seed in seeds:
            for protocol in protocols:
                directory = artifact_dir(args.model, protocol, seed, args.models)
                split = 2 if protocol == "validation" else 3
                first_sample = len(sampler.samples)
                base = {"model": args.model, "protocol": protocol, "seed": seed,
                        "split": str(split), "epochs": args.epochs, "device": device_name}
                try:
                    reused = False
                    if not args.retrain and (directory / "config.json").is_file():
                        try:
                            config = read_config(directory)
                            same = (config.get("epochs"), config.get("mask"), config.get("model"),
                                    config.get("seed"), config.get("protocol"),
                                    config.get("views_sha256"), config.get("lr"),
                                    config.get("batch"))
                            if same == (args.epochs, args.mask, args.model, seed, protocol,
                                        views_sha, args.lr, args.batch):
                                scorer, reused = SequenceScorer.load(directory, device), True
                            else:
                                _log(f"retraining: {directory} was trained with other settings")
                        except ArtifactError as error:
                            _log(f"retraining: {error}")
                    if reused:
                        config = scorer.config
                        _log(f"reusing {directory}")
                    else:
                        scorer, config, _, _ = train_artifact(
                            views, protocol=protocol, model=args.model, seed=seed,
                            epochs=args.epochs, lr=args.lr, batch=args.batch, device=device,
                            mask=args.mask, out=directory, views_path=args.views)
                    ranking, table, costs = evaluate_ranking(scorer, views, split)
                    replayed = replay_table(views, table, split, workers=args.workers,
                                            max_views=args.max_views)
                except Exception as error:  # noqa: BLE001 - a failed run is a skipped row
                    if not is_oom(error):
                        raise
                    reason = f"CUDA out of memory after retries ({type(error).__name__})"
                    _log(f"{protocol} seed {seed}: {reason}")
                    rows = skipped_rows({**base, "train_range": "not trained",
                                         "gpu_contention": sampler.summary(first_sample)}, reason)
                    results += rows
                    runs.append({**base, "status": rows[0]["status"]})
                    continue
                # A reused artifact keeps the train_s of the run that trained
                # it; gpu_contention covers this run only.
                base.update({"train_range": config["train_range"], "train_s": config["train_s"],
                             "infer_us_per_request": round(costs["infer_us_per_request"], 3),
                             "gpu_contention": sampler.summary(first_sample)})
                results += metric_rows(base, ranking, replayed)
                runs.append({**base, "status": "ok", "reused_artifact": reused,
                             "artifact": str(directory), "best_epoch": config.get("best_epoch"),
                             "epoch_log": config.get("epoch_log"), "ranking": ranking,
                             "batch1_us": round(costs["batch1_us"], 1),
                             "oom_retries": config.get("oom_retries")})
                _log(f"{protocol} seed {seed}: {json.dumps(ranking)}")
    out = Path(args.out)
    target = schemas.write_rows("web/seq_metrics.csv", results, out / "seq_metrics.csv")
    selection = select_on_validation(results)
    print(json.dumps({"rows": len(results), "written": str(target), "selection": selection},
                     indent=2))
    if not args.no_provenance:
        command = [sys.executable, "-m", "webprefetch.seqmodel", *args.argv]
        provenance.write_provenance(
            out / "seq", command, inputs=[args.views], study="web", start=start, gpu="auto",
            notes=("E5 GPU sequence model. seq_metrics.csv sits in the parent directory; "
                   "its SHA-256 is in extra.outputs_in_parent. GPU timings are on a shared "
                   "GPU: see extra.gpu_samples and the gpu_contention column. With "
                   f"mask={args.mask!r}: the 'history' mask never ranks an item already in "
                   "the K-view window, so a next view that repeats one is never a ranking "
                   "hit and hit_at_1 / mrr_at_8 are not comparable to an unmasked "
                   "ranker's; compare novel_hit_at_1 and novel_recall_at_8 instead."),
            extra={"outputs_in_parent": {target.name: _sha256(target)},
                   "gpu_samples": sampler.samples, "runs": runs,
                   "validation_selection": selection,
                   "settings": {"seeds": seeds, "protocols": protocols, "epochs": args.epochs,
                                "max_views": args.max_views, "workers": args.workers,
                                "mask": args.mask, "model": args.model,
                                "capacities": list(EVAL_CAPACITIES), "costs": list(EVAL_COSTS)}})
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m webprefetch.seqmodel",
                                     description=__doc__.strip().splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    def common(sub):
        sub.add_argument("--views", type=Path, default=PROCESSED / "views.csv.gz")
        sub.add_argument("--model", choices=MODEL_KINDS, default="gru")
        sub.add_argument("--epochs", type=int, default=20)
        sub.add_argument("--lr", type=float, default=1e-3)
        sub.add_argument("--batch", type=int, default=2048)
        sub.add_argument("--device", default="auto", help="auto | cuda | cuda:N | cpu")
        sub.add_argument("--mask", choices=MASKS, default="history")

    train = commands.add_parser("train", help="train and save one artifact")
    common(train)
    train.add_argument("--protocol", choices=PROTOCOLS, default="validation")
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--out", type=Path, default=None,
                       help="artifact directory (default models/web/seq/<model>_<protocol>_s<seed>)")
    train.add_argument("--no-bf16", action="store_true", help="train in fp32 on CUDA too")
    train.add_argument("--skip-eval", action="store_true", help="do not rank the evaluation split")

    experiment = commands.add_parser("experiment", help="E5: seeds x protocols into seq_metrics.csv")
    common(experiment)
    experiment.add_argument("--seeds", default="0,1,2,3,4")
    experiment.add_argument("--protocols", default="validation,final")
    experiment.add_argument("--models", type=Path, default=SEQ_ROOT, help="artifact root")
    experiment.add_argument("--out", type=Path, default=Path("results/web"))
    experiment.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    experiment.add_argument("--max-views", type=int, default=0,
                            help="replay budget per split (0 = the whole split)")
    experiment.add_argument("--retrain", action="store_true",
                            help="ignore existing artifacts with the same settings")
    experiment.add_argument("--sample-interval-s", type=float, default=10.0)
    experiment.add_argument("--no-provenance", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)
    args.argv = argv
    if args.command == "train":
        command_train(args)
        return 0
    return command_experiment(args)


if __name__ == "__main__":
    sys.exit(main())
