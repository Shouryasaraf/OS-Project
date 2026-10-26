"""Scorer registry: every candidate ranker the web benchmark can replay.

A scorer is any object with ``score(row, index) -> [(item, probability), ...]``
sorted best first. Each registry entry also fixes the index it runs on and
the replay order that index needs:

==============  ============================================  ==============
name            ranking                                       index / order
==============  ============================================  ==============
logistic        the shipped ``FastScorer`` (split-1 ranker)    split 0, session
logistic_novel  the same features, relabelled (P2b, optional)  split 0, session
trans_s0        transition count / source support              split 0, session
trans_prior     the same counts                                splits < s, session
trans_online    the same counts, updated causally              splits < s + s so far, global
seq             GPU sequence model (WP11, optional)            precomputed table, session
==============  ============================================  ==============

An optional scorer whose artifact or module is missing is returned with
``status="skipped: <reason>"`` and no scorer, never a substitute. ``seq``
imports ``webprefetch.seqmodel`` lazily, so torch is needed only when it is
asked for.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .index import transition_index
from .model import FastScorer, features, load_artifacts
from .simulator import TransitionScorer

SCORERS = ("logistic", "logistic_novel", "trans_s0", "trans_prior", "trans_online", "seq")
DEFAULT_SCORERS = ("logistic",)
#: P2(b): a ranker trained with back-navigation and self-repeats as negatives.
#: It has its own artifact name, so the shipped ``ranker.joblib`` still
#: reproduces the committed pivot rows.
NOVEL_RANKER = "ranker_novel.joblib"
NOVEL_SUMMARY = "ranker_novel_summary.json"
#: Where ``seq`` looks for the WP11 artifact unless a path is given.
SEQ_DIRNAME = "seq"


@dataclass
class ScorerSpec:
    """One registry entry resolved for one evaluated split."""
    name: str
    status: str
    scorer: object = None
    index: object = None
    order: str = "session"
    train_range: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "ok"


class TableScorer:
    """Rankings looked up by ``(session_id, position)``.

    For a model that is scored once in a batch (the sequence model): the
    replay stays CPU-only and exact, and the index argument is ignored.
    """

    def __init__(self, table: dict):
        self.table = table

    def score(self, row, index=None) -> list[tuple[int, float]]:
        return list(self.table.get((int(row.session_id), int(row.position)), ()))


#: The frame column a :class:`RankTable` reads; :func:`rank_table` adds it.
RANK_ROW = "rank_row"


class RankTable:
    """Every row's ranking, computed once and stored as numpy arrays.

    ``score(row)`` returns exactly what the live scorer returned for that row
    (the same items, the same float64 probabilities, the same order), so a
    replay through the table equals a replay through the scorer. Many
    policies, capacities and processes then share one scoring pass, and the
    arrays are read-only buffers that forked workers never copy.
    """

    def __init__(self, items: np.ndarray, probs: np.ndarray, lengths: np.ndarray):
        self.items, self.probs, self.lengths = items, probs, lengths

    def score(self, row, index=None) -> list[tuple[int, float]]:
        r = getattr(row, RANK_ROW)
        k = self.lengths[r]
        return list(zip(self.items[r, :k].tolist(), self.probs[r, :k].tolist()))


_SHARED: dict = {}


def _install(shared: dict) -> None:
    _SHARED.clear()
    _SHARED.update(shared)


def _rank_shard(keys) -> list:
    """Score one shard's rows in global time order.

    An online index is advanced before each row exactly as the replay
    advances it (its state depends only on time), so every row gets the
    ranking its replayed decision saw.
    """
    frame, index, scorer = _SHARED["frame"], _SHARED["index"], _SHARED["scorer"]
    if keys is not None:
        frame = frame.loc[frame.session_id.isin(keys)]
    if getattr(index, "online", False):
        index = index.fresh()
    advance = getattr(index, "advance", None)
    keys = ["timestamp", "source_row"] if "source_row" in frame else ["timestamp"]
    out = []
    for row in frame.sort_values(keys, kind="mergesort").itertuples(index=False):
        if advance is not None:
            advance(int(row.timestamp))
        out.append((getattr(row, RANK_ROW), scorer.score(row, index)))
    return out


def map_shards(function, tasks: list, shared: dict, workers: int) -> list:
    """``map`` in a fork pool that shares ``shared``; ``workers=1`` stays here."""
    import multiprocessing
    import os
    if workers <= 1 or len(tasks) <= 1:
        _install(shared)
        try:
            return [function(task) for task in tasks]
        finally:
            _SHARED.clear()
    method = "fork" if "fork" in multiprocessing.get_all_start_methods() else None
    context = multiprocessing.get_context(method)
    processes = min(workers, len(tasks), os.cpu_count() or 1)
    with context.Pool(processes, initializer=_install, initargs=(shared,)) as pool:
        return pool.map(function, tasks, chunksize=1)


def session_shards(frame: pd.DataFrame, workers: int, key: str = "session_id") -> list:
    """Sorted key chunks, one per worker (``[None]`` = everything, serially)."""
    unique = np.sort(pd.unique(frame[key].to_numpy()))
    if workers <= 1 or len(unique) <= 1:
        return [None]
    return [chunk for chunk in np.array_split(unique, min(workers, len(unique))) if len(chunk)]


def rank_table(frame: pd.DataFrame, spec: "ScorerSpec", workers: int = 1) -> RankTable:
    """Score every row of ``frame`` (which must carry :data:`RANK_ROW` =
    0..n-1) once, sharded by session."""
    parts = map_shards(_rank_shard, session_shards(frame, workers),
                       {"frame": frame, "index": spec.index, "scorer": spec.scorer}, workers)
    n = len(frame)
    width = max((len(ranked) for part in parts for _, ranked in part), default=0)
    items = np.full((n, max(1, width)), -1, dtype=np.int64)
    probs = np.zeros((n, max(1, width)), dtype=np.float64)
    lengths = np.zeros(n, dtype=np.int64)
    for part in parts:
        for r, ranked in part:
            lengths[r] = len(ranked)
            if ranked:
                items[r, :len(ranked)] = [item for item, _ in ranked]
                probs[r, :len(ranked)] = [p for _, p in ranked]
    return RankTable(items, probs, lengths)


def parse_scorers(text: str | None) -> tuple[str, ...]:
    """``"logistic,trans_s0"`` -> a validated, de-duplicated tuple."""
    if not text:
        return DEFAULT_SCORERS
    names = tuple(dict.fromkeys(part.strip() for part in text.split(",") if part.strip()))
    unknown = [name for name in names if name not in SCORERS]
    if unknown:
        raise ValueError(f"unknown scorer(s) {unknown}; expected some of {SCORERS}")
    return names


def _skipped(name: str, reason: str) -> ScorerSpec:
    return ScorerSpec(name, f"skipped: {reason}")


def build_scorer(name: str, views: pd.DataFrame, split: int, model_dir: str | Path, *,
                 artifacts: tuple | None = None, seq_path: str | Path | None = None) -> ScorerSpec:
    """Resolve ``name`` for evaluating ``split`` (2 = validation, 3 = test).

    ``artifacts`` is an already loaded ``(index, ranker)`` pair, so callers
    that build several scorers load the joblib files once.
    """
    if name not in SCORERS:
        raise ValueError(f"unknown scorer {name}; expected one of {SCORERS}")
    model_dir = Path(model_dir)
    if name in ("logistic", "logistic_novel"):
        index, ranker = artifacts if artifacts is not None else load_artifacts(model_dir)
        if name == "logistic":
            return ScorerSpec(name, "ok", FastScorer(ranker), index, "session",
                              "ranker: split 1 sample; index: split 0")
        path = model_dir / NOVEL_RANKER
        if not path.exists():
            return _skipped(name, f"no relabelled ranker at {path} (webprefetch train --novel-ranker)")
        return ScorerSpec(name, "ok", FastScorer(joblib.load(path)), index, "session",
                          "relabelled ranker: split 1 sample; index: split 0")
    if name.startswith("trans_"):
        index = transition_index(views, name, split)
        order = "global" if getattr(index, "online", False) else "session"
        ranges = {"trans_s0": "counts: split 0",
                  "trans_prior": f"counts: splits 0-{split - 1}",
                  "trans_online": f"counts: splits 0-{split - 1} + split {split} before each decision"}
        return ScorerSpec(name, "ok", TransitionScorer(), index, order, ranges[name])
    return _seq_scorer(views, split, model_dir, seq_path)


def _seq_scorer(views: pd.DataFrame, split: int, model_dir: Path,
                seq_path: str | Path | None) -> ScorerSpec:
    """The WP11 sequence model, if both its module and an artifact exist.

    Expected protocol: ``seqmodel.SequenceScorer.load(path)`` returns an
    object whose ``precompute(views, splits)`` gives ``{(session_id,
    position): ranked}``; an object that already has ``score(row, index)``
    is used as it is.
    """
    try:
        from . import seqmodel  # noqa: PLC0415 - optional, pulls in torch
    except ImportError as error:
        return _skipped("seq", f"webprefetch.seqmodel unavailable ({error})")
    path = Path(seq_path) if seq_path is not None else model_dir / SEQ_DIRNAME
    if not path.exists():
        return _skipped("seq", f"no trained sequence-model artifact at {path}")
    try:
        loaded = seqmodel.SequenceScorer.load(path)
        if hasattr(loaded, "precompute"):
            scorer = TableScorer(loaded.precompute(views, (split,)))
        elif hasattr(loaded, "score"):
            scorer = loaded
        else:
            return _skipped("seq", "SequenceScorer has neither precompute() nor score()")
    except Exception as error:  # a stale or incompatible artifact is a skip, not a crash
        return _skipped("seq", f"could not load {path}: {error}")
    train_range = getattr(loaded, "train_range", "see the artifact's config")
    return ScorerSpec("seq", "ok", scorer, None, "session", str(train_range))


# ------------------------------------------------------------------ P2(b)

def _novel_label(row, candidate: int) -> int:
    """Positive only for a next view that is neither a self-repeat nor a step back."""
    if not bool(row.eligible) or candidate != row.next_item:
        return 0
    return int(row.next_item != row.itemid and row.next_item != row.prev_item)


def _sample(frame: pd.DataFrame, limit: int) -> pd.DataFrame:
    if len(frame) > limit:
        return frame.iloc[np.linspace(0, len(frame) - 1, limit, dtype=np.int64)]
    return frame


def _matrix(frame: pd.DataFrame, index) -> tuple[np.ndarray, np.ndarray]:
    x, y = [], []
    for row in frame.itertuples(index=False):
        for rank, (item, count) in enumerate(index.candidates(int(row.itemid))):
            x.append(features(row, item, count, rank, index))
            y.append(_novel_label(row, item))
    return np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.int8)


def train_novel_ranker(views: pd.DataFrame, index, max_decisions: int = 30000,
                       validation_decisions: int = 10000):
    """P2(b): the shipped ranker's features and protocol, relabelled.

    The shipped label is ``candidate == next_item``, so the ranker learns
    back-navigation, which a session cache already holds. Here a next view
    that repeats the current item or returns to the previous one is a
    negative for every candidate. Same split-1 sample, same split-2
    validation sample, same C grid chosen by validation log loss.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    train_x, train_y = _matrix(_sample(views.loc[views.split.eq(1)], max_decisions), index)
    val_x, val_y = _matrix(_sample(views.loc[views.split.eq(2)], validation_decisions), index)
    if train_y.sum() < 100 or not val_y.any():
        raise ValueError("too few relabelled positives to train the novel ranker")
    tuning, fitted = [], []
    for c in (0.1, 1.0, 10.0):
        candidate = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=200, solver="lbfgs"))
        candidate.fit(train_x, train_y)
        tuning.append({"C": c, "validation_log_loss":
                       float(log_loss(val_y, candidate.predict_proba(val_x)[:, 1]))})
        fitted.append(candidate)
    winner = min(range(len(tuning)), key=lambda n: tuning[n]["validation_log_loss"])
    summary = {"label": "candidate == next_item and next_item not in (itemid, prev_item)",
               "training_candidate_rows": int(len(train_y)), "positive_candidates": int(train_y.sum()),
               "validation_candidate_rows": int(len(val_y)), "validation_positives": int(val_y.sum()),
               "C_tuning": tuning, "selected_C": tuning[winner]["C"]}
    return fitted[winner], summary


def save_novel_ranker(model_dir: str | Path, model, summary: dict) -> Path:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / NOVEL_RANKER, compress=3)
    (model_dir / NOVEL_SUMMARY).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return model_dir / NOVEL_RANKER

