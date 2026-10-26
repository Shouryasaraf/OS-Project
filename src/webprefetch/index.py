"""Candidate indexes: frozen history counts, and a causal online index.

A frozen index from split 0 alone is what the pivot shipped; it is the
default here so the committed numbers still reproduce. It is also stale: on
the test split, an index that keeps learning from transitions it has already
observed covers far more next views. :class:`OnlineIndex` adds a transition
(a -> b) only once b's view time is strictly earlier than the decision time,
so no replayed decision can see its own or any later transition.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import pandas as pd

MAX_TRANSITIONS = 20
POPULAR_FALLBACK = 8
TRANSITION_VARIANTS = ("trans_s0", "trans_prior", "trans_online")


@dataclass
class CandidateIndex:
    transitions: dict[int, tuple[tuple[int, int], ...]]
    source_total: dict[int, int]
    global_count: dict[int, int]
    popular: tuple[int, ...]

    def candidates(self, current: int) -> list[tuple[int, int]]:
        found: dict[int, int] = {}
        for item, count in self.transitions.get(current, ()):
            if item != current:
                found[item] = count
        for item in self.popular:
            if item != current and item not in found:
                found[item] = 0
        return list(found.items())


def _history(views: pd.DataFrame, max_split: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    history = views.loc[views.split.le(max_split)]
    pairs = history.loc[history.eligible & history.next_item.ge(0), ["itemid", "next_item"]]
    return history, pairs


def _popularity(history: pd.DataFrame) -> tuple[dict[int, int], tuple[int, ...]]:
    popularity = history.itemid.value_counts().rename("count").rename_axis("itemid").reset_index()
    popularity = popularity.sort_values(["count", "itemid"], ascending=[False, True])
    global_count = {int(row.itemid): int(row.count) for row in popularity.itertuples(index=False)}
    return global_count, tuple(int(x) for x in popularity.itemid.iloc[:POPULAR_FALLBACK])


def fit_index(views: pd.DataFrame, max_split: int = 0) -> CandidateIndex:
    """Count transitions and popularity over splits ``<= max_split``.

    ``max_split=0`` is the pivot's split-0 index; ``max_split=s-1`` is the
    freshest index that is still legal for evaluating split ``s``.
    """
    history, pairs = _history(views, max_split)
    counts = pairs.groupby(["itemid", "next_item"], sort=False).size().rename("count").reset_index()
    counts = counts.sort_values(["itemid", "count", "next_item"], ascending=[True, False, True])
    top = counts.groupby("itemid", sort=False).head(MAX_TRANSITIONS)
    transitions: dict[int, list[tuple[int, int]]] = {}
    for row in top.itertuples(index=False):
        transitions.setdefault(int(row.itemid), []).append((int(row.next_item), int(row.count)))
    source_total = {int(k): int(v) for k, v in pairs.groupby("itemid").size().items()}
    global_count, popular = _popularity(history)
    return CandidateIndex({k: tuple(v) for k, v in transitions.items()}, source_total,
                          global_count, popular)


class OnlineIndex:
    """History counts plus every evaluated-split transition observed so far.

    It has the same interface as :class:`CandidateIndex` (``candidates``,
    ``source_total``, ``global_count``, ``popular``), so the transition and
    logistic scorers both run on it. ``advance(ts)`` must be called with
    non-decreasing decision timestamps; the replay engine does so in global
    time order and refuses an online index in session order. Popularity and
    ``global_count`` stay at their history values.
    """

    online = True

    def __init__(self, counts: dict[int, Counter], global_count: dict[int, int],
                 popular: tuple[int, ...], pending: list[tuple[int, int, int]]):
        self._base = counts
        self.global_count = global_count
        self.popular = popular
        # (next_timestamp, source, target), sorted by next_timestamp.
        self.pending = pending
        self._reset()

    def _reset(self) -> None:
        self._counts = {source: Counter(targets) for source, targets in self._base.items()}
        self.source_total = {source: sum(targets.values()) for source, targets in self._base.items()}
        self._sorted: dict[int, tuple[tuple[int, int], ...]] = {}
        self.position = 0
        self.now = None

    @classmethod
    def from_views(cls, views: pd.DataFrame, split: int,
                   history_max_split: int | None = None) -> "OnlineIndex":
        """History from splits ``<= history_max_split`` (default ``split - 1``),
        then the transitions of ``split`` as they become observable."""
        history_max_split = split - 1 if history_max_split is None else history_max_split
        if history_max_split >= split:
            raise ValueError("history must end before the evaluated split")
        history, pairs = _history(views, history_max_split)
        counts: dict[int, Counter] = {}
        for source, target in zip(pairs.itemid.to_numpy(), pairs.next_item.to_numpy()):
            counts.setdefault(int(source), Counter())[int(target)] += 1
        global_count, popular = _popularity(history)
        frame = views.loc[views.split.eq(split) & views.eligible & views.next_item.ge(0),
                          ["next_timestamp", "itemid", "next_item"]]
        frame = frame.sort_values("next_timestamp", kind="mergesort")
        pending = list(zip(frame.next_timestamp.astype("int64").tolist(),
                           frame.itemid.astype("int64").tolist(),
                           frame.next_item.astype("int64").tolist()))
        return cls(counts, global_count, popular, pending)

    def fresh(self) -> "OnlineIndex":
        """A copy at the start of the evaluated split; replays never share state."""
        return OnlineIndex(self._base, self.global_count, self.popular, self.pending)

    def advance(self, timestamp: int) -> None:
        if self.now is not None and timestamp < self.now:
            raise ValueError("online index decisions must be in non-decreasing time order")
        self.now = timestamp
        pending = self.pending
        while self.position < len(pending) and pending[self.position][0] < timestamp:
            _, source, target = pending[self.position]
            self._counts.setdefault(source, Counter())[target] += 1
            self.source_total[source] = self.source_total.get(source, 0) + 1
            self._sorted.pop(source, None)
            self.position += 1

    @property
    def transitions(self) -> dict[int, tuple[tuple[int, int], ...]]:
        return {source: self._top(source) for source in self._counts}

    def _top(self, source: int) -> tuple[tuple[int, int], ...]:
        top = self._sorted.get(source)
        if top is None:
            targets = self._counts.get(source)
            top = tuple(sorted(targets.items(), key=lambda pair: (-pair[1], pair[0]))
                        [:MAX_TRANSITIONS]) if targets else ()
            self._sorted[source] = top
        return top

    def candidates(self, current: int) -> list[tuple[int, int]]:
        found: dict[int, int] = {}
        for item, count in self._top(current):
            if item != current:
                found[item] = count
        for item in self.popular:
            if item != current and item not in found:
                found[item] = 0
        return list(found.items())


def transition_index(views: pd.DataFrame, variant: str, split: int):
    """The index behind each transition-count scorer, for evaluating ``split``.

    ``trans_s0`` freezes split 0, ``trans_prior`` uses every split before
    ``split``, and ``trans_online`` adds ``split``'s own transitions causally.
    """
    if variant == "trans_s0":
        return fit_index(views, max_split=0)
    if split < 1:
        raise ValueError("fresh indexes need at least one earlier split")
    if variant == "trans_prior":
        return fit_index(views, max_split=split - 1)
    if variant == "trans_online":
        return OnlineIndex.from_views(views, split)
    raise ValueError(f"unknown transition variant {variant}; expected {TRANSITION_VARIANTS}")
