"""Session-sharded replay: the serial replay's exact counts, on many processes.

Caches are private to a session (or, with ``cache_scope="visitor"``, to a
visitor), so shards of distinct sessions or visitors never interact and
their integer counters sum to the serial result exactly. Two kinds of state
cross that boundary and are handled separately:

- An :class:`~webprefetch.index.OnlineIndex` depends only on time. Every
  shard receives the whole evaluated split's pending transitions and replays
  its own rows in global time order, so it sees the same index the serial
  replay sees at each decision.
- A ``feedback`` policy's gate is shared by all sessions. Those policies run
  in one extra task over every row, in global order, alongside the shards.

Float sums (the calibration probability mass and the MRR numerator) are
added in a different order than in the serial loop, so they agree to
rounding error rather than bit for bit; every count is exact.
"""

from __future__ import annotations

import multiprocessing
import os

import numpy as np
import pandas as pd

from .cachecore import Counters, Policy
from .data import SESSION_GAP_MS
from .simulator import check_replay, complete_views, finalize, replay_partial

_SHARED: dict = {}


def _install(shared: dict) -> None:
    _SHARED.clear()
    _SHARED.update(shared)


def _run(task: tuple) -> dict:
    keys, policy_positions, count_stats = task
    shared = _SHARED
    frame = shared["frame"]
    if keys is not None:
        frame = frame.loc[frame[shared["key"]].isin(keys)]
    policies = [shared["policies"][n] for n in policy_positions]
    names = [p.name for p in policies if p.name in shared["sessions_for"]]
    return replay_partial(frame, shared["index"], shared["scorer"], policies,
                          shared["capacities"], order=shared["order"],
                          cache_scope=shared["cache_scope"], sessions_for=names,
                          session_gap_ms=shared["session_gap_ms"], count_stats=count_stats)


def _merge(parts: list[dict], capacities: tuple[int, ...]) -> dict:
    totals: dict = {cap: {} for cap in capacities}
    sessions: dict = {cap: {} for cap in capacities}
    visitors: dict = {}
    stats = {"covered": 0, "eligible": 0, "top1": 0, "reciprocal": 0.0,
             "novel_eligible": 0, "novel_top1": 0, "novel_covered": 0,
             "replayed_views": 0, "calibration_bins": [[0, 0.0, 0] for _ in range(10)]}
    for part, counted in parts:
        for cap in capacities:
            for name, counters in part["totals"][cap].items():
                totals[cap].setdefault(name, Counters()).merge(counters)
            for name, records in part["sessions"][cap].items():
                sessions[cap].setdefault(name, {}).update(records)
        visitors.update(part["session_visitors"])
        if counted:
            for key in stats:
                if key == "calibration_bins":
                    for mine, theirs in zip(stats[key], part["stats"][key]):
                        for n in range(3):
                            mine[n] += theirs[n]
                else:
                    stats[key] += part["stats"][key]
    return {"totals": totals, "sessions": sessions, "session_visitors": visitors,
            "stats": stats, "demo": []}


def replay_parallel(views: pd.DataFrame, index, scorer, policies: list[Policy],
                    capacities: tuple[int, ...], workers: int = 1, *,
                    order: str = "session", cache_scope: str = "session",
                    max_views: int = 0, with_sessions: bool = True,
                    session_gap_ms: int = SESSION_GAP_MS) -> dict[int, tuple[list[dict], dict]]:
    """Same result shape as :func:`~webprefetch.simulator.replay_capacities`.

    ``workers`` processes each replay a shard of sessions (visitors under
    ``cache_scope="visitor"``); ``workers=1`` runs the same shards in this
    process. ``scorer`` is any object with ``score(row, index)``.
    """
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if max_views < 0:
        raise ValueError("max_views must be non-negative")
    check_replay(views, index, policies, order, cache_scope)
    capacities = tuple(capacities)
    selected = complete_views(views, max_views)
    key = "visitorid" if cache_scope == "visitor" else "session_id"
    stateless = [n for n, p in enumerate(policies) if not p.stateful]
    stateful = [n for n, p in enumerate(policies) if p.stateful]
    tasks = []
    if stateless:
        # Sorted keys make the shard layout, and so the float sums, repeatable.
        unique = np.sort(pd.unique(selected[key].to_numpy()))
        shards = [chunk for chunk in np.array_split(unique, max(1, min(workers, len(unique))))
                  if len(chunk)]
        tasks += [(chunk, stateless, True) for chunk in shards]
    if stateful:
        tasks.append((None, stateful, not stateless))
    shared = {"frame": selected, "key": key, "index": index, "scorer": scorer,
              "policies": policies, "capacities": capacities, "order": order,
              "cache_scope": cache_scope, "session_gap_ms": session_gap_ms,
              "sessions_for": {p.name for p in policies} if with_sessions else set()}
    if workers == 1 or len(tasks) <= 1:
        _install(shared)
        try:
            parts = [_run(task) for task in tasks]
        finally:
            _SHARED.clear()
    else:
        method = "fork" if "fork" in multiprocessing.get_all_start_methods() else None
        context = multiprocessing.get_context(method)
        processes = min(workers, len(tasks), os.cpu_count() or 1)
        with context.Pool(processes, initializer=_install, initargs=(shared,)) as pool:
            parts = pool.map(_run, tasks, chunksize=1)
    merged = _merge([(part, task[2]) for part, task in zip(parts, tasks)], capacities)
    for cap in capacities:
        for p in policies:
            merged["totals"][cap].setdefault(p.name, Counters())
            if p.name in shared["sessions_for"]:
                merged["sessions"][cap].setdefault(p.name, {})
    return finalize(merged, policies, capacities)
