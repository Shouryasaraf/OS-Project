"""Replay plan for the live experiment, and the simulator reference it must match.

``build_plan`` samples held-out sessions (split 3, at least ``min_views``
views, seed 42), runs :func:`webprefetch.simulator.replay` on exactly those
sessions with ``with_sessions=True``, and writes two files:

* ``plan.json`` -- per session, per view: the item, its timestamp, the
  recorded gap before it, and for every policy the candidates that policy
  asks the cache to prefetch at that view.
* ``sim.json`` -- the simulator's per-session ``[hits, issued, views,
  useful]`` per policy. Live counters must equal these exactly.

The per-view candidate lists come from a decision walk over the candidate
rankings the simulator itself produced (a recording scorer), through the
same :mod:`cachecore` objects: ``Cache``, ``Policy.choose`` and, for the
feedback gate, ``GateTracker``. The walk's per-session counters are checked
against the simulator before anything is written, so a plan that disagrees
with the simulator is refused rather than replayed.

Stateful decisions are taken in **trace time order**. The feedback gate is
shared across sessions and the online index learns from transitions as they
become observable, so both are replayed in global timestamp order exactly as
the simulator does (``order="global"``). The live pass then measures the
latency of those decisions on a time-compressed schedule (capped gaps and a
ramp); it does not re-learn the gate in live time, which would make the
counters order-dependent and break parity. The gate learns only from the
sampled sessions, not from the whole split.

Scorers: ``logistic`` (the trained ranker over the frozen split-0 index,
``models/web``) and the transition-count scorers ``trans_s0``,
``trans_prior`` and ``trans_online`` (:func:`webprefetch.index.transition_index`).
A policy name may end in ``@<scorer>``; without one it uses ``logistic``.
Until WP6's scorer registry is merged, the transition scorers are built here
from ``index.transition_index`` and ``simulator.TransitionScorer`` directly.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

from ..cachecore import Cache, Counters, FeedbackGate, GateTracker, Policy

FORMAT = "webprefetch.live.plan/1"
SCORERS = ("logistic", "trans_s0", "trans_prior", "trans_online")
DEFAULT_SCORER = "logistic"
DEFAULT_SPLIT = 3
DEFAULT_SESSIONS = 2000
DEFAULT_MIN_VIEWS = 2
DEFAULT_SEED = 42
DEFAULT_CAPACITY = 8
#: E6(a): the headline policy set (plan section 3.3).
HEADLINE_POLICIES = ("fixed0", "fixed1", "fixed4", "fixed8", "adaptive_0.002", "adaptive_0.01",
                     "gate_0.01", "feedback_0.01", "adaptive_0.01@trans_online")
BASELINE = "fixed0"
PLAN_NAME = "plan.json"
SIM_NAME = "sim.json"
_NUMBER = r"(\d+(?:\.\d+)?(?:e-?\d+)?)"
_PATTERNS = (
    (re.compile(rf"fixed(\d+)$"), "fixed"),
    (re.compile(rf"gate_{_NUMBER}$"), "gate"),
    (re.compile(rf"adaptive_ca_{_NUMBER}$"), "adaptive_ca"),
    (re.compile(rf"adaptive_{_NUMBER}$"), "adaptive"),
    (re.compile(rf"feedback_{_NUMBER}(?:_e{_NUMBER})?(?:_w{_NUMBER})?$"), "feedback"),
)


class PlanParityError(RuntimeError):
    """The decision walk and the simulator disagree; the plan is not written."""


@dataclass(frozen=True)
class LivePolicy:
    """A cachecore policy plus the scorer whose rankings it reads."""
    name: str
    policy: Policy
    scorer: str = DEFAULT_SCORER

    def record(self) -> dict:
        p = self.policy
        return {"name": self.name, "kind": p.kind, "parameter": p.parameter,
                "epsilon": p.epsilon, "pollution": p.pollution, "scorer": self.scorer}


def parse_policy(name: str) -> LivePolicy:
    """``fixed<k>``, ``gate_<t>``, ``adaptive_<p>``, ``adaptive_ca_<p>`` or
    ``feedback_<c>[_e<epsilon>][_w<pollution>]``, optionally ``@<scorer>``."""
    base, _, scorer = name.partition("@")
    scorer = scorer or DEFAULT_SCORER
    if scorer not in SCORERS:
        raise ValueError(f"unknown scorer {scorer!r} in {name!r}; expected one of {SCORERS}")
    for pattern, kind in _PATTERNS:
        match = pattern.match(base)
        if not match:
            continue
        if kind == "feedback":
            cost, epsilon, pollution = match.groups()
            return LivePolicy(name, Policy(name, kind, float(cost), float(epsilon or 0.0),
                                           float(pollution or 0.0)), scorer)
        return LivePolicy(name, Policy(name, kind, float(match.group(1))), scorer)
    raise ValueError(f"cannot parse policy {name!r}")


def parse_policies(names) -> list[LivePolicy]:
    if isinstance(names, str):
        names = [n for n in names.split(",") if n]
    out = [parse_policy(n) for n in names]
    if len({p.name for p in out}) != len(out):
        raise ValueError("policy names must be unique")
    return out


def make_scorers(views, names, *, models_dir=None, split: int = DEFAULT_SPLIT) -> dict:
    """``{scorer name: (index, scorer)}`` for every name in ``names``."""
    from ..index import transition_index
    from ..simulator import TransitionScorer
    out = {}
    for name in sorted(set(names)):
        if name == "logistic":
            from ..model import FastScorer, load_artifacts
            if models_dir is None:
                raise ValueError("the logistic scorer needs models_dir (webprefetch train)")
            index, ranker = load_artifacts(models_dir)
            out[name] = (index, FastScorer(ranker))
        elif name in SCORERS:
            out[name] = (transition_index(views, name, split), TransitionScorer())
        else:
            raise ValueError(f"unknown scorer {name!r}")
    return out


class _RecordingScorer:
    """Pass-through scorer that keeps the ranking of every (session, position)."""

    def __init__(self, inner):
        self.inner = inner
        self.ranked: dict[tuple[int, int], list[tuple[int, float]]] = {}

    def score(self, row, index):
        ranked = self.inner.score(row, index)
        self.ranked[(int(row.session_id), int(row.position))] = ranked
        return ranked


class _RecordingCache(Cache):
    """A cachecore Cache that remembers every item it was asked to prefetch."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.asked: list[int] = []

    def prefetch(self, item: int) -> bool:
        self.asked.append(int(item))
        return super().prefetch(item)


def _walk(frame, ranked: dict, live: LivePolicy, capacity: int,
          session_gap_ms: int) -> tuple[dict, dict]:
    """Per-view prefetch requests and per-session counters of one policy.

    Mirrors ``simulator.replay_partial`` for a single capacity and session
    scope, in the same global row order; it differs only in recording what
    each view asked for.
    """
    policy = live.policy
    gate = FeedbackGate.for_policy(policy) if policy.stateful else None
    last = ~frame.session_id.duplicated(keep="last").to_numpy()
    caches: dict[int, tuple[_RecordingCache, GateTracker | None]] = {}
    sessions: dict[int, list[int]] = {}
    decisions: dict[tuple[int, int], list[int]] = {}
    expiries: list = []
    sequence = 0
    for position, row in enumerate(frame.itertuples(index=False)):
        now = int(row.timestamp)
        if gate is not None:
            while expiries and expiries[0][0] <= now:
                _, _, expired, tracker = heapq.heappop(expiries)
                tracker.retire(expired)
        sid, item = int(row.session_id), int(row.itemid)
        state = caches.get(sid)
        if state is None:
            tracker = GateTracker(gate) if gate is not None else None
            cache = _RecordingCache(capacity, Counters(),
                                    on_evict=tracker.on_evict if tracker is not None else None)
            state = caches[sid] = (cache, tracker)
        cache, tracker = state
        counters = cache.counters
        before = (counters.hits, counters.issued, counters.useful)
        cache.asked = []
        candidates = ranked[(sid, int(row.position))]
        if tracker is not None:
            tracker.demand(cache, item)
            tracker.prefetch(cache, candidates, sid)
        else:
            cache.demand(item)
            _, nominal = policy.choose(candidates, cache.blocks)
            for candidate, _ in nominal:
                cache.prefetch(candidate)
        decisions[(sid, int(row.position))] = cache.asked
        record = sessions.setdefault(sid, [0, 0, 0, 0])
        record[0] += counters.hits - before[0]
        record[1] += counters.issued - before[1]
        record[2] += 1
        record[3] += counters.useful - before[2]
        if last[position]:
            if tracker is not None:
                heapq.heappush(expiries, (now + session_gap_ms, sequence, sid, tracker))
                sequence += 1
            del caches[sid]
    return decisions, sessions


def select_sessions(views, *, split=DEFAULT_SPLIT, sessions: int = DEFAULT_SESSIONS,
                    min_views: int = DEFAULT_MIN_VIEWS, seed: int = DEFAULT_SEED):
    """The sampled sessions' rows, in prepared (timestamp) order, and their ids."""
    splits = (split,) if isinstance(split, int) else tuple(split)
    test = views.loc[views.split.isin(splits)]
    lengths = test.groupby("session_id", sort=False).size()
    pool = sorted(int(k) for k in lengths.index[lengths.ge(min_views)])
    if not pool:
        raise ValueError(f"no session in split {splits} has at least {min_views} views")
    chosen = sorted(random.Random(seed).sample(pool, min(sessions, len(pool))))
    return test.loc[test.session_id.isin(set(chosen))], chosen


def build_plan(views, policies, scorers: dict, *, split=DEFAULT_SPLIT,
               sessions: int = DEFAULT_SESSIONS, min_views: int = DEFAULT_MIN_VIEWS,
               seed: int = DEFAULT_SEED, capacity: int = DEFAULT_CAPACITY,
               session_gap_ms: int | None = None) -> tuple[dict, dict]:
    """Return ``(plan, sim)``; raises :class:`PlanParityError` on disagreement.

    ``session_gap_ms`` defaults to the 30-minute ``data.SESSION_GAP_MS``.
    """
    from ..data import SESSION_GAP_MS
    from ..simulator import _ordered, replay
    session_gap_ms = SESSION_GAP_MS if session_gap_ms is None else session_gap_ms
    policies = [p if isinstance(p, LivePolicy) else parse_policy(p) for p in policies]
    if len({p.name for p in policies}) != len(policies):
        raise ValueError("policy names must be unique")
    missing = {p.scorer for p in policies} - set(scorers)
    if missing:
        raise ValueError(f"no scorer given for {sorted(missing)}")
    subset, chosen = select_sessions(views, split=split, sessions=sessions,
                                     min_views=min_views, seed=seed)
    frame = _ordered(subset, "global")  # the simulator's own global row order
    decisions: dict[str, dict] = {}
    sim_sessions: dict[str, dict] = {}
    totals: dict[str, dict] = {}
    for scorer_name in sorted({p.scorer for p in policies}):
        group = [p for p in policies if p.scorer == scorer_name]
        index, scorer = scorers[scorer_name]
        recorder = _RecordingScorer(scorer)
        rows, extra, _ = replay(subset, index, None, [p.policy for p in group], capacity,
                                max_views=0, with_sessions=True, scorer=recorder,
                                order="global", session_gap_ms=session_gap_ms)
        for live, row in zip(group, rows):
            simulated = {int(k): list(v) for k, v in extra["sessions"][live.name].items()}
            walked_decisions, walked = _walk(frame, recorder.ranked, live, capacity,
                                             session_gap_ms)
            bad = [sid for sid in chosen if walked.get(sid) != simulated.get(sid)]
            if bad:
                raise PlanParityError(
                    f"{live.name}: decision walk disagrees with the simulator on {len(bad)} "
                    f"sessions (first {bad[0]}: walk {walked.get(bad[0])}, "
                    f"simulator {simulated.get(bad[0])})")
            decisions[live.name] = walked_decisions
            sim_sessions[live.name] = {str(sid): simulated[sid] for sid in chosen}
            totals[live.name] = {k: row[k] for k in (
                "views", "hits", "issued", "useful", "prefetch_hits", "evicted_unused",
                "demand_evicted_by_prefetch", "hit_ratio", "precision", "issued_per_view")}
    plan_sessions: dict[str, list] = {}
    ordered = subset.sort_values(["session_id", "position"], kind="mergesort")
    for row in ordered.itertuples(index=False):
        key = (int(row.session_id), int(row.position))
        pf = {p.name: decisions[p.name][key] for p in policies if decisions[p.name][key]}
        plan_sessions.setdefault(str(key[0]), []).append(
            {"item": int(row.itemid), "ts": int(row.timestamp), "gap_ms": int(row.gap_ms),
             "pf": pf})
    items = sorted({v["item"] for views_ in plan_sessions.values() for v in views_}
                   | {c for views_ in plan_sessions.values() for v in views_
                      for ids in v["pf"].values() for c in ids})
    splits = [split] if isinstance(split, int) else list(split)
    params = {"splits": splits, "sessions_requested": sessions, "min_views": min_views,
              "seed": seed, "capacity": capacity, "session_gap_ms": session_gap_ms,
              "policies": [p.record() for p in policies]}
    plan = {"format": FORMAT, "params": params, "sessions_selected": len(chosen),
            "views": int(len(subset)), "items": items, "sessions": plan_sessions}
    sim = {"format": FORMAT, "params": params, "totals": totals, "sessions": sim_sessions,
           "order": "global", "cache_scope": "session"}
    return plan, sim


def _dump(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def write_plan(plan: dict, sim: dict, directory: str | Path) -> dict:
    """Write ``plan.json`` and ``sim.json``; return their paths and SHA-256."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, obj in ((PLAN_NAME, plan), (SIM_NAME, sim)):
        text = _dump(obj)
        path = directory / name
        tmp = path.with_name(name + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)
        out[name] = {"path": str(path), "sha256": hashlib.sha256(text.encode()).hexdigest()}
    return out


def load_plan(directory: str | Path) -> tuple[dict, dict]:
    directory = Path(directory)
    plan = json.loads((directory / PLAN_NAME).read_text(encoding="utf-8"))
    sim = json.loads((directory / SIM_NAME).read_text(encoding="utf-8"))
    if plan.get("format") != FORMAT or sim.get("format") != FORMAT:
        raise ValueError(f"{directory}: not a {FORMAT} plan; rebuild it")
    return plan, sim


def plan_matches(directory: str | Path, params: dict) -> bool:
    """True when ``directory`` already holds a plan built with ``params``."""
    try:
        plan, _ = load_plan(directory)
    except (OSError, ValueError):
        return False
    return plan["params"] == json.loads(json.dumps(params))
