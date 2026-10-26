"""Online (closed-loop) policy driver for Backend A: no precomputed plan.

The plan-driven backends execute decisions the simulator made offline, which
invites the objection that the policy never ran against a real cache. Here
the policy object runs **inline**: each real request is read from the page
cache first, then fed to the policy, and its proposals are issued as
``WILLNEED`` hints straight away.

* ``deep``/``deep@N``, ``sequential``, ``stride``, ``markov``, ``linux_ra`` and
  the gated ``guard`` / ``depth_adaptive`` / ``correlate`` are supported. The
  window-routed modes (``adaptive``, ``adaptive_evidence``, ``strided``) are
  not: they need the replay loop's window machinery, and are measured through
  the plan instead.
* A user-space **shadow** ``LRUCache`` of the same capacity sees the same
  requests and the same issued prefetches. It deduplicates proposals exactly
  as ``replay`` does (a block the shadow holds is not re-issued) and is what
  ``linux_ra`` reads its window state from (``CacheView``): the emulated
  policy tracks the shadow LRU, never the kernel page cache, whose own policy
  ``kernel_ra`` / ``kernel_seq`` measure.
* The gated modes are fed **measured** usefulness: blocks demanded while
  pending *and* resident in the page cache, as a cumulative count that
  ``GateController.note_useful`` turns into per-interval deltas (the D15
  invariant). Their decisions therefore follow the real cache, so they can
  differ from the simulator's.
* The first window never prefetches, as in ``replay``.

The row's ``sim_*`` columns come from the shadow cache: the simulated outcome
of exactly the prefetches this run issued, not an offline plan.
"""

from __future__ import annotations

from time import perf_counter

from ..guard import GateController, read_ahead
from ..simulator import (DEFAULT_READ_AHEAD_DEPTH, GATED_MODES, CacheView,
                         LatencyModel, Metrics, _demand_read, _issue_prefetches,
                         _make_predictor, candidates)
from ..baselines import DEFAULT_LINUX_RA_MAX_BLOCKS
from .clock import Pacer
from .decisions import ONLINE_PREFIX, PREFETCH, RecordingLRU, _runs, parse_mode

ONLINE_MODES = ("none", "deep", "sequential", "stride", "markov", "linux_ra",
                *GATED_MODES)


def online_mode(label: str) -> tuple[str, dict]:
    """``online-guard`` -> ``("guard", {})``; raises for unsupported modes."""
    if not label.startswith(ONLINE_PREFIX):
        raise ValueError(f"{label!r} is not an online mode")
    base, extra = parse_mode(label[len(ONLINE_PREFIX):])
    if base not in ONLINE_MODES:
        raise ValueError(f"{base!r} cannot run online; supported: {', '.join(ONLINE_MODES)}")
    return base, extra


class OnlineDriver:
    def __init__(self, label: str, capacity: int, *, window_size: int = 32,
                 read_ahead_depth: int = DEFAULT_READ_AHEAD_DEPTH,
                 latency_model: LatencyModel | None = None,
                 policy_interval: int | None = None,
                 linux_ra_max_blocks: int = DEFAULT_LINUX_RA_MAX_BLOCKS):
        base, extra = online_mode(label)
        self.mode = base
        self.depth = extra.get("read_ahead_depth", read_ahead_depth)
        self.window_size = window_size
        self.interval_size = policy_interval or window_size
        self.latency = latency_model or LatencyModel()
        self.metrics = Metrics()
        self.shadow = RecordingLRU(capacity, self.metrics)
        predictor = _make_predictor(
            base, None, window_size=window_size,
            linux_ra_max_blocks=extra.get("linux_ra_max_blocks", linux_ra_max_blocks))
        self.observer = predictor if getattr(predictor, "cache_aware", False) else None
        self.predictor = predictor if self.observer is None else None
        self.view = CacheView(self.shadow) if self.observer is not None else None
        self.gate = (GateController(base, self.latency.break_even_precision())
                     if base in GATED_MODES else None)
        self.inference_us = 0.0

    def _propose(self, request, missed, reused) -> list[int]:
        start = perf_counter()
        if self.observer is not None:
            self.view.prefetch_hits = reused
            proposed = self.observer.observe(request, missed, self.view)
        elif self.predictor is not None:
            proposed = self.predictor.next_candidates(request)
        elif self.gate is not None:
            proposed = self.gate.candidates(request)
        elif self.mode == "deep":
            proposed = read_ahead(request, self.depth)
        elif self.mode == "sequential":
            proposed = candidates(request, "sequential", None)
        else:
            proposed = []
        self.inference_us += (perf_counter() - start) * 1e6
        return proposed

    def run(self, backend, requests, gaps, pacer: Pacer | None = None) -> dict:
        from .backend_pagecache import _Accumulator
        pacer = pacer or Pacer()
        stats = _Accumulator()
        gate, shadow, log = self.gate, self.shadow, self.shadow.log
        interval = 0
        for index, request in enumerate(requests):
            pacer.wait(gaps[index])
            if request.operation == "R":
                stats.add(backend.demand(request))
            missed: list[int] = []
            reused: list[int] = []
            tail_missed = _demand_read(shadow, self.metrics, self.latency, request,
                                       missed, reused)
            proposed = self._propose(request, missed, reused)
            if index >= self.window_size:
                mark = len(log)
                _issue_prefetches(shadow, self.metrics, self.latency, gate, proposed,
                                  request.lba + request.size_blocks, tail_missed)
                issued = [block for op, block, _ in log[mark:] if op == PREFETCH]
                if issued:
                    backend.hint(_runs(issued))
            # The log is only needed for the issued slice; keep it bounded.
            log.clear()
            if gate is not None:
                gate.observe(request)
                gate.note_useful(backend.useful)   # measured, cumulative
                interval += 1
                if interval >= self.interval_size:
                    gate.decide()
                    interval = 0
        if gate is not None:
            if interval:
                gate.note_useful(backend.useful)
                gate.decide()
            self.metrics.policy_switches = gate.switches
        backend.helper.shutdown(wait=True)
        backend.check_hints()
        result = stats.result(backend)
        result["inference_us"] = self.inference_us
        result["gate_switches"] = gate.switches if gate is not None else None
        result["gate_precision"] = (gate.precision if gate is not None and gate.has_data
                                    else None)
        return result
