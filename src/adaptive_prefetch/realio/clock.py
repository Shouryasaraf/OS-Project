"""Think-time models and a precise pacer.

The trace timestamps are whole seconds (oracleGeneral), so the original
inter-arrival times are unknowable below 1 s. Three explicit models instead:

* ``closed``                -- no think time; the next request is issued as soon
                               as the previous one completes. Prefetches get no
                               idle time to complete in.
* ``fixed:US``              -- the same think time before every request.
* ``spread:SCALE:CAP_US``   -- the n requests of one 1-s tick are spread over
                               1/n s each; a gap between ticks is its real
                               length. Every gap is multiplied by SCALE and
                               capped at CAP_US, so idle hours do not stall
                               the run.

The think time is waited *before* each request, after the previous one's
prefetches were submitted, so a prefetch can overlap it. :class:`Pacer` sleeps
to just short of the deadline and spins the rest; every wait's overshoot is
logged as ``think_error_us`` so a model's realised timing is on record.
"""

from __future__ import annotations

import time
from array import array

from .syscalls import set_timer_slack

THINK_MODELS = ("closed", "fixed", "spread")


def parse_think(spec: str) -> tuple[str, tuple[float, ...]]:
    kind, *args = spec.split(":")
    try:
        values = tuple(float(a) for a in args)
    except ValueError:
        raise ValueError(f"bad think-time spec {spec!r}") from None
    if kind == "closed" and not values:
        return kind, values
    if kind == "fixed" and len(values) == 1 and values[0] >= 0:
        return kind, values
    if kind == "spread" and len(values) == 2 and values[0] >= 0 and values[1] >= 0:
        return kind, values
    raise ValueError(f"bad think-time spec {spec!r}; expected closed, fixed:US "
                     "or spread:SCALE:CAP_US")


def gaps_us(requests, spec: str) -> array:
    """Think time before each request, microseconds. The first is always 0."""
    kind, args = parse_think(spec)
    out = array("d", bytes(8 * len(requests)))
    if kind == "closed" or not requests:
        return out
    if kind == "fixed":
        for i in range(1, len(requests)):
            out[i] = args[0]
        return out
    scale, cap = args
    per_tick: dict[float, int] = {}
    for request in requests:
        per_tick[request.timestamp_ms] = per_tick.get(request.timestamp_ms, 0) + 1
    previous = requests[0].timestamp_ms
    for i in range(1, len(requests)):
        ts = requests[i].timestamp_ms
        gap = 1e6 / per_tick[ts] if ts == previous else (ts - previous) * 1e3
        out[i] = min(cap, gap * scale)
        previous = ts
    return out


class Pacer:
    """Wait a target number of microseconds as precisely as user space can.

    Sleeps until ``spin_us`` before the deadline, then busy-waits. The timer
    slack is set to 1 us for this thread (``PR_SET_TIMERSLACK``), so the sleep
    part overshoots by tens of microseconds at most rather than the default
    50 us slack plus scheduler latency.
    """

    def __init__(self, spin_us: float = 200.0):
        self.spin_us = spin_us
        self.slack_set = set_timer_slack(1000)
        # 8 bytes a wait, not a 32-byte float object: anonymous memory grown
        # mid-run comes out of Backend A's page-cache budget.
        self.errors = array("d")
        self.total_us = 0.0

    def wait(self, us: float) -> None:
        if us <= 0:
            return
        start = time.perf_counter()
        deadline = start + us / 1e6
        sleep_for = us - self.spin_us
        if sleep_for > 0:
            time.sleep(sleep_for / 1e6)
        while time.perf_counter() < deadline:
            pass
        waited = (time.perf_counter() - start) * 1e6
        self.errors.append(waited - us)
        self.total_us += waited
