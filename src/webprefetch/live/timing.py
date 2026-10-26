"""One clock, a microsecond timer for asyncio, and an event-loop lag monitor.

asyncio's epoll selector rounds every timeout up to a whole millisecond
(``selectors.EpollSelector.select``), which made every emulated wait in the
prototype 0-1 ms late. :class:`PreciseTimer` keeps the deadlines on one
thread that waits on a ``threading.Condition`` (pthread, microsecond
resolution) and resolves the loop's futures with ``call_soon_threadsafe``.

``time.perf_counter`` is ``CLOCK_MONOTONIC`` on Linux, so the origin and the
client processes read the same clock. Stdlib only.
"""

from __future__ import annotations

import asyncio
import heapq
import itertools
import threading
import time

CLOCK = time.perf_counter
MONITOR_PERIOD_S = 0.005
MONITOR_KEEP = 200_000


def quantile(values, q: float):
    """Nearest-rank quantile of an unsorted sequence; ``None`` when empty.

    The prototype's rule (``sorted[int(q * n)]``, clamped), kept so its lag
    and wait figures are comparable with this package's.
    """
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def _resolve(future) -> None:
    if not future.done():
        future.set_result(None)


class PreciseTimer:
    """``await timer.sleep_until(deadline)`` with microsecond wake-ups."""

    def __init__(self, loop: asyncio.AbstractEventLoop):
        self.loop = loop
        self.heap: list = []
        self.sequence = itertools.count()
        self.condition = threading.Condition()
        self.closed = False
        self.thread = threading.Thread(target=self._run, name="precise-timer", daemon=True)
        self.thread.start()

    def _run(self) -> None:
        while True:
            with self.condition:
                while not self.heap and not self.closed:
                    self.condition.wait()
                if self.closed:
                    return
                when, _, future = self.heap[0]
                delay = when - CLOCK()
                if delay > 0:
                    self.condition.wait(delay)
                    continue
                heapq.heappop(self.heap)
            try:
                self.loop.call_soon_threadsafe(_resolve, future)
            except RuntimeError:  # the loop closed under us
                return

    def sleep_until(self, when: float):
        """A future that resolves at ``when`` (a :data:`CLOCK` reading)."""
        future = self.loop.create_future()
        if when <= CLOCK():
            future.set_result(None)
            return future
        with self.condition:
            heapq.heappush(self.heap, (when, next(self.sequence), future))
            if self.heap[0][2] is future:
                self.condition.notify()
        return future

    def sleep(self, seconds: float):
        return self.sleep_until(CLOCK() + seconds)

    def close(self) -> None:
        with self.condition:
            self.closed = True
            self.condition.notify()


class LagMonitor:
    """How late a periodic precise deadline is serviced by the event loop.

    The timer thread fires on time; the lag is the time the loop then needs
    to run the callback, which is what every emulated wait also pays.
    """

    def __init__(self, timer: PreciseTimer, period_s: float = MONITOR_PERIOD_S):
        self.timer = timer
        self.period_s = period_s
        self.lags: list[float] = []

    async def run(self) -> None:
        while True:
            when = CLOCK() + self.period_s
            await self.timer.sleep_until(when)
            self.lags.append(CLOCK() - when)
            if len(self.lags) > MONITOR_KEEP:
                del self.lags[:MONITOR_KEEP // 2]

    def reset(self) -> None:
        self.lags.clear()

    def summary(self) -> dict:
        lags = self.lags
        pick = (lambda q: quantile(lags, q) * 1e6) if lags else (lambda q: None)
        return {"loop_lag_us_p50": pick(0.5), "loop_lag_us_p99": pick(0.99),
                "loop_lag_us_max": max(lags) * 1e6 if lags else None,
                "loop_lag_samples": len(lags)}
