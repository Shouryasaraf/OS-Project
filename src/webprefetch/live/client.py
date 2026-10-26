"""Replay client: real HTTP fetches, a per-session browser cache, measured waits.

Each session is one emulated browser. Its logical cache **is**
:class:`webprefetch.cachecore.Cache`, the class the simulator replays, so
per-session hits, issued and useful counts can be compared with ``sim.json``
exactly. :class:`SessionCache` only adds what the simulator cannot see, the
physical state of each resident object's fetch (dropped through the cache's
``on_evict`` hook):

* ``ready``    -- the body had arrived before the view: wait 0
* ``inflight`` -- a prefetch was issued but the body is still arriving: the
  view waits for the remainder
* ``miss``     -- not in the cache: a full demand fetch

Requests go through a per-session :class:`Pool` of HTTP/1.1 keep-alive
connections (6 by default, at most 2 carrying speculative requests), in
which a waiting demand is always admitted before a waiting prefetch, as a
browser's idle priority for speculative fetches does. ``--pool fifo`` is the
ablation: one queue in arrival order.

The schedule replays each session's recorded inter-view gaps, capped
(``gap_cap_s``, 2 s by default) and scaled, from the session's start offset
(:func:`session_starts`). A view starts at ``max(previous scheduled time +
gap, previous view ready)``, so the schedule is the same for every policy
unless a wait outlasts a gap. Stdlib only.
"""

from __future__ import annotations

import asyncio
import random
from collections import deque
from dataclasses import dataclass

from ..cachecore import Cache, Counters
from .timing import CLOCK, LagMonitor, PreciseTimer

POOLS = ("priority", "fifo")
DEFAULT_CONNS = 6
DEFAULT_PREFETCH_CONNS = 2
DEFAULT_GAP_CAP_S = 2.0
DEFAULT_RAMP_S = 180.0
_NETWORK_ERRORS = (OSError, asyncio.IncompleteReadError, asyncio.LimitOverrunError, ValueError)


@dataclass
class ClientConfig:
    host: str = "127.0.0.1"
    port: int = 0
    capacity: int = 8
    conns: int = DEFAULT_CONNS
    prefetch_conns: int = DEFAULT_PREFETCH_CONNS
    pool: str = "priority"
    handshake_ms: float = 0.0
    gap_cap_s: float = DEFAULT_GAP_CAP_S
    think_scale: float = 1.0
    local_addr: str | None = None

    def __post_init__(self):
        if self.pool not in POOLS:
            raise ValueError(f"pool must be one of {POOLS}")
        if self.conns < 1 or not 1 <= self.prefetch_conns <= self.conns:
            raise ValueError("need conns >= 1 and 1 <= prefetch_conns <= conns")
        if self.capacity < 1 or self.gap_cap_s < 0 or self.think_scale < 0:
            raise ValueError("capacity must be positive; gap cap and think scale non-negative")


class Conn:
    __slots__ = ("reader", "writer")

    def __init__(self, reader, writer):
        self.reader, self.writer = reader, writer

    async def get(self, path: str, session: str, kind: str) -> tuple[int, dict]:
        self.writer.write(f"GET {path} HTTP/1.1\r\nHost: origin\r\nX-Session: {session}\r\n"
                          f"X-Kind: {kind}\r\n\r\n".encode())
        head = await self.reader.readuntil(b"\r\n\r\n")
        lines = head.decode("latin-1").split("\r\n")
        status = lines[0].split(" ")
        headers = dict(line.split(": ", 1) for line in lines[1:] if ": " in line)
        n = int(headers.get("Content-Length", "0"))
        if n:
            await self.reader.readexactly(n)
        if len(status) < 2 or status[1] not in ("200", "204"):
            raise ValueError(f"HTTP {lines[0]!r} for {path}")
        return n, headers

    def close(self) -> None:
        self.writer.close()


class Pool:
    """A session's connections: at most ``limit``, ``prefetch_limit`` speculative.

    ``acquire``/``release`` are the slot scheduler, kept apart from the
    sockets so the ordering can be tested on its own. A slot freed by
    ``release`` is handed directly to the next admissible waiter.
    """

    def __init__(self, host: str, port: int, limit: int = DEFAULT_CONNS,
                 prefetch_limit: int = DEFAULT_PREFETCH_CONNS, priority: bool = True,
                 handshake_s: float = 0.0, local_addr: str | None = None):
        self.host, self.port = host, port
        self.limit = limit
        # FIFO admits any kind on any connection: that is the ablation.
        self.prefetch_limit = prefetch_limit if priority else limit
        self.priority = priority
        self.handshake_s = handshake_s
        self.local_addr = (local_addr, 0) if local_addr else None
        self.active = 0
        self.speculative_active = 0
        self.waiting_demand: deque = deque()
        self.waiting_speculative: deque = deque()
        self.waiting: deque = deque()  # FIFO mode: (kind, future)
        self.idle: list[Conn] = []
        self.opened = 0
        self.admitted: list[str] = []  # admission order, for tests

    @staticmethod
    def speculative(kind: str) -> bool:
        return kind != "demand"

    def _can(self, kind: str) -> bool:
        return self.active < self.limit and (
            not self.speculative(kind) or self.speculative_active < self.prefetch_limit)

    def _take(self, kind: str) -> None:
        self.active += 1
        self.speculative_active += self.speculative(kind)
        self.admitted.append(kind)

    def _queued_ahead(self, kind: str) -> bool:
        if not self.priority:
            return bool(self.waiting)
        if self.speculative(kind):
            return bool(self.waiting_demand or self.waiting_speculative)
        return bool(self.waiting_demand)

    async def acquire(self, kind: str) -> None:
        if self._can(kind) and not self._queued_ahead(kind):
            self._take(kind)
            return
        future = asyncio.get_running_loop().create_future()
        if not self.priority:
            self.waiting.append((kind, future))
        elif self.speculative(kind):
            self.waiting_speculative.append((kind, future))
        else:
            self.waiting_demand.append((kind, future))
        await future  # the slot was taken for us by _dispatch

    def release(self, kind: str) -> None:
        self.active -= 1
        self.speculative_active -= self.speculative(kind)
        self._dispatch()

    def _dispatch(self) -> None:
        queues = (self.waiting,) if not self.priority else (self.waiting_demand,
                                                            self.waiting_speculative)
        for queue in queues:
            while queue:
                kind, future = queue[0]
                if future.done():  # cancelled while waiting
                    queue.popleft()
                    continue
                if not self._can(kind):
                    break
                queue.popleft()
                self._take(kind)
                future.set_result(None)
            if self.priority and queue is self.waiting_demand and queue:
                return  # a demand still waits: no prefetch may overtake it

    async def get(self, path: str, session: str, kind: str) -> tuple[int, dict]:
        await self.acquire(kind)
        conn = None
        try:
            if self.idle:
                conn = self.idle.pop()
            else:
                reader, writer = await asyncio.open_connection(
                    self.host, self.port, local_addr=self.local_addr)
                if self.handshake_s:
                    await asyncio.sleep(self.handshake_s)  # emulated TCP handshake (ASSUMED)
                conn = Conn(reader, writer)
                self.opened += 1
            result = await conn.get(path, session, kind)
            self.idle.append(conn)
            conn = None
            return result
        finally:
            if conn is not None:
                conn.close()  # a failed connection is never reused
            self.release(kind)

    def close(self) -> None:
        for conn in self.idle:
            conn.close()
        self.idle.clear()


class Fetch:
    """One HTTP fetch of an object; ``task`` resolves when its body arrived."""
    __slots__ = ("kind", "item", "task", "nbytes", "started", "done_at", "error")

    def __init__(self, kind: str, item: int):
        self.kind, self.item = kind, item
        self.task = None
        self.nbytes = 0
        self.started = self.done_at = None
        self.error = None

    @property
    def done(self) -> bool:
        return self.task is not None and self.task.done()


class SessionCache:
    """:class:`cachecore.Cache` plus the fetch behind every resident object."""

    def __init__(self, capacity: int):
        self.counters = Counters()
        self.fetches: dict[int, Fetch] = {}
        self.cache = Cache(capacity, self.counters, on_evict=self._evicted)
        self.used_prefetches: list[Fetch] = []

    def _evicted(self, item: int, speculative: bool) -> None:
        # The bytes of an evicted in-flight prefetch still cross the link;
        # the session keeps its task and drains it at the end.
        self.fetches.pop(item, None)

    def demand(self, item: int, start_fetch) -> tuple[bool, Fetch]:
        """(hit, fetch). On a miss ``start_fetch()`` returns the new demand fetch."""
        speculative = bool(self.cache.blocks.get(item))
        if self.cache.demand(item):
            fetch = self.fetches[item]
            if speculative:
                self.used_prefetches.append(fetch)
            return True, fetch
        fetch = self.fetches[item] = start_fetch()
        return False, fetch

    def prefetch(self, item: int, start_fetch) -> Fetch | None:
        """Start a speculative fetch unless ``item`` is resident (then ``None``)."""
        if not self.cache.prefetch(item):
            return None
        fetch = self.fetches[item] = start_fetch()
        return fetch

    def check(self) -> None:
        """Logical and physical state agree (tests)."""
        if set(self.fetches) != set(self.cache.blocks):
            raise AssertionError("session cache entries diverged from cachecore.Cache")

    def record(self) -> list[int]:
        c = self.counters
        return [c.hits, c.issued, c.views, c.useful]


FETCH_FIELDS = ("kind", "item", "bytes", "start_s", "done_s", "server_total_us", "disk_us",
                "queue_us", "server_lag_us", "model_us", "error")
VIEW_FIELDS = ("k", "item", "cls", "wait_s", "late_s", "view_s")
_TIMING_HEADERS = ("X-Server-Total-Us", "X-Disk-Us", "X-Queue-Us", "X-Server-Lag-Us",
                   "X-Model-Us")


async def _fetch(pool: Pool, fetch: Fetch, session: str, records: list, t0: float) -> Fetch:
    """Fetch one object; append a :data:`FETCH_FIELDS` record (times from ``t0``)."""
    fetch.started = CLOCK()
    headers: dict = {}
    try:
        fetch.nbytes, headers = await pool.get(f"/item/{fetch.item}", session, fetch.kind)
    except _NETWORK_ERRORS as error:
        fetch.error = f"{type(error).__name__}: {error}"
    fetch.done_at = CLOCK()
    records.append((fetch.kind, fetch.item, fetch.nbytes, fetch.started - t0, fetch.done_at - t0,
                    *(int(headers.get(name, 0)) for name in _TIMING_HEADERS),
                    int(fetch.error is not None)))
    return fetch


async def _hint(pool: Pool, items: list[int], session: str, records: list, t0: float) -> None:
    """Ask the origin to WILLNEED ``items``; recorded as a zero-byte ``hint`` fetch."""
    started = CLOCK()
    error = 0
    try:
        await pool.get("/hint?ids=" + ",".join(map(str, items)), session, "hint")
    except _NETWORK_ERRORS:
        error = 1
    records.append(("hint", -1, 0, started - t0, CLOCK() - t0, 0, 0, 0, 0, 0, error))


async def run_session(sid: str, views: list[dict], policy: str, config: ClientConfig,
                      start_at: float, timer: PreciseTimer, t0: float, *,
                      hint: bool = False, tag: str = "") -> dict:
    """Replay one session from ``start_at`` (a CLOCK reading); return its record.

    ``policy`` names the plan's candidate lists. ``hint=True`` sends those
    candidates to the origin's ``/hint`` (server page-cache prefetch) and
    prefetches nothing into the browser cache.
    """
    session = f"{tag}{policy}:{sid}" if not hint else f"{tag}hint:{policy}:{sid}"
    pool = Pool(config.host, config.port, config.conns, config.prefetch_conns,
                config.pool == "priority", config.handshake_ms / 1000, config.local_addr)
    cache = SessionCache(config.capacity)
    fetches: list[tuple] = []
    view_records: list[tuple] = []
    tasks: list = []
    loop = asyncio.get_running_loop()

    def start(kind: str, item: int) -> Fetch:
        fetch = Fetch(kind, item)
        fetch.task = loop.create_task(_fetch(pool, fetch, session, fetches, t0))
        return fetch

    await timer.sleep_until(start_at)
    scheduled = CLOCK()
    previous_ready = scheduled
    for k, view in enumerate(views):
        if k:
            think = min(view["gap_ms"] / 1000, config.gap_cap_s) * config.think_scale
            scheduled = max(scheduled + think, previous_ready)
            await timer.sleep_until(scheduled)
        t_view = CLOCK()
        item = view["item"]
        hit, fetch = cache.demand(item, lambda item=item: start("demand", item))
        if hit and fetch.done:
            cls, wait = "ready", 0.0
        else:
            cls = "inflight" if hit else "miss"
            await asyncio.shield(fetch.task)
            wait = max(0.0, fetch.done_at - t_view)
        previous_ready = CLOCK()
        view_records.append((k, item, cls, wait, t_view - scheduled, t_view - t0))
        candidates = view["pf"].get(policy, ())
        if hint:
            if candidates:
                tasks.append(loop.create_task(_hint(pool, list(candidates), session, fetches, t0)))
            continue
        for candidate in candidates:
            started = cache.prefetch(candidate, lambda c=candidate: start("prefetch", c))
            if started is not None:
                tasks.append(started.task)
    if tasks:
        await asyncio.gather(*tasks)  # wasted prefetch bytes still crossed the link
    pool.close()
    demand_bytes = sum(r[2] for r in fetches if r[0] == "demand")
    prefetch_bytes = sum(r[2] for r in fetches if r[0] == "prefetch")
    useful_bytes = sum(f.nbytes for f in cache.used_prefetches)
    return {"sid": sid, "policy": policy, "hint": hint, "counts": cache.record(),
            "views": view_records, "fetches": fetches, "demand_bytes": demand_bytes,
            "prefetch_bytes": prefetch_bytes, "useful_prefetch_bytes": useful_bytes,
            "conns": pool.opened, "errors": sum(r[10] for r in fetches)}


async def _run_pass(sessions: dict, policy: str, config: ClientConfig, starts: dict,
                    t0: float, hint: bool, tag: str) -> tuple[list[dict], dict]:
    loop = asyncio.get_running_loop()
    timer = PreciseTimer(loop)
    monitor = LagMonitor(timer)
    watch = loop.create_task(monitor.run())
    try:
        results = await asyncio.gather(*(
            run_session(sid, views, policy, config, t0 + starts[sid], timer, t0,
                        hint=hint, tag=tag)
            for sid, views in sessions.items()))
    finally:
        watch.cancel()
        timer.close()
    return list(results), monitor.summary()


def run_pass(sessions: dict, policy: str, config: ClientConfig, starts: dict, *,
             t0: float | None = None, hint: bool = False, tag: str = "") -> tuple[list, dict]:
    """Replay ``sessions`` ({sid: views}) concurrently in a fresh event loop.

    Session ``sid`` starts at ``t0 + starts[sid]``. Returns the per-session
    records and the client loop-lag summary.
    """
    t0 = CLOCK() + 0.2 if t0 is None else t0
    return asyncio.run(_run_pass(sessions, policy, config, starts, t0, hint, tag))


def run_shard(payload: dict) -> tuple[list, dict]:
    """Process-pool entry point: one shard of a pass (see ``run.py``)."""
    config = ClientConfig(**payload["config"])
    return run_pass(payload["sessions"], payload["policy"], config, payload["starts"],
                    t0=payload["t0"], hint=payload["hint"], tag=payload["tag"])


def session_starts(sessions: dict, *, ramp_s: float = DEFAULT_RAMP_S, seed: int = 1,
                   target_rps: float | None = None) -> dict[str, float]:
    """Start offset (s) of every session, identical for every policy.

    Uniform over ``[0, ramp_s]`` by default. With ``target_rps``, sessions
    start one after another (seeded order) so that planned views begin at
    about ``target_rps`` per second.
    """
    ids = sorted(sessions, key=int)
    rng = random.Random(seed)
    if target_rps is None:
        if ramp_s < 0:
            raise ValueError("ramp must be non-negative")
        return {sid: rng.uniform(0, ramp_s) for sid in ids}
    if target_rps <= 0:
        raise ValueError("target rate must be positive")
    rng.shuffle(ids)
    starts, views = {}, 0
    for sid in ids:
        starts[sid] = views / target_rps
        views += len(sessions[sid])
    return starts


def planned_requests(views: list[dict], policy: str, capacity: int, *,
                     hint: bool = False) -> list[int]:
    """Requests each view sends under the plan (logical cache, no timing)."""
    cache = Cache(capacity, Counters())
    out = []
    for view in views:
        n = 0 if cache.demand(view["item"]) else 1
        candidates = view["pf"].get(policy, ())
        if hint:
            n += bool(candidates)
        else:
            n += sum(cache.prefetch(c) for c in candidates)
        out.append(n)
    return out


def planned_rate(sessions: dict, policy: str, starts: dict, config: ClientConfig, *,
                 hint: bool = False, bin_s: float = 1.0) -> dict:
    """Peak and mean offered request rate of the open-loop schedule.

    Every request of a view is placed at the view's scheduled time, so the
    peak is an upper bound on any one-second window the client offers.
    """
    bins: dict[int, int] = {}
    total = 0
    end = 0.0
    for sid, views in sessions.items():
        t = starts[sid]
        for k, (view, n) in enumerate(zip(views, planned_requests(views, policy, config.capacity,
                                                                  hint=hint))):
            if k:
                t += min(view["gap_ms"] / 1000, config.gap_cap_s) * config.think_scale
            bins[int(t // bin_s)] = bins.get(int(t // bin_s), 0) + n
            total += n
            end = max(end, t)
    peak = max(bins.values(), default=0) / bin_s
    return {"peak_rps": peak, "mean_rps": total / end if end > 0 else float(total),
            "requests": total, "schedule_s": end}
