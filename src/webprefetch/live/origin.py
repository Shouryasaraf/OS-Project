"""Origin server: asyncio HTTP/1.1 with an emulated per-session access link.

Timeline of one ``GET /item/<id>``::

    arrive -> wait rtt/2 -> [optional exponential server delay]
           -> REAL os.pread() of the object from the store (page cache or device)
           -> enqueue on the session's link -> transmitted, non-preemptive, at
              size/bandwidth (FIFO, or demand before prefetch with --link priority)
           -> wait rtt/2 -> headers + body

The disk read is real; RTT, bandwidth and server delay are emulated and
**ASSUMED**. Every response carries its timing as headers (microseconds):
``X-Disk-Us``, ``X-Queue-Us`` (time waiting for the link), ``X-Server-Lag-Us``
(how late the delivery deadline was serviced), ``X-Server-Total-Us`` and
``X-Model-Us`` (scheduled delivery minus arrival).

Other endpoints: ``GET /hint?ids=1,2`` issues chunked ``POSIX_FADV_WILLNEED``
on those objects (server-side page-cache prefetch, nothing is sent), and
``GET /stats[?reset=1]`` returns serving counters, event-loop lag and this
process's ``/proc/self/io`` read bytes as JSON (``reset=1`` clears them after
reporting).

Run: ``python -m webprefetch.live origin --store S --index I``. It prints
one JSON line with the bound port once it is ready. Stdlib only.
"""

from __future__ import annotations

import argparse
import asyncio
import heapq
import http.client
import itertools
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from .objects import load_index, process_read_bytes, willneed
from .timing import CLOCK, LagMonitor, PreciseTimer

LINKS = ("fifo", "priority")
DEFAULT_IO_THREADS = 8


class Link:
    """Non-preemptive store-and-forward link of one session.

    Start and end times are computed from the *scheduled* end of the previous
    transmission, so timer jitter never accumulates along a queue. ``busy``
    is set before the server task is created: setting it inside the task (as
    the prototype did) let two sends in one loop tick start two servers.
    """

    def __init__(self, bytes_per_s: float, timer, priority: bool):
        self.bytes_per_s = bytes_per_s
        self.timer = timer
        self.priority = priority
        self.queue: list = []
        self.sequence = itertools.count()
        self.free = 0.0
        self.busy = False
        self.schedule: list[tuple[float, float]] | None = None  # (start, end), tests only

    async def send(self, nbytes: int, demand: bool) -> tuple[float, float, float]:
        """Queue ``nbytes``; return (ready, start, end) once transmitted."""
        ready = CLOCK()
        future = asyncio.get_running_loop().create_future()
        rank = 0 if (demand or not self.priority) else 1
        heapq.heappush(self.queue, (rank, next(self.sequence), nbytes, ready, future))
        if not self.busy:
            self.busy = True
            asyncio.ensure_future(self._serve())
        start, end = await future
        return ready, start, end

    async def _serve(self) -> None:
        try:
            while self.queue:
                _, _, nbytes, ready, future = heapq.heappop(self.queue)
                start = max(ready, self.free)
                self.free = end = start + nbytes / self.bytes_per_s
                if self.schedule is not None:
                    self.schedule.append((start, end))
                await self.timer.sleep_until(end)
                if not future.done():
                    future.set_result((start, end))
        finally:
            self.busy = False


def _reply(status: str, body: bytes = b"", extra: str = "") -> bytes:
    return (f"HTTP/1.1 {status}\r\nContent-Length: {len(body)}\r\n{extra}\r\n").encode() + body


class Origin:
    def __init__(self, store: str, index: dict, loop, *, rtt_ms: float, mbps: float,
                 link: str = "fifo", server_delay_ms: float = 0.0,
                 io_threads: int = DEFAULT_IO_THREADS, seed: int = 7,
                 inject_block_ms: float = 0.0):
        if link not in LINKS:
            raise ValueError(f"link must be one of {LINKS}")
        if rtt_ms < 0 or mbps <= 0:
            raise ValueError("rtt must be >= 0 and bandwidth > 0")
        self.fd = os.open(store, os.O_RDONLY)
        self.index = index
        self.half_rtt = rtt_ms / 2000
        self.bytes_per_s = mbps * 1e6 / 8
        self.server_delay = server_delay_ms / 1000
        self.inject_block = inject_block_ms / 1000
        self.rng = random.Random(seed)
        self.timer = PreciseTimer(loop)
        self.monitor = LagMonitor(self.timer)
        self.priority = link == "priority"
        self.config = {"rtt_ms": rtt_ms, "mbps": mbps, "link": link,
                       "server_delay_ms": server_delay_ms, "io_threads": io_threads,
                       "inject_block_ms": inject_block_ms}
        self.links: dict[str, Link] = {}
        self.pool = ThreadPoolExecutor(io_threads, thread_name_prefix="origin-io")
        self._reset_counters()

    def _reset_counters(self) -> None:
        self.served = self.bytes = self.hints = self.hint_calls = 0
        self.bad_requests = self.connection_errors = 0
        self.read_bytes_start = process_read_bytes()

    def _read(self, item: int) -> tuple[bytes, float]:
        offset, size = self.index[item]
        started = CLOCK()
        data = os.pread(self.fd, size, offset)
        return data, CLOCK() - started

    def _hint(self, items) -> int:
        calls = 0
        for item in items:
            if item in self.index:
                offset, size = self.index[item]
                calls += willneed(self.fd, offset, size)
        return calls

    def stats(self, reset: bool = False) -> dict:
        read_now = process_read_bytes()
        out = {"served": self.served, "bytes": self.bytes, "hints": self.hints,
               "hint_calls": self.hint_calls, "links": len(self.links),
               "bad_requests": self.bad_requests, "connection_errors": self.connection_errors,
               "read_bytes": (read_now - self.read_bytes_start
                              if read_now is not None and self.read_bytes_start is not None
                              else None),
               "pid": os.getpid(), **self.monitor.summary(), "config": self.config}
        if reset:
            self.monitor.reset()
            self.links.clear()
            self._reset_counters()
        return out

    async def handle(self, reader, writer) -> None:
        loop = asyncio.get_running_loop()
        try:
            while True:
                try:
                    head = await reader.readuntil(b"\r\n\r\n")
                except asyncio.IncompleteReadError:
                    return  # the client closed an idle keep-alive connection
                lines = head.decode("latin-1").split("\r\n")
                parts = lines[0].split(" ")
                if len(parts) < 2 or parts[0] != "GET":
                    self.bad_requests += 1
                    writer.write(_reply("400 Bad Request"))
                    await writer.drain()
                    continue
                path = parts[1]
                headers = dict(line.split(": ", 1) for line in lines[1:] if ": " in line)
                if path.startswith("/hint?ids="):
                    ids = [int(x) for x in path[len("/hint?ids="):].split(",") if x]
                    self.hint_calls += await loop.run_in_executor(self.pool, self._hint, ids)
                    self.hints += len(ids)
                    writer.write(_reply("204 No Content"))
                    await writer.drain()
                    continue
                if path.startswith("/stats"):
                    body = json.dumps(self.stats(reset="reset=1" in path)).encode()
                    writer.write(_reply("200 OK", body, "Content-Type: application/json\r\n"))
                    await writer.drain()
                    continue
                try:
                    item = int(path.rsplit("/", 1)[-1]) if path.startswith("/item/") else None
                except ValueError:
                    item = None
                if item is None or item not in self.index:
                    self.bad_requests += 1
                    writer.write(_reply("404 Not Found"))
                    await writer.drain()
                    continue
                session = headers.get("X-Session", "")
                demand = headers.get("X-Kind", "demand") == "demand"
                arrive = CLOCK()
                if self.inject_block:
                    time.sleep(self.inject_block)  # deliberately stalls the loop (tests)
                await self.timer.sleep_until(arrive + self.half_rtt)
                if self.server_delay:
                    await self.timer.sleep_until(
                        CLOCK() + self.rng.expovariate(1 / self.server_delay))
                body, disk_s = await loop.run_in_executor(self.pool, self._read, item)
                link = self.links.get(session)
                if link is None:
                    link = self.links[session] = Link(self.bytes_per_s, self.timer, self.priority)
                ready, start, end = await link.send(len(body), demand)
                deliver = end + self.half_rtt
                await self.timer.sleep_until(deliver)
                now = CLOCK()
                writer.write(
                    (f"HTTP/1.1 200 OK\r\nContent-Length: {len(body)}\r\n"
                     f"X-Disk-Us: {disk_s * 1e6:.0f}\r\nX-Queue-Us: {(start - ready) * 1e6:.0f}\r\n"
                     f"X-Server-Lag-Us: {(now - deliver) * 1e6:.0f}\r\n"
                     f"X-Server-Total-Us: {(now - arrive) * 1e6:.0f}\r\n"
                     f"X-Model-Us: {(deliver - arrive) * 1e6:.0f}\r\n\r\n").encode())
                writer.write(body)
                await writer.drain()
                self.served += 1
                self.bytes += len(body)
        except (ConnectionResetError, BrokenPipeError):
            self.connection_errors += 1
        finally:
            writer.close()

    def close(self) -> None:
        self.timer.close()
        self.pool.shutdown(wait=False)
        os.close(self.fd)


def origin_stats(host: str, port: int, reset: bool = False) -> dict:
    """``GET /stats`` from a running origin (blocking; for the orchestrator)."""
    conn = http.client.HTTPConnection(host, port, timeout=30)
    try:
        conn.request("GET", "/stats?reset=1" if reset else "/stats")
        return json.loads(conn.getresponse().read())
    finally:
        conn.close()


async def serve(args: argparse.Namespace) -> None:
    index = load_index(args.index)
    loop = asyncio.get_running_loop()
    origin = Origin(args.store, index, loop, rtt_ms=args.rtt_ms, mbps=args.mbps,
                    link=args.link, server_delay_ms=args.server_delay_ms,
                    io_threads=args.io_threads, seed=args.seed,
                    inject_block_ms=args.inject_block_ms)
    monitor = asyncio.ensure_future(origin.monitor.run())
    server = await asyncio.start_server(origin.handle, args.host, args.port, backlog=8192)
    port = server.sockets[0].getsockname()[1]
    print(json.dumps({"ready": True, "host": args.host, "port": port, "pid": os.getpid(),
                      **origin.config}), flush=True)
    try:
        async with server:
            await server.serve_forever()
    finally:
        monitor.cancel()
        origin.close()


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--store", required=True, help="store file from build_store")
    parser.add_argument("--index", required=True, help="its JSON index")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0, help="0 picks a free port")
    parser.add_argument("--rtt-ms", type=float, default=50.0, help="ASSUMED round-trip time")
    parser.add_argument("--mbps", type=float, default=10.0, help="ASSUMED access bandwidth")
    parser.add_argument("--server-delay-ms", type=float, default=0.0,
                        help="mean of an exponential server think time (ASSUMED)")
    parser.add_argument("--link", choices=LINKS, default="priority")
    parser.add_argument("--io-threads", type=int, default=DEFAULT_IO_THREADS)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--inject-block-ms", type=float, default=0.0,
                        help="block the event loop this long per item request "
                             "(validity-gate tests only)")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_arguments(parser)
    args = parser.parse_args(argv)
    try:
        asyncio.run(serve(args))
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
