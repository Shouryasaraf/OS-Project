"""Backend B: a user-space LRU over O_DIRECT reads, faithful to the simulator.

The cache is a :class:`~.decisions.ShadowLRU` with exactly ``LRUCache``'s
rules. On every demanded block the backend **asserts** that its hit/miss
equals the plan's flag and raises :class:`~.decisions.CacheDivergence`
otherwise, so a finished run has the simulator's block-level hit/miss
sequence and ``useful_prefetches`` by construction. What it measures is what
the simulator assumes away: a prefetch takes time.

* Demand misses of one request are coalesced into file-contiguous runs and
  read with ``O_DIRECT`` ``preadv`` in the caller's thread.
* Prefetch runs go to a ``ThreadPoolExecutor`` (4 workers). Their blocks enter
  the cache at submission with the run's future attached, and a demand that
  hits a block whose future is not done waits for it: a **late hit**, with the
  wait time recorded. A hit on a completed prefetch is a **ready hit**.
* Buffers come from a preallocated pool of 1 MiB page-aligned ``mmap`` slabs
  (one per worker plus the demand buffer), so the timed path never allocates.
* Lookup and LRU bookkeeping are timed separately (``cpu_bookkeeping_us``),
  so the Python cost of the user-space cache can be subtracted.

No data is written to the device; a trace write is applied to the cache
model only.
"""

from __future__ import annotations

import mmap
import os
import threading
import time
from array import array
from concurrent.futures import ThreadPoolExecutor

from . import BLOCK
from .clock import Pacer
from .decisions import CacheDivergence, PlanStep, ShadowLRU

SLAB = 1 << 20


class _Slabs:
    """One page-aligned 1 MiB slab per worker thread, assigned on first use."""

    def __init__(self, count: int):
        self._pool = mmap.mmap(-1, SLAB * count)
        self._free = [memoryview(self._pool)[i * SLAB:(i + 1) * SLAB] for i in range(count)]
        self._lock = threading.Lock()
        self._local = threading.local()

    def get(self) -> memoryview:
        slab = getattr(self._local, "slab", None)
        if slab is None:
            with self._lock:
                slab = self._local.slab = self._free.pop()
        return slab


class ODirectBackend:
    name = "odirect"

    def __init__(self, path, mapper, capacity: int, *, workers: int = 4):
        self.path = str(path)
        self.fd = os.open(self.path, os.O_RDONLY | os.O_DIRECT)
        self.mapper = mapper
        self.cache = ShadowLRU(capacity)
        self.workers = workers
        self.slabs = _Slabs(workers)
        self.pool = ThreadPoolExecutor(workers, thread_name_prefix="prefetch")
        self.lock = threading.Lock()
        self.inflight = 0
        self.max_inflight = 0
        self.demand_bytes = 0
        self.prefetch_bytes = 0

    def close(self) -> None:
        self.pool.shutdown(wait=True)
        os.close(self.fd)

    # ---------------------------------------------------------------- I/O
    def _read(self, buf: memoryview, offset: int, nbytes: int) -> None:
        done = 0
        while done < nbytes:
            n = min(len(buf), nbytes - done)
            got = os.preadv(self.fd, [buf[:n]], offset + done)
            if got != n:
                raise OSError(f"short O_DIRECT read at {offset + done}: {got} of {n}")
            done += n

    def _prefetch_job(self, offset: int, nbytes: int) -> None:
        with self.lock:
            self.inflight += 1
            if self.inflight > self.max_inflight:
                self.max_inflight = self.inflight
        try:
            self._read(self.slabs.get(), offset, nbytes)
        finally:
            with self.lock:
                self.inflight -= 1

    # ---------------------------------------------------------------- run
    def run(self, requests, plan: list[PlanStep], gaps, pacer: Pacer | None = None) -> dict:
        """Execute ``plan``; returns the raw measurements (see ``runner``).

        Latency is sampled for reads only: a trace write touches the cache
        model and no device, so timing it would only dilute the mean.
        """
        pacer = pacer or Pacer()
        cache, mapper = self.cache, self.mapper
        lat = array("d")
        demand = mmap.mmap(-1, SLAB)
        demand_view = memoryview(demand)
        ready_hits = late_hits = full_hits = 0
        late_wait_us = bookkeeping_us = hit_lat = miss_lat = 0.0
        read_blocks = missed_blocks = 0
        perf = time.perf_counter
        for i, (request, step) in enumerate(zip(requests, plan)):
            pacer.wait(gaps[i])
            if request.operation != "R":
                tb = perf()
                for block in range(request.lba, request.lba + request.size_blocks):
                    cache.write(block)
                self._submit(step)
                bookkeeping_us += (perf() - tb) * 1e6
                continue
            t0 = perf()
            missed: list[int] = []
            waits = set()
            flags = step.hit_flags
            block = request.lba
            for k in range(request.size_blocks):
                entry = cache.read(block)
                if (entry is not None) != bool(flags[k]):
                    raise CacheDivergence(
                        f"request {i} block {block}: backend "
                        f"{'hit' if entry is not None else 'miss'}, plan "
                        f"{'hit' if flags[k] else 'miss'}")
                if entry is None:
                    missed.append(block)
                elif entry[0] is not None:
                    if entry[0].done():
                        entry[0] = None
                    else:
                        waits.add(entry[0])
                block += 1
            bookkeeping_us += (perf() - t0) * 1e6
            for offset, nbytes in mapper.runs(missed):
                self._read(demand_view, offset, nbytes)
                self.demand_bytes += nbytes
            if waits:
                tw = perf()
                for future in waits:
                    future.result()
                late_wait_us += (perf() - tw) * 1e6
                late_hits += 1
            elapsed = (perf() - t0) * 1e6
            lat.append(elapsed)
            read_blocks += request.size_blocks
            missed_blocks += len(missed)
            if missed:
                miss_lat += elapsed
            else:
                full_hits += 1
                hit_lat += elapsed
                if not waits:
                    ready_hits += 1
            # Prefetch submission is asynchronous and outside the demand
            # latency, as a kernel read-ahead submission would be.
            tb = perf()
            self._submit(step)
            bookkeeping_us += (perf() - tb) * 1e6
        self.pool.shutdown(wait=True)
        demand_view.release()
        demand.close()
        miss_requests = len(lat) - full_hits
        return {
            "latencies_us": lat,
            "read_blocks": read_blocks, "missed_blocks": missed_blocks,
            "real_hit_ratio": (read_blocks - missed_blocks) / read_blocks if read_blocks else 0.0,
            "full_hits": full_hits, "ready_hits": ready_hits, "late_hits": late_hits,
            "late_wait_us": late_wait_us,
            "hit_lat_mean_us": hit_lat / full_hits if full_hits else None,
            "miss_lat_mean_us": miss_lat / miss_requests if miss_requests else None,
            "prefetches": cache.prefetches, "useful_prefetches": cache.useful,
            "demand_bytes": self.demand_bytes, "prefetch_bytes": self.prefetch_bytes,
            "max_inflight": self.max_inflight,
            "cpu_bookkeeping_us": bookkeeping_us,
        }

    def _submit(self, step: PlanStep) -> None:
        cache, mapper = self.cache, self.mapper
        for first, count in step.prefetch_runs:
            for offset, nbytes, block in mapper.range_runs(first, count):
                # One job per slab-sized piece, so a worker's buffer suffices.
                for piece in range(0, nbytes, SLAB):
                    size = min(SLAB, nbytes - piece)
                    future = self.pool.submit(self._prefetch_job, offset + piece, size)
                    self.prefetch_bytes += size
                    for b in range(block + piece // BLOCK, block + (piece + size) // BLOCK):
                        if not cache.prefetch(b, future):
                            raise CacheDivergence(f"planned prefetch of resident block {b}")
