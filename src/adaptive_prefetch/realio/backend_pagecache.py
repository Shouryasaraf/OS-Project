"""Backend A: the real kernel page cache, bounded by a memory cgroup.

Demand reads are buffered ``preadv`` calls; prefetches are
``POSIX_FADV_WILLNEED`` hints from a helper thread. The kernel decides what
stays resident (MGLRU, 4 KiB pages), so this backend's hit ratio is a
**measured divergence** from the simulator, reported as such and never
asserted equal.

Kernel facts this relies on, each measured on the test host (kernel 6.14):

* One ``WILLNEED`` (or ``readahead(2)``) call is truncated to
  ``max(read_ahead_kb, max_sectors_kb)`` -- 256 KiB here -- so a longer hint
  must be issued in chunks or most of it is silently dropped.
* ``POSIX_FADV_RANDOM`` turns kernel read-ahead off for this descriptor, so
  only the policy's hints prefetch; ``kernel_ra`` (``FADV_NORMAL``) and
  ``kernel_seq`` (``FADV_SEQUENTIAL``) measure the kernel's own read-ahead.
* ``cachestat(2)`` counts a page that is still under read I/O as cached; a
  ``preadv2(RWF_NOWAIT)`` probe fails on it. Together they classify a request
  as a **ready** hit, a **late** hit (resident but still in flight) or a miss.
  The probe does not start I/O for an absent page under ``FADV_RANDOM``.
* A page already cached and charged to another cgroup is **not** re-charged
  when this cgroup reads it, so the whole backing file is dropped with
  ``FADV_DONTNEED`` before every run, or the budget would not bind.
* The cache is bounded by the cgroup's ``memory.max``. Once everything else
  is allocated the run sets it to the cgroup's non-file memory plus
  ``capacity + BUDGET_OFFSET``, so every file page the cgroup holds -- the
  backing file's and any other -- shares one budget the size of the
  simulated cache. ``memory.stat file`` at the end is recorded so the
  realised budget is on file for every row.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from array import array
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import BLOCK, PAGE
from .clock import Pacer
from .decisions import PlanStep
from .devstat import device_info
from .syscalls import residency_counter

DEFAULT_WILLNEED_CHUNK = 256 << 10
#: Bytes added to the non-file baseline plus the capacity when setting
#: ``memory.max``. Measured with a 32 MiB capacity and 256 MiB of reads: offset
#: 0 left 31.25 MiB of file pages resident, 1 MiB left 32.75 and 2 MiB 33.25,
#: so 1 MiB is the offset at which the page-cache budget matches the simulated
#: capacity to within a few percent. ``cg_file_MiB`` records the outcome.
BUDGET_OFFSET = 1 << 20
#: Headroom for the scope's initial MemoryMax, before the run tightens it.
SCOPE_HEADROOM = 256 << 20

ADVICE = {"random": os.POSIX_FADV_RANDOM, "normal": os.POSIX_FADV_NORMAL,
          "sequential": os.POSIX_FADV_SEQUENTIAL} if hasattr(os, "POSIX_FADV_RANDOM") else {}


class CgroupUnavailable(RuntimeError):
    """No user-delegated memory cgroup to bound the page cache with."""


# -------------------------------------------------------------------- cgroup

_SYSTEMD_RUN_PROBE: tuple[bool, str] | None = None


def systemd_run_available() -> tuple[bool, str]:
    """Whether ``systemd-run --user --scope`` works here (probed once)."""
    global _SYSTEMD_RUN_PROBE
    if _SYSTEMD_RUN_PROBE is None:
        exe = shutil.which("systemd-run")
        if exe is None:
            _SYSTEMD_RUN_PROBE = (False, "systemd-run is not installed")
        else:
            try:
                done = subprocess.run([exe, "--user", "--scope", "-q", "-p",
                                       "MemoryMax=64M", "true"],
                                      capture_output=True, text=True, timeout=30)
                _SYSTEMD_RUN_PROBE = ((True, "") if done.returncode == 0 else
                                      (False, "systemd-run --user failed: "
                                       + (done.stderr.strip() or f"exit {done.returncode}")))
            except (OSError, subprocess.SubprocessError) as error:
                _SYSTEMD_RUN_PROBE = (False, f"systemd-run --user failed: {error}")
    return _SYSTEMD_RUN_PROBE


def scope_command(argv: list[str], capacity_blocks: int) -> list[str]:
    """``argv`` wrapped in a transient user scope with a memory limit.

    Raises :class:`CgroupUnavailable` when ``systemd-run --user`` cannot be
    used, so the caller can skip Backend A cleanly instead of running it
    unbounded.
    """
    ok, reason = systemd_run_available()
    if not ok:
        raise CgroupUnavailable(reason)
    limit = capacity_blocks * BLOCK + SCOPE_HEADROOM
    return [shutil.which("systemd-run") or "systemd-run", "--user", "--scope", "-q",
            "-p", f"MemoryMax={limit}", "-p", "MemorySwapMax=0", *argv]


def own_cgroup() -> Path | None:
    try:
        text = Path("/proc/self/cgroup").read_text().strip()
    except OSError:
        return None
    for line in text.splitlines():
        if line.startswith("0::"):
            return Path("/sys/fs/cgroup" + line[3:])
    return None


class CgroupBudget:
    """This process's cgroup, tightened to a page-cache budget."""

    def __init__(self, path: Path | None = None):
        self.path = path or own_cgroup()
        if self.path is None or not (self.path / "memory.max").exists():
            raise CgroupUnavailable("not in a cgroup v2 with the memory controller")
        if (self.path / "memory.max").read_text().strip() == "max":
            raise CgroupUnavailable(f"{self.path} has no memory limit; run under "
                                    "systemd-run --user --scope -p MemoryMax=...")
        self.limit: int | None = None

    def _stat(self) -> dict:
        return {key: int(value) for key, value in
                (line.split() for line in (self.path / "memory.stat").read_text().splitlines())}

    def tighten(self, capacity_bytes: int, offset: int = BUDGET_OFFSET) -> int:
        """Set ``memory.max`` to non-file usage + ``capacity_bytes`` + ``offset``."""
        current = int((self.path / "memory.current").read_text())
        baseline = current - self._stat().get("file", 0)
        self.limit = baseline + capacity_bytes + offset
        try:
            (self.path / "memory.max").write_text(str(self.limit))
        except OSError as error:
            raise CgroupUnavailable(f"cannot write {self.path}/memory.max: {error}") from error
        return self.limit

    def stats(self) -> dict:
        stat = self._stat()
        events = dict(line.split() for line in
                      (self.path / "memory.events").read_text().splitlines())
        return {"cg_file_MiB": int(stat.get("file", 0)) / (1 << 20),
                "cg_events_max": int(events.get("max", 0)),
                "cg_limit_MiB": self.limit / (1 << 20) if self.limit else None}


# ------------------------------------------------------------------ hints

def willneed_chunk(disk: str | None) -> int:
    """Largest ``WILLNEED`` the kernel honours in one call on ``disk``."""
    info = device_info(disk)
    sizes = [kb << 10 for kb in (info["read_ahead_kb"], info["max_sectors_kb"]) if kb]
    return max(sizes) if sizes else DEFAULT_WILLNEED_CHUNK


def willneed(fd: int, offset: int, nbytes: int, chunk: int) -> int:
    """Hint ``[offset, offset+nbytes)`` in ``chunk``-sized calls; returns calls."""
    start = offset // PAGE * PAGE
    end = -(-(offset + nbytes) // PAGE) * PAGE
    calls = 0
    for pos in range(start, end, chunk):
        os.posix_fadvise(fd, pos, min(chunk, end - pos), os.POSIX_FADV_WILLNEED)
        calls += 1
    return calls


def ready(fd: int, offset: int, nbytes: int, buf) -> int:
    """Bytes of the range readable without blocking (``RWF_NOWAIT`` probe)."""
    try:
        return os.preadv(fd, [buf[:nbytes]], offset, os.RWF_NOWAIT)
    except BlockingIOError:
        return 0


# ---------------------------------------------------------------- backend

class _Pending:
    """Blocks hinted and not yet demanded, without growing during the run.

    Anonymous memory counts against the same cgroup budget as the page
    cache, so a set that grew by a million entries mid-run would quietly
    shrink the cache under test. A bytearray over the mapped regions is
    allocated and touched up front instead; only blocks in regions mapped
    later (online prefetches into new regions) fall back to a set.
    """

    def __init__(self, mapper, extra_regions: int = 0):
        from .layout import REGION_BLOCKS
        self.mapper = mapper
        self.span = (mapper.regions + extra_regions) * REGION_BLOCKS
        self.bits = bytearray(b"\x01") * self.span   # memset touches every page
        self.bits[:] = bytes(self.span)
        self.overflow: set[int] = set()

    def _index(self, block: int) -> int:
        return self.mapper.offset(block) // BLOCK

    def add(self, block: int) -> bool:
        i = self._index(block)
        if i < self.span:
            if self.bits[i]:
                return False
            self.bits[i] = 1
            return True
        if block in self.overflow:
            return False
        self.overflow.add(block)
        return True

    def take(self, block: int) -> bool:
        i = self._index(block)
        if i < self.span:
            was = self.bits[i]
            self.bits[i] = 0
            return bool(was)
        if block in self.overflow:
            self.overflow.discard(block)
            return True
        return False


class PageCacheBackend:
    name = "pagecache"

    def __init__(self, path, mapper, *, advice: str = "random", disk: str | None = None,
                 chunk: int | None = None, extra_regions: int = 0):
        if advice not in ADVICE:
            raise ValueError(f"advice must be one of {sorted(ADVICE)}")
        self.path = str(path)
        self.fd = os.open(self.path, os.O_RDONLY)
        # Mandatory: pages charged to another cgroup would not be re-charged.
        os.posix_fadvise(self.fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.posix_fadvise(self.fd, 0, 0, ADVICE[advice])
        self.advice = advice
        self.mapper = mapper
        self.chunk = chunk or willneed_chunk(disk)
        self.resident, self.residency_source = residency_counter(self.fd)
        self.helper = ThreadPoolExecutor(1, thread_name_prefix="willneed")
        self.pending = _Pending(mapper, extra_regions)
        self.prefetches = 0
        self.useful = 0
        self.prefetch_bytes = 0
        self.hint_calls = 0
        self.hint_errors: list[BaseException] = []
        self._buf = bytearray(1 << 20)
        self._view = memoryview(self._buf)

    def close(self) -> None:
        self.helper.shutdown(wait=True)
        os.posix_fadvise(self.fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(self.fd)

    def _ensure_buffer(self, nbytes: int) -> None:
        if nbytes > len(self._buf):
            self._view.release()
            self._buf = bytearray(nbytes)
            self._view = memoryview(self._buf)

    def residency(self, runs) -> tuple[list[bool], int, int]:
        """Per-page residency of the request's runs (before it is read)."""
        flags: list[bool] = []
        for offset, nbytes, _ in runs:
            p0, p1 = offset // PAGE, (offset + nbytes - 1) // PAGE
            count = p1 - p0 + 1
            cached = self.resident(self.fd, p0 * PAGE, count * PAGE)
            if cached == count or cached == 0:
                flags.extend([cached == count] * count)
            else:
                flags.extend(self.resident(self.fd, p * PAGE, PAGE) == 1
                             for p in range(p0, p1 + 1))
        return flags, sum(flags), len(flags)

    def demand(self, request) -> dict:
        """Read one request; returns its timing and classification."""
        runs = self.mapper.range_runs(request.lba, request.size_blocks)
        # Classification first, untimed: it observes, it does not read.
        page_flags, resident_pages, pages = self.residency(runs)
        block_hits = 0
        useful = 0
        index = 0
        for offset, nbytes, block in runs:
            for b in range(block, block + nbytes // BLOCK):
                off = offset + (b - block) * BLOCK
                hit = page_flags[index + (off // PAGE - offset // PAGE)]
                block_hits += hit
                if self.pending.take(b):
                    useful += hit
            index += (offset + nbytes - 1) // PAGE - offset // PAGE + 1
        self.useful += useful
        self._ensure_buffer(max(n for _, n, _ in runs))
        view = self._view
        waited_on_resident = False
        wait_us = 0.0
        perf = time.perf_counter
        t0 = perf()
        index = 0
        for offset, nbytes, _ in runs:
            count = (offset + nbytes - 1) // PAGE - offset // PAGE + 1
            got = ready(self.fd, offset, nbytes, view)
            if got < nbytes:
                tw = perf()
                while got < nbytes:
                    n = os.preadv(self.fd, [view[got:nbytes]], offset + got)
                    if n <= 0:
                        raise OSError(f"short read at {offset + got}")
                    got += n
                if all(page_flags[index:index + count]):
                    waited_on_resident = True
                    wait_us += (perf() - tw) * 1e6
            index += count
        elapsed = (perf() - t0) * 1e6
        full = resident_pages == pages
        return {"lat": elapsed, "blocks": request.size_blocks, "block_hits": block_hits,
                "full": full, "ready": full and not waited_on_resident,
                "late": waited_on_resident, "late_wait_us": wait_us, "useful": useful}

    def hint(self, blocks_runs) -> list[int]:
        """``WILLNEED`` the ``(first, count)`` block runs from the helper thread.

        Returns the blocks newly marked pending, so a caller can account them.
        """
        ranges = []
        new = []
        for first, count in blocks_runs:
            for b in range(first, first + count):
                if self.pending.add(b):
                    new.append(b)
            for offset, nbytes, _ in self.mapper.range_runs(first, count):
                ranges.append((offset, nbytes))
                start = offset // PAGE * PAGE
                self.prefetch_bytes += -(-(offset + nbytes) // PAGE) * PAGE - start
        self.prefetches += len(new)
        if ranges:
            self.helper.submit(self._hint_job, ranges)
        return new

    def _hint_job(self, ranges) -> None:
        try:
            for offset, nbytes in ranges:
                self.hint_calls += willneed(self.fd, offset, nbytes, self.chunk)
        except BaseException as error:      # its future is never awaited
            self.hint_errors.append(error)
            raise

    def check_hints(self) -> None:
        """Raise if a hint failed on the helper thread (call after shutdown)."""
        if self.hint_errors:
            raise OSError(f"{len(self.hint_errors)} WILLNEED hints failed; first: "
                          f"{self.hint_errors[0]!r}")

    def run(self, requests, plan: list[PlanStep] | None, gaps,
            pacer: Pacer | None = None) -> dict:
        """Replay ``requests``; ``plan`` is ``None`` for the kernel modes."""
        pacer = pacer or Pacer()
        stats = _Accumulator()
        for i, request in enumerate(requests):
            pacer.wait(gaps[i])
            if request.operation == "R":
                stats.add(self.demand(request))
            if plan is not None and plan[i].prefetch_runs:
                self.hint(plan[i].prefetch_runs)
        self.helper.shutdown(wait=True)
        self.check_hints()
        return stats.result(self)


class _Accumulator:
    """Per-run totals shared by the plan-driven and online drivers."""

    def __init__(self):
        self.lat = array("d")
        self.blocks = self.block_hits = 0
        self.full = self.ready = self.late = 0
        self.late_wait_us = self.hit_lat = self.miss_lat = 0.0

    def add(self, r: dict) -> None:
        self.lat.append(r["lat"])
        self.blocks += r["blocks"]
        self.block_hits += r["block_hits"]
        self.late += r["late"]
        self.late_wait_us += r["late_wait_us"]
        if r["full"]:
            self.full += 1
            self.ready += r["ready"]
            self.hit_lat += r["lat"]
        else:
            self.miss_lat += r["lat"]

    def result(self, backend: PageCacheBackend) -> dict:
        misses = len(self.lat) - self.full
        return {
            "latencies_us": self.lat,
            "read_blocks": self.blocks, "missed_blocks": self.blocks - self.block_hits,
            "real_hit_ratio": self.block_hits / self.blocks if self.blocks else 0.0,
            "full_hits": self.full, "ready_hits": self.ready, "late_hits": self.late,
            "late_wait_us": self.late_wait_us,
            "hit_lat_mean_us": self.hit_lat / self.full if self.full else None,
            "miss_lat_mean_us": self.miss_lat / misses if misses else None,
            "prefetches": backend.prefetches, "useful_prefetches": backend.useful,
            "prefetch_bytes": backend.prefetch_bytes, "hint_calls": backend.hint_calls,
            "residency_source": backend.residency_source,
            "willneed_chunk": backend.chunk, "advice": backend.advice,
        }
