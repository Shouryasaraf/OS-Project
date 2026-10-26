"""The backing file and the map from trace blocks to file offsets.

A trace addresses a volume of up to terabytes; the backing file is at most
32 GiB. :class:`RegionMapper` packs the trace's 1 MiB regions into the file in
first-touch order, which keeps every block inside a region at its true offset
relative to its neighbours (so a sequential run stays sequential on the
device) while discarding the empty space between regions. Seek distances
between regions are therefore shorter than on the original volume; that is a
stated limit of the testbed, not hidden.

Demand regions are assigned first, from the request stream alone, so every
mode of one trace lays its demanded data out identically. Regions touched
only by a mode's prefetches are assigned after them. If the file runs out of
slots a region would alias another; that is counted (``wrapped``) and refused
unless the caller explicitly allows it.
"""

from __future__ import annotations

import mmap
import os
import struct
import time
from pathlib import Path

from . import BLOCK, PAGE

REGION_BLOCKS = 2048                 # 1 MiB of 512-B blocks
REGION_BYTES = REGION_BLOCKS * BLOCK
#: Prototype backing-file names, accepted when the canonical one is absent.
#: Same format: non-sparse, written with O_DIRECT, offset-stamped.
BACKING_NAMES = ("backing.bin", "hdd_test.bin", "ssd_test.bin")


class RegionAliasError(RuntimeError):
    """The trace footprint does not fit the backing file without aliasing."""


class BackingFileError(RuntimeError):
    """The backing file is missing, sparse or the wrong size."""


def make_backing_file(path, gib: float, *, overwrite: bool = False,
                      chunk_bytes: int = 64 << 20, progress=None) -> dict:
    """Write a non-sparse, page-stamped file of ``gib`` GiB with ``O_DIRECT``.

    Every 4 KiB page starts with its own byte offset (little-endian u64) and
    the rest is random, so no two pages are identical and nothing in the file
    can be served from a zero page or deduplicated. ``O_DIRECT`` keeps the
    write out of the page cache. Measured at about 200 MB/s on the HDD.
    Refuses to replace an existing file unless ``overwrite``.
    """
    path = Path(path)
    total = int(gib * (1 << 30)) // REGION_BYTES * REGION_BYTES
    if total <= 0:
        raise ValueError("backing file must be at least 1 MiB")
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass overwrite=True to replace it")
    chunk_bytes = max(REGION_BYTES, chunk_bytes // REGION_BYTES * REGION_BYTES)
    buf = mmap.mmap(-1, chunk_bytes)       # page-aligned, as O_DIRECT needs
    buf.write(os.urandom(chunk_bytes))
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_DIRECT, 0o644)
    start = time.perf_counter()
    try:
        written = 0
        while written < total:
            n = min(chunk_bytes, total - written)
            for page in range(0, n, PAGE):
                struct.pack_into("<Q", buf, page, written + page)
            view = memoryview(buf)[:n]
            done = os.pwritev(fd, [view], written)
            view.release()
            if done != n:
                raise OSError(f"short O_DIRECT write at {written}: {done} of {n}")
            written += n
            if progress is not None:
                progress(written, total)
        os.fsync(fd)
    finally:
        os.close(fd)
        buf.close()
    seconds = time.perf_counter() - start
    return {"path": str(path), "bytes": total, "seconds": round(seconds, 3),
            "MiBps": round(total / (1 << 20) / seconds, 1) if seconds else None}


def find_backing(device_dir, name: str | None = None) -> Path:
    """The backing file in ``device_dir``: ``name``, else the first of
    :data:`BACKING_NAMES` that exists."""
    root = Path(device_dir)
    candidates = (name,) if name else BACKING_NAMES
    for candidate in candidates:
        path = root / candidate
        if path.is_file():
            return path
    raise BackingFileError(
        f"no backing file ({', '.join(candidates)}) in {root}; create one with "
        "`python -m adaptive_prefetch.realio mkfile --device-dir DIR --gib N`")


def check_backing(path) -> int:
    """Size of a usable backing file, in bytes, or :class:`BackingFileError`.

    A sparse file would be read from holes without touching the device, so
    allocated blocks must cover the size. The size must be whole regions.
    """
    st = os.stat(path)
    if st.st_size < REGION_BYTES or st.st_size % REGION_BYTES:
        raise BackingFileError(f"{path}: size {st.st_size} is not a positive "
                               "multiple of 1 MiB")
    if st.st_blocks * 512 < st.st_size:
        raise BackingFileError(f"{path} is sparse ({st.st_blocks * 512} of "
                               f"{st.st_size} bytes allocated); rewrite it with mkfile")
    return st.st_size


class RegionMapper:
    """Block -> backing-file offset, by 1 MiB region in first-touch order.

    ``reserved_tail`` bytes at the end of the file are never mapped; the
    drive-cache flush reads them. A region beyond the remaining slots wraps
    onto an already-used slot: it is counted in ``wrapped`` and makes
    :meth:`check` raise unless ``allow_alias``.
    """

    def __init__(self, file_bytes: int, *, reserved_tail: int = 0,
                 allow_alias: bool = False):
        self.nslots = (file_bytes - reserved_tail) // REGION_BYTES
        if self.nslots <= 0:
            raise ValueError("backing file leaves no room for trace regions")
        self.allow_alias = allow_alias
        self.slots: dict[int, int] = {}
        self.wrapped = 0
        self.demand_regions = 0

    @classmethod
    def for_plan(cls, file_bytes: int, requests, plan=None, **kwargs) -> "RegionMapper":
        """Map demand regions first (mode-independent), then prefetch-only ones."""
        mapper = cls(file_bytes, **kwargs)
        for request in requests:
            mapper.touch(request.lba, request.size_blocks)
        mapper.demand_regions = len(mapper.slots)
        if plan is not None:
            for step in plan:
                for first, count in step.prefetch_runs:
                    mapper.touch(first, count)
        return mapper

    @property
    def regions(self) -> int:
        return len(self.slots)

    def _slot(self, region: int) -> int:
        slot = self.slots.get(region)
        if slot is None:
            slot = len(self.slots)
            if slot >= self.nslots:
                self.wrapped += 1
                slot %= self.nslots
            self.slots[region] = slot
        return slot

    def touch(self, first: int, count: int) -> None:
        for region in range(first // REGION_BLOCKS, (first + count - 1) // REGION_BLOCKS + 1):
            self._slot(region)

    def check(self) -> None:
        if self.wrapped and not self.allow_alias:
            raise RegionAliasError(
                f"trace footprint needs {self.regions} regions of 1 MiB but the "
                f"file holds {self.nslots}; {self.wrapped} would alias. Use a "
                "larger backing file, fewer requests, or --allow-alias")

    def offset(self, block: int) -> int:
        region, within = divmod(block, REGION_BLOCKS)
        return self._slot(region) * REGION_BYTES + within * BLOCK

    def runs(self, blocks) -> list[list[int]]:
        """Coalesce blocks into ``[offset, nbytes]`` runs contiguous in the file."""
        out: list[list[int]] = []
        for block in blocks:
            off = self.offset(block)
            if out and out[-1][0] + out[-1][1] == off:
                out[-1][1] += BLOCK
            else:
                out.append([off, BLOCK])
        return out

    def range_runs(self, first: int, count: int) -> list[list[int]]:
        """``[offset, nbytes, first_block]`` runs for blocks ``[first, first+count)``.

        Split at region boundaries and re-merged where consecutive regions
        sit in consecutive slots, so a run is contiguous in both the trace
        and the file.
        """
        out: list[list[int]] = []
        block, end = first, first + count
        while block < end:
            stop = min(end, (block // REGION_BLOCKS + 1) * REGION_BLOCKS)
            off = self.offset(block)
            nbytes = (stop - block) * BLOCK
            if out and out[-1][0] + out[-1][1] == off:
                out[-1][1] += nbytes
            else:
                out.append([off, nbytes, block])
            block = stop
        return out
