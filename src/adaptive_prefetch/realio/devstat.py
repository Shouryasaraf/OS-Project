"""Which disk a file lives on, what it reads, and who else is reading it.

The host is shared, so a measurement is only evidence alongside how much of
the device's traffic was ours. ``/proc/diskstats`` gives the whole disk's
sectors; ``/proc/self/io`` ``read_bytes`` gives this process's (all threads,
O_DIRECT, buffered and WILLNEED read-ahead alike -- checked on this kernel).
The difference is foreign I/O, and a run is flagged when it exceeds 1%.

The drive's own cache (256 MiB on the test HDD) cannot be dropped without
root, so :func:`flush_drive` displaces it by reading an unmapped region of the
backing file with ``O_DIRECT``.
"""

from __future__ import annotations

import mmap
import os
import time
from dataclasses import dataclass
from pathlib import Path

FOREIGN_IO_FLAG = 0.01
SECTOR = 512


def disk_for_path(path) -> str | None:
    """Kernel name of the whole disk holding ``path`` (``sdb``), or ``None``."""
    st = os.stat(path)
    major, minor = os.major(st.st_dev), os.minor(st.st_dev)
    node = Path(f"/sys/dev/block/{major}:{minor}")
    try:
        resolved = node.resolve(strict=True)
    except OSError:
        return None
    if (resolved / "partition").exists():
        resolved = resolved.parent
    return resolved.name


def _read(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def device_info(disk: str | None) -> dict:
    """Model, rotational flag and queue limits from sysfs (``None`` if unknown)."""
    if disk is None:
        return {"model": None, "rotational": None, "read_ahead_kb": None,
                "max_sectors_kb": None, "logical_block_size": None}
    base = Path("/sys/block") / disk
    rot = _read(base / "queue" / "rotational")

    def num(name):
        value = _read(base / "queue" / name)
        return int(value) if value and value.isdigit() else None

    return {"model": _read(base / "device" / "model"),
            "rotational": None if rot is None else rot == "1",
            "read_ahead_kb": num("read_ahead_kb"),
            "max_sectors_kb": num("max_sectors_kb"),
            "logical_block_size": num("logical_block_size")}


def device_label(disk: str | None, rotational: bool | None) -> str:
    """Results label: the kernel name for an HDD (there are several identical
    ones), ``ssd`` for the single solid-state disk."""
    if rotational is False:
        return "ssd"
    return disk or "unknown"


def diskstats(disk: str | None) -> tuple[int, int] | None:
    """``(sectors_read, sectors_written)`` for ``disk`` since boot."""
    if disk is None:
        return None
    try:
        with open("/proc/diskstats") as handle:
            for line in handle:
                fields = line.split()
                if len(fields) > 9 and fields[2] == disk:
                    return int(fields[5]), int(fields[9])
    except OSError:
        return None
    return None


def own_read_bytes() -> int | None:
    """Bytes this process (every thread) caused to be read from storage."""
    try:
        with open("/proc/self/io") as handle:
            for line in handle:
                if line.startswith("read_bytes:"):
                    return int(line.split()[1])
    except OSError:
        return None
    return None


@dataclass
class IOWindow:
    """Disk-wide and own traffic over an interval; ``stop()`` closes it."""
    disk: str | None
    disk_start: tuple[int, int] | None = None
    own_start: int | None = None
    own_bytes: int | None = None
    disk_bytes: int | None = None

    def start(self) -> "IOWindow":
        self.disk_start = diskstats(self.disk)
        self.own_start = own_read_bytes()
        return self

    def stop(self) -> "IOWindow":
        end, own = diskstats(self.disk), own_read_bytes()
        if self.disk_start is not None and end is not None:
            self.disk_bytes = ((end[0] - self.disk_start[0])
                               + (end[1] - self.disk_start[1])) * SECTOR
        if self.own_start is not None and own is not None:
            self.own_bytes = own - self.own_start
        return self

    @property
    def foreign_io_frac(self) -> float | None:
        """Share of the disk's traffic that was not this process's."""
        if self.disk_bytes is None or self.own_bytes is None:
            return None
        if self.disk_bytes <= 0:
            return 0.0
        return max(0.0, self.disk_bytes - self.own_bytes) / self.disk_bytes


def flush_drive(path, mib: int = 1024, *, chunk: int = 1 << 20) -> float:
    """Read the last ``mib`` MiB of ``path`` with ``O_DIRECT``; returns seconds.

    The region must not hold trace data (``RegionMapper`` reserves it), so
    the flush displaces the drive cache without warming anything a run reads.
    """
    if mib <= 0:
        return 0.0
    fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
    buf = mmap.mmap(-1, chunk)
    start = time.perf_counter()
    try:
        size = os.fstat(fd).st_size
        if mib * (1 << 20) > size:
            raise ValueError(f"flush of {mib} MiB exceeds {path} ({size} bytes)")
        for i in range(mib * (1 << 20) // chunk):
            os.preadv(fd, [buf], size - (i + 1) * chunk)
    finally:
        os.close(fd)
        buf.close()
    return time.perf_counter() - start
