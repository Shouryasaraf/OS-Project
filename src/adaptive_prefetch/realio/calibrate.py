"""Device calibration for ``simulator.IOCostModel``: ``t_pos`` and ``t_xfer``.

Four QD1 ``O_DIRECT`` probes on the backing file, none of which touch the page
cache: random 4 KiB reads over the whole file, sequential 4 KiB and 64 KiB
reads, and 1 MiB sequential throughput. From them:

* ``t_xfer_us_per_block = 512 B / sequential 1 MiB throughput``
* ``t_pos_us = mean random 4 KiB latency - 8 * t_xfer`` (the positioning part
  of a random read, with its 8-block transfer taken out)
* ``kappa = t_pos / t_xfer`` -- the device's positioning-to-transfer ratio in
  blocks, compared against each policy's ``kappa*`` in the disk analysis.

The **mean** is used for ``t_pos`` because ``IOCostModel`` sums device time
over I/Os, which is an expectation; the median is recorded beside it. Random
offsets span the whole backing file, which is a property of this file on this
drive, not of the original traced volume. Nothing here is a general property
of HDDs or SSDs (plan honesty rule 15); it is one file on one device.
"""

from __future__ import annotations

import datetime as _dt
import mmap
import os
import random
import time

from . import BLOCK
from .devstat import (IOWindow, device_info, device_label, disk_for_path,
                      flush_drive)

DEFAULT_SAMPLES = {"rand4k": 2000, "seq4k": 4000, "seq64k": 2000, "seq1m": 512}


def _summary(values: list[float]) -> dict:
    ordered = sorted(values)
    n = len(ordered)

    def q(p):
        return ordered[min(n - 1, int(p * n))]

    return {"n": n, "mean": sum(ordered) / n, "p50": q(0.5), "p95": q(0.95),
            "p99": q(0.99), "max": ordered[-1]}


def _timed(fd: int, buf, offsets, nbytes: int) -> list[float]:
    view = memoryview(buf)[:nbytes]
    out = []
    perf = time.perf_counter
    for offset in offsets:
        t = perf()
        got = os.preadv(fd, [view], offset)
        out.append((perf() - t) * 1e6)
        if got != nbytes:
            raise OSError(f"short O_DIRECT read at {offset}: {got} of {nbytes}")
    view.release()
    return out


def measure(path, *, samples: dict | None = None, seed: int = 42,
            flush_mib: int = 1024) -> dict:
    """Run the four probes and return their latency summaries (microseconds)."""
    counts = {**DEFAULT_SAMPLES, **(samples or {})}
    rng = random.Random(seed)
    fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
    buf = mmap.mmap(-1, 1 << 20)
    try:
        size = os.fstat(fd).st_size
        usable = size - flush_mib * (1 << 20)
        if usable < (64 << 20):
            raise ValueError(f"{path} is too small to calibrate on ({size} bytes)")
        flush_drive(path, flush_mib)
        raw: dict = {"file_bytes": size}
        offsets = [rng.randrange(usable // 4096) * 4096 for _ in range(counts["rand4k"])]
        raw["rand4k"] = _summary(_timed(fd, buf, offsets, 4096))
        for name, nbytes in (("seq4k", 4096), ("seq64k", 65536)):
            n = counts[name]
            base = rng.randrange((usable - n * nbytes) // (1 << 20)) * (1 << 20)
            raw[name] = _summary(_timed(fd, buf, [base + i * nbytes for i in range(n)], nbytes))
        n = counts["seq1m"]
        base = rng.randrange((usable - n * (1 << 20)) // (1 << 20)) * (1 << 20)
        start = time.perf_counter()
        lat = _timed(fd, buf, [base + i * (1 << 20) for i in range(n)], 1 << 20)
        seconds = time.perf_counter() - start
        raw["seq1m"] = _summary(lat)
        raw["seq1m_MBps"] = n * (1 << 20) / seconds / 1e6
    finally:
        os.close(fd)
        buf.close()
    return raw


def derive(raw: dict) -> dict:
    """``t_pos_us``, ``t_xfer_us_per_block`` and ``kappa`` from probe summaries."""
    mbps = raw["seq1m_MBps"]
    if mbps <= 0:
        raise ValueError("sequential throughput must be positive")
    t_xfer = BLOCK / mbps                       # bytes / (bytes per us)
    t_pos = max(0.0, raw["rand4k"]["mean"] - (4096 // BLOCK) * t_xfer)
    return {"t_pos_us": t_pos, "t_xfer_us_per_block": t_xfer,
            "kappa": t_pos / t_xfer}


def calibrate(path, *, name: str | None = None, samples: dict | None = None,
              seed: int = 42, flush_mib: int = 1024) -> dict:
    """Measure, derive and return one ``realio/device_<name>.json`` row."""
    disk = disk_for_path(path)
    info = device_info(disk)
    label = name or device_label(disk, info["rotational"])
    try:
        load = [round(x, 2) for x in os.getloadavg()]
    except OSError:
        load = None
    window = IOWindow(disk).start()
    raw = measure(path, samples=samples, seed=seed, flush_mib=flush_mib)
    window.stop()
    constants = derive(raw)
    return {
        "device": label, "path": str(path), "model": info["model"],
        "rotational": info["rotational"], "read_ahead_kb": info["read_ahead_kb"],
        "max_sectors_kb": info["max_sectors_kb"],
        "rand4k_p50_us": raw["rand4k"]["p50"], "rand4k_mean_us": raw["rand4k"]["mean"],
        "seq4k_p50_us": raw["seq4k"]["p50"], "seq64k_p50_us": raw["seq64k"]["p50"],
        "seq1m_MBps": raw["seq1m_MBps"], **constants,
        "samples": sum(raw[k]["n"] for k in ("rand4k", "seq4k", "seq64k", "seq1m")),
        "foreign_io_frac": window.foreign_io_frac, "load_avg": load,
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "extra": {"disk": disk, "probes": raw, "seed": seed, "flush_mib": flush_mib,
                  "t_pos_statistic": "mean random 4 KiB minus 8 block transfers"},
    }
