"""The object store: one file, one contiguous 4 KiB-aligned object per product.

RetailRocket records no page sizes, so every size here is an **ASSUMPTION**:
a lognormal with median 64 KiB and sigma 1.0, clipped to 1 KiB-4 MiB, drawn
deterministically from a hash of (seed, item) so every run, pass and policy
sees the same size for the same product. The layout is shuffled, so file
order is uncorrelated with item id and with session order.

The page-cache helpers need no root: ``POSIX_FADV_DONTNEED`` evicts a file's
clean pages, ``POSIX_FADV_WILLNEED`` starts readahead, and ``preadv(...,
RWF_NOWAIT)`` reads only what is already resident, so a short read or
``EAGAIN`` means "not cached". Two kernel limits were measured on this host
(plan section 0.4): one WILLNEED call reads at most ``max_sectors_kb``
(256 KiB), and one RWF_NOWAIT read returns at most 256 KiB, so both are
issued in 256 KiB chunks. Stdlib only.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import statistics
import time
from pathlib import Path
from statistics import NormalDist

SIZE_MEDIAN = 64 * 1024
SIZE_SIGMA = 1.0
SIZE_MIN = 1024
SIZE_MAX = 4 << 20
SIZE_SEED = 7
ALIGN = 4096
CHUNK = 256 * 1024
STORE_NAME = "store.bin"
INDEX_NAME = "store_index.json"
META_NAME = "store_meta.json"
_FILL = 8 << 20


def object_size(item: int, median: int = SIZE_MEDIAN, sigma: float = SIZE_SIGMA,
                seed: int = SIZE_SEED, lo: int = SIZE_MIN, hi: int = SIZE_MAX) -> int:
    """ASSUMED size in bytes of product ``item`` (deterministic lognormal)."""
    digest = hashlib.blake2b(f"{seed}:{item}".encode(), digest_size=8).digest()
    u = (int.from_bytes(digest, "big") + 0.5) / 2 ** 64
    z = NormalDist().inv_cdf(u)
    return int(min(hi, max(lo, round(median * math.exp(sigma * z)))))


def store_paths(directory: str | Path, median: int = SIZE_MEDIAN,
                sigma: float = SIZE_SIGMA) -> tuple[Path, Path, Path]:
    """(store, index, meta) paths; the size assumption is part of the name,
    so a sensitivity store never overwrites the headline one."""
    directory = Path(directory)
    tag = f"m{median}_s{sigma:g}"
    return (directory / f"{tag}_{STORE_NAME}", directory / f"{tag}_{INDEX_NAME}",
            directory / f"{tag}_{META_NAME}")


def _items_digest(items) -> str:
    return hashlib.sha256(",".join(map(str, items)).encode()).hexdigest()


def build_store(items, directory: str | Path, *, median: int = SIZE_MEDIAN,
                sigma: float = SIZE_SIGMA, seed: int = SIZE_SEED,
                shuffle: bool = True, reuse: bool = True) -> dict:
    """Write the store file and its index, fsync both, return the summary.

    ``reuse=True`` keeps an existing store with the same size assumption and
    layout whose items include every requested one, so a multi-GB store on
    the HDD is built once (by the run with the largest policy set) and the
    ablation runs read the same file.
    """
    items = sorted({int(item) for item in items})
    if not items:
        raise ValueError("an object store needs at least one item")
    store, index_path, meta_path = store_paths(directory, median, sigma)
    wanted = {"items": len(items), "items_sha256": _items_digest(items), "median": median,
              "sigma": sigma, "seed": seed, "shuffle": shuffle, "align": ALIGN,
              "clip": [SIZE_MIN, SIZE_MAX], "size_tag": "ASSUMED"}
    if reuse and store.is_file() and index_path.is_file() and meta_path.is_file():
        meta = json.loads(meta_path.read_text())
        same = all(meta.get(k) == wanted[k] for k in ("median", "sigma", "seed", "shuffle",
                                                       "align", "clip", "size_tag"))
        if same and store.stat().st_size == meta["bytes"] and \
                set(items) <= set(load_index(index_path)):
            return {**meta, "reused": True, "store": str(store), "index": str(index_path)}
    store.parent.mkdir(parents=True, exist_ok=True)
    order = list(items)
    if shuffle:
        random.Random(seed).shuffle(order)
    # Object contents are irrelevant to every measurement; they are seeded
    # bytes so a rebuilt store is byte-identical.
    fill = random.Random(seed).randbytes(_FILL)
    index: dict[int, tuple[int, int]] = {}
    offset = 0
    started = time.perf_counter()
    tmp = store.with_name(store.name + ".tmp")
    with open(tmp, "wb") as handle:
        for item in order:
            size = object_size(item, median, sigma, seed)
            pad = (-offset) % ALIGN
            if pad:
                handle.write(b"\0" * pad)
                offset += pad
            remaining = size
            while remaining:
                n = min(remaining, len(fill))
                start = (item * 4093) % (len(fill) - n + 1)
                handle.write(fill[start:start + n])
                remaining -= n
            index[item] = (offset, size)
            offset += size
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, store)
    index_path.write_text(json.dumps({str(k): list(v) for k, v in sorted(index.items())}))
    sizes = sorted(size for _, size in index.values())
    meta = {**wanted, "bytes": offset, "size_median_bytes": sizes[len(sizes) // 2],
            "size_mean_bytes": statistics.fmean(sizes),
            "size_p99_bytes": sizes[min(len(sizes) - 1, int(len(sizes) * 0.99))],
            "build_s": round(time.perf_counter() - started, 3)}
    meta_path.write_text(json.dumps(meta, indent=1))
    _fsync_dir(store.parent)
    return {**meta, "reused": False, "store": str(store), "index": str(index_path)}


def _fsync_dir(directory: Path) -> None:
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def load_index(path: str | Path) -> dict[int, tuple[int, int]]:
    return {int(k): (int(v[0]), int(v[1])) for k, v in json.loads(Path(path).read_text()).items()}


def drop_page_cache(path: str | Path) -> None:
    """Evict this file's clean pages (no root needed): fsync, then DONTNEED."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def willneed(fd: int, offset: int, size: int, chunk: int = CHUNK) -> int:
    """Hint ``[offset, offset+size)`` in ``chunk``-byte calls; returns the count.

    One WILLNEED call is truncated to ``max_sectors_kb`` (256 KiB here), so a
    single call over a larger object leaves most of it on the device.
    """
    calls = 0
    for position in range(0, size, chunk):
        os.posix_fadvise(fd, offset + position, min(chunk, size - position),
                         os.POSIX_FADV_WILLNEED)
        calls += 1
    return calls


def resident(fd: int, offset: int, size: int, chunk: int = CHUNK) -> bool:
    """True iff every byte of the range is already in the page cache."""
    buffer = bytearray(min(size, chunk))
    position = 0
    while position < size:
        want = min(chunk, size - position)
        try:
            n = os.preadv(fd, [memoryview(buffer)[:want]], offset + position, os.RWF_NOWAIT)
        except BlockingIOError:
            return False
        if n != want:
            return False
        position += n
    return True


def device_of(path: str | Path) -> dict:
    """The block device holding ``path``: name, its whole disk, rotational flag.

    Read from sysfs; fields are ``None`` where sysfs does not say (tmpfs,
    overlay, a container).
    """
    st = os.stat(path)
    major, minor = os.major(st.st_dev), os.minor(st.st_dev)
    info = {"dev": f"{major}:{minor}", "name": None, "disk": None, "rotational": None}
    link = Path(f"/sys/dev/block/{major}:{minor}")
    try:
        real = link.resolve(strict=True)
    except OSError:
        return info
    info["name"] = real.name
    disk = real.parent if (real / "partition").exists() else real
    info["disk"] = disk.name
    try:
        info["rotational"] = (disk / "queue" / "rotational").read_text().strip() == "1"
    except OSError:
        pass
    for key in ("read_ahead_kb", "max_sectors_kb"):
        try:
            info[key] = int((disk / "queue" / key).read_text())
        except (OSError, ValueError):
            pass
    return info


def store_label(path: str | Path) -> str:
    """``hdd`` or ``ssd`` from the rotational flag; ``unknown`` otherwise."""
    rotational = device_of(path).get("rotational")
    return "unknown" if rotational is None else ("hdd" if rotational else "ssd")


def diskstats_sectors_read(name: str | None) -> int | None:
    """Sectors read so far by block device ``name`` (``/proc/diskstats``)."""
    if not name:
        return None
    try:
        with open("/proc/diskstats") as handle:
            for line in handle:
                fields = line.split()
                if len(fields) > 5 and fields[2] == name:
                    return int(fields[5])
    except OSError:
        return None
    return None


def process_read_bytes(pid: int | str = "self") -> int | None:
    """Bytes a process caused to be fetched from storage (``/proc/<pid>/io``)."""
    try:
        with open(f"/proc/{pid}/io") as handle:
            for line in handle:
                if line.startswith("read_bytes:"):
                    return int(line.split()[1])
    except (OSError, ValueError):
        return None
    return None


def _timed_read(fd: int, offset: int, size: int) -> float:
    started = time.perf_counter()
    os.pread(fd, size, offset)
    return (time.perf_counter() - started) * 1e6


def _summary(values: list[float]) -> dict:
    ordered = sorted(values)
    pick = lambda q: ordered[min(len(ordered) - 1, int(q * len(ordered)))]  # noqa: E731
    return {"mean_us": statistics.fmean(ordered), "p50_us": pick(0.5), "p99_us": pick(0.99),
            "n": len(ordered)}


def probe(store: str | Path, index_path: str | Path, *, n: int = 1000,
          leads_ms=(0.0, 1.0, 5.0, 20.0), hinted_n: int = 300, seed: int = 3,
          label: str = "") -> dict:
    """MEASURED page-cache behaviour of one store: cold, warm and hinted reads.

    cold: DONTNEED the whole file, then pread a sample of objects; warm: the
    same reads again; hinted: DONTNEED, chunked WILLNEED on an object, wait
    ``lead`` ms, then check residency and pread it.
    """
    index = list(load_index(index_path).values())
    sample = random.Random(seed).sample(index, min(n, len(index)))
    fd = os.open(store, os.O_RDONLY)
    try:
        out = {"label": label, "store": str(store), "device": device_of(store),
               "objects": len(sample), "load_avg": list(os.getloadavg()), "tag": "MEASURED"}
        os.fsync(fd)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        check = sample[:200]
        out["resident_after_drop"] = sum(resident(fd, o, s) for o, s in check) / len(check)
        cold = [_timed_read(fd, o, s) for o, s in sample]
        warm = [_timed_read(fd, o, s) for o, s in sample]
        out["resident_after_read"] = sum(resident(fd, o, s) for o, s in check) / len(check)
        out["cold"], out["warm"] = _summary(cold), _summary(warm)
        out["hinted"] = {}
        hinted = sample[:hinted_n]
        for lead in leads_ms:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            latencies, hits = [], 0
            for offset, size in hinted:
                willneed(fd, offset, size)
                if lead:
                    time.sleep(lead / 1000)
                hits += resident(fd, offset, size)
                latencies.append(_timed_read(fd, offset, size))
            out["hinted"][f"{lead:g}"] = {"resident_frac": hits / len(hinted),
                                          **_summary(latencies)}
        out["load_avg_end"] = list(os.getloadavg())
        return out
    finally:
        os.close(fd)
