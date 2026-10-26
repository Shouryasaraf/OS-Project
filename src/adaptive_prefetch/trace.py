"""Canonical trace records, trace loading, and reproducible labelled workloads."""

from __future__ import annotations

import csv
import random
import shutil
import struct
import subprocess
import sys
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Iterable, Iterator

from .corpora import CORPUS_OF, ORACLE_MARKER, trace_stem

CLASSES = ("sequential", "strided", "random", "mixed")
MSR_COLUMNS = ("Timestamp", "Hostname", "DiskNumber", "Type", "Offset", "Size",
               "ResponseTime")
IOTTA8_COLUMNS = ("device", "sector", "size", "op", "offset", "timestamp",
                  "lifetime", "count")
ALIBABA_COLUMNS = ("device_id", "opcode", "offset", "length", "timestamp")
REVISED_COLUMNS = ("time_s", "op", "lba", "size_blocks", "seq_or_rand",
                   "t1", "t2", "t3")
#: Every loader profile. ``oracle`` (libCacheSim binary block traces) is
#: never auto-detected: it is not a CSV and has no header to detect from.
FORMATS = ("auto", "normalized", "msr", "iotta8", "alibaba", "revised", "oracle")


@dataclass(frozen=True)
class Request:
    timestamp_ms: float
    lba: int
    size_blocks: int = 1
    operation: str = "R"
    stream_id: str = "default"
    #: Observed device service time in milliseconds, when the source trace
    #: recorded one. Only the ``revised`` profile populates this. ``None`` means
    #: "not measured", which is distinct from a measured zero.
    service_ms: float | None = None
    #: Source access-pattern flag (``seq``/``rand`` in the revised profile).
    #: Retained for ground-truth evaluation only; never used for candidate
    #: generation, which would be lookahead.
    pattern: str | None = None

    def __post_init__(self) -> None:
        if self.timestamp_ms < 0 or self.lba < 0 or self.size_blocks <= 0:
            raise ValueError("timestamp/lba must be non-negative and size_blocks positive")
        if self.operation not in {"R", "W"}:
            raise ValueError("operation must be R or W")
        if self.service_ms is not None and self.service_ms < 0:
            raise ValueError("service_ms must be non-negative")


def iter_csv_chunks(path: str | Path, format: str = "auto",
                    chunk_requests: int = 250_000, **oracle_options):
    """Yield ``list[Request]`` chunks without materialising the whole file.

    ``load_csv`` holds an entire trace as dataclass instances, which costs
    roughly 5x the on-disk size. The MSRC collection is 11 GB across 32
    files, so the whole set cannot be resident. This generator keeps at most
    one chunk alive, which makes the full-collection sweep feasible.

    Chunking is by **request count, not bytes**: a 500 MB slice of a
    ``.revised`` file is about 8.9M requests, which costs ~2.6 GB resident and
    would exhaust a small machine.

    ``revised`` and ``oracle`` stream. ``oracle`` routes to
    ``iter_oracle_chunks`` and is the only profile that takes keyword
    options (``size_unit``, ``merge``, ``max_requests``, ...); passing one with
    any other profile raises ``TypeError``. The small CSV profiles are parsed
    once and then sliced.
    """
    _check_format(format)
    if chunk_requests < 1000:
        raise ValueError("chunk_requests must be at least 1000")
    if format == "oracle":
        yield from iter_oracle_chunks(path, chunk_requests, **oracle_options)
        return
    if oracle_options:
        raise TypeError(f"options {sorted(oracle_options)} apply only to the oracle format")
    if format == "revised":
        yield from _iter_revised_chunks(path, chunk_requests)
        return
    # Other profiles are small enough to load whole; chunk after the fact so
    # callers see one uniform interface. Parse once: re-reading the file for
    # every slice made this quadratic in the number of chunks.
    requests = load_csv(path, format)
    for start in range(0, len(requests), chunk_requests):
        yield requests[start:start + chunk_requests]


def _check_format(format: str) -> None:
    if format not in FORMATS:
        raise ValueError(f"format must be one of {', '.join(FORMATS)}")


def _iter_revised_chunks(path: str | Path, chunk_requests: int):
    """Stream the 8-column revised profile into bounded chunks."""
    chunk: list[Request] = []
    first: float | None = None
    previous: float | None = None
    with Path(path).open(encoding="utf-8-sig") as handle:
        for line_no, raw in enumerate(handle, start=1):
            columns = raw.split()
            if len(columns) != len(REVISED_COLUMNS):
                raise ValueError(
                    f"invalid revised row {line_no}: expected "
                    f"{len(REVISED_COLUMNS)} columns, got {len(columns)}")
            try:
                seconds = float(columns[0])
                op = columns[1]
                lba = int(columns[2])
                size = int(columns[3])
                service_ms = float(columns[5])
            except (ValueError, TypeError) as exc:
                raise ValueError(f"invalid revised row {line_no}: {exc}") from exc
            if op not in {"RS", "WS"}:
                raise ValueError(f"invalid revised row {line_no}: op must be RS or WS")
            if service_ms < 0:
                raise ValueError(
                    f"invalid revised row {line_no}: negative service time")
            if first is None:
                first = seconds
            chunk.append(Request((seconds - first) * 1000, lba, size,
                                 "R" if op == "RS" else "W",
                                 service_ms=service_ms, pattern=columns[4]))
            if len(chunk) >= chunk_requests:
                # Order is checked across the boundary too, by carrying the
                # previous chunk's last timestamp.
                _validate_order(chunk, previous)
                previous = chunk[-1].timestamp_ms
                yield chunk
                chunk = []
    if chunk:
        _validate_order(chunk, previous)
        yield chunk


def load_csv(path: str | Path, format: str = "auto") -> list[Request]:
    """Read a known trace schema; ambiguous units require an explicit profile.

    ``oracle`` materialises the whole streamed trace, so it is for small
    files and prefixes; use ``iter_oracle_chunks`` for anything large.
    """
    _check_format(format)
    if format == "oracle":
        return list(chain.from_iterable(iter_oracle_chunks(path)))
    if format == "auto" and ORACLE_MARKER in Path(path).name:
        raise ValueError(f"{path} is a binary oracleGeneral trace; choose --format oracle")
    if format == "revised":
        return _load_revised_rows(path)
    requests: list[Request] = []
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        if format == "msr" or (format == "auto" and set(MSR_COLUMNS).issubset(columns)):
            if not set(MSR_COLUMNS).issubset(columns):
                raise ValueError("MSR format requires its seven named columns")
            return _load_msr_rows(reader)
        if format == "alibaba" or (format == "auto" and set(ALIBABA_COLUMNS).issubset(columns)):
            if not set(ALIBABA_COLUMNS).issubset(columns):
                raise ValueError("Alibaba format requires device_id,opcode,offset,length,timestamp")
            return _load_alibaba_rows(reader)
        if format == "iotta8" or (format == "auto" and set(IOTTA8_COLUMNS).issubset(columns)):
            if set(IOTTA8_COLUMNS).issubset(columns):
                return _load_iotta8_rows(reader)
            if format == "iotta8":
                handle.seek(0)
                return _load_iotta8_rows(csv.DictReader(handle, fieldnames=IOTTA8_COLUMNS))
        required = {"timestamp_ms", "lba", "size_blocks", "operation"}
        if not required.issubset(columns):
            raise ValueError("unknown CSV schema; choose a documented --format and units")
        for line, row in enumerate(reader, start=2):
            try:
                requests.append(Request(
                    timestamp_ms=float(row["timestamp_ms"]),
                    lba=int(row["lba"]),
                    size_blocks=int(row["size_blocks"]),
                    operation=row["operation"].strip().upper(),
                    stream_id=(row.get("stream_id") or "default").strip(),
                ))
            except (ValueError, TypeError, KeyError, AttributeError) as exc:
                raise ValueError(f"invalid trace row {line}: {exc}") from exc
    if not requests:
        raise ValueError("CSV trace is empty")
    if any(b.timestamp_ms < a.timestamp_ms for a, b in zip(requests, requests[1:])):
        raise ValueError("CSV requests must be sorted by timestamp_ms")
    return requests


def _load_msr_rows(reader: csv.DictReader) -> list[Request]:
    """Convert MSR timestamps and byte ranges to normalized block requests."""
    rows = list(reader)
    if not rows:
        raise ValueError("CSV trace is empty")
    try:
        # Merged MSR samples may concatenate host traces instead of preserving
        # one global timestamp order; replay must use chronological requests.
        rows.sort(key=lambda row: int(row["Timestamp"]))
        first_timestamp = int(rows[0]["Timestamp"])
        requests = []
        for row in rows:
            timestamp = int(row["Timestamp"])
            offset_bytes = int(row["Offset"])
            size_bytes = int(row["Size"])
            if offset_bytes % 512 or size_bytes % 512:
                raise ValueError("Offset and Size must be multiples of 512")
            operation = row["Type"].strip().upper()
            if operation == "READ":
                operation = "R"
            elif operation == "WRITE":
                operation = "W"
            else:
                raise ValueError(f"unsupported Type: {row['Type']}")
            requests.append(Request(
                timestamp_ms=(timestamp - first_timestamp) / 10_000,
                lba=offset_bytes // 512,
                size_blocks=size_bytes // 512,
                operation=operation,
                stream_id=(row.get("Hostname") or "default").strip(),
            ))
    except (ValueError, TypeError) as exc:
        raise ValueError(f"invalid MSR trace row: {exc}") from exc
    if any(b.timestamp_ms < a.timestamp_ms for a, b in zip(requests, requests[1:])):
        raise ValueError("CSV requests must be sorted by timestamp")
    return requests


def _load_iotta8_rows(reader: csv.DictReader) -> list[Request]:
    """Plan-specific eight-column profile: sector/size in 512-B sectors, us time.

    SNIA IOTTA is a repository, not a universal schema. This profile must be
    selected only for a trace whose accompanying metadata confirms these units.
    """
    rows = list(reader)
    if not rows:
        raise ValueError("CSV trace is empty")
    first = float(rows[0]["timestamp"])
    requests = []
    for line, row in enumerate(rows, start=2):
        try:
            op = row["op"].strip().upper()
            if op in {"READ", "0"}:
                op = "R"
            elif op in {"WRITE", "1"}:
                op = "W"
            requests.append(Request(
                timestamp_ms=(float(row["timestamp"]) - first) / 1000,
                lba=int(row["sector"]), size_blocks=int(row["size"]),
                operation=op, stream_id=row["device"].strip(),
            ))
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            raise ValueError(f"invalid IOTTA-8 row {line}: {exc}") from exc
    _validate_order(requests)
    return requests


def _load_alibaba_rows(reader: csv.DictReader) -> list[Request]:
    """Alibaba EBS profile: byte offset/length and microsecond timestamp."""
    rows = list(reader)
    if not rows:
        raise ValueError("CSV trace is empty")
    first = int(rows[0]["timestamp"])
    requests = []
    for line, row in enumerate(rows, start=2):
        try:
            offset = int(row["offset"])
            length = int(row["length"])
            if offset % 512 or length % 512:
                raise ValueError("offset and length must be multiples of 512 bytes")
            requests.append(Request(
                timestamp_ms=(int(row["timestamp"]) - first) / 1000,
                lba=offset // 512, size_blocks=length // 512,
                operation=row["opcode"].strip().upper(),
                stream_id=row["device_id"].strip(),
            ))
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            raise ValueError(f"invalid Alibaba row {line}: {exc}") from exc
    _validate_order(requests)
    return requests


def _load_revised_rows(path: str | Path) -> list[Request]:
    """MSRC-trace-003 'final-trace' profile: 8 whitespace-separated columns.

    Each line is ``time_s op lba size seq|rand t1 t2 t3`` where op is RS (read)
    or WS (write), lba and size are 512-byte sectors, time_s is in seconds.

    Columns 6-8 (``t1``, ``t2``, ``t3``) are the trace's own device timing
    fields and are **retained** as ``service_ms`` / on the record. Measured
    behaviour across the collection: ``t1`` is the total service time in
    milliseconds, bimodal, with 65-87% of requests completing in under 1 us and
    a heavy tail past 10 ms. ``t3`` is zero whenever ``t1`` is tiny, so it is
    the device-queueing component and ``t2`` is a small fixed host overhead.
    ``t1`` is *not* exactly ``t2 + t3`` (the residual reaches 9e-2 ms), so no
    such identity is assumed.

    Column 5 is kept as ``pattern`` for ground-truth scoring only. It must
    never reach candidate generation, which would be lookahead.
    """
    rows: list[tuple[float, str, int, int, str, float]] = []
    with Path(path).open(encoding="utf-8-sig") as handle:
        for line_no, raw in enumerate(handle, start=1):
            columns = raw.split()
            if len(columns) != len(REVISED_COLUMNS):
                raise ValueError(
                    f"invalid revised row {line_no}: expected "
                    f"{len(REVISED_COLUMNS)} columns, got {len(columns)}")
            try:
                seconds = float(columns[0])
                op = columns[1]
                lba = int(columns[2])
                size = int(columns[3])
                service_ms = float(columns[5])
            except (ValueError, TypeError) as exc:
                raise ValueError(f"invalid revised row {line_no}: {exc}") from exc
            if op not in {"RS", "WS"}:
                raise ValueError(f"invalid revised row {line_no}: op must be RS or WS")
            if service_ms < 0:
                raise ValueError(
                    f"invalid revised row {line_no}: negative service time")
            rows.append((seconds, op, lba, size, columns[4], service_ms))
    if not rows:
        raise ValueError("CSV trace is empty")
    first = rows[0][0]
    requests = [Request((seconds - first) * 1000, lba, size,
                        "R" if op == "RS" else "W",
                        service_ms=service_ms, pattern=pattern)
                for seconds, op, lba, size, pattern, service_ms in rows]
    _validate_order(requests)
    return requests


def _validate_order(requests: list[Request], previous: float | None = None) -> None:
    """Raise unless ``requests`` is in time order, continuing from ``previous``.

    ``previous`` is the last timestamp of the preceding chunk, so a stream
    that goes backwards exactly at a chunk boundary is caught as well.
    """
    if requests and previous is not None and requests[0].timestamp_ms < previous:
        raise ValueError("CSV requests must be sorted by timestamp "
                         "(order breaks at a chunk boundary)")
    if any(b.timestamp_ms < a.timestamp_ms for a, b in zip(requests, requests[1:])):
        raise ValueError("CSV requests must be sorted by timestamp")


# --- libCacheSim oracleGeneral block traces -------------------------------
#
# Record layout (24 bytes, little-endian ``<IQIq``), measured on all 1.45 B
# records of the 56 local files:
#
#   uint32 timestamp   whole seconds
#   uint64 obj_id      BYTE offset, always a multiple of 512. Alibaba and
#                      Tencent pack a volume tag (file number x 10) into bits
#                      40 and up, which ORACLE_TAG_MASK drops.
#   uint32 obj_size    bytes (msr, alibaba, tencent, fiu_cheetah) or 512-B
#                      sectors (the other nine fiu files)
#   int64  next_vtime  index of the next record with the same obj_id. Unused:
#                      it is keyed per obj_id, not per block, so it is not a
#                      block-level future oracle.
#
# MSR, Alibaba and Tencent store one record per original request (the first
# 1000 hm_0 offsets equal the committed MSR CSV sample on 1000/1000 rows; sizes
# agree on 968, and 31 of the 32 that differ are writes). FIU stores one
# record per 4 KiB page, so requests are re-assembled by merging same-second
# contiguous pages. There is no read/write flag and no service time.

ORACLE_RECORD = struct.Struct("<IQIq")
#: The low 40 bits of obj_id are the byte offset (max observed 880.9 GB).
ORACLE_TAG_MASK = (1 << 40) - 1
SECTOR_BYTES = 512
#: Largest re-assembled request, in 512-B sectors (512 KiB): Linux's default
#: ``max_sectors_kb`` of the FIU era. A longer contiguous same-second run must
#: have been more than one request, so it is split rather than fused.
DEFAULT_MAX_MERGE_SECTORS = 1024
#: Records examined to choose and check the size unit.
ORACLE_PROBE = 4096
#: Records decoded per read; 1.5 MB buffers.
_READ_RECORDS = 1 << 16
_SIZE_UNITS = ("auto", "bytes", "sectors")
#: ``auto`` is the canonical reconstruction behind every headline number:
#: ``pages`` for fiu_* files, ``none`` elsewhere. ``none`` is one record per
#: request. ``pages`` and ``same_second`` apply the same rule (coalesce
#: consecutive records with equal timestamp whose start is the previous end,
#: up to ``max_merge_sectors``) to any file; ``same_second`` names the
#: sensitivity view, and with ``max_merge_sectors=None`` it is uncapped.
_MERGES = ("auto", "none", "pages", "same_second")


def _zstd_module():
    """The optional ``zstandard`` module, or ``None``. Patched by tests."""
    try:
        import zstandard
    except ImportError:
        return None
    return zstandard


class _ProcessReader:
    """``zstd -dc`` stdout that raises at EOF if the child failed."""

    def __init__(self, process: subprocess.Popen, path: Path) -> None:
        self.process = process
        self.path = path

    def read(self, size: int) -> bytes:
        data = self.process.stdout.read(size)
        if not data and self.process.wait() != 0:
            raise RuntimeError(f"{self.path}: zstd -dc exited with status "
                               f"{self.process.returncode}")
        return data


@contextmanager
def _open_oracle(path: Path):
    """Binary stream of decompressed records.

    A ``.zst`` file is decompressed with the ``zstandard`` module when it is
    importable (the core stays stdlib-only, so it is imported here), else with
    a ``zstd -dc`` child process, else this raises ``RuntimeError``. Any other
    suffix is read raw.
    """
    if path.suffix != ".zst":
        with path.open("rb") as handle:
            yield handle
        return
    zstandard = _zstd_module()
    if zstandard is not None:
        with path.open("rb") as raw, \
                zstandard.ZstdDecompressor().stream_reader(raw, read_size=1 << 20) as stream:
            yield stream
        return
    exe = shutil.which("zstd")
    if exe is None:
        raise RuntimeError(f"{path} is zstd-compressed: install the 'zstandard' "
                           "module (pip install -e .[traces]) or the zstd binary, "
                           "or decompress it first")
    process = subprocess.Popen([exe, "-dc", "--", str(path)],
                               stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    try:
        yield _ProcessReader(process, path)
    finally:
        # Stopping early (max_requests, a probe) leaves the child mid-write.
        process.stdout.close()
        if process.poll() is None:
            process.terminate()
        process.wait()


def _record_buffers(stream, path: Path) -> Iterator[memoryview]:
    """Whole-record slices of the stream; a partial trailing record raises."""
    size = ORACLE_RECORD.size
    want = size * _READ_RECORDS
    tail = b""
    records = 0
    while True:
        data = stream.read(want)
        if not data:
            break
        if tail:
            data = tail + data
        usable = len(data) - len(data) % size
        tail = data[usable:]
        if usable:
            records += usable // size
            yield memoryview(data)[:usable]
    if tail:
        raise ValueError(f"{path}: truncated oracleGeneral record at index "
                         f"{records} ({len(tail)} of {size} bytes)")


def _head_records(path: Path, count: int) -> list[tuple[int, int, int, int]]:
    head: list[tuple[int, int, int, int]] = []
    with _open_oracle(path) as stream:
        for buffer in _record_buffers(stream, path):
            head.extend(ORACLE_RECORD.iter_unpack(buffer))
            if len(head) >= count:
                break
    return head[:count]


def oracle_profile(path: str | Path, probe: int = ORACLE_PROBE) -> dict:
    """Describe how an oracleGeneral file will be decoded, from its head.

    Returns ``size_unit`` (``bytes`` or ``sectors``: sectors when more than
    half of the first ``probe`` sizes are not multiples of 512, which holds
    for nine FIU files and no other), ``merge`` (``pages`` for fiu_* files,
    else ``none``), ``corpus`` (``None`` for a name outside the four
    corpora), ``tagged`` (some obj_id carries a volume tag), plus ``stem``,
    ``probe_records`` and ``contiguity``: the fraction of next records that
    start where the previous one ends under each unit. The loader refuses a
    unit whose contiguity is strictly lower than the other's.
    """
    path = Path(path)
    if probe < 2:
        raise ValueError("probe must be at least 2 records")
    head = _head_records(path, probe)
    if not head:
        raise ValueError(f"{path}: empty oracleGeneral trace")
    odd = sum(1 for record in head if record[2] % SECTOR_BYTES)
    offsets = [record[1] & ORACLE_TAG_MASK for record in head]
    sizes = [record[2] for record in head]
    pairs = len(head) - 1

    def contiguity(unit_bytes: int) -> float:
        if not pairs:
            return 0.0
        hits = sum(1 for i in range(pairs)
                   if offsets[i + 1] == offsets[i] + sizes[i] * unit_bytes)
        return hits / pairs

    stem = trace_stem(path)
    try:
        corpus = CORPUS_OF(stem)
    except ValueError:
        corpus = None
    return {
        "stem": stem,
        "corpus": corpus,
        "size_unit": "sectors" if odd * 2 > len(head) else "bytes",
        "merge": "pages" if stem.startswith("fiu_") else "none",
        "tagged": any(record[1] > ORACLE_TAG_MASK for record in head),
        "probe_records": len(head),
        "contiguity": {"bytes": contiguity(1), "sectors": contiguity(SECTOR_BYTES)},
    }


def _check_size_unit(path: Path, profile: dict, unit: str) -> None:
    """Refuse a size unit that makes the trace look less contiguous.

    Under the wrong unit the next-record contiguity measured 0.000 in every
    local file, so a strictly lower rate is a sharp signal. A tie (both 0 on
    a very random probe) cannot confirm either unit, so it only warns.
    """
    other = "sectors" if unit == "bytes" else "bytes"
    chosen, alternative = profile["contiguity"][unit], profile["contiguity"][other]
    if chosen < alternative:
        raise ValueError(
            f"{path}: size_unit={unit!r} gives next-record contiguity "
            f"{chosen:.4f} but {other!r} gives {alternative:.4f} over the first "
            f"{profile['probe_records']} records; the size unit is wrong")
    if chosen == alternative:
        warnings.warn(
            f"{path}: next-record contiguity is {chosen:.4f} under both size "
            f"units over the first {profile['probe_records']} records, so "
            f"size_unit={unit!r} cannot be confirmed", stacklevel=3)


def iter_oracle_chunks(path: str | Path, chunk_requests: int = 250_000, *,
                       size_unit: str = "auto", merge: str = "auto",
                       max_merge_sectors: int | None = DEFAULT_MAX_MERGE_SECTORS,
                       max_requests: int | None = None, skip_requests: int = 0,
                       stream_id: str | None = None) -> Iterator[list[Request]]:
    """Yield ``list[Request]`` chunks from a libCacheSim oracleGeneral trace.

    size_unit      ``auto`` | ``bytes`` | ``sectors`` (see ``oracle_profile``).
                   Whatever is chosen must pass the contiguity guard.
    merge          ``auto`` | ``none`` | ``pages`` | ``same_second`` (see
                   ``_MERGES``). Merging runs before chunking, so a run that
                   crosses a chunk boundary is still one request.
    max_merge_sectors  cap on a merged request, ``None`` for uncapped. A
                   single record larger than the cap is kept whole.
    skip_requests  canonical (post-merge) requests to drop first; with
                   ``max_requests`` this selects a disjoint later slice.
                   ``timestamp_ms`` stays relative to the file's first
                   record, so slices of one file share a clock.
    max_requests   canonical requests to yield after the skip; ``None`` for
                   all. The decompressor stops as soon as the cap is reached.

    Every request is ``operation="R"`` (the format has no op field), with
    ``lba = (obj_id & ORACLE_TAG_MASK) // 512``, ``service_ms=None``,
    ``pattern=None`` and ``stream_id`` the file stem. Time order is checked
    on every record, so it holds within chunks and across their boundaries;
    an unaligned offset, a zero or fractional-sector size, a timestamp that
    goes backwards or a truncated record raises ``ValueError`` naming the
    0-based record index.
    """
    if size_unit not in _SIZE_UNITS:
        raise ValueError(f"size_unit must be one of {', '.join(_SIZE_UNITS)}")
    if merge not in _MERGES:
        raise ValueError(f"merge must be one of {', '.join(_MERGES)}")
    if chunk_requests < 1000:
        raise ValueError("chunk_requests must be at least 1000")
    if max_merge_sectors is not None and max_merge_sectors < 1:
        raise ValueError("max_merge_sectors must be positive or None")
    if max_requests is not None and max_requests < 0:
        raise ValueError("max_requests must be non-negative or None")
    if skip_requests < 0:
        raise ValueError("skip_requests must be non-negative")
    path = Path(path)
    profile = oracle_profile(path)
    unit = profile["size_unit"] if size_unit == "auto" else size_unit
    _check_size_unit(path, profile, unit)
    do_merge = merge in ("pages", "same_second") or (
        merge == "auto" and profile["merge"] == "pages")
    cap = sys.maxsize if max_merge_sectors is None else max_merge_sectors
    sid = profile["stem"] if stream_id is None else stream_id
    in_bytes = unit == "bytes"
    stop = None if max_requests is None else skip_requests + max_requests
    if stop == 0:
        return

    chunk: list[Request] = []
    emitted = 0                       # canonical requests, skipped ones included
    index = -1                        # 0-based record index, for errors
    first_ts = cur_ts = None          # cur_* is the pending request, in sectors
    cur_start = cur_len = 0
    done = False
    unpack = ORACLE_RECORD.iter_unpack
    with _open_oracle(path) as stream:
        for buffer in _record_buffers(stream, path):
            for ts, obj_id, obj_size, _next in unpack(buffer):
                index += 1
                offset = obj_id & ORACLE_TAG_MASK
                if offset & (SECTOR_BYTES - 1):
                    raise ValueError(f"{path}: record {index}: offset {offset} "
                                     "is not 512-byte aligned")
                if in_bytes:
                    if not obj_size or obj_size & (SECTOR_BYTES - 1):
                        raise ValueError(f"{path}: record {index}: size {obj_size} "
                                         "bytes is not a whole number of sectors")
                    length = obj_size >> 9
                elif obj_size:
                    length = obj_size
                else:
                    raise ValueError(f"{path}: record {index}: zero size")
                start = offset >> 9
                if ts == cur_ts:
                    if (do_merge and start == cur_start + cur_len
                            and cur_len + length <= cap):
                        cur_len += length
                        continue
                elif cur_ts is None:
                    first_ts = ts
                elif ts < cur_ts:
                    raise ValueError(f"{path}: record {index}: timestamp {ts} s "
                                     f"precedes {cur_ts} s; records must be in time order")
                if cur_ts is not None:
                    if emitted >= skip_requests:
                        chunk.append(Request((cur_ts - first_ts) * 1000.0,
                                             cur_start, cur_len, "R", sid))
                        if len(chunk) >= chunk_requests:
                            yield chunk
                            chunk = []
                    emitted += 1
                    if emitted == stop:
                        done = True
                        break
                cur_ts, cur_start, cur_len = ts, start, length
            if done:
                break
    if not done and cur_ts is not None and emitted >= skip_requests:
        chunk.append(Request((cur_ts - first_ts) * 1000.0, cur_start, cur_len,
                             "R", sid))
    if chunk:
        yield chunk


def write_csv(path: str | Path, requests: Iterable[Request]) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp_ms", "lba", "size_blocks", "operation", "stream_id"])
        for req in requests:
            writer.writerow([req.timestamp_ms, req.lba, req.size_blocks, req.operation, req.stream_id])


#: Real-trace geometry. The previous generator drew addresses from
#: ``randrange(1000, 100000)`` (< 100k blocks) and request sizes from
#: ``randint(1, 3)``, while the MSRC volumes span 1e7-1e8 blocks and issue
#: median 8-block, max 1024-block requests. That 350x scale gap is part of why
#: synthetic windows sat ~12 sd away from real ones.
REALISTIC_LBA_BASE = 10 ** 7
REALISTIC_LBA_SPAN = 10 ** 8
#: Typical 4 KiB-aligned request size, in blocks.
TYPICAL_BLOCK_SIZE = 8


def block_stride(rng: random.Random, size: int = TYPICAL_BLOCK_SIZE) -> int:
    """A phase stride that is at least two request lengths.

    A one-request stride is indistinguishable from a sequential read, so the
    two classes must not overlap.
    """
    return max(1, size) * rng.randint(2, 8)


def _window_profile(rng: random.Random) -> dict:
    """Per-window draw controlling how strongly a pattern expresses itself.

    This is what creates *within-class* variance. The old generator used fixed
    constants, so every window of a class had nearly identical features
    (within-class sd 0.0000-0.056 against 0.03-1.28 measured on real traces).
    Real workloads vary in how regular they are, and the classifier has to
    cope with that spread rather than memorising a degenerate distribution.
    """
    return {
        # 0.55-1.0: how cleanly the window follows its class pattern.
        "strength": rng.uniform(0.55, 1.0),
        # Per-window block size, so size is correlated within a window (real
        # workloads issue bursts of uniform-sized I/O) but varies across them.
        "block_size": rng.choice(
            [TYPICAL_BLOCK_SIZE, TYPICAL_BLOCK_SIZE, TYPICAL_BLOCK_SIZE,
             1, 2, 4, 16, 32, 64, 128]),
        # Timing dispersion; drives gap_cv and fast_gap_ratio variance.
        "gap_sigma": rng.uniform(0.3, 2.2),
        # A window is read-heavy or write-heavy rather than a fixed 90% reads.
        # Real traces are strongly bimodal (a write phase, then a read phase).
        "write_probability": rng.choice(
            [0.0, 0.0, 0.0, 0.02, 0.5, 1.0, 1.0, 1.0]),
        # Address span varies per window; a real volume is not one uniform
        # range, and the jump features must not depend on a fixed span.
        "span": rng.choice([REALISTIC_LBA_SPAN, REALISTIC_LBA_SPAN,
                            REALISTIC_LBA_SPAN // 8, REALISTIC_LBA_SPAN // 64,
                            REALISTIC_LBA_SPAN * 4]),
        "base_lba": rng.randrange(REALISTIC_LBA_BASE,
                                  REALISTIC_LBA_BASE + REALISTIC_LBA_SPAN),
    }


def _gap(rng: random.Random, profile: dict, bursty: bool) -> float:
    """Log-normal inter-arrival gap: real service times are heavy-tailed."""
    sigma = profile["gap_sigma"]
    mean = 0.05 if bursty else 0.5
    value = mean * pow(2.718281828, rng.gauss(0.0, sigma))
    return min(value, 5_000.0)


def synthetic_window(label: str, rng: random.Random, n: int = 32,
                     timestamp_start: float = 0.0, start_lba: int | None = None,
                     stride_override: int | None = None,
                     profile: dict | None = None) -> list[Request]:
    """Generate one labelled window.

    The label describes the *construction*, never a real trace. Realism knobs
    (pattern strength, block size, timing dispersion, write ratio, address
    scale, occasional re-access) vary per window so the class distributions
    have the spread real workloads show instead of being near-degenerate.
    """
    if label not in CLASSES or n < 8:
        raise ValueError("label must be a known class and n >= 8")
    if profile is None:
        profile = _window_profile(rng)
    strength = profile["strength"]
    block = profile["block_size"]
    current = (start_lba if start_lba is not None
               else rng.randrange(profile["base_lba"],
                                  profile["base_lba"] + profile["span"]))
    # Strides are whole request lengths, and always at least 2x. A stride of
    # exactly one request length is geometrically identical to a sequential
    # read, so including it made "strided" and "sequential" the same class and
    # the classifier could not separate them (22/40 sequential windows were
    # labelled strided).
    stride = (stride_override if stride_override is not None
              else block * rng.randint(2, 8))
    noise = (1.0 - strength) * 0.5
    time = timestamp_start
    result: list[Request] = []
    seen: list[int] = []
    for i in range(n):
        size = max(1, block + rng.randint(-1, 1) if block > 2 else block)
        revisit = 0.0
        if label == "sequential":
            if i:
                current += size
                if rng.random() < noise:
                    current += block * rng.randint(1, 4)
            # Real scans re-read recently touched blocks occasionally.
            revisit = 0.10 * strength if seen else 0.0
        elif label == "strided":
            if i:
                current += stride
                if rng.random() < noise:
                    current += block * rng.randint(1, 3)
            revisit = 0.08 * strength if seen else 0.0
        elif label == "random":
            # Real "random-looking" windows are not pure uniform draws: some
            # have weak local clusters, some are a single flat scatter. Without
            # this the class had *exactly zero* feature variance, which is
            # what made the previous generator degenerate.
            if rng.random() < 0.15 * strength:
                current += block * rng.randint(1, 3)
            else:
                current = rng.randrange(profile["base_lba"],
                                        profile["base_lba"] + profile["span"])
        else:
            # Interleaved short runs and distant jumps: half the window reads
            # like a sequential burst, half like random access.
            if i % 8 >= 4 or i % 8 == 0:
                current = rng.randrange(profile["base_lba"],
                                        profile["base_lba"] + profile["span"])
            else:
                current += size
            revisit = 0.05 * strength if seen else 0.0
        if revisit and len(seen) > 1 and rng.random() < revisit:
            low = max(0, len(seen) - 8)
            current = seen[rng.randrange(low, len(seen))]
        seen.append(current)
        bursty = label == "mixed" and i % 8 < 4
        time += _gap(rng, profile, bursty)
        result.append(Request(round(time, 4), current, size,
                              "W" if rng.random() < profile["write_probability"]
                              else "R"))
    return result


def synthetic_dataset(windows_per_class: int, seed: int, n: int = 32):
    rng = random.Random(seed)
    examples = [(synthetic_window(label, rng, n), label)
                for label in CLASSES for _ in range(windows_per_class)]
    rng.shuffle(examples)
    return examples


def transition_trace(seed: int, windows_per_class: int = 8, n: int = 32):
    rng = random.Random(seed)
    stream: list[Request] = []
    labels: list[str] = []
    for label in CLASSES:
        phase_stride = block_stride(rng)
        next_lba: int | None = None
        for _ in range(windows_per_class):
            window = synthetic_window(label, rng, n,
                                      stream[-1].timestamp_ms if stream else 0.0,
                                      start_lba=next_lba,
                                      stride_override=phase_stride)
            stream.extend(window)
            labels.append(label)
            if label == "sequential":
                next_lba = window[-1].lba + window[-1].size_blocks
            elif label == "strided":
                next_lba = window[-1].lba + phase_stride
            else:
                next_lba = None
    return stream, labels
