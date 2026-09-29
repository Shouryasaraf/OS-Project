"""Canonical trace records, CSV loading, and reproducible labelled workloads."""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

CLASSES = ("sequential", "strided", "random", "mixed")
MSR_COLUMNS = ("Timestamp", "Hostname", "DiskNumber", "Type", "Offset", "Size",
               "ResponseTime")
IOTTA8_COLUMNS = ("device", "sector", "size", "op", "offset", "timestamp",
                  "lifetime", "count")
ALIBABA_COLUMNS = ("device_id", "opcode", "offset", "length", "timestamp")
REVISED_COLUMNS = ("time_s", "op", "lba", "size_blocks", "seq_or_rand",
                   "t1", "t2", "t3")


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


def load_csv(path: str | Path, format: str = "auto") -> list[Request]:
    """Read a known trace schema; ambiguous units require an explicit profile."""
    if format not in {"auto", "normalized", "msr", "iotta8", "alibaba", "revised"}:
        raise ValueError("format must be auto, normalized, msr, iotta8, alibaba, or revised")
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


def _validate_order(requests: list[Request]) -> None:
    if any(b.timestamp_ms < a.timestamp_ms for a, b in zip(requests, requests[1:])):
        raise ValueError("CSV requests must be sorted by timestamp")


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
