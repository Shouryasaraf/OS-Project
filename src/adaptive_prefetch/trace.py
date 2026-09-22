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

    def __post_init__(self) -> None:
        if self.timestamp_ms < 0 or self.lba < 0 or self.size_blocks <= 0:
            raise ValueError("timestamp/lba must be non-negative and size_blocks positive")
        if self.operation not in {"R", "W"}:
            raise ValueError("operation must be R or W")


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
            except (ValueError, TypeError) as exc:
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

    Each line is  ``time_s op lba size seq|rand t1 t2 t3`` where op is RS
    (read) or WS (write), lba and size are 512-byte sectors, time_s is in
    seconds, and the trailing columns are per-request timing/flag fields
    not needed by the simulator. Timestamps are normalized to elapsed
    milliseconds from the first request, matching the canonical units.
    """
    rows: list[tuple[float, str, int, int]] = []
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
            except (ValueError, TypeError) as exc:
                raise ValueError(f"invalid revised row {line_no}: {exc}") from exc
            if op not in {"RS", "WS"}:
                raise ValueError(f"invalid revised row {line_no}: op must be RS or WS")
            rows.append((seconds, op, lba, size))
    if not rows:
        raise ValueError("CSV trace is empty")
    first = rows[0][0]
    requests = [Request((seconds - first) * 1000, lba, size,
                        "R" if op == "RS" else "W")
                for seconds, op, lba, size in rows]
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


def synthetic_window(label: str, rng: random.Random, n: int = 32,
                     timestamp_start: float = 0.0, start_lba: int | None = None,
                     stride_override: int | None = None) -> list[Request]:
    """Generate one labelled window; label describes the construction, not a real trace."""
    if label not in CLASSES or n < 8:
        raise ValueError("label must be a known class and n >= 8")
    current = start_lba if start_lba is not None else rng.randrange(1000, 100000)
    stride = stride_override if stride_override is not None else rng.randint(4, 16)
    time = timestamp_start
    result: list[Request] = []
    for i in range(n):
        size = rng.randint(1, 3)
        if label == "sequential":
            if i and rng.random() < 0.04:
                current += rng.randint(5, 20)
        elif label == "strided":
            if i:
                current += stride + (rng.randint(1, 5) if rng.random() < 0.04 else 0)
        elif label == "random":
            current = rng.randrange(1000, 100000)
        else:
            # Interleave short sequential runs and distant random jumps.
            if i % 8 == 0:
                current = rng.randrange(1000, 100000)
            elif i % 8 >= 4:
                current = rng.randrange(1000, 100000)
        gap = rng.uniform(0.2, 2.0)
        if label == "mixed" and i % 8 < 4:
            gap = rng.uniform(0.01, 0.1)
        time += gap
        result.append(Request(round(time, 4), current, size,
                              "R" if rng.random() < 0.9 else "W"))
        if label in {"sequential", "mixed"} and (label != "mixed" or i % 8 < 3):
            current += size
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
        phase_stride = rng.randint(4, 16)
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
