"""``provenance.json`` for every ``results/<study>/`` directory.

One file per results directory records what produced it: the command, the
git commit, the host and its load (the host is shared, so a timing without
load is not evidence), the GPU state when ``nvidia-smi`` exists, and the
SHA-256 of every input and every output file. :func:`verify_provenance`
re-hashes the outputs, so a hand-edited results file is detectable.

The keys are :data:`PROVENANCE_KEYS` (always present, ``None`` when a value
was not recorded) plus :data:`OPTIONAL_KEYS`. ``schemas`` registers the same
keys as the ``*/provenance.json`` contract.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
FILENAME = "provenance.json"
FORMAT = "prefetch_common.provenance/1"

#: Present in every provenance file, in this order.
PROVENANCE_KEYS = ("format", "study", "command", "git_sha", "git_dirty",
                   "created_utc", "host", "load_avg", "wall_s", "gpu",
                   "inputs", "outputs", "notes")
#: Allowed in addition. ``source``/``copied_verbatim`` describe results that
#: were carried over rather than generated here (``results/pivot_original``).
OPTIONAL_KEYS = ("source", "copied_verbatim", "extra")

_GPU_QUERY = ("index,name,memory.used,memory.total,utilization.gpu")


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _display(path: Path) -> str:
    """Repo-relative POSIX path when inside the repo, else absolute.

    Not resolved, so ``data/block-traces/...`` stays readable when
    ``data/block-traces`` is a symlink.
    """
    absolute = Path(os.path.abspath(path))
    try:
        return absolute.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return absolute.as_posix()


def _now_utc() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _load() -> list[float] | None:
    try:
        return [round(x, 2) for x in os.getloadavg()]
    except (AttributeError, OSError):  # not on this platform
        return None


def snapshot() -> dict:
    """Host state at the start of a run; pass it to ``write_provenance(start=)``."""
    return {"utc": _now_utc(), "load_avg": _load(), "monotonic": time.monotonic()}


def _git(*args: str) -> str | None:
    try:
        result = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True,
                                text=True, timeout=30, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def git_state() -> tuple[str | None, bool | None]:
    """``(HEAD sha, dirty)``. ``dirty`` ignores untracked files and
    ``results/`` itself, which a run is expected to be writing."""
    sha = _git("rev-parse", "HEAD")
    if sha is None:
        return None, None
    status = _git("status", "--porcelain", "--untracked-files=no", "--",
                  ".", ":(exclude)results")
    return sha, (bool(status) if status is not None else None)


def host_info() -> dict:
    return {"hostname": socket.gethostname(), "platform": platform.platform(),
            "kernel": platform.release(), "python": sys.version.split()[0],
            "cpu_count": os.cpu_count()}


def gpu_info() -> list[dict] | dict | None:
    """One dict per GPU from ``nvidia-smi``; ``None`` when it is not installed.

    The GPU is shared with other tenants, so memory in use and utilisation
    are what make a GPU timing interpretable. Values are nvidia-smi's text
    with units stripped: memory in MiB, utilisation in percent.
    """
    binary = shutil.which("nvidia-smi")
    if binary is None:
        return None
    try:
        result = subprocess.run(
            [binary, f"--query-gpu={_GPU_QUERY}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20, check=False)
    except (OSError, subprocess.SubprocessError) as error:
        return {"error": str(error)}
    if result.returncode != 0:
        return {"error": (result.stderr or result.stdout).strip()[:500]}
    fields = _GPU_QUERY.split(",")
    return [dict(zip(fields, (part.strip() for part in line.split(","))))
            for line in result.stdout.splitlines() if line.strip()]


def hash_inputs(inputs: Iterable[str | Path]) -> dict:
    """``{path: sha256}`` for input files. A missing input is an error, not a
    ``None``: provenance that cannot name its input is not provenance."""
    out = {}
    for item in inputs:
        path = Path(item)
        if path.is_dir():
            raise IsADirectoryError(f"provenance input must be a file: {path}")
        if not path.is_file():
            raise FileNotFoundError(f"provenance input not found: {path}")
        out[_display(path)] = sha256_file(path)
    return out


def hash_outputs(directory: str | Path) -> dict:
    """``{relative path: sha256}`` of every file under ``directory`` except
    ``provenance.json``, sorted by path."""
    root = Path(directory)
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.name != FILENAME)
    return {p.relative_to(root).as_posix(): sha256_file(p) for p in files}


def write_provenance(directory: str | Path, cmd, inputs=(), *,
                     study: str | None = None, start: dict | None = None,
                     outputs: bool = True, gpu: str | bool = "auto",
                     notes: str | None = None, extra: dict | None = None) -> Path:
    """Write ``<directory>/provenance.json`` and return its path.

    ``cmd`` is the command line (a list, or one string). ``inputs`` are the
    files the run read; their SHA-256 is taken now. ``start`` is a :func:`snapshot` taken when the run began, giving the
    start load and the wall time. ``outputs=True`` hashes every file already
    in ``directory``, so call this last. ``gpu="auto"`` queries
    ``nvidia-smi`` when it is installed; ``True`` requires it; ``False``
    skips it. ``study`` defaults to the directory name.
    """
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    if gpu is True and shutil.which("nvidia-smi") is None:
        raise RuntimeError("gpu=True but nvidia-smi is not installed")
    sha, dirty = git_state()
    end_load = _load()
    record = {
        "format": FORMAT,
        "study": study or root.name,
        "command": list(cmd) if not isinstance(cmd, str) else cmd,
        "git_sha": sha,
        "git_dirty": dirty,
        "created_utc": _now_utc(),
        "host": host_info(),
        "load_avg": {"start": start.get("load_avg") if start else None,
                     "end": end_load},
        "wall_s": (round(time.monotonic() - start["monotonic"], 3)
                   if start and "monotonic" in start else None),
        "gpu": gpu_info() if gpu else None,
        "inputs": hash_inputs(inputs),
        "outputs": hash_outputs(root) if outputs else {},
        "notes": notes,
    }
    if extra:
        record["extra"] = extra
    path = root / FILENAME
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, indent=2, sort_keys=False) + "\n")
    os.replace(tmp, path)
    return path


def read_provenance(directory: str | Path) -> dict:
    """Load and key-check ``<directory>/provenance.json``."""
    record = json.loads((Path(directory) / FILENAME).read_text())
    problems = check_keys(record)
    if problems:
        raise ValueError(f"{Path(directory) / FILENAME}: " + "; ".join(problems))
    return record


def check_keys(record: Mapping) -> list[str]:
    """Problems with a provenance record's keys; empty when it conforms."""
    problems = []
    missing = [k for k in PROVENANCE_KEYS if k not in record]
    unknown = [k for k in record if k not in PROVENANCE_KEYS + OPTIONAL_KEYS]
    if missing:
        problems.append(f"missing keys {missing}")
    if unknown:
        problems.append(f"unknown keys {unknown}")
    if "format" in record and record["format"] != FORMAT:
        problems.append(f"format {record.get('format')!r} is not {FORMAT!r}")
    return problems


def verify_provenance(directory: str | Path) -> list[str]:
    """Re-hash ``directory`` against its provenance; returns the mismatches.

    Reports changed, missing and unlisted output files. Inputs are not
    re-hashed (they are usually large and outside the repo); compare them
    with :func:`hash_inputs` when needed.
    """
    record = read_provenance(directory)
    recorded = record["outputs"]
    actual = hash_outputs(directory)
    problems = []
    for name, digest in recorded.items():
        if name not in actual:
            problems.append(f"missing output {name}")
        elif actual[name] != digest:
            problems.append(f"changed output {name}")
    problems += [f"unlisted output {name}" for name in actual if name not in recorded]
    return problems
