"""Shared plumbing for the three paper generators (WP13).

``numbers.py``, ``tables/make_tables.py`` and ``figures/make_figures.py``
read only ``results/**`` (or a tree laid out like it, such as the schema-exact
fixtures in ``tests/fixtures/results/``) and write into ``paper/``. This
module holds what they share:

- :class:`Results` opens a results tree. Every file is read through
  ``prefetch_common.schemas.read_rows``, so a file that breaks its schema
  fails loudly instead of being half-read; a file that is absent is recorded
  as missing, and the caller writes a placeholder. Nothing is ever
  substituted for a missing value.
- :class:`Output` records, per generated file, which inputs it read (with
  their SHA-256 and whether that hash matches the study's
  ``provenance.json``) and which were missing. ``write_manifest`` stores the
  records next to the outputs, so WP14 can show every figure, table and
  number traces back to a provenance-tagged results file.
- LaTeX helpers: escaping, number formatting (fractions in, percent or
  percentage points out), and macro names, which may hold letters only.

A tree that holds a ``FIXTURE`` marker file is synthetic test data. Every
output built from it is stamped "FIXTURE DATA" so it can never pass for a
result.

Stdlib only; the figure generator adds matplotlib itself.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

PAPER_DIR = Path(__file__).resolve().parent
REPO_ROOT = PAPER_DIR.parent
DEFAULT_RESULTS = REPO_ROOT / "results"
FIXTURE_RESULTS = REPO_ROOT / "tests" / "fixtures" / "results"
FIXTURE_MARKER = "FIXTURE"
MANIFEST_NAME = "manifest.json"

if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from prefetch_common import schemas  # noqa: E402
from prefetch_common.provenance import FILENAME as PROVENANCE_FILE  # noqa: E402

#: The headline disk configuration: the C1 collection table on canonical
#: requests with the explicit assumed cost model (MASTER_PLAN 2.3, honesty
#: constraint 13). Other configurations appear only where a figure needs them.
HEADLINE_CONFIG = "C1"
HEADLINE_MERGE = "auto"
HEADLINE_LATENCY = "assumed"
#: Disk capacities of the T2 table, in 512-B blocks (1 MiB and 64 MiB).
HEADLINE_CAPACITIES = (2048, 131072)
GATED_MODES = ("guard", "depth_adaptive", "correlate")

MISSING_TEXT = "MISSING"
FIXTURE_TEXT = "FIXTURE DATA"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Reading results
# ---------------------------------------------------------------------------

@dataclass
class Output:
    """What one generated file read: inputs with hashes, and what was missing."""
    name: str
    inputs: dict = field(default_factory=dict)
    missing: list = field(default_factory=list)
    fixture: bool = False
    note: str | None = None

    @property
    def complete(self) -> bool:
        return not self.missing

    def record(self) -> dict:
        out = {"inputs": dict(sorted(self.inputs.items())),
               "missing": sorted(set(self.missing)), "fixture": self.fixture}
        if self.note:
            out["note"] = self.note
        return out


class Results:
    """A results tree (``results/`` or a fixture tree laid out like it)."""

    def __init__(self, root: Path | str = DEFAULT_RESULTS):
        self.root = Path(root)
        self.fixture = (self.root / FIXTURE_MARKER).is_file()
        self._cache: dict[str, list[dict] | None] = {}
        self._provenance: dict[str, dict | None] = {}

    # -- inputs ------------------------------------------------------------
    def note_input(self, output: Output | None, relative: str) -> None:
        if output is None:
            return
        active, TRACE.active = TRACE.active, False  # bookkeeping, not a claim's read
        try:
            self._note_input(output, relative)
        finally:
            TRACE.active = active

    def _note_input(self, output: Output, relative: str) -> None:
        path = self.root / relative
        digest = sha256_file(path)
        study, _, inner = relative.partition("/")
        record = {"sha256": digest, "provenance_git_sha": None, "provenance_match": None}
        provenance = self.provenance(study)
        if provenance is not None:
            record["provenance_git_sha"] = provenance.get("git_sha")
            recorded = (provenance.get("outputs") or {}).get(inner)
            record["provenance_match"] = None if recorded is None else recorded == digest
        output.inputs[relative] = record

    def provenance(self, study: str) -> dict | None:
        if study not in self._provenance:
            path = self.root / study / PROVENANCE_FILE
            relative = f"{study}/{PROVENANCE_FILE}"
            self._provenance[study] = (traced(schemas.read_rows(relative, path)[0], relative, 0)
                                       if path.is_file() else None)
            if path.is_file():
                TRACE.loaded.add(relative)
        return self._provenance[study]

    def exists(self, relative: str) -> bool:
        return (self.root / relative).is_file()

    def rows(self, relative: str, output: Output | None = None, *,
             required: bool = True) -> list[dict] | None:
        """Rows of a schema'd results file, or ``None`` when it is absent.

        An absent file is added to ``output.missing`` when ``required``.
        A present file that breaks its schema raises ``SchemaError``.
        """
        if relative not in self._cache:
            path = self.root / relative
            self._cache[relative] = ([traced(row, relative, i) for i, row in
                                      enumerate(schemas.read_rows(relative, path))]
                                     if path.is_file() else None)
        rows = self._cache[relative]
        if rows is None:
            if output is not None and required:
                output.missing.append(relative)
            return None
        self.note_input(output, relative)
        TRACE.loaded.add(relative)
        return rows

    def json_object(self, relative: str, output: Output | None = None, *,
                    required: bool = True) -> dict | None:
        """A JSON results file that has no schema yet (MASTER_PLAN notes #14).

        Read as plain JSON; the caller treats every key as optional and
        prints "--" for a key that is not there, never a guessed value.
        """
        path = self.root / relative
        if not path.is_file():
            if output is not None and required:
                output.missing.append(relative)
            return None
        payload = traced(json.loads(path.read_text(encoding="utf-8")), relative, 0)
        self.note_input(output, relative)
        TRACE.loaded.add(relative)
        return payload

    def glob(self, pattern: str, output: Output | None = None, *,
             required: bool = True) -> dict[str, dict]:
        """``{relative path: object}`` for schema'd JSON files matching ``pattern``."""
        found = {}
        for path in sorted(self.root.glob(pattern)):
            relative = path.relative_to(self.root).as_posix()
            found[relative] = self.rows(relative, output)[0]
        if not found and output is not None and required:
            output.missing.append(pattern)
        return found

    @property
    def display_root(self) -> str:
        """The tree's path, repo-relative when it is inside the repository."""
        try:
            return self.root.resolve().relative_to(REPO_ROOT).as_posix()
        except ValueError:
            return self.root.resolve().as_posix()

    def output(self, name: str) -> Output:
        return Output(name, fixture=self.fixture)


def write_manifest(directory: Path, outputs: Iterable[Output], results: Results,
                   name: str = MANIFEST_NAME) -> Path:
    """Write ``directory/name``: per generated file, its inputs and gaps.

    No timestamps, so regenerating from unchanged results leaves the file
    byte-identical.
    """
    payload = {"generator_format": "paper.genlib/1", "results_root": results.display_root,
               "fixture": results.fixture,
               "outputs": {o.name: o.record() for o in sorted(outputs, key=lambda o: o.name)}}
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Row selection
# ---------------------------------------------------------------------------

def select(rows: Sequence[dict] | None, **criteria) -> list[dict]:
    """Rows whose fields equal every ``criteria`` value."""
    if not rows:
        return []
    return [row for row in rows if all(row.get(k) == v for k, v in criteria.items())]


def distinct(rows: Sequence[dict], key: str) -> list:
    seen = []
    for row in rows:
        if row.get(key) not in seen:
            seen.append(row.get(key))
    return seen


def _mode_order() -> tuple[str, ...]:
    try:
        from adaptive_prefetch.benchmark import MODES
    except ImportError:  # the disk package is always importable from src/
        MODES = ()
    return tuple(MODES) + ("learned_depth",)


_MODE_ORDER = _mode_order()


#: A live pass whose store device carried more foreign I/O than this share
#: is flagged and never cited (E6 pre-registration; lead ruling for SSD rows).
LIVE_FOREIGN_LIMIT = 0.01


def live_citable(rows: Sequence[dict]) -> list[dict]:
    """Valid live passes that may be cited: the validity gate passed and the
    store's foreign I/O share was at most ``LIVE_FOREIGN_LIMIT`` (or not
    recorded). Flagged passes stay in the results file; they are only kept
    out of every number the paper quotes."""
    return [r for r in rows if r["valid"] and (r.get("foreign_io_frac") is None
                                               or r["foreign_io_frac"] <= LIVE_FOREIGN_LIMIT)]


def live_comparable(items: Sequence[dict]) -> list[dict]:
    """Valid live passes that carry a delta against their baseline.

    ``webprefetch.live.analyze`` writes NaN deltas on a valid pass whose
    baseline pass failed the validity gate, so such a pass is kept out of the
    savings rather than turning their mean into NaN.
    """
    return [r for r in items if r["valid"] and finite(r["delta_vs_base_ms"])
            and finite(r["delta_vs_base_ms_lo"]) and finite(r["delta_vs_base_ms_hi"])]


def live_saved(items: Sequence[dict]):
    """``(estimate, low, high)`` milliseconds saved per view over comparable
    passes: the mean saving, and the envelope of the per-pass 95% CIs (saved =
    -delta_vs_base_ms; the per-pass CIs are not pooled). ``None`` when no pass
    is comparable."""
    items = live_comparable(items)
    if not items:
        return None
    return (-sum(r["delta_vs_base_ms"] for r in items) / len(items),
            -max(r["delta_vs_base_ms_hi"] for r in items),
            -min(r["delta_vs_base_ms_lo"] for r in items))


def sweep_label(row) -> str:
    """Sweep row -> table policy label (deep@8, linux_ra@256)."""
    if row["mode"] == "deep":
        return f"deep@{row['depth']}"
    if row["mode"] == "linux_ra":
        return f"linux_ra@{row['max_blocks']}"
    return row["mode"]


def disk_utility_rows(sweep, capacity):
    """Mean over traces of each policy's hit ratio and prefetches per block,
    on the traces where every drawn policy has an ok row (paired).

    Only policies with an ok row on every trace any policy covers are drawn,
    so one partially covered mode cannot shrink every curve to its traces.
    """
    rows = [r for r in sweep if r["config"] == HEADLINE_CONFIG and r["merge"] == HEADLINE_MERGE
            and r["latency"] == HEADLINE_LATENCY and r["capacity"] == capacity]
    by_policy = defaultdict(dict)
    for row in rows:
        if row["status"] == "ok":
            by_policy[sweep_label(row)][row["trace"]] = row
    if not by_policy:
        return [], 0
    covered = set.union(*(set(v) for v in by_policy.values()))
    by_policy = {p: v for p, v in by_policy.items() if set(v) == covered}
    if not by_policy:
        return [], 0
    traces = covered
    out = []
    for policy in sorted(by_policy, key=policy_key):
        items = [by_policy[policy][t] for t in sorted(traces)]
        if items:
            out.append({"policy": policy, "requests": 1.0,
                        "hit_ratio": statistics.fmean(r["hit_ratio"] for r in items),
                        "issued_per_request": statistics.fmean(
                            r["total_prefetches"] / r["read_blocks"] for r in items)})
    return out, len(traces)


def policy_key(policy: str) -> tuple:
    """Sort key: benchmark.MODES order, then the numeric ``@`` parameter."""
    base, _, parameter = str(policy).partition("@")
    rank = _MODE_ORDER.index(base) if base in _MODE_ORDER else len(_MODE_ORDER)
    number = re.match(r"\d+", parameter)
    return (rank, base, int(number.group(0)) if number else -1, parameter)


def capacity_label(blocks: int) -> str:
    """512-B blocks as a cache size: 2048 -> '1 MiB'."""
    size = blocks * 512
    for unit, scale in (("GiB", 1 << 30), ("MiB", 1 << 20), ("KiB", 1 << 10)):
        if size >= scale and size % scale == 0:
            return f"{size // scale} {unit}"
    return f"{blocks} blocks"



# ---------------------------------------------------------------------------
# Claim tracing (the claims ledger, results/claims/)
# ---------------------------------------------------------------------------
#
# Every value a generator prints must trace back to a results file. Two
# hooks make that automatic, without changing any printed byte:
#
# - :class:`Results` hands out rows as :class:`TracedRow` (a ``dict``) that
#   tell :data:`TRACE` which file, row and column every read came from;
# - the formatters below (``fnum``, ``fpct``, ``fpp``, ``fratio``,
#   ``fcount``, ``fci``, ``esc``, and any function wrapped in
#   :func:`traced_formatter`) tell it which raw value they printed.
#
# ``numbers.Macros.add`` and ``paper/claims.py`` match the raw values to the
# reads (by object identity, then by equality on the most recent reads), so
# each macro or table cell records ``(file, row, column, raw value,
# formatter)``. Tracing is off unless a caller switches it on, so it costs
# nothing elsewhere.


class _Trace:
    WINDOW = 512

    def __init__(self):
        self.active = False
        #: results-relative files any Results has loaded (kept across resets).
        self.loaded: set = set()
        self.reset()

    def reset(self) -> None:
        from collections import deque
        self.depth = 0
        self.calls: list = []
        self.recent = deque(maxlen=self.WINDOW)
        self.window_rows: set = set()
        self.window_columns: set = set()
        self.window_files: set = set()
        self.floats: dict = {}
        self.files_read: set = set()

    def read(self, row: "TracedRow", key, value) -> None:
        column = row._prefix + str(key)
        where = (row._relative, row._index, column)
        self.recent.append((where, value))
        self.window_rows.add((row._relative, row._index))
        self.window_columns.add((row._relative, column))
        self.window_files.add(row._relative)
        self.files_read.add(row._relative)
        if isinstance(value, float):
            self.floats.setdefault(id(value), (value, where))

    def take(self):
        """The formatter calls and the window of reads since the last take."""
        calls, rows = self.calls, self.window_rows
        columns, files = self.window_columns, self.window_files
        recent = list(self.recent)
        self.calls, self.window_rows, self.window_columns = [], set(), set()
        self.window_files = set()
        self.recent.clear()
        return calls, recent, rows, columns, files

    def locate(self, value, recent) -> tuple | None:
        """``(file, row index, column)`` a raw value was read from, or None."""
        if isinstance(value, float):
            found = self.floats.get(id(value))
            if found is not None and found[0] is value:
                return found[1]
        if value is None or isinstance(value, (dict, list, tuple)):
            return None
        for where, seen in reversed(recent):
            if seen is value:
                return where
        for where, seen in reversed(recent):
            if type(seen) is type(value) and seen == value:
                return where
        if isinstance(value, float):
            # A fraction printed as percent (100 * x) or a saving (-x).
            for where, seen in reversed(recent):
                if isinstance(seen, float):
                    for scale in SCALES:
                        if scale * seen == value:
                            return where + (scale,)
        return None


#: Scalings of a read value that ``_Trace.locate`` recognises; the ledger
#: writes the column as ``column*scale``.
SCALES = (100, -1, -100)


def trace_column(where) -> str:
    """The ledger's column text for a located read: ``col`` or ``col*100``."""
    return where[2] if len(where) == 3 else f"{where[2]}*{where[3]}"


TRACE = _Trace()


class TracedRow(dict):
    """A results row that reports every read to :data:`TRACE` when it is on."""

    __slots__ = ("_relative", "_index", "_prefix")

    def __getitem__(self, key):
        value = dict.__getitem__(self, key)
        if TRACE.active:
            TRACE.read(self, key, value)
        return value

    def get(self, key, default=None):
        if key in self:
            return self[key]
        return default


def traced(value, relative: str, index: int, prefix: str = ""):
    """Wrap a parsed row (recursively for nested JSON) as :class:`TracedRow`."""
    if isinstance(value, dict):
        row = TracedRow()
        row._relative, row._index, row._prefix = relative, index, prefix
        for key, item in value.items():
            dict.__setitem__(row, key, traced(item, relative, index, f"{prefix}{key}."))
        return row
    if isinstance(value, list):
        return [traced(item, relative, index, f"{prefix[:-1]}[{i}].")
                if isinstance(item, (dict, list)) else item for i, item in enumerate(value)]
    return value


def traced_formatter(function=None, *, raw: int = 1):
    """Make a formatter report ``(name, raw values, other args, output)`` to
    :data:`TRACE`. ``raw`` is how many leading positional arguments are the
    values printed. Only the outermost formatter call is recorded (``fpct``
    calling ``fnum`` is one call). A generator that prints a number through a
    function of its own must wrap it with this, or its macro has no traced
    value and ``numbers.Macros.add`` refuses it."""
    import functools

    def wrap(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if not TRACE.active:
                return fn(*args, **kwargs)
            TRACE.depth += 1
            try:
                text = fn(*args, **kwargs)
            finally:
                TRACE.depth -= 1
            if TRACE.depth == 0:
                values = args[:raw]
                TRACE.calls.append((fn.__name__, values, args[raw:], kwargs, text,
                                    [TRACE.locate(v, TRACE.recent) for v in values]))
            return text
        wrapper.traced_raw = raw
        FORMATTERS[fn.__name__] = wrapper
        return wrapper

    return wrap(function) if function is not None else wrap


#: Every traced formatter by name; the ledger re-applies them by this name.
FORMATTERS: dict = {}

# ---------------------------------------------------------------------------
# LaTeX text and numbers
# ---------------------------------------------------------------------------

_ESCAPES = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
            "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
            "^": r"\textasciicircum{}"}


@traced_formatter
def esc(text) -> str:
    """Escape a plain string for LaTeX text mode."""
    return "".join(_ESCAPES.get(ch, ch) for ch in str(text))


def finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


DASH = "--"


@traced_formatter
def fnum(value, digits: int = 2, *, sign: bool = False) -> str:
    """A number in math mode: ``$-0.89$``. ``--`` for None; inf spelled out."""
    if value is None:
        return DASH
    if isinstance(value, float) and math.isinf(value):
        return r"$\infty$" if value > 0 else r"$-\infty$"
    if isinstance(value, float) and math.isnan(value):
        return DASH
    text = f"{value:+.{digits}f}" if sign else f"{value:.{digits}f}"
    if float(text) == 0:
        text = f"{0:.{digits}f}"  # never print -0.00 or +0.00
    return f"${text}$"


@traced_formatter
def fpct(fraction, digits: int = 2) -> str:
    """A fraction as a percentage: 0.35723 -> ``$35.72$\\,\\%``."""
    if not finite(fraction):
        return fnum(fraction)
    return fnum(100.0 * fraction, digits) + r"\,\%"


@traced_formatter
def fpp(fraction, digits: int = 2) -> str:
    """A fractional difference in percentage points: ``$+4.10$\\,pp``."""
    if not finite(fraction):
        return fnum(fraction)
    return fnum(100.0 * fraction, digits, sign=True) + r"\,pp"


@traced_formatter
def fratio(value, digits: int = 3) -> str:
    """A ratio as a multiplier: ``$0.659\\times$``."""
    if not finite(value):
        return fnum(value)
    return f"${value:.{digits}f}\\times$"


@traced_formatter
def fcount(value) -> str:
    """An integer with thin grouping LaTeX accepts in text: ``2{,}756{,}101``."""
    if value is None:
        return DASH
    return f"{int(value):,}".replace(",", "{,}")


@traced_formatter(raw=2)
def fci(lo, hi, formatter, **kwargs) -> str:
    """``[lo, hi]`` with the same formatter as the estimate, minus its unit."""
    if lo is None or hi is None:
        return ""
    strip = (r"\,pp", r"\,\%")
    parts = []
    for value in (lo, hi):
        text = formatter(value, **kwargs)
        for unit in strip:
            text = text.replace(unit, "")
        parts.append(text)
    return f"[{parts[0]}, {parts[1]}]"


_ONES = ("Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine",
         "Ten", "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen",
         "Seventeen", "Eighteen", "Nineteen")
_TENS = ("", "", "Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety")


def int_words(n: int) -> str:
    """An integer as CamelCase English: 256 -> 'TwoHundredFiftySix'."""
    if n < 0:
        return "Minus" + int_words(-n)
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + (_ONES[n % 10] if n % 10 else "")
    for scale, word in ((10 ** 9, "Billion"), (10 ** 6, "Million"), (1000, "Thousand"),
                        (100, "Hundred")):
        if n >= scale:
            head, rest = divmod(n, scale)
            return int_words(head) + word + (int_words(rest) if rest else "")
    raise AssertionError(n)


def camel(text) -> str:
    """Letters-only CamelCase for a macro name part.

    Digits become words ('deep@8' -> 'DeepAtEight', 'msr_hm_0' ->
    'MsrHmZero', '0.01' -> 'ZeroPointZeroOne'), since a TeX control word holds
    letters only.
    """
    text = str(text).replace("@", " at ").replace("+", " plus ")
    out = []
    for token in re.findall(r"[A-Za-z]+|\d+(?:\.\d+)?", text):
        if token[0].isdigit():
            whole, _, fraction = token.partition(".")
            word = int_words(int(whole))
            if fraction:
                word += "Point" + "".join(_ONES[int(d)] for d in fraction)
            out.append(word)
        else:
            out.append(token[:1].upper() + token[1:])
    return "".join(out)


def disk_capacity_word(blocks: int) -> str:
    """'OneMiB' for 2048 blocks: the capacity part of a disk macro name."""
    size, unit = capacity_label(blocks).split(" ")
    return int_words(int(size)) + ("Blocks" if unit == "blocks" else unit)


def is_macro_name(name: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z]+", name))
