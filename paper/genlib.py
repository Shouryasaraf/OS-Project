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
import sys
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
            self._provenance[study] = (schemas.read_rows(f"{study}/{PROVENANCE_FILE}", path)[0]
                                       if path.is_file() else None)
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
            self._cache[relative] = (schemas.read_rows(relative, path)
                                     if path.is_file() else None)
        rows = self._cache[relative]
        if rows is None:
            if output is not None and required:
                output.missing.append(relative)
            return None
        self.note_input(output, relative)
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
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.note_input(output, relative)
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
# LaTeX text and numbers
# ---------------------------------------------------------------------------

_ESCAPES = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
            "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
            "^": r"\textasciicircum{}"}


def esc(text) -> str:
    """Escape a plain string for LaTeX text mode."""
    return "".join(_ESCAPES.get(ch, ch) for ch in str(text))


def finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


DASH = "--"


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


def fpct(fraction, digits: int = 2) -> str:
    """A fraction as a percentage: 0.35723 -> ``$35.72$\\,\\%``."""
    if not finite(fraction):
        return fnum(fraction)
    return fnum(100.0 * fraction, digits) + r"\,\%"


def fpp(fraction, digits: int = 2) -> str:
    """A fractional difference in percentage points: ``$+4.10$\\,pp``."""
    if not finite(fraction):
        return fnum(fraction)
    return fnum(100.0 * fraction, digits, sign=True) + r"\,pp"


def fratio(value, digits: int = 3) -> str:
    """A ratio as a multiplier: ``$0.659\\times$``."""
    if not finite(value):
        return fnum(value)
    return f"${value:.{digits}f}\\times$"


def fcount(value) -> str:
    """An integer with thin grouping LaTeX accepts in text: ``2{,}756{,}101``."""
    if value is None:
        return DASH
    return f"{int(value):,}".replace(",", "{,}")


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
