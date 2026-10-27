"""Flag numbers in the docs that do not come from ``results/``.

    python3 tools/check_doc_numbers.py [--root DIR] [--verbose]

MASTER_PLAN section 5 (honesty constraint 12) and WP15: no number is typed
into README, AGENTS or ``docs/`` unless it is in a results file. This is the
docs' counterpart of ``paper/check.py``'s hand-number lint, and it reuses
that lint's patterns (``hand_numbers``, ``ALLOW_PATTERNS``), so the two agree
on what a figure is and on the allow-list (versions, section and decision
numbers, dimensions). It adds ungrouped integers of four or more digits
("4131 s"), except years and powers of two (cache and block geometry).

What a number is checked against -- its **scope** -- is the point. Membership
anywhere in ``results/`` is nearly vacuous: about 180,000 distinct values sit
there, so almost any two-decimal figure occurs somewhere by accident. A
figure is therefore checked against the narrowest source its text names:

1. ``<!-- source: results/disk/table_main.csv config=C1 capacity=2048 -->``
   (several sources separated by ``;``) scopes every following block, up to
   the next heading or the next ``source`` directive, to the **rows** that
   match the filters. ``<!-- source: none -->`` ends it. This is the strong
   check: a typo or a stale figure almost never matches by chance.
2. Otherwise a block (paragraph, list or table) that names a results file
   or directory (``results/web/decomposition.csv``, a link to
   ``../results/live/``) is checked against those files. A table that names
   none inherits the paragraph just before it.
3. Otherwise the figure is flagged: it names no source. Checking it against
   the whole tree would pass nearly anything, so a figure must say where it
   comes from (claim discipline, AGENTS.md).

Matching. A figure with ``d`` decimals matches a value ``v`` when ``v`` (or
``100 v``: results store fractions, prose prints percent and pp) rounds to it
at ``d`` decimals. A signed figure (``+0.36``, ``-0.94``, Unicode minus)
must match the sign; an unsigned one matches either. ``a/b`` needs both
``a`` and ``b`` in scope. Thousands separators are ignored.

Exempt text, always visible in the file itself rather than hidden here:
fenced code and inline code (commands, flags, identifiers such as
``deep@8``), link targets, HTML comments, and regions between
``<!-- doc-numbers: off (reason) -->`` and ``<!-- doc-numbers: on -->``,
used for history records whose figures came from runs that ``results/`` does
not hold (DECISIONS D1-D17 on MSRC-trace-003). ``EXCLUDED`` lists the two
docs WP15 does not own.

``presentation/slides.tex`` gets ``paper/check.py``'s LaTeX lint: every
number on a slide must be a ``numbers.tex`` macro.

Stdlib only; exit status 1 when anything is flagged.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: The Markdown files checked, as globs relative to the repository root.
DOC_GLOBS = ("README.md", "AGENTS.md", "docs/**/*.md", "presentation/*.md")
#: Docs WP15 does not own, with the reason they are not checked.
EXCLUDED = {
    "docs/MASTER_PLAN.md": "the binding work plan (lead-owned); its figures are pilot "
                           "measurements it labels as orientation only",
    "docs/REFERENCES.md": "bibliography corrections: years, volumes and page ranges",
}
#: LaTeX files whose every number must be a numbers.tex macro.
LATEX_FILES = ("presentation/slides.tex",)

_DIRECTIVE = re.compile(r"<!--\s*(doc-numbers|source)\s*:\s*(.*?)\s*-->", re.S)
_COMMENT = re.compile(r"<!--.*?-->", re.S)
_FENCE = re.compile(r"^(```|~~~).*?^\1[^\n]*$", re.S | re.M)
_CODE_SPAN = re.compile(r"(`+)(.+?)\1", re.S)
_LINK_TARGET = re.compile(r"\]\([^)]*\)")
_URL = re.compile(r"https?://\S+")
_RESULTS_PATH = re.compile(r"(?:\.\./)*results/[\w./*-]*[\w*/]")
_HEADING = re.compile(r"^#{1,6}\s")
_LONG_INT = re.compile(r"(?<![\w.,/+-])\d{4,}(?![\w.,/])")
#: A ratio that ends a sentence ("on 49/56."), which check.py's ratio pattern
#: leaves out because of the full stop.
_FINAL_RATIO = re.compile(r"(?<![\w./])\d+\s*/\s*\d+(?=\.(?:\s|$))", re.M)
#: Cross-references written in lower case ("plan section 2.6"), which
#: check.py's case-sensitive Section pattern does not cover.
DOC_ALLOW_PATTERNS = (re.compile(r"(?i:\bsections?)\s*\d+(?:\.\d+)*"),)
_NUMBER_IN_TEXT = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
_FIGURE = re.compile(r"^([-+−]?)\s*(\d[\d,]*)(?:\.(\d+))?\s*(%|pp|x|×)?$")
MAX_DECIMALS = 8


def load_check(root: Path = ROOT):
    """``paper/check.py``, loaded by path (``paper/`` is not a package)."""
    name = "paper_check"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, root / "paper" / "check.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


# ----------------------------------------------------------------- values

def _numbers(obj, out: set[float]) -> None:
    """Every number in a parsed JSON value, or in the text of a string."""
    if isinstance(obj, bool) or obj is None:
        return
    if isinstance(obj, (int, float)):
        out.add(float(obj))
    elif isinstance(obj, str):
        for match in _NUMBER_IN_TEXT.finditer(re.sub(r"(?<=\d),(?=\d{3})", "", obj)):
            try:
                out.add(float(match.group(0)))
            except ValueError:
                pass
    elif isinstance(obj, dict):
        for value in obj.values():
            _numbers(value, out)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            _numbers(value, out)


def _row_matches(row: dict, filters: dict[str, str]) -> bool:
    return all(str(row.get(key, "")) == value for key, value in filters.items())


class Index:
    """Rounded keys of a set of values, for figure lookups."""

    def __init__(self, values: set[float]):
        self.signed: set[tuple[int, float]] = set()
        self.unsigned: set[tuple[int, float]] = set()
        for value in values:
            if value != value or value in (float("inf"), float("-inf")):
                continue
            for scaled in (value, 100.0 * value):
                for digits in range(MAX_DECIMALS + 1):
                    rounded = round(scaled, digits)
                    self.signed.add((digits, rounded))
                    self.unsigned.add((digits, abs(rounded)))

    def has(self, value: float, digits: int, signed: bool) -> bool:
        if signed:
            return (digits, round(value, digits)) in self.signed
        return (digits, round(abs(value), digits)) in self.unsigned


@dataclass
class Pool:
    """Values from ``results/``, per file and row filter, built lazily."""

    root: Path
    _cache: dict = field(default_factory=dict)

    @property
    def results(self) -> Path:
        return self.root / "results"

    def files(self, target: str) -> list[Path]:
        """The results files a cited path names (a file, a directory or a glob)."""
        relative = target.split("results/", 1)[1].strip("/") if "results/" in target else ""
        base = self.results / relative if relative else self.results
        if any(ch in relative for ch in "*?["):
            return sorted(p for p in self.results.glob(relative) if p.is_file())
        if base.is_dir():
            return sorted(p for p in base.rglob("*") if p.is_file())
        return [base] if base.is_file() else []

    def values(self, path: Path, filters: dict[str, str] | None = None) -> set[float]:
        key = (path, tuple(sorted((filters or {}).items())))
        if key in self._cache:
            return self._cache[key]
        out: set[float] = set()
        suffix = path.suffix
        text = path.read_text(encoding="utf-8", errors="replace")
        if suffix == ".csv":
            for row in csv.DictReader(text.splitlines()):
                if not filters or _row_matches(row, filters):
                    _numbers(list(row.values()), out)
        elif suffix == ".jsonl":
            for line in text.splitlines():
                if line.strip():
                    row = json.loads(line)
                    if not filters or _row_matches(row, filters):
                        _numbers(row, out)
        elif suffix == ".json":
            _numbers(json.loads(text), out)
        elif suffix in (".md", ".txt", ".tex"):
            _numbers(text, out)
        self._cache[key] = out
        return out

    def index(self, sources: tuple) -> Index:
        key = ("index", sources)
        if key not in self._cache:
            values: set[float] = set()
            for target, filters in sources:
                for path in self.files(target):
                    values |= self.values(path, dict(filters) or None)
            self._cache[key] = Index(values)
        return self._cache[key]


def parse_figure(figure: str) -> list[tuple[float, int, bool]] | None:
    """``(value, decimals, signed)`` for each number a figure states.

    ``a/b`` gives two; an unparseable figure gives ``None``.
    """
    figure = figure.strip()
    if "/" in figure:
        parts = [p.strip() for p in figure.split("/")]
        if len(parts) == 2 and all(p.isdigit() for p in parts):
            return [(float(p), 0, False) for p in parts]
        return None
    match = _FIGURE.match(figure.replace(" ", " "))
    if not match:
        return None
    sign, whole, fraction, _unit = match.groups()
    digits = len(fraction) if fraction else 0
    if digits > MAX_DECIMALS:
        return None
    value = float(whole.replace(",", "") + ("." + fraction if fraction else ""))
    if sign in ("-", "−"):
        value = -value
    return [(value, digits, bool(sign))]


# --------------------------------------------------------------- markdown

@dataclass
class Finding:
    path: str
    line: int
    figure: str
    scope: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.figure!r} not in {self.scope}"


@dataclass
class Report:
    findings: list[Finding] = field(default_factory=list)
    checked: dict[str, int] = field(default_factory=lambda: {"rows": 0, "files": 0,
                                                             "uncited": 0})
    exempt_files: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.findings


def _blank_spans(text: str, pattern: re.Pattern[str]) -> str:
    return re.sub(pattern, lambda m: "".join("\n" if c == "\n" else " " for c in m.group(0)),
                  text)


def _parse_sources(spec: str) -> tuple:
    """``results/a.csv k=v; results/b.json`` -> ((target, ((k, v),)), ...)."""
    if spec.strip().lower() in ("none", "end", ""):
        return ()
    sources = []
    for part in spec.split(";"):
        tokens = part.split()
        if not tokens:
            continue
        filters = tuple(tuple(t.split("=", 1)) for t in tokens[1:] if "=" in t)
        sources.append((tokens[0], filters))
    return tuple(sources)


def _line_states(text: str) -> tuple[list[bool], list[tuple | None]]:
    """Per line: is it exempt (``doc-numbers: off``), and its explicit source."""
    lines = text.split("\n")
    exempt = [False] * len(lines)
    explicit: list[tuple | None] = [None] * len(lines)
    events = []
    for match in _DIRECTIVE.finditer(text):
        events.append((text.count("\n", 0, match.start()), match.group(1),
                       match.group(2)))
    off, source = False, None
    pending = {number: (kind, value) for number, kind, value in events}
    for number, line in enumerate(lines):
        if number in pending:
            kind, value = pending[number]
            if kind == "doc-numbers":
                off = value.strip().lower().startswith("off")
            else:
                source = _parse_sources(value) or None
        elif _HEADING.match(line):
            source = None
        exempt[number] = off
        explicit[number] = source
    return exempt, explicit


def _blocks(lines: list[str]) -> list[tuple[int, int]]:
    """``(first, last)`` line indices of each blank-line separated block."""
    blocks, start = [], None
    for number, line in enumerate(lines):
        if line.strip():
            if start is None:
                start = number
        elif start is not None:
            blocks.append((start, number - 1))
            start = None
    if start is not None:
        blocks.append((start, len(lines) - 1))
    return blocks


def check_markdown(path: Path, pool: Pool, check, report: Report, relative: str) -> None:
    raw = path.read_text(encoding="utf-8")
    exempt, explicit = _line_states(raw)
    text = _blank_spans(raw, _FENCE)
    text = _blank_spans(text, _COMMENT)
    # Citations are read before code spans are blanked: a path is usually
    # written as `results/...`.
    cited_lines = text.split("\n")
    text = _blank_spans(text, _CODE_SPAN)
    text = _blank_spans(text, _LINK_TARGET)
    text = _blank_spans(text, _URL)
    lines = text.split("\n")

    block_of: dict[int, tuple] = {}
    previous: tuple = ()
    for first, last in _blocks(cited_lines):
        cites = tuple(sorted({(m.group(0), ()) for n in range(first, last + 1)
                              for m in _RESULTS_PATH.finditer(cited_lines[n])}))
        is_table = cited_lines[first].lstrip().startswith("|")
        scope = cites or (previous if is_table else ())
        for number in range(first, last + 1):
            block_of[number] = scope
        previous = cites or (previous if is_table else ())

    for pattern in DOC_ALLOW_PATTERNS:
        text = check._blank_matches(text, pattern)
    allowed = text
    for pattern in check.ALLOW_PATTERNS:
        allowed = check._blank_matches(allowed, pattern)
    found = check.hand_numbers(text, latex=False)
    for match in _LONG_INT.finditer(allowed):
        value = int(match.group(0))
        if 1900 <= value <= 2099 or value & (value - 1) == 0:
            continue
        found.append((allowed.count("\n", 0, match.start()) + 1, match.group(0)))
    for match in _FINAL_RATIO.finditer(allowed):
        found.append((allowed.count("\n", 0, match.start()) + 1, match.group(0)))

    for line, figure in sorted(set(found)):
        index = line - 1
        if exempt[index] or not lines[index].strip():
            continue
        sources = explicit[index]
        kind = "rows" if sources and any(f for _, f in sources) else "files"
        if not sources:
            sources = block_of.get(index, ())
        if not sources:
            report.checked["uncited"] += 1
            report.findings.append(Finding(relative, line, figure,
                                           "any cited source (the block names no results file)"))
            continue
        parsed = parse_figure(figure)
        report.checked[kind] += 1
        lookup = pool.index(tuple(sources))
        if parsed and all(lookup.has(v, d, s) for v, d, s in parsed):
            continue
        scope = "; ".join(t + (" " + " ".join(f"{k}={v}" for k, v in f) if f else "")
                          for t, f in sources)
        if not any(pool.files(t) for t, _ in sources):
            scope += " (no such results file)"
        report.findings.append(Finding(relative, line, figure, scope))


def check_latex(path: Path, check, report: Report, relative: str) -> None:
    for line, figure in check.hand_numbers(path.read_text(encoding="utf-8"), latex=True):
        report.findings.append(Finding(relative, line, figure,
                                       "numbers.tex (a slide number must be a macro)"))


def doc_files(root: Path) -> list[Path]:
    seen: dict[Path, None] = {}
    for pattern in DOC_GLOBS:
        for path in sorted(root.glob(pattern)):
            if path.is_file():
                seen[path] = None
    return list(seen)


def run(root: Path = ROOT) -> Report:
    root = Path(root)
    check = load_check(root)
    pool = Pool(root)
    report = Report()
    for path in doc_files(root):
        relative = path.relative_to(root).as_posix()
        if relative in EXCLUDED:
            report.exempt_files[relative] = EXCLUDED[relative]
            continue
        check_markdown(path, pool, check, report, relative)
    for relative in LATEX_FILES:
        if (root / relative).is_file():
            check_latex(root / relative, check, report, relative)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    report = run(args.root)
    for finding in report.findings:
        print(finding)
    counts = report.checked
    print(f"check_doc_numbers: {sum(counts.values())} figure(s) checked "
          f"({counts['rows']} against cited rows, {counts['files']} against cited files, "
          f"{counts['uncited']} citing no source); {len(report.findings)} flagged")
    if args.verbose:
        for relative, reason in report.exempt_files.items():
            print(f"  not checked: {relative} ({reason})")
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
