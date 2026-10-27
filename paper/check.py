"""Structural lint for the IEEE paper: the local stand-in for a LaTeX compile.

No LaTeX toolchain can be installed on this machine (MASTER_PLAN L6), so the
students compile on Overleaf and this script is the local gate. It follows
``main.tex`` through every ``\\input``/``\\include`` and checks:

- every ``\\cite`` key exists in ``refs.bib``, and the bibliography itself
  parses (no duplicate key, every entry has a title);
- every ``\\input``/``\\include``/``\\includegraphics`` target exists inside
  the paper directory, or matches a declared generator in ``GENERATED``
  (``numbers.tex``, ``tables/*.tex``, ``figures/*.pdf``);
- braces balance and ``\\begin``/``\\end`` pair up, per file;
- no bare ``_`` outside math in hand-written files, which Overleaf would
  reject;
- the hand-number lint: in ``sections/*.tex`` no decimal, percentage,
  percentage-point, multiplier, ratio (52/56) or thousands-grouped figure
  may appear outside the allow-list (versions, section/decision numbers,
  dimensions). Every number quoted in
  prose must be a ``numbers.tex`` macro written from ``results/``.

- every number macro the prose uses (a control word in one of the
  ``numbers.py`` families, ``NUMBER_MACRO``) is defined in ``numbers.tex``,
  so a misspelt name cannot reach Overleaf as an undefined command.

Modes. The default is for the skeleton: a generated file that does not exist
yet, or a ``\\ref`` whose label may live in one, is a warning. ``--strict``
(after ``python reproduce.py paper``) makes both errors, and an undefined
number macro an error. ``--final`` adds that no ``\\todo`` may remain and
that no macro the prose uses prints ``\\missingnumber`` (a result that
was not generated). Exit status is 1 when any error is found.

Stdlib only, so it runs under system python3 as well as the venv.
"""

from __future__ import annotations

import argparse
import fnmatch
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

PAPER_DIR = Path(__file__).resolve().parent
ROOT_FILE = "main.tex"

# Files a generator writes, keyed by pattern relative to the paper
# directory, with the generator that owns them (MASTER_PLAN section 5; WP13).
GENERATED: tuple[tuple[str, str], ...] = (
    ("numbers.tex", "numbers.py"),
    ("tables/*.tex", "tables/make_tables.py"),
    ("figures/*.pdf", "figures/make_figures.py"),
)

# Only hand-written prose is number-linted. Generated tables carry numbers
# by design.
LINTED_GLOB = "sections/*.tex"

GRAPHICS_SUFFIXES = (".pdf", ".png", ".jpg", ".jpeg", ".eps")

# Commands whose (first) argument is a key, path, dimension or count rather
# than prose. Their arguments are blanked before the number and underscore
# lints, so `\label{fig:f2_depth}` or `\hspace{0.5em}` is never flagged.
_KEY_ARGUMENT = re.compile(
    r"\\(?:label|ref|eqref|autoref|cref|Cref|pageref|nameref|cite[a-zA-Z]*|nocite"
    r"|input|include|includegraphics|InputIfFileExists|url|href|bibliography"
    r"|bibliographystyle|begin|end|hspace|vspace|setlength|addtolength"
    r"|resizebox|scalebox|rule|multicolumn|multirow|IEEEauthorrefmark)\*?"
    r"\s*(?:\[[^\]]*\]\s*)*\{[^{}]*\}"
)

# Spans that look like figures but are not results. Each match is blanked
# before the number patterns run. Extend this only under review (WP14): a
# pattern broad enough to admit a measured figure defeats the lint.
ALLOW_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Lengths: 0.9\columnwidth, 3.5in, 2pt. A unit written as letters must
    # touch the number, or "a gain of 4.10 in mean hit ratio" would pass.
    re.compile(
        r"\d*\.?\d+(?:\s*\\(?:columnwidth|linewidth|textwidth|textheight"
        r"|paperwidth|hsize|baselineskip)|(?:pt|em|ex|in|cm|mm|bp|sp)\b)"
    ),
    # Software, kernel and interface versions: Linux 6.14, Python 3.10.
    re.compile(
        r"(?i:\b(?:linux|kernel|python|pytorch|torch|cuda|numpy|pandas|scikit-learn"
        r"|sklearn|scipy|matplotlib|zstandard|ieeetran|man-pages|glibc|version"
        r"|sata|pcie|nvme|usb))\s*~?\s*\d+(?:\.\d+)+"
    ),
    re.compile(r"\bv\d+(?:\.\d+)+"),
    # Any dotted triple is a version (2.4.13), never a measurement.
    re.compile(r"\b\d+\.\d+\.\d+(?:\.\d+)*\b"),
    # Typed cross-references: Section 2.3, \S 2.1, Fig. 4, Table 2.
    re.compile(
        r"(?:\bSec(?:tion)?s?\.?|\\S|\u00a7|\bFig(?:ure)?s?\.?|\bTables?|\bEqs?\.?"
        r"|\bAlg(?:orithm)?\.?|\bAppendix|\bStep)\s*~?\s*\d+(?:\.\d+)*"
    ),
    # Plan and decision identifiers: D10.5, D14.1, WP7, E3, L4, C13, M16.
    re.compile(r"(?<![A-Za-z])(?:WP|D|W|L|C|E|M|P|T|F)\d+(?:\.\d+)*\b"),
)

_SEP = r"(?:\s|~|\\[,;: ])*"
_NUMBER = r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)"
# The figures a hand-typed result takes: 41.3, .89, 41\%, +4.1 pp, 640x,
# 52/56, 2,756,101 or 2\,756\,101. A bare integer ("four corpora", "56
# traces") is not flagged; the WP14 review covers those.
_HAND_NUMBER_LATEX = re.compile(
    r"(?<![\w.\\])" + _NUMBER + _SEP + r"(?:\\%|pp\b)"
    r"|(?<![\w.\\])" + _NUMBER + _SEP + r"(?:x|\\times|\u00d7)(?![A-Za-z])"
    r"|(?<![\w./\\])\d+\s*/\s*\d+(?![\w./])"
    r"|(?<![\w.\\])\d{1,3}(?:(?:,|\{,\}|\\,)\d{3})+(?!\d)"
    r"|(?<![\w.\\])[-+]?(?:\d+)?\.\d+"
)
# The same for Markdown or plain text, where % is a literal (for WP15's docs).
_HAND_NUMBER_TEXT = re.compile(
    r"(?<![\w.])" + _NUMBER + r"\s*(?:%|pp\b)"
    r"|(?<![\w.])" + _NUMBER + r"\s*(?:x|\u00d7)(?![A-Za-z])"
    r"|(?<![\w./])\d+\s*/\s*\d+(?![\w./])"
    r"|(?<![\w.])\d{1,3}(?:,\d{3})+(?!\d)"
    r"|(?<![\w.])[-+]?(?:\d+)?\.\d+"
)

_INPUT = re.compile(r"\\(input|include)\s*\{([^{}]+)\}")
_GRAPHICS = re.compile(r"\\includegraphics\s*(?:\[[^\]]*\])?\s*\{([^{}]+)\}")
_BIBLIOGRAPHY = re.compile(r"\\bibliography\s*\{([^{}]+)\}")
_CITE = re.compile(r"\\(?:no)?cite[a-zA-Z]*\*?\s*(?:\[[^\]]*\]\s*){0,2}\{([^{}]*)\}")
_LABEL = re.compile(r"\\label\s*\{([^{}]+)\}")
_REF = re.compile(r"\\(?:ref|eqref|autoref|cref|Cref|pageref|nameref)\s*\{([^{}]+)\}")
_ENVIRONMENT = re.compile(r"\\(begin|end)\s*\{([^{}]+)\}")
_TODO = re.compile(r"\\todo\s*\{")
# Control words of the numbers.py families (paper/numbers.py docstring).
NUMBER_MACRO = re.compile(
    r"\\((?:disk|dec|web|live|realio|ld|lstm|corpus|device|host|ci)[A-Z][A-Za-z]*)")
_NEWCOMMAND = re.compile(r"^\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}$", re.M)
_MATH_ENVIRONMENTS = (
    "equation", "equation*", "align", "align*", "gather", "gather*",
    "multline", "multline*", "eqnarray", "eqnarray*", "displaymath", "math",
)


@dataclass(frozen=True)
class Finding:
    level: str  # "error" or "warning"
    path: str  # relative to the paper directory
    line: int  # 1-based; 0 when the finding is about the whole file
    message: str

    def __str__(self) -> str:
        where = f"{self.path}:{self.line}" if self.line else self.path
        return f"{self.level.upper()} {where}: {self.message}"


@dataclass
class BibEntry:
    kind: str
    key: str
    line: int
    fields: dict[str, str] = field(default_factory=dict)


@dataclass
class Report:
    findings: list[Finding] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    cited: set[str] = field(default_factory=set)
    bib_keys: set[str] = field(default_factory=set)

    @property
    def errors(self) -> list[Finding]:
        return [item for item in self.findings if item.level == "error"]

    @property
    def warnings(self) -> list[Finding]:
        return [item for item in self.findings if item.level == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def add(self, level: str, path: str, line: int, message: str) -> None:
        self.findings.append(Finding(level, path, line, message))


# ---------------------------------------------------------------------------
# Text helpers. Each keeps the text's length and newlines, so a match offset
# still maps to the right line after blanking.
# ---------------------------------------------------------------------------

def _line_of(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _blank(text: str, start: int, end: int) -> str:
    span = "".join("\n" if ch == "\n" else " " for ch in text[start:end])
    return text[:start] + span + text[end:]


def _blank_matches(text: str, pattern: re.Pattern[str]) -> str:
    pieces = []
    last = 0
    for match in pattern.finditer(text):
        pieces.append(text[last:match.start()])
        pieces.append("".join("\n" if ch == "\n" else " " for ch in match.group(0)))
        last = match.end()
    pieces.append(text[last:])
    return "".join(pieces)


def strip_comments(text: str) -> str:
    """Blank every LaTeX comment: an unescaped % to the end of its line."""
    out = []
    for line in text.split("\n"):
        cut = None
        index = 0
        while index < len(line):
            char = line[index]
            if char == "\\":
                index += 2  # \% \\ and any other control symbol
                continue
            if char == "%":
                cut = index
                break
            index += 1
        out.append(line if cut is None else line[:cut] + " " * (len(line) - cut))
    return "\n".join(out)


def _blank_math(text: str) -> str:
    for name in _MATH_ENVIRONMENTS:
        escaped = re.escape(name)
        text = _blank_matches(
            text, re.compile(r"\\begin\{" + escaped + r"\}.*?\\end\{" + escaped + r"\}", re.S)
        )
    text = _blank_matches(text, re.compile(r"\$\$.*?\$\$", re.S))
    text = _blank_matches(text, re.compile(r"(?<!\\)\$.*?(?<!\\)\$", re.S))
    text = _blank_matches(text, re.compile(r"\\\(.*?\\\)", re.S))
    return _blank_matches(text, re.compile(r"\\\[.*?\\\]", re.S))


def hand_numbers(text: str, *, latex: bool = True) -> list[tuple[int, str]]:
    """Return (line, figure) for each hand-typed result figure in ``text``.

    With ``latex=True`` comments are ignored and ``\\%`` is the percent
    sign; with ``latex=False`` (Markdown, plain text) ``%`` is literal.
    Spans matching ``ALLOW_PATTERNS`` are never reported.
    """
    if latex:
        text = _blank_matches(strip_comments(text), _KEY_ARGUMENT)
    for pattern in ALLOW_PATTERNS:
        text = _blank_matches(text, pattern)
    pattern = _HAND_NUMBER_LATEX if latex else _HAND_NUMBER_TEXT
    return [(_line_of(text, m.start()), m.group(0).strip()) for m in pattern.finditer(text)]


def bare_underscores(text: str) -> list[int]:
    """Lines with an unescaped ``_`` outside math and key arguments."""
    text = _blank_matches(strip_comments(text), _KEY_ARGUMENT)
    text = _blank_math(text)
    return sorted({_line_of(text, m.start()) for m in re.finditer(r"(?<!\\)_", text)})


def brace_problems(text: str) -> list[tuple[int, str]]:
    """Unmatched ``{``/``}`` outside comments; ``\\{`` and ``\\}`` are literal."""
    text = strip_comments(text)
    problems = []
    opened: list[int] = []
    index = 0
    while index < len(text):
        char = text[index]
        if char == "\\":
            index += 2
            continue
        if char == "{":
            opened.append(index)
        elif char == "}":
            if opened:
                opened.pop()
            else:
                problems.append((_line_of(text, index), "unmatched '}'"))
        index += 1
    for offset in opened:
        problems.append((_line_of(text, offset), "'{' is never closed"))
    return sorted(problems)


def environment_problems(text: str) -> list[tuple[int, str]]:
    """``\\begin``/``\\end`` pairs that do not nest within one file."""
    text = strip_comments(text)
    problems = []
    stack: list[tuple[str, int]] = []
    for match in _ENVIRONMENT.finditer(text):
        kind, name = match.group(1), match.group(2).strip()
        line = _line_of(text, match.start())
        if kind == "begin":
            stack.append((name, line))
        elif not stack:
            problems.append((line, f"\\end{{{name}}} without \\begin"))
        elif stack[-1][0] != name:
            problems.append(
                (line, f"\\end{{{name}}} closes \\begin{{{stack[-1][0]}}} from line {stack[-1][1]}")
            )
            stack.pop()
        else:
            stack.pop()
    for name, line in stack:
        problems.append((line, f"\\begin{{{name}}} is never closed"))
    return problems


# ---------------------------------------------------------------------------
# Bibliography
# ---------------------------------------------------------------------------

def _balanced_end(text: str, start: int, opener: str, closer: str) -> int | None:
    """Index just past the closer matching ``text[start] == opener``."""
    depth = 0
    index = start
    while index < len(text):
        char = text[index]
        if char == "\\":
            index += 2  # an escaped brace never opens or closes
            continue
        if char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return index + 1
        index += 1
    return None


def _parse_fields(body: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    index = 0
    name_pattern = re.compile(r"\s*,?\s*([A-Za-z][\w-]*)\s*=\s*")
    while index < len(body):
        match = name_pattern.match(body, index)
        if not match:
            break
        name = match.group(1).lower()
        index = match.end()
        if index < len(body) and body[index] == "{":
            end = _balanced_end(body, index, "{", "}")
            if end is None:
                break
            fields[name] = body[index + 1:end - 1]
        elif index < len(body) and body[index] == '"':
            end = body.find('"', index + 1)
            if end < 0:
                break
            fields[name] = body[index + 1:end]
            end += 1
        else:
            bare = re.match(r"[^,}\s]+", body[index:])
            if not bare:
                break
            fields[name] = bare.group(0)
            end = index + bare.end()
        index = end
    return fields


def parse_bib(text: str) -> tuple[list[BibEntry], list[tuple[int, str]]]:
    """Parse BibTeX entries; return (entries, [(line, problem), ...])."""
    entries: list[BibEntry] = []
    problems: list[tuple[int, str]] = []
    entry_start = re.compile(r"@\s*([A-Za-z]+)\s*([{(])")
    position = 0
    while True:
        # Scan from the end of the previous entry, so an @ inside a field
        # (a URL, an e-mail address) is never taken for a new entry.
        match = entry_start.search(text, position)
        if not match:
            break
        kind = match.group(1).lower()
        opener = match.group(2)
        closer = "}" if opener == "{" else ")"
        line = _line_of(text, match.start())
        end = _balanced_end(text, match.end() - 1, opener, closer)
        if end is None:
            problems.append((line, f"@{kind} entry is never closed (unbalanced braces)"))
            break
        position = end
        if kind in ("comment", "string", "preamble"):
            continue
        body = text[match.end():end - 1]
        key_match = re.match(r"\s*([^,\s{}()]+)\s*,", body)
        if not key_match:
            problems.append((line, f"@{kind} entry has no citation key"))
            continue
        entries.append(BibEntry(kind, key_match.group(1), line, _parse_fields(body[key_match.end():])))
    return entries, problems


# ---------------------------------------------------------------------------
# The checks
# ---------------------------------------------------------------------------

def _generator_for(relative: str) -> str | None:
    for pattern, generator in GENERATED:
        if fnmatch.fnmatch(relative, pattern):
            return generator
    return None


def _inside(paper_dir: Path, relative: str) -> bool:
    path = PurePosixPath(relative)
    return not path.is_absolute() and ".." not in path.parts


def _check_target(report: Report, paper_dir: Path, source: str, line: int,
                  relative: str, what: str, strict: bool) -> bool:
    """Report on one referenced file; True when it exists on disk."""
    if not _inside(paper_dir, relative):
        report.add("error", source, line,
                   f"{what} '{relative}' is outside the paper directory (Overleaf gets paper/ only)")
        return False
    if (paper_dir / relative).is_file():
        return True
    generator = _generator_for(relative)
    if generator is None:
        report.add("error", source, line, f"{what} '{relative}' does not exist")
        return False
    note = f"{what} '{relative}' is generated by paper/{generator} and has not been generated"
    if not (PAPER_DIR / generator).is_file():
        note += f" (paper/{generator} does not exist yet)"
    if strict:
        report.add("error", source, line, note + "; run `python reproduce.py paper`")
    else:
        report.add("warning", source, line, note)
    return False


def _graphics_target(paper_dir: Path, target: str) -> str:
    if PurePosixPath(target).suffix:
        return target
    for suffix in GRAPHICS_SUFFIXES:
        if (paper_dir / (target + suffix)).is_file():
            return target + suffix
    return target + ".pdf"


#: The claims ledger (paper/claims.py): beside a scratch paper directory, or
#: in results/claims/ for the committed paper.
CLAIMS_LEDGER = ("claims", "paper_claims.csv")


def claims_ledger(paper_dir: Path) -> Path | None:
    """The ledger that belongs to ``paper_dir``: ``paper_dir/claims/`` (a fixture
    or scratch build), else ``results/claims/`` for the repository's paper."""
    local = paper_dir.joinpath(*CLAIMS_LEDGER)
    if local.is_file():
        return local
    if paper_dir.resolve() == PAPER_DIR.resolve():
        return PAPER_DIR.parent / "results" / CLAIMS_LEDGER[0] / CLAIMS_LEDGER[1]
    return None


def read_ledger(path: Path) -> dict[str, str]:
    """``{macro: printed}`` of the ledger's macro rows."""
    import csv
    with open(path, newline="", encoding="utf-8") as handle:
        return {row["claim_id"]: row["printed"] for row in csv.DictReader(handle)
                if not row["claim_id"].startswith("tables/") and row["kind"] != "missing"}


def run_checks(paper_dir: Path | str = PAPER_DIR, *, strict: bool = False,
               final: bool = False) -> Report:
    """Lint the paper rooted at ``paper_dir/main.tex``; see the module doc."""
    paper_dir = Path(paper_dir)
    strict = strict or final
    report = Report()
    ledger_path = claims_ledger(paper_dir)
    ledger = None
    if strict and ledger_path is not None:
        if ledger_path.is_file():
            ledger = read_ledger(ledger_path)
        else:
            report.add("error", ROOT_FILE, 0, f"no claims ledger at {ledger_path}; "
                                              "run `python reproduce.py paper`")
    root = paper_dir / ROOT_FILE
    if not root.is_file():
        report.add("error", ROOT_FILE, 0, "missing")
        return report

    defined: dict[str, str] | None = None
    fixture_numbers = False
    numbers_path = paper_dir / "numbers.tex"
    if numbers_path.is_file():
        numbers_text = numbers_path.read_text(encoding="utf-8")
        defined = dict(_NEWCOMMAND.findall(numbers_text))
        # A fixture build (reproduce.py paper --fixtures) lacks most result
        # rows by design, so an undefined macro there is only a warning.
        fixture_numbers = "\\paperFixtureData}{true}" in numbers_text
    labels: dict[str, tuple[str, int]] = {}
    refs: list[tuple[str, int, str]] = []
    cites: list[tuple[str, int, str]] = []
    bib_files: list[tuple[str, int, str]] = []
    queue = [ROOT_FILE]
    seen: set[str] = set()
    while queue:
        relative = queue.pop(0)
        if relative in seen:
            continue
        seen.add(relative)
        report.files.append(relative)
        raw = (paper_dir / relative).read_text(encoding="utf-8")
        text = strip_comments(raw)
        generated = _generator_for(relative) is not None

        for line, message in brace_problems(raw):
            report.add("error", relative, line, message)
        # "$x$$^{*}$" opens display math: two math shifts must never touch
        # (generated cells append marks to numbers that hold their own $).
        for match in re.finditer(r"(?<!\\)\$\$", text):
            report.add("error", relative, _line_of(text, match.start()),
                       "'$$' (adjacent math shifts open display math)")
        for line, message in environment_problems(raw):
            report.add("error", relative, line, message)

        for match in _INPUT.finditer(text):
            target = match.group(2).strip()
            if not PurePosixPath(target).suffix:
                target += ".tex"
            line = _line_of(text, match.start())
            if _check_target(report, paper_dir, relative, line, target, "\\" + match.group(1), strict):
                queue.append(target)
        for match in _GRAPHICS.finditer(text):
            target = _graphics_target(paper_dir, match.group(1).strip())
            _check_target(report, paper_dir, relative, _line_of(text, match.start()),
                          target, "\\includegraphics", strict)
        for match in _BIBLIOGRAPHY.finditer(text):
            for name in match.group(1).split(","):
                name = name.strip()
                if name:
                    bib_files.append((relative, _line_of(text, match.start()),
                                      name if name.endswith(".bib") else name + ".bib"))
        for match in _CITE.finditer(text):
            line = _line_of(text, match.start())
            for key in match.group(1).split(","):
                key = key.strip()
                if key and key != "*":
                    cites.append((relative, line, key))
        for match in _LABEL.finditer(text):
            name = match.group(1).strip()
            line = _line_of(text, match.start())
            if name in labels:
                first = labels[name]
                report.add("error", relative, line,
                           f"duplicate \\label{{{name}}} (first at {first[0]}:{first[1]})")
            else:
                labels[name] = (relative, line)
        for match in _REF.finditer(text):
            refs.append((relative, _line_of(text, match.start()), match.group(1).strip()))

        if generated:
            if strict and fnmatch.fnmatch(relative, "tables/*.tex"):
                # A generated caption is written by tables/make_tables.py; a
                # figure typed into its caption string is not traced to
                # results/ (the claims ledger traces the cells). Reported,
                # not fatal: see paper/README.md, "The claims ledger".
                for match in re.finditer(r"\\caption\s*\{", raw):
                    end = _balanced_end(raw, match.end() - 1, "{", "}")
                    caption = raw[match.start():end or len(raw)]
                    for _, figure in hand_numbers(caption):
                        report.add("warning", relative, _line_of(raw, match.start()),
                                   f"hand-typed number '{figure}' in a generated caption; "
                                   "print it from results/ in tables/make_tables.py")
            continue
        for line in bare_underscores(raw):
            report.add("error", relative, line, "bare '_' outside math; write '\\_'")
        if fnmatch.fnmatch(relative, LINTED_GLOB):
            for line, figure in hand_numbers(raw):
                report.add("error", relative, line,
                           f"hand-typed number '{figure}'; quote it through a numbers.tex macro")
            if strict and defined is not None:
                for match in NUMBER_MACRO.finditer(text):
                    name = match.group(1)
                    line = _line_of(text, match.start())
                    if name not in defined:
                        report.add("warning" if fixture_numbers else "error", relative, line,
                                   f"\\{name} is not defined in numbers.tex")
                    elif (ledger is not None and "\\missingnumber" not in defined[name]
                          and ledger.get(name) != defined[name]):
                        report.add("error", relative, line,
                                   f"\\{name} is not traced in the claims ledger "
                                   f"({ledger_path.name}): run `python reproduce.py paper`"
                                   if name not in ledger else
                                   f"\\{name}: the claims ledger prints {ledger[name]!r}, "
                                   f"numbers.tex {defined[name]!r} (stale ledger)")
                    elif final and "\\missingnumber" in defined[name]:
                        report.add("error", relative, line,
                                   f"\\{name} prints \\missingnumber (its result is missing)")
        if final:
            for match in _TODO.finditer(text):
                report.add("error", relative, _line_of(text, match.start()), "\\todo remains")

    for path in sorted((paper_dir / "sections").glob("*.tex")):
        relative = path.relative_to(paper_dir).as_posix()
        if relative not in seen:
            report.add("warning", relative, 0, "not reached from main.tex by \\input")

    # Bibliography: parse, then resolve every cite.
    if not bib_files:
        report.add("error", ROOT_FILE, 0, "no \\bibliography command")
    keys: dict[str, BibEntry] = {}
    for source, line, name in bib_files:
        if not _check_target(report, paper_dir, source, line, name, "\\bibliography", strict):
            continue
        entries, problems = parse_bib((paper_dir / name).read_text(encoding="utf-8"))
        for problem_line, message in problems:
            report.add("error", name, problem_line, message)
        for entry in entries:
            folded = entry.key.lower()  # BibTeX keys are case-insensitive
            if folded in keys:
                report.add("error", name, entry.line,
                           f"duplicate key '{entry.key}' (first at line {keys[folded].line})")
                continue
            keys[folded] = entry
            if not entry.fields.get("title"):
                report.add("error", name, entry.line, f"'{entry.key}' has no title")
            # An undated @misc (a dataset listing) is normal; a dated venue
            # without a year is a transcription slip.
            if not entry.fields.get("year") and entry.kind != "misc":
                report.add("warning", name, entry.line, f"'{entry.key}' has no year")
    report.bib_keys = {entry.key for entry in keys.values()}
    for source, line, key in cites:
        if key.lower() not in keys:
            report.add("error", source, line, f"\\cite key '{key}' is not in the bibliography")
        report.cited.add(key)

    for source, line, name in refs:
        if name not in labels:
            report.add("error" if strict else "warning", source, line,
                       f"\\ref{{{name}}} has no \\label")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--paper-dir", type=Path, default=PAPER_DIR,
                        help="directory holding main.tex (default: this script's directory)")
    parser.add_argument("--strict", action="store_true",
                        help="generated files and \\ref labels must exist (after `reproduce.py paper`)")
    parser.add_argument("--final", action="store_true",
                        help="--strict, and no \\todo may remain (the submission gate)")
    parser.add_argument("--quiet", action="store_true", help="print errors only")
    args = parser.parse_args(argv)

    report = run_checks(args.paper_dir, strict=args.strict, final=args.final)
    for finding in report.findings:
        if finding.level == "error" or not args.quiet:
            print(finding)
    uncited = len({key.lower() for key in report.bib_keys} - {key.lower() for key in report.cited})
    print(
        f"check.py: {len(report.errors)} error(s), {len(report.warnings)} warning(s); "
        f"{len(report.files)} file(s), {len(report.cited)} distinct cite key(s), "
        f"{len(report.bib_keys)} bibliography entries ({uncited} not cited)"
    )
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
