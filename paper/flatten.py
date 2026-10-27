"""Write the paper as single files: ``main_flat.tex`` and ``paper_full.txt``.

    python3 paper/flatten.py [--paper-dir DIR] [--check]

Both are generated from the committed paper sources (``main.tex``,
``sections/*.tex``, ``tables/*.tex``, ``numbers.tex``, ``refs.bib``) and are
never edited by hand; ``reproduce.py paper`` runs this after the numbers,
tables and figures, and ``tests/test_paper_flatten.py`` pins the committed
files to a fresh regeneration.

``main_flat.tex``
    ``main.tex`` with every ``\\input``/``\\include`` inlined recursively
    (each marked by ``% >>> begin <file>`` / ``% <<< end <file>`` comments)
    and ``\\IfFileExists`` resolved. ``numbers.tex`` defines about 21,600
    macros, so only the macros the text and tables use (and the macros their
    bodies use) are kept, which keeps TeX's memory small on Overleaf. The
    bibliography is a ``thebibliography`` block formatted here from
    ``refs.bib`` for the cited keys only, numbered in order of first citation
    as IEEEtran's unsorted style numbers them, so neither BibTeX nor
    ``refs.bib`` is needed: the file compiles alone with pdfLaTeX, given the
    ``figures/`` directory beside it. The entries follow IEEE reference
    style; they are close to, not byte-identical with, IEEEtran.bst's output.

``paper_full.txt``
    A plain-text rendering in reading order: title, authors, abstract, every
    heading numbered as IEEEtran numbers it (I, II, ... ; IV-A; Appendix A),
    prose with every macro expanded to its printed value, citations as
    ``[n]`` with the reference list at the place of ``\\bibliography``,
    tables as aligned text with their captions, figures as
    ``[Figure n: caption -- see paper/figures/<file>.pdf]``, and math written
    out in Unicode (``kappa = t_pos/t_xfer``). The converter knows the LaTeX
    this paper uses and degrades gracefully on anything else: an unknown
    command is dropped and its arguments are kept as text, and an undefined
    number-family macro prints ``??`` (as ``\\missingnumber`` does).

Stdlib only, so it runs under system python3 as well as the venv. It loads
``check.py`` by path for the comment stripper and the BibTeX parser, so the
paper has one parser of each.
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import sys
import textwrap
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

PAPER_DIR = Path(__file__).resolve().parent
ROOT_FILE = "main.tex"
NUMBERS_FILE = "numbers.tex"
FLAT_TEX = "main_flat.tex"
FULL_TEXT = "paper_full.txt"
OUTPUTS = (FLAT_TEX, FULL_TEXT)

WIDTH = 78  # plain-text column for prose; tables keep their natural width
CHARS_PER_INCH = 14  # p{} column widths in the plain-text tables
COLUMN_IN, TEXT_IN = 3.5, 7.16  # IEEEtran column and text widths
MAX_AUTHORS = 10  # IEEEtran.bst: more than this prints the first and "et al."
TABLE_SEP = "  "

NBSP = "\u00a0"  # a tie or thin space: never broken by the wrapper
BREAK = "\u2028"  # a forced line break (\\) inside a paragraph


def _check():
    """``check.py``, loaded by path; ``paper/`` is not a package."""
    name = "paper_check"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, PAPER_DIR / "check.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module  # dataclasses look their module up here
        try:
            spec.loader.exec_module(module)
        except BaseException:
            del sys.modules[name]
            raise
    return sys.modules[name]


# ---------------------------------------------------------------------------
# Scanning helpers (on comment-free text unless stated)
# ---------------------------------------------------------------------------

_CONTROL_WORD = re.compile(r"\\([A-Za-z@]+)")


def group_end(text: str, start: int) -> int:
    """Index just past the ``}`` matching ``text[start] == '{'``."""
    depth = 0
    index = start
    while index < len(text):
        char = text[index]
        if char == "\\":
            index += 2
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index + 1
        index += 1
    raise ValueError(f"unbalanced '{{' at offset {start}: {text[start:start + 60]!r}")


def _skip_space(text: str, index: int) -> int:
    """Skip spaces and at most one newline (a blank line is a paragraph)."""
    while index < len(text) and text[index] in " \t":
        index += 1
    if index < len(text) and text[index] == "\n":
        after = index + 1
        while after < len(text) and text[after] in " \t":
            after += 1
        if after >= len(text) or text[after] != "\n":
            index = after
    return index


def read_arg(text: str, index: int) -> tuple[str | None, int]:
    """One mandatory argument: a brace group, a control sequence or a character."""
    index = _skip_space(text, index)
    if index >= len(text):
        return None, index
    if text[index] == "{":
        end = group_end(text, index)
        return text[index + 1:end - 1], end
    if text[index] == "\\":
        match = _CONTROL_WORD.match(text, index)
        end = match.end() if match else index + 2
        return text[index:end], end
    if text[index] in "}":
        return None, index
    return text[index], index + 1


def read_opt(text: str, index: int) -> tuple[str | None, int]:
    """An optional ``[...]`` argument (brace-aware), or None."""
    start = _skip_space(text, index)
    if start >= len(text) or text[start] != "[":
        return None, index
    depth = 0
    position = start + 1
    while position < len(text):
        char = text[position]
        if char == "\\":
            position += 2
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
        elif char == "]" and depth == 0:
            return text[start + 1:position], position + 1
        position += 1
    return None, index


def environment_end(text: str, start: int, name: str) -> tuple[int, int]:
    """(start of ``\\end{name}``, index past it) for the environment whose
    body starts at ``start``; same-name environments may nest."""
    pattern = re.compile(r"\\(begin|end)\s*\{" + re.escape(name) + r"\}")
    depth = 1
    for match in pattern.finditer(text, start):
        depth += 1 if match.group(1) == "begin" else -1
        if depth == 0:
            return match.start(), match.end()
    raise ValueError(f"\\begin{{{name}}} is never closed")


def _split_top(text: str, separator: re.Pattern[str]) -> list[str]:
    """Split on ``separator`` matches at brace depth 0 (escapes skipped)."""
    parts, depth, last, index = [], 0, 0, 0
    while index < len(text):
        char = text[index]
        if char == "\\":
            match = separator.match(text, index) if depth == 0 else None
            if match and match.end() > index:
                parts.append(text[last:index])
                last = index = match.end()
                continue
            index += 2
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
        elif depth == 0:
            match = separator.match(text, index)
            if match and match.end() > index:
                parts.append(text[last:index])
                last = index = match.end()
                continue
        index += 1
    parts.append(text[last:])
    return parts


def roman(number: int) -> str:
    out = []
    for value, digits in ((1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"),
                          (90, "XC"), (50, "L"), (40, "XL"), (10, "X"), (9, "IX"),
                          (5, "V"), (4, "IV"), (1, "I")):
        while number >= value:
            out.append(digits)
            number -= value
    return "".join(out)


def letter(number: int) -> str:
    out = ""
    while number > 0:
        number, rest = divmod(number - 1, 26)
        out = chr(ord("A") + rest) + out
    return out


# ---------------------------------------------------------------------------
# Step 1: inline every \input, resolve \IfFileExists
# ---------------------------------------------------------------------------

_INCLUDE = re.compile(r"\\(input|include)\s*\{([^{}]+)\}|\\IfFileExists\s*\{([^{}]+)\}")


def _tex_target(name: str) -> str:
    name = name.strip()
    return name if PurePosixPath(name).suffix else name + ".tex"


def inline_inputs(text: str, paper_dir: Path, stack: tuple[str, ...] = ()) -> str:
    """``text`` with every ``\\input``/``\\include`` outside comments replaced
    by the file's (recursively inlined) contents."""
    masked = _check().strip_comments(text)  # same length; comments blanked
    out: list[str] = []
    position = 0
    while True:
        match = _INCLUDE.search(masked, position)
        if not match:
            break
        out.append(text[position:match.start()])
        if match.group(3) is not None:  # \IfFileExists{file}{then}{else}
            index = _skip_space(masked, match.end())
            then_end = group_end(masked, index)
            else_start = _skip_space(masked, then_end)
            else_end = group_end(masked, else_start)
            exists = (paper_dir / match.group(3).strip()).is_file()
            branch = text[index + 1:then_end - 1] if exists else text[else_start + 1:else_end - 1]
            out.append(inline_inputs(branch, paper_dir, stack))
            position = else_end
            continue
        target = _tex_target(match.group(2))
        if target in stack:
            raise ValueError(f"\\input cycle: {' -> '.join(stack + (target,))}")
        path = paper_dir / target
        if path.is_file():
            body = inline_inputs(path.read_text(encoding="utf-8"), paper_dir, stack + (target,))
        else:  # the generators' convention: a visible placeholder, never a failure
            body = f"\\textbf{{[MISSING: {target}]}}"
        # The markers are comment lines, which TeX reads as nothing; they add a
        # line end only where the \input had none, so an \input on its own
        # line inside a paragraph never becomes a blank line (a \par).
        last = next((piece for piece in reversed(out) if piece), "")
        before = "" if not last or last.endswith("\n") else "\n"
        after = "" if text.startswith("\n", match.end()) else "\n"
        out.append(f"{before}% >>> begin {target}\n{body.rstrip()}\n% <<< end {target}{after}")
        position = match.end()
    out.append(text[position:])
    return "".join(out)


# ---------------------------------------------------------------------------
# Step 2: keep only the numbers.tex macros the paper uses
# ---------------------------------------------------------------------------

_NEWCOMMAND_LINE = re.compile(r"^\\newcommand\{\\([A-Za-z]+)\}")


def _numbers_region(flat: str) -> tuple[int, int] | None:
    begin = re.search(r"^% >>> begin " + re.escape(NUMBERS_FILE) + r"\n", flat, re.M)
    if not begin:
        return None
    end = re.search(r"^% <<< end " + re.escape(NUMBERS_FILE) + r"\n", flat[begin.end():], re.M)
    if not end:
        raise ValueError("unterminated numbers.tex region")
    return begin.end(), begin.end() + end.start()


def _one_line_definition(rest: str) -> bool:
    """``[n]{body}`` filling the rest of a line, braces balanced."""
    rest = re.sub(r"^\[\d\]", "", rest.rstrip())
    try:
        return rest.startswith("{") and group_end(rest, 0) == len(rest)
    except ValueError:
        return False


def prune_numbers(flat: str) -> tuple[str, int, int]:
    """Drop every numbers.tex macro the rest of ``flat`` never uses, directly
    or through another kept macro. Returns (text, kept, defined)."""
    region = _numbers_region(flat)
    if region is None:
        return flat, 0, 0
    start, end = region
    numbers = flat[start:end]
    lines = numbers.split("\n")
    bodies: dict[str, str] = {}
    for line in lines:
        match = _NEWCOMMAND_LINE.match(line)
        if match:
            bodies[match.group(1)] = line[match.end():]
            if not _one_line_definition(line[match.end():]):
                raise ValueError(f"numbers.tex: \\{match.group(1)} is not defined on one line; "
                                 "flatten.py cannot prune it safely")
    if numbers.count("\\newcommand") != len(bodies):
        raise ValueError("numbers.tex has a \\newcommand that is not one per line; "
                         "flatten.py cannot prune it safely")
    rest = _check().strip_comments(flat[:start] + flat[end:])
    pending = [name for name in _CONTROL_WORD.findall(rest) if name in bodies]
    used: set[str] = set()
    while pending:
        name = pending.pop()
        if name in used:
            continue
        used.add(name)
        pending.extend(n for n in _CONTROL_WORD.findall(bodies[name]) if n in bodies)
    kept: list[str] = []
    group: list[str] = []  # a "% --- <input file>" comment and its macros
    note_written = False
    for line in lines:
        match = _NEWCOMMAND_LINE.match(line)
        if line.startswith("% ---"):
            if len(group) > 1:
                kept.extend(group)
            group = [line]
        elif match:
            if match.group(1) in used:
                group.append(line)
        else:
            if len(group) > 1:
                kept.extend(group)
            group = []
            if not note_written and not line.startswith("%"):
                kept.append(f"% flatten.py keeps only the {len(used)} of {len(bodies)} macros "
                            "that the text and tables use")
                kept.append("% (TeX memory on Overleaf); paper/numbers.tex holds all of them.")
                note_written = True
            kept.append(line)
    if len(group) > 1:
        kept.extend(group)
    return flat[:start] + "\n".join(kept) + flat[end:], len(used), len(bodies)


# ---------------------------------------------------------------------------
# Step 3: the bibliography, formatted from refs.bib
# ---------------------------------------------------------------------------

_CITE = re.compile(r"\\(no)?cite[a-zA-Z]*\*?\s*(?:\[[^\]]*\]\s*){0,2}\{([^{}]*)\}")
_BIBLIOGRAPHY = re.compile(r"\\bibliography\s*\{([^{}]*)\}")
_BIBSTYLE = re.compile(r"\\bibliographystyle\s*\{[^{}]*\}")


@dataclass
class Bibliography:
    entries: dict[str, object] = field(default_factory=dict)  # folded key -> BibEntry
    order: list[str] = field(default_factory=list)  # cited keys, first-citation order

    def number(self, key: str) -> int | None:
        folded = key.lower()
        return self.order.index(folded) + 1 if folded in self.order else None


def citation_order(text: str, all_keys: list[str]) -> list[str]:
    """Folded cite keys in order of first citation; ``\\nocite{*}`` adds the rest."""
    order: list[str] = []
    star = False
    for match in _CITE.finditer(text):
        for key in match.group(2).split(","):
            key = key.strip()
            if key == "*":
                star = True
            elif key and key.lower() not in order:
                order.append(key.lower())
    if star:
        order.extend(k for k in all_keys if k not in order)
    return order


def load_bibliography(paper_dir: Path, flat: str) -> Bibliography:
    check = _check()
    masked = check.strip_comments(flat)
    bib = Bibliography()
    for match in _BIBLIOGRAPHY.finditer(masked):
        for name in match.group(1).split(","):
            name = name.strip()
            path = paper_dir / (name if name.endswith(".bib") else name + ".bib")
            if name and path.is_file():
                entries, _ = check.parse_bib(path.read_text(encoding="utf-8"))
                for entry in entries:
                    bib.entries.setdefault(entry.key.lower(), entry)
    macros, stripped = collect_macros(masked)
    # A key missing from refs.bib gets no number, as with BibTeX: it prints [?].
    bib.order = [key for key in citation_order(expand(stripped, macros), list(bib.entries))
                 if key in bib.entries]
    return bib


def _initial(word: str) -> str:
    pieces = _split_top(word, re.compile(r"-"))
    if len(pieces) > 1:
        return "-".join(_initial(p) for p in pieces if p)
    if re.fullmatch(r"[A-Za-z]{1,2}\.", word):
        return word
    if word.startswith("{"):
        group = word[:group_end(word, 0)]
        inner = group[1:-1]
        unit = group if inner.startswith("\\") else inner[:1]
    elif word.startswith("\\"):
        match = re.match(r"\\.\s*(?:\{[^{}]*\}|[A-Za-z])", word)
        unit = match.group(0) if match else word[:2]
    else:
        unit = word[:1]
    return unit + "."


def _person(name: str) -> str:
    """BibTeX name -> IEEE "F.~M. Last" (von particles stay with the last name)."""
    parts = [p.strip() for p in _split_top(name, re.compile(r","))]
    words_of = lambda s: [w for w in _split_top(s, re.compile(r"\s+")) if w]  # noqa: E731
    if len(parts) == 1:
        words = words_of(parts[0])
        if len(words) == 1:
            return words[0]
        von = next((i for i, w in enumerate(words[:-1])
                    if re.match(r"[a-z]", w.lstrip("{"))), len(words) - 1)
        first, last, jr = words[:von], words[von:], ""
    else:
        last, first = words_of(parts[0]), words_of(parts[-1])
        jr = parts[1] if len(parts) == 3 else ""
    initials = "~".join(_initial(w) for w in first)
    surname = " ".join(last) + (f", {jr}" if jr else "")
    return f"{initials}~{surname}" if initials else surname


def format_names(raw: str) -> str:
    names = [n.strip() for n in _split_top(raw.strip(), re.compile(r"\s+and\s+")) if n.strip()]
    etal = any(n.lower() == "others" for n in names)
    people = []
    for name in names:
        if name.lower() == "others":
            continue
        corporate = name.startswith("{") and group_end(name, 0) == len(name)
        people.append(name if corporate else _person(name))
    if len(people) > MAX_AUTHORS:
        people, etal = people[:1], True
    if etal:
        return ", ".join(people) + " \\emph{et~al.}"
    if len(people) <= 2:
        return " and ".join(people)
    return ", ".join(people[:-1]) + ", and " + people[-1]


def _escape_plain(value: str) -> str:
    return re.sub(r"(?<!\\)([_%&#])", r"\\\1", value)


def format_reference(entry) -> str:
    """One entry in IEEE reference style, as LaTeX (the plain text renders it)."""
    f = {k: v.strip() for k, v in entry.fields.items()}
    kind = entry.kind
    pages = f.get("pages", "")
    pages = (("pp.~" if re.search(r"-", pages) else "p.~") + pages) if pages else ""
    date = " ".join(x for x in (f.get("month", ""), f.get("year", "")) if x)
    imprint = ": ".join(x for x in (f.get("address", ""), f.get("publisher", "")) if x)
    title = f.get("title", "")
    lead = [format_names(f["author"])] if f.get("author") else (
        [format_names(f["editor"]) + ", Ed."] if f.get("editor") else [])
    if kind == "book":
        head = ", ".join(lead + [f"\\emph{{{title}}}"]) if title else ", ".join(lead)
        tail = [", ".join(x for x in (imprint, date) if x)]
        text = head + ". " + ", ".join(x for x in tail if x)
    else:
        if kind == "article":
            venue = [f"\\emph{{{f['journal']}}}" if f.get("journal") else "",
                     f"vol.~{f['volume']}" if f.get("volume") else "",
                     f"no.~{f['number']}" if f.get("number") else "", pages, date]
            rest = ", ".join(x for x in venue if x)
        elif kind in ("inproceedings", "incollection", "conference"):
            first = ", ".join(x for x in (
                f"in \\emph{{{f['booktitle']}}}" if f.get("booktitle") else "",
                f"ser. {f['series']}" if f.get("series") else "",
                f"vol.~{f['volume']}" if f.get("volume") else "") if x)
            second = ", ".join(x for x in (imprint, date, pages) if x)
            # IEEEtran: "in Booktitle, vol. V. Address: Publisher, year, pp."
            joiner = ". " if imprint else ", "
            rest = joiner.join(x for x in (first, second) if x)
        elif kind == "techreport":
            rest = ", ".join(x for x in (f.get("institution", ""), f.get("address", ""),
                                         "Tech. Rep." + (f" {f['number']}" if f.get("number") else ""),
                                         date) if x)
        elif kind in ("phdthesis", "mastersthesis"):
            degree = "Ph.D. dissertation" if kind == "phdthesis" else "Master's thesis"
            rest = ", ".join(x for x in (degree, f.get("school", ""), f.get("address", ""), date) if x)
        else:  # misc, manual, online, unpublished, ...
            rest = ", ".join(x for x in (f.get("howpublished", ""), f.get("organization", ""),
                                         imprint, date) if x)
        if title and rest:
            head = ", ".join(lead + [f"``{title},''"]) + " " + rest
        elif title:
            head = ", ".join(lead + [f"``{title}.''"])
        else:
            head = ", ".join(lead + [rest])
        text = head
    text = text.rstrip(". ")
    if not text.endswith(("''", "?", "!")):
        text += "."
    extras = []
    if f.get("note"):
        extras.append(f["note"].rstrip(".") + ".")
    if f.get("eprint"):
        prefix = f.get("archiveprefix", "arXiv")
        extras.append(f"{prefix}:{_escape_plain(f['eprint'])}.")
    if f.get("doi"):
        extras.append(f"doi: {_escape_plain(f['doi'])}.")
    if f.get("url") and "\\url" not in f.get("howpublished", ""):
        extras.append(f"[Online]. Available: \\url{{{f['url']}}}")
    return " ".join([text] + extras)


def thebibliography(bib: Bibliography) -> str:
    lines = ["% Bibliography: formatted by paper/flatten.py from refs.bib, cited keys only,",
             "% numbered in order of first citation (IEEEtran's unsorted order). It",
             "% replaces \\bibliographystyle{IEEEtran} and \\bibliography{refs}, so this file",
             "% needs neither BibTeX nor refs.bib.",
             f"\\begin{{thebibliography}}{{{'9' * len(str(max(len(bib.order), 1)))}}}"]
    for folded in bib.order:
        entry = bib.entries[folded]
        lines += [f"\\bibitem{{{entry.key}}}", format_reference(entry), ""]
    if lines[-1] == "":
        lines.pop()
    lines.append("\\end{thebibliography}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Building the flat source and main_flat.tex
# ---------------------------------------------------------------------------

@dataclass
class Flat:
    source: str  # main.tex inlined, numbers pruned, bibliography still by key
    bib: Bibliography
    kept: int
    defined: int


def build_flat(paper_dir: Path | str = PAPER_DIR) -> Flat:
    paper_dir = Path(paper_dir)
    root = paper_dir / ROOT_FILE
    flat = inline_inputs(root.read_text(encoding="utf-8"), paper_dir, (ROOT_FILE,))
    flat, kept, defined = prune_numbers(flat)
    return Flat(flat, load_bibliography(paper_dir, flat), kept, defined)


TEX_HEADER = """\
% main_flat.tex -- GENERATED by paper/flatten.py from main.tex; do not edit.
% Rerun `python reproduce.py paper` (or `python3 paper/flatten.py`).
%
% The whole paper in one file, for Overleaf: every \\input (sections/,
% tables/, numbers.tex) is inlined between "% >>> begin" and "% <<< end"
% markers; numbers.tex is cut to the macros the text and tables use; and the
% bibliography is a thebibliography block formatted from refs.bib, so no
% BibTeX run and no refs.bib are needed. Upload this file with the figures/
% directory (figures/*.pdf), set the compiler to pdfLaTeX and the main
% document to main_flat.tex. main.tex with BibTeX remains the reference
% build; paper/README.md has both.
%
"""


def flat_tex(flat: Flat) -> str:
    check = _check()
    masked = check.strip_comments(flat.source)
    out, position = [], 0
    spans = sorted([(m.start(), m.end(), "style") for m in _BIBSTYLE.finditer(masked)]
                   + [(m.start(), m.end(), "bib") for m in _BIBLIOGRAPHY.finditer(masked)])
    for start, end, kind in spans:
        out.append(flat.source[position:start])
        if kind == "style":
            out.append("% (bibliography style: IEEE, formatted by flatten.py below)")
        elif flat.bib.order:
            out.append(thebibliography(flat.bib))
        else:  # an empty thebibliography is a LaTeX error
            out.append("% (no citations: the bibliography is empty)")
        position = end
    out.append(flat.source[position:])
    return TEX_HEADER + "".join(out).rstrip("\n") + "\n"


# ---------------------------------------------------------------------------
# Macros: \newcommand definitions, and expansion
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Macro:
    nargs: int
    default: str | None
    body: str


_DEFINE = re.compile(r"\\(newcommand|renewcommand|providecommand|DeclareRobustCommand)\*?")


def collect_macros(text: str) -> tuple[dict[str, Macro], str]:
    """Every zero- or n-argument command definition, and ``text`` without them."""
    macros: dict[str, Macro] = {}
    out, position = [], 0
    for match in _DEFINE.finditer(text):
        if match.start() < position:
            continue
        index = _skip_space(text, match.end())
        if index < len(text) and text[index] == "{":
            end = group_end(text, index)
            name = text[index + 1:end - 1].strip()
        else:
            word = _CONTROL_WORD.match(text, index)
            if not word:
                continue
            name, end = word.group(0), word.end()
        count, end = read_opt(text, end)
        default, end = read_opt(text, end)
        body_start = _skip_space(text, end)
        if body_start >= len(text) or text[body_start] != "{":
            continue
        body_end = group_end(text, body_start)
        name = name.lstrip("\\")
        if not (match.group(1) == "providecommand" and name in macros):
            macros[name] = Macro(int(count) if count and count.strip().isdigit() else 0,
                                 default, text[body_start + 1:body_end - 1])
        out.append(text[position:match.start()])
        position = body_end
    out.append(text[position:])
    return macros, "".join(out)


def expand(text: str, macros: dict[str, Macro], depth: int = 0) -> str:
    """``text`` with every user macro expanded, recursively. A zero-argument
    control word swallows the spaces after it, as TeX's tokenizer does."""
    if depth > 40:
        raise ValueError("macro expansion does not terminate")
    out, index = [], 0
    while index < len(text):
        slash = text.find("\\", index)
        if slash < 0:
            out.append(text[index:])
            break
        out.append(text[index:slash])
        match = _CONTROL_WORD.match(text, slash)
        if not match:
            out.append(text[slash:slash + 2])
            index = slash + 2
            continue
        name, index = match.group(1), match.end()
        macro = macros.get(name)
        if macro is None:
            out.append(match.group(0))
            continue
        args: list[str] = []
        if macro.nargs:
            if macro.default is not None:
                value, index = read_opt(text, index)
                args.append(macro.default if value is None else value)
            while len(args) < macro.nargs:
                value, index = read_arg(text, index)
                args.append(value or "")
        else:
            index = _skip_space(text, index)
        body = macro.body
        for number, value in enumerate(args, start=1):
            body = body.replace(f"#{number}", value)
        out.append(expand(body, macros, depth + 1))
    return "".join(out)


# ---------------------------------------------------------------------------
# Step 4: the plain-text renderer
# ---------------------------------------------------------------------------

SYMBOLS = {
    # Greek
    **{name: chr(code) for name, code in (
        ("alpha", 0x3B1), ("beta", 0x3B2), ("gamma", 0x3B3), ("delta", 0x3B4),
        ("epsilon", 0x3B5), ("varepsilon", 0x3B5), ("zeta", 0x3B6), ("eta", 0x3B7),
        ("theta", 0x3B8), ("vartheta", 0x3D1), ("iota", 0x3B9), ("kappa", 0x3BA),
        ("lambda", 0x3BB), ("mu", 0x3BC), ("nu", 0x3BD), ("xi", 0x3BE), ("pi", 0x3C0),
        ("rho", 0x3C1), ("sigma", 0x3C3), ("tau", 0x3C4), ("upsilon", 0x3C5),
        ("phi", 0x3C6), ("varphi", 0x3C6), ("chi", 0x3C7), ("psi", 0x3C8), ("omega", 0x3C9),
        ("Gamma", 0x393), ("Delta", 0x394), ("Theta", 0x398), ("Lambda", 0x39B),
        ("Xi", 0x39E), ("Pi", 0x3A0), ("Sigma", 0x3A3), ("Phi", 0x3A6), ("Psi", 0x3A8),
        ("Omega", 0x3A9))},
    # math operators and relations
    "times": "\u00d7", "cdot": "\u00b7", "pm": "\u00b1", "mp": "\u2213", "div": "\u00f7",
    "le": "\u2264", "leq": "\u2264", "ge": "\u2265", "geq": "\u2265", "ne": "\u2260",
    "neq": "\u2260", "approx": "\u2248", "sim": "~", "simeq": "\u2243", "equiv": "\u2261",
    "propto": "\u221d", "ll": "\u226a", "gg": "\u226b", "infty": "\u221e",
    "to": "\u2192", "rightarrow": "\u2192", "leftarrow": "\u2190", "Rightarrow": "\u21d2",
    "Leftarrow": "\u21d0", "leftrightarrow": "\u2194", "mapsto": "\u21a6",
    "in": "\u2208", "notin": "\u2209", "subset": "\u2282", "subseteq": "\u2286",
    "supset": "\u2283", "cup": "\u222a", "cap": "\u2229", "setminus": "\u2216",
    "emptyset": "\u2205", "varnothing": "\u2205", "forall": "\u2200", "exists": "\u2203",
    "partial": "\u2202", "nabla": "\u2207", "sum": "\u03a3", "prod": "\u03a0",
    "int": "\u222b", "cdots": "\u22ef", "vdots": "\u22ee", "prime": "\u2032",
    "star": "\u22c6", "ast": "*", "circ": "\u2218", "bullet": "\u2022",
    "langle": "\u27e8", "rangle": "\u27e9", "lfloor": "\u230a", "rfloor": "\u230b",
    "lceil": "\u2308", "rceil": "\u2309", "mid": "|", "vert": "|", "Vert": "\u2016",
    "lVert": "\u2016", "rVert": "\u2016", "ell": "\u2113", "quad": "  ", "qquad": "    ",
    "lbrace": "{", "rbrace": "}",
    # text symbols
    "ldots": "\u2026", "dots": "\u2026", "textellipsis": "\u2026",
    "textendash": "\u2013", "textemdash": "\u2014", "S": "\u00a7", "P": "\u00b6",
    "dag": "\u2020", "dagger": "\u2020", "ddag": "\u2021", "ddagger": "\u2021",
    "copyright": "\u00a9", "textcopyright": "\u00a9", "textregistered": "\u00ae",
    "texttrademark": "\u2122", "textdegree": "\u00b0", "textbullet": "\u2022",
    "textperiodcentered": "\u00b7", "textasciitilde": "~", "textasciicircum": "^",
    "textbar": "|", "textless": "<", "textgreater": ">", "textunderscore": "_",
    "textquotedblleft": "\u201c", "textquotedblright": "\u201d",
    "textquoteleft": "\u2018", "textquoteright": "\u2019", "slash": "/",
    "textmu": "\u03bc", "checkmark": "\u2713", "textpm": "\u00b1", "texttimes": "\u00d7",
    "i": "\u0131", "o": "\u00f8", "O": "\u00d8", "ss": "\u00df", "ae": "\u00e6",
    "AE": "\u00c6", "oe": "\u0153", "OE": "\u0152", "aa": "\u00e5", "AA": "\u00c5",
    "l": "\u0142", "L": "\u0141",
    "LaTeX": "LaTeX", "TeX": "TeX", "BibTeX": "BibTeX",
}

# Named functions keep their spaces (TeX's spacing is lost in text).
MATH_WORDS = {"log", "ln", "exp", "max", "min", "arg", "argmax", "argmin", "sup", "inf",
              "lim", "sin", "cos", "tan", "det", "Pr", "deg", "dim", "gcd", "mod", "bmod"}

ACCENTS = {"'": "\u0301", "`": "\u0300", '"': "\u0308", "^": "\u0302", "~": "\u0303",
           "=": "\u0304", ".": "\u0307", "c": "\u0327", "v": "\u030c", "u": "\u0306",
           "H": "\u030b", "r": "\u030a", "k": "\u0328"}
MATH_ACCENTS = {"hat": "\u0302", "bar": "\u0304", "tilde": "\u0303", "vec": "\u20d7",
                "dot": "\u0307", "ddot": "\u0308", "widehat": "\u0302", "overline": "\u0304",
                "widetilde": "\u0303"}

# Commands whose argument is printed as it stands (the formatting is lost).
TRANSPARENT = {"textbf", "textit", "emph", "texttt", "textrm", "textsf", "textsc", "textup",
               "textmd", "textsl", "textnormal", "mbox", "hbox", "fbox", "text", "mathrm",
               "mathbf", "mathit", "mathsf", "mathtt", "mathcal", "mathbb", "mathnormal",
               "boldsymbol", "bm", "operatorname", "underline", "ensuremath", "MakeUppercase",
               "uppercase", "IEEEauthorblockN", "IEEEauthorblockA", "IEEEPARstart",
               "texorpdfstring", "markboth", "thanks"}
# Commands printed as nothing, with this many (optional, mandatory) arguments.
DROPPED = {
    "label": (0, 1), "hspace": (0, 1), "vspace": (0, 1), "setlength": (0, 2),
    "addtolength": (0, 2), "setcounter": (0, 2), "addtocounter": (0, 2), "color": (1, 1),
    "typeout": (0, 1), "phantom": (0, 1), "hphantom": (0, 1), "vphantom": (0, 1),
    "index": (0, 1), "cline": (0, 1), "cmidrule": (1, 1), "rule": (1, 2),
    "pagestyle": (0, 1), "thispagestyle": (0, 1), "usepackage": (1, 1),
    "documentclass": (1, 1), "bibliographystyle": (0, 1), "nocite": (0, 1),
    "IEEEauthorrefmark": (0, 1), "renewcommand": (0, 2), "linespread": (0, 1),
    "raisebox": (0, 1), "arraystretch": (0, 0), "includegraphics": (1, 1),
}
SWITCHES = {"centering", "raggedright", "raggedleft", "noindent", "indent", "scriptsize",
            "footnotesize", "small", "normalsize", "large", "Large", "LARGE", "tiny", "huge",
            "Huge", "bfseries", "itshape", "ttfamily", "rmfamily", "sffamily", "mdseries",
            "upshape", "slshape", "scshape", "normalfont", "em", "bf", "it", "tt", "rm", "sf",
            "hline", "toprule", "midrule", "bottomrule", "protect", "relax", "smallskip",
            "medskip", "bigskip", "vfill", "hfill", "clearpage", "newpage", "pagebreak",
            "nolinebreak", "allowbreak", "xspace", "null", "strut", "nonumber", "notag",
            "displaystyle", "textstyle", "scriptstyle", "left", "right", "big", "Big",
            "bigg", "Bigg", "bigl", "bigr", "Bigl", "Bigr", "middle", "limits", "nolimits",
            "IEEEoverridecommandlockouts", "maketitle", "item", "frenchspacing",
            "unskip", "ignorespaces", "leavevmode", "par", "selectfont", "appendices",
            "appendix", "IEEEpeerreviewmaketitle", "balance", "and"}
# Commands whose last mandatory argument is the printed content, by argument
# pattern ("o" optional, "m" mandatory): \multirow{2}{*}{text} prints "text".
CONTENT_LAST = {"multirow": "omom", "makecell": "om", "shortstack": "om", "resizebox": "mmm",
                "scalebox": "mom", "parbox": "omm", "num": "om", "framebox": "oom",
                "rotatebox": "omm", "adjustbox": "mm"}
STARRABLE = {"hspace", "vspace", "section", "subsection", "subsubsection", "paragraph",
             "caption", "footnote", "cite", "newpage", "item"}
REF_COMMANDS = {"ref", "autoref", "cref", "Cref", "nameref", "pageref", "eqref"}
SECTIONING = {"section": 1, "subsection": 2, "subsubsection": 3, "paragraph": 4,
              "subparagraph": 5}
MATH_ENVIRONMENTS = {"equation", "equation*", "align", "align*", "gather", "gather*",
                     "multline", "multline*", "eqnarray", "eqnarray*", "displaymath", "math"}
LIST_ENVIRONMENTS = {"itemize": "-", "enumerate": "#", "description": ""}


def _plain(text: str) -> str:
    """Collapse whitespace (never NBSP) and drop forced breaks."""
    return re.sub(r"[ \t\r\n]+", " ", text.replace(BREAK, " ")).strip()


def _wrap(text: str, width: int, first: str = "", rest: str = "") -> str:
    wrapper = textwrap.TextWrapper(width=width, initial_indent=first, subsequent_indent=rest,
                                   break_long_words=False, break_on_hyphens=False)
    return "\n".join(wrapper.wrap(text)) if text else first.rstrip()


def _underline(text: str, char: str) -> str:
    return f"{text}\n{char * min(len(text), WIDTH)}"


def _compress(numbers: list[int]) -> str:
    """IEEE/cite.sty citation list: [1]--[3], [5]."""
    numbers = sorted(set(numbers))
    out, index = [], 0
    while index < len(numbers):
        run = index
        while run + 1 < len(numbers) and numbers[run + 1] == numbers[run] + 1:
            run += 1
        if run - index >= 2:
            out.append(f"[{numbers[index]}]\u2013[{numbers[run]}]")
        else:
            out.extend(f"[{n}]" for n in numbers[index:run + 1])
        index = run + 1
    return ", ".join(out)


def _accent(base: str, mark: str) -> str:
    if not base:
        return unicodedata.normalize("NFC", mark)
    return unicodedata.normalize("NFC", base[0] + mark + base[1:])


def _script(kind: str, value: str) -> str:
    value = value.strip()
    if kind == "^" and value in ("*", "\u2020", "\u2021", "\u2032", "'", "\u2032\u2032"):
        return value
    if re.fullmatch(r"[+-]?[A-Za-z0-9.]+|.", value):
        return kind + value
    return f"{kind}({value})"


def _number_family(name: str) -> bool:
    return bool(_check().NUMBER_MACRO.fullmatch("\\" + name))


def _column_width(spec: str) -> int:
    match = re.fullmatch(r"\s*([\d.]+)\s*(in|cm|mm|pt|bp|em|ex)\s*", spec)
    if match:
        value, unit = float(match.group(1)), match.group(2)
        inches = value * {"in": 1, "cm": 1 / 2.54, "mm": 1 / 25.4, "pt": 1 / 72.27,
                          "bp": 1 / 72, "em": 1 / 7.2, "ex": 1 / 15}[unit]
        return max(4, round(inches * CHARS_PER_INCH))
    match = re.fullmatch(r"\s*([\d.]*)\s*\\(linewidth|columnwidth|textwidth|hsize)\s*", spec)
    if match:
        fraction = float(match.group(1) or 1)
        base = TEXT_IN if match.group(2) == "textwidth" else COLUMN_IN
        return max(4, round(fraction * base * CHARS_PER_INCH))
    return 30


def parse_colspec(spec: str) -> list[tuple[str, int | None]]:
    """(alignment, wrap width or None) per column of a tabular spec."""
    columns: list[tuple[str, int | None]] = []
    index = 0
    while index < len(spec):
        char = spec[index]
        if char in "lcr":
            columns.append((char, None))
            index += 1
        elif char in "pmb":
            width, index = read_arg(spec, index + 1)
            columns.append(("l", _column_width(width or "")))
        elif char == "X":
            columns.append(("l", 30))
            index += 1
        elif char == "S":
            columns.append(("r", None))
            index += 1
        elif char in "@!<>":
            _, index = read_arg(spec, index + 1)
        elif char == "*":
            count, index = read_arg(spec, index + 1)
            inner, index = read_arg(spec, index)
            columns.extend(parse_colspec(inner or "") * int((count or "1").strip() or 1))
        else:  # | and spaces
            index += 1
    return columns


@dataclass
class Cell:
    text: str
    span: int
    align: str


class TextRenderer:
    """Render the expanded body. Run once to collect labels, then again with
    them (``labels=``) so forward references resolve."""

    def __init__(self, paper_dir: Path, bib: Bibliography, labels: dict[str, str] | None = None):
        self.paper_dir = paper_dir
        self.bib = bib
        self.final_labels = labels or {}
        self.labels: dict[str, str] = {}
        self.counters = {"section": 0, "subsection": 0, "subsubsection": 0,
                         "figure": 0, "table": 0, "equation": 0}
        self.appendix = False
        self.section = ""
        self.subsection = ""
        self.current = ""  # what a \label here refers to
        self.title = ""
        self.author = ""

    # -- inline --------------------------------------------------------------

    def inline(self, text: str, math: bool = False) -> str:
        out: list[str] = []
        index = 0
        n = len(text)
        while index < n:
            char = text[index]
            if char == "\\":
                word = _CONTROL_WORD.match(text, index)
                if word:
                    name, after = word.group(1), word.end()
                    if after < n and text[after] == "*" and name in STARRABLE:
                        after += 1
                    if not math and name not in MATH_WORDS:
                        after = _skip_space(text, after)
                    rendered, index = self.command(name, text, after, math)
                    out.append(rendered)
                else:
                    rendered, index = self.symbol(text[index + 1:index + 2], text, index + 2, math)
                    out.append(rendered)
            elif char == "{":
                end = group_end(text, index)
                out.append(self.inline(text[index + 1:end - 1], math))
                index = end
            elif char == "}":
                index += 1
            elif char == "$":
                display = text.startswith("$$", index)
                start = index + (2 if display else 1)
                end = self._math_end(text, start, "$$" if display else "$")
                # Inline math is short; keep it on one line.
                math_text = re.sub(r"[ \t\r\n]+", " ", self.inline(text[start:end], math=True))
                out.append(math_text.strip().replace(" ", NBSP))
                index = end + (2 if display else 1)
            elif char == "~":
                out.append(NBSP)
                index += 1
            elif char in "^_" and math:
                value, index = self._script_arg(text, index + 1)
                out.append(_script(char, value))
            elif char == "&":
                out.append(" ")
                index += 1
            elif not math and text.startswith("---", index):
                out.append("\u2014")
                index += 3
            elif not math and text.startswith("--", index):
                out.append("\u2013")
                index += 2
            elif not math and text.startswith("``", index):
                out.append("\u201c")
                index += 2
            elif not math and text.startswith("''", index):
                out.append("\u201d")
                index += 2
            else:
                out.append(char)
                index += 1
        return "".join(out)

    @staticmethod
    def _math_end(text: str, start: int, closer: str) -> int:
        index = start
        while index < len(text):
            if text[index] == "\\":
                index += 2
                continue
            if text.startswith(closer, index):
                return index
            index += 1
        return len(text)

    def _script_arg(self, text: str, index: int) -> tuple[str, int]:
        while index < len(text) and text[index] == " ":
            index += 1
        if index >= len(text):
            return "", index
        if text[index] == "{":
            end = group_end(text, index)
            return self.inline(text[index + 1:end - 1], math=True), end
        if text[index] == "\\":
            word = _CONTROL_WORD.match(text, index)
            end = word.end() if word else index + 2
            return self.inline(text[index:end], math=True), end
        return text[index], index + 1

    def _arg(self, text: str, index: int, math: bool) -> tuple[str, int]:
        value, index = read_arg(text, index)
        return (self.inline(value, math) if value else ""), index

    def symbol(self, char: str, text: str, index: int, math: bool) -> tuple[str, int]:
        if char in "%_&#${}":
            return char, index
        if char == ",":
            return NBSP, index
        if char in ";: >\n\t":
            return " ", index
        if char in "!-/@":
            return "", index
        if char == "\\":
            if index < len(text) and text[index] == "*":
                index += 1
            _, index = read_opt(text, index)
            return BREAK, index
        if char == "|":
            return "\u2016", index
        if char in ACCENTS and not math:
            value, index = read_arg(text, index)
            base = self.inline(value or "", math)
            return _accent(base, ACCENTS[char]), index
        return char, index

    def command(self, name: str, text: str, index: int, math: bool) -> tuple[str, int]:
        if name in SYMBOLS:
            return SYMBOLS[name], index
        if name in MATH_WORDS:
            return name, index
        if len(name) == 1 and name in ACCENTS:
            value, index = read_arg(text, index)
            return _accent(self.inline(value or "", math), ACCENTS[name]), index
        if name in MATH_ACCENTS:
            value, index = self._arg(text, index, math)
            return _accent(value, MATH_ACCENTS[name]), index
        if name == "IEEEPARstart":
            first, index = self._arg(text, index, math)
            second, index = self._arg(text, index, math)
            return first + second, index
        if name in ("IEEEauthorblockN", "IEEEauthorblockA"):
            value, index = self._arg(text, index, math)
            return value + BREAK, index
        if name in TRANSPARENT:
            return self._arg(text, index, math)
        if name in ("textsuperscript", "textsubscript"):
            value, index = self._arg(text, index, math)
            return _script("^" if name == "textsuperscript" else "_", value), index
        if name in CONTENT_LAST:  # table and box helpers: print the last argument
            for kind in CONTENT_LAST[name]:
                if kind == "o":
                    _, index = read_opt(text, index)
                else:
                    value, index = read_arg(text, index)
            return (self.inline(value or "", math) if CONTENT_LAST[name] else ""), index
        if name in ("textcolor", "colorbox"):
            _, index = read_opt(text, index)
            _, index = read_arg(text, index)
            return self._arg(text, index, math)
        if name in ("url", "nolinkurl", "path"):
            value, index = read_arg(text, index)
            return (value or "").replace("\\%", "%").replace("\\_", "_"), index
        if name == "href":
            url, index = read_arg(text, index)
            label, index = self._arg(text, index, math)
            return f"{label} ({url})", index
        if name == "footnote":
            _, index = read_opt(text, index)
            value, index = self._arg(text, index, math)
            return f" [Footnote: {_plain(value)}]", index
        if name in ("cite", "citep", "citet", "citealp", "citeauthor"):
            _, index = read_opt(text, index)
            _, index = read_opt(text, index)
            keys, index = read_arg(text, index)
            numbers = [self.bib.number(k.strip()) for k in (keys or "").split(",") if k.strip()]
            known = [n for n in numbers if n is not None]
            rendered = _compress(known) if known else ""
            if len(known) < len(numbers):
                rendered = (rendered + ", " if rendered else "") + "[?]"
            return rendered, index
        if name in REF_COMMANDS:
            key, index = read_arg(text, index)
            value = self.final_labels.get((key or "").strip(), "??")
            return (f"({value})" if name == "eqref" else value), index
        if name == "label":
            key, index = read_arg(text, index)
            self.labels[(key or "").strip()] = self.current
            return "", index
        if name in ("frac", "dfrac", "tfrac"):
            top, index = self._arg(text, index, True)
            bottom, index = self._arg(text, index, True)
            paren = lambda s: f"({s.strip()})" if re.search(r"[\s+\-\u00b7\u00d7/]", s.strip()) else s.strip()  # noqa: E731
            return f"{paren(top)}/{paren(bottom)}", index
        if name == "sqrt":
            _, index = read_opt(text, index)
            value, index = self._arg(text, index, True)
            return f"\u221a({value})" if len(value) > 1 else f"\u221a{value}", index
        if name in ("newline", "linebreak", "break"):
            return BREAK, index
        if name in DROPPED:
            optional, mandatory = DROPPED[name]
            for _ in range(optional):
                _, index = read_opt(text, index)
            for _ in range(mandatory):
                _, index = read_arg(text, index)
            return "", index
        if name in SWITCHES:
            if name in ("left", "right", "bigl", "bigr") and text[index:index + 1] == ".":
                index += 1
            return "", index
        if name in ("caption",):
            _, index = read_opt(text, index)
            return self._arg(text, index, math)
        if _number_family(name):
            return "??", index  # an undefined result macro; TeX would stop here
        return "", index  # unknown: its arguments, if any, print as text

    # -- blocks --------------------------------------------------------------

    def paragraph(self, raw: str, width: int, lead: str = "") -> str:
        text = self.inline(raw)
        pieces = [_plain(p) for p in text.split(BREAK)]
        pieces = [p for p in pieces if p]
        if not pieces:
            return ""
        pieces[0] = lead + pieces[0]
        return "\n".join(_wrap(p, width) for p in pieces)

    def blocks(self, text: str, width: int = WIDTH, lead: str = "") -> list[str]:
        out: list[str] = []
        para: list[str] = []
        leads = [lead]

        def flush() -> None:
            raw = "".join(para)
            para.clear()
            rendered = self.paragraph(raw, width, leads[0])
            if rendered:
                out.append(rendered)
                leads[0] = ""

        index, n = 0, len(text)
        while index < n:
            char = text[index]
            if char == "\\":
                word = _CONTROL_WORD.match(text, index)
                if not word:
                    para.append(text[index:index + 2])
                    index += 2
                    continue
                name, after = word.group(1), word.end()
                star = after < n and text[after] == "*"
                if star:
                    after += 1
                if name in SECTIONING:
                    flush()
                    _, after = read_opt(text, after)
                    title, after = read_arg(text, after)
                    out.append(self.heading(name, star, title or ""))
                    index = after
                elif name == "begin":
                    env, after = read_arg(text, after)
                    env = (env or "").strip()
                    body_end, end = environment_end(text, after, env)
                    flush()
                    out.extend(self.environment(env, text[after:body_end], width))
                    index = end
                elif name in ("title", "author"):
                    _, after = read_opt(text, after)
                    value, after = read_arg(text, after)
                    setattr(self, name, value or "")
                    index = after
                elif name == "maketitle":
                    flush()
                    out.append(self.title_block())
                    index = after
                elif name in ("appendices", "appendix"):
                    flush()
                    self.appendix = True
                    self.counters["section"] = 0
                    index = after
                elif name == "bibliography":
                    flush()
                    _, after = read_arg(text, after)
                    if self.bib.order:
                        out.append(self.references(width))
                    index = after
                elif name == "par":
                    flush()
                    index = after
                else:
                    para.append(text[index:after])
                    index = after
            elif char == "{":
                end = group_end(text, index)
                para.append(text[index:end])
                index = end
            elif char == "$":
                display = text.startswith("$$", index)
                start = index + (2 if display else 1)
                end = self._math_end(text, start, "$$" if display else "$") + (2 if display else 1)
                para.append(text[index:end])
                index = end
            elif char == "\n" and re.match(r"\n[ \t]*\n", text[index:]):
                flush()
                index = re.match(r"\n[ \t\n]*", text[index:]).end() + index
            else:
                para.append(char)
                index += 1
        flush()
        return out

    def heading(self, kind: str, star: bool, title: str) -> str:
        title = _plain(self.inline(title))
        level = SECTIONING[kind]
        if star or level > 3:
            return _underline(title, "=" if level == 1 else "-") if level <= 3 else title
        if level == 1:
            self.counters["section"] += 1
            self.counters["subsection"] = 0
            number = self.counters["section"]
            self.section = letter(number) if self.appendix else roman(number)
            self.current = self.section
            label = f"Appendix {self.section}. {title}" if self.appendix else f"{self.section}. {title}"
            return _underline(label, "=")
        if level == 2:
            self.counters["subsection"] += 1
            self.counters["subsubsection"] = 0
            self.subsection = f"{self.section}-{letter(self.counters['subsection'])}"
            self.current = self.subsection
            return _underline(f"{self.subsection}. {title}", "-")
        self.counters["subsubsection"] += 1
        self.current = f"{self.subsection}{self.counters['subsubsection']}"
        return f"{self.current}) {title}"

    def title_block(self) -> str:
        title = _plain(self.inline(self.title))
        lines = [_wrap(title, WIDTH), ""]
        for author in _split_top(self.author, re.compile(r"\\and(?![A-Za-z])")):
            rendered = [_plain(p) for p in self.inline(author).split(BREAK)]
            rendered = [p for p in rendered if p]
            if rendered:
                lines += rendered + [""]
        return "\n".join(lines).rstrip()

    def references(self, width: int) -> str:
        lines = [_underline("References", "=")]
        tag_width = len(f"[{len(self.bib.order)}] ")
        for number, folded in enumerate(self.bib.order, start=1):
            body = _plain(self.inline(format_reference(self.bib.entries[folded])))
            tag = f"[{number}]".ljust(tag_width)
            lines.append(_wrap(body, width, tag, " " * tag_width))
        return "\n\n".join(lines)

    def environment(self, name: str, body: str, width: int) -> list[str]:
        if name == "document":
            return self.blocks(body, width)
        if name == "abstract":
            return self.blocks(body, width, lead="Abstract\u2014")
        if name == "IEEEkeywords":
            return self.blocks(body, width, lead="Index Terms\u2014")
        if name in ("figure", "figure*", "table", "table*"):
            _, index = read_opt(body, 0)
            if name.startswith("figure"):
                return [self.figure(body[index:], width)]
            return self.table(body[index:], width)
        if name in ("tabular", "tabular*", "tabularx"):
            return [self.tabular(name, body)]
        if name in MATH_ENVIRONMENTS:
            return [self.display_math(body, numbered=not name.endswith("*")
                                      and name not in ("displaymath", "math"), width=width)]
        if name in LIST_ENVIRONMENTS:
            return [self.items(name, body, width)]
        if name in ("verbatim", "lstlisting"):
            return ["\n".join("    " + line for line in body.strip("\n").split("\n"))]
        if name == "minipage":
            _, index = read_opt(body, 0)
            _, index = read_arg(body, index)
            return self.blocks(body[index:], width)
        _, index = read_opt(body, 0)
        return self.blocks(body[index:], width)

    def _labels(self, body: str) -> None:
        for match in re.finditer(r"\\label\s*\{([^{}]*)\}", body):
            self.labels[match.group(1).strip()] = self.current

    def _caption(self, body: str) -> tuple[str, str]:
        """(rendered caption, body without the caption)."""
        match = re.search(r"\\caption\*?", body)
        if not match:
            return "", body
        _, index = read_opt(body, match.end())
        value, index = read_arg(body, index)
        return _plain(self.inline(value or "")), body[:match.start()] + body[index:]

    def figure(self, body: str, width: int) -> str:
        self.counters["figure"] += 1
        number = self.counters["figure"]
        self.current = str(number)
        caption, rest = self._caption(body)
        self._labels(body)
        check = _check()
        paths = []
        for match in re.finditer(r"\\includegraphics\s*(?:\[[^\]]*\])?\s*\{([^{}]+)\}", rest):
            target = check._graphics_target(self.paper_dir, match.group(1).strip())
            paths.append(f"paper/{target}")
        see = f" \u2014 see {', '.join(paths)}" if paths else ""
        text = f"[Figure {number}: {caption}{see}]" if caption else f"[Figure {number}{see}]"
        return _wrap(text, width)

    def table(self, body: str, width: int) -> list[str]:
        self.counters["table"] += 1
        number = roman(self.counters["table"])
        self.current = number
        caption, rest = self._caption(body)
        self._labels(body)
        out = [_wrap(f"TABLE {number}: {caption}" if caption else f"TABLE {number}", width)]
        pattern = re.compile(r"\\begin\s*\{(tabular\*?|tabularx)\}")
        notes, position = [], 0
        while True:
            match = pattern.search(rest, position)
            if not match:
                break
            notes.append(rest[position:match.start()])
            env = match.group(1)
            body_end, end = environment_end(rest, match.end(), env)
            out.append(self.tabular(env, rest[match.end():body_end]))
            position = end
        notes.append(rest[position:])
        note = self.paragraph("".join(notes), width)
        if note:
            out.append(note)
        return out

    def tabular(self, env: str, body: str) -> str:
        index = 0
        if env in ("tabular*", "tabularx"):
            _, index = read_arg(body, index)
        _, index = read_opt(body, index)
        spec, index = read_arg(body, index)
        columns = parse_colspec(spec or "")
        rows: list[list[Cell] | str] = []
        for raw in _split_top(body[index:], re.compile(r"\\\\(?:\*)?(?:\s*\[[^\]]*\])?")):
            raw = raw.strip()
            while True:
                rule = re.match(r"\\(hline|toprule|midrule|bottomrule)\b|\\(cline|cmidrule)\s*"
                                r"(?:\([^)]*\))?\s*\{[^{}]*\}", raw)
                if not rule:
                    break
                rows.append("=" if rule.group(1) in ("toprule", "bottomrule") else "-")
                raw = raw[rule.end():].strip()
            if not raw:
                continue
            cells = []
            for cell in _split_top(raw, re.compile(r"&")):
                cell = cell.strip()
                multi = re.match(r"\\multicolumn\s*\{(\d+)\}\s*\{([^{}]*)\}", cell)
                if multi:
                    content, _ = read_arg(cell, multi.end())
                    spans = parse_colspec(multi.group(2)) or [("l", None)]
                    cells.append(Cell(_plain(self.inline(content or "")), int(multi.group(1)),
                                      spans[0][0]))
                else:
                    column = columns[len(cells)] if len(cells) < len(columns) else ("l", None)
                    cells.append(Cell(_plain(self.inline(cell)), 1, column[0]))
            rows.append(cells)
        return self._layout(columns, rows)

    @staticmethod
    def _layout(columns: list[tuple[str, int | None]], rows: list) -> str:
        count = max([len(columns)] + [sum(c.span for c in r) for r in rows if not isinstance(r, str)])
        columns = columns + [("l", None)] * (count - len(columns))
        widths = [0] * count
        for row in rows:
            if isinstance(row, str):
                continue
            position = 0
            for cell in row:
                if cell.span == 1 and position < count:
                    limit = columns[position][1]
                    if limit:
                        longest = max([len(w) for w in cell.text.split()] + [0])
                        widths[position] = max(widths[position], min(limit, len(cell.text)), longest)
                    else:
                        widths[position] = max(widths[position], len(cell.text))
                position += cell.span
        for row in rows:  # widen the last spanned column for a partial span
            if isinstance(row, str):
                continue
            position = 0
            for cell in row:
                if 1 < cell.span < count:
                    end = min(position + cell.span, count)
                    have = sum(widths[position:end]) + len(TABLE_SEP) * (end - position - 1)
                    if len(cell.text) > have:
                        widths[end - 1] += len(cell.text) - have
                position += cell.span
        total = sum(widths) + len(TABLE_SEP) * (count - 1)
        lines: list[str] = []
        for row in rows:
            if isinstance(row, str):
                lines.append(row * total)
                continue
            position, blocks = 0, []
            for cell in row:
                end = min(position + cell.span, count)
                if cell.span >= count:
                    blocks.append(([cell.text], max(total, len(cell.text)), cell.align))
                    position = end
                    continue
                cell_width = sum(widths[position:end]) + len(TABLE_SEP) * (end - position - 1)
                wrap = columns[position][1] if cell.span == 1 else None
                content = (textwrap.wrap(cell.text, cell_width, break_long_words=False,
                                         break_on_hyphens=False) or [""]) if wrap else [cell.text]
                blocks.append((content, cell_width, cell.align))
                position = end
            while position < count:  # a short row
                blocks.append(([""], widths[position], "l"))
                position += 1
            for line in range(max(len(b[0]) for b in blocks)):
                parts = []
                for content, cell_width, align in blocks:
                    piece = content[line] if line < len(content) else ""
                    parts.append(piece.rjust(cell_width) if align == "r" else
                                 piece.center(cell_width) if align == "c" else
                                 piece.ljust(cell_width))
                lines.append(TABLE_SEP.join(parts).rstrip())
        return "\n".join(lines)

    def display_math(self, body: str, numbered: bool, width: int) -> str:
        labels = re.findall(r"\\label\s*\{([^{}]*)\}", body)
        unnumbered = bool(re.search(r"\\(nonumber|notag)\b", body))
        tag = ""
        if numbered and not unnumbered:
            self.counters["equation"] += 1
            tag = f"({self.counters['equation']})"
            self.current = str(self.counters["equation"])
        for label in labels:
            self.labels[label.strip()] = self.current
        lines = []
        for piece in _split_top(body, re.compile(r"\\\\")):
            text = re.sub(r"[ \t\r\n]+", " ", self.inline(piece, math=True)).replace(NBSP, " ").strip()
            if text:
                lines.append("    " + text)
        if not lines:
            return ""
        if tag:
            pad = width - len(lines[-1]) - len(tag)
            lines[-1] = lines[-1] + " " * max(pad, 4) + tag
        return "\n".join(lines)

    def items(self, name: str, body: str, width: int) -> str:
        # Split on top-level \item, skipping nested environments and groups.
        pieces, starts = [], []
        index = 0
        while index < len(body):
            if body.startswith("\\begin", index):
                env, after = read_arg(body, index + len("\\begin"))
                _, index = environment_end(body, after, (env or "").strip())
                continue
            if body[index] == "{":
                index = group_end(body, index)
                continue
            word = _CONTROL_WORD.match(body, index)
            if word and word.group(1) == "item":
                starts.append((index, word.end()))
                index = word.end()
                continue
            index += 2 if body[index] == "\\" else 1
        for number, (start, after) in enumerate(starts):
            stop = starts[number + 1][0] if number + 1 < len(starts) else len(body)
            label, content_start = read_opt(body, after)
            pieces.append((label, body[content_start:stop]))
        out = []
        for number, (label, content) in enumerate(pieces, start=1):
            marker = LIST_ENVIRONMENTS[name]
            if label is not None:
                bullet = _plain(self.inline(label))
            else:
                bullet = f"{number}." if marker == "#" else marker
            prefix = f"  {bullet} " if bullet else "  "
            inner = self.blocks(content, max(width - len(prefix), 20))
            text = "\n\n".join(inner)
            indented = "\n".join((prefix if i == 0 else " " * len(prefix)) + line if line else ""
                                 for i, line in enumerate(text.split("\n")))
            out.append(indented)
        return "\n".join(out)


TEXT_NOTICE = ("[Plain-text rendering of the paper, GENERATED by paper/flatten.py from "
               "paper/main.tex and the files it inputs; do not edit. Every number is the "
               "value its paper/numbers.tex macro prints (written from results/), and "
               "figures are referenced by file.]")


def drop_comments(text: str) -> str:
    """``text`` without comments, as TeX reads it. An unescaped ``%`` removes
    the rest of its line *and* the line end, so a comment-only line inside a
    paragraph adds nothing (blanking it, as ``check.strip_comments`` does,
    would turn it into a paragraph break) and ``word%`` joins the next line.
    Leading spaces of a line are skipped, and an empty line stays a ``\\par``."""
    masked = _check().strip_comments(text)  # same length; comments blanked
    out: list[str] = []
    for raw, line in zip(text.split("\n"), masked.split("\n")):
        commented = raw != line
        if commented:
            line = line[:next(i for i, (a, b) in enumerate(zip(raw, line)) if a != b)]
        content = line.lstrip(" \t")
        if not content:
            out.append("" if commented else "\n\n")
        else:
            out.append(content + ("" if commented else "\n"))
    return "".join(out)


def render_text(flat: Flat, paper_dir: Path | str = PAPER_DIR) -> str:
    paper_dir = Path(paper_dir)
    masked = drop_comments(flat.source)
    macros, stripped = collect_macros(masked)
    begin = re.search(r"\\begin\s*\{document\}", stripped)
    end = re.search(r"\\end\s*\{document\}", stripped)
    body = stripped[begin.end() if begin else 0:end.start() if end else len(stripped)]
    body = expand(body, macros)
    labels: dict[str, str] = {}
    for _ in range(2):  # the first pass only collects the labels
        renderer = TextRenderer(paper_dir, flat.bib, labels)
        blocks = renderer.blocks(body)
        labels = renderer.labels
    parts = [_wrap(TEXT_NOTICE, WIDTH)] + blocks
    text = "\n\n".join(p.strip("\n") for p in parts if p.strip())
    lines = [line.replace(NBSP, " ").rstrip() for line in text.split("\n")]
    return "\n".join(lines).rstrip("\n") + "\n"


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def render(paper_dir: Path | str = PAPER_DIR) -> dict[str, str]:
    """{file name: contents} for both outputs, without writing anything."""
    flat = build_flat(paper_dir)
    return {FLAT_TEX: flat_tex(flat), FULL_TEXT: render_text(flat, paper_dir)}


def generate(paper_dir: Path | str = PAPER_DIR) -> list[str]:
    """Write both files into ``paper_dir``; returns one summary line each."""
    paper_dir = Path(paper_dir)
    flat = build_flat(paper_dir)
    outputs = {FLAT_TEX: flat_tex(flat), FULL_TEXT: render_text(flat, paper_dir)}
    for name, text in outputs.items():
        (paper_dir / name).write_text(text, encoding="utf-8")
    size = {name: len(text.encode("utf-8")) for name, text in outputs.items()}
    return [f"{FLAT_TEX}: {size[FLAT_TEX]:,} bytes, {flat.kept} of {flat.defined} number "
            f"macros kept, {len(flat.bib.order)} references inlined",
            f"{FULL_TEXT}: {size[FULL_TEXT]:,} bytes, "
            f"{outputs[FULL_TEXT].count(chr(10)):,} lines"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--paper-dir", type=Path, default=PAPER_DIR,
                        help="directory holding main.tex (default: this script's directory)")
    parser.add_argument("--check", action="store_true",
                        help="write nothing; exit 1 if a committed file differs from a fresh render")
    args = parser.parse_args(argv)
    if args.check:
        stale = [name for name, text in render(args.paper_dir).items()
                 if not (args.paper_dir / name).is_file()
                 or (args.paper_dir / name).read_text(encoding="utf-8") != text]
        for name in stale:
            print(f"flatten.py: {name} is stale; rerun `python3 paper/flatten.py`")
        return 1 if stale else 0
    for line in generate(args.paper_dir):
        print(f"flatten.py: wrote {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
