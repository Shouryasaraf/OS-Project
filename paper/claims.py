"""The claims ledger: every number and verdict the paper prints, traced to results/.

    PYTHONPATH=src python paper/claims.py [--results DIR] [--paper-dir DIR] [--out DIR]

``reproduce.py paper`` runs this after the three generators. It rebuilds the
macros of ``numbers.py`` with tracing on (``genlib.TRACE``), regenerates every
table of ``tables/make_tables.py`` with tracing on, and writes, for what the
paper and the slides actually use:

- ``results/claims/paper_claims.csv`` (schema ``claims/paper_claims.csv``,
  through ``prefetch_common.schemas.write_rows``): one row per number macro
  used in ``main.tex``, ``sections/*.tex``, the generated tables or
  ``presentation/slides.tex``; one row per generated table (its files and
  rows); one row per traced table cell. Each row names the results file, the
  row selector (key=value of the identifying columns) or the rows a derived
  value was aggregated over, the column(s), the raw value(s) and the
  formatter that printed them. A ``\\dec`` row adds its rule, verdict,
  intervals and the ``PREREGISTERED.md`` the rule comes from;
- ``results/claims/CLAIMS.md``: the same, readable, grouped by paper section
  and sentence;
- ``results/claims/provenance.json``, rewritten only when the ledger changed
  (so regenerating from unchanged results leaves the tree clean).

It is deterministic: the ledger is a pure function of the committed
``results/`` and ``paper/`` sources, which ``tests/test_claims.py`` checks by
regenerating it. It refuses to run when ``numbers.tex`` or a table on disk is
stale, and ``numbers.Macros`` refuses a macro it cannot trace.

Stdlib only.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.curdir) != _HERE]

import argparse  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
from pathlib import Path  # noqa: E402


def _load(name: str, relative: str):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, Path(_HERE) / relative)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            del sys.modules[name]
            raise
    return sys.modules[name]


g = _load("paper_genlib", "genlib.py")
from prefetch_common import provenance, schemas  # noqa: E402

SCHEMA = "claims/paper_claims.csv"
LEDGER = "paper_claims.csv"
READABLE = "CLAIMS.md"
DEFAULT_OUT = g.DEFAULT_RESULTS / "claims"
SLIDES = g.REPO_ROOT / "presentation" / "slides.tex"
_COMMENT = re.compile(r"(?<!\\)%.*")
_CONTROL = re.compile(r"\\([A-Za-z]+)")
_INPUT = re.compile(r"\\(?:input|include)\s*\{([^{}]+)\}")
_PNUM = re.compile(r"\\pnum\{([A-Za-z]+)\}")


class StaleOutput(RuntimeError):
    """A generated paper file differs from what the results give today."""


# ---------------------------------------------------------------------------
# What the paper uses
# ---------------------------------------------------------------------------

def strip_comments(text: str) -> str:
    return "\n".join(_COMMENT.sub("", line) for line in text.split("\n"))


def paper_files(paper_dir: Path) -> list[str]:
    """``main.tex`` and every file it inputs, in reading order (tables included)."""
    order, queue = [], ["main.tex"]
    while queue:
        relative = queue.pop(0)
        if relative in order or not (paper_dir / relative).is_file():
            continue
        order.append(relative)
        text = strip_comments((paper_dir / relative).read_text(encoding="utf-8"))
        found = []
        for match in _INPUT.finditer(text):
            target = match.group(1).strip()
            found.append(target if target.endswith(".tex") else target + ".tex")
        queue = found + queue
    return [f for f in order if f != "numbers.tex"]


def uses(paper_dir: Path, defined, slides: Path | None) -> dict[str, list[str]]:
    """``{file: [macro, ...]}`` in order of first use, for defined macros only."""
    files = [(f, paper_dir / f) for f in paper_files(paper_dir)]
    if slides is not None and slides.is_file():
        files.append((slides.relative_to(g.REPO_ROOT).as_posix(), slides))
    found = {}
    for name, path in files:
        text = strip_comments(path.read_text(encoding="utf-8"))
        names = [m.group(1) for m in _CONTROL.finditer(text)]
        names += [m.group(1) for m in _PNUM.finditer(text)]
        ordered = list(dict.fromkeys(n for n in names if n in defined))
        if ordered:
            found[name] = ordered
    return found


def sentences(text: str) -> list[str]:
    """Rough sentences of LaTeX prose (enough to give a claim its context)."""
    text = strip_comments(text)
    out = []
    for paragraph in re.split(r"\n\s*\n", text):
        flat = " ".join(paragraph.split())
        out += [s for s in re.split(r"(?<=[.?!])\s+(?=[A-Z\\])", flat) if s]
    return out


# ---------------------------------------------------------------------------
# Selectors
# ---------------------------------------------------------------------------

def _plain(row):
    return {k: dict.__getitem__(row, k) for k in row} if isinstance(row, dict) else row


def key_columns(rows: list[dict]) -> tuple[str, ...] | None:
    """The fewest identifying columns (str/int/bool/None values, schema order)
    that make every row unique, or None when no such set exists."""
    if len(rows) <= 1:
        return ()
    plain = [_plain(r) for r in rows]
    candidates = [k for k in plain[0]
                  if all(v is None or (isinstance(v, (str, int, bool)) and not isinstance(v, float))
                         for v in (r.get(k) for r in plain))]
    chosen = []
    for key in candidates:
        chosen.append(key)
        if len({tuple(r.get(k) for k in chosen) for r in plain}) == len(plain):
            break
    else:
        return None
    for key in list(reversed(chosen)):
        trial = [k for k in chosen if k != key]
        if trial and len({tuple(r.get(k) for k in trial) for r in plain}) == len(plain):
            chosen = trial
    return tuple(chosen)


class Selectors:
    def __init__(self, results):
        self.results = results
        self._keys = {}

    def rows(self, relative: str) -> list[dict]:
        if relative.endswith("/provenance.json"):
            return [self.results.provenance(relative.split("/")[0])]
        if relative in self.results._cache:
            return self.results._cache[relative] or []
        if schemas_has(relative):
            return self.results.rows(relative) or []
        return [self.results.json_object(relative)]

    def selector(self, relative: str, index: int) -> dict:
        if relative not in self._keys:
            self._keys[relative] = key_columns(self.rows(relative))
        keys = self._keys[relative]
        if keys is None:
            return {"_row": index}
        row = _plain(self.rows(relative)[index])
        return {k: row[k] for k in keys}


def schemas_has(relative: str) -> bool:
    try:
        schemas.schema_for(relative)
        return True
    except (KeyError, schemas.SchemaError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Ledger rows
# ---------------------------------------------------------------------------

def _fmt_calls(calls) -> list:
    return [[c[0], c[1], c[2], c[3]] for c in calls]


def _interval_note(meta: dict) -> str:
    parts = [f"scope: {meta['scope']}", f"comparison: {meta['comparison']}"]
    for label, estimate, lo, hi, statistic in meta.get("intervals", ()):
        parts.append(f"{label or 'paired'} interval ({statistic}): estimate {estimate!r}, "
                     f"[{lo!r}, {hi!r}]")
    if meta.get("note"):
        parts.append(meta["note"])
    return "; ".join(parts)


def macro_row(name: str, printed: str, trace: dict, used_in: list[str],
              select: Selectors) -> dict:
    kind = trace["kind"]
    row = {"claim_id": name, "kind": kind, "used_in": ";".join(used_in), "printed": printed,
           "source_file": trace["source"], "selector": None, "rows": None, "columns": None,
           "raw": trace.get("raw", []), "formatter": [c for c in trace.get("calls", [])],
           "rule": None, "verdict": None, "preregistered": None, "note": None}
    notes = []
    if trace.get("alias_of"):
        notes.append(f"alias of \\{trace['alias_of']}")
    if "file" in trace:
        row["source_file"] = trace["file"]
        row["selector"] = select.selector(trace["file"], trace["row"])
        row["rows"] = 1
        row["columns"] = ";".join(trace["columns"])
    elif trace.get("files"):
        row["source_file"] = ";".join(trace["files"])
        row["rows"] = trace.get("rows")
        row["columns"] = ";".join(trace.get("columns") or ()) or None
        if kind == "decision_count":
            notes.append("count of pre-registered verdicts (paper/decisions.py) read from "
                         + ", ".join(trace["files"]))
        else:
            notes.append(f"derived (count, sum, mean or selection) over {trace.get('rows')} "
                         f"row(s) read from {', '.join(trace['files'])}")
    meta = trace.get("meta") or {}
    if meta.get("preregistered"):
        row["preregistered"] = meta["preregistered"]
    if meta.get("rule"):
        row["rule"], row["verdict"] = meta["rule"], meta["verdict"]
        notes.append(_interval_note(meta))
    if kind == "constant":
        notes.append("code constant of the cost model (an assumption, not a result)")
    if kind == "missing":
        notes.append("MISSING: no results row")
    row["note"] = "; ".join(notes) or None
    return row


def table_rows(name: str, calls, files, rows_read, columns, select: Selectors) -> list[dict]:
    claim = f"tables/{name}.tex"
    out = [{"claim_id": claim, "kind": "table", "used_in": claim, "printed": "",
            "source_file": ";".join(sorted(files)) or "none", "selector": None,
            "rows": len(rows_read), "columns": ";".join(sorted(f"{f}:{c}" for f, c in columns))
            or None, "raw": [], "formatter": [], "rule": None, "verdict": None,
            "preregistered": None,
            "note": f"generated table; {len(rows_read)} results row(s) read"}]
    cell = 0
    for function, raws, other, kwargs, printed, where in calls:
        if not any(g.finite(v) for v in raws):
            continue  # labels and dashes, not numbers
        cell += 1
        row = {"claim_id": f"{claim}#{cell}", "used_in": claim, "printed": printed,
               "raw": list(raws), "formatter": [[function, g_jsonable(list(other)),
                                                 g_jsonable(kwargs), printed]],
               "rule": None, "verdict": None, "preregistered": None, "note": None,
               "selector": None, "rows": None, "columns": None}
        if all(where) and len({(w[0], w[1]) for w in where}) == 1:
            row.update(kind="table_cell", source_file=where[0][0],
                       selector=select.selector(where[0][0], where[0][1]), rows=1,
                       columns=";".join(g.trace_column(w) for w in where))
        else:
            used = sorted({w[0] for w in where if w}) or sorted(files)
            row.update(kind="table_derived", source_file=";".join(used) or "none",
                       note="derived in the table generator from the table's input rows")
        out.append(row)
    return out


def g_jsonable(value):
    return sys.modules["paper_numbers"]._jsonable(value)


def build(results, paper_dir: Path, *, slides: Path | None = SLIDES,
          check_current: bool = True):
    """``(ledger rows, readable text)`` for the paper in ``paper_dir``."""
    numbers = _load("paper_numbers", "numbers.py")
    tables = _load("paper_make_tables", "tables/make_tables.py")
    macros, _ = numbers.build(results, trace=True)
    if check_current:
        current = numbers.Macros.render(macros, _.missing, results.display_root)
        on_disk = (paper_dir / numbers.OUTPUT)
        if not on_disk.is_file() or on_disk.read_text(encoding="utf-8") != current:
            raise StaleOutput(f"{on_disk} is stale: run the generators first")
    select = Selectors(results)
    used = uses(paper_dir, macros.values, slides)
    where_used = {}
    for path, names in used.items():
        for name in names:
            where_used.setdefault(name, []).append(path)
    rows = []
    for name in sorted(where_used):
        printed = macros.values[name]
        if macros.fixture and printed != numbers.MISSING:
            printed = r"\fixturemark{" + printed + "}"  # as numbers.tex prints it
        rows.append(macro_row(name, printed, macros.traces[name], where_used[name], select))
    for function in tables.TABLES:
        g.TRACE.reset()
        g.TRACE.active = True
        try:
            text = function(results, results.output(f"tables/{function.__name__}.tex"))
        finally:
            g.TRACE.active = False
        calls, _recent, rows_read, columns, files = g.TRACE.take()
        if check_current:
            target = paper_dir / "tables" / f"{function.__name__}.tex"
            if not target.is_file() or target.read_text(encoding="utf-8") != text:
                raise StaleOutput(f"{target} is stale: run the generators first")
        rows += table_rows(function.__name__, calls, files, rows_read, columns, select)
    g.TRACE.reset()
    return rows, readable(rows, used, paper_dir, slides, macros)


# ---------------------------------------------------------------------------
# CLAIMS.md
# ---------------------------------------------------------------------------

def plain(text: str) -> str:
    """LaTeX value text -> readable Markdown."""
    for old, new in ((r"\,\%", " %"), (r"\%", "%"), (r"\,pp", " pp"), (r"\times", "x"),
                     (r"\,$\mu$s", " us"), (r"\,ms", " ms"), ("{,}", ","), ("~", " "),
                     (r"\infty", "inf"), (r"\_", "_"), ("$", ""), (r"\,", " "), ("|", "/")):
        text = text.replace(old, new)
    return text


def _cell(value) -> str:
    if value is None or value == "":
        return ""
    if not isinstance(value, str):
        value = json.dumps(value, sort_keys=True)
    return value.replace("|", "/").replace("\n", " ")


def readable(rows, used, paper_dir: Path, slides, macros) -> str:
    by_id = {r["claim_id"]: r for r in rows}
    kinds = {}
    for row in rows:
        kinds[row["kind"]] = kinds.get(row["kind"], 0) + 1
    lines = [
        "# Claims ledger",
        "",
        "GENERATED by `paper/claims.py` (`python reproduce.py paper`) from the committed",
        "`results/`; do not edit. Every number and verdict the paper and the slides print",
        "is listed with the results file, the row (key=value of its identifying",
        "columns), the column and the raw value it was printed from. The",
        "machine-readable ledger is `paper_claims.csv` beside this file;",
        "`tests/test_claims.py` re-reads every value from its results file and",
        "re-formats it.",
        "",
        "Kinds: `value` (one results row and column), `derived` (a count, sum, mean or",
        "selection over the rows named), `decision` (a pre-registered verdict or one of",
        "its intervals), `decision_count` (a count of verdicts), `constant` (a cost-model",
        "assumption in code, not a result), `table`/`table_cell`/`table_derived` (the",
        "generated tables).",
        "",
        "Rows: " + ", ".join(f"{k} {v}" for k, v in sorted(kinds.items())) + ".",
        "",
    ]
    for path, names in used.items():
        if path.startswith("tables/"):
            continue
        lines += [f"## {path}", ""]
        source = (paper_dir / path) if not path.startswith("presentation/") else slides
        remaining = list(names)
        for sentence in sentences(source.read_text(encoding="utf-8")):
            present = [n for n in dict.fromkeys(_CONTROL.findall(sentence)
                                                + _PNUM.findall(sentence)) if n in by_id]
            if not present:
                continue
            shown = sentence
            for name in sorted(present, key=len, reverse=True):
                shown = re.sub(r"\\pnum\{" + name + r"\}|\\" + name + r"(?![A-Za-z])",
                               lambda _m, n=name: "**" + plain(by_id[n]["printed"]) + "**",
                               shown)
            if len(shown) > 400:
                shown = shown[:397] + "..."
            lines += [f"> {_cell(shown)}", "",
                      "| macro | printed | kind | source | row | column | raw |",
                      "|---|---|---|---|---|---|---|"]
            for name in present:
                r = by_id[name]
                remaining = [n for n in remaining if n != name]
                row_text = (_cell(r["selector"]) if r["selector"] is not None else
                            f"{r['rows']} row(s)" if r["rows"] else "")
                lines.append(f"| `\\{name}` | {_cell(plain(r['printed']))} | {r['kind']} | "
                             f"{_cell(r['source_file'])} | {row_text} | "
                             f"{_cell(r['columns'])} | {_cell(r['raw'])} |")
            lines.append("")
        if remaining:
            lines += ["Also used (outside a sentence, e.g. in a caption or a list):", "",
                      "| macro | printed | kind | source | row | column | raw |",
                      "|---|---|---|---|---|---|---|"]
            for name in remaining:
                r = by_id[name]
                row_text = (_cell(r["selector"]) if r["selector"] is not None else
                            f"{r['rows']} row(s)" if r["rows"] else "")
                lines.append(f"| `\\{name}` | {_cell(plain(r['printed']))} | {r['kind']} | "
                             f"{_cell(r['source_file'])} | {row_text} | "
                             f"{_cell(r['columns'])} | {_cell(r['raw'])} |")
            lines.append("")
    decisions = [r for r in rows if r["kind"] == "decision" and r["claim_id"].endswith("Verdict")]
    lines += ["## Pre-registered verdicts the paper states", "",
              "| macro | rule | verdict | pre-registration | intervals |", "|---|---|---|---|---|"]
    for r in decisions:
        lines.append(f"| `\\{r['claim_id']}` | {r['rule']} | {_cell(r['verdict'])} | "
                     f"{r['preregistered']} | {_cell(r['note'])} |")
    lines += ["", "## Generated tables", "",
              "| table | results files | rows read | traced cells | derived cells |",
              "|---|---|---|---|---|"]
    for r in rows:
        if r["kind"] != "table":
            continue
        cells = [x for x in rows if x["claim_id"].startswith(r["claim_id"] + "#")]
        lines.append(f"| {r['claim_id']} | {_cell(r['source_file'])} | {r['rows']} | "
                     f"{sum(x['kind'] == 'table_cell' for x in cells)} | "
                     f"{sum(x['kind'] == 'table_derived' for x in cells)} |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def write(rows, text: str, out: Path, *, command=("reproduce.py", "paper"),
          inputs=()) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    ledger = out / LEDGER
    schemas.write_rows(SCHEMA, rows, ledger)
    (out / READABLE).write_text(text, encoding="utf-8")
    written = [ledger, out / READABLE]
    # provenance only when the ledger changed: regenerating from unchanged
    # results must leave the tree clean.
    current = provenance.hash_outputs(out)
    try:
        recorded = provenance.read_provenance(out)["outputs"]
    except (OSError, ValueError):
        recorded = None
    if recorded != current:
        provenance.write_provenance(out, list(command), inputs, study="claims", gpu=False,
                                    notes="regenerated by reproduce.py paper from the "
                                          "committed results/ (deterministic)")
    written.append(out / provenance.FILENAME)
    return written


def generate(results, paper_dir: Path = g.PAPER_DIR, out: Path | None = None) -> list[Path]:
    committed = Path(paper_dir).resolve() == g.PAPER_DIR.resolve() and not results.fixture
    if out is None:
        # Only the committed paper over a real results tree writes
        # results/claims/; a fixture or scratch build keeps its ledger beside it.
        out = results.root / "claims" if committed else Path(paper_dir) / "claims"
    slides = SLIDES if committed else None
    rows, text = build(results, Path(paper_dir), slides=slides)
    inputs = sorted({results.root / f for r in rows for f in r["source_file"].split(";")
                     if (results.root / f).is_file()})
    return write(rows, text, Path(out), inputs=inputs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results", type=Path, default=g.DEFAULT_RESULTS)
    parser.add_argument("--paper-dir", type=Path, default=g.PAPER_DIR)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    written = generate(g.Results(args.results), args.paper_dir, args.out)
    print(f"claims.py: wrote {', '.join(str(p) for p in written)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
