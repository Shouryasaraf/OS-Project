"""paper/check.py, the structural lint that stands in for a LaTeX compile.

Stdlib only, so it runs under system python3 too. These tests pin what the
lint catches (missing cite keys and inputs, unbalanced braces, bare
underscores, hand-typed numbers), what it lets through (versions, section
and decision numbers, lengths, comments), and that the committed skeleton
and bibliography pass it.
"""

import contextlib
import importlib.util
import io
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"

_spec = importlib.util.spec_from_file_location("paper_check", PAPER / "check.py")
check = importlib.util.module_from_spec(_spec)
# paper/ is not a package; register the module before running it, as
# dataclasses look their defining module up in sys.modules.
sys.modules[_spec.name] = check
_spec.loader.exec_module(check)

MINIMAL_BIB = """@article{known2020,
  author = {A. Author},
  title  = {A Title},
  journal = {J},
  year   = {2020}
}
"""


def _main_tex(body: str, preamble: str = "") -> str:
    return (
        "\\documentclass[conference]{IEEEtran}\n" + preamble +
        "\\begin{document}\n" + body + "\n\\bibliographystyle{IEEEtran}\n"
        "\\bibliography{refs}\n\\end{document}\n"
    )


class _PaperDir:
    """A throwaway paper directory holding main.tex, refs.bib and sections/."""

    def __init__(self, main: str, sections: dict[str, str] | None = None, bib: str = MINIMAL_BIB):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name)
        (self.path / "main.tex").write_text(main, encoding="utf-8")
        (self.path / "refs.bib").write_text(bib, encoding="utf-8")
        (self.path / "sections").mkdir()
        for name, text in (sections or {}).items():
            (self.path / "sections" / name).write_text(text, encoding="utf-8")

    def __enter__(self):
        return self.path

    def __exit__(self, *exc):
        self._tmp.cleanup()


def _messages(report, level="error"):
    return [f.message for f in report.findings if f.level == level]


class CommittedPaperTests(unittest.TestCase):
    def test_skeleton_passes_in_default_mode(self):
        report = check.run_checks(PAPER)
        self.assertEqual(report.errors, [], "\n".join(map(str, report.errors)))
        self.assertIn("sections/disk_results.tex", report.files)

    def test_every_section_file_is_reached_from_main(self):
        report = check.run_checks(PAPER)
        reached = set(report.files)
        for path in (PAPER / "sections").glob("*.tex"):
            self.assertIn(path.relative_to(PAPER).as_posix(), reached)

    def test_skeleton_is_not_yet_submittable(self):
        # --final rejects any remaining \todo; the skeleton is all TODOs.
        report = check.run_checks(PAPER, final=True)
        self.assertTrue(any(m == "\\todo remains" for m in _messages(report)))

    def test_bibliography_holds_the_verified_entries(self):
        entries, problems = check.parse_bib((PAPER / "refs.bib").read_text(encoding="utf-8"))
        self.assertEqual(problems, [])
        keys = {entry.key: entry for entry in entries}
        self.assertEqual(len(keys), len(entries), "duplicate keys")
        # 47 from the literature review plus the 8 WP7 additions.
        self.assertEqual(len(entries), 55)
        for key in (
            "wu2007readahead", "linux_mmapi_readahead", "gill2005sarc", "gill2007amp",
            "patterson1995informed", "cao1995integrated", "chakraborttii2020learning",
            "hashemi2018learning", "wong2024baleen", "savva2026adaptive",
            "ramadhan2026capsule", "akgun2021kml", "garg2024crossprefetch",
            "belady1966study", "mattson1970evaluation", "megiddo2003arc",
            "padmanabhan1996using", "kroeger1997exploring", "fan1999web",
            "hidasi2016session", "kang2018sasrec",
            "paszke2019pytorch", "man_posix_fadvise", "man_readahead", "man_cachestat",
            "man_readv", "man_open", "linux_cgroup_v2", "linux_mglru",
        ):
            self.assertIn(key, keys)

    def test_review1_corrections_are_in_the_bibliography(self):
        entries, _ = check.parse_bib((PAPER / "refs.bib").read_text(encoding="utf-8"))
        keys = {entry.key: entry.fields for entry in entries}
        capsule = keys["ramadhan2026capsule"]
        self.assertEqual(capsule["doi"], "10.1145/3820171.3806897")
        self.assertIn("Spatio-Temporal Locality", capsule["title"])
        self.assertEqual((capsule["volume"], capsule["number"], capsule["pages"]), ("54", "1", "76--78"))
        savva = keys["savva2026adaptive"]
        self.assertEqual((savva["journal"], savva["volume"], savva["number"], savva["pages"]),
                         ("Big Data and Cognitive Computing", "10", "1", "28"))
        self.assertIn("HotStorage '21", keys["akgun2021kml"]["booktitle"])
        ecml = keys["chakraborttii2020learning"]
        self.assertEqual((ecml["volume"], ecml["year"]), ("12460", "2021"))
        self.assertIn("Workshop abstract", keys["chakraborttii2022deep"]["note"])

    def test_cli_exit_status(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(check.main(["--paper-dir", str(PAPER), "--quiet"]), 0)


class HandNumberLintTests(unittest.TestCase):
    def assertFlagged(self, text, figure):
        self.assertIn(figure, [found for _, found in check.hand_numbers(text)], text)

    def assertClean(self, text):
        self.assertEqual(check.hand_numbers(text), [], text)

    def test_result_figures_are_flagged(self):
        self.assertFlagged("a hit ratio of 41.3 on hm_0", "41.3")
        self.assertFlagged("it wins by 12\\% overall", "12\\%")
        self.assertFlagged("a gain of +4.10~pp", "+4.10~pp")
        self.assertFlagged("about 640x faster", "640x")
        self.assertFlagged("about $3.5\\times$ the I/O", "3.5\\times")
        self.assertFlagged("worse on 52/56 traces", "52/56")
        self.assertFlagged("2,756,101 events", "2,756,101")
        self.assertFlagged("$0.25$ of requests", "0.25")

    def test_line_numbers_are_reported(self):
        self.assertEqual(check.hand_numbers("first line\nsecond 0.5 here\n"), [(2, "0.5")])

    def test_non_results_are_allowed(self):
        for text in (
            "Linux 6.14 with the multi-gen LRU",
            "readahead(2) appeared in 2.4.13",
            "see Section 2.3 and \\S 4.1",
            "decisions D10.5 and D14.1, work package WP7",
            "\\includegraphics[width=0.9\\columnwidth]{figures/f2_depth}",
            "\\hspace{0.5em} a gap",
            "% a comment that says 41.3\\% is ignored",
            "the macro \\diskAdaptiveVsSeq{} carries the number",
            "a Seagate ST10000NM017B and an RTX PRO 6000",
            "published in 2026, with 12 sections and 8 tables",
            "\\cite{wu2007readahead} and \\label{sec:x1.2}",
            "results/disk/table\\_main.csv",
        ):
            self.assertClean(text)

    def test_escaped_percent_does_not_start_a_comment(self):
        # "5\%" is a percentage, not a comment; the text after it is still prose.
        self.assertEqual([f for _, f in check.hand_numbers("5\\% and 0.7 more")], ["5\\%", "0.7"])

    def test_plain_text_mode_for_markdown(self):
        found = [f for _, f in check.hand_numbers("41.3% and 2x and 1.5", latex=False)]
        self.assertEqual(found, ["41.3%", "2x", "1.5"])
        self.assertEqual(check.hand_numbers("Linux 6.14, D10.5", latex=False), [])


class StructureTests(unittest.TestCase):
    def test_missing_cite_key_is_an_error(self):
        main = _main_tex("\\input{sections/a}")
        with _PaperDir(main, {"a.tex": "See~\\cite{known2020, missing2021}.\n"}) as paper:
            report = check.run_checks(paper)
        self.assertEqual(_messages(report), ["\\cite key 'missing2021' is not in the bibliography"])

    def test_cite_with_optional_argument_and_nocite(self):
        main = _main_tex("\\input{sections/a}\\nocite{*}")
        with _PaperDir(main, {"a.tex": "See~\\cite[p.~3]{known2020}.\n"}) as paper:
            self.assertTrue(check.run_checks(paper).ok)

    def test_hand_number_in_a_section_is_an_error_but_not_in_main(self):
        main = _main_tex("Author note 24.5\n\\input{sections/a}")
        with _PaperDir(main, {"a.tex": "We gain 4.1\\% here.\n"}) as paper:
            report = check.run_checks(paper)
        errors = report.errors
        self.assertEqual(len(errors), 1)
        self.assertEqual((errors[0].path, errors[0].line), ("sections/a.tex", 1))
        self.assertIn("4.1\\%", errors[0].message)

    def test_missing_plain_input_is_an_error(self):
        with _PaperDir(_main_tex("\\input{sections/absent}")) as paper:
            self.assertIn("\\input 'sections/absent.tex' does not exist", _messages(check.run_checks(paper)))

    def test_missing_generated_input_is_a_warning_until_strict(self):
        main = _main_tex("\\input{tables/t2_main}\\includegraphics{figures/f3}",
                         preamble="\\input{numbers}\n")
        with _PaperDir(main) as paper:
            default = check.run_checks(paper)
            strict = check.run_checks(paper, strict=True)
            (paper / "numbers.tex").write_text("\\newcommand{\\x}{1}\n", encoding="utf-8")
            (paper / "tables").mkdir()
            (paper / "tables" / "t2_main.tex").write_text("\\begin{tabular}{c}\n0.25\\\\\n\\end{tabular}\n",
                                                         encoding="utf-8")
            (paper / "figures").mkdir()
            (paper / "figures" / "f3.pdf").write_bytes(b"%PDF-1.4\n")
            generated = check.run_checks(paper, strict=True)
        self.assertTrue(default.ok)
        self.assertEqual(len(default.warnings), 3)
        self.assertEqual(len(strict.errors), 3)
        # Once generated they pass, and a generated table may carry numbers.
        self.assertTrue(generated.ok, "\n".join(map(str, generated.errors)))
        self.assertIn("tables/t2_main.tex", generated.files)

    def test_paths_outside_the_paper_directory_are_rejected(self):
        with _PaperDir(_main_tex("\\includegraphics{../results/plot.pdf}")) as paper:
            messages = _messages(check.run_checks(paper))
        self.assertTrue(any("outside the paper directory" in m for m in messages), messages)

    def test_unbalanced_braces_and_environments(self):
        main = _main_tex("\\input{sections/a}")
        text = "\\textbf{open\n\\begin{figure}\nx\n\\end{table}\n"
        with _PaperDir(main, {"a.tex": text}) as paper:
            messages = _messages(check.run_checks(paper))
        self.assertIn("'{' is never closed", messages)
        self.assertIn("\\end{table} closes \\begin{figure} from line 2", messages)

    def test_escaped_braces_do_not_count(self):
        main = _main_tex("\\input{sections/a}")
        with _PaperDir(main, {"a.tex": "a set \\{x\\} and \\textbf{y}\n"}) as paper:
            self.assertTrue(check.run_checks(paper).ok)

    def test_bare_underscore_outside_math_is_an_error(self):
        main = _main_tex("\\input{sections/a}")
        text = ("ok: linux\\_ra, $x_i$, \\label{s:a_b}, \\url{http://a_b}\n"
                "bad: read_ahead\n")
        with _PaperDir(main, {"a.tex": text}) as paper:
            errors = check.run_checks(paper).errors
        self.assertEqual([(e.line, e.message) for e in errors],
                         [(2, "bare '_' outside math; write '\\_'")])

    def test_bibliography_problems(self):
        bib = MINIMAL_BIB + "@misc{Known2020,\n  title = {Dup}\n}\n@misc{notitle,\n  year = {2020}\n}\n"
        with _PaperDir(_main_tex("\\cite{known2020}"), bib=bib) as paper:
            messages = _messages(check.run_checks(paper))
        self.assertTrue(any(m.startswith("duplicate key 'Known2020'") for m in messages), messages)
        self.assertIn("'notitle' has no title", messages)

    def test_at_sign_inside_a_field_is_not_an_entry(self):
        bib = '@misc{site,\n  title = {T},\n  howpublished = {\\url{mailto:a@b.org}}\n}\n'
        entries, problems = check.parse_bib(bib)
        self.assertEqual([e.key for e in entries], ["site"])
        self.assertEqual(problems, [])

    def test_undefined_ref_is_a_warning_until_strict(self):
        main = _main_tex("\\section{A}\\label{sec:a} see \\ref{sec:a} and \\ref{tab:t2}")
        with _PaperDir(main) as paper:
            self.assertEqual(len(check.run_checks(paper).warnings), 1)
            self.assertEqual(_messages(check.run_checks(paper, strict=True)),
                             ["\\ref{tab:t2} has no \\label"])

    def test_todo_fails_only_in_final_mode(self):
        main = _main_tex("\\input{sections/a}", preamble="\\newcommand{\\todo}[1]{#1}\n")
        with _PaperDir(main, {"a.tex": "\\todo{write this}\n"}) as paper:
            self.assertTrue(check.run_checks(paper).ok)
            self.assertEqual(_messages(check.run_checks(paper, final=True)), ["\\todo remains"])

    def test_cli_returns_one_on_errors(self):
        with _PaperDir(_main_tex("\\cite{nope}")) as paper, contextlib.redirect_stdout(io.StringIO()) as out:
            status = check.main(["--paper-dir", str(paper)])
        self.assertEqual(status, 1)
        self.assertIn("ERROR main.tex:", out.getvalue())


if __name__ == "__main__":
    unittest.main()
