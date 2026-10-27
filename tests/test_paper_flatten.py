"""paper/flatten.py: the paper as one LaTeX file and as one plain-text file.

The committed paper/main_flat.tex and paper/paper_full.txt must be exactly
what flatten.py writes from the committed paper sources today, as
test_paper_decisions.py pins numbers.tex and the tables to results/. Beyond
that the tests check that main_flat.tex stands alone (nothing left to input,
every macro it uses defined, every cite with a \\bibitem, every figure on
disk) and that paper_full.txt is clean text (no LaTeX left, every number the
prose quotes printed), and they pin the converter's pieces on small
synthetic papers.

Stdlib only, so it runs under system python3 too.
"""

import contextlib
import importlib.util
import io
import re
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
sys.path.insert(0, str(ROOT / "src"))


def _load(name, relative):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, PAPER / relative)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


check = _load("paper_check", "check.py")
flatten = _load("paper_flatten", "flatten.py")

# Backslash sequences paper_full.txt may contain. None today; a future source
# that needs one (a Windows path in a reference, say) is added here on review.
ALLOWED_BACKSLASH: tuple[str, ...] = ()

BIB = """@inproceedings{alpha2001,
  author = {Ann B. Smith and Jean-Luc van der Berg and {\\'E}mile Zola},
  title = {Alpha {I/O} Paper},
  booktitle = {Proc. Alpha},
  pages = {1--10},
  publisher = {ACM},
  year = {2001},
  doi = {10.1/abc_def}
}
@article{beta2002,
  author = {Doe, John},
  title = {Beta},
  journal = {J. Beta},
  volume = {3},
  number = {4},
  pages = {28},
  year = {2002}
}
@misc{gamma2003,
  author = {{The Gamma Project}},
  title = {Gamma Manual},
  howpublished = {\\url{https://example.org/g_h}},
  note = {Accessed 2026-10-26},
  year = {2003}
}
@misc{delta2004,
  author = {A. One and B. Two and C. Three and D. Four and E. Five and F. Six and G. Seven and H. Eight and I. Nine and J. Ten and K. Eleven},
  title = {Delta},
  year = {2004}
}
@misc{epsilon2005,
  author = {Juncheng Yang and others},
  title = {Epsilon},
  year = {2005}
}
"""


def _main(body: str, preamble: str = "") -> str:
    return ("\\documentclass[conference]{IEEEtran}\n\\usepackage{url}\n" + preamble +
            "\\begin{document}\n" + body + "\n\\bibliographystyle{IEEEtran}\n"
            "\\bibliography{refs}\n\\end{document}\n")


class _Paper:
    """A throwaway paper directory: main.tex, refs.bib and any other files."""

    def __init__(self, main: str, files: dict[str, str] | None = None, bib: str = BIB):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name)
        (self.path / "main.tex").write_text(main, encoding="utf-8")
        (self.path / "refs.bib").write_text(bib, encoding="utf-8")
        for name, text in (files or {}).items():
            target = self.path / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")

    def __enter__(self):
        return self.path

    def __exit__(self, *exc):
        self._tmp.cleanup()


def _render(body: str, files=None, preamble: str = "") -> dict[str, str]:
    with _Paper(_main(body, preamble), files) as path:
        return flatten.render(path)


def _text(body: str, files=None, preamble: str = "") -> str:
    """The plain text without the generated notice."""
    text = _render(body, files, preamble)[flatten.FULL_TEXT]
    return text.split("]\n\n", 1)[1]


def _flat_text(body: str, files=None, preamble: str = "") -> str:
    """The plain text with every run of whitespace as one space."""
    return re.sub(r"\s+", " ", _text(body, files, preamble))


def _code(tex: str) -> str:
    return check.strip_comments(tex)


class CommittedFlatFilesTests(unittest.TestCase):
    """The committed single-file renderings of the real paper."""

    @classmethod
    def setUpClass(cls):
        cls.rendered = flatten.render(PAPER)
        cls.tex = cls.rendered[flatten.FLAT_TEX]
        cls.text = cls.rendered[flatten.FULL_TEXT]

    def test_committed_files_equal_a_fresh_render(self):
        for name in flatten.OUTPUTS:
            committed = (PAPER / name).read_text(encoding="utf-8")
            self.assertEqual(committed, self.rendered[name],
                             f"paper/{name} is stale: run `python reproduce.py paper` "
                             "(or `python3 paper/flatten.py`) and commit the result")

    def test_flat_tex_stands_alone(self):
        code = _code(self.tex)
        self.assertNotRegex(code, r"\\(input|include|IfFileExists)\b")
        self.assertNotRegex(code, r"\\bibliography(style)?\s*\{")
        self.assertIn("\\begin{thebibliography}", code)
        self.assertEqual(check.brace_problems(self.tex), [])
        self.assertEqual(check.environment_problems(self.tex), [])
        self.assertEqual(code.count("\\begin{document}"), 1)
        for name in ("sections/abstract.tex", "sections/conclusion.tex", "tables/t9_decisions.tex",
                     "numbers.tex"):
            self.assertIn(f"% >>> begin {name}\n", self.tex)

    def test_flat_tex_keeps_exactly_the_number_macros_it_uses(self):
        code = _code(self.tex)
        defined = set(re.findall(r"^\\newcommand\{\\([A-Za-z]+)\}", self.tex, re.M))
        used = set(check.NUMBER_MACRO.findall(code))
        self.assertEqual(used - defined, set())
        everything = set(re.findall(r"^\\newcommand\{\\([A-Za-z]+)\}",
                                    (PAPER / "numbers.tex").read_text(encoding="utf-8"), re.M))
        kept = defined & everything
        body = re.sub(r"^\\newcommand\{\\[A-Za-z]+\}", "", code, flags=re.M)
        self.assertEqual({n for n in kept if f"\\{n}" not in body}, set(),
                         "a kept numbers.tex macro is used nowhere")
        # TeX memory on Overleaf: a small fraction of numbers.tex survives.
        self.assertLess(len(kept), len(everything) // 10)
        self.assertLess(len(self.tex.encode("utf-8")), 500_000)

    def test_flat_tex_labels_references_and_citations_resolve(self):
        code = _code(self.tex)
        labels = re.findall(r"\\label\s*\{([^{}]+)\}", code)
        self.assertEqual(len(labels), len(set(labels)))
        refs = set(re.findall(r"\\(?:ref|eqref)\s*\{([^{}]+)\}", code))
        self.assertEqual(refs - set(labels), set())
        cited = {k.strip().lower() for group in re.findall(r"\\cite\{([^{}]*)\}", code)
                 for k in group.split(",")}
        items = [k.lower() for k in re.findall(r"\\bibitem\{([^{}]+)\}", code)]
        self.assertEqual(sorted(items), sorted(cited))
        self.assertNotIn("MISSING", self.tex)

    def test_flat_tex_figures_exist(self):
        for target in re.findall(r"\\includegraphics\s*(?:\[[^\]]*\])?\s*\{([^{}]+)\}",
                                 _code(self.tex)):
            resolved = check._graphics_target(PAPER, target)
            self.assertTrue((PAPER / resolved).is_file(), resolved)

    def test_text_has_no_latex_left(self):
        leftovers = [m.group(0) for m in re.finditer(r"\\[A-Za-z@]+|\\.", self.text)
                     if m.group(0) not in ALLOWED_BACKSLASH]
        self.assertEqual(leftovers, [])
        self.assertNotIn("$", self.text)
        self.assertEqual(self.text.count("{"), self.text.count("}"))
        self.assertNotRegex(self.text, r"\{[A-Za-z\\]|[A-Za-z]\}")
        self.assertNotIn("~", self.text)
        self.assertNotIn("??", self.text)  # \missingnumber or an unresolved \ref
        self.assertNotIn("[?]", self.text)  # an unknown cite key
        self.assertNotIn("\u00a0", self.text)
        self.assertNotIn("\u2028", self.text)
        self.assertTrue(all(line == line.rstrip() for line in self.text.split("\n")))

    def test_text_numbers_every_heading_in_order(self):
        sections = []
        for name in re.findall(r"\\input\{(sections/[^{}]+)\}",
                               _code((PAPER / "main.tex").read_text(encoding="utf-8"))):
            source = _code((PAPER / f"{name}.tex").read_text(encoding="utf-8"))
            sections += re.findall(r"\\section\{([^{}]+)\}", source)
        headings = re.findall(r"^((?:[IVXLC]+|Appendix [A-Z])\. .+)\n=+$", self.text, re.M)
        self.assertEqual(len(headings), len(sections))
        body = [h for h in headings if not h.startswith("Appendix")]
        self.assertEqual([h.split(".")[0] for h in body],
                         [flatten.roman(i) for i in range(1, len(body) + 1)])
        self.assertIn("IV-A. Disk simulator and policies", self.text)
        self.assertRegex(self.text, r"(?m)^Abstract\u2014", "abstract in IEEE form")
        self.assertIn("Index Terms\u2014", self.text)

    def test_text_title_authors_and_references(self):
        self.assertIn("Depth and Freshness, Not Learned Prediction", self.text)
        self.assertIn("Ashvik Mishra (24BAI0342)", self.text)
        self.assertIn("Shourya Saraf (24BDS0391)", self.text)
        references = self.text.split("\nReferences\n==========\n", 1)
        self.assertEqual(len(references), 2)
        listed = re.findall(r"^\[(\d+)\] ", references[1], re.M)
        items = re.findall(r"\\bibitem\{", self.tex)
        self.assertEqual([int(n) for n in listed], list(range(1, len(items) + 1)))
        cited = {int(n) for n in re.findall(r"\[(\d+)\]", references[0])}
        for first, last in re.findall(r"\[(\d+)\]\u2013\[(\d+)\]", references[0]):
            cited.update(range(int(first), int(last) + 1))
        self.assertEqual(cited, set(range(1, len(items) + 1)), "every reference is cited")

    def test_text_figures_point_at_files(self):
        figures = re.findall(r"\[Figure (\d+): ", self.text)
        self.assertEqual([int(n) for n in figures], list(range(1, len(figures) + 1)))
        flat = re.sub(r"\s+", " ", self.text)
        paths = re.findall(r"see (paper/figures/[\w.-]+\.pdf)\]", flat)
        self.assertEqual(len(paths), len(figures))
        for path in paths:
            self.assertTrue((ROOT / path).is_file(), path)

    def test_text_prints_every_number_the_prose_quotes(self):
        flat = re.sub(r"\s+", " ", self.text)
        macros, _ = flatten.collect_macros(_code(self.tex))
        renderer = flatten.TextRenderer(PAPER, flatten.Bibliography())
        prose = "".join(_code(p.read_text(encoding="utf-8"))
                        for p in sorted((PAPER / "sections").glob("*.tex")))
        names = sorted(set(check.NUMBER_MACRO.findall(prose)))
        self.assertGreater(len(names), 100)
        for name in names:
            value = re.sub(r"\s+", " ", renderer.inline(flatten.expand(f"\\{name}", macros)))
            value = value.replace("\u00a0", " ").strip()
            self.assertIn(value, flat, f"\\{name} = {value!r} is not in paper_full.txt")


class InlineTests(unittest.TestCase):
    def test_inputs_inline_recursively_but_not_in_comments(self):
        files = {"sections/a.tex": "A-text \\input{sections/b}\n",
                 "sections/b.tex": "B-text\n% \\input{sections/missing}\n"}
        tex = _render("\\input{sections/a}\n% \\input{sections/nothere}", files)[flatten.FLAT_TEX]
        self.assertIn("A-text", tex)
        self.assertIn("B-text", tex)
        self.assertIn("% >>> begin sections/b.tex", tex)
        self.assertIn("% \\input{sections/nothere}", tex)  # comments stay verbatim
        self.assertNotIn("MISSING", tex)

    def test_if_file_exists_takes_the_branch_the_file_system_gives(self):
        body = "\\IfFileExists{x.tex}{\\input{x}}{ELSE-BRANCH}"
        tex = _render(body, {"x.tex": "X-CONTENT"})[flatten.FLAT_TEX]
        self.assertIn("X-CONTENT", tex)
        self.assertNotIn("ELSE-BRANCH", tex)
        self.assertIn("ELSE-BRANCH", _render(body)[flatten.FLAT_TEX])

    def test_missing_input_is_a_visible_placeholder(self):
        rendered = _render("Before \\input{tables/t0}")
        self.assertIn("[MISSING: tables/t0.tex]", rendered[flatten.FLAT_TEX])
        self.assertIn("[MISSING: tables/t0.tex]", rendered[flatten.FULL_TEXT])

    def test_an_input_cycle_raises(self):
        with _Paper(_main("\\input{a}"), {"a.tex": "\\input{b}", "b.tex": "\\input{a}"}) as path:
            with self.assertRaisesRegex(ValueError, "cycle"):
                flatten.render(path)

    def test_numbers_are_pruned_to_the_macros_used_directly_or_through_another(self):
        numbers = ("% numbers.tex -- GENERATED\n\\providecommand{\\missingnumber}{\\textbf{??}}\n"
                   "% --- disk/a.csv\n\\newcommand{\\diskUsed}{$1.5$\\,pp}\n"
                   "\\newcommand{\\diskUnused}{$9.9$}\n% --- web/b.csv\n"
                   "\\newcommand{\\webOuter}{see \\webInner}\n\\newcommand{\\webInner}{$7$}\n"
                   "% --- live/c.csv\n\\newcommand{\\liveNever}{1}\n")
        preamble = "\\IfFileExists{numbers.tex}{\\input{numbers}}{}\n"
        rendered = _render("Gain \\diskUsed{} and \\webOuter.", {"numbers.tex": numbers}, preamble)
        tex = rendered[flatten.FLAT_TEX]
        for kept in ("\\diskUsed}", "\\webOuter}", "\\webInner}", "\\missingnumber}",
                     "% --- disk/a.csv", "% --- web/b.csv"):
            self.assertIn(kept, tex)
        for dropped in ("\\diskUnused", "\\liveNever", "% --- live/c.csv"):
            self.assertNotIn(dropped, tex)
        self.assertIn("keeps only the 3 of 5 macros", tex)
        self.assertIn("Gain 1.5 pp and see 7.", rendered[flatten.FULL_TEXT])

    def test_a_multi_line_numbers_macro_is_refused(self):
        numbers = "\\newcommand{\\diskA}{1}\n\\newcommand{\\diskB}{x\n}\n"
        preamble = "\\IfFileExists{numbers.tex}{\\input{numbers}}{}\n"
        with _Paper(_main("\\diskA", preamble), {"numbers.tex": numbers}) as path:
            with self.assertRaisesRegex(ValueError, "on one line"):
                flatten.render(path)


class BibliographyTests(unittest.TestCase):
    def test_citations_are_numbered_by_first_citation_and_compressed(self):
        text = _text("One~\\cite{gamma2003}. Many~\\cite{beta2002,alpha2001,gamma2003,"
                     "epsilon2005}. Again~\\cite{beta2002, gamma2003}.")
        self.assertIn("One [1].", text)
        self.assertIn("Many [1]\u2013[4].", text)  # cite.sty: ranges of three or more
        self.assertIn("Again [1], [2].", text)
        listed = re.findall(r"^\[(\d)\]\s+(.*)$", text, re.M)
        self.assertEqual([n for n, _ in listed], ["1", "2", "3", "4"])
        self.assertTrue(listed[0][1].startswith("The Gamma Project, \u201cGamma Manual,\u201d"))

    def test_unknown_cite_keys_print_a_question_mark_as_with_bibtex(self):
        rendered = _render("X~\\cite{nosuchkey}. Y~\\cite{beta2002,nosuchkey}.")
        self.assertIn("X [?]. Y [1], [?].", rendered[flatten.FULL_TEXT])
        self.assertEqual(re.findall(r"\\bibitem\{([^{}]+)\}", rendered[flatten.FLAT_TEX]),
                         ["beta2002"])

    def test_no_citations_means_no_bibliography(self):
        rendered = _render("No citations here.")
        self.assertNotIn("thebibliography", _code(rendered[flatten.FLAT_TEX]))
        self.assertNotIn("References", rendered[flatten.FULL_TEXT])

    def test_reference_format(self):
        text = re.sub(r"\s+", " ", _text("\\cite{alpha2001,beta2002,gamma2003,delta2004,epsilon2005}"))
        self.assertIn("A. B. Smith, J.-L. van der Berg, and \u00c9. Zola, \u201cAlpha I/O Paper,\u201d"
                      " in Proc. Alpha. ACM, 2001, pp. 1\u201310. doi: 10.1/abc_def.", text)
        self.assertIn("J. Doe, \u201cBeta,\u201d J. Beta, vol. 3, no. 4, p. 28, 2002.", text)
        self.assertIn("https://example.org/g_h, 2003. Accessed 2026-10-26.", text)
        self.assertIn("A. One et al., \u201cDelta,\u201d 2004.", text)  # more than ten authors
        self.assertIn("J. Yang et al., \u201cEpsilon,\u201d 2005.", text)

    def test_thebibliography_is_valid_latex_for_cited_keys_only(self):
        tex = _render("\\cite{beta2002}")[flatten.FLAT_TEX]
        code = _code(tex)
        self.assertEqual(re.findall(r"\\bibitem\{([^{}]+)\}", code), ["beta2002"])
        self.assertIn("\\begin{thebibliography}{9}", code)
        self.assertEqual(check.brace_problems(tex), [])
        self.assertEqual(check.environment_problems(tex), [])
        doi = _render("\\cite{alpha2001}")[flatten.FLAT_TEX]
        self.assertIn("doi: 10.1/abc\\_def.", doi)  # escaped outside \url


class TextTests(unittest.TestCase):
    def test_sections_appendices_floats_and_forward_references(self):
        body = ("\\section{Intro}\\label{sec:intro}\nSee Section~\\ref{sec:two}, "
                "Table~\\ref{tab:t}, Fig.~\\ref{fig:f} and Eq.~\\eqref{eq:u}.\n"
                "\\section{Two}\\label{sec:two}\n\\subsection{Part}\\label{sec:part}\n"
                "\\begin{equation}\nU = \\frac{a - b}{c}\\label{eq:u}\n\\end{equation}\n"
                "\\begin{figure}[t]\\includegraphics[width=\\columnwidth]{figures/f1}"
                "\\caption{A figure.}\\label{fig:f}\\end{figure}\n"
                "\\begin{table}[t]\\caption{A table.}\\label{tab:t}\\begin{tabular}{lr}\n\\hline\n"
                "a & 1 \\\\\n\\hline\n\\end{tabular}\\end{table}\n"
                "Part is \\ref{sec:part}.\n\\appendices\n\\section{Extra}\\label{app:x}\nIn \\ref{app:x}.")
        text = _text(body)
        self.assertIn("See Section II, Table I, Fig. 1 and Eq. (1).", text)
        self.assertIn("I. Intro\n========", text)
        self.assertIn("II-A. Part\n----------", text)
        self.assertIn("Part is II-A.", text)
        self.assertRegex(text, r"    U = \(a - b\)/c +\(1\)")
        self.assertIn("[Figure 1: A figure. \u2014 see paper/figures/f1.pdf]", text)
        self.assertIn("TABLE I: A table.", text)
        self.assertIn("Appendix A. Extra", text)
        self.assertIn("In A.", text)

    def test_tables_align_span_and_wrap(self):
        body = ("\\begin{table*}[t]\\centering\\scriptsize\\caption{T.}\\label{tab:x}\n"
                "\\begin{tabular}{lrp{0.5in}}\n\\hline\nName & Value & Note \\\\\n\\hline\n"
                "\\multicolumn{3}{l}{\\emph{Group one}} \\\\\n"
                "alpha & $1.5\\times$ & a long note that wraps \\\\\n"
                "b\\_c & $-2$ {\\scriptsize [$-3$, $-1$]} & -- \\\\\n\\hline\n\\end{tabular}\n"
                "\\\\[2pt]{\\scriptsize Footnote text.}\n\\end{table*}")
        text = _text(body)
        lines = text.split("\n")
        start = lines.index(next(l for l in lines if l.startswith("Name")))
        # Value is right-aligned to its widest cell; Note wraps at 0.5 in.
        self.assertEqual(lines[start], "Name         Value  Note")
        self.assertEqual(lines[start + 2], "Group one")  # a full-width span
        self.assertEqual(lines[start + 3], "alpha         1.5\u00d7  a long")
        self.assertEqual(lines[start + 4:start + 7], [" " * 20 + w for w in ("note", "that", "wraps")])
        self.assertIn("b_c    -2 [-3, -1]  \u2013", text)
        self.assertTrue(lines[start - 1].startswith("---"))
        self.assertIn("Footnote text.", text)

    def test_math_symbols_and_accents(self):
        text = _flat_text("With $\\kappa = t_{pos}/t_{xfer}$, $c^*$, $x^{\\dagger}$, $10^{-4}$, "
                     "$\\{0,1\\}$, $5$\\,$\\mu$s, 95\\,\\%, O\\_DIRECT, ``quoted'', a--b, "
                     "Ga{\\\"e}l, \\ldots{} done.")
        self.assertIn("With \u03ba = t_pos/t_xfer, c*, x\u2020, 10^-4, {0,1}, 5 \u03bcs, 95 %, "
                      "O_DIRECT, \u201cquoted\u201d, a\u2013b, Ga\u00ebl, \u2026 done.", text)

    def test_unknown_commands_keep_their_text_and_undefined_numbers_print_question_marks(self):
        text = _text("A \\mystery{kept words} and \\diskNoSuchMacro{} end. "
                     "\\textcolor{red}{Red} \\texttt{tt} \\emph{em} \\url{http://a.b/c_d}.")
        self.assertIn("A kept words and ?? end. Red tt em http://a.b/c_d.", text)

    def test_table_helpers_print_their_content(self):
        text = _flat_text("\\multirow{2}{*}{Rows} \\makecell[l]{Cell} \\resizebox{\\columnwidth}{!}{Box} "
                          "\\parbox[t]{2in}{Par}.")
        self.assertIn("Rows Cell Box Par.", text)

    def test_user_macros_with_arguments_expand(self):
        preamble = "\\newcommand{\\pair}[2]{(#1, #2)}\n\\newcommand{\\opt}[2][d]{<#1|#2>}\n"
        text = _text("\\pair{a}{b} \\opt{x} \\opt[y]{z} \\pair ab.", preamble=preamble)
        self.assertIn("(a, b) <d|x> <y|z> (a, b).", text)

    def test_lists_title_and_authors(self):
        body = ("\\title{A Title:\\\\ Sub}\n\\author{\\IEEEauthorblockN{Ann}\\IEEEauthorblockA{Uni\\\\"
                "City}\\and\\IEEEauthorblockN{Bob}\\IEEEauthorblockA{Lab}}\n\\maketitle\n"
                "\\begin{abstract}Short.\\end{abstract}\n"
                "\\begin{itemize}\n\\item First point.\n\\item Second point.\n\\end{itemize}")
        text = _text(body)
        self.assertTrue(text.startswith("A Title: Sub\n\nAnn\nUni\nCity\n\nBob\nLab\n\n"
                                        "Abstract\u2014Short."), text[:120])
        self.assertIn("  - First point.\n  - Second point.", text)

    def test_tex_spaces_after_control_words_are_swallowed(self):
        preamble = "\\newcommand{\\val}{$3$}\n"
        self.assertIn("3ms and 3 ms.", _text("\\val ms and \\val{} ms.", preamble=preamble))


class EntryPointTests(unittest.TestCase):
    def test_generate_writes_both_and_check_detects_a_stale_file(self):
        with _Paper(_main("Hello~\\cite{beta2002}.")) as path:
            quiet = io.StringIO()
            with contextlib.redirect_stdout(quiet):
                self.assertEqual(flatten.main(["--paper-dir", str(path), "--check"]), 1)
                self.assertEqual(flatten.main(["--paper-dir", str(path)]), 0)
                self.assertEqual(flatten.main(["--paper-dir", str(path), "--check"]), 0)
            for name in flatten.OUTPUTS:
                self.assertTrue((path / name).is_file(), name)
            (path / flatten.FULL_TEXT).write_text("edited\n", encoding="utf-8")
            with contextlib.redirect_stdout(quiet):
                self.assertEqual(flatten.main(["--paper-dir", str(path), "--check"]), 1)
            self.assertIn("paper_full.txt is stale", quiet.getvalue())

    def test_reproduce_paper_fixtures_writes_both_files_outside_paper(self):
        import reproduce
        with tempfile.TemporaryDirectory() as tmp:
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                status = reproduce.main(["paper", "--fixtures", "--out", tmp, "--no-check"])
            self.assertIn(status, (0, 4), out.getvalue())  # 4: matplotlib missing
            self.assertIn("[paper] flatten.py: main_flat.tex", out.getvalue())
            text = (Path(tmp) / flatten.FULL_TEXT).read_text(encoding="utf-8")
            self.assertIn("^FIXTURE", text)  # fixture numbers stay marked as such
            self.assertTrue((Path(tmp) / flatten.FLAT_TEX).is_file())


if __name__ == "__main__":
    unittest.main()
