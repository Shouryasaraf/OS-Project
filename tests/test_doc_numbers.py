"""tools/check_doc_numbers.py: the docs quote only numbers that are in results/.

Stdlib only, so it runs under system python3 too. These tests pin how a
figure is matched (rounding, percent and pp scaling, signs), how its scope is
found (a cited file, filtered rows, an inherited table citation, nothing),
what is exempt (code, history regions, link targets), and that the
repository's own README, AGENTS, docs and slides pass.
"""

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location("check_doc_numbers",
                                               ROOT / "tools" / "check_doc_numbers.py")
docnums = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = docnums
_spec.loader.exec_module(docnums)

TABLE = """config,capacity,policy,mean_hit_ratio,delta,ratio,n
C1,2048,none,0.1485499956751096,,1.0,56
C1,2048,deep@8,0.2102649652005244,0.0617,1.0659846857287671,56
C1,131072,deep@8,0.3512932,0.068,1.0054,56
"""


class TempRepo:
    """A minimal repository: paper/check.py from this one, a results file, docs."""

    def __init__(self, docs: dict[str, str]):
        self.dir = tempfile.TemporaryDirectory()
        root = Path(self.dir.name)
        (root / "paper").mkdir()
        (root / "paper" / "check.py").write_text(
            (ROOT / "paper" / "check.py").read_text(encoding="utf-8"), encoding="utf-8")
        (root / "results" / "disk").mkdir(parents=True)
        (root / "results" / "disk" / "table.csv").write_text(TABLE, encoding="utf-8")
        for name, text in docs.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        self.root = root

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.dir.cleanup()

    def flagged(self) -> list[str]:
        return [f.figure for f in docnums.run(self.root).findings]


class FigureParsingTests(unittest.TestCase):
    def test_units_signs_and_groups(self):
        self.assertEqual(docnums.parse_figure("14.85%"), [(14.85, 2, False)])
        self.assertEqual(docnums.parse_figure("-0.94 pp"), [(-0.94, 2, True)])
        self.assertEqual(docnums.parse_figure("−0.94 pp"), [(-0.94, 2, True)])
        self.assertEqual(docnums.parse_figure("1.066x"), [(1.066, 3, False)])
        self.assertEqual(docnums.parse_figure("8,973,201"), [(8973201.0, 0, False)])
        self.assertEqual(docnums.parse_figure("51/5"), [(51.0, 0, False), (5.0, 0, False)])
        self.assertIsNone(docnums.parse_figure("about"))

    def test_index_matches_fractions_as_percent_and_respects_sign(self):
        index = docnums.Index({0.1485499956751096, -0.0094})
        self.assertTrue(index.has(14.85, 2, False))       # 0.14855 printed as percent
        self.assertTrue(index.has(0.149, 3, False))       # the fraction itself
        self.assertFalse(index.has(14.86, 2, False))      # a mis-rounded figure
        self.assertTrue(index.has(-0.94, 2, True))
        self.assertFalse(index.has(0.94, 2, True))        # an explicit + must match
        self.assertTrue(index.has(0.94, 2, False))        # unsigned: magnitude only


class ScopeTests(unittest.TestCase):
    def test_a_cited_file_is_the_scope(self):
        doc = "Mean hit ratio 21.03% (`results/disk/table.csv`), and 99.99% too.\n"
        with TempRepo({"README.md": doc}) as repo:
            self.assertEqual(repo.flagged(), ["99.99%"])

    def test_a_figure_without_a_cited_source_is_flagged(self):
        with TempRepo({"README.md": "A gain of 21.03% with no source.\n"}) as repo:
            self.assertEqual(repo.flagged(), ["21.03%"])

    def test_row_filters_narrow_the_scope(self):
        doc = ("<!-- source: results/disk/table.csv capacity=131072 -->\n"
               "deep@8 at 64 MiB: 35.13%; at 1 MiB: 21.03%.\n"
               "<!-- source: none -->\n")
        with TempRepo({"README.md": doc}) as repo:
            # 21.03% is in the file but not in the filtered rows.
            self.assertEqual(repo.flagged(), ["21.03%"])

    def test_a_source_directive_ends_at_the_next_heading(self):
        doc = ("<!-- source: results/disk/table.csv -->\n"
               "In scope: 21.03%.\n\n"
               "## Next section\n\n"
               "Out of scope: 21.03%.\n")
        with TempRepo({"README.md": doc}) as repo:
            self.assertEqual(repo.flagged(), ["21.03%"])

    def test_a_table_inherits_the_paragraph_before_it(self):
        doc = ("From `results/disk/table.csv`:\n\n"
               "| policy | hit |\n|---|---:|\n| `deep@8` | 21.03% |\n")
        with TempRepo({"README.md": doc}) as repo:
            self.assertEqual(repo.flagged(), [])

    def test_a_ratio_needs_both_numbers_in_scope(self):
        doc = "`results/disk/table.csv`: 56/56 traces, and 57/56.\n"
        with TempRepo({"README.md": doc}) as repo:
            self.assertEqual(repo.flagged(), ["57/56"])


class ExemptionTests(unittest.TestCase):
    def test_code_links_and_history_regions_are_exempt(self):
        doc = ("Run `--per-trace 50,000` and see [the note](x/1.5.md).\n\n"
               "```bash\npython main.py --prefetch-multiple 0.5\n```\n\n"
               "<!-- doc-numbers: off (history: an earlier run) -->\n"
               "D13 measured +4.10 points.\n"
               "<!-- doc-numbers: on -->\n\n"
               "Python 3.12 on kernel 6.14, section 2.6, D10.5, in 2026.\n")
        with TempRepo({"docs/notes.md": doc}) as repo:
            self.assertEqual(repo.flagged(), [])

    def test_link_text_is_prose_and_is_checked(self):
        with TempRepo({"README.md": "See [the 3.14% note](x.md).\n"}) as repo:
            self.assertEqual(repo.flagged(), ["3.14%"])

    def test_unowned_docs_are_not_checked(self):
        with TempRepo({"docs/MASTER_PLAN.md": "+16.32 pp\n"}) as repo:
            report = docnums.run(repo.root)
            self.assertEqual(report.findings, [])
            self.assertIn("docs/MASTER_PLAN.md", report.exempt_files)

    def test_long_integers_are_checked_but_years_and_powers_of_two_are_not(self):
        doc = "Capacity 131072 blocks in 2026; 4131 s and 9324 rows.\n"
        with TempRepo({"README.md": doc}) as repo:
            self.assertEqual(sorted(repo.flagged()), ["4131", "9324"])

    def test_slides_may_hold_no_typed_figure(self):
        tex = ("\\begin{frame}\\includegraphics[width=0.9\\linewidth]{f.pdf}"
               " \\diskTraces{} traces, a gain of 4.1\\,pp\\end{frame}\n")
        with TempRepo({"presentation/slides.tex": tex}) as repo:
            self.assertEqual(repo.flagged(), ["4.1\\,pp"])


class RepositoryTests(unittest.TestCase):
    def test_repository_docs_quote_only_results(self):
        report = docnums.run(ROOT)
        self.assertEqual([str(f) for f in report.findings], [])
        # The check is not vacuous: most figures are matched against cited rows.
        self.assertGreater(report.checked["rows"], report.checked["files"])


if __name__ == "__main__":
    unittest.main()
