"""notebooks/walkthrough.ipynb: it builds, every cell runs, and the committed
copy is a fresh build of the current results/ (re-run
``reproduce.py notebook`` after any change to results/)."""

from __future__ import annotations

import copy
import importlib.util
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
BUILDER = REPO / "notebooks" / "build_notebook.py"
COMMITTED = REPO / "notebooks" / "walkthrough.ipynb"
HAVE = all(importlib.util.find_spec(name) is not None
           for name in ("nbformat", "pandas", "matplotlib"))


def _load_builder():
    spec = importlib.util.spec_from_file_location("notebook_builder_test", BUILDER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _strip_images(notebook: dict) -> tuple[dict, list[int]]:
    """The notebook with PNG payloads blanked, and the code cells that had one."""
    notebook = copy.deepcopy(notebook)
    with_images = []
    for index, cell in enumerate(notebook["cells"]):
        for output in cell.get("outputs", ()):
            data = output.get("data", {})
            if "image/png" in data:
                data["image/png"] = "<png>"
                with_images.append(index)
    return notebook, with_images


@unittest.skipUnless(HAVE, "needs nbformat, pandas and matplotlib (the venv)")
class NotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.builder = _load_builder()
        cls.path = cls.builder.build(Path(cls.tmp.name) / "walkthrough.ipynb")
        cls.fresh = json.loads(cls.path.read_text(encoding="utf-8"))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_every_code_cell_ran_in_order_without_error(self):
        import nbformat
        nbformat.validate(nbformat.reads(self.path.read_text(encoding="utf-8"), 4))
        code = [c for c in self.fresh["cells"] if c["cell_type"] == "code"]
        self.assertEqual([c["execution_count"] for c in code], list(range(1, len(code) + 1)))
        for cell in code:
            self.assertFalse([o for o in cell["outputs"] if o["output_type"] == "error"])
            self.assertTrue(cell["outputs"], "".join(cell["source"])[:80])

    def test_figures_are_embedded_and_the_file_stays_small(self):
        _, with_images = _strip_images(self.fresh)
        self.assertGreaterEqual(len(with_images), 3)
        self.assertLess(self.path.stat().st_size, 5 * 1024 * 1024)

    def test_committed_notebook_is_a_fresh_build(self):
        committed = json.loads(COMMITTED.read_text(encoding="utf-8"))
        ours, our_images = _strip_images(self.fresh)
        theirs, their_images = _strip_images(committed)
        self.assertEqual(our_images, their_images)
        self.assertEqual(theirs, ours, "notebooks/walkthrough.ipynb is stale: "
                         "run `python reproduce.py notebook` and commit it")

    def test_build_is_deterministic(self):
        again = self.builder.build(Path(self.tmp.name) / "again.ipynb")
        self.assertEqual(again.read_bytes(), self.path.read_bytes())

    def test_markdown_types_no_result_numbers(self):
        decimal = re.compile(r"(?<![\w.])\d+\.\d+(?![\w.])")
        for cell in self.fresh["cells"]:
            if cell["cell_type"] == "markdown":
                # headings carry section numbers (2.1); the prose carries none
                text = "\n".join(line for line in "".join(cell["source"]).splitlines()
                                  if not line.startswith("#"))
                self.assertFalse(decimal.findall(text), text[:120])

    def test_first_cell_needs_no_pythonpath(self):
        first = next(c for c in self.fresh["cells"] if c["cell_type"] == "code")
        source = "".join(first["source"])
        self.assertIn('sys.path.insert(0, str(ROOT / "src"))', source)
        self.assertIn("spec_from_file_location", source)
        self.assertNotIn('ROOT / "paper")', source)


class ReproduceNotebookTests(unittest.TestCase):
    def test_missing_extras_return_unavailable(self):
        sys.path.insert(0, str(REPO))
        try:
            import repro
            from repro import notebook
        finally:
            sys.path.remove(str(REPO))
        with mock.patch.object(notebook.importlib.util, "find_spec", return_value=None), \
                mock.patch("builtins.print"):
            self.assertEqual(notebook.run(mock.Mock(out=None)), repro.EXIT_UNAVAILABLE)


if __name__ == "__main__":
    unittest.main()
