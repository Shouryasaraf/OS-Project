"""The committed synthetic web fixture: reproducible, and shaped like events.csv.

The fixture checks are stdlib only; the comparison with the web loader's own
schema constants runs only where the [web] extra is installed.
"""

import csv
import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "data" / "samples" / "web_fixture.csv"
sys.path.insert(0, str(ROOT / "src"))

HAVE_WEB = all(importlib.util.find_spec(name) is not None
               for name in ("numpy", "pandas", "sklearn", "joblib"))


def _generator():
    spec = importlib.util.spec_from_file_location(
        "make_web_fixture", ROOT / "data" / "samples" / "make_web_fixture.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class WebFixtureTests(unittest.TestCase):
    def test_generator_reproduces_the_committed_bytes(self):
        self.assertEqual(_generator().generate(), FIXTURE.read_text(encoding="utf-8"))

    def test_rows_follow_the_retailrocket_conventions(self):
        with FIXTURE.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            header = tuple(reader.fieldnames)
            rows = list(reader)
        self.assertEqual(header, _generator().HEADER)
        self.assertEqual(len(rows), 200)
        events = {row["event"] for row in rows}
        self.assertEqual(events, {"view", "addtocart", "transaction"})
        for row in rows:
            self.assertGreaterEqual(int(row["timestamp"]), 0)
            self.assertGreaterEqual(int(row["visitorid"]), 0)
            self.assertGreaterEqual(int(row["itemid"]), 0)
            # Transaction IDs appear on transactions and nowhere else.
            self.assertEqual(bool(row["transactionid"]), row["event"] == "transaction")
        timestamps = [int(row["timestamp"]) for row in rows]
        self.assertEqual(timestamps, sorted(timestamps))
        # Enough structure for an index: repeat visitors and repeated items.
        self.assertGreater(len({row["visitorid"] for row in rows}), 10)
        self.assertLess(len({row["itemid"] for row in rows}), len(rows) // 4)

    @unittest.skipUnless(HAVE_WEB, "web study needs numpy, pandas, scikit-learn and joblib")
    def test_header_and_events_match_the_web_loader(self):
        from webprefetch.data import EVENTS, EXPECTED
        with FIXTURE.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            self.assertEqual(list(reader.fieldnames), EXPECTED)
            self.assertTrue({row["event"] for row in reader} <= EVENTS)


if __name__ == "__main__":
    unittest.main()
