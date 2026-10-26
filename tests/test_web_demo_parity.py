"""Real held-out replay must agree with the independently saved batch reference.

The demo needs the trained web artifacts (``models/web/``) and the prepared
views (``data/processed/``), which are gitignored; build them once with
``webprefetch prepare && webprefetch train``. The summary tables and the
parity reference are the original pivot run's, committed under
``results/pivot_original/``. That reference is valid only against the
committed full-run selection: a ``--quick`` or partial run must regenerate it
from the same run and never compare across runs.
"""

import importlib.util
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
# The web study needs the [web] extra; system python3 runs only the disk suite.
HAVE_WEB = all(importlib.util.find_spec(name) is not None
               for name in ("numpy", "pandas", "sklearn", "joblib"))
if HAVE_WEB:
    from webprefetch.demo_server import DemoData

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models" / "web"
PROCESSED = ROOT / "data" / "processed"
REFERENCE = ROOT / "results" / "pivot_original"
HAVE_ARTIFACTS = all(path.exists() for path in (
    MODELS / "candidate_index.joblib", MODELS / "ranker.joblib",
    PROCESSED / "views.csv.gz", REFERENCE / "session_parity.json"))


@unittest.skipUnless(HAVE_WEB, "web study needs numpy, pandas, scikit-learn and joblib")
@unittest.skipUnless(HAVE_ARTIFACTS, "demo artifacts absent; run webprefetch prepare && webprefetch train")
class DemoParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = json.loads((REFERENCE / "session_parity.json").read_text())
        cls.data = DemoData(ROOT, models=MODELS, processed=PROCESSED, outputs=REFERENCE)

    def test_real_session_matches_batch_replay(self):
        ref = self.reference
        result = self.data.step({"session_id": ref["session_id"], "step": ref["views"] - 1,
                                 "capacity": ref["capacity"], "cost": ref["cost"],
                                 "policy": ref["policy"]})
        for key in ("views", "hits", "prefetch_hits", "issued", "useful", "unused", "evicted_unused"):
            self.assertEqual(result["totals"][key], ref[key], key)
        counts = {}
        for step in result["steps"]:
            depth = step["selected_depth"]
            counts[str(depth)] = counts.get(str(depth), 0) + 1
        self.assertEqual(counts, ref["depth_counts"])
        self.assertEqual([s["item"] for s in result["steps"]],
                         [v["item"] for v in self.data.session(ref["session_id"])["views"]])

    def test_custom_path_uses_real_index_and_no_heldout_label(self):
        result = self.data.step({"items": [46971, 134947, 46971], "step": 2,
                                 "policy": "fixed1"})
        self.assertIn("hypothetical", result["source"])
        self.assertEqual(result["steps"][0]["actual_next_item"], 134947)
        with self.assertRaises(ValueError):
            self.data.step({"items": [-1], "step": 0})


if __name__ == "__main__":
    unittest.main()
