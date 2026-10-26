"""Real held-out replay must agree with the independently saved batch reference.

The demo needs the trained web artifacts (``models/web/``) and the prepared
views (``data/processed/``), which are gitignored; build them once with
``webprefetch prepare && webprefetch train``.

A parity reference is valid only for the run whose validation selection
produced it. Two references are accepted:

* ``results/web/session_parity.json`` written by ``webprefetch parity``
  (costs 0, 0.01, 0.05 and 0.25), used when it names the SHA-256 of the
  current ``outputs/web/benchmark.json``;
* otherwise the original pivot run's hand-written single case under
  ``results/pivot_original/`` against that run's own summary tables.

Neither is ever compared with another run's selection. The always-on
fixture version of this check is in ``tests/test_web_benchmark.py``.
"""

import hashlib
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
PIVOT = ROOT / "results" / "pivot_original"
RUN_OUTPUTS = ROOT / "outputs" / "web"
RUN_PARITY = ROOT / "results" / "web" / "session_parity.json"
HAVE_ARTIFACTS = all(path.exists() for path in (
    MODELS / "candidate_index.joblib", MODELS / "ranker.joblib",
    PROCESSED / "views.csv.gz", PIVOT / "session_parity.json"))


def reference():
    """(summary directory, parity cases) for the newest same-run reference."""
    if RUN_PARITY.exists() and (RUN_OUTPUTS / "benchmark.json").exists():
        record = json.loads(RUN_PARITY.read_text())
        digest = hashlib.sha256((RUN_OUTPUTS / "benchmark.json").read_bytes()).hexdigest()
        if record.get("benchmark_json_sha256") == digest:
            return RUN_OUTPUTS, record["cases"]
    return PIVOT, [json.loads((PIVOT / "session_parity.json").read_text())]


@unittest.skipUnless(HAVE_WEB, "web study needs numpy, pandas, scikit-learn and joblib")
@unittest.skipUnless(HAVE_ARTIFACTS, "demo artifacts absent; run webprefetch prepare && webprefetch train")
class DemoParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        outputs, cls.cases = reference()
        cls.data = DemoData(ROOT, models=MODELS, processed=PROCESSED, outputs=outputs)

    def test_real_session_matches_batch_replay(self):
        for ref in self.cases:
            with self.subTest(session=ref["session_id"], cost=ref["cost"]):
                result = self.data.step({"session_id": ref["session_id"], "step": ref["views"] - 1,
                                         "capacity": ref["capacity"], "cost": ref["cost"],
                                         "policy": ref["policy"]})
                for key in ("views", "hits", "prefetch_hits", "issued", "useful", "unused",
                            "evicted_unused"):
                    self.assertEqual(result["totals"][key], ref[key], key)
                self.assertEqual(result["totals"]["depth_counts"], ref["depth_counts"])
                if "resolved_policy" in ref:
                    self.assertEqual(result["resolved_policy"], ref["resolved_policy"])
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
