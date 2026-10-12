"""Real held-out replay must agree with the independently saved batch reference."""

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from webprefetch.demo_server import DemoData


class DemoParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]
        cls.reference = json.loads((cls.root / "outputs/session_parity.json").read_text())
        cls.data = DemoData(cls.root)

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
