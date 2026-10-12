"""Small deterministic correctness fixtures; never used as research data."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from webprefetch.model import CandidateIndex, FastScorer, features, fit_index, score_candidates
from webprefetch.simulator import Cache, Counters, Policy, replay


class WorkflowTests(unittest.TestCase):
    def test_candidate_index_uses_only_first_split(self):
        views = pd.DataFrame({
            "split": [0, 0, 1, 2, 3], "eligible": [True] * 5,
            "itemid": [1] * 5, "next_item": [2, 2, 9, 9, 9],
        })
        index = fit_index(views)
        self.assertEqual(index.transitions[1], ((2, 2),))
        self.assertNotIn(9, [item for item, _ in index.candidates(1)])

    def test_fast_scorer_matches_pipeline(self):
        x = np.asarray([[0, 1] * 4 + [0], [1, 0] * 4 + [1],
                        [2, 1] * 4 + [2], [0, 2] * 4 + [3]], dtype=float)
        y = np.asarray([0, 1, 1, 0])
        model = make_pipeline(StandardScaler(), LogisticRegression()).fit(x, y)
        index = CandidateIndex({1: ((2, 3), (3, 1))}, {1: 4}, {2: 3, 3: 1}, ())
        row = SimpleNamespace(itemid=1, prev_item=-1, position=0, gap_ms=0, cart_count=0)
        expected = score_candidates(row, index, model)
        actual = FastScorer(model).score(row, index)
        self.assertEqual([item for item, _ in actual], [item for item, _ in expected])
        for (_, a), (_, b) in zip(actual, expected):
            self.assertAlmostEqual(a, b, places=6)

    def test_speculative_eviction_preserves_demand_object(self):
        counters = Counters()
        cache = Cache(2, counters)
        cache.demand(1)
        cache.prefetch(2)
        cache.prefetch(3)
        self.assertIn(1, cache.blocks)
        self.assertNotIn(2, cache.blocks)
        self.assertEqual(counters.evicted_unused, 1)

    def test_policy_depth_has_zero_option(self):
        self.assertEqual(Policy("a", "adaptive", 0.5).depth([(2, 0.1)]), 0)
        self.assertEqual(Policy("a", "adaptive", 0.01).depth([(2, 0.1)]), 1)

    def test_replay_caches_are_session_local(self):
        model = make_pipeline(StandardScaler(), LogisticRegression()).fit(
            np.asarray([[0, 1] * 4 + [0], [1, 0] * 4 + [1],
                        [2, 1] * 4 + [2], [0, 2] * 4 + [3]], dtype=float),
            [0, 1, 1, 0])
        index = CandidateIndex({}, {}, {}, ())
        views = pd.DataFrame({
            "timestamp": [1, 2, 3], "session_id": [1, 2, 1],
            "itemid": [7, 7, 7], "eligible": [False] * 3,
            "next_item": [-1] * 3, "prev_item": [-1] * 3,
            "position": [0, 0, 1], "gap_ms": [0] * 3, "cart_count": [0] * 3,
        })
        rows, _, _ = replay(views, index, model, [Policy("none", "fixed", 0)], 4, 3)
        self.assertEqual(rows[0]["hits"], 1)


if __name__ == "__main__":
    unittest.main()
