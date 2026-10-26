"""Web core: cachecore, fresh and online indexes, parallel replay, new metrics.

Fixture tests use the committed SYNTHETIC ``data/samples/web_fixture.csv``
through ``build_views(..., verify_source=False)``; it is never research data.
Tests on the real prepared views skip when ``data/processed`` is absent, and
the full-split reproduction also needs ``RUN_SLOW=1``.
"""

import importlib.util
import os
import random
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
# The web study needs the [web] extra; system python3 runs only the disk suite.
HAVE_WEB = all(importlib.util.find_spec(name) is not None
               for name in ("numpy", "pandas", "sklearn", "joblib"))
if HAVE_WEB:
    import pandas as pd

    from webprefetch import cachecore, index as index_module, model, simulator
    from webprefetch.cachecore import Cache, Counters, FeedbackGate, GateTracker, Policy
    from webprefetch.data import build_views
    from webprefetch.index import OnlineIndex, fit_index, transition_index
    from webprefetch.parallel import replay_parallel
    from webprefetch.simulator import (TransitionScorer, break_even_waste, replay,
                                       replay_capacities)

FIXTURE = ROOT / "data" / "samples" / "web_fixture.csv"
PREPARED = ROOT / "data" / "processed" / "views.csv.gz"
MODELS = ROOT / "models" / "web"
RUN_SLOW = os.environ.get("RUN_SLOW") == "1"
STATELESS = None if not HAVE_WEB else [
    Policy("fixed0", "fixed", 0), Policy("fixed1", "fixed", 1), Policy("fixed4", "fixed", 4),
    Policy("adaptive_0.01", "adaptive", 0.01), Policy("adaptive_ca_0.01", "adaptive_ca", 0.01),
    Policy("gate_0.1", "gate", 0.1)]
CAPS = (2, 4, 8)


def fixture_views():
    views, _ = build_views(FIXTURE, verify_source=False)
    return views


def view_frame(rows):
    """Prepared-view records from (session, visitor, timestamp, item) tuples."""
    frame = pd.DataFrame(rows, columns=["session_id", "visitorid", "timestamp", "itemid"])
    frame["source_row"] = range(len(frame))
    frame["cart_count"] = 0
    frame["split"] = 0
    ordered = frame.sort_values(["session_id", "timestamp"], kind="mergesort")
    group = ordered.groupby("session_id", sort=False)
    ordered["position"] = group.cumcount()
    ordered["prev_item"] = group.itemid.shift(1).fillna(-1).astype(int)
    ordered["next_item"] = group.itemid.shift(-1).fillna(-1).astype(int)
    ordered["next_timestamp"] = group.timestamp.shift(-1).fillna(-1).astype(int)
    ordered["gap_ms"] = group.timestamp.diff().fillna(0).astype(int)
    ordered["eligible"] = ordered.next_item.ge(0)
    return ordered.sort_values(["timestamp", "source_row"], kind="mergesort")


def interleaved_views(seed=3, visitors=8, sessions_each=3, length=6):
    """Sessions of different visitors that overlap in time.

    The committed fixture's sessions never overlap, so on it session order
    and global order visit rows in the same order and cannot disagree. Each
    visitor's sessions are more than 30 minutes apart, as prepare() makes them.
    """
    rng = random.Random(seed)
    rows = []
    for visitor in range(visitors):
        for n in range(sessions_each):
            start = n * 3_600_000 + rng.randrange(0, 60_000)
            item = rng.randrange(10)
            for k in range(length):
                rows.append((visitor * sessions_each + n, 100 + visitor,
                             start + k * rng.randrange(1_000, 9_000), item))
                item = (item + rng.choice((1, 1, 2, 5))) % 12
    rng.shuffle(rows)  # source_row order must not decide the replay order
    return view_frame(rows)


def assert_same_replay(case, left, right, capacities):
    for cap in capacities:
        case.assertEqual(left[cap][0], right[cap][0], cap)
        a, b = left[cap][1], right[cap][1]
        case.assertEqual({k: dict(v) for k, v in a["sessions"].items()},
                         {k: dict(v) for k, v in b["sessions"].items()})
        for key in ("oracle_eligible_pairs", "oracle_candidate_coverage_at_8",
                    "ranking_hit_at_1", "novel_eligible_pairs", "novel_hit_at_1",
                    "novel_coverage_at_8", "replayed_views"):
            case.assertEqual(a[key], b[key], key)
        # Float sums are added in shard order, so they agree to rounding only.
        case.assertAlmostEqual(a["ranking_mrr_at_8"], b["ranking_mrr_at_8"], places=12)
        for x, y in zip(a["calibration_bins"], b["calibration_bins"]):
            case.assertEqual((x[0], x[2]), (y[0], y[2]))
            case.assertAlmostEqual(x[1], y[1], places=9)


@unittest.skipUnless(HAVE_WEB, "web study needs numpy, pandas, scikit-learn and joblib")
class CacheCoreTests(unittest.TestCase):
    def test_simulator_and_model_reexport_the_same_objects(self):
        for name in ("Cache", "Counters", "Policy", "DEPTHS", "UTILITY_PREFETCH_COST"):
            self.assertIs(getattr(simulator, name), getattr(cachecore, name), name)
        for name in ("CandidateIndex", "fit_index", "MAX_TRANSITIONS", "POPULAR_FALLBACK"):
            self.assertIs(getattr(model, name), getattr(index_module, name), name)

    def test_cachecore_needs_no_numpy(self):
        source = (ROOT / "src" / "webprefetch" / "cachecore.py").read_text(encoding="utf-8")
        self.assertNotIn("numpy", source.replace("numpy-free", ""))
        self.assertNotIn("pandas", source)

    def test_on_evict_reports_victim_and_kind(self):
        seen = []
        cache = Cache(2, Counters(), on_evict=lambda item, spec: seen.append((item, spec)))
        cache.demand(1)
        self.assertTrue(cache.prefetch(2))
        self.assertFalse(cache.prefetch(2))
        cache.prefetch(3)          # evicts the unused speculative 2 first
        cache.demand(3)
        cache.demand(4)            # no speculative object left: LRU demand 1 goes
        self.assertEqual(seen, [(2, True), (1, False)])

    def test_policy_rejects_unknown_kind_and_bad_epsilon(self):
        with self.assertRaises(ValueError):
            Policy("x", "oracle", 1)
        with self.assertRaises(ValueError):
            Policy("x", "feedback", 0.01, epsilon=1.5)
        with self.assertRaises(ValueError):
            Policy("x", "feedback", 0.01).depth([(1, 0.5)])

    def test_counters_merge_is_additive(self):
        a, b = Counters(views=3, hits=1, issued=2), Counters(views=2, useful=1, issued=1)
        a.depth_counts[1] += 2
        b.depth_counts[1] += 1
        a.merge(b)
        self.assertEqual((a.views, a.hits, a.issued, a.useful, a.depth_counts[1]), (5, 1, 3, 1, 3))

    def test_row_surfaces_per_view_metrics(self):
        c = Counters(views=4, hits=2, prefetch_hits=1, issued=3, useful=1,
                     demand_evicted_by_prefetch=2)
        row = c.row("p", 8)
        self.assertEqual(row["prefetch_hits_per_view"], 0.25)
        self.assertEqual(row["wasted_per_view"], 0.5)
        self.assertEqual(row["demand_evicted_by_prefetch_per_view"], 0.5)
        self.assertIsNone(row["c_star"])


@unittest.skipUnless(HAVE_WEB, "web study needs numpy, pandas, scikit-learn and joblib")
class CacheAwareTests(unittest.TestCase):
    def test_adaptive_ca_never_counts_resident_slots(self):
        rng = random.Random(7)
        ca = Policy("ca", "adaptive_ca", 0.02)
        for _ in range(300):
            ranked = [(item, rng.random() * 0.3) for item in rng.sample(range(20), 10)]
            ranked.sort(key=lambda pair: -pair[1])
            resident = set(rng.sample(range(20), 6))
            depth, nominal = ca.choose(ranked, resident)
            available = [pair for pair in ranked if pair[0] not in resident]
            # depth is the nominal DEPTHS choice; fewer items may be available.
            self.assertEqual(nominal, available[:depth])
            self.assertFalse({item for item, _ in nominal} & resident)

    def test_adaptive_pays_for_resident_items_and_adaptive_ca_does_not(self):
        # Item 5 is resident; adaptive spends its single slot on it (a no-op),
        # adaptive_ca skips it and declines the weak 6.
        ranked = [(5, 0.5), (6, 0.03)]
        self.assertEqual(Policy("a", "adaptive", 0.04).choose(ranked, {5})[1], [(5, 0.5)])
        self.assertEqual(Policy("c", "adaptive_ca", 0.04).choose(ranked, {5}), (0, []))

    def test_every_adaptive_ca_slot_is_a_real_fetch_in_replay(self):
        views = fixture_views()
        test = views.loc[views.split.ge(1)]
        policies = [Policy("a", "adaptive", 0.0), Policy("c", "adaptive_ca", 0.0)]
        out = replay_capacities(test, fit_index(views), None, policies, (4,),
                                scorer=TransitionScorer())
        rows = {row["policy"]: row for row in out[4][0]}
        # With penalty 0 the chosen depth stops where the scores reach 0, so it
        # never exceeds the candidates available here and depth x count is the
        # number of nominal slots.
        slots = {name: sum(depth * n for depth, n in row["depth_counts"].items())
                 for name, row in rows.items()}
        self.assertEqual(slots["c"], rows["c"]["issued"])
        self.assertGreater(slots["a"], rows["a"]["issued"])


@unittest.skipUnless(HAVE_WEB, "web study needs numpy, pandas, scikit-learn and joblib")
class IndexTests(unittest.TestCase):
    def test_fit_index_default_is_split_zero_and_max_split_widens(self):
        views = pd.DataFrame({"split": [0, 0, 1, 2], "eligible": [True] * 4,
                              "itemid": [1, 1, 1, 1], "next_item": [2, 2, 3, 3]})
        self.assertEqual(fit_index(views).transitions[1], ((2, 2),))
        self.assertEqual(fit_index(views, max_split=2).transitions[1], ((2, 2), (3, 2)))
        self.assertEqual(transition_index(views, "trans_prior", 2).transitions[1], ((2, 2), (3, 1)))

    def test_online_index_never_uses_a_future_transition(self):
        views = pd.DataFrame({
            "split": [0, 1, 1, 1, 1], "eligible": [False, True, False, True, False],
            "itemid": [9, 1, 2, 1, 2], "next_item": [-1, 2, -1, 2, -1],
            "timestamp": [0, 10, 20, 30, 40], "next_timestamp": [-1, 20, -1, 40, -1]})
        online = OnlineIndex.from_views(views, split=1)
        online.advance(10)
        self.assertNotIn(2, dict(online.transitions.get(1, ())))
        online.advance(20)   # 1->2 completes at 20: not strictly earlier, still unseen
        self.assertNotIn(2, dict(online.transitions.get(1, ())))
        online.advance(21)
        self.assertEqual(online._top(1), ((2, 1),))
        self.assertEqual(online.source_total[1], 1)
        with self.assertRaises(ValueError):
            online.advance(5)
        self.assertEqual(online.fresh().position, 0)

    def test_replayed_decisions_only_see_earlier_transitions(self):
        views = fixture_views()
        online = transition_index(views, "trans_online", 3)
        checked = []

        class Spy(TransitionScorer):
            def score(self, row, index):
                done = index.pending[:index.position]
                self.check(all(ts < row.timestamp for ts, _, _ in done))
                if index.position < len(index.pending):
                    self.check(index.pending[index.position][0] >= row.timestamp)
                return super().score(row, index)

            def check(self, ok):
                checked.append(ok)

        replay(views.loc[views.split.eq(3)], online, None, [Policy("f1", "fixed", 1)], 4,
               0, scorer=Spy(), order="global")
        self.assertTrue(checked)
        self.assertTrue(all(checked))
        self.assertEqual(online.position, 0)  # the replay worked on a fresh copy

    def test_online_index_and_feedback_refuse_session_order(self):
        views = fixture_views()
        online = transition_index(views, "trans_online", 3)
        with self.assertRaises(ValueError):
            replay(views, online, None, [Policy("f", "fixed", 1)], 4, 0, scorer=TransitionScorer())
        with self.assertRaises(ValueError):
            replay(views, fit_index(views), None, [Policy("g", "feedback", 0.01)], 4, 0,
                   scorer=TransitionScorer())


@unittest.skipUnless(HAVE_WEB, "web study needs numpy, pandas, scikit-learn and joblib")
class ReplayTests(unittest.TestCase):
    def test_global_order_equals_session_order_for_stateless_policies(self):
        views = fixture_views()
        test = views.loc[views.split.ge(1)]
        index = fit_index(views)
        for scope in ("session", "visitor"):
            session = replay_capacities(test, index, None, STATELESS, CAPS,
                                        scorer=TransitionScorer(), cache_scope=scope)
            global_ = replay_capacities(test, index, None, STATELESS, CAPS,
                                        scorer=TransitionScorer(), cache_scope=scope,
                                        order="global")
            assert_same_replay(self, session, global_, CAPS)

    def test_orders_agree_when_sessions_overlap_in_time(self):
        views = interleaved_views()
        by_time = views.sort_values(["timestamp", "source_row"], kind="mergesort").session_id
        # Non-vacuous: the two orders really visit the rows differently.
        self.assertGreater(int(by_time.ne(by_time.shift()).sum()), views.session_id.nunique())
        index = fit_index(views)
        for scope in ("session", "visitor"):
            session = replay_capacities(views, index, None, STATELESS, CAPS,
                                        scorer=TransitionScorer(), cache_scope=scope)
            global_ = replay_capacities(views, index, None, STATELESS, CAPS,
                                        scorer=TransitionScorer(), cache_scope=scope,
                                        order="global")
            assert_same_replay(self, session, global_, CAPS)
            self.assertGreater(session[2][0][2]["issued"], 0)
            parallel = replay_parallel(views, index, TransitionScorer(), STATELESS, CAPS,
                                       workers=3, order="global", cache_scope=scope)
            assert_same_replay(self, session, parallel, CAPS)

    def test_visitor_scope_persists_across_a_visitors_sessions(self):
        views = view_frame([(1, 50, 0, 7), (1, 50, 1, 8), (2, 50, 9_000_000, 7),
                            (3, 60, 10, 7), (4, 61, 20, 7)])
        index = fit_index(views)
        none = [Policy("none", "fixed", 0)]
        by_session = replay(views, index, None, none, 4, 0, scorer=TransitionScorer())[0][0]
        by_visitor = replay(views, index, None, none, 4, 0, scorer=TransitionScorer(),
                            cache_scope="visitor")[0][0]
        self.assertEqual(by_session["hits"], 0)
        self.assertEqual(by_visitor["hits"], 1)
        with self.assertRaises(ValueError):
            replay(views.drop(columns="visitorid"), index, None, none, 4, 0,
                   scorer=TransitionScorer(), cache_scope="visitor")

    def test_novel_metrics_exclude_revisits(self):
        # Session 1 views 1, 2, 1, 3, so 2->1 is a revisit. The index ranks
        # 2 then 3 after 1, and 1 after 2: 1->2 and 2->1 are top-1 hits, but
        # only 1->2 and 1->3 are novel.
        views = view_frame([(1, 5, 0, 1), (1, 5, 1, 2), (1, 5, 2, 1), (1, 5, 3, 3)])
        index = index_module.CandidateIndex({1: ((2, 1), (3, 1)), 2: ((1, 1),)},
                                            {1: 2, 2: 1}, {1: 1, 2: 1, 3: 1}, ())
        _, extra, _ = replay(views, index, None, [Policy("none", "fixed", 0)], 4, 0,
                             scorer=TransitionScorer())
        self.assertEqual(extra["oracle_eligible_pairs"], 3)
        self.assertEqual(extra["novel_eligible_pairs"], 2)   # 1->2 and 1->3
        self.assertAlmostEqual(extra["ranking_hit_at_1"], 2 / 3)   # 1->2 and 2->1
        self.assertEqual(extra["novel_hit_at_1"], 0.5)            # 1->2 only
        self.assertEqual(extra["novel_coverage_at_8"], 1.0)

    def test_break_even_direction_dominates(self):
        sessions = {"a": {1: [5, 1, 5, 1], 2: [2, 0, 3, 0]},
                    "b": {1: [3, 4, 5, 1], 2: [2, 0, 3, 0]}}
        result = break_even_waste(sessions, "a", "b", repetitions=50)
        self.assertEqual(result["direction"], "dominates")
        self.assertIn("dropped_draws", result)
        self.assertEqual(break_even_waste(sessions, "b", "a", repetitions=10)["direction"],
                         "dominated")
        same = {"a": {1: [1, 1, 1, 0]}, "b": {1: [1, 1, 1, 0]}}
        result = break_even_waste(same, "a", "b", repetitions=10)
        self.assertEqual((result["direction"], result["dropped_draws"]), ("constant", 10))
        # Equal waste but more hits still wins at every cost.
        more = {"a": {1: [2, 1, 1, 0]}, "b": {1: [1, 1, 1, 0]}}
        self.assertEqual(break_even_waste(more, "a", "b", repetitions=10)["direction"],
                         "dominates")


@unittest.skipUnless(HAVE_WEB, "web study needs numpy, pandas, scikit-learn and joblib")
class FeedbackGateTests(unittest.TestCase):
    def test_bucket_opens_on_observed_precision(self):
        gate = FeedbackGate(cost=0.3)
        bucket = gate.bucket(0, 0.25)
        self.assertEqual(bucket, (0, 6))
        self.assertTrue(gate.is_open(bucket))      # prior mean 1/2
        for _ in range(4):
            gate.credit(bucket, False)
        self.assertFalse(gate.is_open(bucket))     # 1/6
        polluting = FeedbackGate(cost=0.3, pollution=1.0)
        polluting.displaced[bucket] += 1
        self.assertFalse(polluting.is_open(bucket))  # 1/2 - 1/2

    def test_tracker_credits_use_eviction_and_session_expiry(self):
        gate = FeedbackGate(cost=0.0)
        tracker = GateTracker(gate)
        cache = Cache(3, Counters(), on_evict=tracker.on_evict)
        tracker.demand(cache, 1)
        self.assertEqual(tracker.prefetch(cache, [(2, 0.5), (3, 0.5), (1, 0.9)], session=7), 2)
        tracker.demand(cache, 2)                        # used
        tracker.demand(cache, 4)                        # evicts unused 3
        self.assertEqual(gate.used[(0, 7)], 1)
        self.assertEqual(gate.issued[(1, 7)], 1)
        tracker.prefetch(cache, [(5, 0.5)], session=7)
        tracker.retire(7)                               # 5 expires unused
        self.assertEqual(sum(gate.issued.values()), 3)
        self.assertEqual(sum(gate.used.values()), 1)
        self.assertEqual(tracker.tags, {})

    def test_unused_prefetches_are_credited_only_after_the_session_gap(self):
        # One unused credit closes the bucket (1/3 < 0.4). Session 1's unused
        # prefetch must not count before it has been idle for 30 minutes, so
        # session 2 (10 min later) still issues and session 3 (45 min) does not.
        minute = 60_000
        views = view_frame([(1, 10, 0, 1), (2, 20, 10 * minute, 1), (3, 30, 45 * minute, 1)])
        index = index_module.CandidateIndex({1: ((2, 1),)}, {1: 1}, {1: 3}, ())
        row = replay(views, index, None, [Policy("fb", "feedback", 0.4)], 4, 0,
                     scorer=TransitionScorer(), order="global")[0][0]
        self.assertEqual(row["issued"], 2)

    def test_feedback_replay_is_deterministic_and_parallel_exact(self):
        views = fixture_views()
        test = views.loc[views.split.ge(1)]
        policies = STATELESS[:2] + [Policy("fb", "feedback", 0.05, epsilon=0.1),
                                    Policy("fb_closed", "feedback", 0.9)]
        serial = replay_capacities(test, fit_index(views), None, policies, CAPS,
                                   scorer=TransitionScorer(), order="global")
        again = replay_capacities(test, fit_index(views), None, policies, CAPS,
                                  scorer=TransitionScorer(), order="global")
        assert_same_replay(self, serial, again, CAPS)
        parallel = replay_parallel(test, fit_index(views), TransitionScorer(), policies, CAPS,
                                   workers=3, order="global")
        assert_same_replay(self, serial, parallel, CAPS)
        rows = {row["policy"]: row for row in serial[8][0]}
        self.assertGreater(rows["fb"]["issued"], 0)
        self.assertEqual(rows["fb_closed"]["issued"], 0)


@unittest.skipUnless(HAVE_WEB, "web study needs numpy, pandas, scikit-learn and joblib")
class ParallelTests(unittest.TestCase):
    def test_parallel_equals_serial_on_the_fixture(self):
        views = fixture_views()
        test = views.loc[views.split.ge(1)]
        index = fit_index(views)
        serial = replay_capacities(test, index, None, STATELESS, CAPS, scorer=TransitionScorer())
        for workers in (1, 3):
            for scope in ("session", "visitor"):
                expected = serial if scope == "session" else replay_capacities(
                    test, index, None, STATELESS, CAPS, scorer=TransitionScorer(),
                    cache_scope=scope)
                parallel = replay_parallel(test, index, TransitionScorer(), STATELESS, CAPS,
                                           workers=workers, cache_scope=scope)
                assert_same_replay(self, expected, parallel, CAPS)

    def test_parallel_online_index_equals_serial(self):
        views = fixture_views()
        test = views.loc[views.split.eq(3)]
        online = transition_index(views, "trans_online", 3)
        serial = replay_capacities(test, online, None, STATELESS, CAPS,
                                   scorer=TransitionScorer(), order="global")
        parallel = replay_parallel(test, online, TransitionScorer(), STATELESS, CAPS,
                                   workers=4, order="global")
        assert_same_replay(self, serial, parallel, CAPS)

    def test_logistic_scorer_runs_in_parallel(self):
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        ranker = make_pipeline(StandardScaler(), LogisticRegression()).fit(
            np.asarray([[0, 1] * 4 + [0], [1, 0] * 4 + [1], [2, 1] * 4 + [2], [0, 2] * 4 + [3]],
                       dtype=float), [0, 1, 1, 0])
        views = fixture_views()
        test = views.loc[views.split.ge(1)]
        index = fit_index(views)
        serial = replay_capacities(test, index, ranker, STATELESS, CAPS)
        parallel = replay_parallel(test, index, model.FastScorer(ranker), STATELESS, CAPS, workers=2)
        assert_same_replay(self, serial, parallel, CAPS)


@unittest.skipUnless(HAVE_WEB, "web study needs numpy, pandas, scikit-learn and joblib")
@unittest.skipUnless(PREPARED.is_file(), "prepared views absent; run webprefetch prepare")
class RealPrefixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.views = pd.read_csv(PREPARED)

    def test_parallel_equals_serial_on_a_20k_test_prefix(self):
        test = self.views.loc[self.views.split.eq(3)]
        index = fit_index(self.views)
        serial = replay_capacities(test, index, None, STATELESS, CAPS, max_views=20000,
                                   scorer=TransitionScorer())
        parallel = replay_parallel(test, index, TransitionScorer(), STATELESS, CAPS,
                                   workers=8, max_views=20000)
        assert_same_replay(self, serial, parallel, CAPS)
        self.assertGreater(serial[8][1]["replayed_views"], 15000)
        global_ = replay_capacities(test, index, None, STATELESS, CAPS, max_views=20000,
                                    scorer=TransitionScorer(), order="global")
        assert_same_replay(self, serial, global_, CAPS)

    def test_online_index_parallel_equals_serial_on_a_20k_test_prefix(self):
        test = self.views.loc[self.views.split.eq(3)]
        online = transition_index(self.views, "trans_online", 3)
        policies = STATELESS[:3]
        serial = replay_capacities(test, online, None, policies, (8,), max_views=20000,
                                   scorer=TransitionScorer(), order="global")
        parallel = replay_parallel(test, online, TransitionScorer(), policies, (8,),
                                   workers=8, max_views=20000, order="global")
        assert_same_replay(self, serial, parallel, (8,))

    @unittest.skipUnless(RUN_SLOW, "slow: set RUN_SLOW=1")
    @unittest.skipUnless((MODELS / "ranker.joblib").is_file(), "trained web artifacts absent")
    def test_committed_logistic_fixed1_cap8_reproduces(self):
        index, ranker = model.load_artifacts(MODELS)
        test = self.views.loc[self.views.split.eq(3)]
        out = replay_parallel(test, index, model.FastScorer(ranker),
                              [Policy("fixed1", "fixed", 1)], (8,),
                              workers=min(32, os.cpu_count() or 1), with_sessions=False)
        row = out[8][0][0]
        self.assertEqual((row["hits"], row["issued"]), (65944, 445796))


@unittest.skipUnless(HAVE_WEB, "web study needs numpy, pandas, scikit-learn and joblib")
class FixturePrepareTests(unittest.TestCase):
    def test_fixture_builds_only_with_source_checks_off(self):
        with self.assertRaises(ValueError):
            build_views(FIXTURE)
        views, profile = build_views(FIXTURE, verify_source=False)
        self.assertFalse(profile["source_verified"])
        self.assertEqual(profile["views"], len(views))
        self.assertEqual(sorted(views.split.unique()), [0, 1, 2, 3])
        self.assertEqual(views.groupby("session_id").split.nunique().max(), 1)


if __name__ == "__main__":
    unittest.main()
