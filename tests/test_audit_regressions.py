"""Constructed behavior fixtures, never research data."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import pandas as pd
from webprefetch.data import partition_sessions
from webprefetch.model import CandidateIndex, fit_index
from webprefetch.simulator import (Cache, Counters, Policy, replay, paired_bootstrap,
                                   expected_calibration_error, break_even_waste,
                                   replay_capacities, TransitionScorer)


class NullModel:
    steps = None


class BoundaryTests(unittest.TestCase):
    def test_popularity_ties_are_item_id_ordered(self):
        views=pd.DataFrame({'split':[0,0,0,0], 'eligible':[False]*4,
            'itemid':[9,2,9,2], 'next_item':[-1]*4})
        self.assertEqual(fit_index(views).popular[:2], (2,9))

    def test_partition_breaks_session_and_clears_cross_period_labels(self):
        views = pd.DataFrame({'timestamp':[1, 2, 10, 11], 'source_row':[0,1,2,3],
                              'session_id':[4]*4, 'itemid':[7,8,9,10], 'cart_count':[0]*4})
        out = partition_sessions(views, [5, 20, 30])
        self.assertEqual(out.groupby('session_id').split.nunique().max(), 1)
        self.assertEqual(out.next_item.tolist(), [8, -1, 10, -1])
        self.assertEqual(out.prev_item.tolist(), [-1, 7, -1, 9])
        self.assertEqual(out.eligible.tolist(), [True, False, True, False])

    def test_prefix_excludes_partial_sessions_and_next_label(self):
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        import numpy as np
        model=make_pipeline(StandardScaler(),LogisticRegression()).fit(np.array([[0]*9,[1]*9,[2]*9,[3]*9]),[0,1,0,1])
        index=CandidateIndex({1:((2,1),)}, {1:1}, {2:1}, ())
        v=pd.DataFrame({'session_id':[1,2,1], 'itemid':[1,3,2], 'timestamp':[1,2,3],
                        'source_row':[0,1,2], 'prev_item':[-1,-1,1], 'position':[0,0,1],
                        'gap_ms':[0,0,2], 'cart_count':[0]*3,
                        'eligible':[True,False,False], 'next_item':[2,-1,-1]})
        _, extra, _=replay(v,index,model,[Policy('none','fixed',0)],4,max_views=2)
        self.assertEqual(extra['oracle_eligible_pairs'],0)
        self.assertEqual(extra['replayed_views'],1)


class AccountingTests(unittest.TestCase):
    def test_transition_baseline_ignores_ranker(self):
        index=CandidateIndex({7:((9,4),(8,2))},{7:6},{9:4,8:2},())
        scored=TransitionScorer().score(SimpleNamespace(itemid=7),index)
        self.assertEqual([item for item,_ in scored], [9,8])
        self.assertAlmostEqual(scored[0][1], 4/6)

    def test_multi_capacity_replay_matches_single(self):
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        import numpy as np
        model=make_pipeline(StandardScaler(),LogisticRegression()).fit(np.array([[0]*9,[1]*9,[2]*9,[3]*9]),[0,1,0,1])
        index=CandidateIndex({1:((2,2),)}, {1:2}, {1:2,2:2}, ())
        views=pd.DataFrame({'session_id':[1,1,1], 'itemid':[1,2,1], 'timestamp':[1,2,3],
            'prev_item':[-1,1,2], 'position':[0,1,2], 'gap_ms':[0,1,1], 'cart_count':[0]*3,
            'eligible':[True,True,False], 'next_item':[2,1,-1]})
        policies=[Policy('fixed1','fixed',1), Policy('none','fixed',0)]
        multi=replay_capacities(views,index,model,policies,(2,4))
        for cap in (2,4):
            single,extra,_=replay(views,index,model,policies,cap,0,True)
            self.assertEqual(multi[cap][0],single)
            self.assertEqual(dict(multi[cap][1]['sessions']),dict(extra['sessions']))

    def test_calibration_weighting_and_waste_threshold(self):
        self.assertAlmostEqual(expected_calibration_error([[2, 1.0, 1], [2, 1.6, 0]]), .4)
        sessions={'a':{1:[3,4,5,1]}, 'b':{1:[1,1,5,1]}}
        result=break_even_waste(sessions,'a','b',repetitions=10)
        self.assertAlmostEqual(result['threshold'], 2/3)
        self.assertEqual(result['direction'],'wins_below')

    def test_adaptive_depth_uses_available_count(self):
        # One candidate must cost one request, not nominal depths 2/4/8.
        self.assertEqual(Policy('a','adaptive',0.04).depth([(3,0.05)]),1)

    def test_duplicate_current_eviction_and_unused(self):
        c=Counters(); cache=Cache(2,c)
        cache.demand(1); cache.prefetch(1); cache.prefetch(2); cache.prefetch(2)
        cache.prefetch(3)
        self.assertEqual((c.issued,c.evicted_unused), (2,1))
        self.assertEqual(c.row('p',2)['unused'],2)
        cache.demand(3); cache.demand(3)
        self.assertEqual((c.hits,c.prefetch_hits,c.useful,c.row('p',2)['unused']), (2,1,1,1))

    def test_paired_sessions_interval_degenerate(self):
        sessions={'a':{1:[2,1,3],2:[0,0,1]},'b':{1:[1,1,3],2:[0,0,1]}}
        low,high=paired_bootstrap(sessions,'a','b',repetitions=100)
        self.assertGreaterEqual(low,0)
        self.assertLessEqual(high,1)

if __name__ == '__main__': unittest.main()
