"""Diagnostics around the replay: model comparisons, calibration, freshness,
next-view kinds, think time and the ranker's learning curve.

None of these select a policy; they explain the replay results. Every pass
reads only causal inputs (the index and ranker a decision could legally
use), so they can run on the test split without touching selection.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from .model import features

#: Width-0.1 probability bins, the same bins the replay counts.
CALIBRATION_BINS = 10
#: Bins at or above this lower edge are the reliability table the report
#: prints (P6): 99.8% of candidates sit in bin 0, where any ECE looks tiny.
RELIABLE_FROM = 0.1
CALIBRATION_COLUMNS = ("split", "scorer", "bin", "p_low", "p_high", "count",
                       "mean_probability", "observed_frequency", "gap", "reported",
                       "brier", "brier_reference", "brier_skill_score", "ece")
FRESHNESS_KS = (1, 2, 4, 8, 20, None)
#: The 30k default of ``train_ranker`` and two larger samples (P13).
LEARNING_CURVE_DECISIONS = (30000, 100000, 300000)
LEARNING_CURVE_COLUMNS = ("training_decisions", "candidate_rows", "positive_candidates",
                          "selected_C", "validation_coverage_at_8", "validation_hit_at_1",
                          "validation_mrr_at_8", "validation_views", "capacity", "policy",
                          "cost", "validation_hit_ratio", "validation_issued_per_view",
                          "validation_utility", "train_s")
THINK_THRESHOLDS_S = (0.1, 0.25, 0.5, 1, 2, 5, 10)
THINK_QUANTILES = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9)


def evaluate_models(views, index, ranker, model_dir: str | Path, output_dir: str | Path,
                    max_decisions: int = 10000) -> list[dict]:
    model_dir, output_dir = Path(model_dir), Path(output_dir)
    models = {
        'logistic_full': ranker,
        'logistic_no_context': joblib.load(model_dir / 'ranker_no_context.joblib'),
        'hist_gradient_boosting_diagnostic': joblib.load(model_dir / 'hist_gradient_boosting.joblib'),
    }
    results = []
    for split, label in ((2, 'validation'), (3, 'test')):
        sample = views.loc[views.split.eq(split)]
        if len(sample) > max_decisions:
            sample = sample.iloc[np.linspace(0, len(sample)-1, max_decisions, dtype=np.int64)]
        matrices, truth, groups, transition_hits = [], [], [], 0
        eligible = 0
        coverage = 0
        for row in sample.itertuples(index=False):
            candidates = index.candidates(int(row.itemid))
            groups.append((len(matrices), len(candidates), bool(row.eligible)))
            if bool(row.eligible):
                eligible += 1
                coverage += int(row.next_item in [item for item,_ in candidates])
                transition_hits += int(bool(candidates) and candidates[0][0] == row.next_item)
            for rank,(item,count) in enumerate(candidates):
                matrices.append(features(row,item,count,rank,index))
                truth.append(int(bool(row.eligible) and item == row.next_item))
        x=np.asarray(matrices,dtype=np.float32)
        y=np.asarray(truth,dtype=np.int8)
        for name,model in models.items():
            matrix=x.copy() if name == 'logistic_no_context' else x
            if name == 'logistic_no_context': matrix[:,4:8] = 0
            prob=model.predict_proba(matrix)[:,1]
            top1=0
            for start,size,is_eligible in groups:
                if size and is_eligible:
                    top1 += int(y[start+int(np.argmax(prob[start:start+size]))])
            results.append({'split':label,'model':name,'sample_decisions':len(sample),
                'eligible_pairs':eligible,'candidate_pool_coverage':coverage/eligible,
                'top1_per_eligible':top1/eligible,'candidate_log_loss':float(log_loss(y,prob))})
        results.append({'split':label,'model':'transition_top1_no_ranker',
            'sample_decisions':len(sample),'eligible_pairs':eligible,
            'candidate_pool_coverage':coverage/eligible,
            'top1_per_eligible':transition_hits/eligible,'candidate_log_loss':None})
    pd.DataFrame(results).to_csv(output_dir / 'model_diagnostics.csv',index=False)
    return results


# ------------------------------------------------------------ calibration

def _ordered(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["timestamp", "source_row"] if "source_row" in frame else ["timestamp"]
    return frame.sort_values(keys, kind="mergesort")


def calibration_rows(table, frame: pd.DataFrame, scorer: str, split_label: str) -> list[dict]:
    """Candidate-level reliability table plus the Brier skill score (P6).

    ``table`` is the :class:`~webprefetch.scorers.RankTable` of ``frame``'s
    rows (the rankings the replay used). Every ranked candidate is one
    forecast of "this is the next view", in the replay's width-0.1 bins.
    ``brier_reference`` is the Brier score of always forecasting the base
    rate, so ``brier_skill_score`` > 0 means better than that constant. ECE
    is kept as a secondary number only: almost every candidate falls in the
    lowest bin, where a near-zero ECE hides overconfidence higher up.
    """
    order = np.argsort(frame["rank_row"].to_numpy())
    next_item = frame.next_item.to_numpy()[order]
    eligible = frame.eligible.to_numpy(dtype=bool)[order]
    width = table.items.shape[1]
    valid = np.arange(width)[None, :] < table.lengths[:, None]
    p = table.probs[valid]
    y = ((table.items == next_item[:, None]) & eligible[:, None])[valid].astype(np.float64)
    bins = np.minimum(CALIBRATION_BINS - 1, (p * CALIBRATION_BINS).astype(np.int64))
    count = np.bincount(bins, minlength=CALIBRATION_BINS).astype(np.float64)
    sum_p = np.bincount(bins, weights=p, minlength=CALIBRATION_BINS)
    positives = np.bincount(bins, weights=y, minlength=CALIBRATION_BINS)
    total, base = float(count.sum()), float(y.sum())
    if total:
        brier = float(np.mean((p - y) ** 2))
        rate = base / total
        reference = rate * (1 - rate)
        skill = 1 - brier / reference if reference else None
        ece = float(np.abs(sum_p - positives).sum() / total)
    else:
        brier = reference = skill = ece = None
    rows = []
    for n in range(CALIBRATION_BINS):
        mean = float(sum_p[n] / count[n]) if count[n] else None
        observed = float(positives[n] / count[n]) if count[n] else None
        rows.append({"split": split_label, "scorer": scorer, "bin": n,
                     "p_low": n / CALIBRATION_BINS, "p_high": (n + 1) / CALIBRATION_BINS,
                     "count": int(count[n]), "mean_probability": mean,
                     "observed_frequency": observed,
                     "gap": None if mean is None else observed - mean,
                     "reported": n / CALIBRATION_BINS >= RELIABLE_FROM and bool(count[n]),
                     "brier": brier, "brier_reference": reference,
                     "brier_skill_score": skill, "ece": ece})
    return rows


# -------------------------------------------------------------- freshness

def _transition_counts(frame: pd.DataFrame):
    pairs = frame.loc[frame.eligible & frame.next_item.ge(0), ["itemid", "next_item"]]
    counts: dict[int, Counter] = defaultdict(Counter)
    for source, target in zip(pairs.itemid.to_numpy(), pairs.next_item.to_numpy()):
        counts[int(source)][int(target)] += 1
    popularity = frame.itemid.value_counts().rename("count").rename_axis("itemid").reset_index()
    popularity = popularity.sort_values(["count", "itemid"], ascending=[False, True])
    return counts, tuple(int(x) for x in popularity.itemid.iloc[:8])


def _rank_of(counts, popular, current: int, target: int) -> int | None:
    """1-based rank of ``target`` in the uncapped candidate order.

    The order is ``CandidateIndex.candidates``' (transition count, then item
    id, then the popular fallback) without the 20-transition cap, so k=None
    measures every candidate the counts could ever offer.
    """
    ranked = []
    targets = counts.get(current)
    if targets:
        ranked = [item for item, _ in sorted(((i, n) for i, n in targets.items() if i != current),
                                             key=lambda pair: (-pair[1], pair[0]))]
    seen = set(ranked)
    ranked += [item for item in popular if item != current and item not in seen]
    try:
        return ranked.index(target) + 1
    except ValueError:
        return None


def _coverage_rows(split_label, index_label, ranks, eligible, sources) -> list[dict]:
    rows = []
    for k in FRESHNESS_KS:
        hits = sum(1 for rank in ranks if rank is not None and (k is None or rank <= k))
        rows.append({"split": split_label, "index": index_label, "k": k,
                     "coverage": hits / eligible if eligible else 0.0,
                     "eligible_pairs": eligible, "sources_with_transitions": sources})
    return rows


def freshness_rows(views: pd.DataFrame, split: int, split_label: str) -> list[dict]:
    """Coverage@k of the evaluated split's next views vs index freshness.

    ``split0`` is the shipped index; ``split0..{s-1}`` every earlier split
    (the freshest frozen index that is still legal); ``online`` adds the
    evaluated split's own transitions once their next view is strictly
    earlier than the decision, as :class:`~webprefetch.index.OnlineIndex`
    does. A ranking-only bound: it says nothing about cache hits.
    """
    if split < 1:
        raise ValueError("freshness needs at least one earlier split")
    evaluated = _ordered(views.loc[views.split.eq(split)])
    eligible = evaluated.loc[evaluated.eligible]
    current = eligible.itemid.to_numpy()
    target = eligible.next_item.to_numpy()
    rows = []
    prior_label = "split" + "".join(str(n) for n in range(split))
    for label, max_split in (("split0", 0), (prior_label, split - 1)):
        counts, popular = _transition_counts(views.loc[views.split.le(max_split)])
        ranks = [_rank_of(counts, popular, int(a), int(b)) for a, b in zip(current, target)]
        rows += _coverage_rows(split_label, label, ranks, len(eligible), len(counts))
        if max_split == split - 1:
            prior_counts, prior_popular = counts, popular
    pending = eligible[["next_timestamp", "itemid", "next_item"]].sort_values(
        "next_timestamp", kind="mergesort").to_numpy()
    position = 0
    ranks = []
    for now, a, b in zip(eligible.timestamp.to_numpy(), current, target):
        while position < len(pending) and pending[position][0] < now:
            prior_counts[int(pending[position][1])][int(pending[position][2])] += 1
            position += 1
        ranks.append(_rank_of(prior_counts, prior_popular, int(a), int(b)))
    rows += _coverage_rows(split_label, "online", ranks, len(eligible), None)
    return rows


# ------------------------------------------------- next-view kinds, think

def pair_kinds(views: pd.DataFrame, split: int) -> dict:
    """Eligible next views by kind: self-repeat, earlier in session, novel.

    Only a novel next view can be a prefetch hit that a session cache did
    not already provide.
    """
    frame = views.loc[views.split.eq(split)]
    kinds = Counter()
    items = frame.itemid.to_numpy()
    nexts = frame.next_item.to_numpy()
    eligible = frame.eligible.to_numpy()
    for positions in frame.groupby("session_id", sort=False).indices.values():
        history = set()
        for r in positions:
            history.add(int(items[r]))
            if eligible[r]:
                target = int(nexts[r])
                kinds["self_repeat" if target == items[r] else
                      "earlier_in_session" if target in history else "novel"] += 1
    total = sum(kinds.values())
    return {"eligible": total, "kinds": dict(kinds),
            "kinds_share": {k: v / total for k, v in kinds.items()} if total else {}}


def think_time(views: pd.DataFrame, split: int) -> dict:
    """Seconds between a view and the session's next view (eligible pairs).

    The replay assumes a prefetch completes before the next view; this is
    the time it has to do so.
    """
    frame = views.loc[views.split.eq(split) & views.eligible]
    gap = (frame.next_timestamp.to_numpy() - frame.timestamp.to_numpy()) / 1000.0
    if not len(gap):
        return {"eligible": 0, "quantiles_s": {}, "share_below_s": {}}
    return {"eligible": int(len(gap)),
            "quantiles_s": {str(q): float(np.quantile(gap, q)) for q in THINK_QUANTILES},
            "share_below_s": {str(t): float(np.mean(gap < t)) for t in THINK_THRESHOLDS_S}}


# ---------------------------------------------------------- learning curve

def learning_curve_rows(views: pd.DataFrame, index, decisions=LEARNING_CURVE_DECISIONS, *,
                        capacity: int = 8, cost: float = 0.01, validation_views: int = 0,
                        workers: int = 1) -> list[dict]:
    """P13: does the 30k-decision training sample limit the ranker?

    Each size retrains the shipped ranker (same features, split 1, C chosen
    on validation log loss) and replays validation at ``capacity`` with
    fixed1 and adaptive_<cost>. Validation only; the test split is not read.
    """
    import time

    from .cachecore import Policy
    from .model import FastScorer, train_ranker
    from .parallel import replay_parallel

    validation = views.loc[views.split.eq(2)]
    policies = [Policy("fixed1", "fixed", 1), Policy(f"adaptive_{cost:g}", "adaptive", cost)]
    rows = []
    for n in decisions:
        start = time.monotonic()
        model, summary = train_ranker(views, index, max_decisions=int(n))
        elapsed = time.monotonic() - start
        result = replay_parallel(validation, index, FastScorer(model), policies, (capacity,),
                                 workers, max_views=validation_views, with_sessions=False)
        for row in result[capacity][0]:
            rows.append({"training_decisions": summary["training_decisions"],
                         "candidate_rows": summary["candidate_rows"],
                         "positive_candidates": summary["positive_candidates"],
                         "selected_C": summary["selected_C"],
                         "validation_coverage_at_8": summary["validation_coverage_at_8"],
                         "validation_hit_at_1": summary["validation_hit_at_1"],
                         "validation_mrr_at_8": summary["validation_mrr_at_8"],
                         "validation_views": row["views"], "capacity": capacity,
                         "policy": row["policy"], "cost": cost,
                         "validation_hit_ratio": row["hit_ratio"],
                         "validation_issued_per_view": row["issued_per_view"],
                         "validation_utility": (row["hits"] - cost * row["issued"]) / row["views"]
                         if row["views"] else 0.0,
                         "train_s": round(elapsed, 3)})
    return rows
