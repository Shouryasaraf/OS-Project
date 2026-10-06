"""Next-page candidate retrieval and a causal ranking model."""

from __future__ import annotations

from dataclasses import dataclass
from math import log1p
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


MAX_TRANSITIONS = 20
POPULAR_FALLBACK = 8
FEATURE_NAMES = (
    "log_pair_count", "pair_probability", "log_source_support",
    "log_global_popularity", "candidate_is_previous", "log_session_position",
    "log_preceding_gap_seconds", "cart_count", "retrieval_rank",
)


@dataclass
class CandidateIndex:
    transitions: dict[int, tuple[tuple[int, int], ...]]
    source_total: dict[int, int]
    global_count: dict[int, int]
    popular: tuple[int, ...]

    def candidates(self, current: int) -> list[tuple[int, int]]:
        found: dict[int, int] = {}
        for item, count in self.transitions.get(current, ()):
            if item != current:
                found[item] = count
        for item in self.popular:
            if item != current and item not in found:
                found[item] = 0
        return list(found.items())


def fit_index(views: pd.DataFrame) -> CandidateIndex:
    history = views.loc[views.split.eq(0)]
    pairs = history.loc[history.eligible & history.next_item.ge(0), ["itemid", "next_item"]]
    counts = pairs.groupby(["itemid", "next_item"], sort=False).size().rename("count").reset_index()
    counts.sort_values(["itemid", "count", "next_item"], ascending=[True, False, True], inplace=True)
    top = counts.groupby("itemid", sort=False).head(MAX_TRANSITIONS)
    transitions: dict[int, list[tuple[int, int]]] = {}
    for row in top.itertuples(index=False):
        transitions.setdefault(int(row.itemid), []).append((int(row.next_item), int(row.count)))
    source_total = {int(k): int(v) for k, v in pairs.groupby("itemid").size().items()}
    popularity = history.itemid.value_counts()
    global_count = {int(k): int(v) for k, v in popularity.items()}
    return CandidateIndex({k: tuple(v) for k, v in transitions.items()}, source_total,
                          global_count, tuple(int(x) for x in popularity.index[:POPULAR_FALLBACK]))


def features(row, candidate: int, pair_count: int, retrieval_rank: int,
             index: CandidateIndex) -> list[float]:
    support = index.source_total.get(int(row.itemid), 0)
    return [
        log1p(pair_count), pair_count / support if support else 0.0,
        log1p(support), log1p(index.global_count.get(candidate, 0)),
        float(candidate == row.prev_item), log1p(max(0, int(row.position))),
        log1p(max(0, int(row.gap_ms)) / 1000), float(row.cart_count),
        float(retrieval_rank),
    ]


def train_ranker(views: pd.DataFrame, index: CandidateIndex,
                 max_decisions: int = 30000, validation_decisions: int = 10000):
    decisions = views.loc[views.split.eq(1)]
    if len(decisions) > max_decisions:
        positions = np.linspace(0, len(decisions) - 1, max_decisions, dtype=np.int64)
        decisions = decisions.iloc[positions]
    x: list[list[float]] = []
    y: list[int] = []
    covered = 0
    eligible_count = 0
    for row in decisions.itertuples(index=False):
        candidates = index.candidates(int(row.itemid))
        if bool(row.eligible):
            eligible_count += 1
            covered += int(any(item == row.next_item for item, _ in candidates))
        for rank, (item, count) in enumerate(candidates):
            x.append(features(row, item, count, rank, index))
            y.append(int(bool(row.eligible) and item == row.next_item))
    if not y or sum(y) < 100:
        raise ValueError("fewer than 100 positive candidates; dataset/candidate pool is inadequate")
    validation = views.loc[views.split.eq(2)]
    if len(validation) > validation_decisions:
        positions = np.linspace(0, len(validation) - 1, validation_decisions, dtype=np.int64)
        validation = validation.iloc[positions]
    vx: list[list[float]] = []
    vy: list[int] = []
    for row in validation.itertuples(index=False):
        for rank, (item, count) in enumerate(index.candidates(int(row.itemid))):
            vx.append(features(row, item, count, rank, index))
            vy.append(int(bool(row.eligible) and item == row.next_item))
    if not vx or not any(vy):
        raise ValueError("validation has no positive candidate labels")
    train_x = np.asarray(x, dtype=np.float32)
    train_y = np.asarray(y, dtype=np.int8)
    val_x = np.asarray(vx, dtype=np.float32)
    val_y = np.asarray(vy, dtype=np.int8)
    tuning = []
    fitted = []
    for c in (0.1, 1.0, 10.0):
        candidate_model = make_pipeline(StandardScaler(), LogisticRegression(
            C=c, max_iter=200, solver="lbfgs"))
        candidate_model.fit(train_x, train_y)
        loss = float(log_loss(val_y, candidate_model.predict_proba(val_x)[:, 1]))
        tuning.append({"C": c, "validation_log_loss": loss})
        fitted.append(candidate_model)
    winner = min(range(len(tuning)), key=lambda n: tuning[n]["validation_log_loss"])
    model = fitted[winner]
    scorer = FastScorer(model)
    covered_at_8 = 0
    top1 = 0
    reciprocal_rank = 0.0
    validation_eligible = 0
    for row in validation.itertuples(index=False):
        if not bool(row.eligible):
            continue
        validation_eligible += 1
        ranked = scorer.score(row, index)[:8]
        rank = next((n for n, (item, _) in enumerate(ranked, 1)
                     if item == row.next_item), None)
        if rank is not None:
            covered_at_8 += 1
            top1 += int(rank == 1)
            reciprocal_rank += 1 / rank
    summary = {"training_decisions": int(len(decisions)), "training_eligible_next_views": eligible_count,
               "candidate_rows": len(y),
               "positive_candidates": int(sum(y)),
               "candidate_coverage": covered / eligible_count if eligible_count else 0.0,
               "validation_decisions": int(len(validation)),
               "validation_eligible_next_views": validation_eligible,
               "validation_candidate_rows": len(vy),
               "validation_positives": int(sum(vy)), "C_tuning": tuning,
               "validation_coverage_at_8": covered_at_8 / validation_eligible,
               "validation_hit_at_1": top1 / validation_eligible,
               "validation_mrr_at_8": reciprocal_rank / validation_eligible,
               "selected_C": tuning[winner]["C"], "feature_names": list(FEATURE_NAMES)}
    return model, summary


def score_candidates(row, index: CandidateIndex, model) -> list[tuple[int, float]]:
    candidates = index.candidates(int(row.itemid))
    if not candidates:
        return []
    matrix = np.asarray([features(row, item, count, rank, index)
                         for rank, (item, count) in enumerate(candidates)], dtype=np.float32)
    probabilities = model.predict_proba(matrix)[:, 1]
    ranked = [(item, float(score)) for (item, _), score in zip(candidates, probabilities)]
    ranked.sort(key=lambda pair: (-pair[1], pair[0]))
    return ranked


class FastScorer:
    """Equivalent logistic scoring without a sklearn call for every request."""

    def __init__(self, model):
        scaler, classifier = model.steps[0][1], model.steps[1][1]
        self.weights = classifier.coef_[0] / scaler.scale_
        self.bias = float(classifier.intercept_[0] - np.dot(self.weights, scaler.mean_))

    def score(self, row, index: CandidateIndex) -> list[tuple[int, float]]:
        candidates = index.candidates(int(row.itemid))
        if not candidates:
            return []
        matrix = np.asarray([features(row, item, count, rank, index)
                             for rank, (item, count) in enumerate(candidates)], dtype=np.float64)
        logits = np.clip(matrix @ self.weights + self.bias, -30, 30)
        probabilities = 1 / (1 + np.exp(-logits))
        ranked = [(item, float(score)) for (item, _), score in zip(candidates, probabilities)]
        ranked.sort(key=lambda pair: (-pair[1], pair[0]))
        return ranked


def save_artifacts(directory: str | Path, index: CandidateIndex, model, summary: dict) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(index, directory / "candidate_index.joblib", compress=3)
    joblib.dump(model, directory / "ranker.joblib", compress=3)
    import json
    (directory / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def load_artifacts(directory: str | Path):
    directory = Path(directory)
    # Load only artifacts created by this project; joblib uses Python pickle.
    return joblib.load(directory / "candidate_index.joblib"), joblib.load(directory / "ranker.joblib")
