"""Confidence intervals for paired comparisons, shared by both studies.

* :func:`paired_bootstrap` -- per-unit paired deltas (disk: one value per
  trace), resampled with replacement.
* :func:`cluster_bootstrap` -- two levels: resample clusters (corpora), then
  units (traces) within each drawn cluster. Twin traces are down-weighted with
  :func:`twin_weights` so a family of near-copies counts once.
* :func:`session_bootstrap` -- the web study's ratio estimator over
  per-session counters, ``sum(delta) / sum(views)``.
* :func:`wilcoxon` -- signed-rank p, or ``None`` when scipy is absent.

Every bootstrap is a percentile interval with numpy's default (linear)
quantile rule, and is deterministic for a given ``seed``. The stdlib
resampler draws from :class:`random.Random`, so its intervals are not the
same draws as a numpy-based script's; only ``session_bootstrap(...,
backend="numpy")`` reproduces ``webprefetch.simulator.paired_bootstrap``
exactly.
"""

from __future__ import annotations

import importlib.util
import math
import random
import warnings
from typing import Iterable, Mapping, NamedTuple, Sequence

# scipy is optional and only wilcoxon() needs it; it is imported there, not
# here, so importing this module stays cheap.
HAVE_SCIPY = importlib.util.find_spec("scipy") is not None
#: The note a report prints in place of a p-value when scipy is missing.
WILCOXON_UNAVAILABLE = "Wilcoxon p omitted: scipy is not installed"


class CI(NamedTuple):
    """A point estimate with its bootstrap interval."""
    estimate: float
    low: float
    high: float
    n: int


def _alpha(level: float) -> tuple[float, float]:
    if not 0 < level < 1:
        raise ValueError("level must be in (0, 1)")
    # Rounded so level=0.95 gives exactly 0.025/0.975, the quantiles numpy
    # callers pass literally.
    tail = round((1.0 - level) / 2.0, 12)
    return tail, round(1.0 - tail, 12)


def quantile(values: Sequence[float], q: float) -> float:
    """Linear-interpolation quantile, numpy's default ``method="linear"``."""
    if not values:
        raise ValueError("quantile of no values")
    if not 0 <= q <= 1:
        raise ValueError("q must be in [0, 1]")
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    lower = math.floor(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _interval(estimate: float, draws: list[float], n: int, level: float) -> CI:
    low_q, high_q = _alpha(level)
    return CI(estimate, quantile(draws, low_q), quantile(draws, high_q), n)


def _check_reps(reps: int) -> None:
    if reps < 1:
        raise ValueError("reps must be positive")


def paired_bootstrap(a: Sequence[float], b: Sequence[float], reps: int = 10000,
                     seed: int = 42, *, level: float = 0.95,
                     weights: Sequence[float] | None = None,
                     statistic: str = "difference") -> CI:
    """CI of the mean paired effect of ``a`` over ``b``.

    ``a[i]`` and ``b[i]`` belong to the same unit (trace). ``statistic`` is
    ``"difference"`` (``a - b``) or ``"ratio"`` (``a / b``, e.g. device I/Os
    relative to ``none``; every ``b`` must be non-zero). With ``weights`` the
    mean is ``sum(w * effect) / sum(w)``, and units are resampled together
    with their weights, so request-weighted and unweighted views share one
    resampler. A single unit gives a zero-width interval.
    """
    if len(a) != len(b):
        raise ValueError("a and b must be paired (same length)")
    if not a:
        raise ValueError("paired_bootstrap of no pairs")
    _check_reps(reps)
    if statistic == "difference":
        effects = [float(x) - float(y) for x, y in zip(a, b)]
    elif statistic == "ratio":
        if any(float(y) == 0 for y in b):
            raise ValueError("ratio statistic needs every baseline value non-zero")
        effects = [float(x) / float(y) for x, y in zip(a, b)]
    else:
        raise ValueError(f"unknown statistic {statistic!r}")
    w = [1.0] * len(effects) if weights is None else [float(x) for x in weights]
    if len(w) != len(effects):
        raise ValueError("weights must match the pairs")
    if any(x < 0 for x in w) or sum(w) <= 0:
        raise ValueError("weights must be non-negative with a positive sum")
    n = len(effects)
    estimate = sum(wi * e for wi, e in zip(w, effects)) / sum(w)
    if n == 1:
        return CI(estimate, estimate, estimate, 1)
    rng = random.Random(seed)
    indices = range(n)
    draws = []
    for _ in range(reps):
        chosen = rng.choices(indices, k=n)
        total_w = sum(w[i] for i in chosen)
        # An all-zero-weight resample has no mean; count it as the estimate
        # rather than dropping it, so every rep contributes one draw.
        draws.append(sum(w[i] * effects[i] for i in chosen) / total_w
                     if total_w else estimate)
    return _interval(estimate, draws, n, level)


def twin_weights(names: Sequence[str],
                 groups: Iterable[Iterable[str]]) -> list[float]:
    """Weight ``1/|group|`` per name, so a twin family counts as one trace.

    ``groups`` lists families of near-identical traces (``corpora.TWIN_GROUPS``
    in the disk study). Only members present in ``names`` count towards the
    group size; a name in no group weighs 1. A name in two groups is an error.
    """
    present = set(names)
    size: dict[str, int] = {}
    for group in groups:
        members = [name for name in group if name in present]
        for name in members:
            if name in size:
                raise ValueError(f"{name} is in more than one twin group")
            size[name] = len(members)
    return [1.0 / size.get(name, 1) for name in names]


def cluster_bootstrap(values: Sequence[float], clusters: Sequence,
                      weights: Sequence[float] | None = None,
                      reps: int = 10000, seed: int = 42, *,
                      level: float = 0.95) -> CI:
    """Two-level bootstrap CI of the weighted mean of ``values``.

    Each rep draws as many clusters as there are, with replacement, and then
    re-draws every drawn cluster's units with replacement. The statistic is
    ``sum(w * v) / sum(w)`` over everything drawn; the estimate is the same
    over the original data. With one cluster this is an ordinary bootstrap.
    Clusters are drawn from the sorted ``str(label)`` list, so listing the
    clusters in another order does not change the draws; units keep the
    order they were given in within their cluster.
    """
    if len(values) != len(clusters):
        raise ValueError("values and clusters must have the same length")
    if not values:
        raise ValueError("cluster_bootstrap of no values")
    _check_reps(reps)
    w = [1.0] * len(values) if weights is None else [float(x) for x in weights]
    if len(w) != len(values):
        raise ValueError("weights must match the values")
    if any(x < 0 for x in w) or sum(w) <= 0:
        raise ValueError("weights must be non-negative with a positive sum")
    v = [float(x) for x in values]
    members: dict[str, list[int]] = {}
    for index, label in enumerate(clusters):
        members.setdefault(str(label), []).append(index)
    labels = sorted(members)
    estimate = sum(wi * vi for wi, vi in zip(w, v)) / sum(w)
    if len(v) == 1:
        return CI(estimate, estimate, estimate, 1)
    rng = random.Random(seed)
    draws = []
    for _ in range(reps):
        total_wv = total_w = 0.0
        for label in rng.choices(labels, k=len(labels)):
            group = members[label]
            for i in rng.choices(group, k=len(group)):
                total_wv += w[i] * v[i]
                total_w += w[i]
        draws.append(total_wv / total_w if total_w else estimate)
    return _interval(estimate, draws, len(v), level)


def _session_arrays(sessions: Mapping, policy: str, baseline: str, cost: float):
    # Record layout from webprefetch.simulator.replay(with_sessions=True):
    # [hits, issued, views, useful]. A session absent from one policy counts
    # as zeros there, exactly as webprefetch.simulator.paired_bootstrap does.
    left, right = sessions[policy], sessions[baseline]
    keys = sorted(set(left) | set(right))
    zero = [0, 0, 0]
    delta = [(left.get(k, zero)[0] - right.get(k, zero)[0])
             - cost * (left.get(k, zero)[1] - right.get(k, zero)[1]) for k in keys]
    views = [left.get(k, right.get(k))[2] for k in keys]
    return keys, delta, views


def session_bootstrap(sessions: Mapping, policy: str, baseline: str, *,
                      cost: float = 0.0, reps: int = 500, seed: int = 42,
                      level: float = 0.95, backend: str = "auto") -> CI:
    """Paired session bootstrap of ``U(cost)`` per view, policy minus baseline.

    ``sessions[name][session_id] = [hits, issued, views, useful]`` (webprefetch's
    per-session dicts). The statistic is the ratio estimator
    ``sum(delta) / sum(views)`` over resampled sessions; ``cost=0`` gives the
    hit-ratio difference. ``backend="numpy"`` draws exactly as
    ``webprefetch.simulator.paired_bootstrap`` (default_rng, ``integers``,
    ``np.quantile``) and reproduces its interval; ``"stdlib"`` uses
    :class:`random.Random`; ``"auto"`` picks numpy when it is importable.
    The default ``reps=500`` matches the web study's committed runs.
    """
    if cost < 0:
        raise ValueError("cost must be non-negative")
    _check_reps(reps)
    keys, delta, views = _session_arrays(sessions, policy, baseline, cost)
    if not keys:
        raise ValueError("session_bootstrap of no sessions")
    if sum(views) <= 0:
        raise ValueError("sessions have no views")
    estimate = sum(delta) / sum(views)
    if backend == "auto":
        try:
            import numpy  # noqa: F401
            backend = "numpy"
        except ImportError:
            backend = "stdlib"
    low_q, high_q = _alpha(level)
    n = len(keys)
    if backend == "numpy":
        import numpy as np
        d = np.asarray(delta, dtype=np.float64)
        denominators = np.asarray(views, dtype=np.float64)
        rng = np.random.default_rng(seed)
        samples = np.empty(reps)
        for i in range(reps):
            chosen = rng.integers(0, n, n)
            samples[i] = d[chosen].sum() / denominators[chosen].sum()
        low, high = (float(x) for x in np.quantile(samples, [low_q, high_q]))
        return CI(float(estimate), low, high, n)
    if backend != "stdlib":
        raise ValueError(f"unknown backend {backend!r}")
    rng = random.Random(seed)
    indices = range(n)
    draws = []
    for _ in range(reps):
        chosen = rng.choices(indices, k=n)
        draws.append(sum(delta[i] for i in chosen) / sum(views[i] for i in chosen))
    return _interval(float(estimate), draws, n, level)


def wins(a: Sequence[float], b: Sequence[float],
         tol: float = 0.0) -> tuple[int, int, int]:
    """``(wins, losses, ties)`` of ``a`` over ``b``; ``|a - b| <= tol`` ties."""
    if len(a) != len(b):
        raise ValueError("a and b must be paired (same length)")
    won = lost = 0
    for x, y in zip(a, b):
        d = float(x) - float(y)
        if d > tol:
            won += 1
        elif d < -tol:
            lost += 1
    return won, lost, len(a) - won - lost


def wilcoxon(a: Sequence[float], b: Sequence[float] | None = None) -> float | None:
    """Two-sided Wilcoxon signed-rank p of ``a - b`` (or of ``a``).

    ``None`` when scipy is missing (print :data:`WILCOXON_UNAVAILABLE`), when
    there are fewer than two pairs, or when scipy cannot compute a p.
    """
    if not HAVE_SCIPY:
        return None
    from scipy import stats as scipy_stats
    diffs = [float(x) for x in a] if b is None else [
        float(x) - float(y) for x, y in zip(a, b, strict=True)]
    if len(diffs) < 2:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            p = float(scipy_stats.wilcoxon(diffs).pvalue)
        except ValueError:
            return None
    return None if math.isnan(p) else p
