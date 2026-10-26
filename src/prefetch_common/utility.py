"""One objective for both studies: ``U(c) = (hits - c * issued) / requests``.

``c`` is what one speculative fetch costs, in units of one hit. A policy is
worth running at cost ``c`` exactly when its ``U(c)`` beats no prefetching.

**The disk cost model is U at one particular c.** The modelled mean cost per
read block is ``(hits*h + misses*m + issued*p) / blocks``, which rearranges to

    cost = m - (m - h) * U(c)        with  c = p / (m - h)

so ``argmin`` of modelled cost and ``argmax`` of ``U(c)`` are the same policy
whenever the policies replay the same blocks. :func:`cost_to_c` computes that
``c``. It equals ``LatencyModel.break_even_precision()``: 100/99 = 1.0101 for
the measured shape (hit 1% of a miss, prefetch = one miss) and 50/95 = 0.526
for the 5/100/50 us assumption. At ``c > 1`` a policy has to gain more than
one hit per block it prefetches to beat ``none``, which DECISIONS.md D11 calls
a break-even precision above 100%. That follows from costing every 512-B
block as one device read; it is not a property of hardware.

The web study never had a cost model; its "cost per prefetch" is an assumed
``c``, so the same function and the same columns serve both.

``requests`` is the unit hits are counted in: read **blocks** for the disk
study (the block hit ratio), **views** for the web study.
"""

from __future__ import annotations

import math
from typing import Iterable, Mapping, Sequence

#: The costs every ``U_<c>`` column is reported at. The last one is the disk
#: break-even of the measured cost model, kept exact (100/99) rather than
#: rounded to the 1.0101 its column name shows.
BREAK_EVEN_C = 100.0 / 99.0
UTILITY_CS: tuple[float, ...] = (0.0, 0.001, 0.0025, 0.005, 0.01, 0.02, 0.05,
                                 0.1, 0.25, 0.5, BREAK_EVEN_C)

# Where hits, prefetches and the request denominator live in the rows the two
# studies already produce. Earlier names win. The disk study counts hits per
# block, so ``read_blocks`` must win over ``requests`` (I/O requests).
_REQUEST_KEYS = ("read_blocks", "views", "requests")
_HIT_KEYS = ("hits", "read_hits")
_ISSUED_KEYS = ("issued", "total_prefetches", "prefetches")


def U(hits: float, issued: float, requests: float, c: float) -> float:
    """Utility per request: hits minus ``c`` hit-equivalents per prefetch."""
    if requests <= 0:
        raise ValueError("requests must be positive")
    if c < 0:
        raise ValueError("c must be non-negative")
    return (hits - c * issued) / requests


def cost_to_c(hit_us: float, miss_us: float, prefetch_us: float) -> float:
    """The ``c`` at which ``U`` ranks policies exactly as the cost model does.

    ``prefetch_us / (miss_us - hit_us)``: a prefetch costs ``prefetch_us`` and
    a useful one saves ``miss_us - hit_us``. Infinite when a hit saves nothing,
    matching ``LatencyModel.break_even_precision``.
    """
    if min(hit_us, miss_us, prefetch_us) < 0:
        raise ValueError("latencies must be non-negative")
    saving = miss_us - hit_us
    if saving <= 0:
        return math.inf
    return prefetch_us / saving


def model_c(latency_model) -> float:
    """:func:`cost_to_c` for an object with ``hit_us``, ``demand_miss_us`` and
    ``prefetch_us`` (``adaptive_prefetch.simulator.LatencyModel``)."""
    return cost_to_c(latency_model.hit_us, latency_model.demand_miss_us,
                     latency_model.prefetch_us)


def modelled_cost_us(u: float, hit_us: float, miss_us: float) -> float:
    """Mean modelled cost per block for a policy whose ``U(c)`` is ``u``.

    The inverse of the identity in the module docstring. It reproduces
    ``Metrics.mean_access_latency_us`` when ``u`` was computed at
    ``c = cost_to_c(hit_us, miss_us, prefetch_us)``.
    """
    return miss_us - (miss_us - hit_us) * u


def _field(row, keys: Sequence[str]):
    for key in keys:
        if isinstance(row, Mapping):
            value = row.get(key)
        else:
            value = getattr(row, key, None)
        if value is not None:
            return value
    return None


def row_counts(row) -> tuple[float, float, float]:
    """``(hits, issued, requests)`` from a disk or web result row.

    Accepts a mapping or an object (a disk ``Metrics`` works). Counts are
    preferred; ``hit_ratio`` and ``issued_per_view`` / ``issued_per_request``
    are used only when a count is absent.
    """
    requests = _field(row, _REQUEST_KEYS)
    if requests is None:
        raise KeyError(f"row has none of {_REQUEST_KEYS}")
    requests = float(requests)
    hits = _field(row, _HIT_KEYS)
    if hits is None:
        ratio = _field(row, ("hit_ratio",))
        if ratio is None:
            raise KeyError(f"row has none of {_HIT_KEYS} or hit_ratio")
        hits = float(ratio) * requests
    issued = _field(row, _ISSUED_KEYS)
    if issued is None:
        rate = _field(row, ("issued_per_view", "issued_per_request"))
        if rate is None:
            raise KeyError(f"row has none of {_ISSUED_KEYS} or issued_per_view")
        issued = float(rate) * requests
    return float(hits), float(issued), requests


def row_U(row, c: float) -> float:
    """:func:`U` of one result row."""
    return U(*row_counts(row), c)


def c_star(policy_row, none_row) -> float | None:
    """The cost per prefetch at which ``policy_row`` ties ``none_row``.

    ``(hit_ratio_p - hit_ratio_n) / (issued_rate_p - issued_rate_n)``. When the
    policy issues more than the baseline (the usual case, baseline = no
    prefetch) it beats the baseline exactly for ``c < c*``; a negative ``c*``
    means it loses at every cost. When it issues fewer, the inequality flips.
    ``None`` when both issue at the same rate, where no tie point exists.
    """
    hits_p, issued_p, requests_p = row_counts(policy_row)
    hits_n, issued_n, requests_n = row_counts(none_row)
    if requests_p <= 0 or requests_n <= 0:
        raise ValueError("requests must be positive")
    d_issued = issued_p / requests_p - issued_n / requests_n
    if d_issued == 0:
        return None
    return (hits_p / requests_p - hits_n / requests_n) / d_issued


def utility_column(c: float) -> str:
    """Column name for ``U`` at ``c``: ``U_0.01``, and ``U_1.0101`` for 100/99."""
    return f"U_{c:.6g}"


def utility_columns(row, cs: Iterable[float] = UTILITY_CS) -> dict[str, float]:
    """``{U_<c>: U(c)}`` for one row, the ``U(c)`` block of a results row."""
    hits, issued, requests = row_counts(row)
    return {utility_column(c): U(hits, issued, requests, c) for c in cs}


def _name(row) -> str:
    value = _field(row, ("policy", "mode", "label", "name"))
    return str(value) if value is not None else "?"


def envelope(rows: Sequence, cs: Iterable[float] = UTILITY_CS) -> list[dict]:
    """The best policy at each cost: ``[{c, policy, utility}, ...]``.

    ``rows`` must describe one replay (the same trace slice and capacity, so
    the same requests). A tie keeps the earlier row, so the result depends
    only on the order the rows were given in.
    """
    if not rows:
        raise ValueError("envelope of no rows")
    counts = [row_counts(row) for row in rows]
    out = []
    for c in cs:
        best = 0
        best_u = U(*counts[0], c)
        for index in range(1, len(rows)):
            u = U(*counts[index], c)
            if u > best_u:
                best, best_u = index, u
        out.append({"c": c, "policy": _name(rows[best]), "utility": best_u})
    return out
