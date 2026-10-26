"""Parity arithmetic, the per-repeat summary and the printed conclusions.

Three ways the real measurements are set against the simulator, all
calibrated from the **same run's** ``none`` mode (same device, backend,
think model, trace and repeat), never from an assumed constant:

* ``pred_mean_us`` -- a request-level prediction from the hit ratio alone:
  ``f * c_hit + (1 - f) * c_miss``, where ``f`` is the run's full-hit request
  fraction and ``c_hit``/``c_miss`` are ``none``'s mean latency of full-hit
  and of missing requests. ``pred_err = (pred - measured) / measured`` is how
  far "hit ratio predicts latency" is from the truth.
* ``k_eff`` -- the ``prefetch_multiple`` at which ``LatencyModel``'s per-block
  cost (miss ``m`` per missed block, hit ``0.01 m``, prefetch ``k m`` per
  prefetched block, ``m`` taken from ``none``'s misses) reproduces the
  measured mean. It answers D11 empirically: below 1 a prefetch costs the
  requester less than the demand read it replaces.
* Spearman rank correlation across modes, per (device, backend, think,
  trace), of simulated ``IOCostModel`` device time and of simulated hit
  ratio against measured mean latency.

The summary also carries each mode's measured mean latency minus ``none``'s,
paired by repeat (both runs sit in the same shuffled cell), with a 95%
Student-t interval over the repeats: the primary comparison of
``results/realio/PREREGISTERED.md``.

These are statements about one trace prefix on one device under one think
model (plan honesty rule 15), never about HDDs or SSDs in general.
"""

from __future__ import annotations

import math
from collections import defaultdict

#: Two-sided 95% Student-t critical values by degrees of freedom.
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
        8: 2.306, 9: 2.262, 10: 2.228, 12: 2.179, 15: 2.131, 20: 2.086, 30: 2.042}
HIT_COST_FRACTION = 0.01     # LatencyModel.measured's documented hit cost
GROUP = ("device", "backend", "mode", "think", "trace", "requests")


def latency_summary(values) -> dict:
    ordered = sorted(values)
    n = len(ordered)
    if not n:
        return {"lat_mean_us": 0.0, "lat_p50_us": 0.0, "lat_p95_us": 0.0,
                "lat_p99_us": 0.0, "lat_max_us": 0.0}

    def q(p):
        return ordered[min(n - 1, int(p * n))]

    return {"lat_mean_us": sum(ordered) / n, "lat_p50_us": q(0.5),
            "lat_p95_us": q(0.95), "lat_p99_us": q(0.99), "lat_max_us": ordered[-1]}


def _t(df: int) -> float:
    if df in _T95:
        return _T95[df]
    smaller = [k for k in _T95 if k <= df]
    return _T95[max(smaller)] if df < 30 else 1.96


def t_interval(values) -> tuple[float, float, float]:
    """Mean and 95% Student-t interval over repeats; ``nan`` bounds below 2."""
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, math.nan, math.nan
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
    half = _t(n - 1) * sd / math.sqrt(n)
    return mean, mean - half, mean + half


def _ranks(values) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(xs, ys) -> float | None:
    """Spearman rho with average ranks for ties; ``None`` below 3 points or
    when either side is constant."""
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 3:
        return None
    rx, ry = _ranks([p[0] for p in pairs]), _ranks([p[1] for p in pairs])
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    if sxx == 0 or syy == 0:
        return None
    return sxy / math.sqrt(sxx * syy)


def prediction(run: dict, none: dict) -> tuple[float | None, float | None]:
    """``(pred_mean_us, pred_err)`` for ``run`` calibrated from ``none``."""
    c_hit, c_miss = none.get("hit_lat_mean_us"), none.get("miss_lat_mean_us")
    reads = run.get("read_requests") or 0
    if c_miss is None or not reads or run.get("full_hits") is None:
        return None, None
    c_hit = c_hit if c_hit is not None else 0.0
    f = run["full_hits"] / reads
    pred = f * c_hit + (1 - f) * c_miss
    measured = run["lat_mean_us"]
    return pred, ((pred - measured) / measured if measured else None)


def k_eff(run: dict, none: dict) -> float | None:
    """Effective ``prefetch_multiple``; ``None`` when nothing was prefetched."""
    prefetched = run.get("prefetches") or 0
    reads = run.get("read_requests") or 0
    none_missed = none.get("missed_blocks") or 0
    none_misses = (none.get("read_requests") or 0) - (none.get("full_hits") or 0)
    if not prefetched or not reads or not none_missed or none.get("miss_lat_mean_us") is None:
        return None
    m = none["miss_lat_mean_us"] * none_misses / none_missed   # us per missed block
    missed, read_blocks = run.get("missed_blocks"), run.get("read_blocks")
    if missed is None or read_blocks is None:
        return None
    hit_blocks = read_blocks - missed
    demand = (missed * m + hit_blocks * HIT_COST_FRACTION * m) / reads
    return (run["lat_mean_us"] - demand) * reads / (prefetched * m)


def attach_parity(runs: list[dict]) -> None:
    """Fill ``pred_mean_us``, ``pred_err`` and ``k_eff`` in place.

    Each run is matched to the ``none`` run with the same device, backend,
    think model, trace, requests and repeat. Without one, the fields stay
    ``None`` rather than borrowing another repeat's calibration.
    """
    key = ("device", "backend", "think", "trace", "requests", "repeat")
    nones = {tuple(r[k] for k in key): r for r in runs if r["mode"] == "none"}
    for run in runs:
        none = nones.get(tuple(run[k] for k in key))
        if none is None:
            run["pred_mean_us"] = run["pred_err"] = run["k_eff"] = None
            continue
        run["pred_mean_us"], run["pred_err"] = prediction(run, none)
        run["k_eff"] = k_eff(run, none)


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def summarize(rows: list[dict], details: dict | None = None,
              cost_models: dict | None = None) -> list[dict]:
    """``realio_summary.csv`` rows: mean and CI over repeats per group.

    ``details`` maps a row key to its detail record (for simulated
    ``blocks_moved``); ``cost_models`` maps a device label to its
    ``IOCostModel``. Either may be missing, which leaves the device-time
    columns ``None``.
    """
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in GROUP)].append(row)
    paired = ("device", "backend", "think", "trace", "requests", "repeat")
    none_lat = {tuple(r[k] for k in paired): r["lat_mean_us"] for r in rows
                if r["mode"] == "none"}
    out = []
    for key, members in groups.items():
        first = members[0]
        mean, lo, hi = t_interval([r["lat_mean_us"] for r in members])
        diffs = [r["lat_mean_us"] - none_lat[tuple(r[k] for k in paired)] for r in members
                 if first["mode"] != "none" and tuple(r[k] for k in paired) in none_lat]
        delta = t_interval(diffs) if diffs else (None, None, None)
        device_time = None
        model = (cost_models or {}).get(first["device"])
        if model is not None and first["sim_device_ios"] is not None and details:
            detail = details.get(row_key(first))
            if detail and detail.get("sim_blocks_moved") is not None:
                device_time = (first["sim_device_ios"] * model.t_pos_us
                               + detail["sim_blocks_moved"] * model.t_xfer_us_per_block)
        foreign = [r["foreign_io_frac"] for r in members if r["foreign_io_frac"] is not None]
        out.append({
            **dict(zip(GROUP, key)), "reps": len(members),
            "sim_hit_ratio": first["sim_hit_ratio"], "sim_device_ios": first["sim_device_ios"],
            "sim_device_time_us": device_time,
            "real_hit_ratio": _mean([r["real_hit_ratio"] for r in members]),
            "full_hit_request_frac": _mean([r["full_hit_request_frac"] for r in members]),
            "lat_mean_us": mean, "lat_mean_us_lo": lo, "lat_mean_us_hi": hi,
            "lat_p50_us": _mean([r["lat_p50_us"] for r in members]),
            "lat_p99_us": _mean([r["lat_p99_us"] for r in members]),
            "late_hits": _mean([r["late_hits"] for r in members]),
            "device_MiB": _mean([r["device_MiB"] for r in members]),
            "pred_mean_us": _mean([r["pred_mean_us"] for r in members]),
            "pred_err": _mean([r["pred_err"] for r in members]),
            "k_eff": _mean([r["k_eff"] for r in members]),
            "delta_vs_none_us": delta[0], "delta_vs_none_us_lo": delta[1],
            "delta_vs_none_us_hi": delta[2],
            "spearman_device_time": None, "spearman_hit_ratio": None,
            "foreign_io_frac_max": max(foreign) if foreign else None,
            "load_avg_max": max(r["load_avg"] for r in members),
        })
    by_cell: dict[tuple, list[dict]] = defaultdict(list)
    for row in out:
        by_cell[(row["device"], row["backend"], row["think"], row["trace"],
                 row["requests"])].append(row)
    for cell in by_cell.values():
        lat = [r["lat_mean_us"] for r in cell]
        rho_time = spearman([r["sim_device_time_us"] for r in cell], lat)
        rho_hit = spearman([r["sim_hit_ratio"] for r in cell], lat)
        for r in cell:
            r["spearman_device_time"], r["spearman_hit_ratio"] = rho_time, rho_hit
    out.sort(key=lambda r: tuple(str(r[k]) for k in GROUP))
    return out


def row_key(row: dict) -> tuple:
    return tuple(row[k] for k in ("device", "backend", "mode", "think", "repeat",
                                  "trace", "requests"))


def _fmt(value, spec=".1f") -> str:
    return "n/a" if value is None or (isinstance(value, float) and math.isnan(value)) \
        else format(value, spec)


def conclusions(summary: list[dict], parity_runs: int = 0) -> list[str]:
    """What the run shows, as a few sentences per cell; raw rows are in the CSV."""
    lines = []
    if parity_runs:
        lines.append(f"Backend B parity held on all {parity_runs} odirect runs: block "
                     "hit/miss and useful prefetches equal the simulator's.")
    cells: dict[tuple, list[dict]] = defaultdict(list)
    for row in summary:
        cells[(row["device"], row["backend"], row["think"], row["trace"])].append(row)
    for (device, backend, think, trace), rows in sorted(cells.items()):
        rows = sorted(rows, key=lambda r: r["lat_mean_us"])
        best = rows[0]
        none = next((r for r in rows if r["mode"] == "none"), None)
        head = f"[{device} {backend} {think} {trace}]"
        if none is not None and best["mode"] != "none":
            change = (best["lat_mean_us"] - none["lat_mean_us"]) / none["lat_mean_us"]
            lines.append(f"{head} lowest mean latency: {best['mode']} "
                         f"{_fmt(best['lat_mean_us'])} us vs none "
                         f"{_fmt(none['lat_mean_us'])} us ({change:+.1%}), "
                         f"reps={best['reps']}.")
        else:
            lines.append(f"{head} lowest mean latency: {best['mode']} "
                         f"{_fmt(best['lat_mean_us'])} us, reps={best['reps']}.")
        top_hit = max((r for r in rows if r["sim_hit_ratio"] is not None),
                      key=lambda r: r["sim_hit_ratio"], default=None)
        if top_hit is not None and top_hit["mode"] != best["mode"]:
            lines.append(f"{head} highest simulated hit ratio is {top_hit['mode']} "
                         f"({top_hit['sim_hit_ratio']:.4f}), which is not the "
                         "lowest-latency mode.")
        rho = rows[0]["spearman_hit_ratio"]
        if rho is not None:
            lines.append(f"{head} Spearman(sim hit ratio, measured latency) = {rho:+.2f}"
                         f"; Spearman(sim device time, latency) = "
                         f"{_fmt(rows[0]['spearman_device_time'], '+.2f')}.")
        flagged = [r["mode"] for r in rows
                   if r["foreign_io_frac_max"] is not None and r["foreign_io_frac_max"] > 0.01]
        if flagged:
            lines.append(f"{head} foreign I/O above 1% during: {', '.join(flagged)}; "
                         "treat those latencies as contended.")
    return lines
