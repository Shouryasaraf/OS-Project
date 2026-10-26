"""Per-pass analysis: parity, the validity gate, paired waits and the report.

For every pass (one policy, one repetition) this computes:

* **parity** -- live per-session ``[hits, issued, views, useful]`` against
  ``sim.json``. A ``hint:`` pass prefetches nothing into the browser cache,
  so its reference is the baseline's (no-prefetch) record.
* the **measured** perceived wait of every view, and the **model-ideal**
  wait of the same view: 0 on a hit (ready or in flight), ``rtt +
  size/bandwidth`` on a miss. That ideal is what an analytic simulator with
  the same latency model would predict without contention; the excess is
  decomposed for demand fetches into disk, link queue, server timer lag and
  time outside the server.
* the **validity gate**: per-request server lag p99, origin and client
  event-loop lag p99 all at most 2 ms, no parity mismatch and no connection
  error. An invalid pass is written with ``valid=False`` and its comparison
  columns are NaN: it is never compared.
* the paired session-bootstrap CI of the mean wait per view against the
  baseline pass of the same repetition (ratio estimator ``sum(delta) /
  sum(views)``, via :func:`prefetch_common.stats.session_bootstrap`).
"""

from __future__ import annotations

import math
import statistics
from collections import Counter

from .timing import quantile

GATE_LAG_P99_US = 2000.0
BOOTSTRAP_REPS = 1000
BOOTSTRAP_SEED = 42
#: foreign_io_frac above this is flagged in the report (schema live/live_metrics.csv).
FOREIGN_FLAG = 0.01
NAN = float("nan")


def reference_policy(policy: str) -> tuple[str, bool]:
    """(plan policy name, hint?) of a pass's policy label."""
    if policy.startswith("hint:"):
        return policy[len("hint:"):], True
    return policy, False


def parity(results: list[dict], sim: dict, policy: str, baseline: str) -> dict:
    """Live vs simulator counters for one pass; ``mismatched_sessions`` must be 0."""
    base, hint = reference_policy(policy)
    reference = sim["sessions"][baseline if hint else base]
    live_tot = [0, 0, 0, 0]
    sim_tot = [0, 0, 0, 0]
    mismatched = 0
    seen = set()
    for record in results:
        mine = list(record["counts"])
        theirs = reference.get(str(record["sid"]))
        seen.add(str(record["sid"]))
        if theirs is None or list(theirs) != mine:
            mismatched += 1
        for i in range(4):
            live_tot[i] += mine[i]
            sim_tot[i] += theirs[i] if theirs is not None else 0
    mismatched += sum(1 for sid in reference if sid not in seen)
    return {"sessions": len(results), "sim_hits": sim_tot[0], "live_hits": live_tot[0],
            "sim_issued": sim_tot[1], "live_issued": live_tot[1],
            "sim_useful": sim_tot[3], "live_useful": live_tot[3],
            "mismatched_sessions": mismatched}


def summarise_pass(results: list[dict], sizes: dict[int, int], *, rtt_ms: float,
                   mbps: float) -> dict:
    """Compact per-pass aggregates; the raw records can be dropped afterwards."""
    bytes_per_ms = mbps * 1e6 / 8 / 1e3
    classes: Counter = Counter()
    waits_ms: list[float] = []
    per_session: dict[str, list[float]] = {}
    lags, late_ms = [], []
    demand = []
    errors = 0
    demand_bytes = prefetch_bytes = useful_bytes = 0
    hits = views = issued = useful = 0
    conns = 0
    for record in results:
        wait_sum = ideal_sum = 0.0
        for _, item, cls, wait, late, _ in record["views"]:
            classes[cls] += 1
            ms = wait * 1e3
            waits_ms.append(ms)
            wait_sum += ms
            ideal_sum += 0.0 if cls != "miss" else rtt_ms + sizes[item] / bytes_per_ms
            late_ms.append(late * 1e3)
        per_session[str(record["sid"])] = [wait_sum, ideal_sum, len(record["views"])]
        for fetch in record["fetches"]:
            errors += fetch[10]
            if fetch[0] != "hint":
                lags.append(fetch[8])
            if fetch[0] == "demand" and not fetch[10]:
                demand.append(fetch)
        demand_bytes += record["demand_bytes"]
        prefetch_bytes += record["prefetch_bytes"]
        useful_bytes += record["useful_prefetch_bytes"]
        h, i, v, u = record["counts"]
        hits, issued, views, useful = hits + h, issued + i, views + v, useful + u
        conns += record["conns"]
    breakdown = None
    if demand:
        mean = lambda xs: round(statistics.fmean(xs), 3)  # noqa: E731
        breakdown = {
            "n": len(demand),
            "wall_ms": mean([(f[4] - f[3]) * 1e3 for f in demand]),
            "rtt_plus_tx_ms": mean([rtt_ms + f[2] / bytes_per_ms for f in demand]),
            "disk_ms": mean([f[6] / 1e3 for f in demand]),
            "link_queue_ms": mean([f[7] / 1e3 for f in demand]),
            "server_timer_lag_ms": mean([f[8] / 1e3 for f in demand]),
            "outside_server_ms": mean([(f[4] - f[3]) * 1e3 - f[5] / 1e3 for f in demand]),
            "model_minus_ideal_ms": mean([f[9] / 1e3 - rtt_ms - f[2] / bytes_per_ms
                                          for f in demand]),
        }
    n = len(waits_ms)
    return {
        "sessions": len(results), "views": n, "hits": hits, "issued": issued, "useful": useful,
        "counted_views": views,
        "ready": classes["ready"], "inflight": classes["inflight"], "miss": classes["miss"],
        "mean_wait_ms": statistics.fmean(waits_ms) if n else NAN,
        "model_ideal_wait_ms": (sum(s[1] for s in per_session.values()) / n) if n else NAN,
        "p95_wait_ms": quantile(waits_ms, 0.95) if n else NAN,
        "p99_wait_ms": quantile(waits_ms, 0.99) if n else NAN,
        "view_start_late_ms_p99": quantile(late_ms, 0.99) if n else NAN,
        "demand_bytes": demand_bytes, "prefetch_bytes": prefetch_bytes,
        "useful_prefetch_bytes": useful_bytes,
        "demand_disk_us_mean": statistics.fmean(f[6] for f in demand) if demand else 0.0,
        "req_lag_p99_us": float(quantile(lags, 0.99)) if lags else 0.0,
        "conn_errors": errors, "conns_opened": conns, "breakdown": breakdown,
        "per_session": per_session,
    }


def validity(summary: dict, server: dict, client_lag: dict, mismatched: int, *,
             gate_us: float = GATE_LAG_P99_US) -> tuple[bool, list[str]]:
    """The validity gate; returns (valid, reasons it failed)."""
    reasons = []
    checks = (("per-request server lag p99", summary["req_lag_p99_us"]),
              ("origin loop lag p99", server.get("loop_lag_us_p99")),
              ("client loop lag p99", client_lag.get("loop_lag_us_p99")))
    for label, value in checks:
        if value is None or not math.isfinite(value):
            reasons.append(f"{label} not measured")
        elif value > gate_us:
            reasons.append(f"{label} {value:.0f} us > {gate_us:.0f} us")
    if mismatched:
        reasons.append(f"{mismatched} parity mismatches")
    errors = summary["conn_errors"] + server.get("connection_errors", 0)
    if errors:
        reasons.append(f"{errors} connection errors")
    if server.get("bad_requests"):
        reasons.append(f"{server['bad_requests']} bad requests at the origin")
    return not reasons, reasons


def paired_delta(summary: dict, base: dict, *, reps: int = BOOTSTRAP_REPS,
                 seed: int = BOOTSTRAP_SEED):
    """Mean wait per view, policy minus baseline, with its paired session CI."""
    from prefetch_common.stats import session_bootstrap
    left = {sid: [w, 0, n, 0] for sid, (w, _, n) in summary["per_session"].items()}
    right = {sid: [w, 0, n, 0] for sid, (w, _, n) in base["per_session"].items()}
    if set(left) != set(right):
        raise ValueError("a pass and its baseline must replay the same sessions")
    return session_bootstrap({"p": left, "b": right}, "p", "b", cost=0.0, reps=reps, seed=seed)


def metrics_row(meta: dict, policy: str, baseline: str, summary: dict, base: dict | None,
                *, valid: bool, base_valid: bool, mismatched: int, server: dict,
                client_lag: dict, foreign_io_frac, load_avg: float,
                reps: int = BOOTSTRAP_REPS) -> dict:
    """One ``live/live_metrics.csv`` row."""
    n = summary["views"]
    total = summary["demand_bytes"] + summary["prefetch_bytes"]
    compare = valid and base_valid and base is not None
    if compare:
        ci = paired_delta(summary, base, reps=reps)
        ideal_delta = (sum(s[1] for s in summary["per_session"].values())
                       - sum(s[1] for s in base["per_session"].values())) / n
        base_total = base["demand_bytes"] + base["prefetch_bytes"]
        bytes_ratio = total / base_total if base_total else NAN
        delta, low, high = ci.estimate, ci.low, ci.high
    else:
        delta = low = high = ideal_delta = bytes_ratio = NAN
    server_lag = server.get("loop_lag_us_p99")
    client = client_lag.get("loop_lag_us_p99")
    return {
        **meta, "policy": policy, "baseline": baseline,
        "sessions": summary["sessions"], "views": n,
        "hit_ratio": summary["hits"] / summary["counted_views"] if summary["counted_views"] else NAN,
        "ready": summary["ready"], "inflight": summary["inflight"], "miss": summary["miss"],
        "mean_wait_ms": summary["mean_wait_ms"],
        "model_ideal_wait_ms": summary["model_ideal_wait_ms"],
        "excess_ms": summary["mean_wait_ms"] - summary["model_ideal_wait_ms"],
        "p95_wait_ms": summary["p95_wait_ms"],
        "delta_vs_base_ms": delta, "delta_vs_base_ms_lo": low, "delta_vs_base_ms_hi": high,
        "ideal_delta_vs_base_ms": ideal_delta,
        "kb_per_view": total / n / 1024 if n else NAN,
        "wasted_kb_per_view": ((summary["prefetch_bytes"] - summary["useful_prefetch_bytes"])
                               / n / 1024) if n else NAN,
        "bytes_vs_base": bytes_ratio,
        "demand_disk_us_mean": float(summary["demand_disk_us_mean"]),
        "server_lag_p99_us": float(server_lag) if server_lag is not None else NAN,
        "client_lag_p99_us": float(client) if client is not None else NAN,
        "req_lag_p99_us": float(summary["req_lag_p99_us"]),
        "conn_errors": int(summary["conn_errors"] + server.get("connection_errors", 0)),
        "parity_mismatches": int(mismatched), "valid": bool(valid),
        "foreign_io_frac": foreign_io_frac, "load_avg": float(load_avg),
    }


def parity_row(meta: dict, policy: str, record: dict) -> dict:
    """One ``live/live_parity.csv`` row."""
    keys = ("experiment", "store", "config", "rep")
    return {**{k: meta[k] for k in keys}, "policy": policy, **record}


def foreign_label(value) -> str:
    """``foreign_io_frac`` for the report, flagged above :data:`FOREIGN_FLAG`."""
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "not measured"
    return f"{value:.2%}" + (" (flagged)" if value > FOREIGN_FLAG else "")


def _fmt(value, spec: str) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "n/a"
    return format(value, spec)


def report_markdown(metrics: list[dict], parity_rows: list[dict], runs: dict) -> str:
    """``live_report.md``: every pass, its validity, parity and the decomposition."""
    lines = [
        "# Live HTTP prefetch replay", "",
        "Real HTTP/1.1 fetches over loopback from a real object store, with an **emulated**",
        "network (RTT, bandwidth) and **assumed** object sizes; see `live_config.json` for",
        "every assumption (ASSUMED) and measurement (MEASURED). Waits are measured per view.",
        "The model-ideal wait is 0 on a hit and rtt + size/bandwidth on a miss.",
        "Invalid passes (validity gate failed) are listed but never compared: their delta",
        "columns are n/a. Foreign I/O is the share of the store device's sectors read during",
        f"the pass that the origin did not read (other tenants); above {FOREIGN_FLAG:.0%} it is",
        "flagged, and an SSD row is cited only with it. Generated by",
        "`python -m webprefetch.live run`; do not edit by hand.",
        "",
    ]
    mismatches = sum(r["mismatched_sessions"] for r in parity_rows)
    lines += [f"Parity: {len(parity_rows)} passes, {mismatches} sessions whose live "
              "hits/issued/views/useful differ from the simulator.", ""]
    groups: dict[tuple, list[dict]] = {}
    for row in metrics:
        groups.setdefault((row["experiment"], row["store"], row["config"], row["link"]),
                          []).append(row)
    for (experiment, store, config, link), rows in sorted(groups.items()):
        lines += [f"## {experiment} / {store} / {config} / link {link}", "",
                  "| rep | policy | valid | views | hit | ready/inflight/miss | mean wait ms "
                  "| ideal ms | delta vs base ms [95% CI] | ideal delta | KiB/view | bytes x "
                  "| req lag p99 us | foreign I/O |",
                  "|---:|---|---|---:|---:|---|---:|---:|---|---:|---:|---:|---:|---:|"]
        for r in sorted(rows, key=lambda r: (r["rep"], r["policy"])):
            ci = (f"{_fmt(r['delta_vs_base_ms'], '+.2f')} [{_fmt(r['delta_vs_base_ms_lo'], '+.2f')}, "
                  f"{_fmt(r['delta_vs_base_ms_hi'], '+.2f')}]")
            lines.append(
                f"| {r['rep']} | {r['policy']} | {r['valid']} | {r['views']} | "
                f"{_fmt(r['hit_ratio'], '.4f')} | {r['ready']}/{r['inflight']}/{r['miss']} | "
                f"{_fmt(r['mean_wait_ms'], '.2f')} | {_fmt(r['model_ideal_wait_ms'], '.2f')} | "
                f"{ci} | {_fmt(r['ideal_delta_vs_base_ms'], '+.2f')} | "
                f"{_fmt(r['kb_per_view'], '.1f')} | {_fmt(r['bytes_vs_base'], '.2f')} | "
                f"{_fmt(r['req_lag_p99_us'], '.0f')} | {foreign_label(r.get('foreign_io_frac'))} |")
        lines.append("")
    lines += ["## Demand-fetch decomposition (ms, mean per demand fetch)", "",
              "| run | rep | policy | n | wall | rtt+tx | disk | link queue | timer lag | "
              "outside server | invalid because |",
              "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for tag, run in sorted(runs.items()):
        for p in run.get("passes", []):
            b = p.get("breakdown") or {}
            lines.append(
                f"| {tag} | {p['rep']} | {p['policy']} | {b.get('n', 0)} | "
                f"{_fmt(b.get('wall_ms'), '.2f')} | {_fmt(b.get('rtt_plus_tx_ms'), '.2f')} | "
                f"{_fmt(b.get('disk_ms'), '.3f')} | {_fmt(b.get('link_queue_ms'), '.2f')} | "
                f"{_fmt(b.get('server_timer_lag_ms'), '.3f')} | "
                f"{_fmt(b.get('outside_server_ms'), '.3f')} | "
                f"{'; '.join(p.get('invalid_reasons', [])) or '-'} |")
        for aborted in run.get("aborted", []):
            lines.append(f"| {tag} | {aborted['rep']} | {aborted['policy']} | 0 | n/a | n/a | "
                         f"n/a | n/a | n/a | n/a | aborted: {aborted['reason']} |")
    lines.append("")
    return "\n".join(lines)
