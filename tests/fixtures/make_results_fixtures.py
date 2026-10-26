"""Build the schema-exact FIXTURE results tree under tests/fixtures/results/.

    PYTHONPATH=src python tests/fixtures/make_results_fixtures.py

Every value here is **synthetic**. The files exist so the paper generators
(paper/numbers.py, paper/tables/make_tables.py, paper/figures/make_figures.py)
can be built and tested before any experiment has run; they are not results
and must never be quoted. The tree carries a ``FIXTURE`` marker file, and the
generators stamp everything built from it "FIXTURE DATA".

Rows are written through ``prefetch_common.schemas.write_rows``, so a schema
change breaks this builder rather than silently desynchronising the
fixtures, and ``tests/test_common.py`` validates the committed tree. The
numbers come from a seeded ``random.Random`` and a toy model chosen only so
that every code path has something to draw (a policy that beats ``none``, one
that loses, a skipped row, an infinite kappa*, an invalid live pass). The
output is deterministic: rerunning the builder rewrites identical bytes.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from prefetch_common import schemas, stats, utility  # noqa: E402
from prefetch_common.provenance import FORMAT, PROVENANCE_KEYS, hash_outputs  # noqa: E402

OUT = ROOT / "tests" / "fixtures" / "results"
NOTE = ("FIXTURE: synthetic values from tests/fixtures/make_results_fixtures.py; "
        "not a result, never quote")
REPS = 200  # bootstrap resamples: enough to exercise the code, cheap to build

TRACES = (("msr_hm_0", "msr"), ("msr_web_2", "msr"), ("fiu_homes", "fiu"),
          ("fiu_madmax", "fiu"), ("alibabaBlock_17", "alibaba"), ("tencent_1162", "tencent"))
CAPACITIES = (2048, 131072)
# (label, mode, depth, max_blocks, hit gain scale, prefetch blocks per read block)
C1_POLICIES = (
    ("none", "none", 2, None, 0.0, 0.0),
    ("sequential", "sequential", 2, None, 0.030, 0.30),
    ("adaptive", "adaptive", 2, None, 0.022, 0.28),
    ("adaptive_evidence", "adaptive_evidence", 2, None, 0.027, 0.27),
    ("guard", "guard", 2, None, 0.012, 0.08),
    ("deep@8", "deep", 8, None, 0.070, 1.10),
    ("deep@32", "deep", 32, None, 0.085, 3.20),
    ("deep@256", "deep", 256, None, 0.090, 14.0),
    ("linux_ra@256", "linux_ra", 2, 256, 0.075, 0.90),
)
C2_DEPTHS = (0, 2, 8, 32, 128, 512)


def _w(name: str, rows) -> None:
    schemas.write_rows(name, rows, OUT / name)


def _r(rng: random.Random, lo: float, hi: float) -> float:
    return round(rng.uniform(lo, hi), 6)


# ------------------------------------------------------------------- disk

def _sweep_row(rng, config, trace, corpus, capacity, label, mode, depth, max_blocks,
               gain, rate, base, latency="assumed", status="ok"):
    requests = 20000 + 1000 * len(trace)
    read_blocks = requests * 16
    row = {
        "config": config, "trace": trace, "corpus": corpus, "skip_requests": 0,
        "max_requests": 1_000_000 if config == "C1" else 250_000, "merge": "auto",
        "max_merge_sectors": 1024 if corpus == "fiu" else None,
        "size_unit": "sectors" if corpus == "fiu" else "bytes", "capacity": capacity,
        "mode": mode, "depth": depth, "max_blocks": max_blocks, "latency": latency,
        "hit_us": 5.0 if latency == "assumed" else 1.0, "miss_us": 100.0,
        "prefetch_us": 50.0 if latency == "assumed" else 100.0,
        "break_even": 50 / 95 if latency == "assumed" else 100 / 99, "status": status,
        "requests": requests, "blocks": read_blocks, "hours_covered": 24.0,
        "ops_known": False, "ts_resolution_s": 1.0, "belady_hit_ratio": None,
        "age_table_bound": None, "age_table_evictions": None, "model_seed": None,
        "git_sha": None, "load_start": 12.0, "load_end": 13.0, "wall_s": 1.5,
    }
    metrics = ("read_blocks", "hit_ratio", "precision", "total_prefetches", "useful_prefetches",
               "unused", "mean_us", "policy_switches", "observed_precision",
               "precision_margin", "observed_stride", "demand_ios", "prefetch_ios",
               "prefetch_ios_mergeable", "device_ios", "blocks_moved")
    if status != "ok":
        row.update({key: None for key in metrics})
        row["requests"] = row["blocks"] = row["hours_covered"] = None
        return row
    big = capacity > 2048
    hit = min(0.98, base + (0.08 if big else 0.0) + gain * _r(rng, 0.7, 1.3))
    issued = int(read_blocks * rate * _r(rng, 0.8, 1.2))
    useful = int(round((hit - base - (0.08 if big else 0.0)) * read_blocks))
    useful = max(0, min(issued, useful))
    hits = int(round(hit * read_blocks))
    misses = read_blocks - hits
    demand_ios = max(1, int(misses / 6))
    prefetch_ios = max(0, int(issued / max(depth, 2))) if issued else 0
    mergeable = prefetch_ios // 3
    if mode == "linux_ra" or (mode == "deep" and depth >= 128):
        # Large windows: few, long device I/Os, so kappa* is finite on some traces.
        demand_ios = max(1, demand_ios // 3)
        prefetch_ios = max(1, issued // (256 * 4))
    row.update({
        "read_blocks": read_blocks, "hit_ratio": hits / read_blocks,
        "precision": useful / issued if issued else 0.0, "total_prefetches": issued,
        "useful_prefetches": useful, "unused": issued - useful,
        "mean_us": (hits * row["hit_us"] + misses * row["miss_us"]
                    + issued * row["prefetch_us"]) / read_blocks,
        "policy_switches": 3 if mode in ("adaptive", "guard") else 0,
        "observed_precision": _r(rng, 0.2, 0.6) if mode == "guard" else None,
        "precision_margin": _r(rng, -0.4, 0.1) if mode == "guard" else None,
        "observed_stride": 8 if mode == "guard" else None,
        "demand_ios": demand_ios, "prefetch_ios": prefetch_ios,
        "prefetch_ios_mergeable": mergeable,
        "device_ios": demand_ios + prefetch_ios - mergeable,
        "blocks_moved": misses + issued,
        "belady_hit_ratio": (min(0.99, base + 0.12) if mode == "none" else None),
        "age_table_bound": 4_000_000 if mode == "adaptive" else None,
        "age_table_evictions": 0 if mode == "adaptive" else None,
        "model_seed": 42 if mode == "adaptive" else None,
    })
    return row


def build_disk(rng: random.Random) -> list[dict]:
    rows = []
    bases = {trace: _r(rng, 0.15, 0.45) for trace, _ in TRACES}
    for trace, corpus in TRACES:
        for capacity in CAPACITIES:
            for label, mode, depth, max_blocks, gain, rate in C1_POLICIES:
                rows.append(_sweep_row(rng, "C1", trace, corpus, capacity, label, mode, depth,
                                       max_blocks, gain, rate, bases[trace]))
            rows.append(_sweep_row(rng, "C1", trace, corpus, capacity, "guard", "guard", 2, None,
                                   0.004, 0.01, bases[trace], latency="break_even"))
            rows.append(_sweep_row(rng, "C1", trace, corpus, capacity, "lstm", "lstm", 2, None,
                                   0, 0, bases[trace], status="skipped: evaluated in E2 only"))
            for depth in C2_DEPTHS:
                gain = 0.09 * (1 - 1 / (1 + depth / 6)) - (0.03 if depth >= 512 else 0)
                rows.append(_sweep_row(rng, "C2", trace, corpus, capacity, f"deep@{depth}",
                                       "deep", depth, None, gain, depth * 0.05, bases[trace]))
    _w("disk/sweep.jsonl", rows)
    return rows


def _label(row) -> str:
    if row["mode"] == "deep":
        return f"deep@{row['depth']}"
    if row["mode"] == "linux_ra":
        return f"linux_ra@{row['max_blocks']}"
    return row["mode"]


def _wasted(row):
    return row["unused"] / row["total_prefetches"] if row["total_prefetches"] else None


def build_disk_tables(sweep: list[dict]) -> None:
    groups: dict[tuple, dict[str, dict]] = {}
    for row in sweep:
        key = (row["config"], row["merge"], row["capacity"], row["latency"])
        groups.setdefault(key, {}).setdefault(_label(row), {})[row["trace"]] = row
    # Gated rows at break_even are compared with the assumed-latency baselines.
    main, by_corpus, deltas, kappa = [], [], [], []
    for (config, merge, capacity, latency), policies in sorted(groups.items()):
        assumed = groups[(config, merge, capacity, "assumed")]
        none = assumed.get("none") or assumed.get("deep@0")
        seq = assumed.get("sequential")
        for label, by_trace in sorted(policies.items()):
            for corpus in ("all",) + tuple(sorted({c for _, c in TRACES})):
                traces = [t for t, c in TRACES if corpus in ("all", c)]
                ok = [t for t in traces if by_trace[t]["status"] == "ok"]
                pol = [by_trace[t]["hit_ratio"] for t in ok]
                table = {"config": config, "merge": merge, "capacity": capacity,
                         "latency": latency, "corpus": corpus, "policy": label,
                         "n_traces": len(ok)}
                if not ok:
                    table.update({k: None for k in schemas.columns("disk/table_main.csv")
                                  if k not in table})
                else:
                    blocks = [by_trace[t]["read_blocks"] for t in ok]
                    d_none = stats.paired_bootstrap(pol, [none[t]["hit_ratio"] for t in ok],
                                                    reps=REPS)
                    d_seq = (stats.paired_bootstrap(pol, [seq[t]["hit_ratio"] for t in ok],
                                                    reps=REPS) if seq else None)
                    issued = [t for t in ok if by_trace[t]["total_prefetches"]]
                    won, lost, _ = stats.wins(pol, [none[t]["hit_ratio"] for t in ok])
                    table.update({
                        "mean_hit_ratio": sum(pol) / len(pol),
                        "weighted_hit_ratio": sum(p * b for p, b in zip(pol, blocks)) / sum(blocks),
                        "delta_vs_none": d_none.estimate, "delta_vs_none_lo": d_none.low,
                        "delta_vs_none_hi": d_none.high,
                        "delta_vs_sequential": d_seq.estimate if d_seq else None,
                        "delta_vs_sequential_lo": d_seq.low if d_seq else None,
                        "delta_vs_sequential_hi": d_seq.high if d_seq else None,
                        "mean_precision": (sum(by_trace[t]["precision"] for t in issued)
                                           / len(issued)) if issued else None,
                        "mean_wasted_share": (sum(_wasted(by_trace[t]) for t in issued)
                                              / len(issued)) if issued else None,
                        "device_io_ratio": sum(by_trace[t]["device_ios"] / none[t]["device_ios"]
                                               for t in ok) / len(ok),
                        "blocks_moved_ratio": sum(by_trace[t]["blocks_moved"]
                                                  / none[t]["blocks_moved"] for t in ok) / len(ok),
                        "wins_vs_none": won, "losses_vs_none": lost,
                    })
                (main if corpus == "all" else by_corpus).append(table)
                if not ok or label == "none" or config != "C1":
                    continue
                for metric, statistic, baseline, values, base_values in (
                    ("hit_ratio", "difference", "none", pol, [none[t]["hit_ratio"] for t in ok]),
                    ("hit_ratio", "difference", "sequential", pol,
                     [seq[t]["hit_ratio"] for t in ok]),
                    ("device_ios", "ratio", "none", [by_trace[t]["device_ios"] for t in ok],
                     [none[t]["device_ios"] for t in ok]),
                    ("blocks_moved", "ratio", "none", [by_trace[t]["blocks_moved"] for t in ok],
                     [none[t]["blocks_moved"] for t in ok]),
                ):
                    if label == baseline:
                        continue
                    ci = stats.paired_bootstrap(values, base_values, reps=REPS,
                                                statistic=statistic)
                    effects = ([v - b for v, b in zip(values, base_values)]
                               if statistic == "difference"
                               else [v / b for v, b in zip(values, base_values)])
                    corpora = [dict(TRACES)[t] for t in ok]
                    cluster = stats.cluster_bootstrap(effects, corpora, reps=REPS)
                    tie = 0.0 if statistic == "difference" else 1.0
                    won, lost, ties = stats.wins(effects, [tie] * len(effects))
                    if metric != "hit_ratio":  # fewer device I/Os is the win
                        won, lost = lost, won
                    deltas.append({
                        "config": config, "merge": merge, "capacity": capacity,
                        "latency": latency, "corpus": corpus, "metric": metric,
                        "statistic": statistic, "policy": label, "baseline": baseline,
                        "n": len(ok), "estimate": ci.estimate, "ci_lo": ci.low, "ci_hi": ci.high,
                        "cluster_estimate": cluster.estimate if corpus == "all" else None,
                        "cluster_ci_lo": cluster.low if corpus == "all" else None,
                        "cluster_ci_hi": cluster.high if corpus == "all" else None,
                        "twins": "none",
                        "wilcoxon_p": round(0.5 / (1 + len(ok) * abs(ci.estimate) * 50), 6),
                        "wins": won, "losses": lost, "ties": ties, "reps": REPS, "seed": 42,
                    })
                if corpus == "all" and latency == "assumed":
                    for t in ok:
                        p, n = by_trace[t], none[t]
                        d_ios = p["device_ios"] - n["device_ios"]
                        d_blocks = p["blocks_moved"] - n["blocks_moved"]
                        k = (max(0.0, d_blocks / -d_ios) if d_ios < 0
                             else 0.0 if d_ios == 0 and d_blocks < 0 else float("inf"))
                        kappa.append({"config": config, "merge": merge, "capacity": capacity,
                                      "trace": t, "corpus": dict(TRACES)[t], "policy": label,
                                      "ios_policy": p["device_ios"], "ios_none": n["device_ios"],
                                      "blocks_policy": p["blocks_moved"],
                                      "blocks_none": n["blocks_moved"], "delta_ios": d_ios,
                                      "delta_blocks": d_blocks, "kappa_star": k})
    _w("disk/table_main.csv", main)
    _w("disk/table_by_corpus.csv", by_corpus)
    _w("disk/deltas.csv", deltas)
    _w("disk/kappa.csv", kappa)


def build_corpus_profile(rng) -> None:
    rows = []
    for trace, corpus in TRACES:
        records = int(_r(rng, 1e6, 5e6))
        requests = records if corpus != "fiu" else records // 3
        rows.append({
            "trace": trace, "corpus": corpus,
            "file": f"data/block-traces/{corpus}/{trace}.oracleGeneral.zst",
            "size_unit": "sectors" if corpus == "fiu" else "bytes", "merge": "auto",
            "max_merge_sectors": 1024 if corpus == "fiu" else None, "records": records,
            "requests": requests, "blocks": requests * 16, "footprint_blocks": requests * 4,
            "mean_request_blocks": 16.0, "median_request_blocks": 8.0,
            "contiguous_fraction": _r(rng, 0.05, 0.6), "hours_covered": _r(rng, 20, 170),
            "tagged": corpus in ("alibaba", "tencent"),
            "twin_group": "fiu-homes-family" if trace == "fiu_homes" else None,
            "paper_subset": trace != "fiu_madmax",
        })
    _w("disk/corpus_profile.csv", rows)


def build_learned_depth(rng) -> None:
    rows = []
    for trace, corpus in TRACES[:4]:
        none_hit = _r(rng, 0.2, 0.4)
        for cap in (64, 256):
            for seed in range(3):
                vcf = none_hit + _r(rng, 0.03, 0.08)
                rows.append({
                    "trace": trace, "corpus": corpus, "seed": seed, "status": "ok",
                    "action_cap": cap, "capacity": 2048, "n_requests": 2_000_000,
                    "train_start": 0, "train_end": 1_200_000, "val_start": 1_200_000,
                    "val_end": 1_600_000, "test_start": 1_600_000, "test_end": 2_000_000,
                    "thresholds": {"segments": [0.4, 0.6]},
                    "test_hit_ratio": vcf + _r(rng, -0.01, 0.03), "test_issued": 400_000,
                    "test_precision": _r(rng, 0.2, 0.5), "vcf_depth": 32,
                    "vcf_hit_ratio": vcf, "vcf_issued": 500_000, "relsize_k": 2.0,
                    "relsize_hit_ratio": vcf - 0.005, "deep8_hit_ratio": vcf - 0.01,
                    "none_hit_ratio": none_hit, "oracle_same_actions_hit_ratio": vcf + 0.05,
                    "device": "cuda", "train_s": _r(rng, 30, 90),
                    "infer_us_per_decision": _r(rng, 1, 5),
                    "gpu_contention": "fixture: no GPU was used",
                })
    rows.append({**rows[0], "trace": "fiu_madmax", "corpus": "fiu", "status":
                 "skipped: fewer than 2M canonical requests",
                 **{k: None for k in ("thresholds", "test_hit_ratio", "test_issued",
                                      "test_precision", "vcf_depth", "vcf_hit_ratio",
                                      "vcf_issued", "relsize_k", "relsize_hit_ratio",
                                      "deep8_hit_ratio", "none_hit_ratio",
                                      "oracle_same_actions_hit_ratio", "train_s",
                                      "infer_us_per_decision")}})
    _w("gpu/disk_learned_depth.csv", rows)


# ----------------------------------------------------------------- realio

DEVICES = {"sdb": (True, 5900.0, 2.0), "ssd": (False, 120.0, 1.1)}
REALIO_MODES = (("odirect", "none"), ("odirect", "deep@8"), ("odirect", "linux_ra"),
                ("odirect", "guard"), ("pagecache", "none"), ("pagecache", "deep@8"),
                ("pagecache", "kernel_ra"))
THINKS = ("closed", "spread:1:5000")


def build_realio(rng) -> None:
    for device, (rotational, t_pos, t_xfer) in DEVICES.items():
        _w(f"realio/device_{device}.json", [{
            "device": device, "path": f"/fixture/{device}/backing.bin",
            "model": "FIXTURE-HDD" if rotational else "FIXTURE-SSD", "rotational": rotational,
            "read_ahead_kb": 128, "max_sectors_kb": 256, "rand4k_p50_us": t_pos,
            "rand4k_mean_us": t_pos * 1.05, "seq4k_p50_us": 30.0, "seq64k_p50_us": 230.0,
            "seq1m_MBps": 250.0, "t_pos_us": t_pos, "t_xfer_us_per_block": t_xfer,
            "kappa": t_pos / t_xfer, "samples": 2000, "foreign_io_frac": 0.0,
            "load_avg": [3.0, 3.0, 3.0], "created_utc": "2026-01-01T00:00:00+00:00",
        }])
    raw, summary = [], []
    for device, (rotational, t_pos, _) in DEVICES.items():
        for trace in ("msr_hm_0", "msr_web_2"):
            for think in THINKS:
                for backend, mode in REALIO_MODES:
                    kernel = mode.startswith("kernel")
                    sim_hit = None if kernel else _r(rng, 0.25, 0.4)
                    real_hit = sim_hit if backend == "odirect" else _r(rng, 0.2, 0.4)
                    lat = []
                    for repeat in range(3):
                        mean = t_pos * (1 - real_hit) * _r(rng, 0.9, 1.1)
                        lat.append(mean)
                        raw.append({
                            "device": device, "backend": backend, "mode": mode, "think": think,
                            "repeat": repeat, "trace": trace, "requests": 20000,
                            "sim_hit_ratio": sim_hit, "sim_precision": None if kernel else 0.4,
                            "sim_device_ios": None if kernel else 9000,
                            "real_hit_ratio": real_hit, "full_hit_request_frac": real_hit * 0.9,
                            "ready_hits": 5000, "late_hits": 0 if think != "closed" else 120,
                            "late_wait_us": 1500.0, "lat_mean_us": mean,
                            "lat_p50_us": mean * 0.5, "lat_p95_us": mean * 2.5,
                            "lat_p99_us": mean * 4.0, "lat_max_us": mean * 9.0,
                            "device_MiB": 120.0, "prefetch_MiB": 0.0 if mode == "none" else 40.0,
                            "think_total_s": 0.0 if think == "closed" else 55.0,
                            "think_error_us_p99": None if think == "closed" else 80.0,
                            "wall_s": 13.0, "cg_file_MiB": 31.0 if backend == "pagecache" else None,
                            "cg_events_max": 0 if backend == "pagecache" else None,
                            "foreign_io_frac": 0.0 if rotational else 0.03, "wrapped": 0,
                            "pred_mean_us": mean * 0.8, "pred_err": -0.2, "k_eff": 0.9,
                            "load_avg": 4.0,
                        })
                    ci = stats.paired_bootstrap(lat, [0.0] * 3, reps=REPS)
                    summary.append({
                        "device": device, "backend": backend, "mode": mode, "think": think,
                        "trace": trace, "requests": 20000, "reps": 3, "sim_hit_ratio": sim_hit,
                        "sim_device_ios": None if kernel else 9000,
                        "sim_device_time_us": None if kernel else 9000 * t_pos,
                        "real_hit_ratio": real_hit, "full_hit_request_frac": real_hit * 0.9,
                        "lat_mean_us": ci.estimate, "lat_mean_us_lo": ci.low,
                        "lat_mean_us_hi": ci.high, "lat_p50_us": ci.estimate * 0.5,
                        "lat_p99_us": ci.estimate * 4.0,
                        "late_hits": 0.0 if think != "closed" else 120.0, "device_MiB": 120.0,
                        "pred_mean_us": ci.estimate * 0.8, "pred_err": -0.2, "k_eff": 0.9,
                        "spearman_device_time": 0.6, "spearman_hit_ratio": -0.4,
                        "foreign_io_frac_max": 0.0 if rotational else 0.03, "load_avg_max": 4.0,
                    })
    _w("realio/realio.csv", raw)
    _w("realio/realio_summary.csv", summary)


# -------------------------------------------------------------------- web

WEB_POLICIES = (("fixed0", 0.0), ("fixed1", 1.0), ("fixed2", 2.0), ("fixed4", 4.0),
                ("fixed8", 8.0), ("adaptive_0.01", 1.6), ("feedback_0.01", 1.2))


def build_web(rng) -> None:
    rows = []
    for split in ("validation", "test"):
        for scorer, boost in (("logistic", 0.0), ("trans_online", 0.012)):
            for capacity in (2, 8, 32):
                base = 0.105 + 0.004 * capacity ** 0.5
                for policy, depth in WEB_POLICIES:
                    views = 445_796 if split == "test" else 300_000
                    gain = (0.03 + boost) * (1 - 1 / (1 + depth)) - 0.0015 * max(0, depth - capacity)
                    issued = depth * _r(rng, 0.9, 1.0)
                    hit = base + gain
                    precision = (gain / issued) if issued else 0.0
                    record = {"policy": policy, "views": views, "hit_ratio": hit,
                              "issued_per_view": issued}
                    rows.append({
                        "split": split, "scorer": scorer, "cache_scope": "session",
                        "capacity": capacity, "policy": policy, "views": views, "hit_ratio": hit,
                        "prefetch_hits_per_view": gain, "issued_per_view": issued,
                        "precision": precision, "wasted_per_view": issued - gain,
                        "demand_evicted_by_prefetch_per_view": 0.001 * depth,
                        "c_star": (gain / issued) if issued else None,
                        **utility.utility_columns(record),
                    })
    _w("web/policy_metrics.csv", rows)

    comparisons = []
    for selection in ("per_capacity", "capacity_transfer"):
        for capacity in (2, 8, 32):
            for cost in (0.0, 0.01, 0.05):
                d_best = _r(rng, -0.002, 0.006)
                d0 = d_best + _r(rng, 0.005, 0.02)
                dt = _r(rng, -0.003, 0.004)
                comparisons.append({
                    "split": "test", "selection": selection, "scorer": "logistic",
                    "capacity": capacity, "cost": cost, "adaptive_policy": "adaptive_0.01",
                    "best_fixed_policy": "fixed2", "adaptive_utility": 0.13 + d_best,
                    "best_fixed_utility": 0.13, "fixed0_utility": 0.13 + d_best - d0,
                    "transition_top1_utility": 0.13 + d_best - dt,
                    "vs_best_fixed_same_cap": d_best, "vs_best_fixed_same_cap_lo": d_best - 0.001,
                    "vs_best_fixed_same_cap_hi": d_best + 0.001, "vs_fixed0": d0,
                    "vs_fixed0_lo": d0 - 0.001, "vs_fixed0_hi": d0 + 0.001,
                    "vs_transition_top1": dt, "vs_transition_top1_lo": dt - 0.001,
                    "vs_transition_top1_hi": dt + 0.001, "adaptive_hit_ratio": 0.14,
                    "best_fixed_hit_ratio": 0.138, "adaptive_issued_per_view": 1.6,
                    "best_fixed_issued_per_view": 2.0,
                    "label": "adaptive wins" if d_best - 0.001 > 0 else "no clear winner",
                    "reps": REPS, "seed": 42,
                })
    _w("web/comparisons.csv", comparisons)

    levers = (("ranker", "trans_s0_k1", "logistic_k1", "trans_s0", "logistic", 8, 0.0),
              ("freshness", "fixed4", "fixed4", "trans_online", "logistic", 8, 0.0),
              ("depth", "fixed4", "fixed1", "logistic", "logistic", 8, 0.0),
              ("controller", "adaptive_0.01", "fixed2", "logistic", "logistic", 8, 0.01),
              ("controller", "adaptive_0.01", "fixed4", "logistic", "logistic", 32, 0.01),
              ("feedback", "feedback_0.01", "adaptive_0.01", "logistic", "logistic", 8, 0.01),
              ("cache_scope", "fixed4", "fixed4", "logistic", "logistic", None, 0.0),
              ("sequence_model", "fixed4", "fixed4", "seq", "trans_online", 8, 0.0))
    decomposition = []
    for lever, policy, baseline, scorer, base_scorer, capacity, cost in levers:
        estimate = _r(rng, -0.002, 0.02)
        decomposition.append({
            "lever": lever, "policy": policy, "baseline": baseline, "scorer": scorer,
            "baseline_scorer": base_scorer, "split": "test", "capacity": capacity,
            "cache_scope": "visitor" if lever == "cache_scope" else "session",
            "metric": "hit_ratio" if cost == 0 else "utility_per_view", "cost": cost,
            "estimate": estimate, "ci_lo": estimate - 0.0008, "ci_hi": estimate + 0.0008,
            "sessions": 120_000, "reps": REPS, "seed": 42,
            "note": "fixture" if lever == "sequence_model" else None,
        })
    _w("web/decomposition.csv", decomposition)

    freshness = []
    for index, scale in (("split0", 0.6), ("split012", 0.8), ("online", 1.0)):
        for k in (1, 2, 4, 8, 16, None):
            coverage = scale * (0.35 if k is None else 0.3 * (1 - 1 / (1 + k / 3)))
            freshness.append({"split": "test", "index": index, "k": k, "coverage": coverage,
                              "eligible_pairs": 250_000, "sources_with_transitions": 90_000})
    _w("web/freshness.csv", freshness)

    seq = []
    for protocol in ("validation", "final"):
        for seed in range(3):
            seq.append({
                "model": "gru", "protocol": protocol, "seed": seed, "status": "ok",
                "split": "test" if protocol == "final" else "split2",
                "train_range": "splits 0-2" if protocol == "final" else "split 0 + 90% of split 1",
                "epochs": 20, "hit_at_1": _r(rng, 0.03, 0.05), "mrr_at_8": _r(rng, 0.05, 0.07),
                "novel_hit_at_1": _r(rng, 0.02, 0.04), "novel_recall_at_8": _r(rng, 0.12, 0.15),
                "capacity": 8, "policy": "fixed4", "cost": 0.0, "hit_ratio": _r(rng, 0.13, 0.15),
                "issued_per_view": 3.9, "utility": _r(rng, 0.13, 0.15), "train_s": 260.0,
                "infer_us_per_request": 4.0, "device": "cuda",
                "gpu_contention": "fixture: no GPU was used",
            })
    _w("web/seq_metrics.csv", seq)


# ------------------------------------------------------------------- live

LIVE_POLICIES = ("fixed0", "fixed1", "fixed4", "fixed8", "adaptive_0.01", "feedback_0.01")


def build_live(rng) -> None:
    rows, parity = [], []
    for experiment, stores, link in (("headline", ("ssd", "hdd"), "priority"),
                                     ("fifo", ("ssd",), "fifo")):
        for store in stores:
            for rep in range(3 if experiment == "headline" else 1):
                for policy in LIVE_POLICIES:
                    depth = {"fixed0": 0, "fixed1": 1, "fixed4": 4, "fixed8": 8}.get(policy, 2)
                    saved = 0.0 if depth == 0 else (3.0 * (1 - 1 / (1 + depth))
                                                    - (1.5 if link == "fifo" else 0) * depth / 8)
                    saved *= _r(rng, 0.9, 1.1)
                    delta = -saved
                    valid = not (experiment == "headline" and store == "hdd" and rep == 2
                                 and policy == "fixed8")
                    rows.append({
                        "experiment": experiment, "store": store,
                        "config": f"rtt40-mbps50-{link}", "link": link, "rtt_ms": 40.0,
                        "mbps": 50.0, "size_median_kib": 60.0, "gap_cap_s": 10.0, "ramp_s": 60.0,
                        "rep": rep, "policy": policy, "baseline": "fixed0", "sessions": 200,
                        "views": 900, "hit_ratio": 0.1 + 0.01 * depth, "ready": 80, "inflight": 9,
                        "miss": 811, "mean_wait_ms": 105.0 + delta,
                        "model_ideal_wait_ms": 104.0 + delta * 1.1, "excess_ms": 1.0,
                        "p95_wait_ms": 180.0, "delta_vs_base_ms": delta,
                        "delta_vs_base_ms_lo": delta - 0.4, "delta_vs_base_ms_hi": delta + 0.4,
                        "ideal_delta_vs_base_ms": delta * 1.1,
                        "kb_per_view": 60.0 * (1 + 0.6 * depth),
                        "wasted_kb_per_view": 60.0 * 0.55 * depth,
                        "bytes_vs_base": 1 + 0.6 * depth, "demand_disk_us_mean": 300.0,
                        "server_lag_p99_us": 200.0, "client_lag_p99_us": 250.0,
                        "req_lag_p99_us": 4000.0 if valid else 25000.0, "conn_errors": 0,
                        "parity_mismatches": 0, "valid": valid,
                        "foreign_io_frac": 0.0, "load_avg": 5.0,
                    })
                    parity.append({
                        "experiment": experiment, "store": store,
                        "config": f"rtt40-mbps50-{link}", "rep": rep, "policy": policy,
                        "sessions": 200, "sim_hits": 90 + 9 * depth, "live_hits": 90 + 9 * depth,
                        "sim_issued": 200 * depth, "live_issued": 200 * depth,
                        "sim_useful": 9 * depth, "live_useful": 9 * depth,
                        "mismatched_sessions": 0,
                    })
    _w("live/live_metrics.csv", rows)
    _w("live/live_parity.csv", parity)


# ------------------------------------------------------------- provenance

def write_provenance(study: str) -> None:
    record = {key: None for key in PROVENANCE_KEYS}
    record.update({
        "format": FORMAT, "study": study,
        "command": ["python", "tests/fixtures/make_results_fixtures.py"],
        "git_sha": "0" * 40, "git_dirty": False, "created_utc": "2026-01-01T00:00:00+00:00",
        "host": {"hostname": "fixture-host", "platform": "Linux-fixture", "kernel": "6.14.0",
                 "python": "3.12.0", "cpu_count": 128},
        "load_avg": {"start": [1.0, 1.0, 1.0], "end": [1.0, 1.0, 1.0]}, "wall_s": 0.0,
        "gpu": [{"index": "0", "name": "FIXTURE GPU", "memory.used": "0",
                 "memory.total": "1", "utilization.gpu": "0"}] if study == "gpu" else None,
        "inputs": {}, "outputs": hash_outputs(OUT / study), "notes": NOTE,
    })
    schemas.write_rows(f"{study}/provenance.json", [record], OUT / study / "provenance.json")


def main() -> int:
    rng = random.Random(20261026)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "FIXTURE").write_text(NOTE + "\n", encoding="utf-8")
    (OUT / "README.md").write_text(
        "# Results fixtures (synthetic)\n\n"
        "Every file here is **synthetic test data** with the exact column layout of\n"
        "`prefetch_common.schemas`. It exists so the paper generators can be built and\n"
        "tested before the experiments run. **None of it is a result; never quote it.**\n"
        "Rebuild with `PYTHONPATH=src python tests/fixtures/make_results_fixtures.py`.\n"
        "The `FIXTURE` marker makes the generators stamp their output \"FIXTURE DATA\".\n",
        encoding="utf-8")
    sweep = build_disk(rng)
    build_disk_tables(sweep)
    build_corpus_profile(rng)
    build_learned_depth(rng)
    build_realio(rng)
    build_web(rng)
    build_live(rng)
    for study in ("disk", "gpu", "realio", "web", "live"):
        write_provenance(study)
    print(json.dumps(sorted(p.relative_to(OUT).as_posix() for p in OUT.rglob("*") if p.is_file()),
                     indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
