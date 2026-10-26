"""Orchestrate live passes: plan, store, origin, sequential passes, analysis.

One invocation is one *run*: a fixed experiment label, store, network and
client configuration, and a policy list (the baseline is always added). It

1. builds (or reuses) the plan and its simulator reference,
2. builds (or reuses) the object store and probes its page-cache behaviour,
3. refuses to start when the planned peak offered rate exceeds
   ``--max-rps-per-shard`` x ``--shards`` (loopback timing was measured to be
   accurate below about 250-500 req/s per event loop and to saturate at
   about 2.2k),
4. starts the origin, then runs the passes **one at a time**: ``--reps``
   repetitions, the policy order rotated each repetition, the store's pages
   dropped before every pass, and a pass aborted unless the 1-minute load
   average is below ``--max-load`` (the host is shared),
5. checks parity and validity, and merges the rows into
   ``live_metrics.csv`` and ``live_parity.csv`` (rows with the same
   experiment, store, config, link, rep and policy are replaced), updates
   ``live_config.json`` and ``live_report.md``, and writes provenance last.

Raw per-view and per-fetch records go to ``<work-dir>/raw/`` as ``.csv.gz``
and are not committed.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import gzip
import hashlib
import json
import multiprocessing
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

from ..paths import MODELS, OUTPUTS, PROCESSED
from . import analyze, client, objects, plan as plan_module
from .client import ClientConfig
from .origin import origin_stats
from .timing import CLOCK

RESULTS = Path("results/live")
WORK = OUTPUTS / "live"
CONFIG_NAME = "live_config.json"
REPORT_NAME = "live_report.md"
METRICS_KEY = "live/live_metrics.csv"
PARITY_KEY = "live/live_parity.csv"
EXPERIMENTS = ("headline", "fifo", "hint", "sensitivity", "smoke")
ROW_KEY = ("experiment", "store", "config", "link", "rep", "policy")
EXIT_REFUSED = 2
EXIT_PARITY = 1


def add_arguments(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("plan")
    g.add_argument("--views", type=Path, default=PROCESSED / "views.csv.gz")
    g.add_argument("--models", type=Path, default=MODELS)
    g.add_argument("--policies", default=",".join(plan_module.HEADLINE_POLICIES),
                   help="comma list; 'hint:<policy>' sends that policy's candidates to /hint")
    g.add_argument("--baseline", default=plan_module.BASELINE)
    g.add_argument("--sessions", type=int, default=plan_module.DEFAULT_SESSIONS)
    g.add_argument("--min-views", type=int, default=plan_module.DEFAULT_MIN_VIEWS)
    g.add_argument("--seed", type=int, default=plan_module.DEFAULT_SEED)
    g.add_argument("--split", type=int, default=plan_module.DEFAULT_SPLIT)
    g.add_argument("--capacity", type=int, default=plan_module.DEFAULT_CAPACITY)
    g = parser.add_argument_group("store")
    g.add_argument("--work-dir", type=Path, default=WORK, help="plans and raw records")
    g.add_argument("--store-dir", type=Path, default=None,
                   help="object store directory (default <work-dir>/store, on the repo's disk)")
    g.add_argument("--store-label", default=None, help="ssd | hdd; default from sysfs")
    g.add_argument("--size-median", type=int, default=objects.SIZE_MEDIAN, help="ASSUMED bytes")
    g.add_argument("--size-sigma", type=float, default=objects.SIZE_SIGMA, help="ASSUMED")
    g.add_argument("--no-probe", action="store_true", help="skip disk_probe_<store>.json")
    g = parser.add_argument_group("network (all ASSUMED)")
    g.add_argument("--rtt-ms", type=float, default=50.0)
    g.add_argument("--mbps", type=float, default=10.0)
    g.add_argument("--server-delay-ms", type=float, default=0.0)
    g.add_argument("--link", choices=("priority", "fifo"), default="priority")
    g.add_argument("--io-threads", type=int, default=8)
    g = parser.add_argument_group("client")
    g.add_argument("--pool", choices=client.POOLS, default="priority")
    g.add_argument("--conns", type=int, default=client.DEFAULT_CONNS)
    g.add_argument("--prefetch-conns", type=int, default=client.DEFAULT_PREFETCH_CONNS)
    g.add_argument("--handshake-ms", type=float, default=0.0)
    g.add_argument("--gap-cap-s", type=float, default=client.DEFAULT_GAP_CAP_S)
    g.add_argument("--think-scale", type=float, default=1.0)
    g.add_argument("--ramp-s", type=float, default=client.DEFAULT_RAMP_S)
    g.add_argument("--target-rps", type=float, default=None,
                   help="pace session starts instead of a uniform ramp")
    g.add_argument("--start-seed", type=int, default=1)
    g.add_argument("--shards", type=int, default=1,
                   help="client processes, each from its own 127.0.0.x source address")
    g.add_argument("--max-rps-per-shard", type=float, default=300.0)
    g = parser.add_argument_group("run")
    g.add_argument("--experiment", choices=EXPERIMENTS, default="headline")
    g.add_argument("--reps", type=int, default=3)
    g.add_argument("--results-dir", type=Path, default=RESULTS)
    g.add_argument("--max-load", type=float, default=32.0)
    g.add_argument("--load-wait-s", type=float, default=600.0,
                   help="how long to wait for a quiet host before aborting a pass")
    g.add_argument("--gate-us", type=float, default=analyze.GATE_LAG_P99_US)
    g.add_argument("--bootstrap-reps", type=int, default=analyze.BOOTSTRAP_REPS)
    g.add_argument("--no-raw", action="store_true", help="do not keep per-view .csv.gz")


def config_label(args) -> str:
    label = (f"rtt{args.rtt_ms:g}_bw{args.mbps:g}_med{args.size_median / 1024:g}k"
             f"_cap{args.gap_cap_s:g}")
    if args.link != "priority":
        label += f"_link{args.link}"
    if args.pool != "priority":
        label += f"_pool{args.pool}"
    if args.server_delay_ms:
        label += f"_delay{args.server_delay_ms:g}"
    return label


def _safe(name: str) -> str:
    return name.replace(":", "-").replace("@", "-at-").replace("/", "-")


def resolve_policies(names: str, baseline: str) -> tuple[list[str], list[str]]:
    """(pass labels in order with the baseline first, plan policy names)."""
    labels = [n for n in names.split(",") if n]
    if baseline not in labels:
        labels.insert(0, baseline)
    if len(set(labels)) != len(labels):
        raise ValueError("policies must be unique")
    base_names = []
    for label in labels:
        name = label[len("hint:"):] if label.startswith("hint:") else label
        if name not in base_names:
            base_names.append(name)
    plan_module.parse_policies(base_names)  # fail early on a bad name
    return labels, base_names


def _session_gap_ms() -> int:
    from ..data import SESSION_GAP_MS
    return SESSION_GAP_MS


def ensure_plan(args, base_names: list[str]) -> tuple[Path, dict, dict]:
    params = {"splits": [args.split], "sessions_requested": args.sessions,
              "min_views": args.min_views, "seed": args.seed, "capacity": args.capacity,
              "session_gap_ms": _session_gap_ms(),
              "policies": [p.record() for p in plan_module.parse_policies(base_names)]}
    digest = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()
    directory = Path(args.work_dir) / f"plan_{digest[:12]}"
    if plan_module.plan_matches(directory, params):
        plan, sim = plan_module.load_plan(directory)
        print(f"[live] reusing plan {directory}", flush=True)
        return directory, plan, sim
    from ..data import load_views
    started = time.perf_counter()
    views = load_views(args.views)
    policies = plan_module.parse_policies(base_names)
    scorers = plan_module.make_scorers(views, {p.scorer for p in policies},
                                       models_dir=args.models, split=args.split)
    plan, sim = plan_module.build_plan(views, policies, scorers, split=args.split,
                                       sessions=args.sessions, min_views=args.min_views,
                                       seed=args.seed, capacity=args.capacity)
    plan_module.write_plan(plan, sim, directory)
    print(f"[live] plan {directory}: {plan['sessions_selected']} sessions, {plan['views']} views, "
          f"{len(plan['items'])} objects, {time.perf_counter() - started:.1f} s", flush=True)
    return directory, plan, sim


class OriginProcess:
    """The origin server as a child process (its own event loop and CPU)."""

    def __init__(self, store: str, index: str, args, log: Path):
        command = [sys.executable, "-m", "webprefetch.live", "origin", "--store", store,
                   "--index", index, "--rtt-ms", str(args.rtt_ms), "--mbps", str(args.mbps),
                   "--link", args.link, "--io-threads", str(args.io_threads),
                   "--server-delay-ms", str(args.server_delay_ms)]
        env = dict(os.environ)
        src = str(Path(__file__).resolve().parents[2])
        env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        log.parent.mkdir(parents=True, exist_ok=True)
        self.log = open(log, "a")
        self.process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=self.log,
                                        env=env, text=True)
        line = self.process.stdout.readline()
        if not line:
            self.close()
            raise RuntimeError(f"origin failed to start; see {log}")
        self.ready = json.loads(line)
        self.port = int(self.ready["port"])
        self.pid = self.process.pid

    def stats(self, reset: bool = False) -> dict:
        return origin_stats("127.0.0.1", self.port, reset)

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        if self.process.stdout:
            self.process.stdout.close()
        self.log.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def wait_for_quiet(max_load: float, wait_s: float, poll_s: float = 10.0) -> tuple[bool, float]:
    deadline = time.monotonic() + wait_s
    while True:
        load = os.getloadavg()[0]
        if load < max_load:
            return True, load
        if time.monotonic() >= deadline:
            return False, load
        print(f"[live] load {load:.1f} >= {max_load:g}; waiting", flush=True)
        time.sleep(poll_s)


def execute(sessions: dict, label: str, config: ClientConfig, starts: dict, shards: int,
            tag: str) -> tuple[list, dict]:
    """Run one pass, in-process or over ``shards`` client processes."""
    policy, hint = analyze.reference_policy(label)
    if shards <= 1:
        return client.run_pass(sessions, policy, config, starts, hint=hint, tag=tag)
    ids = sorted(sessions, key=int)
    payloads = []
    t0 = CLOCK() + 3.0  # spawned interpreters need time to import
    for k in range(shards):
        part = {sid: sessions[sid] for sid in ids[k::shards]}
        shard_config = {**asdict(config), "local_addr": f"127.0.0.{2 + k}"}
        payloads.append({"sessions": part, "policy": policy, "config": shard_config,
                         "starts": {sid: starts[sid] for sid in part}, "t0": t0,
                         "hint": hint, "tag": tag})
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(shards, mp_context=context) as pool:
        outputs = list(pool.map(client.run_shard, payloads))
    results = [r for out, _ in outputs for r in out]
    lags = [lag for _, lag in outputs]
    worst = max(lags, key=lambda lag: lag.get("loop_lag_us_p99") or float("inf"))
    return results, {**worst, "shards": lags}


def _write_raw(directory: Path, stem: str, results: list[dict]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    with gzip.open(directory / f"{stem}.views.csv.gz", "wt", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("sid",) + client.VIEW_FIELDS)
        for record in results:
            writer.writerows((record["sid"],) + tuple(v) for v in record["views"])
    with gzip.open(directory / f"{stem}.fetches.csv.gz", "wt", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("sid",) + client.FETCH_FIELDS)
        for record in results:
            writer.writerows((record["sid"],) + tuple(f) for f in record["fetches"])


def _merge(key: str, path: Path, rows: list[dict]) -> list[dict]:
    from prefetch_common.schemas import read_rows, write_rows
    existing = read_rows(key, path) if path.is_file() else []
    fresh = {tuple(r[k] for k in ROW_KEY if k in r) for r in rows}
    kept = [r for r in existing if tuple(r[k] for k in ROW_KEY if k in r) not in fresh]
    merged = kept + rows
    write_rows(key, merged, path)
    return merged


def assumptions(args, store_meta: dict, device: dict) -> dict:
    """Every input of the run, tagged ASSUMED (chosen) or MEASURED (observed)."""
    a = lambda value, note="": {"value": value, "tag": "ASSUMED", **({"note": note} if note else {})}  # noqa: E731
    m = lambda value, note="": {"value": value, "tag": "MEASURED", **({"note": note} if note else {})}  # noqa: E731
    return {
        "rtt_ms": a(args.rtt_ms, "emulated per request at the origin"),
        "bandwidth_mbps": a(args.mbps, "emulated per-session access link"),
        "server_delay_ms": a(args.server_delay_ms, "mean of an exponential; 0 = none"),
        "link_discipline": a(args.link),
        "client_pool": a({"discipline": args.pool, "conns": args.conns,
                          "prefetch_conns": args.prefetch_conns,
                          "handshake_ms": args.handshake_ms}),
        "object_size": a({"dist": "lognormal", "median_bytes": args.size_median,
                          "sigma": args.size_sigma, "clip_bytes": [objects.SIZE_MIN,
                                                                   objects.SIZE_MAX],
                          "seed": objects.SIZE_SEED},
                         "RetailRocket records no page sizes"),
        "think_time": a({"gap_cap_s": args.gap_cap_s, "scale": args.think_scale},
                        "recorded inter-view gaps, capped and scaled"),
        "session_starts": a({"ramp_s": args.ramp_s, "target_rps": args.target_rps,
                             "seed": args.start_seed}),
        "cache": a({"capacity_objects": args.capacity, "scope": "session"}),
        "cold_cache": a(True, "store pages dropped (fsync + DONTNEED) before every pass"),
        "store": m({"path": store_meta["store"], "bytes": store_meta["bytes"],
                    "objects": store_meta["items"], "device": device,
                    "size_median_bytes": store_meta["size_median_bytes"],
                    "size_mean_bytes": store_meta["size_mean_bytes"]}),
        "validity_gate": a({"lag_p99_us_max": args.gate_us, "parity_mismatches": 0,
                            "connection_errors": 0}),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_arguments(parser)
    args = parser.parse_args(argv)
    return run(args, argv if argv is not None else sys.argv[1:])


def run(args, argv) -> int:
    from prefetch_common.provenance import snapshot, write_provenance
    start = snapshot()
    if args.reps < 1 or args.shards < 1:
        raise SystemExit("--reps and --shards must be positive")
    labels, base_names = resolve_policies(args.policies, args.baseline)
    plan_dir, plan, sim = ensure_plan(args, base_names)
    store_dir = Path(args.store_dir) if args.store_dir else Path(args.work_dir) / "store"
    store_meta = objects.build_store(plan["items"], store_dir, median=args.size_median,
                                     sigma=args.size_sigma)
    print(f"[live] store {store_meta['store']}: {store_meta['bytes'] / 2**20:.1f} MiB, "
          f"{'reused' if store_meta['reused'] else 'built'}", flush=True)
    device = objects.device_of(store_meta["store"])
    label = args.store_label or objects.store_label(store_meta["store"])
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_probe:
        probe = objects.probe(store_meta["store"], store_meta["index"], label=label)
        (results_dir / f"disk_probe_{label}.json").write_text(json.dumps(probe, indent=1) + "\n")
    sizes = {item: size for item, (_, size) in objects.load_index(store_meta["index"]).items()}
    sessions = plan["sessions"]
    config = ClientConfig(capacity=args.capacity, conns=args.conns,
                          prefetch_conns=args.prefetch_conns, pool=args.pool,
                          handshake_ms=args.handshake_ms, gap_cap_s=args.gap_cap_s,
                          think_scale=args.think_scale)
    starts = client.session_starts(sessions, ramp_s=args.ramp_s, seed=args.start_seed,
                                   target_rps=args.target_rps)
    limit = args.max_rps_per_shard * args.shards
    rates = {}
    for pass_label in labels:
        policy, hint = analyze.reference_policy(pass_label)
        rates[pass_label] = client.planned_rate(sessions, policy, starts, config, hint=hint)
        print(f"[live] {pass_label}: planned peak {rates[pass_label]['peak_rps']:.0f} req/s, "
              f"mean {rates[pass_label]['mean_rps']:.0f} req/s, "
              f"{rates[pass_label]['requests']} requests", flush=True)
    too_fast = [p for p, r in rates.items() if r["peak_rps"] > limit]
    if too_fast:
        print(f"[live] refused: planned peak rate of {too_fast} exceeds {limit:g} req/s "
              f"({args.shards} shard(s) x {args.max_rps_per_shard:g}); "
              "lengthen --ramp-s, lower --target-rps or add --shards", flush=True)
        return EXIT_REFUSED
    meta = {"experiment": args.experiment, "store": label, "config": config_label(args),
            "link": args.link, "rtt_ms": float(args.rtt_ms), "mbps": float(args.mbps),
            "size_median_kib": args.size_median / 1024, "gap_cap_s": float(args.gap_cap_s),
            "ramp_s": float(args.ramp_s)}
    run_tag = f"{args.experiment}/{label}/{meta['config']}/{args.link}"
    raw_dir = Path(args.work_dir) / "raw" / _safe(run_tag)
    passes, aborted, summaries = [], [], {}
    with OriginProcess(store_meta["store"], store_meta["index"], args,
                       Path(args.work_dir) / "origin.log") as origin:
        config.port = origin.port
        for rep in range(args.reps):
            shift = rep % len(labels)
            for pass_label in labels[shift:] + labels[:shift]:
                quiet, load = wait_for_quiet(args.max_load, args.load_wait_s)
                if not quiet:
                    aborted.append({"rep": rep, "policy": pass_label,
                                    "reason": f"load {load:.1f} >= {args.max_load:g}"})
                    print(f"[live] rep {rep} {pass_label}: aborted, load {load:.1f}", flush=True)
                    continue
                objects.drop_page_cache(store_meta["store"])
                origin.stats(reset=True)
                sectors = objects.diskstats_sectors_read(device.get("name"))
                started = time.perf_counter()
                load_start = os.getloadavg()[0]
                results, client_lag = execute(sessions, pass_label, config, starts,
                                              args.shards, tag=f"r{rep}:")
                wall = time.perf_counter() - started
                server = origin.stats()
                after = objects.diskstats_sectors_read(device.get("name"))
                foreign = None
                if sectors is not None and after is not None and server.get("read_bytes") is not None:
                    device_bytes = (after - sectors) * 512
                    foreign = (max(0, device_bytes - server["read_bytes"]) / device_bytes
                               if device_bytes > 0 else 0.0)
                if not args.no_raw:
                    _write_raw(raw_dir, f"rep{rep}_{_safe(pass_label)}", results)
                summary = analyze.summarise_pass(results, sizes, rtt_ms=args.rtt_ms,
                                                 mbps=args.mbps)
                record = analyze.parity(results, sim, pass_label, args.baseline)
                valid, reasons = analyze.validity(summary, server, client_lag,
                                                  record["mismatched_sessions"],
                                                  gate_us=args.gate_us)
                load_avg = max(load_start, os.getloadavg()[0])
                summaries[(rep, pass_label)] = (summary, record, valid, server, client_lag,
                                                foreign, load_avg)
                passes.append({"rep": rep, "policy": pass_label, "wall_s": round(wall, 3),
                               "load_avg": load_avg, "valid": valid, "invalid_reasons": reasons,
                               "planned_rate": rates[pass_label], "foreign_io_frac": foreign,
                               "server": {k: v for k, v in server.items() if k != "config"},
                               "client_loop_lag": client_lag, "breakdown": summary["breakdown"],
                               "view_start_late_ms_p99": summary["view_start_late_ms_p99"],
                               "p99_wait_ms": summary["p99_wait_ms"],
                               "conns_opened": summary["conns_opened"]})
                print(f"[live] rep {rep} {pass_label}: {wall:.1f} s, mean wait "
                      f"{summary['mean_wait_ms']:.2f} ms, hits {summary['hits']}/"
                      f"{summary['counted_views']}, parity mismatches "
                      f"{record['mismatched_sessions']}, "
                      f"{'valid' if valid else 'INVALID: ' + '; '.join(reasons)}", flush=True)
    metrics, parity_rows = [], []
    for (rep, pass_label), (summary, record, valid, server, lag, foreign, load) in summaries.items():
        base = summaries.get((rep, args.baseline))
        row = analyze.metrics_row(
            {**meta, "rep": rep}, pass_label, args.baseline, summary,
            base[0] if base else None, valid=valid, base_valid=bool(base and base[2]),
            mismatched=record["mismatched_sessions"], server=server, client_lag=lag,
            foreign_io_frac=foreign, load_avg=load, reps=args.bootstrap_reps)
        metrics.append(row)
        parity_rows.append(analyze.parity_row({**meta, "rep": rep}, pass_label, record))
    all_metrics = _merge(METRICS_KEY, results_dir / "live_metrics.csv", metrics)
    all_parity = _merge(PARITY_KEY, results_dir / "live_parity.csv", parity_rows)
    config_path = results_dir / CONFIG_NAME
    live_config = json.loads(config_path.read_text()) if config_path.is_file() else {
        "format": "webprefetch.live.config/1", "runs": {}}
    live_config["runs"][run_tag] = {
        "command": ["python", "-m", "webprefetch.live", "run", *argv],
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "plan": {"dir": str(plan_dir), "params": plan["params"],
                 "sessions": plan["sessions_selected"], "views": plan["views"],
                 "objects": len(plan["items"])},
        "assumptions": assumptions(args, store_meta, device),
        "reps": args.reps, "shards": args.shards, "policies": labels,
        "passes": passes, "aborted": aborted}
    tmp = config_path.with_name(CONFIG_NAME + ".tmp")
    tmp.write_text(json.dumps(live_config, indent=1, sort_keys=True) + "\n")
    tmp.replace(config_path)
    (results_dir / REPORT_NAME).write_text(
        analyze.report_markdown(all_metrics, all_parity, live_config["runs"]))
    inputs = [plan_dir / plan_module.PLAN_NAME, plan_dir / plan_module.SIM_NAME,
              store_meta["index"]]
    write_provenance(results_dir, ["python", "-m", "webprefetch.live", "run", *argv], inputs,
                     study="live", start=start, gpu=False,
                     notes="results accumulate across runs; live_config.json lists every run")
    mismatches = sum(r["mismatched_sessions"] for r in parity_rows)
    invalid = sum(1 for p in passes if not p["valid"])
    print(f"[live] {len(passes)} passes, {invalid} invalid, {len(aborted)} aborted, "
          f"{mismatches} parity mismatches; results in {results_dir}", flush=True)
    return EXIT_PARITY if mismatches else 0
