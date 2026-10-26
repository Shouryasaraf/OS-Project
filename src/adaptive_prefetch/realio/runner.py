"""Command line: ``mkfile``, ``calibrate``, ``run``, ``grid`` and ``report``.

Every measured run executes in a **fresh child process** (``_one``), so no run
inherits another's Python heap, thread pool or page-cache charge. A
page-cache child is re-executed under ``systemd-run --user --scope`` with a
memory limit and then tightens its own ``memory.max`` to the cache budget;
where that is unavailable the page-cache backend is skipped with the reason,
never run unbounded. Within one (trace, think model, repeat), the order of
the (backend, mode) runs is shuffled with a recorded seed, so drift on a
shared host does not line up with one mode.

``results/realio/realio.csv`` gets one row per run (columns are the
``prefetch_common.schemas`` contract); everything that has no column there
(bookkeeping time, layout, budget, parity counters) goes to a detail JSONL
under ``outputs/`` keyed the same way. The console prints conclusions only.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from itertools import chain
from pathlib import Path

from ..simulator import IOCostModel, LatencyModel
from . import DEFAULT_CAPACITY, supported
from .decisions import (KERNEL_MODES, ONLINE_PREFIX, CacheDivergence, parse_mode,
                        record_plan)

PACKAGE_ROOT = Path(__file__).resolve().parents[2]        # .../src
REPO_ROOT = PACKAGE_ROOT.parent
DEFAULT_RESULTS = REPO_ROOT / "results" / "realio"
DEFAULT_DETAILS = REPO_ROOT / "outputs" / "realio" / "runs.jsonl"
RESULT_MARK = "REALIO_RESULT "
BACKENDS = ("odirect", "pagecache")
#: E3's mode set (plan section 2.6). ``learned_depth`` is not in it: lead
#: decision L3 (runtime budget), stated as a limitation in PREREGISTERED.md.
GRID_MODES = ("none", "sequential", "deep", "deep@64", "linux_ra", "guard",
              "adaptive_evidence", "kernel_ra", "kernel_seq", "online-guard")
GRID_TRACES = ("msr/msr_hm_0", "msr/msr_web_2", "msr/msr_prn_0",
               "alibaba/alibabaBlock_17", "fiu/fiu_homes-110108-112108")
GRID_THINK = ("closed", "spread:1:5000")
#: The E3 grids, each a sequence of parts (traces x modes x think models,
#: repeats from ``--repeats``). ``full`` is plan section 2.6 as written
#: (about 10 h on sdb by :func:`estimate_seconds`). ``hdd`` and ``ssd`` are
#: the cut to lead decision L3's ~4 h budget, fixed before any E3 run in
#: ``results/realio/PREREGISTERED.md`` (amendment A1, "runtime budget"):
#: every trace and mode closed loop on the HDD, the spread think model on
#: msr_hm_0 only, and the SSD closed loop on two traces and five modes.
E3_PLANS = {
    "full": ({"traces": GRID_TRACES, "modes": GRID_MODES, "thinks": GRID_THINK},),
    "hdd": ({"traces": GRID_TRACES, "modes": GRID_MODES, "thinks": ("closed",)},
            {"traces": ("msr/msr_hm_0",), "modes": GRID_MODES,
             "thinks": ("spread:1:5000",)}),
    "ssd": ({"traces": ("msr/msr_hm_0", "msr/msr_web_2"),
             "modes": ("none", "deep", "linux_ra", "guard", "kernel_ra"),
             "thinks": ("closed",)},),
}
#: Planning constants for :func:`estimate_seconds` (2026-10-26). Per device
#: class: ``flush_s``, the 1 GiB drive flush before every run (WP8 smoke:
#: 9.6-10.5 s on sdb, 2.6-2.7 s on the SSD); ``closed_s``, the replay wall of
#: one 20k-request closed-loop run, four times the 5k-request Backend B
#: none/deep@64 runs measured on sdb (load average ~120) and on the SSD (WP8
#: smoke; msr_web_2 from a 20k run), multiplied by ``margin``.
#: ``E3_OVERHEAD_S``: child start, trace load and simulator plan (0.5-1.7 s
#: at 20k requests; msr_web_2's plans take 4-9 s). ``E3_SPREAD_THINK_S``:
#: total think time of spread:1:5000 over the first 20k requests, computed
#: from each trace with ``clock.gaps_us``; it can never exceed 20k x 5 ms =
#: 100 s, the fallback for any other trace. Budget numbers, never results.
E3_TIMING = {
    "hdd": {"flush_s": 10.5, "margin": 1.5,
            "closed_s": {"msr_hm_0": 14.6, "msr_web_2": 11.6, "msr_prn_0": 5.0,
                         "alibabaBlock_17": 4.6, "fiu_homes-110108-112108": 5.4},
            "closed_default_s": 15.0},
    "ssd": {"flush_s": 2.7, "margin": 1.5,
            "closed_s": {"msr_hm_0": 2.1, "msr_web_2": 11.9},
            "closed_default_s": 12.0},
}
E3_OVERHEAD_S = {"msr_web_2": 6.0}
E3_OVERHEAD_DEFAULT_S = 2.0
E3_SPREAD_THINK_S = {"msr_hm_0": 53.1, "msr_web_2": 100.0, "msr_prn_0": 96.5,
                     "alibabaBlock_17": 100.0, "fiu_homes-110108-112108": 92.7}
#: How long ``grid`` waits for the load average to drop before one run.
DEFAULT_LOAD_WAIT_S = 3600.0
LOAD_POLL_S = 30.0


# ------------------------------------------------------------------ inputs

def load_requests(path, requests: int | None, skip: int = 0) -> list:
    """The trace slice a run replays (oracleGeneral streams; others load whole)."""
    from ..sweep import trace_format
    from ..trace import iter_oracle_chunks, load_csv
    path = Path(path)
    try:
        fmt = trace_format(path)
    except ValueError:
        fmt = "auto"
    if fmt == "oracle":
        return list(chain.from_iterable(iter_oracle_chunks(
            path, max_requests=requests, skip_requests=skip)))
    loaded = load_csv(path, "revised" if fmt == "revised" else "auto")
    return loaded[skip:skip + requests if requests is not None else None]


def trace_label(path, skip: int = 0) -> str:
    """The row's ``trace``: ``msr_hm_0``, or ``msr_hm_0+skip1600000`` for a run
    that skipped requests. A skipped run replays a different segment (the E2
    test segment for ``learned_depth``), so it must never share a row key, a
    ``none`` calibration or a summary group with the trace's prefix."""
    from ..sweep import trace_name
    name = trace_name(Path(path))
    return f"{name}+skip{skip}" if skip else name


def backend_for(mode: str, backends) -> list[str]:
    """Backends a mode runs on: kernel and online modes are page-cache only."""
    if mode in KERNEL_MODES or mode.startswith(ONLINE_PREFIX):
        return [b for b in backends if b == "pagecache"]
    return list(backends)


def validate_mode(mode: str) -> None:
    if mode in KERNEL_MODES:
        return
    if mode.startswith(ONLINE_PREFIX):
        from .online import online_mode
        online_mode(mode)
        return
    parse_mode(mode)


def plan_parts(name: str, traces=None, modes=None, thinks=None) -> list[dict]:
    """The parts of an :data:`E3_PLANS` grid; ``traces``/``modes``/``thinks``
    replace that field in every part when given."""
    if name not in E3_PLANS:
        raise SystemExit(f"unknown grid plan {name!r}; expected {', '.join(E3_PLANS)}")
    out = []
    for part in E3_PLANS[name]:
        out.append({"traces": tuple(traces or part["traces"]),
                    "modes": tuple(modes or part["modes"]),
                    "thinks": tuple(thinks or part["thinks"])})
    return out


def estimate_seconds(name: str, repeats: int, device_class: str | None = None,
                     traces=None, requests: int = 20_000) -> float:
    """Planned wall time of a grid from :data:`E3_TIMING`; a planning number
    for the budget, never a result. ``device_class`` defaults to the plan's
    own (``hdd`` for ``full``)."""
    timing = E3_TIMING[device_class or ("ssd" if name == "ssd" else "hdd")]
    scale = requests / 20_000
    total = 0.0
    for part in plan_parts(name, traces):
        runs = sum(len(backend_for(m, BACKENDS)) for m in part["modes"])
        for trace in part["traces"]:
            stem = Path(trace).name.split(".")[0]
            run_s = (timing["flush_s"] + E3_OVERHEAD_S.get(stem, E3_OVERHEAD_DEFAULT_S)
                     + timing["margin"] * scale
                     * timing["closed_s"].get(stem, timing["closed_default_s"]))
            for think in part["thinks"]:
                think_s = (0.0 if think == "closed"
                           else scale * E3_SPREAD_THINK_S.get(stem, 100.0))
                total += repeats * runs * (run_s + think_s)
    return total


def _say(text: str) -> None:
    print(text, flush=True)


def wait_for_load(max_load: float, max_wait_s: float, *, poll_s: float = LOAD_POLL_S,
                  log=_say) -> tuple[bool, float, float]:
    """Wait until the 1-min load average is at most ``max_load``, for at most
    ``max_wait_s``. Returns ``(quiet, load, waited_s)``. A transient spike
    therefore delays the next run instead of ending the grid; a host that
    stays busy past the bound still stops it (rerun to resume)."""
    start = time.monotonic()
    load = os.getloadavg()[0]
    next_log = 0.0
    while load > max_load:
        waited = time.monotonic() - start
        if waited >= max_wait_s:
            return False, load, waited
        if waited >= next_log:
            log(f"waiting: load average {load:.1f} exceeds --max-load {max_load:g}; "
                f"rechecking every {poll_s:g} s for up to {max_wait_s - waited:.0f} s")
            next_log = waited + 300.0
        time.sleep(min(poll_s, max_wait_s - waited))
        load = os.getloadavg()[0]
    return True, load, time.monotonic() - start


def write_provenance(results_dir: Path, argv, inputs, start, notes: str) -> Path:
    """``provenance.json`` for this invocation, keeping the earlier ones.

    Calibration and every grid part write the same ``results/realio``
    directory, so each record carries the ones it replaces in
    ``extra.earlier_invocations`` (command, time, git state, load, notes)
    rather than overwriting them.
    """
    from prefetch_common.provenance import read_provenance
    from prefetch_common.provenance import write_provenance as write
    earlier = []
    try:
        previous = read_provenance(results_dir)
    except (OSError, ValueError):
        previous = None
    if previous is not None:
        earlier = list((previous.get("extra") or {}).get("earlier_invocations") or [])
        earlier.append({k: previous.get(k) for k in ("command", "created_utc", "git_sha",
                                                     "git_dirty", "wall_s", "load_avg",
                                                     "inputs", "notes")})
    return write(results_dir, argv, inputs, study="realio", start=start, gpu=False,
                 notes=notes, extra={"earlier_invocations": earlier} if earlier else None)


def parse_latency(text: str) -> LatencyModel:
    hit, miss, prefetch = (float(x) for x in text.split(","))
    return LatencyModel(hit, miss, prefetch)


# --------------------------------------------------------------- one run

def run_one(spec: dict) -> dict:
    """Execute one measured run in this process; returns its result record."""
    from .backend_odirect import ODirectBackend
    from .backend_pagecache import CgroupBudget, PageCacheBackend
    from .clock import Pacer, gaps_us
    from .devstat import IOWindow, flush_drive
    from .layout import RegionMapper, check_backing
    from .online import OnlineDriver
    from .report import latency_summary

    load = os.getloadavg()[0]
    mode, backend_name = spec["mode"], spec["backend"]
    requests = load_requests(spec["trace"], spec["requests"], spec["skip_requests"])
    if not requests:
        raise ValueError(f"{spec['trace']}: no requests after skipping {spec['skip_requests']}")
    _drop_cached(spec["trace"])        # its pages would share the cgroup budget
    latency = parse_latency(spec["latency"])
    budget = None
    if backend_name == "pagecache" and not spec["no_cgroup"]:
        budget = CgroupBudget()          # fail before any work if unbounded
    metrics = plan = None
    if mode not in KERNEL_MODES and not mode.startswith(ONLINE_PREFIX):
        kwargs = dict(spec.get("replay_kw") or {})
        if spec.get("model_path"):
            from ..artifacts import load_model
            kwargs["model"] = load_model(spec["model_path"])
        metrics, plan = record_plan([requests], mode, capacity=spec["capacity"],
                                    window_size=spec["window_size"],
                                    latency_model=latency, **kwargs)
    file_bytes = check_backing(spec["backing"])
    reserved = spec["flush_mib"] << 20
    mapper = RegionMapper.for_plan(file_bytes, requests, plan, reserved_tail=reserved,
                                   allow_alias=spec["allow_alias"])
    mapper.check()
    gaps = gaps_us(requests, spec["think"])
    driver = None
    if backend_name == "odirect":
        backend = ODirectBackend(spec["backing"], mapper, spec["capacity"],
                                 workers=spec["workers"])
    else:
        advice = {"kernel_ra": "normal", "kernel_seq": "sequential"}.get(mode, "random")
        online = mode.startswith(ONLINE_PREFIX)
        backend = PageCacheBackend(spec["backing"], mapper, advice=advice,
                                   disk=spec["disk"],
                                   extra_regions=mapper.demand_regions if online else 0)
        backend._ensure_buffer(max(r.size_blocks for r in requests) * 512 + 8192)
        if online:
            driver = OnlineDriver(mode, spec["capacity"], window_size=spec["window_size"],
                                  latency_model=latency)
    flush_s = flush_drive(spec["backing"], spec["flush_mib"])
    if budget is not None:
        budget.tighten(spec["capacity"] * 512)
    pacer = Pacer()
    window = IOWindow(spec["disk"]).start()
    start = time.perf_counter()
    if driver is not None:
        raw = driver.run(backend, requests, gaps, pacer)
    else:
        raw = backend.run(requests, plan, gaps, pacer)
    wall = time.perf_counter() - start
    window.stop()
    cg = budget.stats() if budget is not None else {}
    backend.close()
    mapper.check()                       # online prefetches may map new regions
    if driver is not None:
        metrics = driver.metrics
    if backend_name == "odirect":
        # The per-block assertion already ran; the totals must agree too.
        if (raw["real_hit_ratio"] != metrics.hit_ratio
                or raw["useful_prefetches"] != metrics.useful_prefetches
                or raw["prefetches"] != metrics.prefetches):
            raise CacheDivergence("Backend B totals differ from the simulator's")
        device_bytes = raw["demand_bytes"] + raw["prefetch_bytes"]
    else:
        device_bytes = window.own_bytes or 0
    reads = len(raw["latencies_us"])
    errors = sorted(pacer.errors)
    row = {
        "device": spec["device"], "backend": backend_name, "mode": mode,
        "think": spec["think"], "repeat": spec["repeat"],
        "trace": trace_label(spec["trace"], spec["skip_requests"]),
        "requests": len(requests),
        "sim_hit_ratio": metrics.hit_ratio if metrics else None,
        "sim_precision": metrics.prefetch_precision if metrics else None,
        "sim_device_ios": metrics.device_ios if metrics else None,
        "real_hit_ratio": raw["real_hit_ratio"],
        "full_hit_request_frac": raw["full_hits"] / reads if reads else 0.0,
        "ready_hits": raw["ready_hits"], "late_hits": raw["late_hits"],
        "late_wait_us": raw["late_wait_us"],
        **latency_summary(raw["latencies_us"]),
        "device_MiB": device_bytes / (1 << 20),
        "prefetch_MiB": raw["prefetch_bytes"] / (1 << 20),
        "think_total_s": pacer.total_us / 1e6,
        "think_error_us_p99": (errors[min(len(errors) - 1, int(0.99 * len(errors)))]
                               if errors else None),
        "wall_s": wall,
        "cg_file_MiB": cg.get("cg_file_MiB"), "cg_events_max": cg.get("cg_events_max"),
        "foreign_io_frac": window.foreign_io_frac, "wrapped": mapper.wrapped,
        "pred_mean_us": None, "pred_err": None, "k_eff": None, "load_avg": load,
    }
    detail = {
        "read_requests": reads, "read_blocks": raw["read_blocks"],
        "missed_blocks": raw["missed_blocks"], "full_hits": raw["full_hits"],
        "prefetches": raw["prefetches"], "useful_prefetches": raw["useful_prefetches"],
        "hit_lat_mean_us": raw["hit_lat_mean_us"], "miss_lat_mean_us": raw["miss_lat_mean_us"],
        "sim_blocks_moved": metrics.blocks_moved if metrics else None,
        "sim_useful_prefetches": metrics.useful_prefetches if metrics else None,
        "sim_prefetches": metrics.prefetches if metrics else None,
        "break_even": latency.break_even_precision(),
        "regions": mapper.regions, "demand_regions": mapper.demand_regions,
        "flush_s": flush_s, "own_read_MiB": (window.own_bytes or 0) / (1 << 20),
        "disk_MiB": (window.disk_bytes / (1 << 20)) if window.disk_bytes is not None else None,
        "timer_slack_set": pacer.slack_set, "parity_checked": backend_name == "odirect",
        "spec": spec,
    }
    for name in ("cpu_bookkeeping_us", "max_inflight", "hint_calls", "residency_source",
                 "willneed_chunk", "advice", "inference_us", "gate_switches",
                 "gate_precision"):
        if name in raw:
            detail[name] = raw[name]
    if "cg_limit_MiB" in cg:
        detail["cg_limit_MiB"] = cg["cg_limit_MiB"]
    return {"row": row, "detail": detail}


def _drop_cached(path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def _child_main(argv) -> int:
    spec = json.loads(argv[0])
    try:
        result = run_one(spec)
    except CacheDivergence as error:
        print(f"CacheDivergence: {error}", file=sys.stderr)
        return 3
    print(RESULT_MARK + json.dumps(result), flush=True)
    return 0


def spawn(spec: dict, *, timeout: float = 7200.0) -> dict:
    """Run ``spec`` in a child process (under a memory scope for Backend A)."""
    from .backend_pagecache import scope_command
    argv = [sys.executable, "-m", "adaptive_prefetch.realio", "_one", json.dumps(spec)]
    if spec["backend"] == "pagecache" and not spec["no_cgroup"]:
        argv = scope_command(argv, spec["capacity"])
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(PACKAGE_ROOT),
                                                      env.get("PYTHONPATH")]))
    done = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, env=env)
    for line in reversed(done.stdout.splitlines()):
        if line.startswith(RESULT_MARK):
            return json.loads(line[len(RESULT_MARK):])
    if done.returncode == 3:
        raise CacheDivergence(done.stderr.strip().splitlines()[-1])
    if done.returncode == -9 and spec["backend"] == "pagecache" and not spec["no_cgroup"]:
        raise RuntimeError(f"run pagecache/{spec['mode']} was killed (SIGKILL), most "
                           "likely by its memory scope's OOM killer: the child's heap "
                           "outgrew capacity + SCOPE_HEADROOM")
    raise RuntimeError(f"run {spec['backend']}/{spec['mode']} failed "
                       f"(exit {done.returncode}): {done.stderr.strip()[-2000:]}")


# ----------------------------------------------------------------- results

def write_results(results_dir: Path, new_rows: list[dict]) -> list[dict]:
    """Append rows to ``realio.csv`` (rewritten atomically); returns all rows."""
    from prefetch_common import schemas
    path = results_dir / "realio.csv"
    rows = schemas.read_rows("realio/realio.csv", path) if path.exists() else []
    rows.extend(new_rows)
    schemas.write_rows("realio/realio.csv", rows, path)
    return rows


def read_details(path: Path) -> dict:
    details = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                details[tuple(record["key"])] = record
    return details


def cost_models(results_dir: Path) -> dict:
    models = {}
    for path in sorted(results_dir.glob("device_*.json")):
        try:
            models[path.stem[len("device_"):]] = IOCostModel.from_device_json(path)
        except ValueError:
            continue
    return models


def write_summary(results_dir: Path, rows: list[dict], details_path: Path) -> list[dict]:
    from prefetch_common import schemas
    from .report import summarize
    summary = summarize(rows, read_details(details_path), cost_models(results_dir))
    schemas.write_rows("realio/realio_summary.csv", summary, results_dir / "realio_summary.csv")
    return summary


# ----------------------------------------------------------------- commands

def _device_label(args, backing: Path) -> tuple[str, str | None]:
    from .devstat import device_info, device_label, disk_for_path
    disk = disk_for_path(backing)
    return (args.device_name or device_label(disk, device_info(disk)["rotational"]), disk)


def describe_device(device_dir, backing: str | None = None,
                    name: str | None = None) -> dict:
    """``{"label", "disk", "rotational", "backing"}`` for the device under
    ``device_dir``: the results label a run there would get, and whether
    sysfs calls it rotational (``None`` when it does not say)."""
    from .devstat import device_info, device_label, disk_for_path
    from .layout import find_backing
    path = find_backing(device_dir, backing)
    disk = disk_for_path(path)
    rotational = device_info(disk)["rotational"]
    return {"label": name or device_label(disk, rotational), "disk": disk,
            "rotational": rotational, "backing": str(path)}


def cmd_mkfile(args) -> int:
    from .layout import make_backing_file
    path = Path(args.device_dir) / (args.backing or "backing.bin")

    def progress(done, total):
        if done == total or done % (1 << 30) == 0:
            print(f"  {done >> 20} / {total >> 20} MiB", flush=True)

    info = make_backing_file(path, args.gib, overwrite=args.overwrite, progress=progress)
    print(f"wrote {info['path']}: {info['bytes'] >> 20} MiB in {info['seconds']} s "
          f"({info['MiBps']} MiB/s)")
    return 0


def cmd_calibrate(args) -> int:
    from prefetch_common import schemas
    from prefetch_common.provenance import snapshot
    from .calibrate import calibrate
    from .layout import check_backing, find_backing
    start = snapshot()
    backing = find_backing(args.device_dir, args.backing)
    check_backing(backing)
    scale = args.samples_scale
    samples = {k: max(16, int(v * scale)) for k, v in
               {"rand4k": 2000, "seq4k": 4000, "seq64k": 2000, "seq1m": 512}.items()}
    row = calibrate(backing, name=args.device_name, samples=samples, seed=args.seed,
                    flush_mib=args.flush_mib)
    results = Path(args.results_dir)
    path = results / f"device_{row['device']}.json"
    schemas.write_rows(f"realio/device_{row['device']}.json", [row], path)
    if not args.no_provenance:
        write_provenance(results, sys.argv, [], start, f"calibrate {row['device']}")
    print(f"{row['device']} ({row['model']}): random 4 KiB mean "
          f"{row['rand4k_mean_us']:.1f} us, p50 {row['rand4k_p50_us']:.1f} us; "
          f"sequential {row['seq1m_MBps']:.1f} MB/s")
    print(f"  t_pos = {row['t_pos_us']:.1f} us, t_xfer = {row['t_xfer_us_per_block']:.3f} "
          f"us/block, kappa = {row['kappa']:.0f} blocks -> {path}")
    if row["foreign_io_frac"] is not None and row["foreign_io_frac"] > 0.01:
        print(f"  WARNING: {row['foreign_io_frac']:.1%} of the disk's traffic was "
              "not ours during calibration")
    return 0


def plan_jobs(args, traces, modes, thinks) -> list[dict]:
    """Every run of the invocation, shuffled within (trace, think, repeat)."""
    from .backend_pagecache import systemd_run_available
    from .layout import check_backing, find_backing
    backing = find_backing(args.device_dir, args.backing)
    check_backing(backing)
    device, disk = _device_label(args, backing)
    backends = [b.strip() for b in args.backend.split(",") if b.strip()]
    for b in backends:
        if b not in BACKENDS:
            raise SystemExit(f"unknown backend {b!r}; expected {', '.join(BACKENDS)}")
    if "pagecache" in backends and not args.no_cgroup:
        ok, reason = systemd_run_available()
        if not ok:
            print(f"skipped: pagecache backend ({reason}); Backend A needs a memory "
                  "cgroup and is never run unbounded")
            backends = [b for b in backends if b != "pagecache"]
    rng = random.Random(args.seed)
    jobs = []
    for trace in traces:
        for think in thinks:
            for repeat in range(1, args.repeats + 1):
                cell = [(b, m) for m in modes for b in backend_for(m, backends)]
                rng.shuffle(cell)
                for order, (backend, mode) in enumerate(cell):
                    jobs.append({
                        "trace": str(trace), "requests": args.requests,
                        "skip_requests": args.skip_requests, "mode": mode,
                        "backend": backend, "think": think, "repeat": repeat,
                        "capacity": args.capacity, "window_size": args.window_size,
                        "backing": str(backing), "flush_mib": args.flush_mib,
                        "workers": args.workers, "allow_alias": args.allow_alias,
                        "latency": args.latency_model, "device": device, "disk": disk,
                        "no_cgroup": args.no_cgroup, "model_path": args.model_path,
                        "replay_kw": parse_replay_args(args.replay_arg),
                        "order": order, "seed": args.seed,
                    })
    done = existing_keys(Path(args.results_dir), Path(args.details))
    if done:
        kept = [j for j in jobs if job_key(j) not in done]
        if len(kept) < len(jobs):
            print(f"resuming: {len(jobs) - len(kept)} of {len(jobs)} runs are already in "
                  f"{Path(args.results_dir) / 'realio.csv'}")
        jobs = kept
    return jobs


def parse_replay_args(items) -> dict:
    """``KEY=VALUE`` pairs for ``replay_stream`` (VALUE parsed as JSON if it can be)."""
    out = {}
    for item in items or ():
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise SystemExit(f"--replay-arg expects KEY=VALUE, got {item!r}")
        try:
            out[key] = json.loads(value)
        except ValueError:
            out[key] = value
    return out


def job_key(job: dict) -> tuple:
    """The ``report.row_key`` a job's row will have when the trace is long
    enough; a row records the requests actually replayed, so a capped run
    is matched through its detail record's spec instead (``existing_keys``)."""
    return (job["device"], job["backend"], job["mode"], job["think"], job["repeat"],
            trace_label(job["trace"], job["skip_requests"]), job["requests"])


def existing_keys(results_dir: Path, details_path: Path | None = None) -> set:
    """Keys of the runs already in ``realio.csv``: their row keys, plus the
    job key of each one's spec, so a run capped by a short trace resumes."""
    from prefetch_common import schemas
    from .report import row_key
    path = results_dir / "realio.csv"
    if not path.exists():
        return set()
    keys = {row_key(r) for r in schemas.read_rows("realio/realio.csv", path)}
    if details_path is not None:
        for key, record in read_details(details_path).items():
            if key in keys and isinstance(record.get("spec"), dict):
                keys.add(job_key(record["spec"]))
    return keys


def execute(args, traces, modes, thinks) -> int:
    from prefetch_common.provenance import snapshot
    from .report import conclusions, row_key
    ok, reason = supported()
    if not ok:
        print(f"skipped: {reason}")
        return 4
    for mode in modes:
        validate_mode(mode)
    from .clock import parse_think
    for think in thinks:
        parse_think(think)
    start = snapshot()
    jobs = plan_jobs(args, traces, modes, thinks)
    results_dir = Path(args.results_dir)
    details_path = Path(args.details)
    details_path.parent.mkdir(parents=True, exist_ok=True)
    if not jobs and not (results_dir / "realio.csv").exists():
        print("nothing to run")
        return 4
    if jobs:
        print(f"{len(jobs)} runs on {jobs[0]['device']} ({jobs[0]['backing']}), "
              f"{args.requests} requests each", flush=True)
    completed, waited_s = 0, 0.0
    for i, job in enumerate(jobs, 1):
        if args.max_load is not None:
            quiet, load, waited = wait_for_load(args.max_load, args.max_load_wait)
            waited_s += waited
            if not quiet:
                print(f"stopped: load average {load:.1f} still exceeds --max-load "
                      f"{args.max_load:g} after {waited:.0f} s; {i - 1} of {len(jobs)} runs "
                      "done (rerun to resume)")
                break
        outcome = spawn(job)
        # Written per run, so an interrupted grid resumes where it stopped.
        write_results(results_dir, [outcome["row"]])
        with open(details_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps({"key": list(row_key(outcome["row"])),
                                     **outcome["detail"]}) + "\n")
        completed += 1
        row = outcome["row"]
        print(f"  [{i}/{len(jobs)}] {row['backend']:9s} {row['mode']:16s} {row['think']:14s} "
              f"rep {row['repeat']}  mean {row['lat_mean_us']:8.1f} us  "
              f"hit {row['real_hit_ratio']:.4f}  wall {row['wall_s']:.1f} s", flush=True)
    rows = finalize(results_dir, details_path)
    summary = write_summary(results_dir, rows, details_path)
    if not args.no_provenance:
        wait_note = (f"; {waited_s:.0f} s spent waiting for the load average"
                     if waited_s >= 1 else "")
        write_provenance(results_dir, sys.argv, [Path(t) for t in traces], start,
                         f"{completed} runs appended; details in {details_path}{wait_note}")
    parity = sum(1 for r in rows if r["backend"] == "odirect")
    print()
    for line in conclusions(summary, parity):
        print(line)
    print(f"\nrows: {results_dir / 'realio.csv'}; summary: "
          f"{results_dir / 'realio_summary.csv'}; details: {details_path}")
    return 0 if completed == len(jobs) else 5


def finalize(results_dir: Path, details_path: Path) -> list[dict]:
    """Recompute the parity columns over every row and rewrite ``realio.csv``.

    A row's ``none`` calibration may come from an earlier invocation (a
    resumed grid), so the computation always spans the whole file, joined
    with the detail records that hold the per-run counters.
    """
    from prefetch_common import schemas
    from .report import attach_parity, row_key
    path = results_dir / "realio.csv"
    rows = schemas.read_rows("realio/realio.csv", path)
    details = read_details(details_path)
    runs = [{**row, **details.get(row_key(row), {})} for row in rows]
    attach_parity(runs)
    for row, run in zip(rows, runs):
        for name in ("pred_mean_us", "pred_err", "k_eff"):
            row[name] = run[name]
    schemas.write_rows("realio/realio.csv", rows, path)
    return rows


def resolve_traces(names) -> list[Path]:
    root = REPO_ROOT / "data" / "block-traces"
    out = []
    for name in names:
        path = Path(name)
        if not path.exists():
            path = root / f"{name}.oracleGeneral.zst"
        if not path.exists():
            raise SystemExit(f"trace not found: {name}")
        out.append(path)
    return out


def cmd_run(args) -> int:
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    thinks = [t.strip() for t in args.think.split(",") if t.strip()]
    return execute(args, resolve_traces(args.trace), modes, thinks)


def cmd_grid(args) -> int:
    modes = [m.strip() for m in args.modes.split(",")] if args.modes else None
    thinks = [t.strip() for t in args.think.split(",")] if args.think else None
    parts = plan_parts(args.plan, args.trace, modes, thinks)
    hours = estimate_seconds(args.plan, args.repeats, traces=args.trace,
                             requests=args.requests) / 3600
    print(f"grid plan {args.plan!r}: {len(parts)} part(s), {args.repeats} repeats; planned "
          f"about {hours:.1f} h (E3_TIMING: a budget estimate, not a measurement)",
          flush=True)
    status = 0
    for part in parts:
        result = execute(args, resolve_traces(part["traces"]), list(part["modes"]),
                         list(part["thinks"]))
        status = status or result
        if result:
            break
    return status


def cmd_report(args) -> int:
    from prefetch_common import schemas
    from .report import conclusions
    results_dir = Path(args.results_dir)
    path = results_dir / "realio.csv"
    if not path.exists():
        print(f"no rows yet: {path}")
        return 4
    rows = finalize(results_dir, Path(args.details))
    summary = write_summary(results_dir, rows, Path(args.details))
    parity = sum(1 for r in rows if r["backend"] == "odirect")
    for line in conclusions(summary, parity):
        print(line)
    return 0


def _common(parser, *, run: bool) -> None:
    parser.add_argument("--device-dir", required=True,
                        help="directory on the device under test holding the backing file")
    parser.add_argument("--backing", default=None,
                        help="backing file name in --device-dir (default: backing.bin, "
                             "else the prototype's hdd_test.bin / ssd_test.bin)")
    parser.add_argument("--device-name", default=None,
                        help="results label (default: kernel disk name, 'ssd' if non-rotational)")
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS))
    parser.add_argument("--flush-mib", type=int, default=1024,
                        help="MiB read from the file's end to displace the drive cache")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-provenance", action="store_true")
    if not run:
        return
    parser.add_argument("--requests", type=int, default=20_000)
    parser.add_argument("--skip-requests", type=int, default=0,
                        help="canonical requests to skip first (e.g. an E2 test segment)")
    parser.add_argument("--backend", default="odirect,pagecache")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--capacity", type=int, default=DEFAULT_CAPACITY,
                        help="cache capacity in 512-B blocks (E3: 65536 = 32 MiB)")
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4, help="Backend B prefetch threads")
    parser.add_argument("--latency-model", default="5,100,50",
                        help="hit,miss,prefetch us for the simulator plan (sets the "
                             "gated modes' break-even; E1 uses the assumed 5,100,50)")
    parser.add_argument("--model-path", default=None, help="classifier for 'adaptive'")
    parser.add_argument("--replay-arg", action="append", default=None, metavar="KEY=VALUE",
                        help="extra replay_stream keyword for the plan (repeatable), e.g. "
                             "an artifact path for an artifact-backed mode")
    parser.add_argument("--allow-alias", action="store_true",
                        help="permit trace regions to alias in a too-small backing file")
    parser.add_argument("--no-cgroup", action="store_true",
                        help="run Backend A without a memory cgroup (tests only: the "
                             "cache is then unbounded and the rows are not E3 evidence)")
    parser.add_argument("--max-load", type=float, default=None,
                        help="before each run, wait while the 1-min load average exceeds this")
    parser.add_argument("--max-load-wait", type=float, default=0.0,
                        help="seconds to wait for --max-load before stopping (grid: "
                             f"{DEFAULT_LOAD_WAIT_S:g})")
    parser.add_argument("--details", default=str(DEFAULT_DETAILS))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m adaptive_prefetch.realio",
                                     description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    mk = sub.add_parser("mkfile", help="write a non-sparse page-stamped backing file")
    mk.add_argument("--device-dir", required=True)
    mk.add_argument("--gib", type=float, required=True)
    mk.add_argument("--backing", default=None)
    mk.add_argument("--overwrite", action="store_true")
    cal = sub.add_parser("calibrate", help="measure t_pos/t_xfer -> device_<dev>.json")
    _common(cal, run=False)
    cal.add_argument("--samples-scale", type=float, default=1.0)
    run = sub.add_parser("run", help="replay traces under modes on one device")
    _common(run, run=True)
    run.add_argument("--trace", action="append", required=True,
                     help="trace path or <corpus>/<stem>; repeatable")
    run.add_argument("--modes", default="none,deep")
    run.add_argument("--think", default="closed")
    grid = sub.add_parser("grid", help="the E3 grid (plan section 2.6) on one device")
    _common(grid, run=True)
    grid.set_defaults(repeats=3, max_load=32.0, max_load_wait=DEFAULT_LOAD_WAIT_S)
    grid.add_argument("--plan", default="full", choices=tuple(E3_PLANS),
                      help="full: plan section 2.6 as written; hdd/ssd: the budgeted E3 "
                           "grids of results/realio/PREREGISTERED.md")
    grid.add_argument("--trace", action="append", default=None)
    grid.add_argument("--modes", default=None)
    grid.add_argument("--think", default=None)
    rep = sub.add_parser("report", help="rewrite the summary and print conclusions")
    rep.add_argument("--results-dir", default=str(DEFAULT_RESULTS))
    rep.add_argument("--details", default=str(DEFAULT_DETAILS))
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "_one":
        return _child_main(argv[1:])
    args = build_parser().parse_args(argv)
    if args.command != "report":
        ok, reason = supported()
        if not ok:
            print(f"skipped: {reason}")
            return 4
    return {"mkfile": cmd_mkfile, "calibrate": cmd_calibrate, "run": cmd_run,
            "grid": cmd_grid, "report": cmd_report}[args.command](args)
