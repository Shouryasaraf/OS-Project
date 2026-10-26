"""Real HTTP prefetch experiment with simulator parity (E6).

``reproduce.py live --store-dir <HDD dir>`` runs E6 as amended for the
runtime budget (``results/live/PREREGISTERED.md``, amendment 1; lead
decision L3) through ``python -m webprefetch.live run`` (plan section 3.3):

* ``headline`` -- six policies, 3 repetitions, on the SSD store and on the
  HDD store: fixed0, fixed4, fixed8, and the adaptive and feedback
  controllers E4 validated at capacity 8 and cost 0.01 for the logistic
  ranker, plus E4's validated adaptive controller for the online index
  (``validated_*`` names, read from ``results/web/comparisons.csv``)
* ``fifo`` -- FIFO link and FIFO client pool for fixed0, fixed4 and fixed8,
  one repetition, SSD store
* ``hint`` -- hint:fixed4 and hint:fixed8 (server-side WILLNEED), one
  repetition, HDD store

That is 42 passes of about 260 s, about 3 h. The HDD store goes under
``/mnt/b18edc65-4395-4a85-976a-bed3428c073b/os-project-prefetch/web-store/``
(lead decision L2); the SSD store defaults to ``outputs/web/live/store`` on
the repository disk. Every run adds its baseline pass. Without E4's
``comparisons.csv`` the validated policies cannot be resolved and the
command stops before any pass (there is no built-in fallback).

``--quick`` is the bounded smoke run (200 sessions, fixed0 and fixed4, 60 s
ramp, one repetition, SSD store) and writes under ``outputs/web/live/``
rather than ``results/live/``. It needs the prepared views and the trained
ranker (``reproduce.py web`` or ``webprefetch prepare && webprefetch
train``); without them, or without ``--store-dir`` for an HDD step, the step
reports itself unavailable. Passes run one at a time and wait for a quiet
host (load below 32); never run this concurrently with E1 or E3.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from . import EXIT_UNAVAILABLE

OWNER = "WP12"
IMPLEMENTED = True
EXPERIMENTS = ("headline", "fifo", "hint")
#: Equal to ``webprefetch.live.plan.HEADLINE_POLICIES`` (a test pins it); a
#: literal here so that ``reproduce.py`` imports nothing from the [web] extra.
HEADLINE = ("fixed0,fixed4,fixed8,validated_adaptive,validated_feedback,"
            "validated_adaptive@trans_online")
HEADLINE_REPS = 3
FIFO_POLICIES = "fixed0,fixed4,fixed8"
HINT_POLICIES = "hint:fixed4,hint:fixed8"
SELECTION = "results/web/comparisons.csv"
SELECTION_COST = 0.01
HDD_STORE = "/mnt/b18edc65-4395-4a85-976a-bed3428c073b/os-project-prefetch/web-store"
#: One pass: the 2000-session seed-42 schedule ends at 249 s (180 s ramp plus
#: the capped sessions; client.planned_rate), and the prototype's passes on
#: the same sample started 251 s apart; plus the page drop, three spawned
#: client shards' start-up and the raw records.
PASS_S = 260
#: Per run: plan (scoring and the online index), store build or reuse, probe.
RUN_SETUP_S = 150
QUICK_PASS_S = 80


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--store-dir", default=None,
                        help=f"HDD object-store directory (L2: on sdb, {HDD_STORE}); "
                             "without it the HDD steps are unavailable")
    parser.add_argument("--ssd-store-dir", default=None,
                        help="SSD object-store directory (default outputs/web/live/store)")
    parser.add_argument("--experiments", default=",".join(EXPERIMENTS),
                        help=f"comma list of {EXPERIMENTS}")
    parser.add_argument("--stores", default="ssd,hdd", help="headline stores: ssd, hdd")
    parser.add_argument("--reps", type=int, default=HEADLINE_REPS, help="headline repetitions")
    parser.add_argument("--sessions", type=int, default=2000)
    parser.add_argument("--policies", default=HEADLINE, help="headline policies")
    parser.add_argument("--selection", default=SELECTION,
                        help="E4's comparisons.csv, read for the validated_* policies")
    parser.add_argument("--selection-cost", type=float, default=SELECTION_COST)
    parser.add_argument("--results-dir", default="results/live")
    parser.add_argument("--max-load", type=float, default=32.0)
    parser.add_argument("--shards", type=int, default=2,
                        help="client processes per pass (each limited to 300 req/s planned)")


def _steps(args) -> list[tuple[str, list[str]]]:
    """(label, ``webprefetch.live run`` arguments) for every step."""
    # Two client processes: at 2000 sessions over a 180 s ramp, the planned
    # peak of fixed8 and the online-index policy exceeds one shard's 300 req/s.
    common = ["--sessions", str(args.sessions), "--results-dir", args.results_dir,
              "--max-load", str(args.max_load), "--shards", str(args.shards),
              "--selection", str(args.selection), "--selection-cost", str(args.selection_cost)]
    ssd = ["--store-dir", args.ssd_store_dir] if args.ssd_store_dir else []
    hdd = ["--store-dir", args.store_dir, "--store-label", "hdd"] if args.store_dir else None
    wanted = [e for e in args.experiments.split(",") if e]
    unknown = sorted(set(wanted) - set(EXPERIMENTS))
    if unknown:
        raise SystemExit(f"unknown experiments {unknown}; expected {EXPERIMENTS}")
    steps = []
    if "headline" in wanted:
        for store in [s for s in args.stores.split(",") if s]:
            target = ssd if store == "ssd" else hdd
            steps.append((f"headline/{store}", None if target is None else [
                *common, *target, "--experiment", "headline", "--policies", args.policies,
                "--reps", str(args.reps)]))
    # The ablations read the headline's store (their objects are a subset)
    # and keep the headline's disk_probe_<store>.json.
    if "fifo" in wanted:
        steps.append(("fifo/ssd", [*common, *ssd, "--experiment", "fifo", "--link", "fifo",
                                   "--pool", "fifo", "--policies", FIFO_POLICIES,
                                   "--reps", "1", "--no-probe"]))
    if "hint" in wanted:
        steps.append(("hint/hdd", None if hdd is None else [
            *common, *hdd, "--experiment", "hint", "--policies", HINT_POLICIES,
            "--reps", "1", "--no-probe"]))
    return steps


def _quick_argv() -> list[str]:
    return ["--sessions", "200", "--min-views", "2", "--policies", "fixed0,fixed4",
            "--ramp-s", "60", "--reps", "1", "--experiment", "smoke",
            "--results-dir", "outputs/web/live/quick-results"]


def plan_passes(args) -> list[tuple[str, int]]:
    """(step, passes) for every runnable step, the baseline pass included."""
    out = []
    for label, argv in _steps(args):
        if argv is None:
            continue
        policies = argv[argv.index("--policies") + 1].split(",")
        reps = int(argv[argv.index("--reps") + 1])
        out.append((label, reps * (len(policies) + ("fixed0" not in policies))))
    return out


def estimate(args: argparse.Namespace) -> str:
    if args.quick:
        return f"about {2 * QUICK_PASS_S // 60 + 1} min (2 passes of about {QUICK_PASS_S} s)"
    try:
        steps = plan_passes(args)
    except SystemExit as error:
        return f"unknown ({error})"
    passes = sum(n for _, n in steps)
    total = passes * PASS_S + len(steps) * RUN_SETUP_S
    detail = ", ".join(f"{label} {n}" for label, n in steps)
    return (f"about {total / 3600:.1f} h ({passes} passes of about {PASS_S} s: {detail}; "
            f"plus about {RUN_SETUP_S // 60} min of plan/store setup per step and any wait "
            "for a quiet host)")


def _inputs_missing() -> str | None:
    from webprefetch.paths import MODELS, PROCESSED
    if not (PROCESSED / "views.csv.gz").is_file():
        return f"{PROCESSED / 'views.csv.gz'} missing (run reproduce.py web or webprefetch prepare)"
    for name in ("candidate_index.joblib", "ranker.joblib"):
        if not (MODELS / name).is_file():
            return f"{MODELS / name} missing (run webprefetch train)"
    return None


def _selection_missing(args) -> str | None:
    """Why the validated policies cannot be resolved, or None."""
    names = [p for _, argv in _steps(args) if argv is not None
             for p in argv[argv.index("--policies") + 1].split(",")]
    if not any(name.split("@")[0].removeprefix("hint:").startswith("validated_")
               for name in names):
        return None
    if not Path(args.selection).is_file():
        return (f"{args.selection} missing: the validated_* policies are E4's per-capacity "
                "selection (run `reproduce.py web` first); there is no built-in fallback")
    return None


def run(args: argparse.Namespace) -> int:
    missing = _inputs_missing()
    if missing:
        print(f"reproduce.py live: unavailable: {missing}")
        return EXIT_UNAVAILABLE
    from webprefetch.live.run import main as live_main
    if args.quick:
        Path("outputs/web/live").mkdir(parents=True, exist_ok=True)
        return live_main(_quick_argv())
    missing = _selection_missing(args)
    if missing:
        print(f"reproduce.py live: unavailable: {missing}")
        return EXIT_UNAVAILABLE
    unavailable = []
    for label, argv in _steps(args):
        if argv is None:
            print(f"reproduce.py live: {label} unavailable: no --store-dir for the HDD store")
            unavailable.append(label)
            continue
        print(f"reproduce.py live: {label}", flush=True)
        status = live_main(argv)
        if status:
            print(f"reproduce.py live: {label} failed (exit {status})")
            return status
    return EXIT_UNAVAILABLE if unavailable else 0
