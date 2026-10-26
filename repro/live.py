"""Real HTTP prefetch experiment with simulator parity (E6).

``reproduce.py live --store-dir <HDD dir>`` runs the core of E6 through
``python -m webprefetch.live run`` (plan section 3.3):

* ``headline`` -- the nine headline policies, ``--reps`` repetitions, on the
  SSD store and on the HDD store
* ``fifo`` -- FIFO link and FIFO client pool for fixed0, fixed4 and fixed8,
  one repetition, SSD store
* ``hint`` -- hint:fixed4 and hint:fixed8 (server-side WILLNEED), one
  repetition, HDD store

Every run adds its baseline pass. ``--quick`` is the bounded smoke run
(200 sessions, fixed0 and fixed4, 60 s ramp, one repetition, SSD store) and
writes under ``outputs/web/live/`` rather than ``results/live/``. It needs
the prepared views and the trained ranker (``reproduce.py web`` or
``webprefetch prepare && webprefetch train``); without them, or without
``--store-dir`` for an HDD step, the step reports itself unavailable.
Passes run one at a time and wait for a quiet host (load below 32); never
run this concurrently with E1 or E3.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from . import EXIT_UNAVAILABLE

OWNER = "WP12"
IMPLEMENTED = True
EXPERIMENTS = ("headline", "fifo", "hint")
HEADLINE = ("fixed0,fixed1,fixed4,fixed8,adaptive_0.002,adaptive_0.01,gate_0.01,"
            "feedback_0.01,adaptive_0.01@trans_online")
PASS_S = 360  # measured on the prototype: 180 s ramp plus the longest capped session
QUICK_PASS_S = 80


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--store-dir", default=None,
                        help="HDD object-store directory (L2: on sdb); without it the HDD "
                             "steps are unavailable")
    parser.add_argument("--ssd-store-dir", default=None,
                        help="SSD object-store directory (default outputs/web/live/store)")
    parser.add_argument("--experiments", default=",".join(EXPERIMENTS),
                        help=f"comma list of {EXPERIMENTS}")
    parser.add_argument("--stores", default="ssd,hdd", help="headline stores: ssd, hdd")
    parser.add_argument("--reps", type=int, default=3, help="headline repetitions")
    parser.add_argument("--sessions", type=int, default=2000)
    parser.add_argument("--policies", default=HEADLINE, help="headline policies")
    parser.add_argument("--results-dir", default="results/live")
    parser.add_argument("--max-load", type=float, default=32.0)
    parser.add_argument("--shards", type=int, default=2,
                        help="client processes per pass (each limited to 300 req/s planned)")


def _steps(args) -> list[tuple[str, list[str]]]:
    """(label, ``webprefetch.live run`` arguments) for every step."""
    # Two client processes: at 2000 sessions over a 180 s ramp, the planned
    # peak of fixed8 and the online-index policy exceeds one shard's 300 req/s.
    common = ["--sessions", str(args.sessions), "--results-dir", args.results_dir,
              "--max-load", str(args.max_load), "--shards", str(args.shards)]
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
    if "fifo" in wanted:
        steps.append(("fifo/ssd", [*common, *ssd, "--experiment", "fifo", "--link", "fifo",
                                   "--pool", "fifo", "--policies", "fixed0,fixed4,fixed8",
                                   "--reps", "1"]))
    if "hint" in wanted:
        steps.append(("hint/hdd", None if hdd is None else [
            *common, *hdd, "--experiment", "hint", "--policies", "hint:fixed4,hint:fixed8",
            "--reps", "1"]))
    return steps


def _quick_argv() -> list[str]:
    return ["--sessions", "200", "--min-views", "2", "--policies", "fixed0,fixed4",
            "--ramp-s", "60", "--reps", "1", "--experiment", "smoke",
            "--results-dir", "outputs/web/live/quick-results"]


def estimate(args: argparse.Namespace) -> str:
    if args.quick:
        return f"about {2 * QUICK_PASS_S // 60 + 1} min (2 passes of about {QUICK_PASS_S} s)"
    try:
        steps = _steps(args)
    except SystemExit as error:
        return f"unknown ({error})"
    passes = 0
    for _, argv in steps:
        if argv is None:
            continue
        policies = argv[argv.index("--policies") + 1].split(",")
        reps = int(argv[argv.index("--reps") + 1])
        passes += reps * (len(policies) + ("fixed0" not in policies))
    return (f"about {passes * PASS_S / 3600:.1f} h ({passes} passes of about {PASS_S} s, "
            "plus waits for a quiet host)")


def _inputs_missing() -> str | None:
    from webprefetch.paths import MODELS, PROCESSED
    if not (PROCESSED / "views.csv.gz").is_file():
        return f"{PROCESSED / 'views.csv.gz'} missing (run reproduce.py web or webprefetch prepare)"
    if not (MODELS / "ranker.joblib").is_file():
        return f"{MODELS / 'ranker.joblib'} missing (run webprefetch train)"
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
