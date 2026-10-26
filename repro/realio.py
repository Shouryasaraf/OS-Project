"""Real Linux prefetch testbed: calibrate, grid and report (E3).

``--device DIR`` names a directory on the device under test holding the
backing file (see ``python -m adaptive_prefetch.realio mkfile``). Without it
the step is unavailable: there is no device to measure, and nothing is
substituted. Lead decision L2 limits E3 to the SSD and the HDD sdb.

* ``--quick``: the acceptance smoke (5,000 requests of msr_hm_0, ``none`` and
  ``deep``, both backends, closed loop, one repeat). Its rows go to
  ``outputs/realio-quick/``, never ``results/``: a smoke run is not evidence.
* default: calibrate the device into ``results/realio/device_<dev>.json``,
  run the E3 grid (plan section 2.6) and write ``realio_summary.csv`` plus
  provenance. The grid stops before a run when the load average exceeds 32
  (E3 needs a quiet host).
"""

from __future__ import annotations

import argparse

from . import EXIT_UNAVAILABLE

OWNER = "WP8"
IMPLEMENTED = True

#: HDD seconds per 20k-request run: closed and spread from the prototype
#: (plan section 2.6); the 1 GiB drive flush measured 9.6-10.0 s on sdb in
#: the WP8 smoke.
_CLOSED_S, _SPREAD_S, _FLUSH_S = 13, 65, 10


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", default=None,
                        help="directory on the device under test; without it the step is unavailable")
    parser.add_argument("--device-name", default=None,
                        help="results label (default: kernel disk name, or 'ssd')")
    parser.add_argument("--reps", type=int, default=3, help="repeats per grid cell")
    parser.add_argument("--trace", action="append", default=None,
                        help="grid trace (<corpus>/<stem> or a path); repeatable")


def estimate(args: argparse.Namespace) -> str:
    if args.quick:
        return "about 1-2 min (4 runs of 5,000 requests plus a 1 GiB drive flush each)"
    from adaptive_prefetch.realio.runner import GRID_MODES, GRID_TRACES, backend_for
    runs = sum(len(backend_for(m, ("odirect", "pagecache"))) for m in GRID_MODES)
    traces = len(args.trace or GRID_TRACES)
    seconds = traces * args.reps * runs * (_CLOSED_S + _SPREAD_S + 2 * _FLUSH_S)
    return (f"about {seconds / 3600:.1f} h on an HDD ({traces} traces x {runs} runs x "
            f"2 think models x {args.reps} reps), plus about 1 min of calibration")


def run(args: argparse.Namespace) -> int:
    if not args.device:
        print("reproduce.py realio: unavailable here: pass --device DIR, a directory on "
              "the device under test holding the backing file")
        return EXIT_UNAVAILABLE
    from adaptive_prefetch.realio import supported
    ok, reason = supported()
    if not ok:
        print(f"reproduce.py realio: unavailable here: {reason}")
        return EXIT_UNAVAILABLE
    from adaptive_prefetch.realio.runner import REPO_ROOT, main
    name = ["--device-name", args.device_name] if args.device_name else []
    if args.quick:
        out = REPO_ROOT / "outputs" / "realio-quick"
        return main(["run", "--trace", "msr/msr_hm_0", "--requests", "5000",
                     "--device-dir", args.device, "--modes", "none,deep",
                     "--backend", "odirect,pagecache", "--think", "closed",
                     "--results-dir", str(out), "--details", str(out / "runs.jsonl"),
                     *name])
    status = main(["calibrate", "--device-dir", args.device, *name])
    if status:
        return status
    traces = [x for t in (args.trace or []) for x in ("--trace", t)]
    return main(["grid", "--device-dir", args.device, "--repeats", str(args.reps),
                 *traces, *name])
