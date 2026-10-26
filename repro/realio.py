"""Real Linux prefetch testbed: calibrate, grid and report (E3).

``--device DIR`` names a directory on the device under test holding the
backing file (see ``python -m adaptive_prefetch.realio mkfile``). Without it
the step is unavailable: there is no device to measure, and nothing is
substituted. Lead decision L2 limits E3 to the SSD and the HDD sdb.

* ``--quick``: the acceptance smoke (5,000 requests of msr_hm_0, ``none`` and
  ``deep``, both backends, closed loop, one repeat). Its rows go to
  ``outputs/realio-quick/``, never ``results/``: a smoke run is not evidence.
* default: the budgeted E3 grid of ``results/realio/PREREGISTERED.md``
  (amendment A1, "runtime budget", lead decision L3): ``--grid auto`` runs
  the ``hdd`` grid on a rotational device and the ``ssd`` subset otherwise
  (``realio.runner.E3_PLANS``); ``--grid full`` is plan section 2.6 as
  written. Run it once per device. The device is calibrated into
  ``results/realio/device_<dev>.json`` first only when that file does not
  exist yet (E0 already wrote it) or ``--recalibrate`` is given. The grid
  waits before a run while the load average exceeds 32 (E3 needs a quiet
  host), for up to an hour each time, then stops; rerunning resumes.
"""

from __future__ import annotations

import argparse

from . import EXIT_UNAVAILABLE

OWNER = "WP8"
IMPLEMENTED = True

GRIDS = ("auto", "hdd", "ssd", "full")


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", default=None,
                        help="directory on the device under test; without it the step is unavailable")
    parser.add_argument("--device-name", default=None,
                        help="results label (default: kernel disk name, or 'ssd')")
    parser.add_argument("--grid", default="auto", choices=GRIDS,
                        help="auto: hdd on a rotational device, else ssd (PREREGISTERED.md A1); "
                             "full: plan section 2.6 as written")
    parser.add_argument("--reps", type=int, default=3, help="repeats per grid cell")
    parser.add_argument("--trace", action="append", default=None,
                        help="grid trace (<corpus>/<stem> or a path); repeatable")
    parser.add_argument("--recalibrate", action="store_true",
                        help="calibrate even when results/realio/device_<dev>.json exists")


def _resolve(args) -> tuple[str | None, dict | None, str]:
    """``(grid, device, note)``: the grid to run on ``args.device``."""
    if not args.device:
        return None, None, "no --device"
    from adaptive_prefetch.realio.layout import BackingFileError
    from adaptive_prefetch.realio.runner import describe_device
    try:
        device = describe_device(args.device, name=args.device_name)
    except (BackingFileError, OSError) as error:
        return None, None, str(error)
    if args.grid != "auto":
        return args.grid, device, ""
    if device["rotational"] is None:
        return None, device, (f"sysfs does not say whether {device['backing']} is on a "
                              "rotational device; pass --grid hdd or --grid ssd")
    return ("hdd" if device["rotational"] else "ssd"), device, ""


def estimate(args: argparse.Namespace) -> str:
    if args.quick:
        return "about 1-2 min (4 runs of 5,000 requests plus a 1 GiB drive flush each)"
    from adaptive_prefetch.realio.runner import estimate_seconds
    grid, device, _ = _resolve(args)

    def hours(name):
        return estimate_seconds(name, args.reps, traces=args.trace) / 3600

    if grid is None:
        both = hours("hdd") + hours("ssd")
        return (f"about {hours('hdd'):.1f} h for the hdd grid on sdb plus {hours('ssd'):.1f} h "
                f"for the ssd subset ({both:.1f} h; one run per --device, {args.reps} reps), "
                "from realio.runner.E3_TIMING; a budget estimate, not a measurement")
    return (f"about {hours(grid):.1f} h: the {grid} grid on {device['label']} "
            f"({args.reps} reps; realio.runner.E3_TIMING, a budget estimate), plus about "
            "1 min of calibration if the device has none yet")


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
    from adaptive_prefetch.realio.runner import DEFAULT_RESULTS, REPO_ROOT, main
    name = ["--device-name", args.device_name] if args.device_name else []
    if args.quick:
        out = REPO_ROOT / "outputs" / "realio-quick"
        return main(["run", "--trace", "msr/msr_hm_0", "--requests", "5000",
                     "--device-dir", args.device, "--modes", "none,deep",
                     "--backend", "odirect,pagecache", "--think", "closed",
                     "--results-dir", str(out), "--details", str(out / "runs.jsonl"),
                     *name])
    grid, device, problem = _resolve(args)
    if grid is None:
        print(f"reproduce.py realio: {problem}")
        return 2
    calibration = DEFAULT_RESULTS / f"device_{device['label']}.json"
    if args.recalibrate or not calibration.is_file():
        status = main(["calibrate", "--device-dir", args.device, *name])
        if status:
            return status
    else:
        print(f"reproduce.py realio: using the existing calibration {calibration} "
              "(--recalibrate to measure again)")
    traces = [x for t in (args.trace or []) for x in ("--trace", t)]
    return main(["grid", "--plan", grid, "--device-dir", args.device,
                 "--repeats", str(args.reps), *traces, *name])
