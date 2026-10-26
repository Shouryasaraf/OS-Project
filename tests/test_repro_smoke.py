"""The reproduce.py dispatcher and the repro/<study>.py interface it relies on.

Stdlib only, so it runs under system python3 too. Study modules are stubbed
by WP0 and replaced by the work package that owns each one; these tests pin
the contract between them, not any study's results.
"""

import contextlib
import importlib
import io
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import repro  # noqa: E402
import reproduce  # noqa: E402


def _quiet(call, *args):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        status = call(*args)
    return status, out.getvalue()


class ReproduceDispatcherTests(unittest.TestCase):
    def test_every_study_module_provides_the_interface(self):
        _, parsers = reproduce.build_parser()
        self.assertEqual(tuple(parsers), repro.STUDIES)
        for name in repro.STUDIES:
            module = importlib.import_module(f"repro.{name}")
            for attribute in ("OWNER", "IMPLEMENTED", "add_arguments", "estimate", "run"):
                self.assertTrue(hasattr(module, attribute), f"repro.{name}.{attribute}")
            # `all` builds each study's arguments without asking, so every
            # option must have a default.
            args = parsers[name].parse_args([])
            self.assertFalse(args.quick)
            self.assertIsInstance(module.estimate(args), str)
        gpu = importlib.import_module("repro.gpu")
        self.assertEqual(set(gpu.OWNERS), set(reproduce.GPU_STUDIES))

    def test_estimate_is_printed_before_the_step_runs(self):
        for name in repro.STUDIES:
            module = importlib.import_module(f"repro.{name}")
            with mock.patch.object(module, "run", side_effect=lambda args: print("RUN") or 0):
                status, out = _quiet(reproduce.main, [name])
            self.assertEqual(status, 0)
            lines = out.splitlines()
            self.assertTrue(lines[0].startswith(f"[{name}] estimated runtime:"), lines)
            self.assertEqual(lines[1], "RUN")

    def test_unimplemented_stub_fails_visibly_and_names_its_owner(self):
        for name in repro.STUDIES:
            module = importlib.import_module(f"repro.{name}")
            if module.IMPLEMENTED:
                continue
            status, out = _quiet(reproduce.main, [name])
            self.assertEqual(status, repro.EXIT_NOT_IMPLEMENTED)
            self.assertIn(f"not yet implemented ({module.OWNER})", out)

    def test_gpu_flag_routes_to_the_gpu_module(self):
        gpu = importlib.import_module("repro.gpu")
        for name in reproduce.GPU_STUDIES:
            calls = []
            with mock.patch.object(gpu, "run", side_effect=lambda study, args: calls.append(study) or 0):
                status, out = _quiet(reproduce.main, [name, "--gpu", "--quick"])
            self.assertEqual((status, calls), (0, [name]))
            self.assertTrue(out.startswith(f"[{name} --gpu] estimated runtime:"))

    def test_all_passes_over_unavailable_steps_but_never_reports_success(self):
        modules = {name: importlib.import_module(f"repro.{name}") for name in repro.STUDIES}
        ran = []

        def fake(name, status):
            return lambda args: ran.append((name, args.quick)) or status

        with contextlib.ExitStack() as stack:
            for name, module in modules.items():
                status = repro.EXIT_UNAVAILABLE if name == "realio" else 0
                stack.enter_context(mock.patch.object(module, "run", side_effect=fake(name, status)))
            status, out = _quiet(reproduce.main, ["all", "--quick"])
        self.assertEqual(status, repro.EXIT_UNAVAILABLE)
        self.assertEqual(ran, [(name, True) for name in repro.STUDIES])
        self.assertIn("unavailable here: realio", out)

    def test_all_stops_at_the_first_real_failure(self):
        modules = {name: importlib.import_module(f"repro.{name}") for name in repro.STUDIES}
        ran = []
        with contextlib.ExitStack() as stack:
            for name, module in modules.items():
                status = 1 if name == "web" else 0
                stack.enter_context(mock.patch.object(
                    module, "run", side_effect=lambda args, n=name, s=status: ran.append(n) or s))
            status, out = _quiet(reproduce.main, ["all"])
        self.assertEqual(status, 1)
        self.assertEqual(ran, list(repro.STUDIES[:repro.STUDIES.index("web") + 1]))
        self.assertIn("stopped at web", out)

    def test_script_bootstraps_its_own_path(self):
        # Run from outside the repository with no PYTHONPATH, as a user would.
        result = subprocess.run([sys.executable, str(ROOT / "reproduce.py"), "--help"],
                                cwd=str(ROOT.parent), capture_output=True, text=True,
                                env={"PATH": "/usr/bin:/bin"}, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        for name in (*repro.STUDIES, "all"):
            self.assertIn(name, result.stdout)


if __name__ == "__main__":
    unittest.main()
