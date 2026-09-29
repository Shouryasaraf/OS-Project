"""Dataset construction shared by ``main.py`` and the mode sweep.

Kept in the package rather than in the entry point so the interactive picker
and the non-interactive default can be reused without importing a script.
"""

from __future__ import annotations

from pathlib import Path

from .simulator import DEFAULT_CAPACITY
from .trace import load_csv, transition_trace

REPO = Path(__file__).resolve().parents[2]

MSR_SAMPLES = {
    "1": REPO / "data" / "samples" / "msr-cambridge1-sample.csv",
    "2": REPO / "data" / "samples" / "msr-cambridge2-sample.csv",
    "3": REPO / "data" / "samples" / "msr-cambridge-sample-merged.csv",
}
MSRC_TRACE_DIR = REPO / "data" / "MSRC-trace-003" / "final-trace"
LSTM_ARTIFACT = REPO / "models" / "lstm_delta.pt"
OUTPUT_DIR = REPO / "outputs"
FORMATS = ("auto", "normalized", "msr", "iotta8", "alibaba", "revised")

#: MSRC traces used for the default run. The full directory holds 32 files up
#: to 2.6 GB each and ``load_csv`` materialises a whole file, so the default
#: run samples a spread of read-heavy volumes rather than loading everything.
#: All of these carry the recorded device service times used to calibrate the
#: measured cost model.
QUICK_MSRC = ("hm_1.revised", "mds_0.revised", "stg_0.revised",
              "proj_3.revised", "ts_0.revised")

#: Requests kept per real trace. These files are large; a full load dominates
#: runtime and adds nothing to a per-dataset hit-ratio comparison.
REAL_TRACE_LIMIT = 40000

#: Name prefixes that mark a dataset as generated rather than measured.
SYNTHETIC_PREFIXES = ("synthetic", "demo")


def is_synthetic(name: str) -> bool:
    """True for generated datasets, false for recorded workloads."""
    return str(name).startswith(SYNTHETIC_PREFIXES)


def default_datasets(window_size: int = 32,
                     capacity: int = DEFAULT_CAPACITY) -> list[tuple[str, list]]:
    """The dataset set used when running non-interactively."""
    datasets: list[tuple[str, list]] = []
    for offset in range(3):
        requests, _ = transition_trace(42 + 200000 + offset, 8, window_size)
        datasets.append((f"synthetic_seed_{42 + offset}", requests))
    for number in ("1", "2"):
        path = MSR_SAMPLES[number]
        if path.is_file():
            datasets.append((path.stem, load_csv(path, "msr")))
    if MSRC_TRACE_DIR.is_dir():
        for name in QUICK_MSRC:
            path = MSRC_TRACE_DIR / name
            if path.is_file():
                requests = load_csv(path, "revised")
                if len(requests) > REAL_TRACE_LIMIT:
                    requests = requests[:REAL_TRACE_LIMIT]
                datasets.append((f"msrc_{path.stem}", requests))
    demo = REPO / "data" / "samples" / "demo.csv"
    if demo.is_file():
        datasets.append(("demo_normalized", load_csv(demo, "auto")))
    return datasets


def dataset_summary(datasets: list[tuple[str, list]]) -> dict:
    """Counts by provenance, for reporting."""
    synthetic = [n for n, _ in datasets if is_synthetic(n)]
    real = [n for n, _ in datasets if not is_synthetic(n)]
    return {
        "total": len(datasets),
        "synthetic": synthetic,
        "real": real,
        "synthetic_count": len(synthetic),
        "real_count": len(real),
    }
