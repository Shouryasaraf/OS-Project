"""Dataset construction shared by ``main.py`` and the mode sweep.

Kept in the package rather than in the entry point so the interactive picker
and the non-interactive default can be reused without importing a script.
"""

from __future__ import annotations

from pathlib import Path

from .simulator import DEFAULT_CAPACITY
# One tuple of loader profiles for every entry point (oracle included); a
# second copy here had already drifted from the loader's.
from .trace import FORMATS, Request, iter_csv_chunks, load_csv, transition_trace

REPO = Path(__file__).resolve().parents[2]

MSR_SAMPLES = {
    "1": REPO / "data" / "samples" / "msr-cambridge1-sample.csv",
    "2": REPO / "data" / "samples" / "msr-cambridge2-sample.csv",
    "3": REPO / "data" / "samples" / "msr-cambridge-sample-merged.csv",
}
MSRC_TRACE_DIR = REPO / "data" / "MSRC-trace-003" / "final-trace"
LSTM_ARTIFACT = REPO / "models" / "lstm_delta.pt"
#: Sidecar recording what ``main.py`` trained :data:`LSTM_ARTIFACT` on. A
#: cached artifact is reused only when this matches the current selection, so
#: an artifact trained on the benchmark traces can never be silently reused.
LSTM_TRAINING_RECORD = LSTM_ARTIFACT.with_suffix(".training.json")
OUTPUT_DIR = REPO / "outputs"

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
#: The same under ``main.py --quick``, which must finish in about 30 s.
#: MSRC-trace-003 is absent on the development machine, so this was measured
#: with five stand-in ``.revised`` files cut from oracle traces (host load
#: ~50): at 40k requests each a quick run took 133 s, dominated by the
#: 13-mode benchmark and the EDA profile; the Belady pass is only 1-2 s per
#: trace, so skipping it alone could not meet the budget. At 5k it took 24 s.
QUICK_REAL_TRACE_LIMIT = 5000

#: Name prefixes that mark a dataset as generated rather than measured.
SYNTHETIC_PREFIXES = ("synthetic", "demo")

#: Prefix of the pooled rows ``main.py`` adds after the per-dataset rows.
#: Synthetic and real datasets are pooled separately (``ALL_SYNTHETIC``,
#: ``ALL_REAL``): one mean over both mixes generator structure into a claim
#: about real workloads.
AGGREGATE_PREFIX = "ALL_"

#: Generator seeds of the synthetic benchmark datasets (``synthetic_seed_N``
#: is built from seed ``200000 + N``), and of the traces the LSTM baseline is
#: trained on. The two ranges must never meet: the LSTM used to be trained on
#: the benchmark traces themselves and then scored on them, in-sample.
BENCHMARK_SEEDS = tuple(42 + 200000 + offset for offset in range(3))
LSTM_TRAINING_SEEDS = tuple(42 + 300000 + offset for offset in range(3))
#: Windows per class of every generated benchmark and training trace.
SYNTHETIC_WINDOWS_PER_CLASS = 8


def is_synthetic(name: str) -> bool:
    """True for generated datasets, false for recorded workloads."""
    return str(name).startswith(SYNTHETIC_PREFIXES)


def is_aggregate(name: str) -> bool:
    """True for the pooled rows, which are not datasets of their own."""
    return str(name).startswith(AGGREGATE_PREFIX)


def synthetic_datasets(window_size: int = 32) -> list[tuple[str, list]]:
    """The labelled synthetic benchmark traces, one per :data:`BENCHMARK_SEEDS`."""
    datasets = []
    for seed in BENCHMARK_SEEDS:
        requests, _ = transition_trace(seed, SYNTHETIC_WINDOWS_PER_CLASS,
                                       window_size)
        datasets.append((f"synthetic_seed_{seed - 200000}", requests))
    return datasets


def trace_head(path: str | Path, limit: int, format: str = "revised") -> list[Request]:
    """The first ``limit`` requests of a trace, read by streaming.

    ``load_csv`` parses the whole file before it can be cut, which for a
    2.6 GB ``.revised`` file is minutes of work for 40k kept requests. The
    head is the same requests ``load_csv(...)[:limit]`` returns; only the
    time-order check stops at the head.
    """
    head: list[Request] = []
    for chunk in iter_csv_chunks(path, format, max(1000, min(limit, 250_000))):
        head.extend(chunk[:limit - len(head)])
        if len(head) >= limit:
            break
    return head


def default_datasets(window_size: int = 32,
                     capacity: int = DEFAULT_CAPACITY, *,
                     msrc_dir: str | Path | None = None,
                     real_trace_limit: int = REAL_TRACE_LIMIT) -> list[tuple[str, list]]:
    """The dataset set used when running non-interactively.

    ``data/samples/demo.csv`` is deliberately not in it. It is one more
    generated transition trace (``export-demo``), so pooling it with the
    seeded synthetic traces only double-counts the generator, and the copy
    committed before the generator fix (D10) carried the largest "gain" of
    any dataset. The interactive picker can still load it as a custom file.
    ``real_trace_limit`` is the number of requests kept per MSRC trace.
    """
    datasets = synthetic_datasets(window_size)
    for number in ("1", "2"):
        path = MSR_SAMPLES[number]
        if path.is_file():
            datasets.append((path.stem, load_csv(path, "msr")))
    trace_dir = Path(msrc_dir) if msrc_dir is not None else MSRC_TRACE_DIR
    if trace_dir.is_dir():
        for name in QUICK_MSRC:
            path = trace_dir / name
            if path.is_file():
                datasets.append((f"msrc_{path.stem}",
                                 trace_head(path, real_trace_limit)))
    return datasets


def _signature(requests: list[Request]) -> tuple:
    return tuple((r.lba, r.size_blocks, r.operation) for r in requests)


def lstm_training_set(datasets: list[tuple[str, list]],
                      window_size: int = 32) -> tuple[list[tuple[str, list]], dict]:
    """Traces the LSTM baseline is trained on, disjoint from ``datasets``.

    Always generated transition traces from :data:`LSTM_TRAINING_SEEDS`, never
    a benchmark dataset. On the synthetic benchmark rows the LSTM is therefore
    scored on held-out traces from the same generator; on real traces it is
    out of domain, which is the honest reading of its row. Raises if any
    training trace coincides with a benchmark dataset, so a seed collision
    cannot reintroduce the leak silently.

    Returns ``(named_traces, record)``; ``record`` is the JSON-safe
    description written next to the artifact.
    """
    training = []
    for seed in LSTM_TRAINING_SEEDS:
        requests, _ = transition_trace(seed, SYNTHETIC_WINDOWS_PER_CLASS,
                                       window_size)
        training.append((f"lstm_train_seed_{seed}", requests))
    benchmark = {_signature(requests) for _, requests in datasets}
    clashes = [name for name, requests in training
               if _signature(requests) in benchmark]
    if clashes:
        raise ValueError(f"LSTM training trace(s) {clashes} also appear in the "
                         "benchmark; training would be in-sample")
    record = {
        "source": "transition_trace",
        "seeds": list(LSTM_TRAINING_SEEDS),
        "windows_per_class": SYNTHETIC_WINDOWS_PER_CLASS,
        "window_size": window_size,
        "requests": sum(len(requests) for _, requests in training),
    }
    return training, record


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
