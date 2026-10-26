"""Full MSRC collection sweep: every trace, every mode, one winner table.

    python sweep_modes.py                 # 250k requests per trace, all 32
    python sweep_modes.py --full          # every request in every file (slow)
    python sweep_modes.py --per-trace 50000
    python sweep_modes.py --modes none sequential adaptive adaptive_evidence
    python sweep_modes.py --trace-dir data/block-traces --no-cost-model

Streams each file in bounded chunks, so memory stays flat regardless of file
size. ``--trace-dir`` may also hold ``.oracleGeneral.zst`` block traces; those
carry no service times, so they never calibrate the cost model. The model is
measured from the first ``.revised`` trace when there is one and is then applied
to every trace, oracle ones included; with no ``.revised`` trace (or with
``--no-cost-model``) every trace runs under the explicit assumed 5/100/50 model.
Per-trace rows go to outputs/msrc_sweep.csv and outputs/msrc_sweep.md.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from time import perf_counter

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

from adaptive_prefetch.datasets import (  # noqa: E402
    MSRC_TRACE_DIR, OUTPUT_DIR, LSTM_ARTIFACT,
)
from adaptive_prefetch.baselines import DEFAULT_LINUX_RA_MAX_BLOCKS  # noqa: E402
from adaptive_prefetch.simulator import (  # noqa: E402
    DEFAULT_CAPACITY, DEFAULT_READ_AHEAD_DEPTH, LatencyModel,
)
from adaptive_prefetch.sweep import (  # noqa: E402
    DEFAULT_CHUNK, DEFAULT_REQUESTS_PER_TRACE, RANKING_MODES, ROW_COLUMNS,
    aggregate_by_mode, discover_traces, per_trace_winners, sweep_collection,
    trace_format, trace_size_bytes, winner_caveats, winners,
)

RULE = "=" * 96
THIN = "-" * 96
#: Measured parse rate of the ``.revised`` text loader (AGENTS.md), so one GB
#: takes about 7.9 minutes. The estimate used to print 0.5 min/GB.
REVISED_PARSE_MB_PER_S = 2.1


def parse_minutes(revised_gb: float) -> float:
    """Minutes to parse ``revised_gb`` of ``.revised`` text at the measured rate."""
    return revised_gb * 1000.0 / REVISED_PARSE_MB_PER_S / 60.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-trace", type=int,
                        default=DEFAULT_REQUESTS_PER_TRACE,
                        help="requests replayed per trace (0 or --full = all)")
    parser.add_argument("--full", action="store_true",
                        help="replay every request in every file (hours)")
    parser.add_argument("--chunk", type=int, default=DEFAULT_CHUNK,
                        help="streaming chunk size in requests")
    parser.add_argument("--cache-blocks", type=int, default=DEFAULT_CAPACITY)
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument("--trace-dir", default=None)
    parser.add_argument("--modes", nargs="+", default=list(RANKING_MODES),
                        choices=list(RANKING_MODES))
    parser.add_argument("--lstm-model", default=None)
    parser.add_argument("--prefetch-multiple", type=float, default=1.0,
                        help="cost of a speculative read relative to a demand "
                             "read; <1.0 models overlapped prefetch")
    parser.add_argument("--no-cost-model", action="store_true",
                        help="use the assumed 5/100/50 us model")
    parser.add_argument("--read-ahead-depth", type=int,
                        default=DEFAULT_READ_AHEAD_DEPTH,
                        help="read-ahead depth of the `deep` mode, in blocks")
    parser.add_argument("--max-blocks", type=int,
                        default=DEFAULT_LINUX_RA_MAX_BLOCKS,
                        help="`linux_ra` window ceiling, in blocks")
    parser.add_argument("--output-dir", default=None,
                        help=f"where the csv and md go (default {OUTPUT_DIR})")
    args = parser.parse_args(argv)

    trace_dir = Path(args.trace_dir) if args.trace_dir else MSRC_TRACE_DIR
    per_trace = None if args.full else (args.per_trace or None)
    traces = discover_traces(trace_dir)
    if not traces:
        print(f"No .revised or .oracleGeneral.zst traces found in {trace_dir}")
        return 1

    total_gb = sum(trace_size_bytes(p) for p in traces) / 1e9
    revised = [p for p in traces if trace_format(p) == "revised"]
    revised_gb = sum(trace_size_bytes(p) for p in revised) / 1e9
    print(RULE)
    print("MSRC COLLECTION SWEEP - every trace, every mode")
    print(RULE)
    print(f"  traces          : {len(traces)}")
    print(f"  collection size : {total_gb:.2f} GB")
    print(f"  modes           : {', '.join(args.modes)}")
    print(f"  cache           : {args.cache_blocks} blocks "
          f"({args.cache_blocks * 512 / 1e6:.1f} MB)")
    print(f"  requests/trace  : "
          f"{'ALL (no cap)' if per_trace is None else f'{per_trace:,}'}")
    print(f"  chunk size      : {args.chunk:,} requests "
          f"(~{args.chunk * 430 / 1e6:.0f} MB resident)")
    print()

    # Calibrate the cost model from a sample of the first .revised trace,
    # which carries the recorded device service times. Oracle block traces
    # record none, so they never calibrate anything.
    latency = None
    if not args.no_cost_model and revised:
        from adaptive_prefetch.trace import iter_csv_chunks
        for chunk in iter_csv_chunks(revised[0], "revised", 20000):
            measured = LatencyModel.measured(
                trace=chunk, prefetch_multiple=args.prefetch_multiple)
            # measured() returns the assumed defaults when the sample has no
            # service times; that must not be labelled MEASURED.
            if any(r.service_ms is not None for r in chunk):
                latency = measured
            break
    if latency is not None:
        print(f"  cost model      : MEASURED from {revised[0].name} - "
              f"hit {latency.hit_us:.1f} / miss {latency.demand_miss_us:.0f} / "
              f"prefetch {latency.prefetch_us:.0f} us")
    else:
        latency = LatencyModel()
        print(f"  cost model      : assumed "
              f"{latency.hit_us:g}/{latency.demand_miss_us:g}/"
              f"{latency.prefetch_us:g} us")
    print(f"  deep depth      : {args.read_ahead_depth} blocks   "
          f"linux_ra max: {args.max_blocks} blocks")
    print()
    print(f"  break-even precision: {latency.break_even_precision():.3f}")
    if latency.break_even_precision() > 1.0:
        print("    Above 1.0, so NO prefetcher can pay for itself and the cost")
        print("    column is degenerate: 'none' wins it by arithmetic. Read the")
        print("    hit-ratio, precision and wasted-I/O columns instead.")
        print("    Use --prefetch-multiple 0.5 to model overlapped prefetch.")

    started = perf_counter()

    def progress(meta, _rows):
        print(f"    {meta['trace']:<12} {meta['size_mb']:>9.1f} MB  "
              f"{meta['seconds']:>6.1f}s", flush=True)

    rows, metas = sweep_collection(
        trace_dir=trace_dir, capacity=args.cache_blocks,
        window_size=args.window_size, lstm_model_path=args.lstm_model,
        requests_per_trace=per_trace, chunk_requests=args.chunk,
        latency_model=latency, modes=tuple(args.modes), progress=progress,
        read_ahead_depth=args.read_ahead_depth, max_blocks=args.max_blocks,
        traces=traces)

    if not rows:
        print("No results produced.")
        return 1

    elapsed = perf_counter() - started
    summary = aggregate_by_mode(rows)
    win_counts = per_trace_winners(rows)
    champ = winners(rows)
    caveats = winner_caveats(rows, champ, latency.break_even_precision())

    scored = max((row["scored"] for row in summary), default=0)
    print()
    print(RULE)
    print(f"RESULTS - {len(rows)} rows over {len(metas)} traces in {elapsed:.0f}s")
    print(RULE)
    print()
    print(f"  {'mode':<20} {'scored':>7} {'mean hit%':>10} {'mean prec%':>11} "
          f"{'mean us/read':>13} {'wasted I/O':>12} {'traces won':>11}")
    print(f"  {'':<20} {'':>7} {'':>10} {'':>11} {'':>13} "
          f"{'(unused/total)':>12}")
    for row in summary:
        if not row["scored"]:
            print(f"  {row['mode']:<20} {'skipped' if row['skipped'] else '-':>7}"
                  f" (no trace produced a result)")
            continue
        total = row["total_prefetches"]
        waste = (f"{row['total_unused']:,}/{total:,}" if total else "-")
        won = win_counts.get(row["mode"], 0)
        print(f"  {row['mode']:<20} {row['scored']:>7} "
              f"{100 * row['mean_hit']:>9.2f}% {100 * row['mean_prec']:>10.2f}% "
              f"{row['mean_us']:>13.2f} {waste:>12} {won:>7}/{scored}")
    print()
    print("  Means above weight every trace equally. Pooled over read blocks")
    print("  (the weighting the benchmark aggregate uses) the hit ratio is:")
    for row in summary:
        if row["scored"]:
            print(f"    {row['mode']:<20} {100 * row['weighted_hit']:>7.2f}%")

    print()
    print("OVERALL WINNERS")
    print(THIN)
    labels = {
        "hit_ratio": "highest mean hit ratio",
        "precision": "highest mean prefetch precision",
        "cost": "lowest mean modelled cost",
        "waste": "fewest wasted prefetches",
    }
    for key, label in labels.items():
        mode = champ.get(key)
        if mode:
            note = f"   ({caveats[key]})" if key in caveats else ""
            print(f"  {label:<34}: {mode}{note}")

    ranked = [row for row in summary if row["scored"]]
    print()
    if not ranked:
        # Every requested mode was skipped (e.g. `--modes lstm` with no
        # artifact): there is nothing to rank, but the rows still get written.
        print("  No mode produced a result, so nothing is ranked.")
    elif (max(ranked, key=lambda r: r["mean_hit"])["mode"]
          == min(ranked, key=lambda r: r["mean_us"])["mode"]):
        print("  The same policy wins on hit ratio and on cost - the clean case.")
    else:
        best_hit = max(ranked, key=lambda r: r["mean_hit"])
        slowest = min(ranked, key=lambda r: r["mean_us"])
        print(f"  NOTE: {best_hit['mode']} wins on hit ratio but "
              f"{slowest['mode']} is cheaper.")
        print("  Prefetching has to be paid for; report both, never one alone.")

    if per_trace is not None:
        print()
        print(f"  SAMPLE: {per_trace:,} requests per trace, not the whole file.")
        if revised:
            print(f"  Parsing the {revised_gb:.2f} GB of .revised text alone is "
                  f"roughly {parse_minutes(revised_gb):.0f} min at the measured "
                  f"{REVISED_PARSE_MB_PER_S} MB/s.")
        print("  Re-run with --full for whole-file numbers (hours).")

    written = write_outputs(rows, summary, win_counts, champ, metas, args,
                            latency, elapsed, caveats)
    print()
    print(f"  wrote {written}")
    return 0


def write_outputs(rows, summary, win_counts, champ, metas, args, latency,
                  elapsed, caveats=None) -> str:
    caveats = caveats or {}
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "msrc_sweep.csv"
    # Feedback-gate diagnostics (`observed_precision`, `precision_margin`,
    # `observed_stride`) explain *why* a gated mode issued the prefetches it
    # did; `observed_precision` is None when the gate never measured anything.
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_COLUMNS)
        writer.writeheader()
        writer.writerows({column: row.get(column) for column in ROW_COLUMNS}
                         for row in rows)

    md_path = output_dir / "msrc_sweep.md"
    # A bare `--full` used to crash here formatting the string 'all' with ','.
    per_trace = ("all" if args.full or not args.per_trace
                 else f"{args.per_trace:,}")
    lines = ["# MSRC collection sweep", ""]
    lines.append(f"- traces: {len(metas)}   rows: {len(rows)}   "
                 f"elapsed: {elapsed:.0f}s")
    lines.append(f"- requests per trace: {per_trace}")
    lines.append(f"- cache: {args.cache_blocks} blocks")
    lines.append(f"- deep depth: {args.read_ahead_depth} blocks; "
                 f"linux_ra max: {args.max_blocks} blocks")
    lines.append(f"- cost model: hit {latency.hit_us:.2f} / "
                 f"miss {latency.demand_miss_us:.1f} / "
                 f"prefetch {latency.prefetch_us:.1f} us "
                 f"(break-even precision {latency.break_even_precision():.4f})")
    lines.append("")
    lines.append("`prefetch_recall` is omitted: it needs whole-stream future")
    lines.append("knowledge, which a bounded-memory streamed pass cannot have.")
    lines.append("Hit ratio, precision, wasted I/O, device I/Os and modelled cost")
    lines.append("are exact.")
    lines.append("")
    lines.append("## Overall winners")
    lines.append("")
    for key, label in (("hit_ratio", "mean hit ratio"),
                       ("precision", "mean prefetch precision"),
                       ("cost", "lowest mean modelled cost"),
                       ("waste", "fewest wasted prefetches")):
        note = f" ({caveats[key]})" if key in caveats else ""
        lines.append(f"- **{label}**: `{champ.get(key)}`{note}")
    lines.append("")
    lines.append("## By mode")
    lines.append("")
    lines.append("`mean` weights every trace equally; `pooled` weights each "
                 "trace by its read blocks, as the benchmark aggregate does.")
    lines.append("")
    lines.append("| mode | traces scored | skipped | mean hit % | pooled hit % "
                 "| mean precision % | pooled precision % | mean us/read "
                 "| device I/Os | blocks moved | traces won |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: "
                 "| ---: | ---: | ---: |")
    for row in summary:
        if not row["scored"]:
            lines.append(f"| `{row['mode']}` | 0 | {row['skipped']} | - | - "
                         "| - | - | - | - | - | 0 |")
            continue
        lines.append(
            f"| `{row['mode']}` | {row['scored']} | {row['skipped']} "
            f"| {100 * row['mean_hit']:.2f} | {100 * row['weighted_hit']:.2f} "
            f"| {100 * row['mean_prec']:.2f} | {100 * row['weighted_prec']:.2f} "
            f"| {row['mean_us']:.2f} | {row['total_device_ios']:,} "
            f"| {row['total_blocks_moved']:,} "
            f"| {win_counts.get(row['mode'], 0)} |")
    lines.append("")
    lines.append("## Per trace")
    lines.append("")
    traces = list(dict.fromkeys(r["trace"] for r in rows))
    modes = list(dict.fromkeys(r["mode"] for r in rows))
    lines.append("| trace | " + " | ".join(f"`{m}`" for m in modes) + " | winner |")
    lines.append("| --- |" + " ---: |" * (len(modes) + 1))
    lookup = {(r["trace"], r["mode"]): r for r in rows}
    for trace in traces:
        cells, best_mode, best_hit = [], None, -1.0
        for mode in modes:
            row = lookup.get((trace, mode))
            if row is None or row.get("hit_ratio") is None:
                cells.append("-")
                continue
            hit = row["hit_ratio"]
            cells.append(f"{100 * hit:.2f}")
            if row["read_blocks"] > 0 and hit > best_hit:
                best_hit, best_mode = hit, mode
        lines.append(f"| `{trace}` | " + " | ".join(cells) + f" | {best_mode} |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return f"{csv_path.name} and {md_path.name}"


if __name__ == "__main__":
    raise SystemExit(main())
