"""Full MSRC collection sweep: every trace, every mode, one winner table.

    python sweep_modes.py                 # 250k requests per trace, all 32
    python sweep_modes.py --full          # every request in every file (slow)
    python sweep_modes.py --per-trace 50000
    python sweep_modes.py --modes none sequential adaptive adaptive_evidence

Streams each file in bounded chunks, so memory stays flat regardless of file
size. Per-trace rows go to outputs/msrc_sweep.csv and outputs/msrc_sweep.md.
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
from adaptive_prefetch.simulator import DEFAULT_CAPACITY, LatencyModel  # noqa: E402
from adaptive_prefetch.sweep import (  # noqa: E402
    DEFAULT_CHUNK, DEFAULT_REQUESTS_PER_TRACE, RANKING_MODES, aggregate_by_mode,
    capped_chunks, discover_traces, per_trace_winners, sweep_collection,
    trace_size_bytes, winners,
)

RULE = "=" * 96
THIN = "-" * 96


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
    args = parser.parse_args(argv)

    trace_dir = Path(args.trace_dir) if args.trace_dir else MSRC_TRACE_DIR
    per_trace = None if args.full else (args.per_trace or None)
    traces = discover_traces(trace_dir)
    if not traces:
        print(f"No .revised traces found in {trace_dir}")
        return 1

    total_gb = sum(trace_size_bytes(p) for p in traces) / 1e9
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

    # Calibrate the cost model from a sample of the first trace, which carries
    # the recorded device service times.
    latency = None
    if not args.no_cost_model:
        from adaptive_prefetch.trace import iter_csv_chunks
        for chunk in iter_csv_chunks(traces[0], "revised", 20000):
            latency = LatencyModel.measured(
                trace=chunk, prefetch_multiple=args.prefetch_multiple)
            break
    if latency is not None:
        print(f"  cost model      : MEASURED from {traces[0].name} - "
              f"hit {latency.hit_us:.1f} / miss {latency.demand_miss_us:.0f} / "
              f"prefetch {latency.prefetch_us:.0f} us")
    else:
        latency = LatencyModel()
        print(f"  cost model      : assumed "
              f"{latency.hit_us:g}/{latency.demand_miss_us:g}/"
              f"{latency.prefetch_us:g} us")
    print()

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
        latency=latency, modes=tuple(args.modes), progress=progress)

    if not rows:
        print("No results produced.")
        return 1

    elapsed = perf_counter() - started
    summary = aggregate_by_mode(rows)
    win_counts = per_trace_winners(rows)
    champ = winners(rows)

    scored = summary[0]["scored"] if summary else 0
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
        total = row["total_prefetches"]
        waste = (f"{row['total_unused']:,}/{total:,}" if total else "-")
        won = win_counts.get(row["mode"], 0)
        print(f"  {row['mode']:<20} {row['scored']:>7} "
              f"{100 * row['mean_hit']:>9.2f}% {100 * row['mean_prec']:>10.2f}% "
              f"{row['mean_us']:>13.2f} {waste:>12} {won:>7}/{scored}")

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
            print(f"  {label:<34}: {mode}")

    slowest = min(summary, key=lambda r: r["mean_us"])
    best_hit = max(summary, key=lambda r: r["mean_hit"])
    print()
    if best_hit["mode"] == slowest["mode"]:
        print("  The same policy wins on hit ratio and on cost - the clean case.")
    else:
        print(f"  NOTE: {best_hit['mode']} wins on hit ratio but "
              f"{slowest['mode']} is cheaper.")
        print("  Prefetching has to be paid for; report both, never one alone.")

    if per_trace is not None:
        print()
        print(f"  SAMPLE: {per_trace:,} requests per trace, not the whole file.")
        print(f"  Collection parse time alone is roughly "
              f"{total_gb * 0.5:.0f} min at the measured 2.1 MB/s.")
        print("  Re-run with --full for whole-file numbers (hours).")

    written = write_outputs(rows, summary, win_counts, champ, metas, args,
                            latency, elapsed)
    print()
    print(f"  wrote {written}")
    return 0


def write_outputs(rows, summary, win_counts, champ, metas, args, latency,
                  elapsed) -> str:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "msrc_sweep.csv"
    fields = ["trace", "mode", "hit_ratio", "precision", "unused",
              "total_prefetches", "mean_us", "policy_switches", "read_blocks",
              # Feedback-gate diagnostics: these explain *why* a gated mode
              # issued the prefetches it did. `observed_precision` is None
              # when the gate never measured anything.
              "observed_precision", "precision_margin", "observed_stride"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    md_path = OUTPUT_DIR / "msrc_sweep.md"
    lines = ["# MSRC collection sweep", ""]
    lines.append(f"- traces: {len(metas)}   rows: {len(rows)}   "
                 f"elapsed: {elapsed:.0f}s")
    lines.append(f"- requests per trace: "
                 f"{'all' if args.full else args.per_trace:,}")
    lines.append(f"- cache: {args.cache_blocks} blocks")
    lines.append(f"- cost model: hit {latency.hit_us:.2f} / "
                 f"miss {latency.demand_miss_us:.1f} / "
                 f"prefetch {latency.prefetch_us:.1f} us")
    lines.append("")
    lines.append("`prefetch_recall` is omitted: it needs whole-stream future")
    lines.append("knowledge, which a bounded-memory streamed pass cannot have.")
    lines.append("Hit ratio, precision, wasted I/O and modelled cost are exact.")
    lines.append("")
    lines.append("## Overall winners")
    lines.append("")
    for key, label in (("hit_ratio", "mean hit ratio"),
                       ("precision", "mean prefetch precision"),
                       ("cost", "lowest mean modelled cost"),
                       ("waste", "fewest wasted prefetches")):
        lines.append(f"- **{label}**: `{champ.get(key)}`")
    lines.append("")
    lines.append("## By mode")
    lines.append("")
    lines.append("| mode | traces scored | mean hit % | mean precision % | "
                 "mean us/read | traces won |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for row in summary:
        lines.append(
            f"| `{row['mode']}` | {row['scored']} | {100 * row['mean_hit']:.2f} "
            f"| {100 * row['mean_prec']:.2f} | {row['mean_us']:.2f} "
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
            if row is None:
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
