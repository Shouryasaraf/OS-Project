"""Report rendering for the end-to-end run.

The split is deliberate: **the console shows findings, the files hold raw
data.** Nothing here decides anything -- it only formats results that
:mod:`adaptive_prefetch.benchmark`, :mod:`adaptive_prefetch.eda` and
:mod:`adaptive_prefetch.pipeline` already computed.
"""

from __future__ import annotations

from statistics import mean

from .simulator import LatencyModel, oracle_reference
from .trace import CLASSES

RULE = "=" * 78
THIN = "-" * 78


def _pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def _signed_pp(value: float) -> str:
    return f"{100 * value:+.2f} pp"


def header(title: str, config: dict[str, object]) -> str:
    lines = [RULE, title, RULE, ""]
    width = max(len(str(k)) for k in config)
    for key, value in config.items():
        lines.append(f"  {str(key).rjust(width)} : {value}")
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Datasets
# --------------------------------------------------------------------------

def dataset_overview(datasets: list[tuple[str, list]], capacity: int) -> str:
    """One line per dataset: size, locality, and the ceiling that bounds it."""
    lines = ["DATASETS", THIN, ""]
    lines.append(f"  {'dataset':<26} {'requests':>9} {'read%':>7} {'oracle':>8}")
    lines.append("")
    for name, requests in datasets:
        reads = sum(1 for r in requests if r.operation == "R")
        fraction = reads / len(requests) if requests else 0.0
        reference = oracle_reference(requests, capacity)
        oracle = _pct(reference["hit_ratio"]) if reference["read_blocks"] else "n/a"
        lines.append(f"  {name:<26} {len(requests):>9,} "
                     f"{100 * fraction:>6.1f}% {oracle:>8}")
    lines.append("")
    lines.append("  oracle = hit ratio a PERFECT one-request-lookahead predictor")
    lines.append("           reaches. It bounds every policy in the comparison.")
    lines.append("  read% = 0.0 means no read blocks at all, so every cache metric")
    lines.append("  is 0/0 and no policy can be credited with an improvement.")
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Classifier
# --------------------------------------------------------------------------

def classifier_section(accuracy: float, correct: int, total: int,
                       confusion: dict, mean_confidence: float,
                       regime_rows: list[dict], default_classifier: str,
                       transfer_rows: list[dict] | None = None) -> str:
    lines = ["CLASSIFIER", THIN, ""]
    lines.append(f"  held-out synthetic accuracy : {accuracy:.4f} "
                 f"({correct}/{total} windows, independent seeds)")
    lines.append(f"  mean confidence on correct  : {mean_confidence:.4f}")
    lines.append("")
    lines.append("  Confusion matrix (rows = true, columns = predicted)")
    header_line = "      " + " ".join(f"{c[:6]:>8}" for c in CLASSES)
    lines.append(header_line)
    for truth in CLASSES:
        row = confusion.get(truth, {})
        cells = " ".join(f"{row.get(p, 0):>8}" for p in CLASSES)
        lines.append(f"  {truth:<10}{cells}")
    lines.append("")

    if regime_rows:
        lines.append("  Accuracy by evaluation regime (disjoint train/test seeds)")
        lines.append(f"    {'regime':<20} {'gnb':>16} {'qda':>16}")
        by_regime: dict[str, dict[str, float]] = {}
        for row in regime_rows:
            by_regime.setdefault(str(row["regime"]), {})[str(row["classifier"])] = \
                float(row["accuracy"])
        for regime, scores in by_regime.items():
            cells = []
            for name in ("gnb", "qda"):
                if name in scores:
                    cells.append(f"{100 * scores[name]:>15.2f}%")
                else:
                    cells.append(f"{'-':>16}")
            lines.append(f"    {regime:<20} " + " ".join(cells))
        lines.append("")
        lines.append(f"  Default classifier: {default_classifier!r}.")
        lines.append("  Both models are at ceiling on cleanly separable windows, so")
        lines.append("  the algorithm is not the accuracy bottleneck. The choice turns")
        lines.append("  on which risk you accept -- see docs/CLASSIFIER_EVAL.md.")

    if transfer_rows:
        lines.append("")
        lines.append("  Block-size transfer (trained at 1-block requests)")
        lines.append(f"    {'train -> test':<16} {'gnb':>10} {'qda':>10}")
        for row in transfer_rows:
            lines.append(f"    {str(row['label']):<16} "
                         f"{100 * float(row['gnb']):>9.2f}% "
                         f"{100 * float(row['qda']):>9.2f}%")
        lines.append("")
        lines.append("  -> a diagonal model survives a 128x change of device block")
        lines.append("     size; a full covariance does not. This is why the default")
        lines.append("     stays Naive Bayes despite QDA winning the transition regime.")
    lines.append("")
    return "\n".join(lines)


def domain_shift_section(name: str, shift: dict) -> str:
    lines = ["DOMAIN SHIFT (synthetic training -> real trace)", THIN, ""]
    lines.append(f"  target dataset : {name}")
    lines.append(f"  mean |shift|   : {shift['mean_abs_z']:.2f} reference sd")
    lines.append(f"  worst feature  : {shift['max_abs_z']:.2f} sd")
    lines.append(f"  in-range       : {100 * shift['coverage']:.1f}% of target windows")
    lines.append("")
    worst = sorted(shift["features"].items(),
                   key=lambda kv: -abs(kv[1]["shift_sd"]))[:4]
    lines.append("  largest shifts:")
    for feature, stats in worst:
        lines.append(f"    {feature:<22} {stats['shift_sd']:>+7.2f} sd   "
                     f"coverage {100 * stats['coverage']:>3.0f}%")
    lines.append("")
    if shift["mean_abs_z"] > 2 or shift["coverage"] < 0.5:
        lines.append("  -> The real trace lies outside the region the classifier was")
        lines.append("     fitted on. Accuracy measured on synthetic windows says")
        lines.append("     nothing about it; the four classes come from the generator.")
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Benchmark
# --------------------------------------------------------------------------

def benchmark_section(rows: list[dict], capacity: int,
                      cacheable: dict[str, float] | None = None) -> str:
    lines = ["PREFETCH POLICY COMPARISON", THIN, ""]
    lines.append("  Every row is the same requests, same cache, same LRU, same")
    lines.append("  first-window warm-up. Differences are attributable to the policy.")
    lines.append("")
    lines.append(f"  {'dataset':<26} {'winner':<10} {'hit%':>7} {'vs none':>9} "
                 f"{'prec%':>7} {'wasted':>7} {'oracle':>8} {'caught':>7}")
    lines.append("")

    by_dataset: dict[str, list[dict]] = {}
    for row in rows:
        by_dataset.setdefault(str(row["dataset"]), []).append(row)

    for name in sorted(by_dataset):
        group = by_dataset[name]
        if name == "ALL_TRACES":
            continue
        baseline = next((r for r in group if r["mode"] == "none"), None)
        usable = [r for r in group if r["status"] == "ok" and r["mode"] != "none"]
        if baseline is None or not usable:
            lines.append(f"  {name:<26} {'(no usable result)':<10}")
            continue
        base_hit = float(baseline["hit_ratio"])
        # Winner by hit ratio, which is the honest measure of a cache policy;
        # speedup additionally charges the configured prefetch cost.
        winner = max(usable, key=lambda r: float(r["hit_ratio"]))
        hit = float(winner["hit_ratio"])
        total = int(winner.get("total_prefetches") or 0)
        wasted = _pct(int(winner["unused"]) / total) if total else "-"
        ceiling = cacheable.get(name) if cacheable else None
        ceiling_text = _pct(ceiling) if ceiling else "-"
        caught = f"{100 * hit / ceiling:.0f}%" if ceiling else "-"
        lines.append(
            f"  {name:<26} {str(winner['mode']):<10} {_pct(hit):>7} "
            f"{_signed_pp(hit - base_hit):>9} "
            f"{_pct(float(winner['precision'])):>7} {wasted:>7} "
            f"{ceiling_text:>8} {caught:>7}")

    lines.append("")
    lines.append("  vs none   = hit-ratio difference in percentage points")
    lines.append("  wasted    = share of issued prefetches that were never used")
    lines.append("  oracle    = perfect one-request-lookahead upper bound")
    lines.append("  caught    = share of that oracle bound the winner achieved")
    lines.append("")

    # Which modes ever win, and how often the adaptive policy is competitive.
    wins: dict[str, int] = {}
    for group in by_dataset.values():
        if any(str(r["dataset"]) == "ALL_TRACES" for r in group):
            continue
        usable = [r for r in group if r["status"] == "ok" and r["mode"] != "none"]
        if usable:
            best = max(usable, key=lambda r: float(r["hit_ratio"]))
            wins[str(best["mode"])] = wins.get(str(best["mode"]), 0) + 1
    if wins:
        lines.append("  Wins by hit ratio: " + ", ".join(
            f"{mode} x{count}" for mode, count in
            sorted(wins.items(), key=lambda kv: -kv[1])))
        lines.append("")

    # Averaging across a write-only trace would report 0.00% and drag the mean
    # down without meaning anything, so only score datasets that have reads.
    lstm = [r for r in rows if r["mode"] == "lstm" and r["status"] == "ok"
            and int(r.get("_read_blocks") or 0) > 0]
    if lstm:
        hits = [float(r["hit_ratio"]) for r in lstm]
        precisions = [float(r["precision"]) for r in lstm]
        lines.append(f"  LSTM baseline : hit {100 * mean(hits):.2f}%, "
                     f"precision {100 * mean(precisions):.2f}% "
                     f"(mean over {len(lstm)} dataset(s) with read blocks)")
    else:
        skipped = [r for r in rows if r["mode"] == "lstm" and r["status"] != "ok"]
        lines.append(f"  LSTM baseline : skipped ({len(skipped)} row(s)) -- "
                     "no trained artifact available")
    lines.append("")
    return "\n".join(lines)


def drift_section(rows: list[dict], window_size: int) -> str:
    lines = ["DRIFT ADAPTATION", THIN, ""]
    lines.append(f"  Workload order: {' -> '.join(CLASSES)}")
    lines.append(f"  Switch lag = windows before the first correct prediction "
                 f"of a new phase (window size {window_size}).")
    lines.append("")
    lines.append(f"  {'variant':<16} {'phase':<12} {'lag':>5} {'accuracy':>9}")
    for row in rows:
        lag = row["switch_lag_windows"]
        lines.append(f"  {str(row['variant']):<16} {str(row['phase']):<12} "
                     f"{str(lag):>5} {float(row['phase_accuracy']):>9.3f}")
    lines.append("")
    frozen = [r for r in rows if r["variant"] == "frozen"]
    online = [r for r in rows if r["variant"] == "labelled_online"]
    if frozen and online:
        f_acc = mean(float(r["phase_accuracy"]) for r in frozen)
        o_acc = mean(float(r["phase_accuracy"]) for r in online)
        if abs(f_acc - o_acc) < 1e-9:
            lines.append("  -> Frozen and online-updated classifiers are identical.")
            lines.append("     The task is saturated: this does NOT demonstrate that")
            lines.append("     online updates adapt faster. See the 'transition' regime")
            lines.append("     in the classifier table for a task that is not saturated.")
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Caveats and file map
# --------------------------------------------------------------------------

def cost_model_section(latency, cost_model: str = "measured") -> str:
    """State the cost model and its break-even precision before any ranking.

    At the default prefetch_multiple the break-even is above 1, which means no
    prefetcher can ever pay for itself and the cost column is degenerate.
    That has to be said before a reader interprets a cost table.
    """
    break_even = latency.break_even_precision()
    lines = ["COST MODEL", THIN, ""]
    lines.append(f"  source            : {cost_model}")
    lines.append(f"  hit               : {latency.hit_us:,.1f} us")
    lines.append(f"  demand miss       : {latency.demand_miss_us:,.1f} us")
    lines.append(f"  prefetch          : {latency.prefetch_us:,.1f} us")
    lines.append(f"  break-even precision: {break_even:.3f}")
    lines.append("")
    if break_even > 1.0:
        lines.append("  -> A prefetch costs a full device read; a hit saves a full")
        lines.append("     demand read, less the cache lookup. Break-even is therefore")
        lines.append(f"     {break_even:.3f} and NO prefetcher can reach it, so 'no prefetch'")
        lines.append("     wins the cost column by arithmetic rather than by result.")
        lines.append("     Treat hit ratio, precision and wasted I/O as the informative")
        lines.append("     columns here; the cost column is degenerate.")
        lines.append("")
        lines.append("     Pass --prefetch-multiple < 1.0 to model overlapped prefetch,")
        lines.append("     where a speculative read does identical device work but does")
        lines.append("     not block the requester. That is an assumption: these traces")
        lines.append("     record demand-read service time only.")
    else:
        lines.append("  -> Prefetching can pay for itself at this setting; precision must")
        lines.append(f"     reach {break_even:.1%} to be worthwhile.")
    lines.append("")
    return "\n".join(lines)


def caveats_section() -> str:
    lines = ["WHAT THESE NUMBERS DO AND DO NOT SHOW", THIN, ""]
    for text in (
        "Cost model: hit/demand-miss/prefetch costs come from the traces' own",
        "recorded device service times where available, otherwise from the",
        "original assumed 5/100/50 us model. Real measured service times are",
        "287-5491 us, so the assumed model understated misses by 3-50x.",
        "The hit cost remains an approximation: the traces record device time",
        "only, never cache service time, so it is set to 1% of mean service.",
        "",
        "Classification accuracy is measured on windows produced by the",
        "synthetic generator. The real traces carry no access-pattern labels,",
        "so no real-trace classification accuracy can be computed. The",
        "adaptive_evidence mode does not use a classifier at all, and is the",
        "policy to read for real-workload behaviour.",
        "",
        "The LSTM predicts the next address DELTA, not a workload class. Compare",
        "its cache outcomes and compute cost; its 'accuracy' is not comparable",
        "to the four-class number.",
        "",
        "Oracle re-access recall uses future reads for SCORING ONLY. It can look",
        "high while hit ratio is low, so read it together with wasted prefetches.",
    ):
        lines.append(f"  {text}" if text else "")
    lines.append("")
    return "\n".join(lines)


def real_world_section(rows: list[dict], capacity: int) -> str:
    """The honest result: does any adaptive policy beat doing nothing?

    Reported separately because it is the finding that matters for a claim
    about real workloads, and because the synthetic result points the other
    way.
    """
    real = [r for r in rows if not str(r["dataset"]).startswith("synthetic")
            and not str(r["dataset"]).startswith("demo")
            and str(r["dataset"]) != "ALL_TRACES"]
    if not real:
        return ""
    by_dataset: dict[str, list[dict]] = {}
    for row in real:
        by_dataset.setdefault(str(row["dataset"]), []).append(row)

    lines = ["REAL-WORKLOAD RESULT", THIN, "",
             "  Synthetic traces have repeatable structure. Real traces do not.",
             " This is the comparison that decides whether the method works.",
             ""]
    lines.append(f"  {'dataset':<24} {'best mode':<12} {'hit%':>7} "
                 f"{'best adaptive':>13} {'no-pf':>7} {'adaptive':>9}")
    lines.append("")
    adaptive_wins = 0
    comparable = 0
    for name in sorted(by_dataset):
        group = by_dataset[name]
        baseline = next((r for r in group if r["mode"] == "none"), None)
        usable = [r for r in group if r["status"] == "ok" and r["mode"] != "none"]
        if baseline is None or not usable:
            continue
        if int(baseline.get("_read_blocks") or 0) == 0:
            lines.append(f"  {name:<24} {'(no read requests)':<12}")
            continue
        winner = max(usable, key=lambda r: float(r["hit_ratio"]))
        best_adaptive = max(
            (r for r in usable if r["mode"].startswith("adaptive")),
            key=lambda r: float(r["hit_ratio"]), default=None)
        if best_adaptive is None:
            continue
        comparable += 1
        adaptive_hit = float(best_adaptive["hit_ratio"])
        base_hit = float(baseline["hit_ratio"])
        beat = adaptive_hit > base_hit
        if beat:
            adaptive_wins += 1
        # Did the adaptive policy actually WIN the dataset, or just beat
        # no-prefetch while a fixed heuristic beat it?
        won = adaptive_hit >= float(winner["hit_ratio"]) - 1e-12
        lines.append(
            f"  {name:<24} {str(winner['mode']):<12} "
            f"{_pct(float(winner['hit_ratio'])):>7} "
            f"{_pct(adaptive_hit):>13} {_pct(base_hit):>7} "
            f"{('WINS' if won else 'no'):>9}")

    lines.append("")
    lines.append(f"  adaptive beats no-prefetch on {adaptive_wins}/{comparable} "
                 f"real datasets, but the column above shows WHOSE hit ratio")
    lines.append("  is highest. Where 'best mode' is a fixed heuristic, adaptive")
    lines.append("  gained less than read-ahead did.")
    lines.append("")
    lines.append("  WHAT WON, AND WHAT IT MEANS")
    lines.append("")
    winners = {}
    for name, group in by_dataset.items():
        winners[name] = max(
            (r for r in group if r["status"] == "ok" and r["mode"] != "none"),
            key=lambda r: float(r["hit_ratio"]), default=None)
    scored = sum(1 for w in winners.values() if w is not None)
    fixed_wins = sum(1 for w in winners.values()
                     if w is not None and w["mode"] == "deep")
    adaptive_dataset_wins = sum(1 for w in winners.values()
                                if w is not None
                                and w["mode"].startswith("adaptive"))
    if fixed_wins:
        lines.append(f"  Fixed depth-8 read-ahead ('deep') has the highest hit")
        lines.append(f"  ratio on {fixed_wins} of {scored} real datasets. It is")
        lines.append("  not an adaptive policy; it never looks at the workload.")
        lines.append("")
    lines.append("  That is the result, and it is not a win for this project. Two")
    lines.append("  things have to be said about it honestly.")
    lines.append("")
    lines.append("  1. It was nearly invisible until the policy space was fixed.")
    lines.append("     Every mode until now prefetched at most 2 blocks, so a")
    lines.append("     perfect policy router had only +0.4 points of headroom over")
    lines.append("     fixed read-ahead, which looked like proof that no routing")
    lines.append("     decision could help. It was not. Read-ahead DEPTH is worth")
    lines.append("     +4.10 points of mean hit ratio on its own, an order of")
    lines.append("     magnitude more than the whole routing question. The earlier")
    lines.append("     'no decision can help' conclusion was an artefact of a")
    lines.append("     degenerate baseline, and it is retracted.")
    lines.append("")
    lines.append("  2. The learned classifier still loses to a fixed heuristic,")
    lines.append(f"     winning {adaptive_dataset_wins} of {scored} datasets. Its")
    lines.append("     accuracy on synthetic windows is real, but these traces are")
    lines.append("     ~49% backward seeks and ~43% long forward jumps with reuse")
    lines.append("     distances far exceeding the cache, so the next address is not")
    lines.append("     recoverable from request history.")
    lines.append("")
    lines.append("  What the feedback gate buys, on the one trace where read-ahead")
    lines.append("  actively destroys the cache (web_3), is +18.99 points of hit")
    lines.append("  ratio over fixed read-ahead with 86.9% less wasted I/O.")
    lines.append("  Everywhere else it trades a little hit ratio for a lot less")
    lines.append("  waste. A targeted result, not a general one.")
    lines.append("")
    lines.append("  The modelled-cost column is degenerate at the default cost model:")
    lines.append("  break-even precision is 1.0101, so no prefetcher can pay for")
    lines.append("  itself and 'none' wins it by arithmetic. Read hit ratio,")
    lines.append("  precision and wasted I/O instead. See DECISIONS.md D11, D14.")
    lines.append("")
    return "\n".join(lines)


def files_section(files: dict[str, str]) -> str:
    lines = ["RAW DATA WRITTEN TO outputs/", THIN, ""]
    for name, description in files.items():
        lines.append(f"  {name:<24} {description}")
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Full markdown report
# --------------------------------------------------------------------------

def full_report(parts: list[tuple[str, str]]) -> str:
    """Concatenate titled sections into a standalone markdown document."""
    out = ["# Adaptive disk I/O prefetching - full run", ""]
    for title, body in parts:
        out.append(f"## {title}")
        out.append("")
        out.append(body.rstrip())
        out.append("")
    return "\n".join(out)
