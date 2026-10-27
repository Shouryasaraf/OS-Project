# MSRC light sample (E7): pre-registered checks and decision rules

Written **before** the full run of `python reproduce.py msrc`, and **after**
experiments E1-E6 had been run and reported. It is therefore not blind to
those results; it fixes the procedure, the label rule, the comparisons and the
words before any number of this study was looked at. Any later change is
added as a dated amendment below; the original text stays.

## 0. What this study is and is not

- **Data.** Every `*.revised` file in `data/samples/msrc/` at run time, sorted
  by name. At the time of writing that is five MSRC-trace-003 files
  (Narayanan et al., FAST 2008): `rsrch_1`, `wdev_1`, `wdev_3`, `web_1`,
  `web_3`. The committed subset was **chosen by file size**, not at random
  and not for workload; it is not representative of the collection. Nothing
  here is a benchmark claim, and no number here is compared with E1 (oracle
  traces, writes replayed as reads) or with the D13-D17 MSRC figures.
- **Purpose.** (a) A labelled-window **transfer check**: these files carry a
  per-request `seq`/`rand` flag (`Request.pattern`), the only real workload
  label in the project. It asks whether a classifier trained on synthetic
  windows, as `adaptive` is, agrees with that flag. (b) A **sanity replay**
  of six modes on the same files.
- The flag is used for scoring only. It never reaches a feature, a
  classifier input, or candidate generation.

## 1. Windows, features and labels

- Non-overlapping windows of 32 requests in trace order, starting at request
  0; a trailing partial window is dropped (as in `simulator.replay`).
- Features: the exact 12-feature contextual vector `adaptive` uses,
  `features.WindowContext().observe(window, dominant_stride(window))`, one
  context per trace, fed every window in order (default age-table bound).
- **Primary label (`majority`)**: a window is `seq` when at least 17 of its
  32 requests carry the flag `seq`, else `not_seq`.
- **Sensitivity label (`quarter`)**: `seq` when at least 8 of 32 do. Reported
  beside the primary, never instead of it.
- **Class mapping.** The synthetic classes are `sequential`, `strided`,
  `random`, `mixed`; the flag has only two values. A prediction of
  `sequential` maps to `seq`; `strided`, `random` and `mixed` map to
  `not_seq`. The full four-class prediction counts are reported too.
  `adaptive_evidence`'s rule (`simulator.route_window`) maps `sequential` to
  `seq` and `strided`/`mixed`/`none` to `not_seq`.

## 2. Classifiers scored (no tuning on these traces)

1. **Shipped**: `benchmark.make_model(seed=42)` with `gnb` (the shipped
   default) and with `qda`.
2. **Synthetic-trained sklearn families**, fitted on exactly the windows
   `make_model(seed=42)` fits (100 per class, contextual stream per class):
   GaussianNB, LDA, QDA (`reg_param=0.01`), logistic regression, kNN (k=5),
   RBF SVM, decision tree, random forest (200 trees), extra trees (200),
   histogram gradient boosting, MLP (64-32). Library defaults otherwise,
   `random_state=0`; logistic, kNN, SVM and MLP behind a `StandardScaler`.
   Fitted on the four synthetic classes and mapped as in section 1.
3. **Within-real baseline, leave-one-trace-out (LOTO)**: the same eleven
   families fitted on the binary labels of the real windows of every other
   trace and scored on the held-out trace. A fold whose training windows hold
   one class only is a `skipped:` row. Same defaults, no tuning.
4. **Trivial baselines**: `majority` (the training folds' majority label in
   LOTO; on the synthetic side, always `not_seq`) and `route_window`.

## 3. Metrics, intervals and n

- Per trace and pooled over traces: accuracy, balanced accuracy (mean of the
  two class recalls) and macro-F1 over {`seq`, `not_seq`}. Balanced accuracy
  and macro-F1 are `None` when the truth holds one class only.
- Intervals: 95% percentile bootstrap resampling **windows** with
  replacement, 2,000 replicates, seed 42. Windows of one trace are serially
  correlated, so these intervals are optimistic; pooled rows are dominated by
  the largest trace and are reported with per-trace n.
- **Small n.** A trace row is flagged `small_n` when it has fewer than 100
  windows or fewer than 10 windows of either class. A flagged row gets no
  verdict word.

## 4. Comparisons and decision rule

- Primary: balanced accuracy (majority label) of every scored classifier
  against (i) the shipped GNB and (ii) `route_window`, per trace and pooled,
  as a paired bootstrap of the difference on the same windows (same
  replicates as section 3).
- **"Better" / "worse"**: the 95% interval of the difference lies wholly
  above / below 0. **"No difference detected"**: it contains 0.
  **"Insufficient n"**: the row is `small_n` or a metric is `None`.
- **"Transfers"** (a synthetic-trained classifier on a trace): the lower end
  of its balanced-accuracy interval is above 0.5.
- Which classifier scores highest is **descriptive only**: no choice made
  here feeds a later result, and a LOTO winner is not a deployable model.

## 5. Sanity replay

- Modes `none`, `sequential`, `adaptive` (shipped GNB, `make_model(42)`),
  `adaptive_evidence`, `deep@8`, `linux_ra@256`; capacity 2048 blocks, window
  32; under `LatencyModel.measured(trace=...)` (mean `t1` service time, hit
  cost 1% of it, prefetch multiple 1.0, so the cost column is degenerate) and
  the assumed `LatencyModel(5, 100, 50)`.
- Two views: `native` (the loader's own semantics: `WS` rows are writes,
  write-allocated into the cache, never trigger prefetch and never count as
  hits; hit ratio is over read blocks) and `writes_as_reads` (every request
  replayed as a read, the oracle-trace convention; a sensitivity view). These
  files are write-dominated, so `native` hit ratios rest on few read blocks;
  read-block counts are reported on every row.
- Every replay figure is **simulated**. Each hit-ratio figure is paired with
  `device_ios` and `blocks_moved` ratios to `none` on the same trace, view
  and cost model. A replay is deterministic and has no sampling unit, so it
  has no interval and no verdict word: these rows are descriptive.

## Amendment A1 (2026-10-27, before the full run; no result seen)

- **Seven traces.** `wdev_2` and `rsrch_2` were committed after the text
  above was written, so the data are now all seven files the sample's README
  lists: `rsrch_1`, `rsrch_2`, `wdev_1`, `wdev_2`, `wdev_3`, `web_1`,
  `web_3`. Section 0's rule ("every `*.revised` file at run time") already
  covers them; LOTO (section 2.3) is now over seven folds.
- **Leave-one-host-family-out (LOFO)**, added as a **secondary** within-real
  protocol. Volumes of one MSRC host (`rsrch_*`, `wdev_*`, `web_*`; the
  family is the file stem before the last `_`) are near-twins, so a LOTO fold
  can train on a sibling of its test volume. LOFO fits the same families on
  the other two hosts' windows and scores every volume of the held-out host.
  Same defaults, same skip rule, same metrics and words. Where LOTO and LOFO
  disagree, the LOFO figure is the one quoted for "within-real" transfer.
