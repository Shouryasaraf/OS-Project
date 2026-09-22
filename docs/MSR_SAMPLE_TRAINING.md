# MSR sample classifier adaptation

The existing four-class Gaussian Naive Bayes classifier was adapted on
`data/samples/msr-cambridge1-sample.csv` using `run_review2.ps1 -TrainMSR`.
It is **not** a model trained from four real workload classes: the CSV has no
workload labels and its small set of windows is overwhelmingly random-like.

- Source: 1,000 requests; 31 complete windows of 32 requests. The final eight
  requests are not used for training.
- Initial training: 100 labelled synthetic windows per class (400 total).
- Real-sample adaptation: 13 windows, all given the *proxy* label `random`.
  Selection required at least 50% reads, at most 20% contiguous transitions,
  at most 20% repeated non-contiguous stride, and at least 75% long address
  jumps. No model prediction or cache-success outcome was used as a label.
- Artifact: `models/msr_sample_gnb.json` (included in the repository). It stores model
  statistics, feature names, rule and source hash for reproducibility.
- Check: held-out **synthetic** accuracy was 1.000 before and 1.000 after the
  update. This says nothing about true classification accuracy on the MSR
  sample because it has no verified four-class labels.

To regenerate and load the artifact from the repository root:

```powershell
.\run_review2.ps1 -TrainMSR
.\run_review2.ps1 -Trace .\data\samples\msr-cambridge1-sample.csv -ModelPath .\models\msr_sample_gnb.json
```

The second command verifies that the saved model can be used in replay. It
uses the same source trace for inference, so its cache metrics must **not** be
reported as held-out real-trace gains. A credible real-data evaluation needs
a separate trace with independent labels or a carefully specified proxy-label
agreement study across separate volumes.

In this in-sample replay, the adapted model selected no prefetch and produced
a 1.54% hit ratio. The earlier synthetic-only adaptive model produced 2.33%
on the same sample. This is a warning that adapting to a tiny, mostly
random-like trace can make prefetch policy selection more conservative; it is
not evidence that either model generalizes better.
