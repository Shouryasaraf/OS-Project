# MSR sample classifier adaptation

The existing four-class Gaussian Naive Bayes classifier was adapted on
`data/samples/msr-cambridge1-sample.csv` using
`python -m adaptive_prefetch train-msr-sample`.
It is **not** a model trained from four real workload classes: the CSV has no
workload labels and its small set of windows is overwhelmingly random-like.

- Source: 1,000 requests; 31 complete windows of 32 requests. The final eight
  requests are not used for training.
- Initial training: 100 labelled synthetic windows per class (400 total).
- Real-sample adaptation: **8** windows, all given the *proxy* label `random`.
  Selection required at least 50% reads, at most 20% contiguous transitions,
  at most 20% repeated non-contiguous stride, and at least 75% long address
  jumps. No model prediction or cache-success outcome was used as a label.
- Artifact: `models/msr_sample_gnb.json` (included in the repository, format
  version 2, JSON rather than pickle). It stores model statistics, the 12
  feature names, class list, decay factor and a source hash for
  reproducibility. `load_model` rejects a mismatch in any of those rather
  than silently scoring against the wrong feature vector.
- Check: held-out **synthetic** accuracy is **0.969 before and 0.956 after**
  the update. The 1.3-point drop is the model shifting mass onto `random` for
  a trace that is 1000 requests of almost nothing else -- which is the
  intended behaviour for weak supervision, and still says nothing about true
  classification accuracy on the MSR sample, because it has no verified
  four-class labels.

*Corrected:* an earlier version of this file reported 13 adapted windows and
a held-out accuracy of 1.000 before and after. The 1.000 was a property of the
old degenerate generator, which had no within-class variance; see
`DECISIONS.md` D10.

To regenerate and load the artifact from the repository root:

```powershell
$env:PYTHONPATH='src'
python -m adaptive_prefetch train-msr-sample
python -m adaptive_prefetch replay .\data\samples\msr-cambridge1-sample.csv --model-path .\models\msr_sample_gnb.json
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
