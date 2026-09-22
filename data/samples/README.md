# Sample traces

- `demo.csv` is a generated normalized trace for inspecting the request schema.
- `msr-cambridge1-sample.csv` is a 1,000-request MSR-format sample used by the
  benchmark command. Its original source and selection method have not been
  independently verified, so do not present it as a full Cambridge benchmark.
- `MSR_trace_format.txt` preserves the supplied format and attribution notes
  for MSR-style traces. The note describes the broader dataset; it does not
  establish the provenance of this particular sample.

From the repository root, run:

```powershell
.\run_review2.ps1 -Trace .\data\samples\msr-cambridge1-sample.csv
```
