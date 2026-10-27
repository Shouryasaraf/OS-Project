# Sample traces

- `demo.csv` is a generated normalized trace for inspecting the request schema.
- `msr-cambridge1-sample.csv` is a 1,000-request MSR-format sample used by the
  benchmark command. Its original source and selection method have not been
  independently verified, so do not present it as a full Cambridge benchmark.
- `MSR_trace_format.txt` preserves the supplied format and attribution notes
  for MSR-style traces. The note describes the broader dataset; it does not
  establish the provenance of this particular sample.
- `msrc/` holds seven of the smallest `MSRC-trace-003` `.revised` files,
  copied unmodified, so the `revised` loader can be exercised without the
  full collection. Size-selected, not representative; see its README.
- `web_fixture.csv` is a **synthetic** 200-row clickstream for the web-study
  tests. It has the RetailRocket `events.csv` header and value conventions but
  contains no RetailRocket data; never use it as research data.
  `make_web_fixture.py` regenerates it byte for byte (stdlib, seed 42), and
  `webprefetch prepare` deliberately rejects it because it checks the real
  file's SHA-256.

From the repository root, run:

```powershell
$env:PYTHONPATH='src'
python -m adaptive_prefetch replay .\data\samples\msr-cambridge1-sample.csv
```
