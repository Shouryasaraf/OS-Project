# Results fixtures (synthetic)

Every file here is **synthetic test data** with the exact column layout of
`prefetch_common.schemas`. It exists so the paper generators can be built and
tested before the experiments run. **None of it is a result; never quote it.**
Rebuild with `PYTHONPATH=src python tests/fixtures/make_results_fixtures.py`.
The `FIXTURE` marker makes the generators stamp their output "FIXTURE DATA".
