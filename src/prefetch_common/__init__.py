"""Code shared by the disk study (``adaptive_prefetch``) and the web study
(``webprefetch``).

* :mod:`prefetch_common.utility` -- the one objective both studies report,
  ``U(c) = (hits - c * issued) / requests``, and the mapping from the disk
  cost model onto ``c``.
* :mod:`prefetch_common.stats` -- paired, cluster and session bootstraps and
  an optional Wilcoxon wrapper.
* :mod:`prefetch_common.provenance` -- ``provenance.json`` for every
  ``results/<study>/`` directory.
* :mod:`prefetch_common.schemas` -- the results contract: the exact columns of
  every results file the paper reads.

Stdlib only. numpy is used when present (``stats.session_bootstrap``) and
scipy when present (``stats.wilcoxon``); neither is required.
"""
