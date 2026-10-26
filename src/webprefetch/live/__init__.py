"""Real HTTP prefetch replay with exact parity to the web simulator (E6).

Ported from the web-realexp prototype. Modules:

* :mod:`.objects` -- the object store (ASSUMED sizes), page-cache helpers
  and the disk probe.
* :mod:`.plan` -- sampled sessions, per-view prefetch decisions and the
  ``simulator.replay`` reference they must reproduce (needs the [web] extra).
* :mod:`.origin` -- asyncio origin: real ``pread`` from the store, emulated
  RTT and per-session link, ``/hint`` (server WILLNEED), ``/stats``.
* :mod:`.client` -- emulated browsers: ``cachecore.Cache`` per session,
  demand-priority connection pool, recorded gaps.
* :mod:`.analyze` -- parity, the validity gate, paired session bootstrap,
  the report.
* :mod:`.run` -- sequential passes and the ``results/live/`` files.

Entry point: ``python -m webprefetch.live {plan,store,probe,origin,run}``.
The origin, client, objects and timing modules are stdlib only; importing
this package imports none of them.
"""
