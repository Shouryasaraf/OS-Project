"""Real Linux prefetch testbed: replay a block trace on a real device.

The simulator (``adaptive_prefetch.simulator``) decides; this package does the
I/O and measures it. Nothing here re-implements a policy for the plan-driven
backends: the simulator's own ``replay_stream`` runs once with a recording
cache (:mod:`.decisions`), and every demand hit/miss and every issued
prefetch becomes the plan a backend executes.

Two backends, two different validity statements:

* **Backend B** (:mod:`.backend_odirect`, ``odirect``) keeps a user-space LRU
  with exactly ``LRUCache``'s rules, reads misses with ``O_DIRECT`` and issues
  prefetches asynchronously on a thread pool. It asserts on every request that
  its hit/miss sequence equals the plan's, so its cache behaviour is the
  simulator's by construction and the measurement isolates *timing*: late
  prefetches, device contention, think time.
* **Backend A** (:mod:`.backend_pagecache`, ``pagecache``) uses the kernel page
  cache inside a memory-limited cgroup, with prefetches as
  ``POSIX_FADV_WILLNEED``. Its hit ratio is a *measured divergence* from the
  simulator (MGLRU, 4 KiB pages), reported as such and never matched.
  ``kernel_ra`` / ``kernel_seq`` measure the kernel's own read-ahead, and
  :mod:`.online` runs a policy inline with no precomputed plan.

Linux-only and stdlib-only. Every entry point checks :func:`supported` and
the unit tests skip cleanly elsewhere. The command line is
``python -m adaptive_prefetch.realio {mkfile,calibrate,run,grid,report}``.

What it is not: a kernel prefetcher. ``linux_ra`` here is the simulator's
emulation of the mm/readahead.c *policy*, driven against the user-space
shadow LRU; the kernel's own read-ahead is what ``kernel_ra`` measures.
"""

from __future__ import annotations

import os
import sys

#: Simulator block size, bytes. Trace LBAs are in these units.
BLOCK = 512
#: Kernel page size; the page cache works in these.
PAGE = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096
#: Cache capacity used for E3, in blocks (32 MiB). A 1 MiB cache cannot be
#: emulated in Backend A: the prototype measured the cgroup's page-cache
#: budget at MemoryMax minus ~7 MiB (plan section 2.6), and the offset this
#: package uses is itself 1 MiB (``backend_pagecache.BUDGET_OFFSET``).
DEFAULT_CAPACITY = 65_536


def supported() -> tuple[bool, str]:
    """Whether this platform can run the testbed, and why not if it cannot."""
    if not sys.platform.startswith("linux"):
        return False, f"the real-device testbed is Linux-only (platform {sys.platform})"
    missing = [name for name in ("O_DIRECT", "preadv", "posix_fadvise", "RWF_NOWAIT")
               if not hasattr(os, name)]
    if missing:
        return False, f"this Python lacks {', '.join(missing)}"
    return True, ""
