"""ctypes wrappers for the few Linux calls the stdlib does not expose.

``cachestat(2)`` (Linux >= 6.5, syscall 451 in the unified table), a
``mincore(2)`` fallback for older kernels, and ``prctl(PR_SET_TIMERSLACK)``.
No root is needed for any of them. Everything degrades: a missing libc or
syscall raises ``OSError`` at call time, never at import time, so importing
this module is safe on any platform.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import errno
import mmap
import os

from . import PAGE

SYS_CACHESTAT = 451
PR_SET_TIMERSLACK = 29

_libc = None


def libc():
    global _libc
    if _libc is None:
        name = ctypes.util.find_library("c")
        if name is None:
            raise OSError(errno.ENOSYS, "libc not found")
        lib = ctypes.CDLL(name, use_errno=True)
        lib.syscall.restype = ctypes.c_long
        lib.mmap.restype = ctypes.c_void_p
        lib.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
                             ctypes.c_int, ctypes.c_int, ctypes.c_long]
        lib.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        lib.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p]
        lib.prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong,
                              ctypes.c_ulong, ctypes.c_ulong]
        _libc = lib
    return _libc


class _Range(ctypes.Structure):
    _fields_ = [("off", ctypes.c_uint64), ("len", ctypes.c_uint64)]


class _Stat(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint64) for name in
                ("nr_cache", "nr_dirty", "nr_writeback", "nr_evicted",
                 "nr_recently_evicted")]


def cachestat(fd: int, offset: int, length: int) -> int:
    """Pages of ``[offset, offset+length)`` in the page cache (0 = to EOF).

    Counts pages that are present, including ones still under read I/O, so it
    answers "is it resident or in flight", not "can it be read now". The
    ``RWF_NOWAIT`` probe in the page-cache backend answers the second.
    """
    rng, stat = _Range(offset, length), _Stat()
    if libc().syscall(SYS_CACHESTAT, fd, ctypes.byref(rng), ctypes.byref(stat), 0) != 0:
        err = ctypes.get_errno()
        raise OSError(err, f"cachestat: {os.strerror(err)}")
    return int(stat.nr_cache)


def mincore_count(fd: int, offset: int, length: int) -> int:
    """Resident pages of a page-aligned range via ``mmap`` + ``mincore``."""
    lib = libc()
    addr = lib.mmap(None, length, mmap.PROT_READ, mmap.MAP_SHARED, fd, offset)
    if addr in (None, ctypes.c_void_p(-1).value):
        err = ctypes.get_errno()
        raise OSError(err, f"mmap: {os.strerror(err)}")
    try:
        pages = (length + PAGE - 1) // PAGE
        vec = ctypes.create_string_buffer(pages)
        if lib.mincore(addr, length, vec) != 0:
            err = ctypes.get_errno()
            raise OSError(err, f"mincore: {os.strerror(err)}")
        return sum(b & 1 for b in vec.raw)
    finally:
        lib.munmap(addr, length)


def residency_counter(fd: int):
    """The best available page-residency counter for ``fd``, and its name.

    ``cachestat`` when the kernel has it and allows it on this file (it
    requires write access or ownership), else ``mincore``. Both have the
    signature ``(fd, offset, length) -> pages``; callers pass page-aligned
    ranges.
    """
    try:
        cachestat(fd, 0, PAGE)
        return cachestat, "cachestat"
    except OSError:
        return mincore_count, "mincore"


def set_timer_slack(nanoseconds: int = 1000) -> bool:
    """Tighten this thread's timer slack (default 50 us) so sleeps overshoot less."""
    try:
        return libc().prctl(PR_SET_TIMERSLACK, nanoseconds, 0, 0, 0) == 0
    except (OSError, AttributeError):
        return False
