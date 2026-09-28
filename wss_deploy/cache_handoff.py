"""Copy committed geometry entries between owner-validated jobs, without sharing files.

The manager supplies only job IDs, never a path from a job record.  Readers still
validate the complete content key in ``GeometryCache.load``; copying an obsolete
or damaged entry can therefore only cause a normal cache miss.  Source writers
publish entries by atomic replacement, so an open file is a stable snapshot.
"""
from __future__ import annotations

import contextlib
import os
from pathlib import Path
import re
import secrets
import stat
from typing import Callable

from . import geometry_cache as GC

_JOB_ID = re.compile(r"[A-Za-z0-9_-]{1,80}")
_ENTRY = re.compile(r"[a-z][a-z0-9_-]*-[0-9a-f]{40}\.npz")


def copy_committed(root: Path, source_id: str, target_id: str, *,
                   valid: Callable[[], bool] = lambda: True) -> int:
    """Best-effort, atomic, no-overwrite transfer; all I/O is outside the manager lock.

    Directory descriptors and ``O_NOFOLLOW`` reject symlinks in every job-relative
    component.  Hard-linked source files, temporary files, and nested directories
    are excluded.  The destination receives its own 0600 copy, so deleting or
    rewriting either job cannot affect the other.  ``valid`` rechecks ownership
    and cancellation between files/chunks and before publication.
    """
    if not GC.enabled() or not all(isinstance(value, str) and _JOB_ID.fullmatch(value)
                                   for value in (source_id, target_id)) or source_id == target_id:
        return 0
    if not valid():
        return 0
    copied = 0
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    try:
        with contextlib.ExitStack() as stack:
            def directory(name, parent=None):
                fd = os.open(name, directory_flags, dir_fd=parent)
                stack.callback(os.close, fd)
                return fd

            root_fd = directory(root)
            source_job = directory(source_id, root_fd)
            target_job = directory(target_id, root_fd)
            source = directory(GC.CACHE_DIRNAME, source_job)
            try:
                os.mkdir(GC.CACHE_DIRNAME, mode=0o700, dir_fd=target_job)
            except FileExistsError:
                pass
            target = directory(GC.CACHE_DIRNAME, target_job)
            for name in os.listdir(source):
                if not _ENTRY.fullmatch(name):
                    continue
                if not valid():
                    break
                try:
                    # Never replace even a damaged local entry: its normal cache
                    # read/recompute path remains authoritative for this job.
                    os.stat(name, dir_fd=target, follow_symlinks=False)
                    continue
                except FileNotFoundError:
                    pass
                temporary = ".handoff-" + secrets.token_hex(12) + ".tmp"
                try:
                    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=source)
                    with os.fdopen(fd, "rb") as incoming:
                        before = os.fstat(incoming.fileno())
                        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                            continue
                        out_fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                                         0o600, dir_fd=target)
                        with os.fdopen(out_fd, "wb") as outgoing:
                            while chunk := incoming.read(1024 * 1024):
                                if not valid():
                                    return copied
                                outgoing.write(chunk)
                        after = os.fstat(incoming.fileno())
                        if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != \
                                (after.st_size, after.st_mtime_ns, after.st_ctime_ns) or not valid():
                            continue
                    # Link the completed PRIVATE temporary copy, not the source.
                    # Unlike replace(), link() atomically fails if the target exists.
                    os.link(temporary, name, src_dir_fd=target, dst_dir_fd=target, follow_symlinks=False)
                    copied += 1
                except OSError:
                    # A concurrent prune/delete, disk error, or unsafe entry only
                    # loses this optimization; stage B computes its own value.
                    continue
                finally:
                    with contextlib.suppress(OSError):
                        os.unlink(temporary, dir_fd=target)
    except OSError:
        pass
    return copied
