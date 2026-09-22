"""Small, dependency-free persistence and provenance helpers."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_job_path(job_dir: Path, value) -> Path:
    """Resolve a path recorded in a job record: relative paths live inside ``job_dir``.

    Records written before 2026-09-20 stored absolute paths; those are returned unchanged so
    historical jobs remain readable.
    """
    path = Path(str(value))
    return path if path.is_absolute() else Path(job_dir) / path


def portable_job_path(job_dir: Path, path) -> str:
    """Store a file inside ``job_dir`` by its relative name so the directory can be moved or copied."""
    path = Path(str(path))
    try:
        return path.resolve().relative_to(Path(job_dir).resolve()).as_posix()
    except (OSError, ValueError):
        return str(path)


def atomic_json(path: Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=1, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
