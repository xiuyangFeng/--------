"""Mesh-only stage-B intermediates cached in the job directory (v0.14, contract: ``pipeline.precompute_geometry_cache``).

Stage B spends most of its CPU time on work that depends only on the uploaded surface and the
centreline, not on the confirmed outlet names or the model release: loading the clean STL, Taubin
smoothing + 0.5 mm resampling, PCA normals and principal curvatures of the resampled cloud, and the
lumen cross-sections of the morphology block.  The job manager runs
:func:`wss_deploy.pipeline.precompute_geometry_cache` while the user is still checking the outlet
names; stage B then finds the results here.

Every entry is keyed by a SHA256 over *exactly the inputs the computation reads* (array bytes,
dtypes and shapes, parameters) plus the source hash of the code that computes it, so a hit is by
construction the value the computation would return (Tier A, bit-identical).  Any mismatch —
different STL, smoothing, sampling seed, atlas, feature program or deployment code — is a miss
and the value is recomputed.  Entries are ``.npz`` files written atomically (temp file +
``os.replace``) and read with ``allow_pickle=False``; an unreadable entry is a miss.

``WSS_DEPLOY_GEOMETRY_CACHE=0`` disables reading and writing.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, Iterable

import numpy as np

CACHE_DIRNAME = "geometry_cache"
CACHE_SCHEMA = "wss-deploy.geometry-cache/v1"
_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()
_SOURCE_HASHES: dict[str, str] = {}


def enabled() -> bool:
    return os.environ.get("WSS_DEPLOY_GEOMETRY_CACHE", "1").strip().lower() not in {"0", "false", "off", "no"}


# ----------------------------------------------------------------------------- keys
def _feed(digest, value) -> None:
    """Feed one key component; arrays contribute dtype, shape and raw bytes."""
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        digest.update(b"A"); digest.update(array.dtype.str.encode()); digest.update(repr(array.shape).encode())
        digest.update(array.tobytes())
    elif isinstance(value, (bytes, bytearray, memoryview)):
        digest.update(b"B"); digest.update(bytes(value))
    elif isinstance(value, str):
        digest.update(b"S"); digest.update(value.encode("utf-8"))
    elif isinstance(value, (list, tuple)):
        digest.update(b"L" + str(len(value)).encode())
        for item in value:
            _feed(digest, item)
    elif isinstance(value, dict):
        digest.update(b"D" + str(len(value)).encode())
        for key in sorted(value, key=str):
            _feed(digest, str(key)); _feed(digest, value[key])
    else:   # numbers, bools, None: repr is exact for Python floats
        digest.update(b"V"); digest.update(repr(value).encode())
    digest.update(b"\0")


def key_of(*parts: Any) -> str:
    """SHA256 hex over the key components (schema version included)."""
    digest = hashlib.sha256()
    _feed(digest, CACHE_SCHEMA)
    for part in parts:
        _feed(digest, part)
    return digest.hexdigest()


def source_hash(*module_names: str) -> str:
    """SHA256 over the source files of the given (already imported or importable) modules.

    Computed once per process and module, as close as possible to the module's import, so the hash
    describes the code this process actually runs.
    """
    import importlib
    digest = hashlib.sha256()
    for name in module_names:
        if name not in _SOURCE_HASHES:
            module = sys.modules.get(name) or importlib.import_module(name)
            path = Path(getattr(module, "__file__", "") or "")
            _SOURCE_HASHES[name] = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "unknown"
        digest.update(name.encode()); digest.update(b"=")
        digest.update(_SOURCE_HASHES[name].encode()); digest.update(b"\n")
    return digest.hexdigest()


def feature_program_hash() -> str:
    """Source hash of the frozen ``wss_features`` program (the same value the release pins)."""
    if "wss_features" not in _SOURCE_HASHES:
        from wss_features import contract_hash
        _SOURCE_HASHES["wss_features"] = contract_hash()
    return _SOURCE_HASHES["wss_features"]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# ----------------------------------------------------------------------------- locking
def job_lock(job_dir: Path) -> threading.Lock:
    """One lock per job directory: the background precompute and stage B never compute the same entry twice."""
    key = str(Path(job_dir).resolve())
    with _LOCKS_GUARD:
        lock = _LOCKS.get(key)
        if lock is None:
            lock = _LOCKS[key] = threading.Lock()
        return lock


# ----------------------------------------------------------------------------- storage
class GeometryCache:
    """``<job_dir>/geometry_cache/<kind>-<key[:40]>.npz`` entries; a disabled cache never hits and never writes."""

    def __init__(self, job_dir: Path | None, *, active: bool | None = None):
        self.job_dir = Path(job_dir).resolve() if job_dir is not None else None
        self.active = bool(enabled() if active is None else active) and self.job_dir is not None
        self.dir = self.job_dir / CACHE_DIRNAME if self.job_dir is not None else None
        self.stats = {"hits": [], "misses": [], "writes": [], "errors": []}
        self.used: set[str] = set()

    def path(self, kind: str, key: str) -> Path:
        return self.dir / f"{kind}-{key[:40]}.npz"

    def load(self, kind: str, key: str) -> dict[str, np.ndarray] | None:
        if not self.active:
            return None
        path = self.path(kind, key)
        if not path.is_file():
            self.stats["misses"].append(kind)
            return None
        try:
            with np.load(path, allow_pickle=False) as data:
                if str(data["__key__"]) != key:
                    raise ValueError("key mismatch")
                arrays = {name: data[name] for name in data.files if name != "__key__"}
        except Exception as exc:  # corrupt / truncated / foreign file: recompute
            self.stats["errors"].append(f"{kind}: {type(exc).__name__}")
            self.stats["misses"].append(kind)
            return None
        self.stats["hits"].append(kind)
        self.used.add(path.name)
        return arrays

    def save(self, kind: str, key: str, arrays: dict[str, np.ndarray], *, compress: bool = False) -> bool:
        """Store one entry atomically.  ``compress`` only pays for integer / JSON payloads (the mesh faces
        halve); float64 geometry is ~95 % incompressible, so it is stored plain to keep the write fast."""
        if not self.active:
            return False
        path = self.path(kind, key)
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(prefix="." + path.stem + ".", suffix=".npz.tmp", dir=self.dir)
            try:
                with os.fdopen(fd, "wb") as stream:
                    (np.savez_compressed if compress else np.savez)(stream, __key__=np.asarray(key), **arrays)
                os.replace(tmp, path)
            finally:
                if os.path.exists(tmp):
                    os.unlink(tmp)
        except Exception as exc:  # a full disk must never fail the job; the value is still returned
            self.stats["errors"].append(f"{kind}: {type(exc).__name__}")
            return False
        self.stats["writes"].append(kind)
        self.used.add(path.name)
        return True

    def prune(self, keep: Iterable[str] | None = None) -> list[str]:
        """Delete entries of this directory that the latest run did not use (stale keys)."""
        if not self.active or not self.dir.is_dir():
            return []
        keep = set(self.used if keep is None else keep)
        removed = []
        for path in self.dir.glob("*.npz"):
            if path.name not in keep:
                with contextlib.suppress(OSError):
                    path.unlink()
                    removed.append(path.name)
        return removed

    def summary(self) -> dict:
        return {"enabled": self.active, "hits": list(self.stats["hits"]), "misses": list(self.stats["misses"]),
                "writes": list(self.stats["writes"]), "errors": list(self.stats["errors"])}


# ----------------------------------------------------------------------------- JSON payloads
def json_array(value: Any) -> np.ndarray:
    """A JSON document as a 0-d unicode array (npz-storable without pickle; floats round-trip exactly)."""
    return np.asarray(json.dumps(value, ensure_ascii=False))


def from_json_array(array: np.ndarray) -> Any:
    return json.loads(str(array))


__all__ = ["CACHE_DIRNAME", "CACHE_SCHEMA", "GeometryCache", "enabled", "feature_program_hash", "file_sha256",
           "from_json_array", "job_lock", "json_array", "key_of", "source_hash"]
