"""wss_features — frozen, dependency-light geometry program for deployment.

This package is a verbatim, self-contained copy of the geometry functions the
deployment tool needs to turn *STL + centreline atlas* into model inputs:

* ``stl``        binary/ASCII STL reader and writer
* ``atlas``      centreline atlas (RMF frames, tube coordinates, outlet semantics)
* ``cloud``      PCA normals, virtual caps, oriented cloud, inside tests, interior queries
* ``frame``      anatomical coordinate frame (flow divider origin, trunk +Z, left CIA +X)
* ``curvature``  principal curvatures from a local Monge quadric
* ``flowref``    along-tree flow-reference features, Murray / cap-fit flow shares
* ``sampling``   Taubin smoothing, voxel decimation, blue-noise-like surface resampling

Nothing here imports the training packages (``wss_v5``, ``training_wss_min``,
``wss_pinn``), reads CFD data, or keeps module-level mutable state: every input
(including the cap-area split rule) is an explicit argument.  The training-side
implementations remain the reference; ``tests/test_wss_features_equivalence.py``
checks bit-for-bit agreement whenever both are importable.

Changing any file in this package changes :func:`contract_hash`.  The recorded
value in ``CONTRACT.json`` must then be bumped deliberately (see
``tests/test_wss_features_contract.py``) so a model release can pin the exact
feature program it was validated with.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

FEATURE_CONTRACT_VERSION = "wss-features/1.0"
_ROOT = Path(__file__).resolve().parent
_SOURCE_FILES = ("atlas.py", "cloud.py", "curvature.py", "flowref.py", "frame.py", "sampling.py", "stl.py")


def contract_hash() -> str:
    """SHA256 over the feature program source (sorted files, LF-normalised)."""
    digest = hashlib.sha256()
    for name in _SOURCE_FILES:
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update((_ROOT / name).read_bytes().replace(b"\r\n", b"\n"))
        digest.update(b"\0")
    return digest.hexdigest()


def recorded_contract() -> dict:
    """The committed contract record (version + hash) shipped next to the code."""
    path = _ROOT / "CONTRACT.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def contract() -> dict:
    """Runtime identity of the feature program, for run manifests and release checks."""
    recorded = recorded_contract()
    actual = contract_hash()
    return {"version": FEATURE_CONTRACT_VERSION, "source_hash": actual,
            "recorded_hash": recorded.get("source_hash"),
            "matches_record": recorded.get("source_hash") == actual}


__all__ = ["FEATURE_CONTRACT_VERSION", "contract", "contract_hash", "recorded_contract"]
