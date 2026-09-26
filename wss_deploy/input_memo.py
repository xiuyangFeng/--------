"""Build an ensemble's model inputs once per case (v0.14 speed-up, companion of ``knn_memo``).

``training_wss_min.evaluate.predict_case_norm`` rebuilds, for every member of an ensemble, the
support indices, the standardised support / query features and the 16-neighbour query patches
(``build_query_patch``: a single-threaded KD-tree query per query chunk).  All members of a
release share the support seed, the input feature list and the frozen feature statistics, so on
one case those calls are repeated with identical arguments — about 0.3 s per member on LV_GUO_YOU.

The three builders are pure functions of their arguments (the case dict, index arrays, the
feature list and statistics, plain options).  Inside :func:`shared_inputs` a call whose arguments
are all equal to an earlier call's — the case normally by identity (the same dict for every member),
otherwise by deep equality of its content; arrays with ``np.array_equal`` (same dtype and shape);
statistics and configs by deep equality — gets a copy of the earlier result, so nothing is
approximated and no caller can modify a shared array.  Outside the scope, and in other threads, the
wrapped functions pass straight through.

Only the deployment process is affected: ``training_wss_min`` is not edited; its module-level
names are wrapped once (the same accepted pattern as ``knn_memo``).  ``WSS_DEPLOY_PATCH_MEMO=0``
switches the reuse off; ``WSS_DEPLOY_PATCH_MEMO_MB`` caps the memory kept per scope (default 3072).
"""
from __future__ import annotations

import contextlib
import inspect
import os
import threading

import numpy as np

_TARGET_MODULE = "training_wss_min.dataset"
_TARGET_NAMES = ("sample_support_indices", "build_features", "build_query_patch")
_local = threading.local()
_install_lock = threading.Lock()


def enabled() -> bool:
    return os.environ.get("WSS_DEPLOY_PATCH_MEMO", "1").strip().lower() not in {"0", "false", "off", "no"}


def _budget_bytes() -> int:
    try:
        return max(0, int(float(os.environ.get("WSS_DEPLOY_PATCH_MEMO_MB", "3072")) * 1024 * 1024))
    except ValueError:
        return 3072 * 1024 * 1024


def _same(a, b) -> bool:
    """Exact equality for the argument types the builders take (identity first, then content)."""
    if a is b:
        return True
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return (isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and a.dtype == b.dtype
                and a.shape == b.shape and bool(np.array_equal(a, b)))
    if isinstance(a, dict) or isinstance(b, dict):
        return (isinstance(a, dict) and isinstance(b, dict) and a.keys() == b.keys()
                and all(_same(a[k], b[k]) for k in a))
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        return (isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)) and len(a) == len(b)
                and all(_same(x, y) for x, y in zip(a, b)))
    if type(a) is not type(b) and not (isinstance(a, (int, float, np.integer, np.floating))
                                       and isinstance(b, (int, float, np.integer, np.floating))):
        return False
    try:
        return bool(a == b)
    except Exception:
        return False


def _copy(value):
    """A private copy of a builder result (ndarray or dict of ndarrays); other values as they are."""
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {k: _copy(v) for k, v in value.items()}
    return value


def _nbytes(value) -> int:
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if isinstance(value, dict):
        return sum(_nbytes(v) for v in value.values())
    return 0


class _MemoBuilder:
    """Drop-in replacement for one dataset builder; memoises only inside :func:`shared_inputs`."""

    def __init__(self, name: str, original):
        self.name = name
        self.original = original
        self.signature = inspect.signature(original)
        self.__wrapped__ = original
        self.__name__ = getattr(original, "__name__", name)
        self.__doc__ = getattr(original, "__doc__", None)

    def __call__(self, *args, **kwargs):
        store = getattr(_local, "store", None)
        if store is None:
            return self.original(*args, **kwargs)
        try:
            bound = self.signature.bind(*args, **kwargs)
            bound.apply_defaults()
            items = tuple(bound.arguments.items())
        except TypeError:
            return self.original(*args, **kwargs)
        stats = _local.stats
        stats["calls"] += 1
        for name, key, value in store["entries"]:
            if name == self.name and len(key) == len(items) and all(
                    k1 == k2 and _same(v1, v2) for (k1, v1), (k2, v2) in zip(key, items)):
                stats["reused"] += 1
                return _copy(value)
        value = self.original(*args, **kwargs)
        size = _nbytes(value)
        if store["bytes"] + size <= store["budget"]:
            # Keep a private copy: the caller (and, on the CPU, the model through torch.from_numpy)
            # may hold the returned arrays; later members must see the pristine result.
            store["entries"].append((self.name, items, _copy(value)))
            store["bytes"] += size
        return value


def install() -> bool:
    """Wrap the dataset builders once (idempotent); returns False when the module is unavailable."""
    import importlib
    with _install_lock:
        try:
            module = importlib.import_module(_TARGET_MODULE)
        except Exception:
            return False
        for name in _TARGET_NAMES:
            current = getattr(module, name, None)
            if current is not None and not isinstance(current, _MemoBuilder):
                setattr(module, name, _MemoBuilder(name, current))
    return True


@contextlib.contextmanager
def shared_inputs():
    """Scope (one ensemble prediction on one case) in which identical input builds are done once.

    Yields ``{"calls", "reused"}`` for the calling thread; other threads are unaffected.  The memo is
    dropped when the scope ends, so nothing outlives the case.
    """
    stats = {"calls": 0, "reused": 0}
    if not enabled() or not install():
        yield stats
        return
    previous = (getattr(_local, "store", None), getattr(_local, "stats", None))
    _local.store, _local.stats = {"entries": [], "bytes": 0, "budget": _budget_bytes()}, stats
    try:
        yield stats
    finally:
        _local.store, _local.stats = previous


__all__ = ["enabled", "install", "shared_inputs"]
