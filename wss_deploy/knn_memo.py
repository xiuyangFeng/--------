"""Share the kNN graph between the members of an ensemble (v0.12.2 speed-up).

Every member of a release predicts on the same case, and the local-refinement block of the
network builds the same wall neighbourhood each time: ``torch_geometric.nn.knn`` on identical
coordinates with identical ``k``.  That call is about two thirds of a member's inference time.

The kNN kernel is deterministic (no atomics: every query keeps its own sorted candidate list), so
a result can be reused whenever *every* input is exactly equal — tensors are compared with
``torch.equal``, the remaining arguments with ``==``.  Nothing is approximated: a reused graph is
the graph the call would have produced.  Measured on LV_GUO_YOU (X5D, 5 members): 14 of 15 calls
reused, ensemble inference 13 s → 5.6 s on a busy GPU; predictions differ from an unmemoised run
by at most 5e-6 Pa, the same size as two unmemoised runs differ from each other (GPU scatter
nondeterminism), p99 equal to six decimals.

Only the deployment process is affected: the training modules are not edited; their module-level
``knn`` name is wrapped once, and the wrapper passes straight through unless the calling thread is
inside :func:`shared_knn`.  ``WSS_DEPLOY_KNN_MEMO=0`` switches the reuse off.
"""
from __future__ import annotations

import contextlib
import inspect
import os
import threading

# v0.14: ``torch_geometric``'s ``knn_interpolate`` calls its own module-level ``knn`` (3 of the 25 calls of an
# X5D ensemble bypassed the training modules' wrapped name); wrapping that name reuses those graphs too.
_TARGET_MODULES = ("training_wss_min.local_refinement", "training_wss_min.baseline_models",
                   "torch_geometric.nn.unpool.knn_interpolate")
_MAX_ENTRIES = 64   # v0.14: knn_interpolate adds one graph per query chunk (large cases have ~20 chunks)
_local = threading.local()
_install_lock = threading.Lock()


def enabled() -> bool:
    return os.environ.get("WSS_DEPLOY_KNN_MEMO", "1").strip().lower() not in {"0", "false", "off", "no"}


def _same(a, b) -> bool:
    import torch
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and a.dtype == b.dtype
                and a.device == b.device and a.shape == b.shape and bool(torch.equal(a, b)))
    return a == b


class _MemoKnn:
    """Drop-in replacement for a module's ``knn``; memoises only inside :func:`shared_knn`."""

    def __init__(self, original):
        self.original = original
        self.signature = inspect.signature(original)
        self.__wrapped__ = original
        self.__name__ = getattr(original, "__name__", "knn")
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
        for key, value in store:
            if len(key) == len(items) and all(k1 == k2 and _same(v1, v2) for (k1, v1), (k2, v2) in zip(key, items)):
                stats["reused"] += 1
                return value.clone() if hasattr(value, "clone") else value
        value = self.original(*args, **kwargs)
        if len(store) < _MAX_ENTRIES:
            store.append((items, value))
        return value


def install() -> list[str]:
    """Wrap ``knn`` in the training modules that are importable (idempotent); returns the wrapped module names."""
    import importlib
    wrapped = []
    with _install_lock:
        for name in _TARGET_MODULES:
            try:
                module = importlib.import_module(name)
            except Exception:
                continue
            current = getattr(module, "knn", None)
            if current is None:
                continue
            if not isinstance(current, _MemoKnn):
                setattr(module, "knn", _MemoKnn(current))
            wrapped.append(name)
    return wrapped


@contextlib.contextmanager
def shared_knn():
    """Scope (one ensemble prediction on one case) in which identical kNN calls are computed once.

    Yields ``{"calls", "reused"}`` for the calling thread; other threads are unaffected.
    """
    stats = {"calls": 0, "reused": 0}
    if not enabled():
        yield stats
        return
    install()
    previous = (getattr(_local, "store", None), getattr(_local, "stats", None))
    _local.store, _local.stats = [], stats
    try:
        yield stats
    finally:
        _local.store, _local.stats = previous


__all__ = ["enabled", "install", "shared_knn"]
