"""Exact, pickle-free cache for confirmed PF6/VF6 geometry.

Only the array/dictionary result of ``build_volume_case`` is persisted.  VTK
surfaces and query objects are rebuilt by their existing consumers.  Stage B
reads and fills it; since v0.16.1 ``volume_pipeline.precompute_volume_case``
also fills it on the proposed outlet mapping while the outlets are being
confirmed (a stage B on the same confirmed mapping then finds it).
"""
from __future__ import annotations

from importlib.metadata import version

import numpy as np

from . import geometry_cache as GC
from . import volume_geometry as VG

KIND = "volume_case"
SCHEMA = "wss-deploy.volume-case-cache/v1"


def _pack(value):
    """Encode supported values without coercing dict keys or NumPy scalars."""
    arrays = {}

    def visit(item):
        if isinstance(item, (np.ndarray, np.generic)):
            array = np.asarray(item)
            if array.dtype.hasobject or array.dtype.fields is not None:
                raise TypeError("object/structured arrays are not volume cache values")
            name = f"array_{len(arrays)}"
            arrays[name] = array
            return ["scalar" if isinstance(item, np.generic) else "array", name]
        if type(item) is dict:
            return ["dict", [[visit(k), visit(v)] for k, v in item.items()]]
        if type(item) in (list, tuple):
            return ["list" if type(item) is list else "tuple", [visit(v) for v in item]]
        if item is None or type(item) in (str, bool, int, float):
            return ["value", item]
        raise TypeError(f"unsupported volume cache value: {type(item).__name__}")

    tree = visit(value)
    arrays["__tree__"] = GC.json_array({"schema": SCHEMA, "tree": tree})
    return arrays


def _unpack(arrays):
    document = GC.from_json_array(arrays["__tree__"])
    if document["schema"] != SCHEMA:
        raise ValueError("volume cache schema mismatch")
    used = {"__tree__"}

    def visit(node):
        kind, value = node
        if kind in {"array", "scalar"}:
            array = arrays[value]
            if array.dtype.hasobject or array.dtype.fields is not None:
                raise ValueError("unsupported cached dtype")
            used.add(value)
            if kind == "scalar":
                if array.shape != ():
                    raise ValueError("cached scalar must be zero-dimensional")
                return array[()]
            return array
        if kind == "dict":
            result = {visit(k): visit(v) for k, v in value}
            if len(result) != len(value):
                raise ValueError("duplicate cached dict keys")
            return result
        if kind in {"list", "tuple"}:
            result = [visit(v) for v in value]
            return result if kind == "list" else tuple(result)
        if kind == "value" and (value is None or type(value) in (str, bool, int, float)):
            return value
        raise ValueError("invalid volume cache node")

    result = visit(document["tree"])
    if used != set(arrays):
        raise ValueError("unexpected volume cache arrays")
    if type(result) is not tuple or len(result) != 2 or any(type(v) is not dict for v in result):
        raise ValueError("volume cache must contain (case, aux) dictionaries")
    return result


def _atlas_inputs(atlas):
    """Include the mapped atlas and the KD-tree's actual data/partition.

    The tree is never stored.  Its partition is part of the key because it
    can determine tie ordering for equidistant centreline samples.
    """
    def partition(node):
        if node is None:
            return None
        return (node.split_dim, node.split, node.start_idx, node.end_idx,
                partition(node.lesser), partition(node.greater))

    tree = atlas.tree
    return {**{name: getattr(atlas, name) for name in (
        "table", "columns", "segments", "semantic_of_segment", "frame_n", "frame_b",
        "tree_rows", "provenance")},
        "tree": {"data": tree.data, "indices": tree.indices, "leafsize": tree.leafsize,
                 "boxsize": tree.boxsize, "mins": tree.mins, "maxes": tree.maxes,
                 "partition": partition(tree.tree)}}


def _canonical_mapping(mapping):
    """The mapping in key order: ``CL.apply_mapping`` walks the atlas segments, so the mapped atlas (and
    the geometry) does not depend on the order of the mapping, and neither does the key — the proposal
    lists outlets in tree order, a confirmation in the order the page sent them."""
    if not isinstance(mapping, dict):
        return mapping
    return {str(key): mapping[key] for key in sorted(mapping, key=str)}


def _cache_key(wall, vertices, faces, atlas, input_features, *, mapping, target,
               case_name, n_internal, seed):
    # Packing inputs rejects unknown objects instead of hashing their repr or
    # silently dropping metadata that may affect geometry in future versions.
    inputs = _pack((wall, vertices, faces, _atlas_inputs(atlas), input_features,
                    _canonical_mapping(mapping), target, case_name, n_internal, seed))
    return GC.key_of(SCHEMA, inputs,
                     GC.source_hash(__name__, "wss_deploy.volume_geometry", "wss_deploy.streamlines", "wss_deploy.geometry_cache",
                                    "wss_features.atlas", "wss_features.cloud", "wss_features.flowref",
                                    "wss_features.frame"),
                     GC.feature_program_hash(),
                     {name: version(name) for name in ("numpy", "scipy", "pyvista", "vtk")})


def volume_case_key(wall, vertices, faces, atlas, input_features, *, mapping=None,
                    target="pressure_velocity", case_name="case", n_internal=20000, seed=0):
    """The exact-input key of a ``build_volume_case`` call (raises on inputs it cannot pack)."""
    return _cache_key(wall, vertices, faces, atlas, input_features, mapping=mapping, target=target,
                      case_name=case_name, n_internal=n_internal, seed=seed)


def load_volume_case(cache, key):
    """The cached ``(case, aux)`` for ``key``, or None (missing, or a damaged entry: then counted a miss)."""
    hits_before = len(cache.stats["hits"])
    payload = cache.load(KIND, key)
    if payload is None:
        return None
    try:
        digest = str(payload.pop("__payload_hash__"))
        if GC.key_of(payload) != digest:
            raise ValueError("volume cache payload checksum mismatch")
        return _unpack(payload)
    except Exception as exc:
        # GeometryCache validates NPZ storage; the typed payload is our
        # responsibility.  A semantically invalid archive is a miss too.
        del cache.stats["hits"][hits_before:]
        cache.used.discard(cache.path(KIND, key).name)
        cache.stats["errors"].append(f"{KIND} payload: {type(exc).__name__}")
        cache.stats["misses"].append(KIND)
        return None


def store_volume_case(cache, key, result) -> None:
    """Persist ``result`` under ``key``; a payload that cannot be stored is only recorded in the stats."""
    try:
        payload = _pack(result)
        _unpack(payload)  # reject unsupported root types before writing
        payload["__payload_hash__"] = np.asarray(GC.key_of(payload))
        cache.save(KIND, key, payload)
    except Exception as exc:
        cache.stats["errors"].append(f"{KIND} payload: {type(exc).__name__}")


def build_volume_case_cached(wall, vertices, faces, atlas, input_features, *, cache=None,
                             mapping=None, target="pressure_velocity", case_name="case",
                             n_internal=20000, seed=0):
    """Reuse exact geometry or recompute on any unavailable/invalid cache entry.

    The caller holds ``GC.job_lock`` just as for the other geometry stages.
    Cache failures cannot turn a valid geometry computation into a failed job.
    """
    kwargs = dict(target=target, case_name=case_name, n_internal=n_internal, seed=seed)
    if cache is None or not cache.active:
        return VG.build_volume_case(wall, vertices, faces, atlas, input_features, **kwargs)
    try:
        key = _cache_key(wall, vertices, faces, atlas, input_features, mapping=mapping, **kwargs)
    except Exception as exc:
        cache.stats["errors"].append(f"{KIND} key: {type(exc).__name__}")
        cache.stats["misses"].append(KIND)
        return VG.build_volume_case(wall, vertices, faces, atlas, input_features, **kwargs)
    hit = load_volume_case(cache, key)
    if hit is not None:
        return hit
    result = VG.build_volume_case(wall, vertices, faces, atlas, input_features, **kwargs)
    store_volume_case(cache, key, result)
    return result
