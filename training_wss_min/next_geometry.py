"""Optional geometry metadata for the direct-WSS matrix.

Every array comes from the frozen view or its explicitly recorded source HDF5.
No CFD field, pressure, velocity, WSS, flux, or RCR value is read here.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


_VIEW_REPORTS = {}


def source_h5_for_bundle(bundle_path):
    """Resolve provenance from the view manifest rather than a newer atlas."""
    bundle_path = Path(bundle_path).resolve()
    view_root = bundle_path.parents[3]
    key = str(view_root)
    if key not in _VIEW_REPORTS:
        manifest = json.loads((view_root / "view_manifest.json").read_text())
        _VIEW_REPORTS[key] = {r["canonical_id"]: r for r in manifest["reports"]}
    unit_id = bundle_path.parent.relative_to(view_root).as_posix()
    return Path(_VIEW_REPORTS[key][unit_id]["source"]["case_h5"])


def _unit_vectors(value, name):
    value = np.asarray(value, dtype=np.float64)
    norm = np.linalg.norm(value, axis=1, keepdims=True)
    if not np.isfinite(value).all() or np.any(norm < 1e-8):
        raise ValueError(f"invalid {name}: nonfinite or zero direction")
    return value / norm


def case_geometry(case):
    """Return [N,8]: aligned normal, aligned tangent, radius_mm, scale_mm.

    Map by Fluent node identity, including the six invalid wall rows removed by
    the frozen view. This cannot silently map a cropped view by row position.
    """
    if "local_geometry" in case:
        return case["local_geometry"]
    import h5py

    bundle_path = Path(case["bundle_path"])
    source = source_h5_for_bundle(bundle_path)
    with np.load(bundle_path, allow_pickle=False) as z, h5py.File(source, "r") as h:
        ids = z["wall_node_id_cas"].astype(np.int64)
        source_ids = h["wall_static/node_id_cas"][:].astype(np.int64)
        order = np.argsort(source_ids)
        at = np.searchsorted(source_ids[order], ids)
        if (len(np.unique(source_ids)) != len(source_ids)
                or np.any(at >= len(order))
                or not np.array_equal(source_ids[order][at], ids)):
            raise ValueError(f"wall node identity mismatch: {bundle_path}")
        rows = order[at]
        raw = h["wall_static/xyz_mm"][:][rows]
        if not np.allclose(raw, z["wall_coords_raw"], atol=1e-4, rtol=1e-7):
            raise ValueError(f"wall coordinates differ from frozen source: {bundle_path}")
        atlas = h["geometry/atlas_table"][:]
        cols = json.loads(h["geometry"].attrs["atlas_columns"])
        atlas_rows = h["wall_static/atlas_row"][:][rows].astype(np.int64)
        if np.any(atlas_rows < 0) or np.any(atlas_rows >= len(atlas)):
            raise ValueError(f"invalid frozen atlas projection: {bundle_path}")
        tangent = atlas[atlas_rows][:, [cols.index(f"tangent_{axis}") for axis in "xyz"]]
        rotation = z["transform_rotation"].astype(np.float64)
        tangent = _unit_vectors(tangent @ rotation.T, "atlas tangent")
        normal = _unit_vectors(z["wall_normal_pca_aligned"], "wall normal")
        radius = z["wall_local_radius"].astype(np.float64)
        scale = float(z["coord_scale"])
        if np.any(radius <= 0) or not np.isfinite(radius).all() or not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"invalid physical geometry scale: {bundle_path}")
        if len(ids) != len(case["pos"]) or not np.allclose(case["pos"], z["wall_coords_norm"], atol=1e-7):
            raise ValueError("local geometry requires the original full wall case")
        case["local_geometry"] = np.column_stack((normal, tangent, radius, np.full(len(ids), scale))).astype(np.float32)
        case["local_geometry_source"] = str(source)
    return case["local_geometry"]


def case_section(case):
    """Return [N,2] = (segment_id, s_local_mm) from the frozen view: atlas metadata only, no labels."""
    if "section" in case:
        return case["section"]
    with np.load(case["bundle_path"], allow_pickle=False) as z:
        if "wall_segment_id" not in z.files or "wall_s_local_mm" not in z.files:
            raise KeyError(f"section tokens require wall_segment_id/wall_s_local_mm: {case['bundle_path']}")
        segment = z["wall_segment_id"].astype(np.int64)
        s_local = z["wall_s_local_mm"].astype(np.float64)
    if len(segment) != len(case["pos"]) or "support_pool" in case or "query_pool" in case:
        raise ValueError("section tokens are defined for the original full-wall WSS case only")
    if segment.min() < 0 or segment.max() > 6 or not np.isfinite(s_local).all() or s_local.min() < -1e-3:
        raise ValueError(f"invalid segment/s_local metadata: {case['bundle_path']}")
    case["section"] = np.column_stack((segment, np.clip(s_local, 0.0, None))).astype(np.float32)
    return case["section"]


def full_wall_tree(case):
    """CPU tree retained on the original complete wall, independent of support."""
    if "_full_wall_tree" not in case:
        if "support_pool" in case or "query_pool" in case:
            raise ValueError("query patches are defined for full-wall WSS cases only")
        case["_full_wall_tree"] = cKDTree(np.asarray(case["pos"], dtype=np.float64))
    return case["_full_wall_tree"]


def build_query_patch(case, query_idx, input_features, feat_stats, nsample=16,
                      frame_index=None, rotation=None, radius_mm=0.0, radius_k=0, offset_norm="mm"):
    """Geometry-only full-wall patches; equal results for train/eval chunks.

    Features are the same standardized point features used by the support.
    Relative offsets are in millimetres. K includes the query point itself.

    2026-09-16 dual scale (all defaults keep the historical output bit-identical):
    ``offset_norm='spacing'`` divides the K offsets by the query's median neighbour distance
    (dimensionless); ``radius_mm>0, radius_k>0`` adds a second set of up to ``radius_k`` points
    inside a fixed millimetre ball (coverage by even ranks along sorted distance; short balls are
    padded with the query itself and masked out) as ``radius_features`` / ``radius_relative_mm``
    / ``radius_mask``.  Geometry only, no labels.
    """
    from .dataset import build_features

    if int(nsample) < 1 or len(case["pos"]) < int(nsample):
        raise ValueError("query patch requires 1 <= nsample <= full wall size")
    query_idx = np.asarray(query_idx, dtype=np.int64)
    _, indices = full_wall_tree(case).query(case["pos"][query_idx], k=int(nsample), workers=1)
    indices = np.asarray(indices, dtype=np.int64).reshape(len(query_idx), int(nsample))
    positions = case["pos"][indices]
    offsets = (positions - case["pos"][query_idx, None]) * float(case["coord_scale_scalar"])
    if rotation is None:
        # One array per feature schema and frame; normal operation has one frame.
        cache_key = (tuple(input_features), json.dumps(feat_stats, sort_keys=True), frame_index)
        cache = case.setdefault("_full_wall_features", {})
        if cache_key not in cache:
            cache[cache_key] = build_features(case, np.arange(len(case["pos"])), input_features,
                                             feat_stats, frame_index=frame_index)
        features = cache[cache_key][indices]
    else:
        # Rotation augmentation must preserve vector metadata and spatial inputs.
        positions = positions @ rotation
        offsets = offsets @ rotation
        features = build_features(case, indices.ravel(), input_features, feat_stats,
                                  pos_override=positions.reshape(-1, 3), frame_index=frame_index)
        features = features.reshape(len(query_idx), int(nsample), -1)
    result = {"features": np.ascontiguousarray(features, dtype=np.float32),
              "relative_mm": np.ascontiguousarray(offsets, dtype=np.float32)}
    if str(offset_norm) == "spacing":
        dist = np.linalg.norm(offsets, axis=-1)
        with np.errstate(all="ignore"):
            spacing = np.nanmedian(np.where(dist > 0, dist, np.nan), axis=1)
        spacing = np.maximum(np.nan_to_num(spacing, nan=1.0), 1e-6)
        result["relative_mm"] = np.ascontiguousarray(offsets / spacing[:, None, None], dtype=np.float32)
    elif str(offset_norm) != "mm":
        raise ValueError("offset_norm must be 'mm' or 'spacing'")
    if float(radius_mm) > 0 and int(radius_k) > 0:
        scale = float(case["coord_scale_scalar"])
        k2 = int(radius_k)
        pos = case["pos"]
        balls = full_wall_tree(case).query_ball_point(pos[query_idx], r=float(radius_mm) / scale, workers=1)
        sel = np.empty((len(query_idx), k2), dtype=np.int64)
        mask = np.zeros((len(query_idx), k2), dtype=np.float32)
        for qi, (q, members) in enumerate(zip(query_idx, balls)):
            members = np.asarray(members, dtype=np.int64)
            if len(members) == 0:
                members = np.array([q], dtype=np.int64)
            order = members[np.argsort(np.linalg.norm(pos[members] - pos[q], axis=1), kind="stable")]
            if len(order) >= k2:
                sel[qi] = order[np.rint(np.linspace(0, len(order) - 1, k2)).astype(np.int64)]
                mask[qi] = 1.0
            else:
                sel[qi, :len(order)] = order
                sel[qi, len(order):] = q
                mask[qi, :len(order)] = 1.0
        positions_r = pos[sel]
        offsets_r = (positions_r - pos[query_idx, None]) * scale
        if rotation is None:
            features_r = cache[cache_key][sel]
        else:
            positions_r = positions_r @ rotation
            offsets_r = offsets_r @ rotation
            features_r = build_features(case, sel.ravel(), input_features, feat_stats,
                                        pos_override=positions_r.reshape(-1, 3), frame_index=frame_index)
            features_r = features_r.reshape(len(query_idx), k2, -1)
        result["radius_features"] = np.ascontiguousarray(features_r, dtype=np.float32)
        result["radius_relative_mm"] = np.ascontiguousarray(offsets_r, dtype=np.float32)
        result["radius_mask"] = mask
    elif float(radius_mm) > 0 or int(radius_k) > 0:
        raise ValueError("dual-scale patch needs both radius_mm > 0 and radius_k > 0")
    return result


def _voxel_groups(case, budget):
    """Cache isotropic spatial cells with at most `budget` occupied cells."""
    key = f"_uniform_voxel_groups_{budget}"
    if key in case:
        return case[key]
    pos = np.asarray(case["pos"], dtype=np.float64)
    pos = pos - pos.min(axis=0)
    extent = float(np.ptp(pos, axis=0).max())
    if extent < 1e-12:
        case[key] = [np.arange(len(pos))]
        return case[key]
    lo, hi = extent / max(len(pos), 2), extent * 1.001
    # Search once per case/budget, never once per epoch.
    best = np.zeros(len(pos), dtype=np.int64)
    for _ in range(16):
        width = (lo + hi) / 2
        cells = np.floor(pos / width).astype(np.int32)
        _, codes = np.unique(cells, axis=0, return_inverse=True)
        count = int(codes.max()) + 1
        if count > budget:
            lo = width
        else:
            hi, best = width, codes
    order = np.argsort(best, kind="stable")
    splits = np.flatnonzero(np.diff(best[order])) + 1
    case[key] = list(np.split(order, splits))
    return case[key]


def spatial_geometry_sample(case, k, seed):
    """70% uniform spatial cells, 15% high curvature, 15% near junction.

    Fixed-size sampling without replacement, using input geometry only. The
    point chosen inside every occupied cell is random, avoiding source-row bias.
    Missing optional junction geometry reallocates its quota to uniform points.
    """
    n = len(case["pos"])
    if k <= 0 or k >= n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    uniform_n = max(1, round(0.70 * k))
    groups = _voxel_groups(case, uniform_n)
    selected = np.zeros(n, dtype=bool)
    out = [int(g[rng.integers(len(g))]) for g in groups]
    selected[out] = True
    if len(out) < uniform_n:
        more = rng.choice(np.flatnonzero(~selected), uniform_n - len(out), replace=False)
        out.extend(more.tolist()); selected[more] = True

    def add_quota(score, count, high):
        nonlocal out
        score = np.asarray(score, dtype=np.float64)
        finite = np.isfinite(score)
        if not finite.any() or count <= 0:
            return
        threshold = np.quantile(score[finite], 0.75 if high else 0.25)
        eligible = finite & ((score >= threshold) if high else (score <= threshold)) & ~selected
        pool = np.flatnonzero(eligible)
        take = rng.choice(pool, min(count, len(pool)), replace=False)
        out.extend(take.tolist()); selected[take] = True

    # Prefer fine-scale wall curvedness when supplied by the existing 25D view.
    curvature = case.get("curvedness", np.abs(case["curvature"]))
    curvature_n = min(round(0.15 * k), k - len(out))
    add_quota(curvature, curvature_n, True)
    if "dist_to_junction_mm" in case:
        add_quota(case["dist_to_junction_mm"], k - uniform_n - curvature_n, False)
    if len(out) < k:
        out.extend(rng.choice(np.flatnonzero(~selected), k - len(out), replace=False).tolist())
    result = np.asarray(out, dtype=np.int64)
    rng.shuffle(result)
    return result
