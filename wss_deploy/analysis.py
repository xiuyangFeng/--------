"""Analysis layer: along-branch profiles, a findings list and trust masks for the reports.

Everything here is derived from the prediction arrays, the confirmed centreline atlas and the
input geometry.  No CFD data is read, no prediction value is changed, and nothing is written
to ``field.npz``: the outputs are JSON blocks for ``summary.json`` plus display arrays that
are embedded in the HTML report only.  Contract: ``wss_deploy/ANALYSIS_CONTRACT.md`` §1–§4.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

PROFILES_SCHEMA = "wss-deploy.profiles/v1"
FINDINGS_SCHEMA = "wss-deploy.findings/v1"
TRUST_SCHEMA = "wss-deploy.trust/v1"
BRANCH_ORDER = ("主动脉", "左髂总", "左髂外", "左髂内", "右髂总", "右髂外", "右髂内")
MAX_POINT_INDICES = 2000
ROUGH_SURFACE_VARIATION = 0.02   # PCA surface variation above which a wall point is "rough"
STENOSIS_LISTING_INDEX = 0.3     # min_radius findings are listed only for branches at or above this
TRUST_BITS = {1: "interpolation_uncovered", 2: "rough_surface", 4: "geometry_out_of_range",
              8: "low_sample_support", 16: "near_opening"}
TRUST_SOURCES = {
    1: {"label": "插值无支撑", "rule": "Gaussian 插值 1.5 mm 内无预测点"},
    2: {"label": "表面粗糙", "rule": f"最近预测点的 PCA 表面变化率 > {ROUGH_SURFACE_VARIATION:g}"},
    4: {"label": "几何越界", "rule": "分支几何超出发布包声明的参考范围（reference_assessment 为 review）"},
    8: {"label": "采样支撑弱", "rule": "内部点 8 近邻半径 > 全局中位数 × 2"},
    16: {"label": "邻近切口", "rule": "到最近切口中心的距离 < 2 × 该切口半径"},
}


# ----------------------------------------------------------------------------- helpers
def _finite(value) -> float | None:
    """JSON-safe float: NaN/inf become null."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _branch_name(branch_names: Mapping[str, str] | None, sid: int) -> str:
    name = (branch_names or {}).get(str(int(sid)))
    return str(name) if isinstance(name, str) and name else f"分支 {int(sid)}"


def _ordered_segments(atlas, branch_names: Mapping[str, str] | None) -> list[dict]:
    """Atlas segments in the clinical order (aorta, left CIA, left ext/int, right ...); unknown names last."""
    def key(segment):
        name = _branch_name(branch_names, int(segment["segment_id"]))
        order = BRANCH_ORDER.index(name) if name in BRANCH_ORDER else len(BRANCH_ORDER)
        return (order, int(segment["segment_id"]))
    return sorted(list(atlas.segments), key=key)


def _clusters(points: np.ndarray, mask: np.ndarray, radius: float, *, min_points: int) -> list[np.ndarray]:
    """Connected components of the masked points at the given linking radius, largest first.

    Same recipe as ``metrics.compute`` (``cKDTree.query_pairs`` + connected components); the
    returned arrays are indices into the *full* point array.
    """
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return []
    pairs = cKDTree(points[idx]).query_pairs(r=float(radius), output_type="ndarray")
    if len(pairs) == 0:
        labels = np.arange(len(idx))
    else:
        graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(idx), len(idx)))
        _, labels = connected_components(graph, directed=False)
    sizes = np.bincount(labels)
    order = np.argsort(-sizes, kind="stable")
    return [idx[labels == label] for label in order if sizes[label] >= max(1, int(min_points))]


def _subsample(indices: np.ndarray, limit: int = MAX_POINT_INDICES) -> list[int]:
    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) > limit:
        indices = indices[np.linspace(0, len(indices) - 1, limit).round().astype(np.int64)]
    return [int(i) for i in indices]


def _extent(points: np.ndarray, centre: np.ndarray, floor: float = 2.0) -> float:
    if len(points) == 0:
        return float(floor)
    return float(max(floor, np.max(np.linalg.norm(points - centre, axis=1))))


def _segment_rows(atlas, sid: int) -> np.ndarray:
    seg = np.asarray(atlas.col("segment_id")).astype(int)
    rows = np.flatnonzero(seg == int(sid))
    order = np.argsort(np.asarray(atlas.col("sample_index"))[rows], kind="stable")
    return rows[order]


def _median_spacing(points: np.ndarray) -> float:
    if len(points) < 2:
        return 1.0
    d, _ = cKDTree(points).query(points, k=2)
    return float(np.median(d[:, 1]))


# ----------------------------------------------------------------------------- §1 profiles
def profiles(atlas, feats: Mapping[str, Any], values_by_field: Mapping[str, Any], *,
             point_mask=None, bin_mm: float = 2.0, branch_names: Mapping[str, str] | None = None) -> dict:
    """Bin predictions along every branch by atlas ``s_local_mm``.

    ``feats`` needs ``segment_id`` and ``s_local_mm`` per point (``wss_features.atlas.map_points``
    output or the pipeline ``geom`` dict).  ``values_by_field`` is ``{"wss": wss_pa}`` for the wall
    family or ``{"speed": speed_m_s, "pressure": pressure_pa}`` for the volume family; the volume
    pressure is the model's relative pressure, so its curve is meaningful for differences only.
    Empty bins keep ``n = 0`` and ``null`` values.
    """
    if not np.isfinite(bin_mm) or bin_mm <= 0:
        raise ValueError("bin_mm must be positive")
    seg_pt = np.asarray(feats["segment_id"]).astype(int)
    s_pt = np.asarray(feats["s_local_mm"], dtype=np.float64)
    n = len(seg_pt)
    if s_pt.shape != (n,):
        raise ValueError("feats.segment_id and feats.s_local_mm must have one value per point")
    mask = np.ones(n, dtype=bool) if point_mask is None else np.asarray(point_mask, dtype=bool)
    if mask.shape != (n,):
        raise ValueError("point_mask must have one value per point")
    fields = {}
    for key, value in values_by_field.items():
        arr = np.asarray(value, dtype=np.float64)
        if arr.shape != (n,):
            raise ValueError(f"values_by_field[{key!r}] must have one value per point")
        fields[key] = arr
    kind = "wss" if "wss" in fields else "volume"
    atlas_seg = np.asarray(atlas.col("segment_id")).astype(int)
    atlas_s = np.asarray(atlas.col("s_local_mm"), dtype=np.float64)
    atlas_root = np.asarray(atlas.col("s_from_root_mm"), dtype=np.float64)
    atlas_r = np.asarray(atlas.col("radius_mm"), dtype=np.float64)
    branches = []
    for segment in _ordered_segments(atlas, branch_names):
        sid = int(segment["segment_id"])
        rows = np.flatnonzero(atlas_seg == sid)
        if len(rows) == 0:
            continue
        length = float(max(atlas_s[rows].max(), 0.0))
        n_bins = max(1, int(math.ceil(length / bin_mm))) if length > 0 else 1
        edges = np.arange(n_bins + 1, dtype=np.float64) * bin_mm
        centres = 0.5 * (edges[:-1] + edges[1:])
        offset = float(segment.get("s_offset_mm", np.min(atlas_root[rows]) - np.min(atlas_s[rows])))

        def bin_of(s):
            return np.clip(np.floor(np.clip(s, 0.0, length) / bin_mm).astype(np.int64), 0, n_bins - 1)

        atlas_bin = bin_of(atlas_s[rows])
        radius, s_root = [], []
        for b in range(n_bins):
            hit = rows[atlas_bin == b]
            radius.append(_finite(np.median(atlas_r[hit])) if len(hit) else None)
            s_root.append(_finite(np.median(atlas_root[hit])) if len(hit) else _finite(offset + centres[b]))
        sel = np.flatnonzero(mask & (seg_pt == sid))
        pbin = bin_of(s_pt[sel]) if len(sel) else np.zeros(0, dtype=np.int64)
        counts = np.bincount(pbin, minlength=n_bins) if len(sel) else np.zeros(n_bins, dtype=np.int64)
        entry = {"segment_id": sid, "name": _branch_name(branch_names, sid), "parent_id": int(segment.get("parent_id", -1)),
                 "length_mm": length, "n_bins": int(n_bins),
                 "s_local_mm": [float(c) for c in centres], "s_from_root_mm": s_root, "radius_mm": radius}
        block: dict[str, list] = {"n": [int(c) for c in counts]}
        if kind == "wss":
            w = fields["wss"][sel]
            mean, p99, vmin = [], [], []
            for b in range(n_bins):
                v = w[pbin == b]
                mean.append(_finite(v.mean()) if len(v) else None)
                p99.append(_finite(np.quantile(v, .99)) if len(v) else None)
                vmin.append(_finite(v.min()) if len(v) else None)
            block.update(mean_pa=mean, p99_pa=p99, min_pa=vmin)
            entry["wss"] = block
        else:
            speed = fields.get("speed")
            pressure = fields.get("pressure")
            sm, sx, pm, pn = [], [], [], []
            for b in range(n_bins):
                pick = sel[pbin == b]
                sv = speed[pick] if speed is not None else np.zeros(0)
                pv = pressure[pick] if pressure is not None else np.zeros(0)
                sm.append(_finite(sv.mean()) if len(sv) else None); sx.append(_finite(sv.max()) if len(sv) else None)
                pm.append(_finite(pv.mean()) if len(pv) else None); pn.append(_finite(pv.min()) if len(pv) else None)
            block.update(speed_mean_m_s=sm, speed_max_m_s=sx, pressure_mean_pa=pm, pressure_min_pa=pn)
            entry["volume"] = block
        branches.append(entry)
    return {"schema_version": PROFILES_SCHEMA, "bin_mm": float(bin_mm), "kind": kind,
            "definition": ("按中心线弧长每 bin_mm 分箱的预测统计；radius_mm 为该弧长处中心线内切半径的中位数；"
                           "体场压力为相对压，曲线只用于差值。"),
            "branches": branches}


# ----------------------------------------------------------------------------- §2 findings
MAX_DIAMETER_DEFINITION = "壁面网格截面的最大 Feret 直径（每 1 mm 一站）"


def _morphology_max_diameter(morphology: Mapping[str, Any] | None, branch_names) -> dict | None:
    """The ``max_diameter`` finding taken from ``morphology.aorta.max`` (contract §17.1)."""
    aorta = (morphology or {}).get("aorta") if isinstance(morphology, Mapping) else None
    if not isinstance(aorta, Mapping):
        return None
    largest = aorta.get("max")
    if not isinstance(largest, Mapping):
        return None
    value = _finite(largest.get("max_diameter_mm"))
    xyz = largest.get("xyz_mm")
    if value is None or not (isinstance(xyz, (list, tuple)) and len(xyz) == 3):
        return None
    sid = int(aorta.get("segment_id", 0))
    name = aorta.get("name") if isinstance(aorta.get("name"), str) else _branch_name(branch_names, sid)
    item = {"kind": "max_diameter", "label": f"{name}最大直径", "branch": name, "segment_id": sid,
            "value": value, "units": "mm", "xyz_mm": [float(v) for v in xyz],
            "s_from_root_mm": _finite(largest.get("s_from_root_mm")),
            "extent_mm": float(max(2.0, 0.75 * value)), "area_mm2": _finite(largest.get("area_mm2")),
            "n_points": 0, "severity": "info", "source": "morphology",
            "equivalent_diameter_mm": _finite(largest.get("equivalent_diameter_mm")),
            "definition": MAX_DIAMETER_DEFINITION}
    if largest.get("oblique"):
        item["oblique"] = True
    return item


def _geometry_findings(atlas, geometry_table: Mapping[str, Any] | None, branch_names, *, start_id: int,
                       morphology: Mapping[str, Any] | None = None) -> list[dict]:
    """``max_diameter`` (global) and ``min_radius`` (per branch with stenosis index ≥ 0.3).

    The diameter comes from the wall-mesh stations when a ``morphology`` block is available
    (contract §17.1) and from the centreline's inscribed radius otherwise.
    """
    items: list[dict] = []
    atlas_seg = np.asarray(atlas.col("segment_id")).astype(int)
    atlas_r = np.asarray(atlas.col("radius_mm"), dtype=np.float64)
    atlas_root = np.asarray(atlas.col("s_from_root_mm"), dtype=np.float64)
    xyz = np.asarray(atlas.xyz, dtype=np.float64)
    table = geometry_table or {}
    from_morphology = _morphology_max_diameter(morphology, branch_names)
    best = None
    for segment in _ordered_segments(atlas, branch_names):
        sid = int(segment["segment_id"])
        rows = np.flatnonzero(atlas_seg == sid)
        if len(rows) < 3:
            continue
        j = rows[int(np.argmax(atlas_r[rows]))]
        if best is None or atlas_r[j] > best[0]:
            best = (float(atlas_r[j]), int(j), sid)
    if from_morphology is not None:
        items.append(from_morphology)
    elif best is not None:
        r, j, sid = best
        items.append({"kind": "max_diameter", "label": f"{_branch_name(branch_names, sid)}最大直径",
                      "branch": _branch_name(branch_names, sid), "segment_id": sid,
                      "value": 2.0 * r, "units": "mm", "xyz_mm": [float(v) for v in xyz[j]],
                      "s_from_root_mm": _finite(atlas_root[j]), "extent_mm": float(max(2.0, 1.5 * r)),
                      "area_mm2": None, "n_points": 0, "severity": "info",
                      "definition": "中心线内切半径最大处的直径（2 × 半径），来自中心线 atlas，不是壁面网格直径"})
    for segment in _ordered_segments(atlas, branch_names):
        sid = int(segment["segment_id"])
        name = _branch_name(branch_names, sid)
        rows = np.flatnonzero(atlas_seg == sid)
        if len(rows) < 3:
            continue
        r = atlas_r[rows]
        entry = table.get(name) if isinstance(table, Mapping) else None
        stenosis = _finite(entry.get("stenosis_index")) if isinstance(entry, Mapping) else None
        if stenosis is None:
            med = float(np.median(r))
            stenosis = float(1 - r.min() / med) if med > 0 else None
        if stenosis is None or stenosis < STENOSIS_LISTING_INDEX:
            continue
        j = rows[int(np.argmin(r))]
        items.append({"kind": "min_radius", "label": f"{name}最小半径（狭窄指数 {stenosis:.2f}）", "branch": name, "segment_id": sid,
                      "value": float(r.min()), "units": "mm", "xyz_mm": [float(v) for v in xyz[j]],
                      "s_from_root_mm": _finite(atlas_root[j]), "extent_mm": float(max(2.0, 3.0 * r.min())),
                      "area_mm2": None, "n_points": 0, "severity": "info", "stenosis_index": float(stenosis),
                      "definition": f"该分支中心线内切半径最小处；狭窄指数 = 1 − 最小半径 / 中位半径，仅当 ≥ {STENOSIS_LISTING_INDEX:g} 时列出"})
    for k, item in enumerate(items):
        item["id"] = f"F{start_id + k}"
    return items


def apply_morphology(findings: dict, morphology: Mapping[str, Any] | None, branch_names=None) -> dict:
    """Swap the ``max_diameter`` finding for the wall-mesh measurement, keeping ids and order.

    Used where the findings must exist before the morphology (the volume family needs the per-branch
    ΔP items to fill ``morphology.branches[].delta_p_pa``); recomputing the findings afterwards would
    repeat the clustering for nothing.
    """
    item = _morphology_max_diameter(morphology, branch_names)
    if item is None or not isinstance(findings, dict):
        return findings
    for index, existing in enumerate(findings.get("items") or []):
        if isinstance(existing, dict) and existing.get("kind") == "max_diameter":
            findings["items"][index] = {**item, "id": existing.get("id"), "rank": existing.get("rank")}
            break
    return findings


def _number_items(items: list[dict]) -> list[dict]:
    for rank, item in enumerate(items, start=1):
        item["id"] = f"F{rank}"
        item["rank"] = rank
    return items


def findings_wall(pts, wss, feats: Mapping[str, Any], atlas, geometry_table: Mapping[str, Any] | None, *,
                  thresholds: Sequence[float], total_area_mm2: float, spacing_mm: float,
                  branch_names: Mapping[str, str] | None = None, min_cluster_points: int = 20,
                  max_clusters: int = 5, morphology: Mapping[str, Any] | None = None) -> dict:
    """Findings for the wall WSS family (contract §2)."""
    pts = np.asarray(pts, dtype=np.float64); wss = np.asarray(wss, dtype=np.float64)
    n = len(pts)
    if wss.shape != (n,) or not np.isfinite(wss).all():
        raise ValueError("wss must contain one finite value per point")
    if not np.isfinite(total_area_mm2) or total_area_mm2 <= 0 or not np.isfinite(spacing_mm) or spacing_mm <= 0:
        raise ValueError("total_area_mm2 and spacing_mm must be positive")
    low, high, very_high = [float(t) for t in thresholds]
    seg = np.asarray(feats["segment_id"]).astype(int)
    s_root = np.asarray(feats["s_from_root_mm"], dtype=np.float64)
    radius = np.asarray(feats["radius_mm"], dtype=np.float64)
    area_per_point = total_area_mm2 / n
    link = 2.0 * spacing_mm
    items: list[dict] = []
    p99 = float(np.quantile(wss, .99))
    for cluster in _clusters(pts, wss >= p99, link, min_points=min_cluster_points)[:max_clusters]:
        top = cluster[int(np.argmax(wss[cluster]))]
        sid = int(seg[top]); value = float(wss[top])
        items.append({"kind": "high_wss_cluster", "label": f"{_branch_name(branch_names, sid)}高 WSS 区", "branch": _branch_name(branch_names, sid),
                      "segment_id": sid, "value": value, "units": "Pa", "xyz_mm": [float(v) for v in pts[top]],
                      "s_from_root_mm": _finite(s_root[top]), "extent_mm": _extent(pts[cluster], pts[top]),
                      "area_mm2": float(len(cluster) * area_per_point), "n_points": int(len(cluster)),
                      "severity": "attention" if value >= very_high else "note",
                      "cluster_mean_pa": float(wss[cluster].mean()),
                      "definition": f"WSS ≥ 空间 p99（{p99:.2f} Pa）的连通簇（连接半径 {link:.2f} mm，≥ {min_cluster_points} 点）；value 为簇内最大值；面积按点占比 × 输入壁面面积估计",
                      "point_indices": _subsample(cluster)})
    imax = int(np.argmax(wss)); sid = int(seg[imax])
    items.append({"kind": "max_wss", "label": f"全场最大 WSS（{_branch_name(branch_names, sid)}）", "branch": _branch_name(branch_names, sid),
                  "segment_id": sid, "value": float(wss[imax]), "units": "Pa", "xyz_mm": [float(v) for v in pts[imax]],
                  "s_from_root_mm": _finite(s_root[imax]), "extent_mm": float(max(2.0, 2.0 * radius[imax])),
                  "area_mm2": None, "n_points": 1, "severity": "attention" if wss[imax] >= very_high else "note",
                  "definition": "预测点云中的单点最大值，对噪声敏感，仅作参考；空间 p99 是主指标"})
    low_clusters = _clusters(pts, wss < low, link, min_points=min_cluster_points)
    low_total_fraction = float(sum(len(c) for c in low_clusters) / n)
    for cluster in low_clusters[:max_clusters]:
        centre = pts[cluster].mean(axis=0)
        rep = cluster[int(np.argmin(np.linalg.norm(pts[cluster] - centre, axis=1)))]
        sid = int(seg[rep])
        items.append({"kind": "low_wss_cluster", "label": f"{_branch_name(branch_names, sid)}低 WSS 区", "branch": _branch_name(branch_names, sid),
                      "segment_id": sid, "value": float(wss[cluster].mean()), "units": "Pa", "xyz_mm": [float(v) for v in pts[rep]],
                      "s_from_root_mm": _finite(s_root[rep]), "extent_mm": _extent(pts[cluster], pts[rep]),
                      "area_mm2": float(len(cluster) * area_per_point), "n_points": int(len(cluster)),
                      "severity": "attention" if low_total_fraction > 0.2 else "note",
                      "cluster_min_pa": float(wss[cluster].min()),
                      "definition": f"WSS < {low:g} Pa 的连通簇（连接半径 {link:.2f} mm，≥ {min_cluster_points} 点）；value 为簇内均值，位置取簇质心最近点；面积按点占比估计",
                      "point_indices": _subsample(cluster)})
    items.extend(_geometry_findings(atlas, geometry_table, branch_names, start_id=len(items) + 1,
                                    morphology=morphology))
    return {"schema_version": FINDINGS_SCHEMA, "family": "wall",
            "thresholds_pa": [low, high, very_high], "p99_threshold_pa": p99,
            "low_wss_total_fraction": low_total_fraction, "connectivity_radius_mm": link,
            "items": _number_items(items)}


LOW_SPEED_LINK_FACTOR = 3.0   # interior samples are sparser than the wall cloud; 2× the median spacing leaves only singletons


def findings_volume(internal_pts, speed, pressure, feats: Mapping[str, Any], atlas,
                    geometry_table: Mapping[str, Any] | None, *, branch_names: Mapping[str, str] | None = None,
                    min_cluster_points: int = 20, max_low_speed: int = 3, end_fraction: float = 0.10,
                    min_end_points: int = 3, link_factor: float = LOW_SPEED_LINK_FACTOR,
                    morphology: Mapping[str, Any] | None = None) -> dict:
    """Findings for the PF6/VF6 volume family (contract §2). ``feats`` indexes interior points only."""
    pts = np.asarray(internal_pts, dtype=np.float64)
    speed = np.asarray(speed, dtype=np.float64); pressure = np.asarray(pressure, dtype=np.float64)
    n = len(pts)
    if speed.shape != (n,) or pressure.shape != (n,) or not np.isfinite(speed).all() or not np.isfinite(pressure).all():
        raise ValueError("speed and pressure must contain one finite value per interior point")
    seg = np.asarray(feats["segment_id"]).astype(int)
    s_local = np.asarray(feats["s_local_mm"], dtype=np.float64)
    s_root = np.asarray(feats["s_from_root_mm"], dtype=np.float64)
    radius = np.asarray(feats["radius_mm"], dtype=np.float64)
    items: list[dict] = []
    i = int(np.argmax(speed)); sid = int(seg[i])
    items.append({"kind": "max_speed", "label": f"最大速度（{_branch_name(branch_names, sid)}）", "branch": _branch_name(branch_names, sid),
                  "segment_id": sid, "value": float(speed[i]), "units": "m/s", "xyz_mm": [float(v) for v in pts[i]],
                  "s_from_root_mm": _finite(s_root[i]), "extent_mm": float(max(2.0, 2.0 * radius[i])),
                  "area_mm2": None, "n_points": 1, "severity": "note",
                  "definition": "内部预测点中的单点最大速度模长；固定收缩期帧，对噪声敏感"})
    i = int(np.argmin(pressure)); sid = int(seg[i])
    items.append({"kind": "min_pressure", "label": f"最低相对压力（{_branch_name(branch_names, sid)}）", "branch": _branch_name(branch_names, sid),
                  "segment_id": sid, "value": float(pressure[i]), "units": "Pa", "xyz_mm": [float(v) for v in pts[i]],
                  "s_from_root_mm": _finite(s_root[i]), "extent_mm": float(max(2.0, 2.0 * radius[i])),
                  "area_mm2": None, "n_points": 1, "severity": "note",
                  "definition": "内部预测点中的单点最低相对压力（相对该帧体积平均压），不是绝对血压"})
    atlas_seg = np.asarray(atlas.col("segment_id")).astype(int)
    atlas_s = np.asarray(atlas.col("s_local_mm"), dtype=np.float64)
    atlas_root = np.asarray(atlas.col("s_from_root_mm"), dtype=np.float64)
    atlas_r = np.asarray(atlas.col("radius_mm"), dtype=np.float64)
    xyz = np.asarray(atlas.xyz, dtype=np.float64)
    drops = []
    for segment in _ordered_segments(atlas, branch_names):
        sid = int(segment["segment_id"])
        rows = _segment_rows(atlas, sid)
        if len(rows) < 3:
            continue
        length = float(atlas_s[rows].max())
        if length <= 0:
            continue
        inside = seg == sid
        prox = inside & (s_local <= end_fraction * length)
        dist = inside & (s_local >= (1.0 - end_fraction) * length)
        if prox.sum() < min_end_points or dist.sum() < min_end_points:
            continue
        mid = rows[int(np.argmin(np.abs(atlas_s[rows] - 0.5 * length)))]
        drop = float(pressure[prox].mean() - pressure[dist].mean())
        name = _branch_name(branch_names, sid)
        drops.append({"kind": "pressure_drop", "label": f"{name}沿程压降", "branch": name, "segment_id": sid,
                      "value": drop, "units": "Pa", "xyz_mm": [float(v) for v in xyz[mid]],
                      "s_from_root_mm": _finite(atlas_root[mid]), "extent_mm": float(max(2.0, 0.5 * length)),
                      "area_mm2": None, "n_points": int(prox.sum() + dist.sum()), "severity": "note",
                      "proximal_mean_pa": float(pressure[prox].mean()), "distal_mean_pa": float(pressure[dist].mean()),
                      "proximal_n": int(prox.sum()), "distal_n": int(dist.sum()),
                      "definition": (f"分支近端 {end_fraction:.0%} 与远端 {end_fraction:.0%} 弧长段内部点相对压力均值之差（近端 − 远端）；"
                                     "正值表示沿血流方向下降；按点等权，不是流量加权的压降"),
                      "point_indices": _subsample(np.flatnonzero(prox | dist))})
    drops.sort(key=lambda item: -abs(item["value"]))
    items.extend(drops)
    if n >= 2:
        link = float(link_factor) * _median_spacing(pts)
        p10 = float(np.quantile(speed, .10))
        for cluster in _clusters(pts, speed < p10, link, min_points=min_cluster_points)[:max_low_speed]:
            centre = pts[cluster].mean(axis=0)
            rep = cluster[int(np.argmin(np.linalg.norm(pts[cluster] - centre, axis=1)))]
            sid = int(seg[rep])
            items.append({"kind": "low_speed_region", "label": f"{_branch_name(branch_names, sid)}低速区", "branch": _branch_name(branch_names, sid),
                          "segment_id": sid, "value": float(speed[cluster].mean()), "units": "m/s", "xyz_mm": [float(v) for v in pts[rep]],
                          "s_from_root_mm": _finite(s_root[rep]), "extent_mm": _extent(pts[cluster], pts[rep]),
                          "area_mm2": None, "n_points": int(len(cluster)), "severity": "note",
                          "definition": f"速度 < 全场 p10（{p10:.3f} m/s）的内部点连通簇（连接半径 {link:.2f} mm，≥ {min_cluster_points} 点）；value 为簇内均值；采样点等权，不是体积占比",
                          "point_indices": _subsample(cluster)})
    items.extend(_geometry_findings(atlas, geometry_table, branch_names, start_id=len(items) + 1,
                                    morphology=morphology))
    return {"schema_version": FINDINGS_SCHEMA, "family": "volume",
            "low_speed_link_mm": float(link_factor) * _median_spacing(pts) if n >= 2 else None,
            "items": _number_items(items)}


# ----------------------------------------------------------------------------- §3 trust
def review_segments(reference_assessment: Mapping[str, Any] | None, branch_names: Mapping[str, str] | None) -> set[int]:
    """Segment ids whose geometry checks are flagged ``review`` in the release reference assessment."""
    out: set[int] = set()
    checks = (reference_assessment or {}).get("checks") if isinstance(reference_assessment, Mapping) else None
    if not isinstance(checks, list):
        return out
    by_name: dict[str, int] = {}
    for sid, name in (branch_names or {}).items():
        try:
            by_name[str(name)] = int(sid)
        except (TypeError, ValueError):
            continue
    for check in checks:
        if not isinstance(check, Mapping) or check.get("status") != "review":
            continue
        path = str(check.get("path", ""))
        if not path.startswith("geometry."):
            continue
        branch = path[len("geometry."):].rsplit(".", 1)[0]
        if branch in by_name:
            out.add(by_name[branch])
    return out


def _trust_block(fractions: dict[str, float], bits: Sequence[int], support: dict[str, str]) -> dict:
    return {"schema_version": TRUST_SCHEMA,
            "bits": {str(b): TRUST_BITS[b] for b in bits},
            "fractions": {TRUST_BITS[b]: float(fractions.get(TRUST_BITS[b], 0.0)) for b in bits},
            "sources": [{"bit": b, "name": TRUST_BITS[b], **TRUST_SOURCES[b], "support": support.get(TRUST_BITS[b], "vertices")} for b in bits],
            "note": "可信区域只标出已知不可靠的位置，不代表其它位置的预测准确率"}


def trust_wall(vertices, vertex_values, pts, surface_variation, vertex_segment, reference_assessment,
               branch_names: Mapping[str, str] | None = None) -> tuple[np.ndarray, dict]:
    """Per-vertex trust bits for the wall family: 1 uncovered, 2 rough, 4 geometry out of range."""
    vertices = np.asarray(vertices, dtype=np.float64)
    values = np.asarray(vertex_values, dtype=np.float64)
    if values.shape != (len(vertices),):
        raise ValueError("vertex_values must contain one value per vertex")
    bits = np.zeros(len(vertices), dtype=np.uint8)
    bits[~np.isfinite(values)] |= 1
    if surface_variation is not None and len(pts):
        variation = np.asarray(surface_variation, dtype=np.float64)
        pts = np.asarray(pts, dtype=np.float64)
        if variation.shape != (len(pts),):
            raise ValueError("surface_variation must contain one value per prediction point")
        _, nearest = cKDTree(pts).query(vertices, k=1)
        bits[variation[nearest] > ROUGH_SURFACE_VARIATION] |= 2
    flagged = review_segments(reference_assessment, branch_names)
    if flagged and vertex_segment is not None:
        vseg = np.asarray(vertex_segment).astype(int)
        bits[np.isin(vseg, list(flagged))] |= 4
    fractions = {TRUST_BITS[b]: float(np.mean((bits & b) > 0)) for b in (1, 2, 4)}
    block = _trust_block(fractions, (1, 2, 4), {})
    block["family"] = "wall"; block["n_vertices"] = int(len(vertices)); block["review_segments"] = sorted(flagged)
    return bits, block


def trust_volume(vertices, vertex_pressure, vertex_segment, interior_pts, interior_segment, caps,
                 reference_assessment, branch_names: Mapping[str, str] | None = None,
                 *, support_k: int = 8, support_factor: float = 2.0, opening_factor: float = 2.0) -> tuple[np.ndarray, np.ndarray, dict]:
    """Trust bits for the volume family: wall vertices (1, 4) and interior points (4, 8, 16)."""
    vertices = np.asarray(vertices, dtype=np.float64)
    vp = np.asarray(vertex_pressure, dtype=np.float64)
    if vp.shape != (len(vertices),):
        raise ValueError("vertex_pressure must contain one value per vertex")
    mesh_bits = np.zeros(len(vertices), dtype=np.uint8)
    mesh_bits[~np.isfinite(vp)] |= 1
    interior = np.asarray(interior_pts, dtype=np.float64).reshape(-1, 3)
    cloud_bits = np.zeros(len(interior), dtype=np.uint8)
    flagged = review_segments(reference_assessment, branch_names)
    if flagged:
        if vertex_segment is not None:
            mesh_bits[np.isin(np.asarray(vertex_segment).astype(int), list(flagged))] |= 4
        if interior_segment is not None and len(interior):
            cloud_bits[np.isin(np.asarray(interior_segment).astype(int), list(flagged))] |= 4
    if len(interior) > support_k:
        d, _ = cKDTree(interior).query(interior, k=support_k + 1)
        local = d[:, -1]
        cloud_bits[local > support_factor * float(np.median(local))] |= 8
    for cap in caps or []:
        try:
            centre = np.asarray(cap["center_mm"], dtype=np.float64).reshape(3)
            r = float(cap["radius_mm"])
        except (KeyError, TypeError, ValueError):
            continue
        if not np.isfinite(r) or r <= 0 or not len(interior):
            continue
        cloud_bits[np.linalg.norm(interior - centre, axis=1) < opening_factor * r] |= 16
    fractions = {TRUST_BITS[1]: float(np.mean((mesh_bits & 1) > 0)),
                 TRUST_BITS[4]: float(np.mean((cloud_bits & 4) > 0)) if len(interior) else 0.0,
                 TRUST_BITS[8]: float(np.mean((cloud_bits & 8) > 0)) if len(interior) else 0.0,
                 TRUST_BITS[16]: float(np.mean((cloud_bits & 16) > 0)) if len(interior) else 0.0}
    block = _trust_block(fractions, (1, 4, 8, 16), {TRUST_BITS[1]: "vertices", TRUST_BITS[4]: "interior_points",
                                                    TRUST_BITS[8]: "interior_points", TRUST_BITS[16]: "interior_points"})
    block["family"] = "volume"; block["n_vertices"] = int(len(vertices)); block["n_interior"] = int(len(interior))
    block["review_segments"] = sorted(flagged); block["wall_geometry_out_of_range_fraction"] = float(np.mean((mesh_bits & 4) > 0))
    return mesh_bits, cloud_bits, block


# ----------------------------------------------------------------------------- §4 frame
def frame_transform(rotation, origin_mm, *, source: str, direction_source: str | None, **extra) -> dict:
    """Summary block for the anatomical frame: ``p_aligned = R @ (p - origin)`` (contract §4)."""
    R = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    o = np.asarray(origin_mm, dtype=np.float64).reshape(3)
    if not np.isfinite(R).all() or not np.isfinite(o).all():
        raise ValueError("frame rotation and origin must be finite")
    return {"source": source, "direction_source": direction_source or "unknown_stl",
            "rotation": R.tolist(), "origin_mm": o.tolist(),
            "convention": "p_aligned = R @ (p - origin)；x = 患者左，z = 指向入口，y = z × x（患者后）",
            **extra}


__all__ = ["BRANCH_ORDER", "FINDINGS_SCHEMA", "MAX_DIAMETER_DEFINITION", "PROFILES_SCHEMA", "TRUST_BITS", "TRUST_SCHEMA",
           "apply_morphology", "findings_volume", "findings_wall", "frame_transform", "profiles", "review_segments",
           "trust_volume", "trust_wall"]
