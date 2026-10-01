"""Analysis layer: along-branch profiles, a findings list and trust masks for the reports.

Everything here is derived from the prediction arrays, the confirmed centreline atlas and the
input geometry.  No CFD data is read, no prediction value is changed, and nothing is written
to ``field.npz``: the outputs are JSON blocks for ``summary.json`` plus display arrays that
are embedded in the HTML report only.  Contract: ``wss_deploy/ANALYSIS_CONTRACT.md`` §1–§4
(cycle-integrated TAWSS / OSI additions: §19.2).
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
ZONES_SCHEMA = "wss-deploy.zones/v1"
# 2026-09-30 listing rules (C6 / U11 / U12, WORKSPACE_V2_CONTRACT.md §5.2).
AREA_FLOOR_MM2 = 100.0            # area-type clusters with an estimated area below 1 cm² are not listed
AREA_KINDS = ("low_wss_cluster", "low_tawss_cluster", "high_osi_cluster", "stagnation_cluster")
HIGH_GRADING_QUANTILE = 0.90      # high-WSS items are 'attention' only when the case p99 reaches the cohort p90
REVIEW_MATCH_MM = 2.0             # a reviewed finding keeps its decision when a new item of the same kind is this close
LEGACY_REVIEW_LABEL = "规则更新前的判定"
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
    family (optionally plus ``tawss`` / ``osi`` / ``rrt`` / ``ecap`` for a three-head release, which add
    blocks of the same names per branch) or ``{"speed": speed_m_s, "pressure": pressure_pa}`` for the volume family; the volume
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
            # Cycle-integrated fields of a three-head release (contract §19.2): same bins, keys only when given.
            q90 = lambda v: np.quantile(v, .90)
            for key, stats in (("tawss", (("mean_pa", np.mean), ("min_pa", np.min))),
                               ("osi", (("mean", np.mean), ("p90", q90))),
                               # v0.13 derived indices (1/Pa): the high tail is the interesting side, like OSI.
                               ("rrt", (("mean", np.mean), ("p90", q90))),
                               ("ecap", (("mean", np.mean), ("p90", q90)))):
                if key in fields:
                    values = fields[key][sel]
                    entry[key] = {name: [_finite(fn(values[pbin == b])) if np.any(pbin == b) else None for b in range(n_bins)]
                                  for name, fn in stats}
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
MAX_DIAMETER_DEFINITION = "管腔截面的最大 Feret 直径（壁面网格每 1 mm 一站；输入是管腔面，不含附壁血栓与管壁）"


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
    item = {"kind": "max_diameter", "label": f"{name}管腔最大直径", "branch": name, "segment_id": sid,
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
        items.append({"kind": "max_diameter", "label": f"{_branch_name(branch_names, sid)}管腔最大直径",
                      "branch": _branch_name(branch_names, sid), "segment_id": sid,
                      "value": 2.0 * r, "units": "mm", "xyz_mm": [float(v) for v in xyz[j]],
                      "s_from_root_mm": _finite(atlas_root[j]), "extent_mm": float(max(2.0, 1.5 * r)),
                      "area_mm2": None, "n_points": 0, "severity": "info",
                      "definition": "中心线内切半径最大处的管腔直径（2 × 半径），来自中心线 atlas，不是壁面网格直径"})
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


CYCLE_MAX_CLUSTERS = 3


def _area_listed(clusters: list[np.ndarray], area_per_point: float, floor_mm2: float, kind: str,
                 suppressed: dict) -> list[np.ndarray]:
    """U12: drop area-type clusters whose estimated area is below ``floor_mm2`` (recorded in ``suppressed``).

    ``clusters`` are sorted largest first, so the filter only removes the tail: taking the first N afterwards
    lists exactly the old first N minus the small ones.
    """
    keep = [c for c in clusters if len(c) * area_per_point >= floor_mm2]
    dropped = [c for c in clusters if len(c) * area_per_point < floor_mm2]
    if dropped:
        suppressed[kind] = {"count": len(dropped), "area_mm2": float(sum(len(c) for c in dropped) * area_per_point)}
    return keep


def _cycle_findings(pts, cycle: Mapping[str, Any], seg, s_root, *, link: float, min_points: int,
                    area_per_point: float, branch_names, area_floor_mm2: float = 0.0,
                    suppressed: dict | None = None) -> list[dict]:
    """Stagnation (TAWSS < 0.4 ∧ OSI > 0.1) and high-OSI (> 0.3) clusters of a three-head release (§19.2).

    2026-09-30 (U12): clusters whose estimated area is below ``area_floor_mm2`` are not listed.
    """
    from .cycle_fields import OSI_THRESHOLDS, STAGNATION
    suppressed = {} if suppressed is None else suppressed
    n = len(pts)
    arrays = {}
    for key in ("tawss", "osi"):
        if cycle.get(key) is None:
            continue
        values = np.asarray(cycle[key], dtype=np.float64)
        if values.shape != (n,) or not np.isfinite(values).all():
            raise ValueError(f"cycle[{key!r}] must contain one finite value per point")
        arrays[key] = values
    tawss, osi = arrays.get("tawss"), arrays.get("osi")
    items: list[dict] = []
    t_lt, o_gt, o_high = float(STAGNATION["tawss_lt_pa"]), float(STAGNATION["osi_gt"]), float(OSI_THRESHOLDS[2])
    if tawss is not None and osi is not None:
        clusters = _area_listed(_clusters(pts, (tawss < t_lt) & (osi > o_gt), link, min_points=min_points),
                                area_per_point, area_floor_mm2, "stagnation_cluster", suppressed)[:CYCLE_MAX_CLUSTERS]
        for k, cluster in enumerate(clusters):
            centre = pts[cluster].mean(axis=0)
            rep = cluster[int(np.argmin(np.linalg.norm(pts[cluster] - centre, axis=1)))]
            sid = int(seg[rep]); name = _branch_name(branch_names, sid); area = float(len(cluster) * area_per_point)
            items.append({"kind": "stagnation_cluster", "label": f"{name}滞留区", "branch": name, "segment_id": sid,
                          "value": area / 100.0, "units": "cm²", "xyz_mm": [float(v) for v in pts[rep]],
                          "s_from_root_mm": _finite(s_root[rep]), "extent_mm": _extent(pts[cluster], pts[rep]),
                          "area_mm2": area, "n_points": int(len(cluster)), "severity": "attention" if k == 0 else "info",
                          "tawss_mean_pa": float(tawss[cluster].mean()), "osi_mean": float(osi[cluster].mean()),
                          "definition": f"TAWSS < {t_lt:g} Pa 且 OSI > {o_gt:g} 的连通簇（连接半径 {link:.2f} mm，≥ {min_points} 点）；value 为簇面积（点占比 × 输入壁面面积），位置取簇质心最近点",
                          "point_indices": _subsample(cluster)})
    if osi is not None:
        for cluster in _area_listed(_clusters(pts, osi > o_high, link, min_points=min_points), area_per_point,
                                    area_floor_mm2, "high_osi_cluster", suppressed)[:CYCLE_MAX_CLUSTERS]:
            top = cluster[int(np.argmax(osi[cluster]))]
            sid = int(seg[top]); name = _branch_name(branch_names, sid)
            item = {"kind": "high_osi_cluster", "label": f"{name}高 OSI 区", "branch": name, "segment_id": sid,
                    "value": float(osi[top]), "units": "1", "xyz_mm": [float(v) for v in pts[top]],
                    "s_from_root_mm": _finite(s_root[top]), "extent_mm": _extent(pts[cluster], pts[top]),
                    "area_mm2": float(len(cluster) * area_per_point), "n_points": int(len(cluster)), "severity": "info",
                    "osi_mean": float(osi[cluster].mean()),
                    "definition": f"OSI > {o_high:g} 的连通簇（连接半径 {link:.2f} mm，≥ {min_points} 点）；value 为簇内最大 OSI，位置取该点",
                    "point_indices": _subsample(cluster)}
            if tawss is not None:
                item["tawss_mean_pa"] = float(tawss[cluster].mean())
            items.append(item)
    return items


def findings_wall(pts, wss, feats: Mapping[str, Any], atlas, geometry_table: Mapping[str, Any] | None, *,
                  thresholds: Sequence[float], total_area_mm2: float, spacing_mm: float,
                  branch_names: Mapping[str, str] | None = None, min_cluster_points: int = 20,
                  max_clusters: int = 5, morphology: Mapping[str, Any] | None = None,
                  cycle: Mapping[str, Any] | None = None, area_floor_mm2: float = AREA_FLOOR_MM2,
                  reference_assessment: Mapping[str, Any] | None = None) -> dict:
    """Findings for the wall WSS family (contract §2; listing rules of WORKSPACE_V2_CONTRACT.md §5.2).

    ``cycle`` = ``{"tawss": array, "osi": array}`` (three-head release) appends stagnation and high-OSI
    clusters after every other item (§19.2); without it the output is exactly the peak-WSS list.

    2026-09-30 rules:

    * U12 — the single-point maximum joins the high-WSS cluster that contains it (``contains_global_max``);
      it is listed on its own only when no listed cluster contains it.
    * U12 — area-type clusters (low WSS, high OSI, stagnation) under ``area_floor_mm2`` (1 cm²) are not
      listed; the statistics tables keep them.  High-WSS clusters keep the ≥ ``min_cluster_points`` rule.
    * U11 — high-WSS items are graded against the release's same-protocol cohort reference
      (:func:`grade_high_findings`); the low-value rules are unchanged.
    """
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
    floor = float(area_floor_mm2 or 0.0)
    link = 2.0 * spacing_mm
    items: list[dict] = []
    suppressed: dict[str, dict] = {}
    p99 = float(np.quantile(wss, .99))
    imax = int(np.argmax(wss))
    merged = False
    for cluster in _clusters(pts, wss >= p99, link, min_points=min_cluster_points)[:max_clusters]:
        top = cluster[int(np.argmax(wss[cluster]))]
        sid = int(seg[top]); value = float(wss[top])
        item = {"kind": "high_wss_cluster", "label": f"{_branch_name(branch_names, sid)}高 WSS 区", "branch": _branch_name(branch_names, sid),
                "segment_id": sid, "value": value, "units": "Pa", "xyz_mm": [float(v) for v in pts[top]],
                "s_from_root_mm": _finite(s_root[top]), "extent_mm": _extent(pts[cluster], pts[top]),
                "area_mm2": float(len(cluster) * area_per_point), "n_points": int(len(cluster)),
                "severity": "note",
                "cluster_mean_pa": float(wss[cluster].mean()),
                "definition": f"峰值 WSS 不低于本例空间 p99（{p99:.2f} Pa）的连通簇（连接半径 {link:.2f} mm，≥ {min_cluster_points} 点）；value 为簇内最大值；面积按点占比 × 输入壁面面积估计",
                "point_indices": _subsample(cluster)}
        if not merged and bool(np.any(cluster == imax)):
            # U12: the global maximum lies in this cluster (its top point is the maximum itself).
            merged = True
            item["contains_global_max"] = True
            item["global_max_pa"] = float(wss[imax])
            item["label"] += "（含全场最大值）"
            item["definition"] += "；含全场单点最大值（对噪声敏感，仅作参考），不再单列"
        items.append(item)
    if not merged:
        sid = int(seg[imax])
        items.append({"kind": "max_wss", "label": f"全场最大 WSS（{_branch_name(branch_names, sid)}）", "branch": _branch_name(branch_names, sid),
                      "segment_id": sid, "value": float(wss[imax]), "units": "Pa", "xyz_mm": [float(v) for v in pts[imax]],
                      "s_from_root_mm": _finite(s_root[imax]), "extent_mm": float(max(2.0, 2.0 * radius[imax])),
                      "area_mm2": None, "n_points": 1, "severity": "note",
                      "definition": "预测点云中的单点最大值，对噪声敏感，仅作参考；空间 p99 是主指标；不在任何已列出的高值簇内"})
    low_clusters = _clusters(pts, wss < low, link, min_points=min_cluster_points)
    low_total_fraction = float(sum(len(c) for c in low_clusters) / n)
    for cluster in _area_listed(low_clusters, area_per_point, floor, "low_wss_cluster", suppressed)[:max_clusters]:
        centre = pts[cluster].mean(axis=0)
        rep = cluster[int(np.argmin(np.linalg.norm(pts[cluster] - centre, axis=1)))]
        sid = int(seg[rep])
        items.append({"kind": "low_wss_cluster", "label": f"{_branch_name(branch_names, sid)}低 WSS 区", "branch": _branch_name(branch_names, sid),
                      "segment_id": sid, "value": float(wss[cluster].mean()), "units": "Pa", "xyz_mm": [float(v) for v in pts[rep]],
                      "s_from_root_mm": _finite(s_root[rep]), "extent_mm": _extent(pts[cluster], pts[rep]),
                      "area_mm2": float(len(cluster) * area_per_point), "n_points": int(len(cluster)),
                      "severity": "attention" if low_total_fraction > 0.2 else "note",
                      "cluster_min_pa": float(wss[cluster].min()),
                      "definition": f"WSS < {low:g} Pa 的连通簇（连接半径 {link:.2f} mm，≥ {min_cluster_points} 点，估计面积 ≥ {floor / 100.0:g} cm²）；value 为簇内均值，位置取簇质心最近点；面积按点占比估计",
                      "point_indices": _subsample(cluster)})
    items.extend(_geometry_findings(atlas, geometry_table, branch_names, start_id=len(items) + 1,
                                    morphology=morphology))
    if cycle is not None:
        items.extend(_cycle_findings(pts, cycle, seg, s_root, link=link, min_points=min_cluster_points,
                                     area_per_point=area_per_point, branch_names=branch_names,
                                     area_floor_mm2=floor, suppressed=suppressed))
    out = {"schema_version": FINDINGS_SCHEMA, "family": "wall",
           "thresholds_pa": [low, high, very_high], "p99_threshold_pa": p99,
           "low_wss_total_fraction": low_total_fraction, "connectivity_radius_mm": link,
           "listing_rules": {"area_floor_mm2": floor, "area_kinds": list(AREA_KINDS),
                             "high_min_points": int(min_cluster_points), "global_max_merged": merged,
                             "suppressed": suppressed,
                             "note": "面积型簇估计面积小于 1 cm² 的不列（统计表保留）；全场最大值并入包含它的高值簇"},
           "items": _number_items(items)}
    if cycle is not None:
        from .cycle_fields import OSI_THRESHOLDS, STAGNATION
        out["cycle_criteria"] = {"stagnation": dict(STAGNATION), "high_osi_gt": float(OSI_THRESHOLDS[2]),
                                 "max_clusters": CYCLE_MAX_CLUSTERS}
    return grade_high_findings(out, reference_assessment)


HIGH_GRADED_KINDS = ("high_wss_cluster", "max_wss")


def grade_high_findings(findings: dict, reference_assessment: Mapping[str, Any] | None) -> dict:
    """U11: severity of the high-WSS items from the release's same-protocol cohort reference (in place).

    * reference present (``reference_assessment.population.status == "pass"``) and the case p99 reaches its
      90th percentile → ``attention``; below it → ``note``;
    * no such reference → ``note`` with ``"grading": "no_reference"`` (无队列参照，不定级).

    The fixed 7 Pa level no longer grades anything; low-value items are left untouched.  Idempotent, so the
    pipeline can grade after the reference assessment exists and a rebuild can re-grade a stored list.
    """
    from .reference import population_p90
    if not isinstance(findings, dict):
        return findings
    population = reference_assessment.get("population") if isinstance(reference_assessment, Mapping) else None
    p90 = population_p90(population)
    case_p99 = _finite((population or {}).get("value_pa")) if p90 is not None else None
    if case_p99 is None:
        case_p99 = _finite(findings.get("p99_threshold_pa"))
    graded = p90 is not None and case_p99 is not None
    severity = ("attention" if case_p99 >= p90 else "note") if graded else "note"
    for item in findings.get("items") or []:
        if isinstance(item, dict) and item.get("kind") in HIGH_GRADED_KINDS:
            item["severity"] = severity
            item["grading"] = "reference_p90" if graded else "no_reference"
    findings["high_grading"] = {
        "method": "case_p99_vs_reference_p90", "quantile": HIGH_GRADING_QUANTILE,
        "status": "graded" if graded else "no_reference", "case_p99_pa": case_p99,
        "reference_p90_pa": p90 if graded else None,
        "reference_count": (population or {}).get("reference_count") if graded else None,
        "severity": severity,
        "note": ("本例 p99 与同口径队列参照的第 90 百分位比较：达到为关注，否则为提示" if graded
                 else "无队列参照，不定级（高值项一律为提示）")}
    return findings


# ----------------------------------------------------------------------------- findings review across rule changes
def _item_distance(a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
    xa, xb = a.get("xyz_mm"), b.get("xyz_mm")
    if not (isinstance(xa, (list, tuple)) and isinstance(xb, (list, tuple)) and len(xa) == 3 and len(xb) == 3):
        return float("inf")
    try:
        d = float(np.linalg.norm(np.asarray(xa, dtype=np.float64) - np.asarray(xb, dtype=np.float64)))
    except (TypeError, ValueError):
        return float("inf")
    return d if math.isfinite(d) else float("inf")


def _same_segment(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    sa, sb = a.get("segment_id"), b.get("segment_id")
    return sa is None or sb is None or sa == sb


def _match(old: Mapping[str, Any], new_items: list[Mapping[str, Any]], taken: set, *, cross_kind: bool,
           tolerance_mm: float) -> str | None:
    """New id for ``old`` by kind + position (same segment, nearest within ``tolerance_mm``), or None."""
    kind = old.get("kind")
    best, best_d = None, float("inf")
    for item in new_items:
        ident = str(item.get("id"))
        if ident in taken:
            continue
        if cross_kind:
            # the old single-point maximum now lives inside the high-WSS cluster that contains it (U12)
            ok = kind == "max_wss" and item.get("kind") == "high_wss_cluster" and item.get("contains_global_max")
        else:
            ok = item.get("kind") == kind
        if not ok or not _same_segment(old, item):
            continue
        d = _item_distance(old, item)
        if d <= tolerance_mm and d < best_d:
            best, best_d = ident, d
    return best


def remap_review(review: Mapping[str, Any] | None, old_items: list | None, new_items: list | None, *,
                 from_version: str | None = None, to_version: str | None = None,
                 tolerance_mm: float = REVIEW_MATCH_MM) -> tuple[dict | None, dict]:
    """Carry reviewer decisions (``findings_review``, keyed by finding id) onto a re-derived findings list.

    The review sidecar stores ``items = {finding_id: {decision, note}}``.  When the listing rules change the
    ids are renumbered, so a decision keyed by the old id could land on another finding.  Each decided old id
    is therefore matched to the new list by **kind + position** (same segment, nearest item within
    ``tolerance_mm``; the old single-point maximum may match the high-WSS cluster that now contains it).
    Matched decisions move to the new id; unmatched ones are kept under ``legacy`` with the old finding's
    kind / label / position and the label 「规则更新前的判定」, and are never applied to a new item.

    Returns ``(review, report)``; ``review`` is the input unchanged (same object) when nothing moved.
    """
    report = {"moved": {}, "legacy": [], "kept": []}
    if not isinstance(review, Mapping):
        return review, report
    decisions = review.get("items") if isinstance(review.get("items"), Mapping) else {}
    if not decisions:
        return review, report
    old_by_id = {str(item.get("id")): item for item in (old_items or []) if isinstance(item, Mapping)}
    new_list = [item for item in (new_items or []) if isinstance(item, Mapping)]
    new_by_id = {str(item.get("id")): item for item in new_list}
    mapping: dict[str, str | None] = {}
    taken: set = set()
    # Same-kind matches first (exact positions win), then the merged maximum.
    for cross in (False, True):
        for ident in decisions:
            if ident in mapping and mapping[ident] is not None:
                continue
            old = old_by_id.get(str(ident))
            if old is None:
                mapping[ident] = None
                continue
            target = _match(old, new_list, taken, cross_kind=cross, tolerance_mm=tolerance_mm)
            if target is not None:
                mapping[ident] = target
                taken.add(target)
            else:
                mapping.setdefault(ident, None)
    if all(mapping.get(ident) == ident for ident in decisions):
        report["kept"] = sorted(decisions)
        return review, report
    items, legacy = {}, [dict(entry) for entry in (review.get("legacy") or []) if isinstance(entry, Mapping)]
    for ident, decision in decisions.items():
        target = mapping.get(ident)
        entry = dict(decision) if isinstance(decision, Mapping) else {"decision": None, "note": ""}
        if target is not None:
            items[target] = entry
            if target != ident:
                report["moved"][ident] = target
            else:
                report["kept"].append(ident)
            continue
        old = old_by_id.get(str(ident)) or {}
        if not entry.get("decision") and not entry.get("note"):
            continue        # an undecided, un-annotated entry carries no information
        kind = old.get("kind")
        reason = ("旧发现在新规则下不再单列或已合并（未找到同类、同位置的发现）" if old
                  else "找不到这条判定对应的旧发现")
        if kind == "max_wss" and any(item.get("contains_global_max") for item in new_list):
            reason = "全场最大值已并入高值簇，该簇已有自己的判定"
        legacy.append({"id": str(ident), "decision": entry.get("decision"), "note": entry.get("note") or "",
                       "kind": kind, "label": old.get("label"), "branch": old.get("branch"),
                       "value": old.get("value"), "units": old.get("units"), "xyz_mm": old.get("xyz_mm"),
                       "s_from_root_mm": old.get("s_from_root_mm"), "status": "rule_update",
                       "status_label": LEGACY_REVIEW_LABEL, "reason": reason,
                       "analysis_version": from_version})
        report["legacy"].append(str(ident))
    out = dict(review)
    out["items"] = items
    out["legacy"] = legacy
    out["remap"] = {"from_analysis_version": from_version, "to_analysis_version": to_version,
                    "map": {str(k): v for k, v in mapping.items()},
                    "rule": f"按 kind + 位置匹配（同一分支、{tolerance_mm:g} mm 内最近）；匹配不上的保留为「{LEGACY_REVIEW_LABEL}」，不套用"}
    return out, report


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


# ----------------------------------------------------------------------------- zones (U10)
# Semantic branch key (paths.BRANCH_CN) -> zone id; the aorta is split further by the morphology block.
ZONE_OF_BRANCH = {"left_cia": "left_cia", "right_cia": "right_cia", "out-le": "left_eia", "out-li": "left_iia",
                  "out-re": "right_eia", "out-ri": "right_iia"}
ZONE_ORDER = ("aorta_proximal", "neck", "sac", "aorta_distal", "aorta",
              "left_cia", "left_eia", "left_iia", "right_cia", "right_eia", "right_iia")
ZONE_LABELS = {"aorta_proximal": ("瘤颈以上主动脉", "aorta above the neck"),
               "aorta_proximal_no_neck": ("瘤体以上主动脉", "aorta above the sac"),
               "neck": ("近端瘤颈", "proximal neck"), "sac": ("瘤体", "aneurysm sac"),
               "aorta_distal": ("瘤体以下主动脉", "aorta below the sac"), "aorta": ("主动脉", "aorta"),
               "left_cia": ("左髂总", "left common iliac"), "right_cia": ("右髂总", "right common iliac"),
               "left_eia": ("左髂外", "left external iliac"), "left_iia": ("左髂内", "left internal iliac"),
               "right_eia": ("右髂外", "right external iliac"), "right_iia": ("右髂内", "right internal iliac")}
ZONE_PAIRS = (("cia", "髂总", "common iliac", "left_cia", "right_cia"),
              ("eia", "髂外", "external iliac", "left_eia", "right_eia"),
              ("iia", "髂内", "internal iliac", "left_iia", "right_iia"))
ZONE_PRIMARY_ORDER = ("tawss", "osi", "wss")


def _zone_stats(field: str, values: np.ndarray, limits: Mapping[str, Sequence[float]]) -> dict:
    """Point-equal statistics of one field inside one zone (``limits[field]`` = (low, high) or (high,) for OSI)."""
    if field == "osi":
        (above,) = tuple(limits.get("osi") or (0.1,))[:1]
        return {"mean": _finite(values.mean()), "p90": _finite(np.quantile(values, .90)),
                "frac_high": _finite(np.mean(values > above))}
    low, high = tuple(limits.get(field) or (0.4, 4.0))[:2]
    if field == "tawss":
        return {"mean": _finite(values.mean()), "median": _finite(np.median(values)),
                "p10": _finite(np.quantile(values, .10)), "p90": _finite(np.quantile(values, .90)),
                "frac_low": _finite(np.mean(values < low)), "frac_high": _finite(np.mean(values > high))}
    return {"mean": _finite(values.mean()), "p99": _finite(np.quantile(values, .99)),
            "frac_low": _finite(np.mean(values < low)), "frac_high": _finite(np.mean(values > high))}


def _aorta_split(morphology: Mapping[str, Any] | None) -> tuple[list[tuple[str, float, float]], dict]:
    """Aorta sub-zones as (id, lower bound, upper bound) on ``s_from_root_mm`` from ``morphology.aorta``.

    Returns ``[]`` when there is no usable sac (→ a single ``aorta`` zone).  Bounds: proximal ``s < a``,
    neck ``a ≤ s < sac start`` (the shoulder between neck and sac joins the neck), sac ``start ≤ s ≤ end``,
    distal ``s > end``; the caller drops zones without points or length.
    """
    aorta = morphology.get("aorta") if isinstance(morphology, Mapping) else None
    sac = aorta.get("sac") if isinstance(aorta, Mapping) else None
    neck = aorta.get("neck") if isinstance(aorta, Mapping) else None
    if not (isinstance(sac, Mapping) and sac.get("present")):
        return [], {}
    s0, s1 = _finite(sac.get("s_start_mm")), _finite(sac.get("s_end_mm"))
    if s0 is None or s1 is None or s1 < s0:
        return [], {}
    info: dict[str, Any] = {}
    parts: list[tuple[str, float, float]] = []
    n0 = _finite(neck.get("s_start_mm")) if isinstance(neck, Mapping) and neck.get("present") else None
    if n0 is not None and n0 < s0:
        parts.append(("aorta_proximal", -math.inf, n0))
        parts.append(("neck", n0, s0))
        n1 = _finite(neck.get("s_end_mm"))
        if n1 is not None:
            info["neck_gap_merged_mm"] = _finite(max(0.0, s0 - n1))
    else:
        parts.append(("aorta_proximal", -math.inf, s0))
        info["neck_present"] = False
    parts.append(("sac", s0, s1))
    parts.append(("aorta_distal", s1, math.inf))
    return parts, info


def zones_wall(segment_id, s_from_root_mm, values_by_field: Mapping[str, Any], *,
               morphology: Mapping[str, Any] | None, branch_names: Mapping[str, str] | None,
               total_area_mm2: float | None, thresholds: Mapping[str, Sequence[float]] | None = None) -> dict | None:
    """Anatomical zone statistics of a wall result (U10, ``wss-deploy.zones/v1``, contract §5.1).

    Zones follow the confirmed branches (``branch_names`` → semantic keys of ``paths.BRANCH_CN``); the aorta
    is split into ``aorta_proximal`` / ``neck`` / ``sac`` / ``aorta_distal`` by the arc-length ranges of
    ``morphology.aorta.neck/sac`` (the shoulder between neck and sac joins the neck; zones with no length or
    no points are omitted), or kept whole as ``aorta`` without a sac.  Every prediction point counts once;
    the area is the point fraction × the input wall area (an estimate).  Only fields present in
    ``values_by_field`` (``wss`` / ``tawss`` / ``osi``) are summarised.  ``per_branch`` and every other block
    are untouched; nothing here changes a prediction.
    """
    from .paths import BRANCH_CN
    seg = np.asarray(segment_id).astype(int).reshape(-1)
    s = np.asarray(s_from_root_mm, dtype=np.float64).reshape(-1)
    n = len(seg)
    if s.shape != (n,) or n == 0:
        return None
    fields: dict[str, np.ndarray] = {}
    for key in ZONE_PRIMARY_ORDER:
        value = values_by_field.get(key) if isinstance(values_by_field, Mapping) else None
        if value is None:
            continue
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
        if arr.shape != (n,):
            raise ValueError(f"values_by_field[{key!r}] must have one value per point")
        fields[key] = arr
    if not fields:
        return None
    from .cycle_fields import OSI_THRESHOLDS, TAWSS_THRESHOLDS_PA
    limits = {"wss": (0.4, 4.0), "tawss": tuple(TAWSS_THRESHOLDS_PA[:2]), "osi": (float(OSI_THRESHOLDS[0]),)}
    for key, value in (thresholds or {}).items():
        if isinstance(value, (list, tuple)) and value:
            limits[key] = tuple(float(v) for v in value)
    area = _finite(total_area_mm2)
    area_per_point = area / n if area is not None and area > 0 else None
    key_of_name = {cn: key for key, cn in BRANCH_CN.items()}
    by_zone: dict[str, dict] = {}
    notes: list[str] = []

    def add(zone_id: str, sid: int, branch_key: str, mask: np.ndarray, s_range, label_key: str | None = None) -> None:
        count = int(mask.sum())
        if count == 0:
            return
        lo, hi = s_range
        if hi is not None and lo is not None and not (hi > lo):
            return
        zh, en = ZONE_LABELS[label_key or zone_id]
        by_zone[zone_id] = {
            "id": zone_id, "label": zh, "label_en": en, "segment_id": int(sid), "branch_key": branch_key,
            "s_range_mm": [_finite(lo), _finite(hi)], "n_points": count,
            "area_mm2": _finite(count * area_per_point) if area_per_point is not None else None,
            "fields": {key: _zone_stats(key, arr[mask], limits) for key, arr in fields.items()}}

    names = {}
    for sid_text, name in (branch_names or {}).items():
        try:
            names[int(sid_text)] = str(name)
        except (TypeError, ValueError):
            continue
    if not names:
        notes.append("缺少分支命名，未能划分解剖分区")
    for sid, name in sorted(names.items()):
        key = key_of_name.get(name)
        on = seg == sid
        if not on.any() or key is None:
            continue
        s_on = s[on]
        if key == "root":
            parts, info = _aorta_split(morphology)
            if not parts:
                add("aorta", sid, key, on, (float(s_on.min()), float(s_on.max())))
                continue
            first, last = float(s_on.min()), float(s_on.max())
            for zone_id, lo, hi in parts:
                if zone_id == "sac":
                    mask = on & (s >= lo) & (s <= hi)
                elif zone_id == "aorta_distal":
                    mask = on & (s > lo)
                else:
                    mask = on & (s >= lo) & (s < hi)
                lo_r = first if not math.isfinite(lo) else lo
                hi_r = last if not math.isfinite(hi) else hi
                label = "aorta_proximal_no_neck" if zone_id == "aorta_proximal" and info.get("neck_present") is False else None
                add(zone_id, sid, key, mask, (lo_r, hi_r), label)
            if "neck" in by_zone and info.get("neck_gap_merged_mm"):
                by_zone["neck"]["gap_merged_mm"] = info["neck_gap_merged_mm"]
        else:
            zone_id = ZONE_OF_BRANCH.get(key)
            if zone_id:
                add(zone_id, sid, key, on, (float(s_on.min()), float(s_on.max())))
    zones = [by_zone[z] for z in ZONE_ORDER if z in by_zone]
    pairs = []
    for pair_id, zh, en, left, right in ZONE_PAIRS:
        if left not in by_zone or right not in by_zone:
            continue
        entry = {"id": pair_id, "label": zh, "label_en": en, "left": left, "right": right, "fields": {}}
        for key in fields:
            lm, rm = by_zone[left]["fields"][key]["mean"], by_zone[right]["fields"][key]["mean"]
            ratio = _finite(lm / rm) if lm is not None and rm not in (None, 0.0) else None
            entry["fields"][key] = {"left_mean": lm, "right_mean": rm, "ratio_left_over_right": ratio}
        pairs.append(entry)
    primary = [key for key in ZONE_PRIMARY_ORDER if key in fields]
    low_wss, high_wss = limits["wss"][:2]
    low_t, high_t = limits["tawss"][:2]
    return {"schema_version": ZONES_SCHEMA,
            "definition": "按中心线分支与弧长划分解剖分区，区内预测点等权统计；面积 = 点占比 × 输入壁面面积（估计）",
            "area_method": "point_fraction_times_input_surface_area",
            "primary_fields": primary,
            "thresholds": {"wss": {"low_lt": low_wss, "high_gt": high_wss}, "tawss": {"low_lt": low_t, "high_gt": high_t},
                           "osi": {"high_gt": limits["osi"][0]}},
            "aorta_split": ("由 morphology.aorta 的瘤颈 / 瘤体弧长范围（s_from_root_mm）划分；瘤颈与瘤体之间的肩部并入瘤颈；"
                            "长度或点数为 0 的分区不列；无瘤时主动脉为一个分区"),
            "zones": zones, "pairs": pairs, "notes": notes}


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


__all__ = ["AREA_FLOOR_MM2", "AREA_KINDS", "BRANCH_ORDER", "CYCLE_MAX_CLUSTERS", "FINDINGS_SCHEMA", "LEGACY_REVIEW_LABEL",
           "MAX_DIAMETER_DEFINITION", "PROFILES_SCHEMA", "TRUST_BITS", "TRUST_SCHEMA", "ZONES_SCHEMA",
           "apply_morphology", "findings_volume", "findings_wall", "frame_transform", "grade_high_findings", "profiles",
           "remap_review", "review_segments", "trust_volume", "trust_wall", "zones_wall"]
