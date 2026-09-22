"""Stage 4a: report metrics from the prediction + geometry only (no CFD, no truth)."""
from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from .paths import BRANCH_CN

LOW_PA, HIGH_PA, VERY_HIGH_PA = 0.4, 4.0, 7.0


def surface_metrics(vertices, faces, values, *, covered=None, face_labels=None,
                    top_fraction: float = .01) -> dict:
    """Area-weighted statistics on the displayed/interpolated wall surface.

    ``values`` are vertex values (normally the Gaussian-interpolated ``vw``),
    so this deliberately reports a separate *surface display* protocol. A
    triangle contributes only when all three vertex values are finite; partial
    triangles are counted as uncovered instead of silently inventing values.
    ``p99`` is the weighted value whose cumulative covered area reaches 99%.
    This is not the historical prediction-point-cloud p99.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    values = np.asarray(values, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("surface vertices must be a finite (N, 3) array")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise ValueError("surface faces must be a non-empty (M, 3) array")
    if np.any(faces < 0) or np.any(faces >= len(vertices)):
        raise ValueError("surface faces contain an out-of-range vertex index")
    if values.shape != (len(vertices),):
        raise ValueError("surface values must contain one value per vertex")
    if not np.isfinite(top_fraction) or not 0 < top_fraction <= 1:
        raise ValueError("top_fraction must be in (0, 1]")
    if covered is None:
        covered = np.isfinite(values)
    else:
        covered = np.asarray(covered, dtype=bool)
        if covered.shape != values.shape:
            raise ValueError("surface coverage mask must contain one value per vertex")
    finite = covered & np.isfinite(values)
    tri = vertices[faces]
    area = .5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    if not np.isfinite(area).all() or np.any(area <= 0):
        raise ValueError("surface contains non-positive or non-finite triangle areas")
    valid_face = finite[faces].all(axis=1)
    partial_face = finite[faces].any(axis=1) & ~valid_face
    valid_area = area[valid_face]
    face_values = values[faces[valid_face]].mean(axis=1)
    total_area = float(area.sum())
    effective_area = float(valid_area.sum())
    if effective_area <= 0:
        raise ValueError("surface has no fully covered finite triangles")
    if not np.isfinite(face_values).all():
        raise ValueError("surface face values are not finite")

    def weighted_quantile(q):
        order = np.argsort(face_values, kind="mergesort")
        sv, sa = face_values[order], valid_area[order]
        target = float(q) * effective_area
        return float(sv[np.searchsorted(np.cumsum(sa), target, side="left").clip(0, len(sv) - 1)])

    p99 = weighted_quantile(.99)
    top_threshold = weighted_quantile(1. - top_fraction)
    top = face_values >= top_threshold
    top_area = float(valid_area[top].sum())
    top_mean = float(np.average(face_values[top], weights=valid_area[top])) if top_area else None
    out = {
        "protocol": "gaussian_interpolated_wall_surface_area_weighted",
        "value_source": "vertex_gaussian_interpolation",
        "weight_source": "triangle_area_mm2",
        "p99_definition": "covered wall triangle area-weighted 99th percentile of triangle mean values",
        "top_area_definition": f"mean over highest {top_fraction:.4g} covered wall area (threshold ties may exceed target)",
        "n_vertices": int(len(vertices)), "n_faces": int(len(faces)),
        "n_valid_faces": int(valid_face.sum()), "n_partial_faces": int(partial_face.sum()),
        "total_area_mm2": total_area, "effective_area_mm2": effective_area,
        "covered_area_fraction": effective_area / total_area,
        "coverage_vertex_fraction": float(finite.mean()),
        "mean_pa": float(np.average(face_values, weights=valid_area)),
        "p95_pa": weighted_quantile(.95), "p99_pa": p99,
        "max_pa": float(face_values.max()), "top_threshold_pa": top_threshold,
        "top_area_mm2": top_area, "top_area_fraction": top_area / effective_area,
        "top_area_mean_pa": top_mean,
    }
    if face_labels is not None:
        labels = np.asarray(face_labels)
        if labels.shape != (len(faces),):
            raise ValueError("face_labels must contain one label per face")
        by_label = {}
        for label in np.unique(labels[valid_face]):
            mask = valid_face & (labels == label)
            label_area = area[mask]
            label_values = np.asarray(values[faces[mask]], dtype=np.float64).mean(axis=1)
            by_label[str(label)] = {"area_mm2": float(label_area.sum()),
                "area_fraction_of_covered": float(label_area.sum() / effective_area),
                "mean_pa": float(np.average(label_values, weights=label_area)),
                "p99_pa": float(np.quantile(label_values, .99)),
                "max_pa": float(label_values.max())}
        out["by_face_label"] = by_label
    return out


# Explicit name used by report/pipeline integration; keep the shorter alias
# available for callers that treat this as a generic surface field utility.
surface_statistics = surface_metrics


def branch_names(segments) -> dict[int, str]:
    out = {}
    for s in segments:
        sid = int(s["segment_id"])
        if s.get("starts_at_root"): out[sid] = BRANCH_CN["root"]
        elif s.get("ends_at_leaf"): out[sid] = BRANCH_CN.get(s.get("outlet_name", ""), f"出口{sid}")
        else:
            d = set(s.get("descendant_outlets", []))
            out[sid] = BRANCH_CN["left_cia"] if d <= {"out-le", "out-li"} else (BRANCH_CN["right_cia"] if d <= {"out-re", "out-ri"} else f"段{sid}")
    return out


def _stats(v):
    v = np.asarray(v, dtype=np.float64)
    if v.ndim != 1 or not v.size or not np.isfinite(v).all():
        raise ValueError("WSS must be a non-empty, finite one-dimensional field")
    return {"n": int(v.size), "mean": float(v.mean()), "median": float(np.median(v)), "p95": float(np.quantile(v, .95)), "p99": float(np.quantile(v, .99)), "max": float(v.max()), "min": float(v.min())}


def geometry_table(atlas, names: dict | None = None) -> dict:
    """Per-branch centreline geometry (length, radii, stenosis index, tortuosity, max curvature).

    Uses only the atlas; shared by the wall and volume families and by the findings list.
    """
    names = names or branch_names(atlas.segments)
    if hasattr(atlas, "table") and hasattr(atlas, "columns"):
        # Same access path as the original inline code (and as the minimal test stubs).
        tab, cols = atlas.table, list(atlas.columns)
        col = lambda c: tab[:, cols.index(c)]
        xyz_all = np.stack([col("x_mm"), col("y_mm"), col("z_mm")], 1)
    else:
        col = lambda c: np.asarray(atlas.col(c))
        xyz_all = np.asarray(atlas.xyz)
    seg_col = col("segment_id").astype(int)
    radius_col = col("radius_mm")
    curvature_col = col("curvature_per_mm")
    geo = {}
    for s in atlas.segments:
        sid = int(s["segment_id"]); mk = seg_col == sid
        if mk.sum() < 3: continue
        r = radius_col[mk]; xyz = xyz_all[mk]
        chord = float(np.linalg.norm(xyz[-1] - xyz[0])); L = float(s.get("length_mm", 0.0))
        geo[names.get(sid, str(sid))] = {"length_mm": L, "radius_min_mm": float(r.min()), "radius_median_mm": float(np.median(r)), "radius_max_mm": float(r.max()),
            "max_diameter_mm": float(2 * r.max()), "stenosis_index": float(1 - r.min() / np.median(r)), "tortuosity": float(L / chord) if chord > 0 else None, "curvature_max_per_mm": float(curvature_col[mk].max())}
    return geo


def compute(pts, wss, geom, atlas, diag, total_area_mm2: float, thresholds=(LOW_PA, HIGH_PA, VERY_HIGH_PA)) -> dict:
    pts = np.asarray(pts, dtype=np.float64)
    wss = np.asarray(wss, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) == 0 or not np.isfinite(pts).all():
        raise ValueError("Prediction coordinates must be a non-empty finite (N, 3) array")
    if wss.shape != (len(pts),) or not np.isfinite(wss).all():
        raise ValueError("WSS must contain one finite value per prediction point")
    if len(thresholds) != 3 or not np.isfinite(thresholds).all() or not 0 <= thresholds[0] < thresholds[1] < thresholds[2]:
        raise ValueError("WSS thresholds must be three finite, strictly increasing non-negative values")
    if not np.isfinite(total_area_mm2) or total_area_mm2 <= 0:
        raise ValueError("Surface area must be finite and positive")
    spacing = float(diag["spacing_mm"])
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("Hotspot connectivity spacing must be finite and positive")
    for key in ("segment_id", "s_from_root_mm", "dist_to_junction_mm", "radius_mm"):
        value = np.asarray(geom[key])
        if value.shape != (len(pts),) or not np.isfinite(value).all():
            raise ValueError(f"Geometry {key} must contain one finite value per prediction point")
    if np.any(np.asarray(geom["radius_mm"]) <= 0):
        raise ValueError("Local radii must be positive")
    low, high, vhigh = thresholds
    seg = np.asarray(geom["segment_id"]).astype(int); s_root = np.asarray(geom["s_from_root_mm"]); dj = np.asarray(geom["dist_to_junction_mm"]); rad = np.asarray(geom["radius_mm"])
    names = branch_names(atlas.segments); n = len(pts)
    thr10 = float(np.quantile(wss, .90)); thr5 = float(np.quantile(wss, .95)); top5 = wss >= thr5; ens10 = wss >= thr10
    tree = cKDTree(pts[top5]); pairs = tree.query_pairs(r=2.0 * spacing, output_type="ndarray"); m = int(top5.sum())
    if m == 0:
        sizes = np.asarray([], dtype=int)
    elif len(pairs) == 0:
        sizes = np.ones(m, dtype=int)
    else:
        g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(m, m)); _, lab = connected_components(g, directed=False)
        sizes = np.bincount(lab)
    imax = int(np.argmax(wss))
    branch = {}
    for sid in sorted(set(seg.tolist())):
        mk = seg == sid; v = wss[mk]
        if mk.sum() < 10: continue
        branch[names.get(sid, str(sid))] = {"segment_id": int(sid), "n_points": int(mk.sum()), "area_mm2": float(total_area_mm2 * mk.mean()),
            "wss_mean_pa": float(v.mean()), "wss_p99_pa": float(np.quantile(v, .99)), "wss_max_pa": float(v.max()),
            "frac_low": float(np.mean(v < low)), "frac_high": float(np.mean(v > high)), "share_of_top10": float(np.sum(mk & ens10) / max(1, np.sum(ens10))), "radius_mean_mm": float(rad[mk].mean())}
    geo = geometry_table(atlas, names)
    return {
        "statistics_protocol": {
            "field": "fixed_peak_systolic_frame", "support": "prediction_point_cloud", "quantile_method": "linear",
            "p99_definition": "固定收缩期帧预测点云的空间第 99 百分位，不是时间最大值",
            "position_definition": "位置和黄色标记对应全场最大值预测点，不对应 p99",
            "area_method": "point_fraction_times_input_surface_area", "area_label": "估计面积",
            "area_note": "点数占比乘输入壁面总面积，未做逐面片面积加权",
            "hotspot_definition": "WSS ≥ 空间 p99 的预测点；同值并列时可超过 1%",
            "top5_connectivity_radius_mm": 2.0 * spacing,
        },
        "wss_field_pa": {**_stats(wss), "top10_threshold_pa": thr10, "top5_threshold_pa": thr5, "top1_threshold_pa": float(np.quantile(wss, .99)),
                         "area_frac_low": float(np.mean(wss < low)), "area_frac_high": float(np.mean(wss > high)), "area_frac_very_high": float(np.mean(wss > vhigh)),
                         "area_low_mm2": float(total_area_mm2 * np.mean(wss < low)), "area_high_mm2": float(total_area_mm2 * np.mean(wss > high)), "thresholds_pa": [low, high, vhigh]},
        "peak": {"definition": "固定收缩期帧空间 p99；位置与标记对应全场最大值", "p99_pa": float(np.quantile(wss, .99)), "max_pa": float(wss[imax]),
                 "branch": names.get(int(seg[imax]), str(seg[imax])), "s_from_inlet_mm": float(s_root[imax]), "dist_to_junction_mm": float(dj[imax]), "local_radius_mm": float(rad[imax]),
                 "xyz_mm": pts[imax].round(2).tolist(), "index": imax, "top5_clusters_ge20": int(np.sum(sizes >= 20)), "top5_cluster_sizes": sorted(sizes.tolist(), reverse=True)[:5]},
        "per_branch": branch, "geometry": geo, "branch_names": {str(k): v for k, v in names.items()},
    }
