"""Stage 4a: report metrics from the prediction + geometry only (no CFD, no truth)."""
from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from .paths import BRANCH_CN

LOW_PA, HIGH_PA, VERY_HIGH_PA = 0.4, 4.0, 7.0


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
    tab, cols = atlas.table, list(atlas.columns); col = lambda c: tab[:, cols.index(c)]
    geo = {}
    for s in atlas.segments:
        sid = int(s["segment_id"]); mk = col("segment_id").astype(int) == sid
        if mk.sum() < 3: continue
        r = col("radius_mm")[mk]; xyz = np.stack([col("x_mm")[mk], col("y_mm")[mk], col("z_mm")[mk]], 1)
        chord = float(np.linalg.norm(xyz[-1] - xyz[0])); L = float(s.get("length_mm", 0.0))
        geo[names.get(sid, str(sid))] = {"length_mm": L, "radius_min_mm": float(r.min()), "radius_median_mm": float(np.median(r)), "radius_max_mm": float(r.max()),
            "max_diameter_mm": float(2 * r.max()), "stenosis_index": float(1 - r.min() / np.median(r)), "tortuosity": float(L / chord) if chord > 0 else None, "curvature_max_per_mm": float(col("curvature_per_mm")[mk].max())}
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
