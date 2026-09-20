#!/usr/bin/env python3
"""Export X5D_v51 best/median/worst wall packages from finished caches; no inference."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial import cKDTree

OUT = Path(__file__).resolve().parent
ROOT = next(p for p in OUT.parents if (p / "training_wss_min").is_dir())
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(OUT))
from prepare_selected_postview import (  # noqa: E402
    dump, gaussian, loadnp, sha, stats, write_csv, write_vtp,
)
from prepare_velwss2_postview import resolve_stl  # noqa: E402
from tools.cfdpost_cloud_export.map_to_stl_surface import _load_stl_mesh  # noqa: E402
from tools.cfdpost_cloud_export.display_fields import (  # noqa: E402
    fit_selfmax, normalization_field_data, selfmax_fields,
)

DATA = ROOT / "data_wss_v5/views_v5_1/wss_min_view_v1"
RUN = ROOT / "training_wss_min/runs/wss_v51_wave1_20260916/X5D_v51_s1234"
PRED = RUN / "eval/ckpt_best/predictions/test"
MODEL = "X5D_v51"
ROLES = ("best", "median", "worst")


def geometry(cid, bundle):
    stl = resolve_stl(cid)
    vertices, triangles = _load_stl_mesh(stl)
    wall = bundle["wall_coords_aligned_mm"].astype(float)
    raw = bundle["wall_coords_raw"].astype(float)
    distance, index = cKDTree(raw).query(vertices)
    bijective = bool(distance.max() < 1e-3 and len(np.unique(index)) == len(raw) == len(vertices))
    if not bijective:
        assert distance.max() < 3.0, (cid, float(distance.max()))
        # STL may be denser or miss some CFD nodes as nearest neighbours; Gaussian still maps CFD → STL.
    rotation = bundle["transform_rotation"].astype(float)
    centroid = bundle["transform_centroid"].astype(float)
    aligned = (vertices - centroid) @ rotation.T
    reconstruction = np.abs((raw - centroid) @ rotation.T - wall).max()
    assert reconstruction < 1e-3 and np.linalg.det(rotation) > 0
    aligned_error = float(np.max(np.linalg.norm(aligned - wall[index], axis=1)))
    assert aligned_error < (1e-3 if bijective else 3.0)
    bbox_ratio = float(np.linalg.norm(np.ptp(raw, axis=0)) / np.linalg.norm(np.ptp(vertices, axis=0)))
    assert abs(bbox_ratio - 1.0) < 1e-3, (cid, bbox_ratio)
    report = json.loads((DATA / cid / "view_report.json").read_text())
    h5_path = Path(report["source"]["case_h5"])
    with h5py.File(h5_path) as handle:
        valid = handle["wall_static/valid"][()].astype(bool)
        xyz = handle["wall_static/xyz_mm"][()][valid]
        ids = handle["wall_static/node_id_cas"][()][valid]
        tri = handle["topology/wall_triangles"][()].astype(np.int64)
        normals = handle["wall_static/normal_out_mesh"][()][valid] @ rotation.T
    np.testing.assert_array_equal(ids, bundle["wall_node_id_cas"])
    assert np.max(np.abs(xyz - raw)) < 1e-3
    remap = np.full(len(valid), -1, np.int64)
    remap[valid] = np.arange(valid.sum())
    tri = remap[tri]
    keep = np.all(tri >= 0, axis=1)
    tri = tri[keep]
    assert len(tri) > 0 and tri.min() >= 0 and tri.max() < len(wall)
    return aligned, triangles, tri, index, normals, {
        "stl_source": str(stl),
        "stl_sha256": sha(stl),
        "snapshot_source": str(h5_path),
        "stl_to_mm_scale": 1.0,
        "scale_source": (
            "identity units verified by bijective STL-to-CFD wall mapping; bbox not fitted"
            if bijective else
            "identity millimetre units verified by matching bounding-box diagonal; STL is denser than CFD wall, bbox not fitted"
        ),
        "stl_cfd_identity": bijective,
        "bbox_ratio_qc_only": bbox_ratio,
        "rotation_convention": "(raw_mm - centroid) @ rotation.T",
        "rotation_determinant": float(np.linalg.det(rotation)),
        "centroid_mm": centroid.tolist(),
        "rotation": rotation.tolist(),
        "reconstruction_max_abs_mm": float(reconstruction),
        "stl_wall_identity_max_distance_mm": float(distance.max()),
        "aligned_max_distance_mm": aligned_error,
        "bijective_node_count": int(len(np.unique(index))),
        "n_stl_vertices": len(vertices),
        "n_stl_triangles": len(triangles),
        "n_cfd_triangles": len(tri),
        "triangles_dropped_invalid": int((~keep).sum()),
        "wall_crop_applied": bool(bundle["wall_crop_applied"]),
        "frame_version": str(bundle["transform_frame_version"]),
        "coordinate_unit": "mm",
        "tolerance_mm": 1e-3 if bijective else 3.0,
    }


def build(role: str, cid: str, selection: dict) -> dict:
    dest = OUT / MODEL / role
    dest.mkdir(parents=True, exist_ok=True)
    pred_path = PRED / cid / "predictions.npz"
    bundle_path = DATA / cid / "bundle.npz"
    metrics_path = RUN / "eval/ckpt_best/metrics.json"
    manifest_path = RUN / "eval/ckpt_best/predictions/manifest.json"
    bundle = loadnp(bundle_path)
    pred = loadnp(pred_path)
    peak = int(bundle["peak_step"])
    peak_index = int(np.flatnonzero(bundle["steps"] == peak)[0])
    assert peak == 1162
    config = json.loads((RUN / "config.json").read_text())
    assert config["train"]["seed"] == 1234
    assert "density_aug_levels" in config["data"]
    official = json.loads(metrics_path.read_text())["test"]["per_case"][cid]["overall"]
    pred_manifest = json.loads(manifest_path.read_text())
    assert pred_manifest["checkpoint"] == "best"
    for key, path in [("checkpoint_sha256", RUN / "ckpt_best.pt"), ("config_sha256", RUN / "config.json")]:
        if key in pred_manifest:
            assert sha(path) == pred_manifest[key], key
    wall = bundle["wall_coords_aligned_mm"].astype(float)
    index = pred["row_index"].astype(np.int64)
    assert np.array_equal(np.sort(index), np.arange(len(wall)))
    xyz = wall[index]
    true = pred["true_pa"].astype(float)
    pred_pa = pred["pred_pa"].astype(float)
    np.testing.assert_array_equal(true, bundle["wall_wss"][peak_index, index])
    assert np.isfinite(pred_pa).all() and pred_pa.min() >= 0
    point_arrays = {
        "wss_cfd_pa": true,
        "wss_pred_pa": pred_pa,
        "wss_error_pred_minus_cfd_pa": pred_pa - true,
        "wss_abs_error_pa": np.abs(pred_pa - true),
        "wss_cfd_norm": pred["true_norm"].astype(float),
        "wss_pred_norm": pred["pred_norm"].astype(float),
        "row_index": index,
    }
    cache_metrics = stats(true, pred_pa)
    assert abs(cache_metrics["r2"] - official["r2"]) < 1e-8, (cache_metrics, official)
    assert abs(cache_metrics["mae"] - official["mae"]) < 1e-6
    stl_xyz, stl_tri, cfd_tri, wall_idx, normals, geom = geometry(cid, bundle)
    order = np.argsort(index)
    source_xyz = xyz[order]
    native = {key: value[order] if np.ndim(value) else value for key, value in point_arrays.items()}
    values = np.column_stack([native["wss_cfd_pa"], native["wss_pred_pa"]])
    prefix, scope, unit = "wss", "wall", "Pa"
    normalization = fit_selfmax(true, pred_pa, prefix, scope, unit)
    field_data = normalization_field_data(normalization)
    point_arrays.update(selfmax_fields(true, pred_pa, normalization))
    native.update(selfmax_fields(native["wss_cfd_pa"], native["wss_pred_pa"], normalization))
    description = (
        f"{MODEL} {cid}; test; seed1234; ckpt_best; peak{peak}; atlas mm; "
        "cached same-point prediction from finished v5.1 X5D_v51"
    )
    pointname = f"{MODEL}__{role}__{cid.replace('/', '__')}__wall_wss.vtp"
    write_vtp(dest / pointname, xyz, point_arrays, description=description, field_data=field_data)
    write_csv(dest / "_export" / "same_point_fields.csv.gz", xyz, point_arrays)
    write_vtp(dest / "aligned_geometry.vtp", stl_xyz, {}, stl_tri, description="Aligned STL geometry; no field interpolation")
    write_vtp(
        dest / "cfd_wall_native.vtp", wall, native, cfd_tri,
        description=description + "; native CFD triangles; no smoothing",
        field_data=field_data,
    )
    mapped, extra, report = gaussian(source_xyz, values, stl_xyz, wall_idx, normals, cfd_tri)
    cfkey, prkey, erkey = "wss_cfd_pa", "wss_pred_pa", "wss_error_pred_minus_cfd_pa"
    surface = {
        cfkey: mapped[:, 0],
        prkey: mapped[:, 1],
        erkey: mapped[:, 1] - mapped[:, 0],
        erkey.replace("error", "abs_error"): np.abs(mapped[:, 1] - mapped[:, 0]),
        **extra,
    }
    surface.update(selfmax_fields(surface[cfkey], surface[prkey], normalization))
    write_vtp(
        dest / "surface_gaussian.vtp", stl_xyz, surface, stl_tri,
        description=description + "; Wall scalar interpolated to original STL vertices",
        field_data=field_data,
    )
    write_csv(dest / "surface_gaussian" / "mapped_vertices.csv", stl_xyz, surface)
    source_arrays = {cfkey: values[:, 0], prkey: values[:, 1], erkey: values[:, 1] - values[:, 0],
                     "wss_abs_error_pa": np.abs(values[:, 1] - values[:, 0])}
    source_arrays.update(selfmax_fields(values[:, 0], values[:, 1], normalization))
    write_csv(dest / "_export" / "gaussian_source.csv.gz", source_xyz, source_arrays)
    assert report["coverage"]["valid_ratio"] == 1.0
    report.update(
        purpose="Wall scalar interpolated to original STL vertices",
        source_points=len(source_xyz),
        source_csv="_export/gaussian_source.csv.gz",
        target_geometry="aligned_geometry.vtp",
        geometry=geom, units=unit,
        topology={"triangles": len(stl_tri), "valid_triangles": int(np.all(extra["map_valid"][stl_tri], axis=1).sum())},
        errors="signed error = interpolated Pred - interpolated CFD; absolute = abs(signed)",
        display_normalization=normalization,
    )
    dump(dest / "mapping_report.json", report)
    sources = {
        str(pred_path): sha(pred_path),
        str(bundle_path): sha(bundle_path),
        str(metrics_path): sha(metrics_path),
        str(RUN / "config.json"): sha(RUN / "config.json"),
        str(RUN / "ckpt_best.pt"): sha(RUN / "ckpt_best.pt"),
        str(manifest_path): sha(manifest_path),
        str(OUT / "selection.json"): sha(OUT / "selection.json"),
    }
    visual = dict(
        official, seed=1234, metric_source=str(metrics_path),
        metric_key=f"test.per_case[{cid}].overall",
        basis="whole evaluated case, not smoothed surface",
        cache_recomputed=cache_metrics,
    )
    chosen = selection[role]
    manifest = {
        "schema_version": 3,
        "model": MODEL,
        "role": role,
        "case_id": cid,
        "run": str(RUN),
        "checkpoint": "ckpt_best.pt",
        "seed": 1234,
        "split": "test",
        "peak_step": peak,
        "peak_index": peak_index,
        "frame": str(bundle["transform_frame_version"]),
        "coordinate_unit": "mm",
        "unit": unit,
        "selection": {
            "basis": "fixed five-seed mean of per-case R2 (1234,7,2025,11,2026); not ensemble; not X5D_long",
            "r2_case_mean": chosen["r2"],
            "r2_sd": chosen["r2_sd"],
            "seed_values": chosen["seed_values"],
            "statistical_median": selection["median"]["statistical_median"],
            "median_rule": selection["median"]["rule"],
        },
        "visualization_metrics": visual,
        "pressure_reference": None,
        "geometry": geom,
        "source_sha256": sources,
        "display_normalization": normalization,
        "verification": {
            "truth_vs_mother_data": "exact",
            "point_index_and_coordinate_identity": "exact",
            "r2_matches_official_abs_tol": 1e-8,
            "mae_matches_official_abs_tol": 1e-6,
            "metric_tolerance_reason": "saved float32 raw cache versus evaluator precision",
            "written_vtp_all_arrays_roundtrip": "exact",
        },
        "files": {
            "pointcloud": pointname,
            "gaussian_surface": "surface_gaussian.vtp",
            "mapping_report": "mapping_report.json",
            "native_cfd_wall": "cfd_wall_native.vtp",
            "aligned_geometry": "aligned_geometry.vtp",
            "original_points_csv": "_export/same_point_fields.csv.gz",
        },
        "surface_display_purpose": "Wall scalar interpolated to original STL vertices",
    }
    manifest["output_sha256"] = {
        key: sha(dest / key)
        for key in [pointname, "surface_gaussian.vtp", "aligned_geometry.vtp", "cfd_wall_native.vtp"]
    }
    dump(dest / "manifest.json", manifest)
    cmax = "invalid" if normalization["cfd_max"] is None else f"{normalization['cfd_max']:.12g}"
    pmax = "invalid" if normalization["pred_max"] is None else f"{normalization['pred_max']:.12g}"
    (dest / "README_打开说明.md").write_text(
        f"# {MODEL} / {role} / {cid}\n\n"
        "ParaView 打开 **surface_gaussian.vtp**，点击 Apply，Representation 选 Surface。\n\n"
        "CFD：`wss_cfd_pa`；Pred：`wss_pred_pa`；有符号误差：`wss_error_pred_minus_cfd_pa`（Pa）。"
        "CFD/Pred 使用相同色标，误差用对称色标。\n\n"
        f"选例 R²（五 seed 均值）：{chosen['r2']:.6f}；当前 s1234 全壁面 R²：{official['r2']:.6f}。"
        f"peak {peak}；v5_atlas_frame_v1；坐标 mm。v5.1 数据更新后重训完成的 X5D_v51，不是纵向几何 X5D_long。\n\n"
        f"壁面 Gaussian 插值：r=3 mm，sharpness=2，max_dist=3 mm，fallback=mask。"
        f"覆盖率 {report['coverage']['valid_ratio']:.3%}。面片内含 `map_valid`、`map_dist_mm`；"
        "配准和跨壁诊断见 `mapping_report.json`。\n\n"
        "`cfd_wall_native.vtp` 保留原生 CFD 三角面上的同点场值，无高斯平滑；用于核对局部热点。"
        f"原点云：`{pointname}`。\n\n"
        "自身最大值归一化：`wss_cfd_selfmax` = CFD / 原始 CFD max；`wss_pred_selfmax` = Pred / 原始 Pred max；"
        "差值 `wss_selfmax_error_pred_minus_cfd`，绝对差 `wss_selfmax_abs_error`。"
        "均无量纲，仅作空间分布对照，幅值差已移除。\n\n"
        f"原始同点域 `{scope}`：CFD max = {cmax} {unit}，Pred max = {pmax} {unit}。"
        "点云/原生面/Gaussian 共享此对分母，插值后不再重求最大值。\n"
    )
    print(json.dumps({
        "model": MODEL, "role": role, "case": cid, "points": len(xyz),
        "surface_triangles": len(stl_tri), "coverage": report["coverage"]["valid_ratio"],
        "s1234_r2": official["r2"],
    }, ensure_ascii=False), flush=True)
    return manifest


def write_batch(cases: list[dict]) -> None:
    dest = OUT / MODEL
    dump(dest / "manifest.json", {
        "schema_version": 3,
        "model": MODEL,
        "note": "Finished v5.1 X5D_v51 caches; package paths relative to each case",
        "selection_source": str(dest / "selection.json"),
        "selection_sha256": sha(dest / "selection.json"),
        "cases": cases,
    })
    with (dest / "打开文件清单.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "role", "case_id", "s1234_whole_case_r2", "five_seed_selection_r2",
                         "surface_file", "original_points", "usage"])
        for case in cases:
            rel = Path(case["role"])
            writer.writerow([
                case["model"], case["role"], case["case_id"],
                case["visualization_metrics"]["r2"], case["selection"]["r2_case_mean"],
                str(rel / case["files"]["gaussian_surface"]),
                str(rel / case["files"]["pointcloud"]),
                case["surface_display_purpose"],
            ])
    with (dest / "归一化分母清单.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "role", "case_id", "seed", "peak_step", "denominator_scope",
                         "source_unit", "cfd_max", "pred_max", "cfd_selfmax_field",
                         "pred_selfmax_field", "cfd_valid", "pred_valid"])
        for case in cases:
            norm = case["display_normalization"]
            writer.writerow([
                case["model"], case["role"], case["case_id"], case["seed"], case["peak_step"],
                norm["denominator_scope"], norm["source_unit"], norm["cfd_max"], norm["pred_max"],
                "wss_cfd_selfmax", "wss_pred_selfmax",
                norm["cfd_denominator"]["valid"], norm["pred_denominator"]["valid"],
            ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roles", nargs="+", default=list(ROLES), choices=ROLES)
    args = parser.parse_args()
    selection = json.loads((OUT / MODEL / "selection.json").read_text())
    for role in args.roles:
        build(role, selection[role]["case_id"], selection)
    existing = []
    for role in ROLES:
        path = OUT / MODEL / role / "manifest.json"
        if path.exists():
            existing.append(json.loads(path.read_text()))
    if existing:
        write_batch(existing)


if __name__ == "__main__":
    main()
