#!/usr/bin/env python3
"""Export VELWSS2 derived-WSS postview packs for best/median/worst; no inference."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = next(p for p in OUT.parents if (p / "training_wss_min").is_dir())
sys.path.insert(0, str(ROOT))
from tools.cfdpost_cloud_export.display_fields import (  # noqa: E402
    fit_selfmax,
    normalization_field_data,
    selfmax_fields,
)

from prepare_selected_postview import (  # noqa: E402
    dump,
    gaussian,
    loadnp,
    sha,
    stats,
    write_csv,
    write_vtp,
)
from tools.cfdpost_cloud_export.map_to_stl_surface import _load_stl_mesh  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
import h5py  # noqa: E402

DATA = ROOT / "data_wss_v5/views/wss_min_view_v1"
EXP = ROOT / "training_wss_min/experiments/vf6_velocity_to_wss_20260916"
SELECT_PATH = OUT.parent / "启发式v2_R2分布_20260914/B_VF6_wss_selection.json"
MODEL = "VF6_wss"
ROLES = ("best", "median", "worst")
VELOCITY_RUN = ROOT / "training_wss_min/runs/wss_local_wave2_20260912/VF6_s1234"


def resolve_stl(cid: str) -> Path:
    folder = ROOT / "data_new" / cid
    naive = folder / (cid.split("/")[-1] + ".stl")
    if naive.is_file():
        return naive
    stls = sorted(p for p in folder.glob("*.stl") if p.is_file())
    if not stls:
        raise FileNotFoundError(f"No STL in {folder}")
    if len(stls) == 1:
        return stls[0]
    preferred = [p for p in stls if p.name.endswith("-sq.stl") or p.stem == cid.split("/")[-2]]
    if len(preferred) == 1:
        return preferred[0]
    raise FileNotFoundError(f"Ambiguous STLs in {folder}: {[p.name for p in stls]}")


def geometry(cid, b):
    stl = resolve_stl(cid)
    v, t = _load_stl_mesh(stl)
    wall = b["wall_coords_aligned_mm"].astype(float)
    raw = b["wall_coords_raw"].astype(float)
    d, idx = cKDTree(raw).query(v)
    bijective = bool(d.max() < 1e-3 and len(np.unique(idx)) == len(raw) == len(v))
    if not bijective:
        assert d.max() < 3.0, (cid, float(d.max()))
        assert len(np.unique(idx)) == len(raw), (cid, len(np.unique(idx)), len(raw), len(v))
    R = b["transform_rotation"].astype(float)
    C = b["transform_centroid"].astype(float)
    aligned = (v - C) @ R.T
    derr = np.abs((raw - C) @ R.T - wall).max()
    assert derr < 1e-3 and np.linalg.det(R) > 0
    aligned_err = float(np.max(np.linalg.norm(aligned - wall[idx], axis=1)))
    if bijective:
        assert aligned_err < 1e-3
    else:
        assert aligned_err < 3.0
    bbox_ratio = float(np.linalg.norm(np.ptp(raw, axis=0)) / np.linalg.norm(np.ptp(v, axis=0)))
    assert abs(bbox_ratio - 1.0) < 1e-3, (cid, bbox_ratio)
    rep = json.loads((DATA / cid / "view_report.json").read_text())
    hpath = Path(rep["source"]["case_h5"])
    with h5py.File(hpath) as h:
        valid = h["wall_static/valid"][()].astype(bool)
        xyz = h["wall_static/xyz_mm"][()][valid]
        ids = h["wall_static/node_id_cas"][()][valid]
        tri = h["topology/wall_triangles"][()].astype(np.int64)
        normals = h["wall_static/normal_out_mesh"][()][valid] @ R.T
    np.testing.assert_array_equal(ids, b["wall_node_id_cas"])
    assert np.max(np.abs(xyz - raw)) < 1e-3
    remap = np.full(len(valid), -1, np.int64)
    remap[valid] = np.arange(valid.sum())
    tri = remap[tri]
    keep = np.all(tri >= 0, axis=1)
    tri = tri[keep]
    assert len(tri) > 0 and tri.min() >= 0 and tri.max() < len(wall)
    report = {
        "stl_source": str(stl),
        "stl_sha256": sha(stl),
        "snapshot_source": str(hpath),
        "stl_to_mm_scale": 1.0,
        "scale_source": (
            "identity units verified by bijective STL-to-CFD wall mapping; bbox not fitted"
            if bijective else
            "identity millimetre units verified by matching bounding-box diagonal; STL is denser than CFD wall, bbox not fitted"
        ),
        "stl_cfd_identity": bijective,
        "bbox_ratio_qc_only": bbox_ratio,
        "rotation_convention": "(raw_mm - centroid) @ rotation.T",
        "rotation_determinant": float(np.linalg.det(R)),
        "centroid_mm": C.tolist(),
        "rotation": R.tolist(),
        "reconstruction_max_abs_mm": float(derr),
        "stl_wall_identity_max_distance_mm": float(d.max()),
        "aligned_max_distance_mm": aligned_err,
        "bijective_node_count": int(len(np.unique(idx))),
        "n_stl_vertices": len(v),
        "n_stl_triangles": len(t),
        "n_cfd_triangles": len(tri),
        "triangles_dropped_invalid": int((~keep).sum()),
        "wall_crop_applied": bool(b["wall_crop_applied"]),
        "frame_version": str(b["transform_frame_version"]),
        "coordinate_unit": "mm",
        "tolerance_mm": 1e-3 if bijective else 3.0,
    }
    return aligned, t, tri, idx, normals, report


def load_selection() -> dict:
    selection = json.loads(SELECT_PATH.read_text())
    return {role: selection[role]["case_id"] for role in ROLES}


def build(role: str, cid: str, selection: dict) -> dict:
    d = OUT / MODEL / role
    d.mkdir(parents=True, exist_ok=True)
    pred_path = EXP / "predictions/legacy" / (cid.replace("/", "__") + ".npz")
    bundle_path = DATA / cid / "bundle.npz"
    metrics_path = EXP / "metrics/legacy.json"
    b = loadnp(bundle_path)
    pred = loadnp(pred_path)
    peak = int(b["peak_step"])
    peakidx = int(np.flatnonzero(b["steps"] == peak)[0])
    assert peak == 1162
    metrics = json.loads(metrics_path.read_text())
    official = metrics["s1234_calibrated"]["per_case"][cid]["overall"]
    wall = b["wall_coords_aligned_mm"].astype(float)
    np.testing.assert_array_equal(pred["truth_wss_pa"], b["wall_wss"][peakidx])
    assert bool(np.asarray(pred["valid_mask"]).all())
    assert len(pred["s1234_calibrated_wss_pa"]) == len(wall)
    # npz wall_mm is raw millimetres; visualization uses the atlas-aligned wall.
    np.testing.assert_array_equal(pred["wall_mm"], b["wall_coords_raw"])
    true = pred["truth_wss_pa"].astype(float)
    pred_pa = pred["s1234_calibrated_wss_pa"].astype(float)
    assert np.isfinite(pred_pa).all() and pred_pa.min() >= 0
    point_arrays = {
        "wss_cfd_pa": true,
        "wss_pred_pa": pred_pa,
        "wss_error_pred_minus_cfd_pa": pred_pa - true,
        "wss_abs_error_pa": np.abs(pred_pa - true),
        "row_index": np.arange(len(wall), dtype=np.int64),
    }
    cache_metrics = stats(true, pred_pa)
    assert abs(cache_metrics["r2"] - official["r2"]) < 1e-12
    assert abs(cache_metrics["mae"] - official["mae"]) < 1e-12
    stlv, stlt, cfdtri, wallidx, normals, geom = geometry(cid, b)
    cfkey, prkey, erkey = "wss_cfd_pa", "wss_pred_pa", "wss_error_pred_minus_cfd_pa"
    vals = np.column_stack([true, pred_pa])
    prefix, scope, unit = "wss", "wall", "Pa"
    normalization = fit_selfmax(true, pred_pa, prefix, scope, unit)
    field_data = normalization_field_data(normalization)
    point_arrays.update(selfmax_fields(true, pred_pa, normalization))
    native = dict(point_arrays)
    description = (
        f"{MODEL} {cid}; test; VF6_s1234 ckpt_best → frozen Profile-Secant V3 calibrated WSS; "
        f"peak{peak}; atlas mm; cached same-point prediction"
    )
    pointname = f"{MODEL}__{role}__{cid.replace('/', '__')}__wall_wss.vtp"
    write_vtp(d / pointname, wall, point_arrays, description=description, field_data=field_data)
    write_csv(d / "_export" / "same_point_fields.csv.gz", wall, point_arrays)
    write_vtp(d / "aligned_geometry.vtp", stlv, {}, stlt, description="Aligned STL geometry; no field interpolation")
    write_vtp(
        d / "cfd_wall_native.vtp", wall, native, cfdtri,
        description=description + "; native CFD triangles; no smoothing",
        field_data=field_data,
    )
    mapped, extra, report = gaussian(wall, vals, stlv, wallidx, normals, cfdtri)
    sf = {
        cfkey: mapped[:, 0],
        prkey: mapped[:, 1],
        erkey: mapped[:, 1] - mapped[:, 0],
        erkey.replace("error", "abs_error"): np.abs(mapped[:, 1] - mapped[:, 0]),
        **extra,
    }
    sf.update(selfmax_fields(sf[cfkey], sf[prkey], normalization))
    write_vtp(
        d / "surface_gaussian.vtp", stlv, sf, stlt,
        description=description + "; Wall scalar interpolated to original STL vertices",
        field_data=field_data,
    )
    write_csv(d / "surface_gaussian" / "mapped_vertices.csv", stlv, sf)
    source_arrays = {cfkey: true, prkey: pred_pa, erkey: pred_pa - true, "wss_abs_error_pa": np.abs(pred_pa - true)}
    source_arrays.update(selfmax_fields(true, pred_pa, normalization))
    write_csv(d / "_export" / "gaussian_source.csv.gz", wall, source_arrays)
    assert report["coverage"]["valid_ratio"] == 1.0
    report.update(
        purpose="Wall scalar interpolated to original STL vertices",
        source_points=len(wall),
        source_csv="_export/gaussian_source.csv.gz",
        target_geometry="aligned_geometry.vtp",
        geometry=geom,
        units=unit,
        topology={
            "triangles": len(stlt),
            "valid_triangles": int(np.all(extra["map_valid"][stlt], axis=1).sum()),
        },
        errors="signed error = interpolated Pred - interpolated CFD; absolute = abs(signed)",
        display_normalization=normalization,
    )
    dump(d / "mapping_report.json", report)
    src = {
        str(pred_path): sha(pred_path),
        str(bundle_path): sha(bundle_path),
        str(metrics_path): sha(metrics_path),
        str(SELECT_PATH): sha(SELECT_PATH),
        str(VELOCITY_RUN / "config.json"): sha(VELOCITY_RUN / "config.json"),
        str(VELOCITY_RUN / "ckpt_best.pt"): sha(VELOCITY_RUN / "ckpt_best.pt"),
    }
    visual = dict(
        official,
        seed=1234,
        metric_source=str(metrics_path),
        metric_key=f"s1234_calibrated.per_case[{cid}].overall",
        basis="whole evaluated wall, not smoothed surface",
        cache_recomputed=cache_metrics,
        derived_from="VF6_s1234 speed → frozen Profile-Secant V3 + within-case calibration",
        radius_source="legacy",
    )
    sel = selection[role]
    manifest = {
        "schema_version": 3,
        "model": MODEL,
        "role": role,
        "case_id": cid,
        "run": str(EXP),
        "velocity_run": str(VELOCITY_RUN),
        "checkpoint": "VF6_s1234 ckpt_best + frozen Profile-Secant V3 calibrated",
        "seed": 1234,
        "split": "test",
        "peak_step": peak,
        "peak_index": peakidx,
        "frame": str(b["transform_frame_version"]),
        "coordinate_unit": "mm",
        "unit": unit,
        "selection": {
            "basis": "fixed three-seed mean of per-case derived-WSS R2 (1234,7,2025); not ensemble",
            "r2_case_mean": sel["r2"],
            "statistical_median": selection["median"]["statistical_median"],
            "median_rule": selection["median"]["rule"],
        },
        "visualization_metrics": visual,
        "pressure_reference": None,
        "geometry": geom,
        "source_sha256": src,
        "display_normalization": normalization,
        "verification": {
            "truth_vs_mother_data": "exact",
            "point_index_and_coordinate_identity": "bundle aligned wall; npz wall_mm is raw mm",
            "r2_matches_official_abs_tol": 1e-12,
            "mae_matches_official_abs_tol": 1e-12,
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
        k: sha(d / k)
        for k in [pointname, "surface_gaussian.vtp", "aligned_geometry.vtp", "cfd_wall_native.vtp"]
    }
    dump(d / "manifest.json", manifest)
    cmax_label = "invalid" if normalization["cfd_max"] is None else f"{normalization['cfd_max']:.12g}"
    pmax_label = "invalid" if normalization["pred_max"] is None else f"{normalization['pred_max']:.12g}"
    note = (
        f"# {MODEL} / {role} / {cid}\n\n"
        "ParaView 打开 **surface_gaussian.vtp**，点击 Apply，Representation 选 Surface。\n\n"
        "CFD：`wss_cfd_pa`；Pred：`wss_pred_pa`；有符号误差：`wss_error_pred_minus_cfd_pa`（Pa）。"
        "CFD/Pred 使用相同色标，误差用对称色标。\n\n"
        f"选例 R²（三种子均值）：{sel['r2']:.6f}；当前 s1234 全壁面派生 WSS R²：{official['r2']:.6f}。"
        f"peak {peak}；v5_atlas_frame_v1；坐标 mm。\n\n"
        f"壁面 Gaussian 插值：r=3 mm，sharpness=2，max_dist=3 mm，fallback=mask。"
        f"覆盖率 {report['coverage']['valid_ratio']:.3%}。面片内含 `map_valid`、`map_dist_mm`；"
        "配准和跨壁诊断见 `mapping_report.json`。\n\n"
        "`cfd_wall_native.vtp` 保留原生 CFD 三角面上的同点场值，无高斯平滑；用于核对局部热点。"
        f"原点云：`{pointname}`。\n\n"
        "这是 VF6 预测速度经冻结 Profile-Secant V3（校准）得到的派生 WSS，不是直接 WSS 回归。\n\n"
        "自身最大值归一化：`wss_cfd_selfmax` = CFD / 原始 CFD max；`wss_pred_selfmax` = Pred / 原始 Pred max；"
        "差值 `wss_selfmax_error_pred_minus_cfd`，绝对差 `wss_selfmax_abs_error`。"
        "均无量纲，仅作空间分布对照，幅值差已移除。\n\n"
        f"原始同点域 `{scope}`：CFD max = {cmax_label} {unit}，Pred max = {pmax_label} {unit}。"
        "点云/原生面/Gaussian 共享此对分母，插值后不再重求最大值；元数据嵌入 VTP FieldData 和 manifest.json。\n"
    )
    (d / "README_打开说明.md").write_text(note)
    print(json.dumps({
        "model": MODEL, "role": role, "case": cid, "points": len(wall),
        "surface_triangles": len(stlt), "coverage": report["coverage"]["valid_ratio"],
        "s1234_r2": official["r2"],
    }, ensure_ascii=False), flush=True)
    return manifest


def write_batch(cases: list[dict]) -> None:
    dump(OUT / MODEL / "manifest.json", {
        "schema_version": 3,
        "model": MODEL,
        "note": "VELWSS2 derived WSS from VF6 caches; package paths relative to each case",
        "selection_source": str(SELECT_PATH),
        "selection_sha256": sha(SELECT_PATH),
        "cases": cases,
    })
    with (OUT / MODEL / "打开文件清单.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        import csv
        writer = csv.writer(handle)
        writer.writerow(["model", "role", "case_id", "s1234_whole_case_r2", "three_seed_selection_r2", "surface_file", "original_points", "usage"])
        for case in cases:
            rel = Path(case["role"])
            writer.writerow([
                case["model"], case["role"], case["case_id"],
                case["visualization_metrics"]["r2"], case["selection"]["r2_case_mean"],
                str(rel / case["files"]["gaussian_surface"]),
                str(rel / case["files"]["pointcloud"]),
                case["surface_display_purpose"],
            ])
    with (OUT / MODEL / "归一化分母清单.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        import csv
        writer = csv.writer(handle)
        writer.writerow(["model", "role", "case_id", "seed", "peak_step", "denominator_scope", "source_unit", "cfd_max", "pred_max", "cfd_selfmax_field", "pred_selfmax_field", "cfd_valid", "pred_valid"])
        for case in cases:
            n = case["display_normalization"]
            writer.writerow([
                case["model"], case["role"], case["case_id"], case["seed"], case["peak_step"],
                n["denominator_scope"], n["source_unit"], n["cfd_max"], n["pred_max"],
                "wss_cfd_selfmax", "wss_pred_selfmax",
                n["cfd_denominator"]["valid"], n["pred_denominator"]["valid"],
            ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roles", nargs="+", default=list(ROLES), choices=ROLES)
    args = parser.parse_args()
    selection = json.loads(SELECT_PATH.read_text())
    cases = []
    for role in args.roles:
        cases.append(build(role, selection[role]["case_id"], selection))
    existing = []
    for role in ROLES:
        path = OUT / MODEL / role / "manifest.json"
        if path.exists():
            existing.append(json.loads(path.read_text()))
    if existing:
        write_batch(existing)


if __name__ == "__main__":
    main()
