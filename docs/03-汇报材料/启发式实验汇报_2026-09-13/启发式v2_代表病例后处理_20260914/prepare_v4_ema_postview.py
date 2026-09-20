#!/usr/bin/env python3
"""Export V4-SP-PN-BC-PDE-EMA best/median/worst derived-WSS packs from full-wall cache.

No volume re-inference. Official WSS R² stays on wss_metrics.json / same-point arrays.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

OUT = Path(__file__).resolve().parent
ROOT = next(p for p in OUT.parents if (p / "training_wss_min").is_dir())
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(OUT))
from prepare_selected_postview import (  # noqa: E402
    dump, gaussian, sha, stats, write_csv, write_vtp,
)
from tools.cfdpost_cloud_export.display_fields import (  # noqa: E402
    fit_selfmax, normalization_field_data, selfmax_fields,
)
from tools.cfdpost_cloud_export.map_to_stl_surface import _load_stl_mesh  # noqa: E402

MODEL = "V4_EMA"
ROLES = ("best", "median", "worst")
ARM = "V4-SP-PN-BC-PDE-EMA-s1234"
METRICS = ROOT / "outputs/wss_pinn/audits/v4_fullwall_20260901/arms" / ARM / "wss_metrics.json"
CACHE = ROOT / "outputs/wss_pinn/audits/v4_fullwall_20260901/arms" / ARM / "point_cache_s0"
V2_TOOL = ROOT / "docs/03-汇报材料/tools/update_wss_pinn_v2_v3_workbook.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def resolve_stl(case_id: str) -> Path:
    case_dir = ROOT / "data_new" / case_id
    last = Path(case_id).parts[-1]
    candidates = [case_dir / f"{last}.stl"]
    if last in {"before", "after"} and len(Path(case_id).parts) >= 2:
        patient = Path(case_id).parts[-2]
        stem = patient.rsplit("-", 1)[0] if "-" in patient else patient
        candidates.extend(
            [
                case_dir / f"{stem}.stl",
                case_dir / f"{stem}-sq.stl",
                case_dir / f"{patient}.stl",
            ]
        )
    for path in candidates:
        if path.is_file():
            return path
    stls = sorted(case_dir.glob("*.stl"))
    sq = [path for path in stls if path.name.endswith("-sq.stl")]
    if len(sq) == 1:
        return sq[0]
    if len(stls) == 1:
        return stls[0]
    raise FileNotFoundError(f"Cannot resolve CFD wall STL for {case_id} in {case_dir}")


def select_cases() -> dict:
    payload = json.loads(METRICS.read_text())
    cases = []
    for row in payload["cases"]:
        cases.append({
            "case_id": row["case_id"],
            "r2": float(row["wss_r2"]),
            "mae": float(row["wss_mae_pa"]),
            "nmae_range": float(row["wss_nmae_range"]),
            "points": int(row["points"]),
        })
    ordered = sorted(cases, key=lambda c: (c["r2"], c["case_id"]))
    values = np.array([c["r2"] for c in ordered])
    median = float(np.median(values))
    selected = {
        "worst": ordered[0],
        "best": ordered[-1],
        "median": min(ordered, key=lambda c: (abs(c["r2"] - median), c["case_id"])),
    }
    return {
        "model": MODEL,
        "arm": ARM,
        "protocol": (
            "历史 V4 PointNet BC+PDE-EMA，seed1234，last_converged（epoch9999，converged:false）。"
            "选例按 full-wall 派生 WSS 物理预测 R²；Median 是距分布中位最近的真实病例。"
            "不是正在训练的 X5D_long，也不是 formal P2V/D2 V4。"
        ),
        "n_cases": len(cases),
        "seeds": [1234],
        "stats": {
            "case_mean": float(values.mean()),
            "case_median": median,
            "case_p10": float(np.percentile(values, 10)),
            "negative_count": int((values < 0).sum()),
        },
        "source_metrics": str(METRICS),
        "source_field": "cases[].wss_r2",
        **{role: selected[role] for role in ROLES},
        "all_cases": cases,
    }


def geometry(cid: str, wall_mm: np.ndarray) -> tuple:
    stl = resolve_stl(cid)
    vertices, triangles = _load_stl_mesh(stl)
    distance, index = cKDTree(wall_mm).query(vertices)
    bijective = bool(distance.max() < 1e-3 and len(np.unique(index)) == len(wall_mm) == len(vertices))
    if not bijective:
        assert distance.max() < 3.0, (cid, float(distance.max()))
    bbox_ratio = float(np.linalg.norm(np.ptp(wall_mm, axis=0)) / np.linalg.norm(np.ptp(vertices, axis=0)))
    assert abs(bbox_ratio - 1.0) < 1e-2, (cid, bbox_ratio)
    report = {
        "stl_source": str(stl),
        "stl_sha256": sha(stl),
        "stl_to_mm_scale": 1.0,
        "scale_source": (
            "identity millimetre units by matching bounding-box diagonal; raw CFD frame, not V5 atlas"
        ),
        "stl_cfd_identity": bijective,
        "stl_wall_identity_max_distance_mm": float(distance.max()),
        "bijective_node_count": int(len(np.unique(index))),
        "n_stl_vertices": len(vertices),
        "n_stl_triangles": len(triangles),
        "n_cfd_wall": int(len(wall_mm)),
        "coordinate_unit": "mm",
        "coordinate_frame": "raw CFD millimetres",
        "tolerance_mm": 1e-3 if bijective else 3.0,
    }
    return vertices, triangles, index, report


def build(role: str, cid: str, selection: dict, arrays: dict) -> dict:
    dest = OUT / MODEL / role
    dest.mkdir(parents=True, exist_ok=True)
    wall = np.asarray(arrays["wall_mm"], dtype=float)
    true = np.asarray(arrays["truth_mag"], dtype=float)
    pred = np.asarray(arrays["pred_mag"], dtype=float)
    official = stats(true, pred)
    expected = float(selection[role]["r2"])
    assert abs(official["r2"] - expected) < 1e-8, (cid, official["r2"], expected)
    stl_xyz, stl_tri, wall_idx, geom = geometry(cid, wall)
    prefix, scope, unit = "wss", "wall", "Pa"
    normalization = fit_selfmax(true, pred, prefix, scope, unit)
    field_data = normalization_field_data(normalization)
    point_arrays = {
        "wss_cfd_pa": true,
        "wss_pred_pa": pred,
        "wss_error_pred_minus_cfd_pa": pred - true,
        "wss_abs_error_pa": np.abs(pred - true),
    }
    point_arrays.update(selfmax_fields(true, pred, normalization))
    description = (
        f"{MODEL} {cid}; historical V4 {ARM}; test35; seed1234; last_converged; "
        "derived full-wall WSS from cached peak volume; raw CFD mm"
    )
    pointname = f"{MODEL}__{role}__{cid.replace('/', '__')}__wall_wss.vtp"
    write_vtp(dest / pointname, wall, point_arrays, description=description, field_data=field_data)
    write_csv(dest / "_export" / "same_point_fields.csv.gz", wall, point_arrays)
    write_vtp(
        dest / "aligned_geometry.vtp", stl_xyz, {}, stl_tri,
        description="Original STL geometry in raw CFD millimetres; no field interpolation",
    )
    values = np.column_stack([true, pred])
    mapped, extra, report = gaussian(wall, values, stl_xyz)
    cfkey, prkey, erkey = "wss_cfd_pa", "wss_pred_pa", "wss_error_pred_minus_cfd_pa"
    surface = {
        cfkey: mapped[:, 0],
        prkey: mapped[:, 1],
        erkey: mapped[:, 1] - mapped[:, 0],
        "wss_abs_error_pa": np.abs(mapped[:, 1] - mapped[:, 0]),
        **extra,
    }
    surface.update(selfmax_fields(surface[cfkey], surface[prkey], normalization))
    write_vtp(
        dest / "surface_gaussian.vtp", stl_xyz, surface, stl_tri,
        description=description + "; Wall scalar interpolated to original STL vertices",
        field_data=field_data,
    )
    write_csv(dest / "surface_gaussian" / "mapped_vertices.csv", stl_xyz, surface)
    source_arrays = {
        cfkey: true, prkey: pred, erkey: pred - true,
        "wss_abs_error_pa": np.abs(pred - true),
    }
    source_arrays.update(selfmax_fields(true, pred, normalization))
    write_csv(dest / "_export" / "gaussian_source.csv.gz", wall, source_arrays)
    assert report["coverage"]["valid_ratio"] == 1.0
    report.update(
        purpose="Wall scalar interpolated to original STL vertices",
        source_points=len(wall),
        source_csv="_export/gaussian_source.csv.gz",
        target_geometry="aligned_geometry.vtp",
        geometry=geom, units=unit,
        topology={
            "triangles": len(stl_tri),
            "valid_triangles": int(np.all(extra["map_valid"][stl_tri], axis=1).sum()),
        },
        errors="signed error = interpolated Pred - interpolated CFD; absolute = abs(signed)",
        display_normalization=normalization,
        native_cfd_wall="not written; V4 cache has wall nodes without native triangle identity in this pack",
    )
    dump(dest / "mapping_report.json", report)
    chosen = selection[role]
    manifest = {
        "schema_version": 3,
        "model": MODEL,
        "role": role,
        "case_id": cid,
        "arm": ARM,
        "seed": 1234,
        "split": "test35",
        "checkpoint": "last_converged",
        "coordinate_unit": "mm",
        "coordinate_frame": "raw CFD millimetres",
        "peak_step": "V4 steady peak (historical, not V5 1162 atlas)",
        "selection": {
            "basis": "full-wall derived WSS physical per-case R2; single seed 1234; not ensemble",
            "r2_case_mean": chosen["r2"],
            "mae": chosen.get("mae"),
            "points": chosen.get("points"),
        },
        "visualization_metrics": {
            "r2": official["r2"],
            "mae": official["mae"],
            "n": official["n"],
            "seed": 1234,
            "metric_source": str(METRICS),
            "metric_key": "cases[case_id].wss_r2",
            "basis": "whole evaluated wall, not smoothed surface",
        },
        "display_normalization": normalization,
        "files": {
            "pointcloud": pointname,
            "gaussian_surface": "surface_gaussian.vtp",
            "mapping_report": "mapping_report.json",
            "native_cfd_wall": None,
            "aligned_geometry": "aligned_geometry.vtp",
            "original_points_csv": "_export/same_point_fields.csv.gz",
        },
        "surface_display_purpose": "Derived WSS interpolated to original STL vertices",
        "source_files": {
            str(METRICS): sha(METRICS),
            str(CACHE / (cid.replace("/", "__") + ".npz")): sha(CACHE / (cid.replace("/", "__") + ".npz")),
        },
    }
    dump(dest / "manifest.json", manifest)
    (dest / "README_打开说明.md").write_text(
        f"# {MODEL} / {role} / {cid}\n\n"
        "ParaView 打开 **surface_gaussian.vtp**，Apply，Representation 选 Surface。\n\n"
        "CFD：`wss_cfd_pa`；Pred：`wss_pred_pa`；误差：`wss_error_pred_minus_cfd_pa`（Pa）。"
        "这是历史 V4 PINN 预测速度经冻结 Profile-Secant V3 得到的派生 WSS，不是直接回归。\n\n"
        f"选例/出图 R²（单 seed 1234）：{official['r2']:.6f}。"
        "checkpoint 文件名为 last_converged，实际 epoch9999、converged:false。\n\n"
        f"壁面 Gaussian：r=3 mm，sharpness=2，max_dist=3 mm，覆盖率 {report['coverage']['valid_ratio']:.3%}。"
        "坐标为原始 CFD 毫米，不是 V5 atlas。本包未写 native CFD 三角面。\n"
    )
    print(json.dumps({
        "model": MODEL, "role": role, "case": cid,
        "points": len(wall), "coverage": report["coverage"]["valid_ratio"],
        "wss_r2": official["r2"],
    }, ensure_ascii=False), flush=True)
    return manifest


def write_batch(cases: list[dict]) -> None:
    dest = OUT / MODEL
    dump(dest / "manifest.json", {
        "schema_version": 3,
        "model": MODEL,
        "arm": ARM,
        "note": "Historical V4 EMA derived WSS from full-wall cache; no volume inference",
        "selection_source": str(dest / "selection.json"),
        "selection_sha256": sha(dest / "selection.json"),
        "cases": cases,
    })
    with (dest / "打开文件清单.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "role", "case_id", "wss_r2", "surface_file", "original_points", "usage"])
        for case in cases:
            rel = Path(case["role"])
            writer.writerow([
                case["model"], case["role"], case["case_id"],
                case["visualization_metrics"]["r2"],
                str(rel / case["files"]["gaussian_surface"]),
                str(rel / case["files"]["pointcloud"]),
                case["surface_display_purpose"],
            ])
    with (dest / "归一化分母清单.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "role", "case_id", "seed", "denominator_scope",
                         "source_unit", "cfd_max", "pred_max", "cfd_valid", "pred_valid"])
        for case in cases:
            norm = case["display_normalization"]
            writer.writerow([
                case["model"], case["role"], case["case_id"], case["seed"],
                norm["denominator_scope"], norm["source_unit"], norm["cfd_max"], norm["pred_max"],
                norm["cfd_denominator"]["valid"], norm["pred_denominator"]["valid"],
            ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roles", nargs="+", default=list(ROLES), choices=ROLES)
    args = parser.parse_args()
    dest = OUT / MODEL
    dest.mkdir(exist_ok=True)
    selection = select_cases()
    (dest / "selection.json").write_text(json.dumps(selection, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "wrote": str(dest / "selection.json"),
        "best": selection["best"]["case_id"],
        "median": selection["median"]["case_id"],
        "worst": selection["worst"]["case_id"],
        "stats": selection["stats"],
    }, ensure_ascii=False), flush=True)
    v2 = _load_module("v2_workbook_for_v4_ema_postview", V2_TOOL)
    _, arrays_by_case = v2._apply_calibrator(CACHE)
    for role in args.roles:
        cid = selection[role]["case_id"]
        build(role, cid, selection, arrays_by_case[cid])
    existing = []
    for role in ROLES:
        path = dest / role / "manifest.json"
        if path.exists():
            existing.append(json.loads(path.read_text()))
    if existing:
        write_batch(existing)


if __name__ == "__main__":
    main()
