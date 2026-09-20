#!/usr/bin/env python3
"""GPU-export V4 EMA speed/pressure onto the WSS-selected best/median/worst cases.

Uses remaining dedicated VRAM (and the driver-allowed remainder) on one 4090.
Does not overwrite WSS ``surface_gaussian.vtp`` or ``same_point_fields.csv.gz``.
Official speed/pressure R² stay on the historical evaluation JSON / same-point arrays.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

OUT = Path(__file__).resolve().parent
ROOT = next(p for p in OUT.parents if (p / "training_wss_min").is_dir())
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(OUT))

from prepare_selected_postview import dump, gaussian, sha, stats, write_csv, write_vtp  # noqa: E402
from prepare_v4_ema_postview import MODEL, ROLES, geometry  # noqa: E402
from tools.cfdpost_cloud_export.display_fields import (  # noqa: E402
    fit_selfmax,
    normalization_field_data,
    selfmax_fields,
)

ARM = "V4-SP-PN-BC-PDE-EMA-s1234"
ARM_INDEX = 3
PROTECTED = {
    "surface_gaussian.vtp",
    "mapping_report.json",
    "aligned_geometry.vtp",
    "_export/same_point_fields.csv.gz",
    "_export/gaussian_source.csv.gz",
}


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def official_row(report: dict, case_id: str) -> dict:
    for row in report["cases"]:
        if row["case_id"] == case_id:
            return row
    raise KeyError(case_id)


def predict_case(exp, arm, dataset, model, device, config, cid: str, chunk_size: int):
    import torch

    by_id = {dataset._case(index)["canonical_id"]: index for index in range(len(dataset))}
    case_index = by_id[cid]
    v4_case = dataset._case(case_index)
    arrays = exp._CaseArrays(dataset, case_index)
    volume = exp._volume_case(v4_case)
    query_time = (
        None
        if arm.temporal_mode == "steady_peak"
        else exp.V4WB._peak_time_s(v4_case, volume.peak_step)
    )
    support_index = arrays.sample_indices(
        "eval_support", int(config["sampling"]["support_points"])
    )
    support_coords = torch.as_tensor(
        np.asarray(arrays.coords[support_index], dtype=np.float32), device=device
    )
    support_features = torch.as_tensor(
        exp.V4WB._support_features(dataset, arrays, support_index), device=device
    )
    support_batch = torch.zeros(len(support_index), dtype=torch.long, device=device)
    bc_vector = torch.as_tensor(arrays.bc_vector, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        encoded = model.encode_support(
            support_coords,
            support_features,
            support_batch,
            bc_vector,
            [arrays.case_id],
            int(config["train"]["seed"]),
        )
        pred_u, pred_p = exp.V4WB._predict_volume(
            model,
            encoded,
            np.asarray(volume.coords),
            query_time,
            device,
            dataset.field_stats,
            chunk_size,
        )
        bundle = exp.BASE._bundle_arrays(volume)
        wall_u, wall_p = exp.V4WB._predict_volume(
            model,
            encoded,
            np.asarray(bundle["wall_coords_norm"]),
            query_time,
            device,
            dataset.field_stats,
            chunk_size,
        )
    del encoded
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return volume, bundle, pred_u, pred_p, wall_u, wall_p


def metric_block(true, pred, official, source, key, basis):
    computed = stats(true, pred)
    official_r2 = float(official["r2"])
    official_mae = float(official["mae"])
    official_n = int(official["count"])
    if computed["n"] != official_n:
        raise ValueError(f"{key} n mismatch: {computed['n']} vs official {official_n}")
    if abs(computed["r2"] - official_r2) > 5e-4:
        raise ValueError(f"{key} R2 mismatch: {computed['r2']} vs official {official_r2}")
    if abs(computed["mae"] - official_mae) > 5e-4:
        raise ValueError(f"{key} MAE mismatch: {computed['mae']} vs official {official_mae}")
    return {
        "r2": computed["r2"],
        "mae": computed["mae"],
        "n": computed["n"],
        "seed": 1234,
        "metric_source": source,
        "metric_key": key,
        "basis": basis,
        "official_json": {"r2": official_r2, "mae": official_mae, "n": official_n},
        "cache_recomputed": computed,
    }


def write_volume(role: str, cid: str, selection: dict, predicted: dict, official: dict, eval_path: str) -> dict:
    dest = OUT / MODEL / role
    existing = json.loads((dest / "manifest.json").read_text())
    volume = predicted["volume"]
    bundle = predicted["bundle"]
    interior_mm = np.asarray(bundle["interior_coords_raw_mm"], dtype=float)
    wall_mm = np.asarray(bundle["wall_coords_raw"], dtype=float)
    if len(interior_mm) != len(volume.coords):
        raise ValueError(f"{cid}: interior raw/volume length {len(interior_mm)} vs {len(volume.coords)}")
    is_wall = np.asarray(volume.is_wall, dtype=bool)
    interior_idx = np.flatnonzero(~is_wall)
    rotation = np.asarray(bundle["rotation"], dtype=float)
    interior_xyz = interior_mm[interior_idx]
    true_u = np.asarray(volume.velocity[interior_idx], dtype=float) @ rotation.T
    pred_u = np.asarray(predicted["velocity"][interior_idx], dtype=float) @ rotation.T
    true_speed = np.linalg.norm(true_u, axis=1)
    pred_speed = np.linalg.norm(pred_u, axis=1)
    true_p_int = np.asarray(volume.pressure[interior_idx], dtype=float)
    pred_p_int = np.asarray(predicted["pressure"][interior_idx], dtype=float)
    p_ref = float(volume.manifest["scales"]["pressure_reference_pa"])
    true_p_wall = np.asarray(bundle["wall_pressure_pa"], dtype=float) - p_ref
    pred_p_wall = np.asarray(predicted["wall_pressure"], dtype=float)
    dist = cKDTree(wall_mm).query(interior_xyz)[0]
    speed_metrics = metric_block(
        true_speed, pred_speed, official["metrics"]["speed"],
        eval_path, "cases[case_id].metrics.speed.r2",
        "official test35 strict-interior speed; not smoothed mesh",
    )
    pressure_metrics = metric_block(
        true_p_int, pred_p_int, official["metrics"]["pressure"],
        eval_path, "cases[case_id].metrics.pressure.r2",
        "official test35 strict-interior relative pressure; visualization also writes wall",
    )
    speed_norm = fit_selfmax(true_speed, pred_speed, "speed", "interior", "m/s")
    pressure_all_true = np.concatenate([true_p_wall, true_p_int])
    pressure_all_pred = np.concatenate([pred_p_wall, pred_p_int])
    pressure_norm = fit_selfmax(
        pressure_all_true, pressure_all_pred, "pressure", "wall_union_interior", "Pa"
    )

    n_int, n_wall = len(interior_idx), len(wall_mm)
    speed_arrays = {
        "speed_cfd": true_speed,
        "speed_pred": pred_speed,
        "speed_error_pred_minus_cfd": pred_speed - true_speed,
        "speed_abs_error": np.abs(pred_speed - true_speed),
        "velocity_cfd_vector": true_u,
        "velocity_pred_vector": pred_u,
        "velocity_error_vector": pred_u - true_u,
        "velocity_vector_error_norm": np.linalg.norm(pred_u - true_u, axis=1),
        "velocity_cfd_x": true_u[:, 0],
        "velocity_cfd_y": true_u[:, 1],
        "velocity_cfd_z": true_u[:, 2],
        "velocity_pred_x": pred_u[:, 0],
        "velocity_pred_y": pred_u[:, 1],
        "velocity_pred_z": pred_u[:, 2],
        "point_kind": np.ones(n_int, dtype=np.int8),
        "dist_to_wall_mm": dist,
        "source_index": interior_idx.astype(np.int64),
    }
    speed_arrays.update(selfmax_fields(true_speed, pred_speed, speed_norm))
    speed_name = f"{MODEL}__{role}__{cid.replace('/', '__')}__interior_velocity.vtp"
    description = (
        f"{MODEL} {cid}; historical V4 {ARM}; test35; seed1234; last_converged; "
        "strict-interior speed; raw CFD mm"
    )
    write_vtp(
        dest / speed_name, interior_xyz, speed_arrays,
        description=description, field_data=normalization_field_data(speed_norm),
    )
    write_csv(dest / "_export" / "same_point_speed.csv.gz", interior_xyz, {
        k: v for k, v in speed_arrays.items() if np.asarray(v).ndim == 1 or k.endswith("_vector")
    })

    kind = np.concatenate([np.zeros(n_wall, dtype=np.int8), np.ones(n_int, dtype=np.int8)])
    pressure_xyz = np.concatenate([wall_mm, interior_xyz], axis=0)
    pressure_arrays = {
        "pressure_cfd_pa": pressure_all_true,
        "pressure_pred_pa": pressure_all_pred,
        "pressure_error_pred_minus_cfd_pa": pressure_all_pred - pressure_all_true,
        "pressure_abs_error_pa": np.abs(pressure_all_pred - pressure_all_true),
        "point_kind": kind,
        "source_index": np.concatenate([
            np.arange(n_wall, dtype=np.int64), interior_idx.astype(np.int64)
        ]),
        "dist_to_wall_mm": np.concatenate([np.zeros(n_wall), dist]),
    }
    pressure_arrays.update(selfmax_fields(pressure_all_true, pressure_all_pred, pressure_norm))
    pressure_name = f"{MODEL}__{role}__{cid.replace('/', '__')}__volume_pressure.vtp"
    p_desc = (
        f"{MODEL} {cid}; historical V4 {ARM}; wall∪interior relative pressure p-p_ref={p_ref:.6f} Pa"
    )
    write_vtp(
        dest / pressure_name, pressure_xyz, pressure_arrays,
        description=p_desc, field_data=normalization_field_data(pressure_norm),
    )
    write_csv(dest / "_export" / "same_point_pressure.csv.gz", interior_xyz, {
        "pressure_cfd_pa": true_p_int,
        "pressure_pred_pa": pred_p_int,
        "pressure_error_pred_minus_cfd_pa": pred_p_int - true_p_int,
        "pressure_abs_error_pa": np.abs(pred_p_int - true_p_int),
        **selfmax_fields(true_p_int, pred_p_int, pressure_norm),
    })

    stl_xyz, stl_tri, _wall_idx, geom = geometry(cid, wall_mm)
    mapped, extra, report = gaussian(
        wall_mm, np.column_stack([true_p_wall, pred_p_wall]), stl_xyz
    )
    surface = {
        "pressure_cfd_pa": mapped[:, 0],
        "pressure_pred_pa": mapped[:, 1],
        "pressure_error_pred_minus_cfd_pa": mapped[:, 1] - mapped[:, 0],
        "pressure_abs_error_pa": np.abs(mapped[:, 1] - mapped[:, 0]),
        **extra,
    }
    surface.update(selfmax_fields(surface["pressure_cfd_pa"], surface["pressure_pred_pa"], pressure_norm))
    write_vtp(
        dest / "pressure_surface_gaussian.vtp", stl_xyz, surface, stl_tri,
        description=p_desc + "; Wall scalar interpolated to original STL vertices",
        field_data=normalization_field_data(pressure_norm),
    )
    write_csv(dest / "pressure_surface_gaussian" / "mapped_vertices.csv", stl_xyz, surface)
    write_csv(dest / "_export" / "pressure_gaussian_source.csv.gz", wall_mm, {
        "pressure_cfd_pa": true_p_wall,
        "pressure_pred_pa": pred_p_wall,
        "pressure_error_pred_minus_cfd_pa": pred_p_wall - true_p_wall,
        **selfmax_fields(true_p_wall, pred_p_wall, pressure_norm),
    })
    if report["coverage"]["valid_ratio"] != 1.0:
        raise ValueError(f"{cid} pressure Gaussian coverage {report['coverage']}")
    report.update(
        purpose="Wall relative pressure interpolated to original STL vertices",
        source_points=n_wall,
        source_csv="_export/pressure_gaussian_source.csv.gz",
        target_geometry="aligned_geometry.vtp",
        geometry=geom,
        units="Pa",
        topology={
            "triangles": len(stl_tri),
            "valid_triangles": int(np.all(extra["map_valid"][stl_tri], axis=1).sum()),
        },
        errors="signed error = interpolated Pred - interpolated CFD; absolute = abs(signed)",
        display_normalization=pressure_norm,
        official_pressure_domain="strict interior; wall is visualization-only",
    )
    dump(dest / "pressure_mapping_report.json", report)

    existing["visualization_metrics_speed"] = speed_metrics
    existing["visualization_metrics_pressure"] = pressure_metrics
    existing["display_normalization_speed"] = speed_norm
    existing["display_normalization_pressure"] = pressure_norm
    existing["pressure_reference"] = {
        "p_ref_pa": p_ref,
        "kind": "V4 volume manifest scales.pressure_reference_pa",
        "posthoc_offset_correction": False,
    }
    existing["files"].update({
        "interior_velocity": speed_name,
        "volume_pressure": pressure_name,
        "pressure_gaussian_surface": "pressure_surface_gaussian.vtp",
        "pressure_mapping_report": "pressure_mapping_report.json",
        "speed_points_csv": "_export/same_point_speed.csv.gz",
        "pressure_points_csv": "_export/same_point_pressure.csv.gz",
    })
    existing["volume_note"] = (
        "Speed/pressure inferred for the WSS-selected cases; not independent speed ranking. "
        "Official R² is test35 strict-interior. Pressure VTP adds wall for visualization."
    )
    dump(dest / "manifest.json", existing)
    (dest / "README_打开说明.md").write_text(
        f"# {MODEL} / {role} / {cid}\n\n"
        "病例按 **派生 WSS R²** 选取（不是速度/压力独立排名）。checkpoint 名为 last_converged，"
        "实际 epoch9999、converged:false。坐标为原始 CFD 毫米，不是 V5 atlas。\n\n"
        "## WSS\n\n"
        "ParaView 打开 **surface_gaussian.vtp**，Representation 选 Surface。\n"
        f"CFD `wss_cfd_pa`；Pred `wss_pred_pa`。选例/出图 R²：{existing['visualization_metrics']['r2']:.6f}。\n\n"
        "## 速度（体内点云）\n\n"
        f"打开 **{speed_name}**。CFD `speed_cfd`；Pred `speed_pred`；向量 `velocity_*_vector`。\n"
        f"正式体内速度 R²：{speed_metrics['r2']:.6f}（n={speed_metrics['n']}）。不做速度壁面插值。\n\n"
        "## 压力（壁面∪体内）\n\n"
        f"点云 **{pressure_name}**（`point_kind` 0 壁面 / 1 体内）。壁面 Gaussian："
        "**pressure_surface_gaussian.vtp**（不覆盖 WSS 的 surface_gaussian.vtp）。\n"
        f"正式体内相对压力 R²：{pressure_metrics['r2']:.6f}（n={pressure_metrics['n']}，不含壁面分母）。"
        f" p_ref = {p_ref:.6f} Pa。压力 selfmax 分母取 wall∪interior。\n"
        f"壁面 Gaussian 覆盖率 {report['coverage']['valid_ratio']:.3%}，r=3 mm / sharpness=2 / max_dist=3 mm。\n"
    )
    for name in PROTECTED:
        path = dest / name if not name.startswith("_export/") else dest / name
        if not path.exists():
            raise FileNotFoundError(f"protected WSS file missing after volume write: {path}")
    print(json.dumps({
        "model": MODEL, "role": role, "case": cid,
        "speed_r2": speed_metrics["r2"], "speed_n": speed_metrics["n"],
        "pressure_r2": pressure_metrics["r2"], "pressure_n": pressure_metrics["n"],
        "pressure_coverage": report["coverage"]["valid_ratio"],
        "wss_r2": existing["visualization_metrics"]["r2"],
    }, ensure_ascii=False), flush=True)
    return existing


def write_batch(cases: list[dict]) -> None:
    dest = OUT / MODEL
    dump(dest / "manifest.json", {
        "schema_version": 3,
        "model": MODEL,
        "arm": ARM,
        "note": (
            "Historical V4 EMA: derived WSS from full-wall cache; speed/pressure from GPU "
            "re-inference of the same three WSS-selected cases."
        ),
        "selection_source": str(dest / "selection.json"),
        "selection_sha256": sha(dest / "selection.json"),
        "cases": cases,
    })
    with (dest / "打开文件清单.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "model", "role", "case_id", "wss_r2", "speed_r2", "pressure_r2",
            "wss_surface", "speed_pointcloud", "pressure_pointcloud", "pressure_surface", "usage",
        ])
        for case in cases:
            rel = Path(case["role"])
            files = case["files"]
            writer.writerow([
                case["model"], case["role"], case["case_id"],
                case["visualization_metrics"]["r2"],
                case["visualization_metrics_speed"]["r2"],
                case["visualization_metrics_pressure"]["r2"],
                str(rel / files["gaussian_surface"]),
                str(rel / files["interior_velocity"]),
                str(rel / files["volume_pressure"]),
                str(rel / files["pressure_gaussian_surface"]),
                "WSS-selected cases; speed/pressure are the same patients",
            ])
    with (dest / "归一化分母清单.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "model", "role", "case_id", "seed", "prefix", "denominator_scope",
            "source_unit", "cfd_max", "pred_max", "cfd_valid", "pred_valid",
        ])
        for case in cases:
            for key, prefix in (
                ("display_normalization", "wss"),
                ("display_normalization_speed", "speed"),
                ("display_normalization_pressure", "pressure"),
            ):
                norm = case[key]
                writer.writerow([
                    case["model"], case["role"], case["case_id"], case["seed"], prefix,
                    norm["denominator_scope"], norm["source_unit"], norm["cfd_max"],
                    norm["pred_max"], norm["cfd_denominator"]["valid"],
                    norm["pred_denominator"]["valid"],
                ])
    lines = [
        "# V4 EMA 代表病例后处理（历史 PointNet BC+PDE-EMA）",
        "",
        "臂：`V4-SP-PN-BC-PDE-EMA-s1234`（矩阵 index 3，seed1234，`last_converged` / epoch9999 / converged:false）。",
        "不是正在训练的 X5D_long，也不是 formal P2V/D2 V4。坐标为原始 CFD 毫米。",
        "",
        "## 选例",
        "",
        "按 test35 全壁面派生 WSS 物理预测 R²；Median 是距分布中位最近的真实病例。",
        "速度/压力是**同一三例**的体场，不是按速度或压力独立选的最好/最差。",
        "",
        "| 角色 | 病例 | WSS R² | 速度 R² | 压力 R² |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for case in cases:
        lines.append(
            f"| {case['role']} | `{case['case_id']}` | "
            f"{case['visualization_metrics']['r2']:.4f} | "
            f"{case['visualization_metrics_speed']['r2']:.4f} | "
            f"{case['visualization_metrics_pressure']['r2']:.4f} |"
        )
    lines += [
        "",
        "## 打开文件",
        "",
        "- WSS 壁面：各例 `surface_gaussian.vtp`（`wss_cfd_pa` / `wss_pred_pa`）",
        "- 速度体内点云：`*__interior_velocity.vtp`（`speed_cfd` / `speed_pred`）",
        "- 压力点云：`*__volume_pressure.vtp`；压力壁面：`pressure_surface_gaussian.vtp`",
        "",
        "压力 Gaussian **不覆盖** WSS 的 `surface_gaussian.vtp`。正式速度/压力 R² 是严格体内同点，",
        "不是插值面。压力 selfmax 分母取 wall∪interior。",
        "",
        "清单：[打开文件清单.csv](打开文件清单.csv) · [归一化分母清单.csv](归一化分母清单.csv)",
        "",
    ]
    (dest / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="2", help="physical GPU index; set as CUDA_VISIBLE_DEVICES")
    parser.add_argument("--memory-fraction", type=float, default=0.70)
    parser.add_argument("--chunk-size", type=int, default=16384)
    parser.add_argument("--roles", nargs="+", default=list(ROLES), choices=ROLES)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch

    exp = _load_module(
        "v4_speed_export_for_ema_volume",
        ROOT / "docs/03-汇报材料/tools/export_v4_speed_r2_postview.py",
    )
    from wss_pinn.v4.config import load_config
    from wss_pinn.v4.data import V4Dataset
    from wss_pinn.v4.models import build_model

    selection = json.loads((OUT / MODEL / "selection.json").read_text())
    arm = exp.V4WB.load_arm(ARM_INDEX)
    if arm.key != ARM:
        raise RuntimeError(f"expected {ARM}, got {arm.key}")
    eval_report = json.loads(arm.official_eval_path.read_text())
    config = load_config(arm.config_path, require_assets=True)
    dataset = V4Dataset(config, roles=("test",))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(float(args.memory_fraction), 0)
        free, total = torch.cuda.mem_get_info(0)
        print(json.dumps({
            "event": "gpu_ready",
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "memory_fraction": args.memory_fraction,
            "free_bytes": int(free),
            "total_bytes": int(total),
            "device": torch.cuda.get_device_name(0),
        }, ensure_ascii=False), flush=True)
    model = build_model(config).to(device=device, dtype=torch.float32)
    payload = torch.load(arm.checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    written = []
    for role in args.roles:
        cid = selection[role]["case_id"]
        print(json.dumps({"event": "case_start", "role": role, "case_id": cid}, ensure_ascii=False), flush=True)
        volume, bundle, pred_u, pred_p, wall_u, wall_p = predict_case(
            exp, arm, dataset, model, device, config, cid, args.chunk_size
        )
        manifest = write_volume(role, cid, selection, {
            "volume": volume, "bundle": bundle,
            "velocity": pred_u, "pressure": pred_p,
            "wall_velocity": wall_u, "wall_pressure": wall_p,
        }, official_row(eval_report, cid), str(arm.official_eval_path))
        written.append(manifest)
        print(json.dumps({"event": "case_done", "role": role, "case_id": cid}, ensure_ascii=False), flush=True)
    existing = []
    for role in ROLES:
        path = OUT / MODEL / role / "manifest.json"
        if path.exists():
            existing.append(json.loads(path.read_text()))
    if all("visualization_metrics_speed" in case for case in existing):
        write_batch(existing)
        print(json.dumps({"event": "batch_written", "n": len(existing)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
