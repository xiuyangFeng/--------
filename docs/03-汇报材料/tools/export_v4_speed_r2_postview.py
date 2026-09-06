#!/usr/bin/env python3
"""Export V4 best/worst official speed-R² cases in the V2/V3 postview layout.

For selected matrix indices the script:

1. ranks official test35 ``metrics.speed.r2``;
2. re-predicts the peak full volume and frozen wall;
3. runs full-wall Profile-Secant V3 WSS;
4. writes mixed point-cloud VTPs;
5. Gaussian-maps wall fields onto the case STL;
6. crops the volume sample to the anatomical ROI.

Official numerical metrics remain on the original evaluation JSON / same-point
arrays.  These files are visualization products only.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from wss_pinn.utils import atomic_write_json, sha256_file, utc_now
from wss_pinn.v4.config import load_config
from wss_pinn.v4.data import V4Dataset
from wss_pinn.v4.evaluate import _CaseArrays
from wss_pinn.v4.models import build_model

V4_TOOL = ROOT / "docs/03-汇报材料/tools/update_wss_pinn_v4_workbook.py"
MAP_TOOL = ROOT / "docs/03-汇报材料/tools/map_wss_pinn_vtp_to_surface.py"
CROP_TOOL = ROOT / "docs/03-汇报材料/tools/crop_wss_pinn_velocity_pointcloud_to_anatomical_roi.py"
COLORMAP = ROOT / "tools/cfdpost_cloud_export/paraview/GNN_blue_white_red.xml"
EXISTING_FULLWALL = ROOT / "outputs/wss_pinn/audits/v4_fullwall_20260901"
DEFAULT_OUTPUT = ROOT / "outputs/wss_pinn/audits/v4_speed_r2_postview_20260902"
DEFAULT_INDICES = (5, 7, 8)
INTERIOR_VISUAL_POINTS = 100_000


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


V4WB = _load_module("v4_workbook_tool_for_postview", V4_TOOL)
BASE = V4WB._load_base()
MAP = _load_module("map_wss_pinn_vtp_to_surface_for_v4", MAP_TOOL)
CROP = _load_module("crop_wss_pinn_velocity_for_v4", CROP_TOOL)


def _safe_case(case_id: str) -> str:
    return case_id.replace("/", "__")


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_stl(case_id: str) -> Path:
    case_dir = ROOT / "data_new" / case_id
    last = Path(case_id).parts[-1]
    candidates: list[Path] = [case_dir / f"{last}.stl"]
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
    candidates.append(case_dir / "0zhang_jin_chun-sq.stl")
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


def _volume_case(v4_case: dict[str, Any]) -> SimpleNamespace:
    volume = json.loads(
        Path(v4_case["files"]["steady_volume_manifest"]["path"]).read_text(encoding="utf-8")
    )
    files = volume["files"]
    return SimpleNamespace(
        case_id=str(v4_case["canonical_id"]),
        manifest=volume,
        coords=np.load(files["interior_coords"]["path"], mmap_mode="r"),
        velocity=np.load(files["velocity_m_s"]["path"], mmap_mode="r"),
        pressure=np.load(files["pressure_relative_pa"]["path"], mmap_mode="r"),
        is_wall=np.asarray(np.load(files["interior_is_wall"]["path"], mmap_mode="r"), dtype=bool),
        peak_step=int(volume["peak_step"]),
    )


def select_speed_cases(arm: V4WB.V4Arm) -> list[tuple[str, str, float]]:
    report = _json(arm.official_eval_path)
    rows = sorted(
        (
            (float(case["metrics"]["speed"]["r2"]), str(case["case_id"]))
            for case in report["cases"]
            if case.get("metrics", {}).get("speed")
        ),
        key=lambda item: (item[0], item[1]),
    )
    if len(rows) < 2:
        raise RuntimeError(f"{arm.key} official eval does not contain enough speed R² rows")
    return [
        ("worst", rows[0][1], rows[0][0]),
        ("best", rows[-1][1], rows[-1][0]),
    ]


def _maybe_reuse_wss_cache(arm: V4WB.V4Arm, case_id: str, dest: Path) -> bool:
    source = EXISTING_FULLWALL / "arms" / arm.key / "point_cache_s0" / (_safe_case(case_id) + ".npz")
    if not source.is_file():
        return False
    if not BASE._cache_valid(source, arm, 0):
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)
    return True


def export_arm(
    arm: V4WB.V4Arm,
    device_name: str,
    output_dir: Path,
    chunk_size: int,
    overwrite: bool,
) -> dict[str, Any]:
    import torch

    selections = select_speed_cases(arm)
    config = load_config(arm.config_path, require_assets=True)
    dataset = V4Dataset(config, roles=("test",))
    by_id = {dataset._case(index)["canonical_id"]: index for index in range(len(dataset))}
    device = torch.device(device_name)
    model = build_model(config).to(device=device, dtype=torch.float32)
    payload = torch.load(arm.checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()

    arm_dir = output_dir / "vtp" / arm.key
    cache_dir = arm_dir / "fullwall_cache"
    arm_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    wss_configs = BASE._load_wss_configs()
    predicted_by_case: dict[str, dict[str, Any]] = {}

    for label, case_id, score in selections:
        case_index = by_id[case_id]
        v4_case = dataset._case(case_index)
        arrays = _CaseArrays(dataset, case_index)
        volume = _volume_case(v4_case)
        query_time = (
            None
            if arm.temporal_mode == "steady_peak"
            else V4WB._peak_time_s(v4_case, volume.peak_step)
        )
        support_index = arrays.sample_indices(
            "eval_support", int(config["sampling"]["support_points"])
        )
        support_coords = torch.as_tensor(
            np.asarray(arrays.coords[support_index], dtype=np.float32), device=device
        )
        support_features = torch.as_tensor(
            V4WB._support_features(dataset, arrays, support_index), device=device
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
            pred_u, pred_p = V4WB._predict_volume(
                model,
                encoded,
                np.asarray(volume.coords),
                query_time,
                device,
                dataset.field_stats,
                chunk_size,
            )
        bundle = BASE._bundle_arrays(volume)
        with torch.no_grad():
            wall_u, wall_p = V4WB._predict_volume(
                model,
                encoded,
                np.asarray(bundle["wall_coords_norm"]),
                query_time,
                device,
                dataset.field_stats,
                chunk_size,
            )
        cache_path = cache_dir / (_safe_case(case_id) + ".npz")
        if overwrite or not BASE._cache_valid(cache_path, arm, 0):
            reused = False if overwrite else _maybe_reuse_wss_cache(arm, case_id, cache_path)
            if overwrite or not reused:
                wss_arrays = BASE._build_wss_cache_arrays(
                    volume,
                    bundle,
                    pred_u,
                    arm,
                    0,
                    *wss_configs[:4],
                )
                BASE._write_npz(cache_path, wss_arrays)
        predicted_by_case[case_id] = {
            "label": label,
            "speed_r2": score,
            "volume": volume,
            "bundle": bundle,
            "velocity": pred_u,
            "pressure": pred_p,
            "wall_velocity": wall_u,
            "wall_pressure": wall_p,
        }
        del encoded
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(
            json.dumps(
                {
                    "event": "v4_postview_predicted",
                    "arm": arm.key,
                    "selection": label,
                    "case_id": case_id,
                    "official_speed_r2": score,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    _, wss_by_case = BASE._apply_calibrator(cache_dir)
    manifest_rows = []
    for case_id, item in predicted_by_case.items():
        volume = item["volume"]
        bundle = item["bundle"]
        wss = wss_by_case[case_id]
        strict = np.flatnonzero(~np.asarray(volume.is_wall, dtype=bool))
        take = min(int(INTERIOR_VISUAL_POINTS), len(strict))
        seed = int.from_bytes(hashlib.sha256(case_id.encode("utf-8")).digest()[:8], "little")
        selected = np.sort(np.random.default_rng(seed).choice(strict, size=take, replace=False))
        rotation = np.asarray(bundle["rotation"], dtype=np.float64)
        interior_points = np.asarray(bundle["interior_coords_raw_mm"], dtype=np.float64)[selected]
        wall_points = np.asarray(bundle["wall_coords_raw"], dtype=np.float64)
        points = np.concatenate([interior_points, wall_points], axis=0)
        n_interior = len(interior_points)
        n_wall = len(wall_points)
        truth_velocity = np.concatenate(
            [
                (np.asarray(volume.velocity[selected], dtype=np.float64) @ rotation.T).astype(np.float32),
                np.zeros((n_wall, 3), dtype=np.float32),
            ],
            axis=0,
        )
        pred_velocity = np.concatenate(
            [
                (np.asarray(item["velocity"][selected], dtype=np.float64) @ rotation.T).astype(np.float32),
                (np.asarray(item["wall_velocity"], dtype=np.float64) @ rotation.T).astype(np.float32),
            ],
            axis=0,
        ).astype(np.float32)
        pressure_reference = float(volume.manifest["scales"]["pressure_reference_pa"])
        truth_pressure = np.concatenate(
            [
                np.asarray(volume.pressure[selected], dtype=np.float32),
                (bundle["wall_pressure_pa"] - pressure_reference).astype(np.float32),
            ],
            axis=0,
        )
        pred_pressure = np.concatenate(
            [item["pressure"][selected], item["wall_pressure"]], axis=0
        ).astype(np.float32)
        zero_wss = np.zeros((n_interior, 3), dtype=np.float32)
        truth_wss_vec = np.concatenate([zero_wss, wss["truth_vec"].astype(np.float32)], axis=0)
        pred_wss_vec = np.concatenate([zero_wss, wss["pred_vec"].astype(np.float32)], axis=0)
        truth_wss_mag = np.concatenate(
            [np.zeros(n_interior, dtype=np.float32), wss["truth_mag"].astype(np.float32)]
        )
        pred_wss_mag = np.concatenate(
            [np.zeros(n_interior, dtype=np.float32), wss["pred_mag"].astype(np.float32)]
        )
        truth_speed = np.linalg.norm(truth_velocity, axis=1).astype(np.float32)
        pred_speed = np.linalg.norm(pred_velocity, axis=1).astype(np.float32)
        point_kind = np.concatenate(
            [np.zeros(n_interior, dtype=np.int8), np.ones(n_wall, dtype=np.int8)]
        )
        arrays = {
            "point_kind_0_volume_1_wall": point_kind,
            "field_valid": np.ones(len(points), dtype=np.int8),
            "wss_valid": point_kind.copy(),
            "velocity_truth_m_s": truth_velocity,
            "velocity_pred_m_s": pred_velocity,
            "velocity_error_m_s": pred_velocity - truth_velocity,
            "speed_truth_m_s": truth_speed,
            "speed_pred_m_s": pred_speed,
            "speed_abs_error_m_s": np.abs(pred_speed - truth_speed),
            "pressure_truth_relative_pa": truth_pressure,
            "pressure_pred_relative_pa": pred_pressure,
            "pressure_error_signed_pa": pred_pressure - truth_pressure,
            "pressure_abs_error_pa": np.abs(pred_pressure - truth_pressure),
            "wss_truth_vector_pa": truth_wss_vec,
            "wss_pred_vector_pa": pred_wss_vec,
            "wss_truth_pa": truth_wss_mag,
            "wss_pred_pa": pred_wss_mag,
            "wss_error_signed_pa": pred_wss_mag - truth_wss_mag,
            "wss_abs_error_pa": np.abs(pred_wss_mag - truth_wss_mag),
            "source_index": np.concatenate(
                [selected.astype(np.int64), np.arange(n_wall, dtype=np.int64)]
            ),
        }
        output_path = arm_dir / f"{item['label']}__{_safe_case(case_id)}.vtp"
        if output_path.exists() and not overwrite:
            raise FileExistsError(output_path)
        BASE._write_vtp(
            output_path,
            points,
            arrays,
            {
                "arm": arm.key,
                "matrix_index": int(arm.index),
                "run_id": arm.key,
                "case_id": case_id,
                "selection": item["label"],
                "selection_official_speed_r2": float(item["speed_r2"]),
                "coordinate_frame": "raw CFD frame, millimetres",
                "wss_algorithm": BASE.ALGORITHM_NAME,
            },
        )
        stl_path = resolve_stl(case_id)
        mapping_report = MAP.process_case(
            source_path=output_path,
            stl_path=stl_path,
            case_id=case_id,
            selection=item["label"],
            arm=arm.key,
            radius=3.0,
            sharpness=2.0,
            overwrite=True,
        )
        crop_report = CROP.process_case(
            source_path=output_path,
            stl_path=stl_path,
            arm=arm.key,
            selection=item["label"],
            case_id=case_id,
            hole_size=1_000_000.0,
            tolerance=1e-6,
            overwrite=True,
        )
        row = {
            "selection": item["label"],
            "case_id": case_id,
            "official_speed_r2": float(item["speed_r2"]),
            "stl": str(stl_path),
            "interior_visual_points": n_interior,
            "wall_points": n_wall,
            "vtp": str(output_path),
            "vtp_sha256": sha256_file(output_path),
            "surface_wall_vtp": mapping_report["output_surface_vtp"],
            "surface_coverage": mapping_report["coverage"],
            "anatomical_roi_vtp": crop_report["output_velocity_pointcloud_vtp"],
            "anatomical_roi": crop_report.get("counts", crop_report),
        }
        manifest_rows.append(row)
        print(json.dumps({"event": "v4_postview_written", **row}, ensure_ascii=False, default=str), flush=True)

    manifest = {
        "schema_version": 1,
        "created_at": utc_now(),
        "status": "completed",
        "arm": arm.key,
        "matrix_index": int(arm.index),
        "checkpoint": str(arm.checkpoint_path),
        "checkpoint_sha256": sha256_file(arm.checkpoint_path),
        "selection_metric": "official test35 metrics.speed.r2",
        "coordinate_frame": "raw CFD frame in mm",
        "point_kind": {"0": "strict-volume visualization sample", "1": "all frozen wall points"},
        "wss_valid_rule": "use wss_valid==1 or threshold point_kind==1 for WSS fields",
        "metric_basis": "visualization only; official speed R2 remains on evaluation JSON",
        "cases": manifest_rows,
    }
    path = arm_dir / "manifest.json"
    atomic_write_json(path, manifest)
    return manifest


def write_batch_readme(output_dir: Path, manifests: list[dict[str, Any]]) -> Path:
    lines = [
        "# V4 速度 R² best/worst 后处理包（ParaView）",
        "",
        f"生成时间：{utc_now()}",
        "口径对齐 2026-08-06 V2/V3 审计：`outputs/wss_pinn/audits/v2_v3_profile_secant_wss_20260806/`。",
        "排序指标：各臂 official test35 `metrics.speed.r2`（同点体域速度，不是面片插值 R²）。",
        "WSS：冻结 Profile-Secant V3，全壁面节点；可视化字段含 `wss_pred_over_wss_pred_max`。",
        "",
        "## 在 ParaView 里打开",
        "",
        "1. 看壁面 WSS / `wss_pred/wss_pred_max`：打开 `*__surface_wall.vtp`，Representation=Surface，Coloring 选标量。",
        "2. 看解剖腔内速度点云：打开 `*__anatomical_roi_velocity_pointcloud.vtp`，Representation=Points。",
        "3. 源点云（体点+壁面点混合，对照用）：打开 `best__*.vtp` / `worst__*.vtp`，用 `point_kind_0_volume_1_wall` 过滤。",
        "4. 色标可导入同目录 `GNN_blue_white_red.xml`。CFD 与 Pred 请用同一色标范围。",
        "5. 连续截面需要再把体点云转 VTU；纯 VTP 点云 Slice 只会切到稀疏点。",
        "",
        "## 推荐字段",
        "",
        "| 目的 | 字段 |",
        "| --- | --- |",
        "| 速度 | `speed_truth_m_s` / `speed_pred_m_s` / `vel_mag_cfd` / `vel_mag_pred` |",
        "| WSS 绝对值 | `wss_truth_pa` / `wss_pred_pa` / `wss_cfd` / `wss_pred` |",
        "| WSS 相对预测峰值 | `wss_pred_over_wss_pred_max` |",
        "| WSS 相对真值峰值 | `wss_truth_over_wss_truth_max` / `wss_pred_over_wss_truth_max` |",
        "",
        "面片上的归一化分母写在 VTP Field Data / `*_mapping_report.json`，不要在插值面上重算正式 R²。",
        "",
        "## 本批病例",
        "",
    ]
    for manifest in manifests:
        lines.append(
            f"### index {manifest['matrix_index']} `{manifest['arm']}`"
        )
        lines.append("")
        for case in manifest["cases"]:
            lines.append(
                f"- **{case['selection']}**: `{case['case_id']}`，official speed R²=`{case['official_speed_r2']:.6f}`"
            )
            lines.append(f"  - 面片：`{case['surface_wall_vtp']}`")
            lines.append(f"  - 速度点云：`{case['anatomical_roi_vtp']}`")
            lines.append(
                f"  - 映射覆盖率：{case['surface_coverage']['valid_ratio']:.2%}"
            )
        lines.append("")
    path = output_dir / "README_后处理打开说明.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indices", type=int, nargs="+", default=list(DEFAULT_INDICES))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--chunk-size", type=int, default=16384)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if COLORMAP.is_file():
        shutil.copy2(COLORMAP, output_dir / COLORMAP.name)

    manifests = []
    for index in args.indices:
        arm = V4WB.load_arm(index)
        print(
            json.dumps(
                {
                    "event": "v4_postview_arm_started",
                    "matrix_index": index,
                    "arm": arm.key,
                    "checkpoint": str(arm.checkpoint_path),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        manifests.append(
            export_arm(
                arm,
                args.device,
                output_dir,
                args.chunk_size,
                args.overwrite,
            )
        )

    batch = {
        "schema_version": 1,
        "created_at": utc_now(),
        "status": "completed",
        "selection_metric": "official test35 metrics.speed.r2",
        "matrix_indices": list(args.indices),
        "output_dir": str(output_dir),
        "arms": manifests,
    }
    batch_path = output_dir / "vtp" / "batch_manifest.json"
    atomic_write_json(batch_path, batch)
    readme = write_batch_readme(output_dir, manifests)
    print(
        json.dumps(
            {"event": "completed", "batch_manifest": str(batch_path), "readme": str(readme)},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
