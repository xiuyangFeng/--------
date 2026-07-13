#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""WSS-min 病例后处理可视化：完整壁面推理 → wall CSV → Gaussian 回插 STL → ParaView 包。

对齐 ``.cursor/skills/postview-surface-viz``：
  - 插值：gaussian r=3 mm, sharpness=2, max_dist=3 mm
  - 标量：wss_cfd / wss_pred / err_wss / abs_err_wss
  - 指标口径：同点 wall CSV（不在面片上算正式 R2）

坐标系：bundle 刚性配准帧（mm）；STL 先按 bbox 对角缩放再应用同一刚性变换，
与 ``visualize_sampling.py`` 一致。

示例：
  python -m training_wss_min.export_wss_postview \\
    --run-dir training_wss_min/runs/r5_a0e_b1_ctrl_s1234 \\
    --cases slow/WU_FENG_YAN,fast/RAN_QING_BO \\
    --output-dir docs/03-汇报材料/figures/WSS最小路线_20260712/postview_a0e_ctrl
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from training_wss_min import config as C
from training_wss_min import dataset as D
from training_wss_min import metrics as M
from training_wss_min.evaluate import load_model_from_run, load_wss_stats_for_run, predict_case_norm

REPO = C.PROJECT_ROOT
POSTVIEW_XML = (
    REPO / "outputs/field/postview/v3p_i6diag_t016_report"
    / "GUO_XI_JIANG__result_features_merged-1146/GNN_blue_white_red.xml"
)
SCALARS = ["wss_cfd", "wss_pred", "err_wss", "abs_err_wss"]


def _bbox_diag(xyz: np.ndarray) -> float:
    extents = xyz.max(axis=0) - xyz.min(axis=0)
    return float(np.linalg.norm(extents))


def _load_stl(stl_file: Path) -> Tuple[np.ndarray, np.ndarray]:
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    reader = vtk.vtkSTLReader()
    reader.SetFileName(str(stl_file))
    reader.Update()
    poly = reader.GetOutput()
    if poly.GetNumberOfPoints() == 0 or poly.GetNumberOfPolys() == 0:
        raise RuntimeError(f"无法读取 STL: {stl_file}")
    verts = vtk_to_numpy(poly.GetPoints().GetData()).astype(np.float64)
    tris = vtk_to_numpy(poly.GetPolys().GetData()).reshape(-1, 4)[:, 1:4].astype(np.int64)
    return verts, tris


def _write_stl(verts: np.ndarray, tris: np.ndarray, out_file: Path) -> None:
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray

    pts = vtk.vtkPoints()
    pts.SetData(numpy_to_vtk(np.ascontiguousarray(verts, dtype=np.float64), deep=True))
    poly = vtk.vtkPolyData()
    poly.SetPoints(pts)
    cells = vtk.vtkCellArray()
    packed = np.hstack([
        np.full((len(tris), 1), 3, dtype=np.int64),
        np.asarray(tris, dtype=np.int64),
    ]).ravel()
    cells.SetCells(len(tris), numpy_to_vtkIdTypeArray(packed, deep=True))
    poly.SetPolys(cells)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    writer = vtk.vtkSTLWriter()
    writer.SetFileName(str(out_file))
    writer.SetInputData(poly)
    writer.SetFileTypeToBinary()
    if writer.Write() != 1:
        raise RuntimeError(f"STL 写入失败: {out_file}")


def _write_pointcloud_vtp(xyz: np.ndarray, scalars: Dict[str, np.ndarray], out_file: Path) -> None:
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk

    pts = vtk.vtkPoints()
    pts.SetData(numpy_to_vtk(np.ascontiguousarray(xyz, dtype=np.float64), deep=True))
    poly = vtk.vtkPolyData()
    poly.SetPoints(pts)
    verts = vtk.vtkCellArray()
    for i in range(len(xyz)):
        verts.InsertNextCell(1)
        verts.InsertCellPoint(i)
    poly.SetVerts(verts)
    for name, values in scalars.items():
        arr = numpy_to_vtk(np.ascontiguousarray(values, dtype=np.float32), deep=True)
        arr.SetName(name)
        poly.GetPointData().AddArray(arr)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(out_file))
    writer.SetInputData(poly)
    if writer.Write() != 1:
        raise RuntimeError(f"VTP 写入失败: {out_file}")


def _case_label(case: Dict) -> str:
    return f"{case['cohort'].removeprefix('AG/')}/{case['case']}"


def _find_partition(split_file: Path, label: str) -> str:
    split = D.load_split(split_file)
    for part, key in (
        ("train", "train_cases"),
        ("val", "val_cases"),
        ("test", "test_cases"),
        ("excluded", "excluded_cases"),
    ):
        if label in split.get(key, []):
            return part
    return "unknown"


def _readme(case_label: str, run_name: str, peak_step: int, partition: str) -> str:
    short = case_label.split("/")[-1]
    return f"""# 后处理软件打开说明（WSS-min · ParaView）

## 病例 / 模型

- 病例：`{case_label}`（split 分区：**{partition}**）
- 模型：`{run_name}` · `ckpt_best.pt`
- 帧口径：WSS-min **peak 收缩期**（`peak_step={peak_step}`），非 V3P merged-1146
- 插值：Gaussian `r=3 mm, sharpness=2, max_dist=3 mm`
- 坐标系：bundle 刚性配准帧（mm）；STL 已变换到同一帧

## 主文件

| 文件 | 用途 |
| --- | --- |
| **`{short}__surface_wall.vtp`** | ParaView 面片云图（含 wss_cfd / wss_pred / err_wss / abs_err_wss） |
| `{short}__pointcloud_wall.vtp` | 原始壁面点云（未插值） |
| `plots/fig_wss_triptych.png` | CFD | Pred | Error 三联预览 |
| `_export/{short}__wall.csv` | 同点指标口径源（正式 R2 只在此算） |
| `GNN_blue_white_red.xml` | 蓝-白-红色标 |

## ParaView 步骤

1. Open `{short}__surface_wall.vtp` -> Apply
2. Representation = **Surface**
3. Coloring：`wss_cfd` / `wss_pred` / `err_wss` / `abs_err_wss`
4. 导入同目录 `GNN_blue_white_red.xml`；CFD 与 Pred **共用同一 Data Range**
5. 正式数字读 `_export/*__wall.csv` 或 `manifest_bundle.json` 的 `pointcloud_metrics`，不要在面片上算 R2
"""


def export_case(
    case: Dict,
    *,
    model,
    cfg: C.ExpConfig,
    feat_stats: Dict,
    wss_stats: Dict,
    device: str,
    out_root: Path,
    run_name: str,
    partition: str,
) -> Dict:
    from tools.cfdpost_cloud_export.map_to_stl_surface import map_csv_to_stl
    from tools.cfdpost_cloud_export.plot_stl_mapped_triptych import plot_triptych

    label = _case_label(case)
    short = case["case"]
    tag = label.replace("/", "__")
    case_dir = out_root / f"{tag}__peak_wss"
    export_dir = case_dir / "_export"
    surf_dir = case_dir / "surface_gaussian"
    pc_dir = case_dir / "pointcloud"
    plots_dir = case_dir / "plots"
    for d in (export_dir, surf_dir, pc_dir, plots_dir):
        d.mkdir(parents=True, exist_ok=True)

    y_pred_norm = predict_case_norm(model, case, cfg.data.input_features, feat_stats, device)
    y_pred = np.clip(D.denormalize_wss(y_pred_norm, wss_stats), 0, None).astype(np.float64)
    y_true = case["y_raw"].astype(np.float64)
    err = y_pred - y_true
    abs_err = np.abs(err)

    with np.load(case["bundle_path"], allow_pickle=True) as data:
        wall_raw = data["wall_coords_raw"].astype(np.float64)
        coord_scale = float(data["coord_scale"])
        centroid = data["transform_centroid"].astype(np.float64)
        rotation = data["transform_rotation"].astype(np.float64)
        extent_mismatch = bool(data["unit_extent_mismatch"])

    wall_xyz = case["pos"].astype(np.float64) * coord_scale
    if len(wall_xyz) != len(y_true):
        raise RuntimeError(f"{label}: xyz / n_wss 长度不一致")

    wall_csv = export_dir / f"{short}__wall.csv"
    pd.DataFrame({
        "x": wall_xyz[:, 0], "y": wall_xyz[:, 1], "z": wall_xyz[:, 2],
        "wss_cfd": y_true, "wss_pred": y_pred, "err_wss": err, "abs_err_wss": abs_err,
    }).to_csv(wall_csv, index=False, float_format="%.8g")

    _write_pointcloud_vtp(
        wall_xyz,
        {"wss_cfd": y_true, "wss_pred": y_pred, "err_wss": err, "abs_err_wss": abs_err},
        pc_dir / f"{short}__wall.vtp",
    )
    shutil.copy2(pc_dir / f"{short}__wall.vtp", case_dir / f"{short}__pointcloud_wall.vtp")

    stl_src = REPO / "data_new" / case["cohort"] / case["case"] / f"{case['case']}.stl"
    if not stl_src.is_file():
        raise FileNotFoundError(stl_src)
    stl_raw, stl_tris = _load_stl(stl_src)
    stl_scale = _bbox_diag(wall_raw) / max(_bbox_diag(stl_raw), 1e-12)
    stl_xyz = (stl_raw * stl_scale - centroid) @ rotation
    stl_aligned = case_dir / f"{short}.stl"
    _write_stl(stl_xyz, stl_tris, stl_aligned)

    mapped = map_csv_to_stl(
        csv_path=wall_csv,
        stl_path=stl_aligned,
        output_dir=surf_dir,
        method="gaussian",
        max_dist=3.0,
        radius=3.0,
        sharpness=2.0,
        k=12,
        power=2.0,
        fallback="mask",
        scalar_columns=SCALARS,
        report_json=surf_dir / f"{short}__mapping_report_wall.json",
    )
    surface_vtp = Path(mapped["vtp"])
    surface_csv = Path(mapped["csv"])
    mapping_json = Path(mapped["report"])
    shutil.copy2(surface_vtp, case_dir / f"{short}__surface_wall.vtp")
    shutil.copy2(surface_csv, case_dir / f"{short}__surface_wall.csv")
    shutil.copy2(mapping_json, case_dir / f"{short}__mapping_report.json")
    if POSTVIEW_XML.is_file():
        shutil.copy2(POSTVIEW_XML, case_dir / "GNN_blue_white_red.xml")

    triptych = plots_dir / "fig_wss_triptych.png"
    triptych_report = plot_triptych(
        vtp_path=surface_vtp,
        cfd_key="wss_cfd",
        pred_key="wss_pred",
        err_key="err_wss",
        output_path=triptych,
        title=f"{label} · peak_step={case['peak_step']} · {run_name}",
        unit="Pa",
        plane="xz",
        render="surface",
        field_cmap="GNN_BWR",
        err_cmap="GNN_BWR",
    )
    (plots_dir / "fig_wss_triptych_report.json").write_text(
        json.dumps(triptych_report, indent=2, ensure_ascii=False), encoding="utf-8",
    )

    pc_metrics = M.basic_metrics(y_true, y_pred)
    hotspot = M.hotspot_localization_metrics(y_true, y_pred, case["pos"])
    mapping_report = json.loads(mapping_json.read_text(encoding="utf-8"))
    (case_dir / "README_后处理打开说明.md").write_text(
        _readme(label, run_name, int(case["peak_step"]), partition), encoding="utf-8",
    )
    manifest = {
        "purpose": "WSS-min peak-WSS postview package (CFD | Pred | Error on STL)",
        "skill": "postview-surface-viz",
        "run": run_name,
        "checkpoint": "ckpt_best.pt",
        "case": label,
        "partition": partition,
        "peak_step": int(case["peak_step"]),
        "n_wall_points": int(len(y_true)),
        "interpolation": {
            "method": "gaussian",
            "radius_mm": 3.0,
            "sharpness": 2.0,
            "max_dist_mm": 3.0,
        },
        "coordinate_system": (
            "bundle registration frame in mm; STL bbox-scaled then rigid-transformed"
        ),
        "stl_to_pipeline_scale": float(stl_scale),
        "unit_extent_mismatch": extent_mismatch,
        "scalars": SCALARS,
        "pointcloud_metrics": {
            "r2": float(pc_metrics["r2"]),
            "mae": float(pc_metrics["mae"]),
            "rmse": float(pc_metrics.get("rmse", float("nan"))),
            "top10_iou": float(hotspot.get("top10_iou", float("nan"))),
            "metric_basis": "same-point wall CSV only; not STL-mapped surface",
        },
        "mapping_coverage": mapping_report.get("coverage"),
        "files": {
            "surface_wall_vtp": f"{short}__surface_wall.vtp",
            "pointcloud_wall_vtp": f"{short}__pointcloud_wall.vtp",
            "wall_csv": f"_export/{short}__wall.csv",
            "triptych": "plots/fig_wss_triptych.png",
            "mapping_report": f"{short}__mapping_report.json",
        },
    }
    (case_dir / "manifest_bundle.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(
        f"[ok] {label}  R2={pc_metrics['r2']:.4f}  "
        f"valid={mapping_report['coverage']['valid_ratio']:.1%}  -> {case_dir}"
    )
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description="WSS-min postview surface package export")
    ap.add_argument("--run-dir", required=True, type=str)
    ap.add_argument(
        "--cases",
        default="slow/WU_FENG_YAN,fast/RAN_QING_BO",
        help="comma-separated, e.g. slow/A,fast/B",
    )
    ap.add_argument("--output-dir", required=True, type=str)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = REPO / run_dir
    out_root = Path(args.output_dir)
    if not out_root.is_absolute():
        out_root = REPO / out_root
    out_root.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    cfg, feat_stats, model, ckpt = load_model_from_run(run_dir, device)
    wss_stats = load_wss_stats_for_run(run_dir)
    model.eval()
    split_file = Path(cfg.data.split_path)
    labels = [x.strip() for x in args.cases.split(",") if x.strip()]

    manifests: List[Dict] = []
    for label in labels:
        subset, case_name = label.split("/", 1)
        cohort_rel = f"AG/{subset}"
        bundle = C.DATA_ROOT / cohort_rel / case_name / "bundle.npz"
        if not bundle.is_file():
            raise FileNotFoundError(f"missing bundle: {bundle}")
        case = D.load_case(cohort_rel, case_name, wss_stats)
        part = _find_partition(split_file, f"{subset}/{case_name}")
        print(f"[load] {subset}/{case_name}  partition={part}  device={device}")
        manifests.append(
            export_case(
                case,
                model=model,
                cfg=cfg,
                feat_stats=feat_stats,
                wss_stats=wss_stats,
                device=device,
                out_root=out_root,
                run_name=run_dir.name,
                partition=part,
            )
        )

    batch = {
        "run": run_dir.name,
        "checkpoint": "ckpt_best.pt",
        "best_epoch": int(ckpt.get("epoch", -1)) if isinstance(ckpt, dict) else -1,
        "output_dir": str(out_root),
        "cases": manifests,
        "notes": (
            "Both cases are in dev1 train; surface VTP is for visualization only. "
            "Formal R2 is in each package manifest pointcloud_metrics. "
            "A0E-ctrl is the latest Round-5 anchor (val field/casemean 0.3587/0.2300)."
        ),
    }
    (out_root / "README.md").write_text(
        "# WSS-min postview batch\n\n"
        f"- model: `{run_dir.name}` / `ckpt_best.pt` (best epoch={batch['best_epoch']})\n"
        "- interpolation: Gaussian r=3 mm, sharpness=2 (postview-surface-viz default)\n"
        "- cases: " + ", ".join(m["case"] for m in manifests) + "\n"
        "- open each `*__surface_wall.vtp`; preview at `plots/fig_wss_triptych.png`\n"
        "- formal metrics: `_export/*__wall.csv` / `manifest_bundle.json`\n",
        encoding="utf-8",
    )
    (out_root / "batch_manifest.json").write_text(
        json.dumps(batch, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(f"[done] batch -> {out_root}")


if __name__ == "__main__":
    main()
