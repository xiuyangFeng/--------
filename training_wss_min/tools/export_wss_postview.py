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
  python -m training_wss_min.tools.export_wss_postview \\
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
from training_wss_min import surface as S
from training_wss_min.evaluate import load_model_from_run, load_wss_stats_for_run, predict_case_norm

REPO = C.PROJECT_ROOT
POSTVIEW_XML = (
    REPO / "outputs/field/postview/v3p_i6diag_t016_report"
    / "GUO_XI_JIANG__result_features_merged-1146/GNN_blue_white_red.xml"
)
# ``*_over_cfd_max`` uses one per-case CFD wall maximum for truth, prediction
# and error, so ParaView does not give the prediction a separate display range.
LEGACY_PHYSICAL_SCALARS = [
    "wss_cfd", "wss_pred", "err_wss", "abs_err_wss",
    "wss_cfd_over_cfd_max", "wss_pred_over_cfd_max",
    "err_wss_over_cfd_max", "abs_err_wss_over_cfd_max",
]
# 导师要求的“双自最大值”空间：CFD 除以 CFD max，Pred 除以 Pred max。
# 这组字段只比较病例内空间分布；不能评价绝对 WSS 幅值是否恢复。
SELF_MAX_SCALARS = [
    "wss_cfd_over_cfd_max", "wss_pred_over_pred_max",
    "err_wss_selfmax", "abs_err_wss_selfmax",
]
NORMALIZED_SCALARS = [
    "true_norm", "pred_norm", "err_norm", "abs_err_norm",
    "true_highrisk", "pred_highrisk", "highrisk_overlap",
    "highrisk_missed", "highrisk_false_positive",
]


def selfmax_scalar_data(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    eps: float = 1e-12,
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """分别按 CFD/Pred 自身病例最大值归一化，不裁剪预测。"""
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    if yt.shape != yp.shape:
        raise ValueError(f"self-max true/pred shape mismatch: {yt.shape} vs {yp.shape}")
    if not np.isfinite(yt).all() or not np.isfinite(yp).all():
        raise ValueError("self-max inputs must be finite")
    true_max = float(np.max(yt))
    pred_max = float(np.max(yp))
    if true_max <= eps:
        raise ValueError(f"CFD WSS max must be > {eps:g}, got {true_max:g}")
    if pred_max <= eps:
        raise ValueError(f"Pred WSS max must be > {eps:g}, got {pred_max:g}")
    true_selfmax = yt / true_max
    pred_selfmax = yp / pred_max
    err = pred_selfmax - true_selfmax
    return {
        "wss_cfd_over_cfd_max": true_selfmax,
        "wss_pred_over_pred_max": pred_selfmax,
        "err_wss_selfmax": err,
        "abs_err_wss_selfmax": np.abs(err),
    }, {
        "wss_cfd_max": true_max,
        "wss_pred_max": pred_max,
        "wss_pred_max_over_cfd_max": pred_max / true_max,
        "pred_below_zero_fraction": float(np.mean(yp < 0.0)),
    }


def build_postview_hotspot_payload(
    case: Dict,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    surface_metric_mode: str,
) -> Dict:
    """Build PostView masks without making vertex-only exports consume STL area.

    ``legacy_vertex`` keeps the historical top-10%-by-point masks.  Only
    ``both_strict`` loads mapped surface-area weights and emits area-labelled
    metrics.  The generic ``true_highrisk``/``pred_highrisk`` scalar names are
    retained for old ParaView readers, while the manifest records their basis.
    """
    if surface_metric_mode not in {"legacy_vertex", "both_strict"}:
        raise ValueError(f"unsupported surface_metric_mode={surface_metric_mode!r}")

    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    legacy_true_thr = float(np.percentile(yt, 90.0))
    legacy_pred_thr = float(np.percentile(yp, 90.0))
    legacy_true = yt >= legacy_true_thr
    legacy_pred = yp >= legacy_pred_thr
    legacy_hotspot = M.hotspot_localization_metrics(yt, yp, case["pos"])

    area_hotspot = None
    area_report = None
    if surface_metric_mode == "both_strict":
        if "surface_area_weights" not in case:
            case["surface_area_weights"], case["surface_area_report"] = \
                S.area_weights_for_case(case, strict=True)
        weights = case["surface_area_weights"]
        selected_true = M.area_top_fraction_mask(yt, weights)
        selected_pred = M.area_top_fraction_mask(yp, weights)
        area_hotspot = M.area_hotspot_metrics(yt, yp, case["pos"], weights)
        area_report = case.get("surface_area_report")
        basis = "surface_area"
    else:
        selected_true = legacy_true
        selected_pred = legacy_pred
        basis = "legacy_vertex"

    selected_overlap = selected_true & selected_pred
    scalars = {
        "true_highrisk": selected_true.astype(np.float32),
        "pred_highrisk": selected_pred.astype(np.float32),
        "highrisk_overlap": selected_overlap.astype(np.float32),
        "highrisk_missed": (selected_true & ~selected_pred).astype(np.float32),
        "highrisk_false_positive": (selected_pred & ~selected_true).astype(np.float32),
        "legacy_vertex_true_highrisk": legacy_true.astype(np.float32),
        "legacy_vertex_pred_highrisk": legacy_pred.astype(np.float32),
        "legacy_vertex_highrisk_overlap": (legacy_true & legacy_pred).astype(np.float32),
    }
    return {
        "basis": basis,
        "scalars": scalars,
        "selected_true": selected_true,
        "selected_pred": selected_pred,
        "legacy_hotspot": legacy_hotspot,
        "area_hotspot": area_hotspot,
        "surface_area_report": area_report,
    }


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


def _crop_stl_to_wall(verts: np.ndarray, tris: np.ndarray, wall_xyz: np.ndarray,
                      max_dist: float = 3.0) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Remove original-STL tail triangles outside an explicitly cropped CFD wall."""
    from scipy.spatial import cKDTree

    dist = cKDTree(wall_xyz).query(verts, k=1)[0]
    keep_tri = np.all(dist[tris] <= max_dist, axis=1)
    kept = tris[keep_tri]
    if not len(kept):
        raise RuntimeError("STL crop removed every triangle; frame/path mapping is invalid")
    used = np.unique(kept)
    remap = np.full(len(verts), -1, dtype=np.int64)
    remap[used] = np.arange(len(used), dtype=np.int64)
    report = {
        "applied": True,
        "max_dist_mm": float(max_dist),
        "vertices_before": int(len(verts)),
        "vertices_after": int(len(used)),
        "triangles_before": int(len(tris)),
        "triangles_after": int(len(kept)),
    }
    return verts[used], remap[kept], report


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
    return D.canonical_unit_id(case.get("unit_id", f"{case['cohort']}/{case['case']}"))


def _resolve_bundle_path(data_root: str | Path, label: str) -> Tuple[str, Path]:
    """Resolve legacy AG and canonical AG/AAA IDs without injecting a cohort prefix."""
    canonical = D.canonical_unit_id(label)
    cohort, subset, case_name = canonical.split("/", 2)
    return canonical, Path(data_root) / cohort / subset / case_name / "bundle.npz"


def _existing_complete_manifest(
    out_root: Path,
    canonical: str,
    surface_metric_mode: str | None = None,
) -> Tuple[Dict, Path] | None:
    tags = [canonical.replace("/", "__")]
    if canonical.startswith("AG/"):
        tags.append(canonical.removeprefix("AG/").replace("/", "__"))
    for tag in tags:
        case_dir = out_root / f"{tag}__peak_wss"
        manifest_path = case_dir / "manifest_bundle.json"
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (surface_metric_mode is not None
                and manifest.get("surface_metric_mode") != surface_metric_mode):
            # A partial package written before the metric-mode split may contain
            # area-labelled masks even for a vertex-only run.  Do not resume it.
            continue
        required = [case_dir / value for value in manifest.get("files", {}).values()]
        coverage = (manifest.get("mapping_coverage") or {}).get("valid_ratio")
        if (required and all(path.is_file() for path in required)
                and coverage is not None and float(coverage) >= 0.999999):
            manifest["case"] = canonical
            return manifest, case_dir
    return None


def _find_partition(split_file: Path, label: str) -> str:
    split = D.load_split(split_file)
    wanted = D.canonical_unit_id(label)
    for part, key in (
        ("train", "train_cases"),
        ("val", "val_cases"),
        ("test", "test_cases"),
        ("excluded", "excluded_cases"),
    ):
        if any(D.canonical_unit_id(item) == wanted for item in split.get(key, [])):
            return part
    return "unknown"


def _readme(case_label: str, run_name: str, peak_step: int, partition: str,
            checkpoint: str, normalization_mode: str,
            surface_metric_mode: str) -> str:
    short = case_label.split("/")[-1]
    return f"""# 后处理软件打开说明（WSS-min · ParaView）

## 病例 / 模型

- 病例：`{case_label}`（split 分区：**{partition}**）
- 模型：`{run_name}` · `ckpt_{checkpoint}.pt`
- 目标空间：`{normalization_mode}`；正式指标使用同点 `true_norm/pred_norm`
- high-risk 口径：`{surface_metric_mode}`；`legacy_vertex` 不读取或伪造面积权重
- 帧口径：WSS-min **peak 收缩期**（`peak_step={peak_step}`），非 V3P merged-1146
- 插值：Gaussian `r=3 mm, sharpness=2, max_dist=3 mm`
- 坐标系：bundle 刚性配准帧（mm）；STL 已变换到同一帧

## 主文件

| 文件 | 用途 |
| --- | --- |
| **`{short}__surface_wall.vtp`** | ParaView 面片云图（normalized truth/pred/error/high-risk mask） |
| `{short}__pointcloud_wall.vtp` | 原始壁面点云（未插值） |
| `plots/fig_wss_triptych.png` | CFD | Pred | Error 三联预览 |
| `plots/fig_wss_selfmax_triptych.png` | CFD/CFDmax | Pred/Predmax | Error 三联预览 |
| `_export/{short}__wall.csv` | 同点指标口径源（正式 R2 只在此算） |
| `GNN_blue_white_red.xml` | 蓝-白-红色标 |

## ParaView 步骤

1. Open `{short}__surface_wall.vtp` -> Apply
2. Representation = **Surface**
3. Coloring：`true_norm` / `pred_norm` / `err_norm` / `abs_err_norm` /
   `true_highrisk` / `pred_highrisk` / `wss_cfd_over_cfd_max` /
   `wss_pred_over_pred_max` / `err_wss_selfmax`
4. 导入同目录 `GNN_blue_white_red.xml`；CFD 与 Pred **共用同一 Data Range**
5. 正式数字读 `_export/*__wall.csv` 或 `manifest_bundle.json` 的 `pointcloud_metrics`，不要在面片上算 R2
"""


def _plot_top10_overlay(pos: np.ndarray, true_mask: np.ndarray, pred_mask: np.ndarray,
                        out_file: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    ax.scatter(pos[:, 2], pos[:, 0], s=1, c="#d0d0d0", alpha=0.35, label="wall")
    overlap = true_mask & pred_mask
    missed = true_mask & ~pred_mask
    false_positive = pred_mask & ~true_mask
    ax.scatter(pos[overlap, 2], pos[overlap, 0], s=7, c="#7b2cbf",
               alpha=0.8, label="overlap")
    ax.scatter(pos[missed, 2], pos[missed, 0], s=6, c="#d62728",
               alpha=0.8, label="missed true")
    ax.scatter(pos[false_positive, 2], pos[false_positive, 0], s=5, c="#1f77b4",
               alpha=0.8, label="false positive")
    ax.set_aspect("equal")
    ax.set_xlabel("z")
    ax.set_ylabel("x")
    ax.set_title(title)
    ax.legend(loc="best", markerscale=3)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=140)
    plt.close(fig)


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
    checkpoint: str = "best",
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

    pred_norm = np.asarray(
        predict_case_norm(model, case, cfg.data.input_features, feat_stats, device, cfg=cfg),
        dtype=np.float64,
    )
    true_norm = np.asarray(case["y_norm"], dtype=np.float64)
    err_norm = pred_norm - true_norm
    abs_err_norm = np.abs(err_norm)
    hotspot_payload = build_postview_hotspot_payload(
        case, true_norm, pred_norm, cfg.eval.surface_metric_mode,
    )
    true_highrisk = hotspot_payload["selected_true"]
    pred_highrisk = hotspot_payload["selected_pred"]

    scalar_data: Dict[str, np.ndarray] = {
        "true_norm": true_norm,
        "pred_norm": pred_norm,
        "err_norm": err_norm,
        "abs_err_norm": abs_err_norm,
        **hotspot_payload["scalars"],
    }
    if cfg.data.target_normalization == "global_stats":
        y_true = case["y_raw"].astype(np.float64)
        y_pred = D.denormalize_wss(pred_norm, wss_stats).astype(np.float64)
        if wss_stats.get("method") == "log_z":
            y_pred = np.clip(y_pred, 0, None)
        err = y_pred - y_true
        abs_err = np.abs(err)
        wss_cfd_max = max(float(np.max(y_true)), 1e-12)
        scalar_data.update({
            "wss_cfd": y_true, "wss_pred": y_pred,
            "err_wss": err, "abs_err_wss": abs_err,
            "wss_cfd_over_cfd_max": y_true / wss_cfd_max,
            "wss_pred_over_cfd_max": y_pred / wss_cfd_max,
            "err_wss_over_cfd_max": err / wss_cfd_max,
            "abs_err_wss_over_cfd_max": abs_err / wss_cfd_max,
        })

    # GLOBAL 使用已恢复的物理 WSS；CASE 的未知病例尺度在“各除以自身 max”
    # 后严格抵消，因此可直接用 true_norm/pred_norm 计算而无需恢复 Pa。
    selfmax_source_true = scalar_data.get("wss_cfd", true_norm)
    selfmax_source_pred = scalar_data.get("wss_pred", pred_norm)
    selfmax_scalars, selfmax_denominators = selfmax_scalar_data(
        selfmax_source_true, selfmax_source_pred,
    )
    scalar_data.update(selfmax_scalars)

    with np.load(case["bundle_path"], allow_pickle=True) as data:
        wall_raw = data["wall_coords_raw"].astype(np.float64)
        coord_scale = float(data["coord_scale"])
        centroid = data["transform_centroid"].astype(np.float64)
        rotation = data["transform_rotation"].astype(np.float64)
        extent_mismatch = bool(data["unit_extent_mismatch"])

    wall_xyz = case["pos"].astype(np.float64) * coord_scale
    if len(wall_xyz) != len(true_norm):
        raise RuntimeError(f"{label}: xyz / n_wss 长度不一致")

    wall_csv = export_dir / f"{short}__wall.csv"
    wall_columns = {
        "x": wall_xyz[:, 0], "y": wall_xyz[:, 1], "z": wall_xyz[:, 2],
        **scalar_data,
    }
    pd.DataFrame(wall_columns).to_csv(wall_csv, index=False, float_format="%.8g")

    _write_pointcloud_vtp(
        wall_xyz,
        scalar_data,
        pc_dir / f"{short}__wall.vtp",
    )
    shutil.copy2(pc_dir / f"{short}__wall.vtp", case_dir / f"{short}__pointcloud_wall.vtp")

    stl_src = Path(case.get("original_stl_path", ""))
    if not stl_src.is_file():
        stl_src = REPO / "data_new" / case["cohort"] / case["case"] / f"{case['case']}.stl"
    if not stl_src.is_file():
        raise FileNotFoundError(stl_src)
    stl_raw, stl_tris = _load_stl(stl_src)
    stl_scale = float(case.get("original_stl_scale_to_mm", float("nan")))
    if not np.isfinite(stl_scale) or stl_scale <= 0:
        stl_scale = _bbox_diag(wall_raw) / max(_bbox_diag(stl_raw), 1e-12)
    stl_xyz = (stl_raw * stl_scale - centroid) @ rotation
    crop_report = {"applied": False}
    if case.get("wall_crop_applied", False):
        stl_xyz, stl_tris, crop_report = _crop_stl_to_wall(
            stl_xyz, stl_tris, wall_xyz, max_dist=3.0,
        )
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
        scalar_columns=list(scalar_data),
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
        cfd_key="true_norm",
        pred_key="pred_norm",
        err_key="err_norm",
        output_path=triptych,
        title=f"{label} · peak_step={case['peak_step']} · {run_name}",
        unit="normalized",
        plane="xz",
        render="surface",
        field_cmap="GNN_BWR",
        err_cmap="GNN_BWR",
    )
    (plots_dir / "fig_wss_triptych_report.json").write_text(
        json.dumps(triptych_report, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    selfmax_triptych = plots_dir / "fig_wss_selfmax_triptych.png"
    selfmax_triptych_report = plot_triptych(
        vtp_path=surface_vtp,
        cfd_key="wss_cfd_over_cfd_max",
        pred_key="wss_pred_over_pred_max",
        err_key="err_wss_selfmax",
        output_path=selfmax_triptych,
        title=f"{label} · CFD/CFDmax vs Pred/Predmax · {run_name}",
        unit="self-max normalized",
        plane="xz",
        render="surface",
        field_cmap="GNN_BWR",
        err_cmap="GNN_BWR",
    )
    (plots_dir / "fig_wss_selfmax_triptych_report.json").write_text(
        json.dumps(selfmax_triptych_report, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    _plot_top10_overlay(
        case["pos"], true_highrisk, pred_highrisk,
        plots_dir / "fig_top10_overlay.png",
        f"{label} · {hotspot_payload['basis']} top10%",
    )

    pc_metrics = M.basic_metrics(true_norm, pred_norm)
    selfmax_metrics = M.basic_metrics(
        selfmax_scalars["wss_cfd_over_cfd_max"],
        selfmax_scalars["wss_pred_over_pred_max"],
    )
    hotspot = hotspot_payload["legacy_hotspot"]
    area_hotspot = hotspot_payload["area_hotspot"]
    mapping_report = json.loads(mapping_json.read_text(encoding="utf-8"))
    (case_dir / "README_后处理打开说明.md").write_text(
        _readme(
            label, run_name, int(case["peak_step"]), partition, checkpoint,
            cfg.data.target_normalization, cfg.eval.surface_metric_mode,
        ), encoding="utf-8",
    )
    pointcloud_metrics = {
        "r2": float(pc_metrics["r2"]),
        "mae": float(pc_metrics["mae"]),
        "rmse": float(pc_metrics.get("rmse", float("nan"))),
        "top10_iou": float(hotspot.get("top10_iou", float("nan"))),
        "legacy_vertex_top10_iou": float(hotspot.get("top10_iou", float("nan"))),
        "peak_point_dist_over_bbox": float(
            hotspot.get("peak_point_dist_over_bbox", float("nan"))
        ),
        "hotspot_centroid_dist_over_bbox": float(
            hotspot.get("hotspot_centroid_dist_over_bbox", float("nan"))
        ),
        "metric_basis": "same-point wall CSV only; not STL-mapped surface",
    }
    if area_hotspot is not None:
        pointcloud_metrics.update({
            "area_top10_iou": float(area_hotspot.get("area_top10_iou", float("nan"))),
            "area_top10_precision": float(
                area_hotspot.get("area_top10_precision", float("nan"))
            ),
            "area_top10_recall": float(
                area_hotspot.get("area_top10_recall", float("nan"))
            ),
            "area_hotspot_centroid_dist": float(
                area_hotspot.get("area_hotspot_centroid_dist", float("nan"))
            ),
        })
    manifest = {
        "purpose": "WSS-min normalized target postview package (truth | pred | error on STL)",
        "skill": "postview-surface-viz",
        "run": run_name,
        "checkpoint": f"ckpt_{checkpoint}.pt",
        "case": label,
        "partition": partition,
        "peak_step": int(case["peak_step"]),
        "n_wall_points": int(len(true_norm)),
        "surface_metric_mode": cfg.eval.surface_metric_mode,
        "highrisk_mask_basis": hotspot_payload["basis"],
        "surface_area_metrics_status": (
            "complete_strict" if area_hotspot is not None else "not_requested"
        ),
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
        "stl_crop_to_cfd_wall": crop_report,
        "unit_extent_mismatch": extent_mismatch,
        "scalars": list(scalar_data),
        "normalization": {
            **case.get("target_normalization_meta", {}),
            "mode": cfg.data.target_normalization,
            "prediction_requires_case_wss_max": False,
            "physical_recovery": (
                "available from frozen train-only global stats"
                if cfg.data.target_normalization == "global_stats"
                else "disabled; would require an externally supplied or separately predicted case scale"
            ),
        },
        "pointcloud_metrics": pointcloud_metrics,
        "selfmax_metrics": {
            "definition": "WSS_CFD/WSS_CFD,max vs WSS_Pred/WSS_Pred,max",
            "r2": float(selfmax_metrics["r2"]),
            "mae": float(selfmax_metrics["mae"]),
            "rmse": float(selfmax_metrics.get("rmse", float("nan"))),
            **selfmax_denominators,
            "metric_basis": "same-point wall CSV; prediction is not clipped",
            "interpretation": "spatial distribution only; absolute WSS amplitude removed",
        },
        "mapping_coverage": mapping_report.get("coverage"),
        "files": {
            "surface_wall_vtp": f"{short}__surface_wall.vtp",
            "pointcloud_wall_vtp": f"{short}__pointcloud_wall.vtp",
            "wall_csv": f"_export/{short}__wall.csv",
            "triptych": "plots/fig_wss_triptych.png",
            "selfmax_triptych": "plots/fig_wss_selfmax_triptych.png",
            "top10_overlay": "plots/fig_top10_overlay.png",
            "mapping_report": f"{short}__mapping_report.json",
        },
    }
    if hotspot_payload["surface_area_report"] is not None:
        manifest["surface_area_mapping"] = hotspot_payload["surface_area_report"]
    (case_dir / "manifest_bundle.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    (case_dir / "normalization.json").write_text(
        json.dumps(manifest["normalization"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        f"[ok] {label}  R2={pc_metrics['r2']:.4f}  "
        f"valid={mapping_report['coverage']['valid_ratio']:.1%}  -> {case_dir}"
    )
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description="WSS-min postview surface package export")
    ap.add_argument("--run-dir", required=True, type=str)
    selector = ap.add_mutually_exclusive_group(required=True)
    selector.add_argument(
        "--cases",
        default=None,
        help="comma-separated, e.g. slow/A,fast/B",
    )
    selector.add_argument("--partition", choices=("train", "val", "test"),
                          help="export every case in one split partition")
    ap.add_argument("--allow-test", action="store_true",
                    help="explicitly unlock export of the config's pre-authorized test partition")
    ap.add_argument("--checkpoint", choices=("best", "last"), default="best")
    ap.add_argument("--output-dir", required=True, type=str)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--resume", action="store_true",
                    help="reuse complete existing case packages and export only missing cases")
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
    cfg, feat_stats, model, ckpt = load_model_from_run(run_dir, device, args.checkpoint)
    wss_stats = load_wss_stats_for_run(run_dir)
    model.eval()
    split_file = Path(cfg.data.split_path)
    if not split_file.is_absolute():
        split_file = REPO / split_file
    if args.partition:
        labels = [
            f"{cohort}/{case}"
            for cohort, case in D.load_split_cases(split_file, args.partition)
        ]
    else:
        labels = [x.strip() for x in args.cases.split(",") if x.strip()]
    if not labels:
        raise ValueError("selected no cases for postview export")

    manifests: List[Dict] = []
    for label in labels:
        canonical, bundle = _resolve_bundle_path(cfg.data.data_root, label)
        cohort, subset, case_name = canonical.split("/", 2)
        cohort_rel = f"{cohort}/{subset}"
        if not bundle.is_file():
            raise FileNotFoundError(f"missing bundle: {bundle}")
        part = _find_partition(split_file, canonical)
        if part == "test" and not args.allow_test:
            raise PermissionError("test export requires --allow-test")
        if args.resume:
            existing = _existing_complete_manifest(
                out_root, canonical, surface_metric_mode=cfg.eval.surface_metric_mode
            )
            if existing is not None:
                manifest, case_dir = existing
                manifests.append(manifest)
                print(f"[resume] {canonical} -> {case_dir}")
                continue
        case = D.load_case(
            cohort_rel, case_name, wss_stats, target=cfg.data.target,
            target_normalization=cfg.data.target_normalization,
            data_root=cfg.data.data_root,
            required_frame_version=cfg.data.required_frame_version,
        )
        print(f"[load] {canonical}  partition={part}  device={device}")
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
                checkpoint=args.checkpoint,
            )
        )

    batch = {
        "run": run_dir.name,
        "checkpoint": f"ckpt_{args.checkpoint}.pt",
        "best_epoch": int(ckpt.get("epoch", -1)) if isinstance(ckpt, dict) else -1,
        "output_dir": str(out_root),
        "surface_metric_mode": cfg.eval.surface_metric_mode,
        "cases": manifests,
        "notes": (
            "Surface VTP is for visualization only. Formal R2 is in each package "
            "manifest pointcloud_metrics and the run-level eval/metrics.json."
        ),
    }
    (out_root / "README.md").write_text(
        "# WSS-min postview batch\n\n"
        f"- model: `{run_dir.name}` / `ckpt_{args.checkpoint}.pt` (epoch={batch['best_epoch']})\n"
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
