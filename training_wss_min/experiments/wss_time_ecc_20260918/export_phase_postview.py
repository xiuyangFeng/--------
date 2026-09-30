#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T-null / T0 / TB8：固定中位病例 × 四个相位的 Gaussian 壁面后处理包。

选例：cv3 留出折 136 例，按 T0 ckpt_best 峰值帧 physical_overall_r2 取距中位数最近的一例。
相位代表帧对齐 analysis_20260919 窗口：加速 / 峰值 1162 / 减速 / 谷底。
T-null 为 D2 B_scale（折外峰值 ln × 训练折 σ 缩放），不是神经网络。
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

REPO = Path("/public/newhome/cy/Digital_twin/GNN")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.cfdpost_cloud_export.display_fields import (
    fit_selfmax, selfmax_fields, normalization_field_data,
)
from tools.cfdpost_cloud_export.map_to_stl_surface import map_csv_to_stl, _write_vtp
from tools.cfdpost_cloud_export.plot_stl_mapped_triptych import (
    _load_mesh_vtp, _plot_surface_panel, _register_gnn_colormaps, FIELD_CMAP_NAME,
)
from training_wss_min import config as C
from training_wss_min import dataset as D
from training_wss_min import metrics as M
from training_wss_min import time_basis as TB
from training_wss_min.evaluate import load_model_from_run, load_wss_stats_for_run, predict_case_norm
from training_wss_min.tools.export_wss_postview import (
    _bbox_diag, _crop_stl_to_wall, _load_stl, _write_pointcloud_vtp, _write_stl,
)

EXP = REPO / "training_wss_min/experiments/wss_time_ecc_20260918"
RUNS = REPO / "training_wss_min/runs/wss_time_ecc_20260918"
VIEW = REPO / "data_wss_v5/views_v5_1/wss_min_view_v1"
CASCADE = REPO / "data_wss_v5/views_v5_1/wss_min_cascade_v1"
POSTVIEW_XML = (
    REPO / "outputs/field/postview/v3p_i6diag_t016_report"
    / "GUO_XI_JIANG__result_features_merged-1146/GNN_blue_white_red.xml"
)
FLOOR, EPS, PEAK_INDEX = 0.05, 1e-6, 21
GAUSSIAN = dict(method="gaussian", max_dist=3.0, radius=3.0, sharpness=2.0,
                k=12, power=2.0, fallback="mask")

PHASES = (
    ("accel", 13, "加速（Q 上升、尚未到 0.8 Qpeak）"),
    ("peak", 21, "峰值帧（协议步 1162，Q/Qpeak=1）"),
    ("decel", 35, "减速射血（峰值窗之后）"),
    ("trough", 48, "谷底（协议波形最低 Q）"),
)
ARM_LABELS = (("T-null", "T-null B_scale"), ("T0", "T0 相位查询"), ("TB8", "TB8 时间基头 K=8"))


def ln_pa(tau: np.ndarray) -> np.ndarray:
    return np.log(np.clip(np.asarray(tau, dtype=np.float64), FLOOR, None) + EPS)


def load_waveform() -> Tuple[np.ndarray, np.ndarray]:
    wf = json.loads((EXP / "offline/protocol_inlet_waveform_v51.json").read_text())
    return np.asarray(wf["steps"], np.int64), np.asarray(wf["q_norm"], np.float64)


def pick_median_case() -> Dict:
    rows = []
    for fold in range(3):
        df = pd.read_csv(RUNS / f"T0_f{fold}_s1234/eval/ckpt_best/per_case_metrics.csv")
        df["fold"] = fold
        rows.append(df[["case", "fold", "physical_overall_r2", "overall_r2"]])
    t0 = pd.concat(rows, ignore_index=True)
    median = float(t0["physical_overall_r2"].median())
    t0["abs_med"] = (t0["physical_overall_r2"] - median).abs()
    pick = t0.sort_values(["abs_med", "case"]).iloc[0]
    tb = pd.concat(
        [pd.read_csv(RUNS / f"TB8_f{f}_s1234/eval/ckpt_best/per_case_metrics.csv").assign(fold=f)
         for f in range(3)],
        ignore_index=True,
    )
    tb_hit = tb.loc[tb["case"] == pick["case"]].iloc[0]
    return {
        "case": str(pick["case"]),
        "fold": int(pick["fold"]),
        "ranking_metric": "T0 ckpt_best physical_overall_r2 (peak frame 1162, held-out 136 cases)",
        "n_ranked": int(len(t0)),
        "median_r2": median,
        "T0_peak_physical_r2": float(pick["physical_overall_r2"]),
        "T0_peak_norm_r2": float(pick["overall_r2"]),
        "TB8_peak_physical_r2": float(tb_hit["physical_overall_r2"]),
        "delta_to_median": float(pick["abs_med"]),
        "r2_min": float(t0["physical_overall_r2"].min()),
        "r2_max": float(t0["physical_overall_r2"].max()),
    }


def load_model_bundle(arm: str, fold: int, device: str):
    run_dir = RUNS / f"{arm}_f{fold}_s1234"
    cfg, feat_stats, model, ckpt = load_model_from_run(run_dir, device, "best")
    wss_stats = load_wss_stats_for_run(run_dir)
    model.eval()
    return run_dir, cfg, feat_stats, wss_stats, model, ckpt


def load_eval_case(cfg: C.ExpConfig, wss_stats: Dict, canonical: str) -> Dict:
    cohort, subset, name = canonical.split("/", 2)
    return D.load_case(
        f"{cohort}/{subset}", name, wss_stats,
        target=cfg.data.target,
        target_normalization=cfg.data.target_normalization,
        data_root=cfg.data.data_root,
        required_frame_version=cfg.data.required_frame_version,
        timesteps=getattr(cfg.data, "timesteps", "peak"),
        waveform_path=getattr(cfg.data, "waveform_path", None),
        extra_point_features=C.v6_point_features(cfg),
        point_features_root=getattr(cfg.data, "point_features_root", None),
        time_basis_path=getattr(cfg.data, "time_basis_path", None),
        time_basis_k=int(getattr(cfg.model, "time_basis_k", 0)),
    )


def tnull_fields(canonical: str, fold: int, frame_index: int) -> Tuple[np.ndarray, np.ndarray]:
    with np.load(VIEW / canonical / "bundle.npz", allow_pickle=False) as z:
        ids = z["wall_node_id_cas"]
        true_pa = np.asarray(z["wall_wss"][frame_index], dtype=np.float64)
    with np.load(CASCADE / canonical / "features.npz", allow_pickle=False) as z:
        cid_ids = z["wall_node_id_cas"]
        base = z["wall_log_wss_base"].astype(np.float64)
    if not np.array_equal(ids, cid_ids):
        order = {int(v): i for i, v in enumerate(cid_ids)}
        base = base[np.array([order[int(v)] for v in ids])]
    lp = ln_pa(np.exp(base))
    stats = json.loads((EXP / "offline" / f"wss_frame_stats_fold{fold}.json").read_text())
    mu = np.asarray(stats["frame"]["log_mean"], np.float64)
    sd = np.asarray(stats["frame"]["log_std"], np.float64)
    z_peak = (lp - mu[PEAK_INDEX]) / sd[PEAK_INDEX]
    pred_ln = mu[frame_index] + sd[frame_index] * z_peak
    pred_pa = np.clip(np.exp(pred_ln) - EPS, 0.0, None)
    return true_pa, pred_pa


def predict_t0_frame(model, case, cfg, feat_stats, wss_stats, device, frame_index: int):
    D.select_frame(case, frame_index)
    pred_norm = np.asarray(
        predict_case_norm(model, case, cfg.data.input_features, feat_stats, device,
                          cfg=cfg, frame_index=frame_index),
        dtype=np.float64,
    )
    stats_k = D.frame_stats_view(wss_stats, frame_index)
    true_pa = np.asarray(case["y_raw"], dtype=np.float64)
    pred_pa = np.clip(D.denormalize_wss(pred_norm, stats_k), 0.0, None)
    return true_pa, pred_pa, pred_norm, np.asarray(case["y_norm"], dtype=np.float64)


def predict_tb8_all(model, case, cfg, feat_stats, wss_stats, device, frame_indices: List[int]):
    coef = np.asarray(
        predict_case_norm(model, case, cfg.data.input_features, feat_stats, device,
                          cfg=cfg, return_all_channels=True),
        dtype=np.float64,
    )
    basis = case["time_basis"]
    out = {}
    for k in frame_indices:
        pred_norm = TB.reconstruct_frame(coef, basis, k)
        stats_k = D.frame_stats_view(wss_stats, k)
        D.select_frame(case, k)
        true_pa = np.asarray(case["y_raw"], dtype=np.float64)
        pred_pa = np.clip(D.denormalize_wss(pred_norm, stats_k), 0.0, None)
        out[k] = (true_pa, pred_pa, np.asarray(pred_norm, dtype=np.float64),
                  np.asarray(case["y_norm"], dtype=np.float64))
    return out


def align_stl(case: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    with np.load(case["bundle_path"], allow_pickle=True) as data:
        wall_raw = data["wall_coords_raw"].astype(np.float64)
        coord_scale = float(data["coord_scale"])
        centroid = data["transform_centroid"].astype(np.float64)
        rotation = data["transform_rotation"].astype(np.float64)
    wall_xyz = case["pos"].astype(np.float64) * coord_scale
    stl_src = Path(case.get("original_stl_path", ""))
    if not stl_src.is_file():
        stl_src = REPO / "data_new" / case["cohort"] / case["case"] / f"{case['case']}.stl"
    if not stl_src.is_file():
        raise FileNotFoundError(stl_src)
    stl_raw, stl_tris = _load_stl(stl_src)
    stl_scale = float(case.get("original_stl_scale_to_mm", float("nan")))
    if not np.isfinite(stl_scale) or stl_scale <= 0:
        stl_scale = _bbox_diag(wall_raw) / max(_bbox_diag(stl_raw), 1e-12)
    err_r = float(np.abs((wall_raw - centroid) @ rotation - wall_xyz).max())
    err_rt = float(np.abs((wall_raw - centroid) @ rotation.T - wall_xyz).max())
    rotation_apply = rotation if err_r <= err_rt else rotation.T
    if min(err_r, err_rt) > 1e-2:
        raise RuntimeError(f"cannot reproduce wall frame (err {err_r:.3g}/{err_rt:.3g} mm)")
    stl_xyz = (stl_raw * stl_scale - centroid) @ rotation_apply
    crop = {"applied": False}
    if case.get("wall_crop_applied", False):
        stl_xyz, stl_tris, crop = _crop_stl_to_wall(stl_xyz, stl_tris, wall_xyz, max_dist=3.0)
    return wall_xyz, stl_xyz, stl_tris, {
        "stl_source": str(stl_src),
        "stl_scale_to_mm": stl_scale,
        "rotation_convention": "@R" if err_r <= err_rt else "@R.T",
        "repro_err_mm": min(err_r, err_rt),
        "crop": crop,
        "coord_scale": coord_scale,
    }


def physical_arrays(true_pa: np.ndarray, pred_pa: np.ndarray) -> Dict[str, np.ndarray]:
    err = pred_pa - true_pa
    return {
        "wss_cfd": true_pa.astype(np.float64),
        "wss_pred": pred_pa.astype(np.float64),
        "err_wss": err.astype(np.float64),
        "abs_err_wss": np.abs(err).astype(np.float64),
    }


def attach_fielddata(vtp_path: Path, extra_point: Dict[str, np.ndarray], field_data: Dict) -> None:
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk

    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(vtp_path))
    reader.Update()
    poly = reader.GetOutput()
    for name, values in extra_point.items():
        arr = numpy_to_vtk(np.ascontiguousarray(values, dtype=np.float32), deep=True)
        arr.SetName(name)
        poly.GetPointData().AddArray(arr)
    fd = poly.GetFieldData()
    for name, value in field_data.items():
        if isinstance(value, str):
            arr = vtk.vtkStringArray()
            arr.SetName(name)
            arr.InsertNextValue(value)
            fd.AddArray(arr)
        else:
            arr = numpy_to_vtk(np.asarray([value], dtype=np.float64), deep=True)
            arr.SetName(name)
            fd.AddArray(arr)
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(vtp_path))
    writer.SetInputData(poly)
    if writer.Write() != 1:
        raise RuntimeError(f"VTP rewrite failed: {vtp_path}")


def plot_signed_triptych(vtp_path: Path, out_png: Path, title: str, *,
                         cfd_key: str, pred_key: str, err_key: str, unit: str) -> Dict:
    import matplotlib.pyplot as plt

    _register_gnn_colormaps()
    verts, tris, scalars = _load_mesh_vtp(vtp_path)
    cfd, pred, err = scalars[cfd_key], scalars[pred_key], scalars[err_key]
    if "map_valid" in scalars:
        mask = scalars["map_valid"].astype(bool)
        cfd = np.where(mask, cfd, np.nan)
        pred = np.where(mask, pred, np.nan)
        err = np.where(mask, err, np.nan)
    valid = np.isfinite(cfd) & np.isfinite(pred) & np.isfinite(err)
    vmin = float(np.nanmin(np.concatenate([cfd[valid], pred[valid]])))
    vmax = float(np.nanmax(np.concatenate([cfd[valid], pred[valid]])))
    err_vmax = float(np.percentile(np.abs(err[valid]), 99.0)) or 1.0
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    specs = (
        (cfd, f"CFD ({unit})", vmin, vmax, FIELD_CMAP_NAME),
        (pred, f"Pred ({unit})", vmin, vmax, FIELD_CMAP_NAME),
        (err, f"Pred−CFD ({unit})", -err_vmax, err_vmax, FIELD_CMAP_NAME),
    )
    for ax, (arr, lab, lo, hi, cmap) in zip(axes, specs):
        mappable = _plot_surface_panel(
            ax, verts, tris, arr, plane=(0, 2), cmap=cmap, vmin=lo, vmax=hi,
            axis_names=("x", "y", "z"),
        )
        ax.set_title(lab, fontsize=11, fontweight="bold")
        if mappable is not None:
            fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(title, fontsize=12, y=1.02)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return {"png": str(out_png), "vmin": vmin, "vmax": vmax, "err_p99": err_vmax,
            "n_tri": int(len(tris)), "valid_vertices": int(valid.sum())}


def export_one(*, out_dir: Path, arm: str, phase: str, step: int, frame_index: int,
               q_norm: float, wall_xyz: np.ndarray, stl_path: Path,
               true_pa: np.ndarray, pred_pa: np.ndarray, canonical: str,
               extra_meta: Dict) -> Dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    export_dir = out_dir / "_export"
    surf_dir = out_dir / "surface_gaussian"
    plots_dir = out_dir / "plots"
    for d in (export_dir, surf_dir, plots_dir):
        d.mkdir(parents=True, exist_ok=True)

    scalars = physical_arrays(true_pa, pred_pa)
    contract = fit_selfmax(true_pa, pred_pa, prefix="wss",
                           denominator_scope="original colocated wall, this case this frame",
                           source_unit="Pa")
    scalars.update(selfmax_fields(true_pa, pred_pa, contract))
    pc_metrics = M.basic_metrics(true_pa, pred_pa)

    csv_path = export_dir / "same_point_fields.csv"
    pd.DataFrame({"x": wall_xyz[:, 0], "y": wall_xyz[:, 1], "z": wall_xyz[:, 2], **scalars}
                 ).to_csv(csv_path, index=False, float_format="%.8g")
    _write_pointcloud_vtp(wall_xyz, scalars, out_dir / "pointcloud.vtp")

    mapped = map_csv_to_stl(
        csv_path=csv_path, stl_path=stl_path, output_dir=surf_dir,
        scalar_columns=["wss_cfd", "wss_pred", "err_wss", "abs_err_wss"],
        report_json=out_dir / "mapping_report.json", **GAUSSIAN,
    )
    surface_vtp = Path(mapped["vtp"])
    shutil.copy2(surface_vtp, out_dir / "surface_gaussian.vtp")
    shutil.copy2(mapped["csv"], out_dir / "surface_gaussian.csv")
    if POSTVIEW_XML.is_file():
        shutil.copy2(POSTVIEW_XML, out_dir / "GNN_blue_white_red.xml")

    verts, tris, mapped_sc = _load_mesh_vtp(out_dir / "surface_gaussian.vtp")
    mapped_cfd = np.asarray(mapped_sc["wss_cfd"], dtype=np.float64)
    mapped_pred = np.asarray(mapped_sc["wss_pred"], dtype=np.float64)
    surface_selfmax = selfmax_fields(mapped_cfd, mapped_pred, contract)
    field_data = {
        **normalization_field_data(contract),
        "wss_cfd_max": float(contract["cfd_max"]),
        "wss_pred_max": float(contract["pred_max"]),
        "frame_index": float(frame_index),
        "peak_step": float(step),
        "q_norm": float(q_norm),
        "arm": arm,
        "phase": phase,
        "case": canonical,
        "n_source_points": float(len(true_pa)),
        "pointcloud_r2_pa": float(pc_metrics["r2"]),
        "pointcloud_mae_pa": float(pc_metrics["mae"]),
    }
    attach_fielddata(out_dir / "surface_gaussian.vtp", surface_selfmax, field_data)
    attach_fielddata(out_dir / "pointcloud.vtp", {}, field_data)

    pa_plot = plot_signed_triptych(
        out_dir / "surface_gaussian.vtp", plots_dir / "fig_wss_pa_triptych.png",
        f"{canonical} · {arm} · {phase} · step {step}",
        cfd_key="wss_cfd", pred_key="wss_pred", err_key="err_wss", unit="Pa",
    )
    sm_plot = plot_signed_triptych(
        out_dir / "surface_gaussian.vtp", plots_dir / "fig_wss_selfmax_triptych.png",
        f"{canonical} · {arm} · {phase} · selfmax",
        cfd_key="wss_cfd_selfmax", pred_key="wss_pred_selfmax",
        err_key="wss_selfmax_error_pred_minus_cfd", unit="selfmax",
    )
    mapping = json.loads((out_dir / "mapping_report.json").read_text())
    manifest = {
        "case": canonical, "arm": arm, "phase": phase,
        "frame_index": int(frame_index), "step": int(step), "q_norm": float(q_norm),
        "n_wall_points": int(len(true_pa)),
        "n_surface_vertices": int(len(verts)), "n_triangles": int(len(tris)),
        "interpolation": GAUSSIAN,
        "selfmax": contract,
        "pointcloud_metrics": {
            "r2": float(pc_metrics["r2"]), "mae": float(pc_metrics["mae"]),
            "rmse": float(pc_metrics.get("rmse", float("nan"))),
            "metric_basis": "same-point wall CSV; not STL-mapped surface",
        },
        "mapping_coverage": mapping.get("coverage"),
        "files": {
            "surface_gaussian": "surface_gaussian.vtp",
            "pointcloud": "pointcloud.vtp",
            "same_point_csv": "_export/same_point_fields.csv",
            "mapping_report": "mapping_report.json",
            "pa_preview": "plots/fig_wss_pa_triptych.png",
            "selfmax_preview": "plots/fig_wss_selfmax_triptych.png",
        },
        **extra_meta,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return {**manifest, "_plots": {"pa": pa_plot, "selfmax": sm_plot}}


def verify_vtp(path: Path, n_src: int | None = None) -> Dict:
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    poly = reader.GetOutput()
    n_pts = int(poly.GetNumberOfPoints())
    n_tri = int(poly.GetNumberOfPolys())
    names = [poly.GetPointData().GetArrayName(i) for i in range(poly.GetPointData().GetNumberOfArrays())]
    required = ["wss_cfd", "wss_pred"]
    missing = [k for k in required if k not in names]
    cfd = vtk_to_numpy(poly.GetPointData().GetArray("wss_cfd")) if "wss_cfd" in names else None
    pred = vtk_to_numpy(poly.GetPointData().GetArray("wss_pred")) if "wss_pred" in names else None
    finite_ok = bool(cfd is not None and pred is not None
                     and np.isfinite(cfd).mean() > 0.9 and np.isfinite(pred).mean() > 0.9)
    return {
        "path": str(path.relative_to(path.parents[4]) if len(path.parts) > 4 else path),
        "n_points": n_pts, "n_triangles": n_tri, "fields": names,
        "has_wss_cfd": "wss_cfd" in names, "has_wss_pred": "wss_pred" in names,
        "missing": missing, "finite_ok": finite_ok,
        "source_count_match": (n_src is None) or (n_pts == n_src),
        "has_faces": n_tri > 0 if path.name.startswith("surface") or "gaussian" in path.name else True,
    }


def write_readme(out_root: Path, selection: Dict, steps: np.ndarray, q: np.ndarray,
                 packages: List[Dict]) -> None:
    phase_rows = "\n".join(
        f"| `{name}` | {k} | {int(steps[k])} | {q[k]:.3f} | {desc} |"
        for name, k, desc in PHASES
    )
    files = "\n".join(
        f"| `{p['arm']}` | `{p['phase']}` | `{p['arm']}/{p['phase']}/surface_gaussian.vtp` |"
        for p in packages
    )
    text = f"""# T-null / T0 / TB8 中位病例相位后处理包

- 病例：`{selection['case']}`（cv3 fold {selection['fold']} 留出）
- 选例：{selection['ranking_metric']}
- 中位数 R²={selection['median_r2']:.4f}；本例 T0 峰值 R²={selection['T0_peak_physical_r2']:.4f}（Δ={selection['delta_to_median']:.6f}）
- checkpoint：`ckpt_best.pt`；T-null = D2 B_scale（cascade 折外峰值 × 训练折帧统计）
- 插值：Gaussian r=3 mm, sharpness=2, max_dist=3 mm
- 正式 R²/MAE 只在同点 CSV 上算，不要在面片上重算

## 相位代表帧

| 相位 | frame | step | Q/Qpeak | 定义 |
| --- | ---: | ---: | ---: | --- |
{phase_rows}

## 打开这些 Gaussian 面片

| 臂 | 相位 | VTP |
| --- | --- | --- |
{files}

ParaView：Open `surface_gaussian.vtp` → Representation=Surface → Coloring 选 `wss_cfd` / `wss_pred` / `err_wss`。CFD 与 Pred 共用同一 Data Range（Pa）。`wss_*_selfmax` 无量纲，只比分布。
"""
    (out_root / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:3")
    ap.add_argument("--output-dir", default=str(EXP / "postview_median_phases_20260920"))
    args = ap.parse_args()
    out_root = Path(args.output_dir)
    if not out_root.is_absolute():
        out_root = REPO / out_root
    out_root.mkdir(parents=True, exist_ok=True)

    device = args.device
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    print(f"[start] device={device}  out={out_root}", flush=True)

    selection = pick_median_case()
    canonical = selection["case"]
    fold = selection["fold"]
    steps, q = load_waveform()
    frame_indices = [k for _, k, _ in PHASES]
    print(f"[select] {canonical} fold={fold} T0_r2={selection['T0_peak_physical_r2']:.4f} "
          f"median={selection['median_r2']:.4f}", flush=True)

    print("[load] T0 model + case", flush=True)
    t0_run, t0_cfg, t0_feat, t0_stats, t0_model, t0_ckpt = load_model_bundle("T0", fold, device)
    t0_case = load_eval_case(t0_cfg, t0_stats, canonical)
    wall_xyz, stl_xyz, stl_tris, geom_meta = align_stl(t0_case)
    case_root = out_root / canonical.replace("/", "__")
    case_root.mkdir(parents=True, exist_ok=True)
    stl_path = case_root / "aligned_geometry.stl"
    _write_stl(stl_xyz, stl_tris, stl_path)

    print("[infer] T0 four frames", flush=True)
    t0_pred = {}
    for k in frame_indices:
        t0_pred[k] = predict_t0_frame(t0_model, t0_case, t0_cfg, t0_feat, t0_stats, device, k)
        print(f"  T0 frame {k} r2={M.r2_score(t0_pred[k][0], t0_pred[k][1]):.4f}", flush=True)
    del t0_model
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()

    print("[load] TB8 model + case", flush=True)
    tb_run, tb_cfg, tb_feat, tb_stats, tb_model, tb_ckpt = load_model_bundle("TB8", fold, device)
    tb_case = load_eval_case(tb_cfg, tb_stats, canonical)
    print("[infer] TB8 coefficients → four frames", flush=True)
    tb_pred = predict_tb8_all(tb_model, tb_case, tb_cfg, tb_feat, tb_stats, device, frame_indices)
    for k in frame_indices:
        print(f"  TB8 frame {k} r2={M.r2_score(tb_pred[k][0], tb_pred[k][1]):.4f}", flush=True)
    del tb_model
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()

    print("[build] T-null B_scale", flush=True)
    tnull_pred = {k: tnull_fields(canonical, fold, k) for k in frame_indices}
    for k in frame_indices:
        print(f"  T-null frame {k} r2={M.r2_score(tnull_pred[k][0], tnull_pred[k][1]):.4f}", flush=True)

    peak_check = abs(float(t0_pred[21][0].shape[0] and M.r2_score(*t0_pred[21][:2]))
                     - selection["T0_peak_physical_r2"])
    print(f"[check] T0 peak r2 vs CSV |Δ|={peak_check:.6g}", flush=True)

    arm_data = {
        "T-null": {"pred": tnull_pred, "run": None, "ckpt_epoch": None,
                   "definition": "D2 B_scale: cascade OOF peak ln × fold train frame μ/σ"},
        "T0": {"pred": t0_pred, "run": str(t0_run), "ckpt_epoch": int(t0_ckpt.get("epoch", -1)),
               "definition": "phase-query PointNet++ , ckpt_best, frame_stats denorm"},
        "TB8": {"pred": tb_pred, "run": str(tb_run), "ckpt_epoch": int(tb_ckpt.get("epoch", -1)),
                "definition": "time-basis K=8 reconstruct_frame, ckpt_best, frame_stats denorm"},
    }

    packages: List[Dict] = []
    for arm, _label in ARM_LABELS:
        for phase, k, _desc in PHASES:
            print(f"[export] {arm} {phase} step={int(steps[k])}", flush=True)
            true_pa, pred_pa = arm_data[arm]["pred"][k][:2]
            pack = export_one(
                out_dir=case_root / arm / phase,
                arm=arm, phase=phase, step=int(steps[k]), frame_index=k,
                q_norm=float(q[k]), wall_xyz=wall_xyz, stl_path=stl_path,
                true_pa=true_pa, pred_pa=pred_pa, canonical=canonical,
                extra_meta={
                    "run": arm_data[arm]["run"],
                    "checkpoint": "ckpt_best.pt",
                    "ckpt_epoch": arm_data[arm]["ckpt_epoch"],
                    "definition": arm_data[arm]["definition"],
                    "geometry": geom_meta,
                    "skill": "postview-surface-viz",
                },
            )
            packages.append(pack)

    listing_rows = []
    verifications = []
    for pack in packages:
        rel = Path(pack["arm"]) / pack["phase"]
        surf = case_root / rel / "surface_gaussian.vtp"
        cloud = case_root / rel / "pointcloud.vtp"
        verifications.append(verify_vtp(surf))
        verifications.append(verify_vtp(cloud, n_src=pack["n_wall_points"]))
        listing_rows.append({
            "arm": pack["arm"], "phase": pack["phase"], "step": pack["step"],
            "q_norm": pack["q_norm"],
            "surface_gaussian": str((rel / "surface_gaussian.vtp").as_posix()),
            "pointcloud": str((rel / "pointcloud.vtp").as_posix()),
            "wss_cfd": "yes", "wss_pred": "yes",
            "pointcloud_r2_pa": pack["pointcloud_metrics"]["r2"],
            "mapping_valid_ratio": (pack.get("mapping_coverage") or {}).get("valid_ratio"),
        })

    batch = {
        "experiment": "wss_time_ecc_20260918",
        "output_dir": str(out_root),
        "selection": selection,
        "phases": [
            {"name": name, "frame_index": k, "step": int(steps[k]), "q_norm": float(q[k]), "note": desc}
            for name, k, desc in PHASES
        ],
        "geometry": geom_meta,
        "n_wall_points": int(len(wall_xyz)),
        "T0_peak_r2_vs_csv_abs_delta": float(peak_check),
        "packages": [{k: v for k, v in p.items() if not k.startswith("_")} for p in packages],
        "notes": [
            "Formal R2/MAE are same-point wall metrics; Gaussian VTP is visualization only.",
            "selfmax uses original colocated CFD/Pred maxima of this case/frame; not refit on the surface.",
            "T-null is not a trained model; Pred is B_scale of cascade OOF peak prediction.",
            "cv3 held-out fold is the eval set; test34 was not used.",
        ],
    }
    (out_root / "manifest.json").write_text(json.dumps(batch, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_root / "case_selection.json").write_text(json.dumps(selection, indent=2, ensure_ascii=False), encoding="utf-8")
    with (out_root / "打开文件清单.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(listing_rows[0]))
        writer.writeheader()
        writer.writerows(listing_rows)
    failed = [v for v in verifications if v["missing"] or not v["finite_ok"]
              or (v["path"].endswith("surface_gaussian.vtp") and not v["has_faces"])]
    verification = {
        "n_checked": len(verifications), "n_failed": len(failed),
        "T0_peak_r2_vs_csv_abs_delta": float(peak_check),
        "peak_r2_match_ok": bool(peak_check < 1e-3),
        "items": verifications, "failed": failed,
    }
    (out_root / "verification.json").write_text(json.dumps(verification, indent=2, ensure_ascii=False), encoding="utf-8")
    write_readme(out_root, selection, steps, q, packages)
    shutil.copy2(out_root / "README.md", case_root / "README_打开说明.md")
    print(json.dumps({"done": str(out_root), "failed": len(failed),
                      "peak_r2_delta": peak_check}, indent=2), flush=True)
    if failed:
        raise SystemExit(f"verification failed for {len(failed)} files")


if __name__ == "__main__":
    main()
