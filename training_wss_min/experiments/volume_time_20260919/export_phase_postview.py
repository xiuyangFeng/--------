#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""体场 PT0/PTB8、VT0/VTB4：与 WSS 中位病例同一例 × 四个相位的 ParaView 包。

选例固定为 wss_time_ecc_20260918/postview_median_phases_20260920 的
AAA/unruputer/SHEN_FANG_JIN（cv3 fold2 留出），不按体场 R² 重排。
速度只写体内点云（含向量）；压力写完整/壁面/体内点云，以及 wall-only Gaussian 面片和 native 壁面。
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
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
from training_wss_min.evaluate import (
    denormalize_volume_prediction, load_model_from_run, load_wss_stats_for_run, predict_case_norm,
)
from training_wss_min.tools.add_cfd_wall_mesh_vtp import write_mesh_vtp
from training_wss_min.tools.export_wss_postview import (
    _bbox_diag, _crop_stl_to_wall, _load_stl, _write_stl,
)
from wss_v5 import contract as VC

EXP = REPO / "training_wss_min/experiments/volume_time_20260919"
WSS_POST = REPO / "training_wss_min/experiments/wss_time_ecc_20260918/postview_median_phases_20260920"
RUNS = REPO / "training_wss_min/runs/volume_time_20260919"
POSTVIEW_XML = (
    REPO / "outputs/field/postview/v3p_i6diag_t016_report"
    / "GUO_XI_JIANG__result_features_merged-1146/GNN_blue_white_red.xml"
)
GAUSSIAN = dict(method="gaussian", max_dist=3.0, radius=3.0, sharpness=2.0,
                k=12, power=2.0, fallback="mask")
PHASES = (
    ("accel", 13, "加速（Q 上升、尚未到 0.8 Qpeak）"),
    ("peak", 21, "峰值帧（协议步 1162，Q/Qpeak=1）"),
    ("decel", 35, "减速射血（峰值窗之后）"),
    ("trough", 48, "谷底（协议波形最低 Q）"),
)
PRESSURE_ARMS = ("PT0", "PTB8")
VELOCITY_ARMS = ("VT0", "VTB4")
PEAK_CSV = {
    "PT0": 0.6542682668051394,
    "PTB8": 0.6020723792017169,
    "VT0": 0.8278360926536754,
    "VTB4": 0.7703768726358045,
}


def load_waveform() -> Tuple[np.ndarray, np.ndarray]:
    wf = json.loads((REPO / "training_wss_min/experiments/wss_time_ecc_20260918"
                     / "offline/protocol_inlet_waveform_v51.json").read_text())
    return np.asarray(wf["steps"], np.int64), np.asarray(wf["q_norm"], np.float64)


def wss_selection() -> Dict:
    selection = json.loads((WSS_POST / "case_selection.json").read_text())
    selection["copied_from"] = str(WSS_POST / "case_selection.json")
    selection["volume_note"] = (
        "Same case as the WSS median-phase package; not re-ranked on volume R2."
    )
    return selection


def load_model_bundle(arm: str, fold: int, device: str):
    run_dir = RUNS / f"{arm}_f{fold}_s1234"
    cfg, feat_stats, model, ckpt = load_model_from_run(run_dir, device, "best")
    stats = load_wss_stats_for_run(run_dir)
    model.eval()
    return run_dir, cfg, feat_stats, stats, model, ckpt


def load_eval_case(cfg: C.ExpConfig, stats: Dict, canonical: str) -> Dict:
    cohort, subset, name = canonical.split("/", 2)
    return D.load_case(
        f"{cohort}/{subset}", name, stats,
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
        volume_time_sidecar_root=getattr(cfg.data, "volume_time_sidecar_root", None),
        volume_h5_root=getattr(cfg.data, "volume_h5_root", None),
    )


def predict_t0_frames(model, case, cfg, feat_stats, stats, device, frame_indices: List[int], target: str):
    out = {}
    for k in frame_indices:
        D.select_frame(case, k)
        pred_norm = np.asarray(
            predict_case_norm(
                model, case, cfg.data.input_features, feat_stats, device,
                cfg=cfg, return_all_channels=(target == "velocity"), frame_index=k,
            ),
            dtype=np.float64,
        )
        pred = denormalize_volume_prediction(pred_norm, target, D.frame_stats_view(stats, k))
        rows = D.query_rows(case)
        truth = np.asarray(case["y_raw"][rows], dtype=np.float64)
        if pred.shape != truth.shape:
            raise RuntimeError(f"shape mismatch frame {k}: pred {pred.shape} truth {truth.shape}")
        out[k] = {"truth": truth, "pred": pred, "rows": rows.copy(), "p_ref": _p_ref(case, k)}
        print(f"    frame {k} r2={_scalar_r2(target, truth, pred):.4f}", flush=True)
    return out


def predict_tb_frames(model, case, cfg, feat_stats, stats, device, frame_indices: List[int], target: str):
    coef = np.asarray(
        predict_case_norm(
            model, case, cfg.data.input_features, feat_stats, device,
            cfg=cfg, return_all_channels=True,
        ),
        dtype=np.float64,
    )
    basis = case["time_basis"]
    rows = D.query_rows(case)
    out = {}
    for k in frame_indices:
        pred_norm = TB.reconstruct_frame(coef, basis, k)
        pred = denormalize_volume_prediction(pred_norm, target, D.frame_stats_view(stats, k))
        D.select_frame(case, k)
        truth = np.asarray(case["y_raw"][rows], dtype=np.float64)
        if pred.shape != truth.shape:
            raise RuntimeError(f"TB shape mismatch frame {k}: pred {pred.shape} truth {truth.shape}")
        out[k] = {"truth": truth, "pred": pred, "rows": rows.copy(), "p_ref": _p_ref(case, k)}
        print(f"    frame {k} r2={_scalar_r2(target, truth, pred):.4f}", flush=True)
    return out


def _p_ref(case: Dict, frame_index: int) -> float:
    src = case.get("volume_frames")
    if src is None:
        return float("nan")
    return float(np.asarray(src.p_volume_mean_pa, dtype=np.float64)[int(frame_index)])


def _scalar_r2(target: str, truth: np.ndarray, pred: np.ndarray) -> float:
    if target == "velocity":
        return float(M.basic_metrics(np.linalg.norm(truth, axis=1), np.linalg.norm(pred, axis=1))["r2"])
    return float(M.basic_metrics(truth, pred)["r2"])


def align_wall(case: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    n_wall = int(case["n_wall"])
    with np.load(case["bundle_path"], allow_pickle=True) as data:
        wall_raw = data["wall_coords_raw"].astype(np.float64)
        coord_scale = float(data["coord_scale"])
        centroid = data["transform_centroid"].astype(np.float64)
        rotation = data["transform_rotation"].astype(np.float64)
    full_xyz = case["pos"].astype(np.float64) * coord_scale
    wall_xyz = full_xyz[:n_wall]
    if len(wall_raw) != n_wall:
        raise RuntimeError(f"wall_coords_raw {len(wall_raw)} != n_wall {n_wall}")
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
    return wall_xyz, full_xyz, stl_xyz, stl_tris, {
        "stl_source": str(stl_src),
        "stl_scale_to_mm": stl_scale,
        "rotation_convention": "@R" if err_r <= err_rt else "@R.T",
        "repro_err_mm": min(err_r, err_rt),
        "crop": crop,
        "coord_scale": coord_scale,
        "n_wall": n_wall,
        "n_full": int(len(full_xyz)),
    }


def wall_triangles(canonical: str, wall_xyz: np.ndarray) -> np.ndarray:
    h5_path = VC.case_dir(canonical) / "case.h5"
    with h5py.File(h5_path, "r") as h5:
        valid = h5["wall_static/valid"][()].astype(bool)
        tri = h5["topology/wall_triangles"][()].astype(np.int64)
    if int(valid.sum()) != len(wall_xyz):
        raise RuntimeError(f"{canonical}: valid wall {int(valid.sum())} != {len(wall_xyz)}")
    remap = np.full(len(valid), -1, dtype=np.int64)
    remap[valid] = np.arange(valid.sum())
    tri = remap[tri]
    return tri[(tri >= 0).all(axis=1)]


def write_points(path: Path, xyz: np.ndarray, arrays: Dict[str, np.ndarray]) -> None:
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy

    xyz = np.ascontiguousarray(xyz, dtype=np.float64)
    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(xyz, deep=True))
    poly = vtk.vtkPolyData()
    poly.SetPoints(points)
    cells = vtk.vtkCellArray()
    cells.SetData(
        numpy_to_vtkIdTypeArray(np.arange(len(xyz) + 1, dtype=np.int64), deep=True),
        numpy_to_vtkIdTypeArray(np.arange(len(xyz), dtype=np.int64), deep=True),
    )
    poly.SetVerts(cells)
    for name, data in arrays.items():
        arr_np = np.ascontiguousarray(data)
        if arr_np.dtype == np.bool_ or arr_np.dtype == np.uint8:
            arr_np = arr_np.astype(np.uint8)
        elif np.issubdtype(arr_np.dtype, np.integer):
            arr_np = arr_np.astype(np.int64 if arr_np.ndim == 1 else arr_np.dtype)
        else:
            arr_np = arr_np.astype(np.float32)
        arr = numpy_to_vtk(arr_np, deep=True)
        arr.SetName(name)
        poly.GetPointData().AddArray(arr)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(poly)
    writer.SetDataModeToBinary()
    if writer.Write() != 1:
        raise RuntimeError(f"VTP write failed: {path}")
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    loaded = reader.GetOutput()
    if int(loaded.GetNumberOfPoints()) != len(xyz):
        raise RuntimeError(f"VTP reread point count mismatch: {path}")
    if not np.allclose(vtk_to_numpy(loaded.GetPoints().GetData()), xyz, atol=1e-8):
        raise RuntimeError(f"VTP coordinate mismatch: {path}")


def attach_fielddata(vtp_path: Path, extra_point: Dict[str, np.ndarray], field_data: Dict) -> None:
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk

    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(vtp_path))
    reader.Update()
    poly = reader.GetOutput()
    for name, values in extra_point.items():
        arr_np = np.ascontiguousarray(values)
        if arr_np.dtype == np.uint8:
            arr = numpy_to_vtk(arr_np, deep=True)
        else:
            arr = numpy_to_vtk(arr_np.astype(np.float32), deep=True)
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
    writer.SetDataModeToBinary()
    if writer.Write() != 1:
        raise RuntimeError(f"VTP rewrite failed: {vtp_path}")


def plot_signed_triptych(vtp_path: Path, out_png: Path, title: str, *,
                         cfd_key: str, pred_key: str, err_key: str, unit: str) -> Dict:
    import matplotlib
    matplotlib.use("Agg")
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


def plot_cloud_triptych(xyz: np.ndarray, cfd: np.ndarray, pred: np.ndarray,
                        out_png: Path, title: str, unit: str) -> Dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _register_gnn_colormaps()
    err = pred - cfd
    idx = np.arange(len(xyz))
    if len(idx) > 85000:
        idx = np.random.default_rng(1234).choice(idx, 85000, replace=False)
    idx = idx[np.argsort(xyz[idx, 1])]
    vmin = float(min(cfd.min(), pred.min()))
    vmax = float(max(cfd.max(), pred.max()))
    err_vmax = float(np.percentile(np.abs(err), 99.0)) or 1.0
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.2))
    specs = (
        (cfd, f"CFD ({unit})", vmin, vmax, FIELD_CMAP_NAME),
        (pred, f"Pred ({unit})", vmin, vmax, FIELD_CMAP_NAME),
        (err, f"Pred−CFD ({unit})", -err_vmax, err_vmax, FIELD_CMAP_NAME),
    )
    for ax, (arr, lab, lo, hi, cmap) in zip(axes, specs):
        sc = ax.scatter(xyz[idx, 0], xyz[idx, 2], c=arr[idx], cmap=cmap, vmin=lo, vmax=hi,
                        s=0.6, linewidths=0, rasterized=True)
        ax.set_aspect("equal")
        ax.set_title(lab, fontsize=11, fontweight="bold")
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Z (mm)")
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(title + "\nPNG thinned for display; VTP keeps the full interior cloud", fontsize=11, y=1.02)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return {"png": str(out_png), "vmin": vmin, "vmax": vmax, "err_p99": err_vmax,
            "displayed_points": int(len(idx)), "source_points": int(len(xyz))}


def fielddata_base(contract: Dict, *, arm: str, phase: str, canonical: str,
                   frame_index: int, step: int, q_norm: float, n_src: int,
                   r2: float, mae: float, p_ref: float) -> Dict:
    prefix = contract["prefix"]
    return {
        **normalization_field_data(contract),
        f"{prefix}_cfd_max": float(contract["cfd_max"]),
        f"{prefix}_pred_max": float(contract["pred_max"]),
        "frame_index": float(frame_index),
        "peak_step": float(step),
        "q_norm": float(q_norm),
        "arm": arm,
        "phase": phase,
        "case": canonical,
        "n_source_points": float(n_src),
        "pointcloud_r2": float(r2),
        "pointcloud_mae": float(mae),
        "p_volume_mean_pa": float(p_ref),
    }


def export_pressure(*, out_dir: Path, arm: str, phase: str, step: int, frame_index: int,
                    q_norm: float, full_xyz: np.ndarray, wall_xyz: np.ndarray, tri: np.ndarray,
                    stl_path: Path, truth: np.ndarray, pred: np.ndarray, rows: np.ndarray,
                    n_wall: int, canonical: str, extra_meta: Dict) -> Dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    export_dir = out_dir / "_export"
    surf_dir = out_dir / "surface_gaussian"
    plots_dir = out_dir / "plots"
    for d in (export_dir, surf_dir, plots_dir):
        d.mkdir(parents=True, exist_ok=True)

    xyz = full_xyz[rows]
    kind = (rows >= n_wall).astype(np.int8)
    if not np.array_equal(xyz[:n_wall], wall_xyz):
        raise RuntimeError("pressure wall slice is not concat(wall, interior)[:n_wall]")
    if not (kind[:n_wall] == 0).all() or not (kind[n_wall:] == 1).all():
        raise RuntimeError("pressure point_kind identity failed")

    err = pred - truth
    scalars = {
        "pressure_cfd": truth.astype(np.float64),
        "pressure_pred": pred.astype(np.float64),
        "err_pressure": err.astype(np.float64),
        "abs_err_pressure": np.abs(err).astype(np.float64),
        "point_kind_0_wall_1_interior": kind.astype(np.int64),
        "source_index": rows.astype(np.int64),
        "pressure_reference_pa": np.full(len(truth), extra_meta["p_ref"], dtype=np.float64),
    }
    contract = fit_selfmax(
        truth, pred, prefix="pressure",
        denominator_scope="original colocated wall∪interior, this case this frame",
        source_unit="Pa",
    )
    scalars.update(selfmax_fields(truth, pred, contract))
    pc = M.basic_metrics(truth, pred)
    wall_m = M.basic_metrics(truth[:n_wall], pred[:n_wall])
    int_m = M.basic_metrics(truth[n_wall:], pred[n_wall:])

    csv_path = export_dir / "same_point_fields.csv.gz"
    pd.DataFrame({"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2], **scalars}).to_csv(
        csv_path, index=False, float_format="%.8g"
    )
    wall_src = export_dir / "gaussian_source_wall.csv"
    wall_df = pd.DataFrame({
        "x": wall_xyz[:, 0], "y": wall_xyz[:, 1], "z": wall_xyz[:, 2],
        "pressure_cfd": truth[:n_wall], "pressure_pred": pred[:n_wall],
    })
    wall_df.to_csv(wall_src, index=False, float_format="%.8g")

    write_points(out_dir / "full_pointcloud.vtp", xyz, scalars)
    write_points(out_dir / "interior_pointcloud.vtp", xyz[n_wall:], {k: v[n_wall:] for k, v in scalars.items()})
    write_points(out_dir / "wall_pointcloud.vtp", wall_xyz, {k: v[:n_wall] for k, v in scalars.items()})
    write_mesh_vtp(wall_xyz, tri, {k: v[:n_wall] for k, v in scalars.items() if np.asarray(v).ndim == 1},
                   out_dir / "cfd_wall_native.vtp")

    mapped = map_csv_to_stl(
        csv_path=wall_src, stl_path=stl_path, output_dir=surf_dir,
        scalar_columns=["pressure_cfd", "pressure_pred"],
        report_json=out_dir / "mapping_report.json", **GAUSSIAN,
    )
    surface_vtp = Path(mapped["vtp"])
    shutil.copy2(surface_vtp, out_dir / "surface_gaussian.vtp")
    shutil.copy2(mapped["csv"], out_dir / "surface_gaussian.csv")
    if POSTVIEW_XML.is_file():
        shutil.copy2(POSTVIEW_XML, out_dir / "GNN_blue_white_red.xml")

    verts, tris, mapped_sc = _load_mesh_vtp(out_dir / "surface_gaussian.vtp")
    mapped_cfd = np.asarray(mapped_sc["pressure_cfd"], dtype=np.float64)
    mapped_pred = np.asarray(mapped_sc["pressure_pred"], dtype=np.float64)
    mapped_err = mapped_pred - mapped_cfd
    surface_extra = {
        "err_pressure": mapped_err.astype(np.float32),
        "abs_err_pressure": np.abs(mapped_err).astype(np.float32),
        **selfmax_fields(mapped_cfd, mapped_pred, contract),
    }
    fd = fielddata_base(
        contract, arm=arm, phase=phase, canonical=canonical, frame_index=frame_index,
        step=step, q_norm=q_norm, n_src=len(truth), r2=pc["r2"], mae=pc["mae"],
        p_ref=extra_meta["p_ref"],
    )
    for path in (
        out_dir / "surface_gaussian.vtp",
        out_dir / "full_pointcloud.vtp",
        out_dir / "interior_pointcloud.vtp",
        out_dir / "wall_pointcloud.vtp",
        out_dir / "cfd_wall_native.vtp",
    ):
        attach_fielddata(path, surface_extra if path.name == "surface_gaussian.vtp" else {}, fd)

    pa_plot = plot_signed_triptych(
        out_dir / "surface_gaussian.vtp", plots_dir / "fig_pressure_pa_triptych.png",
        f"{canonical} · {arm} · {phase} · step {step} · wall Gaussian",
        cfd_key="pressure_cfd", pred_key="pressure_pred", err_key="err_pressure", unit="Pa",
    )
    sm_plot = plot_signed_triptych(
        out_dir / "surface_gaussian.vtp", plots_dir / "fig_pressure_selfmax_triptych.png",
        f"{canonical} · {arm} · {phase} · pressure selfmax",
        cfd_key="pressure_cfd_selfmax", pred_key="pressure_pred_selfmax",
        err_key="pressure_selfmax_error_pred_minus_cfd", unit="selfmax",
    )
    int_plot = plot_cloud_triptych(
        xyz[n_wall:], truth[n_wall:], pred[n_wall:],
        plots_dir / "fig_pressure_interior_projection.png",
        f"{canonical} · {arm} · {phase} · interior pressure", "Pa",
    )
    mapping = json.loads((out_dir / "mapping_report.json").read_text())
    y0 = float(np.median(xyz[n_wall:, 1]))
    slab = np.abs(xyz[:, 1] - y0) <= 1.5
    slab &= kind == 1
    write_points(out_dir / "interior_3mm_slab.vtp", xyz[slab], {k: v[slab] for k, v in scalars.items()})
    attach_fielddata(out_dir / "interior_3mm_slab.vtp", {}, fd)
    manifest = {
        "case": canonical, "arm": arm, "field": "pressure", "phase": phase,
        "frame_index": int(frame_index), "step": int(step), "q_norm": float(q_norm),
        "n_query_points": int(len(truth)), "n_wall_points": int(n_wall),
        "n_interior_points": int(len(truth) - n_wall),
        "n_surface_vertices": int(len(verts)), "n_triangles": int(len(tris)),
        "n_native_triangles": int(len(tri)),
        "interpolation": GAUSSIAN,
        "selfmax": contract,
        "pointcloud_metrics": {
            "overall": {"r2": float(pc["r2"]), "mae": float(pc["mae"]), "rmse": float(pc.get("rmse", float("nan")))},
            "wall": {"r2": float(wall_m["r2"]), "mae": float(wall_m["mae"])},
            "interior": {"r2": float(int_m["r2"]), "mae": float(int_m["mae"])},
            "metric_basis": "same-point wall∪interior; not STL-mapped surface",
        },
        "mapping_coverage": mapping.get("coverage"),
        "files": {
            "surface_gaussian": "surface_gaussian.vtp",
            "cfd_wall_native": "cfd_wall_native.vtp",
            "full_pointcloud": "full_pointcloud.vtp",
            "wall_pointcloud": "wall_pointcloud.vtp",
            "interior_pointcloud": "interior_pointcloud.vtp",
            "interior_3mm_slab": "interior_3mm_slab.vtp",
            "same_point_csv": "_export/same_point_fields.csv.gz",
            "gaussian_source": "_export/gaussian_source_wall.csv",
            "mapping_report": "mapping_report.json",
            "pa_preview": "plots/fig_pressure_pa_triptych.png",
            "selfmax_preview": "plots/fig_pressure_selfmax_triptych.png",
            "interior_preview": "plots/fig_pressure_interior_projection.png",
        },
        **extra_meta,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return {**manifest, "_plots": {"pa": pa_plot, "selfmax": sm_plot, "interior": int_plot}}


def export_velocity(*, out_dir: Path, arm: str, phase: str, step: int, frame_index: int,
                    q_norm: float, full_xyz: np.ndarray, wall_xyz: np.ndarray, tri: np.ndarray,
                    truth: np.ndarray, pred: np.ndarray, rows: np.ndarray,
                    n_wall: int, canonical: str, extra_meta: Dict) -> Dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    export_dir = out_dir / "_export"
    plots_dir = out_dir / "plots"
    for d in (export_dir, plots_dir):
        d.mkdir(parents=True, exist_ok=True)

    xyz = full_xyz[rows]
    if rows.min() < n_wall:
        raise RuntimeError("velocity query rows must be interior-only")
    if not np.array_equal(xyz, full_xyz[n_wall:]):
        raise RuntimeError("velocity coordinates are not concat(wall, interior)[n_wall:]")
    source_index = rows - n_wall
    speed_cfd = np.linalg.norm(truth, axis=1)
    speed_pred = np.linalg.norm(pred, axis=1)
    err_vec = pred - truth
    err_speed = speed_pred - speed_cfd
    contract = fit_selfmax(
        speed_cfd, speed_pred, prefix="speed",
        denominator_scope="original colocated interior, this case this frame",
        source_unit="m/s",
    )
    scalars = {
        "speed_cfd": speed_cfd,
        "speed_pred": speed_pred,
        "err_speed": err_speed,
        "abs_err_speed": np.abs(err_speed),
        "velocity_cfd_aligned_m_s": truth.astype(np.float32),
        "velocity_pred_aligned_m_s": pred.astype(np.float32),
        "velocity_error_aligned_m_s": err_vec.astype(np.float32),
        "point_kind_0_wall_1_interior": np.ones(len(rows), dtype=np.int64),
        "source_index": source_index.astype(np.int64),
        "query_idx": rows.astype(np.int64),
    }
    scalars.update(selfmax_fields(speed_cfd, speed_pred, contract))
    pc = M.basic_metrics(speed_cfd, speed_pred)

    from scipy.spatial import cKDTree
    node_dist = cKDTree(wall_xyz).query(xyz, k=1)[0]
    identity = {
        "n_interior": int(len(xyz)),
        "query_idx_min": int(rows.min()),
        "query_idx_max": int(rows.max()),
        "n_wall": int(n_wall),
        "wall_node_distance_mm": {
            "min": float(node_dist.min()),
            "p01": float(np.percentile(node_dist, 1)),
            "p50": float(np.percentile(node_dist, 50)),
            "note": "nearest wall node, not face distance; near-wall low speed kept",
        },
    }
    csv_cols = {
        "x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2],
        "speed_cfd": speed_cfd, "speed_pred": speed_pred,
        "err_speed": err_speed, "abs_err_speed": np.abs(err_speed),
        "u_cfd": truth[:, 0], "v_cfd": truth[:, 1], "w_cfd": truth[:, 2],
        "u_pred": pred[:, 0], "v_pred": pred[:, 1], "w_pred": pred[:, 2],
        "source_index": source_index, "query_idx": rows,
        **{k: v for k, v in scalars.items() if np.asarray(v).ndim == 1 and k.startswith("speed_")},
    }
    pd.DataFrame(csv_cols).to_csv(export_dir / "same_point_fields.csv.gz", index=False, float_format="%.8g")
    write_points(out_dir / "interior_pointcloud.vtp", xyz, scalars)
    write_mesh_vtp(wall_xyz, tri, {"geometry_only": np.ones(len(wall_xyz))}, out_dir / "wall_geometry_only.vtp")
    y0 = float(np.median(xyz[:, 1]))
    slab = np.abs(xyz[:, 1] - y0) <= 1.5
    write_points(out_dir / "interior_3mm_slab.vtp", xyz[slab], {k: v[slab] for k, v in scalars.items()})
    if POSTVIEW_XML.is_file():
        shutil.copy2(POSTVIEW_XML, out_dir / "GNN_blue_white_red.xml")

    fd = fielddata_base(
        contract, arm=arm, phase=phase, canonical=canonical, frame_index=frame_index,
        step=step, q_norm=q_norm, n_src=len(speed_cfd), r2=pc["r2"], mae=pc["mae"],
        p_ref=extra_meta.get("p_ref", float("nan")),
    )
    for path in (out_dir / "interior_pointcloud.vtp", out_dir / "interior_3mm_slab.vtp"):
        attach_fielddata(path, {}, fd)

    speed_plot = plot_cloud_triptych(
        xyz, speed_cfd, speed_pred, plots_dir / "fig_speed_triptych.png",
        f"{canonical} · {arm} · {phase} · step {step} · |u| interior", "m/s",
    )
    sm_plot = plot_cloud_triptych(
        xyz, scalars["speed_cfd_selfmax"], scalars["speed_pred_selfmax"],
        plots_dir / "fig_speed_selfmax_triptych.png",
        f"{canonical} · {arm} · {phase} · speed selfmax", "selfmax",
    )
    manifest = {
        "case": canonical, "arm": arm, "field": "velocity", "phase": phase,
        "frame_index": int(frame_index), "step": int(step), "q_norm": float(q_norm),
        "n_interior_points": int(len(xyz)),
        "interpolation": "none (interior point cloud only)",
        "selfmax": contract,
        "identity": identity,
        "pointcloud_metrics": {
            "r2": float(pc["r2"]), "mae": float(pc["mae"]),
            "rmse": float(pc.get("rmse", float("nan"))),
            "metric_basis": "same-point interior |u|; not a wall projection",
        },
        "files": {
            "interior_pointcloud": "interior_pointcloud.vtp",
            "interior_3mm_slab": "interior_3mm_slab.vtp",
            "wall_geometry_only": "wall_geometry_only.vtp",
            "same_point_csv": "_export/same_point_fields.csv.gz",
            "speed_preview": "plots/fig_speed_triptych.png",
            "selfmax_preview": "plots/fig_speed_selfmax_triptych.png",
        },
        **extra_meta,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return {**manifest, "_plots": {"speed": speed_plot, "selfmax": sm_plot}}


def verify_vtp(path: Path, *, expect_faces: bool, required: List[str], n_src: int | None = None) -> Dict:
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    poly = reader.GetOutput()
    n_pts = int(poly.GetNumberOfPoints())
    n_tri = int(poly.GetNumberOfPolys())
    names = [poly.GetPointData().GetArrayName(i) for i in range(poly.GetPointData().GetNumberOfArrays())]
    missing = [k for k in required if k not in names]
    finite_ok = True
    for key in required:
        if key not in names:
            finite_ok = False
            break
        arr = vtk_to_numpy(poly.GetPointData().GetArray(key))
        finite_ok = finite_ok and np.isfinite(arr).mean() > 0.9
    rel = str(path.relative_to(EXP)) if str(path).startswith(str(EXP)) else str(path)
    return {
        "path": rel, "n_points": n_pts, "n_triangles": n_tri, "fields": names,
        "missing": missing, "finite_ok": bool(finite_ok),
        "source_count_match": (n_src is None) or (n_pts == n_src),
        "has_faces": n_tri > 0 if expect_faces else True,
    }


def write_readme(out_root: Path, selection: Dict, steps: np.ndarray, q: np.ndarray) -> None:
    phase_rows = "\n".join(
        f"| `{name}` | {k} | {int(steps[k])} | {q[k]:.3f} | {desc} |"
        for name, k, desc in PHASES
    )
    text = f"""# 体场中位病例相位后处理包（与 WSS 同一例）

- 病例：`{selection['case']}`（cv3 fold {selection['fold']} 留出）
- 选例：沿用 WSS `postview_median_phases_20260920`，**不按体场 R² 重排**
- checkpoint：各臂 fold2 `ckpt_best.pt`
- 压力：相对压 `p(x,t) − p_volume_mean(t)`（Pa）；壁面 Gaussian + native 面片，体内点云另存
- 速度：体内点云 `|u|` 与三分量向量（m/s），**不回插壁面**
- 正式 R²/MAE 只在同点 CSV 上算

## 相位代表帧

| 相位 | frame | step | Q/Qpeak | 定义 |
| --- | ---: | ---: | ---: | --- |
{phase_rows}

## 打开这些文件

压力（ParaView：Surface；路径相对本包根目录）：

- `{selection['case'].replace('/', '__')}/pressure/<臂>/<相位>/surface_gaussian.vtp` — STL Gaussian 壁面（`pressure_cfd` / `pressure_pred` / `err_pressure`）
- `{selection['case'].replace('/', '__')}/pressure/<臂>/<相位>/cfd_wall_native.vtp` — 原生 CFD 壁面三角，无插值
- `{selection['case'].replace('/', '__')}/pressure/<臂>/<相位>/interior_pointcloud.vtp` — 体内压力点云
- `{selection['case'].replace('/', '__')}/pressure/<臂>/<相位>/full_pointcloud.vtp` — 壁面∪体内完整点云

速度（ParaView：Points，可用 Glyph）：

- `{selection['case'].replace('/', '__')}/velocity/<臂>/<相位>/interior_pointcloud.vtp` — `speed_*` 与 `velocity_*_aligned_m_s`
- `{selection['case'].replace('/', '__')}/velocity/<臂>/<相位>/wall_geometry_only.vtp` — 透明外壳，无速度场

CFD 与 Pred 共用同一 Data Range。`*_selfmax` 无量纲，只比分布。压力分母来自该帧完整 wall∪interior；速度分母来自该帧体内 `|u|`。
"""
    (out_root / "README.md").write_text(text, encoding="utf-8")


def _release(model) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--output-dir", default=str(EXP / "postview_median_phases_20260921"))
    args = ap.parse_args()
    out_root = Path(args.output_dir)
    if not out_root.is_absolute():
        out_root = REPO / out_root
    out_root.mkdir(parents=True, exist_ok=True)

    device = args.device
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    print(f"[start] device={device}  out={out_root}", flush=True)

    selection = wss_selection()
    canonical = selection["case"]
    fold = int(selection["fold"])
    if fold != 2:
        raise RuntimeError(f"expected fold 2 for {canonical}, got {fold}")
    steps, q = load_waveform()
    frame_indices = [k for _, k, _ in PHASES]
    print(f"[select] {canonical} fold={fold} (copied from WSS median package)", flush=True)

    print("[load] PT0 geometry + four frames", flush=True)
    pt0_run, pt0_cfg, pt0_feat, pt0_stats, pt0_model, pt0_ckpt = load_model_bundle("PT0", fold, device)
    pt0_case = load_eval_case(pt0_cfg, pt0_stats, canonical)
    wall_xyz, full_xyz, stl_xyz, stl_tris, geom_meta = align_wall(pt0_case)
    n_wall = int(pt0_case["n_wall"])
    tri = wall_triangles(canonical, wall_xyz)
    case_root = out_root / canonical.replace("/", "__")
    case_root.mkdir(parents=True, exist_ok=True)
    stl_path = case_root / "aligned_geometry.stl"
    _write_stl(stl_xyz, stl_tris, stl_path)
    print(f"  n_wall={n_wall} n_full={len(full_xyz)} n_tri={len(tri)}", flush=True)

    print("[infer] PT0", flush=True)
    pt0_pred = predict_t0_frames(pt0_model, pt0_case, pt0_cfg, pt0_feat, pt0_stats, device, frame_indices, "pressure_mixed")
    peak_deltas = {"PT0": abs(_scalar_r2("pressure_mixed", pt0_pred[21]["truth"], pt0_pred[21]["pred"]) - PEAK_CSV["PT0"])}
    print(f"  PT0 peak |ΔR²|={peak_deltas['PT0']:.6g}", flush=True)
    _release(pt0_model)

    print("[infer] PTB8", flush=True)
    ptb_run, ptb_cfg, ptb_feat, ptb_stats, ptb_model, ptb_ckpt = load_model_bundle("PTB8", fold, device)
    ptb_case = load_eval_case(ptb_cfg, ptb_stats, canonical)
    ptb_pred = predict_tb_frames(ptb_model, ptb_case, ptb_cfg, ptb_feat, ptb_stats, device, frame_indices, "pressure_mixed")
    peak_deltas["PTB8"] = abs(_scalar_r2("pressure_mixed", ptb_pred[21]["truth"], ptb_pred[21]["pred"]) - PEAK_CSV["PTB8"])
    print(f"  PTB8 peak |ΔR²|={peak_deltas['PTB8']:.6g}", flush=True)
    _release(ptb_model)

    print("[infer] VT0", flush=True)
    vt0_run, vt0_cfg, vt0_feat, vt0_stats, vt0_model, vt0_ckpt = load_model_bundle("VT0", fold, device)
    vt0_case = load_eval_case(vt0_cfg, vt0_stats, canonical)
    vt0_pred = predict_t0_frames(vt0_model, vt0_case, vt0_cfg, vt0_feat, vt0_stats, device, frame_indices, "velocity")
    peak_deltas["VT0"] = abs(_scalar_r2("velocity", vt0_pred[21]["truth"], vt0_pred[21]["pred"]) - PEAK_CSV["VT0"])
    print(f"  VT0 peak |ΔR²|={peak_deltas['VT0']:.6g}", flush=True)
    _release(vt0_model)

    print("[infer] VTB4", flush=True)
    vtb_run, vtb_cfg, vtb_feat, vtb_stats, vtb_model, vtb_ckpt = load_model_bundle("VTB4", fold, device)
    vtb_case = load_eval_case(vtb_cfg, vtb_stats, canonical)
    vtb_pred = predict_tb_frames(vtb_model, vtb_case, vtb_cfg, vtb_feat, vtb_stats, device, frame_indices, "velocity")
    peak_deltas["VTB4"] = abs(_scalar_r2("velocity", vtb_pred[21]["truth"], vtb_pred[21]["pred"]) - PEAK_CSV["VTB4"])
    print(f"  VTB4 peak |ΔR²|={peak_deltas['VTB4']:.6g}", flush=True)
    _release(vtb_model)

    arm_meta = {
        "PT0": {"pred": pt0_pred, "run": str(pt0_run), "ckpt_epoch": int(pt0_ckpt.get("epoch", -1)),
                "definition": "phase-query PointNet++ pressure, ckpt_best, frame_stats denorm"},
        "PTB8": {"pred": ptb_pred, "run": str(ptb_run), "ckpt_epoch": int(ptb_ckpt.get("epoch", -1)),
                 "definition": "time-basis K=8 pressure reconstruct_frame, ckpt_best, frame_stats denorm"},
        "VT0": {"pred": vt0_pred, "run": str(vt0_run), "ckpt_epoch": int(vt0_ckpt.get("epoch", -1)),
                "definition": "phase-query PointNet++ velocity, ckpt_best, frame_stats denorm"},
        "VTB4": {"pred": vtb_pred, "run": str(vtb_run), "ckpt_epoch": int(vtb_ckpt.get("epoch", -1)),
                 "definition": "time-basis K=4 velocity reconstruct_frame, ckpt_best, frame_stats denorm"},
    }

    packages: List[Dict] = []
    for arm in PRESSURE_ARMS:
        for phase, k, _desc in PHASES:
            print(f"[export] pressure {arm} {phase}", flush=True)
            rec = arm_meta[arm]["pred"][k]
            packages.append(export_pressure(
                out_dir=case_root / "pressure" / arm / phase,
                arm=arm, phase=phase, step=int(steps[k]), frame_index=k, q_norm=float(q[k]),
                full_xyz=full_xyz, wall_xyz=wall_xyz, tri=tri, stl_path=stl_path,
                truth=rec["truth"], pred=rec["pred"], rows=rec["rows"], n_wall=n_wall,
                canonical=canonical,
                extra_meta={
                    "run": arm_meta[arm]["run"], "checkpoint": "ckpt_best.pt",
                    "ckpt_epoch": arm_meta[arm]["ckpt_epoch"],
                    "definition": arm_meta[arm]["definition"],
                    "geometry": geom_meta, "p_ref": rec["p_ref"],
                    "skill": "postview-surface-viz",
                },
            ))

    for arm in VELOCITY_ARMS:
        for phase, k, _desc in PHASES:
            print(f"[export] velocity {arm} {phase}", flush=True)
            rec = arm_meta[arm]["pred"][k]
            packages.append(export_velocity(
                out_dir=case_root / "velocity" / arm / phase,
                arm=arm, phase=phase, step=int(steps[k]), frame_index=k, q_norm=float(q[k]),
                full_xyz=full_xyz, wall_xyz=wall_xyz, tri=tri,
                truth=rec["truth"], pred=rec["pred"], rows=rec["rows"], n_wall=n_wall,
                canonical=canonical,
                extra_meta={
                    "run": arm_meta[arm]["run"], "checkpoint": "ckpt_best.pt",
                    "ckpt_epoch": arm_meta[arm]["ckpt_epoch"],
                    "definition": arm_meta[arm]["definition"],
                    "geometry": geom_meta, "p_ref": rec["p_ref"],
                    "skill": "postview-surface-viz",
                },
            ))

    listing_rows = []
    verifications = []
    case_tag = canonical.replace("/", "__")
    for pack in packages:
        field = pack["field"]
        rel = Path(field) / pack["arm"] / pack["phase"]
        listed = Path(case_tag) / rel
        if field == "pressure":
            surf = case_root / rel / "surface_gaussian.vtp"
            native = case_root / rel / "cfd_wall_native.vtp"
            full = case_root / rel / "full_pointcloud.vtp"
            interior = case_root / rel / "interior_pointcloud.vtp"
            wall_pc = case_root / rel / "wall_pointcloud.vtp"
            verifications.append(verify_vtp(surf, expect_faces=True, required=["pressure_cfd", "pressure_pred"]))
            verifications.append(verify_vtp(native, expect_faces=True, required=["pressure_cfd", "pressure_pred"],
                                            n_src=pack["n_wall_points"]))
            verifications.append(verify_vtp(full, expect_faces=False, required=["pressure_cfd", "pressure_pred"],
                                            n_src=pack["n_query_points"]))
            verifications.append(verify_vtp(interior, expect_faces=False, required=["pressure_cfd", "pressure_pred"],
                                            n_src=pack["n_interior_points"]))
            verifications.append(verify_vtp(wall_pc, expect_faces=False, required=["pressure_cfd", "pressure_pred"],
                                            n_src=pack["n_wall_points"]))
            listing_rows.append({
                "field": field, "arm": pack["arm"], "phase": pack["phase"], "step": pack["step"],
                "q_norm": pack["q_norm"],
                "wall_surface": str((listed / "surface_gaussian.vtp").as_posix()),
                "wall_native": str((listed / "cfd_wall_native.vtp").as_posix()),
                "interior_pointcloud": str((listed / "interior_pointcloud.vtp").as_posix()),
                "full_pointcloud": str((listed / "full_pointcloud.vtp").as_posix()),
                "r2": pack["pointcloud_metrics"]["overall"]["r2"],
            })
        else:
            interior = case_root / rel / "interior_pointcloud.vtp"
            verifications.append(verify_vtp(
                interior, expect_faces=False,
                required=["speed_cfd", "speed_pred", "velocity_cfd_aligned_m_s", "velocity_pred_aligned_m_s"],
                n_src=pack["n_interior_points"],
            ))
            listing_rows.append({
                "field": field, "arm": pack["arm"], "phase": pack["phase"], "step": pack["step"],
                "q_norm": pack["q_norm"],
                "wall_surface": "",
                "wall_native": "",
                "interior_pointcloud": str((listed / "interior_pointcloud.vtp").as_posix()),
                "full_pointcloud": "",
                "r2": pack["pointcloud_metrics"]["r2"],
            })

    batch = {
        "experiment": "volume_time_20260919",
        "output_dir": str(out_root),
        "selection": selection,
        "checkpoint": "ckpt_best.pt",
        "phases": [
            {"name": name, "frame_index": k, "step": int(steps[k]), "q_norm": float(q[k]), "note": desc}
            for name, k, desc in PHASES
        ],
        "geometry": geom_meta,
        "n_wall_points": int(n_wall),
        "n_full_points": int(len(full_xyz)),
        "peak_r2_vs_csv_abs_delta": peak_deltas,
        "packages": [{k: v for k, v in p.items() if not k.startswith("_")} for p in packages],
        "notes": [
            "Case copied from the WSS median-phase package; volume R2 was not used to re-select.",
            "Pressure selfmax denominators are the original wall∪interior maxima of this case/frame.",
            "Velocity has no wall interpolation; Glyph vectors are velocity_*_aligned_m_s.",
            "Formal R2/MAE remain same-point metrics; Gaussian VTP is visualization only.",
            "cv3 held-out fold is the eval set; test34 was not used.",
        ],
    }
    (out_root / "manifest.json").write_text(json.dumps(batch, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_root / "case_selection.json").write_text(json.dumps(selection, indent=2, ensure_ascii=False), encoding="utf-8")
    with (out_root / "打开文件清单.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(listing_rows[0]))
        writer.writeheader()
        writer.writerows(listing_rows)
    failed = [v for v in verifications if v["missing"] or not v["finite_ok"] or not v["has_faces"]
              or v["source_count_match"] is False]
    verification = {
        "n_checked": len(verifications), "n_failed": len(failed),
        "peak_r2_vs_csv_abs_delta": peak_deltas,
        "peak_r2_match_ok": all(d < 1e-3 for d in peak_deltas.values()),
        "items": verifications, "failed": failed,
    }
    (out_root / "verification.json").write_text(json.dumps(verification, indent=2, ensure_ascii=False), encoding="utf-8")
    write_readme(out_root, selection, steps, q)
    shutil.copy2(out_root / "README.md", case_root / "README_打开说明.md")
    print(json.dumps({"done": str(out_root), "failed": len(failed),
                      "peak_r2_delta": peak_deltas}, indent=2), flush=True)
    if failed:
        raise SystemExit(f"verification failed for {len(failed)} files")


if __name__ == "__main__":
    main()
