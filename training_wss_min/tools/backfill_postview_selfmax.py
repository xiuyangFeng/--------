#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""无需模型推理，为既有 PostView 包补齐 CFD/Pred 各自 max 归一化产物。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

from training_wss_min import metrics as M
from training_wss_min.tools.export_wss_postview import (
    SELF_MAX_SCALARS,
    selfmax_scalar_data,
)
from tools.cfdpost_cloud_export.plot_stl_mapped_triptych import plot_triptych


REPO = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO / "training_wss_min/runs/pointnet_distribution_matrix/outputs"
DEFAULT_RUNS = ("e2_global", "e3_global", "e23_global", "e2_case", "e3_case")


def _source_columns(columns: Iterable[str]) -> Tuple[str, str, str]:
    names = set(columns)
    if {"wss_cfd", "wss_pred"}.issubset(names):
        return "wss_cfd", "wss_pred", "physical_wss_pa"
    if {"true_norm", "pred_norm"}.issubset(names):
        return "true_norm", "pred_norm", "normalized_target_scale_cancels"
    raise ValueError("CSV/VTP lacks both physical and normalized true/pred fields")


def _with_selfmax_columns(
    frame: pd.DataFrame,
    *,
    true_max: float | None = None,
    pred_max: float | None = None,
) -> Tuple[pd.DataFrame, Dict[str, float], str]:
    true_key, pred_key, source_space = _source_columns(frame.columns)
    yt = frame[true_key].to_numpy(dtype=np.float64)
    yp = frame[pred_key].to_numpy(dtype=np.float64)
    if true_max is None or pred_max is None:
        fields, meta = selfmax_scalar_data(yt, yp)
    else:
        if true_max <= 1e-12 or pred_max <= 1e-12:
            raise ValueError(f"invalid fixed self-max denominators: {true_max}, {pred_max}")
        true_self = yt / true_max
        pred_self = yp / pred_max
        err = pred_self - true_self
        fields = {
            "wss_cfd_over_cfd_max": true_self,
            "wss_pred_over_pred_max": pred_self,
            "err_wss_selfmax": err,
            "abs_err_wss_selfmax": np.abs(err),
        }
        meta = {
            "wss_cfd_max": float(true_max),
            "wss_pred_max": float(pred_max),
            "wss_pred_max_over_cfd_max": float(pred_max / true_max),
            "pred_below_zero_fraction": float(np.mean(yp < 0.0)),
        }
    out = frame.copy()
    for name, values in fields.items():
        out[name] = values
    return out, meta, source_space


def _read_vtp(path: Path):
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    poly = reader.GetOutput()
    if poly.GetNumberOfPoints() == 0:
        raise RuntimeError(f"empty VTP: {path}")
    arrays = {}
    point_data = poly.GetPointData()
    for i in range(point_data.GetNumberOfArrays()):
        arr = point_data.GetArray(i)
        if arr is not None and arr.GetName():
            arrays[arr.GetName()] = vtk_to_numpy(arr).astype(np.float64)
    return poly, arrays


def _update_vtp(path: Path, true_max: float, pred_max: float) -> None:
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk

    poly, arrays = _read_vtp(path)
    true_key, pred_key, _ = _source_columns(arrays)
    yt, yp = arrays[true_key], arrays[pred_key]
    true_self = yt / true_max
    pred_self = yp / pred_max
    err = pred_self - true_self
    fields = {
        "wss_cfd_over_cfd_max": true_self,
        "wss_pred_over_pred_max": pred_self,
        "err_wss_selfmax": err,
        "abs_err_wss_selfmax": np.abs(err),
    }
    point_data = poly.GetPointData()
    for name, values in fields.items():
        if point_data.HasArray(name):
            point_data.RemoveArray(name)
        arr = numpy_to_vtk(np.ascontiguousarray(values, dtype=np.float32), deep=True)
        arr.SetName(name)
        point_data.AddArray(arr)
    tmp = path.with_name(path.stem + ".selfmax_tmp.vtp")
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(tmp))
    writer.SetInputData(poly)
    if writer.Write() != 1:
        raise RuntimeError(f"failed to rewrite VTP: {path}")
    tmp.replace(path)


def _append_readme(case_dir: Path) -> None:
    path = case_dir / "README_后处理打开说明.md"
    if not path.is_file():
        return
    text = path.read_text(encoding="utf-8")
    marker = "## 双自最大值归一化（导师新增口径）"
    if marker in text:
        return
    text += f"""

{marker}

- `wss_cfd_over_cfd_max` = `WSS_CFD / max(WSS_CFD)`
- `wss_pred_over_pred_max` = `WSS_Pred / max(WSS_Pred)`
- `err_wss_selfmax` = 后者减前者；正式 R²/MAE/RMSE 只在同点 CSV 计算。
- `plots/fig_wss_selfmax_triptych.png` 使用 CFD/Pred 共享色标展示病例内空间分布。
- 该口径移除了真实与预测各自的绝对幅值，不用于判断预测最大 WSS 是否正确。
"""
    path.write_text(text, encoding="utf-8")


def _update_manifest(
    case_dir: Path,
    *,
    metrics: Dict[str, float],
    denominators: Dict[str, float],
    source_space: str,
) -> Dict:
    path = case_dir / "manifest_bundle.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    scalars = list(manifest.get("scalars", []))
    for name in SELF_MAX_SCALARS:
        if name not in scalars:
            scalars.append(name)
    manifest["scalars"] = scalars
    manifest["selfmax_metrics"] = {
        "definition": "WSS_CFD/WSS_CFD,max vs WSS_Pred/WSS_Pred,max",
        "source_space": source_space,
        "r2": float(metrics["r2"]),
        "mae": float(metrics["mae"]),
        "rmse": float(metrics["rmse"]),
        **denominators,
        "metric_basis": "same-point wall CSV; prediction is not clipped",
        "interpretation": "spatial distribution only; absolute WSS amplitude removed",
    }
    manifest.setdefault("files", {})["selfmax_triptych"] = (
        "plots/fig_wss_selfmax_triptych.png"
    )
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def backfill_case(case_dir: Path) -> Tuple[Dict, np.ndarray, np.ndarray, Dict]:
    short = case_dir.name.split("__")[-2] if "__" in case_dir.name else case_dir.name
    manifest_path = case_dir / "manifest_bundle.json"
    manifest_old = json.loads(manifest_path.read_text(encoding="utf-8"))
    short = manifest_old["case"].split("/")[-1]

    point_csv = case_dir / "_export" / f"{short}__wall.csv"
    point_frame = pd.read_csv(point_csv)
    point_frame, denominators, source_space = _with_selfmax_columns(point_frame)
    point_frame.to_csv(point_csv, index=False, float_format="%.8g")
    yt = point_frame["wss_cfd_over_cfd_max"].to_numpy(dtype=np.float64)
    yp = point_frame["wss_pred_over_pred_max"].to_numpy(dtype=np.float64)
    metrics = M.basic_metrics(yt, yp)

    true_max = denominators["wss_cfd_max"]
    pred_max = denominators["wss_pred_max"]
    csv_paths = [
        case_dir / f"{short}__surface_wall.csv",
        case_dir / "surface_gaussian" / f"{short}__stl_mapped_wall.csv",
    ]
    for path in csv_paths:
        if path.is_file():
            frame, _, _ = _with_selfmax_columns(
                pd.read_csv(path), true_max=true_max, pred_max=pred_max,
            )
            frame.to_csv(path, index=False, float_format="%.8g")

    vtp_paths = [
        case_dir / f"{short}__pointcloud_wall.vtp",
        case_dir / "pointcloud" / f"{short}__wall.vtp",
        case_dir / f"{short}__surface_wall.vtp",
        case_dir / "surface_gaussian" / f"{short}__stl_mapped_wall.vtp",
    ]
    for path in vtp_paths:
        if path.is_file():
            _update_vtp(path, true_max, pred_max)

    surface_vtp = case_dir / f"{short}__surface_wall.vtp"
    plot_path = case_dir / "plots" / "fig_wss_selfmax_triptych.png"
    report = plot_triptych(
        vtp_path=surface_vtp,
        cfd_key="wss_cfd_over_cfd_max",
        pred_key="wss_pred_over_pred_max",
        err_key="err_wss_selfmax",
        output_path=plot_path,
        title=(
            f"{manifest_old['case']} · CFD/CFDmax vs Pred/Predmax · "
            f"{manifest_old['run']}"
        ),
        unit="self-max normalized",
        plane="xz",
        render="surface",
        field_cmap="GNN_BWR",
        err_cmap="GNN_BWR",
    )
    (case_dir / "plots" / "fig_wss_selfmax_triptych_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    _append_readme(case_dir)
    manifest = _update_manifest(
        case_dir,
        metrics=metrics,
        denominators=denominators,
        source_space=source_space,
    )
    return manifest, yt, yp, metrics


def backfill_run(run_dir: Path, partition: str) -> Dict:
    root = run_dir / "postview/ckpt_best" / partition
    case_dirs = sorted(p for p in root.glob("*__peak_wss") if p.is_dir())
    if not case_dirs:
        raise FileNotFoundError(f"no PostView case directories under {root}")
    manifests = []
    true_by_case = []
    pred_by_case = []
    rows = []
    for case_dir in case_dirs:
        manifest, yt, yp, metrics = backfill_case(case_dir)
        manifests.append(manifest)
        true_by_case.append(yt)
        pred_by_case.append(yp)
        sm = manifest["selfmax_metrics"]
        rows.append({
            "case": manifest["case"],
            "r2": float(metrics["r2"]),
            "mae": float(metrics["mae"]),
            "rmse": float(metrics["rmse"]),
            "wss_cfd_max": sm["wss_cfd_max"],
            "wss_pred_max": sm["wss_pred_max"],
            "wss_pred_max_over_cfd_max": sm["wss_pred_max_over_cfd_max"],
            "pred_below_zero_fraction": sm["pred_below_zero_fraction"],
            "source_space": sm["source_space"],
        })
        print(f"[selfmax] {run_dir.name} {manifest['case']} R2={metrics['r2']:.4f}")

    r2s = np.asarray([r["r2"] for r in rows], dtype=np.float64)
    pooled = M.basic_metrics(np.concatenate(true_by_case), np.concatenate(pred_by_case))
    casebalanced = M.casebalanced_field_metrics(true_by_case, pred_by_case)
    summary = {
        "run": run_dir.name,
        "partition": partition,
        "definition": "WSS_CFD/WSS_CFD,max vs WSS_Pred/WSS_Pred,max",
        "n_cases": len(rows),
        "field_pooled": pooled,
        "field_casebalanced": casebalanced,
        "case_r2": {
            "mean": float(np.mean(r2s)),
            "median": float(np.median(r2s)),
            "p10": float(np.percentile(r2s, 10.0)),
            "negative_cases": int(np.sum(r2s < 0.0)),
        },
        "interpretation": (
            "spatial distribution after each field uses its own per-case maximum; "
            "absolute amplitude calibration is intentionally removed"
        ),
    }
    with (root / "selfmax_per_case_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (root / "selfmax_metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8",
    )

    batch_path = root / "batch_manifest.json"
    if batch_path.is_file():
        batch = json.loads(batch_path.read_text(encoding="utf-8"))
        batch["cases"] = manifests
        batch["selfmax_summary"] = summary
        batch_path.write_text(json.dumps(batch, indent=2, ensure_ascii=False), encoding="utf-8")
    readme = root / "README.md"
    if readme.is_file():
        text = readme.read_text(encoding="utf-8")
        marker = "## Self-max distribution"
        if marker not in text:
            text += (
                f"\n{marker}\n\n"
                "- fields: `wss_cfd_over_cfd_max` and `wss_pred_over_pred_max`\n"
                "- summary: `selfmax_metrics.json` / `selfmax_per_case_metrics.csv`\n"
                "- preview: each case `plots/fig_wss_selfmax_triptych.png`\n"
            )
            readme.write_text(text, encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--runs", nargs="+", default=list(DEFAULT_RUNS))
    parser.add_argument("--partition", default="test", choices=("train", "val", "test"))
    args = parser.parse_args()
    root = args.root if args.root.is_absolute() else REPO / args.root
    summaries = []
    for run in args.runs:
        summaries.append(backfill_run(root / run, args.partition))
    out = root / f"selfmax_{args.partition}_summary.json"
    out.write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] {len(summaries)} runs -> {out}")


if __name__ == "__main__":
    main()
