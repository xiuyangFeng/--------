#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 GNN 场预测导出为 CFD-Post 可用的物理量点云 CSV。

优先从 processed/features/*.csv 读取原始 Fluent 坐标与 CFD 真值；
若 features 缺失，则从 .pt 的归一化坐标经 inverse_transform 还原。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.coord_normalize import inverse_transform  # noqa: E402


def _load_norm_stats(path: Path) -> Dict[str, Dict[str, float]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    stats = data.get("statistics")
    if not isinstance(stats, dict):
        raise SystemExit(f"归一化文件缺少 statistics: {path}")
    return stats


def _denorm_zscore(values: np.ndarray, stats: Dict[str, Dict[str, float]], key: str) -> np.ndarray:
    entry = stats[key]
    return values * float(entry["std"]) + float(entry["mean"])


def _vel_mag(u: np.ndarray, v: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.sqrt(u * u + v * v + w * w)


def _find_manifest_item(items: list, sample_id: str, case_name: str) -> Dict[str, Any]:
    matches = [
        it for it in items
        if str(it.get("sample_id")) == sample_id and str(it.get("case_name")) == case_name
    ]
    if not matches:
        raise SystemExit(
            f"manifest 中未找到 sample_id={sample_id!r} 且 case_name={case_name!r} 的条目"
        )
    if len(matches) > 1:
        raise SystemExit(f"manifest 中存在重复条目: {sample_id!r} / {case_name!r}")
    return matches[0]


def _resolve_path(path_str: str, base: Path) -> Path:
    p = Path(path_str)
    if p.is_file():
        return p.resolve()
    alt = (base / path_str).resolve()
    if alt.is_file():
        return alt
    alt2 = (REPO_ROOT / path_str).resolve()
    if alt2.is_file():
        return alt2
    raise FileNotFoundError(f"文件不存在: {path_str}")


def _features_csv_path(case_name: str, sample_id: str) -> Path:
    return REPO_ROOT / "data_new" / "AG" / case_name / "processed" / "features" / f"{sample_id}.csv"


def _transform_params_path(case_name: str) -> Path:
    return (
        REPO_ROOT / "data_new" / "AG" / case_name / "processed" / "coord_normalized" / "transform_params.json"
    )


def _load_features_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"x", "y", "z", "u", "v", "w", "p", "wss", "is_wall"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"features CSV 缺少列: {sorted(missing)} ({path})")
    return df


def _load_prediction(path: Path) -> Dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    for key in ("x", "y_true", "y_pred", "y_wss_true", "y_wss_pred", "wall_mask"):
        if key not in payload:
            raise SystemExit(f"预测文件缺少字段 {key!r}: {path}")
    return payload


def _coords_from_features(df: pd.DataFrame) -> Tuple[np.ndarray, str]:
    coords = df[["x", "y", "z"]].to_numpy(dtype=np.float64)
    return coords, "features_csv"


def _coords_from_inverse_transform(
    x_norm: np.ndarray,
    case_name: str,
) -> Tuple[np.ndarray, str]:
    tp_path = _transform_params_path(case_name)
    if not tp_path.is_file():
        raise SystemExit(
            f"features CSV 不可用且缺少 transform_params.json: {tp_path}\n"
            "请确认病例已运行 coord_normalize，或补齐 features CSV。"
        )
    transform_params = json.loads(tp_path.read_text(encoding="utf-8"))
    coords, _ = inverse_transform(x_norm, transform_params=transform_params)
    return coords.astype(np.float64), "inverse_transform"


def _build_export_frame(
    *,
    coords: np.ndarray,
    coord_source: str,
    is_wall: np.ndarray,
    cfd_u: np.ndarray,
    cfd_v: np.ndarray,
    cfd_w: np.ndarray,
    cfd_p: np.ndarray,
    cfd_wss: np.ndarray,
    pred_u: np.ndarray,
    pred_v: np.ndarray,
    pred_w: np.ndarray,
    pred_p: np.ndarray,
    pred_wss: np.ndarray,
) -> pd.DataFrame:
    cfd_vm = _vel_mag(cfd_u, cfd_v, cfd_w)
    pred_vm = _vel_mag(pred_u, pred_v, pred_w)

    df = pd.DataFrame(
        {
            "x": coords[:, 0],
            "y": coords[:, 1],
            "z": coords[:, 2],
            "is_wall": is_wall.astype(np.int8),
            "u_cfd": cfd_u,
            "v_cfd": cfd_v,
            "w_cfd": cfd_w,
            "p_cfd": cfd_p,
            "wss_cfd": cfd_wss,
            "vel_mag_cfd": cfd_vm,
            "u_pred": pred_u,
            "v_pred": pred_v,
            "w_pred": pred_w,
            "p_pred": pred_p,
            "wss_pred": pred_wss,
            "vel_mag_pred": pred_vm,
            "err_u": pred_u - cfd_u,
            "err_v": pred_v - cfd_v,
            "err_w": pred_w - cfd_w,
            "err_p": pred_p - cfd_p,
            "err_wss": pred_wss - cfd_wss,
            "err_vel_mag": pred_vm - cfd_vm,
        }
    )
    df.attrs["coord_source"] = coord_source
    return df


def export_sample(
    *,
    manifest_path: Path,
    sample_id: str,
    case_name: str,
    output_dir: Path,
    norm_params_path: Path,
) -> Dict[str, Path]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    items = manifest.get("items", [])
    if not isinstance(items, list):
        raise SystemExit("manifest.json 缺少 items 列表")

    item = _find_manifest_item(items, sample_id, case_name)
    pred_path = _resolve_path(str(item["prediction_path"]), manifest_path.parent)
    payload = _load_prediction(pred_path)

    n_nodes = int(payload["x"].shape[0])
    stats = _load_norm_stats(norm_params_path)

    y_true = payload["y_true"].detach().cpu().numpy()
    y_pred = payload["y_pred"].detach().cpu().numpy()
    y_wss_true = payload["y_wss_true"].detach().cpu().numpy()
    y_wss_pred = payload["y_wss_pred"].detach().cpu().numpy()
    x_norm = payload["x"].detach().cpu().numpy()[:, 0:3]
    wall_mask_pt = payload["wall_mask"].detach().cpu().numpy().astype(bool)

    features_path = _features_csv_path(case_name, sample_id)
    if features_path.is_file():
        feat_df = _load_features_csv(features_path)
        if len(feat_df) != n_nodes:
            raise SystemExit(
                f"行数不一致: features={len(feat_df)} vs .pt={n_nodes} ({features_path})"
            )
        coords, coord_source = _coords_from_features(feat_df)
        is_wall = feat_df["is_wall"].to_numpy(dtype=np.float64) > 0.5
        cfd_u = feat_df["u"].to_numpy(dtype=np.float64)
        cfd_v = feat_df["v"].to_numpy(dtype=np.float64)
        cfd_w = feat_df["w"].to_numpy(dtype=np.float64)
        cfd_p = feat_df["p"].to_numpy(dtype=np.float64)
        cfd_wss = feat_df["wss"].to_numpy(dtype=np.float64)
    else:
        print(f"[warn] 未找到 features CSV，使用坐标逆变换: {features_path}")
        coords, coord_source = _coords_from_inverse_transform(x_norm, case_name)
        is_wall = wall_mask_pt
        cfd_u = _denorm_zscore(y_true[:, 0], stats, "u")
        cfd_v = _denorm_zscore(y_true[:, 1], stats, "v")
        cfd_w = _denorm_zscore(y_true[:, 2], stats, "w")
        cfd_p = _denorm_zscore(y_true[:, 3], stats, "p")
        cfd_wss = _denorm_zscore(y_wss_true[:, 0], stats, "wss")

    pred_u = _denorm_zscore(y_pred[:, 0], stats, "u")
    pred_v = _denorm_zscore(y_pred[:, 1], stats, "v")
    pred_w = _denorm_zscore(y_pred[:, 2], stats, "w")
    pred_p = _denorm_zscore(y_pred[:, 3], stats, "p")
    pred_wss = _denorm_zscore(y_wss_pred[:, 0], stats, "wss")

    export_df = _build_export_frame(
        coords=coords,
        coord_source=coord_source,
        is_wall=is_wall,
        cfd_u=cfd_u,
        cfd_v=cfd_v,
        cfd_w=cfd_w,
        cfd_p=cfd_p,
        cfd_wss=cfd_wss,
        pred_u=pred_u,
        pred_v=pred_v,
        pred_w=pred_w,
        pred_p=pred_p,
        pred_wss=pred_wss,
    )

    case_slug = case_name.replace("/", "__")
    stem = f"{case_slug}__{sample_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "all": output_dir / f"{stem}__all.csv",
        "wall": output_dir / f"{stem}__wall.csv",
        "interior": output_dir / f"{stem}__interior.csv",
    }
    export_df.to_csv(paths["all"], index=False, float_format="%.8g")
    export_df.loc[export_df["is_wall"] == 1].to_csv(paths["wall"], index=False, float_format="%.8g")
    export_df.loc[export_df["is_wall"] == 0].to_csv(paths["interior"], index=False, float_format="%.8g")

    n_wall = int((export_df["is_wall"] == 1).sum())
    n_interior = int((export_df["is_wall"] == 0).sum())
    print(f"样本: {sample_id} | 病例: {case_name}")
    print(f"坐标来源: {coord_source}")
    print(f"节点总数: {n_nodes} (壁面 {n_wall}, 腔内 {n_interior})")
    print(f"输出:")
    for label, p in paths.items():
        print(f"  [{label}] {p}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="导出 GNN 预测为 CFD-Post 点云 CSV（物理单位：Pa、m/s）"
    )
    parser.add_argument("--manifest", required=True, help="predictions_test(_best_wss)/manifest.json 路径")
    parser.add_argument("--sample-id", required=True, help="样本 ID，如 result_features_merged-1120")
    parser.add_argument("--case-name", required=True, help="病例名，如 slow/GUO_XI_JIANG")
    parser.add_argument(
        "--output-dir",
        default="tools/cfdpost_cloud_export/output",
        help="输出目录（相对仓库根目录或绝对路径）",
    )
    parser.add_argument(
        "--norm-params",
        default="data_new/AG/normalization_params_global.json",
        help="全局 z-score 反归一化参数",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    if not manifest_path.is_file():
        raise SystemExit(f"manifest 不存在: {manifest_path}")

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir

    norm_path = Path(args.norm_params)
    if not norm_path.is_absolute():
        norm_path = REPO_ROOT / norm_path
    if not norm_path.is_file():
        raise SystemExit(f"归一化参数不存在: {norm_path}")

    export_sample(
        manifest_path=manifest_path,
        sample_id=args.sample_id,
        case_name=args.case_name,
        output_dir=output_dir,
        norm_params_path=norm_path,
    )


if __name__ == "__main__":
    main()
