#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CROWN/Beihang 预测 → 与 GNN 相同格式的 CFD-Post 点云 CSV（壁面/全点）。

默认 ``--inference-mode full``：在 CROWN 预处理全点云上推理（对齐 paper_full 全局 max-pool），
再将壁面 features CSV 坐标最近邻映射到预测场，便于与 V3P 同一壁面点 / STL 插值流程对比。

``--inference-mode features_csv``：仅在 GNN features CSV 的 15000 点上推理（可视化对齐，非 paper 口径）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from external_baselines.crown_beihang.data import load_pkl_records, load_train_p_stats  # noqa: E402
from external_baselines.crown_beihang.model import CrownPointNet  # noqa: E402
from external_baselines.crown_beihang.utils import load_config, project_root, resolve_device  # noqa: E402

from tools.cfdpost_cloud_export.export_for_cfdpost import (  # noqa: E402
    _build_export_frame,
    _features_csv_path,
    _load_features_csv,
    _vel_mag,
)


def _frame_key_from_sample_id(sample_id: str) -> str:
    prefix = "result_features_"
    if sample_id.startswith(prefix):
        return sample_id[len(prefix) :]
    return sample_id


def _load_crown_record(
    preprocessed_root: Path,
    case_name: str,
    frame_key: str,
    point_filter: str,
) -> Dict[str, Any]:
    partial = preprocessed_root / "pkl" / "partial" / f"{case_name.replace('/', '__')}_{point_filter}.pkl"
    if not partial.is_file():
        raise SystemExit(f"CROWN partial pkl 不存在: {partial}")
    target_sid = f"{case_name}/{frame_key}"
    for rec in load_pkl_records(partial):
        if rec["sample_id"] == target_sid:
            return rec
    raise SystemExit(f"partial pkl 中未找到 {target_sid!r}: {partial}")


def _load_checkpoint(model: CrownPointNet, checkpoint: Path, device: torch.device) -> None:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if isinstance(payload, dict):
        if "model" in payload:
            state = payload["model"]
        elif "state_dict" in payload:
            state = payload["state_dict"]
        else:
            state = payload
    else:
        state = payload
    model.load_state_dict(state)
    model.eval()


@torch.no_grad()
def _forward_points(
    model: CrownPointNet,
    xyz: np.ndarray,
    device: torch.device,
    chunk_size: int,
) -> np.ndarray:
    """xyz: (N, 3) physical coords → pred (N, 4) u,v,w,p_norm."""
    n = xyz.shape[0]
    if n <= chunk_size:
        feat = torch.from_numpy(xyz.T.astype(np.float32)).unsqueeze(0).to(device)
        out = model(feat)[0].transpose(0, 1).cpu().numpy()
        return out

    chunks = []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        feat = torch.from_numpy(xyz[start:end].T.astype(np.float32)).unsqueeze(0).to(device)
        pred = model(feat)[0].transpose(0, 1).cpu()
        chunks.append(pred)
    return torch.cat(chunks, dim=0).numpy()


def _denorm_p(p_norm: np.ndarray, p_min: float, p_max: float) -> np.ndarray:
    return p_norm * (p_max - p_min) + p_min


def _nn_map_predictions(
    source_xyz: np.ndarray,
    source_pred: np.ndarray,
    query_xyz: np.ndarray,
) -> np.ndarray:
    tree = cKDTree(source_xyz)
    _, idx = tree.query(query_xyz, k=1, workers=-1)
    return source_pred[idx]


def export_crown_sample(
    *,
    config_path: Path,
    checkpoint_path: Path,
    case_name: str,
    sample_id: str,
    output_dir: Path,
    inference_mode: str = "full",
    chunk_size: int = 65536,
    device_name: str = "auto",
) -> Dict[str, Path]:
    config = load_config(config_path)
    data_cfg = config["data"]
    preprocessed_root = Path(data_cfg.get("preprocessed_root", "external_baselines/CROWN_Beihang/private_preprocessed_raw_ascii_v1"))
    if not preprocessed_root.is_absolute():
        preprocessed_root = project_root() / preprocessed_root
    point_filter = data_cfg.get("point_filter", "volume")
    if point_filter == "interior":
        point_filter = "volume"

    stats = load_train_p_stats(preprocessed_root / "stats" / "train_stats.json", point_filter)
    p_min, p_max = float(stats["p_min"]), float(stats["p_max"])

    features_path = _features_csv_path(case_name, sample_id)
    if not features_path.is_file():
        raise SystemExit(f"features CSV 不存在: {features_path}")
    feat_df = _load_features_csv(features_path)

    device = resolve_device(device_name)
    model = CrownPointNet(input_dim=3, output_dim=4).to(device)
    _load_checkpoint(model, checkpoint_path, device)

    frame_key = _frame_key_from_sample_id(sample_id)

    if inference_mode == "features_csv":
        source_xyz = feat_df[["x", "y", "z"]].to_numpy(dtype=np.float64)
        pred_all = _forward_points(model, source_xyz, device, chunk_size)
        pred_u, pred_v, pred_w = pred_all[:, 0], pred_all[:, 1], pred_all[:, 2]
        pred_p = _denorm_p(pred_all[:, 3], p_min, p_max)
        coords = source_xyz
        coord_source = "features_csv_crown_infer"
    elif inference_mode == "full":
        rec = _load_crown_record(preprocessed_root, case_name, frame_key, point_filter)
        full_xyz = rec["features"].T.astype(np.float64)  # (N, 3)
        pred_full = _forward_points(model, full_xyz, device, chunk_size)
        query_xyz = feat_df[["x", "y", "z"]].to_numpy(dtype=np.float64)
        pred_mapped = _nn_map_predictions(full_xyz, pred_full, query_xyz)
        pred_u, pred_v, pred_w = pred_mapped[:, 0], pred_mapped[:, 1], pred_mapped[:, 2]
        pred_p = _denorm_p(pred_mapped[:, 3], p_min, p_max)
        coords = query_xyz
        coord_source = f"crown_full_nn_map(n={full_xyz.shape[0]:,})"
        print(f"[crown_export] full 推理 {full_xyz.shape[0]:,} 点 → NN 映射至 features {len(feat_df):,} 点")
    else:
        raise SystemExit(f"未知 inference_mode: {inference_mode}")

    is_wall = feat_df["is_wall"].to_numpy(dtype=np.float64) > 0.5
    cfd_u = feat_df["u"].to_numpy(dtype=np.float64)
    cfd_v = feat_df["v"].to_numpy(dtype=np.float64)
    cfd_w = feat_df["w"].to_numpy(dtype=np.float64)
    cfd_p = feat_df["p"].to_numpy(dtype=np.float64)
    cfd_wss = feat_df["wss"].to_numpy(dtype=np.float64)
    # CROWN 不预测 WSS；插值/出图仅压力使用 pred，WSS 列填 NaN
    pred_wss = np.full_like(cfd_wss, np.nan)

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
    print(f"样本: {sample_id} | 病例: {case_name} | CROWN {inference_mode}")
    print(f"坐标/真值来源: features CSV · 预测: CROWN checkpoint")
    print(f"coord_source: {coord_source}")
    print(f"节点总数: {len(export_df)} (壁面 {n_wall}, 腔内 {n_interior})")
    for label, p in paths.items():
        print(f"  [{label}] {p}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="导出 CROWN 预测为 CFD-Post 点云 CSV")
    parser.add_argument("--config", required=True, help="CROWN config.json")
    parser.add_argument("--checkpoint", required=True, help="best_model.pt")
    parser.add_argument("--case-name", required=True, help="如 slow/GUO_XI_JIANG")
    parser.add_argument("--sample-id", required=True, help="如 result_features_merged-1146")
    parser.add_argument("--output-dir", default="tools/cfdpost_cloud_export/output/crown")
    parser.add_argument(
        "--inference-mode",
        choices=("full", "features_csv"),
        default="full",
        help="full=全点云推理+壁面NN；features_csv=仅在 features 15000 点上推理",
    )
    parser.add_argument("--chunk-size", type=int, default=65536)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir

    export_crown_sample(
        config_path=Path(args.config).resolve(),
        checkpoint_path=Path(args.checkpoint).resolve(),
        case_name=args.case_name,
        sample_id=args.sample_id,
        output_dir=output_dir,
        inference_mode=args.inference_mode,
        chunk_size=args.chunk_size,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
