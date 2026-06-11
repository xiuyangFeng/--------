#!/usr/bin/env python3
"""G4 · 2D 网格 WSS 预测反映射到壁面点并计算 R²。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .grid import UnwrapGridConfig, graph_to_2d_sample, load_norm_stats, r2_score, remap_grid_to_wall_points

REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser(description="G4 · 2D→3D WSS 反映射评估")
    ap.add_argument("--graph", type=Path, required=True)
    ap.add_argument("--pred-npz", type=Path, required=True, help="含 pred_wss (H,W) 或 (1,H,W)")
    ap.add_argument("--grid-s", type=int, default=64)
    ap.add_argument("--sectors", type=int, default=4)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    cfg = UnwrapGridConfig(grid_s=args.grid_s, n_sectors=args.sectors)
    stats = load_norm_stats()
    sample = graph_to_2d_sample(args.graph.resolve(), cfg=cfg, norm_stats=stats)
    if sample is None:
        raise SystemExit(f"无法展开: {args.graph}")

    pred_data = np.load(args.pred_npz)
    pred = pred_data["pred_wss"] if "pred_wss" in pred_data else pred_data["y_pred"]
    pred_pts, gt_norm = remap_grid_to_wall_points(pred, sample["remap"], stats=stats)
    _, gt_phys = remap_grid_to_wall_points(sample["y_wss_phys"], sample["remap"], use_phys=True, stats=stats)

    pred_phys = pred_pts * float(stats.get("wss", {}).get("std", 1.0)) + float(stats.get("wss", {}).get("mean", 0.0))
    report = {
        "graph": str(args.graph.resolve()),
        "r2_wss_norm_points": r2_score(gt_norm, pred_pts),
        "r2_wss_phys_points": r2_score(sample["remap"]["wss_phys"], pred_phys),
        "r2_wss_grid": r2_score(sample["y_wss_phys"], pred[np.newaxis] if pred.ndim == 2 else pred),
    }
    print(json.dumps(report, indent=2))
    if args.output:
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
