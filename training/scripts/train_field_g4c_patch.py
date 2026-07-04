#!/usr/bin/env python3
"""G4-c Phase 1 · Geodesic Patch CNN 单 case 过拟合（V3P · M-E 结构探针）。"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from pipeline.wall_unwrap.geodesic_patch import (
    GeodesicPatchConfig,
    extract_graph_patch_samples,
    merged_graph_point_r2,
)
from pipeline.wall_unwrap.grid import collect_graph_paths, load_norm_stats, r2_score
from training.core.denylist import filter_case_names
from training.core.splits import SplitSpec
from training.core.utils import dump_json, ensure_dir, resolve_device, set_seed, timestamp
from training.models.unet2d_wss import PatchConvWSS, UNet2DWSS

REPO_ROOT = Path(__file__).resolve().parents[2]

# I6-diag best_wss · 汇报三例 81 帧 pooled（postview 同口径 · 仅供 Phase 1 对照）
I6_DIAG_CASE_R2 = {
    "slow/GUO_XI_JIANG": 0.5703,
    "slow/ZHANG_JUN_HUA": 0.3520,
    "fast/CHEN_SHI_MING": 0.3630,
}


class PatchDataset(Dataset):
    def __init__(self, samples: List[Dict[str, Any]]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self.samples[idx]
        return {
            "x": torch.from_numpy(s["x_feat"]),
            "y": torch.from_numpy(s["y_wss_norm"]),
            "y_phys": torch.from_numpy(s["y_wss_phys"]),
            "occupied": torch.from_numpy(s["occupied"]),
        }


def _collect_samples(
    graph_paths: List[Path],
    cfg: GeodesicPatchConfig,
    stats: Dict,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for gp in graph_paths:
        out.extend(extract_graph_patch_samples(gp, cfg=cfg, stats=stats))
    return out


def _masked_loss(pred: torch.Tensor, target: torch.Tensor, occ: torch.Tensor) -> torch.Tensor:
    mask = occ.unsqueeze(1) > 0.5
    if not mask.any():
        return pred.sum() * 0.0
    return nn.functional.smooth_l1_loss(pred[mask], target[mask])


def _eval_patches(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    stats: Dict,
    samples_raw: List[Dict[str, Any]],
) -> Dict[str, float]:
    model.eval()
    wss_std = float(stats.get("wss", {}).get("std", 1.0))
    wss_mean = float(stats.get("wss", {}).get("mean", 0.0))
    grid_preds: List[np.ndarray] = []
    grid_gts: List[np.ndarray] = []
    pt_preds: List[np.ndarray] = []
    pt_gts: List[np.ndarray] = []
    offset = 0
    graph_preds: Dict[str, List[np.ndarray]] = {}
    graph_samples: Dict[str, List[Dict[str, Any]]] = {}

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            pred = model(x).cpu().numpy()
            y_phys = batch["y_phys"].numpy()
            bs = pred.shape[0]
            batch_raw = samples_raw[offset : offset + bs]
            offset += bs
            for i in range(bs):
                h, w = batch_raw[i]["y_wss_phys"].shape[1], batch_raw[i]["y_wss_phys"].shape[2]
                pred_phys = pred[i, 0, :h, :w] * wss_std + wss_mean
                gt_phys = y_phys[i, 0, :h, :w]
                occ = batch_raw[i]["occupied"]
                m = occ > 0.5
                grid_preds.append(pred_phys[m])
                grid_gts.append(gt_phys[m])
                remap = batch_raw[i]["remap"]
                pred_pts = pred_phys[remap["u_bin"], remap["v_bin"]]
                pt_preds.append(pred_pts)
                pt_gts.append(remap["wss_phys"])
                gp = batch_raw[i]["graph_path"]
                graph_preds.setdefault(gp, []).append(pred[i])
                graph_samples.setdefault(gp, []).append(batch_raw[i])

    gp_all = np.concatenate(grid_preds) if grid_preds else np.array([])
    gg_all = np.concatenate(grid_gts) if grid_gts else np.array([])
    pp_all = np.concatenate(pt_preds) if pt_preds else gp_all
    pg_all = np.concatenate(pt_gts) if pt_gts else gg_all

    merged_r2s = [
        merged_graph_point_r2(Path(g), graph_preds[g], graph_samples[g])
        for g in graph_preds
        if graph_samples.get(g)
    ]
    merged_r2s = [m for m in merged_r2s if np.isfinite(m)]

    return {
        "r2_wss_grid_phys": r2_score(gg_all, gp_all) if gg_all.size else float("nan"),
        "r2_wss_points_phys": r2_score(pg_all, pp_all) if pg_all.size else float("nan"),
        "r2_wss_merged_graph_phys": float(np.mean(merged_r2s)) if merged_r2s else float("nan"),
    }


def _resolve_cases(mode: str, split: SplitSpec, data_root: Path, case_arg: str) -> List[str]:
    train = filter_case_names(split.train_cases, data_root)
    if mode == "overfit1c":
        return [case_arg or train[0]]
    if case_arg:
        return [c.strip() for c in case_arg.split(",") if c.strip()]
    return train[:1]


def main() -> None:
    ap = argparse.ArgumentParser(description="V3P G4-c Geodesic Patch Phase 1 过拟合")
    ap.add_argument("--mode", choices=["overfit1c"], default="overfit1c")
    ap.add_argument("--case", required=True, help="如 fast/CHEN_SHI_MING")
    ap.add_argument("--split", type=Path, default=REPO_ROOT / "training/splits/split_AG_v1.json")
    ap.add_argument("--data-root", type=Path, default=REPO_ROOT / "data_new/AG")
    ap.add_argument("--patch-h", type=int, default=12)
    ap.add_argument("--patch-w", type=int, default=12)
    ap.add_argument("--geodesic-radius-mm", type=float, default=3.0)
    ap.add_argument("--knn-k", type=int, default=20)
    ap.add_argument("--max-seeds-per-graph", type=int, default=40)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--exp-id", default="V3P-M-E-G4c-PatchPhase1-post5463")
    ap.add_argument("--output-root", type=Path, default=REPO_ROOT / "outputs/field")
    ap.add_argument("--go-r2-grid", type=float, default=0.95)
    ap.add_argument("--go-delta-i6", type=float, default=0.05, help="merged 3D R² 相对 I6-diag 同病例最小 Δ")
    args = ap.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)
    split = SplitSpec.from_json(args.split.resolve())
    data_root = args.data_root.resolve()
    stats = load_norm_stats()

    patch_cfg = GeodesicPatchConfig(
        patch_h=args.patch_h,
        patch_w=args.patch_w,
        geodesic_radius_mm=args.geodesic_radius_mm,
        knn_k=args.knn_k,
        max_seeds_per_graph=args.max_seeds_per_graph,
    )

    cases = _resolve_cases(args.mode, split, data_root, args.case)
    graph_paths = collect_graph_paths(cases, data_root)
    if not graph_paths:
        raise RuntimeError(f"无 graph: {cases}")

    print(f"收集 patch 样本: {len(graph_paths)} graphs · case {cases[0]}", flush=True)
    samples = _collect_samples(graph_paths, patch_cfg, stats)
    if not samples:
        raise RuntimeError(f"无 patch 样本: {cases}")
    print(f"patch 样本数: {len(samples)}", flush=True)

    case_slug = cases[0].replace("/", "__")
    run_name = (
        f"g4c_patch_{args.mode}_{case_slug}_p{args.patch_h}_r{args.geodesic_radius_mm}_k{args.knn_k}"
        f"_seed{args.seed}_{timestamp()}"
    )
    run_dir = ensure_dir(args.output_root / run_name)

    ds = PatchDataset(samples)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    eval_loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    use_patch_cnn = min(args.patch_h, args.patch_w) < 32
    if use_patch_cnn:
        model = PatchConvWSS(in_channels=4, base_channels=32, out_channels=1).to(device)
        print("model: PatchConvWSS (小 patch)", flush=True)
    else:
        model = UNet2DWSS(in_channels=4, base_channels=16, out_channels=1).to(device)
        print("model: UNet2DWSS", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    i6_ref = I6_DIAG_CASE_R2.get(cases[0])
    history: List[Dict[str, float]] = []
    best_score = -1e9
    best_epoch = 0
    go_overfit = False

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        n_batches = 0
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            occ = batch["occupied"].to(device)
            pred = model(x)
            loss = _masked_loss(pred, y, occ)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            train_loss += float(loss.item())
            n_batches += 1

        metrics = _eval_patches(model, eval_loader, device, stats, samples)
        row = {"epoch": epoch, "train_loss": train_loss / max(n_batches, 1), **metrics}
        history.append(row)

        score = metrics["r2_wss_merged_graph_phys"]
        if score > best_score:
            best_score = score
            best_epoch = epoch
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "patch_cfg": asdict(patch_cfg)},
                run_dir / "best_model.pt",
            )

        delta_i6 = score - i6_ref if i6_ref is not None else float("nan")
        grid_ok = metrics["r2_wss_grid_phys"] >= args.go_r2_grid
        merged_ok = np.isfinite(delta_i6) and delta_i6 >= args.go_delta_i6
        if grid_ok and merged_ok:
            go_overfit = True
            print(f"Phase 1 Go at ep {epoch}: grid={metrics['r2_wss_grid_phys']:.4f} merged={score:.4f} Δi6={delta_i6:+.4f}")
            break

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"ep {epoch:3d} loss {row['train_loss']:.5f} "
                f"grid {metrics['r2_wss_grid_phys']:.4f} "
                f"pt {metrics['r2_wss_points_phys']:.4f} "
                f"merged {score:.4f} Δi6 {delta_i6:+.4f}"
            )

    summary = {
        "exp_id": args.exp_id,
        "mode": args.mode,
        "device": str(device),
        "patch_cfg": asdict(patch_cfg),
        "train_cases": cases,
        "n_train_graphs": len(graph_paths),
        "n_train_patches": len(samples),
        "best_epoch": best_epoch,
        "best_r2_wss_merged_graph_phys": best_score,
        "i6_diag_case_r2_ref": i6_ref,
        "delta_vs_i6_diag": best_score - i6_ref if i6_ref is not None else None,
        "go_overfit": go_overfit,
        "go_thresholds": {
            "r2_wss_grid_phys": args.go_r2_grid,
            "delta_vs_i6_diag": args.go_delta_i6,
        },
        "final_metrics": history[-1] if history else {},
        "run_dir": str(run_dir),
        "phase0_ref": "v3p_g4c_patch_sweep_summary_20260630.json · leading p12/r3/k20",
    }
    dump_json(summary, run_dir / "summary.json")

    with (run_dir / "history.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader()
        w.writerows(history)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(run_dir)


if __name__ == "__main__":
    main()
