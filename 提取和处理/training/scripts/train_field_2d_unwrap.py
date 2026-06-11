#!/usr/bin/env python3
"""G4 · 2D 壁面展开 U-Net 训练（Phase 1 过拟合 / Phase 2 Probe）。"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from pipeline.wall_unwrap.grid import (
    UnwrapGridConfig,
    collect_graph_paths,
    graph_to_2d_sample,
    load_norm_stats,
    r2_score,
    remap_grid_to_wall_points,
)
from training.core.denylist import filter_case_names
from training.core.splits import SplitSpec
from training.core.utils import dump_json, ensure_dir, resolve_device, set_seed, timestamp
from training.models.unet2d_wss import UNet2DWSS

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class TrainConfig:
    mode: str
    cases: List[str]
    grid_s: int = 64
    n_sectors: int = 4
    epochs: int = 150
    lr: float = 1e-3
    batch_size: int = 8
    seed: int = 1
    device: str = "auto"
    exp_id: str = "V3P-G-60-2DUnwrap"
    go_r2: float = 0.95


class Unwrap2DDataset(Dataset):
    def __init__(self, graph_paths: List[Path], cfg: UnwrapGridConfig, stats: Dict, pad_square: bool = True) -> None:
        self.samples: List[Dict[str, Any]] = []
        self.cfg = cfg
        self.stats = stats
        self.pad_square = pad_square
        for gp in graph_paths:
            s = graph_to_2d_sample(gp, cfg=cfg, norm_stats=stats)
            if s is not None:
                self.samples.append(s)

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _pad_hw(arr: np.ndarray, size: int) -> np.ndarray:
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]
        c, h, w = arr.shape
        out = np.pad(
            arr,
            ((0, 0), (0, max(0, size - h)), (0, max(0, size - w))),
            mode="edge",
        )[:, :size, :size]
        return out

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self.samples[idx]
        size = self.cfg.grid_s
        x = self._pad_hw(s["x_feat"], size) if self.pad_square else s["x_feat"]
        y = self._pad_hw(s["y_wss_norm"], size) if self.pad_square else s["y_wss_norm"]
        y_phys = self._pad_hw(s["y_wss_phys"], size) if self.pad_square else s["y_wss_phys"]
        return {
            "x": torch.from_numpy(x),
            "y": torch.from_numpy(y),
            "y_phys": torch.from_numpy(y_phys),
            "occupied": torch.from_numpy(s["occupied"].astype(np.float32)),
        }


def _resolve_cases(mode: str, split: SplitSpec, data_root: Path, case_arg: str) -> Tuple[List[str], List[str], List[str]]:
    train = filter_case_names(split.train_cases, data_root)
    val = filter_case_names(split.val_cases, data_root)
    test = filter_case_names(split.test_cases, data_root)
    if mode == "overfit1c":
        c = case_arg or train[0]
        return [c], [], []
    if mode == "probe":
        return train, val, test
    if case_arg:
        return [c.strip() for c in case_arg.split(",") if c.strip()], [], []
    return train[:1], [], []


def _eval_loader(
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
    grid_gts_phys: List[np.ndarray] = []
    pt_preds: List[np.ndarray] = []
    pt_gts_phys: List[np.ndarray] = []
    offset = 0
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
                grid_preds.append(pred_phys)
                grid_gts_phys.append(y_phys[i, 0, :h, :w])
                if i < len(batch_raw):
                    pp, _ = remap_grid_to_wall_points(pred[i, 0, :h, :w], batch_raw[i]["remap"], stats=stats)
                    pt_preds.append(pp * wss_std + wss_mean)
                    pt_gts_phys.append(batch_raw[i]["remap"]["wss_phys"])
    gp = np.concatenate([g.ravel() for g in grid_preds])
    gg = np.concatenate([g.ravel() for g in grid_gts_phys])
    pp_arr = np.concatenate(pt_preds) if pt_preds else gp
    pg = np.concatenate(pt_gts_phys) if pt_gts_phys else gg
    return {
        "r2_wss_grid_phys": r2_score(gg, gp),
        "r2_wss_points_phys": r2_score(pg, pp_arr),
    }


def _build_loader(paths: List[Path], cfg: UnwrapGridConfig, stats: Dict, batch_size: int, shuffle: bool) -> Tuple[DataLoader, List[Dict[str, Any]]]:
    ds = Unwrap2DDataset(paths, cfg, stats)
    if len(ds) == 0:
        raise RuntimeError("无可用 2D 样本")
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False), ds.samples


def main() -> None:
    ap = argparse.ArgumentParser(description="V3P G4 2D U-Net 训练")
    ap.add_argument("--mode", choices=["overfit1c", "probe"], default="overfit1c")
    ap.add_argument("--case", default="", help="overfit1c 指定 case，默认 train 首例")
    ap.add_argument("--split", type=Path, default=REPO_ROOT / "training/splits/split_AG_v1.json")
    ap.add_argument("--data-root", type=Path, default=REPO_ROOT / "data_new/AG")
    ap.add_argument("--grid-s", type=int, default=64)
    ap.add_argument("--sectors", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--exp-id", default="V3P-G-60-2DUnwrap")
    ap.add_argument("--output-root", type=Path, default=REPO_ROOT / "outputs/field")
    ap.add_argument("--go-r2", type=float, default=0.95)
    args = ap.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)
    split = SplitSpec.from_json(args.split.resolve())
    data_root = args.data_root.resolve()
    stats = load_norm_stats()
    cfg = UnwrapGridConfig(grid_s=args.grid_s, n_sectors=args.sectors)

    train_cases, val_cases, test_cases = _resolve_cases(args.mode, split, data_root, args.case)
    train_paths = collect_graph_paths(train_cases, data_root)
    val_paths = collect_graph_paths(val_cases, data_root) if val_cases else []
    test_paths = collect_graph_paths(test_cases, data_root) if test_cases else []

    run_name = f"g4_2d_unwrap_{args.mode}_s{cfg.n_sectors}_seed{args.seed}_{timestamp()}"
    run_dir = ensure_dir(args.output_root / run_name)

    train_loader, train_raw = _build_loader(train_paths, cfg, stats, args.batch_size, shuffle=True)
    train_eval_loader, _ = _build_loader(train_paths, cfg, stats, args.batch_size, shuffle=False)
    val_loader, val_raw = (None, [])
    if val_paths:
        val_loader, val_raw = _build_loader(val_paths, cfg, stats, args.batch_size, shuffle=False)

    model = UNet2DWSS(in_channels=4, base_channels=32, out_channels=1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    loss_fn = nn.SmoothL1Loss()

    history: List[Dict[str, float]] = []
    best_r2 = -1e9
    best_epoch = 0
    go_overfit = False

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            pred = model(x)
            loss = loss_fn(pred, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            train_loss += float(loss.item())
            n_batches += 1

        metrics = _eval_loader(model, train_eval_loader, device, stats, train_raw)
        row = {"epoch": epoch, "train_loss": train_loss / max(n_batches, 1), **metrics}
        val_metrics: Dict[str, float] = {}
        if val_loader is not None:
            val_metrics = _eval_loader(model, val_loader, device, stats, val_raw)
            row.update({f"val_{k}": v for k, v in val_metrics.items()})
        history.append(row)

        score_key = "val_r2_wss_grid_phys" if val_metrics else "r2_wss_grid_phys"
        score = row.get(score_key, row["r2_wss_grid_phys"])
        if score > best_r2:
            best_r2 = score
            best_epoch = epoch
            torch.save({"model": model.state_dict(), "epoch": epoch, "cfg": asdict(cfg)}, run_dir / "best_model.pt")

        if args.mode == "overfit1c" and row["r2_wss_grid_phys"] >= args.go_r2:
            go_overfit = True
            break

        if epoch % 10 == 0 or epoch == 1:
            val_s = f" val_r2 {row.get('val_r2_wss_grid_phys', float('nan')):.4f}" if val_metrics else ""
            print(
                f"ep {epoch:3d} loss {row['train_loss']:.5f} "
                f"train_r2 {row['r2_wss_grid_phys']:.4f}{val_s} "
                f"pt_r2 {row['r2_wss_points_phys']:.4f}"
            )

    # test eval for probe mode
    test_metrics: Dict[str, float] = {}
    if test_paths and args.mode == "probe":
        ckpt = torch.load(run_dir / "best_model.pt", map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        test_loader, test_raw = _build_loader(test_paths, cfg, stats, args.batch_size, shuffle=False)
        test_metrics = _eval_loader(model, test_loader, device, stats, test_raw)

    summary = {
        "exp_id": args.exp_id,
        "mode": args.mode,
        "device": str(device),
        "grid_s": cfg.grid_s,
        "n_sectors": cfg.n_sectors,
        "train_cases": train_cases,
        "n_train_graphs": len(train_paths),
        "best_epoch": best_epoch,
        "best_score_key": "val_r2_wss_grid_phys" if val_paths else "r2_wss_grid_phys",
        "best_r2_wss_grid_phys": best_r2,
        "go_overfit": go_overfit,
        "go_threshold": args.go_r2,
        "test_metrics": test_metrics,
        "run_dir": str(run_dir),
    }
    dump_json(summary, run_dir / "summary.json")

    hist_path = run_dir / "history.csv"
    with hist_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader()
        w.writerows(history)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(run_dir)


if __name__ == "__main__":
    main()
