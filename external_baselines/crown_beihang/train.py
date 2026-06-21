from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from training.core.splits import SplitSpec

from .data import build_datasets, collate_crown, load_train_p_stats
from .evaluate import evaluate_checkpoint
from .model import CrownPointNet
from .physics import physics_residual_loss
from .reporting import write_run_reports
from .source_utils import wall_mask_from_velocity
from .utils import default_private_preprocessed_root, dump_json, load_config, project_root, resolve_device, set_seed, timestamp


def build_run_dir(config: Dict[str, Any], split_version: str) -> Path:
    run_cfg = config["run"]
    system_cfg = config["system"]
    name = f"{run_cfg['experiment_name']}_{split_version}_seed{system_cfg['seed']}_{timestamp()}"
    run_dir = Path(run_cfg.get("output_root", "outputs/external_baselines/crown_beihang")) / name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _p_stats(config: Dict[str, Any]) -> tuple[float, float]:
    root = project_root()
    data_cfg = config["data"]
    output_root = Path(data_cfg.get("preprocessed_root", str(default_private_preprocessed_root())))
    if not output_root.is_absolute():
        output_root = root / output_root
    point_filter = data_cfg.get("point_filter", "volume")
    stats = load_train_p_stats(output_root / "stats" / "train_stats.json", point_filter)
    return float(stats["p_min"]), float(stats["p_max"])


def _prepare_minibatch(
    batch: Dict[str, Any],
    sample_points: int,
    device: torch.device,
    separate_phy: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """与源码一致：每个样本随机抽 sample_points；PINN 时 data/phy 用不同 idx。"""
    input_indices = batch["input_indices"]
    pv_batch: List[torch.Tensor] = []
    label_batch: List[torch.Tensor] = []
    pv_phy_batch: List[torch.Tensor] = []

    for feat, lab in zip(batch["features"], batch["targets"]):
        feat = feat.to(device)
        lab = lab.to(device)
        n = feat.shape[1]
        choice = min(sample_points, n)
        idx = torch.randperm(n, device=device)[:choice]
        model_in = feat[input_indices][:, idx]
        pv_batch.append(model_in)
        label_batch.append(lab[:, idx])

        if separate_phy:
            idx_phy = torch.randperm(n, device=device)[:choice]
            xyz = feat[0:3][:, idx_phy]
            if model_in.shape[0] > 3:
                extra = feat[input_indices[3:]][:, idx_phy] if len(input_indices) > 3 else None
                if extra is not None and extra.shape[0] > 0:
                    pv_phy_batch.append(torch.cat([xyz, extra], dim=0))
                else:
                    pv_phy_batch.append(xyz)
            else:
                pv_phy_batch.append(xyz)
        else:
            pv_phy_batch.append(model_in)

    return (
        torch.stack(pv_batch, dim=0),
        torch.stack(label_batch, dim=0),
        torch.stack(pv_phy_batch, dim=0),
    )


def _wall_loss_source(
    model: CrownPointNet,
    pv_data: torch.Tensor,
    labels: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    """model_train_pinn.py：速度近零作壁面，no-slip 在壁面坐标点上。"""
    wall_mask = wall_mask_from_velocity(labels, threshold)
    bsz = pv_data.size(0)
    loss_wall = pv_data.new_tensor(0.0)
    count = 0
    for b in range(bsz):
        mask = wall_mask[b]
        if not mask.any():
            continue
        pv_b = pv_data[b : b + 1, 0:3, mask]
        pred_w = model(pv_b)
        u, v, w = pred_w[:, 0], pred_w[:, 1], pred_w[:, 2]
        loss_wall = loss_wall + (u.square() + v.square() + w.square()).mean()
        count += 1
    if count == 0:
        return loss_wall
    return loss_wall / count


def train_step_nopinn(
    model: CrownPointNet,
    optimizer: torch.optim.Optimizer,
    batch: Dict[str, Any],
    scaler: GradScaler,
    device: torch.device,
    sample_points: int,
) -> tuple[float, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    pv_data, labels, _ = _prepare_minibatch(batch, sample_points, device, separate_phy=False)

    with autocast():
        pred = model(pv_data)
        diff = pred - labels
        loss = diff.square().mean()
        mae = diff.abs().mean()

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return float(loss.item()), float(mae.item())


def train_step_pinn(
    model: CrownPointNet,
    optimizer: torch.optim.Optimizer,
    batch: Dict[str, Any],
    scaler: GradScaler,
    device: torch.device,
    lphy: float,
    reynolds: float,
    sample_points: int,
    wall_threshold: float,
) -> Dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    pv_data, labels, pv_phy = _prepare_minibatch(batch, sample_points, device, separate_phy=True)

    pred = model(pv_data)
    loss_data = (pred - labels).square().mean()
    loss_wall = _wall_loss_source(model, pv_data, labels, wall_threshold)
    loss_phy, loss_pde, loss_cont = physics_residual_loss(model, pv_phy, reynolds)
    func = loss_phy + loss_wall
    loss = loss_data + lphy * func

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    return {
        "loss": float(loss.item()),
        "loss_data": float(loss_data.item()),
        "loss_phy": float(loss_phy.item()),
        "loss_pde": float(loss_pde.item()),
        "loss_continuity": float(loss_cont.item()),
        "loss_wall": float(loss_wall.item()),
        "func": float(func.item()),
    }


@torch.no_grad()
def eval_step_nopinn(
    model: CrownPointNet,
    batch: Dict[str, Any],
    device: torch.device,
    sample_points: int,
) -> tuple[float, float]:
    model.eval()
    pv_data, labels, _ = _prepare_minibatch(batch, sample_points, device, separate_phy=False)
    pred = model(pv_data)
    diff = pred - labels
    return float(diff.square().mean().item()), float(diff.abs().mean().item())


def eval_step_pinn(
    model: CrownPointNet,
    batch: Dict[str, Any],
    device: torch.device,
    lphy: float,
    reynolds: float,
    sample_points: int,
    wall_threshold: float,
) -> Dict[str, float]:
    model.eval()
    with torch.enable_grad():
        pv_data, labels, pv_phy = _prepare_minibatch(batch, sample_points, device, separate_phy=True)
        pred = model(pv_data)
        loss_data = (pred - labels).square().mean()
        loss_wall = _wall_loss_source(model, pv_data, labels, wall_threshold)
        loss_phy, loss_pde, loss_cont = physics_residual_loss(model, pv_phy, reynolds)
        func = loss_phy + loss_wall
        loss = loss_data + lphy * func
    return {
        "loss": float(loss.item()),
        "loss_data": float(loss_data.item()),
        "loss_phy": float(loss_phy.item()),
        "loss_pde": float(loss_pde.item()),
        "loss_continuity": float(loss_cont.item()),
        "loss_wall": float(loss_wall.item()),
        "func": float(func.item()),
    }


def run_epoch_nopinn(
    model: CrownPointNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    sample_points: int,
) -> Dict[str, float]:
    scaler = GradScaler()
    is_train = optimizer is not None
    loss_sum = 0.0
    mae_sum = 0.0
    n = 0
    for batch in loader:
        if is_train:
            loss, mae = train_step_nopinn(model, optimizer, batch, scaler, device, sample_points)
        else:
            loss, mae = eval_step_nopinn(model, batch, device, sample_points)
        loss_sum += loss
        mae_sum += mae
        n += 1
    return {"loss": loss_sum / max(n, 1), "mae": mae_sum / max(n, 1)}


def run_epoch_pinn(
    model: CrownPointNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    lphy: float,
    reynolds: float,
    sample_points: int,
    wall_threshold: float,
) -> Dict[str, float]:
    scaler = GradScaler()
    is_train = optimizer is not None
    keys = ["loss", "loss_data", "loss_phy", "loss_pde", "loss_continuity", "loss_wall", "func"]
    sums = {k: 0.0 for k in keys}
    n = 0
    for batch in loader:
        if is_train:
            metrics = train_step_pinn(
                model, optimizer, batch, scaler, device, lphy, reynolds, sample_points, wall_threshold
            )
        else:
            metrics = eval_step_pinn(model, batch, device, lphy, reynolds, sample_points, wall_threshold)
        for k in keys:
            sums[k] += metrics[k]
        n += 1
    return {k: sums[k] / max(n, 1) for k in keys}


def dry_run(config: Dict[str, Any]) -> None:
    input_dim = len(config["data"]["input_features"])
    model = CrownPointNet(input_dim=input_dim, output_dim=4)
    params = sum(p.numel() for p in model.parameters())
    print(f"CROWN dry-run OK input_dim={input_dim} params={params} pinn={config.get('physics', {}).get('enabled', False)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="CROWN/Beihang external baseline trainer")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.dry_run:
        dry_run(config)
        return

    split = SplitSpec.from_json(config["data"]["split_file"])
    set_seed(config["system"]["seed"], deterministic=config["system"].get("deterministic", True))
    device = resolve_device(config["system"].get("device", "auto"))
    datasets = build_datasets(config)
    data_cfg = config["data"]
    sample_points = int(data_cfg.get("sample_points", 10000))
    physics_cfg = config.get("physics", {})
    wall_threshold = float(physics_cfg.get("wall_velocity_threshold", 0.01))

    loaders = {
        name: DataLoader(
            ds,
            batch_size=data_cfg.get("batch_size", 16),
            shuffle=(name == "train"),
            num_workers=data_cfg.get("num_workers", 0),
            pin_memory=data_cfg.get("pin_memory", False),
            collate_fn=collate_crown,
        )
        for name, ds in datasets.items()
    }

    input_dim = datasets["train"].input_dim
    model = CrownPointNet(input_dim=input_dim, output_dim=4).to(device)
    optim_cfg = config["optim"]
    optimizer = torch.optim.Adam(model.parameters(), lr=optim_cfg.get("lr", 3e-3))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.98, patience=10, min_lr=1e-4
    )

    use_pinn = bool(physics_cfg.get("enabled", False))
    reynolds = float(physics_cfg.get("reynolds", 300))
    lphy = float(physics_cfg.get("lphy_init", 1.0))

    run_dir = build_run_dir(config, split.split_version)
    shutil.copy2(args.config, run_dir / "config.json")
    history_path = run_dir / "history.csv"

    if use_pinn:
        fieldnames = [
            "epoch", "train_loss", "val_loss", "loss_data", "loss_phy",
            "loss_pde", "loss_continuity", "loss_wall", "func", "lphy",
        ]
    else:
        fieldnames = ["epoch", "train_loss", "val_loss", "train_mae", "val_mae"]

    best_val = float("inf")
    best_epoch = 0
    patience = optim_cfg.get("early_stopping_patience")
    bad_epochs = 0
    log_every = int(optim_cfg.get("log_every", 10))
    epochs = int(optim_cfg.get("epochs", 20000))

    with open(history_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for epoch in range(1, epochs + 1):
            if use_pinn:
                train_m = run_epoch_pinn(
                    model, loaders["train"], optimizer, device, lphy, reynolds, sample_points, wall_threshold
                )
                val_m = run_epoch_pinn(
                    model, loaders["val"], None, device, lphy, reynolds, sample_points, wall_threshold
                )
                row = {
                    "epoch": epoch,
                    "train_loss": train_m["loss"],
                    "val_loss": val_m["loss"],
                    "loss_data": train_m["loss_data"],
                    "loss_phy": train_m["loss_phy"],
                    "loss_pde": train_m["loss_pde"],
                    "loss_continuity": train_m["loss_continuity"],
                    "loss_wall": train_m["loss_wall"],
                    "func": train_m["func"],
                    "lphy": lphy,
                }
                val_loss = val_m["loss"]
            else:
                train_m = run_epoch_nopinn(model, loaders["train"], optimizer, device, sample_points)
                val_m = run_epoch_nopinn(model, loaders["val"], None, device, sample_points)
                row = {
                    "epoch": epoch,
                    "train_loss": train_m["loss"],
                    "val_loss": val_m["loss"],
                    "train_mae": train_m["mae"],
                    "val_mae": val_m["mae"],
                }
                val_loss = val_m["loss"]

            writer.writerow(row)
            f.flush()
            scheduler.step(val_loss)

            if epoch % log_every == 0:
                if use_pinn:
                    if val_m["func"] > 0:
                        lphy = max(min(train_m["loss_data"] / (val_m["func"] + 1e-9), 100.0), 0.001)
                    print(
                        f"Epoch {epoch:04d} train={train_m['loss']:.5f} val={val_m['loss']:.5f} "
                        f"data={train_m['loss_data']:.5f} func={train_m['func']:.5f} lphy={lphy:.5f}",
                        flush=True,
                    )
                else:
                    print(
                        f"Epoch {epoch:04d} train={train_m['loss']:.5f} val={val_m['loss']:.5f} "
                        f"mae={train_m['mae']:.5f}",
                        flush=True,
                    )

            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch
                bad_epochs = 0
                torch.save(
                    {"model": model.state_dict(), "config": config, "epoch": epoch},
                    run_dir / "best_model.pt",
                )
            else:
                bad_epochs += 1

            if patience is not None and bad_epochs >= patience:
                print(f"early stopping at epoch={epoch}, best_epoch={best_epoch}", flush=True)
                break

    p_min, p_max = _p_stats(config)
    test_metrics = evaluate_checkpoint(
        run_dir / "best_model.pt",
        split_name="test",
        output_dir=run_dir,
        p_min=p_min,
        p_max=p_max,
    )
    manifest = {
        "baseline": "CROWN_Beihang",
        "experiment_name": config["run"]["experiment_name"],
        "split_version": split.split_version,
        "seed": config["system"]["seed"],
        "device": str(device),
        "input_features": config["data"]["input_features"],
        "point_filter": data_cfg.get("point_filter", "volume"),
        "physics_enabled": use_pinn,
        "output_names": list(datasets["train"].target_names),
        "dataset_sizes": {name: len(ds) for name, ds in datasets.items()},
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "p_min": p_min,
        "p_max": p_max,
        "test_metrics": test_metrics,
    }
    dump_json(run_dir / "manifest.json", manifest)
    write_run_reports(run_dir, manifest)
    print(f"run_dir={run_dir}", flush=True)


if __name__ == "__main__":
    main()
