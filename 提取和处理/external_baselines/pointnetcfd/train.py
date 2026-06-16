from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from typing import Dict, List

import torch
from torch import nn
from torch.utils.data import DataLoader

from training.core.splits import SplitSpec

from .data import TARGET_MODES, PointCloudBatch, build_datasets, collate_pointclouds
from .evaluate import evaluate_checkpoint
from .metrics import regression_metrics
from .model import PointNetCFD
from .reporting import write_run_reports
from .utils import dump_json, load_config, move_batch, resolve_device, set_seed, timestamp


def run_epoch(
    model: PointNetCFD,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    output_names: List[str],
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    loss_fn = nn.MSELoss()
    preds: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []
    total_loss = 0.0
    total_nodes = 0

    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_train):
            pred = model(batch.node_input, batch.global_cond, batch.batch)
            loss = loss_fn(pred, batch.target)
            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
        total_loss += float(loss.item()) * batch.target.shape[0]
        total_nodes += batch.target.shape[0]
        preds.append(pred.detach().cpu())
        targets.append(batch.target.detach().cpu())

    metrics = regression_metrics(torch.cat(preds, dim=0), torch.cat(targets, dim=0), output_names)
    metrics["loss"] = total_loss / max(total_nodes, 1)
    return metrics


def build_run_dir(config: Dict[str, object], split_version: str) -> Path:
    run_cfg = config["run"]
    system_cfg = config["system"]
    name = f"{run_cfg['experiment_name']}_{split_version}_seed{system_cfg['seed']}_{timestamp()}"
    run_dir = Path(run_cfg.get("output_root", "outputs/external_baselines/pointnetcfd")) / name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def dry_run(config: Dict[str, object]) -> None:
    target_mode = config["target"]["mode"]
    if target_mode not in TARGET_MODES:
        raise KeyError(f"未知 target.mode={target_mode}; 可选={sorted(TARGET_MODES)}")
    input_dim = len(config["data"]["node_features"]) + len(config["data"]["global_features"])
    global_dim = len(config["data"]["global_features"])
    output_dim = len(TARGET_MODES[target_mode])
    model = PointNetCFD(
        input_dim=input_dim,
        global_dim=global_dim,
        output_dim=output_dim,
        **config["model"],
    )
    param_count = sum(p.numel() for p in model.parameters())
    print("PointNetCFD dry-run OK")
    print(f"input_dim={input_dim}, global_dim={global_dim}, output_dim={output_dim}, params={param_count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="PointNetCFD-style external baseline trainer")
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
    output_names = datasets["train"].output_names

    loader_cfg = config["data"]
    loaders = {
        split_name: DataLoader(
            dataset,
            batch_size=loader_cfg.get("batch_size", 2),
            shuffle=(split_name == "train"),
            num_workers=loader_cfg.get("num_workers", 0),
            pin_memory=loader_cfg.get("pin_memory", False),
            collate_fn=collate_pointclouds,
        )
        for split_name, dataset in datasets.items()
    }

    model = PointNetCFD(
        input_dim=datasets["train"].input_dim,
        global_dim=datasets["train"].global_dim,
        output_dim=len(output_names),
        **config["model"],
    ).to(device)
    optim_cfg = config["optim"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=optim_cfg.get("lr", 5e-4),
        weight_decay=optim_cfg.get("weight_decay", 1e-4),
    )

    run_dir = build_run_dir(config, split.split_version)
    shutil.copy2(args.config, run_dir / "config.json")

    history_path = run_dir / "history.csv"
    best_val = float("inf")
    best_epoch = 0
    patience = optim_cfg.get("early_stopping_patience", 30)
    bad_epochs = 0

    with open(history_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "val_rmse", "val_mae"])
        writer.writeheader()

        for epoch in range(1, optim_cfg.get("epochs", 200) + 1):
            train_metrics = run_epoch(model, loaders["train"], device, optimizer, output_names)
            val_metrics = run_epoch(model, loaders["val"], device, None, output_names)
            writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": train_metrics["loss"],
                    "val_loss": val_metrics["loss"],
                    "val_rmse": val_metrics["rmse"],
                    "val_mae": val_metrics["mae"],
                }
            )
            f.flush()

            if val_metrics["loss"] < best_val:
                best_val = val_metrics["loss"]
                best_epoch = epoch
                bad_epochs = 0
                torch.save({"model": model.state_dict(), "config": config, "epoch": epoch}, run_dir / "best_model.pt")
            else:
                bad_epochs += 1

            print(
                f"epoch={epoch:04d} train_loss={train_metrics['loss']:.6f} "
                f"val_loss={val_metrics['loss']:.6f} val_rmse={val_metrics['rmse']:.6f}"
            )
            if bad_epochs >= patience:
                print(f"early stopping at epoch={epoch}, best_epoch={best_epoch}")
                break

    test_metrics = evaluate_checkpoint(run_dir / "best_model.pt", split_name="test", output_dir=run_dir)
    manifest = {
        "baseline": "PointNetCFD",
        "experiment_name": config["run"]["experiment_name"],
        "split_version": split.split_version,
        "seed": config["system"]["seed"],
        "device": str(device),
        "target_mode": config["target"]["mode"],
        "output_names": output_names,
        "node_features": config["data"]["node_features"],
        "global_features": config["data"]["global_features"],
        "dataset_sizes": {name: len(dataset) for name, dataset in datasets.items()},
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "test_metrics": test_metrics,
    }
    dump_json(run_dir / "manifest.json", manifest)
    write_run_reports(run_dir, manifest)
    print(f"run_dir={run_dir}")


if __name__ == "__main__":
    main()
