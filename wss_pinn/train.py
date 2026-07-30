from __future__ import annotations

import argparse
import json
import os
import platform
import random
import sys
from pathlib import Path

import numpy as np
import torch

from .config import ExperimentConfig
from .data.dataset import PhysicsDataset, as_tensor
from .models import WSSPINNModel
from .objectives import stage_losses
from .utils import atomic_write_json, git_state, guard_write_path, utc_now


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(requested: str, override: str | None = None) -> torch.device:
    name = override or requested
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(name)


def make_model(config: ExperimentConfig, device: torch.device, dtype: torch.dtype):
    return WSSPINNModel(config["model"]).to(device=device, dtype=dtype)


def _save_checkpoint(
    path: Path,
    *,
    model,
    optimizer,
    epoch: int,
    best_loss: float,
    config: ExperimentConfig,
) -> None:
    target = guard_write_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    torch.save(
        {
            "schema_version": 1,
            "epoch": int(epoch),
            "best_loss": float(best_loss),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "resolved_config": config.as_dict(),
            "resolved_config_sha256": config.resolved_sha256,
        },
        temporary,
    )
    os.replace(temporary, target)


def run(
    config: ExperimentConfig,
    *,
    dry_run: bool = False,
    device_override: str | None = None,
) -> dict:
    train_config = config["train"]
    seed_everything(int(train_config["seed"]))
    dtype = torch.float64 if train_config["precision"] == "float64" else torch.float32
    device = resolve_device(train_config["device"], device_override)
    dataset = PhysicsDataset(
        config["sampling"]["manifest_path"],
        verify=True,
        roles=config["data"]["train_roles"],
    )
    model = make_model(config, device, dtype)
    init_checkpoint = train_config.get("init_checkpoint")
    if init_checkpoint:
        payload = torch.load(init_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(payload["model"], strict=True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_config["learning_rate"]),
        weight_decay=float(train_config["weight_decay"]),
    )
    start_epoch = 0
    best_loss = float("inf")
    resume = train_config.get("resume")
    if resume:
        payload = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        start_epoch = int(payload["epoch"]) + 1
        best_loss = float(payload.get("best_loss", best_loss))

    epochs = 1 if dry_run else int(train_config["epochs"])
    run_dir = config.run_dir
    if not dry_run:
        guard_write_path(run_dir).mkdir(parents=True, exist_ok=True)
        if (run_dir / "checkpoints/last.pt").exists() and not resume:
            raise FileExistsError(f"refusing to overwrite existing run: {run_dir}")
        atomic_write_json(run_dir / "resolved_config.json", config.as_dict())
        atomic_write_json(
            run_dir / "environment.json",
            {
                "created_at": utc_now(),
                "python": sys.version,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_device": (
                    torch.cuda.get_device_name(device) if device.type == "cuda" else None
                ),
                "git": git_state(),
            },
        )

    history_path = run_dir / "training_progress.jsonl"
    last_report = {}
    for epoch in range(start_epoch, start_epoch + epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        accumulated = {}
        case_samples = dataset.epoch_samples(train_config, epoch)
        for sample in case_samples:
            descriptor = as_tensor(sample["descriptor"], device, dtype)
            wall_coords = as_tensor(sample["wall_coords"], device, dtype)
            wall_wss = as_tensor(sample["wall_wss"], device, dtype)
            interior_coords = as_tensor(sample["interior_coords"], device, dtype)
            velocity = as_tensor(sample["velocity"], device, dtype)
            pressure = as_tensor(sample["pressure"], device, dtype)
            losses = stage_losses(
                model,
                descriptor,
                wall_coords,
                wall_wss,
                interior_coords,
                velocity,
                pressure,
                config["loss"],
            )
            (losses["total"] / len(case_samples)).backward()
            for name, value in losses.items():
                accumulated[name] = accumulated.get(name, 0.0) + float(value.detach())
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        accumulated = {name: value / len(case_samples) for name, value in accumulated.items()}
        if not all(np.isfinite(value) for value in accumulated.values()):
            raise FloatingPointError(f"non-finite training loss at epoch {epoch}: {accumulated}")
        last_report = {
            "timestamp": utc_now(),
            "epoch": epoch,
            "stage": config.stage,
            "case_ids": [str(sample["case_id"]) for sample in case_samples],
            "loss": accumulated,
            "device": str(device),
        }
        if dry_run:
            break
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(last_report, ensure_ascii=False) + "\n")
        if accumulated["total"] < best_loss:
            best_loss = accumulated["total"]
            _save_checkpoint(
                run_dir / "checkpoints/best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_loss=best_loss,
                config=config,
            )
        if (
            (epoch + 1) % int(train_config["checkpoint_every"]) == 0
            or epoch + 1 == start_epoch + epochs
        ):
            _save_checkpoint(
                run_dir / "checkpoints/last.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_loss=best_loss,
                config=config,
            )
        if epoch % int(train_config["log_every"]) == 0:
            print(json.dumps(last_report, ensure_ascii=False), flush=True)
    result = {
        "status": "dry_run_passed" if dry_run else "completed",
        "stage": config.stage,
        "config_sha256": config.resolved_sha256,
        "last": last_report,
    }
    if not dry_run:
        atomic_write_json(run_dir / "training_summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda"])
    args = parser.parse_args()
    config = ExperimentConfig.from_json(args.config)
    result = run(config, dry_run=args.dry_run, device_override=args.device)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
