"""Train one V4 arm using train-only convergence and no automatic test read."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from wss_pinn.utils import (
    atomic_write_json,
    git_state,
    guard_write_path,
    sha256_file,
    utc_now,
)
from wss_pinn.volume_utils import tensor_state_sha256

from .config import load_config
from .controller import PDEWeightController
from .data import V4Dataset, collate_v4
from .models import build_model
from .physics import compute_raw_losses, gradient_diagnostics


def seed_everything(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _move(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, non_blocking=True)
    if isinstance(value, dict):
        return {key: _move(item, device) for key, item in value.items()}
    return value


def _scheduler(
    optimizer: torch.optim.Optimizer,
    max_epochs: int,
    warmup_epochs: int,
    base_lr: float,
    min_lr: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    minimum = min_lr / base_lr

    def scale(epoch: int) -> float:
        if epoch < warmup_epochs:
            return max((epoch + 1) / max(warmup_epochs, 1), 1.0e-8)
        progress = min(
            max((epoch - warmup_epochs) / max(max_epochs - warmup_epochs, 1), 0.0),
            1.0,
        )
        return minimum + (1.0 - minimum) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def _atomic_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    target = guard_write_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _weighted_losses(
    raw: dict[str, torch.Tensor],
    *,
    lambda_bc: float,
    lambda_pde: float,
    transient: bool,
) -> dict[str, torch.Tensor]:
    values = dict(raw)
    bc_divisor = 3.0 if transient else 2.0
    values["no_slip_weighted"] = raw["no_slip_raw"] * lambda_bc / bc_divisor
    values["inlet_bc_weighted"] = raw["inlet_bc_raw"] * lambda_bc / bc_divisor
    values["rcr_bc_weighted"] = (
        raw["rcr_bc_raw"] * lambda_bc / 3.0 if transient else raw["rcr_bc_raw"] * 0.0
    )
    for outlet in range(4):
        values[f"rcr_bc_outlet_{outlet}_weighted"] = (
            raw[f"rcr_bc_outlet_{outlet}_raw"] * lambda_bc / 12.0
            if transient
            else raw[f"rcr_bc_outlet_{outlet}_raw"] * 0.0
        )
    values["bc_weighted"] = raw["bc_total"] * lambda_bc
    values["continuity_weighted"] = raw["continuity_raw"] * lambda_pde / 2.0
    for axis in ("x", "y", "z"):
        values[f"momentum_{axis}_weighted"] = (
            raw[f"momentum_{axis}_raw"] * lambda_pde / 6.0
        )
    values["momentum_group_weighted"] = raw["momentum_group_raw"] * lambda_pde / 2.0
    values["pde_weighted"] = raw["pde_total"] * lambda_pde
    values["physics_total"] = values["bc_weighted"] + values["pde_weighted"]
    values["total"] = raw["data_total"] + values["physics_total"]
    return values


def _checkpoint_payload(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    controller: PDEWeightController,
    config: Any,
    epoch: int,
    global_step: int,
    initialization_sha256: str,
    dataset_snapshot: dict[str, Any],
    converged: bool,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "route": config["route"],
        "epoch": int(epoch),
        "global_step": int(global_step),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "controller": controller.state_dict(),
        "initialization_state_sha256": initialization_sha256,
        "resolved_config": config.as_dict(),
        "resolved_config_sha256": config.resolved_sha256,
        "dataset_snapshot": dataset_snapshot,
        "train_only_converged": bool(converged),
    }


def _relative_change(left: float, right: float) -> float:
    return abs(right - left) / max(abs(left), 1.0e-30)


def _converged(
    history: list[dict[str, Any]], config: Any
) -> tuple[bool, dict[str, Any]]:
    contract = config["convergence"]
    current_epoch = len(history)
    if current_epoch < int(contract["min_epochs"]):
        return False, {"reason": "min_epochs"}
    window = int(contract["window_epochs"])
    consecutive = int(contract["consecutive_windows"])
    required = window * consecutive
    if len(history) < required:
        return False, {"reason": "insufficient_windows"}
    metrics = ["mean_data_total"]
    if bool(config["physics"]["bc_enabled"]):
        metrics.extend(["mean_no_slip_raw", "mean_inlet_bc_raw"])
        if str(config["experiment"]["temporal_mode"]) == "transient_81":
            metrics.append("mean_rcr_bc_raw")
    if bool(config["physics"]["pde_enabled"]):
        metrics.extend(["mean_continuity_raw", "mean_momentum_group_raw"])
    windows = []
    for index in range(consecutive):
        start = len(history) - required + index * window
        rows = history[start : start + window]
        windows.append(
            {metric: float(np.mean([row[metric] for row in rows])) for metric in metrics}
        )
    changes = {
        metric: max(
            _relative_change(windows[index - 1][metric], windows[index][metric])
            for index in range(1, consecutive)
        )
        for metric in metrics
    }
    threshold = float(contract["relative_change_threshold"])
    stable = all(value <= threshold for value in changes.values())
    last = history[-window:]
    clip_fraction = float(np.mean([row["gradient_clipping_fraction"] for row in last]))
    bound_fraction = float(
        np.mean(
            [
                row["lambda_clamped_fraction"]
                for row in last
            ]
        )
    )
    lr_ready = True
    if bool(contract["require_min_lr"]):
        lr_ready = history[-1]["learning_rate_next_epoch"] <= float(
            config["train"]["min_learning_rate"]
        ) * 1.05
    diagnostics = {
        "changes": changes,
        "threshold": threshold,
        "gradient_clipping_fraction": clip_fraction,
        "lambda_bound_fraction": bound_fraction,
        "lr_ready": lr_ready,
    }
    return (
        stable
        and clip_fraction <= float(contract["gradient_clip_fraction_limit"])
        and bound_fraction <= float(contract["lambda_bound_fraction_limit"])
        and lr_ready,
        diagnostics,
    )


def run(config_path: str | Path, *, dry_run: bool = False, device_override: str | None = None) -> dict[str, Any]:
    config = load_config(config_path, require_assets=True)
    seed_everything(int(config["train"]["seed"]))
    device_name = device_override or str(config["train"]["device"])
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(device_name)
    dataset = V4Dataset(config, roles=("train",))
    if len(dataset) != 138:
        raise ValueError("formal V4 training must see exactly train138")
    dataset_snapshot = dataset.dataset_snapshot
    model = build_model(config).to(device=device, dtype=torch.float32)
    initialization_sha256 = tensor_state_sha256(model.state_dict())
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["train"]["learning_rate"]),
        weight_decay=float(config["train"]["weight_decay"]),
    )
    scheduler = _scheduler(
        optimizer,
        int(config["train"]["max_epochs"]),
        int(config["train"]["warmup_epochs"]),
        float(config["train"]["learning_rate"]),
        float(config["train"]["min_learning_rate"]),
    )
    controller = PDEWeightController(config["loss_balancing"])
    start_epoch = 0
    global_step = 0
    epoch_history: list[dict[str, Any]] = []
    resume = config["train"].get("resume")
    if resume:
        checkpoint_path = Path(resume)
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if payload.get("route") != config["route"]:
            raise ValueError("resume route mismatch")
        if payload.get("initialization_state_sha256") != initialization_sha256:
            raise ValueError("resume initialization hash mismatch")
        saved_snapshot = payload.get("dataset_snapshot")
        if saved_snapshot is not None and saved_snapshot != dataset_snapshot:
            raise ValueError("resume dataset snapshot mismatch")
        if dataset.centerline_v2_schema and saved_snapshot is None:
            raise ValueError("Centerline V2 resume checkpoint lacks a dataset snapshot")
        old_config = payload.get("resolved_config", {})
        for key in ("id", "temporal_mode", "backbone", "training_mode", "seed"):
            if old_config.get("experiment", {}).get(key) != config["experiment"][key]:
                raise ValueError(f"resume experiment.{key} mismatch")
        model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        controller.load_state_dict(payload["controller"])
        start_epoch = int(payload["epoch"]) + 1
        global_step = int(payload["global_step"])
        history_path = config.run_dir / "epoch_progress.jsonl"
        if history_path.is_file():
            epoch_history = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines() if line]

    run_dir = config.run_dir
    if not dry_run:
        guard_write_path(run_dir).mkdir(parents=True, exist_ok=True)
        if (run_dir / "checkpoints/last.pt").exists() and not resume:
            raise FileExistsError(f"refusing to overwrite existing run: {run_dir}")
        if not resume:
            atomic_write_json(run_dir / "resolved_config.json", config.as_dict())
            atomic_write_json(
                run_dir / "environment.json",
                {
                    "created_at": utc_now(),
                    "python": sys.version,
                    "platform": platform.platform(),
                    "torch": torch.__version__,
                    "device": str(device),
                    "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                    "git": git_state(),
                    "manifest": {
                        "path": config["paths"]["manifest"],
                        "sha256": sha256_file(config["paths"]["manifest"]),
                    },
                    "stats": {
                        "path": config["paths"]["stats"],
                        "sha256": sha256_file(config["paths"]["stats"]),
                    },
                    "split": {
                        "path": config["paths"]["split"],
                        "sha256": sha256_file(config["paths"]["split"]),
                    },
                    "dataset_snapshot": dataset_snapshot,
                    "initialization_state_sha256": initialization_sha256,
                    "test35_read_during_training": False,
                    "warm_start": False,
                },
            )
            _atomic_checkpoint(
                run_dir / "checkpoints/initialization.pt",
                {
                    "schema_version": 1,
                    "route": config["route"],
                    "model": model.state_dict(),
                    "initialization_state_sha256": initialization_sha256,
                    "resolved_config_sha256": config.resolved_sha256,
                    "dataset_snapshot": dataset_snapshot,
                },
            )

    step_path = run_dir / "training_progress.jsonl"
    epoch_path = run_dir / "epoch_progress.jsonl"
    csv_path = run_dir / "history.csv"
    step_handle = epoch_handle = csv_handle = None
    csv_writer = None
    stopped = False
    stop_diagnostics: dict[str, Any] = {}
    executed_epochs = 0
    try:
        if not dry_run:
            step_handle = step_path.open("a", encoding="utf-8")
            epoch_handle = epoch_path.open("a", encoding="utf-8")
            csv_exists = csv_path.exists() and csv_path.stat().st_size > 0
            csv_handle = csv_path.open("a", encoding="utf-8", newline="")
        max_epoch = start_epoch + 1 if dry_run else int(config["train"]["max_epochs"])
        for epoch in range(start_epoch, max_epoch):
            dataset.set_epoch(epoch)
            generator = torch.Generator()
            generator.manual_seed(int(config["train"]["seed"]) + epoch * 104729)
            loader = DataLoader(
                dataset,
                batch_size=int(config["train"]["batch_cases"]),
                shuffle=True,
                generator=generator,
                num_workers=int(config["train"]["num_workers"]),
                collate_fn=collate_v4,
                pin_memory=device.type == "cuda",
            )
            model.train()
            sums: dict[str, float] = {}
            steps = 0
            clipped = 0
            lambda_clamped = 0
            for step_in_epoch, cpu_batch in enumerate(loader):
                batch = _move(cpu_batch, device)
                optimizer.zero_grad(set_to_none=True)
                raw, _ = compute_raw_losses(
                    model, batch, config, field_stats=dataset.field_stats
                )
                control = controller.update(
                    raw["data_total"],
                    raw["pde_total"],
                    global_step=global_step + 1,
                    epoch=epoch,
                )
                losses = _weighted_losses(
                    raw,
                    lambda_bc=float(config["physics"]["lambda_bc"]),
                    lambda_pde=control.lambda_pde,
                    transient=config.temporal_mode == "transient_81",
                )
                if not torch.isfinite(losses["total"]):
                    raise FloatingPointError(
                        f"non-finite total at epoch={epoch} step={step_in_epoch}"
                    )
                gradient_record = {
                    "grad_norm_data": 0.0,
                    "grad_norm_physics": 0.0,
                    "cos_grad_data_continuity": 0.0,
                    "cos_grad_data_momentum": 0.0,
                    "cos_grad_data_no_slip": 0.0,
                    "cos_grad_data_inlet_bc": 0.0,
                    "cos_grad_data_rcr_bc": 0.0,
                }
                if (global_step + 1) % int(
                    config["loss_balancing"]["gradient_check_interval_steps"]
                ) == 0:
                    gradient_record = gradient_diagnostics(model, losses)
                losses["total"].backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(config["train"]["grad_clip_norm"])
                )
                if not torch.isfinite(gradient_norm):
                    raise FloatingPointError("non-finite gradient norm")
                clip_flag = float(gradient_norm.detach().cpu()) > float(
                    config["train"]["grad_clip_norm"]
                )
                clipped += int(clip_flag)
                optimizer.step()
                global_step += 1
                steps += 1
                lambda_clamped += int(
                    control.lambda_clamped_low or control.lambda_clamped_high
                )
                record = {
                    "created_at": utc_now(),
                    "epoch": epoch,
                    "step_in_epoch": step_in_epoch,
                    "global_step": global_step,
                    "case_ids": "|".join(batch["case_ids"]),
                    "frame_indices": "|".join(
                        "none" if value is None else str(value)
                        for value in batch["frame_indices"]
                    ),
                    **{
                        name: float(value.detach().cpu())
                        for name, value in losses.items()
                    },
                    "lambda_phy": control.lambda_pde,
                    "lambda_bc": float(config["physics"]["lambda_bc"]),
                    "loss_ratio_raw": control.loss_ratio_raw,
                    "ema_data": control.ema_data,
                    "ema_physics": control.ema_pde,
                    "lambda_target": control.lambda_target,
                    "lambda_applied": control.lambda_applied,
                    "lambda_clamped_low": control.lambda_clamped_low,
                    "lambda_clamped_high": control.lambda_clamped_high,
                    "lambda_update_flag": control.lambda_update_flag,
                    **gradient_record,
                    "gradient_norm": float(gradient_norm.detach().cpu()),
                    "gradient_clip_flag": bool(clip_flag),
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                }
                numeric = [value for value in record.values() if isinstance(value, (int, float))]
                if not np.isfinite(numeric).all():
                    raise FloatingPointError("non-finite value in training record")
                for name, value in losses.items():
                    sums[name] = sums.get(name, 0.0) + float(value.detach().cpu())
                sums["lambda_phy"] = sums.get("lambda_phy", 0.0) + control.lambda_pde
                if not dry_run and (
                    global_step == 1
                    or global_step % int(config["train"]["log_every_steps"]) == 0
                ):
                    step_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    step_handle.flush()
                    if csv_writer is None:
                        csv_writer = csv.DictWriter(csv_handle, fieldnames=list(record))
                        if not csv_exists:
                            csv_writer.writeheader()
                    csv_writer.writerow(record)
                    csv_handle.flush()
                if dry_run:
                    break
            scheduler.step()
            executed_epochs += 1
            epoch_record = {
                "created_at": utc_now(),
                "epoch": epoch,
                "global_step": global_step,
                "steps": steps,
                **{f"mean_{name}": value / max(steps, 1) for name, value in sums.items()},
                "gradient_clipping_fraction": clipped / max(steps, 1),
                "lambda_clamped_fraction": lambda_clamped / max(steps, 1),
                "learning_rate_next_epoch": float(optimizer.param_groups[0]["lr"]),
            }
            epoch_history.append(epoch_record)
            if not dry_run:
                epoch_handle.write(json.dumps(epoch_record, ensure_ascii=False) + "\n")
                epoch_handle.flush()
                print(
                    json.dumps(
                        {
                            "event": "v4_epoch_completed",
                            "experiment_id": config["experiment"]["id"],
                            **epoch_record,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            converged, convergence_diagnostics = _converged(epoch_history, config)
            reached_max = epoch + 1 >= int(config["train"]["max_epochs"])
            if not dry_run:
                payload = _checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    controller=controller,
                    config=config,
                    epoch=epoch,
                    global_step=global_step,
                    initialization_sha256=initialization_sha256,
                    dataset_snapshot=dataset_snapshot,
                    converged=converged,
                )
                completed_epoch = epoch + 1
                if completed_epoch in set(int(value) for value in config["train"]["milestone_epochs"]):
                    _atomic_checkpoint(
                        run_dir / f"checkpoints/epoch_{completed_epoch:05d}.pt", payload
                    )
                if completed_epoch % int(config["train"]["checkpoint_every_epochs"]) == 0 or converged or reached_max:
                    _atomic_checkpoint(run_dir / "checkpoints/last.pt", payload)
                if converged or reached_max:
                    _atomic_checkpoint(run_dir / "checkpoints/last_converged.pt", payload)
            if converged or reached_max:
                stopped = True
                stop_diagnostics = {
                    "reason": "train_only_plateau" if converged else "max_epochs",
                    "converged": converged,
                    **convergence_diagnostics,
                }
                break
            if dry_run:
                break
    finally:
        for handle in (step_handle, epoch_handle, csv_handle):
            if handle is not None:
                handle.close()

    result = {
        "schema_version": 1,
        "route": config["route"],
        "experiment_id": config["experiment"]["id"],
        "dry_run": dry_run,
        "executed_epochs": executed_epochs,
        "global_step": global_step,
        "initialization_state_sha256": initialization_sha256,
        "stopped": stopped,
        "stop": stop_diagnostics,
        "test35_read_during_training": False,
    }
    if not dry_run:
        atomic_write_json(run_dir / "training_summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device")
    args = parser.parse_args()
    result = run(args.config, dry_run=args.dry_run, device_override=args.device)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
