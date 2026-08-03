"""Train PointNet/PointNet++ peak uvwp data-only and PINN experiments."""

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

from .config import VolumeExperimentConfig
from .data import VolumeFieldDataset, collate_volume_samples
from .losses import compute_losses
from .models import build_volume_model
from .utils import tensor_state_sha256


def seed_everything(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(name: str, override: str | None = None) -> torch.device:
    selected = override or name
    if selected == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(selected)


def _move_batch(batch: dict[str, Any], device: torch.device, dtype: torch.dtype) -> dict:
    output: dict[str, Any] = {}
    floating = {
        "support_coords",
        "support_features",
        "query_coords",
        "query_target",
        "physics_coords",
        "physics_length_m",
        "wall_coords",
    }
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            if key in floating:
                output[key] = value.to(device=device, dtype=dtype, non_blocking=True)
            else:
                output[key] = value.to(device=device, non_blocking=True)
        else:
            output[key] = value
    return output


def _scheduler(optimizer, epochs: int, warmup: int, min_lr: float, base_lr: float):
    minimum_ratio = float(min_lr) / float(base_lr)

    def scale(epoch: int) -> float:
        if warmup > 0 and epoch < warmup:
            return max((epoch + 1) / warmup, 1e-8)
        denominator = max(epochs - warmup, 1)
        progress = min(max((epoch - warmup) / denominator, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return minimum_ratio + (1.0 - minimum_ratio) * cosine

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


def _checkpoint_payload(
    *,
    model,
    optimizer,
    scheduler,
    epoch: int,
    global_step: int,
    best_data: float,
    best_total: float,
    config: VolumeExperimentConfig,
    initialization_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "route": "volume_uvwp_peak_v1",
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_data": float(best_data),
        "best_total": float(best_total),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "initialization_state_sha256": initialization_sha256,
        "resolved_config": config.as_dict(),
        "resolved_config_sha256": config.resolved_sha256,
    }


def _rms(value: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean(torch.square(value.detach().float()))).cpu())


def _quantile(value: torch.Tensor, q: float) -> float:
    return float(torch.quantile(value.detach().float(), float(q)).cpu())


def _step_record(
    *,
    losses: dict[str, torch.Tensor],
    diagnostics: dict[str, torch.Tensor],
    epoch: int,
    step_in_epoch: int,
    global_step: int,
    case_ids: list[str],
    batch: dict[str, Any],
    learning_rate: float,
    gradient_norm: float,
    mode: str,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "created_at": utc_now(),
        "epoch": int(epoch),
        "step_in_epoch": int(step_in_epoch),
        "global_step": int(global_step),
        "mode": mode,
        "case_ids": "|".join(case_ids),
        "batch_cases": len(case_ids),
        "support_points": int(len(batch["support_coords"])),
        "query_points": int(len(batch["query_coords"])),
        "physics_points": int(len(batch["physics_coords"])) if mode == "pinn" else 0,
        "wall_points": int(len(batch["wall_coords"])) if mode == "pinn" else 0,
        "learning_rate": float(learning_rate),
        "gradient_norm": float(gradient_norm),
    }
    record.update({name: float(value.detach().cpu()) for name, value in losses.items()})
    if mode == "pinn":
        record.update(
            {
                "continuity_rms_dimensionless": _rms(diagnostics["continuity_hat"]),
                "continuity_rms_s_inv": _rms(diagnostics["continuity_phys_s_inv"]),
                "momentum_x_rms_dimensionless": _rms(
                    diagnostics["momentum_hat"][:, 0]
                ),
                "momentum_y_rms_dimensionless": _rms(
                    diagnostics["momentum_hat"][:, 1]
                ),
                "momentum_z_rms_dimensionless": _rms(
                    diagnostics["momentum_hat"][:, 2]
                ),
                "momentum_rms_pa_per_m": _rms(diagnostics["momentum_phys_pa_per_m"]),
                "convective_rms_dimensionless": _rms(
                    diagnostics["momentum_convective_hat"]
                ),
                "pressure_gradient_rms_dimensionless": _rms(
                    diagnostics["momentum_pressure_hat"]
                ),
                "viscous_rms_dimensionless": _rms(
                    diagnostics["momentum_viscous_hat"]
                ),
                "shear_rate_p50_s_inv": _quantile(
                    diagnostics["shear_rate_s_inv"], 0.50
                ),
                "shear_rate_p95_s_inv": _quantile(
                    diagnostics["shear_rate_s_inv"], 0.95
                ),
                "viscosity_p50_pa_s": _quantile(
                    diagnostics["viscosity_pa_s"], 0.50
                ),
                "viscosity_p95_pa_s": _quantile(
                    diagnostics["viscosity_pa_s"], 0.95
                ),
                "wall_speed_rms_m_s": _rms(diagnostics["wall_speed_m_s"]),
                "wall_speed_p95_m_s": _quantile(
                    diagnostics["wall_speed_m_s"], 0.95
                ),
                "wall_speed_max_m_s": float(
                    diagnostics["wall_speed_m_s"].detach().max().cpu()
                ),
            }
        )
    finite_values = [
        value for value in record.values() if isinstance(value, (int, float))
    ]
    record["all_finite"] = bool(np.isfinite(finite_values).all())
    return record


def run(
    config: VolumeExperimentConfig,
    *,
    dry_run: bool = False,
    device_override: str | None = None,
) -> dict[str, Any]:
    train_cfg = config["train"]
    seed_everything(int(train_cfg["seed"]))
    device = resolve_device(str(train_cfg["device"]), device_override)
    dtype = torch.float64 if train_cfg["precision"] == "float64" else torch.float32
    field_stats_path = Path(config["paths"]["field_stats"])
    manifest_path = Path(config["paths"]["sidecar_manifest"])
    if not field_stats_path.exists() or not manifest_path.exists():
        raise FileNotFoundError(
            "volume-field sidecar manifest/stats are required before training"
        )
    field_stats = json.loads(field_stats_path.read_text(encoding="utf-8"))
    dataset = VolumeFieldDataset(
        manifest_path,
        field_stats_path,
        roles=config["data"]["train_roles"],
        input_variant=config.input_variant,
        sampling=config["sampling"],
        seed=int(train_cfg["seed"]),
    )
    model = build_volume_model(config).to(device=device, dtype=dtype)
    initialization_sha256 = tensor_state_sha256(model.state_dict())
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    total_epochs = int(train_cfg["epochs"])
    scheduler = _scheduler(
        optimizer,
        total_epochs,
        int(train_cfg["warmup_epochs"]),
        float(train_cfg["min_learning_rate"]),
        float(train_cfg["learning_rate"]),
    )
    start_epoch = 0
    global_step = 0
    best_data = float("inf")
    best_total = float("inf")
    resume = train_cfg.get("resume")
    resume_sha256 = None
    if resume:
        resume_path = Path(resume).resolve()
        expected_checkpoint_dir = (config.run_dir / "checkpoints").resolve()
        if resume_path.parent != expected_checkpoint_dir:
            raise ValueError(
                "resume must point to a checkpoint from the same run directory; "
                "cross-run initialization is forbidden"
            )
        payload = torch.load(resume_path, map_location=device, weights_only=False)
        if payload.get("route") != "volume_uvwp_peak_v1":
            raise ValueError("resume checkpoint route mismatch")
        checkpoint_config = payload.get("resolved_config", {})
        for section, key in (
            ("experiment", "id"),
            ("experiment", "mode"),
            ("experiment", "architecture"),
            ("experiment", "input_variant"),
        ):
            if checkpoint_config.get(section, {}).get(key) != config[section][key]:
                raise ValueError(f"resume checkpoint mismatch: {section}.{key}")
        if payload.get("initialization_state_sha256") != initialization_sha256:
            raise ValueError("resume initialization hash does not match this experiment")
        model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        start_epoch = int(payload["epoch"]) + 1
        global_step = int(payload["global_step"])
        best_data = float(payload["best_data"])
        best_total = float(payload["best_total"])
        resume_sha256 = sha256_file(resume_path)

    end_epoch = start_epoch + 1 if dry_run else total_epochs
    requested_epochs = end_epoch - start_epoch
    if requested_epochs <= 0:
        raise ValueError(
            f"run already reached configured train.epochs={total_epochs}; "
            "increase the explicit total only if a continuation is intended"
        )

    run_dir = config.run_dir
    if not dry_run:
        guard_write_path(run_dir).mkdir(parents=True, exist_ok=True)
        if (run_dir / "checkpoints/last.pt").exists() and not resume:
            raise FileExistsError(f"refusing to overwrite existing run: {run_dir}")
        initial_path = run_dir / "checkpoints/initialization.pt"
        if resume:
            if not initial_path.exists():
                raise FileNotFoundError(
                    f"same-run resume is missing initialization evidence: {initial_path}"
                )
            initial_payload = torch.load(
                initial_path, map_location="cpu", weights_only=False
            )
            if (
                initial_payload.get("initialization_state_sha256")
                != initialization_sha256
            ):
                raise ValueError("stored initialization evidence hash drift")
            resume_events = run_dir / "resume_events.jsonl"
            with resume_events.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "created_at": utc_now(),
                            "resume_checkpoint": str(Path(resume).resolve()),
                            "resume_checkpoint_sha256": resume_sha256,
                            "start_epoch": start_epoch,
                            "configured_total_epochs": total_epochs,
                            "resolved_config_sha256": config.resolved_sha256,
                            "warm_start": False,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        else:
            atomic_write_json(run_dir / "resolved_config.json", config.as_dict())
            atomic_write_json(
                run_dir / "environment.json",
                {
                    "created_at": utc_now(),
                    "python": sys.version,
                    "platform": platform.platform(),
                    "torch": torch.__version__,
                    "device": str(device),
                    "cuda_device": (
                        torch.cuda.get_device_name(device)
                        if device.type == "cuda"
                        else None
                    ),
                    "git": git_state(),
                    "sidecar_manifest": {
                        "path": str(manifest_path.resolve()),
                        "sha256": sha256_file(manifest_path),
                    },
                    "field_stats": {
                        "path": str(field_stats_path.resolve()),
                        "sha256": sha256_file(field_stats_path),
                    },
                    "initialization_state_sha256": initialization_sha256,
                    "warm_start": False,
                },
            )
            _atomic_checkpoint(
                initial_path,
                {
                    "schema_version": 1,
                    "route": "volume_uvwp_peak_v1",
                    "model": model.state_dict(),
                    "initialization_state_sha256": initialization_sha256,
                    "resolved_config_sha256": config.resolved_sha256,
                },
            )

    jsonl_path = run_dir / "training_progress.jsonl"
    epoch_jsonl_path = run_dir / "epoch_progress.jsonl"
    csv_path = run_dir / "history.csv"
    json_handle = None
    epoch_json_handle = None
    csv_handle = None
    csv_writer = None
    last_record: dict[str, Any] = {}
    try:
        if not dry_run:
            json_handle = jsonl_path.open("a", encoding="utf-8")
            csv_exists = csv_path.exists() and csv_path.stat().st_size > 0
            csv_handle = csv_path.open("a", encoding="utf-8", newline="")
            epoch_json_handle = epoch_jsonl_path.open("a", encoding="utf-8")
        for epoch in range(start_epoch, end_epoch):
            dataset.set_epoch(epoch)
            generator = torch.Generator()
            generator.manual_seed(int(train_cfg["seed"]) + epoch * 104729)
            loader = DataLoader(
                dataset,
                batch_size=int(train_cfg["batch_cases"]),
                shuffle=True,
                generator=generator,
                num_workers=int(train_cfg["num_workers"]),
                collate_fn=collate_volume_samples,
                pin_memory=device.type == "cuda",
            )
            model.train()
            epoch_data = 0.0
            epoch_total = 0.0
            epoch_steps = 0
            epoch_loss_sums: dict[str, float] = {}
            for step_in_epoch, cpu_batch in enumerate(loader):
                batch = _move_batch(cpu_batch, device, dtype)
                optimizer.zero_grad(set_to_none=True)
                losses, diagnostics = compute_losses(
                    model,
                    batch,
                    field_stats=field_stats,
                    config=config,
                    epoch=epoch,
                )
                if not torch.isfinite(losses["total"]):
                    raise FloatingPointError(
                        f"non-finite total loss at epoch={epoch} step={step_in_epoch}"
                    )
                losses["total"].backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(train_cfg["grad_clip_norm"])
                )
                if not torch.isfinite(gradient_norm):
                    raise FloatingPointError("non-finite gradient norm")
                optimizer.step()
                global_step += 1
                epoch_steps += 1
                epoch_data += float(losses["data_total"].detach().cpu())
                epoch_total += float(losses["total"].detach().cpu())
                for name, value in losses.items():
                    epoch_loss_sums[name] = epoch_loss_sums.get(name, 0.0) + float(
                        value.detach().cpu()
                    )
                last_record = _step_record(
                    losses=losses,
                    diagnostics=diagnostics,
                    epoch=epoch,
                    step_in_epoch=step_in_epoch,
                    global_step=global_step,
                    case_ids=batch["case_ids"],
                    batch=batch,
                    learning_rate=float(optimizer.param_groups[0]["lr"]),
                    gradient_norm=float(gradient_norm.detach().cpu()),
                    mode=config.mode,
                )
                if not last_record["all_finite"]:
                    raise FloatingPointError("training log contains NaN/Inf")
                if not dry_run and (
                    global_step == 1
                    or global_step % int(train_cfg["log_every_steps"]) == 0
                ):
                    json_handle.write(json.dumps(last_record, ensure_ascii=False) + "\n")
                    json_handle.flush()
                    if csv_writer is None:
                        csv_writer = csv.DictWriter(csv_handle, fieldnames=list(last_record))
                        if not csv_exists:
                            csv_writer.writeheader()
                    csv_writer.writerow(last_record)
                    csv_handle.flush()
            scheduler.step()
            mean_data = epoch_data / max(epoch_steps, 1)
            mean_total = epoch_total / max(epoch_steps, 1)
            data_improved = mean_data < best_data
            total_improved = mean_total < best_total
            best_data = min(best_data, mean_data)
            best_total = min(best_total, mean_total)
            epoch_record = {
                "created_at": utc_now(),
                "epoch": int(epoch),
                "global_step": int(global_step),
                "steps": int(epoch_steps),
                "learning_rate_next_epoch": float(optimizer.param_groups[0]["lr"]),
                "best_data": float(best_data),
                "best_total": float(best_total),
                **{
                    f"mean_{name}": total / max(epoch_steps, 1)
                    for name, total in epoch_loss_sums.items()
                },
            }
            if not dry_run:
                epoch_json_handle.write(
                    json.dumps(epoch_record, ensure_ascii=False) + "\n"
                )
                epoch_json_handle.flush()
            if not dry_run:
                payload = _checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    global_step=global_step,
                    best_data=best_data,
                    best_total=best_total,
                    config=config,
                    initialization_sha256=initialization_sha256,
                )
                if data_improved:
                    _atomic_checkpoint(run_dir / "checkpoints/best_data.pt", payload)
                if total_improved:
                    _atomic_checkpoint(run_dir / "checkpoints/best_total.pt", payload)
                completed_epoch = epoch + 1
                if completed_epoch in {
                    int(value)
                    for value in train_cfg.get("milestone_epochs", [])
                }:
                    _atomic_checkpoint(
                        run_dir / f"checkpoints/epoch_{completed_epoch:05d}.pt",
                        payload,
                    )
                if (
                    (epoch + 1) % int(train_cfg["checkpoint_every_epochs"]) == 0
                    or epoch == start_epoch + requested_epochs - 1
                ):
                    _atomic_checkpoint(run_dir / "checkpoints/last.pt", payload)
        summary = {
            "schema_version": 1,
            "created_at": utc_now(),
            "status": "dry_run_completed" if dry_run else "completed",
            "route": "volume_uvwp_peak_v1",
            "experiment_id": config["experiment"]["id"],
            "mode": config.mode,
            "architecture": config.architecture,
            "input_variant": config.input_variant,
            "warm_start": False,
            "resume_checkpoint_sha256": resume_sha256,
            "initialization_state_sha256": initialization_sha256,
            "epochs_executed": requested_epochs,
            "start_epoch": start_epoch,
            "end_epoch_exclusive": end_epoch,
            "configured_total_epochs": total_epochs,
            "global_steps": global_step,
            "best_data": best_data,
            "best_total": best_total,
            "last_record": last_record,
        }
        if not dry_run:
            atomic_write_json(run_dir / "training_summary.json", summary)
        return summary
    finally:
        if json_handle is not None:
            json_handle.close()
        if epoch_json_handle is not None:
            epoch_json_handle.close()
        if csv_handle is not None:
            csv_handle.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"))
    args = parser.parse_args()
    config = VolumeExperimentConfig.from_json(args.config)
    report = run(config, dry_run=args.dry_run, device_override=args.device)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
