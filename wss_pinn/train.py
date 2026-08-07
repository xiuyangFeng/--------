"""训练 PointNet / PointNet++ 峰值体域 ``u,v,w,p`` 的 data-only 与 PINN 实验。

学习要点
--------
1. **配对公平**：同一 seed 下 data-only / PINN 共享初始化哈希、采样协议与
   数据监督；PINN 禁止从 data-only checkpoint 热启动。
2. **resume ≠ 热启动**：``train.resume`` 只允许指向**本 run** 的
   ``checkpoints/``，并校验 route、实验字段与 ``initialization_state_sha256``。
3. **精简日志**：step 级日志只保存采样规模、优化器状态和 loss；物理场诊断
   统一放在评估阶段，避免训练入口承担重复分析职责。
4. **checkpoint**：``initialization.pt``（初始化证据）、``best_data`` /
   ``best_total``、``last``，以及可选 ``milestone_epochs`` 定点快照。
"""

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

from .config import ExperimentConfig
from .data import VolumeFieldDataset, collate_volume_samples
from .losses import compute_losses
from .models import build_model
from .validation import aggregate_case_validation
from .volume_utils import tensor_state_sha256


def seed_everything(seed: int) -> None:
    """固定 Python / NumPy / Torch（含 CUDA）随机源，尽量可复现。

    ``CUBLAS_WORKSPACE_CONFIG`` 与 ``use_deterministic_algorithms(warn_only=True)``
    尽量开启确定性；部分 CUDA kernel 仍可能 warn，但不直接失败。
    """
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(name: str, override: str | None = None) -> torch.device:
    """解析训练设备；请求 CUDA 但不可用时立即报错，避免静默掉到 CPU。"""
    selected = override or name
    if selected == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(selected)


def _move_batch(batch: dict[str, Any], device: torch.device, dtype: torch.dtype) -> dict:
    """把 batch 迁到目标设备；浮点坐标/场量转成训练精度，整数索引等只搬设备。"""
    output: dict[str, Any] = {}
    floating = {
        "support_coords",
        "support_features",
        "query_coords",
        "query_target",
        "physics_coords",
        "physics_length_m",
        "wall_coords",
        "inlet_coords",
        "inlet_velocity_m_s",
        "outlet_coords",
        "outlet_pressure_relative_pa",
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
    """线性 warmup + cosine 衰减到 ``min_lr`` 的 LambdaLR。

    ``scale(epoch)`` 返回相对 ``base_lr`` 的倍率：warmup 阶段线性爬升，之后
    按余弦从 1 降到 ``min_lr/base_lr``。
    """
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
    """原子写 checkpoint：临时文件 → ``os.replace``，防止半截 ``.pt``。"""
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
    config: ExperimentConfig,
    initialization_sha256: str,
) -> dict[str, Any]:
    """组装可 resume 的完整 checkpoint 字典（含配置哈希与初始化证据）。"""
    return {
        "schema_version": 1,
        "route": config.route,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_data": float(best_data),
        "best_total": float(best_total),
        "best_validation_field_score_cb": (
            float(best_data)
            if config.route == "volume_uvwp_peak_field_v4"
            else None
        ),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "initialization_state_sha256": initialization_sha256,
        "resolved_config": config.as_dict(),
        "resolved_config_sha256": config.resolved_sha256,
    }


def _step_record(
    *,
    losses: dict[str, torch.Tensor],
    epoch: int,
    step_in_epoch: int,
    global_step: int,
    case_ids: list[str],
    batch: dict[str, Any],
    learning_rate: float,
    gradient_norm: float,
    mode: str,
) -> dict[str, Any]:
    """把一步训练压成稳定、可写入 JSONL/CSV 的扁平记录。"""
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
        "physics_points": int(len(batch["physics_coords"]))
        if mode in {"pinn", "data_bc_pde"}
        else 0,
        "wall_points": int(len(batch["wall_coords"]))
        if mode in {"pinn", "data_bc", "data_bc_pde"}
        else 0,
        "inlet_points": int(len(batch.get("inlet_coords", []))),
        "outlet_points": int(len(batch.get("outlet_coords", []))),
        "learning_rate": float(learning_rate),
        "gradient_norm": float(gradient_norm),
    }
    record.update({name: float(value.detach().cpu()) for name, value in losses.items()})
    finite_values = [
        value for value in record.values() if isinstance(value, (int, float))
    ]
    record["all_finite"] = bool(np.isfinite(finite_values).all())
    return record


def _validate_field_v4(
    *,
    model,
    dataset: VolumeFieldDataset,
    field_stats: dict[str, Any],
    config: ExperimentConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Evaluate the frozen val15 one case at a time for exact case balance."""
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(config["train"]["num_workers"]),
        collate_fn=collate_volume_samples,
        pin_memory=device.type == "cuda",
    )
    model.eval()
    cases: list[dict[str, Any]] = []
    with torch.no_grad():
        for cpu_batch in loader:
            batch = _move_batch(cpu_batch, device, dtype)
            losses = compute_losses(
                model,
                batch,
                field_stats=field_stats,
                config=config,
                epoch=0,
            )
            components = {
                name: float(losses[f"data_{name}_raw"].detach().cpu())
                for name in ("u", "v", "w", "pressure")
            }
            score = float(np.mean(list(components.values())))
            cases.append(
                {
                    "case_id": batch["case_ids"][0],
                    "points": int(len(batch["query_coords"])),
                    "field_score": score,
                    **{f"data_{name}": value for name, value in components.items()},
                    "query_sha256": batch["fixed_validation_query_sha256"][0],
                }
            )
    audit = aggregate_case_validation(
        cases, legacy_batch_cases=int(config["train"]["batch_cases"])
    )
    audit["cases"] = cases
    means = {
        "data_total": audit["validation_field_score_cb"],
        "total": audit["validation_field_score_cb"],
        **{
            f"data_{name}_raw": float(
                np.mean([row[f"data_{name}"] for row in cases])
            )
            for name in ("u", "v", "w", "pressure")
        },
    }
    model.train()
    return means, audit


def run(
    config: ExperimentConfig,
    *,
    dry_run: bool = False,
    device_override: str | None = None,
) -> dict[str, Any]:
    """执行一次训练（或 dry-run 冒烟）。

    Parameters
    ----------
    config :
        已通过科学合同校验的实验配置。
    dry_run :
        True 时只跑 1 个 epoch、不写 run 目录产物，供 GPU preflight。
    device_override :
        CLI ``--device`` 覆盖配置中的 device。

    Returns
    -------
    summary :
        ``training_summary.json`` 同结构的字典（dry_run 也返回，但不落盘）。
    """
    train_cfg = config["train"]
    seed_everything(int(train_cfg["seed"]))
    device = resolve_device(str(train_cfg["device"]), device_override)
    dtype = torch.float64 if train_cfg["precision"] == "float64" else torch.float32

    # ---- 数据与模型：必须先有 schema-v2 sidecar / field_stats ----
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
    validation_dataset = None
    if config["data"].get("validation_roles"):
        validation_dataset = VolumeFieldDataset(
            manifest_path,
            field_stats_path,
            roles=config["data"]["validation_roles"],
            input_variant=config.input_variant,
            sampling=config["sampling"],
            seed=int(train_cfg["seed"]) + 90_001,
        )
        validation_dataset.set_epoch(0)
    model = build_model(config).to(device=device, dtype=dtype)
    # 初始化哈希是配对实验与 resume 护栏的核心证据
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
    epochs_without_improvement = 0
    selection_history: list[dict[str, Any]] = []

    # ---- 同 run 断点续训（显式拒绝跨 run / 热启动）----
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
        if payload.get("route") != config.route:
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
        # 重新随机初始化后的哈希必须与保存时一致，否则说明 seed/架构已变
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

    # dry_run：只执行 1 个 epoch；正式跑：从 start_epoch 到 total_epochs
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
        # 已有 last.pt 且未声明 resume → 拒绝覆盖，防止误跑冲掉正式结果
        if (run_dir / "checkpoints/last.pt").exists() and not resume:
            raise FileExistsError(f"refusing to overwrite existing run: {run_dir}")
        initial_path = run_dir / "checkpoints/initialization.pt"
        if resume:
            # 续训必须能找到当初的初始化证据，并与当前哈希一致
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
            # 新 run：落盘配置、环境元数据与初始化 checkpoint
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
                    "route": config.route,
                    "model": model.state_dict(),
                    "initialization_state_sha256": initialization_sha256,
                    "resolved_config_sha256": config.resolved_sha256,
                },
            )

    # ---- 训练主循环 ----
    jsonl_path = run_dir / "training_progress.jsonl"
    epoch_jsonl_path = run_dir / "epoch_progress.jsonl"
    csv_path = run_dir / "history.csv"
    json_handle = None
    epoch_json_handle = None
    csv_handle = None
    csv_writer = None
    last_record: dict[str, Any] = {}
    executed_epochs = 0
    stopped_early = False
    try:
        if not dry_run:
            json_handle = jsonl_path.open("a", encoding="utf-8")
            csv_exists = csv_path.exists() and csv_path.stat().st_size > 0
            csv_handle = csv_path.open("a", encoding="utf-8", newline="")
            epoch_json_handle = epoch_jsonl_path.open("a", encoding="utf-8")
        for epoch in range(start_epoch, end_epoch):
            # 每 epoch 重采样 support/query；DataLoader shuffle 也按 epoch 种子
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
            epoch_clipped_steps = 0
            epoch_loss_sums: dict[str, float] = {}
            for step_in_epoch, cpu_batch in enumerate(loader):
                batch = _move_batch(cpu_batch, device, dtype)
                optimizer.zero_grad(set_to_none=True)
                losses = compute_losses(
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
                if float(gradient_norm.detach().cpu()) > float(
                    train_cfg["grad_clip_norm"]
                ):
                    epoch_clipped_steps += 1
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
                # 按 log_every_steps 写 step 日志（第 1 步必写）
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
                if dry_run:
                    break

            # epoch 结束：更新 LR；V3 用固定 val15 统一选 checkpoint。
            scheduler.step()
            mean_data = epoch_data / max(epoch_steps, 1)
            mean_total = epoch_total / max(epoch_steps, 1)
            validation_means: dict[str, float] = {}
            validation_audit: dict[str, Any] | None = None
            if validation_dataset is not None and (
                (epoch + 1) % int(train_cfg["validation_every_epochs"]) == 0
                or epoch == start_epoch + requested_epochs - 1
            ):
                if config.route == "volume_uvwp_peak_field_v4":
                    validation_means, validation_audit = _validate_field_v4(
                        model=model,
                        dataset=validation_dataset,
                        field_stats=field_stats,
                        config=config,
                        device=device,
                        dtype=dtype,
                    )
                else:
                    validation_generator = torch.Generator()
                    validation_generator.manual_seed(int(train_cfg["seed"]) + 700_001)
                    validation_loader = DataLoader(
                        validation_dataset,
                        batch_size=int(train_cfg["batch_cases"]),
                        shuffle=False,
                        generator=validation_generator,
                        num_workers=int(train_cfg["num_workers"]),
                        collate_fn=collate_volume_samples,
                        pin_memory=device.type == "cuda",
                    )
                    model.eval()
                    validation_sums: dict[str, float] = {}
                    validation_steps = 0
                    with torch.enable_grad():
                        for validation_cpu_batch in validation_loader:
                            validation_batch = _move_batch(
                                validation_cpu_batch, device, dtype
                            )
                            validation_losses = compute_losses(
                                model,
                                validation_batch,
                                field_stats=field_stats,
                                config=config,
                                epoch=0,
                            )
                            if not torch.isfinite(validation_losses["total"]):
                                raise FloatingPointError("non-finite validation loss")
                            validation_steps += 1
                            for name, value in validation_losses.items():
                                validation_sums[name] = validation_sums.get(name, 0.0) + float(
                                    value.detach().cpu()
                                )
                            if dry_run:
                                break
                    validation_means = {
                        name: total / max(validation_steps, 1)
                        for name, total in validation_sums.items()
                    }
                    model.train()
            selection_data = validation_means.get("data_total", mean_data)
            selection_total = validation_means.get("total", mean_total)
            min_delta = (
                float(train_cfg.get("early_stopping_min_delta", 0.0))
                if config.route == "volume_uvwp_peak_field_v4"
                else 0.0
            )
            data_improved = selection_data < best_data - min_delta
            total_improved = selection_total < best_total - min_delta
            if data_improved:
                best_data = selection_data
                epochs_without_improvement = 0
            elif config.route == "volume_uvwp_peak_field_v4":
                epochs_without_improvement += 1
            if total_improved:
                best_total = selection_total
            epoch_record = {
                "created_at": utc_now(),
                "epoch": int(epoch),
                "global_step": int(global_step),
                "steps": int(epoch_steps),
                "gradient_clipped_steps": int(epoch_clipped_steps),
                "gradient_clipping_fraction": float(
                    epoch_clipped_steps / max(epoch_steps, 1)
                ),
                "learning_rate_next_epoch": float(optimizer.param_groups[0]["lr"]),
                "best_data": float(best_data),
                "best_total": float(best_total),
                **{
                    f"mean_{name}": total / max(epoch_steps, 1)
                    for name, total in epoch_loss_sums.items()
                },
                **{
                    f"validation_{name}": value
                    for name, value in validation_means.items()
                },
            }
            if validation_audit is not None:
                epoch_record.update(
                    {
                        "validation_field_score_cb": validation_audit[
                            "validation_field_score_cb"
                        ],
                        "validation_field_score_point_weighted": validation_audit[
                            "validation_field_score_point_weighted"
                        ],
                        "validation_field_score_legacy_batch_weighted": validation_audit[
                            "validation_field_score_legacy_batch_weighted"
                        ],
                        "validation_legacy_overweighted_case_ids": validation_audit[
                            "legacy_overweighted_case_ids"
                        ],
                        "validation_legacy_weight_ratio_max_to_min": validation_audit[
                            "legacy_weight_ratio_max_to_min"
                        ],
                        "validation_cases": validation_audit["cases"],
                        "epochs_without_field_improvement": epochs_without_improvement,
                    }
                )
                selection_history.append(
                    {
                        "epoch": int(epoch),
                        "case_balanced": validation_audit[
                            "validation_field_score_cb"
                        ],
                        "point_weighted": validation_audit[
                            "validation_field_score_point_weighted"
                        ],
                        "legacy_batch_weighted": validation_audit[
                            "validation_field_score_legacy_batch_weighted"
                        ],
                    }
                )
            if not dry_run:
                epoch_json_handle.write(
                    json.dumps(epoch_record, ensure_ascii=False) + "\n"
                )
                epoch_json_handle.flush()
                print(
                    json.dumps(
                        {
                            "event": "training_epoch_completed",
                            "experiment_id": config["experiment"]["id"],
                            **epoch_record,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
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
                data_checkpoint = (
                    "best_validation_field_cb.pt"
                    if config.route == "volume_uvwp_peak_field_v4"
                    else (
                        "best_validation_data.pt"
                        if config.route == "volume_uvwp_peak_qs_smooth_v3"
                        else "best_data.pt"
                    )
                )
                total_checkpoint = (
                    "best_validation_total.pt"
                    if config.route
                    in {
                        "volume_uvwp_peak_qs_smooth_v3",
                        "volume_uvwp_peak_field_v4",
                    }
                    else "best_total.pt"
                )
                if data_improved:
                    _atomic_checkpoint(run_dir / "checkpoints" / data_checkpoint, payload)
                if total_improved:
                    _atomic_checkpoint(run_dir / "checkpoints" / total_checkpoint, payload)
                completed_epoch = epoch + 1  # 人类可读的「已完成 epoch 数」
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

            executed_epochs += 1
            if (
                not dry_run
                and config.route == "volume_uvwp_peak_field_v4"
                and epochs_without_improvement
                >= int(train_cfg["early_stopping_patience"])
            ):
                _atomic_checkpoint(run_dir / "checkpoints/last.pt", payload)
                stopped_early = True
                break

        selection_audit_path = None
        if not dry_run and config.route == "volume_uvwp_peak_field_v4":
            if not selection_history:
                raise RuntimeError("V4 training produced no validation selection history")
            best_epochs = {
                key: min(selection_history, key=lambda row: row[key])["epoch"]
                for key in ("case_balanced", "point_weighted", "legacy_batch_weighted")
            }
            last_validation = validation_audit or {}
            selection_audit = {
                "schema_version": 1,
                "created_at": utc_now(),
                "route": config.route,
                "selection_metric": "validation_field_score_cb",
                "selection_formula": (
                    "mean_case((MSE_u + MSE_v + MSE_w + MSE_p)/4), "
                    "all components standardized by frozen train123 statistics"
                ),
                "best_epoch_zero_based": best_epochs,
                "checkpoint_epoch_difference": {
                    "legacy_minus_case_balanced": best_epochs[
                        "legacy_batch_weighted"
                    ]
                    - best_epochs["case_balanced"],
                    "point_minus_case_balanced": best_epochs["point_weighted"]
                    - best_epochs["case_balanced"],
                },
                "legacy_overweighted_case_ids": last_validation.get(
                    "legacy_overweighted_case_ids", []
                ),
                "legacy_weight_ratio_max_to_min": last_validation.get(
                    "legacy_weight_ratio_max_to_min"
                ),
                "fixed_validation_query_hashes": {
                    row["case_id"]: row["query_sha256"]
                    for row in last_validation.get("cases", [])
                },
                "history": selection_history,
            }
            selection_audit_path = atomic_write_json(
                run_dir / "selection_audit.json", selection_audit
            )
        summary = {
            "schema_version": 1,
            "created_at": utc_now(),
            "status": "dry_run_completed" if dry_run else "completed",
            "route": config.route,
            "experiment_id": config["experiment"]["id"],
            "mode": config.mode,
            "architecture": config.architecture,
            "input_variant": config.input_variant,
            "warm_start": False,
            "resume_checkpoint_sha256": resume_sha256,
            "initialization_state_sha256": initialization_sha256,
            "epochs_executed": executed_epochs,
            "start_epoch": start_epoch,
            "end_epoch_exclusive": start_epoch + executed_epochs,
            "configured_total_epochs": total_epochs,
            "stopped_early": stopped_early,
            "global_steps": global_step,
            "best_data": best_data,
            "best_total": best_total,
            "best_validation_field_score_cb": (
                best_data if config.route == "volume_uvwp_peak_field_v4" else None
            ),
            "selection_audit": (
                {
                    "path": str(selection_audit_path),
                    "sha256": sha256_file(selection_audit_path),
                }
                if selection_audit_path is not None
                else None
            ),
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
    """CLI：``python -m wss_pinn.train --config <json> [--dry-run] [--device]``。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"))
    args = parser.parse_args()
    config = ExperimentConfig.from_json(args.config)
    report = run(config, dry_run=args.dry_run, device_override=args.device)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
