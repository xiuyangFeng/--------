from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

import torch

from ..core.config import ExperimentConfig, resolve_wss_effective_dim, resolve_wss_runtime_names
from ..core.data import (
    FieldGraphDataset,
    build_dataloader,
    build_feature_mask,
    build_required_data_keys,
)
from ..core.io import append_experiment_index, load_checkpoint, load_pretrained_encoder
from ..core.models import build_field_model_from_config
from ..core.splits import SplitSpec
from ..core.denylist import filter_case_names, skipped_case_names
from ..core.trainer import FieldTrainer
from ..core.utils import dump_json, ensure_dir, resolve_device, set_seed, timestamp


_BACKBONE_PREFIXES = ("in_proj.", "blocks.", "shared_decoder.")


def apply_training_freeze(model: torch.nn.Module, *, freeze_backbone: bool, freeze_field_head: bool) -> None:
    """I6-a 两阶段：按前缀冻结参数；默认全 False 时无操作（向后兼容）。"""
    frozen = 0
    for name, param in model.named_parameters():
        if freeze_backbone and name.startswith(_BACKBONE_PREFIXES):
            param.requires_grad = False
            frozen += param.numel()
        elif freeze_field_head and name.startswith("field_head."):
            param.requires_grad = False
            frozen += param.numel()
    if frozen:
        print(f"已冻结参数: {frozen:,}（backbone={freeze_backbone}, field_head={freeze_field_head}）")


def build_run_dir(config: ExperimentConfig, split: SplitSpec) -> Path:
    # run_dir 把 experiment_name / split / seed / timestamp 全部带上，方便后续追溯单次实验。
    # 按“实验名_划分版本_seedX_时间戳”生成本次运行目录名。
    run_name = f"{config.run.experiment_name}_{split.split_version}_seed{config.system.seed}_{timestamp()}"
    # 真正创建目录并返回。
    return ensure_dir(Path(config.run.output_root) / run_name)


def build_run_manifest(
    *,
    config: ExperimentConfig,
    split: SplitSpec,
    device: torch.device,
    run_dir: Path,
    fit_result: Dict[str, object],
    test_metrics: Dict[str, float],
    dataset_sizes: Dict[str, int],
    test_metrics_best_wss: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    # run_manifest 是给“训练结束后的世界”看的，而不是给训练过程看的。
    # 后面做任务 B/C、实验记录汇总、论文表格回溯时，都尽量读这个文件。
    # 返回结构化运行清单，尽可能把复现实验所需的信息放全。
    manifest: Dict[str, object] = {
        "task": config.meta.task,
        "exp_id": config.meta.exp_id,
        "experiment_name": config.run.experiment_name,
        "stage": config.meta.stage,
        "study_group": config.meta.study_group,
        "question": config.meta.question,
        "feature_set": config.meta.feature_set,
        "ablation_axis": config.meta.ablation_axis,
        "tags": config.meta.tags,
        "notes": config.meta.notes,
        "generated_from": config.meta.generated_from,
        "device": str(device),
        "run_dir": str(run_dir),
        "output_root": config.run.output_root,
        "init_checkpoint": config.run.init_checkpoint,
        "pretrained_encoder": config.run.pretrained_encoder,
        "encoder_lr_ratio": config.optim.encoder_lr_ratio,
        "split_version": split.split_version,
        "split_source": split.source,
        "split_notes": split.notes,
        "seed": config.system.seed,
        "deterministic": config.system.deterministic,
        "model": {
            "name": config.model.name,
            "hidden_dim": config.model.hidden_dim,
            "num_layers": config.model.num_layers,
            "dropout": config.model.dropout,
            "heads": config.model.heads,
            "head_layout": config.model.head_layout,
            "wss_head_dropout": config.model.wss_head_dropout,
        },
        "data": {
            "data_root": config.data.data_root,
            "graphs_subdir": config.data.graphs_subdir,
            "batch_size": config.data.batch_size,
            "augment": config.data.augment,
            "augment_config": config.data.augment_config,
            "enabled_node_features": config.data.enabled_node_features,
            "enabled_global_features": config.data.enabled_global_features,
        },
        "optim": {
            "epochs": config.optim.epochs,
            "lr": config.optim.lr,
            "weight_decay": config.optim.weight_decay,
            "warmup_epochs": config.optim.warmup_epochs,
            "scheduler_factor": config.optim.scheduler_factor,
            "scheduler_patience": config.optim.scheduler_patience,
            "early_stopping_patience": config.optim.early_stopping_patience,
            "early_stop_min_delta": config.optim.early_stop_min_delta,
            "val_score_ema_alpha": config.optim.val_score_ema_alpha,
            "target_weights": config.optim.target_weights,
            "interior_loss_boost": config.optim.interior_loss_boost,
            "grad_clip_norm": config.optim.grad_clip_norm,
            "wss_loss_weight": config.optim.wss_loss_weight,
            "wss_weights": config.optim.wss_weights,
            "val_score_wss_weights": config.optim.val_score_wss_weights,
            "wss_loss_type": config.optim.wss_loss_type,
            "wss_huber_beta": config.optim.wss_huber_beta,
            "early_stop_wss_weight": config.optim.early_stop_wss_weight,
            "domain_loss_enabled": config.optim.domain_loss.enabled,
        },
        "physics": {
            "enabled": config.physics.enabled,
            "warmup_epochs": config.physics.warmup_epochs,
            "density": config.physics.density,
            "viscosity": config.physics.viscosity,
            "coord_scales": config.physics.coord_scales,
            "time_scale": config.physics.time_scale,
            "continuity_weight": config.physics.continuity_weight,
            "momentum_weight": config.physics.momentum_weight,
            "no_slip_weight": config.physics.no_slip_weight,
            "equation": config.physics.equation,
            "denormalize_fields": config.physics.denormalize_fields,
            "length_unit_to_meter": config.physics.length_unit_to_meter,
            "no_slip_mode": config.physics.no_slip_mode,
            "dynamic_weight": config.physics.dynamic_weight,
        },
        "dataset_sizes": dataset_sizes,
        "best_epoch": fit_result["best_epoch"],
        "best_val_score": fit_result["best_val_score"],
        "best_val_loss": fit_result["best_val_loss"],
        "best_wss_epoch": fit_result.get("best_wss_epoch", 0),
        "best_val_wss_r2": fit_result.get("best_val_wss_r2"),
        "test_metrics": test_metrics,
    }
    if test_metrics_best_wss is not None:
        manifest["test_metrics_best_wss"] = test_metrics_best_wss
    return manifest


def main() -> None:
    # 构造命令行解析器。
    parser = argparse.ArgumentParser(description="任务A场重建训练入口")
    # 本脚本只要求一个 JSON 配置路径。
    parser.add_argument("--config", required=True, help="JSON 配置文件路径")
    # 解析命令行参数。
    args = parser.parse_args()

    # 从 JSON 读取实验配置。
    config = ExperimentConfig.from_json(args.config)
    # 对配置做显式校验。
    config.validate()
    # 读取训练/验证/测试划分。
    split = SplitSpec.from_json(config.data.split_file)

    train_cases = filter_case_names(split.train_cases, config.data.data_root)
    val_cases = filter_case_names(split.val_cases, config.data.data_root)
    test_cases = filter_case_names(split.test_cases, config.data.data_root)
    denied = skipped_case_names(
        split.train_cases + split.val_cases + split.test_cases,
        config.data.data_root,
    )
    if denied:
        print(f"⏭ PREPROCESS_DENYLIST 跳过 {len(denied)} 例: {sorted(denied)}")

    # physics 尺度必须在构建 trainer 前解析完成，否则 loss 会退回默认 1.0。
    # 先把训练病例目录路径组织出来，供 physics 自动读取尺度。
    train_case_dirs = [Path(config.data.data_root) / case_name for case_name in train_cases]
    config.physics.resolve_scales_from_data(
        data_root=config.data.data_root,
        graphs_subdir=config.data.graphs_subdir,
        case_dirs=train_case_dirs,
    )

    # 固定随机种子。
    set_seed(config.system.seed, deterministic=config.system.deterministic)
    # 解析运行设备。
    device = resolve_device(config.system.device)

    # 根据配置生成节点特征与全局特征的屏蔽 mask。
    feature_mask = build_feature_mask(
        enabled_node_features=config.data.enabled_node_features,
        enabled_global_features=config.data.enabled_global_features,
    )
    eff_wss_dim = resolve_wss_effective_dim(
        config.model.wss_dim,
        config.model.wss_output_mode,
        config.model.wss_metric_dim,
    )
    required_data_keys = build_required_data_keys(
        config.model.name,
        wss_dim=eff_wss_dim,
        wss_target_frame=config.data.wss_target_frame,
    )
    wss_target_names = resolve_wss_runtime_names(
        config.data.wss_target_frame,
        config.model.wss_dim,
        config.model.wss_output_mode,
        config.model.wss_metric_dim,
    )

    # 训练/验证/测试共用同一套 split 文件，保证后续任务 B、C 可以回溯到统一划分。
    # 构建训练集数据对象。
    train_dataset = FieldGraphDataset(
        root=config.data.data_root,
        case_names=train_cases,
        graphs_subdir=config.data.graphs_subdir,
        augment=config.data.augment,
        augment_config=config.data.augment_config,
        preload=config.data.preload,
        feature_mask=feature_mask,
        required_keys=required_data_keys,
        wss_target_frame=config.data.wss_target_frame,
        wss_domain_norm=config.data.wss_domain_norm,
        wss_domain_norm_stats=config.data.wss_domain_norm_stats,
    )
    # 构建验证集数据对象；验证集不做增强。
    val_dataset = FieldGraphDataset(
        root=config.data.data_root,
        case_names=val_cases,
        graphs_subdir=config.data.graphs_subdir,
        augment=False,
        preload=config.data.preload,
        feature_mask=feature_mask,
        required_keys=required_data_keys,
        wss_target_frame=config.data.wss_target_frame,
        wss_domain_norm=config.data.wss_domain_norm,
        wss_domain_norm_stats=config.data.wss_domain_norm_stats,
    )
    # 构建测试集数据对象；测试集同样不做增强。
    test_dataset = FieldGraphDataset(
        root=config.data.data_root,
        case_names=test_cases,
        graphs_subdir=config.data.graphs_subdir,
        augment=False,
        preload=config.data.preload,
        feature_mask=feature_mask,
        required_keys=required_data_keys,
        wss_target_frame=config.data.wss_target_frame,
        wss_domain_norm=config.data.wss_domain_norm,
        wss_domain_norm_stats=config.data.wss_domain_norm_stats,
    )

    # 训练 DataLoader 需要打乱顺序。
    train_loader = build_dataloader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        seed=config.system.seed,
    )
    # 验证 DataLoader 不打乱顺序。
    val_loader = build_dataloader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
    )
    # 测试 DataLoader 也不打乱顺序。
    test_loader = build_dataloader(
        test_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
    )

    model = build_field_model_from_config(config).to(device)

    if config.run.init_checkpoint:
        if config.data.wss_target_frame == "local":
            raise ValueError(
                "local frame 训练禁止 warm-start（wss_head 形状与 global ckpt 不兼容）"
            )
        load_checkpoint(model, config.run.init_checkpoint, device)
        print(f"已加载初始 checkpoint: {config.run.init_checkpoint}")
    elif config.run.pretrained_encoder:
        stats = load_pretrained_encoder(model, config.run.pretrained_encoder, device)
        print(
            f"已加载 SSL encoder backbone: {config.run.pretrained_encoder} "
            f"(loaded={stats['loaded']}, skipped={stats['skipped']})"
        )

    if config.optim.freeze_backbone or config.optim.freeze_field_head:
        apply_training_freeze(
            model,
            freeze_backbone=config.optim.freeze_backbone,
            freeze_field_head=config.optim.freeze_field_head,
        )

    # 统计模型总参数量。
    total_params = sum(p.numel() for p in model.parameters())
    # 统计可训练参数量。
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型: {config.model.name} | 总参数: {total_params:,} | 可训练: {trainable_params:,}")

    trainable_params_list = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params_list:
        raise ValueError("无可训练参数：请检查 freeze_backbone/freeze_field_head 配置")

    # 优化器使用 Adam；G3 SSL 微调时对 backbone 使用较低学习率。
    if config.run.pretrained_encoder and config.optim.encoder_lr_ratio != 1.0:
        encoder_params = [
            p for n, p in model.named_parameters()
            if n.startswith(_BACKBONE_PREFIXES) and p.requires_grad
        ]
        head_params = [
            p for n, p in model.named_parameters()
            if not n.startswith(_BACKBONE_PREFIXES) and p.requires_grad
        ]
        encoder_lr = config.optim.lr * config.optim.encoder_lr_ratio
        param_groups = []
        if encoder_params:
            param_groups.append({"params": encoder_params, "lr": encoder_lr})
        if head_params:
            param_groups.append({"params": head_params, "lr": config.optim.lr})
        optimizer = torch.optim.Adam(param_groups, weight_decay=config.optim.weight_decay)
        print(f"discriminative LR: encoder={encoder_lr:g}, head={config.optim.lr:g}")
    else:
        optimizer = torch.optim.Adam(
            trainable_params_list,
            lr=config.optim.lr,
            weight_decay=config.optim.weight_decay,
        )

    # 先读出 warmup epoch 数。
    warmup_epochs = config.optim.warmup_epochs
    # 默认不启用 warmup 调度器。
    warmup_scheduler = None
    # 只有 warmup_epochs > 0 时才创建线性 warmup。
    if warmup_epochs > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=warmup_epochs,
        )
    # 主调度器使用按验证损失自适应降学习率的 Plateau。
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min",
        factor=config.optim.scheduler_factor,
        patience=config.optim.scheduler_patience,
    )

    # 创建本次训练的输出目录。
    run_dir = build_run_dir(config, split)

    wss_weights_tensor = (
        torch.tensor(config.optim.wss_weights, dtype=torch.float32) if eff_wss_dim > 0 else None
    )

    # V3: normalization_params_global.json 路径推导。
    norm_params_path = str(Path(config.data.data_root) / "normalization_params_global.json")

    trainer = FieldTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        loss_weights=torch.tensor(config.optim.target_weights, dtype=torch.float32),
        grad_clip_norm=config.optim.grad_clip_norm,
        physics_config=config.physics,
        interior_loss_boost=config.optim.interior_loss_boost,
        accumulate_grad_batches=config.optim.accumulate_grad_batches,
        log_dir=run_dir / "tb_logs",
        use_amp=config.system.amp,
        warmup_scheduler=warmup_scheduler,
        warmup_epochs=warmup_epochs,
        wss_loss_weight=config.optim.wss_loss_weight,
        wss_weights=wss_weights_tensor,
        early_stop_wss_weight=config.optim.early_stop_wss_weight,
        wss_loss_type=config.optim.wss_loss_type,
        wss_huber_beta=config.optim.wss_huber_beta,
        domain_loss_config=config.optim.domain_loss,
        norm_params_path=norm_params_path,
        early_stop_min_delta=config.optim.early_stop_min_delta,
        val_score_ema_alpha=config.optim.val_score_ema_alpha,
        val_score_wss_weights=config.optim.val_score_wss_weights,
        wss_target_names=wss_target_names,
        wss_target_frame=config.data.wss_target_frame,
        wss_output_mode=config.model.wss_output_mode,
        wss_metric_dim=config.model.wss_metric_dim if config.model.wss_output_mode == "vel_diff" else 0,
        vel_diff_variant=config.model.vel_diff_variant,
        select_best_on_data_loss=config.optim.select_best_on_data_loss,
        i6_grad_probe=config.optim.i6_grad_probe,
        i6_grad_probe_interval=config.optim.i6_grad_probe_interval,
    )
    # 保存本次训练用到的完整配置快照。
    dump_json(config.to_dict(), run_dir / "config.snapshot.json")
    # 保存当前 split 快照，保证结果可回溯。
    dump_json(split.to_dict(), run_dir / "split.snapshot.json")

    # 训练快照和 split 快照一起落盘，后续迁移到服务器或复现实验时不依赖外部状态。
    # 开始训练。
    fit_result = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config.optim.epochs,
        early_stopping_patience=config.optim.early_stopping_patience,
        run_dir=run_dir,
        save_every=config.run.save_every,
        save_best_only=config.run.save_best_only,
    )
    # 用最佳模型在测试集上做最终评估。
    test_metrics = trainer.evaluate(test_loader, checkpoint_path=run_dir / "best_model.pt")
    ckpt_wss = run_dir / "best_wss_model.pt"
    test_metrics_best_wss: Optional[Dict[str, float]] = None
    if ckpt_wss.is_file():
        test_metrics_best_wss = trainer.evaluate(test_loader, checkpoint_path=ckpt_wss)

    # 记录三个数据子集对应的图数量。
    dataset_sizes = {
        "num_train_graphs": len(train_dataset),
        "num_val_graphs": len(val_dataset),
        "num_test_graphs": len(test_dataset),
    }
    # summary 维持轻量，适合快速查看；完整信息放在 run_manifest。
    # 汇总最常看的关键信息。
    summary = {
        "device": str(device),
        "exp_id": config.meta.exp_id,
        "study_group": config.meta.study_group,
        "feature_set": config.meta.feature_set,
        "experiment_name": config.run.experiment_name,
        "split_version": split.split_version,
        "seed": config.system.seed,
        "model": config.model.name,
        "physics_enabled": config.physics.enabled,
        **dataset_sizes,
        "best_epoch": fit_result["best_epoch"],
        "best_val_score": fit_result["best_val_score"],
        "best_val_loss": fit_result["best_val_loss"],
        "best_wss_epoch": fit_result.get("best_wss_epoch", 0),
        "best_val_wss_r2": fit_result.get("best_val_wss_r2"),
        "test_metrics": test_metrics,
    }
    if test_metrics_best_wss is not None:
        summary["test_metrics_best_wss"] = test_metrics_best_wss
    # 写 summary.json。
    dump_json(summary, run_dir / "summary.json")
    # 写更完整的 run_manifest.json。
    dump_json(
        build_run_manifest(
            config=config,
            split=split,
            device=device,
            run_dir=run_dir,
            fit_result=fit_result,
            test_metrics=test_metrics,
            dataset_sizes=dataset_sizes,
            test_metrics_best_wss=test_metrics_best_wss,
        ),
        run_dir / "run_manifest.json",
    )
    # 同时把关键信息追加入总实验索引 CSV。
    append_experiment_index(
        output_root=config.run.output_root,
        row={
            "task": config.meta.task,
            "exp_id": config.meta.exp_id,
            "experiment_name": config.run.experiment_name,
            "study_group": config.meta.study_group,
            "feature_set": config.meta.feature_set,
            "ablation_axis": config.meta.ablation_axis,
            "run_dir": str(run_dir),
            "split_version": split.split_version,
            "seed": config.system.seed,
            "model": config.model.name,
            "enabled_node_features": "|".join(config.data.enabled_node_features),
            "enabled_global_features": "|".join(config.data.enabled_global_features),
            "augment": config.data.augment,
            "physics_enabled": config.physics.enabled,
            "physics_continuity_weight": config.physics.continuity_weight,
            "physics_momentum_weight": config.physics.momentum_weight,
            "physics_no_slip_weight": config.physics.no_slip_weight,
            "best_epoch": fit_result["best_epoch"],
            "best_val_score": fit_result["best_val_score"],
            "best_val_loss": fit_result["best_val_loss"],
            "best_wss_epoch": fit_result.get("best_wss_epoch", 0),
            "best_val_wss_r2": fit_result.get("best_val_wss_r2", ""),
            "test_rmse": test_metrics["rmse"],
            "test_rmse_vel_mag": test_metrics["rmse_vel_mag"],
            "test_rmse_p": test_metrics["rmse_p"],
            "test_r2_u": test_metrics["r2_u"],
            "test_r2_v": test_metrics["r2_v"],
            "test_r2_w": test_metrics["r2_w"],
            "test_r2_p": test_metrics["r2_p"],
            "test_r2_vel_mag": test_metrics["r2_vel_mag"],
            "head_layout": config.model.head_layout,
            "domain_loss_enabled": config.optim.domain_loss.enabled,
            "sampling_profile": "wall13000_near2000" if config.optim.domain_loss.enabled else "default",
        },
        fieldnames=[
            "task",
            "exp_id",
            "experiment_name",
            "study_group",
            "feature_set",
            "ablation_axis",
            "run_dir",
            "split_version",
            "seed",
            "model",
            "enabled_node_features",
            "enabled_global_features",
            "augment",
            "physics_enabled",
            "physics_continuity_weight",
            "physics_momentum_weight",
            "physics_no_slip_weight",
            "best_epoch",
            "best_val_score",
            "best_val_loss",
            "best_wss_epoch",
            "best_val_wss_r2",
            "test_rmse",
            "test_rmse_vel_mag",
            "test_rmse_p",
            "test_r2_u",
            "test_r2_v",
            "test_r2_w",
            "test_r2_p",
            "test_r2_vel_mag",
            "head_layout",
            "domain_loss_enabled",
            "sampling_profile",
        ],
    )

    if test_metrics_best_wss is not None:
        keys_p_wss = (
            "rmse_p",
            "mae_p",
            "r2_p",
            "loss_wall_pressure",
            "loss_wall_wss",
            "wss_rmse_wss",
            "wss_r2_wss",
            "wss_rmse",
        )

        def _pick(m: Dict[str, float], ks: tuple[str, ...]) -> Dict[str, float]:
            return {k: float(m[k]) for k in ks if k in m}

        print("\n[test 对比] best_model.pt vs best_wss_model.pt（压力 + 壁面 WSS）")
        print("  best_model:     ", _pick(test_metrics, keys_p_wss))
        print("  best_wss_model: ", _pick(test_metrics_best_wss, keys_p_wss))
    elif not ckpt_wss.is_file():
        print("\n[test] 未找到 best_wss_model.pt，无二次评估。")

    # 终端打印训练输出目录。
    print(f"训练完成，结果保存在: {run_dir}")
    # 终端打印 summary 便于快速查看。
    print(summary)


if __name__ == "__main__":
    # 作为脚本直接运行时，进入主函数。
    main()
