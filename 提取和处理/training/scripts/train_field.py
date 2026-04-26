from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import torch

from ..core.config import ExperimentConfig
from ..core.data import (
    FieldGraphDataset,
    build_dataloader,
    build_feature_mask,
    build_required_data_keys,
)
from ..core.io import append_experiment_index
from ..core.models import build_model
from ..core.splits import SplitSpec
from ..core.trainer import FieldTrainer
from ..core.utils import dump_json, ensure_dir, resolve_device, set_seed, timestamp


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
) -> Dict[str, object]:
    # run_manifest 是给“训练结束后的世界”看的，而不是给训练过程看的。
    # 后面做任务 B/C、实验记录汇总、论文表格回溯时，都尽量读这个文件。
    # 返回结构化运行清单，尽可能把复现实验所需的信息放全。
    return {
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
            "scheduler_factor": config.optim.scheduler_factor,
            "scheduler_patience": config.optim.scheduler_patience,
            "early_stopping_patience": config.optim.early_stopping_patience,
            "target_weights": config.optim.target_weights,
            "interior_loss_boost": config.optim.interior_loss_boost,
            "grad_clip_norm": config.optim.grad_clip_norm,
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
        },
        "dataset_sizes": dataset_sizes,
        "best_epoch": fit_result["best_epoch"],
        "best_val_loss": fit_result["best_val_loss"],
        "test_metrics": test_metrics,
    }


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

    # physics 尺度必须在构建 trainer 前解析完成，否则 loss 会退回默认 1.0。
    # 先把训练病例目录路径组织出来，供 physics 自动读取尺度。
    train_case_dirs = [Path(config.data.data_root) / case_name for case_name in split.train_cases]
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
    required_data_keys = build_required_data_keys(config.model.name, wss_dim=config.model.wss_dim)

    # 训练/验证/测试共用同一套 split 文件，保证后续任务 B、C 可以回溯到统一划分。
    # 构建训练集数据对象。
    train_dataset = FieldGraphDataset(
        root=config.data.data_root,
        case_names=split.train_cases,
        graphs_subdir=config.data.graphs_subdir,
        augment=config.data.augment,
        augment_config=config.data.augment_config,
        preload=config.data.preload,
        feature_mask=feature_mask,
        required_keys=required_data_keys,
    )
    # 构建验证集数据对象；验证集不做增强。
    val_dataset = FieldGraphDataset(
        root=config.data.data_root,
        case_names=split.val_cases,
        graphs_subdir=config.data.graphs_subdir,
        augment=False,
        preload=config.data.preload,
        feature_mask=feature_mask,
        required_keys=required_data_keys,
    )
    # 构建测试集数据对象；测试集同样不做增强。
    test_dataset = FieldGraphDataset(
        root=config.data.data_root,
        case_names=split.test_cases,
        graphs_subdir=config.data.graphs_subdir,
        augment=False,
        preload=config.data.preload,
        feature_mask=feature_mask,
        required_keys=required_data_keys,
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

    model = build_model(
        model_name=config.model.name,
        hidden_dim=config.model.hidden_dim,
        num_layers=config.model.num_layers,
        dropout=config.model.dropout,
        heads=config.model.heads,
        use_transformer_prenorm=config.model.use_transformer_prenorm,
        wss_dim=config.model.wss_dim,
    ).to(device)

    # 统计模型总参数量。
    total_params = sum(p.numel() for p in model.parameters())
    # 统计可训练参数量。
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型: {config.model.name} | 总参数: {total_params:,} | 可训练: {trainable_params:,}")

    # 优化器使用 Adam。
    optimizer = torch.optim.Adam(
        model.parameters(),
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

    wss_weights_tensor = torch.tensor(config.optim.wss_weights, dtype=torch.float32) if config.model.wss_dim > 0 else None
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
        "best_val_loss": fit_result["best_val_loss"],
        "test_metrics": test_metrics,
    }
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
            "best_val_loss": fit_result["best_val_loss"],
            "test_rmse": test_metrics["rmse"],
            "test_rmse_vel_mag": test_metrics["rmse_vel_mag"],
            "test_rmse_p": test_metrics["rmse_p"],
            "test_r2_u": test_metrics["r2_u"],
            "test_r2_v": test_metrics["r2_v"],
            "test_r2_w": test_metrics["r2_w"],
            "test_r2_p": test_metrics["r2_p"],
            "test_r2_vel_mag": test_metrics["r2_vel_mag"],
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
            "best_val_loss",
            "test_rmse",
            "test_rmse_vel_mag",
            "test_rmse_p",
            "test_r2_u",
            "test_r2_v",
            "test_r2_w",
            "test_r2_p",
            "test_r2_vel_mag",
        ],
    )

    # 终端打印训练输出目录。
    print(f"训练完成，结果保存在: {run_dir}")
    # 终端打印 summary 便于快速查看。
    print(summary)


if __name__ == "__main__":
    # 作为脚本直接运行时，进入主函数。
    main()
