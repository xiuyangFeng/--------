from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import torch

from .config import ExperimentConfig
from .data import FieldGraphDataset, build_dataloader, build_feature_mask
from .io import append_experiment_index
from .models import build_model
from .splits import SplitSpec
from .trainer import FieldTrainer
from .utils import dump_json, ensure_dir, resolve_device, set_seed, timestamp


def build_run_dir(config: ExperimentConfig, split: SplitSpec) -> Path:
    # run_dir 把 experiment_name / split / seed / timestamp 全部带上，方便后续追溯单次实验。
    run_name = f"{config.run.experiment_name}_{split.split_version}_seed{config.system.seed}_{timestamp()}"
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
    parser = argparse.ArgumentParser(description="任务A场重建训练入口")
    parser.add_argument("--config", required=True, help="JSON 配置文件路径")
    args = parser.parse_args()

    config = ExperimentConfig.from_json(args.config)
    config.validate()
    split = SplitSpec.from_json(config.data.split_file)

    # physics 尺度必须在构建 trainer 前解析完成，否则 loss 会退回默认 1.0。
    train_case_dirs = [Path(config.data.data_root) / case_name for case_name in split.train_cases]
    config.physics.resolve_scales_from_data(
        data_root=config.data.data_root,
        graphs_subdir=config.data.graphs_subdir,
        case_dirs=train_case_dirs,
    )

    set_seed(config.system.seed, deterministic=config.system.deterministic)
    device = resolve_device(config.system.device)

    feature_mask = build_feature_mask(
        enabled_node_features=config.data.enabled_node_features,
        enabled_global_features=config.data.enabled_global_features,
    )

    # 训练/验证/测试共用同一套 split 文件，保证后续任务 B、C 可以回溯到统一划分。
    train_dataset = FieldGraphDataset(
        root=config.data.data_root,
        case_names=split.train_cases,
        graphs_subdir=config.data.graphs_subdir,
        augment=config.data.augment,
        augment_config=config.data.augment_config,
        preload=config.data.preload,
        feature_mask=feature_mask,
    )
    val_dataset = FieldGraphDataset(
        root=config.data.data_root,
        case_names=split.val_cases,
        graphs_subdir=config.data.graphs_subdir,
        augment=False,
        preload=config.data.preload,
        feature_mask=feature_mask,
    )
    test_dataset = FieldGraphDataset(
        root=config.data.data_root,
        case_names=split.test_cases,
        graphs_subdir=config.data.graphs_subdir,
        augment=False,
        preload=config.data.preload,
        feature_mask=feature_mask,
    )

    train_loader = build_dataloader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
    )
    val_loader = build_dataloader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
    )
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
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.optim.lr,
        weight_decay=config.optim.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.optim.scheduler_factor,
        patience=config.optim.scheduler_patience,
    )

    trainer = FieldTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        loss_weights=torch.tensor(config.optim.target_weights, dtype=torch.float32),
        grad_clip_norm=config.optim.grad_clip_norm,
        physics_config=config.physics,
    )

    run_dir = build_run_dir(config, split)
    dump_json(config.to_dict(), run_dir / "config.snapshot.json")
    dump_json(split.to_dict(), run_dir / "split.snapshot.json")

    # 训练快照和 split 快照一起落盘，后续迁移到服务器或复现实验时不依赖外部状态。
    fit_result = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config.optim.epochs,
        early_stopping_patience=config.optim.early_stopping_patience,
        run_dir=run_dir,
        save_every=config.run.save_every,
        save_best_only=config.run.save_best_only,
    )
    test_metrics = trainer.evaluate(test_loader, checkpoint_path=run_dir / "best_model.pt")

    dataset_sizes = {
        "num_train_graphs": len(train_dataset),
        "num_val_graphs": len(val_dataset),
        "num_test_graphs": len(test_dataset),
    }
    # summary 维持轻量，适合快速查看；完整信息放在 run_manifest。
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
    dump_json(summary, run_dir / "summary.json")
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
            "test_rmse_p": test_metrics["rmse_p"],
            "test_r2_p": test_metrics["r2_p"],
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
            "test_rmse_p",
            "test_r2_p",
        ],
    )

    print(f"训练完成，结果保存在: {run_dir}")
    print(summary)


if __name__ == "__main__":
    main()
