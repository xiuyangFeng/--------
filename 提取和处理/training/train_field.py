from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .config import ExperimentConfig
from .data import FieldGraphDataset, build_dataloader, build_feature_mask
from .io import append_experiment_index
from .models import build_model
from .splits import SplitSpec
from .trainer import FieldTrainer
from .utils import dump_json, ensure_dir, resolve_device, set_seed, timestamp


def build_run_dir(config: ExperimentConfig, split: SplitSpec) -> Path:
    run_name = f"{config.run.experiment_name}_{split.split_version}_seed{config.system.seed}_{timestamp()}"
    return ensure_dir(Path(config.run.output_root) / run_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="任务A场重建训练入口")
    parser.add_argument("--config", required=True, help="JSON 配置文件路径")
    args = parser.parse_args()

    config = ExperimentConfig.from_json(args.config)
    config.validate()
    split = SplitSpec.from_json(config.data.split_file)

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

    summary = {
        "device": str(device),
        "experiment_name": config.run.experiment_name,
        "split_version": split.split_version,
        "seed": config.system.seed,
        "model": config.model.name,
        "num_train_graphs": len(train_dataset),
        "num_val_graphs": len(val_dataset),
        "num_test_graphs": len(test_dataset),
        "best_epoch": fit_result["best_epoch"],
        "best_val_loss": fit_result["best_val_loss"],
        "test_metrics": test_metrics,
    }
    dump_json(summary, run_dir / "summary.json")
    append_experiment_index(
        output_root=config.run.output_root,
        row={
            "experiment_name": config.run.experiment_name,
            "run_dir": str(run_dir),
            "split_version": split.split_version,
            "seed": config.system.seed,
            "model": config.model.name,
            "best_epoch": fit_result["best_epoch"],
            "best_val_loss": fit_result["best_val_loss"],
            "test_rmse": test_metrics["rmse"],
            "test_rmse_p": test_metrics["rmse_p"],
            "test_r2_p": test_metrics["r2_p"],
        },
        fieldnames=[
            "experiment_name",
            "run_dir",
            "split_version",
            "seed",
            "model",
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
