from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .config import ExperimentConfig
from .data import FieldGraphDataset, build_dataloader, build_feature_mask
from .models import build_model
from .splits import SplitSpec
from .trainer import FieldTrainer
from .utils import dump_json, ensure_dir, resolve_device, set_seed


def resolve_cases(split: SplitSpec, subset: str):
    mapping = {
        "train": split.train_cases,
        "val": split.val_cases,
        "test": split.test_cases,
    }
    if subset not in mapping:
        raise ValueError(f"未知 subset: {subset}")
    return mapping[subset]


def main() -> None:
    parser = argparse.ArgumentParser(description="任务A独立评估脚本")
    parser.add_argument("--config", required=True, help="JSON 配置文件路径")
    parser.add_argument("--checkpoint", required=True, help="模型权重路径")
    parser.add_argument(
        "--subset",
        default="test",
        choices=["train", "val", "test"],
        help="评估子集",
    )
    parser.add_argument(
        "--output",
        default="",
        help="评估结果保存目录，默认保存到 checkpoint 同级 eval_<subset>",
    )
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
    dataset = FieldGraphDataset(
        root=config.data.data_root,
        case_names=resolve_cases(split, args.subset),
        graphs_subdir=config.data.graphs_subdir,
        augment=False,
        preload=config.data.preload,
        feature_mask=feature_mask,
    )
    loader = build_dataloader(
        dataset,
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
    optimizer = torch.optim.Adam(model.parameters(), lr=config.optim.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")

    trainer = FieldTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        loss_weights=torch.tensor(config.optim.target_weights, dtype=torch.float32),
        grad_clip_norm=config.optim.grad_clip_norm,
    )
    metrics = trainer.evaluate(loader, checkpoint_path=Path(args.checkpoint))

    output_dir = (
        ensure_dir(args.output)
        if args.output
        else ensure_dir(Path(args.checkpoint).resolve().parent / f"eval_{args.subset}")
    )
    # 评估结果单独导出，避免后续重新训练才能拿到测试指标。
    dump_json(
        {
            "subset": args.subset,
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "num_graphs": len(dataset),
            "metrics": metrics,
        },
        output_dir / "metrics.json",
    )
    print(metrics)


if __name__ == "__main__":
    main()
