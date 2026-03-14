from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import torch

from .config import ExperimentConfig
from .data import FieldGraphDataset, build_dataloader, build_feature_mask
from .io import load_checkpoint
from .losses import build_loss_plugin
from .metrics import RegressionMeter
from .models import build_model
from .splits import SplitSpec
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


def evaluate_checkpoint(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    loss_weights: torch.Tensor,
    physics_config=None,
    eval_epoch: int = 10**9,
    checkpoint_path: Path | str | None = None,
) -> Dict[str, float]:
    """Standalone evaluation that matches training-time metric semantics."""
    if checkpoint_path is not None:
        load_checkpoint(model, checkpoint_path, device)
    model.eval()
    meter = RegressionMeter()
    weights = loss_weights.to(device)
    loss_plugin = build_loss_plugin(physics_config)
    extra_totals: Dict[str, float] = {}
    num_batches = 0

    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)
        breakdown = loss_plugin.build_loss(
            model=model,
            batch=batch,
            pred=pred,
            target=batch.y,
            data_weights=weights,
            epoch=eval_epoch,
            train=False,
        )
        meter.update(pred, batch.y, breakdown.total_loss.item())
        for key, value in breakdown.scalar_dict().items():
            extra_totals[key] = extra_totals.get(key, 0.0) + value
        num_batches += 1

    metrics = meter.compute()
    for key, total in extra_totals.items():
        metrics[key] = total / max(1, num_batches)
    metrics["physics_enabled"] = float(loss_plugin.is_enabled(eval_epoch))
    return metrics


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

    metrics = evaluate_checkpoint(
        model=model,
        loader=loader,
        device=device,
        loss_weights=torch.tensor(config.optim.target_weights, dtype=torch.float32),
        physics_config=config.physics,
        checkpoint_path=Path(args.checkpoint),
    )

    output_dir = (
        ensure_dir(args.output)
        if args.output
        else ensure_dir(Path(args.checkpoint).resolve().parent / f"eval_{args.subset}")
    )
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
