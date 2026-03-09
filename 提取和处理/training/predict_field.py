from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .config import ExperimentConfig
from .data import FieldGraphDataset, build_dataloader, build_feature_mask
from .io import load_checkpoint, sanitize_batch_metadata
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


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="任务A预测导出脚本")
    parser.add_argument("--config", required=True, help="JSON 配置文件路径")
    parser.add_argument("--checkpoint", required=True, help="模型权重路径")
    parser.add_argument(
        "--subset",
        default="test",
        choices=["train", "val", "test"],
        help="导出子集",
    )
    parser.add_argument(
        "--output",
        default="",
        help="导出目录，默认保存到 checkpoint 同级 predictions_<subset>",
    )
    args = parser.parse_args()

    config = ExperimentConfig.from_json(args.config)
    config.validate()
    split = SplitSpec.from_json(config.data.split_file)

    set_seed(config.system.seed, deterministic=config.system.deterministic)
    device = resolve_device(config.system.device)
    output_dir = (
        ensure_dir(args.output)
        if args.output
        else ensure_dir(Path(args.checkpoint).resolve().parent / f"predictions_{args.subset}")
    )

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
        batch_size=1,
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
    load_checkpoint(model, args.checkpoint, device)
    model.eval()

    manifest = []
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)
        batch_cpu = batch.cpu()
        pred_cpu = pred.cpu()

        sample_ids = sanitize_batch_metadata(batch.sample_id)
        case_names = sanitize_batch_metadata(batch.case_name)
        graph_paths = sanitize_batch_metadata(batch.graph_path)

        data_list = batch_cpu.to_data_list()
        # DataLoader batch 后预测会拼接到一起，这里按单图节点数切回逐图结果。
        pred_list = pred_cpu.split([data.y.size(0) for data in data_list], dim=0)

        for data, pred_item, sample_id, case_name, graph_path in zip(
            data_list, pred_list, sample_ids, case_names, graph_paths
        ):
            save_path = output_dir / f"{sample_id}.pt"
            torch.save(
                {
                    "sample_id": sample_id,
                    "case_name": case_name,
                    "graph_path": graph_path,
                    # 保留输入与真值，方便后续任务 B 直接读取并做指标恢复。
                    "x": data.x,
                    "global_cond": data.global_cond,
                    "edge_index": data.edge_index,
                    "y_true": data.y,
                    "y_pred": pred_item,
                },
                save_path,
            )
            manifest.append(
                {
                    "sample_id": sample_id,
                    "case_name": case_name,
                    "graph_path": graph_path,
                    "prediction_path": str(save_path),
                }
            )

    dump_json(
        {
            "subset": args.subset,
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "num_predictions": len(manifest),
            "items": manifest,
        },
        output_dir / "manifest.json",
    )
    print(f"已导出 {len(manifest)} 个预测样本到: {output_dir}")


if __name__ == "__main__":
    main()
