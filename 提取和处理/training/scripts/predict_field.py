from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ..core.config import ExperimentConfig, resolve_wss_effective_dim, resolve_wss_runtime_names
from ..core.data import FieldGraphDataset, build_dataloader, build_feature_mask, build_required_data_keys
from ..core.io import load_checkpoint, sanitize_batch_metadata
from ..core.models import build_field_model_from_config, split_model_output
from ..core.splits import SplitSpec
from ..core.denylist import resolve_split_subset
from ..core.utils import dump_json, ensure_dir, resolve_device, set_seed
from pipeline.config import NODE_FEATURE_NAMES, TARGET_NAMES


def resolve_cases(split: SplitSpec, subset: str, data_root: str):
    return resolve_split_subset(split, subset, data_root)


def extract_time_value(global_cond: torch.Tensor) -> torch.Tensor:
    # global_cond 在单图和 batch 后的形状可能不同，这里统一抽成标量时间值。
    if global_cond.ndim == 1:
        return global_cond[0:1].clone()
    if global_cond.ndim == 2:
        return global_cond[0, 0:1].clone()
    return global_cond.reshape(-1)[:1].clone()


def _prediction_pt_filename(case_name: str, sample_id: str) -> str:
    """导出文件名在测试集上必须全局唯一（不同病例常有相同图 stem / sample_id）。"""
    safe_case = str(case_name).replace("/", "__").replace(" ", "_")
    safe_sid = str(sample_id).replace("/", "_")
    return f"{safe_case}__{safe_sid}.pt"


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
    eff_wss_dim = resolve_wss_effective_dim(
        config.model.wss_dim,
        config.model.wss_output_mode,
        config.model.wss_metric_dim,
    )
    wss_target_names = resolve_wss_runtime_names(
        config.data.wss_target_frame,
        config.model.wss_dim,
        config.model.wss_output_mode,
        config.model.wss_metric_dim,
    )
    required_data_keys = build_required_data_keys(
        config.model.name,
        wss_dim=eff_wss_dim,
        wss_target_frame=config.data.wss_target_frame,
    )
    dataset = FieldGraphDataset(
        root=config.data.data_root,
        case_names=resolve_cases(split, args.subset, config.data.data_root),
        graphs_subdir=config.data.graphs_subdir,
        augment=False,
        preload=config.data.preload,
        feature_mask=feature_mask,
        required_keys=required_data_keys,
        wss_target_frame=config.data.wss_target_frame,
    )
    loader = build_dataloader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
    )

    model = build_field_model_from_config(config).to(device)
    load_checkpoint(model, args.checkpoint, device)
    model.eval()

    manifest = []
    for batch in loader:
        batch = batch.to(device)
        pred, wss_pred = split_model_output(model(batch))
        batch_cpu = batch.cpu()
        pred_cpu = pred.cpu()
        wss_pred_cpu = wss_pred.cpu() if wss_pred is not None else None

        sample_ids = sanitize_batch_metadata(batch.sample_id)
        case_names = sanitize_batch_metadata(batch.case_name)
        graph_paths = sanitize_batch_metadata(batch.graph_path)

        data_list = batch_cpu.to_data_list()
        # DataLoader batch 后预测会拼接到一起，这里按单图节点数切回逐图结果。
        pred_list = pred_cpu.split([data.y.size(0) for data in data_list], dim=0)
        wss_pred_list = (
            wss_pred_cpu.split([data.y.size(0) for data in data_list], dim=0)
            if wss_pred_cpu is not None
            else [None] * len(data_list)
        )

        for data, pred_item, wss_pred_item, sample_id, case_name, graph_path in zip(
            data_list, pred_list, wss_pred_list, sample_ids, case_names, graph_paths
        ):
            save_path = output_dir / _prediction_pt_filename(case_name, sample_id)
            payload = {
                "sample_id": sample_id,
                "case_name": case_name,
                # regional_eval 优先从 graph_path 读未 mask 的完整节点特征，区域口径与模型输入开关无关。
                "graph_path": graph_path,
                "node_feature_names": NODE_FEATURE_NAMES,
                "target_names": TARGET_NAMES,
                "wall_mask": data.x[:, NODE_FEATURE_NAMES.index("is_wall")].bool(),
                "time_value": extract_time_value(data.global_cond),
                # 这里故意把输入、真值和预测一起保存。
                # 这样任务 B 做 CFD vs AI 对照时，不需要再额外回原始图目录找数据。
                "x": data.x,
                "global_cond": data.global_cond,
                "edge_index": data.edge_index,
                "y_true": data.y,
                "y_pred": pred_item,
            }
            if wss_pred_item is not None:
                payload["wss_target_names"] = wss_target_names
                payload["wss_target_frame"] = config.data.wss_target_frame
                payload["y_wss_pred"] = wss_pred_item
                if hasattr(data, "y_wss") and data.y_wss is not None:
                    payload["y_wss_true"] = data.y_wss
                if hasattr(data, "y_wss_global") and data.y_wss_global is not None:
                    payload["y_wss_global_true"] = data.y_wss_global
            torch.save(payload, save_path)
            manifest.append(
                {
                    "sample_id": sample_id,
                    "case_name": case_name,
                    "graph_path": graph_path,
                    "prediction_path": str(save_path),
                    "wall_nodes": int(data.x[:, NODE_FEATURE_NAMES.index("is_wall")].sum().item()),
                }
            )

    dump_json(
        {
            "subset": args.subset,
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "config_path": str(Path(args.config).resolve()),
            "model_name": config.model.name,
            "split_version": split.split_version,
            "num_predictions": len(manifest),
            "items": manifest,
        },
        output_dir / "manifest.json",
    )
    print(f"已导出 {len(manifest)} 个预测样本到: {output_dir}")


if __name__ == "__main__":
    main()
