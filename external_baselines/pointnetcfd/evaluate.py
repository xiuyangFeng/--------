from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import TARGET_MODES, PointCloudBatch, PointNetCFDDataset, collate_pointclouds
from .metrics import expand_graph_labels, grouped_metrics, regression_metrics, write_metric_rows
from .model import PointNetCFD
from .utils import dump_json, load_config, move_batch, resolve_device
from training.core.splits import SplitSpec


def _build_dataset(config: Dict[str, object], split_name: str) -> PointNetCFDDataset:
    data_cfg = config["data"]
    split = SplitSpec.from_json(data_cfg["split_file"])
    split_cases = {
        "train": split.train_cases,
        "val": split.val_cases,
        "test": split.test_cases,
    }
    if split_name not in split_cases:
        raise KeyError(f"未知 split={split_name}; 可选 train/val/test")
    return PointNetCFDDataset(
        data_root=data_cfg["data_root"],
        case_names=split_cases[split_name],
        graphs_subdir=data_cfg.get("graphs_subdir", "processed/graphs"),
        node_features=data_cfg["node_features"],
        global_features=data_cfg["global_features"],
        target_mode=config["target"]["mode"],
    )


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    split_name: str,
    output_dir: str | Path | None = None,
    save_predictions: bool = False,
) -> Dict[str, float]:
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config")
    if config is None:
        config_path = checkpoint_path.parent / "config.json"
        config = load_config(config_path)

    device = resolve_device(config["system"].get("device", "auto"))
    dataset = _build_dataset(config, split_name)
    loader = DataLoader(
        dataset,
        batch_size=config["data"].get("batch_size", 2),
        shuffle=False,
        num_workers=config["data"].get("num_workers", 0),
        pin_memory=config["data"].get("pin_memory", False),
        collate_fn=collate_pointclouds,
    )
    output_names = TARGET_MODES[config["target"]["mode"]]
    model = PointNetCFD(
        input_dim=dataset.input_dim,
        global_dim=dataset.global_dim,
        output_dim=len(output_names),
        **config["model"],
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    preds: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []
    case_groups: List[str] = []
    sample_groups: List[str] = []
    with torch.no_grad():
        for raw_batch in loader:
            batch: PointCloudBatch = move_batch(raw_batch, device)
            pred = model(batch.node_input, batch.global_cond, batch.batch)
            preds.append(pred.cpu())
            targets.append(batch.target.cpu())
            case_groups.extend(expand_graph_labels(batch.batch, batch.case_names))
            sample_labels = [f"{c}/{s}" for c, s in zip(batch.case_names, batch.sample_ids)]
            sample_groups.extend(expand_graph_labels(batch.batch, sample_labels))

    pred_all = torch.cat(preds, dim=0)
    target_all = torch.cat(targets, dim=0)
    metrics = regression_metrics(pred_all, target_all, output_names)
    metrics["n_points"] = int(target_all.shape[0])

    out_dir = Path(output_dir) if output_dir is not None else checkpoint_path.parent
    dump_json(out_dir / f"metrics_{split_name}.json", metrics)
    write_metric_rows(
        out_dir / f"metrics_{split_name}_by_case.csv",
        grouped_metrics(pred_all, target_all, case_groups, output_names),
    )
    write_metric_rows(
        out_dir / f"metrics_{split_name}_by_sample.csv",
        grouped_metrics(pred_all, target_all, sample_groups, output_names),
    )
    if save_predictions:
        np.savez_compressed(
            out_dir / f"predictions_{split_name}.npz",
            pred=pred_all.numpy(),
            target=target_all.numpy(),
            case=np.asarray(case_groups),
            sample=np.asarray(sample_groups),
            output_names=np.asarray(output_names),
        )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a PointNetCFD checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Path to best_model.pt")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--save-predictions", action="store_true")
    args = parser.parse_args()
    metrics = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        split_name=args.split,
        output_dir=args.output_dir,
        save_predictions=args.save_predictions,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
