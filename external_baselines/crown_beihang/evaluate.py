from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from .data import build_datasets, collate_crown, load_train_p_stats
from .metrics import grouped_metrics, regression_metrics, write_metric_rows
from .model import CrownPointNet
from .utils import default_private_preprocessed_root, dump_json, load_config, project_root, resolve_device


def evaluate_model(
    model: CrownPointNet,
    loader: DataLoader,
    device: torch.device,
    output_names: List[str],
    sample_points: int,
    eval_seed: int,
) -> tuple[torch.Tensor, torch.Tensor, List[str], List[str]]:
    model.eval()
    preds: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []
    case_names: List[str] = []
    sample_ids: List[str] = []
    g = torch.Generator(device=device)
    g.manual_seed(eval_seed)

    for batch in loader:
        input_indices = batch["input_indices"]
        for feat, targ, case_name, sample_id in zip(
            batch["features"], batch["targets"], batch["case_names"], batch["sample_ids"]
        ):
            feat = feat.to(device)
            targ = targ.to(device)
            n = feat.shape[1]
            choice = min(sample_points, n)
            idx = torch.randperm(n, generator=g, device=device)[:choice]
            model_in = feat[input_indices][:, idx]
            with torch.no_grad():
                pred = model(model_in.unsqueeze(0))[0].transpose(0, 1)
            preds.append(pred.cpu())
            targets.append(targ[:, idx].transpose(0, 1).cpu())
            case_names.extend([case_name] * choice)
            sample_ids.extend([sample_id] * choice)

    return torch.cat(preds, dim=0), torch.cat(targets, dim=0), case_names, sample_ids


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    split_name: str = "test",
    output_dir: str | Path | None = None,
    p_min: float | None = None,
    p_max: float | None = None,
) -> Dict[str, float]:
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config") or load_config(checkpoint_path.parent / "config.json")

    device = resolve_device(config["system"].get("device", "auto"))
    datasets = build_datasets(config)
    dataset = datasets[split_name]
    loader = DataLoader(
        dataset,
        batch_size=config["data"].get("batch_size", 16),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_crown,
    )

    if p_min is None or p_max is None:
        root = project_root()
        data_cfg = config["data"]
        output_root = Path(data_cfg.get("preprocessed_root", str(default_private_preprocessed_root())))
        if not output_root.is_absolute():
            output_root = root / output_root
        stats = load_train_p_stats(
            output_root / "stats" / "train_stats.json",
            data_cfg.get("point_filter", "volume"),
        )
        p_min = float(stats["p_min"])
        p_max = float(stats["p_max"])

    model = CrownPointNet(input_dim=dataset.input_dim, output_dim=4).to(device)
    model.load_state_dict(checkpoint["model"])
    output_names = list(dataset.target_names)
    sample_points = int(config["data"].get("sample_points", 10000))
    eval_seed = int(config["system"].get("seed", 1)) + 17

    pred_all, target_all, case_names, sample_ids = evaluate_model(
        model, loader, device, output_names, sample_points, eval_seed
    )

    metrics = regression_metrics(pred_all, target_all, output_names, p_min=p_min, p_max=p_max)
    metrics["n_points"] = int(target_all.shape[0])

    vel_wall = (
        target_all[:, 0].square() + target_all[:, 1].square() + target_all[:, 2].square()
    ) <= 0.01
    if vel_wall.any():
        wm = regression_metrics(
            pred_all[vel_wall], target_all[vel_wall], output_names, p_min=p_min, p_max=p_max
        )
        for k, v in wm.items():
            metrics[f"wall_{k}"] = v
    interior = ~vel_wall
    if interior.any():
        im = regression_metrics(
            pred_all[interior], target_all[interior], output_names, p_min=p_min, p_max=p_max
        )
        for k, v in im.items():
            metrics[f"interior_{k}"] = v

    out_dir = Path(output_dir) if output_dir else checkpoint_path.parent
    dump_json(out_dir / f"metrics_{split_name}.json", metrics)
    write_metric_rows(
        out_dir / f"metrics_{split_name}_by_case.csv",
        grouped_metrics(pred_all, target_all, case_names, output_names, p_min=p_min, p_max=p_max),
    )
    write_metric_rows(
        out_dir / f"metrics_{split_name}_by_sample.csv",
        grouped_metrics(pred_all, target_all, sample_ids, output_names, p_min=p_min, p_max=p_max),
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CROWN checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    metrics = evaluate_checkpoint(args.checkpoint, args.split, args.output_dir)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
