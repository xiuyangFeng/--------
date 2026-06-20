from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader

from .data import build_datasets, collate_crown, load_train_p_stats
from .metrics import grouped_metrics, metric_ranges, regression_metrics, write_metric_rows
from .model import CrownPointNet
from .utils import default_private_preprocessed_root, dump_json, load_config, project_root, resolve_device


def _prepare_eval_minibatch(
    batch: Dict[str, Any],
    sample_points: int,
    device: torch.device,
    eval_seed: int,
    sample_offset: int,
) -> tuple[torch.Tensor, torch.Tensor, List[tuple[str, str, int]]]:
    input_indices = batch["input_indices"]
    pv_batch: List[torch.Tensor] = []
    label_batch: List[torch.Tensor] = []
    meta: List[tuple[str, str, int]] = []

    for local_idx, (feat, targ, case_name, sample_id) in enumerate(
        zip(batch["features"], batch["targets"], batch["case_names"], batch["sample_ids"])
    ):
        feat = feat.to(device, non_blocking=True)
        targ = targ.to(device, non_blocking=True)
        n = feat.shape[1]
        choice = min(sample_points, n)
        g = torch.Generator(device=device)
        g.manual_seed(eval_seed + sample_offset + local_idx)
        idx = torch.randperm(n, generator=g, device=device)[:choice]
        pv_batch.append(feat[input_indices][:, idx])
        label_batch.append(targ[:, idx])
        meta.append((case_name, sample_id, choice))

    return torch.stack(pv_batch, dim=0), torch.stack(label_batch, dim=0), meta


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
    sample_offset = 0

    for batch in loader:
        pv_data, label_data, meta = _prepare_eval_minibatch(
            batch, sample_points, device, eval_seed, sample_offset
        )
        with torch.no_grad():
            pred = model(pv_data)

        for b, (case_name, sample_id, choice) in enumerate(meta):
            preds.append(pred[b].transpose(0, 1).cpu())
            targets.append(label_data[b].transpose(0, 1).cpu())
            case_names.extend([case_name] * choice)
            sample_ids.extend([sample_id] * choice)

        sample_offset += len(meta)

    return torch.cat(preds, dim=0), torch.cat(targets, dim=0), case_names, sample_ids


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    split_name: str = "test",
    output_dir: str | Path | None = None,
    p_min: float | None = None,
    p_max: float | None = None,
    lazy_load: bool | None = None,
) -> Dict[str, float]:
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config") or load_config(checkpoint_path.parent / "config.json")

    device = resolve_device(config["system"].get("device", "auto"))
    datasets = build_datasets(config, splits=(split_name,), lazy_load=lazy_load)
    dataset = datasets[split_name]

    data_cfg = config["data"]
    eval_batch_size = int(data_cfg.get("eval_batch_size", data_cfg.get("batch_size", 16)))
    loader = DataLoader(
        dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=int(data_cfg.get("eval_num_workers", 0)),
        pin_memory=bool(data_cfg.get("pin_memory", False)),
        collate_fn=collate_crown,
    )

    if p_min is None or p_max is None:
        root = project_root()
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

    nmae_ranges = metric_ranges(target_all, output_names, p_min=p_min, p_max=p_max)
    metrics = regression_metrics(
        pred_all,
        target_all,
        output_names,
        p_min=p_min,
        p_max=p_max,
        nmae_ranges=nmae_ranges,
    )
    metrics["n_points"] = int(target_all.shape[0])

    vel_wall = (
        target_all[:, 0].square() + target_all[:, 1].square() + target_all[:, 2].square()
    ) <= 0.01
    if vel_wall.any():
        wm = regression_metrics(
            pred_all[vel_wall],
            target_all[vel_wall],
            output_names,
            p_min=p_min,
            p_max=p_max,
            nmae_ranges=nmae_ranges,
        )
        for k, v in wm.items():
            metrics[f"wall_{k}"] = v
    interior = ~vel_wall
    if interior.any():
        im = regression_metrics(
            pred_all[interior],
            target_all[interior],
            output_names,
            p_min=p_min,
            p_max=p_max,
            nmae_ranges=nmae_ranges,
        )
        for k, v in im.items():
            metrics[f"interior_{k}"] = v

    out_dir = Path(output_dir) if output_dir else checkpoint_path.parent
    dump_json(out_dir / f"metrics_{split_name}.json", metrics)
    write_metric_rows(
        out_dir / f"metrics_{split_name}_by_case.csv",
        grouped_metrics(
            pred_all,
            target_all,
            case_names,
            output_names,
            p_min=p_min,
            p_max=p_max,
            nmae_ranges=nmae_ranges,
        ),
    )
    write_metric_rows(
        out_dir / f"metrics_{split_name}_by_sample.csv",
        grouped_metrics(
            pred_all,
            target_all,
            sample_ids,
            output_names,
            p_min=p_min,
            p_max=p_max,
            nmae_ranges=nmae_ranges,
        ),
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CROWN checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--eager-load",
        action="store_true",
        help="强制读取 merged pkl（默认 lazy partial）",
    )
    args = parser.parse_args()
    lazy_load = False if args.eager_load else None
    metrics = evaluate_checkpoint(args.checkpoint, args.split, args.output_dir, lazy_load=lazy_load)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
