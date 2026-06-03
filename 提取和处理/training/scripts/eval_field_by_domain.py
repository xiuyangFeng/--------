#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按数据域（AAA / AG / ILO）汇总 field 模型指标，供 V3D 探针门禁分域汇报。

示例:
  python -m training.scripts.eval_field_by_domain \\
    --config outputs/field/<run>/config.snapshot.json \\
    --checkpoint outputs/field/<run>/best_wss_model.pt \\
    --subset test
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import torch

from ..core.config import ExperimentConfig, resolve_wss_effective_dim, resolve_wss_runtime_names
from ..core.data import FieldGraphDataset, build_dataloader, build_feature_mask
from ..core.models import build_field_model_from_config
from ..core.splits import SplitSpec
from ..core.utils import dump_json, ensure_dir, resolve_device, set_seed
from .eval_field import evaluate_checkpoint, resolve_cases


def case_domain(case_name: str) -> str:
    return case_name.split("/")[0]


def group_cases_by_domain(case_names: List[str]) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = defaultdict(list)
    for name in case_names:
        groups[case_domain(name)].append(name)
    return dict(sorted(groups.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description="按 AAA/AG/ILO 分域评估 field 模型")
    parser.add_argument("--config", required=True, help="config.snapshot.json")
    parser.add_argument("--checkpoint", required=True, help="模型权重")
    parser.add_argument(
        "--subset",
        default="test",
        choices=["train", "val", "test"],
    )
    parser.add_argument(
        "--output",
        default="",
        help="输出目录，默认 <checkpoint_parent>/eval_by_domain_<subset>",
    )
    parser.add_argument(
        "--domains",
        nargs="*",
        default=None,
        help="仅评估指定域，如 AAA AG ILO；默认全部",
    )
    args = parser.parse_args()

    config = ExperimentConfig.from_json(args.config)
    config.validate()
    split = SplitSpec.from_json(config.data.split_file)
    all_cases = resolve_cases(split, args.subset)
    by_domain = group_cases_by_domain(all_cases)

    if args.domains:
        want = set(args.domains)
        by_domain = {k: v for k, v in by_domain.items() if k in want}

    set_seed(config.system.seed, deterministic=config.system.deterministic)
    device = resolve_device(config.system.device)
    feature_mask = build_feature_mask(
        enabled_node_features=config.data.enabled_node_features,
        enabled_global_features=config.data.enabled_global_features,
    )

    wss_target_names = resolve_wss_runtime_names(
        config.data.wss_target_frame,
        config.model.wss_dim,
        config.model.wss_output_mode,
        config.model.wss_metric_dim,
    )
    eff_wss_dim = resolve_wss_effective_dim(
        config.model.wss_dim,
        config.model.wss_output_mode,
        config.model.wss_metric_dim,
    )
    model = build_field_model_from_config(config).to(device)

    ckpt = Path(args.checkpoint).resolve()
    loss_weights = torch.tensor(config.optim.target_weights, dtype=torch.float32)
    wss_weights = (
        torch.tensor(config.optim.wss_weights, dtype=torch.float32)
        if eff_wss_dim > 0
        else None
    )

    global_metrics = None
    domain_results: Dict[str, Dict] = {}

    # 全局（与 eval_field 一致）
    full_ds = FieldGraphDataset(
        root=config.data.data_root,
        case_names=all_cases,
        graphs_subdir=config.data.graphs_subdir,
        augment=False,
        preload=config.data.preload,
        feature_mask=feature_mask,
        wss_target_frame=config.data.wss_target_frame,
        wss_domain_norm=config.data.wss_domain_norm,
        wss_domain_norm_stats=config.data.wss_domain_norm_stats,
    )
    full_loader = build_dataloader(
        full_ds,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
    )
    global_metrics = evaluate_checkpoint(
        model=model,
        loader=full_loader,
        device=device,
        loss_weights=loss_weights,
        physics_config=config.physics,
        interior_loss_boost=config.optim.interior_loss_boost,
        wss_loss_weight=config.optim.wss_loss_weight,
        wss_weights=wss_weights,
        wss_loss_type=config.optim.wss_loss_type,
        wss_huber_beta=config.optim.wss_huber_beta,
        checkpoint_path=ckpt,
        wss_target_names=wss_target_names or None,
        wss_target_frame=config.data.wss_target_frame,
    )

    for domain, cases in by_domain.items():
        ds = FieldGraphDataset(
            root=config.data.data_root,
            case_names=cases,
            graphs_subdir=config.data.graphs_subdir,
            augment=False,
            preload=config.data.preload,
            feature_mask=feature_mask,
            wss_target_frame=config.data.wss_target_frame,
            wss_domain_norm=config.data.wss_domain_norm,
            wss_domain_norm_stats=config.data.wss_domain_norm_stats,
        )
        if len(ds) == 0:
            domain_results[domain] = {"num_graphs": 0, "num_cases": len(cases), "metrics": {}}
            continue
        loader = build_dataloader(
            ds,
            batch_size=config.data.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
            pin_memory=config.data.pin_memory,
        )
        metrics = evaluate_checkpoint(
            model=model,
            loader=loader,
            device=device,
            loss_weights=loss_weights,
            physics_config=config.physics,
            interior_loss_boost=config.optim.interior_loss_boost,
            wss_loss_weight=config.optim.wss_loss_weight,
            wss_weights=wss_weights,
            wss_loss_type=config.optim.wss_loss_type,
            wss_huber_beta=config.optim.wss_huber_beta,
            checkpoint_path=ckpt,
            wss_target_names=wss_target_names or None,
            wss_target_frame=config.data.wss_target_frame,
        )
        domain_results[domain] = {
            "num_cases": len(cases),
            "num_graphs": len(ds),
            "metrics": metrics,
        }
        r2_p = metrics.get("r2_p", metrics.get("r2", float("nan")))
        wss_r2 = metrics.get("wss_r2_wss", float("nan"))
        print(f"[{domain}] cases={len(cases)} graphs={len(ds)}  r2_p={r2_p:.4f}  wss_r2_wss={wss_r2:.4f}")

    out_dir = (
        Path(args.output).resolve()
        if args.output
        else ckpt.parent / f"eval_by_domain_{args.subset}"
    )
    ensure_dir(out_dir)
    report = {
        "subset": args.subset,
        "checkpoint": str(ckpt),
        "split_file": config.data.split_file,
        "global": {"num_cases": len(all_cases), "num_graphs": len(full_ds), "metrics": global_metrics},
        "by_domain": domain_results,
    }
    dump_json(report, out_dir / "metrics_by_domain.json")
    print(f"\n已保存: {out_dir / 'metrics_by_domain.json'}")


if __name__ == "__main__":
    main()
