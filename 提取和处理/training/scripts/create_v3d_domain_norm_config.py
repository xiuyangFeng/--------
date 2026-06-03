#!/usr/bin/env python3
"""Create the TODO-19 V3D per-domain WSS normalization config.

The script computes train-only WSS statistics from existing graph assets, grouped
by the first path segment of each case name (AAA / AG / ILO), then writes a
single-run config that applies those stats on the fly in FieldGraphDataset.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import torch

from pipeline.config import NODE_FEATURE_NAMES
from pipeline.dataset import load_graph_data
from training.core.utils import dump_json


def _domain(case_name: str) -> str:
    return case_name.split("/", 1)[0]


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_domain_wss_stats(data_root: Path, split_file: Path, graphs_subdir: str) -> Dict[str, Dict[str, list]]:
    split = _load_json(split_file)
    train_cases = split["train_cases"]
    is_wall_idx = NODE_FEATURE_NAMES.index("is_wall")
    accum: Dict[str, Dict[str, torch.Tensor | int]] = {}

    for case_name in train_cases:
        domain = _domain(case_name)
        graph_dir = data_root / case_name / graphs_subdir
        if not graph_dir.is_dir():
            raise FileNotFoundError(f"missing graph dir: {graph_dir}")
        for graph_path in sorted(graph_dir.glob("*.pt")):
            graph = load_graph_data(graph_path)
            if not hasattr(graph, "y_wss") or graph.y_wss is None:
                raise ValueError(f"missing y_wss: {graph_path}")
            is_wall = graph.x[:, is_wall_idx].bool()
            y = graph.y_wss[is_wall].to(torch.float64)
            finite = torch.isfinite(y).all(dim=1)
            y = y[finite]
            if y.numel() == 0:
                continue
            if domain not in accum:
                accum[domain] = {
                    "sum": torch.zeros(y.shape[1], dtype=torch.float64),
                    "sumsq": torch.zeros(y.shape[1], dtype=torch.float64),
                    "count": 0,
                }
            accum[domain]["sum"] += y.sum(dim=0)
            accum[domain]["sumsq"] += (y * y).sum(dim=0)
            accum[domain]["count"] += int(y.shape[0])

    stats: Dict[str, Dict[str, list]] = {}
    for domain, values in sorted(accum.items()):
        count = int(values["count"])
        if count < 2:
            raise ValueError(f"not enough wall WSS samples for domain {domain}: {count}")
        mean = values["sum"] / count
        var = values["sumsq"] / count - mean * mean
        std = torch.sqrt(torch.clamp(var, min=1e-12))
        stats[domain] = {
            "mean": [float(v) for v in mean.tolist()],
            "std": [float(v) for v in std.tolist()],
        }
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Create V3D TODO-19 DomainNorm config")
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("training/configs/field/generated/v3_data_new_tri_domain/V3D-Probe-WSS-01_seed1.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("training/configs/field/generated/v3_data_new_tri_domain/V3D-Probe-WSS-DomainNorm_seed1.json"),
    )
    args = parser.parse_args()

    config = _load_json(args.base_config)
    data_root = Path(config["data"]["data_root"])
    split_file = Path(config["data"]["split_file"])
    graphs_subdir = config["data"].get("graphs_subdir", "processed/graphs")

    stats = compute_domain_wss_stats(data_root, split_file, graphs_subdir)

    config["run"]["experiment_name"] = "field_v3d_pointnext_localpool_probe_wss_domainnorm_wall13000_near2000"
    config["data"]["wss_domain_norm"] = "per_domain"
    config["data"]["wss_domain_norm_stats"] = stats
    config["meta"]["exp_id"] = "V3D-Probe-WSS-DomainNorm"
    config["meta"]["question"] = (
        "TODO-19 C2a: V3D WSS-only probe with train-only AAA/AG/ILO per-domain WSS z-score targets."
    )
    config["meta"]["ablation_axis"] = "v3d_wss_domain_norm"
    config["meta"]["tags"] = sorted(set(config["meta"].get("tags", []) + ["todo19", "domain-norm"]))
    config["meta"]["notes"] = (
        "单变量相对 V3D-Probe-WSS-01：仅将 y_wss 按 train split 内 AAA/AG/ILO 分域 mean/std "
        "在线标准化；模型、split、loss、图资产不变。Go: 全局/分域 WSS 明确改善，尤其 ILO 向 AG 靠近。"
    )
    config["meta"]["generated_from"] = str(args.base_config)

    dump_json(config, args.output)
    print(f"wrote {args.output}")
    for domain, domain_stats in stats.items():
        print(f"{domain}: mean={domain_stats['mean']} std={domain_stats['std']}")


if __name__ == "__main__":
    main()
