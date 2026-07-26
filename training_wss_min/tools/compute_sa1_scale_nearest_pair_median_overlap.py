#!/usr/bin/env python3
"""Compute SA1 nearest-centre-pair median overlap for sa1_scale configs.

口径对齐 07-19 Q2V 重叠审计：
- 对每个 SA1 center 取空间最近的另一个 center，去重得到 nearest-pair 集合；
- overlap = |shared unique sources| / min(|group_a|, |group_b|)；
- 报告每例 median，再报告跨例 median-of-medians / mean-of-medians，以及 pooled median。
同时保留既有审计的 max_pair_overlap（全共享 pair 最大值）便于对照。
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch_geometric.nn import fps

from training_wss_min import dataset as D
from training_wss_min.baseline_models import PointNetSetAbstraction, _group
from training_wss_min.config import ExpConfig
from training_wss_min.surface import stable_seed


def support_points(cfg: ExpConfig) -> int:
    return int(cfg.data.support_n_points or cfg.data.wall_n_points)


def is_proportional(cfg: ExpConfig) -> bool:
    return not cfg.model.sa_center_counts


def nearest_pairs(center_xyz: np.ndarray) -> list[tuple[int, int]]:
    _, neighbour = cKDTree(center_xyz).query(center_xyz, k=2)
    return sorted({tuple(sorted((int(row), int(index))))
                   for row, index in enumerate(neighbour[:, 1])})


def group_members(row: np.ndarray, col: np.ndarray, n_centers: int) -> list[set[int]]:
    members = [set() for _ in range(n_centers)]
    for center, source in zip(row.tolist(), col.tolist()):
        members[int(center)].add(int(source))
    return members


def max_pair_overlap(row: np.ndarray, col: np.ndarray, sizes: np.ndarray) -> float:
    memberships: dict[int, list[int]] = defaultdict(list)
    for center, source in zip(row.tolist(), col.tolist()):
        memberships[int(source)].append(int(center))
    shared: dict[tuple[int, int], int] = defaultdict(int)
    for centers in memberships.values():
        unique = sorted(set(centers))
        for i, left in enumerate(unique):
            for right in unique[i + 1:]:
                shared[(left, right)] += 1
    max_ratio = 0.0
    for (left, right), count in shared.items():
        denom = min(int(sizes[left]), int(sizes[right]))
        if denom:
            max_ratio = max(max_ratio, count / denom)
    return float(max_ratio)


def nearest_pair_rates(center_xyz: np.ndarray, members: list[set[int]]) -> np.ndarray:
    rates = []
    for a, b in nearest_pairs(center_xyz):
        first, second = members[a], members[b]
        denom = min(len(first), len(second))
        rates.append((len(first & second) / denom) if denom else 0.0)
    return np.asarray(rates, dtype=float)


def audit_config(config_path: Path, max_cases: int | None) -> dict:
    cfg = ExpConfig.from_json(config_path)
    stats = D.load_wss_stats(cfg.data.wss_stats_path)
    cases = D.load_partition(
        cfg.data.split_path, "train", stats, strict=True, target=cfg.data.target,
        target_normalization=cfg.data.target_normalization, data_root=cfg.data.data_root,
        required_frame_version=cfg.data.required_frame_version,
    )
    if max_cases is not None and max_cases < len(cases):
        by_size = sorted(range(len(cases)), key=lambda i: len(cases[i]["pos"]))
        keep = {by_size[0], by_size[-1]}
        for i in range(len(cases)):
            if len(keep) >= max_cases:
                break
            keep.add(i)
        cases = [cases[i] for i in sorted(keep)]

    mode = cfg.model.sa_grouping[0] if cfg.model.sa_grouping else "ball"
    proportional = is_proportional(cfg)
    per_case = []
    pooled_rates: list[float] = []

    for case_index, case in enumerate(cases):
        case["geom_seed"] = cfg.train.seed + 7919 * case_index
        seed = stable_seed(cfg.train.seed, 0, case["unit_id"], "support")
        idx = D.sample_indices(
            case, cfg.data, seed, epoch=0, case_index=case_index,
            run_seed=cfg.train.seed, stream="support",
            n_points=support_points(cfg),
            sampling=cfg.data.support_sampling or cfg.data.sampling,
        )
        pos = torch.as_tensor(case["pos"][idx], dtype=torch.float32, device="cuda")
        batch = torch.zeros(len(pos), dtype=torch.long, device="cuda")
        torch.manual_seed(stable_seed(cfg.train.seed, case["unit_id"], "sa1_audit") % (2**63 - 1))
        if proportional:
            centers = fps(pos, batch, ratio=float(cfg.model.sa_ratios[0]), random_start=False)
        else:
            sa = PointNetSetAbstraction(
                1, 1, cfg.model.sa_ratios[0], cfg.model.sa_radius[0], cfg.model.sa_nsample[0],
                center_count=cfg.model.sa_center_counts[0],
                center_sampling=cfg.model.sa_center_sampling,
                stage=0, grouping=mode,
                adaptive_candidate_centers=cfg.model.sa_adaptive_candidate_centers,
                overlap_cap=cfg.model.sa_overlap_cap,
            ).to("cuda").train()
            centers = sa._fixed_indices(
                pos, batch, unit_ids=[case["unit_id"]], epoch=0,
                global_seed=cfg.train.seed, evaluation=False,
            )
            del sa
        row, col = _group(
            pos, batch, pos[centers], batch[centers], cfg.model.sa_radius[0],
            cfg.model.sa_nsample[0], grouping=mode,
            adaptive_candidate_centers=cfg.model.sa_adaptive_candidate_centers,
            overlap_cap=cfg.model.sa_overlap_cap,
        )
        row_np = row.detach().cpu().numpy()
        col_np = col.detach().cpu().numpy()
        center_xyz = pos[centers].detach().cpu().numpy()
        sizes = np.bincount(row_np, minlength=len(centers))
        members = group_members(row_np, col_np, len(centers))
        rates = nearest_pair_rates(center_xyz, members)
        record = {
            "unit_id": case["unit_id"],
            "support_points": int(len(pos)),
            "centers": int(len(centers)),
            "nearest_pair_count": int(len(rates)),
            "coverage": float(np.unique(col_np).size / len(pos)),
            "median_overlap": float(np.median(rates)),
            "mean_overlap": float(np.mean(rates)),
            "p10_overlap": float(np.quantile(rates, 0.10)),
            "p90_overlap": float(np.quantile(rates, 0.90)),
            "max_nearest_pair_overlap": float(np.max(rates)),
            "max_pair_overlap": max_pair_overlap(row_np, col_np, sizes),
            "fraction_le_1_3": float(np.mean(rates <= 1 / 3)),
            "fraction_exact_100": float(np.mean(rates == 1.0)),
            "group_size_min": int(sizes.min()),
            "group_size_median": float(np.median(sizes)),
            "group_size_max": int(sizes.max()),
        }
        per_case.append(record)
        pooled_rates.extend(rates.tolist())
        del pos, batch, row, col
        torch.cuda.empty_cache()

    case_medians = np.asarray([r["median_overlap"] for r in per_case], dtype=float)
    pooled = np.asarray(pooled_rates, dtype=float)
    return {
        "config": str(config_path.resolve()),
        "label": config_path.stem,
        "grouping": mode,
        "center_mode": "proportional_ratio" if proportional else "fixed_counts",
        "sa_center_counts": list(cfg.model.sa_center_counts),
        "sa_nsample0": int(cfg.model.sa_nsample[0]),
        "cases": len(per_case),
        "coverage_min": min(r["coverage"] for r in per_case),
        "median_of_case_medians": float(np.median(case_medians)),
        "mean_of_case_medians": float(np.mean(case_medians)),
        "pooled_median_overlap": float(np.median(pooled)),
        "pooled_p10_overlap": float(np.quantile(pooled, 0.10)),
        "pooled_p90_overlap": float(np.quantile(pooled, 0.90)),
        "pooled_fraction_le_1_3": float(np.mean(pooled <= 1 / 3)),
        "pooled_fraction_exact_100": float(np.mean(pooled == 1.0)),
        "max_pair_overlap": max(r["max_pair_overlap"] for r in per_case),
        "per_case": per_case,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-cases", type=int, default=None,
                        help="抽样上限；默认全量 train（固定5k合同）")
    parser.add_argument("--max-cases-large", type=int, default=4,
                        help="全点/比例制合同抽样上限")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("requires CUDA")

    results = []
    for path in args.configs:
        cfg = ExpConfig.from_json(path)
        large = (support_points(cfg) <= 0) or is_proportional(cfg) or (
            (cfg.data.support_sampling or cfg.data.sampling) in {"random", "area_random"}
            and support_points(cfg) > 10000
        )
        # 全点：support_n_points=0/None 且 wall 全量；本矩阵全点合同 wall_n_points=0
        is_allpts = int(cfg.data.wall_n_points or 0) == 0 and int(cfg.data.support_n_points or 0) == 0
        max_cases = args.max_cases_large if (is_allpts or is_proportional(cfg)) else args.max_cases
        print(f"[audit] {path.name} max_cases={max_cases}", flush=True)
        results.append(audit_config(path, max_cases))

    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "definition": (
            "nearest-centre-pair overlap = shared unique sources / min(group sizes); "
            "median matches 2026-07-19 Q2V SA overlap audit"
        ),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {args.output}", flush=True)
    print(f"{'label':40} {'k':>4} {'centers':>12} {'med(case)':>10} {'pooled':>10} {'max_all':>10}")
    for r in results:
        centers = r["sa_center_counts"][:1] if r["sa_center_counts"] else ["prop"]
        print(
            f"{r['label'][:40]:40} {r['sa_nsample0']:>4} {str(centers[0]):>12} "
            f"{r['median_of_case_medians']:>10.4f} {r['pooled_median_overlap']:>10.4f} "
            f"{r['max_pair_overlap']:>10.4f}"
        )


if __name__ == "__main__":
    main()
