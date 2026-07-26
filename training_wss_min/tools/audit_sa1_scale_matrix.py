#!/usr/bin/env python3
"""Audit SA1 coverage/overlap for the sa1_scale matrix (fixed or ratio centers).

与 audit_sa1_grouping_matrix.py 同源，两点扩展：
1. 兼容 ``sa_center_counts=[]`` 的比例制 center（fps ratio 路径，eval 口径
   ``random_start=False`` 确定性选取）。
2. 全点/比例制合同的 overlap 统计是 O(edges×memberships) 纯 Python 循环，
   按 ``--max-cases-large`` 抽样审计（默认 4 例）；固定 5k/10k 合同仍可全量。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch_geometric.nn import fps

from training_wss_min import dataset as D
from training_wss_min.baseline_models import PointNetSetAbstraction, _group
from training_wss_min.config import ExpConfig
from training_wss_min.surface import stable_seed


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def overlap_metrics(row: np.ndarray, col: np.ndarray, group_sizes: np.ndarray) -> tuple[float, int]:
    memberships: dict[int, list[int]] = defaultdict(list)
    for center, source in zip(row.tolist(), col.tolist()):
        memberships[source].append(center)
    shared: dict[tuple[int, int], int] = defaultdict(int)
    max_memberships = 0
    for centers in memberships.values():
        unique = sorted(set(centers))
        max_memberships = max(max_memberships, len(unique))
        for i, left in enumerate(unique):
            for right in unique[i + 1:]:
                shared[(left, right)] += 1
    max_ratio = 0.0
    for (left, right), count in shared.items():
        denom = min(int(group_sizes[left]), int(group_sizes[right]))
        if denom:
            max_ratio = max(max_ratio, count / denom)
    return max_ratio, max_memberships


def support_points(cfg: ExpConfig) -> int:
    return int(cfg.data.support_n_points or cfg.data.wall_n_points)


def is_proportional(cfg: ExpConfig) -> bool:
    return not cfg.model.sa_center_counts


def contract_key(cfg: ExpConfig) -> tuple:
    counts = cfg.model.sa_center_counts
    return (
        cfg.data.support_sampling or cfg.data.sampling,
        support_points(cfg),
        cfg.model.sa_grouping[0] if cfg.model.sa_grouping else "ball",
        int(cfg.model.sa_nsample[0]),
        int(counts[0]) if counts else -1,
        float(cfg.model.sa_ratios[0]),
        float(cfg.model.sa_radius[0]),
        int(cfg.model.sa_adaptive_candidate_centers),
        float(cfg.model.sa_overlap_cap),
    )


def audit_contract(config_path: Path, max_cases: int | None) -> dict:
    cfg = ExpConfig.from_json(config_path)
    stats = D.load_wss_stats(cfg.data.wss_stats_path)
    cases = D.load_partition(
        cfg.data.split_path, "train", stats, strict=True, target=cfg.data.target,
        target_normalization=cfg.data.target_normalization, data_root=cfg.data.data_root,
        required_frame_version=cfg.data.required_frame_version,
    )
    if max_cases is not None:
        # 抽样时保证覆盖点数极端：取最小/最大例 + 首序补齐
        by_size = sorted(range(len(cases)), key=lambda i: len(cases[i]["pos"]))
        keep = {by_size[0], by_size[-1]}
        for i in range(len(cases)):
            if len(keep) >= max_cases:
                break
            keep.add(i)
        cases = [cases[i] for i in sorted(keep)]
    mode = cfg.model.sa_grouping[0] if cfg.model.sa_grouping else "ball"
    proportional = is_proportional(cfg)
    records = []
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
        sizes = np.bincount(row_np, minlength=len(centers))
        coverage = np.unique(col_np).size / len(pos)
        overlap, max_memberships = overlap_metrics(row_np, col_np, sizes)
        record = {
            "unit_id": case["unit_id"], "support_points": len(pos), "centers": len(centers),
            "edges": len(row_np), "coverage": coverage,
            "group_size_min": int(sizes.min()), "group_size_median": float(np.median(sizes)),
            "group_size_max": int(sizes.max()), "max_pair_overlap": overlap,
            "max_memberships_per_source": max_memberships,
        }
        if mode in {"knn_cover", "adaptive_cover"} and abs(coverage - 1.0) > 1e-12:
            raise RuntimeError(f"{mode} coverage gate failed: {record}")
        if mode == "adaptive_cover" and overlap > cfg.model.sa_overlap_cap + 1e-12:
            raise RuntimeError(f"adaptive overlap gate failed: {record}")
        records.append(record)
        del pos, batch, row, col
    return {
        "config": str(config_path.resolve()), "config_sha256": sha(config_path),
        "contract": list(contract_key(cfg)), "grouping": mode,
        "center_mode": "proportional_ratio" if proportional else "fixed_counts",
        "hard_gate": mode in {"knn_cover", "adaptive_cover"},
        "cases": len(records),
        "coverage_min": min(r["coverage"] for r in records),
        "coverage_median": float(np.median([r["coverage"] for r in records])),
        "max_pair_overlap": max(r["max_pair_overlap"] for r in records),
        "group_size_min": min(r["group_size_min"] for r in records),
        "group_size_max": max(r["group_size_max"] for r in records),
        "per_case": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-cases", type=int, default=None,
                        help="固定小点数合同的抽样上限；默认全量 train")
    parser.add_argument("--max-cases-large", type=int, default=4,
                        help="全点/比例制合同的抽样上限（纯 Python overlap 统计代价高）")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("formal SA1 audit requires CUDA")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = manifest["configs"]
    unique: dict[tuple, Path] = {}
    represented: dict[tuple, list[str]] = defaultdict(list)
    for row in rows:
        path = Path(row["config"])
        cfg = ExpConfig.from_json(path)
        key = contract_key(cfg)
        unique.setdefault(key, path)
        represented[key].append(row["experiment_id"])
    results = []
    for index, (key, path) in enumerate(unique.items(), 1):
        cfg = ExpConfig.from_json(path)
        large = support_points(cfg) <= 0 or is_proportional(cfg)
        limit = args.max_cases_large if large else args.max_cases
        print(f"audit {index}/{len(unique)} {path.stem} (max_cases={limit})", flush=True)
        result = audit_contract(path, limit)
        result["represents"] = represented[key]
        result["sampled_audit"] = limit is not None
        results.append(result)
    payload = {
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "manifest": str(args.manifest.resolve()), "manifest_sha256": sha(args.manifest),
        "contracts": len(results), "matrix_rows": len(rows), "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "contracts": len(results),
                      "rows": len(rows), "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
