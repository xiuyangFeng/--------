#!/usr/bin/env python3
"""Build one case's fixed/multistart FPS5000 cache for the Q1V/Q2V matrix."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from training_wss_min import config as C
from training_wss_min.dataset import (
    _coords_sha256, build_fps_pool, farthest_point_sample, load_case, load_split_cases,
)
from training_wss_min.surface import stable_seed


REPO = Path(__file__).resolve().parents[2]
DEFAULT_SPLIT = REPO / "training/splits/split_AG_AAA_wss_min_v4_stratified_seed1234.json"
DEFAULT_STATS = REPO / "data_wss_min/fold_stats/v4/AG_AAA_v4_stratified_train106_global_stats.json"
DEFAULT_CACHE = REPO / "training_wss_min/preflight/fps_support_cache_v4"


def indexed_cases(split: Path) -> list[tuple[str, int, str, str]]:
    rows: list[tuple[str, int, str, str]] = []
    for partition in ("train", "val", "test"):
        for index, (cohort, case) in enumerate(load_split_cases(split, partition)):
            rows.append((partition, index, cohort, case))
    return rows


def build_one(*, split: Path, stats_path: Path, data_root: Path, cache_dir: Path,
              row_index: int, k: int, pool_size: int, seed: int) -> Path:
    rows = indexed_cases(split)
    if not (0 <= row_index < len(rows)):
        raise IndexError(f"row-index {row_index} outside 0..{len(rows)-1}")
    partition, partition_index, cohort, case_name = rows[row_index]
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    case = load_case(
        cohort, case_name, stats, data_root=data_root,
        required_frame_version="stl_landmarks_v4",
    )
    pos = case["pos"]
    k_eff = min(len(pos), int(k))
    unit_id = case["unit_id"]
    fixed_seed = stable_seed(seed, 0, unit_id, "support")
    geom_seed = seed + 7919 * partition_index
    fixed = farthest_point_sample(pos, k_eff, int(fixed_seed))
    pool = build_fps_pool(pos, k_eff, int(pool_size), int(geom_seed))
    output = cache_dir / unit_id / f"fps_k{k_eff}_pool{int(pool_size)}.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    with temp.open("wb") as handle:
        np.savez_compressed(
            handle, fixed=fixed.astype(np.int64), pool=pool.astype(np.int64),
            n_total=np.int64(len(pos)), k=np.int64(k_eff), pool_size=np.int64(pool_size),
            coords_sha256=np.asarray(_coords_sha256(pos)), partition=np.asarray(partition),
            partition_index=np.int64(partition_index), unit_id=np.asarray(unit_id),
            fixed_seed=np.asarray(str(int(fixed_seed))), geom_seed=np.asarray(str(int(geom_seed))),
        )
    temp.replace(output)
    print(json.dumps({
        "status": "built", "row_index": row_index, "partition": partition,
        "partition_index": partition_index, "unit_id": unit_id,
        "n_total": len(pos), "k": k_eff, "pool_size": pool_size,
        "output": str(output),
    }, ensure_ascii=False))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--stats", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--data-root", type=Path, default=REPO / "data_wss_min")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--row-index", type=int, required=True)
    parser.add_argument("--k", type=int, default=5000)
    parser.add_argument("--pool-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    build_one(
        split=args.split, stats_path=args.stats, data_root=args.data_root,
        cache_dir=args.cache_dir, row_index=args.row_index, k=args.k,
        pool_size=args.pool_size, seed=args.seed,
    )


if __name__ == "__main__":
    main()
