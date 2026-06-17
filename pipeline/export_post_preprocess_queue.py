#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
导出 `data_new` 下 AAA + AG + ILO「全流程（步骤 1–5）尚未齐套」的病例相对路径，供集群从步骤 2 跑到 5。

排除 `export_gap_preprocess_queue.PREPROCESS_DENYLIST`（预处理/全流程导出/实验共用）；
排除已完成病例（merged/features/coord_normalized/normalized/graphs 数量均等于 matched_frame_count）。

默认包含输入尚不足以跑步骤 2 的病例（仍会入队，步骤 2 将失败），可用 --only-feature-ready 去掉。
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Set

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from pipeline.audit_inputs import discover_case_dirs
    from pipeline.config import (
        COORD_NORMALIZED_DIR,
        DATA_ROOT,
        FEATURES_DIR,
        GRAPHS_DIR,
        MERGED_DIR,
        NORMALIZED_DIR,
    )
    from pipeline.export_gap_preprocess_queue import PREPROCESS_DENYLIST
    from pipeline.validation import inspect_case_inputs
else:
    from .audit_inputs import discover_case_dirs
    from .config import (
        COORD_NORMALIZED_DIR,
        DATA_ROOT,
        FEATURES_DIR,
        GRAPHS_DIR,
        MERGED_DIR,
        NORMALIZED_DIR,
    )
    from .export_gap_preprocess_queue import PREPROCESS_DENYLIST
    from .validation import inspect_case_inputs


def _count_merged(case: Path) -> int:
    d = case / MERGED_DIR
    return len(list(d.glob("merged-*.csv"))) if d.is_dir() else 0


def _count_features(case: Path) -> int:
    d = case / FEATURES_DIR
    return len(list(d.glob("result_features_*.csv"))) if d.is_dir() else 0


def _count_csv(dir_path: Path) -> int:
    if not dir_path.is_dir():
        return 0
    return sum(1 for p in dir_path.iterdir() if p.suffix == ".csv")


def _count_graphs(case: Path) -> int:
    d = case / GRAPHS_DIR
    return len(list(d.glob("*.pt"))) if d.is_dir() else 0


def is_pipeline_fully_complete(case: Path, n_exp: int) -> bool:
    if n_exp <= 0:
        return False
    m = _count_merged(case)
    f = _count_features(case)
    c = _count_csv(case / COORD_NORMALIZED_DIR)
    n = _count_csv(case / NORMALIZED_DIR)
    g = _count_graphs(case)
    return m == n_exp == f == c == n == g


def main() -> None:
    parser = argparse.ArgumentParser(
        description="导出 AAA+AG+ILO 全流程未完成病例列表（相对 data_root）"
    )
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--only-feature-ready",
        action="store_true",
        help="仅保留 inspect_case_inputs 中 has_feature_inputs 的病例（可跑 extract_features）",
    )
    args = parser.parse_args()

    root = args.data_root.resolve()
    deny: Set[str] = set(PREPROCESS_DENYLIST)

    lines: List[str] = []
    for d in discover_case_dirs(root, ["AAA", "AG", "ILO"], []):
        rel = d.resolve().relative_to(root).as_posix()
        if rel in deny:
            continue
        rep = inspect_case_inputs(d)
        n_exp = int(rep["matched_frame_count"] or 0)
        if is_pipeline_fully_complete(d, n_exp):
            continue
        if args.only_feature_ready and not rep["has_feature_inputs"]:
            continue
        lines.append(rel)

    lines = sorted(set(lines))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    print(f"写入 {len(lines)} 条 -> {args.output}")


if __name__ == "__main__":
    main()
