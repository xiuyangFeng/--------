"""训练 / 评估侧 PREPROCESS_DENYLIST 过滤（与 pipeline 共用同一常量）。"""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Set

from pipeline.export_gap_preprocess_queue import PREPROCESS_DENYLIST


def denylist_hit(case_rel: str, data_root: str | Path) -> bool:
    """case 是否在 PREPROCESS_DENYLIST 中（兼容 data_root=data_new/AG 时的 AG/ 前缀键）。"""
    rel = case_rel.replace("\\", "/")
    if rel in PREPROCESS_DENYLIST:
        return True
    root_name = Path(data_root).name
    return f"{root_name}/{rel}" in PREPROCESS_DENYLIST


def filter_case_names(case_names: Sequence[str], data_root: str | Path) -> List[str]:
    """剔除 denylist 病例，保持原顺序。"""
    return [c for c in case_names if not denylist_hit(c, data_root)]


def skipped_case_names(case_names: Sequence[str], data_root: str | Path) -> Set[str]:
    """返回输入中被 denylist 命中的病例集合。"""
    return {c for c in case_names if denylist_hit(c, data_root)}


def resolve_split_subset(split, subset: str, data_root: str | Path) -> List[str]:
    """按 train/val/test 取 case 并剔除 PREPROCESS_DENYLIST。"""
    mapping = {
        "train": split.train_cases,
        "val": split.val_cases,
        "test": split.test_cases,
    }
    if subset not in mapping:
        raise ValueError(f"未知 subset: {subset}")
    return filter_case_names(mapping[subset], data_root)
