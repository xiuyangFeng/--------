from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np


@dataclass
class SplitSpec:
    split_version: str
    # 始终按患者级划分，避免同一患者不同时间帧泄漏到不同子集。
    train_cases: List[str]
    val_cases: List[str]
    test_cases: List[str]
    source: Optional[str] = None
    notes: str = ""

    @classmethod
    def from_json(cls, path: str | Path) -> "SplitSpec":
        # split 是整个实验体系的根输入之一，因此这里保持极简、显式、不可猜。
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return cls(
            split_version=raw["split_version"],
            train_cases=raw["train_cases"],
            val_cases=raw["val_cases"],
            test_cases=raw["test_cases"],
            source=raw.get("source"),
            notes=raw.get("notes", ""),
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "split_version": self.split_version,
            "train_cases": self.train_cases,
            "val_cases": self.val_cases,
            "test_cases": self.test_cases,
            "source": self.source,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Patient-level k-fold cross-validation
# ---------------------------------------------------------------------------


def _validate_cv_inputs(case_names: Sequence[str], val_ratio: float) -> List[str]:
    names = list(case_names)
    if len(names) != len(set(names)):
        raise ValueError("case_names 存在重复项，无法生成患者级划分")
    if len(names) < 3:
        raise ValueError("至少需要 3 个病例，才能同时构造 train / val / test")
    if not (0.0 < val_ratio < 1.0):
        raise ValueError("val_ratio 必须在 (0, 1) 范围内")
    return names


def _split_train_val(remaining: Sequence[str], val_ratio: float) -> tuple[List[str], List[str]]:
    if len(remaining) < 2:
        raise ValueError("验证集划分后必须至少保留 1 个训练病例")
    val_size = max(1, int(round(len(remaining) * val_ratio)))
    val_size = min(val_size, len(remaining) - 1)
    val = list(remaining[:val_size])
    train = list(remaining[val_size:])
    if not train:
        raise ValueError("当前 val_ratio 导致训练集为空，请减小 val_ratio 或增加病例数")
    return train, val


def generate_kfold_splits(
    case_names: Sequence[str],
    k: int = 5,
    seed: int = 42,
    val_ratio: float = 0.15,
    prefix: str = "kfold",
) -> List[SplitSpec]:
    """Generate *k* patient-level splits for cross-validation.

    Each fold uses a disjoint set of patients as the test set. From the
    remaining patients a ``val_ratio`` fraction is held out for validation.

    Returns a list of ``SplitSpec`` objects, one per fold.
    """
    rng = np.random.default_rng(seed)
    names = _validate_cv_inputs(case_names, val_ratio)
    if k < 2:
        raise ValueError("k-fold 交叉验证要求 k >= 2")
    if k > len(names):
        raise ValueError("k-fold 交叉验证要求 k <= 病例数")
    rng.shuffle(names)
    test_folds = [list(fold) for fold in np.array_split(np.array(names, dtype=object), k)]

    folds: List[SplitSpec] = []
    for i, test_fold in enumerate(test_folds):
        test = list(test_fold)
        if not test:
            raise ValueError("生成了空测试 fold，请检查 k 与病例数设置")
        remaining = [c for c in names if c not in test]
        train, val = _split_train_val(remaining, val_ratio)

        folds.append(
            SplitSpec(
                split_version=f"{prefix}_fold{i}",
                train_cases=train,
                val_cases=val,
                test_cases=test,
                source="auto_kfold",
                notes=f"k={k}, seed={seed}, fold={i}/{k}",
            )
        )
    return folds


def generate_loocv_splits(
    case_names: Sequence[str],
    seed: int = 42,
    val_ratio: float = 0.15,
    prefix: str = "loocv",
) -> List[SplitSpec]:
    """Generate leave-one-patient-out splits."""
    names = _validate_cv_inputs(case_names, val_ratio)
    rng = np.random.default_rng(seed)
    rng.shuffle(names)

    folds: List[SplitSpec] = []
    for i, test_case in enumerate(names):
        remaining = [c for c in names if c != test_case]
        rng_fold = np.random.default_rng(seed + i)
        rng_fold.shuffle(remaining)
        train, val = _split_train_val(remaining, val_ratio)

        folds.append(
            SplitSpec(
                split_version=f"{prefix}_{i}",
                train_cases=train,
                val_cases=val,
                test_cases=[test_case],
                source="auto_loocv",
                notes=f"leave-out={test_case}, seed={seed}",
            )
        )
    return folds


def save_splits(folds: List[SplitSpec], output_dir: str | Path) -> List[Path]:
    """Save a list of SplitSpec objects as individual JSON files."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    for fold in folds:
        p = out / f"{fold.split_version}.json"
        with open(p, "w", encoding="utf-8") as f:
            json.dump(fold.to_dict(), f, indent=2, ensure_ascii=False)
        paths.append(p)
    return paths
