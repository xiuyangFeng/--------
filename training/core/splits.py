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
        # 读取 split JSON 文件。
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        # 将 JSON 字段映射成强类型 SplitSpec。
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
    # 先转成 list，便于后续重复使用与打乱顺序。
    names = list(case_names)
    # 患者名不能重复，否则交叉验证会发生数据泄漏。
    if len(names) != len(set(names)):
        raise ValueError("case_names 存在重复项，无法生成患者级划分")
    # 至少要能分出 train / val / test 三部分。
    if len(names) < 3:
        raise ValueError("至少需要 3 个病例，才能同时构造 train / val / test")
    # 验证集比例必须是严格的 0 到 1 之间。
    if not (0.0 < val_ratio < 1.0):
        raise ValueError("val_ratio 必须在 (0, 1) 范围内")
    # 校验通过后返回标准 list。
    return names


def _split_train_val(remaining: Sequence[str], val_ratio: float) -> tuple[List[str], List[str]]:
    # 至少要留下 1 个训练病例和 1 个验证病例。
    if len(remaining) < 2:
        raise ValueError("验证集划分后必须至少保留 1 个训练病例")
    # 先按比例计算验证集大小。
    val_size = max(1, int(round(len(remaining) * val_ratio)))
    # 再限制验证集不能把全部样本拿走。
    val_size = min(val_size, len(remaining) - 1)
    # 这里约定剩余序列前半段给验证集。
    val = list(remaining[:val_size])
    # 后半段给训练集。
    train = list(remaining[val_size:])
    # 防御式检查：训练集不能为空。
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
    # 用独立随机数生成器，避免污染全局 NumPy 随机状态。
    rng = np.random.default_rng(seed)
    # 先做输入合法性检查。
    names = _validate_cv_inputs(case_names, val_ratio)
    # k-fold 至少需要 2 折。
    if k < 2:
        raise ValueError("k-fold 交叉验证要求 k >= 2")
    # 折数不能多于病例数。
    if k > len(names):
        raise ValueError("k-fold 交叉验证要求 k <= 病例数")
    # 先整体打乱病例顺序。
    rng.shuffle(names)
    # 把病例切成 k 个互斥测试 fold。
    test_folds = [list(fold) for fold in np.array_split(np.array(names, dtype=object), k)]

    folds: List[SplitSpec] = []
    # 逐个 fold 生成一份 train / val / test 划分。
    for i, test_fold in enumerate(test_folds):
        # 当前 fold 的测试集就是这一折的病例。
        test = list(test_fold)
        # 不允许出现空测试折。
        if not test:
            raise ValueError("生成了空测试 fold，请检查 k 与病例数设置")
        # 剩余病例用于划分 train / val。
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
    # 先做基础输入校验。
    names = _validate_cv_inputs(case_names, val_ratio)
    # 全局随机打乱 leave-one-out 的测试顺序。
    rng = np.random.default_rng(seed)
    rng.shuffle(names)

    folds: List[SplitSpec] = []
    # 每次拿 1 个病例做测试。
    for i, test_case in enumerate(names):
        # 当前测试病例之外的其余病例进入 train / val 划分池。
        remaining = [c for c in names if c != test_case]
        # 为每个 fold 单独构造随机数生成器，避免所有 fold 验证集都一样。
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
    # 规范化输出目录路径。
    out = Path(output_dir)
    # 递归创建输出目录。
    out.mkdir(parents=True, exist_ok=True)
    # 收集每个写出的 JSON 路径。
    paths: List[Path] = []
    # 逐个 fold 落盘成单独的 JSON 文件。
    for fold in folds:
        p = out / f"{fold.split_version}.json"
        with open(p, "w", encoding="utf-8") as f:
            json.dump(fold.to_dict(), f, indent=2, ensure_ascii=False)
        paths.append(p)
    return paths
