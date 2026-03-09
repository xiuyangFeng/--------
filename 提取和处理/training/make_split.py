from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List


def load_cases(path: str | Path) -> List[str]:
    path = Path(path)
    if path.suffix.lower() == ".json":
        # 兼容两种最常见输入：
        # 1. ["case_a", "case_b", ...]
        # 2. {"cases": ["case_a", "case_b", ...]}
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict) and "cases" in raw:
            return [str(x) for x in raw["cases"]]
        if isinstance(raw, list):
            return [str(x) for x in raw]
        raise ValueError("JSON 文件必须是病例列表，或包含 'cases' 字段。")

    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="根据病例名单生成患者级 split 文件")
    parser.add_argument("--cases-file", required=True, help="病例名单 txt/json 文件")
    parser.add_argument("--output", required=True, help="输出 split.json 路径")
    parser.add_argument("--split-version", required=True, help="划分版本号，如 split_v1")
    parser.add_argument("--source", default="", help="数据源标识，如 AG/fast")
    parser.add_argument("--seed", type=int, default=1, help="随机种子")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="训练集比例")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="验证集比例")
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="测试集比例，默认与前两者互补",
    )
    args = parser.parse_args()

    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError("train/val/test 比例之和必须为 1.0")

    cases = load_cases(args.cases_file)
    if len(cases) < 3:
        raise ValueError("病例数至少需要 3 个才能生成 train/val/test 划分")

    # 这里先去重再随机打乱，保证同一病例不会因为输入文件重复而影响划分比例。
    random.seed(args.seed)
    cases = sorted(set(cases))
    random.shuffle(cases)

    n_total = len(cases)
    n_train = max(1, int(n_total * args.train_ratio))
    n_val = max(1, int(n_total * args.val_ratio))
    if n_train + n_val >= n_total:
        n_val = max(1, n_total - n_train - 1)
    n_test = n_total - n_train - n_val
    if n_test < 1:
        raise ValueError("测试集病例数不足，请调整比例。")

    split = {
        "split_version": args.split_version,
        "source": args.source,
        "train_cases": cases[:n_train],
        "val_cases": cases[n_train : n_train + n_val],
        "test_cases": cases[n_train + n_val :],
        "notes": "由 training/make_split.py 自动生成。请确认患者级划分满足项目要求。",
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(split, f, ensure_ascii=False, indent=2)

    print(f"已生成 split 文件: {output}")
    print(
        f"train={len(split['train_cases'])}, "
        f"val={len(split['val_cases'])}, "
        f"test={len(split['test_cases'])}"
    )


if __name__ == "__main__":
    main()
