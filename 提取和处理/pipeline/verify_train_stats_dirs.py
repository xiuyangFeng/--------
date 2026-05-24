#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
校验 normalize 步骤的 train-only 统计目录是否与 split 的 train_cases 一一对应。

Gate-0：len(stats_dirs) == len(train_cases)，extra=0，missing=0。

示例:
  python -m pipeline.verify_train_stats_dirs \\
    --data-root data_new \\
    --train-split training/splits/split_data_new_v3.json \\
    --sources AAA/ruputer AAA/unruputer AG/fast AG/slow ILO
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from .config import DATA_ROOT, get_case_dirs
    from .normalize import select_train_stats_dirs
except ImportError:
    from config import DATA_ROOT, get_case_dirs
    from pipeline.normalize import select_train_stats_dirs


def load_train_cases(split_path: Path) -> list[str]:
    with open(split_path, encoding="utf-8") as f:
        split = json.load(f)
    train_cases = split.get("train_cases", split.get("train", split.get("cases", [])))
    if not train_cases:
        print(f"❌ split 中未找到 train_cases: {split_path}")
        sys.exit(1)
    return train_cases


def main() -> None:
    parser = argparse.ArgumentParser(description="校验 train-only 统计目录（Gate-0）")
    parser.add_argument("--data-root", type=str, default=None, help="数据根目录")
    parser.add_argument(
        "--train-split",
        type=str,
        required=True,
        help="训练集 split JSON（含 train_cases）",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=None,
        metavar="SOURCE",
        help="数据源子路径（与 normalize 批处理一致）",
    )
    parser.add_argument(
        "--no-strict",
        action="store_true",
        help="仅打印 extra/missing，不因校验失败退出",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root) if args.data_root else DATA_ROOT
    split_path = Path(args.train_split)
    train_cases = load_train_cases(split_path)
    case_dirs = get_case_dirs(data_root, sources=args.sources)

    stats_dirs, extra, missing = select_train_stats_dirs(
        case_dirs,
        data_root,
        train_cases,
        strict=not args.no_strict,
    )

    print(f"data_root:     {data_root.resolve()}")
    print(f"train_split:   {split_path}")
    print(f"case_dirs:     {len(case_dirs)}")
    print(f"train_cases:   {len(train_cases)}")
    print(f"stats_dirs:    {len(stats_dirs)}")
    print(f"extra:         {len(extra)}")
    print(f"missing:       {len(missing)}")

    if extra:
        print("\nextra:")
        for k in sorted(extra):
            print(f"  {k}")
    if missing:
        print("\nmissing:")
        for k in sorted(missing):
            print(f"  {k}")

    passed = (
        len(stats_dirs) == len(train_cases)
        and not extra
        and not missing
    )
    if passed:
        print("\n✅ Gate-0 通过")
    else:
        print("\n❌ Gate-0 未通过")
        if not args.no_strict:
            sys.exit(1)


if __name__ == "__main__":
    main()
