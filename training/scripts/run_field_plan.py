from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List


def load_manifest(path: str | Path) -> List[dict]:
    # run_field_plan 不解析具体 config 内容，只依赖 manifest 里已经固化好的 config_path。
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return list(raw.get("items", []))


def filter_items(
    items: Iterable[dict],
    *,
    study_group: str,
    exp_id: str,
    seed: int | None,
) -> List[dict]:
    # 这里保持最朴素的筛选逻辑，方便后面扩展成按 tag / model / feature_set 过滤。
    selected = []
    for item in items:
        if study_group and item.get("study_group") != study_group:
            continue
        if exp_id and item.get("exp_id") != exp_id:
            continue
        if seed is not None and int(item.get("seed", -1)) != seed:
            continue
        selected.append(item)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="按 manifest 顺序批量执行任务A训练")
    parser.add_argument(
        "--manifest",
        default="training/configs/field/generated/manifest.json",
        help="由 training.scripts.make_field_plan 生成的 manifest.json",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="用于启动训练的 Python 解释器",
    )
    parser.add_argument("--study-group", default="", help="只运行指定 study_group")
    parser.add_argument("--exp-id", default="", help="只运行指定 exp_id")
    parser.add_argument("--seed", type=int, default=None, help="只运行指定 seed")
    parser.add_argument("--limit", type=int, default=0, help="最多运行多少个实验")
    parser.add_argument("--dry-run", action="store_true", help="只打印命令，不实际执行")
    args = parser.parse_args()

    items = load_manifest(args.manifest)
    items = filter_items(
        items,
        study_group=args.study_group,
        exp_id=args.exp_id,
        seed=args.seed,
    )
    if args.limit > 0:
        items = items[: args.limit]

    if not items:
        raise ValueError("筛选后没有可执行的实验项")

    for idx, item in enumerate(items, start=1):
        # 这里直接调用 `python -m training.scripts.train_field`，避免批量入口和单实验入口出现两套逻辑。
        command = [
            args.python,
            "-m",
            "training.scripts.train_field",
            "--config",
            item["config_path"],
        ]
        print(
            f"[{idx}/{len(items)}] {item['exp_id']} seed={item['seed']} "
            f"group={item['study_group']} -> {' '.join(command)}"
        )
        if args.dry_run:
            continue
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
