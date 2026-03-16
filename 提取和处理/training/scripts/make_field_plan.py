from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from ..analysis.plan import build_task_a_plan


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def dump_json(data, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def parse_groups(value: str) -> List[str]:
    # 生成器只接受有限的实验组名字，防止拼写错误时默默生成空结果。
    groups = [item.strip() for item in value.split(",") if item.strip()]
    valid = {"baseline", "input", "geometry", "augment", "coord"}
    unknown = sorted(set(groups) - valid)
    if unknown:
        raise ValueError(f"未知实验组: {unknown}")
    return groups


def parse_coord_variants(values: List[str]) -> Dict[str, str]:
    # 坐标归一化实验显式要求 name=subdir，避免把目录逻辑硬编码进 plan.py。
    variants: Dict[str, str] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"坐标变体格式必须是 name=subdir，收到: {item}")
        name, subdir = item.split("=", 1)
        variants[name.strip()] = subdir.strip()
    return variants


def main() -> None:
    parser = argparse.ArgumentParser(description="任务A实验计划生成器")
    parser.add_argument("--data-root", required=True, help="训练数据根目录")
    parser.add_argument("--split-file", required=True, help="患者级 split JSON 路径")
    parser.add_argument(
        "--graphs-subdir",
        default="processed/graphs",
        help="默认图数据子目录，baseline/input/geometry/augment 组共用",
    )
    parser.add_argument(
        "--output-dir",
        default="training/configs/field/generated",
        help="生成配置文件的目录",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/field",
        help="训练产物输出根目录，将写入每个配置文件的 run.output_root",
    )
    parser.add_argument(
        "--groups",
        default="baseline,input,geometry,augment",
        help="生成哪些实验组，逗号分隔，可选 baseline,input,geometry,augment,coord",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[1, 2, 3],
        help="为每个实验组生成的随机种子列表",
    )
    parser.add_argument(
        "--coord-variant",
        action="append",
        default=[],
        help="坐标归一化变体，格式 name=subdir，可重复传入",
    )
    args = parser.parse_args()

    groups = parse_groups(args.groups)
    coord_variants = parse_coord_variants(args.coord_variant)
    output_dir = ensure_dir(args.output_dir)

    items = build_task_a_plan(
        data_root=args.data_root,
        split_file=args.split_file,
        graphs_subdir=args.graphs_subdir,
        output_root=args.output_root,
        seeds=args.seeds,
        groups=groups,
        coord_variants=coord_variants,
    )

    manifest_items = []
    for item in items:
        config_path = output_dir / item.output_relpath
        ensure_dir(config_path.parent)
        dump_json(item.config.to_dict(), config_path)

        # manifest 的作用是给 run_field_plan.py 和后续服务器批量调度器做唯一事实来源。
        row = item.manifest_row()
        row["config_path"] = str(config_path)
        row["split_file"] = args.split_file
        row["graphs_subdir"] = item.config.data.graphs_subdir
        manifest_items.append(row)

    manifest = {
        "generated_at": timestamp(),
        "data_root": args.data_root,
        "split_file": args.split_file,
        "groups": groups,
        "seeds": args.seeds,
        "items": manifest_items,
    }
    dump_json(manifest, Path(output_dir) / "manifest.json")

    print(f"已生成 {len(manifest_items)} 个任务A配置文件到: {output_dir}")
    print(Path(output_dir) / "manifest.json")


if __name__ == "__main__":
    main()
