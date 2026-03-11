#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
独立的数据输入审计入口。

用途：
1. 在正式 preprocess / run_all / make_split 前，批量检查原始病例是否完整
2. 支持同时扫描 AAA / AG / ILO 大类，或指定更细的数据源
3. 输出 JSON / CSV / 可用于 split 的病例名单
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Sequence

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from pipeline.config import DATA_ROOT
    from pipeline.validation import build_batch_issue_report, save_batch_issue_report
else:
    from .config import DATA_ROOT
    from .validation import build_batch_issue_report, save_batch_issue_report


def _iter_source_dirs(data_root: Path, groups: Sequence[str]) -> Iterable[Path]:
    if not data_root.exists():
        return
    normalized = {item.strip().upper() for item in groups if item.strip()}
    for group_dir in sorted(data_root.iterdir()):
        if not group_dir.is_dir() or group_dir.name.startswith("."):
            continue
        if normalized and group_dir.name.upper() not in normalized:
            continue

        child_dirs = [item for item in sorted(group_dir.iterdir()) if item.is_dir() and not item.name.startswith(".")]
        case_like_children = [item for item in child_dirs if (item / "ascii").is_dir() or (item / "ascii_in").is_dir()]
        if case_like_children:
            yield group_dir
            continue

        for child_dir in child_dirs:
            yield child_dir


def discover_case_dirs(data_root: Path, groups: Sequence[str], sources: Sequence[str]) -> List[Path]:
    if sources:
        source_dirs = [data_root / src for src in sources]
    else:
        source_dirs = list(_iter_source_dirs(data_root, groups))

    case_dirs: List[Path] = []
    for source_dir in source_dirs:
        if not source_dir.exists() or not source_dir.is_dir():
            continue
        for case_dir in sorted(source_dir.iterdir()):
            if case_dir.is_dir() and not case_dir.name.startswith("."):
                case_dirs.append(case_dir)
    return sorted(case_dirs, key=lambda path: (str(path.parent), path.name))


def write_case_list(report: dict, output_path: Path, require_named_stl: bool) -> int:
    ready_cases = []
    for case in report["cases"]:
        if not case["has_feature_inputs"]:
            continue
        if require_named_stl and not case["has_named_surface_model"]:
            continue
        ready_cases.append(case["case_name"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for case_name in ready_cases:
            f.write(f"{case_name}\n")
    return len(ready_cases)


def main() -> None:
    parser = argparse.ArgumentParser(description="批量审计原始病例输入是否完整")
    parser.add_argument("--data-root", default=str(DATA_ROOT), help="数据根目录，默认使用 pipeline.config.DATA_ROOT")
    parser.add_argument(
        "--groups",
        nargs="+",
        default=["AAA", "AG", "ILO"],
        help="要扫描的大类目录，默认同时扫描 AAA AG ILO",
    )
    parser.add_argument(
        "--sources",
        nargs="*",
        default=[],
        help="更细粒度的数据源，如 AG/fast AAA/rupture；提供后将覆盖 --groups",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="报告输出目录，默认写到 <data-root>/pipeline_reports",
    )
    parser.add_argument(
        "--report-name",
        default="raw_input_audit",
        help="报告文件名前缀，不带扩展名",
    )
    parser.add_argument(
        "--ready-cases-output",
        default=None,
        help="可选：导出通过审计的病例名单 txt，建议在单一数据源审计时供 training.make_split 使用",
    )
    parser.add_argument(
        "--require-named-stl",
        action="store_true",
        help="导出病例名单时要求必须存在同名 <case>.stl",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir) if args.output_dir else data_root / "pipeline_reports"

    if not data_root.exists():
        print(f"❌ 数据根目录不存在: {data_root}")
        return

    case_dirs = discover_case_dirs(data_root, args.groups, args.sources)
    if not case_dirs:
        print(f"❌ 未找到任何病例目录: data_root={data_root}")
        return

    report = build_batch_issue_report(case_dirs)
    json_path, csv_path = save_batch_issue_report(report, output_dir, args.report_name)

    print("=" * 60)
    print("🔎 原始输入审计完成")
    print("=" * 60)
    print(f"📁 数据根目录: {data_root}")
    if args.sources:
        print(f"📚 扫描数据源: {', '.join(args.sources)}")
    else:
        print(f"📚 扫描大类: {', '.join(args.groups)}")
    print(f"📊 病例总数: {report['case_count']}")
    print(f"📊 可跑 preprocess 的病例: {report['preprocess_ready_count']}")
    print(f"📊 可跑 extract_features 的病例: {report['feature_ready_count']}")
    print(f"📊 存在同名 <case>.stl 的病例: {report['named_stl_ready_count']}")
    print(f"📊 存在问题的病例: {report['issue_case_count']}")
    print(f"📝 JSON 报告: {json_path}")
    print(f"📝 CSV 报告: {csv_path}")

    if report["source_case_counts"]:
        print("📚 分数据源统计:")
        for source, total in report["source_case_counts"].items():
            preprocess_ready = report["source_preprocess_ready_counts"].get(source, 0)
            feature_ready = report["source_feature_ready_counts"].get(source, 0)
            print(f"  - {source}: total={total}, preprocess_ready={preprocess_ready}, feature_ready={feature_ready}")

    if report["issue_type_counts"]:
        print("⚠️ 问题类型统计:")
        for code, count in sorted(report["issue_type_counts"].items()):
            print(f"  - {code}: {count}")

    if args.ready_cases_output:
        ready_output = Path(args.ready_cases_output)
        ready_count = write_case_list(report, ready_output, require_named_stl=args.require_named_stl)
        print(f"✅ 可用于 split 的病例名单: {ready_output} ({ready_count} cases)")


if __name__ == "__main__":
    main()
