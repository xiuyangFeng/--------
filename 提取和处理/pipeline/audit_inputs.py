#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
独立的数据输入审计入口。

用途：
1. 在正式 preprocess / run_all / make_split 前，批量检查原始病例是否完整
2. 支持同时扫描 AAA / AG / ILO 大类，或指定更细的数据源
3. 正确处理 ILO 三层嵌套目录 (ILO/<患者>-<0|1>/<before|after>)
4. 输出 JSON / CSV / 可用于 split 的病例名单
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from pipeline.config import DATA_ROOT
    from pipeline.validation import build_batch_issue_report, save_batch_issue_report
else:
    from .config import DATA_ROOT
    from .validation import build_batch_issue_report, save_batch_issue_report

# ---------------------------------------------------------------------------
# 目录发现
# ---------------------------------------------------------------------------

_CASE_MARKER_DIRS = {"ascii", "ascii_in"}
_ILO_PHASES = {"before", "after"}


def _has_case_markers(d: Path) -> bool:
    return any((d / m).is_dir() for m in _CASE_MARKER_DIRS)


def _visible_subdirs(d: Path) -> List[Path]:
    if not d.is_dir():
        return []
    return sorted(c for c in d.iterdir() if c.is_dir() and not c.name.startswith("."))


def _iter_source_dirs(data_root: Path, groups: Sequence[str]) -> Iterable[Path]:
    """
    对每个 group (AAA / AG / ILO) 找到所有包含叶级 case 目录的 *source dir*。

    AAA / AG:  group_dir/subgroup 是 source_dir, 其子目录是 case_dir
    ILO:       group_dir 本身是 source_dir, 其中 <患者>-0|1/before|after 是 case_dir
               此函数 yield group_dir, 由 discover_case_dirs() 进一步展开
    """
    if not data_root.exists():
        return
    normalized = {item.strip().upper() for item in groups if item.strip()}

    for group_dir in sorted(data_root.iterdir()):
        if not group_dir.is_dir() or group_dir.name.startswith("."):
            continue
        if normalized and group_dir.name.upper() not in normalized:
            continue

        child_dirs = _visible_subdirs(group_dir)
        if not child_dirs:
            continue

        # 检查第一层子目录是否直接是 case (含 ascii/)
        case_like = [d for d in child_dirs if _has_case_markers(d)]
        if case_like:
            yield group_dir
            continue

        # 检查是否为 ILO 嵌套: child 含 before/after 且其子目录含 case markers
        ilo_like = any(
            any(gc.name in _ILO_PHASES and _has_case_markers(gc) for gc in _visible_subdirs(child))
            for child in child_dirs[:5]
        )
        if ilo_like:
            yield group_dir
            continue

        # 否则是 AAA/AG 那样的二层: subgroup -> case
        for child_dir in child_dirs:
            yield child_dir


def _expand_ilo_cases(source_dir: Path) -> List[Path]:
    """展开 ILO 嵌套: source_dir/<患者>-0|1/before|after -> list of leaf case dirs."""
    cases: List[Path] = []
    for patient_dir in _visible_subdirs(source_dir):
        leaves = _visible_subdirs(patient_dir)
        phase_leaves = [d for d in leaves if d.name in _ILO_PHASES and _has_case_markers(d)]
        if phase_leaves:
            cases.extend(phase_leaves)
        elif _has_case_markers(patient_dir):
            cases.append(patient_dir)
    return cases


def discover_case_dirs(
    data_root: Path, groups: Sequence[str], sources: Sequence[str]
) -> List[Path]:
    """
    发现所有叶级 case 目录。

    对于 ILO, 自动展开三层嵌套 (ILO/<患者>-<0|1>/<before|after>)。
    对于 AAA/AG, 行为与之前一致。
    """
    if sources:
        source_dirs = [data_root / src for src in sources]
    else:
        source_dirs = list(_iter_source_dirs(data_root, groups))

    case_dirs: List[Path] = []
    for source_dir in source_dirs:
        if not source_dir.exists() or not source_dir.is_dir():
            continue

        # 尝试 ILO 嵌套展开
        ilo_cases = _expand_ilo_cases(source_dir)
        if ilo_cases:
            case_dirs.extend(ilo_cases)
            continue

        # 标准二层: source_dir/<case>/
        for child in _visible_subdirs(source_dir):
            case_dirs.append(child)

    return sorted(case_dirs, key=lambda p: (str(p.parent.parent), str(p.parent), p.name))


# ---------------------------------------------------------------------------
# 导出可用于 split 的病例名单
# ---------------------------------------------------------------------------

def write_case_list(report: dict, output_path: Path, require_named_stl: bool) -> int:
    ready_cases = []
    for case in report["cases"]:
        if not case["has_feature_inputs"]:
            continue
        if require_named_stl and not case["has_named_surface_model"]:
            continue
        ready_cases.append(case["case_id"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for case_id in ready_cases:
            f.write(f"{case_id}\n")
    return len(ready_cases)


# ---------------------------------------------------------------------------
# 终端格式化输出
# ---------------------------------------------------------------------------

def _print_table(headers: List[str], rows: List[List[str]], indent: int = 2) -> None:
    """简易对齐表格打印。"""
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    prefix = " " * indent
    header_line = prefix + "  ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
    sep_line = prefix + "  ".join("-" * col_widths[i] for i in range(len(headers)))
    print(header_line)
    print(sep_line)
    for row in rows:
        print(prefix + "  ".join(cell.ljust(col_widths[i]) for i, cell in enumerate(row)))


def _print_report(report: dict, data_root: Path, scan_label: str) -> None:
    print()
    print("=" * 64)
    print("  原始输入审计完成")
    print("=" * 64)
    print(f"  数据根目录  : {data_root}")
    print(f"  扫描范围    : {scan_label}")
    print()

    # ---- 总览 ----
    print("  [总览]")
    overview_headers = ["指标", "数量"]
    overview_rows = [
        ["病例总数", str(report["case_count"])],
        ["可跑 preprocess", str(report["preprocess_ready_count"])],
        ["可跑 extract_features", str(report["feature_ready_count"])],
        ["存在同名 STL", str(report["named_stl_ready_count"])],
        ["存在问题", str(report["issue_case_count"])],
    ]
    _print_table(overview_headers, overview_rows, indent=4)
    print()

    # ---- 分数据源统计 ----
    if report["source_case_counts"]:
        print("  [分数据源统计]")
        src_headers = ["数据源", "总数", "preprocess就绪", "feature就绪"]
        src_rows = []
        for source in sorted(report["source_case_counts"].keys()):
            total = report["source_case_counts"][source]
            pp = report["source_preprocess_ready_counts"].get(source, 0)
            ft = report["source_feature_ready_counts"].get(source, 0)
            src_rows.append([source, str(total), str(pp), str(ft)])
        _print_table(src_headers, src_rows, indent=4)
        print()

    # ---- 问题类型统计 (附带受影响 case_id 预览) ----
    if report.get("issue_summary"):
        print("  [问题类型统计]")
        for item in report["issue_summary"]:
            code = item["code"]
            count = item["count"]
            cases_preview = item["affected_cases"][:5]
            more = f" ... 等 {count} 例" if count > 5 else ""
            print(f"    {code}: {count}")
            print(f"      例: {', '.join(cases_preview)}{more}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="批量审计原始病例输入是否完整")
    parser.add_argument(
        "--data-root",
        default=str(DATA_ROOT),
        help="数据根目录，默认使用 pipeline.config.DATA_ROOT",
    )
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
        help="更细粒度的数据源，如 AG/fast AAA/ruputer；提供后将覆盖 --groups",
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
        help="导出病例名单时要求必须存在同名 <case>.stl (ILO 除外)",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir) if args.output_dir else data_root / "pipeline_reports"

    if not data_root.exists():
        print(f"错误: 数据根目录不存在: {data_root}")
        return

    case_dirs = discover_case_dirs(data_root, args.groups, args.sources)
    if not case_dirs:
        print(f"错误: 未找到任何病例目录: data_root={data_root}")
        return

    report = build_batch_issue_report(case_dirs)
    json_path, csv_path = save_batch_issue_report(report, output_dir, args.report_name)

    scan_label = ", ".join(args.sources) if args.sources else ", ".join(args.groups)
    _print_report(report, data_root, scan_label)

    print(f"  JSON 报告: {json_path}")
    print(f"  CSV  报告: {csv_path}")

    if args.ready_cases_output:
        ready_output = Path(args.ready_cases_output)
        ready_count = write_case_list(report, ready_output, require_named_stl=args.require_named_stl)
        print(f"  可用名单 : {ready_output} ({ready_count} cases)")

    print()


if __name__ == "__main__":
    main()
