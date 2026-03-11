#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量处理前的数据完备性检查。

目标：
1. 不改动原有处理逻辑
2. 在批量运行前识别缺失输入
3. 输出可复查的问题清单，便于后续补数据
"""

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from pipeline.config import BC_DIR, BC_FILE_MAPPING, INNER_DIR, SURFACE_DIR
else:
    from .config import BC_DIR, BC_FILE_MAPPING, INNER_DIR, SURFACE_DIR


def _collect_numbered_files(folder: Path) -> Dict[str, Path]:
    numbered_files: Dict[str, Path] = {}
    if not folder.is_dir():
        return numbered_files

    for path in folder.iterdir():
        if not path.is_file():
            continue
        stem = path.stem
        if "-" not in stem:
            continue
        number_part = stem.split("-")[-1]
        if number_part.isdigit():
            numbered_files[number_part] = path
    return numbered_files


def _find_surface_file(case_dir: Path) -> Optional[Path]:
    case_name = case_dir.name
    possible_names = [
        f"{case_name}.stl",
        f"{case_name.replace('_', ' ')}.stl",
        f"{case_name.replace(' ', '_')}.stl",
    ]

    for name in possible_names:
        candidate = case_dir / name
        if candidate.exists():
            return candidate

    surface_files = sorted(list(case_dir.glob("*.stl")) + list(case_dir.glob("*.vtp")))
    return surface_files[0] if surface_files else None


def _infer_source_name(case_dir: Path) -> str:
    parent = case_dir.parent
    grandparent = parent.parent
    if grandparent.name.upper() in {"AAA", "AG", "ILO"}:
        return f"{grandparent.name}/{parent.name}"
    if parent.name.upper() in {"AAA", "AG", "ILO"}:
        return parent.name
    return parent.name


def _add_issue(issues: List[Dict[str, object]], code: str, message: str, severity: str = "error") -> None:
    issues.append(
        {
            "code": code,
            "severity": severity,
            "message": message,
        }
    )


def inspect_case_inputs(case_dir: Path) -> Dict[str, object]:
    """
    检查单病例在批量处理中的输入完备性。

    返回字段同时覆盖：
    - 步骤1 preprocess 所需：ascii / ascii_in
    - 步骤2 extract_features 额外所需：stl / Global_conditions
    """
    case_dir = Path(case_dir)
    surface_dir = case_dir / SURFACE_DIR
    inner_dir = case_dir / INNER_DIR
    bc_dir = case_dir / BC_DIR

    issues: List[Dict[str, object]] = []

    surface_files = _collect_numbered_files(surface_dir)
    inner_files = _collect_numbered_files(inner_dir)
    matched_steps = sorted(set(surface_files) & set(inner_files), key=int)
    surface_only_steps = sorted(set(surface_files) - set(inner_files), key=int)
    inner_only_steps = sorted(set(inner_files) - set(surface_files), key=int)

    if not surface_dir.is_dir():
        _add_issue(issues, "missing_ascii_dir", f"缺少目录: {SURFACE_DIR}")
    elif not surface_files:
        _add_issue(issues, "empty_ascii_dir", f"{SURFACE_DIR} 目录存在，但未找到可识别时间步文件")

    if not inner_dir.is_dir():
        _add_issue(issues, "missing_ascii_in_dir", f"缺少目录: {INNER_DIR}")
    elif not inner_files:
        _add_issue(issues, "empty_ascii_in_dir", f"{INNER_DIR} 目录存在，但未找到可识别时间步文件")

    if surface_only_steps:
        _add_issue(
            issues,
            "ascii_missing_in_ascii_in",
            f"{SURFACE_DIR} 存在 {len(surface_only_steps)} 个未配对时间步",
            severity="warning",
        )

    if inner_only_steps:
        _add_issue(
            issues,
            "ascii_in_missing_in_ascii",
            f"{INNER_DIR} 存在 {len(inner_only_steps)} 个未配对时间步",
            severity="warning",
        )

    if surface_dir.is_dir() and inner_dir.is_dir() and not matched_steps:
        _add_issue(issues, "no_matched_ascii_frames", "ascii 与 ascii_in 没有任何可配对时间步")

    named_surface_model = case_dir / f"{case_dir.name}.stl"
    has_named_surface_model = named_surface_model.exists()
    surface_model = _find_surface_file(case_dir)
    if surface_model is None:
        _add_issue(issues, "missing_surface_model", "缺少 STL/VTP 表面模型，无法提取几何特征")
    elif not has_named_surface_model:
        _add_issue(
            issues,
            "missing_named_stl",
            f"缺少同名 STL 文件: {case_dir.name}.stl",
            severity="warning",
        )

    bc_files_complete = False
    if not bc_dir.is_dir():
        _add_issue(issues, "missing_global_conditions_dir", f"缺少目录: {BC_DIR}")
    else:
        inlet_primary = bc_dir / BC_FILE_MAPPING["inlet"]["primary"]
        inlet_fallback = bc_dir / BC_FILE_MAPPING["inlet"]["fallback"]
        has_inlet_bc = inlet_primary.exists() or inlet_fallback.exists()
        if not has_inlet_bc:
            _add_issue(
                issues,
                "missing_inlet_bc",
                f"缺少入口边界条件文件: {inlet_primary.name} / {inlet_fallback.name}",
            )

        missing_outlet_count = 0
        for outlet_key in ("O1", "O2", "O3", "O4"):
            outlet_file = bc_dir / BC_FILE_MAPPING[outlet_key]
            if not outlet_file.exists():
                missing_outlet_count += 1
                _add_issue(
                    issues,
                    f"missing_{outlet_key.lower()}_bc",
                    f"缺少出口边界条件文件: {outlet_file.name}",
                )
        bc_files_complete = has_inlet_bc and missing_outlet_count == 0

    has_preprocess_inputs = surface_dir.is_dir() and inner_dir.is_dir() and bool(matched_steps)
    has_feature_inputs = has_preprocess_inputs and (surface_model is not None) and bc_files_complete

    return {
        "case_name": case_dir.name,
        "case_path": str(case_dir),
        "source": _infer_source_name(case_dir),
        "surface_dir_exists": surface_dir.is_dir(),
        "inner_dir_exists": inner_dir.is_dir(),
        "bc_dir_exists": bc_dir.is_dir(),
        "bc_files_complete": bc_files_complete,
        "has_named_surface_model": has_named_surface_model,
        "surface_model_path": str(surface_model) if surface_model else None,
        "surface_frame_count": len(surface_files),
        "inner_frame_count": len(inner_files),
        "matched_frame_count": len(matched_steps),
        "surface_only_count": len(surface_only_steps),
        "inner_only_count": len(inner_only_steps),
        "surface_only_steps": [int(step) for step in surface_only_steps],
        "inner_only_steps": [int(step) for step in inner_only_steps],
        "matched_steps": [int(step) for step in matched_steps],
        "has_preprocess_inputs": has_preprocess_inputs,
        "has_feature_inputs": has_feature_inputs,
        "issues": issues,
    }


def build_batch_issue_report(case_dirs: Iterable[Path]) -> Dict[str, object]:
    case_reports = [inspect_case_inputs(case_dir) for case_dir in case_dirs]
    cases_with_issues = [report for report in case_reports if report["issues"]]
    preprocess_ready = sum(1 for report in case_reports if report["has_preprocess_inputs"])
    feature_ready = sum(1 for report in case_reports if report["has_feature_inputs"])
    named_stl_ready = sum(1 for report in case_reports if report["has_named_surface_model"])

    issue_type_counts: Dict[str, int] = {}
    source_counts: Dict[str, int] = defaultdict(int)
    source_feature_ready: Dict[str, int] = defaultdict(int)
    source_preprocess_ready: Dict[str, int] = defaultdict(int)
    for report in case_reports:
        source = str(report["source"])
        source_counts[source] += 1
        if report["has_preprocess_inputs"]:
            source_preprocess_ready[source] += 1
        if report["has_feature_inputs"]:
            source_feature_ready[source] += 1
        for issue in report["issues"]:
            code = str(issue["code"])
            issue_type_counts[code] = issue_type_counts.get(code, 0) + 1

    return {
        "case_count": len(case_reports),
        "preprocess_ready_count": preprocess_ready,
        "feature_ready_count": feature_ready,
        "named_stl_ready_count": named_stl_ready,
        "issue_case_count": len(cases_with_issues),
        "issue_type_counts": issue_type_counts,
        "source_case_counts": dict(sorted(source_counts.items())),
        "source_preprocess_ready_counts": dict(sorted(source_preprocess_ready.items())),
        "source_feature_ready_counts": dict(sorted(source_feature_ready.items())),
        "cases": case_reports,
    }


def _report_paths(output_dir: Path, report_name: str) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{report_name}.json", output_dir / f"{report_name}.csv"


def save_batch_issue_report(report: Dict[str, object], output_dir: Path, report_name: str) -> tuple[Path, Path]:
    json_path, csv_path = _report_paths(output_dir, report_name)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case_name",
                "source",
                "has_preprocess_inputs",
                "has_feature_inputs",
                "has_named_surface_model",
                "bc_files_complete",
                "surface_frame_count",
                "inner_frame_count",
                "matched_frame_count",
                "surface_only_count",
                "inner_only_count",
                "issue_count",
                "issue_codes",
                "issue_messages",
            ],
        )
        writer.writeheader()
        for case_report in report["cases"]:
            writer.writerow(
                {
                    "case_name": case_report["case_name"],
                    "source": case_report["source"],
                    "has_preprocess_inputs": case_report["has_preprocess_inputs"],
                    "has_feature_inputs": case_report["has_feature_inputs"],
                    "has_named_surface_model": case_report["has_named_surface_model"],
                    "bc_files_complete": case_report["bc_files_complete"],
                    "surface_frame_count": case_report["surface_frame_count"],
                    "inner_frame_count": case_report["inner_frame_count"],
                    "matched_frame_count": case_report["matched_frame_count"],
                    "surface_only_count": case_report["surface_only_count"],
                    "inner_only_count": case_report["inner_only_count"],
                    "issue_count": len(case_report["issues"]),
                    "issue_codes": ";".join(str(issue["code"]) for issue in case_report["issues"]),
                    "issue_messages": " | ".join(str(issue["message"]) for issue in case_report["issues"]),
                }
            )

    return json_path, csv_path
