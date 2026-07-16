#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按数据层白名单正式预处理 AAA/ILO 新队列。

默认候选严格来自 audit_classified.csv 的 F_clean_pass / E_dense_clean：缺文件、WSS
全零、百万点混合区均不会进入。支持 Slurm array 通过 SLURM_ARRAY_TASK_ID 每任务一例。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from pipeline_wss_min import config as C, preprocess, reporting
from pipeline_wss_min.new_cohorts.common import candidate_units, split_unit


def summarize(units):
    rows = []
    for unit in units:
        cohort, case_name = split_unit(unit)
        report_path = C.out_case_dir(cohort, case_name) / "report.json"
        if not report_path.is_file():
            rows.append({"unit_id": unit, "status": "missing", "reason": "report.json 不存在"})
            continue
        row = json.loads(report_path.read_text())
        row["unit_id"] = unit
        rows.append(row)
    out_json = C.OUT_ROOT / "pipeline_reports" / "new_cohorts_v4_preprocess_summary.json"
    out_csv = C.OUT_ROOT / "pipeline_reports" / "new_cohorts_v4_preprocess_summary.csv"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    summary = {
        "n_expected": len(units),
        "n_ok": sum(r.get("status") == "ok" for r in rows),
        "n_missing": sum(r.get("status") == "missing" for r in rows),
        "n_other": sum(r.get("status") not in ("ok", "missing") for r in rows),
        "centerline_translation_repaired_units": [
            r["unit_id"] for r in rows if r.get("centerline_translation_applied")
        ],
        "frame_versions": sorted({r.get("frame_version", "") for r in rows if r.get("status") == "ok"}),
        "outputs": {"json": str(out_json), "csv": str(out_csv)},
    }
    out_json.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unit-id", default=None)
    ap.add_argument("--index", type=int, default=None, help="0-based 候选索引")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    units = candidate_units()

    if args.list:
        for i, unit in enumerate(units):
            print(f"{i}\t{unit}")
        return
    if args.summarize:
        summarize(units)
        return

    index = args.index
    if index is None and "SLURM_ARRAY_TASK_ID" in os.environ:
        index = int(os.environ["SLURM_ARRAY_TASK_ID"])
    if args.unit_id:
        selected = [args.unit_id]
    elif index is not None:
        if not (0 <= index < len(units)):
            raise SystemExit(f"index 越界: {index}, 候选数={len(units)}")
        selected = [units[index]]
    else:
        selected = units

    for unit in selected:
        if unit not in units:
            raise SystemExit(f"unit 不在数据层+坐标架双白名单中: {unit}")
        cohort, case_name = split_unit(unit)
        stage = "new_" + "_".join(unit.split("/")[-2:]).replace("-", "_")
        reporting.setup_logging(stage)
        report = preprocess.preprocess_case(cohort, case_name, C.DEFAULT)
        if report.get("frame_version") != "stl_landmarks_v4":
            raise RuntimeError(f"{unit} 未使用 v4 坐标架")
        if report.get("landmark_source") != "original_stl":
            raise RuntimeError(f"{unit} 未使用原始 STL")


if __name__ == "__main__":
    main()
