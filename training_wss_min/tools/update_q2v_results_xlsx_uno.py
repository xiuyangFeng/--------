#!/usr/bin/env python3
"""Idempotently backfill Q2V/ILO and Point++ result rows into the reporting xlsx."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

import uno
from com.sun.star.beans import PropertyValue
from com.sun.star.table import CellAddress, CellRangeAddress


REPO = Path(__file__).resolve().parents[2]
WORKBOOK = REPO / "docs/03-汇报材料/WSS_PointNet实验矩阵与结果汇总last.xlsx"
ANALYSIS = REPO / "training_wss_min/preflight/q2v_ilo_arch_matrix_results_analysis.json"
SUBMISSION = REPO / "training_wss_min/preflight/q2v_ilo_arch_matrix_submission.json"


def prop(name: str, value) -> PropertyValue:
    item = PropertyValue()
    item.Name = name
    item.Value = value
    return item


def set_cell(sheet, col: int, row: int, value) -> None:
    cell = sheet.getCellByPosition(col, row)
    if value is None or value == "":
        cell.String = ""
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        cell.Value = float(value)
    else:
        cell.String = str(value)


def clear_row(sheet, row: int, end_col: int) -> None:
    for col in range(end_col + 1):
        set_cell(sheet, col, row, "")


def copy_row(sheet, source_row: int, target_row: int, end_col: int) -> None:
    sheet.copyRange(
        CellAddress(Sheet=sheet.RangeAddress.Sheet, Column=0, Row=target_row),
        CellRangeAddress(
            Sheet=sheet.RangeAddress.Sheet,
            StartColumn=0,
            EndColumn=end_col,
            StartRow=source_row,
            EndRow=source_row,
        ),
    )


def find_row(sheet, col: int, value: str, start: int = 0, end: int = 400) -> int | None:
    for row in range(start, end):
        if sheet.getCellByPosition(col, row).String == value:
            return row
    return None


def last_nonempty_row(sheet, col: int, start: int = 0, end: int = 400) -> int:
    last = start
    for row in range(start, end):
        if sheet.getCellByPosition(col, row).String.strip():
            last = row
    return last


def conclusions() -> dict[str, str]:
    return {
        "q2v_ilo_d1_fixed_frozen": "同test27较Q2V R²_cb -0.0208；仅IoU改善，不支持整体增益",
        "q2v_ilo_d1_fixed_refit": "较D1 frozen R²_cb +0.0027；方向混合，统计重算收益可忽略",
        "q2v_ilo_d2_extended_frozen": "同test36较zero-shot R²_cb -0.0153；负例2→1、IoU改善但误差/高尾变差",
        "q2v_ilo_d3_pool2025_control": "D3同test36控制锚点；ILO未参与训练",
        "q2v_ilo_d3_pool2025_frozen": "较D3 control R²_cb -0.0066；负例7→3但pooled/高尾回退",
        "q2v_ilo_d3_pool2025_refit": "较D3 frozen R²_cb -0.0457；本协议不支持重算统计",
        "q2v_arch_dev_n16_w32": "Q2V-structure dev control；物理/归一化R²_cb 0.2456/0.5481",
        "q2v_arch_dev_n16_w64": "归一化R²_cb 0.5709数值第一；物理主指标-0.0089，不改写主排名",
        "q2v_arch_dev_n32_w32": "val第二，R²_cb 0.2589；0.224M参数的效率候选",
        "q2v_arch_dev_n32_w64": "预注册物理R²_cb第一0.2603；仅领先0.0014且参数约4倍，待确认",
        "q2v_arch_dev_n64_w32": "n64在width32回退；R²_cb 0.2327",
        "q2v_arch_dev_n64_w64": "扩宽有所恢复但仍低于n32；不优先",
        "Q2V-10477-zero-shot-D2-test36": "原Q2V对扩展test36的零样本锚点；R²_cb 0.2646",
    }


def build_rows(analysis: dict, submission: dict) -> list[dict]:
    submitted = {item["experiment_id"]: item for item in submission["configs"]}
    notes = conclusions()
    rows = []
    for record in analysis["records"]:
        item = submitted[record["experiment_id"]]
        split = json.loads(Path(item["split"]).read_text(encoding="utf-8"))
        config = json.loads(Path(item["config"]).read_text(encoding="utf-8"))
        model = config["model"]
        best = record["best"]
        if record["family"] == "architecture":
            partition = f"{len(split['train_cases'])}/{len(split['val_cases'])}/{len(split['test_cases'])}（只看val21）"
            protocol = "定量完成；test27未访问"
            control = record["control_group"]
            change = record["unique_change"]
            if record["experiment_id"] == "q2v_arch_dev_n16_w32":
                protocol = "Q2V-structure dev control；test27未访问"
                control = "Q2V结构（dev重训；非原checkpoint）"
                change = "仅改dev85/val21 split与train85统计；模型/采样与Q2V一致"
        else:
            partition = f"{len(split['train_cases'])}/{len(split['val_cases'])}/{len(split['test_cases'])}"
            pv = record["postview"]
            if pv["status"] == "passed":
                protocol = f"定量+PostView {pv['verified_cases']}/{pv['expected_cases']}通过"
            else:
                coverage = pv["coverage"]["valid_ratio"]
                protocol = f"定量完成；PostView 35/36门禁失败（单例覆盖{coverage:.1%}）"
            control = record["control_group"]
            change = record["unique_change"]
        rows.append({
            "experiment_id": record["experiment_id"],
            "family": record["family"],
            "partition": partition,
            "model_label": "PointNet++ SA3",
            "capacity": (
                f"5000→500→125→32；r=0.05/0.10/0.20；nsample={model['sa_nsample'][0]}；"
                f"width={model['width']}"
            ),
            "sampling": "vertex random5000 / SEP；FPS center",
            "points": 5000,
            "metrics": best,
            "status": f"{record['slurm_state']} | {record['job_id']}",
            "protocol": protocol,
            "control": control,
            "change": change,
            "conclusion": notes[record["experiment_id"]],
            "source": str(Path(record["run_dir"]) / "eval/ckpt_best/metrics.json"),
        })
    zero = next(item for item in analysis["anchors"] if item["experiment_id"].endswith("test36"))
    rows.append({
        "experiment_id": zero["experiment_id"],
        "family": "read_only_evaluation",
        "partition": "106/0/36（只读override）",
        "model_label": "PointNet++ SA3",
        "capacity": "5000→500→125→32；r=0.05/0.10/0.20；nsample=16；width=32",
        "sampling": "vertex random5000 / SEP；FPS center",
        "points": 5000,
        "metrics": zero["best"],
        "status": "COMPLETED | 10490",
        "protocol": "只读零样本test36；checkpoint未改变",
        "control": "Q2V-10477",
        "change": "仅将评估split override为D2 test36",
        "conclusion": notes[zero["experiment_id"]],
        "source": zero["metrics_path"],
    })
    return rows


METRIC_KEYS = (
    "physical_r2_casebalanced", "physical_r2_pooled", "physical_r2_case_mean",
    "physical_r2_case_median", "physical_r2_case_p10", "physical_r2_negative_cases",
    "physical_rmse", "physical_mae", "physical_nrmse_range_pooled",
    "physical_nrmse_range_case_mean", "physical_nmae_pooled", "physical_nmae_case_mean",
    "high_wss_r2", "physical_top10_amplitude_ratio", "physical_top10_iou",
    "normalized_r2_casebalanced", "normalized_r2_pooled", "normalized_r2_case_mean",
    "normalized_r2_case_median", "normalized_r2_case_p10", "normalized_mae", "normalized_rmse",
    "normalized_nrmse_range_pooled", "normalized_nrmse_range_case_mean",
    "normalized_nmae_pooled", "normalized_nmae_case_mean", "spearman_case_mean",
    "high_wss_spearman_case_mean", "normalized_top10_amplitude_ratio",
    "physical_p99_amplitude_ratio",
)


def update_overview(sheet, rows: list[dict]) -> None:
    headers = ("状态 / Job", "协议闭环", "主对照", "唯一变化", "本轮判读", "指标真源")
    set_cell(sheet, 36, 0, "Q2V/ILO 与架构矩阵结果（2026-07-18）")
    for col, title in enumerate(headers, start=36):
        set_cell(sheet, col, 1, title)
    # Phase-V rows already contain the unique Q2V test27 anchor.  Their point-count
    # column was correct, but Q1V/Q2V/Q3V inherited Q0's 2000-support capacity text.
    phase_v_capacity = "5000→500→125→32；r=0.05/0.10/0.20；nsample=16；width=32"
    for experiment_id in ("Q1V-10476", "Q2V-10477", "Q3V-10478"):
        row = find_row(sheet, 0, experiment_id, 2)
        if row is None:
            raise RuntimeError(f"missing existing Phase-V row: {experiment_id}")
        set_cell(sheet, 3, row, phase_v_capacity)
    duplicate_row = find_row(sheet, 0, "Q2V-10477-reference-test27", 2)
    if duplicate_row is not None:
        clear_row(sheet, duplicate_row, 41)
    next_row = max(21, last_nonempty_row(sheet, 0, 2) + 1)
    for item in rows:
        row = find_row(sheet, 0, item["experiment_id"], 2)
        if row is None:
            row = next_row
            next_row += 1
            copy_row(sheet, 20, row, 41)
        clear_row(sheet, row, 41)
        for col, value in enumerate((
            item["experiment_id"], item["partition"], item["model_label"], item["capacity"],
            item["sampling"], item["points"],
        )):
            set_cell(sheet, col, row, value)
        for offset, key in enumerate(METRIC_KEYS, start=6):
            set_cell(sheet, offset, row, item["metrics"].get(key))
        for col, value in enumerate((
            item["status"], item["protocol"], item["control"], item["change"],
            item["conclusion"], item["source"],
        ), start=36):
            set_cell(sheet, col, row, value)
    for col, width in enumerate((4200, 7200, 5200, 9000, 9800, 12000), start=36):
        sheet.Columns.getByIndex(col).Width = width
    sheet.setPrintAreas((
        CellRangeAddress(
            Sheet=sheet.RangeAddress.Sheet,
            StartColumn=0,
            EndColumn=35,
            StartRow=0,
            # Calc's xlsx round-trip stores the supplied end row one row short.
            # Include one trailing row so the final reference row remains in print range.
            EndRow=max(find_row(sheet, 0, item["experiment_id"], 2) for item in rows) + 1,
        ),
    ))


def update_teacher(sheet, rows: list[dict]) -> None:
    headers = ("状态 / Job", "协议闭环", "主对照", "唯一变化", "本轮判读", "指标真源")
    for col, title in enumerate(headers, start=15):
        set_cell(sheet, col, 2, title)
    duplicate_row = find_row(sheet, 1, "Q2V-10477-reference-test27", 3)
    if duplicate_row is not None:
        clear_row(sheet, duplicate_row, 20)
    section_text = "Ⅴ Q2V 数据扩容与 Point++ 架构结果（2026-07-18；单seed开发/归因）"
    section_row = find_row(sheet, 0, section_text, 3)
    if section_row is None:
        section_row = max(29, last_nonempty_row(sheet, 0, 3) + 1)
        copy_row(sheet, 19, section_row, 20)
        clear_row(sheet, section_row, 20)
        try:
            sheet.getCellRangeByPosition(0, section_row, 20, section_row).merge(True)
        except Exception:
            pass
        set_cell(sheet, 0, section_row, section_text)
    next_row = max(section_row + 1, last_nonempty_row(sheet, 1, section_row + 1) + 1)
    for item in rows:
        row = find_row(sheet, 1, item["experiment_id"], section_row + 1)
        if row is None:
            row = next_row
            next_row += 1
            copy_row(sheet, 20, row, 20)
        clear_row(sheet, row, 20)
        metric = item["metrics"]
        family = {
            "data": "数据扩容",
            "architecture": "架构开发（val）",
            "read_only_evaluation": "只读评估",
        }[item["family"]]
        values = (
            family, item["experiment_id"], item["model_label"], item["sampling"],
            metric["physical_r2_casebalanced"], metric["physical_r2_case_mean"],
            metric["physical_rmse"], metric["physical_nmae_pooled"],
            metric["physical_nmae_case_mean"], metric["high_wss_r2"],
            metric["normalized_r2_casebalanced"], metric["normalized_nmae_pooled"],
            metric["spearman_case_mean"], metric["physical_top10_iou"],
            metric["physical_p99_amplitude_ratio"],
            item["status"], item["protocol"], item["control"], item["change"],
            item["conclusion"], item["source"],
        )
        for col, value in enumerate(values):
            set_cell(sheet, col, row, value)
    for col, width in enumerate((4200, 6800, 5200, 9000, 9800, 12000), start=15):
        sheet.Columns.getByIndex(col).Width = width
    sheet.setPrintAreas((
        CellRangeAddress(
            Sheet=sheet.RangeAddress.Sheet,
            StartColumn=0,
            EndColumn=14,
            StartRow=0,
            EndRow=max(find_row(sheet, 1, item["experiment_id"], section_row + 1) for item in rows) + 1,
        ),
    ))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workbook", type=Path, default=WORKBOOK)
    parser.add_argument("--analysis", type=Path, default=ANALYSIS)
    parser.add_argument("--submission", type=Path, default=SUBMISSION)
    parser.add_argument(
        "--uno-url", default="uno:socket,host=localhost,port=2085;urp;StarOffice.ComponentContext"
    )
    args = parser.parse_args()
    analysis = json.loads(args.analysis.read_text(encoding="utf-8"))
    submission = json.loads(args.submission.read_text(encoding="utf-8"))
    rows = build_rows(analysis, submission)
    if len(rows) != 13:
        raise RuntimeError(f"expected 13 result rows, got {len(rows)}")

    temp_dir = Path(tempfile.mkdtemp(prefix="q2v_xlsx_"))
    temp_book = temp_dir / args.workbook.name
    shutil.copy2(args.workbook, temp_book)
    local = uno.getComponentContext()
    resolver = local.ServiceManager.createInstanceWithContext(
        "com.sun.star.bridge.UnoUrlResolver", local
    )
    ctx = resolver.resolve(args.uno_url)
    desktop = ctx.ServiceManager.createInstanceWithContext("com.sun.star.frame.Desktop", ctx)
    doc = desktop.loadComponentFromURL(
        uno.systemPathToFileUrl(str(temp_book.resolve())), "_blank", 0,
        (prop("Hidden", True), prop("ReadOnly", False)),
    )
    try:
        update_teacher(doc.Sheets.getByName("教师汇报视图"), rows)
        update_overview(doc.Sheets.getByName("实验矩阵总览"), rows)
        page_styles = doc.StyleFamilies.getByName("PageStyles")
        for sheet_name in ("教师汇报视图", "实验矩阵总览"):
            sheet = doc.Sheets.getByName(sheet_name)
            style = page_styles.getByName(sheet.PageStyle)
            style.ScaleToPagesX = 1
            style.ScaleToPagesY = 2
        doc.store()
    finally:
        doc.close(True)
    if not zipfile.is_zipfile(temp_book):
        raise RuntimeError("updated workbook is not a valid xlsx zip container")
    shutil.copy2(temp_book, args.workbook)
    shutil.rmtree(temp_dir)
    print(json.dumps({"status": "updated", "rows": len(rows), "workbook": str(args.workbook)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
