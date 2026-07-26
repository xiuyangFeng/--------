#!/usr/bin/env python3
"""Backfill the completed historical-test27 sampling/radius exploration into xlsx."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

import uno
from com.sun.star.beans import PropertyValue
from com.sun.star.table import CellAddress, CellRangeAddress


REPO = Path(__file__).resolve().parents[2]
WORKBOOK = REPO / "docs/03-汇报材料/WSS_PointNet实验矩阵与结果汇总last.xlsx"
ANALYSIS = REPO / "training_wss_min/preflight/q2v_sampling_radius_test27_results_analysis.json"


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
    found = start
    for row in range(start, end):
        if sheet.getCellByPosition(col, row).String.strip():
            found = row
    return found


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


def row_definitions(analysis: dict) -> list[dict]:
    comparisons = analysis["paired_comparisons"]
    notes = {
        "q2v_radius_r80_sep": (
            f"较Q2V R²_cb {comparisons['r80_vs_q2v']['treatment_minus_control']['physical_r2_casebalanced']:+.4f}；缩半径无增益"
        ),
        "q2v_radius_r60_sep": (
            f"较Q2V R²_cb {comparisons['r60_vs_q2v']['treatment_minus_control']['physical_r2_casebalanced']:+.4f}；进一步缩小仍回退"
        ),
        "q1v_fpsmultistart5000_same": (
            f"较Q1V R²_cb {comparisons['fps_multistart_vs_q1v']['treatment_minus_control']['physical_r2_casebalanced']:+.4f}；不支持替代vertex-random"
        ),
        "q2v_to_q1v_same_finetune": (
            f"较Q1V R²_cb {comparisons['warm_start_vs_q1v']['treatment_minus_control']['physical_r2_casebalanced']:+.4f}；warm-start无稳定增益"
        ),
    }
    meta = {
        "q2v_radius_r80_sep": {
            "capacity": "5000→500→125→32；r=0.04/0.08/0.16；nsample=16；width=32",
            "sampling": "vertex random5000 / SEP；FPS center",
            "control": "Q2V-10477", "change": "仅半径缩至原始80%",
        },
        "q2v_radius_r60_sep": {
            "capacity": "5000→500→125→32；r=0.03/0.06/0.12；nsample=16；width=32",
            "sampling": "vertex random5000 / SEP；FPS center",
            "control": "Q2V-10477", "change": "仅半径缩至原始60%",
        },
        "q1v_fpsmultistart5000_same": {
            "capacity": "5000→500→125→32；r=0.05/0.10/0.20；nsample=16；width=32",
            "sampling": "fps_multistart5000 / SAME；FPS center",
            "control": "Q1V-10476", "change": "仅support/query采样：vertex-random→fps_multistart",
        },
        "q2v_to_q1v_same_finetune": {
            "capacity": "5000→500→125→32；r=0.05/0.10/0.20；nsample=16；width=32",
            "sampling": "vertex random5000 / SAME；FPS center",
            "control": "Q1V-10476", "change": "Q2V best权重warm-start；Q1V SAME；优化器重置",
        },
    }
    rows = []
    for record in analysis["records"]:
        experiment_id = record["experiment_id"]
        if experiment_id not in meta:
            raise RuntimeError(f"unexpected experiment id: {experiment_id}")
        rows.append({
            "experiment_id": experiment_id,
            "model_label": "PointNet++ SA3",
            "partition": "106/0/27（历史test27探索）",
            "points": 5000,
            "metrics": record["best"],
            "capacity": meta[experiment_id]["capacity"],
            "sampling": meta[experiment_id]["sampling"],
            "status": f"COMPLETED | {record['job_id']}",
            "protocol": "探索性test27；400 epoch+best/last；PostView 27/27通过",
            "control": meta[experiment_id]["control"],
            "change": meta[experiment_id]["change"],
            "conclusion": notes[experiment_id],
            "source": str(Path(record["run_dir"]) / "eval/ckpt_best/metrics.json"),
        })
    return rows


def update_overview(sheet, rows: list[dict]) -> None:
    section = "Q2V/test27 半径与采样探索（2026-07-18；非确认性）"
    section_row = find_row(sheet, 0, section, 2)
    if section_row is None:
        section_row = last_nonempty_row(sheet, 0, 2) + 1
        copy_row(sheet, 21, section_row, 35)
        clear_row(sheet, section_row, 35)
        try:
            sheet.getCellRangeByPosition(0, section_row, 35, section_row).merge(True)
        except Exception:
            pass
        set_cell(sheet, 0, section_row, section)
    next_row = max(section_row + 1, last_nonempty_row(sheet, 0, section_row + 1) + 1)
    for item in rows:
        row = find_row(sheet, 0, item["experiment_id"], section_row + 1)
        if row is None:
            row = next_row
            next_row += 1
            copy_row(sheet, 19, row, 35)
        clear_row(sheet, row, 35)
        for col, value in enumerate((
            item["experiment_id"], item["partition"], item["model_label"], item["capacity"],
            item["sampling"], item["points"],
        )):
            set_cell(sheet, col, row, value)
        for offset, key in enumerate(METRIC_KEYS, start=6):
            set_cell(sheet, offset, row, item["metrics"].get(key))
    sheet.setPrintAreas((CellRangeAddress(
        Sheet=sheet.RangeAddress.Sheet, StartColumn=0, EndColumn=35, StartRow=0,
        EndRow=max(find_row(sheet, 0, item["experiment_id"], section_row + 1) for item in rows) + 1,
    ),))


def update_teacher(sheet, rows: list[dict]) -> None:
    section = "Ⅵ Q2V/test27 半径与采样探索（2026-07-18；非确认性）"
    section_row = find_row(sheet, 0, section, 3)
    if section_row is None:
        section_row = last_nonempty_row(sheet, 0, 3) + 1
        copy_row(sheet, 30, section_row, 14)
        clear_row(sheet, section_row, 14)
        try:
            sheet.getCellRangeByPosition(0, section_row, 14, section_row).merge(True)
        except Exception:
            pass
        set_cell(sheet, 0, section_row, section)
    next_row = max(section_row + 1, last_nonempty_row(sheet, 1, section_row + 1) + 1)
    for item in rows:
        row = find_row(sheet, 1, item["experiment_id"], section_row + 1)
        if row is None:
            row = next_row
            next_row += 1
            copy_row(sheet, 32, row, 14)
        clear_row(sheet, row, 14)
        metric = item["metrics"]
        values = (
            "探索性test27", item["experiment_id"], item["model_label"], item["sampling"],
            metric["physical_r2_casebalanced"], metric["physical_r2_case_mean"],
            metric["physical_rmse"], metric.get("physical_nmae_pooled"),
            metric.get("physical_nmae_case_mean"), metric["high_wss_r2"],
            metric["normalized_r2_casebalanced"], metric.get("normalized_nmae_pooled"),
            metric["spearman_case_mean"], metric["physical_top10_iou"],
            metric["physical_p99_amplitude_ratio"],
        )
        for col, value in enumerate(values):
            set_cell(sheet, col, row, value)
    sheet.setPrintAreas((CellRangeAddress(
        Sheet=sheet.RangeAddress.Sheet, StartColumn=0, EndColumn=14, StartRow=0,
        EndRow=max(find_row(sheet, 1, item["experiment_id"], section_row + 1) for item in rows) + 1,
    ),))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workbook", type=Path, default=WORKBOOK)
    parser.add_argument("--analysis", type=Path, default=ANALYSIS)
    parser.add_argument(
        "--uno-url", default="uno:socket,host=localhost,port=2085;urp;StarOffice.ComponentContext"
    )
    args = parser.parse_args()
    analysis = json.loads(args.analysis.read_text(encoding="utf-8"))
    if analysis.get("test27_policy") != "user-authorized historical anchor; not independent confirmation":
        raise RuntimeError("refusing to write a non-exploratory test27 matrix")
    rows = row_definitions(analysis)
    if len(rows) != 4:
        raise RuntimeError(f"expected 4 rows, got {len(rows)}")
    temp_dir = Path(tempfile.mkdtemp(prefix="q2vsr_xlsx_"))
    temp_book = temp_dir / args.workbook.name
    shutil.copy2(args.workbook, temp_book)
    local = uno.getComponentContext()
    resolver = local.ServiceManager.createInstanceWithContext("com.sun.star.bridge.UnoUrlResolver", local)
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
            style.ScaleToPagesY = 3
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
