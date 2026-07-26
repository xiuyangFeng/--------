#!/usr/bin/env python3
"""Idempotently backfill the PointNet++ QAD three-seed result into the report xlsx."""

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
ANALYSIS = REPO / "training_wss_min/preflight/pointnetpp_qad_results_analysis.json"

OVERVIEW_TITLE = "PointNet++ QAD-Lite 三种子配对（2026-07-19；val21）"
TEACHER_TITLE = "Ⅷ PointNet++ QAD-Lite 三种子配对（2026-07-19；val21）"


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


def find_row(sheet, col: int, value: str, start: int = 0, end: int = 300) -> int | None:
    for row in range(start, end):
        if sheet.getCellByPosition(col, row).String == value:
            return row
    return None


def last_nonempty_row(sheet, col: int, start: int = 0, end: int = 300) -> int:
    last = start
    for row in range(start, end):
        if sheet.getCellByPosition(col, row).String.strip():
            last = row
    return last


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


def mean_metrics(records: list[dict]) -> dict[str, float]:
    result = {}
    for key in METRIC_KEYS:
        values = [float(record["best"][key]) for record in records]
        result[key] = sum(values) / len(values)
    return result


def build_rows(analysis: dict) -> tuple[list[dict], list[dict]]:
    records = analysis["records"]
    if len(records) != 6 or {int(item["seed"]) for item in records} != {7, 1234, 2025}:
        raise RuntimeError("expected the exact 3-seed R0/R1 QAD matrix")
    if analysis["jobs"].get("10540", "").split(";")[0] != "5/5 COMPLETED (0:0)":
        raise RuntimeError("refusing to update workbook before Job 10540 is fully complete")

    decoder_order = {"interpolate": 0, "qad_lite": 1}
    ordered = sorted(records, key=lambda item: (int(item["seed"]), decoder_order[item["decoder"]]))
    raw_rows = []
    for item in ordered:
        decoder = "R0插值解码" if item["decoder"] == "interpolate" else "R1-QAD"
        raw_rows.append({
            "id": f"{decoder}(n32w32,s{item['seed']})",
            "source_id": item["experiment_id"],
            "stage": f"val21单种子（seed {item['seed']}）",
            "metrics": item["best"],
            "decoder": item["decoder"],
        })

    mean_rows = []
    for decoder, label in (("interpolate", "R0插值解码"), ("qad_lite", "R1-QAD")):
        selected = [item for item in records if item["decoder"] == decoder]
        mean_rows.append({
            "id": f"{label}(n32w32,3seed均值)",
            "source_id": f"{decoder}_n32_w32_three_seed_mean",
            "stage": "三种子均值（val21）",
            "metrics": mean_metrics(selected),
            "decoder": decoder,
        })
    return raw_rows, mean_rows


def capacity(decoder: str) -> str:
    suffix = "interpolate" if decoder == "interpolate" else "QAD-Lite(+2,625 params)"
    return f"5000→500→125→32；r=0.05/0.10/0.20；nsample=32；width=32；decoder={suffix}"


def overview_values(item: dict) -> tuple:
    metric = item["metrics"]
    return (
        item["id"], "85/21/27（只看val21）", "PointNet++ SA3", capacity(item["decoder"]),
        "vertex random5000 / SEP；FPS center", 5000,
        *(metric[key] for key in METRIC_KEYS), item["source_id"],
    )


def teacher_values(item: dict) -> tuple:
    metric = item["metrics"]
    return (
        item["stage"], item["id"], "PointNet++ SA3", "vertex random5000 / SEP；FPS center",
        metric["physical_r2_casebalanced"], metric["physical_r2_case_mean"],
        metric["physical_rmse"], metric["physical_nmae_pooled"],
        metric["physical_nmae_case_mean"], metric["high_wss_r2"],
        metric["normalized_r2_casebalanced"], metric["normalized_nmae_pooled"],
        metric["spearman_case_mean"], metric["physical_top10_iou"],
        metric["physical_p99_amplitude_ratio"],
    )


def update_overview(sheet, raw_rows: list[dict], mean_rows: list[dict]) -> int:
    section_row = find_row(sheet, 0, OVERVIEW_TITLE, 2)
    if section_row is None:
        section_row = last_nonempty_row(sheet, 0, 2) + 1
        copy_row(sheet, 40, section_row, 36)  # reuse the preceding section style
        try:
            sheet.getCellRangeByPosition(0, section_row, 35, section_row).merge(True)
        except Exception:
            pass
    clear_row(sheet, section_row, 36)
    set_cell(sheet, 0, section_row, OVERVIEW_TITLE)
    rows = raw_rows + mean_rows
    for offset, item in enumerate(rows, start=1):
        row = section_row + offset
        copy_row(sheet, 41, row, 36)  # reuse a standard data-row style
        clear_row(sheet, row, 36)
        for col, value in enumerate(overview_values(item)):
            set_cell(sheet, col, row, value)
    end_row = section_row + len(rows)
    sheet.setPrintAreas((CellRangeAddress(
        Sheet=sheet.RangeAddress.Sheet, StartColumn=0, EndColumn=36,
        StartRow=0, EndRow=end_row + 1,
    ),))
    return end_row


def update_teacher(sheet, mean_rows: list[dict]) -> int:
    section_row = find_row(sheet, 0, TEACHER_TITLE, 3)
    if section_row is None:
        section_row = last_nonempty_row(sheet, 0, 3) + 1
        copy_row(sheet, 50, section_row, 14)  # reuse the preceding section style
        try:
            sheet.getCellRangeByPosition(0, section_row, 14, section_row).merge(True)
        except Exception:
            pass
    clear_row(sheet, section_row, 14)
    set_cell(sheet, 0, section_row, TEACHER_TITLE)
    for offset, item in enumerate(mean_rows, start=1):
        row = section_row + offset
        copy_row(sheet, 51, row, 14)
        clear_row(sheet, row, 14)
        for col, value in enumerate(teacher_values(item)):
            set_cell(sheet, col, row, value)
    end_row = section_row + len(mean_rows)
    sheet.setPrintAreas((CellRangeAddress(
        Sheet=sheet.RangeAddress.Sheet, StartColumn=0, EndColumn=14,
        StartRow=0, EndRow=end_row + 1,
    ),))
    return end_row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workbook", type=Path, default=WORKBOOK)
    parser.add_argument("--analysis", type=Path, default=ANALYSIS)
    parser.add_argument(
        "--uno-url", default="uno:socket,host=localhost,port=2085;urp;StarOffice.ComponentContext"
    )
    args = parser.parse_args()
    analysis = json.loads(args.analysis.read_text(encoding="utf-8"))
    raw_rows, mean_rows = build_rows(analysis)

    temp_dir = Path(tempfile.mkdtemp(prefix="qad_xlsx_"))
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
        overview_end = update_overview(doc.Sheets.getByName("实验矩阵总览"), raw_rows, mean_rows)
        teacher_end = update_teacher(doc.Sheets.getByName("教师汇报视图"), mean_rows)
        page_styles = doc.StyleFamilies.getByName("PageStyles")
        for sheet_name, pages_y in (("实验矩阵总览", 1), ("教师汇报视图", 2)):
            sheet = doc.Sheets.getByName(sheet_name)
            style = page_styles.getByName(sheet.PageStyle)
            style.ScaleToPagesX = 1
            style.ScaleToPagesY = pages_y
        doc.store()
    finally:
        doc.close(True)
    if not zipfile.is_zipfile(temp_book):
        raise RuntimeError("updated workbook is not a valid xlsx zip container")
    shutil.copy2(temp_book, args.workbook)
    shutil.rmtree(temp_dir)
    print(json.dumps({
        "status": "updated", "overview_rows": len(raw_rows) + len(mean_rows),
        "teacher_rows": len(mean_rows), "overview_end": overview_end + 1,
        "teacher_end": teacher_end + 1, "workbook": str(args.workbook),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
