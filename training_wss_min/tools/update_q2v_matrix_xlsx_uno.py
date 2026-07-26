#!/usr/bin/env python3
"""Append QUEUED-only Q2V/ILO and architecture preregistration rows via LibreOffice UNO.

Run with the system Python while a headless LibreOffice listener is available;
this avoids rebuilding the established workbook or fabricating result cells.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import uno
from com.sun.star.beans import PropertyValue
from com.sun.star.table import CellAddress, CellRangeAddress


REPO = Path(__file__).resolve().parents[2]
WORKBOOK = REPO / "docs/03-汇报材料/WSS_PointNet实验矩阵与结果汇总last.xlsx"
SUBMISSION = REPO / "training_wss_min/preflight/q2v_ilo_arch_matrix_submission.json"


def prop(name: str, value) -> PropertyValue:
    item = PropertyValue()
    item.Name = name
    item.Value = value
    return item


def set_text(sheet, col: int, row: int, value) -> None:
    cell = sheet.getCellByPosition(col, row)
    cell.String = "" if value is None else str(value)


def clear_row(sheet, row: int, start_col: int, end_col: int) -> None:
    for col in range(start_col, end_col + 1):
        set_text(sheet, col, row, "")


def copy_row(sheet, source_row: int, target_row: int, end_col: int) -> None:
    sheet.copyRange(
        CellAddress(Sheet=sheet.RangeAddress.Sheet, Column=0, Row=target_row),
        CellRangeAddress(
            Sheet=sheet.RangeAddress.Sheet,
            StartColumn=0, EndColumn=end_col,
            StartRow=source_row, EndRow=source_row,
        ),
    )


def submission_rows(payload: dict) -> list[dict]:
    if payload.get("status") != "submitted":
        raise RuntimeError("xlsx may only be updated from a completed submission manifest")
    job_by_family = {}
    for job in payload["jobs"]:
        if job["role"] == "data_training_array":
            job_by_family["data"] = job["job_id"]
        elif job["role"] == "architecture_dev_array_val_only":
            job_by_family["architecture"] = job["job_id"]
    rows = []
    for item in payload["configs"]:
        split = json.loads(Path(item["split"]).read_text(encoding="utf-8"))
        cfg = json.loads(Path(item["config"]).read_text(encoding="utf-8"))
        model = cfg["model"]
        family = item["family"]
        task = item["array_task_id"]
        status = f"QUEUED | {job_by_family[family]}_{task}"
        rows.append({
            **item,
            "status": status,
            "partition": (
                f"{len(split['train_cases'])}/{len(split['val_cases'])}/{len(split['test_cases'])}"
            ),
            "model_label": "PointNet++ SA3",
            "capacity": (
                f"500→125→32；r=0.05/0.10/0.20；nsample={model['sa_nsample'][0]}；"
                f"width={model['width']}"
            ),
            "sampling": "vertex random5000 / SEP；FPS center",
        })
    return rows


def update_overview(sheet, rows: list[dict]) -> None:
    existing = {sheet.getCellByPosition(0, row).String for row in range(2, 200)}
    if any(item["experiment_id"] in existing for item in rows):
        raise RuntimeError("overview already contains at least one Q2V preregistration row")
    set_text(sheet, 36, 0, "预注册与执行状态（结果列保持空白）")
    for col, title in enumerate(("状态 / Job", "主对照", "唯一变化", "配置路径"), start=36):
        set_text(sheet, col, 1, title)
    start = 21
    for offset, item in enumerate(rows):
        row = start + offset
        copy_row(sheet, 20, row, 39)
        clear_row(sheet, row, 0, 39)
        values = (
            item["experiment_id"], item["partition"], item["model_label"],
            item["capacity"], item["sampling"], "5000",
        )
        for col, value in enumerate(values):
            set_text(sheet, col, row, value)
        set_text(sheet, 36, row, item["status"])
        set_text(sheet, 37, row, item["control_group"])
        set_text(sheet, 38, row, item["unique_change"])
        set_text(sheet, 39, row, item["config"])
    sheet.Columns.getByIndex(36).Width = 4200
    sheet.Columns.getByIndex(37).Width = 5000
    sheet.Columns.getByIndex(38).Width = 9000
    sheet.Columns.getByIndex(39).Width = 10000


def update_teacher(sheet, rows: list[dict]) -> None:
    existing = {sheet.getCellByPosition(1, row).String for row in range(3, 200)}
    if any(item["experiment_id"] in existing for item in rows):
        raise RuntimeError("teacher view already contains at least one Q2V preregistration row")
    for col, title in enumerate(("状态 / Job", "主对照", "唯一变化"), start=15):
        set_text(sheet, col, 2, title)
    section_row = 29
    copy_row(sheet, 19, section_row, 17)
    clear_row(sheet, section_row, 0, 17)
    try:
        sheet.getCellRangeByPosition(0, section_row, 17, section_row).merge(True)
    except Exception:
        pass
    set_text(sheet, 0, section_row, "Ⅴ Q2V 数据扩容与 Point++ 架构预注册（QUEUED；结果留空）")
    for offset, item in enumerate(rows, start=1):
        row = section_row + offset
        copy_row(sheet, 20, row, 17)
        clear_row(sheet, row, 0, 17)
        set_text(sheet, 0, row, "数据扩容" if item["family"] == "data" else "架构开发")
        set_text(sheet, 1, row, item["experiment_id"])
        set_text(sheet, 2, row, item["model_label"])
        set_text(sheet, 3, row, item["sampling"])
        set_text(sheet, 15, row, item["status"])
        set_text(sheet, 16, row, item["control_group"])
        set_text(sheet, 17, row, item["unique_change"])
    sheet.Columns.getByIndex(15).Width = 4200
    sheet.Columns.getByIndex(16).Width = 5000
    sheet.Columns.getByIndex(17).Width = 9000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workbook", type=Path, default=WORKBOOK)
    ap.add_argument("--submission", type=Path, default=SUBMISSION)
    ap.add_argument(
        "--uno-url", default="uno:socket,host=localhost,port=2085;urp;StarOffice.ComponentContext"
    )
    args = ap.parse_args()
    rows = submission_rows(json.loads(args.submission.read_text(encoding="utf-8")))
    if len(rows) != 12:
        raise RuntimeError(f"expected 12 preregistration rows, got {len(rows)}")

    local = uno.getComponentContext()
    resolver = local.ServiceManager.createInstanceWithContext(
        "com.sun.star.bridge.UnoUrlResolver", local
    )
    ctx = resolver.resolve(args.uno_url)
    desktop = ctx.ServiceManager.createInstanceWithContext("com.sun.star.frame.Desktop", ctx)
    doc = desktop.loadComponentFromURL(
        uno.systemPathToFileUrl(str(args.workbook.resolve())), "_blank", 0,
        (prop("Hidden", True), prop("ReadOnly", False)),
    )
    try:
        update_teacher(doc.Sheets.getByName("教师汇报视图"), rows)
        update_overview(doc.Sheets.getByName("实验矩阵总览"), rows)
        doc.store()
    finally:
        doc.close(True)
    print(json.dumps({"status": "updated", "rows": len(rows),
                      "workbook": str(args.workbook.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
