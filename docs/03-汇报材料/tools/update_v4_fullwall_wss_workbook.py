#!/usr/bin/env python3
"""Backfill the eight V4 WSS columns with the full-wall audit metrics."""

from __future__ import annotations

import json
import shutil
from copy import copy
from pathlib import Path

from openpyxl import load_workbook
from openpyxl.comments import Comment
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter


ROOT = Path(__file__).resolve().parents[3]
XLSX = ROOT / "docs/03-汇报材料/WSS_PINN_V1_V2_V3_field-v4实验矩阵与指标汇总_2026-08-06.xlsx"
BACKUP = XLSX.with_name(XLSX.stem + "_备份_补V4_fullwall_WSS前.xlsx")
AUDIT = ROOT / "outputs/wss_pinn/audits/v4_fullwall_20260901"
PLOT_ROOT = ROOT / "docs/03-汇报材料/V4_BC-PDE_FIX_EMA_横向R2散点图_2026-09-01"


def _fmt(summary: dict, decimals: int = 4) -> str:
    if not summary or int(summary.get("n", 0)) <= 0:
        return ""
    return f"{float(summary['mean']):.{decimals}f} ± {float(summary['std_population']):.{decimals}f}"


def _fmt_pct(summary: dict) -> str:
    if not summary or int(summary.get("n", 0)) <= 0:
        return ""
    return f"{100.0 * float(summary['mean']):.2f}% ± {100.0 * float(summary['std_population']):.2f}%"


def _arms() -> list[str]:
    return [
        "V4-SP-PN-BC-PDE-F-s1234",
        "V4-SP-PN-BC-PDE-EMA-s1234",
        "V4-SP-PNPP-BC-PDE-F-s1234",
        "V4-SP-PNPP-BC-PDE-EMA-s1234",
        "V4-TR-PN-BC-PDE-F-s1234",
        "V4-TR-PN-BC-PDE-EMA-s1234",
        "V4-TR-PNPP-BC-PDE-F-s1234",
        "V4-TR-PNPP-BC-PDE-EMA-s1234",
    ]


def _load_audit(arm: str) -> tuple[dict, dict]:
    arm_dir = AUDIT / "arms" / arm
    wss = json.loads((arm_dir / "wss_metrics.json").read_text(encoding="utf-8"))
    downstream = json.loads((arm_dir / "downstream_metrics.json").read_text(encoding="utf-8"))
    return wss, downstream


def _write_summary_sheet(wb, audits: dict[str, tuple[dict, dict]]) -> None:
    name = "V4 Full-wall WSS"
    if name in wb.sheetnames:
        del wb[name]
    ws = wb.create_sheet(name)
    headers = [
        "实验ID", "阶段", "架构", "训练模式", "wall点数（每例范围）",
        "WSS R²_cb", "WSS scaled R²_cb", "WSS NMAE",
        "WSS fit a_cb", "WSS fit b_cb (Pa)", "WSS fit R²_cb",
        "worst", "most frequent", "best", "WSS回归图",
    ]
    navy, blue = "17365D", "2F75B5"
    thin = Side(style="thin", color="B7B7B7")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=len(headers))
    ws.cell(1, 1, "V4 full-wall WSS 下游审计（seed1234；仅用于替换原 1200 点 WSS 指标）")
    ws.cell(1, 1).font = Font(name="Microsoft YaHei", size=14, bold=True, color="FFFFFF")
    ws.cell(1, 1).fill = PatternFill("solid", fgColor=navy)
    ws.merge_cells(start_row=2, start_column=1, end_row=2, end_column=len(headers))
    ws.cell(2, 1, "协议：test35 × 每例全部冻结壁面节点；速度输入为 peak full-volume prediction；WSS 使用 Profile-Secant V3。")
    ws.cell(2, 1).fill = PatternFill("solid", fgColor="D9EAF7")
    ws.cell(2, 1).alignment = Alignment(wrap_text=True)
    for col, header in enumerate(headers, 1):
        cell = ws.cell(3, col, header)
        cell.font = Font(name="Microsoft YaHei", size=9, bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor=blue)
        cell.border = border
    plot_manifest = json.loads(
        (PLOT_ROOT / "V4_fullwall_wss_manifest.json").read_text(encoding="utf-8")
    )["arms"]
    row = 4
    for arm, (wss, downstream) in audits.items():
        bits = arm.split("-")
        temporal = "steady peak" if "SP" in bits else "transient 81"
        backbone = "PointNet++" if "PNPP" in arm else "PointNet"
        mode = "BC+PDE-EMA" if "EMA" in arm else "BC+PDE-FIXED"
        cases = wss["cases"]
        manifest = plot_manifest[arm]
        report_folder = manifest["report_folder"]
        values = [
            arm,
            temporal,
            backbone,
            mode,
            f"{min(x['points'] for x in cases)}–{max(x['points'] for x in cases)}",
            _fmt(wss["wss_r2"]),
            _fmt(wss["wss_scaled_r2"]),
            _fmt_pct(wss["wss_nmae_range"]),
            _fmt(downstream["wss_fit"]["slope_a"]),
            _fmt(downstream["wss_fit"]["intercept_b"]),
            _fmt(downstream["wss_fit"]["regression_r2"]),
            "",
            "",
            "",
            str(PLOT_ROOT / report_folder / "wss_representative_regressions.png"),
        ]
        # The manifest contains the full-wall OLS representatives.
        reps = manifest["wss_representatives"]
        values[11:14] = [reps["worst"], reps["most_frequent"], reps["best"]]
        for col, value in enumerate(values, 1):
            cell = ws.cell(row, col, value)
            cell.border = border
            cell.font = Font(name="Microsoft YaHei", size=9)
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        row += 1
    widths = [38, 16, 12, 18, 20, 18, 20, 18, 16, 18, 18, 32, 32, 32, 80]
    for col, width in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(col)].width = width
    ws.freeze_panes = "B4"
    ws.sheet_view.zoomScale = 80


def main() -> None:
    if not BACKUP.is_file():
        shutil.copy2(XLSX, BACKUP)
    wb = load_workbook(XLSX)
    ws = wb["实验矩阵汇总"]
    ws.cell(1, 1, "WSS_PINN V2 / V3 / V4 实验矩阵与核心指标汇总（2026-09-01 full-wall WSS 更新）")
    # Keep the workbook-level protocol note synchronized with the replacement
    # WSS evaluation.  The previous note described the superseded 1200-point
    # sampling protocol.
    ws.cell(3, 1, "V4 行是 seed1234 official test35 screen：场指标来自全部评估宇宙（瞬态=eligible×81 帧）；E_rel_l2 为设计 primary。WSS 为冻结 Profile-Secant V3、峰值全场速度、test35×每例全部冻结壁面节点（full-wall）。index 0–15（seed1234 16 臂）已填。单 seed，禁止与 V2 SAME5K / V3 val-selected 裸比。")
    ws.cell(3, 1).alignment = Alignment(wrap_text=True, vertical="center")
    headers = [ws.cell(4, col).value for col in range(1, ws.max_column + 1)]
    idx = {name: col for col, name in enumerate(headers, 1) if name}
    audits: dict[str, tuple[dict, dict]] = {}
    for arm in _arms():
        audits[arm] = _load_audit(arm)
    audit_protocol = "Profile-Secant V3; test35×all frozen wall nodes (full-wall); peak full-volume predicted velocity"
    for row in range(5, ws.max_row + 1):
        experiment_id = ws.cell(row, idx["实验ID"]).value
        if experiment_id not in audits:
            continue
        wss, downstream = audits[experiment_id]
        values = {
            "WSS R²_cb (mean±std)": _fmt(wss["wss_r2"]),
            "WSS scaled R²_cb (mean±std)": _fmt(wss["wss_scaled_r2"]),
            "WSS NMAE (mean±std)": _fmt_pct(wss["wss_nmae_range"]),
            "WSS下游协议": audit_protocol,
            "WSS fit a_cb (mean±std)": _fmt(downstream["wss_fit"]["slope_a"]),
            "WSS fit b_cb (Pa, mean±std)": _fmt(downstream["wss_fit"]["intercept_b"]),
            "WSS fit R²_cb (mean±std)": _fmt(downstream["wss_fit"]["regression_r2"]),
        }
        for name, value in values.items():
            cell = ws.cell(row, idx[name], value)
            cell.comment = Comment("V4 full-wall replacement: test35 × all frozen wall nodes; Profile-Secant V3.", "Codex")
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    _write_summary_sheet(wb, audits)
    wb.save(XLSX)
    print(json.dumps({"updated_workbook": str(XLSX), "backup": str(BACKUP), "arms": _arms(), "protocol": audit_protocol}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
