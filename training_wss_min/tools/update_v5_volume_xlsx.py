#!/usr/bin/env python3
"""Add a separate volume-target section (pressure / velocity) to the WSS workbook.

Adds to 《WSS_PointNet实验矩阵与结果汇总last.xlsx》, kept apart from the WSS rows:
* 实验矩阵总览: a title row + one row per result (pressure all / wall / interior, velocity |u|);
  the metric columns keep their positions but are read in the target's units (Pa or m/s), and the
  "high-WSS" column means "top-10% true-value region";
* 教师汇报视图: section Ⅺ with the same rows (P..S formulas point at the overview rows);
* 汇总对比: a dated section with the WSS R4 same-configuration row as the reference.
Idempotent. Uses /usr/bin/python3 (openpyxl).
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from openpyxl import load_workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill

from training_wss_min.tools.update_v5_rerun_xlsx import (
    BLUE, BOOK, EXP, EXPECTED_SHEETS, GREEN, LSA2, NAVY, RUNS, THIN, WHITE, YELLOW, copy_row_style, load_metrics, styled,
    unmerge_from, delete_section,
)

BACKUP = EXP / "WSS_PointNet实验矩阵与结果汇总last_backup_before_volume_20260907.xlsx"
OVERVIEW_TITLE = "体场目标（压力 / 速度，2026-09-07）—— 下列行的指标列按目标单位解释（压力 Pa 相对值、速度 m/s），“high-WSS”列 = 高值区（top10% 真值）R²，归一化为线性 z（R² 与物理相同）"
TEACHER_TITLE = "Ⅺ 体场目标（2026-09-07；★R4 s1234 同配置换回归目标：压力 = 壁面∪内部混合 query、速度 = 内部三分量只评 |u|；qad_lite 解码；test34 全量体单元评估）"
SECTION_TITLE = "2026-09-07｜体场目标（压力 / 速度）：R4 同配置换回归目标（qad_lite 解码，test34 全量内部单元）"
MODEL = LSA2 + "（qad_lite query 解码）"
SPLIT = "V5 train138/test34；support = 壁面 5000，query = {q}；seed=1234"

P_RUN = RUNS / "v5_rerun_20260906/outputs/r5p_pressure_mixed_qad_s1234/eval/ckpt_best/metrics.json"
V_RUN = RUNS / "v5_rerun_20260906/outputs/r5v_velocity_qad_s1234/eval/ckpt_best/metrics.json"
R4_RUN = RUNS / "v5_rerun_20260906/outputs/r4_lsa2_h2_logradius_v5feat_s1234/eval/ckpt_best/metrics.json"


def rows():
    mp, mv = load_metrics(P_RUN), load_metrics(V_RUN)
    return [
        # (id, label, unit, change, split, metrics dict, target name)
        ("v5_volume/pressure_mixed_all", "体场·压力 p−p_ref（壁面∪内部，全部 query 点）", "Pa", "R4 配置换目标：相对压力（p_ref=该例峰值步体积平均压），query 壁面 2500+内部 2500/步，线性 z，无 pinball，18D（+到壁距离）", SPLIT.format(q="全部壁面节点 + 全部内部单元"), mp, "pressure"),
        ("v5_volume/pressure_mixed_wall", "体场·压力 —— 仅壁面节点分组", "Pa", "同上模型，只在壁面节点上评估", SPLIT.format(q="全部壁面节点（1.23M 点）"), mp["query_groups"]["wall"], "pressure"),
        ("v5_volume/pressure_mixed_interior", "体场·压力 —— 仅内部单元分组", "Pa", "同上模型，只在内部单元上评估", SPLIT.format(q="全部内部单元（14.1M 点）"), mp["query_groups"]["interior"], "pressure"),
        ("v5_volume/velocity_speed", "体场·速度幅值 |u|（内部单元；模型出三分量）", "m/s", "R4 配置换目标：速度三分量（解剖系，逐分量线性 z，out_dim=3），只评 |u|；分量 R² 0.735–0.765，方向余弦 0.89", SPLIT.format(q="全部内部单元（14.1M 点）"), mv, "velocity"),
    ]


def overview_values(row):
    wid, label, unit, change, split, m, _ = row
    n = m["normalized"]
    return [
        label, split, MODEL, f"[{unit}] " + change, "random5000 / 独立 query 5000", 5000,
        m["field_casebalanced"]["r2"], m["field"]["r2"], m["aggregate"]["r2_casemean"], m["aggregate"]["r2_casemed"], m["aggregate"]["r2_casep10"],
        f"{m['aggregate']['r2_negative_cases']}/{m['aggregate']['n_cases']}",
        m["field"]["rmse"], m["field"]["mae"], m["field"]["nrmse_range"], m["aggregate"]["nrmse_casemean"], m["field"].get("nmae_range"), m["aggregate"].get("nmae_casemean"),
        m["regional_field"]["high_wss"]["r2"], m["calibration"]["top10_pred_true_ratio"], m["hotspot"]["top10_iou_casemean"],
        n["field_casebalanced"]["r2"], n["field"]["r2"], n["aggregate"]["r2_casemean"], n["aggregate"]["r2_casemed"], n["aggregate"]["r2_casep10"],
        n["field_casebalanced"]["mae"], n["field_casebalanced"]["rmse"], n["field"]["nrmse_range"], n["aggregate"]["nrmse_casemean"], n["field"].get("nmae_range"), n["aggregate"].get("nmae_casemean"),
        m["hotspot"]["spearman_all_casemean"], m["hotspot"]["spearman_high_wss_casemean"], n["calibration"]["top10_pred_true_ratio"], n["calibration"]["p99_pred_true_ratio"],
        wid,
        m["field_casebalanced"].get("r2_linear_fit"), m["field_casebalanced"].get("linear_fit_slope"), m["field_casebalanced"].get("linear_fit_intercept"),
        m["field"].get("r2_linear_fit"), m["field"].get("linear_fit_slope"), m["field"].get("linear_fit_intercept"),
        n["field_casebalanced"].get("r2_linear_fit"), n["field_casebalanced"].get("linear_fit_slope"), n["field_casebalanced"].get("linear_fit_intercept"),
        n["field"].get("r2_linear_fit"), n["field"].get("linear_fit_slope"), n["field"].get("linear_fit_intercept"),
    ]


def teacher_values(row, overview_row: int):
    wid, label, unit, change, split, m, _ = row
    n = m["normalized"]
    # 字面值而非跨表公式，理由同 update_v5_rerun_xlsx。
    fit = [m["field_casebalanced"].get("r2_linear_fit"), m["field"].get("r2_linear_fit"),
           m["field"].get("linear_fit_slope"), m["field"].get("linear_fit_intercept")]
    return ["V5 体场目标", label, MODEL, "random5000 / 独立 query", m["field_casebalanced"]["r2"], m["aggregate"]["r2_casemean"], m["field"]["rmse"],
            m["field"].get("nmae_range"), m["aggregate"].get("nmae_casemean"), m["regional_field"]["high_wss"]["r2"], n["field_casebalanced"]["r2"],
            n["field"].get("nmae_range"), m["hotspot"]["spearman_all_casemean"], m["hotspot"]["top10_iou_casemean"], n["calibration"]["p99_pred_true_ratio"]] + fit


def main() -> None:
    for p in (P_RUN, V_RUN, R4_RUN):
        if not p.is_file():
            raise FileNotFoundError(p)
    data = rows()
    ids = [r[0] for r in data]
    labels = {r[0]: r[1] for r in data}
    if not BACKUP.exists():
        shutil.copy2(BOOK, BACKUP)
    with tempfile.TemporaryDirectory(prefix="wss_v5_vol_xlsx_") as tmp:
        cand = Path(tmp) / BOOK.name
        shutil.copy2(BOOK, cand)
        wb = load_workbook(cand, data_only=False)
        if wb.sheetnames != EXPECTED_SHEETS:
            raise RuntimeError(f"unexpected sheets {wb.sheetnames}")
        # ---- 实验矩阵总览: title row + rows
        ov = wb["实验矩阵总览"]
        stale = [c.row for c in ov["AK"] if c.value in set(ids)] + [c.row for c in ov["A"] if c.value == OVERVIEW_TITLE]
        if stale:
            unmerge_from(ov, min(stale))
        for r in sorted(set(stale), reverse=True):
            ov.delete_rows(r, 1)
        base_ov = ov.max_row
        r = ov.max_row + 1
        ov.merge_cells(start_row=r, start_column=1, end_row=r, end_column=20)
        cell = ov.cell(r, 1, OVERVIEW_TITLE)
        cell.font = Font(bold=True, color=WHITE)
        cell.fill = PatternFill("solid", fgColor=NAVY)
        cell.alignment = Alignment(vertical="center", wrap_text=True)
        ov.row_dimensions[r].height = 34
        ov_row_of = {}
        for row in data:
            r = ov.max_row + 1
            copy_row_style(ov, base_ov, r, 49)
            for c, v in enumerate(overview_values(row), 1):
                ov.cell(r, c, v)
            ov_row_of[row[0]] = r
        # ---- 教师汇报视图: section Ⅺ
        te = wb["教师汇报视图"]
        stale = [c.row for c in te["B"] if c.value in set(labels.values())] + [c.row for c in te["A"] if c.value == TEACHER_TITLE]
        if stale:
            unmerge_from(te, min(stale))
        stale += [c.row for c in te["A"] if c.value == "V5 体场目标" and te.cell(c.row, 2).value is None]
        for r in sorted(set(stale), reverse=True):
            te.delete_rows(r, 1)
        base_te = te.max_row
        header_src = max(c.row for c in te["A"] if isinstance(c.value, str) and c.value[:1] in "ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩⅪ")
        r = te.max_row + 1
        copy_row_style(te, header_src, r, 19)
        te.merge_cells(start_row=r, start_column=1, end_row=r, end_column=15)
        te.cell(r, 1, TEACHER_TITLE)
        for row in data:
            r = te.max_row + 1
            copy_row_style(te, base_te, r, 19)
            for c, v in enumerate(teacher_values(row, ov_row_of[row[0]]), 1):
                te.cell(r, c, v)
        # ---- 汇总对比: dated section with WSS R4 reference
        su = wb["汇总对比"]
        old = [c.row for c in su["A"] if c.value == SECTION_TITLE]
        if old:
            unmerge_from(su, old[0])
            delete_section(su, old[0])
        r = su.max_row + 1
        su.merge_cells(start_row=r, start_column=1, end_row=r, end_column=15)
        cell = su.cell(r, 1, SECTION_TITLE)
        cell.font = Font(bold=True, color=WHITE, size=12)
        cell.fill = PatternFill("solid", fgColor=NAVY)
        cell.alignment = Alignment(vertical="center")
        su.row_dimensions[r].height = 23
        r += 1
        headers = ["臂", "目标 / 单位 / 唯一变化", "R²_cb（物理=归一化）", "R² pooled", "case mean / P10", "MAE", "RMSE", "高值区(top10% 真值) R²", "top10 幅值比", "p99 比", "top10 IoU", "Spearman", "分域 R²_cb AG / AAA / ILO", "补充", "判定"]
        for c, h in enumerate(headers, 1):
            cell = su.cell(r, c, h)
            cell.font = Font(bold=True)
            cell.fill = PatternFill("solid", fgColor=BLUE)
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            cell.border = Border(top=THIN, bottom=THIN, left=THIN, right=THIN)
        su.row_dimensions[r].height = 36
        r += 1
        mp, mv, m4 = load_metrics(P_RUN), load_metrics(V_RUN), load_metrics(R4_RUN)
        vec = mv["vector"]
        def group_txt(m):
            g = m["group_casebalanced"]
            return f"{g['AG']['r2']:.3f} / {g['AAA']['r2']:.3f} / {g['ILO']['r2']:.3f}"
        def vals(label, unit_change, m, extra, verdict):
            return [label, unit_change, m["field_casebalanced"]["r2"], m["field"]["r2"],
                    f"{m['aggregate']['r2_casemean']:.4f} / {m['aggregate']['r2_casep10']:.4f}", m["field"]["mae"], m["field"]["rmse"],
                    m["regional_field"]["high_wss"]["r2"], m["calibration"]["top10_pred_true_ratio"], m["calibration"]["p99_pred_true_ratio"],
                    m["hotspot"]["top10_iou_casemean"], m["hotspot"]["spearman_all_casemean"], group_txt(m), extra, verdict]
        table = [
            (vals("参照：WSS R4 s1234（同配置，interpolate 解码）", "WSS Pa（log_z）", m4, "归一化 R²_cb 0.786（log 目标与物理不同）", "参照"), BLUE),
            (vals("体场·压力 p−p_ref（壁面∪内部）", "Pa 相对（p_ref=该例峰值步体积平均压）；query 混合 50/50；线性 z；无 pinball；qad_lite", mp,
                  f"壁面 {mp['query_groups']['wall']['field_casebalanced']['r2']:.3f} / 内部 {mp['query_groups']['interior']['field_casebalanced']['r2']:.3f}；3 例负 R² 为真值压差≈100 Pa 的病例",
                  "压力比 WSS 好学 +0.13；壁面与内部同分"), GREEN),
            (vals("体场·速度幅值 |u|（内部；模型出三分量）", "m/s；逐分量线性 z，out_dim=3；只评 |u|；qad_lite", mv,
                  f"分量 R²_cb u/v/w {vec['components']['u']['r2_casebalanced']:.3f}/{vec['components']['v']['r2_casebalanced']:.3f}/{vec['components']['w']['r2_casebalanced']:.3f}；方向余弦 {vec['direction_cosine_casemean']:.2f}（速度加权 {vec['direction_cosine_speedweighted_casemean']:.2f}）",
                  "速度比 WSS 好学 +0.18；高速区仍欠估"), YELLOW),
        ]
        for values, fill in table:
            for c, v in enumerate(values, 1):
                cell = su.cell(r, c, v)
                fmt = "0.0000" if c in (3, 4, 8, 9, 10, 11, 12) else ("0.000" if c in (6, 7) else None)
                styled(cell, fill, fmt)
            su.row_dimensions[r].height = 48
            r += 1
        if hasattr(wb, "calculation"):
            wb.calculation.fullCalcOnLoad = True
            wb.calculation.forceFullCalc = True
        wb.save(cand)
        wb.close()
        chk = load_workbook(cand, read_only=True, data_only=False)
        got_ids = {v[0] for v in chk["实验矩阵总览"].iter_rows(min_col=37, max_col=37, values_only=True)}
        got_labels = {v[0] for v in chk["教师汇报视图"].iter_rows(min_col=2, max_col=2, values_only=True)}
        titles = {v[0] for v in chk["汇总对比"].iter_rows(min_col=1, max_col=1, values_only=True)}
        assert set(ids) <= got_ids and set(labels.values()) <= got_labels and SECTION_TITLE in titles
        counts = (chk["实验矩阵总览"].max_row, chk["教师汇报视图"].max_row, chk["汇总对比"].max_row)
        chk.close()
        shutil.copy2(cand, BOOK)
    print(json.dumps({"rows_added": len(data), "overview_rows": ov_row_of, "sheet_max_rows": counts, "backup": str(BACKUP)}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
