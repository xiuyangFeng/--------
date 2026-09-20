#!/usr/bin/env python3
"""Backfill the compact WSS workbook with the V5 rerun (new data) results.

Adds to 《WSS_PointNet实验矩阵与结果汇总last.xlsx》:
* 实验矩阵总览: one row per new run (best checkpoint, test34) and one row per legacy
  checkpoint re-evaluated read-only on the 34-case overlap (reference);
* 教师汇报视图: a new section Ⅹ with the same rows (P..S formulas point at the overview rows);
* 汇总对比: a dated section with new-vs-legacy deltas per arm.
Idempotent: previously written V5 rows/sections are removed before re-adding.
Uses /usr/bin/python3 (openpyxl); a backup of the workbook is written next to the results.
"""
from __future__ import annotations

import json
import shutil
import tempfile
from copy import copy
from pathlib import Path

from openpyxl import load_workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side

ROOT = Path(__file__).resolve().parents[2]
BOOK = ROOT / "docs/03-汇报材料/WSS_PointNet实验矩阵与结果汇总last.xlsx"
RUNS = ROOT / "training_wss_min/runs"
EXP = ROOT / "training_wss_min/experiments/v5_rerun_20260906"
BACKUP = EXP / "WSS_PointNet实验矩阵与结果汇总last_backup_before_v5_20260907.xlsx"
EXPECTED_SHEETS = ["汇总对比", "教师汇报视图", "RCR Oracle", "指标说明", "实验矩阵总览"]
SECTION_TITLE = "2026-09-07｜V5 新数据重跑（train138/test34）：旧骨干只换数据 + R4 V5 几何特征"
TEACHER_TITLE = "Ⅹ V5 新数据重跑（2026-09-07；train138/test34；旧骨干只换数据 + R4 V5 特征）"
NAVY, BLUE, GREEN, YELLOW, WHITE = "1F4E78", "D9EAF7", "E2F0D9", "FFF2CC", "FFFFFF"
THIN = Side(style="thin", color="B7B7B7")

LSA2 = "PointNeXt-R + LocalGeoPE + L-SA2"
S3 = "PointNeXt-R + LocalGeoPE"
V5_SPLIT = "V5 train138/test34（AG/AAA/ILO；anatomy_pointcloud_v5 新数据；seed={seed}）"
LEGACY_SPLIT = "旧 138/0/36 训练的 checkpoint 在 test34 交集上只读重评（旧 data_wss_min）"

# (workbook id, label, model, change, split, metrics path, seed, reference id or None)
ROWS = [
    ("legacy_overlap34/r1", "旧ckpt·R1 L-SA2+H2+logR s1234·交集34", LSA2, "旧模型参照：MSE+q90 pinball λ0.20，7D 含 ln(local_radius)", LEGACY_SPLIT, EXP / "legacy_overlap34/r1/metrics.json", 1234, None),
    ("legacy_overlap34/r2", "旧ckpt·R2 L-SA2 MSE s1234·交集34", LSA2, "旧模型参照：纯 MSE，6D", LEGACY_SPLIT, EXP / "legacy_overlap34/r2/metrics.json", 1234, None),
    ("legacy_overlap34/r3", "旧ckpt·R3 S3-GEOPE MSE s1234·交集34", S3, "旧模型参照：S3-GEOPE，MSE，6D", LEGACY_SPLIT, EXP / "legacy_overlap34/r3/metrics.json", 1234, None),
    ("legacy_overlap34/r3s7", "旧ckpt·R3 S3-GEOPE MSE s7·交集34", S3, "旧模型参照：S3-GEOPE，MSE，6D", LEGACY_SPLIT, EXP / "legacy_overlap34/r3s7/metrics.json", 7, None),
    ("legacy_overlap34/r3s2025", "旧ckpt·R3 S3-GEOPE MSE s2025·交集34", S3, "旧模型参照：S3-GEOPE，MSE，6D", LEGACY_SPLIT, EXP / "legacy_overlap34/r3s2025/metrics.json", 2025, None),
    ("v5_rerun_20260906/r1_lsa2_h2_logradius_s1234", "V5·R1 L-SA2+H2+logR s1234", LSA2, "只换数据：MSE+q90 pinball λ0.20，7D 含 ln(local_radius)", V5_SPLIT, RUNS / "v5_rerun_20260906/outputs/r1_lsa2_h2_logradius_s1234/eval/ckpt_best/metrics.json", 1234, "legacy_overlap34/r1"),
    ("v5_rerun_20260906/r1_lsa2_h2_logradius_s7", "V5·R1 L-SA2+H2+logR s7", LSA2, "只换数据：同 R1，seed 7", V5_SPLIT, RUNS / "v5_rerun_20260906/outputs/r1_lsa2_h2_logradius_s7/eval/ckpt_best/metrics.json", 7, "legacy_overlap34/r1"),
    ("v5_rerun_20260906/r1_lsa2_h2_logradius_s2025", "V5·R1 L-SA2+H2+logR s2025", LSA2, "只换数据：同 R1，seed 2025", V5_SPLIT, RUNS / "v5_rerun_20260906/outputs/r1_lsa2_h2_logradius_s2025/eval/ckpt_best/metrics.json", 2025, "legacy_overlap34/r1"),
    ("v5_rerun_20260906/r2_lsa2_mse_s1234", "V5·R2 L-SA2 MSE s1234", LSA2, "只换数据：纯 MSE，6D", V5_SPLIT, RUNS / "v5_rerun_20260906/outputs/r2_lsa2_mse_s1234/eval/ckpt_best/metrics.json", 1234, "legacy_overlap34/r2"),
    ("v5_rerun_20260906/r3_s3_geope_mse_s1234", "V5·R3 S3-GEOPE MSE s1234", S3, "只换数据：S3-GEOPE，MSE，6D", V5_SPLIT, RUNS / "v5_rerun_20260906/outputs/r3_s3_geope_mse_s1234/eval/ckpt_best/metrics.json", 1234, "legacy_overlap34/r3"),
    ("v5_rerun_20260906/r3_s3_geope_mse_s7", "V5·R3 S3-GEOPE MSE s7", S3, "只换数据：同 R3，seed 7", V5_SPLIT, RUNS / "v5_rerun_20260906/outputs/r3_s3_geope_mse_s7/eval/ckpt_best/metrics.json", 7, "legacy_overlap34/r3s7"),
    ("v5_rerun_20260906/r3_s3_geope_mse_s2025", "V5·R3 S3-GEOPE MSE s2025", S3, "只换数据：同 R3，seed 2025", V5_SPLIT, RUNS / "v5_rerun_20260906/outputs/r3_s3_geope_mse_s2025/eval/ckpt_best/metrics.json", 2025, "legacy_overlap34/r3s2025"),
    ("v5_rerun_20260906/r4_lsa2_h2_logradius_v5feat_s1234", "★ V5·R4 R1 + V5 十维几何特征 s1234", LSA2, "R1 配方 + ρ/sinθ/cosθ/dR·ds/到分叉·端点距离/端区/对齐 PCA 法向，17D", V5_SPLIT, RUNS / "v5_rerun_20260906/outputs/r4_lsa2_h2_logradius_v5feat_s1234/eval/ckpt_best/metrics.json", 1234, "v5_rerun_20260906/r1_lsa2_h2_logradius_s1234"),
    ("v5_rerun_20260906/r4_lsa2_h2_logradius_v5feat_s7", "V5·R4 R1 + V5 十维几何特征 s7", LSA2, "同 R4，seed 7（Wave 3 复现）", V5_SPLIT, RUNS / "v5_rerun_20260906/outputs/r4_lsa2_h2_logradius_v5feat_s7/eval/ckpt_best/metrics.json", 7, "v5_rerun_20260906/r4_lsa2_h2_logradius_v5feat_s1234"),
    ("v5_rerun_20260906/r4_lsa2_h2_logradius_v5feat_s2025", "V5·R4 R1 + V5 十维几何特征 s2025", LSA2, "同 R4，seed 2025（Wave 3 复现）", V5_SPLIT, RUNS / "v5_rerun_20260906/outputs/r4_lsa2_h2_logradius_v5feat_s2025/eval/ckpt_best/metrics.json", 2025, "v5_rerun_20260906/r4_lsa2_h2_logradius_v5feat_s1234"),
    ("v5_rerun_20260906/r4abl_notube_s1234", "V5·R4 消融：去管道坐标 ρ/sinθ/cosθ（14D）", LSA2, "留一组消融：R4 去掉 rho/theta_sin/theta_cos", V5_SPLIT, RUNS / "v5_rerun_20260906/outputs/r4abl_notube_s1234/eval/ckpt_best/metrics.json", 1234, "v5_rerun_20260906/r4_lsa2_h2_logradius_v5feat_s1234"),
    ("v5_rerun_20260906/r4abl_nodist_s1234", "V5·R4 消融：去分叉/端点距离+端区（14D）", LSA2, "留一组消融：R4 去掉 dist_to_junction/dist_to_endpoint/end_zone", V5_SPLIT, RUNS / "v5_rerun_20260906/outputs/r4abl_nodist_s1234/eval/ckpt_best/metrics.json", 1234, "v5_rerun_20260906/r4_lsa2_h2_logradius_v5feat_s1234"),
    ("v5_rerun_20260906/r4abl_nodrds_s1234", "V5·R4 消融：去 dR/ds（16D）", LSA2, "留一组消融：R4 去掉 dr_ds", V5_SPLIT, RUNS / "v5_rerun_20260906/outputs/r4abl_nodrds_s1234/eval/ckpt_best/metrics.json", 1234, "v5_rerun_20260906/r4_lsa2_h2_logradius_v5feat_s1234"),
    ("v5_rerun_20260906/r4abl_nonormal_s1234", "V5·R4 消融：去对齐 PCA 法向（14D）", LSA2, "留一组消融：R4 去掉 nx/ny/nz_aligned", V5_SPLIT, RUNS / "v5_rerun_20260906/outputs/r4abl_nonormal_s1234/eval/ckpt_best/metrics.json", 1234, "v5_rerun_20260906/r4_lsa2_h2_logradius_v5feat_s1234"),
    ("v5_rerun_20260906/r4tail_q95_s1234", "V5·R4 尾部目标：pinball q95 λ0.20", LSA2, "R4 配方，pinball 分位 0.90→0.95", V5_SPLIT, RUNS / "v5_rerun_20260906/outputs/r4tail_q95_s1234/eval/ckpt_best/metrics.json", 1234, "v5_rerun_20260906/r4_lsa2_h2_logradius_v5feat_s1234"),
    ("v5_rerun_20260906/r4tail_q90l05_s1234", "V5·R4 尾部目标：pinball q90 λ0.50", LSA2, "R4 配方，pinball 权重 0.20→0.50", V5_SPLIT, RUNS / "v5_rerun_20260906/outputs/r4tail_q90l05_s1234/eval/ckpt_best/metrics.json", 1234, "v5_rerun_20260906/r4_lsa2_h2_logradius_v5feat_s1234"),
    ("v5_rerun_20260906/r4nll_s1234", "V5·R4 高斯 NLL 头（μ 通道评估）", LSA2, "R4 配方，loss=gaussian_nll（μ、logσ² 双通道），无 pinball；表内为 μ 直接回变换的指标", V5_SPLIT, RUNS / "v5_rerun_20260906/outputs/r4nll_s1234/eval/ckpt_best/metrics.json", 1234, "v5_rerun_20260906/r4_lsa2_h2_logradius_v5feat_s1234"),
    ("v5_rerun_20260906/r4_lsa2_h2_logradius_v5feat_s3407", "V5·R4 R1 + V5 十维几何特征 s3407", LSA2, "同 R4，seed 3407（2026-09-08 为 BT 矩阵配对检验补种，使 R4 基准达 5 seed）", V5_SPLIT, RUNS / "v5_rerun_20260906/outputs/r4_lsa2_h2_logradius_v5feat_s3407/eval/ckpt_best/metrics.json", 3407, "v5_rerun_20260906/r4_lsa2_h2_logradius_v5feat_s1234"),
    ("v5_rerun_20260906/r4_lsa2_h2_logradius_v5feat_s7919", "V5·R4 R1 + V5 十维几何特征 s7919", LSA2, "同 R4，seed 7919（2026-09-08 为 BT 矩阵配对检验补种，使 R4 基准达 5 seed）", V5_SPLIT, RUNS / "v5_rerun_20260906/outputs/r4_lsa2_h2_logradius_v5feat_s7919/eval/ckpt_best/metrics.json", 7919, "v5_rerun_20260906/r4_lsa2_h2_logradius_v5feat_s1234"),
    ("v5_rerun_20260906/t1_lsa2_h2_logradius_time_s1234", "V5·T1 = R1 + t（随机帧训练，表内为峰值帧 1162 口径）", LSA2, "R1 + 协议波形相位特征 q_norm/dq_norm/t_sin/t_cos（11D）；timesteps=random_frame，每例每 epoch 随机 1 帧（峰值 1/9 保底）；全 81 帧 log_z 统计（floor 0.05 Pa）；81 帧 pooled 0.502、TAWSS R² 0.443 见跟踪文档 §10", V5_SPLIT, RUNS / "v5_rerun_20260906/outputs/t1_lsa2_h2_logradius_time_s1234/eval/ckpt_best/metrics.json", 1234, "v5_rerun_20260906/r1_lsa2_h2_logradius_s1234"),
    ("v5_rerun_20260906/t4_lsa2_h2_logradius_v5feat_time_s1234", "V5·T4 = R4 + t（随机帧训练，表内为峰值帧 1162 口径）", LSA2, "R4 + 协议波形相位特征（21D）；其余同 T1；81 帧 pooled 0.500、TAWSS R² 0.410；峰值帧掉 0.11 主因幅值收缩（峰值帧仅占训练样本 12%）", V5_SPLIT, RUNS / "v5_rerun_20260906/outputs/t4_lsa2_h2_logradius_v5feat_time_s1234/eval/ckpt_best/metrics.json", 1234, "v5_rerun_20260906/r4_lsa2_h2_logradius_v5feat_s1234"),
]
IDS = [r[0] for r in ROWS]
LABELS = {r[0]: r[1] for r in ROWS}


def load_metrics(path: Path) -> dict:
    m = json.loads(path.read_text(encoding="utf-8"))
    return m.get("test", m)


def overview_values(row) -> list:
    wid, label, model, change, split, path, seed, _ = row
    m = load_metrics(path)
    n = m["normalized"]
    return [
        label, split.format(seed=seed), model, change, "random5000 / SAME；FPS center", 5000,
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


def teacher_values(row, overview_row: int) -> list:
    wid, label, model, change, split, path, seed, _ = row
    m = load_metrics(path)
    n = m["normalized"]
    stage = "V5 参照（旧 ckpt·34 例）" if wid.startswith("legacy") else "V5 新数据重跑"
    # 字面值而非跨表公式：openpyxl 的 delete_rows 不会重写公式引用，后续任一分节重建都会让这些
    # 公式指向错误的行（2026-09-08 实测 186 个公式里 30 个指错）。这里直接写数值。
    fit = [m["field_casebalanced"].get("r2_linear_fit"), m["field"].get("r2_linear_fit"),
           m["field"].get("linear_fit_slope"), m["field"].get("linear_fit_intercept")]
    return [stage, label, model, "random5000 / SAME", m["field_casebalanced"]["r2"], m["aggregate"]["r2_casemean"], m["field"]["rmse"],
            m["field"].get("nmae_range"), m["aggregate"].get("nmae_casemean"), m["regional_field"]["high_wss"]["r2"], n["field_casebalanced"]["r2"],
            n["field"].get("nmae_range"), m["hotspot"]["spearman_all_casemean"], m["hotspot"]["top10_iou_casemean"], n["calibration"]["p99_pred_true_ratio"]] + fit


def copy_row_style(ws, source_row: int, target_row: int, columns: int) -> None:
    ws.row_dimensions[target_row].height = ws.row_dimensions[source_row].height
    for col in range(1, columns + 1):
        source, target = ws.cell(source_row, col), ws.cell(target_row, col)
        if source.has_style:
            target._style = copy(source._style)
        target.alignment = copy(source.alignment)


def delete_section(ws, start_row: int) -> int:
    """删除「汇总对比」里从 start_row 开始的一整个分节：一直删到下一个分节标题之前（或表尾）。

    不能用 `2 + len(ROWS)` 这类"按当前行数删"的写法：那是用**新**的分节长度去删**旧**的分节，
    只要行数变过（补种、加臂）就会多删下一节的标题或少删残行。分节标题以 '2026-' 开头。
    """
    end = ws.max_row
    for r in range(start_row + 1, ws.max_row + 1):
        v = ws.cell(r, 1).value
        if isinstance(v, str) and v.startswith("2026-"):
            end = r - 1
            break
    n = end - start_row + 1
    ws.delete_rows(start_row, n)
    return n


def unmerge_from(ws, first_row: int) -> None:
    """openpyxl 删除行时不会移动合并区，必须先把待改区域（first_row 及以后）的合并区全部拆掉，
    否则旧合并区会盖住后来的数据行并清空其单元格。"""
    for rng in [r for r in ws.merged_cells.ranges if r.min_row >= first_row]:
        ws.unmerge_cells(str(rng))


def first_row_with(ws, col_letter: str, values: set) -> int | None:
    rows = [c.row for c in ws[col_letter] if c.value in values]
    return min(rows) if rows else None


def styled(cell, fill, fmt=None, bold=False, wrap=True):
    cell.fill = PatternFill("solid", fgColor=fill)
    cell.border = Border(bottom=THIN)
    cell.alignment = Alignment(vertical="center", wrap_text=wrap)
    if bold:
        cell.font = Font(bold=True)
    if fmt:
        cell.number_format = fmt


def main() -> None:
    for row in ROWS:
        if not row[5].is_file():
            raise FileNotFoundError(row[5])
    BACKUP.parent.mkdir(parents=True, exist_ok=True)
    if not BACKUP.exists():
        shutil.copy2(BOOK, BACKUP)
    with tempfile.TemporaryDirectory(prefix="wss_v5_xlsx_") as tmp:
        cand = Path(tmp) / BOOK.name
        shutil.copy2(BOOK, cand)
        wb = load_workbook(cand, data_only=False)
        if wb.sheetnames != EXPECTED_SHEETS:
            raise RuntimeError(f"unexpected sheets {wb.sheetnames}")
        # ---- 实验矩阵总览
        ov = wb["实验矩阵总览"]
        first = first_row_with(ov, "AK", set(IDS))
        if first is not None:
            unmerge_from(ov, first)
        for r in sorted([c.row for c in ov["AK"] if c.value in set(IDS)], reverse=True):
            ov.delete_rows(r, 1)
        base_ov = ov.max_row
        ov_row_of = {}
        for row in ROWS:
            r = ov.max_row + 1
            copy_row_style(ov, base_ov, r, 49)
            for c, v in enumerate(overview_values(row), 1):
                ov.cell(r, c, v)
            ov_row_of[row[0]] = r
        # ---- 教师汇报视图
        te = wb["教师汇报视图"]
        stale = [c.row for c in te["B"] if c.value in set(LABELS.values())] + [c.row for c in te["A"] if c.value == TEACHER_TITLE]
        if stale:
            unmerge_from(te, min(stale))
        # 被旧合并区清空的残行：A 列仍是本节 stage 文本但 B 列已空
        stale += [c.row for c in te["A"] if c.value in {"V5 新数据重跑", "V5 参照（旧 ckpt·34 例）"} and te.cell(c.row, 2).value is None]
        for r in sorted(set(stale), reverse=True):
            te.delete_rows(r, 1)
        base_te = te.max_row
        header_src = max(c.row for c in te["A"] if isinstance(c.value, str) and c.value[:1] in "ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ")
        r = te.max_row + 1
        copy_row_style(te, header_src, r, 19)
        te.merge_cells(start_row=r, start_column=1, end_row=r, end_column=15)
        te.cell(r, 1, TEACHER_TITLE)
        for row in ROWS:
            r = te.max_row + 1
            copy_row_style(te, base_te, r, 19)
            for c, v in enumerate(teacher_values(row, ov_row_of[row[0]]), 1):
                te.cell(r, c, v)
        # ---- 汇总对比 section
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
        headers = ["臂", "唯一变化", "R²_cb（物理）", "ΔR²_cb vs 参照", "R²_cb（归一化）", "Δ归一化", "MAE (Pa)", "ΔMAE", "high-WSS R²", "Δhigh-WSS", "top10 IoU", "ΔIoU", "case mean / P10", "参照", "判定"]
        for c, h in enumerate(headers, 1):
            cell = su.cell(r, c, h)
            cell.font = Font(bold=True)
            cell.fill = PatternFill("solid", fgColor=BLUE)
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            cell.border = Border(top=THIN, bottom=THIN, left=THIN, right=THIN)
        su.row_dimensions[r].height = 36
        r += 1
        metrics = {row[0]: load_metrics(row[5]) for row in ROWS}
        for row in ROWS:
            wid, label, _, change, _, _, _, ref = row
            m = metrics[wid]
            k = lambda mm: (mm["field_casebalanced"]["r2"], mm["normalized"]["field_casebalanced"]["r2"], mm["field"]["mae"], mm["regional_field"]["high_wss"]["r2"], mm["hotspot"]["top10_iou_casemean"])
            cur = k(m)
            if ref:
                base = k(metrics[ref])
                deltas = [cur[i] - base[i] for i in range(5)]
                ref_txt = LABELS[ref]
                if "_time_" in wid:
                    verdict = "加时间维 t（随机帧训练）后的峰值帧口径；对比同配方单帧 run"
                    fill = BLUE
                else:
                    verdict = "R4 vs R1：新增几何特征增益" if wid.endswith("v5feat_s1234") else "同配方只换数据"
                    fill = YELLOW if wid.endswith("v5feat_s1234") else GREEN
            else:
                deltas = [None] * 5
                ref_txt = "—"
                verdict = "参照（旧数据、旧模型）"
                fill = BLUE
            values = [label, change, cur[0], deltas[0], cur[1], deltas[1], cur[2], deltas[2], cur[3], deltas[3], cur[4], deltas[4],
                      f"{m['aggregate']['r2_casemean']:.4f} / {m['aggregate']['r2_casep10']:.4f}", ref_txt, verdict]
            for c, v in enumerate(values, 1):
                cell = su.cell(r, c, v)
                fmt = None
                if 3 <= c <= 12:
                    fmt = "+0.0000;-0.0000;0.0000" if c % 2 == 0 else "0.0000"
                styled(cell, fill, fmt)
            su.row_dimensions[r].height = 40
            r += 1
        if hasattr(wb, "calculation"):
            wb.calculation.fullCalcOnLoad = True
            wb.calculation.forceFullCalc = True
        wb.save(cand)
        wb.close()
        chk = load_workbook(cand, read_only=True, data_only=False)
        ids = {v[0] for v in chk["实验矩阵总览"].iter_rows(min_col=37, max_col=37, values_only=True)}
        labels = {v[0] for v in chk["教师汇报视图"].iter_rows(min_col=2, max_col=2, values_only=True)}
        titles = {v[0] for v in chk["汇总对比"].iter_rows(min_col=1, max_col=1, values_only=True)}
        assert set(IDS) <= ids and set(LABELS.values()) <= labels and SECTION_TITLE in titles
        counts = (chk["实验矩阵总览"].max_row, chk["教师汇报视图"].max_row, chk["汇总对比"].max_row)
        chk.close()
        shutil.copy2(cand, BOOK)
    print(json.dumps({"rows_added": len(ROWS), "overview_rows": ov_row_of, "sheet_max_rows": counts, "backup": str(BACKUP)}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
