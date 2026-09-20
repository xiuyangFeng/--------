#!/usr/bin/env python3
"""Backfill the WSS workbook with the bottleneck-transformer matrix (teacher method adapted to our features).

Kept as its own section, separate from the WSS rerun and volume-target sections:
* 实验矩阵总览: a title row + one row PER RUN (每个 seed 一行), consistent with how the V5 rerun seeds were logged;
* 教师汇报视图: section Ⅻ with one row PER ARM (multi-seed arms show the seed mean) plus the verdict;
* 汇总对比: a dated section comparing每个臂 against the R4 THREE-SEED MEAN with the correct difference bands.

Statistics follow tools/report_bt_matrix.py: the reference is R4's three-seed mean, the noise band is a band on a
difference (sd*sqrt(1/n_arm + 1/n_base)), and a single-seed arm gets no verdict. Idempotent.
"""
from __future__ import annotations

import json
import math
import shutil
import tempfile
from pathlib import Path

import numpy as np
from openpyxl import load_workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill

from training_wss_min.tools.update_v5_rerun_xlsx import (
    BLUE, BOOK, EXP, EXPECTED_SHEETS, GREEN, LSA2, NAVY, RUNS, THIN, WHITE, YELLOW,
    copy_row_style, load_metrics, styled, unmerge_from, delete_section,
)
from training_wss_min.tools.report_bt_matrix import ARMS, BASE_ARM, NORM, PHYS, SEEDS, dig, seeds_of, verdict_for

BACKUP = EXP / "WSS_PointNet实验矩阵与结果汇总last_backup_before_bt_20260908.xlsx"
OVERVIEW_TITLE = ("Bottleneck Transformer 矩阵（导师方法适配我们的几何特征，2026-09-08）—— 每行一个 run（含 seed）；"
                  "参照为 R4 三 seed 均值，判定带见 教师汇报视图 Ⅻ 节与 汇总对比")
TEACHER_TITLE = ("Ⅻ Bottleneck Transformer（2026-09-08；导师 model_train_1.py 的最粗层全局自注意力，"
                 "token 条件改用我们的 17 维几何特征；核心臂三 seed、变体单 seed 不判定。"
                 "本节每行 = 一个臂：多 seed 臂的所有指标（含线性拟合）均为该臂各 seed 的均值，"
                 "而「实验矩阵总览」是每个 run 一行，因此两处对同一臂的数值不同属正常）")
# 标题一旦改写，旧标题行不会再被完整匹配删除，因此统一按前缀去重。
TEACHER_TITLE_PREFIX = "Ⅻ Bottleneck Transformer"
SECTION_TITLE = "2026-09-08｜Bottleneck Transformer 矩阵：导师全局特征方法接入 R4（参照 = R4 全部 seed 的均值）"
# 旧标题：改过标题后仍要能删掉上一轮写入的分节，否则会重复追加。
LEGACY_SECTION_TITLES = ("2026-09-08｜Bottleneck Transformer 矩阵：导师全局特征方法接入 R4（参照 = R4 三 seed 均值）",)
MODEL = LSA2 + "（+ 最粗层瓶颈全局注意力）"
SPLIT = "V5 train138/test34；support 壁面 5000；seed={seed}"


def arm_rows():
    """[(label, stem, {seed: metrics}), ...] for the base arm and every BT arm that has at least one run."""
    out = [(BASE_ARM[0], BASE_ARM[1], seeds_of(BASE_ARM[1]))]
    for label, stem in ARMS:
        runs = seeds_of(stem)
        if runs:
            out.append((label, stem, runs))
    return out


def cfg_of(stem, seed):
    p = RUNS.parent.parent.parent / "configs" / "v5_rerun_20260906" / f"{stem}_s{seed}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}


def arm_shape(stem, seed):
    mc = cfg_of(stem, seed).get("model", {})
    tok = (mc.get("sa_center_counts") or [None, None, None])[-1]
    if mc.get("bottleneck_transformer"):
        layers = mc.get("bottleneck_layers", 1)
        attn = f"瓶颈×{layers}（{mc.get('bottleneck_heads', 8)}头）" if layers else "无（仅条件 MLP）"
        cond = {"input_features": "输入几何特征 MLP", "xyz": "绝对 xyz MLP", "none": "无"}.get(mc.get("bottleneck_pos_enc"), "?")
        if mc.get("bottleneck_geo_bias"):
            cond += "+相对几何偏置"
        gamma = mc.get("bottleneck_residual_scale_init") or mc.get("residual_scale_init")
        drop = mc.get("bottleneck_dropout", 0.0)
        extra = f"；γ0={gamma}；dropout={drop}"
    elif mc.get("coarse_attention"):
        attn, cond, extra = f"coarse_attention（{mc.get('coarse_attention_heads', 4)}头）", "相对几何偏置", ""
    elif tuple(mc.get("local_transformer_stages", ())) == (2, 3):
        attn, cond, extra = "SA2+SA3 局部 Transformer（无全局）", "—", ""
    else:
        attn, cond, extra = "无", "—", ""
    return tok, attn, cond, extra


def overview_values(label, stem, seed, m):
    n = m["normalized"]
    tok, attn, cond, extra = arm_shape(stem, seed)
    change = f"粗层 token {tok}；注意力 {attn}；token 条件 {cond}{extra}"
    return [
        f"{label}·s{seed}", SPLIT.format(seed=seed), MODEL, change, "random5000 / SAME；FPS center", 5000,
        m["field_casebalanced"]["r2"], m["field"]["r2"], m["aggregate"]["r2_casemean"], m["aggregate"]["r2_casemed"], m["aggregate"]["r2_casep10"],
        f"{m['aggregate']['r2_negative_cases']}/{m['aggregate']['n_cases']}",
        m["field"]["rmse"], m["field"]["mae"], m["field"]["nrmse_range"], m["aggregate"]["nrmse_casemean"], m["field"].get("nmae_range"), m["aggregate"].get("nmae_casemean"),
        m["regional_field"]["high_wss"]["r2"], m["calibration"]["top10_pred_true_ratio"], m["hotspot"]["top10_iou_casemean"],
        n["field_casebalanced"]["r2"], n["field"]["r2"], n["aggregate"]["r2_casemean"], n["aggregate"]["r2_casemed"], n["aggregate"]["r2_casep10"],
        n["field_casebalanced"]["mae"], n["field_casebalanced"]["rmse"], n["field"]["nrmse_range"], n["aggregate"]["nrmse_casemean"], n["field"].get("nmae_range"), n["aggregate"].get("nmae_casemean"),
        m["hotspot"]["spearman_all_casemean"], m["hotspot"]["spearman_high_wss_casemean"], n["calibration"]["top10_pred_true_ratio"], n["calibration"]["p99_pred_true_ratio"],
        f"bt_matrix/{stem}_s{seed}",
        m["field_casebalanced"].get("r2_linear_fit"), m["field_casebalanced"].get("linear_fit_slope"), m["field_casebalanced"].get("linear_fit_intercept"),
        m["field"].get("r2_linear_fit"), m["field"].get("linear_fit_slope"), m["field"].get("linear_fit_intercept"),
        n["field_casebalanced"].get("r2_linear_fit"), n["field_casebalanced"].get("linear_fit_slope"), n["field_casebalanced"].get("linear_fit_intercept"),
        n["field"].get("r2_linear_fit"), n["field"].get("linear_fit_slope"), n["field"].get("linear_fit_intercept"),
    ]


def main() -> None:
    rows = arm_rows()
    base = rows[0][2]
    if not base:
        raise SystemExit("R4 baseline runs missing")
    bp = np.array([dig(m, PHYS) for m in base.values()], float)
    bn = np.array([dig(m, NORM) for m in base.values()], float)
    n_base = len(bp)
    sd_p, sd_n = float(bp.std(ddof=1)), float(bn.std(ddof=1))
    if not BACKUP.exists():
        shutil.copy2(BOOK, BACKUP)
    ids = [f"bt_matrix/{stem}_s{s}" for _, stem, runs in rows[1:] for s in runs]
    labels = [f"{label}（{len(runs)} seed{' 均值' if len(runs) > 1 else ''}）" for label, stem, runs in rows]
    label_prefixes = [label for label, _, _ in rows]  # seed 数会随补种变化，去重必须按前缀
    with tempfile.TemporaryDirectory(prefix="wss_bt_xlsx_") as tmp:
        cand = Path(tmp) / BOOK.name
        shutil.copy2(BOOK, cand)
        wb = load_workbook(cand, data_only=False)
        if wb.sheetnames != EXPECTED_SHEETS:
            raise RuntimeError(f"unexpected sheets {wb.sheetnames}")
        # ---- 实验矩阵总览
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
        cell.font = Font(bold=True, color=WHITE); cell.fill = PatternFill("solid", fgColor=NAVY)
        cell.alignment = Alignment(vertical="center", wrap_text=True); ov.row_dimensions[r].height = 34
        ov_row_of = {}
        for label, stem, runs in rows[1:]:
            for seed in sorted(runs):
                r = ov.max_row + 1
                copy_row_style(ov, base_ov, r, 49)
                for c, v in enumerate(overview_values(label, stem, seed, runs[seed]), 1):
                    ov.cell(r, c, v)
                ov_row_of[f"{stem}_s{seed}"] = r
        # ---- 教师汇报视图 Ⅻ：每臂一行
        te = wb["教师汇报视图"]
        stale = [c.row for c in te["B"]
                 if isinstance(c.value, str) and any(c.value.startswith(pfx) for pfx in label_prefixes)]
        stale += [c.row for c in te["A"] if isinstance(c.value, str) and c.value.startswith(TEACHER_TITLE_PREFIX)]
        if stale:
            unmerge_from(te, min(stale))
        stale += [c.row for c in te["A"] if c.value == "V5 瓶颈全局注意力" and te.cell(c.row, 2).value is None]
        for r in sorted(set(stale), reverse=True):
            te.delete_rows(r, 1)
        base_te = te.max_row
        header_src = max(c.row for c in te["A"] if isinstance(c.value, str) and c.value[:1] in "ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩⅪⅫ")
        r = te.max_row + 1
        copy_row_style(te, header_src, r, 19)
        te.merge_cells(start_row=r, start_column=1, end_row=r, end_column=15)
        te.cell(r, 1, TEACHER_TITLE)
        for label, stem, runs in rows:
            ap = np.array([dig(m, PHYS) for m in runs.values()], float)
            an = np.array([dig(m, NORM) for m in runs.values()], float)
            ex = list(runs.values())[0]
            r = te.max_row + 1
            copy_row_style(te, base_te, r, 19)
            first_seed = sorted(runs)[0]
            ref = ov_row_of.get(f"{stem}_s{first_seed}")
            # 字面值（各 seed 的均值）而非跨表公式，理由同 update_v5_rerun_xlsx。
            fit = [float(np.mean([m["field_casebalanced"].get("r2_linear_fit") for m in runs.values()])),
                   float(np.mean([m["field"].get("r2_linear_fit") for m in runs.values()])),
                   float(np.mean([m["field"].get("linear_fit_slope") for m in runs.values()])),
                   float(np.mean([m["field"].get("linear_fit_intercept") for m in runs.values()]))]
            for c, v in enumerate(["V5 瓶颈全局注意力", f"{label}（{len(runs)} seed{' 均值' if len(runs) > 1 else ''}）", MODEL, "random5000 / SAME",
                                   float(ap.mean()), float(np.mean([m["aggregate"]["r2_casemean"] for m in runs.values()])),
                                   float(np.mean([m["field"]["rmse"] for m in runs.values()])),
                                   float(np.mean([m["field"]["nmae_range"] for m in runs.values()])),
                                   float(np.mean([m["aggregate"]["nmae_casemean"] for m in runs.values()])),
                                   float(np.mean([dig(m, ("regional_field", "high_wss", "r2")) for m in runs.values()])),
                                   float(an.mean()),
                                   float(np.mean([dig(m, ("normalized", "field", "nmae_range")) for m in runs.values()])),
                                   float(np.mean([m["hotspot"]["spearman_all_casemean"] for m in runs.values()])),
                                   float(np.mean([m["hotspot"]["top10_iou_casemean"] for m in runs.values()])),
                                   float(np.mean([dig(m, ("normalized", "calibration", "p99_pred_true_ratio")) for m in runs.values()]))] + fit, 1):
                te.cell(r, c, v)
        # ---- 汇总对比
        su = wb["汇总对比"]
        old = [c.row for c in su["A"] if c.value in (SECTION_TITLE,) + LEGACY_SECTION_TITLES]
        if old:
            unmerge_from(su, old[0])
            end = su.max_row
            delete_section(su, old[0])  # 本节永远是表尾，整段删净再重建
        r = su.max_row + 1
        su.merge_cells(start_row=r, start_column=1, end_row=r, end_column=15)
        cell = su.cell(r, 1, SECTION_TITLE)
        cell.font = Font(bold=True, color=WHITE, size=12); cell.fill = PatternFill("solid", fgColor=NAVY)
        cell.alignment = Alignment(vertical="center"); su.row_dimensions[r].height = 23
        r += 1
        band_note = (f"噪声由 R4 {n_base} 个 seed 现场标定：物理 sd={sd_p:.4f}、归一化 sd={sd_n:.4f}；"
                     f"单 seed 对单 seed 95% 带 ±{1.96*sd_p*math.sqrt(2):.3f}/±{1.96*sd_n*math.sqrt(2):.3f}；"
                     f"三 seed 均值之间 ±{1.96*sd_p*math.sqrt(2/3):.3f}/±{1.96*sd_n*math.sqrt(2/3):.3f}。判定需两指标方向一致且各自超带。")
        su.merge_cells(start_row=r, start_column=1, end_row=r, end_column=15)
        cell = su.cell(r, 1, band_note); cell.font = Font(italic=True); cell.alignment = Alignment(vertical="center", wrap_text=True)
        su.row_dimensions[r].height = 30
        r += 1
        headers = ["臂", "结构（粗 token / 注意力 / 条件）", "seed 数", "物理 R²_cb", "Δ物理", "归一化 R²_cb", "Δ归一化",
                   "逐例均值", "MAE (Pa)", "high-WSS R²", "top10 比", "top10 IoU", "分域 AG/AAA/ILO", "学到的 γ", "判定"]
        for c, h in enumerate(headers, 1):
            cell = su.cell(r, c, h); cell.font = Font(bold=True); cell.fill = PatternFill("solid", fgColor=BLUE)
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            cell.border = Border(top=THIN, bottom=THIN, left=THIN, right=THIN)
        su.row_dimensions[r].height = 36
        r += 1
        # γ（学到的残差强度）来自 ckpt，需要 torch；本工具跑在 /usr/bin/python3（有 openpyxl 无 torch），
        # 因此优先读 GNN 环境预先导出的旁路 JSON，读不到再尝试 torch。
        gamma_side = {}
        side = EXP / "bt_gammas.json"
        if side.is_file():
            gamma_side = json.loads(side.read_text(encoding="utf-8"))
        for label, stem, runs in rows:
            ap = np.array([dig(m, PHYS) for m in runs.values()], float)
            an = np.array([dig(m, NORM) for m in runs.values()], float)
            n = len(ap)
            dp, dn = ap.mean() - bp.mean(), an.mean() - bn.mean()
            verdict, _, _ = verdict_for(dp, dn, n, n_base, sd_p, sd_n, is_base=(stem == BASE_ARM[1]))
            verdict = verdict.replace("**", "")
            fill = BLUE if stem == BASE_ARM[1] else (GREEN if "提升" in verdict else YELLOW)
            seed0 = sorted(runs)[0]
            tok, attn, cond, extra = arm_shape(stem, seed0)
            gtxt = gamma_side.get(f"{stem}_s{seed0}", "—")
            grp = {k: float(np.mean([dig(m, ("group_casebalanced", k, "r2")) for m in runs.values()])) for k in ("AG", "AAA", "ILO")}
            values = [label, f"{tok} / {attn} / {cond}{extra}", n, float(ap.mean()), None if stem == BASE_ARM[1] else float(dp),
                      float(an.mean()), None if stem == BASE_ARM[1] else float(dn),
                      float(np.mean([m["aggregate"]["r2_casemean"] for m in runs.values()])),
                      float(np.mean([m["field"]["mae"] for m in runs.values()])),
                      float(np.mean([dig(m, ("regional_field", "high_wss", "r2")) for m in runs.values()])),
                      float(np.mean([m["calibration"]["top10_pred_true_ratio"] for m in runs.values()])),
                      float(np.mean([m["hotspot"]["top10_iou_casemean"] for m in runs.values()])),
                      f"{grp['AG']:.3f} / {grp['AAA']:.3f} / {grp['ILO']:.3f}", gtxt, verdict]
            for c, v in enumerate(values, 1):
                cell = su.cell(r, c, v)
                fmt = None
                if c in (4, 6, 8, 10, 11, 12):
                    fmt = "0.0000"
                elif c in (5, 7):
                    fmt = "+0.0000;-0.0000;0.0000"
                elif c == 9:
                    fmt = "0.000"
                styled(cell, fill, fmt)
            su.row_dimensions[r].height = 44
            r += 1
        if hasattr(wb, "calculation"):
            wb.calculation.fullCalcOnLoad = True
            wb.calculation.forceFullCalc = True
        wb.save(cand); wb.close()
        chk = load_workbook(cand, read_only=True, data_only=False)
        got = {v[0] for v in chk["实验矩阵总览"].iter_rows(min_col=37, max_col=37, values_only=True)}
        assert set(ids) <= got, sorted(set(ids) - got)[:3]
        counts = (chk["实验矩阵总览"].max_row, chk["教师汇报视图"].max_row, chk["汇总对比"].max_row)
        chk.close()
        shutil.copy2(cand, BOOK)
    print(json.dumps({"runs_written": len(ids), "arms": len(rows), "overview_rows": ov_row_of,
                      "sheet_max_rows": counts, "backup": str(BACKUP)}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
