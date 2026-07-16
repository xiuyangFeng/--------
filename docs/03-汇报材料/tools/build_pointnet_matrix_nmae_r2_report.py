#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""为老师打包：NMAE 表 + R² regression 散点图，并回填汇总 xlsx。

口径：
  NMAE_range_pooled = MAE / (max(true)-min(true))，test 全点
  NMAE_range_casemean = mean_i [ MAE_i / (max_i-min_i) ]
  物理 Pa：GLOBAL 用 wss_cfd / wss_pred；CASE 无物理恢复，仅报 norm / case-max
  归一化空间：true_norm / pred_norm（GLOBAL=log-z；CASE=WSS/WSSmax）
"""
from __future__ import annotations

import json
from copy import copy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import openpyxl
from openpyxl.styles import Alignment, Font, PatternFill, Border, Side
from openpyxl.utils import get_column_letter

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "docs/03-汇报材料/figures/WSS_PointNet矩阵_NMAE与R2_20260715"
XLSX = ROOT / "docs/03-汇报材料/WSS_PointNet实验矩阵与结果汇总.xlsx"

RUNS = {
    "E0-GLOBAL": {
        "root": ROOT / "training_wss_min/runs/pointnet_trainloss_e400/outputs/pointnet_xyzgeom",
        "metrics": "eval/metrics.json",
        "postview": None,  # 无 PostView；物理 NMAE 从 metrics 推算
        "eval_npz": OUT / "e0_global_test.npz",  # 既有 eval-mode 全点重推理缓存
        "has_physical": True,
        "norm_mode": "global_log_z",
    },
    "E2-GLOBAL": {
        "root": ROOT / "training_wss_min/runs/pointnet_distribution_matrix/outputs/e2_global",
        "metrics": "eval/ckpt_best/metrics.json",
        "postview": "postview/ckpt_best/test",
        "has_physical": True,
        "norm_mode": "global_log_z",
    },
    "E3-GLOBAL": {
        "root": ROOT / "training_wss_min/runs/pointnet_distribution_matrix/outputs/e3_global",
        "metrics": "eval/ckpt_best/metrics.json",
        "postview": "postview/ckpt_best/test",
        "has_physical": True,
        "norm_mode": "global_log_z",
    },
    "E23-GLOBAL": {
        "root": ROOT / "training_wss_min/runs/pointnet_distribution_matrix/outputs/e23_global",
        "metrics": "eval/ckpt_best/metrics.json",
        "postview": "postview/ckpt_best/test",
        "has_physical": True,
        "norm_mode": "global_log_z",
    },
    "E4-DEEP-GLOBAL": {
        "root": ROOT / "training_wss_min/runs/pointnet_deeper/outputs/e4_deep_global",
        "metrics": "eval/ckpt_best/metrics.json",
        "postview": "postview/ckpt_best/test",
        "has_physical": True,
        "norm_mode": "global_log_z",
    },
    "E2-CASE": {
        "root": ROOT / "training_wss_min/runs/pointnet_distribution_matrix/outputs/e2_case",
        "metrics": "eval/ckpt_best/metrics.json",
        "postview": "postview/ckpt_best/test",
        "has_physical": False,
        "norm_mode": "case_max",
    },
    "E3-CASE": {
        "root": ROOT / "training_wss_min/runs/pointnet_distribution_matrix/outputs/e3_case",
        "metrics": "eval/ckpt_best/metrics.json",
        "postview": "postview/ckpt_best/test",
        "has_physical": False,
        "norm_mode": "case_max",
    },
}


def _r2(y_true, y_pred):
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - yt.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan")


def _nmae(y_true, y_pred):
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    mae = float(np.mean(np.abs(yt - yp)))
    rng = float(yt.max() - yt.min())
    nmae = mae / rng if rng > 1e-12 else float("nan")
    return mae, rng, nmae


def _nmae_from_field(field):
    mae = float(field["mae"])
    rmse = float(field["rmse"])
    nrmse = float(field.get("nrmse_range", float("nan")))
    if not np.isfinite(nrmse) or nrmse <= 0:
        return mae, float("nan"), float("nan")
    rng = rmse / nrmse
    return mae, rng, mae / rng


def load_wall_cases(postview_dir: Path):
    rows = []
    for csv_path in sorted(postview_dir.glob("*__peak_wss/_export/*__wall.csv")):
        data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
        names = set(data.dtype.names or ())
        label = csv_path.name.replace("__wall.csv", "")
        row = {
            "case": label,
            "true_norm": np.asarray(data["true_norm"], dtype=np.float64),
            "pred_norm": np.asarray(data["pred_norm"], dtype=np.float64),
        }
        if "wss_cfd" in names and "wss_pred" in names:
            row["wss_cfd"] = np.asarray(data["wss_cfd"], dtype=np.float64)
            row["wss_pred"] = np.asarray(data["wss_pred"], dtype=np.float64)
        if "wss_cfd_over_cfd_max" in names and "wss_pred_over_cfd_max" in names:
            row["wss_cfd_over_max"] = np.asarray(data["wss_cfd_over_cfd_max"], dtype=np.float64)
            row["wss_pred_over_max"] = np.asarray(data["wss_pred_over_cfd_max"], dtype=np.float64)
        else:
            # CASE 导出只有 true_norm=WSS/WSSmax
            row["wss_cfd_over_max"] = row["true_norm"]
            row["wss_pred_over_max"] = row["pred_norm"]
        rows.append(row)
    return rows


def summarize_space(case_true_pred):
    """case_true_pred: list of (true, pred)."""
    maes, nmaes, r2s = [], [], []
    all_t, all_p = [], []
    for yt, yp in case_true_pred:
        mae, rng, nmae = _nmae(yt, yp)
        maes.append(mae)
        nmaes.append(nmae)
        r2s.append(_r2(yt, yp))
        all_t.append(yt)
        all_p.append(yp)
    yt = np.concatenate(all_t)
    yp = np.concatenate(all_p)
    mae_p, rng_p, nmae_p = _nmae(yt, yp)
    # case-balanced R2
    case_means = np.asarray([t.mean() for t, _ in case_true_pred])
    gmean = float(case_means.mean())
    ss_res = float(np.mean([np.mean((t - p) ** 2) for t, p in case_true_pred]))
    ss_tot = float(np.mean([np.mean((t - gmean) ** 2) for t, _ in case_true_pred]))
    r2_cb = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
    return {
        "mae_pooled": mae_p,
        "range_pooled": rng_p,
        "nmae_pooled": nmae_p,
        "mae_casemean": float(np.mean(maes)),
        "nmae_casemean": float(np.mean(nmaes)),
        "r2_pooled": _r2(yt, yp),
        "r2_cb": r2_cb,
        "r2_casemed": float(np.median(r2s)),
        "r2_casemean": float(np.mean(r2s)),
        "n_points": int(yt.size),
        "n_cases": len(case_true_pred),
        "true_all": yt,
        "pred_all": yp,
    }


def collect_run(run_id: str, meta: dict) -> dict:
    root = meta["root"]
    metrics_path = root / meta["metrics"]
    mtest = json.loads(metrics_path.read_text())["test"]
    out = {
        "run_id": run_id,
        "norm_mode": meta["norm_mode"],
        "has_physical": meta["has_physical"],
        "metrics_path": str(metrics_path),
    }

    # metrics.json physical / top-level
    if meta["has_physical"] and "field" in mtest:
        mae, rng, nmae = _nmae_from_field(mtest["field"])
        out["phys_from_metrics"] = {
            "mae_pooled": mae,
            "range_pooled": rng,
            "nmae_pooled": nmae,
            "r2_cb": float(mtest["field_casebalanced"]["r2"]),
            "r2_pooled": float(mtest["field"]["r2"]),
            "mae_casemean": float(mtest["aggregate"].get("mae_casemean", float("nan"))),
        }
        # case-mean NMAE from per_case if present
        nmaes = []
        for reg in mtest.get("per_case", {}).values():
            ov = reg.get("overall", {})
            if "mae" in ov and "nrmse_range" in ov and "rmse" in ov and ov["nrmse_range"]:
                rng_i = ov["rmse"] / ov["nrmse_range"]
                if rng_i > 1e-12:
                    nmaes.append(ov["mae"] / rng_i)
        out["phys_from_metrics"]["nmae_casemean"] = float(np.mean(nmaes)) if nmaes else float("nan")
        out["source_phys"] = "metrics.json"

    norm_block = mtest.get("normalized")
    if norm_block is None and not meta["has_physical"]:
        norm_block = mtest
    if norm_block and "field" in norm_block:
        mae, rng, nmae = _nmae_from_field(norm_block["field"])
        out["norm_from_metrics"] = {
            "mae_pooled": mae,
            "range_pooled": rng,
            "nmae_pooled": nmae,
            "r2_cb": float(norm_block["field_casebalanced"]["r2"]),
            "r2_pooled": float(norm_block["field"]["r2"]),
            "mae_casemean": float(norm_block["aggregate"].get("mae_casemean", float("nan"))),
        }
        nmaes = []
        for reg in norm_block.get("per_case", {}).values():
            ov = reg.get("overall", {})
            if "mae" in ov and "nrmse_range" in ov and "rmse" in ov and ov["nrmse_range"]:
                rng_i = ov["rmse"] / ov["nrmse_range"]
                if rng_i > 1e-12:
                    nmaes.append(ov["mae"] / rng_i)
        out["norm_from_metrics"]["nmae_casemean"] = float(np.mean(nmaes)) if nmaes else float("nan")
        out["source_norm"] = "metrics.json"

    eval_npz = meta.get("eval_npz")
    if eval_npz and Path(eval_npz).exists():
        cached = np.load(eval_npz)
        counts = np.asarray(cached["case_n"], dtype=np.int64)
        stops = np.cumsum(counts)
        starts = np.concatenate(([0], stops[:-1]))
        phys_pairs = [
            (cached["true_phys"][a:b], cached["pred_phys"][a:b])
            for a, b in zip(starts, stops)
        ]
        norm_pairs = [
            (cached["true_norm"][a:b], cached["pred_norm"][a:b])
            for a, b in zip(starts, stops)
        ]
        out["phys_wall"] = summarize_space(phys_pairs)
        out["norm_wall"] = summarize_space(norm_pairs)
        out["n_postview_cases"] = int(len(counts))
        out["source_phys"] = "npz_evalmode"
        out["source_norm"] = "npz_evalmode"
        return out

    if meta["postview"]:
        cases = load_wall_cases(root / meta["postview"])
        out["n_postview_cases"] = len(cases)
        if meta["has_physical"] and cases and "wss_cfd" in cases[0]:
            out["phys_wall"] = summarize_space([(c["wss_cfd"], c["wss_pred"]) for c in cases])
            out["source_phys"] = "wall.csv"
        else:
            out["phys_wall"] = None
        out["norm_wall"] = summarize_space([(c["true_norm"], c["pred_norm"]) for c in cases])
        out["source_norm"] = "wall.csv"
        # CASE 额外：真值已是 /WSSmax，NMAE≈MAE（分母≈1）
        out["casemax_wall"] = summarize_space(
            [(c["wss_cfd_over_max"], c["wss_pred_over_max"]) for c in cases]
        )
    else:
        out["n_postview_cases"] = 0
    return out


def plot_regression(true, pred, title, out_path: Path, xlabel, ylabel, r2_cb=None):
    yt = np.asarray(true, dtype=np.float64)
    yp = np.asarray(pred, dtype=np.float64)
    m = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[m], yp[m]
    # subsample for plot density
    rng = np.random.default_rng(0)
    if yt.size > 80000:
        idx = rng.choice(yt.size, 80000, replace=False)
        yt_s, yp_s = yt[idx], yp[idx]
    else:
        yt_s, yp_s = yt, yp

    r2 = _r2(yt, yp)
    mae, _, nmae = _nmae(yt, yp)

    fig, ax = plt.subplots(figsize=(5.2, 5.0), dpi=160)
    hb = ax.hexbin(yt_s, yp_s, gridsize=80, cmap="viridis", mincnt=1, bins="log")
    lo = float(min(yt_s.min(), yp_s.min()))
    hi = float(max(yt_s.max(), yp_s.max()))
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.2, label="y = x")
    cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("log10(count)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    txt = f"$R^2$={r2:.3f}"
    if r2_cb is not None and np.isfinite(r2_cb):
        txt += f"\n$R^2_{{cb}}$={r2_cb:.3f}"
    txt += f"\nMAE={mae:.3g}\nNMAE={nmae:.3%}"
    ax.text(0.04, 0.96, txt, transform=ax.transAxes, va="top", ha="left",
            fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_grid(summaries, space_key, out_path: Path, title: str, xlabel, ylabel):
    ids = [s["run_id"] for s in summaries if s.get(space_key)]
    n = len(ids)
    if n == 0:
        return
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4.0 * nrows), dpi=150)
    axes = np.atleast_1d(axes).ravel()
    for ax, sid in zip(axes, ids):
        s = next(x for x in summaries if x["run_id"] == sid)
        block = s[space_key]
        yt, yp = block["true_all"], block["pred_all"]
        rng = np.random.default_rng(0)
        if yt.size > 50000:
            idx = rng.choice(yt.size, 50000, replace=False)
            yt_s, yp_s = yt[idx], yp[idx]
        else:
            yt_s, yp_s = yt, yp
        ax.hexbin(yt_s, yp_s, gridsize=60, cmap="viridis", mincnt=1, bins="log")
        lo = float(min(yt_s.min(), yp_s.min()))
        hi = float(max(yt_s.max(), yp_s.max()))
        ax.plot([lo, hi], [lo, hi], "r--", lw=1.0)
        ax.set_title(
            f"{sid}\n$R^2$={block['r2_pooled']:.3f} | $R^2_{{cb}}$={block['r2_cb']:.3f}\n"
            f"NMAE={block['nmae_pooled']:.2%}",
            fontsize=9,
        )
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.grid(True, alpha=0.2)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(title, fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def write_summary_csv(summaries, path: Path):
    lines = [
        "run_id,norm_mode,"
        "phys_nmae_pooled,phys_nmae_casemean,phys_mae_pooled,phys_r2_cb,"
        "norm_nmae_pooled,norm_nmae_casemean,norm_mae_pooled,norm_r2_cb,"
        "source_phys,source_norm"
    ]
    for s in summaries:
        phys = s.get("phys_wall") or s.get("phys_from_metrics") or {}
        norm = s.get("norm_wall") or s.get("norm_from_metrics") or {}
        # CASE: prefer casemax_wall as the "norm" reporting space (same as true_norm)
        if not s["has_physical"] and s.get("casemax_wall"):
            norm = s["casemax_wall"]
        src_p = s.get("source_phys", "")
        src_n = s.get("source_norm", "")
        lines.append(
            f"{s['run_id']},{s['norm_mode']},"
            f"{phys.get('nmae_pooled', float('nan'))},"
            f"{phys.get('nmae_casemean', float('nan'))},"
            f"{phys.get('mae_pooled', float('nan'))},"
            f"{phys.get('r2_cb', float('nan'))},"
            f"{norm.get('nmae_pooled', float('nan'))},"
            f"{norm.get('nmae_casemean', float('nan'))},"
            f"{norm.get('mae_pooled', float('nan'))},"
            f"{norm.get('r2_cb', float('nan'))},"
            f"{src_p},{src_n}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_xlsx(summaries):
    """在原表旁新增「NMAE与R2」工作表；主表插入 NMAE 列（若尚无）。"""
    wb = openpyxl.load_workbook(XLSX)
    main = wb["实验矩阵总览"]
    badge_values = [main.cell(3, c).value for c in (1, 9, 21, 35)]

    # openpyxl 3.0 不会在 insert_cols 时正确搬移 merged-cell 占位符。
    # 先解除后续需要扩展的整行合并区，插列后再统一重建。
    for merged in list(main.merged_cells.ranges):
        if merged.min_row != merged.max_row:
            continue
        row_index = merged.min_row
        first_value = main.cell(row_index, 1).value
        if row_index <= 6 or (
            isinstance(first_value, str)
            and (first_value.startswith("注 ") or first_value.startswith("结论｜"))
        ):
            main.unmerge_cells(str(merged))

    # --- sheet: NMAE与R2 ---
    name = "NMAE与R2"
    if name in wb.sheetnames:
        del wb[name]
    ws = wb.create_sheet(name, 0)
    header = [
        "实验 ID", "目标归一化",
        "物理 NMAE pooled", "物理 NMAE case-mean", "物理 MAE (Pa)", "物理 R² cb",
        "归一化 NMAE pooled", "归一化 NMAE case-mean", "归一化 MAE", "归一化 R² cb",
        "数据来源(物理)", "数据来源(归一化)", "备注",
    ]
    ws.append(["给老师：正式矩阵 + E4 deeper 追加探针 NMAE + R²（ckpt_best · test16）"])
    ws.append([
        "NMAE = MAE / (max(true)−min(true))。"
        "GLOBAL 物理空间与 log-z 归一化空间分列；CASE 无物理恢复，归一化列即 WSS/WSSmax（分母≈1，故 NMAE≈MAE）。"
        "E0 无 PostView，物理 NMAE 由 metrics.json 的 MAE 与 nrmse_range 反推 range。"
    ])
    ws.append(header)
    notes = {
        "E0-GLOBAL": "model.eval() 全点重推理；与官方 metrics 对齐",
        "E2-GLOBAL": "主增量：容量",
        "E3-GLOBAL": "只改 random-5000",
        "E23-GLOBAL": "交互；未超 E2",
        "E4-DEEP-GLOBAL": "相对 E2 只改深度；test 退化，No-Go",
        "E2-CASE": "相对 E2-GLOBAL No-Go",
        "E3-CASE": "相对 E3-GLOBAL No-Go",
    }
    for s in summaries:
        phys = s.get("phys_wall") or s.get("phys_from_metrics") or {}
        norm = s.get("norm_wall") or s.get("norm_from_metrics") or {}
        if not s["has_physical"] and s.get("casemax_wall"):
            norm = s["casemax_wall"]
        src_p = s.get("source_phys", "—")
        src_n = s.get("source_norm", "—")
        ws.append([
            s["run_id"], s["norm_mode"],
            _fmt_pct(phys.get("nmae_pooled")),
            _fmt_pct(phys.get("nmae_casemean")),
            _fmt(phys.get("mae_pooled")),
            _fmt(phys.get("r2_cb")),
            _fmt_pct(norm.get("nmae_pooled")),
            _fmt_pct(norm.get("nmae_casemean")),
            _fmt(norm.get("mae_pooled")),
            _fmt(norm.get("r2_cb")),
            src_p, src_n, notes.get(s["run_id"], ""),
        ])
    ws.append([])
    ws.append(["配套图目录", str(OUT.relative_to(ROOT))])
    ws.append(["单 run 回归图", "r2_regression_<ID>_{physical|normalized}.png"])
    ws.append(["汇总网格", "r2_regression_grid_normalized.png / r2_regression_grid_physical.png"])

    # style
    ws.sheet_view.showGridLines = False
    ws.freeze_panes = "A4"
    ws.sheet_properties.pageSetUpPr.fitToPage = True
    ws.page_setup.orientation = "landscape"
    ws.page_setup.paperSize = ws.PAPERSIZE_A3
    ws.page_setup.fitToWidth = 1
    ws.page_setup.fitToHeight = 1
    ws.print_title_rows = "1:3"
    ws.merge_cells("A1:M1")
    ws.merge_cells("A2:M2")
    ws["A1"].font = Font(name="Noto Sans CJK SC", size=15, bold=True, color="1F4E79")
    ws["A1"].alignment = Alignment(horizontal="left", vertical="center")
    fill = PatternFill("solid", fgColor="1F4E79")
    font = Font(color="FFFFFF", bold=True)
    for cell in ws[3]:
        cell.fill = fill
        cell.font = font
        cell.alignment = Alignment(wrap_text=True, vertical="center")
    widths = [18, 16, 17, 18, 15, 13, 18, 19, 15, 15, 17, 19, 38]
    for col, width in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(col)].width = width
    ws.column_dimensions["M"].width = 36
    ws.row_dimensions[1].height = 28
    ws.row_dimensions[2].height = 42
    ws["A2"].alignment = Alignment(wrap_text=True)
    ws.print_area = f"A1:M{ws.max_row}"

    # --- insert NMAE columns into 实验矩阵总览 if missing ---
    # Find header row 7; insert after Z (MAE Pa) -> AA for phys NMAE if not present
    headers = [main.cell(7, c).value for c in range(1, main.max_column + 1)]
    if "NMAE pooled" not in headers and "物理 NMAE" not in headers:
        # Insert two columns before AA (currently high-WSS R² at 27): after Z=26
        main.insert_cols(27, 2)
        main.cell(7, 27).value = "NMAE pooled"
        main.cell(7, 28).value = "NMAE case-mean"
        for c in (27, 28):
            main.cell(7, c)._style = copy(main.cell(7, 26)._style)
            main.cell(7, c).alignment = copy(main.cell(7, 26).alignment)
        # merge header band row6: extend 物理指标? leave as-is; label in row7
        id_to_row = {}
        for r in range(8, main.max_row + 1):
            eid = main.cell(r, 2).value
            if eid:
                id_to_row[eid] = r
        for s in summaries:
            r = id_to_row.get(s["run_id"])
            if not r:
                continue
            phys = s.get("phys_wall") or s.get("phys_from_metrics") or {}
            for c in (27, 28):
                main.cell(r, c)._style = copy(main.cell(r, 26)._style)
                main.cell(r, c).alignment = copy(main.cell(r, 26).alignment)
            if s["has_physical"]:
                main.cell(r, 27).value = phys.get("nmae_pooled")
                main.cell(r, 28).value = phys.get("nmae_casemean")
                for c in (27, 28):
                    main.cell(r, c).number_format = "0.00%"

    # normalized NMAE：插在归一化块的 MAE 列之后（表头恰为 "MAE"，不是 "MAE (Pa)"）
    headers2 = [main.cell(7, c).value for c in range(1, main.max_column + 1)]
    if "norm NMAE pooled" not in headers2:
        mae_norm_col = next((i for i, h in enumerate(headers2, 1) if h == "MAE"), None)
        if mae_norm_col is not None:
            insert_at = mae_norm_col + 1
            main.insert_cols(insert_at, 2)
            main.cell(7, insert_at).value = "norm NMAE pooled"
            main.cell(7, insert_at + 1).value = "norm NMAE case-mean"
            for c in (insert_at, insert_at + 1):
                main.cell(7, c)._style = copy(main.cell(7, insert_at - 1)._style)
                main.cell(7, c).alignment = copy(main.cell(7, insert_at - 1).alignment)
            id_to_row = {}
            for r in range(8, main.max_row + 1):
                eid = main.cell(r, 2).value
                if eid:
                    id_to_row[eid] = r
            for s in summaries:
                r = id_to_row.get(s["run_id"])
                if not r:
                    continue
                norm = s.get("norm_wall") or s.get("norm_from_metrics") or {}
                if not s["has_physical"] and s.get("casemax_wall"):
                    norm = s["casemax_wall"]
                for c in (insert_at, insert_at + 1):
                    main.cell(r, c)._style = copy(main.cell(r, insert_at - 1)._style)
                    main.cell(r, c).alignment = copy(main.cell(r, insert_at - 1).alignment)
                if norm:
                    main.cell(r, insert_at).value = norm.get("nmae_pooled")
                    main.cell(r, insert_at + 1).value = norm.get("nmae_casemean")
                    for c in (insert_at, insert_at + 1):
                        main.cell(r, c).number_format = "0.00%"

    # openpyxl 插列不会自动扩展合并区；重建顶部色带和页尾，保证 52 列版式完整。
    last_col = main.max_column
    last_letter = get_column_letter(last_col)

    def unmerge_row(row_index):
        for merged in list(main.merged_cells.ranges):
            if merged.min_row == row_index and merged.max_row == row_index:
                main.unmerge_cells(str(merged))

    for row_index in (1, 2, 4, 5):
        value = main.cell(row_index, 1).value
        unmerge_row(row_index)
        main.merge_cells(start_row=row_index, start_column=1, end_row=row_index, end_column=last_col)
        main.cell(row_index, 1).value = value

    unmerge_row(3)
    for (start, end), value in zip(((1, 8), (9, 20), (21, 34), (35, last_col)), badge_values):
        main.merge_cells(start_row=3, start_column=start, end_row=3, end_column=end)
        main.cell(3, start).value = value

    group_specs = [
        (1, 5, "实验身份", "17365D"),
        (6, 15, "模型与训练协议", "2F75B5"),
        (16, 18, "变量与配对", "BF9000"),
        (19, 31, "物理 WSS 指标（Pa 空间）", "548235"),
        (32, 49, "归一化空间分布与 high-risk 指标", "7030A0"),
        (50, last_col, "选模与结论", "C65911"),
    ]
    unmerge_row(6)
    for start, end, value, color in group_specs:
        main.merge_cells(start_row=6, start_column=start, end_row=6, end_column=end)
        cell = main.cell(6, start, value)
        cell.fill = PatternFill("solid", fgColor=color)
        cell.font = Font(name="Noto Sans CJK SC", size=10, bold=True, color="FFFFFF")
        cell.alignment = Alignment(horizontal="center", vertical="center")

    data_rows = [r for r in range(8, main.max_row + 1) if main.cell(r, 2).value]
    end_data = max(data_rows)
    evidence_col = last_col
    for row_index in data_rows:
        evidence = main.cell(row_index, evidence_col)
        for col_index in range(1, last_col + 1):
            cell = main.cell(row_index, col_index)
            if cell.hyperlink:
                cell.hyperlink = None
        if evidence.comment and evidence.comment.text:
            target = Path(evidence.comment.text)
            evidence.hyperlink = target.as_uri()
            evidence.style = "Hyperlink"
            evidence.font = Font(name="Noto Sans CJK SC", size=8.5, color="0563C1", underline="single")

    for row_index in range(end_data + 1, main.max_row + 1):
        text = main.cell(row_index, 1).value
        if isinstance(text, str) and (text.startswith("注 ") or text.startswith("结论｜")):
            unmerge_row(row_index)
            main.merge_cells(start_row=row_index, start_column=1, end_row=row_index, end_column=last_col)
    main.auto_filter.ref = f"A7:{last_letter}{end_data}"
    content_rows = [
        r for r in range(1, main.max_row + 1)
        if any(main.cell(r, c).value is not None for c in range(1, last_col + 1))
    ]
    main.print_area = f"A1:{last_letter}{max(content_rows)}"
    main.column_dimensions[get_column_letter(27)].width = 12
    main.column_dimensions[get_column_letter(28)].width = 12
    main.column_dimensions[get_column_letter(39)].width = 12
    main.column_dimensions[get_column_letter(40)].width = 12
    main.column_dimensions[get_column_letter(50)].width = 11
    main.column_dimensions[get_column_letter(51)].width = 35
    main.column_dimensions[get_column_letter(52)].width = 14

    wb.save(XLSX)


def _fmt(v):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    return round(float(v), 6)


def _fmt_pct(v):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    return f"{100.0 * float(v):.2f}%"


def write_readme(summaries):
    lines = [
        "# WSS PointNet 正式矩阵 + E4 deeper · NMAE 与 R² regression（给老师）",
        "",
        "生成日期：2026-07-15",
        "",
        "## 口径",
        "",
        "- **NMAE** = MAE / (max(true) − min(true))",
        "  - `pooled`：test16 全部壁面点合在一起算",
        "  - `case-mean`：每例先算 NMAE 再平均（更不受极端高峰病例主导）",
        "- **物理空间**：Pa；仅 GLOBAL 可逆回；CASE 不报物理 NMAE",
        "- **归一化空间**：GLOBAL = train61 全局 log-z；CASE = WSS/WSSmax（真值 max≈1，故 NMAE≈MAE）",
        "- **R² regression 图**：横轴 true、纵轴 pred，红色虚线 y=x；色块为 log-count hexbin",
        "- 主模型：`ckpt_best(train_loss)`；分区：test16",
        "",
        "## 数字摘要",
        "",
        "| Run | 物理 NMAE pooled | 物理 R² cb | 归一化 NMAE pooled | 归一化 R² cb |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for s in summaries:
        phys = s.get("phys_wall") or s.get("phys_from_metrics") or {}
        norm = s.get("norm_wall") or s.get("norm_from_metrics") or {}
        if not s["has_physical"] and s.get("casemax_wall"):
            norm = s["casemax_wall"]
        lines.append(
            f"| {s['run_id']} | {_fmt_pct(phys.get('nmae_pooled'))} | "
            f"{_fmt(phys.get('r2_cb'))} | {_fmt_pct(norm.get('nmae_pooled'))} | "
            f"{_fmt(norm.get('r2_cb'))} |"
        )
    lines += [
        "",
        "## 文件",
        "",
        "- `nmae_r2_summary.csv` / Excel 工作表 `NMAE与R2`",
        "- `r2_regression_<RUN>_{physical|normalized}.png`",
        "- `r2_regression_grid_physical.png`、`r2_regression_grid_normalized.png`",
        "",
        "## 注意",
        "",
        "- 物理 NMAE（range）因个别高峰点把分母拉大，数值会偏小；判读请同时看 R² / Spearman / top10，不要单靠 NMAE 宣布 Go。",
        "- CASE 与 GLOBAL 的归一化 MAE/NMAE **不可直接横比**（目标尺度不同）。",
        "- E0 使用已有 `model.eval()` 全点重推理缓存；E2/E3/E23/E4/CASE 使用各自 best PostView 同点 CSV。",
        "",
    ]
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summaries = [collect_run(rid, meta) for rid, meta in RUNS.items()]

    # per-run plots
    for s in summaries:
        rid = s["run_id"]
        if s.get("phys_wall"):
            plot_regression(
                s["phys_wall"]["true_all"], s["phys_wall"]["pred_all"],
                f"{rid} · physical WSS (Pa) · test16",
                OUT / f"r2_regression_{rid}_physical.png",
                "CFD WSS (Pa)", "Pred WSS (Pa)",
                r2_cb=s["phys_wall"]["r2_cb"],
            )
        if s.get("norm_wall"):
            xlabel = "true log-z" if s["norm_mode"] == "global_log_z" else "true WSS/WSSmax"
            ylabel = "pred log-z" if s["norm_mode"] == "global_log_z" else "pred WSS/WSSmax"
            plot_regression(
                s["norm_wall"]["true_all"], s["norm_wall"]["pred_all"],
                f"{rid} · normalized target · test16",
                OUT / f"r2_regression_{rid}_normalized.png",
                xlabel, ylabel,
                r2_cb=s["norm_wall"]["r2_cb"],
            )

    plot_grid(
        [s for s in summaries if s.get("phys_wall")],
        "phys_wall",
        OUT / "r2_regression_grid_physical.png",
        "Formal GLOBAL · physical WSS R2 regression (test16)",
        "CFD WSS (Pa)", "Pred WSS (Pa)",
    )
    plot_grid(
        [s for s in summaries if s.get("norm_wall")],
        "norm_wall",
        OUT / "r2_regression_grid_normalized.png",
        "Formal matrix · normalized-target R2 regression (test16)",
        "true (normalized)", "pred (normalized)",
    )

    write_summary_csv(summaries, OUT / "nmae_r2_summary.csv")
    write_readme(summaries)
    update_xlsx(summaries)

    # also dump json for audit
    serializable = []
    for s in summaries:
        d = {k: v for k, v in s.items() if k not in ("phys_wall", "norm_wall", "casemax_wall")}
        for key in ("phys_wall", "norm_wall", "casemax_wall", "phys_from_metrics", "norm_from_metrics"):
            block = s.get(key)
            if not block:
                continue
            d[key] = {kk: vv for kk, vv in block.items() if kk not in ("true_all", "pred_all")}
        serializable.append(d)
    (OUT / "nmae_r2_summary.json").write_text(
        json.dumps(serializable, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"[ok] wrote {OUT}")
    print(f"[ok] updated {XLSX}")


if __name__ == "__main__":
    main()
