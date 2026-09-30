"""Idempotent backfill of the dedicated workbook sheet for the cycle-integrated (TAWSS / OSI) regression line.

Sheet ``TAWSS_OSI周期量矩阵`` (table ``CycleResults``) is owned entirely by this tool and rebuilt from scratch on every run:

  * stage 1  ``wss_cycle_20260920``          9 fold arms (A1 / O1 / O2 × cv3) + 9 free-baseline rows (C1, per fold) + 6 stagnation rows
  * stage 3  ``wss_cycle_stage3_20260921``   9 train136 arms (× seed) + 3 three-seed ensembles + 2 ensemble stagnation rows
                                             (ensembles from offline/stage3_report_best.json when present)
  * stage 2  ``wss_cycle_m1_20260921``       M1 three-head arms, one row per output channel (peak WSS / TAWSS / OSI)
  * OSI tail ``wss_osi_tail_20260924``       M1r / K1 / K2 / K5 single-change arms on the M1 recipe, one row per output channel;
                                             reference = stored same-fold M1, gate text from experiments/<exp>/report_best.json
  * OSI struct ``wss_osi_struct_20260925``  V1 / V2 auxiliary-channel arms (+ derived-OSI rows for V1), same references
  * v5.2 M1  ``wss_cycle_m1_v52_20260925``  M1cap × CV5 × 3 seeds, one row per run and channel; peak reference = paired X5Dcap run

Every cell of every other sheet (value, style, comment, hyperlink), merged ranges and dimensions are verified unchanged after
saving; a timestamped backup is written first.  Arms without results keep ``—``.

    python -m training_wss_min.tools.update_wss_cycle_xlsx [--workbook PATH] [--name NAME ...]
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
from openpyxl import load_workbook
from openpyxl.comments import Comment
from openpyxl.worksheet.table import Table, TableStyleInfo

from training_wss_min import config as C
from training_wss_min import metrics as M
from training_wss_min.tools.report_wss_recovery import cell_record
from training_wss_min.tools.update_wss_local_wave1_xlsx import DESCRIPTIONS

ROOT = C.PROJECT_ROOT
RUNS = ROOT / "training_wss_min/runs"
BOOK = ROOT / "docs/03-汇报材料/WSS_PointNet实验矩阵与结果汇总last.xlsx"
SHEET = "TAWSS_OSI周期量矩阵"
TABLE = "CycleResults"
STYLE_SHEET = "速度与压力实验矩阵"
MISSING = "—"
DEFAULT_NAMES = ("wss_cycle_20260920", "wss_cycle_stage3_20260921", "wss_cycle_m1_20260921", "wss_cycle_m1_stage3_20260922")
GROUPS = {
    "wss_cycle_20260920": "WSS周期积分量TAWSS/OSI直接回归阶段1｜2026-09-20（A1/O1/O2×cv3）",
    "wss_cycle_stage3_20260921": "WSS周期积分量阶段3确认｜2026-09-21（A1/O1/O2×seed 1234/7/2025，train136→test34）",
    "wss_cycle_m1_20260921": "WSS周期积分量阶段2 M1三头｜2026-09-21（M1×cv3，逐通道行）",
    "wss_cycle_m1_stage3_20260922": "WSS周期积分量M1三头三seed确认｜2026-09-22（M1×seed 1234/7/2025，train136→test34，逐通道行+集成）",
}
PEAK_SEED_RUN = "wss_v51_wave1_20260916"
PEAK_FOLD_RUN = "wss_v51_wave2a_20260916"

SECTIONS = [
    ("01  实验与评估协议", ["实验矩阵", "实验 / 模型", "预测对象", "检查点 / 汇总", "评估范围 / 划分", "折", "seed", "目标空间", "标签定义"]),
    ("02  训练相关", ["输入维", "epochs", "参数量", "训练时长 s", "best epoch", "train loss @best", "train loss @last", "训练例数", "留出例数"]),
    ("03  精度（与 WSS 表同口径）", ["归一化 R²\n病例等权 cb", "物理 R²\n病例等权 cb", "逐例均值 R²", "逐例中位 R²", "逐例 p10 R²", "负 R² 例", "MAE\npooled", "RMSE\npooled",
                              "NMAE_range\npooled", "逐例 NMAE_range\n均值", "Spearman\n逐例均值", "r²_fit cb\n（Pearson²）", "斜率 cb", "top10 幅值比", "top10 IoU\n逐例均值", "Δ物理 R²\nlast−best"]),
    ("04  落地一致性（cycle_agreement）", ["CCC 空间", "CCC 逐例中位", "CCC 逐例 p10", "CCC 最差例", "逐点相对误差中位\n（逐例中位）", "±30% 内点份额\n（逐例中位）", "top10% 热点 Dice\n中位", "top10% 热点 Dice\np10",
                                       "分段均值相对误差\n逐例中位", "分段最差段相对误差\n逐例中位", "病例均值 BA 偏差\n（相对）", "病例均值 BA LoA 低", "病例均值 BA LoA 高", "病例均值 CCC\n（病例级）",
                                       "掩膜 1", "掩膜 1 Dice 中位", "掩膜 1 Dice p10", "掩膜 1 Dice 最差", "掩膜 1 真值面积份额中位", "掩膜 1 面积份额 BA 偏差", "掩膜 1 面积份额 LoA 低", "掩膜 1 面积份额 LoA 高",
                                       "掩膜 2", "掩膜 2 Dice 中位", "掩膜 2 Dice p10", "掩膜 2 Dice 最差", "掩膜 2 真值面积份额中位", "掩膜 2 面积份额 BA 偏差", "掩膜 2 面积份额 LoA 低", "掩膜 2 面积份额 LoA 高"]),
    ("05  阈值掩膜 IoU（threshold_masks）", ["掩膜 1 IoU 逐例均值", "掩膜 1 precision", "掩膜 1 recall", "掩膜 1 面积份额误差 med", "掩膜 2 IoU 逐例均值", "掩膜 2 precision", "掩膜 2 recall", "掩膜 2 面积份额误差 med"]),
    ("06  参照与门控", ["同折/同seed峰值底座\n归一化 R²_cb", "Δ归一化 R²\nvs 峰值底座", "同折/同seed峰值底座\n物理 R²_cb", "自由基线 / 单头参照", "参照主值", "Δ主值 vs 参照", "参照 log R²\n（TAWSS-null）", "参照掩膜 IoU", "Δ掩膜 IoU", "门控结果"]),
    ("07  来源", ["run id", "配置路径", "metrics 路径", "备注"]),
]
HEADERS = [h for _, hs in SECTIONS for h in hs]
COL = {h: i + 1 for i, h in enumerate(HEADERS)}

GLOSSARY = [
    ("CCC（Lin 一致性相关）", "2·cov(t,p) / (var t + var p + (mean t − mean p)²)。同时惩罚型态错和幅值/偏置错，=1 才是逐点完全一致；TAWSS 在 ln(max(y,0.05 Pa)) 空间算，OSI 线性。与 R² 的差别：R² 按方差加权、被高值尾部主导，CCC 不放大尾部。"),
    ("逐点相对误差中位 / ±30% 内点份额", "|pred − true| / max(true, floor) 的逐例中位，以及相对误差 ≤ 0.3 的点份额（eval.cycle_rel_tolerance；OSI 分母下限 0.01）。回答「单点读数的容差」。"),
    ("top10% 热点 Dice", "真值与预测各自前 10% 的点集的 Dice = 2|A∩B| / (|A|+|B|)。热点位置是否对，与幅值无关。"),
    ("阈值掩膜 Dice / IoU / 面积份额", "临床阈值掩膜（TAWSS < 0.4 Pa；OSI > 0.1 / > 0.3）的 Dice 与 IoU（顶点计数），配真值面积份额与预测面积份额；区域很小时 Dice 不稳，必须与面积份额一起读。"),
    ("分段均值相对误差", "按 wss_v5 语义血管段（trunk / 左右 CIA / 四末支，bundle wall_semantic_id，≥200 点）算段均值 |pred − true| / true；逐例取中位与最差段。对应报告按血管段给数的粒度。"),
    ("Bland–Altman 偏差 / LoA", "病例级报告数（病例均值 TAWSS 取相对差；掩膜面积份额取绝对差）的偏差 mean(p − t) 与 95% 一致性界 bias ± 1.96·sd。回答「这个数系统偏多少、95% 病例落在哪个范围」。"),
    ("自由基线", "不训练就有的参照：TAWSS-null = 峰值折外预测 × 训练折波形按 80 帧重算 TAWSS；TAWSS-ratio = 常数 c × 峰值预测；OSI-null = 训练折等渗回归 OSI ~ ln τ̂_peak 套到留出折/test34。模型必须超过它们才算学到周期信息。"),
    ("门控（预注册）", "阶段 1：G1.1 归一化 R²_cb ≥ 同折峰值底座 − 0.02；G1.2 Pa R²_cb ≥ max(TAWSS-null, ratio) + 0.05 且 3/3 折同向；G2.1 OSI R²_cb > 0；G2.2 OSI>0.1 IoU ≥ OSI-null + 0.05；G2.3 面积份额误差中位 ≤ 0.05；G3 滞留区 IoU ≥ 基线组合 + 0.05。阶段 2 M1：峰值通道归一化 R²_cb 相对同折底座掉 ≤ 0.02（3/3），两头相对单头 A1/O2 ≥ −0.01。阶段 3：确认，不设门。"),
    ("三 seed 集成（ens3）", "同一目标三个 seed 的 test34 逐点 Pa 预测取均值（OSI 裁 [0,0.5]）后重算全套指标；归一化空间用 train136 统计量把集成值映回 z。与已部署 X5D_v51 峰值集成同形态。"),
    ("M1 逐通道行", "一次前向输出 [峰值 WSS, TAWSS, OSI] 三通道；每通道按各自统计量走单目标评估，行内数字与对应单目标臂同口径；参照列：峰值通道对同折 X5D_v51、TAWSS 对 A1_f{k}、OSI 对 O2_f{k}。"),
    ("标签定义", "wss_min_cycle_v1：帧 0–79（步 1120–1278）各权 1/80（帧 80 = 帧 0 + T）；TAWSS = 时间平均 |τ|；OSI = ½(1 − |Σ τ| / Σ |τ|) ∈ [0, 0.5]；从 bundle wall_wss_vec 直接算，不新跑 CFD。"),
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rj(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else default


def num(v, nd=4):
    if v is None:
        return MISSING
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v
    return MISSING if not math.isfinite(f) else round(f, nd)


def dig(d, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def describe(name, aid):
    return DESCRIPTIONS.get(f"{name}:{aid}") or DESCRIPTIONS.get(aid) or aid


def blank():
    return {h: MISSING for h in HEADERS}


def fill_accuracy(v, best, last=None):
    """03 精度（与 WSS 表同口径）+ 04 落地一致性 + 05 阈值掩膜，来源 = 单目标 result dict（metrics.json['test'] 或 heads[...]）。"""
    fcb, fld, agg, hot, cal = best["field_casebalanced"], best["field"], best["aggregate"], best.get("hotspot", {}), best.get("calibration", {})
    v.update({
        "归一化 R²\n病例等权 cb": num(dig(best, "normalized", "field_casebalanced", "r2")), "物理 R²\n病例等权 cb": num(fcb["r2"]),
        "逐例均值 R²": num(agg["r2_casemean"]), "逐例中位 R²": num(agg["r2_casemed"]), "逐例 p10 R²": num(agg["r2_casep10"]),
        "负 R² 例": f"{int(agg['r2_negative_cases'])}/{int(agg['n_cases'])}", "MAE\npooled": num(fld["mae"]), "RMSE\npooled": num(fld["rmse"]),
        "NMAE_range\npooled": num(fld.get("nmae_range")), "逐例 NMAE_range\n均值": num(agg.get("nmae_casemean")),
        "Spearman\n逐例均值": num(hot.get("spearman_all_casemean"), 3), "r²_fit cb\n（Pearson²）": num(fcb.get("r2_linear_fit")), "斜率 cb": num(fcb.get("linear_fit_slope"), 3),
        "top10 幅值比": num(cal.get("top10_pred_true_ratio"), 3), "top10 IoU\n逐例均值": num(hot.get("top10_iou_casemean"), 3),
        "Δ物理 R²\nlast−best": num(last["field_casebalanced"]["r2"] - fcb["r2"]) if last else MISSING,
    })
    fill_agreement(v, best.get("cycle_agreement"))
    for j, (name, blk) in enumerate(sorted((best.get("threshold_masks") or {}).items())[:2], 1):
        v.update({f"掩膜 {j} IoU 逐例均值": num(blk["iou_casemean"], 3), f"掩膜 {j} precision": num(blk["precision_casemean"], 3),
                  f"掩膜 {j} recall": num(blk["recall_casemean"], 3), f"掩膜 {j} 面积份额误差 med": num(blk["frac_abs_err_casemed"], 3)})
        if v[f"掩膜 {j}"] == MISSING:
            v[f"掩膜 {j}"] = name


def fill_agreement(v, ca):
    if not ca or not ca.get("ccc"):
        return
    v.update({
        "CCC 空间": ca.get("ccc_space", MISSING), "CCC 逐例中位": num(ca["ccc"]["med"], 3), "CCC 逐例 p10": num(ca["ccc"]["p10"], 3), "CCC 最差例": num(ca["ccc"]["min"], 3),
        "逐点相对误差中位\n（逐例中位）": num(ca["rel_err_med"]["med"], 3), "±30% 内点份额\n（逐例中位）": num(ca["within_tol"]["med"], 3),
        "top10% 热点 Dice\n中位": num(ca["top_dice"]["med"], 3), "top10% 热点 Dice\np10": num(ca["top_dice"]["p10"], 3),
        "分段均值相对误差\n逐例中位": num(ca["seg_rel_err_med"]["med"], 3), "分段最差段相对误差\n逐例中位": num(ca["seg_rel_err_max"]["med"], 3),
        "病例均值 BA 偏差\n（相对）": num(ca["case_mean"]["bias"], 3), "病例均值 BA LoA 低": num(ca["case_mean"]["loa_low"], 3),
        "病例均值 BA LoA 高": num(ca["case_mean"]["loa_high"], 3), "病例均值 CCC\n（病例级）": num(ca["case_mean"]["ccc"], 3),
    })
    for j, (name, blk) in enumerate(sorted(ca.get("masks", {}).items())[:2], 1):
        v.update({f"掩膜 {j}": name, f"掩膜 {j} Dice 中位": num(blk["dice_med"], 3), f"掩膜 {j} Dice p10": num(blk["dice_p10"], 3), f"掩膜 {j} Dice 最差": num(blk["dice_min"], 3),
                  f"掩膜 {j} 真值面积份额中位": num(blk["true_frac_med"], 3), f"掩膜 {j} 面积份额 BA 偏差": num(blk["area_share"]["bias"], 3),
                  f"掩膜 {j} 面积份额 LoA 低": num(blk["area_share"]["loa_low"], 3), f"掩膜 {j} 面积份额 LoA 高": num(blk["area_share"]["loa_high"], 3)})


def fill_training(v, cfg, run, diag, hist):
    score = [h.get("selection_score", h.get("train_loss")) for h in hist]
    best_ep = int(np.argmin(score)) if score else None
    split = rj(cfg["data"]["split_path"], {})
    v.update({"输入维": len(cfg["data"]["input_features"]), "epochs": cfg["train"]["epochs"], "参数量": diag.get("parameter_count", MISSING),
              "训练时长 s": num(diag.get("elapsed_seconds", hist[-1].get("elapsed_s") if hist else None), 0),
              "best epoch": best_ep if best_ep is not None else MISSING,
              "train loss @best": num(hist[best_ep]["train_loss"], 5) if hist and best_ep is not None else MISSING,
              "train loss @last": num(hist[-1]["train_loss"], 5) if hist else MISSING,
              "训练例数": len(split.get("train_cases", [])), "留出例数": len(split.get("test_cases", []))})


def peak_reference(v, best, ref_metrics):
    if not ref_metrics:
        return
    bn, bp = dig(ref_metrics, "test", "normalized", "field_casebalanced", "r2"), dig(ref_metrics, "test", "field_casebalanced", "r2")
    v.update({"同折/同seed峰值底座\n归一化 R²_cb": num(bn), "Δ归一化 R²\nvs 峰值底座": num(dig(best, "normalized", "field_casebalanced", "r2") - bn), "同折/同seed峰值底座\n物理 R²_cb": num(bp)})


def free_baseline_reference(v, best, cf, is_osi):
    if not cf:
        return
    fcb = best["field_casebalanced"]
    if is_osi:
        on = cf["OSI_null"]; iou0 = dig(on, "threshold_masks", "above_0.1", "iou")
        v.update({"自由基线 / 单头参照": "OSI-null（等渗 OSI~ln τ̂_peak）", "参照主值": num(on["r2_cb"]), "Δ主值 vs 参照": num(fcb["r2"] - on["r2_cb"]),
                  "参照掩膜 IoU": num(iou0, 3), "Δ掩膜 IoU": num(dig(best, "threshold_masks", "above_0.1", "iou_casemean") - iou0, 3) if iou0 is not None else MISSING})
    else:
        tn, tr = cf["TAWSS_null"], cf["TAWSS_ratio"]; bmax = max(tn["r2_cb"], tr["r2_cb"])
        iou0 = max(dig(tn, "threshold_masks", "below_0.4", "iou"), dig(tr, "threshold_masks", "below_0.4", "iou"))
        v.update({"自由基线 / 单头参照": f"max(TAWSS-null {tn['r2_cb']:.4f}, TAWSS-ratio {tr['r2_cb']:.4f})", "参照主值": num(bmax), "Δ主值 vs 参照": num(fcb["r2"] - bmax),
                  "参照 log R²\n（TAWSS-null）": num(tn["log_r2_cb"]), "参照掩膜 IoU": num(iou0, 3),
                  "Δ掩膜 IoU": num(dig(best, "threshold_masks", "below_0.4", "iou_casemean") - iou0, 3)})


def load_arm(name, arm, configs):
    cfg = rj(configs / arm["config"]); run = RUNS / cfg["name"]
    mets = {ck: rj(run / f"eval/ckpt_{ck}/metrics.json") for ck in ("best", "last")}
    mets = {ck: (m["test"] if m else None) for ck, m in mets.items()}
    diag = rj(run / "training_diagnostics.json", {})
    hist = [json.loads(l) for l in (run / "history.jsonl").read_text().splitlines() if l.strip()] if (run / "history.jsonl").is_file() else []
    return cfg, run, mets, diag, hist


# ----------------------------------------------------------------------------------------------------------------------------
def rows_stage1(name, configs, exp, matrix, queue, group):
    c1 = rj(exp / "offline/c1_baselines.json", {}); gate = rj(exp / "offline/gate_report_best.json", {})
    rows = []
    for arm in matrix["arms"]:
        aid = arm["id"]; fold = arm.get("fold"); kind = aid.split("_")[0]
        cfg, run, mets, diag, hist = load_arm(name, arm, configs); best, last = mets["best"], mets["last"]
        rec = queue["arms"].get(aid, {}); is_osi = cfg["data"]["target"] == "osi"
        stats = rj(cfg["data"]["wss_stats_path"], {})
        space = {"tawss": "log_z（ln(max(TAWSS,0.05)) z-score）", "osi_linear": "linear z（评估裁 [0,0.5]）", "osi_logit": "logit_z（0.5·sigmoid 反变换）"}
        key = "tawss" if not is_osi else ("osi_logit" if stats.get("method") == "logit_z" else "osi_linear")
        v = blank(); split = rj(cfg["data"]["split_path"], {})
        v.update({"实验矩阵": group, "实验 / 模型": f"{aid}｜{describe(name, aid)}｜{'已完成' if rec.get('status') == 'complete' else rec.get('status', '待跑')}",
                  "预测对象": "OSI · 无量纲 [0,0.5]" if is_osi else "TAWSS · Pa", "检查点 / 汇总": "best（last 见本格批注）",
                  "评估范围 / 划分": f"V5.1 cv3_v51 fold{fold} 留出折（{len(split.get('test_cases', []))} 例）；test34 未用", "折": fold, "seed": arm.get("seed", 1234),
                  "目标空间": space[key], "标签定义": "wss_min_cycle_v1：帧 0–79 各权 1/80；" + ("OSI=½(1−|Στ|/Σ|τ|)" if is_osi else "TAWSS=时间平均|τ|"),
                  "run id": cfg["name"], "配置路径": str(configs / arm["config"]), "metrics 路径": str(run / "eval/ckpt_best/metrics.json")})
        fill_training(v, cfg, run, diag, hist)
        note = f"Job {queue.get('job_id', '')}；{arm['title']}；单 seed，只作阶段 1 筛选。"
        if best:
            fill_accuracy(v, best, last)
            if not is_osi:
                peak_reference(v, best, rj(RUNS / PEAK_FOLD_RUN / f"X5D_v51_f{fold}_s1234/eval/ckpt_best/metrics.json"))
            free_baseline_reference(v, best, c1.get(f"fold{fold}", {}), is_osi)
            g = gate.get("gates", {})
            if kind == "A1":
                v["门控结果"] = f"G1.1 {'PASS' if dig(g, 'G1.1', 'pass') else 'FAIL'}；G1.2 {'PASS' if dig(g, 'G1.2', 'pass') else 'FAIL'}（三折均值判，见 gate_report_best.txt）"
            else:
                v["门控结果"] = (f"G2.1 {'PASS' if g.get(f'{kind}.G2.1') else 'FAIL'}；G2.2 {'PASS' if dig(g, f'{kind}.G2.2', 'pass') else 'FAIL'}；"
                                 f"G2.3 {'PASS' if dig(g, f'{kind}.G2.3', 'pass') else 'FAIL'}；G3 A1×{kind} {'PASS' if dig(g, f'G3.A1x{kind}', 'pass') else 'FAIL'}（三折均值判）")
            if last:
                lca = last.get("cycle_agreement") or {}
                note += (f" last：归一化 R²_cb={num(dig(last, 'normalized', 'field_casebalanced', 'r2'))}，物理 R²_cb={num(last['field_casebalanced']['r2'])}，"
                         f"CCC 中位={num(dig(lca, 'ccc', 'med'), 3)}，分段误差中位={num(dig(lca, 'seg_rel_err_med', 'med'), 3)}。")
        v["备注"] = note
        rows.append(("arm", v))
    rows += baseline_rows(group, exp, c1)
    rows += combo_rows(group, exp, RUNS / name, c1)
    return rows


def baseline_rows(group, exp, c1):
    rows = []
    for fold in range(3):
        cf = c1.get(f"fold{fold}")
        if not cf:
            continue
        n = cf.get("n_cases")
        for key, label, target in (("TAWSS_null", "TAWSS-null：峰值折外预测 × 训练折波形（σ 缩放）按 80 帧重算 TAWSS；不训练", "TAWSS · Pa"),
                                   ("TAWSS_ratio", f"TAWSS-ratio：c × 峰值折外预测，c = 训练折逐点 TAWSS/峰值 中位（c={cf['TAWSS_ratio'].get('ratio_c', float('nan')):.3f}）；不训练", "TAWSS · Pa"),
                                   ("OSI_null", "OSI-null：训练折等渗回归 OSI ~ ln τ̂_peak 套到留出折峰值折外预测；不训练", "OSI · 无量纲 [0,0.5]")):
            b = cf[key]; v = blank()
            v.update({"实验矩阵": group, "实验 / 模型": f"{key}_f{fold}｜{label}｜自由基线（C1，作业 15333）", "预测对象": target, "检查点 / 汇总": "无训练",
                      "评估范围 / 划分": f"V5.1 cv3_v51 fold{fold} 留出折（{n} 例）；折外峰值预测来自 wss_min_cascade_v1", "折": fold, "目标空间": "物理空间直接构造",
                      "标签定义": "同模型臂", "物理 R²\n病例等权 cb": num(b["r2_cb"]), "逐例均值 R²": num(b["r2_casemean"]), "逐例中位 R²": num(b["r2_casemed"]), "逐例 p10 R²": num(b["r2_casep10"]),
                      "负 R² 例": f"{int(b['r2_negative_cases'])}/{n}", "MAE\npooled": num(b["mae_cb"]), "Spearman\n逐例均值": num(b["spearman_casemean"], 3),
                      "r²_fit cb\n（Pearson²）": num(b["r2_fit_cb"]), "斜率 cb": num(b["slope_cb"], 3),
                      "run id": f"c1_baselines.{key}.fold{fold}", "metrics 路径": str(exp / "offline/c1_baselines.json"),
                      "备注": "自由基线，供 G1.2 / G2.2 / G3 参照；log R²（TAWSS）与掩膜 IoU 见对应列"})
            if key.startswith("TAWSS"):
                v["归一化 R²\n病例等权 cb"] = num(b.get("log_r2_cb"))
            for j, (mname, blk) in enumerate(sorted(b.get("threshold_masks", {}).items())[:2], 1):
                v.update({f"掩膜 {j}": mname, f"掩膜 {j} IoU 逐例均值": num(blk["iou"], 3), f"掩膜 {j} precision": num(blk["precision"], 3), f"掩膜 {j} recall": num(blk["recall"], 3),
                          f"掩膜 {j} 面积份额误差 med": num(blk.get("frac_abs_err_casemed"), 3)})
            rows.append(("baseline", v))
    return rows


def combo_rows(group, exp, runs, c1):
    rows = []
    for fold in range(3):
        a_dir = runs / f"A1_f{fold}_s1234/eval/ckpt_best/predictions/test"
        if not a_dir.is_dir():
            continue
        for o in ("O1", "O2"):
            o_dir = runs / f"{o}_f{fold}_s1234/eval/ckpt_best/predictions/test"
            if not o_dir.is_dir():
                continue
            dice, iou, tf, pf = [], [], [], []
            for pa in sorted(a_dir.glob("*/*/*/predictions.npz")):
                po = o_dir / pa.relative_to(a_dir)
                if not po.is_file():
                    continue
                with np.load(pa) as za, np.load(po) as zo:
                    tm = (za["true_pa"] < 0.4) & (zo["true_pa"] > 0.1); pm = (za["pred_pa"] < 0.4) & (zo["pred_pa"] > 0.1)
                u = np.count_nonzero(tm | pm)
                iou.append(np.count_nonzero(tm & pm) / u if u else np.nan); dice.append(M.dice_masks(tm, pm)); tf.append(tm.mean()); pf.append(pm.mean())
            dice = np.asarray(dice); iou = np.asarray(iou); ba = M._bland_altman(tf, pf, relative=False)
            cf = c1.get(f"fold{fold}", {}) if c1 else {}
            base = max(dig(cf, "stagnation_null_null", "iou", default=float("nan")), dig(cf, "stagnation_ratio_null", "iou", default=float("nan")))
            v = blank()
            v.update({"实验矩阵": group, "实验 / 模型": f"A1×{o}_f{fold}｜滞留区组合：A1 的 TAWSS < 0.4 Pa ∧ {o} 的 OSI > 0.1（同点预测离线组合）｜已完成",
                      "预测对象": "滞留区掩膜（TAWSS<0.4 ∧ OSI>0.1）", "检查点 / 汇总": "best × best", "评估范围 / 划分": f"V5.1 cv3_v51 fold{fold} 留出折（{len(dice)} 例）", "折": fold, "seed": 1234,
                      "目标空间": "两臂物理空间预测的掩膜交集", "标签定义": "同模型臂",
                      "掩膜 1": "stagnation", "掩膜 1 Dice 中位": num(np.nanmedian(dice), 3), "掩膜 1 Dice p10": num(np.nanquantile(dice, .1), 3), "掩膜 1 Dice 最差": num(np.nanmin(dice), 3),
                      "掩膜 1 真值面积份额中位": num(np.median(tf), 3), "掩膜 1 面积份额 BA 偏差": num(ba["bias"], 3), "掩膜 1 面积份额 LoA 低": num(ba["loa_low"], 3), "掩膜 1 面积份额 LoA 高": num(ba["loa_high"], 3),
                      "掩膜 1 IoU 逐例均值": num(np.nanmean(iou), 3), "掩膜 1 面积份额误差 med": num(np.median(np.abs(np.asarray(pf) - np.asarray(tf))), 3),
                      "自由基线 / 单头参照": "max(TAWSS-null×OSI-null, TAWSS-ratio×OSI-null) 滞留区 IoU", "参照掩膜 IoU": num(base, 3), "Δ掩膜 IoU": num(np.nanmean(iou) - base, 3),
                      "门控结果": "G3 判据：三折均值 Δ ≥ +0.05 且 2/3 同向（见 gate_report_best.txt）",
                      "run id": f"A1_f{fold}_s1234 × {o}_f{fold}_s1234", "metrics 路径": str(exp / "offline/gate_report_best.json"),
                      "备注": "滞留区 = 低 TAWSS 且高 OSI 的交集区域；IoU/Dice 顶点计数；面积份额 Bland–Altman 为绝对差"})
            rows.append(("combo", v))
    return rows


# ----------------------------------------------------------------------------------------------------------------------------
def rows_stage3(name, configs, exp, matrix, queue, group):
    c1 = rj(ROOT / "training_wss_min/experiments/wss_cycle_20260920/offline/c1_baselines.json", {}).get("test34", {})
    report = rj(exp / "offline/stage3_report_best.json", {})
    rows = []
    for arm in matrix["arms"]:
        aid = arm["id"]; seed = arm.get("seed", 1234); kind = aid.split("_")[0]
        cfg, run, mets, diag, hist = load_arm(name, arm, configs); best, last = mets["best"], mets["last"]
        rec = queue["arms"].get(aid, {}); is_osi = cfg["data"]["target"] == "osi"
        stats = rj(cfg["data"]["wss_stats_path"], {})
        space = {"tawss": "log_z（train136 统计）", "osi_linear": "linear z（评估裁 [0,0.5]）", "osi_logit": "logit_z（0.5·sigmoid 反变换）"}
        key = "tawss" if not is_osi else ("osi_logit" if stats.get("method") == "logit_z" else "osi_linear")
        v = blank()
        v.update({"实验矩阵": group, "实验 / 模型": f"{aid}｜{describe(name, aid)}｜{'已完成' if rec.get('status') == 'complete' else rec.get('status', '待跑')}",
                  "预测对象": "OSI · 无量纲 [0,0.5]" if is_osi else "TAWSS · Pa", "检查点 / 汇总": "best（last 见本格批注）",
                  "评估范围 / 划分": "V5.1 train136 → test34（34 例，锁定确认，读一次）", "折": "train136→test34", "seed": seed,
                  "目标空间": space[key], "标签定义": "wss_min_cycle_v1：帧 0–79 各权 1/80；" + ("OSI=½(1−|Στ|/Σ|τ|)" if is_osi else "TAWSS=时间平均|τ|"),
                  "run id": cfg["name"], "配置路径": str(configs / arm["config"]), "metrics 路径": str(run / "eval/ckpt_best/metrics.json")})
        fill_training(v, cfg, run, diag, hist)
        note = f"Job {queue.get('job_id', '')}；{arm['title']}；阶段 3 确认（不设门），test34 只读一次。"
        if best:
            fill_accuracy(v, best, last)
            if not is_osi:
                peak_reference(v, best, rj(RUNS / PEAK_SEED_RUN / f"X5D_v51_s{seed}/eval/ckpt_best/metrics.json"))
            free_baseline_reference(v, best, c1, is_osi)
            v["门控结果"] = "阶段 3 确认：不设门；与同 seed 已部署峰值模型 / test34 自由基线并列读"
            if last:
                note += f" last：归一化 R²_cb={num(dig(last, 'normalized', 'field_casebalanced', 'r2'))}，物理 R²_cb={num(last['field_casebalanced']['r2'])}。"
        v["备注"] = note
        rows.append(("arm", v))
    for kind, ens in (report.get("ensembles") or {}).items():
        if kind.startswith("stagnation"):
            v = blank(); ba = ens["area_share_ba"]
            v.update({"实验矩阵": group, "实验 / 模型": f"{kind.replace('stagnation_', '')}_ens3｜滞留区：A1 三 seed 集成 TAWSS < 0.4 ∧ {kind[-2:]} 三 seed 集成 OSI > 0.1｜已完成",
                      "预测对象": "滞留区掩膜（TAWSS<0.4 ∧ OSI>0.1）", "检查点 / 汇总": "三 seed Pa 均值集成", "评估范围 / 划分": "test34（34 例）", "折": "train136→test34", "seed": "1234+7+2025",
                      "掩膜 1": "stagnation", "掩膜 1 Dice 中位": num(ens["dice_med"], 3), "掩膜 1 IoU 逐例均值": num(ens["iou_casemean"], 3),
                      "掩膜 1 面积份额 BA 偏差": num(ba["bias"], 3), "掩膜 1 面积份额 LoA 低": num(ba["loa_low"], 3), "掩膜 1 面积份额 LoA 高": num(ba["loa_high"], 3),
                      "自由基线 / 单头参照": "test34 基线组合滞留区 IoU", "参照掩膜 IoU": num(ens["baseline_iou"], 3), "Δ掩膜 IoU": num(ens["iou_casemean"] - ens["baseline_iou"], 3),
                      "run id": f"{kind}_ens3", "metrics 路径": str(exp / "offline/stage3_report_best.json"), "备注": "集成预测的掩膜交集；顶点计数"})
            rows.append(("combo", v)); continue
        is_osi = kind != "A1"; v = blank()
        v.update({"实验矩阵": group, "实验 / 模型": f"{kind}_ens3｜{kind} 三 seed（1234/7/2025）test34 逐点 Pa 预测均值集成{'（OSI 裁 [0,0.5]）' if is_osi else ''}；与已部署 X5D_v51 集成同形态｜已完成",
                  "预测对象": "OSI · 无量纲 [0,0.5]" if is_osi else "TAWSS · Pa", "检查点 / 汇总": "三 seed best 集成", "评估范围 / 划分": "V5.1 train136 → test34（34 例）", "折": "train136→test34", "seed": "1234+7+2025",
                  "目标空间": "集成在物理空间；归一化列用 train136 统计量映回 z", "标签定义": "同单 seed 臂",
                  "归一化 R²\n病例等权 cb": num(ens["norm_r2cb"]), "物理 R²\n病例等权 cb": num(ens["pa_r2cb"]), "逐例均值 R²": num(ens["r2_casemean"]), "逐例中位 R²": num(ens["r2_casemed"]),
                  "逐例 p10 R²": num(ens["r2_casep10"]), "负 R² 例": f"{ens['neg']}/{ens['n_cases']}", "MAE\npooled": num(ens["mae_cb"]), "Spearman\n逐例均值": num(ens["spearman"], 3),
                  "r²_fit cb\n（Pearson²）": num(ens["r2_fit_cb"]), "斜率 cb": num(ens["slope_cb"], 3),
                  "run id": f"{kind}_ens3", "metrics 路径": str(exp / "offline/stage3_report_best.json"),
                  "备注": f"三 seed 单模型物理 R²_cb 均值 {num(dig(ens, 'seed_mean', 'pa_r2cb'))} ± {num(dig(ens, 'seed_sd', 'pa_r2cb'))}（sd）；归一化 {num(dig(ens, 'seed_mean', 'norm_r2cb'))} ± {num(dig(ens, 'seed_sd', 'norm_r2cb'))}"})
        fill_agreement(v, ens.get("agreement"))
        for j, (mname, blk) in enumerate(sorted(ens.get("threshold_masks", {}).items())[:2], 1):
            v.update({f"掩膜 {j} IoU 逐例均值": num(blk["iou"], 3), f"掩膜 {j} 面积份额误差 med": num(blk["frac_abs_err_casemed"], 3)})
            if v[f"掩膜 {j}"] == MISSING:
                v[f"掩膜 {j}"] = mname
        pseudo = {"field_casebalanced": {"r2": ens["pa_r2cb"]}, "normalized": {"field_casebalanced": {"r2": ens["norm_r2cb"]}},
                  "threshold_masks": {k: {"iou_casemean": b["iou"]} for k, b in ens.get("threshold_masks", {}).items()}}
        free_baseline_reference(v, pseudo, c1, is_osi)
        if not is_osi:
            peak_ens = rj(ROOT / "training_wss_min/experiments/wss_cycle_stage3_20260921/offline/peak_ensemble_reference.json")
            if peak_ens:
                v.update({"同折/同seed峰值底座\n归一化 R²_cb": num(peak_ens.get("norm_r2cb")), "Δ归一化 R²\nvs 峰值底座": num(ens["norm_r2cb"] - peak_ens["norm_r2cb"]), "同折/同seed峰值底座\n物理 R²_cb": num(peak_ens.get("pa_r2cb"))})
        v["门控结果"] = "阶段 3 确认：不设门"
        rows.append(("ensemble", v))
    return rows


# ----------------------------------------------------------------------------------------------------------------------------
def rows_m1(name, configs, exp, matrix, queue, group):
    gate = rj(exp / "offline/m1_report_best.json", {})
    rows = []
    for arm in matrix["arms"]:
        aid = arm["id"]; fold = arm.get("fold")
        cfg, run, mets, diag, hist = load_arm(name, arm, configs); best, last = mets["best"], mets["last"]
        rec = queue["arms"].get(aid, {})
        for ch in C.MULTI_CHANNELS:
            res = best if ch == "wss" else (best or {}).get("heads", {}).get(ch)
            res_last = last if ch == "wss" else (last or {}).get("heads", {}).get(ch)
            v = blank(); split = rj(cfg["data"]["split_path"], {})
            label = {"wss": "峰值帧 WSS 通道（与 X5D 同口径）", "tawss": "TAWSS 通道（log_z）", "osi": "OSI 通道（logit_z）"}[ch]
            v.update({"实验矩阵": group, "实验 / 模型": f"{aid}[{ch}]｜M1 三头单模型 {label}：{describe(name, aid)}｜{'已完成' if rec.get('status') == 'complete' else rec.get('status', '待跑')}",
                      "预测对象": {"wss": "WSS · Pa（峰值帧）", "tawss": "TAWSS · Pa", "osi": "OSI · 无量纲 [0,0.5]"}[ch], "检查点 / 汇总": "best（last 见本格批注）",
                      "评估范围 / 划分": f"V5.1 cv3_v51 fold{fold} 留出折（{len(split.get('test_cases', []))} 例）；test34 未用", "折": fold, "seed": arm.get("seed", 1234),
                      "目标空间": "三通道 [峰值 log_z, TAWSS log_z, OSI logit_z]，等权 MSE，pinball 关", "标签定义": "峰值帧 wall_wss + wss_min_cycle_v1（帧 0–79）",
                      "run id": f"{cfg['name']}[{ch}]", "配置路径": str(configs / arm["config"]), "metrics 路径": str(run / "eval/ckpt_best/metrics.json") + ("" if ch == "wss" else f"→heads.{ch}")})
            fill_training(v, cfg, run, diag, hist)
            note = f"Job {queue.get('job_id', '')}；{arm['title']}；通道 {ch}。"
            if res:
                fill_accuracy(v, res, res_last)
                if ch == "wss":
                    peak_reference(v, res, rj(RUNS / PEAK_FOLD_RUN / f"X5D_v51_f{fold}_s1234/eval/ckpt_best/metrics.json"))
                    v["自由基线 / 单头参照"] = f"同折峰值底座 X5D_v51_f{fold}（Δ 见峰值底座列）"
                else:
                    ref_id = "A1" if ch == "tawss" else "O2"
                    ref = rj(RUNS / "wss_cycle_20260920" / f"{ref_id}_f{fold}_s1234/eval/ckpt_best/metrics.json")
                    if ref:
                        rn, rp = dig(ref, "test", "normalized", "field_casebalanced", "r2"), dig(ref, "test", "field_casebalanced", "r2")
                        v.update({"自由基线 / 单头参照": f"单头 {ref_id}_f{fold}（阶段 1）归一化 R²_cb {rn:.4f} / 物理 {rp:.4f}", "参照主值": num(rn),
                                  "Δ主值 vs 参照": num(dig(res, "normalized", "field_casebalanced", "r2") - rn)})
                g = gate.get("gate", {})
                if g:
                    v["门控结果"] = (f"峰值掉≤0.02 {'PASS' if g.get('peak_drop_le_0.02_all_folds') else 'FAIL'}（Δ均值 {num(g.get('peak_delta_mean'))}）；"
                                     f"TAWSS 头≥A1−0.01 {'PASS' if g.get('tawss_head_ge_A1_minus_0.01') else 'FAIL'}；OSI 头≥O2−0.01 {'PASS' if g.get('osi_head_ge_O2_minus_0.01') else 'FAIL'}")
                if res_last:
                    note += f" last：归一化 R²_cb={num(dig(res_last, 'normalized', 'field_casebalanced', 'r2'))}，物理 R²_cb={num(res_last['field_casebalanced']['r2'])}。"
            v["备注"] = note
            rows.append(("m1", v))
    return rows


# ----------------------------------------------------------------------------------------------------------------------------
def rows_m1_stage3(name, configs, exp, matrix, queue, group):
    """M1 三头三 seed 确认（train136 → test34）：逐 seed 逐通道行 + 逐通道三 seed 集成行 + 集成滞留区行（offline/m1_stage3_report_best.json）。"""
    report = rj(exp / "offline/m1_stage3_report_best.json", {})
    c1 = rj(ROOT / "training_wss_min/experiments/wss_cycle_20260920/offline/c1_baselines.json", {}).get("test34", {})
    rows = []
    for arm in matrix["arms"]:
        aid = arm["id"]; seed = arm.get("seed", 1234)
        cfg, run, mets, diag, hist = load_arm(name, arm, configs); best, last = mets["best"], mets["last"]
        rec = queue["arms"].get(aid, {})
        for ch in C.MULTI_CHANNELS:
            res = best if ch == "wss" else (best or {}).get("heads", {}).get(ch)
            res_last = last if ch == "wss" else (last or {}).get("heads", {}).get(ch)
            v = blank()
            label = {"wss": "峰值帧 WSS 通道（与 X5D 同口径）", "tawss": "TAWSS 通道（log_z）", "osi": "OSI 通道（logit_z）"}[ch]
            v.update({"实验矩阵": group, "实验 / 模型": f"{aid}[{ch}]｜M1 三头单模型 {label}：{describe(name, aid)}｜{'已完成' if rec.get('status') == 'complete' else rec.get('status', '待跑')}",
                      "预测对象": {"wss": "WSS · Pa（峰值帧）", "tawss": "TAWSS · Pa", "osi": "OSI · 无量纲 [0,0.5]"}[ch], "检查点 / 汇总": "best（last 见本格批注）",
                      "评估范围 / 划分": "V5.1 train136 → test34（34 例，锁定确认，读一次）", "折": "train136→test34", "seed": seed,
                      "目标空间": "三通道 [峰值 log_z, TAWSS log_z, OSI logit_z]（train136 统计），等权 MSE，pinball 关", "标签定义": "峰值帧 wall_wss + wss_min_cycle_v1（帧 0–79）",
                      "run id": f"{cfg['name']}[{ch}]", "配置路径": str(configs / arm["config"]), "metrics 路径": str(run / "eval/ckpt_best/metrics.json") + ("" if ch == "wss" else f"→heads.{ch}")})
            fill_training(v, cfg, run, diag, hist)
            note = f"Job {queue.get('job_id', '')}；{arm['title']}；通道 {ch}；确认阶段不设门。"
            if res:
                fill_accuracy(v, res, res_last)
                if ch == "wss":
                    peak_reference(v, res, rj(RUNS / PEAK_SEED_RUN / f"X5D_v51_s{seed}/eval/ckpt_best/metrics.json"))
                    v["自由基线 / 单头参照"] = f"同 seed 已部署峰值模型 X5D_v51_s{seed}（Δ 见峰值底座列）"
                else:
                    ref_id = "A1" if ch == "tawss" else "O2"
                    ref = rj(RUNS / "wss_cycle_stage3_20260921" / f"{ref_id}_s{seed}/eval/ckpt_best/metrics.json")
                    if ref:
                        rn, rp = dig(ref, "test", "normalized", "field_casebalanced", "r2"), dig(ref, "test", "field_casebalanced", "r2")
                        v.update({"自由基线 / 单头参照": f"同 seed 单头 {ref_id}_s{seed}（阶段 3）归一化 R²_cb {rn:.4f} / 物理 {rp:.4f}", "参照主值": num(rn),
                                  "Δ主值 vs 参照": num(dig(res, "normalized", "field_casebalanced", "r2") - rn)})
                    free_baseline_reference(v, res, c1, ch == "osi") if ch != "wss" else None
                v["门控结果"] = "确认阶段：不设门；与同 seed 峰值 / 单头模型并列读"
                if res_last:
                    note += f" last：归一化 R²_cb={num(dig(res_last, 'normalized', 'field_casebalanced', 'r2'))}，物理 R²_cb={num(res_last['field_casebalanced']['r2'])}。"
            v["备注"] = note
            rows.append(("m1", v))
    refs = report.get("references", {})
    for ch, ens in (report.get("ensembles") or {}).items():
        if ch == "stagnation":
            ba = ens["area_share_ba"]; v = blank()
            v.update({"实验矩阵": group, "实验 / 模型": "M1_ens3 滞留区｜M1 三 seed 集成的 TAWSS 通道 < 0.4 Pa ∧ OSI 通道 > 0.1｜已完成", "预测对象": "滞留区掩膜（TAWSS<0.4 ∧ OSI>0.1）",
                      "检查点 / 汇总": "三 seed Pa 均值集成", "评估范围 / 划分": "test34（34 例）", "折": "train136→test34", "seed": "1234+7+2025",
                      "掩膜 1": "stagnation", "掩膜 1 Dice 中位": num(ens["dice_med"], 3), "掩膜 1 IoU 逐例均值": num(ens["iou_casemean"], 3),
                      "掩膜 1 面积份额 BA 偏差": num(ba["bias"], 3), "掩膜 1 面积份额 LoA 低": num(ba["loa_low"], 3), "掩膜 1 面积份额 LoA 高": num(ba["loa_high"], 3),
                      "自由基线 / 单头参照": f"test34 基线组合 IoU {num(ens['baseline_iou'], 3)}；单头集成 A1×O2 IoU {num(ens.get('single_head_A1xO2'), 3)}", "参照掩膜 IoU": num(ens["baseline_iou"], 3),
                      "Δ掩膜 IoU": num(ens["iou_casemean"] - ens["baseline_iou"], 3), "run id": "M1_ens3_stagnation", "metrics 路径": str(exp / "offline/m1_stage3_report_best.json"),
                      "备注": "单模型三输出的集成掩膜交集；顶点计数"})
            rows.append(("combo", v)); continue
        is_osi = ch == "osi"; v = blank()
        ref, rname = {"wss": (refs.get("peak_ens3"), "已部署峰值三 seed 集成"), "tawss": (refs.get("A1_ens3"), "单头 A1_ens3"), "osi": (refs.get("O2_ens3"), "单头 O2_ens3")}[ch]
        v.update({"实验矩阵": group, "实验 / 模型": f"M1_ens3[{ch}]｜M1 三 seed（1234/7/2025）test34 逐点 Pa 均值集成，通道 {ch}{'（裁 [0,0.5]）' if is_osi else ''}｜已完成",
                  "预测对象": {"wss": "WSS · Pa（峰值帧）", "tawss": "TAWSS · Pa", "osi": "OSI · 无量纲 [0,0.5]"}[ch], "检查点 / 汇总": "三 seed best 集成",
                  "评估范围 / 划分": "V5.1 train136 → test34（34 例）", "折": "train136→test34", "seed": "1234+7+2025", "目标空间": "集成在物理空间；归一化列用 train136 统计量映回 z",
                  "标签定义": "同单 seed 臂", "归一化 R²\n病例等权 cb": num(ens["norm_r2cb"]), "物理 R²\n病例等权 cb": num(ens["pa_r2cb"]), "逐例均值 R²": num(ens["r2_casemean"]),
                  "逐例 p10 R²": num(ens["r2_casep10"]), "负 R² 例": f"{ens['neg']}/{ens['n_cases']}", "Spearman\n逐例均值": num(ens["spearman"], 3), "斜率 cb": num(ens["slope_cb"], 3),
                  "top10 IoU\n逐例均值": num(ens.get("top10_iou"), 3), "run id": f"M1_ens3[{ch}]", "metrics 路径": str(exp / "offline/m1_stage3_report_best.json"),
                  "备注": f"三 seed 单模型物理 R²_cb 均值 {num(dig(ens, 'seed_mean', 'pa_r2cb'))} ± {num(dig(ens, 'seed_sd', 'pa_r2cb'))}；归一化 {num(dig(ens, 'seed_mean', 'norm_r2cb'))} ± {num(dig(ens, 'seed_sd', 'norm_r2cb'))}"})
        fill_agreement(v, ens.get("agreement"))
        for j, (mname, blk) in enumerate(sorted(ens.get("masks", {}).items())[:2], 1):
            v.update({f"掩膜 {j} IoU 逐例均值": num(blk["iou"], 3), f"掩膜 {j} 面积份额误差 med": num(blk["frac_abs_err_casemed"], 3)})
            if v[f"掩膜 {j}"] == MISSING:
                v[f"掩膜 {j}"] = mname
        if ref:
            v.update({"自由基线 / 单头参照": rname, "参照主值": num(ref["norm_r2cb"]), "Δ主值 vs 参照": num(ens["norm_r2cb"] - ref["norm_r2cb"]),
                      "同折/同seed峰值底座\n归一化 R²_cb": num(ref["norm_r2cb"]) if ch == "wss" else MISSING,
                      "Δ归一化 R²\nvs 峰值底座": num(ens["norm_r2cb"] - ref["norm_r2cb"]) if ch == "wss" else MISSING,
                      "同折/同seed峰值底座\n物理 R²_cb": num(ref["pa_r2cb"]) if ch == "wss" else MISSING})
        v["门控结果"] = "确认阶段：不设门"
        rows.append(("ensemble", v))
    return rows


OSI_TAIL_ARMS = {
    "M1r": ("同硬件对照（M1 配方不改）", "node04 A100（Slurm 之外）"),
    "K1": ("OSI 通道软掩膜 BCE（λ 0.1，OSI>0.2/0.3，温度 0.25 z）", "node04 A100（Slurm 之外）"),
    "K2": ("OSI 通道尾部加权 MSE（α 2，OSI>0.2，批内归一）", "master 4090（Slurm 15673）"),
    "K5": ("+8 维分离几何输入（F3 上游历史 5 + bif_angle_cos + carina_cos + log_r_over_rdistal）", "master 4090（Slurm 15673）"),
}


def rows_osi_tail(name, configs, exp, matrix, queue, group):
    """2026-09-24 OSI 尾部矩阵：M1 配方单变量臂 × cv3，逐通道行；参照 = 同折已存 M1（wss_cycle_m1_20260921）同通道；门控 = report_best.json。"""
    report = rj(exp / "report_best.json", {})
    status = {}
    for fname in ("queue_status_node04_train_wave1.json", "queue_status_node04_eval.json", "queue_status_master.json"):
        for aid, rec in (rj(exp / fname, {}) or {}).get("arms", {}).items():
            status.setdefault(aid, {}).update(rec.get("stages", {}))
    rows = []
    for arm in matrix["arms"]:
        aid, fold, key = arm["id"], arm["fold"], arm["arm"]
        cfg, run, mets, diag, hist = load_arm(name, arm, configs); best, last = mets["best"], mets["last"]
        ref_run = RUNS / "wss_cycle_m1_20260921" / f"M1_f{fold}_s1234/eval/ckpt_best/metrics.json"
        ref = (rj(ref_run) or {}).get("test")
        title, hw = OSI_TAIL_ARMS[key]
        done = all(status.get(aid, {}).get(s, {}).get("state") in ("done", "done_before_start") for s in ("train", "eval_best", "eval_last"))
        verdict = dig(report, "verdicts", key)
        for ch in C.MULTI_CHANNELS:
            res = best if ch == "wss" else (best or {}).get("heads", {}).get(ch)
            res_last = last if ch == "wss" else (last or {}).get("heads", {}).get(ch)
            rres = ref if ch == "wss" else (ref or {}).get("heads", {}).get(ch)
            v = blank(); split = rj(cfg["data"]["split_path"], {})
            label = {"wss": "峰值帧 WSS 通道", "tawss": "TAWSS 通道（log_z）", "osi": "OSI 通道（logit_z）"}[ch]
            v.update({"实验矩阵": group, "实验 / 模型": f"{aid}[{ch}]｜M1 三头 {label}：{title}｜{'已完成' if done else '未完成'}",
                      "预测对象": {"wss": "WSS · Pa（峰值帧）", "tawss": "TAWSS · Pa", "osi": "OSI · 无量纲 [0,0.5]"}[ch], "检查点 / 汇总": "best（last 见本格批注）",
                      "评估范围 / 划分": f"V5.1 cv3_v51 fold{fold} 留出折（{len(split.get('test_cases', []))} 例）；test34 未用；{hw}", "折": fold, "seed": arm.get("seed", 1234),
                      "目标空间": "三通道 [峰值 log_z, TAWSS log_z, OSI logit_z]，等权 MSE + 本臂改动", "标签定义": "峰值帧 wall_wss + wss_min_cycle_v1（帧 0–79）",
                      "run id": f"{cfg['name']}[{ch}]", "配置路径": str(configs / arm["config"]), "metrics 路径": str(run / "eval/ckpt_best/metrics.json") + ("" if ch == "wss" else f"→heads.{ch}")})
            fill_training(v, cfg, run, diag, hist)
            note = f"{title}；执行 {hw}；通道 {ch}；参照同折已存 M1_f{fold}（master 4090）。"
            if res:
                fill_accuracy(v, res, res_last)
                if rres:
                    rn = dig(rres, "normalized", "field_casebalanced", "r2")
                    v.update({"自由基线 / 单头参照": f"同折已存 M1_f{fold} 同通道 归一化 R²_cb {rn:.4f}", "参照主值": num(rn),
                              "Δ主值 vs 参照": num(dig(res, "normalized", "field_casebalanced", "r2") - rn)})
                    if ch == "osi":
                        i0 = dig(rres, "threshold_masks", "above_0.3", "iou_casemean")
                        i1 = dig(res, "threshold_masks", "above_0.3", "iou_casemean")
                        if i0 is not None and i1 is not None:
                            v.update({"参照掩膜 IoU": num(i0, 3), "Δ掩膜 IoU": num(i1 - i0, 3)})
                if verdict and key != "M1r":
                    parts = [f"对 {rn_}（{d['role']}）：{d['verdict']}（原始 IoU@0.3 Δ{d['delta']['iou_0.3']:+.3f}、校准后 Δ{d['delta']['cal_iou_0.3']:+.3f}、"
                             f"OSI R² Δ{d['delta']['r2_cb']:+.3f}）" for rn_, d in verdict.items()]
                    v["门控结果"] = "；".join(parts)
                elif key == "M1r":
                    jit = dig(report, "fold_mean", "jitter_M1r_minus_M1")
                    v["门控结果"] = (f"对照臂：M1r − 已存 M1 抖动 OSI R² {jit['r2_cb']:+.4f}、IoU@0.3 {jit['iou_0.3']:+.4f}" if jit else "对照臂")
                if res_last:
                    note += f" last：归一化 R²_cb={num(dig(res_last, 'normalized', 'field_casebalanced', 'r2'))}，物理 R²_cb={num(res_last['field_casebalanced']['r2'])}。"
            v["备注"] = note
            rows.append(("m1", v))
    return rows


OSI_STRUCT_ARMS = {
    "V1": "辅助通道 (mean_axial, mean_circ)：帧 0–79 平均 WSS 向量在 (轴向, 周向) 上的分量 ÷ 平均幅值；另报派生 OSI 头",
    "V2": "辅助通道 rev_frac（与峰值方向反向的帧份额）",
}


def rows_osi_struct(name, configs, exp, matrix, queue, group):
    """2026-09-25 任务 2：M1 配方 + 辅助通道 × v5.1 cv3，逐通道行（V1 另加派生 OSI 行）；参照 = 同折已存 M1 同通道；门控 = report_best.json。"""
    report = rj(exp / "report_best.json", {})
    rows = []
    for arm in matrix["arms"]:
        aid, fold, key = arm["id"], arm["fold"], arm["arm"]
        cfg, run, mets, diag, hist = load_arm(name, arm, configs); best, last = mets["best"], mets["last"]
        ref = (rj(RUNS / "wss_cycle_m1_20260921" / f"M1_f{fold}_s1234/eval/ckpt_best/metrics.json") or {}).get("test")
        heads = list(C.MULTI_CHANNELS) + (["osi_derived"] if key == "V1" else [])
        for ch in heads:
            res = best if ch == "wss" else (best or {}).get("heads", {}).get(ch)
            res_last = last if ch == "wss" else (last or {}).get("heads", {}).get(ch)
            rres = ref if ch == "wss" else (ref or {}).get("heads", {}).get("osi" if ch == "osi_derived" else ch)
            v = blank(); split = rj(cfg["data"]["split_path"], {})
            label = {"wss": "峰值帧 WSS 通道", "tawss": "TAWSS 通道", "osi": "OSI 直接头", "osi_derived": "派生 OSI = ½(1−min(1,‖(r_a,r_c)‖))"}[ch]
            v.update({"实验矩阵": group, "实验 / 模型": f"{aid}[{ch}]｜M1 + {OSI_STRUCT_ARMS[key]}｜{label}",
                      "预测对象": {"wss": "WSS · Pa（峰值帧）", "tawss": "TAWSS · Pa"}.get(ch, "OSI · 无量纲 [0,0.5]"), "检查点 / 汇总": "best（last 见本格批注）",
                      "评估范围 / 划分": f"V5.1 cv3_v51 fold{fold} 留出折（{len(split.get('test_cases', []))} 例）；test34 未用；node04 A100",
                      "折": fold, "seed": arm.get("seed", 1234), "目标空间": f"{cfg['model']['out_dim']} 通道等权 MSE（三头 + 辅助 {cfg['data'].get('multi_aux_channels')}）",
                      "标签定义": "峰值帧 wall_wss + wss_min_cycle_v1（帧 0–79）+ bundle wall_wss_vec 派生辅助通道",
                      "run id": f"{cfg['name']}[{ch}]", "配置路径": str(configs / arm["config"]),
                      "metrics 路径": str(run / "eval/ckpt_best/metrics.json") + ("" if ch == "wss" else f"→heads.{ch}")})
            fill_training(v, cfg, run, diag, hist)
            note = f"{OSI_STRUCT_ARMS[key]}；{label}；参照同折已存 M1_f{fold}。"
            if res:
                fill_accuracy(v, res, res_last)
                if rres:
                    rn = dig(rres, "normalized", "field_casebalanced", "r2")
                    v.update({"自由基线 / 单头参照": f"同折已存 M1_f{fold} {'OSI' if ch == 'osi_derived' else ch} 头 归一化 R²_cb {rn:.4f}", "参照主值": num(rn),
                              "Δ主值 vs 参照": num(dig(res, "normalized", "field_casebalanced", "r2") - rn)})
                    if ch in ("osi", "osi_derived"):
                        i0, i1 = dig(rres, "threshold_masks", "above_0.3", "iou_casemean"), dig(res, "threshold_masks", "above_0.3", "iou_casemean")
                        if i0 is not None and i1 is not None:
                            v.update({"参照掩膜 IoU": num(i0, 3), "Δ掩膜 IoU": num(i1 - i0, 3)})
                verdict = dig(report, "verdicts", f"{key}:{ch}")
                if verdict:
                    v["门控结果"] = "；".join(f"对 {r}（{d['reference_role']}）：{d['verdict']}（R² Δ{d['delta']['r2_cb']:+.3f}、校准后 IoU@0.3 Δ{d['delta']['cal_iou_0.3']:+.3f}）"
                                             for r, d in verdict.items())
                if res_last:
                    note += f" last：归一化 R²_cb={num(dig(res_last, 'normalized', 'field_casebalanced', 'r2'))}，物理 R²_cb={num(res_last['field_casebalanced']['r2'])}。"
            v["备注"] = note
            rows.append(("m1", v))
    return rows


def rows_m1_v52(name, configs, exp, matrix, queue, group):
    """2026-09-25 任务 1：M1cap × v5.2 × CV5 × 3 seed，逐 run 逐通道行；峰值通道参照 = 同折同 seed X5Dcap；测量不设门（汇总见 report_best.json）。"""
    report = rj(exp / "report_best.json", {})
    rows = []
    for arm in matrix["arms"]:
        aid, fold, seed = arm["id"], arm["fold"], arm["seed"]
        cfg, run, mets, diag, hist = load_arm(name, arm, configs); best, last = mets["best"], mets["last"]
        ref = (rj(RUNS / "wss_v52_20260923" / f"X5Dcap_v52cv_f{fold}_s{seed}/eval/ckpt_best/metrics.json") or {}).get("test")
        for ch in C.MULTI_CHANNELS:
            res = best if ch == "wss" else (best or {}).get("heads", {}).get(ch)
            res_last = last if ch == "wss" else (last or {}).get("heads", {}).get(ch)
            v = blank(); split = rj(cfg["data"]["split_path"], {})
            v.update({"实验矩阵": group, "实验 / 模型": f"{aid}[{ch}]｜M1 三头 × v5.2 X5Dcap 配方 CV5｜{ch}",
                      "预测对象": {"wss": "WSS · Pa（峰值帧）", "tawss": "TAWSS · Pa", "osi": "OSI · 无量纲 [0,0.5]"}[ch], "检查点 / 汇总": "best（last 见本格批注）",
                      "评估范围 / 划分": f"V5.2 cv5_v52 fold{fold} 留出折（{len(split.get('test_cases', []))} 例）；master 4090",
                      "折": fold, "seed": seed, "目标空间": "三通道 [峰值 log_z, TAWSS log_z, OSI logit_z]，等权 MSE，pinball 关",
                      "标签定义": "峰值帧 wall_wss + v5.2 wss_min_cycle_v1（帧 0–79）",
                      "run id": f"{cfg['name']}[{ch}]", "配置路径": str(configs / arm["config"]),
                      "metrics 路径": str(run / "eval/ckpt_best/metrics.json") + ("" if ch == "wss" else f"→heads.{ch}")})
            fill_training(v, cfg, run, diag, hist)
            note = f"任务 1（v5.2 数据效应测量）；通道 {ch}。"
            if res:
                fill_accuracy(v, res, res_last)
                if ch == "wss" and ref:
                    peak_reference(v, res, {"test": ref})
                    v["自由基线 / 单头参照"] = f"同折同 seed X5Dcap_v52cv_f{fold}_s{seed}（Δ 见峰值底座列）"
                ms = dig(report, "seed_mean")
                v["门控结果"] = ("测量不设门；护栏：峰值归一化逐折掉 ≤ 0.02 " + ("全部满足" if dig(report, "guard_summary", "peak_norm_drop_le_0.02_all") else "有不满足")
                                 + (f"；三 seed 261 例折外 OSI R² {ms['r2_cb']:.3f}、校准后 IoU@0.3 {ms['cal_iou_0.3']:.3f}" if ms else ""))
                if res_last:
                    note += f" last：归一化 R²_cb={num(dig(res_last, 'normalized', 'field_casebalanced', 'r2'))}，物理 R²_cb={num(res_last['field_casebalanced']['r2'])}。"
            v["备注"] = note
            rows.append(("m1", v))
    return rows


GROUPS["wss_osi_struct_20260925"] = "WSS周期积分量OSI目标结构矩阵｜2026-09-25（V1 平均切向向量/V2 rev_frac × cv3，逐通道 + 派生 OSI 行；node04 A100）"
GROUPS["wss_cycle_m1_v52_20260925"] = "WSS周期积分量M1三头×v5.2 X5Dcap CV5｜2026-09-25（5 折 × seed 1234/7/2025，逐 run 逐通道行；master 4090）"
GROUPS["wss_osi_tail_20260924"] ="WSS周期积分量OSI尾部矩阵｜2026-09-24（M1r/K1/K2/K5×cv3，逐通道行；M1r/K1 node04 A100，K2/K5 master 4090）"
DEFAULT_NAMES = DEFAULT_NAMES + ("wss_osi_tail_20260924", "wss_cycle_m1_v52_20260925", "wss_osi_struct_20260925")
BUILDERS = {"wss_cycle_20260920": rows_stage1, "wss_cycle_stage3_20260921": rows_stage3, "wss_cycle_m1_20260921": rows_m1,
            "wss_cycle_m1_stage3_20260922": rows_m1_stage3, "wss_osi_tail_20260924": rows_osi_tail,
            "wss_osi_struct_20260925": rows_osi_struct, "wss_cycle_m1_v52_20260925": rows_m1_v52}


def update_workbook(book: Path, names):
    records = []
    for name in names:
        configs = ROOT / "training_wss_min/configs" / name; exp = ROOT / "training_wss_min/experiments" / name
        matrix = rj(configs / "matrix.json")
        if not matrix:
            print(f"skip {name}: no matrix.json"); continue
        queue = rj(exp / "queue_status.json", {"arms": {}})
        records += BUILDERS[name](name, configs, exp, matrix, queue, GROUPS[name])
    if not any(r[0] == "arm" and r[1]["物理 R²\n病例等权 cb"] != MISSING for r in records):
        raise RuntimeError("no arm row carries metrics: check runs/<experiment>/<arm>/eval paths")
    before = sha(book)
    wb = load_workbook(book)
    others = [s for s in wb.worksheets if s.title != SHEET]
    preserved = {(s.title, c.coordinate): cell_record(c) for s in others for row in s for c in row}
    dims = {s.title: (copy.deepcopy(s.column_dimensions), copy.deepcopy(s.row_dimensions)) for s in others}
    merges = {s.title: tuple(map(str, s.merged_cells.ranges)) for s in others}
    if SHEET in wb.sheetnames:
        del wb[SHEET]
    style_src = wb[STYLE_SHEET]
    ws = wb.create_sheet(SHEET, index=wb.sheetnames.index(STYLE_SHEET) + 1)
    kinds = {k: sum(r[0] == k for r in records) for k in ("arm", "baseline", "combo", "ensemble", "m1")}
    ws["A1"] = f"{SHEET}  |  周期积分量（TAWSS / OSI）直接回归实验矩阵与落地一致性指标"
    ws["A2"] = ("每行一个臂 / 自由基线 / 组合 / 集成 / M1 通道；WSS 峰值帧线仍在「WSS实验矩阵」页按 R² / NMAE 口径报，本页只收周期积分量目标（M1 的峰值通道除外，作为多头对照）。"
                "03 节与 WSS 表同口径（O 臂的物理列为 OSI 无量纲）；04–05 节为 2026-09-21 新增的落地一致性指标（evaluate.py result['cycle_agreement'] 与 threshold_masks）。")
    ws["A3"] = (f"{len([n for n in names if (ROOT / 'training_wss_min/configs' / n / 'matrix.json').is_file()])} 组 · {len(records)} 条记录"
                f"（{kinds['arm']} 单目标臂 + {kinds['baseline']} 基线 + {kinds['combo']} 滞留区组合 + {kinds['ensemble']} 集成 + {kinds['m1']} M1 通道行）  |  — = 未产出或不适用")
    for r in (1, 2, 3):
        ws.cell(r, 1)._style = copy.copy(style_src.cell(r, 1)._style)
    col = 1
    for title, hs in SECTIONS:
        ws.cell(4, col).value = title; ws.cell(4, col)._style = copy.copy(style_src.cell(4, 1)._style)
        col += len(hs)
    for j, h in enumerate(HEADERS, 1):
        c = ws.cell(5, j); c.value = h; c._style = copy.copy(style_src.cell(5, 1)._style)
    for i, (kind, v) in enumerate(records):
        row = 6 + i
        for j, h in enumerate(HEADERS, 1):
            c = ws.cell(row, j); c.value = v[h]; c._style = copy.copy(style_src.cell(7 if i % 2 else 6, 1)._style)
        ws.cell(row, COL["备注"]).comment = Comment(str(v["备注"])[:32000], "cycle report")
        if kind in ("arm", "m1"):
            ws.cell(row, COL["检查点 / 汇总"]).comment = Comment(str(v["备注"])[:32000], "cycle report")
        ws.row_dimensions[row].height = 72
    last_row = 5 + len(records)
    ws.add_table(Table(displayName=TABLE, ref=f"A5:{ws.cell(5, len(HEADERS)).column_letter}{last_row}",
                       tableStyleInfo=TableStyleInfo(name="TableStyleMedium2", showRowStripes=True)))
    ws.freeze_panes = "D6"
    ws.column_dimensions["A"].width = 24; ws.column_dimensions["B"].width = 48
    for j in range(3, len(HEADERS) + 1):
        ws.column_dimensions[ws.cell(5, j).column_letter].width = 14
    ws.column_dimensions[ws.cell(5, COL["备注"]).column_letter].width = 60
    g0 = last_row + 3
    ws.cell(g0, 1).value = "指标说明（本页新增指标的口径）"; ws.cell(g0, 1)._style = copy.copy(style_src.cell(4, 1)._style)
    for i, (k, text) in enumerate(GLOSSARY, 1):
        ws.cell(g0 + i, 1).value = k; ws.cell(g0 + i, 2).value = text
        ws.cell(g0 + i, 1)._style = copy.copy(style_src.cell(6, 1)._style); ws.cell(g0 + i, 2)._style = copy.copy(style_src.cell(6, 2)._style)
        ws.row_dimensions[g0 + i].height = 48
    backup = book.with_name(book.stem + f"_backup_before_cycle_sheet_{time.strftime('%Y%m%d_%H%M%S')}.xlsx")
    backup.write_bytes(book.read_bytes())
    wb.save(book)
    check = load_workbook(book)
    for s in check.worksheets:
        if s.title == SHEET:
            continue
        for row in s:
            for c in row:
                if cell_record(c) != preserved[(s.title, c.coordinate)]:
                    raise AssertionError(f"historical cell changed: {s.title}!{c.coordinate}")
        assert tuple(map(str, s.merged_cells.ranges)) == merges[s.title], f"merged ranges changed: {s.title}"
        old_cols, old_rows = dims[s.title]
        assert {k: dict(v) for k, v in s.column_dimensions.items()} == {k: dict(v) for k, v in old_cols.items()}, f"column dims changed: {s.title}"
        assert {k: dict(v) for k, v in s.row_dimensions.items()} == {k: dict(v) for k, v in old_rows.items()}, f"row dims changed: {s.title}"
    evidence = {"passed": True, "sheet": SHEET, "table": TABLE, "experiments": list(names), "before_sha256": before, "after_sha256": sha(book), "backup": str(backup),
                "records": len(records), "record_kinds": kinds, "rows": {f"{r[1]['run id']}": 6 + i for i, r in enumerate(records)},
                "table_ref": f"A5:{ws.cell(5, len(HEADERS)).column_letter}{last_row}", "historical_cells_verified": len(preserved),
                "with_cycle_agreement": sum(1 for r in records if r[0] in ("arm", "m1", "ensemble") and r[1]["CCC 逐例中位"] != MISSING),
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (ROOT / "training_wss_min/experiments/wss_cycle_20260920/xlsx_cycle_sheet_acceptance.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2))
    return evidence


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--workbook", type=Path, default=BOOK)
    parser.add_argument("--name", action="append", default=None, help="experiment name(s); default = all three cycle experiments that have a matrix.json")
    args = parser.parse_args(argv)
    names = tuple(args.name) if args.name else DEFAULT_NAMES
    print(json.dumps(update_workbook(args.workbook, names), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
