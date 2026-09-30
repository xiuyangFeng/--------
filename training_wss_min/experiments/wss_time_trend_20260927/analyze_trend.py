#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""汇总 trend_eval.py 的逐例结果：对账表、趋势指标表、对 T-null 的配对差、分队列、残差分析、选例、图。

只读 per_case/*.jsonl 与 fold_summary/*.json，写 trend_report.md、trend_summary.json、case_selection.json、figs/*.png。
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

EXP = Path(__file__).resolve().parent
for ttc in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",):
    if Path(ttc).is_file():
        fm.fontManager.addfont(ttc)
plt.rcParams.update({"font.family": ["Noto Sans CJK JP", "Droid Sans Fallback", "DejaVu Sans"],
                     "axes.unicode_minus": False, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#8a8a85", "axes.labelcolor": "#3b3b38", "xtick.color": "#5c5c58",
                     "ytick.color": "#5c5c58", "axes.grid": True, "grid.color": "#e6e5df", "grid.linewidth": 0.8})
INK, INK2 = "#1f1f1d", "#5c5c58"
ARMS = os.environ.get("TREND_ARMS", "Tnull,C-raw,TT-raw,TT-raw-noattn,TT-warm,TT-warm-noattn,T0").split(",")   # 预览时可去掉未跑完的臂
FIG_ARMS = [a for a in ("Tnull", "C-raw", "TT-raw", "TT-warm", "T0") if a in ARMS]      # 图只画主臂；对角注意力两臂只进表
LABEL = {"Tnull": "T-null", "C-raw": "C-raw", "TT-raw": "TT-raw", "TT-warm": "TT-warm", "T0": "T0",
         "TT-raw-noattn": "TT-raw-noattn", "TT-warm-noattn": "TT-warm-noattn"}
COLOR = {"Tnull": "#2a78d6", "C-raw": "#eb6834", "T0": "#1baf7a", "TT-raw": "#eda100", "TT-warm": "#e87ba4"}
PERIOD, PEAK_WIN, TROUGH = 80, list(range(17, 27)), list(range(5, 10)) + list(range(43, 58))

# (列名, 中文说明, 方向：+1 越大越好 / -1 越小越好 / 0 越接近 1 越好)
TREND = [
    ("spearman_cycle", "逐帧空间 Spearman（全周期）", 1),
    ("spearman_peak", "  峰值窗", 1), ("spearman_decel", "  减速段", 1),
    ("spearman_trough", "  谷底 20 帧", 1), ("spearman_plateau", "  舒张平台", 1),
    ("lowq_dice_cycle", "最低 20 % 区位置 Dice", 1), ("highq_dice_cycle", "最高 10 % 区位置 Dice", 1),
    ("lowwss_dice_cycle", "低 WSS（<0.4 Pa）区 Dice", 1), ("lowwss_frac_mae", "低 WSS 面积占比曲线 MAE", -1),
    ("lowwss_frac_corr", "低 WSS 面积占比曲线相关", 1),
    ("global_corr", "全壁平均 WSS 曲线相关", 1), ("global_amp_ratio", "全壁平均 WSS 峰谷幅度比（预测/真值）", 0),
    ("global_shape_nrmse", "全壁平均 WSS 归一化形状误差", -1),
    ("ts_corr_ln", "逐点时序相关（ln，81 帧，原口径）", 1), ("ts_corr_pa", "逐点时序相关（Pa）", 1),
    ("shape_nrmse_med", "逐点归一化波形误差（中位）", -1),
    ("peak_time_err_p90", "峰时误差 P90（帧）", -1), ("trough_time_err_med", "谷时误差中位（帧）", -1),
    ("trough_time_err_p90", "谷时误差 P90（帧）", -1),
    ("ts_corr_ln_sync", "逐点时序相关：同步区", 1), ("ts_corr_ln_mid", "  过渡区", 1),
    ("ts_corr_ln_async", "  非同步区", 1),
]
REPRO = [("cycle_r2cb_pa", "全周期 Pa R²_cb"), ("trough_r2cb_pa", "谷底 Pa R²_cb"), ("peak_r2cb_pa", "峰值帧 Pa R²_cb"),
         ("tawss_r2cb_pa", "TAWSS Pa R²_cb"), ("ts_corr_ln", "ln 时序相关"), ("peak_time_err_p90", "峰时 P90（帧）")]


def load() -> pd.DataFrame:
    rows = []
    for arm in ARMS:
        for f in sorted((EXP / "per_case").glob(f"{arm}_f*.jsonl")):
            for line in f.read_text().splitlines():
                r = json.loads(line)
                if r["arm"] != arm:
                    continue
                r.pop("moments", None)
                rows.append(r)
    df = pd.DataFrame(rows).drop_duplicates(["arm", "unit_id"])
    counts = df.groupby("arm").size().to_dict()
    for arm in ARMS:
        if counts.get(arm) != 136:
            raise RuntimeError(f"{arm}: {counts.get(arm)} cases, expected 136")
    return df


def fold_mean(df: pd.DataFrame, arm: str, col: str):
    s = df[df.arm == arm].groupby("fold")[col].mean()
    return float(s.mean()), float(s.min()), float(s.max())


def boot_ci(d: np.ndarray, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    d = d[np.isfinite(d)]
    means = rng.choice(d, (n, len(d)), replace=True).mean(1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def paired(df: pd.DataFrame, arm: str, ref: str, col: str, sign: int) -> Dict:
    a = df[df.arm == arm].set_index("unit_id")
    b = df[df.arm == ref].set_index("unit_id")
    if sign == 0:
        d = (np.abs(np.log(b[col])) - np.abs(np.log(a.loc[b.index, col]))).to_numpy()  # >0 = arm 更接近 1
    else:
        d = sign * (a.loc[b.index, col] - b[col]).to_numpy()                            # >0 = arm 更好
    folds = a.loc[b.index, "fold"].to_numpy()
    fold_signs = [float(np.nanmean(d[folds == f])) for f in range(3)]
    lo, hi = boot_ci(d)
    return {"mean_improvement": float(np.nanmean(d)), "ci95": [lo, hi], "win_rate": float(np.mean(d[np.isfinite(d)] > 0)),
            "folds_positive": int(sum(v > 0 for v in fold_signs)), "fold_means": fold_signs}


def frame_matrix(df: pd.DataFrame, arm: str, key: str) -> np.ndarray:
    return np.array([r[key] for r in df[df.arm == arm]["frame"]], dtype=np.float64)


def fig_frame_curves(df: pd.DataFrame, q: np.ndarray, t: np.ndarray, path: Path):
    fig = plt.figure(figsize=(10, 7.2))
    gs = fig.add_gridspec(3, 1, height_ratios=[0.55, 1.6, 1.3], hspace=0.35)
    axq = fig.add_subplot(gs[0]); ax1 = fig.add_subplot(gs[1], sharex=axq); ax2 = fig.add_subplot(gs[2], sharex=axq)
    axq.plot(t, q, color=INK2, lw=2); axq.set_ylabel("Q/Qpeak"); axq.set_title("入口流量（协议波形，所有病例相同）", fontsize=10, loc="left")
    for ax in (axq, ax1, ax2):
        ax.axvspan(t[PEAK_WIN[0]], t[PEAK_WIN[-1]], color="#efeee9", lw=0, zorder=0)
        ax.axvspan(t[43], t[57], color="#f6efe4", lw=0, zorder=0)
    ends = []
    for arm in FIG_ARMS:
        m = np.nanmean(frame_matrix(df, arm, "spearman"), axis=0)
        ax1.plot(t, m, color=COLOR[arm], lw=2, label=LABEL[arm])
        ends.append((m[-1], arm))
        d = np.nanmean(frame_matrix(df, arm, "lowq_dice"), axis=0)
        ax2.plot(t, d, color=COLOR[arm], lw=2)
    ax1.set_ylabel("空间 Spearman"); ax1.set_title("逐帧空间 Spearman（136 例留出均值；与幅值无关，只看「哪里高哪里低」）", fontsize=10, loc="left")
    ax2.set_ylabel("Dice"); ax2.set_title("逐帧最低 20 % 区位置 Dice（136 例均值）", fontsize=10, loc="left")
    ax2.set_xlabel("t (s)；灰底 = 峰值窗 10 帧，浅橙底 = 谷底段")
    ax1.legend(ncol=5, fontsize=9, frameon=False, loc="lower left")
    for ax in (axq, ax1):
        plt.setp(ax.get_xticklabels(), visible=False)
    fig.savefig(path, dpi=140, bbox_inches="tight"); plt.close(fig)


def fig_strata(df: pd.DataFrame, path: Path):
    strata = [("ts_corr_ln_sync", "同步区\n(r≥0.9)"), ("ts_corr_ln_mid", "过渡区\n(0.5–0.9)"), ("ts_corr_ln_async", "非同步区\n(r<0.5)")]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    width = 0.8 / len(FIG_ARMS)
    for i, arm in enumerate(FIG_ARMS):
        vals = [fold_mean(df, arm, c)[0] for c, _ in strata]
        x = np.arange(len(strata)) + (i - (len(FIG_ARMS) - 1) / 2) * width
        ax.bar(x, vals, width * 0.9, color=COLOR[arm], label=LABEL[arm])
    af = [fold_mean(df, "Tnull", c.replace("ts_corr_ln", "area_frac"))[0] for c, _ in strata]
    ax.set_xticks(range(len(strata)))
    ax.set_xticklabels([f"{s}\n占壁面 {a * 100:.0f} %" for (_, s), a in zip(strata, af)])
    ax.set_ylim(0, 1); ax.set_ylabel("逐点 ln 时序相关（面积加权）")
    ax.set_title("按真值「是否跟随全壁脉动」分层的逐点时序相关（分层只用真值：r = 该点 ln 波形与本例全壁平均 ln 波形的相关）", fontsize=9.5, loc="left")
    ax.legend(ncol=5, fontsize=9, frameon=False, loc="upper right")
    fig.savefig(path, dpi=140, bbox_inches="tight"); plt.close(fig)


def fig_peak_vs_trend(df: pd.DataFrame, path: Path) -> Dict:
    arms = [a for a in ("Tnull", "C-raw", "T0") if a in ARMS]
    fig, axes = plt.subplots(1, len(arms), figsize=(12, 3.9), sharey=True)
    out = {}
    for ax, arm in zip(axes, arms):
        d = df[df.arm == arm]
        x, y = d["peak_r2_case_pa"].clip(lower=-1), d["spearman_cycle"]
        rho = spearmanr(d["peak_r2_case_pa"], d["spearman_cycle"]).correlation
        rho2 = spearmanr(d["peak_r2_case_pa"], d["ts_corr_ln"]).correlation
        out[arm] = {"rho_peak_r2_vs_spearman_cycle": float(rho), "rho_peak_r2_vs_ts_corr_ln": float(rho2)}
        ax.scatter(x, y, s=22, color=COLOR[arm], alpha=0.75, edgecolors="white", linewidths=0.6)
        ax.set_title(f"{LABEL[arm]}：Spearman ρ = {rho:.2f}", fontsize=10, loc="left")
        ax.set_xlabel("该例峰值帧 R²（Pa，截到 ≥ −1）")
    axes[0].set_ylabel("该例全周期空间 Spearman")
    fig.suptitle("残差分析：峰值帧误差会不会传到全周期趋势（每点 = 一个留出病例）", fontsize=10.5, x=0.01, y=1.05, ha="left")
    fig.savefig(path, dpi=140, bbox_inches="tight"); plt.close(fig)
    return out


def select_cases(df: pd.DataFrame, best: str) -> List[Dict]:
    b = df[df.arm == best].set_index("unit_id")
    tn = df[df.arm == "Tnull"].set_index("unit_id")
    picks: List[Dict] = []
    used = set()
    for cohort in ("AAA", "AG", "ILO"):
        sub = b[b.cohort == cohort]
        med = sub["spearman_cycle"].median()
        uid = (sub["spearman_cycle"] - med).abs().sort_values().index[0]
        picks.append({"unit_id": uid, "reason": f"{cohort} 队列中位（{best} 全周期空间 Spearman {b.loc[uid, 'spearman_cycle']:.2f}，队列中位 {med:.2f}）"})
        used.add(uid)
    gain = (b["spearman_cycle"] - tn.loc[b.index, "spearman_cycle"]).drop(index=list(used))
    uid = gain.sort_values().index[-1]
    picks.append({"unit_id": uid, "reason": f"{best} 相对 T-null 全周期空间 Spearman 提升最大（+{gain[uid]:.2f}）"}); used.add(uid)
    worst = b["spearman_cycle"].drop(index=list(used)).sort_values().index[0]
    picks.append({"unit_id": worst, "reason": f"{best} 全周期空间 Spearman 最低（{b.loc[worst, 'spearman_cycle']:.2f}，失败例）"})
    for p in picks:
        p["fold"] = int(b.loc[p["unit_id"], "fold"])
        p["metrics"] = {a: {k: float(df[(df.arm == a) & (df.unit_id == p["unit_id"])][k].iloc[0])
                            for k in ("spearman_cycle", "spearman_trough", "ts_corr_ln", "peak_r2_case_pa")}
                        for a in ARMS}
    return picks


def fmt(v, nd=3):
    return "—" if v is None or not np.isfinite(v) else f"{v:.{nd}f}"


def main():
    df = load()
    wf = json.loads(Path("/public/newhome/cy/Digital_twin/GNN/training_wss_min/experiments/wss_time_ecc_20260918/offline/protocol_inlet_waveform_v51.json").read_text())
    q, t = np.asarray(wf["q_norm"][:PERIOD]), np.asarray(wf["time_s"][:PERIOD]) - wf["time_s"][0]
    figs = EXP / "figs"; figs.mkdir(exist_ok=True)
    summary: Dict = {"n_cases_per_arm": 136, "protocol": "v5.1 cv3 三折留出、seed 1234；趋势指标帧 0–79；fold 均值再三折算术平均"}
    # 对账
    recon = {}
    for arm in ARMS:
        fs = [json.loads((EXP / "fold_summary" / f"{arm}_f{f}.json").read_text()) for f in range(3)]
        recon[arm] = {"max_abs_diff": max(s["reference_max_abs_diff"] for s in fs),
                      **{k: float(np.mean([s["reproduced"][k] for s in fs])) for k, _ in REPRO}}
    summary["reconciliation"] = recon
    # 趋势表
    table = {arm: {c: fold_mean(df, arm, c) for c, _, _ in TREND} for arm in ARMS}
    summary["trend"] = {arm: {c: v[0] for c, v in tab.items()} for arm, tab in table.items()}
    learned = [a for a in ARMS if a != "Tnull"]
    main_learned = [a for a in learned if "noattn" not in a]
    anchored = [a for a in main_learned if a != "T0"]     # 选例只看锚定臂（峰值帧 = 部署 X5D，才可能直接进展示），与 T0 是否跑完无关
    best = max(anchored, key=lambda a: table[a]["spearman_cycle"][0])
    summary["best_learned_by_spearman_cycle"] = best
    # 配对差
    pairs = {arm: {c: paired(df, arm, "Tnull", c, s) for c, _, s in TREND} for arm in learned}
    pairs_t0 = {c: paired(df, "C-raw", "T0", c, s) for c, _, s in TREND} if "T0" in ARMS else {}
    summary["paired_vs_tnull"], summary["paired_craw_vs_t0"] = pairs, pairs_t0
    KEY = [("spearman_cycle", 1), ("spearman_trough", 1), ("lowq_dice_cycle", 1), ("ts_corr_ln_async", 1),
           ("shape_nrmse_med", -1), ("global_shape_nrmse", -1)]
    others = [a for a in main_learned if a != "C-raw"]
    attn_pairs = {f"{a}−{a}-noattn": {c: paired(df, a, f"{a}-noattn", c, s) for c, s in KEY}
                  for a in ("TT-warm", "TT-raw") if a in ARMS and f"{a}-noattn" in ARMS}
    summary["paired_attention"] = attn_pairs
    pairs_vs_craw = {a: {c: paired(df, a, "C-raw", c, s) for c, s in KEY} for a in others}
    summary["paired_vs_craw"] = pairs_vs_craw
    # 残差：峰值窗空间格局好坏能否预示谷底格局好坏
    summary["peakwin_vs_trough"] = {a: float(spearmanr(df[df.arm == a]["spearman_peak"], df[df.arm == a]["spearman_trough"]).correlation)
                                    for a in ARMS}
    # 分队列
    key_cols = ["spearman_cycle", "spearman_trough", "lowq_dice_cycle", "lowwss_frac_mae", "ts_corr_ln_async", "area_frac_async"]
    cohort = {arm: {c: {k: float(df[(df.arm == arm) & (df.cohort == c)][k].mean()) for k in key_cols}
                    for c in ("AAA", "AG", "ILO")} for arm in ARMS}
    ncoh = df[df.arm == "Tnull"].groupby("cohort").size().to_dict()
    summary["by_cohort"], summary["cohort_n"] = cohort, ncoh
    # 图 + 残差
    fig_frame_curves(df, q, t, figs / "fig1_frame_spearman_dice.png")
    fig_strata(df, figs / "fig2_sync_strata.png")
    summary["peak_vs_trend"] = fig_peak_vs_trend(df, figs / "fig3_peak_vs_trend.png")
    # T-null 构造性质：逐帧排序与峰值预测相同
    summary["tnull_rank_invariance_note"] = "T-null ln = μ(t)+σ(t)·z(x)，σ>0 → 每帧空间排序与峰值预测完全相同，空间 Spearman 的时间变化只来自真值自身的变化"
    picks = select_cases(df, best)
    (EXP / "case_selection.json").write_text(json.dumps({"rule": f"按 {best}（锚定学习臂中全周期空间 Spearman 最高；T0 不锚定、不参与选例）：三队列各取距队列中位最近的一例 + 相对 T-null 提升最大一例 + 最低一例；选例只看指标，不看动画",
                                                         "cases": picks}, indent=1, ensure_ascii=False))
    (EXP / "trend_summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    # ---- markdown
    L = ["# 全周期趋势指标零训练复评 · 自动汇总", "",
         "由 `analyze_trend.py` 生成；逐例数据在 `per_case/`，对账在 `fold_summary/`。口径：v5.1 cv3 三折留出 136 例、seed 1234；"
         "趋势指标用帧 0–79（一个 0.8 s 周期），每折病例均值再三折算术平均；括号内为三折最小–最大。", "",
         "## 1. 对账（复现已存主指标）", "",
         "| 臂 | " + " | ".join(n for _, n in REPRO) + " | 与已存值最大绝对差 |", "|---|" + "---:|" * (len(REPRO) + 1)]
    for arm in ARMS:
        L.append(f"| {LABEL[arm]} | " + " | ".join(fmt(recon[arm][k], 4) for k, _ in REPRO) + f" | {recon[arm]['max_abs_diff']:.1e} |")
    L += ["", "## 2. 趋势指标（三折均值；括号 = 三折范围）", "",
          "| 指标 | 方向 | " + " | ".join(LABEL[a] for a in ARMS) + " |", "|---|:-:|" + "---:|" * len(ARMS)]
    for c, name, s in TREND:
        arrow = "↑" if s == 1 else ("↓" if s == -1 else "≈1")
        L.append(f"| {name} | {arrow} | " + " | ".join(f"{table[a][c][0]:.3f} ({table[a][c][1]:.3f}–{table[a][c][2]:.3f})" for a in ARMS) + " |")
    af = {k: fold_mean(df, "Tnull", f"area_frac_{k}")[0] for k in ("sync", "mid", "async")}
    L += ["", f"分层面积占比（真值定义，与臂无关）：同步区 {af['sync'] * 100:.1f} %、过渡区 {af['mid'] * 100:.1f} %、非同步区 {af['async'] * 100:.1f} %。", "",
          "## 3. 相对 T-null 的配对改进（136 例；正 = 更好；CI 为病例自助法 95 %）", ""]
    L += ["| 指标 | " + " | ".join(LABEL[a] for a in main_learned) + " |", "|---|" + "---|" * len(main_learned)]
    for c, name, s in TREND:
        cells = []
        for a in main_learned:
            p = pairs[a][c]
            cells.append(f"{p['mean_improvement']:+.3f} [{p['ci95'][0]:+.3f}, {p['ci95'][1]:+.3f}] 胜 {p['win_rate'] * 100:.0f} % 折 {p['folds_positive']}/3")
        L.append(f"| {name} | " + " | ".join(cells) + " |")
    L += ["", "## 4. C-raw 对 T0（正 = C-raw 更好）", "", "| 指标 | 均值差 [95 % CI] | 胜率 | 同向折 |", "|---|---|---:|---:|"]
    for c, name, s in (TREND if pairs_t0 else []):
        p = pairs_t0[c]
        L.append(f"| {name} | {p['mean_improvement']:+.3f} [{p['ci95'][0]:+.3f}, {p['ci95'][1]:+.3f}] | {p['win_rate'] * 100:.0f} % | {p['folds_positive']}/3 |")
    L += ["", "## 4b. 其他学习臂对 C-raw（正 = 该臂更好；CI 为病例自助法 95 %）", "",
          "| 指标 | " + " | ".join(LABEL[a] for a in others) + " |", "|---|" + "---|" * len(others)]
    for c, s in KEY:
        name = dict((k, n) for k, n, _ in TREND)[c]
        L.append(f"| {name.strip()} | " + " | ".join(
            f"{pairs_vs_craw[a][c]['mean_improvement']:+.3f} [{pairs_vs_craw[a][c]['ci95'][0]:+.3f}, {pairs_vs_craw[a][c]['ci95'][1]:+.3f}] 胜 {pairs_vs_craw[a][c]['win_rate'] * 100:.0f} % 折 {pairs_vs_craw[a][c]['folds_positive']}/3"
            for a in others) + " |")
    L += ["", "## 4c. 跨帧注意力的贡献（全注意力 − 对角注意力，同参数同初始化；正 = 注意力更好）", "",
          "| 指标 | " + " | ".join(attn_pairs) + " |", "|---|" + "---|" * len(attn_pairs)]
    for c, s in KEY:
        name = dict((k, n) for k, n, _ in TREND)[c]
        L.append(f"| {name.strip()} | " + " | ".join(
            f"{v[c]['mean_improvement']:+.3f} [{v[c]['ci95'][0]:+.3f}, {v[c]['ci95'][1]:+.3f}] 胜 {v[c]['win_rate'] * 100:.0f} % 折 {v[c]['folds_positive']}/3"
            for v in attn_pairs.values()) + " |")
    L += ["", f"## 5. 分队列（病例均值；AAA {ncoh.get('AAA')} / AG {ncoh.get('AG')} / ILO {ncoh.get('ILO')} 例）", "",
          "| 臂 | 队列 | " + " | ".join(key_cols) + " |", "|---|---|" + "---:|" * len(key_cols)]
    for arm in ARMS:
        for c in ("AAA", "AG", "ILO"):
            L.append(f"| {LABEL[arm]} | {c} | " + " | ".join(fmt(cohort[arm][c][k]) for k in key_cols) + " |")
    L += ["", "## 6. 残差分析：峰值帧误差与趋势", "", "| 臂 | ρ(该例峰值 R², 全周期空间 Spearman) | ρ(该例峰值 R², ln 时序相关) |", "|---|---:|---:|"]
    for arm, v in summary["peak_vs_trend"].items():
        L.append(f"| {LABEL[arm]} | {v['rho_peak_r2_vs_spearman_cycle']:.2f} | {v['rho_peak_r2_vs_ts_corr_ln']:.2f} |")
    L += ["", "峰值窗空间 Spearman 与谷底空间 Spearman 的跨病例秩相关（峰值格局准的病例，谷底格局是否也准）：" +
          "、".join(f"{LABEL[a]} {v:.2f}" for a, v in summary["peakwin_vs_trough"].items()) + "。"]
    L += ["", "## 7. 动画选例", "", f"规则：{json.loads((EXP / 'case_selection.json').read_text())['rule']}", "",
          "| 病例 | 折 | 理由 | T-null Spearman | C-raw | T0 |", "|---|---:|---|---:|---:|---:|"]
    for p in picks:
        m = p["metrics"]
        L.append(f"| {p['unit_id']} | {p['fold']} | {p['reason']} | {m['Tnull']['spearman_cycle']:.3f} | {m['C-raw']['spearman_cycle']:.3f} | {m.get('T0', {}).get('spearman_cycle', float('nan')):.3f} |")
    L += ["", "## 8. 图", "", "![逐帧空间 Spearman 与低剪切区 Dice](figs/fig1_frame_spearman_dice.png)", "",
          "![按同步性分层的时序相关](figs/fig2_sync_strata.png)", "", "![峰值误差与趋势](figs/fig3_peak_vs_trend.png)", ""]
    (EXP / "trend_report.md").write_text("\n".join(L), encoding="utf-8")
    print(f"best learned arm: {best}; picks: {[p['unit_id'] for p in picks]}")


if __name__ == "__main__":
    main()
