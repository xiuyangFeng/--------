#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成《WSS_PointNet实验矩阵指标汇总》PPT 的结果图。

数据源:docs/03-汇报材料/WSS_PointNet实验矩阵与结果汇总last.xlsx 的「实验矩阵总览」
(该 sheet 为真源;教师汇报视图/汇总对比 的 SA1 段存在列错位,不作数据源)。
输出:docs/03-汇报材料/figures/WSS_PointNet矩阵汇总_20260721/*.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import openpyxl
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
XLSX = ROOT / "WSS_PointNet实验矩阵与结果汇总last.xlsx"
OUT = ROOT / "figures" / "WSS_PointNet矩阵汇总_20260721"
OUT.mkdir(parents=True, exist_ok=True)

FONT = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
if FONT.exists():
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(FONT)).get_name()
plt.rcParams["axes.unicode_minus"] = False

# dataviz 参考调色板(light):PointNet=蓝,PointNet++=青,强调=黄,No-Go/负向=红,中性=灰
C_PN = "#2a78d6"      # PointNet
C_PNPP = "#1baf7a"    # PointNet++ SA3
C_HL = "#eda100"      # 高亮/候选
C_BAD = "#e34948"     # 负向/No-Go
C_VIOLET = "#4a3aa7"
C_GRAY = "#8a8a85"
GRID = dict(color="#e5e5e3", linewidth=0.8)
SEQ_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
            "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]


def style_ax(ax, ygrid=True):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c9c9c5")
    ax.tick_params(colors="#555555", labelsize=9)
    if ygrid:
        ax.yaxis.grid(True, **GRID)
        ax.set_axisbelow(True)


def load_overview():
    wb = openpyxl.load_workbook(XLSX, data_only=True)
    ws = wb["实验矩阵总览"]
    rows = {}
    order = []
    for r in ws.iter_rows(min_row=3, values_only=True):
        if r[0] is None or r[2] is None:
            continue
        key = str(r[0]).strip()
        rows[key] = r
        order.append(key)
    return rows, order


ROWS, ORDER = load_overview()

# 总览列号(0-based):6 物理R²cb,7 R²raw,8 case mean,11 负例,12 RMSE,13 MAE,
# 14/15 NRMSE pooled/case-mean,16/17 NMAE pooled/case-mean,18 highWSS R²,
# 19 top10幅值比,20 top10 IoU,21 归一化R²cb,22 归一化R²raw,31 Spearman,
# 32 highWSS Spearman,34 p99比
COL = dict(r2cb=6, r2raw=7, casemean=8, neg=11, rmse=12, mae=13,
           nrmse_p=14, nrmse_c=15, nmae_p=16, nmae_c=17, high=18,
           top10ratio=19, iou=20, nr2cb=21, nr2raw=22, spear=32,
           highspear=33, p99=35)


def v(key, col):
    row = ROWS.get(key)
    if row is None:
        raise KeyError(key)
    x = row[COL[col]]
    return None if x in (None, "") else float(x)


def bar_labels(ax, bars, fmt="{:.3f}", dy=0.004, fs=9, color="#333333"):
    for b in bars:
        h = b.get_height()
        va = "bottom" if h >= 0 else "top"
        off = dy if h >= 0 else -dy
        ax.text(b.get_x() + b.get_width() / 2, h + off, fmt.format(h),
                ha="center", va=va, fontsize=fs, color=color)


def save(fig, name):
    fig.savefig(OUT / name, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


# ---------------------------------------------------------------- F1 test16 基线
def fig_test16():
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 3.9),
                             gridspec_kw={"width_ratios": [3, 2]})
    ax = axes[0]
    keys = ["Point初始模版", "Point容量6-256-512", "Point初始模版+random采样",
            "Point容量6-256-512+random采样", "Point初始模版+容量6-64-128-256-512"]
    labels = ["E0\n原始模版", "E2\n宽通道", "E3\nrandom5000", "E23\n宽+5000", "E4\n加深"]
    vals = [v(k, "r2cb") for k in keys]
    colors = [C_PN if i == 1 else "#9ec5f4" for i in range(5)]
    bars = ax.bar(labels, vals, color=colors, width=0.62)
    bar_labels(ax, bars)
    ax.set_ylabel("物理 R²_cb(test16)")
    ax.set_title("容量是主增量:宽通道 E2 一次性 +0.073;点数/深度无附加收益", fontsize=11, loc="left")
    ax.set_ylim(0, 0.27)
    style_ax(ax)

    ax = axes[1]
    pairs = [("E2", v("Point容量6-256-512", "nr2cb"), v("Point容量6-256-512+目标wss/wssmax", "nr2cb")),
             ("E3", v("Point初始模版+random采样", "nr2cb"), v("Point初始模版+random采样+目标wss/wssmax", "nr2cb"))]
    x = np.arange(2)
    w = 0.32
    b1 = ax.bar(x - w / 2, [p[1] for p in pairs], w, color=C_PN, label="global log-z")
    b2 = ax.bar(x + w / 2, [p[2] for p in pairs], w, color=C_BAD, label="WSS/WSSmax")
    bar_labels(ax, b1); bar_labels(ax, b2)
    ax.set_xticks(x); ax.set_xticklabels([p[0] for p in pairs])
    ax.set_ylabel("归一化 R²_cb(test16)")
    ax.set_title("逐病例 WSS/WSSmax:No-Go", fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9)
    ax.set_ylim(0, 0.55)
    style_ax(ax)
    fig.tight_layout()
    save(fig, "f1_test16_baseline.png")


# ---------------------------------------------------------------- F2 v4/分层锚点
def fig_v4_anchors():
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 3.9),
                             gridspec_kw={"width_ratios": [5, 3]})
    ax = axes[0]
    keys = ["Point容量6-256-512+AG(v4)", "Point容量6-256-512+AG+AAA数据(v4)",
            "Point++SA3+AG(v4)", "Point++SA3+AG+AAA数据(锁定划分)"]
    labels = ["E2 AG61", "E2 AG61+AAA57", "SA3 AG61", "SA3 AG+AAA(锁定)"]
    vals = [v(k, "r2cb") for k in keys]
    colors = [C_PN, C_PN, C_PNPP, C_PNPP]
    bars = ax.bar(labels, vals, color=colors, width=0.58)
    bar_labels(ax, bars)
    ax.axhline(0.2597, color=C_GRAY, linestyle="--", linewidth=1.2)
    ax.text(3.45, 0.262, "旧 v3 E2 锚点 0.2597", fontsize=8.5, color="#666666", ha="right")
    ax.set_ylabel("物理 R²_cb(common-test15)")
    ax.set_title("v4 发布后 AG 重跑低于旧锚点;AAA 扩容仅小幅补偿", fontsize=11, loc="left")
    ax.set_ylim(0, 0.30)
    style_ax(ax)

    ax = axes[1]
    keys = ["Point++SA3+AG+AAA数据(分层划分)", "Point容量6-256-512+AG+AAA数据(分层划分)"]
    labels = ["SA3 / 9169", "E2 / 9170"]
    vals = [v(k, "r2cb") for k in keys]
    bars = ax.bar(labels, vals, color=[C_PNPP, C_PN], width=0.5)
    bar_labels(ax, bars)
    negs = ["6/27", "12/27"]
    for b, n in zip(bars, negs):
        ax.text(b.get_x() + b.get_width() / 2, 0.006, f"负例 {n}",
                ha="center", fontsize=8.5, color="white")
    ax.set_ylabel("物理 R²_cb(test27)")
    ax.set_title("分层 test27 锚点(FPS2000)", fontsize=11, loc="left")
    ax.set_ylim(0, 0.30)
    style_ax(ax)
    fig.tight_layout()
    save(fig, "f2_v4_anchors.png")


# ---------------------------------------------------------------- F3 Phase-V 主矩阵
def fig_phasev():
    keys = ["Point容量6-256-512+AG+AAA数据(分层划分)", "P1V:PointNet顶点采样(SAME)",
            "P2V:PointNet顶点采样(SEP)", "Point++SA3+AG+AAA数据(分层划分)",
            "Q0:Point++SA3+FPS采样(SAME)", "Q1V:Point++SA3顶点采样(SAME)",
            "Q2V:Point++SA3顶点采样(SEP)", "Q3V:Point++SA3顶点采样(SAME+随机中心)"]
    # xlsx 里 ID 用全角冒号
    keys = [k.replace(":", "：") for k in keys]
    labels = ["B0\nFPS2000", "P1V\nrandom5000\nSAME", "P2V ★\nrandom5000\nSEP",
              "B1\nFPS2000\nlegacy", "Q0\nFPS2000\n新推理", "Q1V ★\nrandom5000\nSAME",
              "Q2V\nrandom5000\nSEP", "Q3V\n随机中心"]
    vals = [v(k, "r2cb") for k in keys]
    high = [v(k, "high") for k in keys]
    negs = [ROWS[k][COL["neg"]] for k in keys]
    colors = [C_PN, C_PN, C_HL, C_PNPP, C_PNPP, C_HL, C_PNPP, C_PNPP]

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11.6, 5.6), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 2], "hspace": 0.12})
    x = np.arange(len(keys))
    bars = ax.bar(x, vals, color=colors, width=0.62)
    bar_labels(ax, bars)
    for i, n in enumerate(negs):
        ax.text(i, 0.008, f"负例 {n}", ha="center", fontsize=8, color="white")
    ax.axvline(2.5, color="#c9c9c5", linewidth=1)
    ax.text(1.0, 0.30, "PointNet(E2 宽通道)", fontsize=10, color=C_PN, ha="center")
    ax.text(5.5, 0.30, "PointNet++ SA3", fontsize=10, color="#128a5f", ha="center")
    ax.set_ylabel("物理 R²_cb")
    ax.set_ylim(0, 0.33)
    ax.set_title("Phase-V test27(106/0/27,legacy vertex):随机5000点采样是主要增益;★=本轮优先候选",
                 fontsize=11.5, loc="left")
    style_ax(ax)

    b2 = ax2.bar(x, high, color=[c if c != C_HL else C_HL for c in colors], width=0.62)
    bar_labels(ax2, b2, fmt="{:.2f}", dy=0.02)
    ax2.set_ylabel("high-WSS R²")
    ax2.set_ylim(-1.05, 0)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8.5)
    ax2.text(0.995, 0.06, "全部为负:高 WSS 尾部仍未解决", transform=ax2.transAxes,
             ha="right", fontsize=9.5, color=C_BAD)
    style_ax(ax2)
    save(fig, "f3_phasev_test27.png")


# ---------------------------------------------------------------- F4 半径探索
def fig_radius():
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    scale = [0.6, 0.8, 1.0, 1.2, 1.5]
    q1v = [v("Q1V半径探索(0.6×补充)", "r2cb"), v("Q1V半径探索(0.8×)", "r2cb"),
           v("Q1V：Point++SA3顶点采样(SAME)", "r2cb"),
           v("Q1V半径探索(1.2×)", "r2cb"), v("Q1V半径探索(1.5×)", "r2cb")]
    q2v = [v("Q2V半径探索(0.6×)", "r2cb"), v("Q2V半径探索(0.8×)", "r2cb"),
           v("Q2V：Point++SA3顶点采样(SEP)", "r2cb"), None, None]
    ax.plot(scale, q1v, "-o", color=C_PNPP, linewidth=2, markersize=7, label="Q1V / SAME")
    ax.plot(scale[:3], q2v[:3], "-o", color=C_VIOLET, linewidth=2, markersize=7, label="Q2V / SEP")
    for s_, y_ in zip(scale, q1v):
        ax.annotate(f"{y_:.3f}", (s_, y_), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=9)
    for s_, y_ in zip(scale[:3], q2v[:3]):
        ax.annotate(f"{y_:.3f}", (s_, y_), textcoords="offset points", xytext=(0, -16),
                    ha="center", fontsize=9, color=C_VIOLET)
    ax.plot([1.2], [q1v[3]], "o", markersize=13, markerfacecolor="none",
            markeredgecolor=C_HL, markeredgewidth=2.2)
    ax.text(1.2, q1v[3] - 0.014, "唯一正向臂\n仅 +0.0085,登记复核候选", ha="center",
            fontsize=8.5, color="#a06e00", va="top")
    ax.set_xticks(scale)
    ax.set_xticklabels(["0.6×", "0.8×", "1.0×\n(基准 0.05/0.10/0.20)", "1.2×", "1.5×"])
    ax.set_xlabel("三层 SA ball 半径缩放")
    ax.set_ylabel("物理 R²_cb(test27)")
    ax.set_title("半径响应非单调:缩小半径两个方向证据均为负;仅 1.2× 微弱正向", fontsize=11.5, loc="left")
    ax.legend(frameon=False, fontsize=9.5, loc="lower right")
    ax.set_ylim(0.21, 0.30)
    style_ax(ax)
    fig.tight_layout()
    save(fig, "f4_radius_scan.png")


# ---------------------------------------------------------------- F5 nsample×width
def heat(ax, data, rows, cols, title, vmin, vmax, star=None):
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blues", SEQ_BLUE)
    ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, fontsize=9.5)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows, fontsize=9.5)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            dark = (data[i, j] - vmin) / (vmax - vmin) > 0.55
            txt = f"{data[i, j]:.4f}"
            if star == (i, j):
                txt += " ★"
            ax.text(j, i, txt, ha="center", va="center", fontsize=10,
                    color="white" if dark else "#1a1a19")
    ax.set_title(title, fontsize=10.5, loc="left")
    ax.tick_params(colors="#555555")
    for s in ax.spines.values():
        s.set_visible(False)


def fig_nsample_width():
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 3.6))
    dev = np.array([
        [v("Q2V架构搜索(邻域16·宽32)", "r2cb"), v("Q2V架构搜索(邻域16·宽64)", "r2cb")],
        [v("Q2V架构搜索(邻域32·宽32)", "r2cb"), v("Q2V架构搜索(邻域32·宽64)", "r2cb")],
        [v("Q2V架构搜索(邻域64·宽32)", "r2cb"), v("Q2V架构搜索(邻域64·宽64)", "r2cb")]])
    heat(axes[0], dev, ["nsample=16", "nsample=32", "nsample=64"],
         ["width=32(0.22M 参数)", "width=64(0.88M 参数)"],
         "①grouped dev 85/21 · val21(Q2V/SEP 协议)", 0.22, 0.30, star=(1, 1))
    t27 = np.array([
        [v("Q1V random5000 + ball16(复用)".replace("(", "（").replace(")", "）"), "r2cb"),
         v("Q1V n16 / w64", "r2cb")],
        [v("Q1V n32 / w32", "r2cb"), v("Q1V n32 / w64", "r2cb")]])
    heat(axes[1], t27, ["nsample=16", "nsample=32"], ["width=32", "width=64"],
         "②historical test27 · 单 seed(Q1V/SAME 协议)", 0.22, 0.30, star=(0, 0))
    fig.suptitle("nsample × width:val21 上 n32_w64 物理第一(仅 +0.0014);test27 上基线 n16_w32 反而最好 → 无稳定容量增益",
                 fontsize=11.5, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    save(fig, "f5_nsample_width.png")


# ---------------------------------------------------------------- F6 SA1 grouping
def fig_grouping():
    cfgs = [("ball16(基准)", "Q1V random5000 + ball16（复用）", "0.20/0.30", "1.00"),
            ("ball32", "Q1V ball32", "0.38/0.55", "1.00"),
            ("raw KNN-8", "Q1V raw KNN-8", "0.51/0.68", "0.875"),
            ("raw KNN-10", "Q1V raw KNN-10", "0.58/0.78", "0.900"),
            ("KNN-8+cover", "Q1V KNN-8-cover", "1.00/1.00", "0.875"),
            ("KNN-10+cover", "Q1V KNN-10-cover", "1.00/1.00", "0.900"),
            ("adaptive_cover", "Q1V adaptive_cover", "1.00/1.00", "≤0.333")]
    vals = [v(k, "r2cb") for _, k, _, _ in cfgs]
    fig, ax = plt.subplots(figsize=(11.6, 4.4))
    x = np.arange(len(cfgs))
    colors = [C_PNPP] + ["#86b6ef"] * 6
    bars = ax.bar(x, vals, color=colors, width=0.6)
    bar_labels(ax, bars, fmt="{:.4f}")
    ax.axhline(vals[0], color=C_GRAY, linestyle="--", linewidth=1)
    for i, (_, _, cov, ov) in enumerate(cfgs):
        ax.text(i, 0.020, f"覆盖率\n{cov}", ha="center", fontsize=8, color="white")
        ax.text(i, -0.028, f"重叠 {ov}", ha="center", fontsize=8, color="#555555")
    ax.set_xticks(x)
    ax.set_xticklabels([c[0] for c in cfgs], fontsize=9.5)
    ax.set_ylabel("物理 R²_cb(test27)")
    ax.set_ylim(-0.04, 0.31)
    ax.set_title("SA1 分组方式(Q1V/SAME,单 seed):没有任何替代分组超过 ball16 基准;"
                 "覆盖率提高(cover/adaptive)并不带来精度收益", fontsize=11.5, loc="left")
    style_ax(ax)
    ax.text(0.995, 0.97, "覆盖率 = SA1 support 被分组覆盖比例(min/median);重叠 = 最大组间 |交|/min(|G|)",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color="#666666")
    fig.tight_layout()
    save(fig, "f6_sa1_grouping.png")


# ---------------------------------------------------------------- F7 support 矩阵
def fig_support():
    q1v = {("random", "ball"): "Q1V random5000 + ball16（复用）",
           ("random", "adaptive"): "Q1V adaptive_cover",
           ("fixed-FPS", "ball"): "Q1V fixed-FPS + ball16",
           ("fixed-FPS", "adaptive"): "Q1V fixed-FPS + adaptive",
           ("FPS-multistart", "ball"): "Q1V FPS-multistart5000 + ball16（历史复用）",
           ("FPS-multistart", "adaptive"): "Q1V FPS-multistart + adaptive"}
    q2v = {("random", "ball"): "Q2V random5000 + ball16（复用）",
           ("random", "adaptive"): "Q2V random + adaptive",
           ("fixed-FPS", "ball"): "Q2V fixed-FPS + ball16",
           ("fixed-FPS", "adaptive"): "Q2V fixed-FPS + adaptive",
           ("FPS-multistart", "ball"): "Q2V FPS-multistart + ball16",
           ("FPS-multistart", "adaptive"): "Q2V FPS-multistart + adaptive"}
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.0), sharey=True)
    for ax, mp, name in [(axes[0], q1v, "Q1V / SAME"), (axes[1], q2v, "Q2V / SEP")]:
        sups = ["random", "fixed-FPS", "FPS-multistart"]
        x = np.arange(3)
        w = 0.32
        b1 = ax.bar(x - w / 2, [v(mp[(s, "ball")], "r2cb") for s in sups], w,
                    color=C_PNPP, label="ball16")
        b2 = ax.bar(x + w / 2, [v(mp[(s, "adaptive")], "r2cb") for s in sups], w,
                    color="#9ec5f4", label="adaptive_cover")
        bar_labels(ax, b1, fmt="{:.3f}"); bar_labels(ax, b2, fmt="{:.3f}")
        ax.set_xticks(x)
        ax.set_xticklabels(["random5000\n(每epoch重采)", "fixed-FPS5000\n(固定)", "FPS-multistart\n5000(pool8)"], fontsize=9)
        ax.set_title(name, fontsize=10.5, loc="left")
        ax.set_ylim(0, 0.31)
        style_ax(ax)
    axes[0].set_ylabel("物理 R²_cb(test27)")
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("support 采样 × 分组(单 seed):random5000+ball16 在两种合同下都最好;"
                 "固定 FPS/多起点 FPS 均回退", fontsize=11.5, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save(fig, "f7_support_matrix.png")


# ---------------------------------------------------------------- F8 扩容+NormLoss
def fig_expand_norm():
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.2), gridspec_kw={"width_ratios": [5, 4]})
    ax = axes[0]
    groups = [
        ("D1\n(test27)", [("Q2V 基线", v("Q2V：Point++SA3顶点采样(SEP)", "r2cb"), C_PNPP),
                          ("+ILO41 冻结", v("Q2V扩数据D1(冻结)", "r2cb"), "#86b6ef"),
                          ("+ILO41 重算", v("Q2V扩数据D1(重训)", "r2cb"), "#86b6ef")]),
        ("D2\n(test36)", [("zero-shot", v("Q2V零样本迁移评估(test36)", "r2cb"), C_PNPP),
                          ("+ILO32 冻结", v("Q2V扩数据D2(冻结)", "r2cb"), "#86b6ef")]),
        ("D3\n(test36)", [("control", v("Q2V数据池2025(对照)", "r2cb"), C_PNPP),
                          ("+ILO32 冻结", v("Q2V数据池2025(冻结)", "r2cb"), "#86b6ef"),
                          ("+ILO32 重算", v("Q2V数据池2025(重训)", "r2cb"), "#86b6ef")]),
    ]
    xpos, ticks, tlabels = 0, [], []
    for gname, items in groups:
        for lab, val, c in items:
            b = ax.bar(xpos, val, color=c, width=0.7)
            bar_labels(ax, b, fmt="{:.3f}")
            ticks.append(xpos); tlabels.append(lab)
            xpos += 1
        ax.text(xpos - len(items) / 2 - 0.5, 0.325, gname, ha="center", fontsize=9.5, color="#444444")
        xpos += 0.6
    ax.set_xticks(ticks); ax.set_xticklabels(tlabels, fontsize=8, rotation=20)
    ax.set_ylabel("物理 R²_cb")
    ax.set_ylim(0, 0.36)
    ax.set_title("数据扩容(加 ILO 训练):三个主配对物理 R²_cb 全为负差 → 不晋级", fontsize=11, loc="left")
    style_ax(ax)

    ax = axes[1]
    keys = [("A0 点池化\n(对照)", "NormLoss A0｜Q2V pool2025（原 point-pooled 对照，复用）"),
            ("A1 逐病例\n等权", "NormLoss A1｜逐病例等权 log-z + MSE"),
            ("A1b 逐父系\n等权 ★", "NormLoss A1b｜逐父系等权 log-z + MSE"),
            ("A2 A1+\ncohort特征", "NormLoss A2｜A1 + cohort one-hot"),
            ("A3 A1+\nrawHuber", "NormLoss A3｜A1 + raw-Huber λ=0.2"),
            ("A4 A2+\nrawHuber", "NormLoss A4｜A1 + cohort one-hot + raw-Huber λ=0.2")]
    vals = [v(k, "r2cb") for _, k in keys]
    colors = ["#86b6ef", "#86b6ef", C_HL, "#86b6ef", "#86b6ef", "#86b6ef"]
    bars = ax.bar([k[0] for k in keys], vals, color=colors, width=0.62)
    bar_labels(ax, bars, fmt="{:.3f}")
    ax.axhline(vals[0], color=C_GRAY, linestyle="--", linewidth=1)
    ax.set_ylim(0, 0.31)
    ax.tick_params(axis="x", labelsize=8)
    ax.set_title("归一化口径×尾部损失(pool2025 test36,单seed):\n仅 A1b(逐父系等权 log-z)全面正向,保留为确认候选", fontsize=10.5, loc="left")
    style_ax(ax)
    fig.tight_layout()
    save(fig, "f8_expansion_normloss.png")


# ---------------------------------------------------------------- F9 QAD 配对
def fig_qad():
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 3.9))
    ax = axes[0]
    seeds = ["seed1234", "seed7", "seed2025"]
    r0 = [0.52937, 0.53601, 0.54864]
    r1 = [0.56612, 0.55877, 0.53189]
    for i, s in enumerate(seeds):
        ax.plot([0, 1], [r0[i], r1[i]], "-o", color=C_PNPP if r1[i] > r0[i] else C_BAD,
                linewidth=1.8, markersize=6)
        ax.annotate(s, (1, r1[i]), textcoords="offset points", xytext=(8, 0), fontsize=9)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["R0 插值 decoder", "R1 QAD-Lite"])
    ax.set_xlim(-0.25, 1.55)
    ax.set_ylabel("归一化 R²_cb(val21)")
    ax.set_title("QAD-Lite val21 三种子:均值 +0.0143 < 预注册门 +0.015", fontsize=10.5, loc="left")
    style_ax(ax)

    ax = axes[1]
    r0 = [0.62097, 0.59272, 0.60092]
    r1 = [0.59873, 0.60015, 0.60147]
    for i, s in enumerate(seeds):
        ax.plot([0, 1], [r0[i], r1[i]], "-o", color=C_PNPP if r1[i] > r0[i] else C_BAD,
                linewidth=1.8, markersize=6)
        ax.annotate(s, (1, r1[i]), textcoords="offset points", xytext=(8, 0), fontsize=9)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["R0 插值 decoder", "R1 QAD-Lite"])
    ax.set_xlim(-0.25, 1.55)
    ax.set_ylabel("归一化 R²_cb(test27)")
    ax.set_title("精确 Q2V 协议三种子复核:均值 −0.0048 → QAD 判 No-Go", fontsize=10.5, loc="left")
    style_ax(ax)
    fig.tight_layout()
    save(fig, "f9_qad_paired.png")


# ---------------------------------------------------------------- F10 指标冗余证据
def fig_metric_redundancy():
    rows = [r for r in ROWS.values() if r[COL["r2cb"]] not in (None, "")]

    def col(c):
        return np.array([float(r[COL[c]]) for r in rows
                         if r[COL[c]] not in (None, "") and r[COL["r2raw"]] not in (None, "")])

    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.7))
    ax = axes[0]
    a = np.array([float(r[COL["r2cb"]]) for r in rows if r[COL["r2raw"]] not in (None, "")])
    b = np.array([float(r[COL["r2raw"]]) for r in rows if r[COL["r2raw"]] not in (None, "")])
    ax.scatter(a, b, s=22, color=C_PN, alpha=0.75, edgecolors="white", linewidths=0.5)
    r = np.corrcoef(a, b)[0, 1]
    lo, hi = min(a.min(), b.min()) - 0.01, max(a.max(), b.max()) + 0.01
    ax.plot([lo, hi], [lo, hi], "--", color=C_GRAY, linewidth=1)
    ax.set_xlabel("物理 R²_cb(主指标)"); ax.set_ylabel("物理 R²_raw")
    ax.set_title(f"R²_raw ≈ R²_cb(Pearson r={r:.3f})\n→ 汇报表删 raw,归档保留", fontsize=10, loc="left")
    style_ax(ax)

    ax = axes[1]
    pairs = [(float(r[COL["nrmse_c"]]), float(r[COL["nmae_c"]])) for r in rows
             if r[COL["nrmse_c"]] not in (None, "") and r[COL["nmae_c"]] not in (None, "")]
    a = np.array([p[0] for p in pairs]); b = np.array([p[1] for p in pairs])
    r = np.corrcoef(a, b)[0, 1]
    ax.scatter(a, b, s=22, color=C_VIOLET, alpha=0.75, edgecolors="white", linewidths=0.5)
    ax.set_xlabel("range-NRMSE case-mean"); ax.set_ylabel("NMAE case-mean")
    ax.set_title(f"NRMSE 与 NMAE 高度共线(r={r:.3f})\n→ 相对误差保留 NMAE 一族即可", fontsize=10, loc="left")
    style_ax(ax)

    ax = axes[2]
    hs = np.array([float(r[COL["highspear"]]) for r in rows if r[COL["highspear"]] not in (None, "")])
    ax.hist(hs, bins=21, color="#9ec5f4", edgecolor="white")
    ax.axvline(0, color=C_GRAY, linewidth=1)
    ax.set_xlabel("high-WSS Spearman(全矩阵各行取值)")
    ax.set_ylabel("行数")
    ax.set_title(f"high-WSS Spearman 全部≈0\n(范围 {hs.min():.3f}~{hs.max():.3f},从未参与判定)→ 删", fontsize=10, loc="left")
    style_ax(ax)
    fig.tight_layout()
    save(fig, "f10_metric_redundancy.png")


if __name__ == "__main__":
    fig_test16()
    fig_v4_anchors()
    fig_phasev()
    fig_radius()
    fig_nsample_width()
    fig_grouping()
    fig_support()
    fig_expand_norm()
    fig_qad()
    fig_metric_redundancy()
    print("done ->", OUT)
