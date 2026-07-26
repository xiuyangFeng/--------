#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成《WSS_PointNet实验矩阵指标汇总》PPT 的结构/机制示意图。

事实来源:training_wss_min/baseline_models.py、config.py、dataset.py 与
pipeline_wss_min(细节见 docs/02-推进与变更 相关文档)。
输出:docs/03-汇报材料/figures/WSS_PointNet矩阵汇总_20260721/*.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "WSS_PointNet矩阵汇总_20260721"
OUT.mkdir(parents=True, exist_ok=True)

FONT = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
if FONT.exists():
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(FONT)).get_name()
plt.rcParams["axes.unicode_minus"] = False

C_BLUE = "#2a78d6"
C_BLUE_L = "#cde2fb"
C_AQUA = "#1baf7a"
C_AQUA_L = "#d9f2e8"
C_HL = "#eda100"
C_HL_L = "#fdeecd"
C_RED = "#e34948"
C_VIOLET = "#4a3aa7"
C_GRAY = "#8a8a85"
INK = "#1a1a19"


def box(ax, x, y, w, h, text, fc, ec=None, fs=9.5, tc=INK, lw=1.2, sub=None, subfs=7.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                facecolor=fc, edgecolor=ec or fc, linewidth=lw))
    cy = y + h * (0.62 if sub else 0.5)
    ax.text(x + w / 2, cy, text, ha="center", va="center", fontsize=fs, color=tc)
    if sub:
        ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center",
                fontsize=subfs, color="#555555")


def arrow(ax, x1, y1, x2, y2, color="#777777", lw=1.6, style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=13, color=color, linewidth=lw))


def blank(ax, xlim=(0, 10), ylim=(0, 10)):
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.axis("off")


def save(fig, name):
    fig.savefig(OUT / name, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


# ---------------------------------------------------- A1 PointNet E2 结构
def fig_arch_pointnet():
    fig, ax = plt.subplots(figsize=(11.6, 4.6))
    blank(ax, (0, 12), (0, 6))
    ax.text(0.1, 5.65, "PointNet(E2 宽通道)结构 —— 792,833 参数,dropout=0,逐点回归 WSS",
            fontsize=13, color=INK, weight="bold")

    box(ax, 0.2, 2.3, 1.7, 1.5, "输入点云\nN × 6", C_BLUE_L, fs=10,
        sub="x, y, z + abscissa_norm\nlocal_radius, curvature")
    arrow(ax, 1.95, 3.05, 2.55, 3.05)

    box(ax, 2.6, 2.3, 2.5, 1.5, "逐点局部编码 MLP\n6 → 256 → 512", C_BLUE, tc="white", fs=10.5,
        sub="每层 Linear+BN+ReLU")
    arrow(ax, 5.15, 3.05, 5.75, 3.05)

    box(ax, 5.8, 3.4, 2.1, 1.0, "逐病例\nglobal max-pool → 512", C_BLUE, tc="white", fs=9.5)
    box(ax, 5.8, 1.7, 2.1, 1.0, "逐点局部特征 512", C_BLUE_L, fs=9.5)
    arrow(ax, 7.95, 3.9, 8.55, 3.3)
    arrow(ax, 7.95, 2.2, 8.55, 2.8)

    box(ax, 8.6, 2.3, 1.5, 1.5, "拼接\n1024", "#9ec5f4", fs=10)
    arrow(ax, 10.15, 3.05, 10.6, 3.05)
    box(ax, 10.65, 2.3, 1.25, 1.5, "解码\n1024→512\n→256→1", C_BLUE, tc="white", fs=9)

    # width 旋钮标注
    box(ax, 2.3, 0.15, 4.6, 1.15, "「width / 容量」旋钮:局部编码通道宽度", C_HL_L, ec=C_HL, fs=9.5,
        sub="E0 原始 32→64→128(0.09M)|E2 宽通道 256→512(0.79M,本轮锚点)\nE4 加深 6→64→128→256→512(0.87M,No-Go)")
    arrow(ax, 3.9, 1.3, 3.85, 2.2, color=C_HL)
    ax.text(9.35, 4.35, "对每个采样点输出 1 个 WSS 预测", fontsize=8.5, color="#555555", ha="center")
    save(fig, "a1_arch_pointnet.png")


# ---------------------------------------------------- A2 PointNet++ SA3 结构
def fig_arch_pointnetpp():
    fig, ax = plt.subplots(figsize=(12.6, 6.6))
    blank(ax, (0, 13), (0, 8))
    ax.text(0.1, 7.65, "PointNet++ SA3 结构与全部调参旋钮 —— 224,001 参数(width32/head64)",
            fontsize=13, color=INK, weight="bold")

    # 主链
    box(ax, 0.15, 4.6, 1.55, 1.3, "support\n5000 点", C_AQUA_L, fs=10,
        sub="(B1/Q0 为 2000)")
    arrow(ax, 1.72, 5.25, 2.18, 5.25)
    box(ax, 2.2, 4.6, 1.15, 1.3, "stem\n6→32→32", C_AQUA, tc="white", fs=9)
    arrow(ax, 3.37, 5.25, 3.83, 5.25)
    box(ax, 3.85, 4.6, 1.85, 1.3, "SA1:500 中心\nr=0.05|MLP 64", C_AQUA, tc="white", fs=9,
        sub="ball ≤16 邻居/中心")
    arrow(ax, 5.72, 5.25, 6.18, 5.25)
    box(ax, 6.2, 4.6, 1.85, 1.3, "SA2:125 中心\nr=0.10|MLP 128", C_AQUA, tc="white", fs=9,
        sub="ball ≤16")
    arrow(ax, 8.07, 5.25, 8.53, 5.25)
    box(ax, 8.55, 4.6, 1.85, 1.3, "SA3:32 中心\nr=0.20|MLP 256", C_AQUA, tc="white", fs=9,
        sub="ball ≤16")
    arrow(ax, 10.42, 5.25, 10.88, 5.25)
    box(ax, 10.9, 4.6, 1.95, 1.3, "FP×3 上采样\nk=3 插值", "#66c9a3", tc="white", fs=9,
        sub="384→128|192→64|96→32")
    ax.text(6.4, 6.15, "完整 SA 链:5000 support → 500 → 125 → 32(后三层是 SA 中心数,不是输入点数)",
            fontsize=9, color="#128a5f", ha="center")

    arrow(ax, 11.85, 4.55, 11.85, 3.95)
    box(ax, 10.9, 2.6, 1.95, 1.3, "head\n32→64→1", C_AQUA, tc="white", fs=9.5,
        sub="逐 query 点输出 WSS")

    # 旋钮标注(五个)
    box(ax, 0.15, 2.7, 2.6, 1.35, "① support 采样旋钮", C_HL_L, ec=C_HL, fs=9.5,
        sub="vertex-random5000(每epoch重采)\nFPS2000 / fixed-FPS5000\nFPS-multistart5000(pool8)")
    arrow(ax, 1.0, 4.05, 0.95, 4.5, color=C_HL)

    box(ax, 0.15, 0.75, 2.6, 1.35, "② query 合同旋钮", C_HL_L, ec=C_HL, fs=9.5,
        sub="SAME:query=support\nSEP:独立重采 5000\n评估恒为 full-cloud query")
    arrow(ax, 1.45, 2.1, 1.45, 2.65, color=C_HL)

    box(ax, 3.2, 2.7, 2.6, 1.35, "③ SA1 分组旋钮(grouping)", C_HL_L, ec=C_HL, fs=9.5,
        sub="ball16/ball32|raw KNN-8/10\nKNN+cover 修复|adaptive_cover\n(仅改 SA1,SA2/3 保持 ball)")
    arrow(ax, 4.5, 4.05, 4.6, 4.5, color=C_HL)

    box(ax, 3.2, 0.75, 2.6, 1.35, "④ 半径旋钮(radius)", C_HL_L, ec=C_HL, fs=9.5,
        sub="三层同步缩放 0.6×~1.5×\n基准 0.05/0.10/0.20(归一化坐标)")
    arrow(ax, 5.3, 2.1, 5.6, 2.65, color=C_HL)

    box(ax, 6.25, 2.7, 2.6, 1.35, "⑤ nsample / width 旋钮", C_HL_L, ec=C_HL, fs=9.5,
        sub="nsample:每中心邻居上限 16/32/64\nwidth:基通道 32→64\n(通道=width×2^层,0.22M→0.88M)")
    arrow(ax, 7.5, 4.05, 7.4, 4.5, color=C_HL)

    box(ax, 6.25, 0.75, 2.6, 1.35, "⑥ SA 中心采样旋钮", C_HL_L, ec=C_HL, fs=9.5,
        sub="FPS(确定性,基准)\nRandom(Q3V,整体回退)")
    arrow(ax, 8.4, 2.1, 8.9, 4.5, color=C_HL)

    box(ax, 9.35, 0.75, 3.35, 1.35, "⑦ decoder 旋钮", C_HL_L, ec=C_HL, fs=9.5,
        sub="3-NN interpolate(基准)\nQAD-Lite 门控残差(+2,625 参数,No-Go)")
    arrow(ax, 11.3, 2.1, 11.6, 2.55, color=C_HL)
    save(fig, "a2_arch_pointnetpp.png")


# ---------------------------------------------------- A3 SA1 grouping 机制
def fig_grouping_schematic():
    rng = np.random.default_rng(7)
    pts = rng.uniform(0, 10, size=(90, 2))
    centers = np.array([[2.6, 6.4], [5.4, 3.2], [7.6, 6.8]])
    r = 2.0

    fig, axes = plt.subplots(1, 4, figsize=(12.6, 3.8))
    titles = [
        ("ball query(基准 ball16)", "半径 r 内取 ≤nsample 个\n覆盖率低(min≈0.20)但最稳", 0),
        ("raw KNN-8/10", "无半径约束取 K 近邻\n覆盖率 0.51~0.78,负对照", 1),
        ("KNN + coverage 修复", "未覆盖点补连最近中心\n覆盖 100%,重叠仍高(0.875+)", 2),
        ("adaptive_cover", "先最近中心全分配(覆盖必 100%)\n组间重叠硬上限 ≤1/3,组大小 1~84", 3)]

    for ax, (title, sub, mode) in zip(axes, titles):
        blank(ax, (-0.5, 10.5), (-0.7, 10.5))
        d = np.linalg.norm(pts[:, None, :] - centers[None, :, :], axis=2)
        nearest = d.argmin(axis=1)
        cols = [C_BLUE, C_AQUA, C_VIOLET]

        if mode == 0:
            covered = (d.min(axis=1) <= r)
            for ci, c in enumerate(centers):
                ax.add_patch(Circle(c, r, fill=False, color=cols[ci], linewidth=1.6, linestyle="--"))
            for i, p in enumerate(pts):
                if covered[i]:
                    ax.plot(*p, "o", color=cols[nearest[i]], markersize=4)
                else:
                    ax.plot(*p, "o", color="#cccccc", markersize=3.5)
            ax.text(9.6, -0.35, "灰点=未覆盖", fontsize=8, ha="right", color="#888888")
        elif mode in (1, 2):
            k = 8
            assigned = np.zeros(len(pts), dtype=bool)
            for ci, c in enumerate(centers):
                idx = np.argsort(d[:, ci])[:k]
                assigned[idx] = True
                for i in idx:
                    ax.plot([c[0], pts[i, 0]], [c[1], pts[i, 1]],
                            color=cols[ci], linewidth=0.7, alpha=0.55)
            for i, p in enumerate(pts):
                if assigned[i]:
                    ax.plot(*p, "o", color=cols[nearest[i]], markersize=4)
                elif mode == 2:
                    ax.plot([centers[nearest[i], 0], p[0]], [centers[nearest[i], 1], p[1]],
                            color=C_HL, linewidth=0.9, alpha=0.9)
                    ax.plot(*p, "o", color=C_HL, markersize=4)
                else:
                    ax.plot(*p, "o", color="#cccccc", markersize=3.5)
            if mode == 2:
                ax.text(9.6, -0.35, "黄边=覆盖修复", fontsize=8, ha="right", color="#a06e00")
            else:
                ax.text(9.6, -0.35, "灰点=永不被分组", fontsize=8, ha="right", color="#888888")
        else:
            for i, p in enumerate(pts):
                ax.plot([centers[nearest[i], 0], p[0]], [centers[nearest[i], 1], p[1]],
                        color=cols[nearest[i]], linewidth=0.55, alpha=0.4)
                ax.plot(*p, "o", color=cols[nearest[i]], markersize=4)
            mid = (centers[0] + centers[2]) / 2 + np.array([0.2, -0.6])
            ax.text(mid[0], mid[1], "共享点预算\n≤⌊min|G|/3⌋", fontsize=8, ha="center", color="#555555")

        for ci, c in enumerate(centers):
            ax.plot(*c, "*", color=cols[ci], markersize=15, markeredgecolor="white",
                    markeredgewidth=0.8)
        ax.set_title(title, fontsize=10.5, loc="left")
        ax.text(0, 9.9, sub, fontsize=8.2, color="#555555", va="top")
    fig.suptitle("SA1 分组机制对比(★=SA1 中心;颜色=归属分组)—— test27 结论:替代分组均未超过 ball16",
                 fontsize=12, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    save(fig, "a3_sa1_grouping_schematic.png")


# ---------------------------------------------------- A4 support/query 采样示意
def fig_sampling_schematic():
    fig, ax = plt.subplots(figsize=(12.6, 5.2))
    blank(ax, (0, 13), (0, 7))
    ax.text(0.1, 6.65, "训练采样合同:support(网络输入)与 query(监督位置)是两个独立旋钮",
            fontsize=13, color=INK, weight="bold")

    box(ax, 0.15, 4.4, 2.3, 1.5, "全壁面点云\n(AG ~1.3万 / ILO ~8万)", "#eeeeec", ec="#c9c9c5", fs=9.5,
        sub="每例 peak-systole 壁面 WSS")
    arrow(ax, 2.47, 5.15, 3.0, 5.15)

    box(ax, 3.05, 5.15, 3.1, 1.0, "vertex-random5000", C_AQUA, tc="white", fs=9.5,
        sub="等概率无放回,每 epoch 重采 → 最优")
    box(ax, 3.05, 3.95, 3.1, 1.0, "FPS(2000/固定5000)", "#9ec5f4", fs=9.5,
        sub="确定性最远点覆盖,固定不变 → 回退")
    box(ax, 3.05, 2.75, 3.1, 1.0, "FPS-multistart5000", "#9ec5f4", fs=9.5,
        sub="8 个起点池轮换 → 回退")
    ax.text(4.6, 6.3, "support 采样(编码器看到的点)", fontsize=10, color="#128a5f", ha="center")

    arrow(ax, 6.2, 4.6, 7.0, 4.6)
    box(ax, 7.05, 3.9, 2.6, 1.5, "编码-解码\nPointNet / PointNet++", "#eeeeec", ec="#c9c9c5", fs=10)
    arrow(ax, 9.7, 4.6, 10.5, 4.6)

    box(ax, 10.55, 4.7, 2.3, 1.15, "SAME 合同", C_BLUE, tc="white", fs=10,
        sub="query = support 同一批点")
    box(ax, 10.55, 3.3, 2.3, 1.15, "SEP 合同", C_VIOLET, tc="white", fs=10,
        sub="query 独立重采 5000 点\n(监督位置 ≠ 输入位置)")
    ax.text(11.7, 6.1, "query 合同(在哪些点上算 loss)", fontsize=10, color="#3a3090", ha="center")

    box(ax, 3.05, 0.5, 9.8, 1.6, "评估合同(全部实验统一):fixed-support / full-cloud query", "#fdeecd",
        ec=C_HL, fs=10.5,
        sub="推理时 support 固定,query = 全壁面点;正式指标在完整壁面同点 CSV 上计算(legacy vertex 口径)\n"
            "Phase-V 结论:B0→P1V/P2V、Q0→Q1V 的主要增益来自「随机 5000 点 + 每 epoch 重采」;SEP 只对 PointNet 有额外增益")
    arrow(ax, 8.35, 3.85, 8.35, 2.15, color=C_HL)
    save(fig, "a4_sampling_schematic.png")


# ---------------------------------------------------- A5 数据管线
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(12.6, 6.4))
    blank(ax, (0, 13), (0, 8))
    ax.text(0.1, 7.65, "数据管线:CFD 原始导出 → 训练 bundle → 训练/评估 → PostView",
            fontsize=13, color=INK, weight="bold")

    y1 = 5.7
    box(ax, 0.15, y1, 2.9, 1.5, "① CFD 原始数据(Fluent ASCII)", C_BLUE_L, fs=9.5,
        sub="壁面节点:坐标+压力+WSS(标量+矢量)\n内部 cell:速度/压力|VMTK 中心线\n原始 STL|入口波形 vf-in")
    arrow(ax, 3.07, y1 + 0.75, 3.5, y1 + 0.75)
    box(ax, 3.55, y1, 3.1, 1.5, "② 预处理 preprocess", C_BLUE, tc="white", fs=9.5,
        sub="peak-systole 选相(入口流量最大步)\n单位→mm|stl_landmarks_v4 解剖坐标帧\n"
            "刚性配准|逐病例保形归一化到 [-1,1]")
    arrow(ax, 6.67, y1 + 0.75, 7.1, y1 + 0.75)
    box(ax, 7.15, y1, 2.55, 1.5, "③ bundle.npz", C_AQUA, tc="white", fs=9.5,
        sub="完整壁面点云+WSS\n几何特征+坐标帧元数据")
    arrow(ax, 9.72, y1 + 0.75, 10.15, y1 + 0.75)
    box(ax, 10.2, y1, 2.65, 1.5, "④ split / train-only stats", C_AQUA_L, fs=9.2,
        sub="病例级 train/val/test\n目标与输入统计只由 train 计算")

    arrow(ax, 11.5, y1 - 0.05, 11.5, 4.75)
    box(ax, 8.8, 3.05, 4.05, 1.65, "⑤ 配置驱动训练", "#eeeeec", ec="#c9c9c5", fs=10,
        sub="support/query 采样|PointNet/PointNet++\ntrain-loss 选模|ckpt_best / ckpt_last")
    arrow(ax, 8.75, 3.85, 7.95, 3.85)
    box(ax, 4.8, 3.05, 3.1, 1.65, "⑥ 完整壁面评估", C_HL_L, ec=C_HL, fs=10,
        sub="fixed support / full-cloud query\nR²、MAE、RMSE、high-WSS 与逐病例 CSV")
    arrow(ax, 4.75, 3.85, 3.95, 3.85)
    box(ax, 0.15, 3.05, 3.75, 1.65, "⑦ PostView / 汇报产物", C_VIOLET, tc="white", fs=10,
        sub="同点真值/预测/误差|STL 映射\n图件、表格与可追溯 manifest")

    box(ax, 0.15, 0.55, 12.7, 1.55, "全链路硬门禁", "#fdeecd", ec=C_HL, fs=10.5,
        sub="canonical 病例 ID 与 split 防泄漏|frame_version 与文件 SHA|SA 几何合同|"
            "CUDA 前后向与 checkpoint 严格重载|full/chunk 一致性|NaN/Inf 拒绝")
    save(fig, "a5_data_pipeline.png")


if __name__ == "__main__":
    fig_arch_pointnet()
    fig_arch_pointnetpp()
    fig_grouping_schematic()
    fig_sampling_schematic()
    fig_pipeline()
    print("done ->", OUT)
