#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成《WSS_PointNet实验矩阵指标汇总》PPT 的结构/机制示意图。

事实来源:baseline_models.py / dataset.py / config.py / preprocess.py 等
(由两个 Explore 代理核对,行号见 PPT 备注)。
输出:docs/03-汇报材料/figures/WSS_PointNet矩阵汇总_20260721/*.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "WSS_PointNet矩阵汇总_20260721"
OUT.mkdir(parents=True, exist_ok=True)

FONT = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
if FONT.exists():
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(FONT)).get_name()
plt.rcParams["axes.unicode_minus"] = False

C_PN = "#2a78d6"
C_PNPP = "#1baf7a"
C_HL = "#eda100"
C_BAD = "#e34948"
C_VIOLET = "#4a3aa7"
C_INK = "#1a1a19"
C_MUT = "#6b6b66"


def box(ax, x, y, w, h, text, fc, ec=None, fs=10, tc="white", lw=0, rounded=0.02, weight="normal"):
    ec = ec or fc
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={rounded}",
                       facecolor=fc, edgecolor=ec, linewidth=lw, mutation_aspect=1)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=tc, weight=weight, linespacing=1.35)


def arrow(ax, x0, y0, x1, y1, color=C_MUT, lw=1.6, style="-|>", ms=12):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
                                 mutation_scale=ms, color=color, linewidth=lw,
                                 shrinkA=0, shrinkB=0))


def save(fig, name):
    fig.savefig(OUT / name, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


# ============================================================ S1 数据/CFD 管线
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    ax.text(1, 97.5, "从 CFD 原始结果到训练样本:四阶段预处理管线", fontsize=15, weight="bold", color=C_INK)
    ax.text(1, 93.5, "ANSYS Fluent(ASCII 导出) → 峰值收缩期壁面 WSS 场 → 几何配准与归一化 → FPS/random 稀疏化",
            fontsize=10.5, color=C_MUT)

    # 上排:原始输入
    inputs = [
        ("壁面 ASCII\nnode·x/y/z·pressure\nwall-shear(WSS,Pa)+3矢量分量", C_PN),
        ("内部点 ASCII\ncell·x/y/z·pressure\nvelocity(近壁掩码≤1.5mm)", "#6da7ec"),
        ("入口波形 vf-in\n定义 peak systole\n(缺失回退区间中点)", "#9ec5f4"),
        ("原始 STL 曲面\n+ VMTK 中心线\n(abscissa/半径/曲率)", C_PNPP),
    ]
    x = 2
    for t, c in inputs:
        box(ax, x, 80, 22.5, 10, t, c, fs=8.6)
        x += 24
    ax.text(1, 76.5, "① preprocess(逐病例,严格顺序)", fontsize=10.5, weight="bold", color=C_INK)

    steps = [
        "单位→mm\n(中心线包围盒反推)",
        "stl_landmarks_v4\n坐标帧:原点=分叉点\n+Z 主干 / −Z 双髂支",
        "刚性配准+尾段裁剪\ndet(R)=+1",
        "坐标各向同性\nmax_abs→[−1,1]",
        "几何特征:最近中心线点\nabscissa/local_radius/curvature",
    ]
    x = 2
    for i, t in enumerate(steps):
        box(ax, x, 64, 18.2, 10, t, "#eef3f8", ec="#c9d6e5", lw=1, tc=C_INK, fs=8.0)
        if i < len(steps) - 1:
            arrow(ax, x + 18.2, 69, x + 19.6, 69, color="#c9d6e5")
        x += 19.6

    # 下排:后三阶段
    y2 = 47
    for i, (t, sub) in enumerate([
        ("② qa-gate", "peak/all 零值率≤1%\nn_wall≤50000(AG护栏)\n裁剪/居中偏移≤5%"),
        ("③ global-stats", "train-only·peak\nlog-z:mean/std\n(病例共享,可恢复Pa)"),
        ("④ build-samples", "FPS-2000 或 random-5000\n每样本 coords(N,3)+geom(N,4)\ny(log-z)·y_raw(Pa)"),
    ]):
        bx = 2 + i * 33
        box(ax, bx, y2, 30, 13, "", "#f3f0fb", ec="#d7cff0", lw=1.2, tc=C_INK)
        ax.text(bx + 1.5, y2 + 10, t, fontsize=11, weight="bold", color=C_VIOLET)
        ax.text(bx + 1.5, y2 + 4.2, sub, fontsize=8.4, color=C_INK, va="center")
        if i < 2:
            arrow(ax, bx + 30, y2 + 6.5, bx + 33, y2 + 6.5, color="#d7cff0")

    # bundle 输出
    box(ax, 2, 28, 96, 14,
        "bundle.npz(每病例):壁面几何 wall_coords_norm/raw · 几何特征 abscissa/radius/curvature · "
        "全时间步 wall_wss(Pa)/pressure · 内部点 · 坐标帧与变换元信息\n"
        "→ 训练输入固定 6 维 xyzgeom = [x, y, z, abscissa_norm, local_radius, curvature]"
        "(curvature 做 signed_log1p;dist_to_wall 壁面恒0已移除)",
        "#fbf3e0", ec=C_HL, lw=1.4, tc=C_INK, fs=8.9)
    arrow(ax, 50, 47, 50, 42, color=C_HL, lw=2)

    ax.text(1, 22.5, "队列与划分", fontsize=11, weight="bold", color=C_INK)
    ax.text(1, 16,
            "• AG(主动脉-髂动脉)76 例(v4)→ 训练用 61;AAA(腹主动脉瘤)达标 65→入训 57;ILO(术前)审核通过 41(仅 */before)\n"
            "• 主力划分 stratified seed1234 = 106/0/27(train 61/21/24,test 15/6/6),按 AG / AAA-rupture / AAA-unrupture 分层\n"
            "• WSS 单位恒为 Pa;bundle 存原始值,图件才做 log1p 显示裁剪;v4 cutover 2026-07-16 原子迁移完成(AG 84→76)",
            fontsize=9, color=C_INK, linespacing=1.7, va="top")
    save(fig, "s1_data_pipeline.png")


# ============================================================ S2 PointNet E2 结构
def fig_pointnet_e2():
    fig, ax = plt.subplots(figsize=(12.2, 4.6))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    ax.text(1, 95, "PointNet E2(导师宽通道 baseline)", fontsize=15, weight="bold", color=C_PN)
    ax.text(1, 89.5, "逐点共享 MLP 编码 → 逐病例 global max-pool → 拼接后解码;参数量 792,833,dropout=0",
            fontsize=10.5, color=C_MUT)

    box(ax, 1, 60, 15, 16, "输入点云\nN×6\nxyzgeom", "#dbe8f8", ec=C_PN, lw=1.3, tc=C_INK, fs=9.5)
    # 局部编码
    box(ax, 20, 60, 12, 16, "共享 MLP\n6 → 256", C_PN, fs=9.5)
    box(ax, 34, 60, 12, 16, "共享 MLP\n256 → 512", C_PN, fs=9.5)
    arrow(ax, 16, 68, 20, 68, color=C_PN)
    arrow(ax, 32, 68, 34, 68, color=C_PN)
    ax.text(33, 78.5, "局部编码(逐点特征 512 维)", fontsize=9.5, color=C_PN, ha="center")

    # global pool
    box(ax, 50, 60, 13, 16, "global\nmax-pool\n(逐病例)", C_HL, fs=9.5)
    arrow(ax, 46, 68, 50, 68, color=C_PN)
    box(ax, 50, 40, 13, 14, "全局特征\n512 维", "#fbf3e0", ec=C_HL, lw=1.3, tc=C_INK, fs=9.5)
    arrow(ax, 56.5, 60, 56.5, 54, color=C_HL)

    # concat
    box(ax, 40, 22, 8, 26, "拼接\n1024", "#e7e3f5", ec=C_VIOLET, lw=1.3, tc=C_INK, fs=9)
    arrow(ax, 50, 47, 48, 40, color=C_VIOLET)  # global→concat
    ax.annotate("", xy=(44, 48), xytext=(38, 68),
                arrowprops=dict(arrowstyle="-|>", color=C_PN, lw=1.4,
                                connectionstyle="arc3,rad=-0.3"))
    ax.text(30, 52, "逐点 512\n(每个点)", fontsize=8.5, color=C_PN, ha="center")

    # decoder
    box(ax, 53, 26, 10, 12, "MLP\n1024→512", C_VIOLET, fs=9)
    box(ax, 66, 26, 10, 12, "MLP\n512→256", C_VIOLET, fs=9)
    box(ax, 79, 26, 10, 12, "Linear\n256→1", C_BAD, fs=9)
    arrow(ax, 48, 32, 53, 32, color=C_VIOLET)
    arrow(ax, 63, 32, 66, 32, color=C_VIOLET)
    arrow(ax, 76, 32, 79, 32, color=C_VIOLET)
    box(ax, 91, 26, 8, 12, "逐点\nWSS", "#dbe8f8", ec=C_PN, lw=1.3, tc=C_INK, fs=9)
    arrow(ax, 89, 32, 91, 32, color=C_BAD)

    ax.text(1, 12,
            "对照臂:E0 原始窄通道(32→64→128) · E3 random-5000 采样 · E23 宽+5000 · E4 加深(6→64→128→256→512)\n"
            "结论:宽通道 E2 相对 E0 一次性 +0.073(test16 物理 R²_cb);点数/深度均无稳定附加收益,E4 加深过拟合 No-Go",
            fontsize=9.3, color=C_INK, linespacing=1.6)
    save(fig, "s2_pointnet_e2.png")


# ============================================================ S3 PointNet++ SA3 结构
def fig_pointnetpp():
    fig, ax = plt.subplots(figsize=(12.4, 5.6))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    ax.text(1, 96, "PointNet++ SA3(三层 Set Abstraction + Feature Propagation)", fontsize=15, weight="bold", color=C_PNPP)
    ax.text(1, 90.5, "编码器逐层下采样并聚合局部邻域,解码器 3-NN 插值上采样回全部点;参数量 224,001(width=32)",
            fontsize=10.5, color=C_MUT)

    # encoder SA layers
    enc = [("support\n5000", "#d6efe4", C_INK),
           ("SA1\ncenter 500\nr=0.05·邻域16\nMLP 35→64→64", C_PNPP, "white"),
           ("SA2\ncenter 125\nr=0.10·邻域16\nMLP 67→128→128", "#128a5f", "white"),
           ("SA3\ncenter 32\nr=0.20·邻域16\nMLP 131→256→256", "#0d6b49", "white")]
    x = 2
    for i, (t, c, tc) in enumerate(enc):
        w = 15 if i == 0 else 20
        box(ax, x, 64, w, 18, t, c, ec=C_PNPP if i == 0 else c, lw=1.3 if i == 0 else 0, tc=tc, fs=8.8)
        if i < 3:
            arrow(ax, x + w, 73, x + w + 2.2, 73, color=C_PNPP)
        x += w + 2.2
    ax.text(2, 85.5, "编码器:每层 FPS 选中心 → ball-query 分组 → 逐点相对坐标+特征 → scatter-max", fontsize=9.2, color="#128a5f")

    # decoder FP
    dec = [("FP3\n384→128→128", "#7fcdb0"),
           ("FP2\n192→64→64", "#a5dcc8"),
           ("FP1\n96→32→32", "#c8ebe0")]
    x = 24
    ax.text(2, 55, "解码器:3-NN interpolate 逐层上采样(k=3),skip 连接编码器同层", fontsize=9.2, color="#128a5f")
    for i, (t, c) in enumerate(dec):
        box(ax, x, 40, 18, 11, t, c, tc=C_INK, fs=8.8)
        if i < 2:
            arrow(ax, x + 18, 45.5, x + 20, 45.5, color="#7fcdb0")
        x += 20
    arrow(ax, 62, 64, 55, 51, color="#7fcdb0", lw=1.4)  # SA3->FP3
    box(ax, 86, 40, 12, 11, "head\n32→64→1", C_BAD, fs=9)
    arrow(ax, 82, 45.5, 86, 45.5, color=C_BAD)
    box(ax, 86, 64, 12, 12, "逐点\nWSS", "#d6efe4", ec=C_PNPP, lw=1.3, tc=C_INK, fs=9)
    arrow(ax, 92, 51, 92, 64, color=C_BAD)

    # 旋钮说明
    box(ax, 2, 5, 96, 27, "", "#f7fbf9", ec="#bfe4d6", lw=1.3, tc=C_INK)
    ax.text(4, 28, "本轮扫过的四类结构旋钮(单变量对照)", fontsize=11.5, weight="bold", color="#0d6b49")
    knobs = [
        ("width(通道基数)", "32↔64:通道=[width·2^i];width64≈4× 参数(0.22M→0.88M)。val21 上 n32_w64 物理第一但仅 +0.0014"),
        ("nsample(邻域点数)", "16↔32↔64:ball-query 每中心最多取多少邻居。无跨 width 稳定增益"),
        ("radius(ball 半径)", "三层 0.05/0.10/0.20 为基准,缩放 0.6×~1.5×;响应非单调,仅 1.2× 微弱正向(+0.0085)"),
        ("SA1 分组方式", "ball / raw-KNN / KNN+coverage / adaptive_cover;覆盖率提高不带来精度收益,ball16 仍最优"),
        ("SA center 采样", "FPS(确定性最远点)↔ Random;Q3V 随机中心整体回退"),
    ]
    yk = 23.5
    for name, desc in knobs:
        ax.text(4.5, yk, "●", fontsize=9, color=C_HL)
        ax.text(6.5, yk, name, fontsize=9.3, weight="bold", color=C_INK)
        ax.text(27, yk, desc, fontsize=8.7, color="#333333")
        yk -= 4.1
    save(fig, "s3_pointnetpp_sa3.png")


# ============================================================ S4 SA1 分组机制对比
def fig_grouping_mechanism():
    fig, axes = plt.subplots(1, 4, figsize=(12.6, 3.9))
    rng = np.random.RandomState(7)
    pts = rng.rand(80, 2)
    centers = np.array([[0.3, 0.35], [0.68, 0.62], [0.5, 0.2]])

    def base(ax, title, sub, tcolor):
        ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.15, 1.12); ax.axis("off")
        ax.scatter(pts[:, 0], pts[:, 1], s=13, color="#c7c7c2", zorder=1)
        ax.scatter(centers[:, 0], centers[:, 1], s=110, marker="*",
                   color=C_HL, edgecolors="#7a5600", linewidths=0.8, zorder=5)
        ax.text(0.5, 1.06, title, ha="center", fontsize=11, weight="bold", color=tcolor)
        ax.text(0.5, -0.11, sub, ha="center", fontsize=8.3, color="#444444")

    def dist(p, c):
        return np.hypot(pts[:, 0] - c[0], pts[:, 1] - c[1])

    # ball
    ax = axes[0]
    base(ax, "ball query(基准)", "半径内取≤nsample个;\n覆盖率低(0.20/0.30)、可漏点", C_PNPP)
    for c in centers:
        ax.add_patch(Circle(c, 0.22, fill=False, edgecolor=C_PNPP, lw=1.5, alpha=0.8))
        d = dist(pts, c)
        idx = np.where(d <= 0.22)[0]
        for i in idx:
            ax.plot([c[0], pts[i, 0]], [c[1], pts[i, 1]], color=C_PNPP, lw=0.5, alpha=0.5, zorder=2)

    # raw knn
    ax = axes[1]
    base(ax, "raw KNN-k", "每中心固定取 k 个最近;\n不看半径,覆盖仍不全(负对照)", C_VIOLET)
    for c in centers:
        d = dist(pts, c)
        idx = np.argsort(d)[:8]
        for i in idx:
            ax.plot([c[0], pts[i, 0]], [c[1], pts[i, 1]], color=C_VIOLET, lw=0.5, alpha=0.55, zorder=2)

    # knn cover
    ax = axes[2]
    base(ax, "KNN + coverage 修复", "KNN 后给未覆盖点补最近中心;\n覆盖=100%,但重叠仍高", C_PN)
    assigned = set()
    for c in centers:
        d = dist(pts, c)
        idx = np.argsort(d)[:8]
        for i in idx:
            assigned.add(i)
            ax.plot([c[0], pts[i, 0]], [c[1], pts[i, 1]], color=C_PN, lw=0.5, alpha=0.5, zorder=2)
    for i in range(len(pts)):
        if i not in assigned:
            cc = np.argmin([np.hypot(*(pts[i] - c)) for c in centers])
            ax.plot([centers[cc][0], pts[i, 0]], [centers[cc][1], pts[i, 1]],
                    color=C_BAD, lw=0.8, alpha=0.85, zorder=3)

    # adaptive
    ax = axes[3]
    base(ax, "adaptive_cover", "先按最近中心 100% 覆盖;\n每对中心重叠硬约束 ≤1/3", C_HL)
    prim = np.array([np.argmin([np.hypot(*(p - c)) for c in centers]) for p in pts])
    cols = [C_PNPP, C_PN, C_VIOLET]
    for i, p in enumerate(pts):
        c = centers[prim[i]]
        ax.plot([c[0], p[0]], [c[1], p[1]], color=cols[prim[i]], lw=0.5, alpha=0.55, zorder=2)

    fig.suptitle("SA1 分组方式对比:核心权衡是「邻域覆盖率」vs「组间重叠」——但本轮证据显示提高覆盖并不改善 WSS 精度",
                 fontsize=12, y=1.02, x=0.5)
    fig.tight_layout()
    save(fig, "s4_grouping_mechanism.png")


# ============================================================ S5 support/query 采样
def fig_sampling():
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.4), gridspec_kw={"width_ratios": [3, 2]})

    ax = axes[0]
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    ax.text(2, 95, "support 点采样(4 种)", fontsize=13, weight="bold", color=C_INK)
    rows = [
        ("FPS-2000", "确定性最远点采样,固定 2000 点", "#9ec5f4"),
        ("vertex random-5000", "壁面顶点等概率无放回,每 epoch 重采", C_PNPP),
        ("fixed-FPS-5000", "离线持久化的确定性 FPS 子集", "#6da7ec"),
        ("fps_multistart-5000", "8 个不同起点 FPS,按 seed/epoch 轮选", C_VIOLET),
    ]
    y = 80
    for name, desc, c in rows:
        box(ax, 3, y, 26, 11, name, c, fs=9.6)
        ax.text(31, y + 5.5, desc, fontsize=9.2, color=C_INK, va="center")
        y -= 15
    ax.text(2, 14, "★ Phase-V 结论:random-5000(每 epoch 重采)是主要增益来源;\n"
            "   固定/多起点 FPS 均回退。random-5000 让网络每 epoch 看到更完整的病例内 WSS 型态。",
            fontsize=9.3, color="#0d6b49", linespacing=1.6)

    ax = axes[1]
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    ax.text(2, 95, "SAME vs SEP(query 合同)", fontsize=13, weight="bold", color=C_INK)
    # SAME
    box(ax, 6, 66, 40, 20, "SAME\nquery = support\n(同一子集)", C_PNPP, fs=10)
    ax.text(6, 60, "支撑点与被预测点是同一批", fontsize=8.8, color=C_MUT)
    # SEP
    box(ax, 6, 30, 18, 20, "support\n随机子集", "#9ec5f4", fs=9)
    box(ax, 30, 30, 18, 20, "query\n独立子集", C_HL, fs=9)
    ax.text(24, 40, "≠", fontsize=16, ha="center", va="center", color=C_BAD)
    ax.text(6, 24, "SEP=独立重采一批 query 点", fontsize=8.8, color=C_MUT)
    ax.text(2, 14, "PointNet 分支 P2V(SEP)优于 P1V(SAME);\n"
            "PointNet++ 分支 Q2V(SEP)与 Q1V(SAME)无一致方向,\n以 Q1V 为主候选。",
            fontsize=9, color=C_INK, linespacing=1.6)
    fig.tight_layout()
    save(fig, "s5_sampling.png")


# ============================================================ S6 归一化口径
def fig_normalization():
    fig, ax = plt.subplots(figsize=(12.2, 4.0))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    ax.text(1, 95, "目标 WSS 归一化口径(决定网络学的是什么)", fontsize=14, weight="bold", color=C_INK)
    cards = [
        ("global log-z(主口径)", C_PNPP,
         "y=(log(WSS+ε)−μ)/σ\nμ,σ 由 train 病例峰值壁面点共享\n保留病例间幅值差异,可反变换回 Pa",
         "两条队列合并时点池化统计被密网格病例主导,\n是 high-WSS 幅值压缩的混杂因素"),
        ("case-max:WSS/WSSmax", C_BAD,
         "每例真值 max=1,移除病例间幅值\n测试时只输出相对场,不恢复 Pa",
         "No-Go:R²_cb/Spearman 明显下降,\nmax 对单点尖峰敏感,输出可能出负值"),
        ("逐病例/逐父系等权 log-z", C_HL,
         "log-z 统计改用病例等权(A1)\n或 AG/AAA/ILO 父系等权(A1b)聚合",
         "A1b 保留为确认候选:\n主 R²、RMSE 与高尾护栏同步最好"),
    ]
    x = 2
    for title, c, body, note in cards:
        box(ax, x, 20, 30, 66, "", "white", ec=c, lw=1.8, tc=C_INK)
        ax.text(x + 1.5, 80, title, fontsize=10.3, weight="bold", color=c)
        ax.text(x + 1.5, 62, body, fontsize=8.9, color=C_INK, va="center", linespacing=1.6)
        ax.text(x + 1.5, 34, note, fontsize=8.4, color=C_MUT, va="center", linespacing=1.55)
        ax.plot([x + 1.5, x + 28.5], [45, 45], color="#e0e0dc", lw=1)
        x += 32
    ax.text(1, 12, "另有 cohort one-hot 特征(A2/A4):把 AG/AAA/ILO 队列标签作为 0/1 输入维,单独未见增益。",
            fontsize=9, color=C_MUT)
    ax.text(1, 6, "high-risk 主定义固定为每例真值 top10% 点(正比例缩放不改排序),用 IoU / 峰值距离 / 质心距离评定。",
            fontsize=9, color=C_MUT)
    save(fig, "s6_normalization.png")


if __name__ == "__main__":
    fig_pipeline()
    fig_pointnet_e2()
    fig_pointnetpp()
    fig_grouping_mechanism()
    fig_sampling()
    fig_normalization()
    print("done ->", OUT)
