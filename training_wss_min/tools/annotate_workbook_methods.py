"""Plain-language method annotations for the results workbook.

Two things, both idempotent:

1. A new sheet ``方法说明对照`` with (a) a vocabulary block for the recurring jargon
   (support/query, SAME/SEP, FPS, knn_cover, LocalGeoPE, PointNeXt-R, DropPath, log_z,
   pinball, ...) and (b) one row per experiment family / code explaining in plain words
   what was changed, relative to which base model, with the key settings and the source
   document section.
2. Column B (实验 / 模型) of both result sheets gets ``｜说明：<one sentence>`` appended
   to every data row, so a reader sees what the code means without leaving the row.
   Existing text is kept; the suffix is only added once (re-running is a no-op).

Every other cell (values, styles, comments, hyperlinks), merged range and dimension is
verified unchanged after saving; a backup is written next to the experiment records.

    python -m training_wss_min.tools.annotate_workbook_methods [--workbook PATH] [--dry-run]
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import os
import re
import tempfile
import time
from pathlib import Path

from openpyxl import load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from training_wss_min import config as C
from training_wss_min.tools.report_wss_recovery import cell_record

ROOT = C.PROJECT_ROOT
BOOK = ROOT / "docs/03-汇报材料/WSS_PointNet实验矩阵与结果汇总last.xlsx"
BACKUP_DIR = ROOT / "training_wss_min/experiments/workbook_method_annotations"
SHEET = "方法说明对照"
SUFFIX = "｜说明："
DOC_V5 = "docs/02-推进与变更/WSS_PINN/WSS_V5_训练实验跟踪.md"
DOC_OLD = "docs/02-推进与变更/WSS最小化_训练实验跟踪.md"

# ----------------------------------------------------------------------------------------------
# 1) vocabulary
# ----------------------------------------------------------------------------------------------
VOCAB = [
    ("support / query", "support = 编码器看到的壁面点（每例随机 5000 点）；query = 要出预测的点。训练时 query 也是 5000 点，评估时 query = 全部壁面节点。"),
    ("SAME / SEP（IND、独立 query）", "SAME：训练时 query 与 support 是同一批点，解码器退化为恒等；SEP/IND/independent：训练时 query 另外独立随机采样，解码器（3-NN 插值等）真正被训练。"),
    ("FPS / random / fps_multistart", "support 点的采样方式：最远点采样（固定子集）/ 每 epoch 随机 / 多起点 FPS 池。"),
    ("SA1/SA2/SA3、center、ball、knn、knn_cover、adaptive_cover", "PointNet++ 的三级集合抽象：从 support 点里选 center（125/125/32 个），每个 center 聚合一个邻域；邻域可用固定半径球（ball）、k 近邻（knn）、k 近邻+未覆盖点补边（knn_cover）或自适应覆盖（adaptive_cover）。"),
    ("D2-K64 / c125×k64", "SA1 用 125 个 center、每个 center 聚合 64 个近邻（knn_cover），SA2 125 center/16 邻居，SA3 32 center/16 邻居；是 V5 之前确定的骨干底座。"),
    ("PointNeXt-R（PNXR、InvRes、sa_blocks=(1,1,0)）", "在 SA1、SA2 各加一个倒残差块（inverted residual MLP，γ 初值 1e-3），提高局部编码容量。"),
    ("LocalGeoPE（GeoPE）", "局部几何位置编码：每条邻域边的消息里加入相对坐标 Δxyz/r、距离和三个语义几何量的差分（Δ弧长、Δ局部半径、Δ曲率）；纯 XYZ 版只用前两项。"),
    ("L-SA2（局部 Transformer）", "在 SA2 每个 center 的邻域内部，把邻居消息当 token 做一层多头自注意力后再 max 聚合；不是 center 之间的全局注意力。"),
    ("coarse attention / Bottleneck Transformer（BT）", "在最粗一级（SA3 的 32 或 125 个 center）之间做全局自注意力；BT 是导师版本（token 条件 MLP + 相对几何偏置 + γ 缩放残差）。"),
    ("QAD-Lite / SEP-Kernel / local_attn 解码器", "从 support 特征插值到 query 点的方式：3-NN 逆距离插值是默认；QAD-Lite 加一个由 query 自身特征驱动的门控残差；SEP-Kernel 学非负归一化的插值权重；local_attn 用 k 个邻居的几何条件权重 + 零初始化残差头。"),
    ("局部壁面分支（local branch，A1）", "在 support 分辨率（5000 点）上另开一条多尺度球邻域消息通路（半径 0.015/0.03/0.06 ≈ 2.5/5/10 mm），零初始化后加到解码前特征。"),
    ("query patch / FiLM（E3）", "对每个 query 点，从完整壁面点云取 K=16 个最近邻组成 patch，用编码器的逐例全局向量做 FiLM 调制，零初始化残差加到输出。"),
    ("方向邻域（E2）", "局部分支的邻居按切平面的轴向/周向四个象限配额选取，并用法向门控去掉跨壁面的邻居。"),
    ("log_z、log_z 全局统计、casebal/cohortbal", "目标 ln(WSS) 用训练集统计做 z-score；point-pooled 统计被高密度病例主导，casebal/cohortbal 改为逐病例/逐父系（AG/AAA/ILO）等权计算统计。"),
    ("H1 / H2 / pinball q90 λ0.2", "H1：加一个热点分类通道（top10% BCE）；H2：主 MSE 之外再加 0.2 倍的 90 分位 pinball 损失，抬高上尾。"),
    ("raw-Huber / Pa-MSE", "把预测换回 Pa 后再加一项物理空间的 Huber 或 MSE 辅助损失。"),
    ("DropPath / HeadDrop / NeighborDrop（REG-P10）", "正则化：随机跳过残差块 / 输出头 dropout / 邻域内随机丢边；REG-P10 = DropPath 最大 0.10。"),
    ("logR", "把 ln(local_radius) 作为第 7 维输入（R1 配方的一部分）。"),
    ("V5 十维几何特征（R4）", "管道坐标 ρ、sinθ、cosθ、半径梯度 dR/ds、到分叉距离、到端点距离、端区标记、对齐坐标架下的 PCA 法向 3 维。"),
    ("壁面主曲率族（A4）", "在 PCA 法向切平面里做局部二次拟合得到的主曲率 k1/k2、高斯曲率、curvedness（细 32 邻居 / 粗 128 邻居两个尺度）和中心线切向·法向。"),
    ("Jensen 回变换 / NLL 头", "log 空间预测换回 Pa 时用 exp(μ+σ²/2) 而不是 exp(μ)；σ 由高斯 NLL 双通道头逐点给出。"),
    ("Murray 分支流量先验（F6，X5）", "按协议常量（主干 1、左右髂总各 0.5）和 Murray 定律（末支按半径³）算每支流量份额 log Q，以及 Poiseuille 尺度 log τ0 = log Q − 3 log R；只用中心线半径。"),
    ("截面 token 上下文（S3，X11）", "把 support 点按（血管段, 4 mm 弧长 bin）池化成 token，在整棵血管树上做 2 层 Transformer，零初始化残差加到 query patch 的 FiLM 上下文。"),
    ("同期对照（contemporaneous control）", "把底座配置原样再训一次，用来量化同配置的运行抖动；X0 对 C1 的差 −0.040 说明单 seed 之间 ±0.03–0.04 的物理 R² 差异读不出来。"),
    ("test16 / test15 / test27 / test36 / test34 / val21", "不同阶段的测试划分：AG 16 例 → AG 15 例 → AG+AAA 分层 27 例 → 含 ILO 的 36 例 → V5 的 34 例（去掉 2 例坏数据）；val21 是 Q2V 阶段的开发验证集。"),
    ("best / last / ema", "best = 训练损失最低的 epoch；last = 第 400 轮；ema = 权重指数滑动平均的影子模型。"),
    ("物理 R²_cb / 归一化 R²_cb", "病例等权的 R²：分别在 Pa 空间和 log_z 空间计算；物理空间被高 WSS 尾部主导，归一化空间噪声小。"),
]

# ----------------------------------------------------------------------------------------------
# 2) family-level base descriptions (what the base model of a matrix is)
# ----------------------------------------------------------------------------------------------
BASE = {
    "E2": "PointNet 导师宽通道版（逐点 MLP 6→256→512，全局池化后 1024→512→256→1），输入 xyz+弧长/半径/曲率，FPS-2000 点，全局 log-z",
    "PNPP": "经典 PointNet++ 三级 SA（2000→500→125→32，半径 0.05/0.10/0.20，FP k=3，0.22M 参数）",
    "Q1V": "PointNet++ 三级 SA，每例随机 5000 顶点为 support，SAME（query=support），FPS center 500/125/32，nsample 16，width 32，AG/AAA 106/0/27 划分",
    "Q2V": "同 Q1V 但 SEP（query 独立采样）",
    "D2K64": "D2-K64 骨干：随机 5000 support，固定 center 125/125/32，SA1 用 knn_cover k=64，SA2/SA3 16 邻居，width 32；先在 AG/AAA test27，后在 mixed 138/0/36（含 ILO）",
    "S3": "S3-GEOPE：D2-K64 + PointNeXt-R 倒残差块(1,1,0) + LocalGeoPE 7 维边编码，输入 6 维（xyz+弧长/半径/曲率），纯 MSE，mixed 138/0/36",
    "REGP10": "REG-P10：S3-GEOPE + DropPath 最大 0.10",
    "LSA2": "REG-P10-LSA2：REG-P10 + SA2 邻域内局部 Transformer（4 头）",
    "R1": "R1：REG-P10-LSA2 + H2（MSE + 0.2×q90 pinball）+ 第 7 维 ln(local_radius)，SAME，V5 干净数据 train138/test34",
    "R4": "R4：R1 配方 + V5 十维几何特征（管道坐标、dR/ds、到分叉/端点距离、端区、对齐 PCA 法向），17 维输入，多 seed 均值物理 R² 0.556",
    "A5": "A5：R4 + 多尺度局部壁面分支（2.5/5/10 mm）+ 8 维壁面主曲率族，25 维输入，SAME，物理 R² 0.623",
    "M2": "M2：A5 改为独立 query（训练时 query 与 support 分开随机采，解码器真正被训练），固定 3-NN 解码，物理 R² 0.628，是 §16/§18 的底座",
    "C1": "C1：M2 + E2 切平面方向邻域 + E3 完整壁面 K16 query patch/FiLM，物理 R² 0.6575，是波 1/1b/2 的底座",
    "R5": "R5：R4 同配置只换回归目标（压力/速度），support 仍是壁面 5000 点，query 取内部单元（压力混合壁面），qad_lite 解码，18 维输入（多一维到壁距离）",
}

# ----------------------------------------------------------------------------------------------
# 3) code → (plain description, base key, key settings, source)
#    matched by regular expression against "<group>|<label>|<run id>"
# ----------------------------------------------------------------------------------------------
DOC_LM = "WSS_V5_局部形态实验矩阵_2026-09-17.md"
GLOSS: list[tuple[str, str, str, str, str]] = [
    # ---- early PointNet (test16) ----
    (r"^早期 PointNet.*\|Point初始模版\+random采样\+目标wss/wssmax", "最初 PointNet 模版（32→64→128 逐点 MLP + 全局池化），每 epoch 随机 5000 点重采样，目标改为逐病例 WSS/WSSmax 归一化（E3-CASE）", "", "test16；No-Go（逐例归一化让整体分布退化）", f"{DOC_OLD}｜PointNet baseline 新矩阵 2026-07-15"),
    (r"^早期 PointNet.*\|Point初始模版\+random采样", "最初 PointNet 模版，只把固定 FPS-2000 改成每 epoch 随机 5000 点（E3-GLOBAL）", "", "test16；单独增益弱", f"{DOC_OLD}｜PointNet baseline 新矩阵 2026-07-15"),
    (r"^早期 PointNet.*\|Point初始模版\+容量", "最初模版加深到 6→64→128→256→512（E4-DEEP，导师追加深度探针）", "", "train 拟合更好但 test 变差；No-Go", f"{DOC_OLD}｜PointNet E4 deeper 2026-07-15"),
    (r"^早期 PointNet.*\|Point初始模版", "最初的 PointNet 模版：逐点 MLP 32→64→128，全局 max-pool，头 256→128→64→1；输入 xyz+几何，FPS-2000，AG 61/0/16（E0-GLOBAL，共同对照）", "", "物理 R² 0.141", f"{DOC_OLD}｜PointNet baseline 新矩阵 2026-07-15"),
    (r"^早期 PointNet.*\|Point容量6-256-512\+目标wss/wssmax", "导师宽通道 PointNet（6→256→512）+ 逐病例 WSS/WSSmax 目标（E2-CASE）", "E2", "No-Go", f"{DOC_OLD}｜PointNet baseline 新矩阵 2026-07-15"),
    (r"^早期 PointNet.*\|Point容量6-256-512\+random采样", "导师宽通道 PointNet + 每 epoch 随机 5000 点（E23-GLOBAL）", "E2", "未超过 E2，无协同", f"{DOC_OLD}｜PointNet baseline 新矩阵 2026-07-15"),
    (r"^早期 PointNet.*\|Point容量6-256-512", "导师要求的宽通道 PointNet（6→256→512；1024→512→256→1，E2-GLOBAL），本轮最强，容量是主要有效因素", "E2", "物理 R² 0.214（test16）", f"{DOC_OLD}｜PointNet baseline 新矩阵 2026-07-15"),
    # ---- v4 / AAA data (test15) ----
    (r"^v4架构与AAA数据.*\|Point容量.*AG\+AAA", "宽通道 PointNet 在 v4 数据上、训练集加入 AAA（mixed 118/0/15，锁定 AG test15）", "E2", "", f"{DOC_OLD}｜v4 PointNet++ 三协议结果 2026-07-16/17"),
    (r"^v4架构与AAA数据.*\|Point容量", "宽通道 PointNet 在 v4 数据上只用 AG（61/0/15）", "E2", "", f"{DOC_OLD}｜v4 PointNet++ 三协议结果"),
    (r"^v4架构与AAA数据.*\|Point\+\+SA3\+AG\+AAA", "PointNet++ 三级 SA 在 AG+AAA 混合训练（118/0/15，锁定 AG test15）", "PNPP", "common-test15 上未超过 E2", f"{DOC_OLD}｜v4 PointNet++ 三协议结果"),
    (r"^v4架构与AAA数据.*\|Point\+\+SA3", "PointNet++ 三级 SA 只用 AG（61/0/15）", "PNPP", "", f"{DOC_OLD}｜v4 PointNet++ 三协议结果"),
    # ---- stratified split / Phase-V (test27) ----
    (r"^分层划分.*\|Point容量", "宽通道 PointNet 在新的分层划分 106/0/27（AG+AAA）上", "E2", "B0 参照", f"{DOC_OLD}｜v4 PointNet++ 三协议结果 / test27 Support-Query 两阶段矩阵"),
    (r"^分层划分.*\|P1V", "PointNet + 每例随机 5000 顶点、SAME（query=support）", "E2", "0.159", f"{DOC_OLD}｜test27 Support/Query 两阶段矩阵 2026-07-18"),
    (r"^分层划分.*\|P2V", "PointNet + 随机 5000 顶点、SEP（query 独立采样）；PointNet 分支优先候选，也是 PINN 线的 PointNet 锚点", "E2", "0.219", f"{DOC_OLD}｜test27 Support/Query 两阶段矩阵"),
    (r"^分层划分.*\|Point\+\+SA3", "PointNet++ 三级 SA 在分层划分 106/0/27 上（B1/9169）", "PNPP", "test27 上超过 E2；结论依赖划分", f"{DOC_OLD}｜v4 PointNet++ 三协议结果"),
    (r"^分层划分.*\|Q0", "PointNet++ + FPS-2000 support、SAME、固定 FPS center：从旧 ratio 采样切到固定中心/固定 support 全 query 评估的桥接组", "PNPP", "0.221", f"{DOC_OLD}｜test27 Support/Query 两阶段矩阵"),
    (r"^分层划分.*\|Q1V", "PointNet++ + 每例随机 5000 顶点 support、SAME、FPS center 500/125/32；PointNet++ 分支优先候选（07-18 阶段候选）", "Q1V", "0.276", f"{DOC_OLD}｜test27 Support/Query 两阶段矩阵"),
    (r"^分层划分.*\|Q2V", "Q1V 改成 SEP（query 与 support 独立采样）", "Q1V", "0.272，混合结果", f"{DOC_OLD}｜test27 Support/Query 两阶段矩阵"),
    (r"^分层划分.*\|Q3V", "Q1V 的 FPS center 改成随机 center", "Q1V", "0.252，回退", f"{DOC_OLD}｜test27 Support/Query 两阶段矩阵"),
    # ---- Q2V data expansion ----
    (r"^Q2V数据扩容.*\|Q2V扩数据D1\(冻结\)", "Q2V 底座、训练集加入 ILO41（train147），保持原 test27 与冻结的 Q2V 统计", "Q2V", "R² −0.021：加 ILO 反而变差", f"{DOC_OLD}｜Q2V 数据扩容 2026-07-18"),
    (r"^Q2V数据扩容.*\|Q2V扩数据D1\(重训\)", "同上但重算 train147 的目标统计", "Q2V", "只回补 +0.003", f"{DOC_OLD}｜Q2V 数据扩容"),
    (r"^Q2V数据扩容.*\|Q2V扩数据D2", "Q2V 底座、训练集加入 ILO32，在含 ILO 的 test36 上评，统计冻结", "Q2V", "相对零样本 −0.015", f"{DOC_OLD}｜Q2V 数据扩容"),
    (r"^Q2V数据扩容.*\|Q2V数据池2025\(对照\)", "pool2025 划分（138/0/36）上的 Q2V 对照（不含 ILO 入训）", "Q2V", "0.298", f"{DOC_OLD}｜Q2V 数据扩容"),
    (r"^Q2V数据扩容.*\|Q2V数据池2025\(冻结\)", "pool2025 划分、加入 ILO32 训练、统计冻结", "Q2V", "0.292", f"{DOC_OLD}｜Q2V 数据扩容"),
    (r"^Q2V数据扩容.*\|Q2V数据池2025\(重训\)", "pool2025 划分、加入 ILO32、重算 point-pooled 统计（后来发现 pooled 统计被高密度队列主导是负迁移主因）", "Q2V", "0.246", f"{DOC_OLD}｜Q2V 数据扩容 / 负迁移归因"),
    (r"^Q2V架构搜索.*邻域(\d+)·宽(\d+)", "Q2V 结构在 dev85/val21 开发划分上扫描 SA 邻居数 nsample 与通道宽度 width（{m1} 邻居 / 宽 {m2}）", "Q2V", "n32 最好；width64 参数 4 倍仅 +0.0014", f"{DOC_OLD}｜Q2V 架构搜索 2026-07-18"),
    (r"^Q2V零样本迁移", "用 Q2V（只在 AG/AAA 上训练）的 checkpoint 直接在含 ILO 的 test36 上评估（零样本）", "Q2V", "0.265", f"{DOC_OLD}｜Q2V 数据扩容"),
    (r"^Q2V半径与采样探索.*\|Q1V多起点FPS", "Q1V 的 support 采样从随机顶点改为多起点 FPS 池", "Q1V", "−0.047，不支持", f"{DOC_OLD}｜Q2V/test27 半径与采样探索"),
    (r"^Q2V半径与采样探索.*\|Q2V半径探索\(0\.6×\)", "Q2V 三级 ball 半径缩到 0.6 倍（0.03/0.06/0.12）", "Q2V", "−0.032", f"{DOC_OLD}｜Q2V/test27 半径与采样探索"),
    (r"^Q2V半径与采样探索.*\|Q2V半径探索\(0\.8×\)", "Q2V 三级 ball 半径缩到 0.8 倍（0.04/0.08/0.16）", "Q2V", "−0.029", f"{DOC_OLD}｜Q2V/test27 半径与采样探索"),
    (r"^Q2V半径与采样探索.*\|Q2V→Q1V", "用 Q2V 的权重热启动，再按 Q1V（SAME）设定微调", "Q2V", "−0.009 vs Q1V", f"{DOC_OLD}｜Q2V/test27 半径与采样探索"),
    (r"^Q1V半径.*\((\d\.\d)×", "Q1V 三级 ball 半径整体缩放到 {m1} 倍", "Q1V", "所有缩放都不优于 1.0×", f"{DOC_OLD}｜Q1V 半径探索"),
    # ---- QAD ----
    (r"^QAD-Lite三seed.*\|R0插值", "对照：Q2V 结构（nsample 32/width 32）+ 默认 3-NN 逆距离插值解码，dev85/val21{seedtxt}", "Q2V", "", f"{DOC_OLD}｜PointNet++ R1 QAD-Lite 2026-07-19"),
    (r"^QAD-Lite三seed.*\|R1-QAD", "同上但解码器换成 QAD-Lite（query 自身特征 + 相对位移驱动的门控残差）{seedtxt}", "Q2V", "3 seed 归一化 +0.014 未过 +0.015 门；No-Go", f"{DOC_OLD}｜PointNet++ R1 QAD-Lite"),
    (r"^QAD精确Q2V.*\|R0插值", "对照：精确 Q2V-10477 协议（106/0/27，nsample 16）+ 3-NN 插值{seedtxt}", "Q2V", "", f"{DOC_OLD}｜QAD 精确 Q2V 协议三种子 2026-07-19"),
    (r"^QAD精确Q2V.*\|R1-QAD", "精确 Q2V 协议 + QAD-Lite 解码{seedtxt}", "Q2V", "3 seed 归一化 −0.005、物理 −0.018；No-Go", f"{DOC_OLD}｜QAD 精确 Q2V 协议三种子"),
    # ---- SA1 grouping matrix ----
    (r"^SA1覆盖与重叠.*\|(Q1V|Q2V) (random5000|FPS-multistart5000|fixed-FPS|FPS-multistart|random) \+ (ball16|adaptive)", "{m1} 底座，support 用 {m2}（{e2}），SA1 邻域用 {m3}（{e3}）", "Q1V", "", f"{DOC_OLD}｜SA1 分组矩阵（2026-07-20）"),
    (r"^SA1覆盖与重叠.*\|Q1V n(\d+) / w(\d+)", "Q1V 底座，SA 邻居数 {m1}、通道宽 {m2}", "Q1V", "", f"{DOC_OLD}｜SA1 分组矩阵"),
    (r"^SA1覆盖与重叠.*\|Q1V adaptive_cover", "Q1V 的 SA1 分组改为 adaptive_cover（先全覆盖主分配，再选最多一个第二归属）", "Q1V", "", f"{DOC_OLD}｜SA1 分组矩阵"),
    (r"^SA1覆盖与重叠.*\|Q1V ball32", "Q1V 的 SA1 ball 邻居上限 16→32", "Q1V", "", f"{DOC_OLD}｜SA1 分组矩阵"),
    (r"^SA1覆盖与重叠.*\|Q1V raw KNN-(\d+)", "Q1V 的 SA1 改为纯 k 近邻（k={m1}），不补未覆盖点", "Q1V", "", f"{DOC_OLD}｜SA1 分组矩阵"),
    (r"^SA1覆盖与重叠.*\|Q1V KNN-(\d+)-cover", "Q1V 的 SA1 改为 k 近邻 + 未覆盖点补边（knn_cover，k={m1}；导师标准分组）", "Q1V", "KNN-8-cover 0.259", f"{DOC_OLD}｜SA1 分组矩阵"),
    # ---- norm/loss ----
    (r"^归一化口径×尾部损失.*\|NormLoss A0", "pool2025 划分上的 Q2V 对照：目标统计按全部点 pooled（被高密度病例主导）", "Q2V", "0.246", f"{DOC_OLD}｜Q2V-pool2025 归一化口径×尾部损失 2026-07-20"),
    (r"^归一化口径×尾部损失.*\|NormLoss A1b", "目标 log-z 统计改为逐父系（AG/AAA/ILO）等权计算 + MSE", "Q2V", "0.275（+0.030），唯一候选", f"{DOC_OLD}｜Q2V-pool2025 归一化口径×尾部损失"),
    (r"^归一化口径×尾部损失.*\|NormLoss A1(?!b)", "目标 log-z 统计改为逐病例等权计算 + MSE", "Q2V", "0.245，不增", f"{DOC_OLD}｜Q2V-pool2025 归一化口径×尾部损失"),
    (r"^归一化口径×尾部损失.*\|NormLoss A2", "A1 + 输入加 cohort one-hot（AG/AAA/ILO）", "Q2V", "不优于 A1", f"{DOC_OLD}｜Q2V-pool2025 归一化口径×尾部损失"),
    (r"^归一化口径×尾部损失.*\|NormLoss A3", "A1 + 物理空间 raw-Huber 辅助损失 λ=0.2", "Q2V", "回退", f"{DOC_OLD}｜Q2V-pool2025 归一化口径×尾部损失"),
    (r"^归一化口径×尾部损失.*\|NormLoss A4", "A1 + cohort one-hot + raw-Huber", "Q2V", "最差", f"{DOC_OLD}｜Q2V-pool2025 归一化口径×尾部损失"),
    # ---- SA1 scale matrix ----
    (r"^SA1尺度与容量.*\|bridge random5000 \+ 500c \+ k64 SEP", "bridge（随机 5000 support、固定 500/125/32 center、SA1 knn_cover k=64）改 SEP", "Q1V", "No-Go", f"{DOC_OLD}｜bridge SEP 对照 2026-07-22"),
    (r"^SA1尺度与容量.*\|bridge", "bridge：随机 5000 support、固定 500/125/32 center、SA1 knn_cover k=64（连接 Q1V 与 D2 家族的桥接臂）", "Q1V", "0.244", f"{DOC_OLD}｜SA1-scale 矩阵 2026-07-21"),
    (r"^SA1尺度与容量.*\|D1-fixed 全点 \+ k(\d+)", "D1-fixed：用全部壁面点做 support（约 1.3 万–7 万点），固定 500/125/32 center，SA1 k={m1}", "Q1V", "全点不优于随机 5000", f"{DOC_OLD}｜SA1-scale 矩阵"),
    (r"^SA1尺度与容量.*\|D1-prop 全点比例center \+ k(\d+)", "D1-prop：全部壁面点 + center 数按点数比例（0.1N/0.25/0.25），SA1 k={m1}，batch 2", "Q1V", "全败", f"{DOC_OLD}｜SA1-scale 矩阵"),
    (r"^SA1尺度与容量.*\|D2 c(\d+) ?× ?k(\d+)", "D2：随机 5000 support，SA1 center 减到 {m1} 个、邻域放大到 k={m2}（更少 center × 更大邻域）", "Q1V", "c125×k128 = 0.2765 全轮最好，也是 PINN 线的 PointNet++ 锚点", f"{DOC_OLD}｜SA1-scale 矩阵"),
    (r"^SA1尺度与容量.*\|D3 random10000 \+ k(\d+)", "D3：support 增到随机 10000 点，固定 500/125/32 center，SA1 k={m1}", "Q1V", "不优于 5000", f"{DOC_OLD}｜SA1-scale 矩阵"),
    (r"^SA1尺度与容量.*\|D4 KNN-(\d+)-cover w64", "D4：KNN-{m1}-cover 分组 + 通道宽 64", "Q1V", "cover 分组下加宽反而退化", f"{DOC_OLD}｜SA1-scale 矩阵"),
    (r"^SA1尺度与容量.*\|D5-A stem", "D5-A：在 w64 KNN-8-cover 上把 stem 改成 6→32→64 瓶颈", "Q1V", "+0.013 相对父配置，仍低于 w32", f"{DOC_OLD}｜SA1-scale 矩阵"),
    # ---- ILO / S0–S5 ----
    (r"^ILO数据扩容与S0.*\|F1", "D2-K64 骨干，AG/AAA 训练集加入 ILO41（train147），仍在 AG/AAA test27 评", "D2K64", "−0.017：加 ILO 无跨域正迁移", f"{DOC_OLD}｜D2 c125×k64：ILO 两协议 2026-07-23"),
    (r"^ILO数据扩容与S0.*\|M0", "D2-K64 只在 AG/AAA 训练，在 mixed test36（含 ILO）零样本评", "D2K64", "", f"{DOC_OLD}｜D2 c125×k64：ILO 两协议"),
    (r"^ILO数据扩容与S0.*\|M1/S0", "M1 = S0：D2-K64 在 mixed138（含 ILO32）训练、test36 评，是 S1–S5 结构臂的公共父控制", "D2K64", "0.261", f"{DOC_OLD}｜D2 c125×k64：ILO 两协议"),
    (r"^ILO数据扩容与S0.*\|S1", "S1：M1 + 两个额外输入（半径梯度 radius_gradient、病例尺度 coord_scale，8 维）", "D2K64", "+0.002，不支持", f"{DOC_OLD}｜结构模块结果 2026-07-23"),
    (r"^ILO数据扩容与S0.*\|S2", "S2：M1 + PointNeXt-R 倒残差块 (1,1,0)", "D2K64", "−0.009 单独不稳", f"{DOC_OLD}｜结构模块结果"),
    (r"^ILO数据扩容与S0.*\|S3", "S3：S2 + LocalGeoPE（邻域边加相对坐标/距离/Δ弧长/Δ半径/Δ曲率）；唯一对 AG/AAA/ILO 同时提高的臂，三 seed 确认 +0.022", "D2K64", "0.292", f"{DOC_OLD}｜结构模块结果 / 三种子确认"),
    (r"^ILO数据扩容与S0.*\|S4", "S4：S2 + SA3 32 个粗中心之间的全局注意力（coarse attention）", "D2K64", "+0.018 弱正", f"{DOC_OLD}｜结构模块结果"),
    (r"^ILO数据扩容与S0.*\|S5C", "S5C：S2 改独立 query + 3-NN 插值（S5 的匹配控制）", "D2K64", "0.250", f"{DOC_OLD}｜结构模块结果"),
    (r"^ILO数据扩容与S0.*\|S5 ", "S5：S5C + Conservative SEP-Kernel 解码（只学非负归一化的插值权重）", "D2K64", "+0.024 vs S5C，条件正", f"{DOC_OLD}｜结构模块结果"),
    # ---- S3 root-cause / regularisation ----
    (r"S3口径与域条件根因.*\|RC0", "S3 底座，目标统计改为 train138 全部点 pooled 重算", "S3", "−0.014", f"{DOC_OLD}｜S3-GEOPE 根因矩阵 2026-07-23→24"),
    (r"S3口径与域条件根因.*\|RC1", "S3 底座，目标 log-z 统计改为逐病例等权（casebal）", "S3", "三 seed 均值 +0.004，跨零", f"{DOC_OLD}｜S3-GEOPE 根因矩阵"),
    (r"S3口径与域条件根因.*\|RC2", "S3 底座，目标统计改为逐父系等权（cohortbal）；AAA 回补但 ILO 回落", "S3", "均值 −0.002", f"{DOC_OLD}｜S3-GEOPE 根因矩阵"),
    (r"S3口径与域条件根因.*\|RC3", "S3 底座，输入加 cohort one-hot（6→9 维）", "S3", "均值 −0.008；No-Go", f"{DOC_OLD}｜S3-GEOPE 根因矩阵"),
    (r"S3口径与域条件根因.*\|RC4", "S3 底座，加物理空间 raw-Huber λ=0.2", "S3", "−0.009", f"{DOC_OLD}｜S3-GEOPE 根因矩阵"),
    (r"S3口径与域条件根因.*\|RC5", "S3 底座，cohortbal 统计 + cohort one-hot 同时用", "S3", "−0.027 最差", f"{DOC_OLD}｜S3-GEOPE 根因矩阵"),
    (r"S3正则化三seed.*\|REG-P05", "S3 + DropPath 最大 0.05（残差块随机跳过，沿深度线性增加）", "S3", "三 seed 全正；Weak-Go", f"{DOC_OLD}｜S3 正则化三种子矩阵 2026-07-24/25"),
    (r"S3正则化三seed.*\|REG-P10", "S3 + DropPath 最大 0.10；后来成为 REG-P10 底座", "S3", "三 seed 小幅正", f"{DOC_OLD}｜S3 正则化三种子矩阵"),
    (r"S3正则化三seed.*\|REG-H", "S3 + 输出头 dropout 0.10", "S3", "不稳", f"{DOC_OLD}｜S3 正则化三种子矩阵"),
    (r"S3正则化三seed.*\|REG-N05", "S3 + NeighborDrop 0.05（邻域内随机丢边、保留最近边）", "S3", "均值负；关闭", f"{DOC_OLD}｜S3 正则化三种子矩阵"),
    (r"^三锚点纯XYZ.*\|D2", "D2 c125×k64 只用 xyz 输入（去掉弧长/半径/曲率）", "D2K64", "明显下降：几何特征是必要信息", f"{DOC_OLD}｜三锚点纯 XYZ 对照 2026-07-24/25"),
    (r"^三锚点纯XYZ.*\|M1/S0", "M1/S0 只用 xyz 输入", "D2K64", "明显下降", f"{DOC_OLD}｜三锚点纯 XYZ 对照"),
    (r"^三锚点纯XYZ.*\|S2-PNXR", "S2 PointNeXt-R 只用 xyz 输入", "D2K64", "明显下降", f"{DOC_OLD}｜三锚点纯 XYZ 对照"),
    (r"^S3 SA3全局注意力", "S3 + SA3 32 个粗中心间的全局注意力（4 头），单 seed", "S3", "Weak-Go", f"{DOC_OLD}｜S3 + SA3 coarse attention 2026-07-24"),
    (r"^正则化强度与两两交叉.*\|droppath0(\d+)", "S3 + DropPath 0.{m1}", "S3", "0.15 退化，0.20 近零", f"{DOC_OLD}｜S3 正则化强度补齐 + 两两交叉 2026-07-25"),
    (r"^正则化强度与两两交叉.*\|head_dropout0(\d+)", "S3 + 输出头 dropout 0.{m1}", "S3", "0.15 最优但小", f"{DOC_OLD}｜S3 正则化强度补齐"),
    (r"^正则化强度与两两交叉.*\|neighbordrop0(\d+)", "S3 + NeighborDrop 0.{m1}", "S3", "全负", f"{DOC_OLD}｜S3 正则化强度补齐"),
    (r"^正则化强度与两两交叉.*\|pair_", "S3 + 两种正则同时开（{label}）", "S3", "无协同", f"{DOC_OLD}｜S3 正则化强度补齐"),
    (r"^S2 XYZ \+ LocalGeoPE", "纯 xyz 输入的 S2 + 4 维 LocalGeoPE（只有相对坐标与距离）", "D2K64", "明显拉升，但仍不及带几何输入", f"{DOC_OLD}｜纯 XYZ + LocalGeoPE 2026-07-25"),
    (r"^D2 \+ PointNeXt-R \+ LocalGeoPE", "AG/AAA test27 协议下的 D2 c125×k64 + PointNeXt-R + LocalGeoPE", "D2K64", "复现正增益", f"{DOC_OLD}｜D2 c125×k64 PNXR+GeoPE 2026-07-26"),
    (r"^REG-P10局部SA注意力.*\|REG-P10 \+ G-SA3", "REG-P10 + SA3 粗中心全局注意力（对照）", "REGP10", "无增益", f"{DOC_OLD}｜REG-P10 局部 SA Transformer 矩阵 2026-07-26"),
    (r"^REG-P10局部SA注意力.*\|REG-P10 \+ L-SA(\d+)", "REG-P10 + 在 SA{m1} 的每个邻域内部做局部 Transformer（邻居消息作 token）", "REGP10", "只有 L-SA2 过门", f"{DOC_OLD}｜REG-P10 局部 SA Transformer 矩阵"),
    (r"^D2局部SA注意力.*L-SA(\d+)", "D2 PNXR+GeoPE（AG/AAA test27）+ SA{m1} 邻域内局部 Transformer", "D2K64", "SA1 退化、SA2 持平", f"{DOC_OLD}｜固定 D2 的 local-SA 对照 2026-07-27"),
    (r"^REG-P10 EdgeConv.*EC-SA(\d+)", "REG-P10 + 在 SA{m1} 既有邻域内加静态 EdgeConv 残差消息 [x_i, x_j−x_i, Δp]", "REGP10", "三臂全 No-Go", f"{DOC_OLD}｜REG-P10 静态 EdgeConv 2026-07-28"),
    (r"^RCR输入对照.*\|RCR-O0", "S3 只用几何输入（6 维），RCR 探针的对照", "S3", "对照", f"{DOC_OLD}｜S3 RCR Oracle 2026-07-28"),
    (r"^RCR输入对照.*\|RCR-O1a", "S3 + 四个出口真实 RCR 参数（log R1/R2/C，12 维病例常量；oracle，部署不可得）", "S3", "明显提升（oracle）：边界条件确有信息", f"{DOC_OLD}｜S3 RCR Oracle"),
    (r"^RCR输入对照.*\|RCR-O2", "S3 + 分层内打乱的 RCR（负对照）", "S3", "无增益", f"{DOC_OLD}｜S3 RCR Oracle"),
    (r"^Hotspot BCE.*\|O0-H1", "S3（O0）+ 第二输出通道做病例相对 top10% 热点 BCE（λ=0.2）", "S3", "小幅正，未达门", f"{DOC_OLD}｜O0 热点/高 WSS 两臂 2026-07-28"),
    (r"^Hotspot BCE.*\|O0-H2", "S3（O0）+ 0.2×q90 pinball 辅助损失（抬上尾）", "S3", "小幅正，未达门", f"{DOC_OLD}｜O0 热点/高 WSS 两臂"),
    (r"^LSA2 SAME/IND.*LSA2 (IND|SAME) (H1|H2|MSE)", "REG-P10-LSA2 底座，query 方式 {m1}（SAME=与 support 同点；IND=独立采样），损失 {m2}（MSE / H1 热点 BCE / H2 q90 pinball）", "LSA2", "SAME-H2 唯一 Go", f"{DOC_OLD}｜REG-P10-LSA2 SAME/IND × MSE/H1/H2 2026-07-29"),
    (r"^LSA2并发复现.*log\(radius\)", "LSA2-SAME-H2 + 第 7 维 ln(local_radius)（train 统计标准化）；晋级为 R1 配方", "LSA2", "过门，晋级 R1", f"{DOC_OLD}｜SAME-H2 并发复现 / +log(local_radius) 2026-07-29"),
    (r"^LSA2并发复现.*concurrent", "LSA2-SAME-H2 的同 seed 并发复现（控制 CUDA 训练轨迹漂移）", "LSA2", "", f"{DOC_OLD}｜SAME-H2 并发复现"),
    # ---- V5 line ----
    (r"^旧checkpoint.*\|旧ckpt·R1", "旧数据训练的 R1 checkpoint，在新旧共有的 34 例交集上只读重评（证明数据换新前后可比）", "R1", "0.351", f"{DOC_V5}｜§3"),
    (r"^旧checkpoint.*\|旧ckpt·R2", "旧数据的 R2（LSA2 纯 MSE，6 维）在 34 例交集上重评", "LSA2", "", f"{DOC_V5}｜§3"),
    (r"^旧checkpoint.*\|旧ckpt·R3", "旧数据的 R3（S3-GEOPE 纯 MSE，6 维）在 34 例交集上重评{seedtxt}", "S3", "", f"{DOC_V5}｜§3"),
    (r"^V5新数据.*\|V5·R1 ", "R1 配方只换成 V5 干净数据（真实 mm、atlas 特征、train138/test34）{seedtxt}", "R1", "3 seed 均值 0.457：只换数据 +0.10", f"{DOC_V5}｜§2–§4"),
    (r"^V5新数据.*\|V5·R2", "R2：LSA2 纯 MSE、6 维，只换数据", "LSA2", "", f"{DOC_V5}｜§3"),
    (r"^V5新数据.*\|V5·R3", "R3：S3-GEOPE 纯 MSE、6 维，只换数据{seedtxt}", "S3", "3 seed", f"{DOC_V5}｜§3–§4"),
    (r"^V5新数据.*\|★ V5·R4|^V5新数据.*\|V5·R4 R1 \+ V5 十维几何特征", "R4：R1 + V5 十维几何特征（管道坐标、dR/ds、到分叉/端点距离、端区、对齐 PCA 法向；17 维）{seedtxt}", "R4", "多 seed 均值 0.556：法向贡献最大", f"{DOC_V5}｜§4–§5"),
    (r"^V5新数据.*\|V5·R4 消融：去分叉/端点距离", "R4 去掉到分叉/端点距离与端区三维（14 维）", "R4", "下降，作用在尾部", f"{DOC_V5}｜§5.1(b)"),
    (r"^V5新数据.*\|V5·R4 消融：去 dR/ds", "R4 去掉半径梯度 dR/ds（16 维）", "R4", "下降", f"{DOC_V5}｜§5.1(b)"),
    (r"^V5新数据.*\|V5·R4 消融：去对齐 PCA 法向", "R4 去掉对齐坐标架下的 PCA 法向三维（14 维）", "R4", "最大单项下降", f"{DOC_V5}｜§5.1(b)"),
    (r"^V5新数据.*\|V5·R4 消融：去管道坐标", "R4 去掉管道坐标 ρ、sinθ、cosθ（14 维）", "R4", "噪声内", f"{DOC_V5}｜§5.1(b)"),
    (r"^V5新数据.*\|V5·R4 高斯 NLL", "R4 配方改为高斯 NLL 双通道头（μ、log σ²），无 pinball；表内为 exp(μ) 口径，逐点 Jensen 口径为 0.600", "R4", "μ 通道 0.549；Jensen 0.600", f"{DOC_V5}｜§5.1(d)"),
    (r"^V5新数据.*\|V5·R4 尾部目标：pinball q90 λ0\.50", "R4 的 pinball 权重 0.2→0.5", "R4", "噪声内", f"{DOC_V5}｜§5.1(c)"),
    (r"^V5新数据.*\|V5·R4 尾部目标：pinball q95", "R4 的 pinball 分位 0.90→0.95", "R4", "噪声内", f"{DOC_V5}｜§5.1(c)"),
    (r"^时间条件模型.*\|V5·T1", "T1：R1 + 协议波形相位特征（q_norm/dq_norm/t_sin/t_cos），训练每例每 epoch 随机一帧（81 帧），表内为峰值帧口径", "R1", "峰值帧持平", f"{DOC_V5}｜§10"),
    (r"^时间条件模型.*\|V5·T4", "T4：R4 + 相位特征，随机帧训练；峰值帧监督被稀释到 12%", "R4", "峰值帧下降", f"{DOC_V5}｜§10"),
    # ---- BT matrix ----
    (r"^Bottleneck Transformer.*\|对照·密度", "BT-0 对照：R4 把最粗层 center 从 32 加到 125，不加注意力（分离 token 密度效应）", "R4", "平", f"{DOC_V5}｜§11"),
    (r"^Bottleneck Transformer.*\|对照·容量", "BT-C 对照：125 token + SA3 邻域内局部 Transformer（有容量、无全局可达）", "R4", "平", f"{DOC_V5}｜§11"),
    (r"^Bottleneck Transformer.*\|对照·条件", "BT-6 对照：125 token，只有 token 条件 MLP、layers=0 无注意力", "R4", "平", f"{DOC_V5}｜§11"),
    (r"^Bottleneck Transformer.*\|BT-1", "BT-1：我们原有的 coarse_attention（32 token，相对几何偏置）", "R4", "单 seed", f"{DOC_V5}｜§11"),
    (r"^Bottleneck Transformer.*\|BT-2", "BT-2：导师的瓶颈 Transformer 接在 32 个粗 token 上（1 层，token 条件用我们的 17 维几何特征，γ 近恒等启动）", "R4", "归一化小幅、边缘显著", f"{DOC_V5}｜§11"),
    (r"^Bottleneck Transformer.*\|BT-3", "BT-3 主候选：瓶颈 Transformer 接在 125 个粗 token 上", "R4", "主候选，归一化小幅、边缘显著", f"{DOC_V5}｜§11"),
    (r"^Bottleneck Transformer.*\|BT-4", "BT-4：125 token、瓶颈 2 层", "R4", "单 seed", f"{DOC_V5}｜§11"),
    (r"^Bottleneck Transformer.*\|BT-5b", "BT-5b：BT-3 去掉相对几何偏置", "R4", "单 seed", f"{DOC_V5}｜§11"),
    (r"^Bottleneck Transformer.*\|BT-5", "BT-5：导师原版的 token 条件方式（绝对 xyz 的 MLP）", "R4", "无增益", f"{DOC_V5}｜§11"),
    (r"^Bottleneck Transformer.*\|BT-7", "BT-7：导师原版残差强度 γ=1.0（标准 Transformer）", "R4", "不显著", f"{DOC_V5}｜§11"),
    (r"^Bottleneck Transformer.*\|BT-8", "BT-8：导师原版 dropout 0.1", "R4", "不显著", f"{DOC_V5}｜§11"),
    (r"^Bottleneck Transformer.*\|R4 基准", "R4 基准（32 粗 token、无注意力）多 seed 均值，BT 矩阵的参照", "R4", "多 seed 均值参照", f"{DOC_V5}｜§11.6"),
    # ---- V6 single frame ----
    (r"^V6单帧.*\|V6·A0 ", "A0：R4 改高斯 NLL 头，物理空间用逐点 Jensen 回变换 exp(μ+σ²/2)", "R4", "收益全在回变换", f"{DOC_V5}｜§12"),
    (r"^V6单帧.*\|V6·A1a", "A1a：只把 SA1 中心 125→256 加密", "R4", "归一化小幅", f"{DOC_V5}｜§12"),
    (r"^V6单帧.*\|V6·A1b", "A1b：只把 SA1 半径 0.05→0.03", "R4", "带内", f"{DOC_V5}｜§12"),
    (r"^V6单帧.*\|V6·A1 ", "A1：新增 support 分辨率上的多尺度局部壁面分支（2.5/5/10 mm 球邻域 + 法向差分，零初始化）", "R4", "归一化超带", f"{DOC_V5}｜§12"),
    (r"^V6单帧.*\|V6·A2a", "A2a：只把训练 query 改成独立采样（修训练/评估解码不一致）", "R4", "平：收益不在协议", f"{DOC_V5}｜§12"),
    (r"^V6单帧.*\|V6·A2b", "A2b：A2a + 16 邻域几何条件的可学解码器（local_attn）", "R4", "归一化小幅", f"{DOC_V5}｜§12"),
    (r"^V6单帧.*\|V6·A3", "A3：病例级幅值头（最粗层池化→逐例标量加到输出）", "R4", "塌成常数", f"{DOC_V5}｜§12"),
    (r"^V6单帧.*\|V6·A4b", "A4b：输入加分支语义 one-hot + 分支内相对弧长", "R4", "无增量", f"{DOC_V5}｜§12"),
    (r"^V6单帧.*\|V6·A4 ", "A4：输入加 8 维壁面主曲率族（细/粗两尺度 k1/k2/高斯曲率/curvedness + 切向·法向）", "R4", "归一化超带", f"{DOC_V5}｜§12"),
    (r"^V6单帧.*\|V6·A5b", "A5b：A5 + 高斯 NLL 头 + Jensen 回变换", "R4", "幅值校准最好", f"{DOC_V5}｜§12"),
    (r"^V6单帧.*\|V6·A5 ", "A5：A1 局部分支 + A4 壁面曲率族一起用（25 维输入）；本轮最好单帧配方", "R4", "本轮最好单帧配方", f"{DOC_V5}｜§12"),
    # ---- V6 followup D/L/M ----
    (r"^V6-A5后续.*·D1 ", "D1：A5 在标准化后把管道坐标 ρ/sinθ/cosθ 三列屏蔽为 0", "A5", "下降", f"{DOC_V5}｜§13"),
    (r"^V6-A5后续.*·D2 ", "D2：屏蔽到分叉/端点距离与端区", "A5", "持平", f"{DOC_V5}｜§13"),
    (r"^V6-A5后续.*·D3 ", "D3：屏蔽半径梯度 dR/ds", "A5", "略降", f"{DOC_V5}｜§13"),
    (r"^V6-A5后续.*·D4 ", "D4：屏蔽显式法向（局部分支的法向差分也一起去掉）", "A5", "下降", f"{DOC_V5}｜§13"),
    (r"^V6-A5后续.*·D5 ", "D5：屏蔽细尺度主曲率族", "A5", "下降", f"{DOC_V5}｜§13"),
    (r"^V6-A5后续.*·D6 ", "D6：屏蔽粗尺度主曲率族", "A5", "持平", f"{DOC_V5}｜§13"),
    (r"^V6-A5后续.*·D7 ", "D7：屏蔽中心线切向·壁面法向点积", "A5", "略降", f"{DOC_V5}｜§13"),
    (r"^V6-A5后续.*·L1 ", "L1：A5 去掉 pinball，纯 MSE", "A5", "下降", f"{DOC_V5}｜§13"),
    (r"^V6-A5后续.*·L2 ", "L2：A5 损失再加 0.1×Pa 空间 Huber", "A5", "下降", f"{DOC_V5}｜§13"),
    (r"^V6-A5后续.*·L3 ", "L3：纯 MSE + Pa-Huber", "A5", "下降", f"{DOC_V5}｜§13"),
    (r"^V6-A5后续.*·M1 ", "M1：把局部邻域分支换成参数量相同的逐点 MLP（容量对照，不看邻居）", "A5", "下降：邻域本身有用", f"{DOC_V5}｜§13"),
    (r"^V6-A5后续.*·M2 ", "M2：A5 只把 SAME 改为独立 query，固定 3-NN 解码；成为后续 §16/§18 的底座", "A5", "略高于 A5，成为底座", f"{DOC_V5}｜§13"),
    (r"^V6-A5后续.*·M3 ", "M3：M2 的 query 插值 3-NN→16-NN", "A5", "略低于 M2", f"{DOC_V5}｜§13"),
    (r"^V6-A5后续.*·M4 ", "M4：16 邻居可学几何条件解码 + query 自身几何 + 残差头（完整可学解码方案）", "A5", "归一化本矩阵最好，物理略低", f"{DOC_V5}｜§13"),
    (r"^V6-A5后续.*·M5 ", "M5：三条局部分支半径都设 0.03（去掉多尺度，参数量不变）", "A5", "下降", f"{DOC_V5}｜§13"),
    # ---- multiradius BT ----
    (r"^V6多半径BT.*·MS1 ", "MS1：M2 的 SA3 换成新的统一分组实现（单半径 0.20，最近 32 点）+ 中心条件 MLP，无全局 BT（桥接对照）", "M2", "未超 M2", f"{DOC_V5}｜§14"),
    (r"^V6多半径BT.*·MS2 ", "MS2：MS1 + 单路全局瓶颈 Transformer", "M2", "未超 M2", f"{DOC_V5}｜§14"),
    (r"^V6多半径BT.*·MS3 ", "MS3：SA3 按三个半径 0.10/0.20/0.40 分组并融合，无 BT", "M2", "未超 M2", f"{DOC_V5}｜§14"),
    (r"^V6多半径BT.*·MS4 ", "MS4 导师完整方案：三个半径各配独立 BT，再按中心融合", "M2", "未超 M2", f"{DOC_V5}｜§14"),
    (r"^V6多半径BT.*·MS5 ", "MS5：三支都用 0.20 半径 + 三个 BT（控制分支数/参数量）", "M2", "未超 M2", f"{DOC_V5}｜§14"),
    (r"^V6多半径BT.*·MS6 ", "MS6：三半径分组 + 近等参数的逐 token FFN 替代 BT", "M2", "本矩阵最高但仍在带内", f"{DOC_V5}｜§14"),
    (r"^预测速度派生WSS", "VELWSS1：用已训练的 R5V 速度模型预测内部速度，再用冻结的 Profile-Secant V3 算法推 WSS（不直接回归 WSS）", "R5", "低于直接回归", f"{DOC_V5}｜§15"),
    (r"^V5速度派生WSS·VELWSS2.*\|VELWSS2_legacy_s", "VELWSS2：用已训练的 VF6 速度模型（V07 底座＋Murray 先验）预测内部速度，再用冻结的 Profile-Secant V3 推 WSS；legacy 半径与 VELWSS1 同口径{seedtxt}", "", "三 seed 均值 0.547，高于 VELWSS1 的 0.491，仍低于直接回归 X5 0.717；CFD 速度同算子 0.965", f"{DOC_V5}｜§27"),
    (r"^V5速度派生WSS·VELWSS2.*\|VELWSS2_legacy_mean", "VELWSS2 三 seed 均值（legacy 半径，与 VELWSS1 同口径）", "", "0.547 ± 0.003；比 VELWSS1 +0.056，比 X5 直接回归 −0.170", f"{DOC_V5}｜§27"),
    (r"^V5速度派生WSS·VELWSS2.*\|VELWSS2_legacy_exact_mean", "VELWSS2 三 seed 均值，半径改为旧 bundle 半径按 1000/unit_factor 精确逐点对应（去掉旧映射的补偿，属诊断臂）", "", "0.544；比 legacy 低 0.003，说明旧映射其实是补偿而非错误", f"{DOC_V5}｜§27"),
    (r"^V5速度派生WSS·VELWSS2.*\|VELWSS2_v5atlas_mean", "VELWSS2 三 seed 均值，半径改为 V5 bundle wall_local_radius（当前部署输入的半径定义，无映射）", "", "0.549；比 legacy 高 0.002，三种半径来源差异在 ±0.003 内", f"{DOC_V5}｜§27"),
    # ---- M2 optimisation ----
    (r"^M2优化.*·MO0", "MO0：M2 配置同期从零重训一次（同期对照）", "M2", "低于历史 M2（同配置抖动）", f"{DOC_V5}｜§16"),
    (r"^M2优化.*·historical_m2", "历史 M2 本身（原 25 维、原损失、独立 query/3NN）", "M2", "历史底座", f"{DOC_V5}｜§16"),
    (r"^M2优化.*·MO-L(\d)", "MO-L{m1}：M2 损失加物理空间项（{struct}）", "M2", "无一过门", f"{DOC_V5}｜§16"),
    (r"^M2优化.*·MO-P(\d)", "MO-P{m1}：M2 只改一个训练超参（{struct}）", "M2", "无一过门", f"{DOC_V5}｜§16"),
    (r"^M2优化.*·MO-S(\d)", "MO-S{m1}：M2 只改一个结构细节（{struct}）", "M2", "无一过门", f"{DOC_V5}｜§16"),
    # ---- E / C ----
    (r"^WSS直接回归纠正.*\|E0", "E0：M2 独立 query 配置的同期重跑（对照）", "M2", "同期对照", f"{DOC_V5}｜§18"),
    (r"^WSS直接回归纠正.*\|E1", "E1：A5 SAME-query 配置的同期重跑（协议对照）", "A5", "协议对照", f"{DOC_V5}｜§18"),
    (r"^WSS直接回归纠正.*\|E2", "E2：局部分支的邻居按切平面轴向/周向象限配额选取 + 法向门控（方向邻域）", "M2", "带内，进入 C1", f"{DOC_V5}｜§18"),
    (r"^WSS直接回归纠正.*\|E3", "E3：每个 query 从完整壁面点云取 K16 patch，用粗层逐例上下文做 FiLM 调制，零初始化残差", "M2", "最强单结构，进入 C1", f"{DOC_V5}｜§18"),
    (r"^WSS直接回归纠正.*\|E4", "E4：局部分支池化改为 max + 加权 mean + std", "M2", "带内", f"{DOC_V5}｜§18"),
    (r"^WSS直接回归纠正.*\|E5", "E5：局部分支邻域改为固定毫米邻域 + 按局部半径比例的邻域", "M2", "下降", f"{DOC_V5}｜§18"),
    (r"^WSS直接回归纠正.*\|E6", "E6：加物理毫米范围内近邻差分损失（法向/切平面相容的邻居对）", "M2", "尾部病例改善", f"{DOC_V5}｜§18"),
    (r"^WSS直接回归纠正.*\|E7", "E7：加逐病例配额的 pairwise 排序损失", "M2", "带内", f"{DOC_V5}｜§18"),
    (r"^WSS直接回归纠正.*\|E8", "E8：support 采样改为 70% 空间覆盖 + 15% 高曲率 + 15% 近分叉", "M2", "带内", f"{DOC_V5}｜§18"),
    (r"^WSS直接回归纠正.*\|E9", "E9：输入加真实入口面积与公共名义流量构成的 Q/A 参考速度（病例常量）", "M2", "带内", f"{DOC_V5}｜§18"),
    (r"^WSS直接回归纠正.*\|C1", "C1 = E2 方向邻域 + E3 完整壁面 patch/FiLM；本轮优先保留，成为波 1 的底座", "M2", "本轮最好，波 1 底座", f"{DOC_V5}｜§18"),
    (r"^WSS直接回归纠正.*\|C2", "C2 = E2 + E6", "M2", "组合未超 C1", f"{DOC_V5}｜§18"),
    (r"^WSS直接回归纠正.*\|C3", "C3 = E3 + E7", "M2", "组合未超 C1", f"{DOC_V5}｜§18"),
    (r"^WSS直接回归纠正.*\|C4", "C4 = E3 + E9（按预登记规则选出的结构 + 入口 Q/A）", "M2", "组合未超 C1", f"{DOC_V5}｜§18"),
    # ---- wave 1 / 1b ----
    (r"^WSS局部信息筛选矩阵1b.*\|X0_s", "C1 配置换 seed 重跑（同 seed 配对初始化的对照）", "C1", "C1 配置四次运行跨度 0.617–0.682", f"{DOC_V5}｜§20.3"),
    (r"^WSS局部信息筛选矩阵1b.*\|X5_s", "X5 Murray 先验换 seed 复核", "C1", "三 seed 均值 0.717（sd 0.004），同 seed 配对 Δ 全部超带", f"{DOC_V5}｜§20.3"),
    (r"^WSS局部信息筛选矩阵1b.*\|X5q", "X5 只保留分支流量份额 log Q", "C1", "0.716 ≈ X5：增益来自 log Q", f"{DOC_V5}｜§20.3"),
    (r"^WSS局部信息筛选矩阵1b.*\|X5t", "X5 只保留 Poiseuille 尺度 log τ0", "C1", "0.701", f"{DOC_V5}｜§20.3"),
    (r"^WSS局部信息筛选矩阵1b.*\|X5F1", "X5 + X2 弯曲参考角", "C1", "0.715，物理无增益", f"{DOC_V5}｜§20.3"),
    (r"^WSS局部信息筛选矩阵1b.*\|X5S1a", "X5 + X7 切平面坐标架 patch", "C1", "0.709", f"{DOC_V5}｜§20.3"),
    (r"^WSS局部信息筛选矩阵｜.*\|X0｜", "X0：C1 配置同期重跑（对照）", "C1", "0.617（历史 C1 0.6575：同配置抖动 −0.04）", f"{DOC_V5}｜§20"),
    (r"^WSS局部信息筛选矩阵｜.*\|X1｜", "X1：C1 + 权重 EMA 0.999", "C1", "带内", f"{DOC_V5}｜§20"),
    (r"^WSS局部信息筛选矩阵｜.*\|X2｜", "X2：输入加 5 维弯曲参考角（周向角相对中心线曲率方向、上游 2D/5D 滞后、挠率）", "C1", "恰在带上", f"{DOC_V5}｜§20"),
    (r"^WSS局部信息筛选矩阵｜.*\|X3｜", "X3：输入加 4 维分叉参考角（嵴侧/髋部、分叉平面、分叉角、兄弟支半径比）", "C1", "带内", f"{DOC_V5}｜§20"),
    (r"^WSS局部信息筛选矩阵｜.*\|X4｜", "X4：输入加 5 维上游历史（s/D、上游半径极值比、上游最大 κR）", "C1", "带内", f"{DOC_V5}｜§20"),
    (r"^WSS局部信息筛选矩阵｜.*\|X5｜", "X5：输入加 Murray 分支流量先验（log Q 份额、log τ0）；本轮唯一大效应", "C1", "0.718（+0.100）", f"{DOC_V5}｜§20"),
    (r"^WSS局部信息筛选矩阵｜.*\|X6｜", "X6：输入加 train138 人群先验 ln WSS（按血管段/弧长/周向角分 bin，训练例留一）", "C1", "带内", f"{DOC_V5}｜§20"),
    (r"^WSS局部信息筛选矩阵｜.*\|X7｜", "X7：query patch 的相对偏移改到 query 自己的切平面坐标架（轴向/周向/法向）", "C1", "带内", f"{DOC_V5}｜§20"),
    (r"^WSS局部信息筛选矩阵｜.*\|X8｜", "X8：patch 的 mean 池化改为 query 条件注意力池化（零初始化）", "C1", "带内", f"{DOC_V5}｜§20"),
    (r"^WSS局部信息筛选矩阵｜.*\|X9｜", "X9：X7 + X8 + patch K 16→32", "C1", "带内", f"{DOC_V5}｜§20"),
    (r"^WSS局部信息筛选矩阵｜.*\|X10｜", "X10：加切平面 WSS 方向辅助头（预测轴向/周向方向，1−cos 损失 λ0.1）", "C1", "带内", f"{DOC_V5}｜§20"),
    (r"^WSS局部信息筛选矩阵｜.*\|X11｜", "X11：中心线截面 token 上下文（血管段 × 4 mm bin 的 token 上做 2 层 Transformer，零初始化残差入 FiLM）", "C1", "0.670（+0.053）：结构侧唯一超带", f"{DOC_V5}｜§20"),
    (r"^WSS局部信息筛选矩阵｜.*\|X12｜", "X12：输出头换成 4 专家混合（专家复制原头、门控看输入特征）", "C1", "带内", f"{DOC_V5}｜§20"),
    (r"^WSS局部信息筛选矩阵｜.*\|X13a｜", "X13a：81 帧全帧预训练（随机帧 + 相位特征 + 全帧统计）", "C1", "带内", f"{DOC_V5}｜§20"),
    (r"^WSS局部信息筛选矩阵｜.*\|X13b｜", "X13b：从 X13a 热启动，只训峰值帧 150 轮", "C1", "带内", f"{DOC_V5}｜§20"),
    (r"^WSS局部信息筛选矩阵｜.*\|X15｜", "X15：X2+X3+X4+X5 的 16 维特征一起追加", "C1", "0.712（F6 主导）", f"{DOC_V5}｜§20"),
    (r"^WSS局部信息筛选矩阵｜.*\|X16｜", "X16：X9 结构 + X15 特征 + X10 方向头", "C1", "0.714 / 归一化 0.864 最高", f"{DOC_V5}｜§20"),
    # ---- wave 4 (WSS sheet) ----
    (r"^WSS局部信息筛选矩阵4.*\|X5A_s(\d+)｜", "X5A：C1 + 切口盖面拟合的分流份额（log(Q外/Q内)=1.072·log(盖面面积比)+0.164，train138 拟合，只用点云盖面半径）替换 Murray R³ 份额，seed {m1}；与同 seed 的 X5 配对", "C1", "五 seed 配对：物理 −0.015/−0.006/−0.019/−0.010/+0.007（均值 −0.009），归一化 +0.004～+0.009 全正；AG/AAA 升、ILO 每个 seed 降 0.04；未取代 X5", f"{DOC_V5}｜§20.7"),
    (r"^WSS局部信息筛选矩阵4.*\|X5_s(\d+)｜", "X5（C1 + Murray 先验）再补 seed {m1}，作 5 seed 集成成员", "C1", "5 seed 集成 0.7354（3 seed 0.7325，单 seed 均值 0.717）", f"{DOC_V5}｜§20.7"),
    (r"^WSS局部信息筛选矩阵4.*\|X5B_s1234", "X5B：Murray 份额与 capfit 份额四列一起给模型", "C1", "带内，无额外价值", f"{DOC_V5}｜§20.7"),
    # ---- wave 5 (WSS sheet): stenosis index + cv3 ----
    (r"^WSS局部信息筛选矩阵5.*\|X5I_s(\d+)｜", "X5I：X5（C1 + Murray 先验）再加一列狭窄指数 ln(R_local/R_distal(segment))（sidecar v1.3，Murray 分流同一末端半径配方），seed {m1}；与同 seed 的 X5 配对", "C1", "三 seed 配对 Δ物理 −0.002/+0.006/+0.001（均值 +0.001）、归一化 0.000；零效应，X5 仍为底座", f"{DOC_V5}｜§21.6"),
    (r"^WSS局部信息筛选矩阵5.*\|X5AI_s(\d+)｜", "X5AI：X5A（capfit 分流）再加一列狭窄指数，seed {m1}；与同 seed 的 X5A 配对", "C1", "三 seed 配对 Δ物理 +0.004/+0.001/−0.003（均值 +0.001）；ILO 仍比 X5 低 0.02–0.06，狭窄列补不回 capfit 的损失", f"{DOC_V5}｜§21.6"),
    (r"^WSS局部信息筛选矩阵5.*\|X5_f(\d)_s(\d+)｜", "X5 配方在 train138 患者分组 3-fold 的 fold{m1}（train = 其余两折 91–93 例，逐折 log_z 与特征统计），seed {m2}；指标是折外 45/47/46 例，不是 test34", "C1", "折外 0.686/0.682/0.669，合并 138 例 0.6802/0.8461；折模型在冻结 test34 0.701/0.672/0.700", f"{DOC_V5}｜§21.5"),
    (r"^WSS局部信息筛选矩阵5.*\|X5X11_f(\d)_s(\d+)｜", "X5 + S3 截面 token 上下文在 cv3 fold{m1}，seed {m2}；与同折 X5_f{m1} 配对（同一参考链）；指标为折外，不是 test34", "C1", "三折 Δ −0.004/−0.009/+0.018，合并 +0.000/−0.003，逐例 54 胜 84 负；high-WSS R² 三折升、IoU 三折略降；不进主线", f"{DOC_V5}｜§21.5"),
    # ---- wave 5b (WSS sheet): T3 cascade + cv3 seed 7 ----
    (r"^WSS局部信息筛选矩阵5b.*\|T3_s(\d+)｜", "T3 热点级联：X5 输入再加一阶段（cv3 折外 X5）预测 log_wss_base，回归目标改为残差 ln(WSS)−log_wss_base，训练损失聚焦一阶段预测的逐例 top20% 区（区外权重 0.2），评估只在该区叠加残差、区外保留一阶段预测，seed {m1}；与同 seed 的 X5 配对", "C1", "三 seed 配对 Δ物理 +0.004/−0.000/+0.014（均值 +0.006，带内）、归一化 −0.008（3/3 负）；high-WSS R² +0.02～+0.04；不进主线", f"{DOC_V5}｜§21.7"),
    (r"^WSS局部信息筛选矩阵5b.*\|T3n_s(\d+)｜", "残差堆叠对照：同 T3 的输入与残差目标，但不聚焦、不门控（全场学残差），seed {m1}；与同 seed 的 X5 配对", "C1", "三 seed 配对 Δ物理 −0.003/+0.019/+0.027（均值 +0.014，符号不一致）、归一化 −0.004（3/3 负）、逐例负多于胜；high-WSS R² +0.01～+0.10、top10 比最高 0.767；尾部候选，不进主线", f"{DOC_V5}｜§21.7"),
    (r"^WSS局部信息筛选矩阵5b.*\|X5_f(\d)_s7｜", "X5 配方在 cv3 fold{m1}，seed 7（train = 其余两折 91–93 例，指标为折外，不是 test34）", "C1", "折外 0.693/0.666/0.678，合并 138 例 0.6786/0.8472（seed 1234 为 0.6802）", f"{DOC_V5}｜§21.5"),
    # ---- v5.1 wave 1 (2026-09-16): corrected labels (26 RCR reruns), 2 duplicates removed, train136/test34 ----
    (r"^WSS_v5\.1重训波1.*\|X5D_v51_s(\d+)｜", "X5D 配方（C1 + Murray 分流先验 + 密度增广）在 v5.1 数据上重训：26 例出口面积修正重算、剔除 2 例重复，train136/test34，特征标准化在 train136 重算，seed {m1}；C1_s{m1} 配对初始化", "C1", "新 test34 五 seed 0.7461/0.7571/0.7657/0.7510/0.7542（均值 0.7548）；五 seed 集成 Pa 均值 0.7749 / log 均值 0.7718，逐例均值/p10 0.791/0.680，high-WSS R² 0.387，AG/AAA/ILO 0.818/0.739/0.747；v5.0 同 seed 0.706–0.741，只反映标签修正效应（4 例 test 标签已变）", f"{DOC_V5}｜§28"),
    (r"^WSS_v5\.1重训波1.*\|X5Dcap_s(\d+)｜", "协议版 capfit：X5D 的两列 Murray 输入换成按出口盖面半径的 Murray 份额与 log_tau0（sidecar v1.4，照抄 CFD 出口协议），seed {m1}；与同 seed X5D_v51 配对", "C1", "三 seed 配对 Δ物理 +0.015/−0.009/+0.003（均值 +0.003，带内）、Δ归一化 +0.007/+0.006/+0.003（3/3 正，均值 +0.005）；MAE −3.7%/−3.8%/−1.3%，IoU +0.015，AAA 三 seed 全正；三 seed 集成 0.7770 vs 0.7736（+0.0035）；小而一致的改善，候选", f"{DOC_V5}｜§28"),
    (r"^WSS_v5\.1重训波1.*\|X5Ddual_s(\d+)｜", "双尺度 query patch：K16 偏移按局部中位间距归一 + 固定 4 mm 球内 16 个覆盖采样点（mask），零初始化第二分支，seed {m1}；与同 seed X5D_v51 配对", "C1", "全密度 test34 三 seed Δ物理 +0.004/−0.013/−0.011（均值 −0.007）、Δ归一化 0.000；两个 seed 尾部变差；三 seed 集成 −0.0075。抽稀复评：冻结特征层损失 50/25/10% 从 −0.017/−0.060/−0.146 缩到 −0.004/−0.023/−0.056（三 seed 同向），重算变体 10% −0.183→−0.143；STL 重采样 0.5/0.8/1.2 mm 净差 −0.002/+0.005/−0.005（被全密度代价抵消）→ 不进底座，只作点云粗于 1.5 mm 的备选", f"{DOC_V5}｜§28"),
    (r"^WSS_v5\.1重训波1.*\|X5Dnoise_s(\d+)｜", "边界噪声增广：每例每轮 p=0.5 换成沿法向相关噪声（σ 0.1/0.2/0.3 mm，corr 2 mm）的点云并重算法向/曲率族，密度增广仍 p=0.6，seed {m1}；与同 seed X5D_v51 配对", "C1", "全密度 test34 三 seed Δ物理 +0.002/−0.004/−0.012（均值 −0.005）、Δ归一化 −0.002/−0.002/−0.007（3/3 负）、逐例胜 14/14/7；全密度小代价。几何噪声复评：边界噪声 0.2/0.4 mm 的损失 −0.066/−0.138 → −0.051/−0.095，但干净面低 0.006、加 1 mm 平滑后只差 +0.002 → 不进默认底座，部署以重采样+平滑为主防线", f"{DOC_V5}｜§28"),
    (r"^WSS_v5\.1波2a三折.*\|X5D_v51_f(\d)_s1234｜", "X5D_v51 配方在 cv3_v51 第 {m1} 折上训练（train = 其余两折 90–92 例，折训统计与 z-score），指标为留出折折外，不是 test34；同时是 T3n 级联的阶段一", "C1", "折外 0.7366/0.6701/0.7323，合并 136 例 0.7100/0.8544（v5.0 数据 X5 为 0.680）；ILO 折外 0.624 最弱", f"{DOC_V5}｜§28.6"),
    (r"^WSS_v5\.1波2b级联.*\|T3n_s(\d+)｜", "T3n 残差堆叠（v5.1）：X5D_v51 再加一列阶段一预测 log_wss_base（cv3_v51 折外/折模型轮转）作输入并改学残差，不聚焦不门控，seed {m1}；与同 seed X5D_v51 配对", "C1", "test34 三 seed Δ物理 −0.006/−0.009/−0.009（均值 −0.008，3/3 负）、归一化 −0.002、逐例胜 17/12/13；三 seed 集成 0.7662 vs 底座 0.7736（−0.007）；top10 比略升但整体降 → 不进主线", f"{DOC_V5}｜§28.7"),
    (r"^WSS_v5\.1波2b级联.*\|T3n_f(\d)_s1234｜", "T3n 残差堆叠在 cv3_v51 第 {m1} 折（折训 offset 统计，嵌套），与同折 X5D_v51_f{m1} 配对；指标为留出折折外", "C1", "三折 Δ −0.027/−0.011/−0.024，合并 136 例 0.6900 vs 底座 0.7100（Δ −0.020，逐例仅 30/136 胜）；top10 比 +0.045 但 R² 一致变差 → 折外证据否定 T3n", f"{DOC_V5}｜§28.7"),
    # ---- longitudinal geometry channels onto the X5D baseline (2026-09-17) ----
    (r"^WSS沿程几何接入.*\|X5D_long_s(\d+)｜", "X5D_v51 输入追加 §26.1 导出的 32 维沿程几何通道（参考面与点云各 8 个 value + 8 个 mask：log 面积、面积坡度、圆度、偏心率、上下游最窄面积比与距离），共 59 维；统计量只用 train136 的 valid 点，缺失值标准化为 0 并由 mask 通道告知；27→59 配对零初始化，seed {m1}；与同 seed X5D_v51 配对", "C1", "test34 三 seed Δ物理 +0.024/−0.001/−0.009（均值 +0.005，sd 0.014，仅 1/3 正）、Δ归一化 +0.007/+0.006/−0.002（均值 +0.003，2/3 正）；top10 IoU 3/3 升（+0.015）、MAE 2/3 降、high-WSS 集成 +0.024；三 seed 集成 0.7789 vs 同 seed 数底座 0.7736（+0.005）；与 capfit 同级 → 候选，不改部署底座", f"{DOC_V5}｜§29"),
    (r"^WSS沿程几何剪枝.*\|L(\d)_s1234｜", "沿程通道剪枝阶梯：在 X5D_v51 上只追加 {m1} 组参考面沿程 value/mask 通道（L8=ref 全 8 个、L5=前向选出 5 个、L2=圆度+上游最窄距离、L1=只留截面圆度），通道排序依据是 train136 折外残差的边际增量，选择不碰 test34；seed 1234", "C1", "归一化增益从 2 列到 32 列基本持平（L1 +0.0060 / L8 +0.0063 / 全 32 列 +0.0065，差在 ±0.004 噪声带内）；L1 是唯一在几何缺失区不变差的臂（同桶 ln-MSE C +1.34% / D +1.86%），全场 +3.74% 优于全 32 列的 +3.07%，覆盖率 0.788 最高 → 最终保留集 = geom_ref_roundness 两列", f"{DOC_V5}｜§29.12"),
    # ---- local morphology wave A: query-patch receptive field ladder (2026-09-17) ----
    (r"^WSS局部形态感受野.*\|W0(\d)(K32)?_s1234｜", "局部形态 Wave A：在 K16 query patch 之外增加固定 {m1} mm 球的覆盖采样（零初始化第二分支，其余 X5D_v51 逐位不变）；依据 Phase 0——K16 半径仅 2.06 mm＝8.3% 圆周，而区间内残差的 m=1..3 周向模态净份额 0.589、误差相干长度约 5 mm；seed 1234", "C1", "四臂全部不进主线。精度列不可排名（物理 Δ +0.0009/+0.0006/+0.0134/+0.0099 非单调且全在单 seed 带 ±0.014 内，W06 与 W06K32 只差采样点数却差 0.0093）；机制验证决定性——m=1..3 净份额 底座 +0.5706 → 各臂 +0.5625~+0.5728，±0.008 内等于没动。失败机制＝球分支的 masked mean 池化抹平角度排列，够得着≠分得清 → 收窄到 B2 扇区 token", f"{DOC_LM}"),
    (r"^WSS局部形态扇区token.*\|(S6R2|S4|S6|S8)_s1234｜", "局部形态 Wave B（B2）：球不变（6 mm × 32 = W06K32），只把 masked mean 换成周向扇区 × 距离环 token，token 间一层 4 头自注意力后汇总，零初始化接入；{m1}；seed 1234", "C1", "作业 15046 完训完评。vs W06K32：S4/S6/S8/S6R2 物理 Δ +0.0074/+0.0000/+0.0158/+0.0006，归一化全在 +0.0027~+0.0033。精度列不可排名（非单调、S8 刚过矩阵 ±0.014）；主判据 m=1..3 净份额尚未算 → 不宣布胜出、不进底座", f"{DOC_LM}"),
    (r"^WSS偏心与全周期时间阶段2.*\|T0_f(\d)_s1234｜", "T0 直接相位查询：同折 X5D_v51 + 4 相位列（q_norm/dq_norm/t_sin/t_cos），random_frame + 峰值保底 1/9，frame_stats，EMA(train_loss, α0.2) 选模；只评 cv3_v51 fold{m1} 留出折 81 帧，不用 test34", "C1", "作业 15071。三折均值峰值/全周期/谷底/TAWSS 0.6967/0.5166/0.3218/0.7164；过 G2.1/G2.4，G2.3 均值过但 fold2 峰值掉 0.033 → 不过部署门，不补 T0-800", f"{DOC_V5}｜§30.3"),
    (r"^WSS偏心与全周期时间阶段2.*\|TB8_f(\d)_s1234｜", "时间基头 K*=8（out_dim=9），无时间输入，系数 MSE；fold{m1} 只评留出折 81 帧", "C1", "作业 15071。均值峰值/全周期 0.6094/0.4497，低于 T-null 0.4613，峰值相对 X5D −0.104；G2.1/G2.2/G2.3/G2.5 FAIL → No-Go，不补 seed、不开 2b", f"{DOC_V5}｜§30.3"),
    (r"^WSS偏心与全周期时间阶段2.*\|TB16_f(\d)_s1234｜", "时间基头 2K*=16（out_dim=17），无时间输入，系数 MSE；fold{m1} 只评留出折 81 帧", "C1", "作业 15071。均值峰值/全周期 0.6040/0.4490，相对 TB8 无增益；同样 No-Go", f"{DOC_V5}｜§30.3"),
    (r"^WSS局部信息筛选矩阵5b.*\|X5X11_f(\d)_s7｜", "X5 + S3 截面 token 上下文在 cv3 fold{m1}，seed 7；与同折 X5_f{m1}_s7 配对；指标为折外", "C1", "三折 Δ +0.004/+0.017/+0.005，合并 +0.010/−0.001，逐例 62 胜 76 负；两个 seed 合读六折尾部一致更好、IoU 六折全降；不进主线", f"{DOC_V5}｜§21.5"),
    # ---- wave 6 (WSS sheet): density augmentation ----
    (r"^WSS局部信息筛选矩阵6.*\|X5D_s(\d+)｜", "X5D：X5 配方 + 训练期密度增广（每例每轮 p=0.6 换成 70/50/35/25% 体素抽稀云，法向/曲率族/tn_dot 在抽稀云上重算；输入与结构逐位不变），seed {m1}；与同 seed 的 X5 配对", "C1", "全密度 test34 三 seed 配对 Δ物理 −0.012/−0.001/+0.021（均值 +0.003，带内）、归一化 +0.004（3/3 正）；抽稀 50/25/10% 损失 −0.022/−0.068/−0.171（X5 为 −0.061/−0.183/−0.397）；STL 1.2 mm −0.047（X5 −0.147）；定为部署底座候选", f"{DOC_V5}｜§22.2–22.3"),
    # ---- wave 2 (WSS sheet) ----
    (r"^WSS局部信息筛选矩阵2.*\|X5X11_s(\d+)｜", "X5X11：C1 + Murray 分支流量先验（X5）+ 中心线截面 token 上下文（X11），seed {m1}；与同 seed 的 X5、X0 配对", "C1", "三 seed 0.7247/0.7262/0.7274：比同 seed 的 X5 高 +0.005～+0.014（带内），归一化持平；高值区 R²/top10/p99 比一致上升", f"{DOC_V5}｜§20.4"),
    (r"^WSS局部信息筛选矩阵2.*\|X5e233", "X5 的 Murray 分流指数 3→2.33（只用 log Q 份额）", "C1", "低于 X5q：指数 3 更好", f"{DOC_V5}｜§20.4"),
    (r"^WSS局部信息筛选矩阵2.*\|X5cap", "X5 的末支半径改用几何程序的虚拟盖半径（只用 log Q 份额）", "C1", "低于 X5q：atlas 半径更好", f"{DOC_V5}｜§20.4"),
    (r"^WSS局部信息筛选矩阵2.*\|X5res", "X5 输入不变，回归目标改为残差 ln(WSS) − log τ0（评估换回标准 log_z）", "C1", "与 X5 持平；AG 升、ILO 降", f"{DOC_V5}｜§20.4"),
    # ---- wave 2 (volume sheet) ----
    (r"^波2 F6接体场.*\|PF6_s1234", "压力 PF6：P02（SA3 后全局 BT）+ Murray 分支流量先验 F6（内部单元按所在血管段广播 log Q 份额，log τ0 用单元处 atlas 半径）", "R5", "比同期对照 P02r +0.035，MAE −11%，29/34 例变好", f"{DOC_V5}｜§20.4"),
    (r"^波2 F6接体场.*\|P02r_s1234", "P02 同配置同期重跑（对照，量化同配置抖动）", "R5", "与历史 P02 差 +0.0006", f"{DOC_V5}｜§20.4"),
    (r"^波2 F6接体场.*\|VF6_s1234", "速度 VF6：V07（新 SA3 单半径 0.20 + BT）+ Murray 分支流量先验 F6", "R5", "比同期对照 V07r +0.023，高速区 R² 0.17→0.32，24/34 例变好", f"{DOC_V5}｜§20.4"),
    (r"^波2 F6接体场.*\|V07r_s1234", "V07 同配置同期重跑（对照）", "R5", "与历史 V07 差 −0.004", f"{DOC_V5}｜§20.4"),
    # ---- wave 3 (volume sheet, three-seed confirmation) ----
    (r"^波3 F6接体场三seed.*\|PF6_s(\d+)", "压力 PF6：P02 + Murray 分支流量先验 F6，seed {m1}；与同 seed、同继承初始权重的 P02_s{m1} 配对", "R5", "三 seed 配对 Δ +0.035/+0.062/+0.039（均值 +0.045），MAE 每个 seed 降 8–12%；晋级为压力底座", f"{DOC_V5}｜§20.5"),
    (r"^波3 F6接体场三seed.*\|VF6_s(\d+)", "速度 VF6：V07 + Murray 分支流量先验 F6，seed {m1}；与同 seed 的 V07_s{m1} 配对", "R5", "三 seed 配对 Δ +0.023/+0.026/+0.033（均值 +0.027），高速区 R² 0.17→0.29–0.32，向量 RMSE −2.6～−5.0%；晋级为速度底座", f"{DOC_V5}｜§20.5"),
    (r"^波3 F6接体场三seed.*\|P02_s(\d+)", "P02 同 seed 对照（seed {m1}，从 seed 配对参考链从零训练）", "R5", "配对对照", f"{DOC_V5}｜§20.5"),
    (r"^波3 F6接体场三seed.*\|V07_s(\d+)", "V07 同 seed 对照（seed {m1}）", "R5", "配对对照", f"{DOC_V5}｜§20.5"),
    # ---- volume sheet ----
    (r"^V5 体场目标.*\|R5P-all", "R5P：R4 同配置改为回归相对压力（壁面 + 内部单元混合 query），全部 query 评估", "R5", "0.707", f"{DOC_V5}｜§7"),
    (r"^V5 体场目标.*\|R5P-interior", "R5P 只看内部单元的压力指标", "R5", "0.707", f"{DOC_V5}｜§7"),
    (r"^V5 体场目标.*\|R5P-wall", "R5P 只看壁面点的压力指标", "R5", "0.702", f"{DOC_V5}｜§7"),
    (r"^V5 体场目标.*\|R5V", "R5V：R4 同配置改为回归内部单元速度三分量，按速度幅值评", "R5", "0.748", f"{DOC_V5}｜§7"),
    (r"^体场全局注意力.*\|R5P\|", "历史 R5P（压力）checkpoint 重评，作为参照", "R5", "0.707", f"{DOC_V5}｜§17"),
    (r"^体场全局注意力.*\|R5V\|", "历史 R5V（速度）checkpoint 重评，作为参照", "R5", "0.748", f"{DOC_V5}｜§17"),
    (r"^体场全局注意力.*\|([PV])00", "{kind}00：R5 配置同期从零重训（对照）", "R5", "", f"{DOC_V5}｜§17"),
    (r"^体场全局注意力.*\|([PV])01", "{kind}01：R5 + L（M2 的多尺度局部壁面分支）", "R5", "", f"{DOC_V5}｜§17"),
    (r"^体场全局注意力.*\|P02", "压力 P02：R5 的 SA3 之后直接加 G（32 token 的全局瓶颈 Transformer）；压力线最终选择", "R5", "压力线最终选择", f"{DOC_V5}｜§17"),
    (r"^体场全局注意力.*\|V02", "速度 V02：R5 的 SA3 之后直接加 G（32 token 的全局瓶颈 Transformer）", "R5", "", f"{DOC_V5}｜§17"),
    (r"^体场全局注意力.*\|([PV])03", "{kind}03：R5 + L + G", "R5", "", f"{DOC_V5}｜§17"),
    (r"^体场全局注意力.*\|([PV])04", "{kind}04：R5 + L + C（只有 token 条件编码，无注意力）", "R5", "", f"{DOC_V5}｜§17"),
    (r"^体场全局注意力.*\|([PV])05", "{kind}05：R5 + L + C + F（条件编码 + 近等参数逐 token FFN，无跨 token 交互）", "R5", "", f"{DOC_V5}｜§17"),
    (r"^体场全局注意力.*\|([PV])06", "{kind}06：R5 + L + 新 SA3 单半径 0.20 分组，无 BT", "R5", "", f"{DOC_V5}｜§17"),
    (r"^体场全局注意力.*\|P07", "压力 P07：新 SA3 单半径 0.20 + BT", "R5", "", f"{DOC_V5}｜§17"),
    (r"^体场全局注意力.*\|V07", "速度 V07：新 SA3 单半径 0.20 + BT；速度线候选", "R5", "速度线候选", f"{DOC_V5}｜§17"),
    (r"^体场全局注意力.*\|([PV])08", "{kind}08：新 SA3 三半径 0.10/0.20/0.40 分组，无 BT", "R5", "", f"{DOC_V5}｜§17"),
    (r"^体场全局注意力.*\|([PV])09", "{kind}09：三半径各配独立 BT（导师完整方案）", "R5", "", f"{DOC_V5}｜§17"),
    (r"^体场全局注意力.*\|([PV])10", "{kind}10：三支都用 0.20 半径 + 独立 BT", "R5", "", f"{DOC_V5}｜§17"),
    (r"^体场全局注意力.*\|([PV])11", "{kind}11：三半径 + 近等参数逐 token FFN 替代 BT", "R5", "", f"{DOC_V5}｜§17"),
    (r"^体场全局注意力.*\|J00", "J00：共享骨干/FP/QAD、压力与速度各一个头的联合双任务模型", "R5", "未双赢", f"{DOC_V5}｜§17"),
    (r"^体场全局注意力.*\|J01", "J01：J00 + 全局 BT", "R5", "", f"{DOC_V5}｜§17"),
]


TOKEN_EXPLAIN = {
    "ball16": "半径球，每个 center 最多 16 个邻居", "adaptive": "自适应覆盖分组，先全覆盖再补第二归属",
    "random5000": "每 epoch 随机 5000 顶点", "random": "每 epoch 随机顶点", "fixed-FPS": "固定的最远点采样子集",
    "FPS-multistart5000": "多起点最远点采样池", "FPS-multistart": "多起点最远点采样池", "Q1V": "SAME", "Q2V": "SEP",
}


def _seed_of(text: str) -> str:
    m = re.search(r"[·_ (]s(\d{1,5})\b", text) or re.search(r"seed(\d{1,5})", text)
    return m.group(1) if m else "1234"


def gloss_for(group: str, label: str, run_id: str, struct: str) -> tuple[str, str, str, str] | None:
    """Return (description, base_key, key results, source) for one row."""
    key = f"{group}|{label}|{run_id}"
    for pattern, desc, base, note, source in GLOSS:
        m = re.search(pattern, key)
        if not m:
            continue
        fill = {f"m{i}": g for i, g in enumerate(m.groups(), 1)}
        fill.update({f"e{i}": TOKEN_EXPLAIN.get(g, g) for i, g in enumerate(m.groups(), 1)})
        seed = "" if "均值" in label else _seed_of(label + " " + run_id)
        fill.update(label=label, seed=seed, seedtxt=(f"，seed {seed}" if seed else ""), struct=struct or "",
                    kind={"P": "压力 P", "V": "速度 V"}.get(fill.get("m1", ""), fill.get("m1", "")))
        try:
            text = desc.format(**fill)
        except (KeyError, IndexError):
            text = desc
        return text, base, note, source
    return None


def build_description(group: str, label: str, run_id: str, struct: str) -> str | None:
    hit = gloss_for(group, label, run_id, struct)
    if hit is None:
        return None
    text, base, _, _ = hit
    if base and BASE.get(base) and base not in text:
        text = f"{text}（底座 {base}，见「方法说明对照」）"
    if "均值" in label and "seed" in label:
        text += "；本行为多 seed 均值"
    if "｜last" in label:
        text += "；本行为 last checkpoint"
    return text


# ----------------------------------------------------------------------------------------------
# workbook operations
# ----------------------------------------------------------------------------------------------
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def annotate(book: Path, dry_run: bool = False, listing: Path | None = None) -> dict:
    wb = load_workbook(book)
    sheets = {"WSS实验矩阵": (2, 165), "速度与压力实验矩阵": (2, 132)}   # (label column, run-id column)
    struct_col = {"WSS实验矩阵": 162, "速度与压力实验矩阵": 129}
    changed_cells = set()
    glossary_rows: dict[tuple[str, str], list] = {}
    unmatched = []
    for name, (label_col, id_col) in sheets.items():
        ws = wb[name]
        for r in range(6, ws.max_row + 1):
            group = ws.cell(r, 1).value
            label = ws.cell(r, label_col).value
            if not group or label is None:
                continue
            label = str(label)
            run_id = str(ws.cell(r, id_col).value or "")
            struct = str(ws.cell(r, struct_col[name]).value or "")
            base_label = label.split(SUFFIX)[0]
            hit = gloss_for(str(group), base_label, run_id, struct)
            if hit is None:
                unmatched.append((name, r, str(group), base_label[:60]))
                continue
            description = build_description(str(group), base_label, run_id, struct)
            key = (str(group), re.sub(r"[·_ (]s\d{1,5}\b|·s\d+|\(3 seed 均值\)|（\d seed 均值）|｜last|｜3seed均值", "", base_label).strip())
            if key not in glossary_rows:
                r2 = ws.cell(r, 6).value
                note = hit[2]
                if isinstance(r2, (int, float)):
                    note = f"{note}；表内物理 R²_cb={r2:.4f}" if note else f"表内物理 R²_cb={r2:.4f}"
                glossary_rows[key] = [name, str(group), key[1], hit[0], BASE.get(hit[1], hit[1]), struct[:300], note, hit[3]]
            new_value = f"{base_label}{SUFFIX}{description}"
            if label != new_value:
                changed_cells.add((name, ws.cell(r, label_col).coordinate))
                if not dry_run:
                    ws.cell(r, label_col).value = new_value
    # ---- glossary sheet ----
    if SHEET in wb.sheetnames:
        del wb[SHEET]
    gs = wb.create_sheet(SHEET)
    title_font = Font(bold=True, size=13)
    head_font = Font(bold=True)
    head_fill = PatternFill("solid", fgColor="DDEBF7")
    wrap = Alignment(wrap_text=True, vertical="top")
    gs["A1"] = "方法说明对照  |  代号 → 人话（自动从跟踪文档提取，2026-09-12）"
    gs["A1"].font = title_font
    gs["A2"] = ("读法：先看下面的术语表，再查每个实验代号做了什么。两张结果表的「实验 / 模型」列已在原标签后追加「｜说明：…」。"
                "结果要点只摘主指标（物理 R²_cb，best），完整数字以结果表为准。")
    gs["A2"].alignment = wrap
    row = 4
    gs.cell(row, 1, "一、术语表").font = head_font
    row += 1
    for c, h in enumerate(("术语 / 缩写", "含义"), 1):
        cell = gs.cell(row, c, h); cell.font = head_font; cell.fill = head_fill
    row += 1
    for term, meaning in VOCAB:
        gs.cell(row, 1, term).alignment = wrap
        gs.cell(row, 2, meaning).alignment = wrap
        row += 1
    row += 1
    gs.cell(row, 1, "二、底座模型（其它臂都是在这些底座上改一处）").font = head_font
    row += 1
    for c, h in enumerate(("底座代号", "是什么"), 1):
        cell = gs.cell(row, c, h); cell.font = head_font; cell.fill = head_fill
    row += 1
    for key, text in BASE.items():
        gs.cell(row, 1, key).alignment = wrap
        gs.cell(row, 2, text).alignment = wrap
        row += 1
    row += 1
    gs.cell(row, 1, "三、逐实验说明（顺序与结果表一致；同一臂的多 seed / last 行合并为一条）").font = head_font
    row += 1
    headers = ("结果表", "实验矩阵（列 A）", "实验 / 代号", "具体做了什么（人话）", "底座", "结构 / 关键设置（结果表原文）", "结果要点", "文档出处")
    for c, h in enumerate(headers, 1):
        cell = gs.cell(row, c, h); cell.font = head_font; cell.fill = head_fill; cell.alignment = wrap
    row += 1
    for values in glossary_rows.values():
        for c, v in enumerate(values, 1):
            gs.cell(row, c, v).alignment = wrap
        row += 1
    widths = (14, 30, 34, 70, 40, 44, 26, 44)
    for c, w in enumerate(widths, 1):
        gs.column_dimensions[get_column_letter(c)].width = w
    gs.column_dimensions["B"].width = max(gs.column_dimensions["B"].width, 60)
    gs.freeze_panes = "A4"
    summary = {"changed_label_cells": len(changed_cells), "glossary_entries": len(glossary_rows),
               "vocabulary": len(VOCAB), "unmatched": unmatched}
    if listing is not None:
        listing.write_text("\n".join(f"[{v[1]}] {v[2]}\n    → {v[3]}\n    结果：{v[6]}" for v in glossary_rows.values()))
    if dry_run:
        return summary
    # ---- save with full verification of untouched cells ----
    original = load_workbook(book)
    preserved = {}
    for s in original:
        if s.title == SHEET:
            continue
        for r in s.iter_rows():
            for c in r:
                if (s.title, c.coordinate) not in changed_cells:
                    preserved[(s.title, c.coordinate)] = cell_record(c)
    merges = {s.title: tuple(map(str, s.merged_cells.ranges)) for s in original if s.title != SHEET}
    fd, temp = tempfile.mkstemp(dir=book.parent, prefix=".methods_", suffix=".xlsx")
    os.close(fd)
    try:
        wb.save(temp)
        checked = load_workbook(temp)
        for (sheet, coordinate), record in preserved.items():
            if cell_record(checked[sheet][coordinate]) != record:
                raise AssertionError(f"untouched cell changed: {sheet}!{coordinate}")
        assert {s.title: tuple(map(str, s.merged_cells.ranges)) for s in checked if s.title != SHEET} == merges
        assert SHEET in checked.sheetnames
        checked.close()
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        backup = BACKUP_DIR / f"{book.stem}_backup_before_methods_{time.strftime('%Y%m%d_%H%M%S')}.xlsx"
        backup.write_bytes(book.read_bytes())
        before = sha(book)
        os.replace(temp, book)
    finally:
        if Path(temp).exists():
            Path(temp).unlink()
    summary.update(backup=str(backup), before_sha256=before, after_sha256=sha(book), verified_cells=len(preserved))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", type=Path, default=BOOK)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--listing", type=Path, default=None, help="write the code→description listing here")
    args = parser.parse_args(argv)
    summary = annotate(args.workbook, dry_run=args.dry_run, listing=args.listing)
    unmatched = summary.pop("unmatched")
    print(summary)
    if unmatched:
        print(f"UNMATCHED rows ({len(unmatched)}):")
        for item in unmatched:
            print("  ", item)


if __name__ == "__main__":
    main()
