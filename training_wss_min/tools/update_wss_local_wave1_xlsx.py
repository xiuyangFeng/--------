"""Idempotent workbook backfill for the wave-1 local-information screening matrix.

Owns exactly the 17 rows whose column A equals GROUP on sheet ``WSS实验矩阵`` (168 columns,
table ``WSSResults``); every historical cell (value, style, comment, hyperlink), merged range
and dimension of all three sheets is verified unchanged after saving.  Arms without a
complete queue record and metrics file keep the placeholder ``—`` in every result cell.

    python -m training_wss_min.tools.update_wss_local_wave1_xlsx [--workbook PATH]
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

from openpyxl import load_workbook
from openpyxl.comments import Comment

from training_wss_min import config as C
from training_wss_min.tools.report_wss_recovery import METRIC_COLUMNS, cell_record, get_metric, metric_values

ROOT = C.PROJECT_ROOT
NAME = "wss_local_wave1_20260912"
CONFIGS = ROOT / "training_wss_min/configs" / NAME
EXP = ROOT / "training_wss_min/experiments" / NAME
RUNS = ROOT / "training_wss_min/runs"
BOOK = ROOT / "docs/03-汇报材料/WSS_PointNet实验矩阵与结果汇总last.xlsx"
GROUP = "WSS局部信息筛选矩阵｜2026-09-12（17臂）"
EXTRA_EXPERIMENTS: list[str] = []
MISSING = "—"
DESCRIPTIONS = {
    "X0": "C1同配置同期重跑（同期对照）", "X1": "权重EMA 0.999（另评ckpt_ema）",
    "X2": "+5维弯曲参考角（RMF输运曲率方向、上游2D/5D滞后、挠率）",
    "X3": "+4维分叉参考角（嵴侧/髋部、分叉平面、分叉角、兄弟支半径比）",
    "X4": "+5维上游历史（s/D、上游半径极值比、上游最大κR）",
    "X5": "+2维Murray分支流量先验（log Q份额、log τ0）", "X6": "+1维train138人群先验（训练例留一）",
    "X7": "query patch偏移改到query切平面坐标架", "X8": "patch注意力池化（零初始化logit）",
    "X9": "X7＋X8＋K16→32", "X10": "切平面WSS方向辅助头（1−cos，λ0.1）",
    "X11": "中心线截面token上下文（4mm bin，2层Transformer，零初始化残差入FiLM）",
    "X12": "混合专家头（4个head拷贝＋输入特征门控）",
    "X13a": "全帧预训练（random_frame＋相位特征＋全帧统计）", "X13b": "从X13a热启动只训峰值帧150轮",
    "X15": "F1＋F2＋F3＋F6共16维一起追加", "X16": "X9＋X15＋X10组合",
    "X5X11_s1234": "X5 Murray分支流量先验＋X11截面token上下文（seed 1234；与同seed的X5/X0配对）",
    "X5X11_s7": "X5 Murray分支流量先验＋X11截面token上下文（seed 7）",
    "X5X11_s2025": "X5 Murray分支流量先验＋X11截面token上下文（seed 2025）",
    "X5e233_s1234": "F6变体：Murray分流指数3→2.33（只用log Q份额）",
    "X5cap_s1234": "F6变体：末支半径改用几何程序虚拟盖半径（只用log Q份额）",
    "X5res_s1234": "F6变体：X5输入不变，回归目标改为残差ln(WSS)−log τ0（评估换回标准log_z）",
    "X5A_s1234": "切口盖面拟合分流（capfit）替换Murray份额，seed 1234（与同seed X5配对）",
    "X5A_s7": "capfit分流替换Murray，seed 7", "X5A_s2025": "capfit分流替换Murray，seed 2025",
    "X5A_s11": "capfit分流替换Murray，seed 11", "X5A_s2026": "capfit分流替换Murray，seed 2026",
    "X5_s11": "X5 Murray先验再补seed 11（5 seed集成成员）", "X5_s2026": "X5 Murray先验再补seed 2026（5 seed集成成员）",
    "X5B_s1234": "Murray份额＋capfit份额四列一起，seed 1234",
    # wave 5 (2026-09-13): stenosis index + cv3
    "X5I_s1234": "X5＋狭窄指数ln(R_local/R_distal)一列（sidecar v1.3），seed 1234（与同seed X5配对）",
    "X5I_s7": "X5＋狭窄指数，seed 7", "X5I_s2025": "X5＋狭窄指数，seed 2025",
    "X5AI_s1234": "X5A（capfit分流）＋狭窄指数一列，seed 1234（与同seed X5A配对）",
    "X5AI_s7": "X5A＋狭窄指数，seed 7", "X5AI_s2025": "X5A＋狭窄指数，seed 2025",
    "X5_f0_s1234": "X5 在 train138 患者分组3-fold 的 fold0（指标为折外45例，非test34）",
    "X5_f1_s1234": "X5 cv3 fold1（折外47例，非test34）", "X5_f2_s1234": "X5 cv3 fold2（折外46例，非test34）",
    "X5X11_f0_s1234": "X5＋X11截面token上下文 cv3 fold0（折外45例，与X5_f0配对）",
    "X5X11_f1_s1234": "X5＋X11 cv3 fold1（折外47例，与X5_f1配对）", "X5X11_f2_s1234": "X5＋X11 cv3 fold2（折外46例，与X5_f2配对）",
    "X5_f0_s7": "X5 cv3 fold0，seed 7（折外45例）", "X5_f1_s7": "X5 cv3 fold1，seed 7（折外47例）", "X5_f2_s7": "X5 cv3 fold2，seed 7（折外46例）",
    "X5X11_f0_s7": "X5＋X11 cv3 fold0，seed 7（与X5_f0_s7配对）", "X5X11_f1_s7": "X5＋X11 cv3 fold1，seed 7", "X5X11_f2_s7": "X5＋X11 cv3 fold2，seed 7",
    "T3_s1234": "T3热点级联：X5输入＋一阶段折外预测log_wss_base，残差目标，预测top20%区域聚焦损失＋评估门控，seed 1234（与同seed X5配对）",
    "T3_s7": "T3热点级联，seed 7", "T3_s2025": "T3热点级联，seed 2025",
    "T3n_s1234": "残差堆叠对照：同T3输入与残差目标，不聚焦不门控，seed 1234", "T3n_s7": "残差堆叠对照，seed 7", "T3n_s2025": "残差堆叠对照，seed 2025",
    # wave 6 (2026-09-15): density augmentation
    "X5D_s1234": "X5＋训练期密度增广（每例每轮 p=0.6 换成 70/50/35/25% 体素抽稀云，法向/曲率族重算；输入与结构不变），seed 1234（与同seed X5配对）",
    "X5D_s7": "X5＋密度增广，seed 7", "X5D_s2025": "X5＋密度增广，seed 2025",
    # v5.1 wave 1 (2026-09-16): retrain on corrected/deduplicated data + three follow-up arms
    **{f"X5D_v51_s{s}": f"X5D 配方在 v5.1 数据上重训（26 例 RCR 出口面积修正重算、剔除 2 例重复，train136/test34；特征标准化在 train136 重算），seed {s}（对照 = v5.0 数据同 seed X5D，只反映标签变化）" for s in (1234, 7, 2025, 11, 2026)},
    **{f"X5Dcap_s{s}": f"协议版 capfit：X5D 的两列 Murray 输入换成按出口盖面半径的 Murray 份额与 log_tau0（sidecar v1.4，照抄 CFD 出口协议），seed {s}（与同 seed X5D_v51 配对）" for s in (1234, 7, 2025)},
    **{f"X5Ddual_s{s}": f"双尺度 query patch：K16 偏移按局部中位间距归一 + 固定 4 mm 球内 16 个覆盖采样点（mask），零初始化第二分支，seed {s}（与同 seed X5D_v51 配对）" for s in (1234, 7, 2025)},
    **{f"X5Dnoise_s{s}": f"边界噪声增广：每例每轮 p=0.5 换成沿法向相关噪声（σ 0.1/0.2/0.3 mm，corr 2 mm）的点云并重算法向/曲率族，密度增广仍 p=0.6，seed {s}（与同 seed X5D_v51 配对）" for s in (1234, 7, 2025)},
    # learning curve (2026-09-20): X5D_v51 recipe on nested 25/50/75/100 % subsets of each cv3_v51 fold, 3 seeds
    **{f"wss_learning_curve_20260920:LC{p}_f{k}_s{sd}": (f"Learning curve：X5D_v51 配方只用 cv3_v51 fold{k} 训练例的 {p} %（嵌套分层子集，子集自算 log_z 统计与特征 z-score），seed {sd}；指标为留出折折外（非 test34），三折合并后与同 seed 其他分数配对读斜率" if p < 100 else f"Learning curve 100 % 档：X5D_v51 配方用 cv3_v51 fold{k} 全部训练例，seed {sd}（seed 1234 的 100 % 档 = 波 2a 折模型，不重训）；指标为留出折折外（非 test34）")
      for p in (25, 50, 75, 100) for k in range(3) for sd in (1234, 7, 2025) if not (p == 100 and sd == 1234)},
    # v5.1 wave 2a/2b (2026-09-16): cv3_v51 fold models and T3n out-of-fold check
    **{f"X5D_v51_f{k}_s1234": f"X5D_v51 在 cv3_v51 第 {k} 折上训练（折训统计与 z-score），只读留出折；作 T3n 的阶段一" for k in range(3)},
    "wss_v51_wave2b_20260916:T3n_s1234": "T3n 残差堆叠（X5D_v51 + 阶段一折外预测 log_wss_base 输入，残差目标，不聚焦不门控），seed 1234（与同 seed X5D_v51 配对）",
    "wss_v51_wave2b_20260916:T3n_s7": "T3n 残差堆叠（v5.1），seed 7", "wss_v51_wave2b_20260916:T3n_s2025": "T3n 残差堆叠（v5.1），seed 2025",
    # 沿程几何 32 通道接入 X5D 底座（2026-09-17）
    "wss_x5d_longitudinal_20260917:X5D_long_s1234": "X5D_v51 输入追加 32 维沿程几何 value/mask 通道（参考面/点云各 8 个：log面积、面积坡度、圆度、偏心率、上下游最窄面积比与距离；共 59 维），统计量只用 train136 valid 点，27→59 配对零初始化，seed 1234（与同 seed X5D_v51 配对）",
    "wss_x5d_longitudinal_20260917:X5D_long_s7": "X5D_v51＋32 维沿程几何通道，seed 7",
    "wss_x5d_longitudinal_20260917:X5D_long_s2025": "X5D_v51＋32 维沿程几何通道，seed 2025",
    # 沿程通道剪枝阶梯（单 seed 1234，2026-09-17）；通道排序依据=train136 折外残差的边际增量
    "wss_x5d_long_prune_20260917:L8_s1234": "剪枝 L8：只留参考面 8 个沿程 value+mask（去掉与 ref 相关 0.93–1.00 的全部点云列），43 维；折外解释力 0.00435＝全 32 列天花板的 92%，逐点覆盖率 0.651→0.747",
    "wss_x5d_long_prune_20260917:L5_s1234": "剪枝 L5：前向选出的 5 个 ref 通道（坡度/圆度/上游面积比/下游面积比/上游距离），37 维；折外 0.00404＝85%",
    "wss_x5d_long_prune_20260917:L2_s1234": "剪枝 L2：只留圆度＋上游最窄距离，31 维；折外 0.00291＝62%",
    "wss_x5d_long_prune_20260917:L1_s1234": "剪枝 L1：只留截面圆度一路（geom_ref_roundness＋mask），29 维；折外 0.00250＝53%（单通道第一），覆盖率 0.788 全阶梯最高＝最终保留集",
    # 局部形态 Wave A：query patch 感受野阶梯（单 seed 1234，2026-09-17）
    "wss_local_morph_radius_20260917:W04_s1234": "感受野 W04：K16 最近邻之外再加固定 4 mm 球的 16 点覆盖采样（零初始化第二分支，offset_norm 保持 mm），圆周覆盖约 9%",
    "wss_local_morph_radius_20260917:W06_s1234": "感受野 W06：固定 6 mm 球 × 16 点，圆周覆盖约 14%（D7 测得误差相干长度约 5 mm）",
    "wss_local_morph_radius_20260917:W08_s1234": "感受野 W08：固定 8 mm 球 × 16 点，圆周覆盖约 18%",
    "wss_local_morph_radius_k_20260917:W06K32_s1234": "感受野 W06K32：固定 6 mm 球 × 32 点——与 W06 只差采样点数，用于分离\u201c够得着\u201d与\u201c采得密\u201d",
    # 局部形态 Wave B：B2 扇区 token（单 seed 1234，2026-09-18，作业 15046）
    "wss_local_morph_sectors_20260918:S4_s1234": "扇区 S4：球不变（6 mm × 32 点 = W06K32），只把 masked mean 换成 4 扇形 × 1 环 = 4 token，token 间一层自注意力后汇总，零初始化接入",
    "wss_local_morph_sectors_20260918:S6_s1234": "扇区 S6：同上，6 扇形 × 1 环 = 6 token",
    "wss_local_morph_sectors_20260918:S8_s1234": "扇区 S8：同上，8 扇形 × 1 环 = 8 token",
    "wss_local_morph_sectors_20260918:S6R2_s1234": "扇区 S6R2：同上，6 扇形 × 2 环 = 12 token（径向再分一档）",
    **{f"T3n_f{k}_s1234": f"T3n 残差堆叠在 cv3_v51 第 {k} 折上（折训 offset 统计，嵌套），与 X5D_v51_f{k} 配对，只读留出折" for k in range(3)},
    # 偏心与全周期时间阶段 2（cv3 三折 × seed 1234，作业 15071）
    **{f"wss_time_ecc_20260918:T0_f{k}_s1234": f"T0 直接相位查询：X5D_v51 fold{k} + 4 相位列（q_norm/dq_norm/t_sin/t_cos），random_frame + 峰值保底 1/9，frame_stats，EMA(train_loss, α0.2) 选模；只评留出折 81 帧，不用 test34" for k in range(3)},
    **{f"wss_time_ecc_20260918:TB8_f{k}_s1234": f"时间基头 K*=8（out_dim=9），无时间输入，系数 MSE；fold{k} 只评留出折 81 帧" for k in range(3)},
    **{f"wss_time_ecc_20260918:TB16_f{k}_s1234": f"时间基头 2K*=16（out_dim=17），无时间输入，系数 MSE；fold{k} 只评留出折 81 帧" for k in range(3)},
    # 周期积分量 TAWSS / OSI 直接回归阶段 1（cv3 三折 × seed 1234，作业 15335；标签 wss_min_cycle_v1 帧 0–79）
    **{f"wss_cycle_20260920:A1_f{k}_s1234": f"A1 TAWSS 直接回归：X5D_v51 fold{k} 配方只换标签为周期平均 |τ|（log_z，训练折统计）；表内 Pa 列 = TAWSS(Pa)；只读留出折" for k in range(3)},
    **{f"wss_cycle_20260920:O1_f{k}_s1234": f"O1 OSI 直接回归（线性 z，评估裁 [0,0.5]）：X5D_v51 fold{k} 配方只换标签；表内「Pa」列实为 OSI 无量纲；只读留出折" for k in range(3)},
    **{f"wss_cycle_20260920:O2_f{k}_s1234": f"O2 OSI 直接回归（logit_z 目标空间，反变换 0.5·sigmoid）：与 O1 只差目标空间；表内「Pa」列实为 OSI 无量纲；只读留出折" for k in range(3)},
    # 周期积分量阶段 3 确认（train136 → test34 读一次，seed 1234/7/2025，作业 15394）与 M1 三头（cv3 作业 15452；三 seed 作业 15476）
    **{f"wss_cycle_stage3_20260921:A1_s{sd}": f"A1 TAWSS 直接回归：已部署 X5D_v51_s{sd} 配方只换标签为周期平均 |τ|（log_z，train136 统计）；test34 读一次" for sd in (1234, 7, 2025)},
    **{f"wss_cycle_stage3_20260921:O1_s{sd}": f"O1 OSI 直接回归（线性 z，裁 [0,0.5]）：已部署 X5D_v51_s{sd} 配方只换标签；test34 读一次" for sd in (1234, 7, 2025)},
    **{f"wss_cycle_stage3_20260921:O2_s{sd}": f"O2 OSI 直接回归（logit_z）：已部署 X5D_v51_s{sd} 配方只换标签；test34 读一次" for sd in (1234, 7, 2025)},
    **{f"wss_cycle_m1_20260921:M1_f{k}_s1234": f"M1 三头：X5D_v51 fold{k} 配方 out_dim 3 = [峰值 WSS log_z, TAWSS log_z, OSI logit_z]，等权 MSE，pinball 关，输出层配对重建；只读留出折" for k in range(3)},
    **{f"wss_cycle_m1_stage3_20260922:M1_s{sd}": f"M1 三头三 seed 确认：已部署 X5D_v51_s{sd} 配方 out_dim 3 = [峰值 WSS log_z, TAWSS log_z, OSI logit_z]（train136 多通道统计），等权 MSE，pinball 关；test34 读一次" for sd in (1234, 7, 2025)},
    # v5.2p4 全量 265 例训练（2026-09-30，队列 16192）：X5Dcap_asym2 配方只换数据，test = recover8
    **{f"wss_v52p4_full265_20260930:X5Dcap_asym2_full265_s{sd}": f"全量 265 例 X5Dcap_asym2 seed {sd}：配方 = wss_v52_phys_20260926/X5Dcap_asym2_s{sd} 逐位不变，只换数据（v5.2 261 + YANG_BAO_KUI + 3 个同病人搭档单元；val 空、按训练损失选模）；test = recover8（8 例，病人不在训练集）" for sd in (1234, 7, 2025)},
}
FULL265 = "wss_v52p4_full265"  # 2026-09-30: rows of the full265 data-version training (protocol / note / paired IND reference)


def describe(aid):
    """2026-09-16: experiment-scoped override first (same arm id reused across waves), then the plain id."""
    return DESCRIPTIONS.get(f"{NAME}:{aid}") or DESCRIPTIONS[aid]


def external_refs_for(arm, matrix):
    """Paired references declared in matrix.json (wave 2): same-seed X5 for X5X11, X5q/X5 for the F6 variants."""
    if NAME.startswith(FULL265):
        # the X5Dcap_asym2 IND (train170) run of the same seed, evaluated on recover8 like the queue evaluated this arm
        # (frozen copy, GPU, best checkpoint, full265 view root; experiments/<name>/ref_ind_recover8.slurm)
        seed = arm.get("seed", 1234)
        ref = EXP / "ref_ind_recover8" / f"X5Dcap_asym2_s{seed}" / "metrics.json"
        return [(f"IND同seed{seed}（train170，同一recover8）", str(ref))] if ref.is_file() else []
    runs = {key.split(" (")[0]: value for key, value in matrix.get("external_reference_runs", {}).items()}
    for other in matrix.get("arms", []):  # 2026-09-16: same-experiment references (e.g. X5Dcap vs X5D_v51 of the same seed)
        runs.setdefault(other["id"], str(ROOT / "training_wss_min/runs" / NAME / other["id"] / "eval/ckpt_best/metrics.json"))
    aid, seed = arm["id"], arm.get("seed", 1234)
    if aid.startswith("X5X11"):
        wanted = [(f"X5同seed{seed}", f"X5_s{seed}"), ("X11", "X11_s1234")]
    elif aid.startswith(("X5e233", "X5cap")):
        wanted = [("X5q", "X5q_s1234"), ("X5", "X5_s1234")]
    elif aid.startswith("X5res"):
        wanted = [("X5", "X5_s1234")]
    elif aid.startswith("X5AI"):
        wanted = [(f"X5A同seed{seed}", f"X5A_s{seed}"), (f"X5同seed{seed}", f"X5_s{seed}")]
    elif aid.startswith(("X5A", "X5B")):
        wanted = [(f"X5同seed{seed}", f"X5_s{seed}")]
    elif aid.startswith("X5D_v51_f") or aid.startswith("T3n_f"):
        wanted = []  # fold arms: paired within the experiment (parent), no external run
    elif aid.startswith("X5D_v51"):
        wanted = [(f"v5.0数据X5D同seed{seed}", f"X5D_s{seed}")]
    elif aid.startswith(("X5Dcap", "X5Ddual", "X5Dnoise")) or (aid.startswith("T3n_") and "v51" in NAME):
        wanted = [(f"X5D_v51同seed{seed}", f"X5D_v51_s{seed}")]
    elif aid.startswith(("X5I", "X5D")) or aid.startswith(("T3_", "T3n_")):
        wanted = [(f"X5同seed{seed}", f"X5_s{seed}")]
    elif aid.startswith("X5X11_f"):
        wanted = [(f"X5同折同seed", f"X5_f{arm.get('fold')}_s{seed}")]
    elif NAME.startswith("wss_local_morph_sectors") or aid.startswith(("S4_", "S6_", "S8_", "S6R2_")):
        wanted = [("W06K32同球无扇区", "W06K32_s1234"), ("X5D_v51同seed1234", "X5D_v51_s1234")]
    elif NAME.startswith("wss_time_ecc") or aid.startswith(("T0_f", "TB8_f", "TB16_f")):
        fold = arm.get("fold")
        wanted = [(f"X5D_v51同折峰值底座", f"X5D_v51_f{fold}_s1234")]
        if aid.startswith(("TB8_f", "TB16_f")):
            wanted.append((f"T0同折", f"T0_f{fold}_s1234"))
    elif NAME.startswith("wss_cycle") or aid.startswith(("A1_f", "O1_f", "O2_f")):
        # 周期线：第一参照 = 同折峰值底座（G1.1 用归一化 Δ；对 OSI 臂只是跨目标并列）；O2 另配 O1 同折（目标空间对照）
        fold = arm.get("fold")
        wanted = [(f"X5D_v51同折峰值底座", f"X5D_v51_f{fold}_s1234")]
        if aid.startswith("O2_f"):
            wanted.append((f"O1同折（线性目标空间）", f"O1_f{fold}_s1234"))
    else:
        wanted = []
    return [(name, runs[key]) for name, key in wanted if key in runs and Path(runs[key]).is_file()]
STATUS_TEXT = {"complete": "已完成（队列退出码0；best/last评估齐全）", "failed": "失败", "blocked": "阻塞",
               "cancelled": "取消", "waiting": "等待依赖", "pending": "等待执行"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else default


def arm_records():
    matrix = read_json(CONFIGS / "matrix.json")
    queue = read_json(EXP / "queue_status.json", {"arms": {}})
    # 2026-09-29: a matrix may have been executed by several queues (e.g. wss_v52_20260923 IND in 15618, its CV5
    # arms in experiments/wss_v52_20260923_cv5 by 15636); records missing/incomplete here are taken from EXTRA_EXPERIMENTS.
    for extra in EXTRA_EXPERIMENTS:
        other = read_json(ROOT / "training_wss_min/experiments" / extra / "queue_status.json", {"arms": {}})["arms"]
        for aid, rec in other.items():
            if queue["arms"].get(aid, {}).get("status") != "complete" and rec.get("status") == "complete":
                queue["arms"][aid] = rec
    for arm in matrix["arms"]:
        DESCRIPTIONS.setdefault(arm["id"], arm["title"])
    anchor = read_json(Path(matrix["anchor_run"]) / "eval/ckpt_best/metrics.json")["test"]
    items = []
    for arm in matrix["arms"]:
        aid = arm["id"]
        if arm.get("base", "wss") != "wss":      # volume arms live on the other sheet
            continue
        config = read_json(CONFIGS / arm["config"])
        record = queue["arms"].get(aid, {})
        run = RUNS / config["name"]
        metrics = {}
        for checkpoint in arm["evaluate"]:
            stage = record.get("stages", {}).get(f"eval_{checkpoint}", {})
            path = run / f"eval/ckpt_{checkpoint}/metrics.json"
            if stage.get("returncode") == 0 and path.is_file():
                metrics[checkpoint] = {"path": str(path), "metrics": read_json(path)["test"]}
        accepted = record.get("status") == "complete" and all(c in metrics for c in arm["evaluate"])
        history = [json.loads(l) for l in (run / "history.jsonl").read_text().splitlines() if l.strip()] \
            if (run / "history.jsonl").is_file() else []
        items.append({**arm, "config_path": str(CONFIGS / arm["config"]), "run_name": config["name"],
                      "run_dir": str(run), "epochs": config["train"]["epochs"],
                      "input_dim": len(config["data"]["input_features"]),
                      "queue_status": record.get("status", "pending"),
                      "status": STATUS_TEXT.get(record.get("status", "pending"), record.get("status", "pending")),
                      "accepted": accepted, "metrics": metrics,
                      "diagnostics": read_json(run / "training_diagnostics.json", {}),
                      "elapsed": history[-1].get("elapsed_s") if history else None,
                      "history_rows": len(history)})
    return matrix, queue, anchor, items


def update_workbook(book: Path):
    matrix, queue, anchor, arms = arm_records()
    before = sha(book)
    wb = load_workbook(book)
    # 2026-09-29: the workbook gained sheets (TAWSS_OSI周期量矩阵 at index 2, 方法说明对照); require presence, not position
    if not {"WSS实验矩阵", "速度与压力实验矩阵", "指标说明"} <= set(wb.sheetnames):
        raise ValueError("unexpected workbook sheet schema")
    ws = wb["WSS实验矩阵"]
    if ws.max_column != 168 or ws["FI5"].value != "原始实验 ID" or "WSSResults" not in ws.tables:
        raise ValueError("unexpected workbook 168-column schema")
    owned = [r for r in range(6, ws.max_row + 1) if ws.cell(r, 1).value == GROUP]
    if owned and len(owned) != len(arms):
        raise ValueError(f"owned rows must number exactly {len(arms)}")
    locations = ({ws.cell(r, 165).value: r for r in owned} if owned else
                 {a["run_name"]: ws.max_row + i + 1 for i, a in enumerate(arms)})
    if set(locations) != {a["run_name"] for a in arms}:
        raise ValueError("run identities differ from the workbook's owned rows")
    preserved = {(s.title, c.coordinate): cell_record(c) for s in wb for row in s for c in row
                 if not (s.title == ws.title and c.row in owned)}
    dimensions = {s.title: (copy.deepcopy(s.column_dimensions), copy.deepcopy(s.row_dimensions)) for s in wb}
    merges = {s.title: tuple(map(str, s.merged_cells.ranges)) for s in wb}
    control = next((a for a in arms if a["id"] == "X0"), None)
    control_best = control["metrics"]["best"]["metrics"] if control and control["accepted"] else None
    pair_paths = ["field_casebalanced.r2", "normalized.field_casebalanced.r2", "field.mae",
                  "hotspot.spearman_all_casemean", "hotspot.top10_iou_casemean"]
    for i, arm in enumerate(arms):
        row = locations[arm["run_name"]]
        values = [MISSING] * 168
        is_cycle = NAME.startswith("wss_cycle")
        is_osi = is_cycle and arm["id"].startswith("O")
        protocol = ("v5.2p4 full265：train 265 / val 0 → test recover8（8 例，病人不在训练集）；峰值1162；n=8 描述性读数"
                    if NAME.startswith(FULL265) else
                    "V5.1 cv3 留出折 81 帧；test34 未用；表内 R²_cb=峰值帧 1162，周期指标见备注"
                    if NAME.startswith("wss_time_ecc") else
                    ("V5.1 cv3 留出折；test34 未用；目标 = 周期积分量（wss_min_cycle_v1，帧 0–79）：表内物理列 = "
                     + ("OSI∈[0,0.5]（无量纲，非 Pa）" if is_osi else "TAWSS(Pa)") + "；阈值掩膜与自由基线见备注")
                    if is_cycle else
                    "V5 train138/test34；峰值1162；test34开发筛选")
        values[:5] = [GROUP, f"{arm['id']}｜{describe(arm['id'])}｜{arm['status']}",
                      ("OSI · 无量纲" if is_osi else "TAWSS · Pa") if is_cycle else "WSS · Pa",
                      "best主结果；last/ema见逐指标批注" if arm["accepted"] else "尚无完整结果",
                      protocol]
        values[13] = "X0（同期对照）；C1（历史锚点）" if (arm["id"] != "X0" and control_best is not None) else "C1（历史锚点）"
        if NAME.startswith("wss_time_ecc"):
            note = (f"Job {queue.get('job_id', '尚未提交')}；{arm['status']}；history {arm['history_rows']}/{arm['epochs']}。"
                    f"cv3_v51 fold{arm.get('fold')} 留出折 81 帧，test34 未用；表内物理/归一化 R²_cb 是峰值帧 1162，不是全周期。"
                    f"单seed{arm.get('seed', 1234)}，只作阶段 2 筛选；{arm['title']}。")
        elif is_cycle:
            note = (f"Job {queue.get('job_id', '尚未提交')}；{arm['status']}；history {arm['history_rows']}/{arm['epochs']}。"
                    f"cv3_v51 fold{arm.get('fold')} 留出折，test34 未用；目标 = {'OSI（无量纲，表内「Pa」列实为 OSI）' if is_osi else 'TAWSS（Pa）'}，"
                    f"标签 wss_min_cycle_v1（wall_wss_vec 帧 0–79 各权 1/80）。单seed{arm.get('seed', 1234)}，只作阶段 1 筛选；{arm['title']}。")
        elif NAME.startswith(FULL265):
            note = (f"Job {queue.get('job_id', '尚未提交')}；{arm['status']}；history {arm['history_rows']}/{arm['epochs']}。"
                    f"全量训练 train 265（v5.2 261 + YANG_BAO_KUI + 3 个同病人搭档单元），val 空（按训练损失选模），test = recover8；"
                    f"test91 已在训练集内，recover8 是唯一留出集，仅 8 例只作描述。单seed{arm.get('seed', 1234)}；"
                    f"配方 = wss_v52_phys_20260926/X5Dcap_asym2_s{arm.get('seed', 1234)} 逐位不变（只改数据路径），init_reference_config=C1_s{arm.get('seed', 1234)}。"
                    f"ΔR²列相对 IND 同 seed（train170）在同一 recover8 上的读数（同冻结代码、GPU、best）。"
                    f"三 seed 集成读数（CPU，evaluate_recover8.sh）：full265 0.8393 对 IND 三 seed 0.8352 / 五 seed 0.8393，见 outputs/cfd_auto_trial_20260927/_recover/eval/readout_recover8_full265_vs_ind.md；{arm['title']}。")
        else:
            note = (f"Job {queue.get('job_id', '尚未提交')}；{arm['status']}；history {arm['history_rows']}/{arm['epochs']}。"
                    f"单seed{arm.get('seed', 1234)}、已暴露test34：只作筛选，不作显著性或泛化结论；单seed对单seed 95%带约±0.034 Pa R²/±0.009归一化。"
                    f"底座C1（E2＋E3），init_reference_config=C1（继承张量初始权重逐位相同）；{arm['title']}。")
        sources = [arm["config_path"], str(Path(arm["run_dir"]) / "history.jsonl"), str(EXP / "results.json")]
        other = {}
        if arm["accepted"]:
            best = arm["metrics"]["best"]["metrics"]
            last = arm["metrics"]["last"]["metrics"]
            for col, value in metric_values(best).items():
                if value is not None:
                    values[col - 1] = value
            values[11] = f"{int(best['aggregate']['r2_negative_cases'])}/{int(best['aggregate']['n_cases'])}"
            values[60] = arm["diagnostics"].get("parameter_count", MISSING)
            values[112] = get_metric(last, "field_casebalanced.r2") - get_metric(best, "field_casebalanced.r2")
            values[139] = 1
            values[147] = arm["diagnostics"].get("parameter_count", MISSING)
            values[148] = arm["diagnostics"].get("elapsed_seconds", arm["elapsed"])
            refs = []
            if control_best is not None and arm["id"] != "X0":
                refs.append(("X0", control_best, arm and control["metrics"]["best"]["path"]))
            for ref_name, ref_path in external_refs_for(arm, matrix):
                refs.append((ref_name, read_json(ref_path)["test"], ref_path))
            refs.append(("C1历史", anchor, str(Path(matrix["anchor_run"]) / "eval/ckpt_best/metrics.json")))
            for j, (ref, ref_metrics, ref_path) in enumerate(refs[:4]):
                base = 64 + j * 6
                values[base] = f"{ref}；{ref_path}"
                for k, dotted in enumerate(pair_paths, 1):
                    own, ours = get_metric(best, dotted), get_metric(ref_metrics, dotted)
                    if own is not None and ours is not None:
                        values[base + k] = own - ours
                if ref == "X0":
                    values[12] = get_metric(best, "field_casebalanced.r2") - get_metric(ref_metrics, "field_casebalanced.r2")
            if control_best is None or arm["id"] == "X0":
                values[12] = get_metric(best, "field_casebalanced.r2") - get_metric(refs[0][1], "field_casebalanced.r2")
                values[13] = "；".join(r[0] for r in refs) + "（ΔR²列相对第一个参照）"
            values[88] = "；".join(r[0] for r in refs) + "；best同checkpoint配对；单seed，无跨seed统计"
            other = {ck: metric_values(m["metrics"]) for ck, m in arm["metrics"].items() if ck != "best"}
            sources += [m["path"] for m in arm["metrics"].values()]
            if NAME.startswith("wss_time_ecc"):
                cyc = best.get("cycle") or {}
                tawss = (best.get("tawss") or {}).get("field_casebalanced") or {}
                note += (f" 周期 best：cycle_R²_cb={cyc.get('cycle_r2cb_pa')} trough={cyc.get('trough_r2cb_pa')} "
                         f"TAWSS_R²_cb={tawss.get('r2')} peak_time_err_med={cyc.get('peak_time_err_med_frames')}。"
                         f"ΔR²列相对同折 X5D_v51 峰值底座（G2.3）；TB 第二参照为同折 T0。")
            elif is_cycle:
                masks = best.get("threshold_masks") or {}
                mask_text = "；".join(f"{name}: IoU={blk.get('iou_casemean'):.3f} 面积份额误差med={blk.get('frac_abs_err_casemed'):.3f}"
                                     for name, blk in masks.items())
                c1 = (read_json(EXP / "offline/c1_baselines.json") or {}).get(f"fold{arm.get('fold')}", {})
                if is_osi:
                    on = c1.get("OSI_null", {})
                    base_text = (f"OSI-null（训练折等渗 OSI~ln τ̂_peak）同折：R²_cb={on.get('r2_cb')} "
                                 f">0.1 IoU={(on.get('threshold_masks') or {}).get('above_0.1', {}).get('iou')}")
                else:
                    tn, tr = c1.get("TAWSS_null", {}), c1.get("TAWSS_ratio", {})
                    base_text = (f"TAWSS-null（峰值折外预测×波形，80 帧）同折 Pa R²_cb={tn.get('r2_cb')} log={tn.get('log_r2_cb')}；"
                                 f"TAWSS-ratio Pa={tr.get('r2_cb')}")
                note += (f" 阈值掩膜 best：{mask_text}。自由基线 {base_text}。"
                         f"ΔR²列相对同折 X5D_v51 峰值底座（跨目标并列，G1.1 用归一化 Δ）；门控读数见 experiments/wss_cycle_20260920/offline/gate_report_best.txt。")
        values[159:] = ["OSI（无量纲）" if is_osi else "Pa", "PointNeXt-R＋LocalGeoPE＋L-SA2＋local branch＋E2方向邻域＋E3 patch/FiLM（C1）",
                        describe(arm["id"]),
                        f"seed{arm.get('seed', 1234)} / {arm['epochs']}ep / {arm['input_dim']}D / support5000 / query5000 independent",
                        5000, arm["run_name"], note[:32000], "\n".join(sources),
                        GROUP]
        for col, value in enumerate(values, 1):
            cell = ws.cell(row, col)
            cell._style = copy.copy(ws.cell(310 if i % 2 else 311, col)._style)
            cell.value = value
            cell.comment = None
            cell.hyperlink = None
            if arm["accepted"] and col in METRIC_COLUMNS and other:
                text = "；".join(f"{ck}={vals[col] if vals[col] is not None else MISSING}" for ck, vals in other.items())
                cell.comment = Comment(f"本单元格=best；同口径 {text}\n路径={METRIC_COLUMNS[col]}", "wave1 report")
        ws.cell(row, 166).comment = Comment(note[:32000], "wave1 report")
        ws.cell(row, 167).comment = Comment("\n".join(sources), "wave1 report")
        ws.row_dimensions[row].height = 72
    ws.tables["WSSResults"].ref = f"A5:FL{ws.max_row}"
    ws.tables["WSSResults"].autoFilter.ref = f"A5:FL{ws.max_row}"
    ws.print_area = f"A1:N{ws.max_row}"
    fd, temp = tempfile.mkstemp(dir=book.parent, prefix=".wss_local_wave1_", suffix=".xlsx")
    os.close(fd)
    try:
        wb.save(temp)
        checked = load_workbook(temp)
        for (sheet, coordinate), record in preserved.items():
            if cell_record(checked[sheet][coordinate]) != record:
                raise AssertionError(f"historical cell changed: {sheet}!{coordinate}")
        assert {s.title: tuple(map(str, s.merged_cells.ranges)) for s in checked} == merges
        for sheet in checked:
            old_cols, old_rows = dimensions[sheet.title]
            assert {k: dict(v) for k, v in sheet.column_dimensions.items()} == {k: dict(v) for k, v in old_cols.items()}
            for key, value in old_rows.items():
                if sheet.title != ws.title or key not in owned:
                    assert dict(sheet.row_dimensions[key]) == dict(value)
        for arm in arms:
            if not arm["accepted"]:
                for column in [*range(6, 14), *range(15, 160)]:
                    assert checked[ws.title].cell(locations[arm["run_name"]], column).value == MISSING
        checked.close()
        if sha(book) != before:
            raise RuntimeError("workbook changed concurrently; refusing replacement")
        backup = book.with_name(book.stem + f"_backup_before_wave1_{time.strftime('%Y%m%d_%H%M%S')}.xlsx")
        backup.write_bytes(book.read_bytes())
        os.replace(temp, book)
    finally:
        if Path(temp).exists():
            Path(temp).unlink()
    evidence = {"passed": True, "before_sha256": before, "after_sha256": sha(book), "backup": str(backup),
                "rows": locations, "records": len(arms), "accepted_result_rows": sum(a["accepted"] for a in arms),
                "historical_cells_verified": len(preserved), "table_ref": ws.tables["WSSResults"].ref}
    (EXP / "xlsx_acceptance.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2))
    return evidence


def main(argv=None):
    global NAME, CONFIGS, EXP, GROUP, EXTRA_EXPERIMENTS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", type=Path, default=BOOK)
    parser.add_argument("--name", default=NAME, help="experiment name (configs/<name>, experiments/<name>)")
    parser.add_argument("--group", default=GROUP, help="workbook column-A group label that owns the rows")
    parser.add_argument("--extra-experiment", action="append", default=[], help="experiments/<name> whose complete queue records fill arms not completed in the primary experiment")
    args = parser.parse_args(argv)
    NAME, GROUP = args.name, args.group
    EXTRA_EXPERIMENTS = list(args.extra_experiment)
    CONFIGS = ROOT / "training_wss_min/configs" / NAME
    EXP = ROOT / "training_wss_min/experiments" / NAME
    print(json.dumps(update_workbook(args.workbook), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
