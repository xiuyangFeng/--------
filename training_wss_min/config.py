#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""training_wss_min 实验配置（配置驱动 baseline）。

一个实验 = 一个 JSON 配置。用 `ExpConfig.from_json` 读、`to_json` 写；
训练/评估只读取已登记的 config 路径，不在运行时生成或改写源配置。

约定：所有相对路径都相对仓库根 `PROJECT_ROOT`；产物落在 `runs/<name>/`。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Tuple

try:  # keep ``python training_wss_min/config.py`` smoke compatibility
    from . import longitudinal_geometry as LG
except ImportError:  # pragma: no cover - direct script invocation only
    import longitudinal_geometry as LG  # type: ignore

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data_wss_min"
BUNDLE_GLOB = "AG/*/*/bundle.npz"          # data_wss_min 下 cohort/case
GLOBAL_STATS = DATA_ROOT / "wss_global_stats.json"
DEFAULT_SPLIT = PROJECT_ROOT / "training" / "splits" / "split_AG_wss_min_v1.json"
RUNS_ROOT = Path(__file__).resolve().parent / "runs"


# 可用的逐点输入特征（壁面 dist_to_wall 恒为 0，已从可选列表移除）
# cohort_ag/cohort_aaa/cohort_ilo 为逐病例 cohort one-hot（同一病例内恒定），
# 供通用模型按疾病域条件化；不参与几何 z-score 统计，直接以 0/1 入网络。
COHORT_FEATURE_KEYS = ("cohort_ag", "cohort_aaa", "cohort_ilo")
RCR_OUTLETS = ("outle", "outli", "outri", "outre")
RCR_PARAMETERS = ("r1", "r2", "c")
CASE_FEATURE_KEYS = tuple(
    f"log_rcr_{outlet}_{parameter}"
    for outlet in RCR_OUTLETS
    for parameter in RCR_PARAMETERS
) + ("inlet_velocity_nominal_m_s",)
# V5 视图（data_wss_v5/views/wss_min_view_v1）额外逐点几何特征：管道坐标 rho/theta、半径梯度、
# 到分叉/端点距离、端区标记、对齐坐标架下的 PCA 法向；旧 bundle 不含这些键时 load_case 不会附加。
V5_POINT_FEATURE_KEYS = (
    "rho", "theta_sin", "theta_cos", "dr_ds", "dist_to_junction_mm", "dist_to_endpoint_mm",
    "end_zone", "nx_aligned", "ny_aligned", "nz_aligned",
    "dist_to_wall_mm",  # volume targets: 0 on wall points, cell-to-wall distance for interior points
    # geometry-only scale channels used by the optional E5/E9 data arms
    "radius_over_coord_scale", "log_radius_over_coord_scale",
)
# 时间/相位特征（timesteps='random_frame' 时可用）：来自 172 例共享的协议入口波形，逐点广播，不做 z-score（本身有界）
TIME_FEATURE_KEYS = ("q_norm", "dq_norm", "t_sin", "t_cos", "t_norm")

# V6 单帧矩阵（configs/v6_singleframe_20260909）的局部微分几何与分支语义特征。
# 曲面主曲率/形状指数来自壁面点云 + PCA 法向的局部二次拟合（sidecar，data.point_features_root
# 指向 data_wss_v5/views/wss_min_geom_v2）；tn_dot 是中心线切向与壁面法向的夹角余弦。
# 分支语义 one-hot 与分支内弧长由既有 bundle 的 wall_semantic_id/wall_segment_id/wall_s_local_mm 派生。
# 全部只依赖部署可得的输入（壁面点云 + 中心线 atlas），不含任何 CFD 网格/解量。
V6_SURFACE_FEATURE_KEYS = (
    "curv_k1", "curv_k2", "curv_mean", "curv_gauss", "curvedness", "shape_index",
    "curv_k1_c", "curv_k2_c", "curv_mean_c", "curv_gauss_c", "curvedness_c", "shape_index_c",
    "tn_dot",
)
V6_SEMANTIC_FEATURE_KEYS = (
    "sem_trunk", "sem_lcia", "sem_rcia", "sem_lext", "sem_lint", "sem_rext", "sem_rint",
)
V6_BRANCH_FEATURE_KEYS = ("s_local_mm", "s_local_frac")
V6_POINT_FEATURE_KEYS = V6_SURFACE_FEATURE_KEYS + V6_SEMANTIC_FEATURE_KEYS + V6_BRANCH_FEATURE_KEYS
# 波 1 局部信息矩阵（configs/wss_local_wave1_20260912）的流动参考几何特征，来自
# wss_v5/views/wall_flowref_v1.py 的 sidecar（data.point_features_root 可给多个根目录）：
# F1 弯曲参考角（RMF 输运的曲率方向 + 上游 2D/5D 滞后）、F2 分叉参考角（嵴侧/髋部、分叉平面、分叉角、
# 兄弟支半径比）、F3 上游历史（s/D、上游窗口极值半径比、上游最大 κR）、F6 Murray 分支流量先验、
# P5 train138 人群先验（训练例留一）。除 atlas_prior_logwss 外全部只依赖点云 + 中心线。
V7_FLOW_FEATURE_KEYS = (
    "bend_cos", "bend_signed", "torsion_rr", "bend_cos_up2", "bend_cos_up5",
    "carina_cos", "bif_plane_cos", "bif_angle_cos", "sibling_radius_ratio",
    "s_over_d", "up_min_r_ratio_2d", "up_min_r_ratio_5d", "up_max_r_ratio_5d", "up_kappa_5d",
    "log_q_branch_murray", "log_tau0_murray", "atlas_prior_logwss",
    "log_q_branch_murray233", "log_q_branch_murray_cap",  # sidecar v1.1 variants: exponent 2.33 / virtual-cap radii
    "log_q_branch_capfit", "log_tau0_capfit",  # sidecar v1.2: train138-fitted cap-area split rule (deployable)
    "log_r_over_rdistal",  # sidecar v1.3: ln(R_local / R_distal(segment)), the stenosis term hidden inside Murray's log_tau0
    "log_tau0_murray_cap",  # sidecar v1.4 (2026-09-16): log_q_branch_murray_cap - 3 ln R, i.e. the CFD outlet protocol (Murray on cap radii)
)
# 波 5 T3 级联：一阶段（部署可得）模型的 ln(WSS_pred) 作为二阶段的输入与残差偏移，来自 wss_min_cascade_v1 sidecar
# （train138 为患者分组 3-fold 的折外预测，test34 为折模型预测；见 tools/make_cascade_sidecar.py）
V8_CASCADE_FEATURE_KEYS = ("log_wss_base",)
# 2026-09-26 一维物理先验 sidecar（wss_v5.views.wall_phys1d_v1）：协议常量（峰值入口流量、Carreau 高剪切黏度）+ 开口半径版
# Murray 分流 + 点的 atlas 半径 → 泊肃叶壁面剪切 log τ₁D（Pa）；wom = 同一 1D 管在协议波形下的 Womersley 脉动解在峰值步的壁面剪切
# （随半径的相位/幅值修正）；可作输入或残差目标（data.target_log_offset_feature）。
V9_PHYS1D_FEATURE_KEYS = ("log_tau_1d_pois", "log_tau_1d_wom", "log_wom_alpha", "log_re_1d")
# 需要从 sidecar 读取的逐点特征（按 data.point_features_root 的顺序查找）
LONGITUDINAL_FEATURE_KEYS = LG.FEATURE_KEYS
SIDECAR_FEATURE_KEYS = V6_SURFACE_FEATURE_KEYS + V7_FLOW_FEATURE_KEYS + V8_CASCADE_FEATURE_KEYS + LONGITUDINAL_FEATURE_KEYS + V9_PHYS1D_FEATURE_KEYS
# 体场目标（support=壁面，query=内部单元）可用的 sidecar 键：分支流量份额按 vol_segment_id 广播到内部单元，
# log_tau0 = log_q - 3 log(内部单元的 atlas 半径)。其余 sidecar 键是壁面专属，不能用于体场目标。
VOLUME_EXTENDABLE_SIDECAR_KEYS = (
    "log_q_branch_murray", "log_q_branch_murray233", "log_q_branch_murray_cap", "log_tau0_murray",
    "log_q_branch_capfit", "log_tau0_capfit",
)
# 与 curvature 同口径处理的重尾曲率族：signed_log1p + p99 截断
CURVATURE_LIKE_KEYS = (
    "curvature", "curv_k1", "curv_k2", "curv_mean", "curv_gauss", "curvedness",
    "curv_k1_c", "curv_k2_c", "curv_mean_c", "curv_gauss_c", "curvedness_c",
    "bend_signed", "torsion_rr", "up_kappa_5d",
)

FEATURE_KEYS = (
    "x", "y", "z", "abscissa_norm",
    "local_radius", "log_local_radius", "curvature", "coord_scale",
    "radius_gradient",
    *V5_POINT_FEATURE_KEYS,
    *V6_POINT_FEATURE_KEYS,
    *V7_FLOW_FEATURE_KEYS,
    *V8_CASCADE_FEATURE_KEYS,
    *LONGITUDINAL_FEATURE_KEYS,
    *V9_PHYS1D_FEATURE_KEYS,
    *TIME_FEATURE_KEYS,
    *COHORT_FEATURE_KEYS,
    *CASE_FEATURE_KEYS,
)

# 禁止作为输入的常量/废弃列
FORBIDDEN_FEATURE_KEYS = ("dist_to_wall",)

# targets whose query set comes from the volume view (support stays the wall point cloud)
VOLUME_TARGETS = ("pressure_mixed", "velocity", "velocity_pressure")
# 2026-09-20 周期积分量标量目标（wss_min_cycle_v1 视图：从 81 帧向量 WSS 派生，帧 0–79 各权 1/80）：
# 'tawss' = 周期平均 |τ|（Pa，log_z 统计量）；'osi' = 振荡剪切指数 ∈ [0, 0.5]（linear 或 logit_z 统计量）。
# 与峰值帧标量 WSS 共用同一套单帧训练/评估路径，只换标签来源与统计量文件。
CYCLE_TARGETS = ("tawss", "osi")
OSI_MAX = 0.5
# 2026-09-21 M1 三头单模型：一次输出 [峰值帧 WSS(log_z), TAWSS(log_z), OSI(logit_z 或 linear)]，out_dim=3，各通道各自统计量
# （stats 文件 method='multi'，含 channels 与逐通道子统计），损失 = 三通道等权 MSE（objectives 的向量分支），评估逐通道走单目标路径，
# 顶层结果 = 峰值帧通道（与 X5D 同口径），另两头在 result["heads"]。
MULTI_TARGET = "wss_cycle_multi"
MULTI_CHANNELS = ("wss", "tawss", "osi")
# 2026-09-25 M1 辅助通道（data.multi_aux_channels，缺省空 = 历史三通道，逐位不变）：追加在三通道之后，线性 z（训练折统计量），
# 只进等权 MSE 与评估，不进部署输出。rev_frac = 与峰值帧方向反向的帧份额（cycle.npz）；mean_axial / mean_circ =
# 帧 0–79 平均 WSS 向量在局部 (轴向, 周向) 架上的分量 ÷ 同帧 Σ‖τ‖/80（与方向头同一坐标架，部署可得几何），
# 两者同开时评估另报派生 OSI = ½(1 − min(1, ‖(r_a, r_c)‖))（result["heads"]["osi_derived"]）。
MULTI_AUX_CHANNELS = ("rev_frac", "mean_axial", "mean_circ")


@dataclass
class DataConfig:
    # 显式数据根；默认值保持历史 run 行为。v4 配置必须显式写入该字段。
    data_root: str = str(DATA_ROOT)
    # None 保持旧 bundle 兼容；v4 配置强制 stl_landmarks_v4。
    required_frame_version: Optional[str] = None
    split_path: str = str(DEFAULT_SPLIT)
    wss_stats_path: str = str(GLOBAL_STATS)
    # 可选冻结几何特征统计。None 表示严格从当前 train partition 重算；
    # 数据扩容归因实验可显式引用控制组的 feature_stats.json。
    feature_stats_path: Optional[str] = None
    # 可选逐病例常量特征表。当前用于 RCR oracle；只有 input_features 中
    # 显式列出的 CASE_FEATURE_KEYS 会从该表读取并广播到病例内所有壁面点。
    case_features_path: Optional[str] = None
    # V6 逐点几何 sidecar 根目录（<root>/<cohort>/<subset>/<case>/features.npz）。
    # None 保持旧行为；只有 input_features 里出现 SIDECAR_FEATURE_KEYS 时才需要。
    # 可为一个路径或路径列表（按顺序查找每个 sidecar 键；波 1 同时用 wss_min_geom_v2 与 wss_min_flowref_v1）。
    point_features_root: Optional[str] = None
    # 波 1：切平面 WSS 方向标签（由 bundle 的 wall_wss_vec 峰值帧投到 (轴向, 周向) 单位向量）；仅供 direction_head。
    direction_target: bool = False
    # 波 1：把 (segment_id, s_local_mm) 作为 support/query 的截面 token 元数据传给模型；仅供 section_context。
    section_tokens: bool = False
    # 波 2：残差目标。非空时训练目标为 ln(WSS+eps) − <该逐点特征>（如 log_tau0_murray），用 wss_stats 里的
    # ``offset`` 统计做 z-score；评估前把预测换回标准 log_z（stats["log"]），所以指标口径与其它臂一致。
    # 该特征必须出现在 input_features 里（由 sidecar 加载）。None = 历史行为。
    target_log_offset_feature: Optional[str] = None
    # 2026-09-15 密度增广（部署鲁棒性）：训练时每例每轮以 density_aug_prob 的概率换成预计算的抽稀云
    # （<density_aug_root>/L<pct>/<case>/features.npz，tools/build_density_sidecars），抽稀云上重算过的法向、
    # 曲率族与 tn_dot 覆盖对应输入，support/query/patch 邻域都在抽稀云上取；评估不受影响。
    # None / 空 / 0 = 关闭（旧行为逐位不变）。
    density_aug_root: Optional[str] = None
    density_aug_levels: Tuple[int, ...] = ()
    density_aug_prob: float = 0.0
    # 2026-09-16 边界噪声增广（部署鲁棒性，§22.6 B：噪声 0.2 mm 掉 0.035、平滑几乎免费）：训练时每例每轮以
    # noise_aug_prob 的概率换成预计算的噪声云（<noise_aug_root>/S<sigma>/<case>/features.npz，tools/build_noise_sidecars）：
    # 全部点沿法向加相关长度 2 mm 的噪声后的位置，连同噪声云上重算的法向/曲率族/tn_dot 一起替换。抽中噪声视图的
    # 那一轮不再叠加密度增广。默认关闭 = 旧行为逐位不变（独立抽签流，不消耗密度增广的随机数）。
    noise_aug_root: Optional[str] = None
    noise_aug_levels: Tuple[str, ...] = ()
    noise_aug_prob: float = 0.0
    # 训练稀疏化：每例采样多少壁面点；<=0 或 >=全量 表示用全部点
    wall_n_points: int = 2000
    sampling: str = "fps"                 # ... | 'geom_stratified'
    # Support/Query 新协议；None 时严格回退上面的旧字段。
    support_n_points: Optional[int] = None
    support_sampling: Optional[str] = None  # 'fps' | 'random' | 'area_random'
    query_mode: str = "same"               # 'same' | 'independent'
    # volume targets (pressure_mixed): share of query points drawn from the wall pool; rest from interior cells
    query_wall_fraction: float = 0.5
    query_n_points: Optional[int] = None
    query_sampling: Optional[str] = None
    # random 族采样默认要求每例点数 >= support_n；置 True 时欠点病例自动
    # 回退全点（sample_indices 的 k>=n 分支），用于 10k 采样含 <10k 病例的矩阵。
    support_allow_undersized: bool = False
    fps_pool_size: int = 8                # fps_multistart 预计算子集数
    # 可选持久化 FPS 索引缓存。新 FPS 矩阵要求 cache_required=True，避免每个
    # DataLoader worker 重复执行 O(N*k) 的 CPU FPS。
    fps_cache_dir: Optional[str] = None
    fps_cache_required: bool = False
    # geom_weighted 采样权重：w = 1 + a*curv_norm + b*(1/local_radius)_norm + c*near_bifurcation
    geom_weight_curv: float = 1.0
    geom_weight_invradius: float = 1.0
    geom_weight_bifurcation: float = 0.0  # 近原点(=分叉)加权；0 关闭
    # Optional geometry-stratified support/query sampling. Empty tuple preserves
    # all historical sampling exactly; values name equal-probability strata.
    stratified_bins: Tuple[int, ...] = ()
    # Fixed-mm dual-scale metadata for data-side experiments. The model may use
    # radius_over_mm_<value> dynamic feature names; no CFD/label data involved.
    fixed_mm_scales: Tuple[float, ...] = ()
    # Raw atlas metadata, separate from standardised network input channels.
    local_geometry: bool = False
    query_patch_nsample: int = 0
    # 2026-09-16 双尺度 query patch（§22.4 / §22.6 A2：K16 patch 的抽稀损失一半是偏移尺度漂移、一半是邻居变远）：
    # 在 K16 最近邻之外再取固定毫米半径 query_patch_radius_mm 内的覆盖采样点（按距离排序等间隔取 radius_k 个，
    # 不足补 self 并 mask），偏移单位 mm；query_patch_offset_norm='spacing' 时把 K16 的偏移除以该 query 的邻居
    # 中位间距（无量纲）。默认全关 = 旧行为逐位不变。
    query_patch_radius_mm: float = 0.0
    query_patch_radius_k: int = 0
    query_patch_offset_norm: str = "mm"   # 'mm' | 'spacing'
    # 每个 epoch 重新采样（增广，仅训练集）。val/test 恒用完整点云评估。
    resample_each_epoch: bool = True
    # 训练期随机 3D 旋转增广
    rot_aug: bool = False
    # 输入特征（网络实际吃的列）
    input_features: Tuple[str, ...] = ("x", "y", "z")
    # curvature 输入的鲁棒变换；旧 run 的 feature_stats 不含 transform 时仍按旧口径评估
    curvature_transform: str = "signed_log1p"  # 'signed_log1p' | 'none'
    # 时间步：'peak' = 峰值收缩期单帧（历史行为）；'random_frame' = 训练每例每 epoch 随机一帧、
    # 评估按 eval.eval_frames；需要 input_features 含 TIME_FEATURE_KEYS 且给出 waveform_path。
    timesteps: str = "peak"               # 'peak' | 'random_frame' | 'time_basis'
    waveform_path: Optional[str] = None   # 协议入口波形 JSON（steps/q_norm/dq_norm/t_sin/t_cos/t_norm）
    peak_frame_prob: float = 1.0 / 9.0    # random_frame 训练时抽到峰值帧的保底概率
    # 时间基输出头（timesteps='time_basis'）：训练折时间 PCA 基文件（time_basis.py 合同）；每点目标 = [b0, a_1..a_K]，
    # 一次推理重建全部帧。需要 model.time_basis_k>0、model.out_dim=K+1、target_normalization='frame_stats'。
    time_basis_path: Optional[str] = None
    # 体场目标的全周期标签来源（timesteps != 'peak' 时必填）：A0 产出的逐例 sidecar 根目录
    # （壁面行的最近内部单元索引 + 逐帧体积平均压）与快照 case.h5 的 cases 根目录。
    # 81 帧体场标签太大不入内存，训练/评估按帧从 case.h5 懒读，见 dataset.VolumeFrameSource。
    volume_time_sidecar_root: Optional[str] = None
    volume_h5_root: Optional[str] = None
    target: str = "wss"                   # 'wss' | 'pressure'（均为标量）| 'tawss' | 'osi'（周期积分标量，见 CYCLE_TARGETS）
    # 周期积分量视图根目录（wss_v5.views.wall_cycle_v1 的 wss_min_cycle_v1）；仅 target ∈ CYCLE_TARGETS 时必填，
    # 逐点标签 wall_tawss / wall_osi 按视图 bundle 行序读取并用 wall_node_id_cas 校验。None = 旧行为。
    cycle_view_root: Optional[str] = None
    # 2026-09-25 M1 辅助通道（仅 target=wss_cycle_multi；取值见 MULTI_AUX_CHANNELS）；缺省空 = 历史三通道。
    multi_aux_channels: Tuple[str, ...] = ()
    # 目标标准化；global_stats 保持历史行为，case_max 用逐病例完整壁面 WSS 最大值；
    # frame_stats = 逐帧 log 均值/标准差（stats 文件 'frame' 块，训练折），仅多帧模式（random_frame / time_basis）可用。
    target_normalization: str = "global_stats"  # 'global_stats' | 'case_max' | 'frame_stats'
    num_workers: int = 4
    # 第四轮默认 False：避免 persistent worker 持有过期 epoch 状态
    persistent_workers: bool = False


@dataclass
class ModelConfig:
    # ``pointnext_s`` is retained for historical runs.  The minimal 2x3
    # baseline uses ``mlp``, ``pointnet`` and ``pointnetpp``.
    name: str = "pointnext_s"
    width: int = 32                       # stem 通道
    # 可选显式 stem 通道序列（不含输入维）。空 tuple 保持历史 (width, width)；
    # 显式设置时末元素必须等于 width，仅 pointnetpp 支持。
    stem_channels: Tuple[int, ...] = ()
    # 逐点 MLP 的隐藏层宽度；仅 ``name='mlp'`` 使用。
    mlp_hidden: Tuple[int, ...] = (64, 64, 64)
    # 4 个 SA stage 的下采样比、ball 半径（归一化坐标下）、每组邻居上限、残差块数
    sa_ratios: Tuple[float, ...] = (0.25, 0.25, 0.25, 0.25)
    sa_radius: Tuple[float, ...] = (0.05, 0.10, 0.20, 0.40)
    sa_nsample: Tuple[int, ...] = (16, 16, 16, 16)
    sa_blocks: Tuple[int, ...] = (1, 1, 1, 1)   # 每 stage InvResMLP 残差块数
    # pointnetpp 的 PointNeXt-R 可控残差核心。历史 baseline 配置的
    # sa_blocks=(0,...) 保持完全关闭；非零时才读取下列参数。
    invres_expansion: int = 2
    residual_scale_init: float = 1e-3
    # 空 tuple 保留历史 ratio-SA；显式 counts 启用固定中心和 Support/Query 解码。
    sa_center_sampling: str = "fps"        # 'fps' | 'random'
    sa_center_counts: Tuple[int, ...] = ()
    # SA grouping；空 tuple 保持历史三层 ball-query 行为。新矩阵可逐层指定，
    # 例如 ("adaptive_cover", "ball", "ball") 只改变 SA1。
    sa_grouping: Tuple[str, ...] = ()       # ball | knn | knn_cover | adaptive_cover
    # adaptive_cover 先做 source->最近 center 的全覆盖主分配，再从以下数量的
    # 最近 center 候选中选择至多一个第二归属。目标 group 大小沿用 sa_nsample。
    sa_adaptive_candidate_centers: int = 16
    sa_overlap_cap: float = 1.0 / 3.0
    # full-query 解码器。qad_lite 是历史无界残差对照；sep_kernel 只学习
    # 非负且和为 1 的凸插值权重。默认 interpolate 严格保持历史行为。
    query_decoder: str = "interpolate"     # interpolate | qad_lite | sep_kernel | local_attn
    # Fixed query interpolation only; 0 preserves historical fp_knn. Does not
    # change the encoder's FeaturePropagation neighbourhoods.
    query_interpolation_k: int = 0
    # Zero selected already-standardised input channels at the model boundary.
    # Keeps feature order, train statistics and all geometric indices unchanged.
    input_feature_mask_indices: Tuple[int, ...] = ()
    qad_hidden: int = 32
    sep_hidden: int = 32
    sep_beta_init: float = 2.0
    # local_attn（V6 A2）：k 近邻几何条件凸插值 + 可选零初始化局部残差头。
    # 初始化恒等于 1/d^2 的 k 近邻插值，k<=0 时回退 fp_knn（即 sep_kernel 的推广）。
    # 训练必须用 query_mode='independent'，否则 query==support 时解码器退化为恒等、学不到东西。
    query_decoder_k: int = 0
    query_decoder_hidden: int = 32
    query_decoder_residual: bool = False
    query_decoder_feature_indices: Tuple[int, ...] = ()
    # 高分辨率局部壁面分支（V6 A1）：在 support 分辨率上做多尺度邻域消息，
    # 零初始化后加到解码前的逐点特征；False 时不建模块、state_dict 与旧 run 一致。
    local_branch: bool = False
    local_branch_mode: str = "neighborhood"  # neighborhood | pointwise (capacity control)
    pointwise_branch_hidden: int = 34
    local_branch_radii: Tuple[float, ...] = (0.015, 0.03, 0.06)
    local_branch_nsample: Tuple[int, ...] = (16, 16, 16)
    local_branch_channels: int = 32
    local_branch_feature_indices: Tuple[int, ...] = ()
    # M2 optimisation: modulation of the pooled local-scale channels; disabled
    # by default to preserve historical parameter keys and forward operations.
    local_branch_modulation: str = "none"
    local_branch_modulation_hidden: int = 16
    local_branch_directional: bool = False
    local_branch_pool: str = "max"
    local_branch_radius_mode: str = "normalized"
    local_branch_radii_mm: Tuple[float, ...] = (2.5, 5.0)
    local_branch_radius_ratio: float = 0.5
    query_patch_film: bool = False
    query_patch_k: int = 16
    query_patch_hidden: int = 64
    query_patch_radius_k: int = 0  # 2026-09-16 双尺度 patch 第二组（固定毫米范围）点数，须等于 data.query_patch_radius_k；0 = 关
    # 2026-09-18 扇区 token（B2）：把固定毫米球的邻居按 query 自身 (轴向, 周向) 平面的角度分成
    # query_patch_sectors 个扇形 × query_patch_rings 个距离环，每格一个 token，token 间做一层自注意力
    # 再汇总；输出零初始化。0 = 关，旧配置逐位不变。需要 query_patch_radius_k > 0。
    query_patch_sectors: int = 0
    query_patch_rings: int = 1
    query_patch_sector_heads: int = 4
    # 波 1（wss_local_wave1）：query patch 的相对坐标架与池化方式；默认 = C1 行为。
    # 'tangent' 把 patch 相对偏移表示为 query 的 (轴向, 周向, 法向) 局部坐标；'attention' 用零初始化
    # 的 logit 做 query 条件注意力池化（初始 = 均值池化，逐位等于 C1）。
    query_patch_frame: str = "atlas"       # 'atlas' | 'tangent'
    query_patch_pool: str = "mean"         # 'mean' | 'attention'
    # 波 1：混合专家头。experts 份 head 拷贝 + 以 query 输入特征为门控（门控末层零初始化 =>
    # 均匀权重 => 初始输出与单头逐位相同）。0 = 关闭，不增加 state_dict key。
    head_moe_experts: int = 0
    head_moe_hidden: int = 16
    # 波 1：切平面 WSS 方向辅助头：在通道 0 之后追加 (轴向, 周向) 两个输出通道，通道 0 的计算不变；
    # 需要 data.direction_target 与 train.loss_direction_lambda > 0。
    direction_head: bool = False
    direction_head_hidden: int = 32
    # 波 1：中心线截面 token 上下文。support 特征按 (segment, s_local/bin_mm) 池化成树上的 token，
    # 做 layers 层 Transformer，零初始化残差加到 query patch 的 FiLM 上下文上（初始逐位等于 C1）。
    section_context: bool = False
    section_context_bin_mm: float = 4.0
    section_context_max_bins: int = 64
    section_context_layers: int = 2
    section_context_heads: int = 4
    section_context_hidden: int = 128
    # 病例级幅值头（V6 A3）：最粗一级 SA 的全局池化 -> 逐例标量，加到输出通道 0。
    # 标准化 log 空间的逐例加性偏置 == 物理空间的逐例乘性尺度；零初始化 = 初始恒等。
    case_scale_head: bool = False
    case_scale_hidden: int = 64
    # LocalGeoPE 在每层 SA 消息中加入相对 xyz/距离及可选三个语义几何量
    # 的差分；默认索引对应 abscissa_norm/local_radius/curvature。空 tuple
    # 表示纯 XYZ 版，只使用相对 xyz/距离，不访问不存在的语义特征。
    local_geope: bool = False
    local_geope_feature_indices: Tuple[int, ...] = (3, 4, 5)
    # S3-GEOPE 后续正则化：DropPath 仅作用于 PointNeXt-R 残差块，
    # 并按全部残差块深度从 0 线性增加到该最大值；NeighborDrop 在每个
    # SA/InvRes 邻域内随机丢边，但始终保留离中心最近的一条边。
    drop_path_rate: float = 0.0
    neighbor_drop_rate: float = 0.0
    # SA 邻域内 Transformer：1-based stage 编号；空 tuple 完全关闭并保持
    # 历史 state_dict。它在逐边 MLP/LocalGeoPE 后、max 聚合前运行。
    local_transformer_stages: Tuple[int, ...] = ()
    local_transformer_heads: int = 4
    local_transformer_ffn_ratio: int = 2
    local_transformer_dropout: float = 0.0
    # 几何分组内的静态 EdgeConv 残差消息：1-based stage 编号；空 tuple
    # 完全关闭且不增加 state_dict key。邻域仍由既有 SA grouping 决定，
    # 只检验 [x_i, x_j-x_i, Δp_ij] 消息是否补充局部关系建模。
    edgeconv_stages: Tuple[int, ...] = ()
    # 只在最后一级 SA（D2 为 32 centers）使用一个相对几何 bias 全局块。
    # 该模块已经等价承担“SA3 全局 Transformer”，不要再叠加同义开关。
    coarse_attention: bool = False
    coarse_attention_heads: int = 4
    # 导师的 bottleneck transformer（最粗一级全局自注意力）按我们的数据适配：token 的位置/上下文编码
    # 默认用中心点的完整输入几何特征（pos_enc='input_features'），可选 'xyz'（导师原版）或 'none'；
    # 注意力 logits 仍可叠加相对几何偏置；每层 γ 缩放残差（初始 = 恒等）。默认关闭，不改旧 run。
    bottleneck_transformer: bool = False
    # Same coarse-token conditioning and location; token_ffn is the parameter-
    # matched pointwise control for the attention layer (no token mixing).
    bottleneck_mode: str = "attention"  # attention | token_ffn
    bottleneck_layers: int = 1
    bottleneck_heads: int = 8
    bottleneck_ffn_ratio: int = 2
    bottleneck_dropout: float = 0.0   # 与 local_transformer_dropout/dropout 一致；导师原版 0.1 需在配置里显式写
    bottleneck_pos_enc: str = "input_features"  # 'input_features' | 'xyz' | 'none'
    bottleneck_geo_bias: bool = True
    # 残差强度：None = 沿用骨干的 residual_scale_init(1e-3，近恒等启动)；导师的标准 Transformer 相当于 1.0
    bottleneck_residual_scale_init: Optional[float] = None
    # Parallel radius-set grouping at the final SA stage.  Each branch shares
    # FPS centres but uses its own ball neighbourhood; optional branch-wise
    # bottleneck Transformer or pointwise FFN is fused back to model width.
    multiradius_values: Tuple[float, ...] = ()
    multiradius_bottleneck: bool = False
    multiradius_ffn: bool = False
    invres_radius_scale: float = 1.0      # InvResMLP 分组半径 = sa_radius * scale
    fp_knn: int = 3                       # FP 插值近邻数
    head_hidden: int = 64
    # PointNet 可显式指定通道以复现导师网络；空 tuple 保持 width 推导的历史结构。
    pointnet_local_channels: Tuple[int, ...] = ()
    pointnet_decoder_channels: Tuple[int, ...] = ()
    out_dim: int = 1                      # 标量=1；矢量=3（二期）；NLL=2；时间基头=K+1
    # 时间基输出头的模态数 K（0 = 关闭）；>0 时必须 data.timesteps='time_basis' 且 out_dim=K+1
    time_basis_k: int = 0
    output_head: str = "single"          # single | velocity_pressure (ux, uy, uz, p_rel)
    dropout: float = 0.0


@dataclass
class TrainConfig:
    epochs: int = 160
    batch_cases: int = 8                  # 每步几个病例（点云）
    lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup_epochs: int = 10
    min_lr: float = 1e-5
    grad_clip: float = 1.0
    seed: int = 1234
    # loss：标准化空间 MSE；可几何加权（更看重狭窄/分叉/高梯度处）
    loss: str = "mse"                     # 'mse' | 'huber' | 'gaussian_nll'
    huber_delta: float = 1.0
    loss_geom_weight: bool = False        # True 则按几何量加权 loss
    loss_weight_curv: float = 1.0
    loss_weight_invradius: float = 1.0
    # 按目标幅值加权 loss
    loss_weight_target: bool = False
    loss_weight_target_alpha: float = 2.0
    # 2026-09-26 尾部方向性损失：预测 < 真值（低估）的逐点损失乘以该系数；1.0 = 关闭（旧行为逐位不变）。标量 log 空间目标专用。
    loss_asym_under_weight: float = 1.0
    # 固定 train-only 分位阈值；None 表示回退 batch 分位（仅 B0 行为对照）
    loss_weight_fixed_quantiles: bool = True
    y_norm_q02: Optional[float] = None
    y_norm_q98: Optional[float] = None
    curv_q02: Optional[float] = None
    curv_q98: Optional[float] = None
    invr_q02: Optional[float] = None
    invr_q98: Optional[float] = None
    # C1：log 主损失 + raw-space Huber 辅助；0=关闭
    loss_raw_huber_lambda: float = 0.0
    raw_huber_delta: float = 1.0
    loss_raw_mse_lambda: float = 0.0
    # Optional local-structure objectives (next direct-WSS matrix). Disabled
    # by default to preserve historical loss and checkpoint selection.
    loss_local_diff_lambda: float = 0.0
    local_diff_min_distance: float = 0.0
    local_diff_max_distance: float = 0.0
    local_diff_max_pairs: int = 0
    local_diff_min_distance_mm: float = 2.0
    local_diff_max_distance_mm: float = 5.0
    local_diff_normal_cos_min: float = 0.5
    local_diff_normal_displacement_max: float = 0.5
    loss_pairwise_rank_lambda: float = 0.0
    pairwise_rank_margin: float = 0.0
    pairwise_rank_max_pairs: int = 0
    log_loss_components: bool = False
    # Pair architecture experiments with a fresh seeded reference model, never
    # with its trained checkpoint. None leaves historical construction intact.
    init_reference_config: Optional[str] = None
    # NLL（C3）
    nll_logvar_min: float = -6.0
    nll_logvar_max: float = 2.0
    nll_logvar_reg: float = 1e-4
    # O0 高值优化：保持标准化空间 MSE 为主损失，只增加一个可归因的辅助项。
    # hotspot BCE 使用第二输出通道作为逐点 logit；pinball 仍使用第一回归通道。
    loss_hotspot_bce_lambda: float = 0.0
    hotspot_quantile: float = 0.90
    loss_pinball_lambda: float = 0.0
    pinball_quantile: float = 0.90
    # 2026-09-24 OSI 尾部辅助项（仅 target='wss_cycle_multi'，作用在 OSI 通道；跟踪 §34.11 / P0c：误差集中在高 OSI 且为幅值压缩）。
    # tail：OSI 通道逐点平方误差乘 w = 1 + alpha·1[OSI_true > osi_tail_threshold]，批内按 w 均值归一（通道总权重不变）。
    # mask：逐阈值软掩膜 BCE，logit = (ẑ_osi − z(T)) / osi_mask_temperature（z 为 OSI 通道归一化空间），标签 1[OSI_true > T]，阈值间平均。
    # 0 = 关闭，旧配置逐位不变。
    loss_osi_tail_alpha: float = 0.0
    osi_tail_threshold: float = 0.2
    loss_osi_mask_lambda: float = 0.0
    osi_mask_thresholds: Tuple[float, ...] = (0.2, 0.3)
    osi_mask_temperature: float = 0.25
    # 波 1：切平面 WSS 方向辅助损失（1 - cos），作用在 direction_head 的两个通道上；0 = 关闭
    loss_direction_lambda: float = 0.0
    # 波 1：权重 EMA（0 = 关闭；>0 时每步更新影子权重并另存 ckpt_ema.pt，best/last 不变）
    ema_decay: float = 0.0
    # 波 5 T3 级联：按某个输入特征（一阶段预测 log_wss_base）的逐例分位定义热点区，区内主损失权重 1、区外
    # loss_region_outside_weight；辅助项（pinball 等）不加权。None = 关闭，旧配置逐位不变。
    loss_region_feature: Optional[str] = None
    loss_region_quantile: float = 0.8
    loss_region_outside_weight: float = 0.2
    amp: bool = True
    eval_every: int = 10                  # 每多少 epoch 在 val 完整点云上评估一次
    ckpt_metric: str = "val_selection_score"  # 兼容旧名；实际用 selection_rule
    # 'train_loss_ema'：按训练损失的指数滑动平均选 best（随机帧训练时逐轮损失取决于抽到的帧，直接取最小有偏）
    selection_rule: str = "r4_composite_v1"
    selection_ema_alpha: float = 0.2      # 仅 selection_rule='train_loss_ema' 使用
    ckpt_top_k: int = 3
    early_stop_patience: int = 6          # 按 eval 次数计；<=0 关闭早停
    min_epoch: int = 40
    log_every_steps: int = 0              # >0 时按 step 打点；0 只按 epoch
    # 可选 warm-start：仅加载模型权重，优化器、学习率日程和 checkpoint 选择均重新开始。
    # 这不是中断恢复，路径与哈希会写入新 run 的 initialization.json。
    init_checkpoint_path: Optional[str] = None
    init_checkpoint_strict: bool = True


@dataclass
class EvalConfig:
    # False 时保持历史“完整点云既是 support 又是 query”的评估行为。
    fixed_support: bool = False
    full_cloud_query: bool = True
    query_chunk_size: int = 0
    support_seed: int = 1234
    stability_seeds: Tuple[int, ...] = (1234, 2025, 3407, 7919, 104729)
    # legacy_vertex 保持旧 run/旧 config 不依赖 STL 面积映射；both_strict 才启用
    # 面积加权指标，并要求每例严格通过 surface-area mapping，禁止静默均匀退化。
    surface_metric_mode: str = "legacy_vertex"  # 'legacy_vertex' | 'both_strict'
    # gaussian_nll 的部署口径：物理空间用逐点 exp(mu + sigma^2/2)（Jensen 修正）而不是 exp(mu)。
    # False = 历史行为（只用通道 0）；归一化空间指标恒为通道 0，不受影响。
    pointwise_jensen: bool = False
    # timesteps='random_frame' 时的评估帧：'peak' | 'all' | 逗号分隔 Fluent 步号（如 "1120,1162,1216"）。
    # 顶层结果恒为峰值帧（与单帧 run 同口径），其余帧在 result["frames"]，另有 pooled 与 TAWSS。
    eval_frames: str = "peak"
    # 多帧模式的周期口径附加块 result["cycle"]：逐帧 R²_cb 均值、谷底四分位（Q 最低 20 帧）、峰时误差、逐点时序相关等；
    # 需要 data.waveform_path（定义谷底帧）。False = 旧输出逐位不变。
    time_metrics: bool = False
    # 波 5 T3 级联评估门控：二阶段残差只在偏移特征（一阶段预测）的逐例 top-(1-q) 区域生效，区外回退为
    # 一阶段预测（残差 0）。需要 data.target_log_offset_feature；None = 关闭（旧行为逐位不变）。
    residual_gate_quantile: Optional[float] = None
    # 2026-09-20 物理空间阈值掩膜指标（周期积分量矩阵 §7）：对每个阈值算 真值>阈 与 预测>阈（above）/ <阈（below）
    # 的逐例 IoU / precision / recall / 面积份额误差，写入 result["threshold_masks"]。空 = 旧输出逐位不变。
    threshold_masks_above: Tuple[float, ...] = ()
    threshold_masks_below: Tuple[float, ...] = ()
    # 2026-09-21 周期积分量落地一致性块 result["cycle_agreement"]（仅 target ∈ CYCLE_TARGETS 时计算；wss 口径不变）：
    # 逐例 Lin CCC（TAWSS 在 ln 空间、OSI 线性）、逐点相对误差中位与 ≤ 容差份额、top10% 热点 Dice、阈值掩膜 Dice、
    # 按语义血管段（bundle wall_semantic_id）的段均值相对误差；病例级 Bland–Altman（病例均值、掩膜面积份额）。
    cycle_agreement: bool = True
    cycle_rel_tolerance: float = 0.3
    cycle_min_segment_points: int = 200
    # 相对误差分母下限：TAWSS 用统计量文件的 floor（0.05 Pa）；OSI 是 [0, 0.5] 的有界量，近零点相对误差无意义，用 0.01
    cycle_osi_rel_floor: float = 0.01


@dataclass
class ExpConfig:
    name: str = "baseline_xyz_fps2000_peak"
    notes: str = ""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    # ---- 派生路径 ----
    @property
    def run_dir(self) -> Path:
        return RUNS_ROOT / self.name

    # ---- 序列化 ----
    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False))
        return path

    @staticmethod
    def from_dict(d: dict) -> "ExpConfig":
        def _mk(cls, sub):
            sub = dict(sub or {})
            # tuple 字段 JSON 里是 list，转回 tuple 保持类型一致
            for k, v in list(sub.items()):
                if isinstance(v, list):
                    sub[k] = tuple(v)
            # 忽略未知字段（向前兼容旧 config）
            known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore
            sub = {k: v for k, v in sub.items() if k in known}
            return cls(**sub)
        cfg = ExpConfig(
            name=d.get("name", "unnamed"),
            notes=d.get("notes", ""),
            data=_mk(DataConfig, d.get("data")),
            model=_mk(ModelConfig, d.get("model")),
            train=_mk(TrainConfig, d.get("train")),
            eval=_mk(EvalConfig, d.get("eval")),
        )
        validate_features(cfg)
        return cfg

    @staticmethod
    def from_json(path: str | Path) -> "ExpConfig":
        return ExpConfig.from_dict(json.loads(Path(path).read_text()))


def validate_features(cfg: ExpConfig) -> None:
    longitudinal = LG.validate_feature_pairs(cfg.data.input_features)
    if longitudinal:
        # No new channels are enabled by default. Geometry values and their
        # masks are explicit pairs and never serve as labels/loss selectors.
        if cfg.data.target_log_offset_feature in LG.FEATURE_KEYS or cfg.train.loss_region_feature in LG.FEATURE_KEYS:
            raise ValueError("longitudinal geometry/masks are inputs only, not target offsets or supervision selectors")
        if cfg.data.feature_stats_path:
            LG.validate_stats_file(cfg.data.feature_stats_path, cfg.data.input_features, cfg.data.split_path)
    if cfg.model.local_branch_pool not in {"max", "multistat"}:
        raise ValueError("local_branch_pool must be max or multistat")
    if cfg.model.local_branch_radius_mode not in {"normalized", "physical_dual"}:
        raise ValueError("local_branch_radius_mode must be normalized or physical_dual")
    new_local = (cfg.model.local_branch_directional or cfg.model.local_branch_pool != "max"
                 or cfg.model.local_branch_radius_mode != "normalized")
    if new_local and (not cfg.model.local_branch or cfg.model.local_branch_mode != "neighborhood"):
        raise ValueError("local neighborhood options require the enabled neighborhood branch")
    need_geometry = (cfg.model.local_branch_directional
                     or cfg.model.local_branch_radius_mode == "physical_dual"
                     or cfg.train.loss_local_diff_lambda > 0)
    if need_geometry and not cfg.data.local_geometry:
        raise ValueError("directional/physical neighborhoods and local difference require data.local_geometry")
    if cfg.model.local_branch_radius_mode == "physical_dual":
        if (len(cfg.model.local_branch_radii_mm) != 2
                or any(not math.isfinite(r) or r <= 0 for r in cfg.model.local_branch_radii_mm)
                or not math.isfinite(cfg.model.local_branch_radius_ratio)
                or cfg.model.local_branch_radius_ratio <= 0
                or len(cfg.model.local_branch_nsample) != 3):
            raise ValueError("physical_dual requires two positive mm radii, a positive radius ratio and three branches")
    if cfg.model.query_patch_film:
        if (cfg.model.name != "pointnetpp" or cfg.data.query_mode != "independent"
                or not cfg.data.local_geometry
                or not cfg.eval.fixed_support or not cfg.model.sa_center_counts
                or cfg.model.query_patch_hidden <= 0 or cfg.model.query_patch_k < 2
                or cfg.data.query_patch_nsample != cfg.model.query_patch_k):
            raise ValueError("query_patch_film requires independent fixed-support pointnetpp queries and matching positive patch sizes")
    elif cfg.data.query_patch_nsample:
        raise ValueError("query_patch_nsample would be unused without query_patch_film")
    if cfg.data.query_patch_offset_norm not in {"mm", "spacing"}:
        raise ValueError("query_patch_offset_norm must be 'mm' or 'spacing'")
    if cfg.data.query_patch_offset_norm != "mm" and not cfg.model.query_patch_film:
        raise ValueError("query_patch_offset_norm is unused without query_patch_film")
    if int(cfg.data.query_patch_radius_k) > 0 or float(cfg.data.query_patch_radius_mm) > 0 or int(cfg.model.query_patch_radius_k) > 0:
        if (not cfg.model.query_patch_film or int(cfg.data.query_patch_radius_k) < 1
                or not math.isfinite(float(cfg.data.query_patch_radius_mm)) or float(cfg.data.query_patch_radius_mm) <= 0
                or int(cfg.model.query_patch_radius_k) != int(cfg.data.query_patch_radius_k)):
            raise ValueError("dual-scale patch requires query_patch_film, a positive query_patch_radius_mm and matching positive radius_k in data and model")
    for weight in (cfg.train.loss_local_diff_lambda, cfg.train.loss_pairwise_rank_lambda):
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("local and ranking loss weights must be finite and non-negative")
    # ---- 波 1（wss_local_wave1）新增字段的校验；默认值全部等于 C1/历史行为 ----
    if int(cfg.model.query_patch_sectors) > 0:
        if int(cfg.model.query_patch_radius_k) < 1:
            raise ValueError("query_patch_sectors operates on the fixed-mm ball branch; set query_patch_radius_k > 0")
        if not cfg.data.local_geometry:
            raise ValueError("query_patch_sectors needs data.local_geometry for the query tangent frame")
        if int(cfg.model.query_patch_sectors) < 2 or int(cfg.model.query_patch_rings) < 1 \
                or int(cfg.model.query_patch_sector_heads) < 1 \
                or int(cfg.model.query_patch_hidden) % int(cfg.model.query_patch_sector_heads):
            raise ValueError("query_patch_sectors >= 2, rings >= 1, and query_patch_hidden must divide by sector_heads")
    elif int(cfg.model.query_patch_rings) != 1 or int(cfg.model.query_patch_sector_heads) != 4:
        raise ValueError("query_patch_rings / query_patch_sector_heads are unused without query_patch_sectors")
    if cfg.model.query_patch_frame not in {"atlas", "tangent"}:
        raise ValueError("query_patch_frame must be 'atlas' or 'tangent'")
    if cfg.model.query_patch_pool not in {"mean", "attention"}:
        raise ValueError("query_patch_pool must be 'mean' or 'attention'")
    if (cfg.model.query_patch_frame != "atlas" or cfg.model.query_patch_pool != "mean") and not cfg.model.query_patch_film:
        raise ValueError("query_patch_frame/query_patch_pool require query_patch_film=true")
    if int(cfg.model.head_moe_experts) < 0 or int(cfg.model.head_moe_hidden) <= 0:
        raise ValueError("head_moe_experts must be >= 0 and head_moe_hidden positive")
    if int(cfg.model.head_moe_experts) and (cfg.model.name != "pointnetpp" or cfg.model.output_head != "single"):
        raise ValueError("the mixture-of-experts head requires the single-head pointnetpp model")
    direction_weight = float(cfg.train.loss_direction_lambda)
    if not math.isfinite(direction_weight) or direction_weight < 0:
        raise ValueError("loss_direction_lambda must be finite and non-negative")
    if bool(cfg.model.direction_head) != (direction_weight > 0):
        raise ValueError("direction_head and loss_direction_lambda>0 must be enabled together")
    if cfg.data.direction_target != bool(cfg.model.direction_head):
        raise ValueError("data.direction_target and model.direction_head must be enabled together")
    if cfg.model.direction_head:
        if (cfg.model.name != "pointnetpp" or cfg.data.target != "wss" or not cfg.data.local_geometry
                or cfg.train.loss == "gaussian_nll" or int(cfg.model.out_dim) != 1
                or float(cfg.train.loss_hotspot_bce_lambda) > 0 or cfg.data.timesteps != "peak"
                or cfg.model.output_head != "single" or int(cfg.model.direction_head_hidden) <= 0):
            raise ValueError("direction_head requires peak-frame scalar WSS, local_geometry, out_dim=1, single head, no NLL/hotspot channels")
    if cfg.data.section_tokens != bool(cfg.model.section_context):
        raise ValueError("data.section_tokens and model.section_context must be enabled together")
    if cfg.model.section_context:
        if not cfg.model.query_patch_film:
            raise ValueError("section_context feeds the query patch FiLM context; it requires query_patch_film=true")
        if (float(cfg.model.section_context_bin_mm) <= 0 or int(cfg.model.section_context_max_bins) < 1
                or int(cfg.model.section_context_layers) < 1 or int(cfg.model.section_context_heads) < 1
                or int(cfg.model.section_context_hidden) <= 0
                or int(cfg.model.section_context_hidden) % int(cfg.model.section_context_heads) != 0):
            raise ValueError("invalid section_context bin/token/layer/head/hidden settings")
    if not (0.0 <= float(cfg.train.ema_decay) < 1.0):
        raise ValueError("ema_decay must be in [0, 1)")
    offset_feature = cfg.data.target_log_offset_feature
    if offset_feature:
        if (cfg.data.target != "wss" or cfg.data.timesteps != "peak" or cfg.data.target_normalization != "global_stats"
                or cfg.train.loss == "gaussian_nll" or float(cfg.train.loss_raw_huber_lambda or 0) > 0
                or float(cfg.train.loss_raw_mse_lambda or 0) > 0 or cfg.eval.pointwise_jensen
                or float(cfg.train.loss_hotspot_bce_lambda) > 0):
            raise ValueError("target_log_offset_feature requires peak-frame scalar WSS with global_stats, plain log-space losses")
        if offset_feature not in cfg.data.input_features or offset_feature not in SIDECAR_FEATURE_KEYS:
            raise ValueError("target_log_offset_feature must be a sidecar point feature listed in input_features")
    density_prob = float(cfg.data.density_aug_prob or 0.0)
    if density_prob < 0.0 or density_prob > 1.0:
        raise ValueError("density_aug_prob must be in [0, 1]")
    if density_prob > 0.0:
        if not cfg.data.density_aug_root or not cfg.data.density_aug_levels:
            raise ValueError("density augmentation requires density_aug_root and density_aug_levels")
        if any(int(l) <= 0 or int(l) >= 100 for l in cfg.data.density_aug_levels):
            raise ValueError("density_aug_levels are percentages strictly between 0 and 100")
        if cfg.data.target not in ("wss", *CYCLE_TARGETS, MULTI_TARGET):
            # 周期积分量（tawss/osi/多头）是逐点壁面标量，抽稀行索引同样切标签（dataset.subset_case），与 wss 同路径
            raise ValueError("density augmentation supports wall scalar targets only (wss / tawss / osi / wss_cycle_multi)")
        # 多帧模式（random_frame / time_basis）：抽稀 sidecar 的行索引同样切多帧标签（dataset.subset_case），几何覆盖不变
        for sampling in ((cfg.data.support_sampling or cfg.data.sampling), (cfg.data.query_sampling or cfg.data.support_sampling or cfg.data.sampling)):
            if sampling != "random":
                raise ValueError("density augmentation requires random support/query sampling (FPS pools are tied to the full cloud)")
    noise_prob = float(cfg.data.noise_aug_prob or 0.0)
    if noise_prob < 0.0 or noise_prob > 1.0:
        raise ValueError("noise_aug_prob must be in [0, 1]")
    if noise_prob > 0.0:
        if not cfg.data.noise_aug_root or not cfg.data.noise_aug_levels:
            raise ValueError("noise augmentation requires noise_aug_root and noise_aug_levels")
        try:
            sigmas = [float(t) for t in cfg.data.noise_aug_levels]
        except (TypeError, ValueError) as exc:
            raise ValueError("noise_aug_levels are sigma tags in mm, e.g. [\"0.2\", \"0.3\"]") from exc
        if any(s <= 0.0 or s > 2.0 for s in sigmas):
            raise ValueError("noise_aug_levels must be sigma values in (0, 2] mm")
        if cfg.data.target != "wss" or cfg.data.timesteps != "peak":
            raise ValueError("noise augmentation supports peak-frame WSS only")
        for sampling in ((cfg.data.support_sampling or cfg.data.sampling), (cfg.data.query_sampling or cfg.data.support_sampling or cfg.data.sampling)):
            if sampling != "random":
                raise ValueError("noise augmentation requires random support/query sampling (FPS pools are tied to the clean cloud)")
    region_feature = cfg.train.loss_region_feature
    if region_feature:
        if region_feature not in cfg.data.input_features:
            raise ValueError("loss_region_feature must be listed in input_features")
        if cfg.data.target != "wss" or cfg.data.timesteps != "peak":
            raise ValueError("loss_region_feature requires peak-frame scalar WSS")
        if not (0.0 < float(cfg.train.loss_region_quantile) < 1.0) or float(cfg.train.loss_region_outside_weight) < 0.0:
            raise ValueError("loss_region_quantile must be in (0, 1) and loss_region_outside_weight >= 0")
    gate_quantile = cfg.eval.residual_gate_quantile
    if gate_quantile is not None:
        if not offset_feature:
            raise ValueError("residual_gate_quantile requires data.target_log_offset_feature")
        if not (0.0 < float(gate_quantile) < 1.0):
            raise ValueError("residual_gate_quantile must be in (0, 1)")
    if cfg.data.target in VOLUME_TARGETS:
        wall_only = set(cfg.data.input_features) & (set(SIDECAR_FEATURE_KEYS) - set(VOLUME_EXTENDABLE_SIDECAR_KEYS))
        if wall_only:
            raise ValueError(f"wall-only sidecar features cannot be used with volume targets: {sorted(wall_only)}")
    roots = cfg.data.point_features_root
    if roots is not None and not isinstance(roots, str):
        if not roots or any(not isinstance(r, str) or not r for r in roots):
            raise ValueError("point_features_root must be a path or a non-empty list of paths")
    if cfg.train.loss_local_diff_lambda:
        if not (0 <= cfg.train.local_diff_min_distance_mm < cfg.train.local_diff_max_distance_mm
                and math.isfinite(cfg.train.local_diff_max_distance_mm)
                and cfg.train.local_diff_max_pairs > 0
                and 0 <= cfg.train.local_diff_normal_cos_min <= 1
                and 0 <= cfg.train.local_diff_normal_displacement_max <= 1):
            raise ValueError("invalid physical surface-pair parameters")
    if cfg.train.loss_pairwise_rank_lambda and (cfg.train.pairwise_rank_max_pairs <= 0
            or not math.isfinite(cfg.train.pairwise_rank_margin) or cfg.train.pairwise_rank_margin < 0):
        raise ValueError("ranking requires a positive per-case pair budget and nonnegative finite margin")
    if (need_geometry or cfg.model.query_patch_film) and cfg.data.target not in ("wss", *CYCLE_TARGETS, MULTI_TARGET):
        raise ValueError("new local geometry/patch experiments currently require a wall scalar target (wss / tawss / osi / wss_cycle_multi)")
    if cfg.model.output_head not in {"single", "velocity_pressure"}:
        raise ValueError("output_head must be single or velocity_pressure")
    if cfg.model.output_head == "velocity_pressure" and cfg.data.target != "velocity_pressure":
        raise ValueError("velocity_pressure output_head requires the joint velocity_pressure target")
    if cfg.model.bottleneck_mode not in {"attention", "token_ffn"}:
        raise ValueError("bottleneck_mode must be attention or token_ffn")
    if cfg.model.bottleneck_mode == "token_ffn" and (
        not cfg.model.bottleneck_transformer or cfg.model.multiradius_values
        or int(cfg.model.bottleneck_layers) < 1
    ):
        raise ValueError("token_ffn requires an enabled single bottleneck with positive layers")
    for f in cfg.data.input_features:
        if f in FORBIDDEN_FEATURE_KEYS:
            raise ValueError(
                f"feature {f!r} is forbidden (constant on wall points); "
                f"remove it from input_features"
            )
        dynamic_fixed_mm = f.startswith("radius_over_mm_")
        if dynamic_fixed_mm:
            try:
                if float(f.removeprefix("radius_over_mm_").removesuffix("mm")) <= 0:
                    raise ValueError
            except ValueError:
                raise ValueError(f"invalid fixed-mm feature {f!r}")
        if f not in FEATURE_KEYS and not dynamic_fixed_mm:
            raise ValueError(f"unknown feature {f!r}; allowed={FEATURE_KEYS}")
    requested_case_features = set(cfg.data.input_features) & set(CASE_FEATURE_KEYS)
    if requested_case_features and not cfg.data.case_features_path:
        raise ValueError(
            "case-level input features require data.case_features_path; "
            f"requested={sorted(requested_case_features)}"
        )
    requested_surface_features = set(cfg.data.input_features) & set(SIDECAR_FEATURE_KEYS)
    if requested_surface_features and not cfg.data.point_features_root:
        raise ValueError(
            "sidecar point features require data.point_features_root "
            f"(wall_geom_v2 / wall_flowref_v1 sidecars); requested={sorted(requested_surface_features)}"
        )
    if cfg.data.target_normalization not in {"global_stats", "case_max", "frame_stats"}:
        raise ValueError(
            "target_normalization must be 'global_stats', 'case_max' or 'frame_stats', got "
            f"{cfg.data.target_normalization!r}"
        )
    if cfg.data.target_normalization == "case_max" and cfg.data.target != "wss":
        raise ValueError("target_normalization='case_max' is only supported for target='wss'")
    local = tuple(cfg.model.pointnet_local_channels)
    decoder = tuple(cfg.model.pointnet_decoder_channels)
    if bool(local) != bool(decoder):
        raise ValueError(
            "pointnet_local_channels and pointnet_decoder_channels must both be set or both be empty"
        )
    if any(int(ch) <= 0 for ch in (*local, *decoder)):
        raise ValueError("explicit PointNet channels must all be positive")
    support_sampling = cfg.data.support_sampling or cfg.data.sampling
    if support_sampling not in {"fps", "fps_multistart", "random", "geom_weighted", "geom_stratified", "area_random"}:
        raise ValueError(f"unsupported support sampling {support_sampling!r}")
    if cfg.data.support_allow_undersized and support_sampling not in {"random", "area_random", "geom_stratified"}:
        raise ValueError(
            "support_allow_undersized=True is only supported for random-family sampling"
        )
    if cfg.data.fps_cache_required and not cfg.data.fps_cache_dir:
        raise ValueError("fps_cache_required=True requires fps_cache_dir")
    if cfg.data.query_mode not in {"same", "independent"}:
        raise ValueError("query_mode must be 'same' or 'independent'")
    if cfg.data.query_mode == "independent":
        if not cfg.data.query_n_points or int(cfg.data.query_n_points) <= 0:
            raise ValueError("independent query_mode requires positive query_n_points")
        if (cfg.data.query_sampling or support_sampling) not in {"random", "area_random", "geom_stratified"}:
            raise ValueError("independent query sampling must be random, area_random, or geom_stratified")
    if cfg.model.sa_center_sampling not in {"fps", "random"}:
        raise ValueError("sa_center_sampling must be 'fps' or 'random'")
    if cfg.model.query_decoder not in {"interpolate", "qad_lite", "sep_kernel", "local_attn"}:
        raise ValueError(
            "query_decoder must be 'interpolate', 'qad_lite', 'sep_kernel', or 'local_attn'"
        )
    if int(cfg.model.query_interpolation_k) < 0:
        raise ValueError("query_interpolation_k must be >= 0 (0 preserves fp_knn)")
    if cfg.model.query_interpolation_k and (
        cfg.model.name != "pointnetpp" or cfg.model.query_decoder != "interpolate"
    ):
        raise ValueError("query_interpolation_k only applies to pointnetpp interpolate queries")
    mask_indices = tuple(cfg.model.input_feature_mask_indices)
    if any(not isinstance(i, int) or isinstance(i, bool) for i in mask_indices):
        raise ValueError("input_feature_mask_indices must contain integer indices")
    if len(set(mask_indices)) != len(mask_indices) or any(
        i < 0 or i >= len(cfg.data.input_features) for i in mask_indices
    ):
        raise ValueError("input_feature_mask_indices must be unique and within data.input_features")
    if mask_indices and cfg.model.name != "pointnetpp":
        raise ValueError("input_feature_mask_indices currently requires pointnetpp")
    if cfg.model.local_branch_mode not in {"neighborhood", "pointwise"}:
        raise ValueError("local_branch_mode must be neighborhood or pointwise")
    if cfg.model.local_branch_mode == "pointwise" and not cfg.model.local_branch:
        raise ValueError("pointwise local_branch_mode requires local_branch=true")
    if int(cfg.model.pointwise_branch_hidden) <= 0:
        raise ValueError("pointwise_branch_hidden must be positive")
    if int(cfg.model.query_decoder_hidden) <= 0:
        raise ValueError("query_decoder_hidden must be positive")
    if cfg.model.query_decoder == "local_attn":
        if int(cfg.model.query_decoder_k) < 0:
            raise ValueError("query_decoder_k must be >= 0 (0 falls back to fp_knn)")
        if cfg.data.query_mode != "independent":
            # query == support makes every query its own nearest neighbour at distance 0,
            # so the decoder trains as the identity and only changes behaviour at eval.
            raise ValueError(
                "query_decoder='local_attn' requires data.query_mode='independent' so the "
                "decoder is exercised during training"
            )
        indices = tuple(int(i) for i in cfg.model.query_decoder_feature_indices)
        if len(set(indices)) != len(indices):
            raise ValueError("query_decoder_feature_indices must be unique")
        if indices and (min(indices) < 0 or max(indices) >= len(cfg.data.input_features)):
            raise ValueError("query_decoder_feature_indices are outside data.input_features")
    if int(cfg.model.qad_hidden) <= 0:
        raise ValueError("qad_hidden must be positive")
    if int(cfg.model.sep_hidden) <= 0:
        raise ValueError("sep_hidden must be positive")
    if float(cfg.model.sep_beta_init) < 0:
        raise ValueError("sep_beta_init must be non-negative")
    if cfg.model.query_decoder in {"qad_lite", "sep_kernel", "local_attn"}:
        if cfg.model.name != "pointnetpp":
            raise ValueError(
                f"{cfg.model.query_decoder} currently requires model.name='pointnetpp'"
            )
        if not cfg.model.sa_center_counts:
            raise ValueError(
                f"{cfg.model.query_decoder} requires fixed support/query via sa_center_counts"
            )
    if cfg.model.sa_center_counts:
        if len(cfg.model.sa_center_counts) != len(cfg.model.sa_radius):
            raise ValueError("sa_center_counts must match the number of SA stages")
        if any(int(n) <= 0 for n in cfg.model.sa_center_counts):
            raise ValueError("sa_center_counts must all be positive")
    if cfg.model.name == "pointnetpp":
        if len(cfg.model.sa_blocks) != len(cfg.model.sa_radius):
            raise ValueError("pointnetpp sa_blocks must match the number of SA stages")
        if any(int(n) < 0 for n in cfg.model.sa_blocks):
            raise ValueError("pointnetpp sa_blocks must all be non-negative")
        if int(cfg.model.invres_expansion) <= 0:
            raise ValueError("invres_expansion must be positive")
        if float(cfg.model.residual_scale_init) < 0:
            raise ValueError("residual_scale_init must be non-negative")
        if not (0.0 <= float(cfg.model.drop_path_rate) < 1.0):
            raise ValueError("drop_path_rate must be in [0, 1)")
        if not (0.0 <= float(cfg.model.neighbor_drop_rate) < 1.0):
            raise ValueError("neighbor_drop_rate must be in [0, 1)")
        if float(cfg.model.drop_path_rate) > 0 and not any(
            int(n) > 0 for n in cfg.model.sa_blocks
        ):
            raise ValueError("drop_path_rate > 0 requires at least one PointNeXt-R block")
        if cfg.model.local_branch:
            radii = tuple(float(r) for r in cfg.model.local_branch_radii)
            nsample = tuple(int(n) for n in cfg.model.local_branch_nsample)
            if not radii or len(radii) != len(nsample):
                raise ValueError("local_branch_radii and local_branch_nsample must be non-empty and equal length")
            if any(r <= 0 for r in radii) or any(n <= 0 for n in nsample):
                raise ValueError("local_branch radii/nsample must all be positive")
            if int(cfg.model.local_branch_channels) <= 0:
                raise ValueError("local_branch_channels must be positive")
            branch_indices = tuple(int(i) for i in cfg.model.local_branch_feature_indices)
            if len(set(branch_indices)) != len(branch_indices):
                raise ValueError("local_branch_feature_indices must be unique")
            if branch_indices and (
                min(branch_indices) < 0 or max(branch_indices) >= len(cfg.data.input_features)
            ):
                raise ValueError("local_branch_feature_indices are outside data.input_features")
        if cfg.model.case_scale_head and int(cfg.model.case_scale_hidden) <= 0:
            raise ValueError("case_scale_hidden must be positive")
        if cfg.model.local_geope:
            indices = tuple(int(i) for i in cfg.model.local_geope_feature_indices)
            if len(indices) not in {0, 3} or len(set(indices)) != len(indices):
                raise ValueError(
                    "local_geope_feature_indices must be empty (xyz-only) or contain three unique indices"
                )
            if indices and (
                min(indices) < 0 or max(indices) >= len(cfg.data.input_features)
            ):
                raise ValueError(
                    "local_geope_feature_indices are outside data.input_features"
                )
        local_transformer_stages = tuple(
            int(stage) for stage in cfg.model.local_transformer_stages
        )
        if len(set(local_transformer_stages)) != len(local_transformer_stages):
            raise ValueError("local_transformer_stages must not contain duplicates")
        if any(
            stage < 1 or stage > len(cfg.model.sa_radius)
            for stage in local_transformer_stages
        ):
            raise ValueError(
                "local_transformer_stages must use 1-based SA stage numbers"
            )
        if int(cfg.model.local_transformer_heads) <= 0:
            raise ValueError("local_transformer_heads must be positive")
        if int(cfg.model.local_transformer_ffn_ratio) <= 0:
            raise ValueError("local_transformer_ffn_ratio must be positive")
        if not (0.0 <= float(cfg.model.local_transformer_dropout) < 1.0):
            raise ValueError("local_transformer_dropout must be in [0, 1)")
        for stage in local_transformer_stages:
            stage_channels = int(cfg.model.width) * (2 ** stage)
            if stage_channels % int(cfg.model.local_transformer_heads) != 0:
                raise ValueError(
                    f"SA{stage} channels must be divisible by local_transformer_heads"
                )
        edgeconv_stages = tuple(int(stage) for stage in cfg.model.edgeconv_stages)
        if len(set(edgeconv_stages)) != len(edgeconv_stages):
            raise ValueError("edgeconv_stages must not contain duplicates")
        if any(
            stage < 1 or stage > len(cfg.model.sa_radius)
            for stage in edgeconv_stages
        ):
            raise ValueError(
                "edgeconv_stages must use 1-based SA stage numbers"
            )
        if int(cfg.model.coarse_attention_heads) <= 0:
            raise ValueError("coarse_attention_heads must be positive")
        coarse_channels = int(cfg.model.width) * (2 ** len(cfg.model.sa_radius))
        multi = tuple(cfg.model.multiradius_values)
        if (cfg.model.multiradius_bottleneck or cfg.model.multiradius_ffn) and not multi:
            raise ValueError("multi-radius context requires multiradius_values")
        if multi:
            if any(isinstance(r, bool) or not isinstance(r, (int, float)) or not math.isfinite(r) or r <= 0 for r in multi):
                raise ValueError("multiradius_values must contain finite positive radii")
            if cfg.model.multiradius_bottleneck and cfg.model.multiradius_ffn:
                raise ValueError("multiradius_bottleneck and multiradius_ffn are alternatives")
            if cfg.model.bottleneck_transformer or cfg.model.coarse_attention:
                raise ValueError("multi-radius replaces the single coarse global module")
            if not cfg.model.sa_center_counts or int(cfg.model.sa_center_counts[-1]) <= 0:
                raise ValueError("multi-radius requires fixed positive final-SA centre count")
            final_stage = len(cfg.model.sa_radius)
            if cfg.model.sa_blocks[-1] or final_stage in local_transformer_stages or final_stage in edgeconv_stages:
                raise ValueError("multi-radius currently requires a plain final-SA stage")
            if cfg.model.neighbor_drop_rate:
                raise ValueError("multi-radius currently requires neighbor_drop_rate=0")
            if cfg.model.bottleneck_pos_enc != "input_features":
                raise ValueError("all multi-radius arms require input-feature centre conditioning")
            if cfg.model.bottleneck_layers < 1:
                raise ValueError("multi-radius context layers must be positive")
        if cfg.model.bottleneck_transformer or multi:
            if cfg.model.coarse_attention:
                raise ValueError("bottleneck_transformer and coarse_attention are alternatives at SA3; enable one")
            if int(cfg.model.bottleneck_layers) < 0:
                raise ValueError("bottleneck_layers must be >= 0 (0 = token conditioning only, no attention)")
            if int(cfg.model.bottleneck_heads) < 1:
                raise ValueError("bottleneck_heads must be >= 1")
            if int(cfg.model.bottleneck_ffn_ratio) <= 0:
                raise ValueError("bottleneck_ffn_ratio must be positive")
            if int(cfg.model.bottleneck_layers) == 0 and cfg.model.bottleneck_pos_enc == "none":
                raise ValueError("bottleneck_layers=0 with pos_enc='none' is a no-op module")
            if cfg.model.bottleneck_residual_scale_init is not None and float(cfg.model.bottleneck_residual_scale_init) < 0:
                raise ValueError("bottleneck_residual_scale_init must be non-negative")
            if coarse_channels % int(cfg.model.bottleneck_heads) != 0:
                raise ValueError("coarse PointNet++ channels must be divisible by bottleneck_heads")
            if cfg.model.bottleneck_pos_enc not in {"input_features", "xyz", "none"}:
                raise ValueError("bottleneck_pos_enc must be 'input_features', 'xyz' or 'none'")
            if not (0.0 <= float(cfg.model.bottleneck_dropout) < 1.0):
                raise ValueError("bottleneck_dropout must be in [0, 1)")
        if cfg.model.coarse_attention and (
            coarse_channels % int(cfg.model.coarse_attention_heads) != 0
        ):
            raise ValueError(
                "coarse PointNet++ channels must be divisible by coarse_attention_heads"
            )
    elif (
        cfg.model.local_geope
        or cfg.model.local_branch
        or cfg.model.case_scale_head
        or cfg.model.coarse_attention
        or cfg.model.bottleneck_transformer
        or cfg.model.multiradius_values
        or cfg.model.multiradius_bottleneck
        or cfg.model.multiradius_ffn
        or cfg.model.local_transformer_stages
        or cfg.model.edgeconv_stages
    ):
        raise ValueError(
            "LocalGeoPE, EdgeConv, the local wall branch, the case-scale head and "
            "local/global attention require model.name='pointnetpp'"
        )
    grouping = tuple(cfg.model.sa_grouping)
    if grouping:
        if len(grouping) != len(cfg.model.sa_radius):
            raise ValueError("sa_grouping must match the number of SA stages")
        allowed_grouping = {"ball", "knn", "knn_cover", "adaptive_cover"}
        unknown = sorted(set(grouping) - allowed_grouping)
        if unknown:
            raise ValueError(f"unsupported sa_grouping values: {unknown}")
        if cfg.model.name != "pointnetpp":
            raise ValueError("custom sa_grouping currently requires model.name='pointnetpp'")
    if cfg.model.stem_channels:
        if cfg.model.name != "pointnetpp":
            raise ValueError("stem_channels currently requires model.name='pointnetpp'")
        if any(int(ch) <= 0 for ch in cfg.model.stem_channels):
            raise ValueError("stem_channels must all be positive")
        if int(cfg.model.stem_channels[-1]) != int(cfg.model.width):
            raise ValueError("stem_channels[-1] must equal model.width")
    if int(cfg.model.sa_adaptive_candidate_centers) < 2:
        raise ValueError("sa_adaptive_candidate_centers must be >= 2")
    if not (0.0 <= float(cfg.model.sa_overlap_cap) <= 1.0 / 3.0 + 1e-12):
        raise ValueError("sa_overlap_cap must be in [0, 1/3]")
    if cfg.eval.query_chunk_size < 0:
        raise ValueError("query_chunk_size must be >= 0")
    if cfg.eval.surface_metric_mode not in {"legacy_vertex", "both_strict"}:
        raise ValueError(
            "surface_metric_mode must be 'legacy_vertex' or 'both_strict'"
        )
    if float(cfg.train.loss_hotspot_bce_lambda) < 0:
        raise ValueError("loss_hotspot_bce_lambda must be non-negative")
    raw_mse_weight = float(cfg.train.loss_raw_mse_lambda)
    if not math.isfinite(raw_mse_weight) or raw_mse_weight < 0:
        raise ValueError("loss_raw_mse_lambda must be finite and non-negative")
    if raw_mse_weight and (
        cfg.data.target != "wss" or cfg.data.target_normalization != "global_stats"
        or cfg.train.loss == "gaussian_nll" or int(cfg.model.out_dim) != 1
    ):
        raise ValueError("Pa-MSE requires scalar WSS with global_stats and a non-NLL head")
    if cfg.train.init_reference_config and cfg.train.init_checkpoint_path:
        raise ValueError("fresh paired initialization and checkpoint warm-start are alternatives")
    if cfg.model.local_branch_modulation not in {"none", "gate", "additive"}:
        raise ValueError("local_branch_modulation must be none, gate or additive")
    if int(cfg.model.local_branch_modulation_hidden) < 1:
        raise ValueError("local_branch_modulation_hidden must be positive")
    if cfg.model.local_branch_modulation != "none" and (
        not cfg.model.local_branch or cfg.model.local_branch_mode != "neighborhood"
    ):
        raise ValueError("local branch modulation requires the neighbourhood wall branch")
    if float(cfg.train.loss_pinball_lambda) < 0:
        raise ValueError("loss_pinball_lambda must be non-negative")
    if not (0.0 < float(cfg.train.hotspot_quantile) < 1.0):
        raise ValueError("hotspot_quantile must be in (0, 1)")
    if not (0.0 < float(cfg.train.pinball_quantile) < 1.0):
        raise ValueError("pinball_quantile must be in (0, 1)")
    osi_tail_alpha = float(getattr(cfg.train, "loss_osi_tail_alpha", 0.0) or 0.0)
    osi_mask_lambda = float(getattr(cfg.train, "loss_osi_mask_lambda", 0.0) or 0.0)
    if osi_tail_alpha < 0 or osi_mask_lambda < 0:
        raise ValueError("loss_osi_tail_alpha and loss_osi_mask_lambda must be non-negative")
    if (osi_tail_alpha > 0 or osi_mask_lambda > 0) and cfg.data.target != MULTI_TARGET:
        raise ValueError("OSI tail / mask losses act on the OSI channel of target='wss_cycle_multi' only")
    if osi_tail_alpha > 0 and not (0.0 < float(cfg.train.osi_tail_threshold) < OSI_MAX):
        raise ValueError("osi_tail_threshold must be in (0, 0.5)")
    if osi_mask_lambda > 0:
        thresholds = tuple(cfg.train.osi_mask_thresholds or ())
        if not thresholds or any(not (0.0 < float(t) < OSI_MAX) for t in thresholds):
            raise ValueError("osi_mask_thresholds must be a non-empty list of OSI values in (0, 0.5)")
        if not (float(cfg.train.osi_mask_temperature) > 0):
            raise ValueError("osi_mask_temperature must be positive")
    if cfg.train.loss == "gaussian_nll" and (
        float(cfg.train.loss_hotspot_bce_lambda) > 0
        or float(cfg.train.loss_pinball_lambda) > 0
    ):
        raise ValueError(
            "hotspot/pinball auxiliaries are not supported with gaussian_nll"
        )
    if float(cfg.train.loss_hotspot_bce_lambda) > 0 and int(cfg.model.out_dim) != 2:
        raise ValueError("hotspot BCE requires model.out_dim=2")
    if cfg.data.target in CYCLE_TARGETS or cfg.data.target == MULTI_TARGET:
        if not cfg.data.cycle_view_root:
            raise ValueError(f"target={cfg.data.target!r} requires data.cycle_view_root (wss_min_cycle_v1 view)")
        if cfg.data.timesteps != "peak" or cfg.data.target_normalization != "global_stats":
            raise ValueError("cycle targets (tawss/osi/wss_cycle_multi) are single-frame scalars: timesteps='peak' with global_stats only")
        if (cfg.train.loss == "gaussian_nll" or float(cfg.train.loss_hotspot_bce_lambda) > 0 or cfg.model.direction_head
                or cfg.data.target_log_offset_feature or cfg.eval.pointwise_jensen or cfg.data.direction_target):
            raise ValueError("cycle targets support plain scalar heads only (no NLL/Jensen, hotspot BCE, direction head or residual offset)")
        if cfg.data.target == MULTI_TARGET:
            if (cfg.train.loss != "mse" or float(cfg.train.loss_pinball_lambda) > 0 or cfg.train.loss_weight_target
                    or float(cfg.train.loss_raw_huber_lambda or 0) > 0
                    or float(getattr(cfg.train, "loss_raw_mse_lambda", 0) or 0) > 0 or cfg.train.loss_geom_weight
                    or cfg.model.output_head not in (None, "single")):
                raise ValueError("wss_cycle_multi supports plain equal-weight channel MSE with a single head only "
                                 "(no pinball / target weighting / raw losses / geometry weighting)")
            aux = tuple(getattr(cfg.data, "multi_aux_channels", ()) or ())
            if len(set(aux)) != len(aux) or any(a not in MULTI_AUX_CHANNELS for a in aux):
                raise ValueError(f"data.multi_aux_channels must be distinct names from {MULTI_AUX_CHANNELS}")
            if ("mean_axial" in aux) != ("mean_circ" in aux):
                raise ValueError("mean_axial and mean_circ must be enabled together (the derived OSI needs both components)")
            if int(cfg.model.out_dim) != len(MULTI_CHANNELS) + len(aux):
                raise ValueError(f"wss_cycle_multi requires model.out_dim={len(MULTI_CHANNELS) + len(aux)} "
                                 f"({len(MULTI_CHANNELS)} channels + {len(aux)} auxiliary)")
    elif cfg.data.cycle_view_root:
        raise ValueError("data.cycle_view_root is only meaningful with target='tawss'|'osi'|'wss_cycle_multi'")
    if tuple(getattr(cfg.data, "multi_aux_channels", ()) or ()) and cfg.data.target != MULTI_TARGET:
        raise ValueError("data.multi_aux_channels requires target='wss_cycle_multi'")
    if not (0.0 < float(getattr(cfg.eval, "cycle_rel_tolerance", 0.3)) < 10.0) or int(getattr(cfg.eval, "cycle_min_segment_points", 200)) < 1 \
            or not (0.0 < float(getattr(cfg.eval, "cycle_osi_rel_floor", 0.01)) < 0.5):
        raise ValueError("eval.cycle_rel_tolerance must be in (0, 10), cycle_min_segment_points positive, cycle_osi_rel_floor in (0, 0.5)")
    for name in ("threshold_masks_above", "threshold_masks_below"):
        values = tuple(getattr(cfg.eval, name, ()) or ())
        if values and (cfg.data.target in VOLUME_TARGETS or cfg.data.target == "pressure"):
            raise ValueError(f"eval.{name} is defined for scalar wall targets (wss/tawss/osi) only")
        if any((not isinstance(v, (int, float))) or not math.isfinite(float(v)) for v in values):
            raise ValueError(f"eval.{name} must be finite numbers")
    expected_out = {"velocity": 3, "velocity_pressure": 4,
                    MULTI_TARGET: len(MULTI_CHANNELS) + len(tuple(getattr(cfg.data, "multi_aux_channels", ()) or ()))}.get(cfg.data.target, 1)
    if cfg.data.timesteps == "time_basis":
        # 时间基头：每个目标分量各出一套 [b0, a_1..a_K]（速度 3 分量共用同一组基向量）
        expected_out = expected_out * (int(cfg.model.time_basis_k) + 1)
    if (
        float(cfg.train.loss_hotspot_bce_lambda) == 0
        and cfg.train.loss != "gaussian_nll"
        and int(cfg.model.out_dim) != expected_out
    ):
        raise ValueError(
            f"target={cfg.data.target!r} without NLL/hotspot channels requires model.out_dim={expected_out}"
        )
    if cfg.eval.pointwise_jensen:
        if cfg.train.loss != "gaussian_nll" or int(cfg.model.out_dim) != 2:
            raise ValueError(
                "eval.pointwise_jensen needs the gaussian NLL head (loss='gaussian_nll', out_dim=2)"
            )
        if cfg.data.target_normalization != "global_stats":
            raise ValueError("eval.pointwise_jensen requires target_normalization='global_stats'")
    time_feats = [f for f in cfg.data.input_features if f in TIME_FEATURE_KEYS]
    if cfg.data.timesteps not in {"peak", "random_frame", "time_basis"}:
        raise ValueError("timesteps must be 'peak', 'random_frame' or 'time_basis'")
    if cfg.data.target_normalization == "frame_stats" and cfg.data.timesteps == "peak":
        raise ValueError("target_normalization='frame_stats' requires a multi-frame mode (random_frame / time_basis)")
    if (int(cfg.model.time_basis_k) > 0) != (cfg.data.timesteps == "time_basis"):
        raise ValueError("model.time_basis_k>0 if and only if data.timesteps='time_basis'")
    if cfg.data.timesteps == "time_basis":
        if not cfg.data.time_basis_path:
            raise ValueError("timesteps='time_basis' requires data.time_basis_path")
        if cfg.data.target not in {"wss", "pressure_mixed", "velocity"} or cfg.data.target_normalization != "frame_stats":
            raise ValueError("timesteps='time_basis' requires target='wss'|'pressure_mixed'|'velocity' "
                             "with target_normalization='frame_stats'")
        if int(cfg.model.out_dim) != expected_out:
            raise ValueError(f"time_basis head requires model.out_dim = {expected_out} "
                             "(target components × (time_basis_k + 1))")
        if cfg.train.loss not in {"mse", "huber"}:
            raise ValueError("time_basis head supports loss='mse' or 'huber' on the coefficient vector only")
        if any(float(getattr(cfg.train, key, 0.0) or 0.0) != 0.0 for key in (
                "loss_pinball_lambda", "loss_hotspot_bce_lambda", "loss_raw_huber_lambda",
                "loss_raw_mse_lambda", "loss_direction_lambda")) or cfg.train.loss_region_feature:
            raise ValueError("time_basis head does not support pinball/hotspot/raw-space/direction/region auxiliaries")
        if getattr(cfg.eval, "pointwise_jensen", False):
            raise ValueError("time_basis head has no NLL channel for pointwise_jensen")
        if cfg.eval.time_metrics and not cfg.data.waveform_path:
            raise ValueError("eval.time_metrics needs data.waveform_path (defines the low-flow frames)")
    if cfg.eval.time_metrics and cfg.data.timesteps == "peak":
        raise ValueError("eval.time_metrics needs a multi-frame mode")
    if time_feats and cfg.data.timesteps != "random_frame":
        raise ValueError(f"time features {time_feats} require timesteps='random_frame'")
    if cfg.data.timesteps == "random_frame":
        if not time_feats:
            raise ValueError("timesteps='random_frame' needs at least one TIME_FEATURE_KEYS entry in input_features")
        if not cfg.data.waveform_path:
            raise ValueError("timesteps='random_frame' requires data.waveform_path")
        if (cfg.data.target not in {"wss", "pressure_mixed", "velocity"}
                or cfg.data.target_normalization not in {"global_stats", "frame_stats"}):
            raise ValueError("timesteps='random_frame' supports target='wss'|'pressure_mixed'|'velocity' "
                             "with global_stats or frame_stats normalization")
        if cfg.eval.time_metrics and not cfg.data.waveform_path:
            raise ValueError("eval.time_metrics needs data.waveform_path (defines the low-flow frames)")
        if not (0.0 <= float(cfg.data.peak_frame_prob) <= 1.0):
            raise ValueError("peak_frame_prob must be in [0, 1]")
    if cfg.data.target in VOLUME_TARGETS:
        if cfg.data.query_mode != "independent":
            raise ValueError(f"target={cfg.data.target!r} needs query_mode='independent' (support=wall, query=volume pool)")
        if (cfg.data.support_sampling or cfg.data.sampling) != "random" or (cfg.data.query_sampling or "random") != "random":
            raise ValueError("volume targets only support random support/query sampling")
        if not (0.0 <= float(cfg.data.query_wall_fraction) <= 1.0):
            raise ValueError("query_wall_fraction must be in [0, 1]")
        if cfg.data.timesteps != "peak":
            # 体场全周期：81 帧标签不入内存，按帧从快照 case.h5 懒读，行映射来自 A0 的 sidecar
            if cfg.data.target not in {"pressure_mixed", "velocity"}:
                raise ValueError("multi-frame volume targets support 'pressure_mixed'|'velocity' only")
            if not cfg.data.volume_time_sidecar_root or not cfg.data.volume_h5_root:
                raise ValueError("multi-frame volume targets require data.volume_time_sidecar_root and data.volume_h5_root")
    if cfg.data.target == "velocity":
        if cfg.train.loss == "gaussian_nll" or float(cfg.train.loss_pinball_lambda) > 0 or cfg.train.loss_weight_target \
                or float(cfg.train.loss_hotspot_bce_lambda) > 0 or float(cfg.train.loss_raw_huber_lambda or 0) > 0:
            raise ValueError("velocity (vector) target supports plain MSE/Huber only")
    if cfg.data.target == "velocity_pressure":
        if (cfg.model.name != "pointnetpp" or cfg.model.output_head != "velocity_pressure"
                or int(cfg.model.out_dim) != 4 or cfg.model.query_decoder != "qad_lite"
                or not cfg.model.sa_center_counts or cfg.model.case_scale_head):
            raise ValueError("velocity_pressure requires PointNet++ joint 3+1 heads, fixed support, QAD and no case-scale head")
        if (cfg.data.timesteps != "peak" or cfg.data.target_normalization != "global_stats"
                or not cfg.eval.fixed_support or cfg.data.rot_aug or cfg.data.query_wall_fraction != 0.5):
            raise ValueError("velocity_pressure requires peak, global_stats, fixed eval support, pressure wall fraction 0.5 and no rotation augmentation")
        if (cfg.train.loss != "mse" or cfg.train.loss_geom_weight or cfg.train.loss_weight_target
                or cfg.train.loss_pinball_lambda or cfg.train.loss_hotspot_bce_lambda
                or cfg.train.loss_raw_huber_lambda or cfg.train.loss_raw_mse_lambda):
            raise ValueError("velocity_pressure supports only equally weighted masked task MSEs")
        if cfg.train.selection_rule != "train_loss" or cfg.train.ckpt_metric != "train_loss":
            raise ValueError("velocity_pressure uses one checkpoint selected by joint train_loss")


def input_dim(cfg: ExpConfig) -> int:
    return len(cfg.data.input_features)


def v6_point_features(cfg: ExpConfig) -> Tuple[str, ...]:
    """input_features 中需要额外加载/派生的 V6/V7 逐点特征（空 tuple = 旧数据路径）。"""
    return tuple(f for f in cfg.data.input_features
                 if f in V6_POINT_FEATURE_KEYS or f in V7_FLOW_FEATURE_KEYS or f in V8_CASCADE_FEATURE_KEYS
                 or f in LONGITUDINAL_FEATURE_KEYS or f in V9_PHYS1D_FEATURE_KEYS)


if __name__ == "__main__":
    # 冒烟：默认配置能否读写往返
    c = ExpConfig()
    p = c.to_json(RUNS_ROOT / "_tmp_config_check.json")
    c2 = ExpConfig.from_json(p)
    assert c2.model.sa_radius == c.model.sa_radius, "tuple 往返失败"
    assert input_dim(c2) == 3
    p.unlink()
    print("config OK:", c.name, "input_dim", input_dim(c2), "run_dir", c.run_dir)
