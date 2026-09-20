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
)
# V5 视图（data_wss_v5/views/wss_min_view_v1）额外逐点几何特征：管道坐标 rho/theta、半径梯度、
# 到分叉/端点距离、端区标记、对齐坐标架下的 PCA 法向；旧 bundle 不含这些键时 load_case 不会附加。
V5_POINT_FEATURE_KEYS = (
    "rho", "theta_sin", "theta_cos", "dr_ds", "dist_to_junction_mm", "dist_to_endpoint_mm",
    "end_zone", "nx_aligned", "ny_aligned", "nz_aligned",
    "dist_to_wall_mm",  # volume targets: 0 on wall points, cell-to-wall distance for interior points
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
# 与 curvature 同口径处理的重尾曲率族：signed_log1p + p99 截断
CURVATURE_LIKE_KEYS = (
    "curvature", "curv_k1", "curv_k2", "curv_mean", "curv_gauss", "curvedness",
    "curv_k1_c", "curv_k2_c", "curv_mean_c", "curv_gauss_c", "curvedness_c",
)

FEATURE_KEYS = (
    "x", "y", "z", "abscissa_norm",
    "local_radius", "log_local_radius", "curvature", "coord_scale",
    "radius_gradient",
    *V5_POINT_FEATURE_KEYS,
    *V6_POINT_FEATURE_KEYS,
    *TIME_FEATURE_KEYS,
    *COHORT_FEATURE_KEYS,
    *CASE_FEATURE_KEYS,
)

# 禁止作为输入的常量/废弃列
FORBIDDEN_FEATURE_KEYS = ("dist_to_wall",)

# targets whose query set comes from the volume view (support stays the wall point cloud)
VOLUME_TARGETS = ("pressure_mixed", "velocity")


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
    # None 保持旧行为；只有 input_features 里出现 V6_SURFACE_FEATURE_KEYS 时才需要。
    point_features_root: Optional[str] = None
    # 训练稀疏化：每例采样多少壁面点；<=0 或 >=全量 表示用全部点
    wall_n_points: int = 2000
    sampling: str = "fps"                 # 'fps' | 'fps_multistart' | 'random' | 'geom_weighted'
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
    timesteps: str = "peak"               # 'peak' | 'random_frame'
    waveform_path: Optional[str] = None   # 协议入口波形 JSON（steps/q_norm/dq_norm/t_sin/t_cos/t_norm）
    peak_frame_prob: float = 1.0 / 9.0    # random_frame 训练时抽到峰值帧的保底概率
    target: str = "wss"                   # 'wss' | 'pressure'（均为标量）
    # 目标标准化；global_stats 保持历史行为，case_max 用逐病例完整壁面 WSS 最大值。
    target_normalization: str = "global_stats"  # 'global_stats' | 'case_max'
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
    out_dim: int = 1                      # 标量=1；矢量=3（二期）；NLL=2
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
    amp: bool = True
    eval_every: int = 10                  # 每多少 epoch 在 val 完整点云上评估一次
    ckpt_metric: str = "val_selection_score"  # 兼容旧名；实际用 selection_rule
    selection_rule: str = "r4_composite_v1"
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
    for f in cfg.data.input_features:
        if f in FORBIDDEN_FEATURE_KEYS:
            raise ValueError(
                f"feature {f!r} is forbidden (constant on wall points); "
                f"remove it from input_features"
            )
        if f not in FEATURE_KEYS:
            raise ValueError(f"unknown feature {f!r}; allowed={FEATURE_KEYS}")
    requested_case_features = set(cfg.data.input_features) & set(CASE_FEATURE_KEYS)
    if requested_case_features and not cfg.data.case_features_path:
        raise ValueError(
            "case-level input features require data.case_features_path; "
            f"requested={sorted(requested_case_features)}"
        )
    requested_surface_features = set(cfg.data.input_features) & set(V6_SURFACE_FEATURE_KEYS)
    if requested_surface_features and not cfg.data.point_features_root:
        raise ValueError(
            "local surface-geometry features require data.point_features_root "
            f"(wall_geom_v2 sidecar); requested={sorted(requested_surface_features)}"
        )
    if cfg.data.target_normalization not in {"global_stats", "case_max"}:
        raise ValueError(
            "target_normalization must be 'global_stats' or 'case_max', got "
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
    if support_sampling not in {"fps", "fps_multistart", "random", "geom_weighted", "area_random"}:
        raise ValueError(f"unsupported support sampling {support_sampling!r}")
    if cfg.data.support_allow_undersized and support_sampling not in {"random", "area_random"}:
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
        if (cfg.data.query_sampling or support_sampling) not in {"random", "area_random"}:
            raise ValueError("independent query sampling must be 'random' or 'area_random'")
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
    if cfg.train.loss == "gaussian_nll" and (
        float(cfg.train.loss_hotspot_bce_lambda) > 0
        or float(cfg.train.loss_pinball_lambda) > 0
    ):
        raise ValueError(
            "hotspot/pinball auxiliaries are not supported with gaussian_nll"
        )
    if float(cfg.train.loss_hotspot_bce_lambda) > 0 and int(cfg.model.out_dim) != 2:
        raise ValueError("hotspot BCE requires model.out_dim=2")
    expected_out = 3 if cfg.data.target == "velocity" else 1
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
    if cfg.data.timesteps not in {"peak", "random_frame"}:
        raise ValueError("timesteps must be 'peak' or 'random_frame'")
    if time_feats and cfg.data.timesteps != "random_frame":
        raise ValueError(f"time features {time_feats} require timesteps='random_frame'")
    if cfg.data.timesteps == "random_frame":
        if not time_feats:
            raise ValueError("timesteps='random_frame' needs at least one TIME_FEATURE_KEYS entry in input_features")
        if not cfg.data.waveform_path:
            raise ValueError("timesteps='random_frame' requires data.waveform_path")
        if cfg.data.target != "wss" or cfg.data.target_normalization != "global_stats":
            raise ValueError("timesteps='random_frame' currently supports target='wss' with global_stats normalization")
        if not (0.0 <= float(cfg.data.peak_frame_prob) <= 1.0):
            raise ValueError("peak_frame_prob must be in [0, 1]")
    if cfg.data.target in VOLUME_TARGETS:
        if cfg.data.query_mode != "independent":
            raise ValueError(f"target={cfg.data.target!r} needs query_mode='independent' (support=wall, query=volume pool)")
        if (cfg.data.support_sampling or cfg.data.sampling) != "random" or (cfg.data.query_sampling or "random") != "random":
            raise ValueError("volume targets only support random support/query sampling")
        if not (0.0 <= float(cfg.data.query_wall_fraction) <= 1.0):
            raise ValueError("query_wall_fraction must be in [0, 1]")
    if cfg.data.target == "velocity":
        if cfg.train.loss == "gaussian_nll" or float(cfg.train.loss_pinball_lambda) > 0 or cfg.train.loss_weight_target \
                or float(cfg.train.loss_hotspot_bce_lambda) > 0 or float(cfg.train.loss_raw_huber_lambda or 0) > 0:
            raise ValueError("velocity (vector) target supports plain MSE/Huber only")


def input_dim(cfg: ExpConfig) -> int:
    return len(cfg.data.input_features)


def v6_point_features(cfg: ExpConfig) -> Tuple[str, ...]:
    """input_features 中需要额外加载/派生的 V6 逐点特征（空 tuple = 旧数据路径）。"""
    return tuple(f for f in cfg.data.input_features if f in V6_POINT_FEATURE_KEYS)


if __name__ == "__main__":
    # 冒烟：默认配置能否读写往返
    c = ExpConfig()
    p = c.to_json(RUNS_ROOT / "_tmp_config_check.json")
    c2 = ExpConfig.from_json(p)
    assert c2.model.sa_radius == c.model.sa_radius, "tuple 往返失败"
    assert input_dim(c2) == 3
    p.unlink()
    print("config OK:", c.name, "input_dim", input_dim(c2), "run_dir", c.run_dir)
