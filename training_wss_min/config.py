#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""training_wss_min 实验配置（配置驱动 baseline）。

一个实验 = 一个 JSON 配置。用 `ExpConfig.from_json` 读、`to_json` 写；
训练/评估只读取已登记的 config 路径，不在运行时生成或改写源配置。

约定：所有相对路径都相对仓库根 `PROJECT_ROOT`；产物落在 `runs/<name>/`。
"""

from __future__ import annotations

import json
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
FEATURE_KEYS = (
    "x", "y", "z", "abscissa_norm",
    "local_radius", "curvature", "coord_scale", "radius_gradient",
    *COHORT_FEATURE_KEYS,
)

# 禁止作为输入的常量/废弃列
FORBIDDEN_FEATURE_KEYS = ("dist_to_wall",)


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
    # 训练稀疏化：每例采样多少壁面点；<=0 或 >=全量 表示用全部点
    wall_n_points: int = 2000
    sampling: str = "fps"                 # 'fps' | 'fps_multistart' | 'random' | 'geom_weighted'
    # Support/Query 新协议；None 时严格回退上面的旧字段。
    support_n_points: Optional[int] = None
    support_sampling: Optional[str] = None  # 'fps' | 'random' | 'area_random'
    query_mode: str = "same"               # 'same' | 'independent'
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
    # 时间步：baseline 用峰值收缩期单步
    timesteps: str = "peak"               # 'peak'（其余暂不支持，占位）
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
    query_decoder: str = "interpolate"     # interpolate | qad_lite | sep_kernel
    qad_hidden: int = 32
    sep_hidden: int = 32
    sep_beta_init: float = 2.0
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
    # 只在最后一级 SA（D2 为 32 centers）使用一个相对几何 bias 全局块。
    # 该模块已经等价承担“SA3 全局 Transformer”，不要再叠加同义开关。
    coarse_attention: bool = False
    coarse_attention_heads: int = 4
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
    # NLL（C3）
    nll_logvar_min: float = -6.0
    nll_logvar_max: float = 2.0
    nll_logvar_reg: float = 1e-4
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
    if cfg.model.query_decoder not in {"interpolate", "qad_lite", "sep_kernel"}:
        raise ValueError(
            "query_decoder must be 'interpolate', 'qad_lite', or 'sep_kernel'"
        )
    if int(cfg.model.qad_hidden) <= 0:
        raise ValueError("qad_hidden must be positive")
    if int(cfg.model.sep_hidden) <= 0:
        raise ValueError("sep_hidden must be positive")
    if float(cfg.model.sep_beta_init) < 0:
        raise ValueError("sep_beta_init must be non-negative")
    if cfg.model.query_decoder in {"qad_lite", "sep_kernel"}:
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
        if int(cfg.model.coarse_attention_heads) <= 0:
            raise ValueError("coarse_attention_heads must be positive")
        coarse_channels = int(cfg.model.width) * (2 ** len(cfg.model.sa_radius))
        if cfg.model.coarse_attention and (
            coarse_channels % int(cfg.model.coarse_attention_heads) != 0
        ):
            raise ValueError(
                "coarse PointNet++ channels must be divisible by coarse_attention_heads"
            )
    elif (
        cfg.model.local_geope
        or cfg.model.coarse_attention
        or cfg.model.local_transformer_stages
    ):
        raise ValueError(
            "LocalGeoPE and local/global attention require model.name='pointnetpp'"
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


def input_dim(cfg: ExpConfig) -> int:
    return len(cfg.data.input_features)


if __name__ == "__main__":
    # 冒烟：默认配置能否读写往返
    c = ExpConfig()
    p = c.to_json(RUNS_ROOT / "_tmp_config_check.json")
    c2 = ExpConfig.from_json(p)
    assert c2.model.sa_radius == c.model.sa_radius, "tuple 往返失败"
    assert input_dim(c2) == 3
    p.unlink()
    print("config OK:", c.name, "input_dim", input_dim(c2), "run_dir", c.run_dir)
