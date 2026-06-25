from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from pipeline.config import GLOBAL_COND_NAMES, NODE_FEATURE_NAMES, TARGET_NAMES, WSS_TARGET_NAMES
from pipeline.config import WSS_LOCAL_COMPONENT_NAMES, WSS_LOCAL_TARGET_NAMES


def resolve_wss_target_names(wss_target_frame: str, wss_dim: int) -> List[str]:
    """按 frame 与 wss_dim 解析 WSS 监督/指标名列表。"""
    if wss_target_frame == "local":
        if wss_dim == 3:
            return list(WSS_LOCAL_COMPONENT_NAMES)
        if wss_dim == 4:
            return list(WSS_LOCAL_TARGET_NAMES)
        raise ValueError(f"local frame 下 wss_dim 须为 3 或 4，收到 {wss_dim}")
    if wss_dim == 1:
        return ["wss"]
    if wss_dim != len(WSS_TARGET_NAMES):
        raise ValueError(
            f"global frame 下 wss_dim 须为 1（magnitude-only）或 {len(WSS_TARGET_NAMES)}，收到 {wss_dim}"
        )
    return list(WSS_TARGET_NAMES)


def resolve_wss_target_column_indices(wss_target_frame: str, wss_dim: int) -> List[int]:
    """从图数据 y_wss 全列中选取与 wss_dim / frame 对应的列索引。"""
    names = resolve_wss_target_names(wss_target_frame, wss_dim)
    if wss_target_frame == "local":
        local_names = list(WSS_LOCAL_TARGET_NAMES)
        return [local_names.index(n) for n in names]
    return [WSS_TARGET_NAMES.index(n) for n in names]


def resolve_wss_effective_dim(
    wss_dim: int,
    wss_output_mode: str = "head",
    wss_metric_dim: int = 1,
) -> int:
    """训练/评估时 WSS 指标与 y_wss 对齐的有效维度（vel_diff 模式无 wss_head）。"""
    if wss_output_mode == "vel_diff":
        return wss_metric_dim
    return wss_dim


def resolve_wss_runtime_names(
    wss_target_frame: str,
    wss_dim: int,
    wss_output_mode: str = "head",
    wss_metric_dim: int = 1,
) -> List[str]:
    """按 model 配置解析 WSS 监督/指标名；vel_diff 时用 wss_metric_dim。"""
    eff = resolve_wss_effective_dim(wss_dim, wss_output_mode, wss_metric_dim)
    if eff <= 0:
        return []
    return resolve_wss_target_names(wss_target_frame, eff)


@dataclass
class RunConfig:
    # 训练 run 的名字会进入输出目录名，因此这里最好保持“任务-模型-特征集”可读。
    experiment_name: str
    output_root: str = "outputs/field"
    save_every: int = 10
    save_best_only: bool = True
    init_checkpoint: Optional[str] = None
    # G3 SSL 微调：仅加载 FieldPointNeXt backbone（in_proj/blocks/shared_decoder），head 随机初始化。
    pretrained_encoder: Optional[str] = None


@dataclass
class DataConfig:
    # data_root 指向病例目录的根；graphs_subdir 指向每个病例下已经构好的图数据目录。
    data_root: str
    split_file: str
    graphs_subdir: str = "processed/graphs"
    preload: bool = False
    batch_size: int = 2
    num_workers: int = 0
    pin_memory: bool = False
    enabled_node_features: List[str] = field(
        default_factory=lambda: NODE_FEATURE_NAMES.copy()
    )
    # 通过特征名而不是硬编码索引控制消融，避免改图文件结构。
    enabled_global_features: List[str] = field(
        default_factory=lambda: GLOBAL_COND_NAMES.copy()
    )
    augment: bool = True
    augment_config: Dict[str, Any] = field(default_factory=dict)
    # WSS 监督坐标系：global（默认，向后兼容）或 local（local_v1 口径）。
    wss_target_frame: str = "global"
    # TODO-19: WSS 目标按数据域重新标准化。none=关闭；per_domain=按 AAA/AG/ILO 统计。
    wss_domain_norm: str = "none"
    # 形如 {"AAA": {"mean": [...], "std": [...]}, ...}，维度与当前 y_wss 列一致。
    wss_domain_norm_stats: Dict[str, Dict[str, List[float]]] = field(default_factory=dict)


@dataclass
class ModelConfig:
    name: str = "transformer"
    hidden_dim: int = 128
    num_layers: int = 3
    dropout: float = 0.1
    heads: int = 4
    use_transformer_prenorm: bool = False
    # WSS 预测头输出维度。0 = 不启用 WSS 头，4 = [wss, wss_x, wss_y, wss_z]。
    wss_dim: int = 0
    # V3: 输出头结构。"single_linear" = 单层 nn.Linear（默认，与旧 checkpoint 兼容）；
    # "mlp2" = 2 层 MLP（含 LayerNorm + GELU），仅对 FieldPointNeXt 生效。
    head_layout: str = "single_linear"
    # V3: 仅 wss_head 在 mlp2 中间层使用的 Dropout 概率；0 = 不插入（与旧 checkpoint 兼容）。
    wss_head_dropout: float = 0.0
    # TODO-17: 近壁速度梯度上下文，仅拼接进 wss_head 输入（不改 field_head）。
    wss_vel_context: bool = False
    wss_vel_context_dim: int = 4
    # 思路2 输入侧：壁面 |∇p| 仅拼接进 wss_head 输入（非 loss）；默认关闭。
    wss_pgrad_context: bool = False
    # mag=旧口径 1 维 |∇p|；rich=拼接 [gx, gy, gz, |∇p|]，承接 PC-v2 结构探针。
    wss_pgrad_context_mode: str = "mag"
    # G2-b：壁面 query × 近壁内部 (p, Δp, vel) kernel-attention；替代 concat 版 wss_pgrad_context。
    wss_kernel_attention: bool = False
    wss_kernel_attn_dim: int = 64
    wss_kernel_attn_heads: int = 4
    # G1-a0：Vector-Neuron 等变 WSS 头（VNHeadPlain）；替代标量 wss_head。
    wss_vn_head: bool = False
    wss_vn_channels: int = 32
    wss_vn_layers: int = 2
    # 近壁速度差分代理作为 direct WSS head 侧带输入；比 vel_diff 直接输出更柔性，默认关闭。
    wss_boundary_layer_context: bool = False
    wss_boundary_layer_variant: str = "tang_normal"
    # TODO-30 结构版：WSS 输出方式。"head"=wss_head（默认）；"vel_diff"=近壁速度差分推 |WSS|。
    wss_output_mode: str = "head"
    # vel_diff 模式下 WSS 指标/合成输出维度（通常 1=magnitude-only）；head 模式忽略。
    wss_metric_dim: int = 1
    # vel_diff 差分变体："naive"=|Δvel|/欧氏距（5276 旧口径）；"tang_normal"=切向速度/法向距（oracle v2）。
    vel_diff_variant: str = "naive"
    # TODO-18: 多档 k 邻域 mean+max pool；空列表=仅用 data.edge_index（与旧 checkpoint 兼容）。
    pool_k_tiers: List[int] = field(default_factory=list)


@dataclass
class DomainLossConfig:
    """V3 双域 mask loss 配置。enabled=False 时完全不影响旧训练路径。"""
    enabled: bool = False
    lambda_vel_int: float = 0.3
    lambda_vel_noslip: float = 0.1
    lambda_p_int: float = 0.5
    lambda_p_wall: float = 1.0
    lambda_wss: float = 0.1
    # TODO-9：4 维 head 模长一致性 |pred[wss] − √(x²+y²+z²)|；0=关闭（母版默认）。
    lambda_wss_mag_consist: float = 0.0
    # TODO-42：高 WSS top-k ListNet 排序辅助 loss；0=关闭。
    lambda_wss_rank: float = 0.0
    wss_rank_top_frac: float = 0.10
    wss_rank_max_nodes: int = 512
    # TODO-30：预测 WSS 模长 ↔ 近壁预测速度差分代理 的逐图标准化模式一致性；0=关闭。
    #          需要速度被监督（lambda_vel_int/noslip>0）才有意义。
    lambda_wss_vel_consist: float = 0.0
    # TODO-33：近壁预测速度差分代理 ↔ GT |WSS| 的逐图标准化模式监督；0=关闭。
    #          直接塑造预测速度场的近壁梯度，使其差分出正确 WSS 模式。
    lambda_wss_slope: float = 0.0
    # 思路 2：预测 WSS 模长 ↔ 壁面预测 |∇p| 的逐图标准化模式一致性；0=关闭。
    #          仅依赖压力（已 R²≈0.96），不需要速度监督。
    lambda_wss_pgrad_consist: float = 0.0
    normalize_by_target_std: bool = False
    norm_consts: Dict[str, float] = field(default_factory=dict)
    weight_calibration: str = ""


@dataclass
class OptimConfig:
    # target_weights 对应 [u, v, w, p] 四个输出维度的损失权重。
    epochs: int = 200
    lr: float = 5e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 0
    scheduler_factor: float = 0.5
    scheduler_patience: int = 10
    early_stopping_patience: int = 30
    # 复合 val_score 早停：仅当新分数低于历史最佳超过该阈值才算刷新（0=任意下降）。
    early_stop_min_delta: float = 0.0
    # 双域 val_score 的 EMA 系数；0=关闭，早停与 best 模型完全基于原始 val_score。
    val_score_ema_alpha: float = 0.0
    # I6 诊断：每 N 个训练 step 记录共享 backbone 上 cos(∇L_p, ∇L_wss) 与梯度范数比；
    # 默认关闭（仅诊断 run 打开，额外 2 次反向传播，常规训练不受影响）。
    i6_grad_probe: bool = False
    i6_grad_probe_interval: int = 50
    target_weights: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0])
    interior_loss_boost: float = 1.0
    grad_clip_norm: Optional[float] = 1.0
    accumulate_grad_batches: int = 1
    # WSS 多任务损失：L_total = field_loss + wss_loss_weight * wss_loss
    # wss_weights 对应 [wss, wss_x, wss_y, wss_z] 四个维度的损失权重。
    wss_loss_weight: float = 0.0
    wss_weights: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0])
    # val_score 中 WSS 分项的加权系数，顺序同 WSS_TARGET_NAMES。
    # 默认 [1,1,1,1] 保持与历史均匀聚合等价；非对称实验可与 wss_weights 同步设置。
    val_score_wss_weights: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0])
    # 早停/模型选择的混合指标权重。> 0 时验证分数为
    # data_loss + early_stop_wss_weight * wss_loss，取代默认的 total loss。
    early_stop_wss_weight: float = 0.0
    # WSS 监督项形式：mse（默认）或 huber / smooth_l1（PyTorch Smooth L1，beta 见下）。
    wss_loss_type: str = "mse"
    wss_huber_beta: float = 1.0
    # V3 双域 mask loss（默认关闭，旧配置不受影响）。
    domain_loss: DomainLossConfig = field(default_factory=DomainLossConfig)
    # G3 SSL 微调：encoder backbone 学习率 = lr × encoder_lr_ratio（head 仍用 lr）。
    encoder_lr_ratio: float = 1.0
    # V1 PINN：True 时 best/早停按 val data_loss 选优（physics 仅作训练正则，不污染选模型）。
    # 仅在非双域路径生效；默认 False 保持旧行为（按 total loss 选优）。
    select_best_on_data_loss: bool = False
    # I6-a 两阶段：warm-start 后冻结 backbone（in_proj/blocks/shared_decoder）。
    freeze_backbone: bool = False
    # I6-a 两阶段：冻结 field_head，仅续训 wss_head（须 init_checkpoint）。
    freeze_field_head: bool = False


@dataclass
class SystemConfig:
    # deterministic=True 更适合论文实验；如果后面追求吞吐量，可以在服务器上再放宽。
    seed: int = 1
    device: str = "auto"
    deterministic: bool = True
    amp: bool = False


@dataclass
class MetaConfig:
    # meta 不参与模型训练本身，只服务实验追踪、命名、后处理和导出。
    task: str = "field"
    exp_id: str = ""
    stage: str = "task_a"
    study_group: str = ""
    question: str = ""
    feature_set: str = ""
    ablation_axis: str = ""
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    generated_from: str = ""


@dataclass
class PhysicsConfig:
    enabled: bool = False
    warmup_epochs: int = 0
    density: float = 1060.0
    viscosity: float = 0.0035
    coord_scales: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    time_scale: float = 1.0
    continuity_weight: float = 0.0
    momentum_weight: float = 0.0
    no_slip_weight: float = 0.0
    auto_load_scales: bool = True
    # --- V1 PINN 扩展（默认值=旧行为，对现有 V1/V3 路径零影响）---
    # N-S 方程形式："unsteady"（旧默认，含 ∂/∂t）或 "steady"（去掉时间项，按帧当稳态）。
    equation: str = "unsteady"
    # True 时在算残差前把预测 u/v/w/p 反归一化到物理单位（SI），使连续性/动量量纲闭合。
    denormalize_fields: bool = False
    # 形如 {"u": {"mean":..,"std":..}, "v":..., "w":..., "p":...}；
    # auto_load_scales 时由 normalization_params_global.json 自动填充。
    field_norm_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # 坐标物理单位换算到米的系数。坐标以 mm 存储时设 0.001，使导数落在 SI（1/m）。
    length_unit_to_meter: float = 1.0
    # no-slip 目标："normalized_zero"（旧，逼归一化速度→0）或
    # "physical_zero"（逼反归一化壁面速度→物理 0，需 field_norm_stats）。
    no_slip_mode: str = "normalized_zero"
    # 动态权重 omega（论文式）：True 时每 N 轮按 data/physics loss 比例刷新总物理项权重。
    dynamic_weight: bool = False
    dynamic_weight_update_every: int = 10
    omega_init: float = 1.0
    # physics 分支最大节点数；0=全图。>0 时随机子采样以控 PINN 反传显存。
    max_physics_nodes: int = 0
    # 验证阶段是否计算 physics 残差（需坐标 autograd）。
    # false（默认）：val 仅 data+wss，与 trainer 的 no_grad 验证路径及
    # optim.select_best_on_data_loss 兼容；true 时在内层 enable_grad 下算 physics（监控用）。
    eval_physics_on_val: bool = False

    def resolve_scales_from_data(self, data_root: str, graphs_subdir: str, case_dirs: list) -> None:
        """从 pipeline 产物自动加载物理损失所需的坐标/时间尺度与场归一化统计。

        coord_normalize 会为每个病例保存 ``transform_params.json``，其中
        含有该病例的 ``scale_factor``。物理损失在归一化坐标上求导后，
        需要用这些尺度还原回原始物理空间，否则 continuity / momentum
        的量纲会失真。这里读取可用病例的尺度，并用中位数作为稳健代表值。

        当 ``denormalize_fields`` 开启且 ``field_norm_stats`` 为空时，
        从 ``normalization_params_global.json`` 读取 u/v/w/p 的 z-score
        mean/std，供把预测反归一化回物理单位。
        """
        # 如果没启用自动读取或物理损失本身关闭，就直接返回。
        if not self.auto_load_scales or not self.enabled:
            return

        # 函数内部局部导入，避免配置模块在普通场景下引入额外依赖。
        import json
        from pathlib import Path

        # 收集训练病例里可用的坐标缩放因子。
        scale_factors: List[float] = []
        for case_dir in case_dirs:
            # 每个病例的坐标归一化参数都保存在固定位置。
            params_path = Path(case_dir) / "processed" / "coord_normalized" / "transform_params.json"
            # 缺文件就跳过，不阻塞整个实验。
            if not params_path.exists():
                continue
            try:
                with open(params_path, "r", encoding="utf-8") as f:
                    params = json.load(f)
                # 读出当前病例的空间尺度因子。
                sf = params.get("scale_factor", 1.0)
                # 过滤掉明显非法的非正尺度。
                if sf > 1e-6:
                    scale_factors.append(sf)
            except (json.JSONDecodeError, KeyError):
                # 单个病例参数损坏时忽略，尽量让训练继续。
                continue

        # 如果收集到了有效尺度，就用中位数作为三轴共同尺度。
        if scale_factors:
            import statistics
            median_sf = statistics.median(scale_factors)
            self.coord_scales = [median_sf, median_sf, median_sf]

        # t_norm 的标准差保存在全局归一化参数里，用于把 dt 还原回原始时间尺度。
        norm_params_path = Path(data_root) / "normalization_params_global.json"
        # 全局归一化参数存在时，再尝试恢复时间尺度（以及场归一化统计）。
        if norm_params_path.exists():
            try:
                with open(norm_params_path, "r", encoding="utf-8") as f:
                    norm_params = json.load(f)
                stats = norm_params.get("statistics", {})
                t_stats = stats.get("t_norm")
                # 如果能读到 t_norm 的标准差，就把它当作时间缩放尺度。
                if t_stats and t_stats.get("std", 0) > 1e-10:
                    self.time_scale = t_stats["std"]
                # 反归一化所需的场统计；仅在需要且未显式提供时自动填充。
                if self.denormalize_fields and not self.field_norm_stats:
                    loaded: Dict[str, Dict[str, float]] = {}
                    for name in ("u", "v", "w", "p"):
                        s = stats.get(name)
                        if s and "mean" in s and "std" in s:
                            loaded[name] = {"mean": float(s["mean"]), "std": float(s["std"])}
                    if loaded:
                        self.field_norm_stats = loaded
            except (json.JSONDecodeError, KeyError):
                # 时间归一化参数读失败时保持默认值 1.0。
                pass


@dataclass
class ExperimentConfig:
    # 训练入口只接受这一层统一配置，避免 CLI 参数逐步失控。
    run: RunConfig
    data: DataConfig
    model: ModelConfig
    optim: OptimConfig
    system: SystemConfig = field(default_factory=SystemConfig)
    meta: MetaConfig = field(default_factory=MetaConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        # 这里显式展开每个子配置，便于后续新增字段时保持类型边界清晰。
        # OptimConfig 内含嵌套的 DomainLossConfig，需要先 pop 出来单独构造。
        optim_raw = dict(data["optim"])
        domain_loss_raw = optim_raw.pop("domain_loss", None)
        optim_obj = OptimConfig(**optim_raw)
        if domain_loss_raw is not None:
            optim_obj.domain_loss = DomainLossConfig(**domain_loss_raw)
        return cls(
            run=RunConfig(**data["run"]),
            data=DataConfig(**data["data"]),
            model=ModelConfig(**data["model"]),
            optim=optim_obj,
            system=SystemConfig(**data.get("system", {})),
            meta=MetaConfig(**data.get("meta", {})),
            physics=PhysicsConfig(**data.get("physics", {})),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "ExperimentConfig":
        # 从 JSON 文件读取原始配置字典。
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        # 再委托给 from_dict 构造强类型配置对象。
        return cls.from_dict(raw)

    def to_dict(self) -> Dict[str, Any]:
        # dataclass 原生转字典，便于快照和导出。
        return asdict(self)

    def validate(self) -> None:
        # 训练阶段只做“特征屏蔽”，因此这里必须保证名字和 pipeline 导出的 schema 一致。
        # 检查节点特征名是否都在 schema 中。
        unknown_node = sorted(set(self.data.enabled_node_features) - set(NODE_FEATURE_NAMES))
        # 检查全局特征名是否都在 schema 中。
        unknown_global = sorted(
            set(self.data.enabled_global_features) - set(GLOBAL_COND_NAMES)
        )
        if unknown_node:
            raise ValueError(f"未知节点特征: {unknown_node}")
        if unknown_global:
            raise ValueError(f"未知全局特征: {unknown_global}")
        # 模型名必须属于注册表里支持的骨干网络。
        if self.model.name not in {"mlp", "graphsage", "transformer", "meshgraphnet", "pointnetpp", "pointnext"}:
            raise ValueError(f"不支持的模型: {self.model.name}")
        # 监督损失权重维度必须与目标维度一致。
        if len(self.optim.target_weights) != len(TARGET_NAMES):
            raise ValueError("target_weights 维度必须与目标输出一致")
        eff_wss_dim = resolve_wss_effective_dim(
            self.model.wss_dim,
            self.model.wss_output_mode,
            self.model.wss_metric_dim,
        )
        wss_names = (
            resolve_wss_runtime_names(
                self.data.wss_target_frame,
                self.model.wss_dim,
                self.model.wss_output_mode,
                self.model.wss_metric_dim,
            )
            if eff_wss_dim > 0
            else []
        )
        if eff_wss_dim > 0:
            if len(self.optim.wss_weights) != len(wss_names):
                raise ValueError(
                    f"wss_weights 维度 ({len(self.optim.wss_weights)}) 须与 "
                    f"{self.data.wss_target_frame} frame 目标 ({len(wss_names)}) 一致"
                )
            if len(self.optim.val_score_wss_weights) != len(wss_names):
                raise ValueError(
                    f"val_score_wss_weights 维度须与 {self.data.wss_target_frame} frame 目标一致"
                )
        if self.data.wss_target_frame not in ("global", "local"):
            raise ValueError("data.wss_target_frame 须为 global 或 local")
        if self.data.wss_domain_norm not in ("none", "per_domain"):
            raise ValueError("data.wss_domain_norm 须为 none 或 per_domain")
        if self.data.wss_domain_norm == "per_domain":
            if self.data.wss_target_frame != "global":
                raise ValueError("data.wss_domain_norm=per_domain 当前仅支持 global WSS 目标")
            if not self.data.wss_domain_norm_stats:
                raise ValueError("data.wss_domain_norm=per_domain 时必须提供 wss_domain_norm_stats")
            for domain, stats in self.data.wss_domain_norm_stats.items():
                if "mean" not in stats or "std" not in stats:
                    raise ValueError(f"{domain} 缺少 mean/std")
                if len(stats["mean"]) != len(WSS_TARGET_NAMES) or len(stats["std"]) != len(WSS_TARGET_NAMES):
                    raise ValueError(f"{domain} mean/std 维度须为 {len(WSS_TARGET_NAMES)}")
                if any(float(v) <= 1e-10 for v in stats["std"]):
                    raise ValueError(f"{domain} std 必须为正")
        if any(w < 0 for w in self.optim.val_score_wss_weights):
            raise ValueError("val_score_wss_weights 不得为负")
        if sum(self.optim.val_score_wss_weights) <= 0:
            raise ValueError("val_score_wss_weights 权重之和须 > 0")
        if self.model.wss_dim > 0 and self.data.wss_target_frame == "global":
            if self.model.wss_dim not in (1, len(WSS_TARGET_NAMES)):
                raise ValueError(
                    f"global frame 下 wss_dim 须为 0、1（magnitude-only）或 {len(WSS_TARGET_NAMES)}"
                )
        if self.model.wss_dim > 0 and self.data.wss_target_frame == "local":
            if self.model.wss_dim not in (3, 4):
                raise ValueError("local frame 下 wss_dim 须为 3 或 4")
        # 物理坐标尺度必须提供 3 个值，对应 x/y/z。
        if len(self.physics.coord_scales) != 3:
            raise ValueError("physics.coord_scales 必须包含 3 个坐标尺度")
        if self.physics.equation not in ("steady", "unsteady"):
            raise ValueError("physics.equation 须为 steady 或 unsteady")
        if self.physics.no_slip_mode not in ("normalized_zero", "physical_zero"):
            raise ValueError("physics.no_slip_mode 须为 normalized_zero 或 physical_zero")
        if self.physics.length_unit_to_meter <= 0:
            raise ValueError("physics.length_unit_to_meter 必须 > 0")
        if self.physics.dynamic_weight and self.physics.dynamic_weight_update_every < 1:
            raise ValueError("physics.dynamic_weight_update_every 必须 >= 1")
        if self.physics.max_physics_nodes < 0:
            raise ValueError("physics.max_physics_nodes 必须 >= 0")
        # field_norm_stats 若显式提供，须含 mean/std；为空时由 resolve_scales_from_data 自动加载。
        for ch, st in self.physics.field_norm_stats.items():
            if "mean" not in st or "std" not in st:
                raise ValueError(f"physics.field_norm_stats[{ch}] 缺少 mean/std")
        # batch_size 至少为 1。
        if self.data.batch_size < 1:
            raise ValueError("batch_size 必须 >= 1")
        # 学习率必须为正。
        if self.optim.lr <= 0:
            raise ValueError("lr 必须 > 0")
        if self.optim.encoder_lr_ratio <= 0:
            raise ValueError("encoder_lr_ratio 必须 > 0")
        if self.run.pretrained_encoder and self.run.init_checkpoint:
            raise ValueError("pretrained_encoder 与 init_checkpoint 互斥")
        if (self.optim.freeze_backbone or self.optim.freeze_field_head) and not self.run.init_checkpoint:
            raise ValueError("freeze_backbone/freeze_field_head 须配合 init_checkpoint")
        if self.optim.freeze_field_head and self.model.wss_dim <= 0:
            raise ValueError("freeze_field_head 须 wss_dim > 0")
        if self.run.pretrained_encoder:
            if self.model.name != "pointnext":
                raise ValueError("pretrained_encoder 仅支持 pointnext")
            if (
                self.model.wss_vel_context
                or self.model.wss_pgrad_context
                or self.model.wss_kernel_attention
                or self.model.wss_vn_head
                or self.model.wss_boundary_layer_context
                or self.model.pool_k_tiers
            ):
                raise ValueError("pretrained_encoder 仅适用于标准 AsymW-a pointnext 配方")
        # 隐藏维度必须为正整数。
        if self.model.hidden_dim < 1:
            raise ValueError("hidden_dim 必须 >= 1")
        # 梯度累积步数至少为 1。
        if self.optim.accumulate_grad_batches < 1:
            raise ValueError("accumulate_grad_batches 必须 >= 1")
        if self.optim.interior_loss_boost <= 0:
            raise ValueError("interior_loss_boost 必须 > 0")
        if self.model.wss_dim > 0 and self.optim.wss_loss_weight > 0:
            lt = (self.optim.wss_loss_type or "mse").lower()
            if lt not in ("mse", "huber", "smooth_l1"):
                raise ValueError("wss_loss_type 须为 mse、huber 或 smooth_l1")
        if self.optim.wss_huber_beta <= 0:
            raise ValueError("wss_huber_beta 必须 > 0")
        if self.optim.early_stop_min_delta < 0:
            raise ValueError("early_stop_min_delta 不得为负")
        if not 0.0 <= self.optim.val_score_ema_alpha <= 1.0:
            raise ValueError("val_score_ema_alpha 须在 [0, 1] 内")
        if self.model.head_layout not in ("single_linear", "mlp2"):
            raise ValueError(f"head_layout 须为 single_linear 或 mlp2，收到: {self.model.head_layout}")
        if not 0.0 <= self.model.wss_head_dropout < 1.0:
            raise ValueError("wss_head_dropout 须在 [0, 1) 内")
        if self.model.wss_head_dropout > 0 and self.model.head_layout != "mlp2":
            raise ValueError("wss_head_dropout > 0 时 head_layout 须为 mlp2")
        if self.model.wss_vel_context:
            if self.model.name != "pointnext":
                raise ValueError("wss_vel_context 仅支持 pointnext")
            if self.model.wss_dim <= 0:
                raise ValueError("wss_vel_context 须 wss_dim > 0")
            if self.model.wss_vel_context_dim < 1:
                raise ValueError("wss_vel_context_dim 须 >= 1")
        if self.model.wss_vel_context and self.run.init_checkpoint:
            raise ValueError(
                "wss_vel_context 训练禁止 warm-start（wss_head 输入维与 global ckpt 不兼容）"
            )
        if self.model.wss_pgrad_context:
            if self.model.name != "pointnext":
                raise ValueError("wss_pgrad_context 仅支持 pointnext")
            if self.model.wss_dim <= 0:
                raise ValueError("wss_pgrad_context 须 wss_dim > 0")
            if self.model.wss_output_mode != "head":
                raise ValueError("wss_pgrad_context 仅适用于 wss_output_mode=head")
            if self.model.wss_pgrad_context_mode not in ("mag", "rich"):
                raise ValueError("wss_pgrad_context_mode 须为 mag 或 rich")
        if self.model.wss_pgrad_context and self.run.init_checkpoint:
            raise ValueError(
                "wss_pgrad_context 训练禁止 warm-start（wss_head 输入维与 global ckpt 不兼容）"
            )
        if self.model.wss_kernel_attention:
            if self.model.name != "pointnext":
                raise ValueError("wss_kernel_attention 仅支持 pointnext")
            if self.model.wss_dim <= 0:
                raise ValueError("wss_kernel_attention 须 wss_dim > 0")
            if self.model.wss_output_mode != "head":
                raise ValueError("wss_kernel_attention 仅适用于 wss_output_mode=head")
            if self.model.wss_kernel_attn_dim < 8:
                raise ValueError("wss_kernel_attn_dim 须 >= 8")
            if self.model.wss_kernel_attn_heads < 1:
                raise ValueError("wss_kernel_attn_heads 须 >= 1")
            if self.model.wss_kernel_attn_dim % self.model.wss_kernel_attn_heads != 0:
                raise ValueError("wss_kernel_attn_dim 须能被 wss_kernel_attn_heads 整除")
            if self.model.wss_pgrad_context:
                raise ValueError("wss_kernel_attention 与 wss_pgrad_context 互斥（G2-b 升级替换 concat）")
        if self.model.wss_kernel_attention and self.run.init_checkpoint:
            raise ValueError(
                "wss_kernel_attention 训练禁止 warm-start（wss_head 输入维与 global ckpt 不兼容）"
            )
        if self.model.wss_vn_head:
            if self.model.name != "pointnext":
                raise ValueError("wss_vn_head 仅支持 pointnext")
            if self.model.wss_dim != 4:
                raise ValueError("wss_vn_head 须 wss_dim=4")
            if self.model.wss_output_mode != "head":
                raise ValueError("wss_vn_head 仅适用于 wss_output_mode=head")
            if self.model.wss_vn_channels < 4:
                raise ValueError("wss_vn_channels 须 >= 4")
            if self.model.wss_vn_layers < 1:
                raise ValueError("wss_vn_layers 须 >= 1")
            if (
                self.model.wss_vel_context
                or self.model.wss_pgrad_context
                or self.model.wss_kernel_attention
                or self.model.wss_boundary_layer_context
            ):
                raise ValueError(
                    "wss_vn_head 与 wss_vel_context / wss_pgrad_context / "
                    "wss_kernel_attention / wss_boundary_layer_context 互斥（G1-a0 单变量）"
                )
        if self.model.wss_vn_head and self.run.init_checkpoint:
            raise ValueError(
                "wss_vn_head 训练禁止 warm-start（wss_head 结构与 global ckpt 不兼容）"
            )
        if self.model.wss_boundary_layer_context:
            if self.model.name != "pointnext":
                raise ValueError("wss_boundary_layer_context 仅支持 pointnext")
            if self.model.wss_dim <= 0:
                raise ValueError("wss_boundary_layer_context 须 wss_dim > 0")
            if self.model.wss_output_mode != "head":
                raise ValueError("wss_boundary_layer_context 仅适用于 wss_output_mode=head")
            if self.model.wss_boundary_layer_variant not in ("naive", "tang_normal"):
                raise ValueError("wss_boundary_layer_variant 须为 naive 或 tang_normal")
        if self.model.wss_boundary_layer_context and self.run.init_checkpoint:
            raise ValueError(
                "wss_boundary_layer_context 训练禁止 warm-start（wss_head 输入维与 global ckpt 不兼容）"
            )
        if self.model.wss_output_mode not in ("head", "vel_diff"):
            raise ValueError(
                f"wss_output_mode 须为 head 或 vel_diff，收到: {self.model.wss_output_mode}"
            )
        if self.model.wss_output_mode == "vel_diff":
            if self.model.name != "pointnext":
                raise ValueError("wss_output_mode=vel_diff 仅支持 pointnext")
            if self.model.wss_dim > 0:
                raise ValueError("vel_diff 模式须 wss_dim=0（不使用 wss_head）")
            if self.data.wss_target_frame != "global":
                raise ValueError("vel_diff 模式当前仅支持 global WSS 目标")
            if self.model.wss_metric_dim not in (1, len(WSS_TARGET_NAMES)):
                raise ValueError(
                    f"vel_diff 模式 wss_metric_dim 须为 1 或 {len(WSS_TARGET_NAMES)}"
                )
            if (
                self.model.wss_vel_context
                or self.model.wss_pgrad_context
                or self.model.wss_kernel_attention
                or self.model.wss_vn_head
                or self.model.wss_boundary_layer_context
            ):
                raise ValueError(
                    "vel_diff 模式与 wss_vel_context / wss_pgrad_context / "
                    "wss_kernel_attention / wss_vn_head / wss_boundary_layer_context 互斥"
                )
            if self.run.init_checkpoint:
                raise ValueError("vel_diff 模式禁止 warm-start（无 wss_head 参数）")
            if self.model.vel_diff_variant not in ("naive", "tang_normal"):
                raise ValueError(
                    f"vel_diff_variant 须为 naive 或 tang_normal，收到: {self.model.vel_diff_variant}"
                )
        tiers = self.model.pool_k_tiers
        if tiers:
            if self.model.name != "pointnext":
                raise ValueError("pool_k_tiers 仅支持 pointnext")
            if len(tiers) < 2:
                raise ValueError("pool_k_tiers 须至少 2 档（如 [6, 18, 36]）")
            if any(k < 1 for k in tiers):
                raise ValueError("pool_k_tiers 各档 k 须 >= 1")
            if tiers != sorted(set(tiers)):
                raise ValueError("pool_k_tiers 须为严格递增且无重复")
            if tiers[0] != 6:
                raise ValueError("pool_k_tiers 首档须为 6（与 pipeline GRAPH_CONFIG k_neighbors 一致）")
        if tiers and len(tiers) >= 2 and self.run.init_checkpoint:
            raise ValueError(
                "pool_k_tiers 多档训练禁止 warm-start（backbone block MLP 输入维与旧 ckpt 不兼容）"
            )
        dl = self.optim.domain_loss
        eff_wss_dim = resolve_wss_effective_dim(
            self.model.wss_dim,
            self.model.wss_output_mode,
            self.model.wss_metric_dim,
        )
        if dl.enabled:
            for attr in (
                "lambda_vel_int",
                "lambda_vel_noslip",
                "lambda_p_int",
                "lambda_p_wall",
                "lambda_wss",
                "lambda_wss_mag_consist",
                "lambda_wss_rank",
                "lambda_wss_vel_consist",
                "lambda_wss_slope",
                "lambda_wss_pgrad_consist",
            ):
                if getattr(dl, attr) < 0:
                    raise ValueError(f"domain_loss.{attr} 不得为负")
            if (dl.lambda_wss_vel_consist > 0 or dl.lambda_wss_pgrad_consist > 0) and eff_wss_dim < 1:
                raise ValueError(
                    "lambda_wss_vel_consist / lambda_wss_pgrad_consist 须有效 WSS 输出维 >=1"
                )
            if dl.lambda_wss_rank > 0:
                if not 0.0 < dl.wss_rank_top_frac <= 1.0:
                    raise ValueError("wss_rank_top_frac 须在 (0, 1] 内")
                if dl.wss_rank_max_nodes < 2:
                    raise ValueError("wss_rank_max_nodes 须 >= 2")
            if dl.lambda_wss_mag_consist > 0:
                if self.model.wss_dim != 4:
                    raise ValueError("lambda_wss_mag_consist 须 wss_dim=4（global 四分量 head）")
                if self.run.init_checkpoint:
                    raise ValueError(
                        "lambda_wss_mag_consist 训练禁止 warm-start（loss 项与旧 ckpt 训练轨迹不兼容）"
                    )
