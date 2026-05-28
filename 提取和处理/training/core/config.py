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


@dataclass
class RunConfig:
    # 训练 run 的名字会进入输出目录名，因此这里最好保持“任务-模型-特征集”可读。
    experiment_name: str
    output_root: str = "outputs/field"
    save_every: int = 10
    save_best_only: bool = True
    init_checkpoint: Optional[str] = None


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

    def resolve_scales_from_data(self, data_root: str, graphs_subdir: str, case_dirs: list) -> None:
        """从 pipeline 产物自动加载物理损失所需的坐标/时间尺度。

        coord_normalize 会为每个病例保存 ``transform_params.json``，其中
        含有该病例的 ``scale_factor``。物理损失在归一化坐标上求导后，
        需要用这些尺度还原回原始物理空间，否则 continuity / momentum
        的量纲会失真。这里读取可用病例的尺度，并用中位数作为稳健代表值。
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
        # 全局归一化参数存在时，再尝试恢复时间尺度。
        if norm_params_path.exists():
            try:
                with open(norm_params_path, "r", encoding="utf-8") as f:
                    norm_params = json.load(f)
                stats = norm_params.get("statistics", {})
                t_stats = stats.get("t_norm")
                # 如果能读到 t_norm 的标准差，就把它当作时间缩放尺度。
                if t_stats and t_stats.get("std", 0) > 1e-10:
                    self.time_scale = t_stats["std"]
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
        wss_names = resolve_wss_target_names(self.data.wss_target_frame, self.model.wss_dim) if self.model.wss_dim > 0 else []
        if self.model.wss_dim > 0:
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
        # batch_size 至少为 1。
        if self.data.batch_size < 1:
            raise ValueError("batch_size 必须 >= 1")
        # 学习率必须为正。
        if self.optim.lr <= 0:
            raise ValueError("lr 必须 > 0")
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
        if dl.enabled:
            for attr in (
                "lambda_vel_int",
                "lambda_vel_noslip",
                "lambda_p_int",
                "lambda_p_wall",
                "lambda_wss",
                "lambda_wss_mag_consist",
            ):
                if getattr(dl, attr) < 0:
                    raise ValueError(f"domain_loss.{attr} 不得为负")
            if dl.lambda_wss_mag_consist > 0:
                if self.model.wss_dim != 4:
                    raise ValueError("lambda_wss_mag_consist 须 wss_dim=4（global 四分量 head）")
                if self.run.init_checkpoint:
                    raise ValueError(
                        "lambda_wss_mag_consist 训练禁止 warm-start（loss 项与旧 ckpt 训练轨迹不兼容）"
                    )
