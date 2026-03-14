from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from pipeline.config import GLOBAL_COND_NAMES, NODE_FEATURE_NAMES, TARGET_NAMES


@dataclass
class RunConfig:
    # 训练 run 的名字会进入输出目录名，因此这里最好保持“任务-模型-特征集”可读。
    experiment_name: str
    output_root: str = "outputs/field"
    save_every: int = 10
    save_best_only: bool = True


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


@dataclass
class ModelConfig:
    # 这里故意只暴露最小一组 backbone 超参数，避免第一阶段实验维度失控。
    name: str = "transformer"
    hidden_dim: int = 128
    num_layers: int = 3
    dropout: float = 0.1
    heads: int = 4


@dataclass
class OptimConfig:
    # target_weights 对应 [u, v, w, p] 四个输出维度的损失权重。
    epochs: int = 200
    lr: float = 5e-4
    weight_decay: float = 1e-4
    scheduler_factor: float = 0.5
    scheduler_patience: int = 10
    early_stopping_patience: int = 30
    target_weights: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0])
    grad_clip_norm: Optional[float] = 1.0


@dataclass
class SystemConfig:
    # deterministic=True 更适合论文实验；如果后面追求吞吐量，可以在服务器上再放宽。
    seed: int = 1
    device: str = "auto"
    deterministic: bool = True


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
        if not self.auto_load_scales or not self.enabled:
            return

        import json
        from pathlib import Path

        scale_factors: List[float] = []
        for case_dir in case_dirs:
            params_path = Path(case_dir) / "processed" / "coord_normalized" / "transform_params.json"
            if not params_path.exists():
                continue
            try:
                with open(params_path, "r", encoding="utf-8") as f:
                    params = json.load(f)
                sf = params.get("scale_factor", 1.0)
                if sf > 1e-6:
                    scale_factors.append(sf)
            except (json.JSONDecodeError, KeyError):
                continue

        if scale_factors:
            import statistics
            median_sf = statistics.median(scale_factors)
            self.coord_scales = [median_sf, median_sf, median_sf]

        # t_norm 的标准差保存在全局归一化参数里，用于把 dt 还原回原始时间尺度。
        norm_params_path = Path(data_root) / "normalization_params_global.json"
        if norm_params_path.exists():
            try:
                with open(norm_params_path, "r", encoding="utf-8") as f:
                    norm_params = json.load(f)
                stats = norm_params.get("statistics", {})
                t_stats = stats.get("t_norm")
                if t_stats and t_stats.get("std", 0) > 1e-10:
                    self.time_scale = t_stats["std"]
            except (json.JSONDecodeError, KeyError):
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
        return cls(
            run=RunConfig(**data["run"]),
            data=DataConfig(**data["data"]),
            model=ModelConfig(**data["model"]),
            optim=OptimConfig(**data["optim"]),
            system=SystemConfig(**data.get("system", {})),
            meta=MetaConfig(**data.get("meta", {})),
            physics=PhysicsConfig(**data.get("physics", {})),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "ExperimentConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return cls.from_dict(raw)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        # 训练阶段只做“特征屏蔽”，因此这里必须保证名字和 pipeline 导出的 schema 一致。
        unknown_node = sorted(set(self.data.enabled_node_features) - set(NODE_FEATURE_NAMES))
        unknown_global = sorted(
            set(self.data.enabled_global_features) - set(GLOBAL_COND_NAMES)
        )
        if unknown_node:
            raise ValueError(f"未知节点特征: {unknown_node}")
        if unknown_global:
            raise ValueError(f"未知全局特征: {unknown_global}")
        if self.model.name not in {"mlp", "graphsage", "transformer", "meshgraphnet", "pointnetpp"}:
            raise ValueError(f"不支持的模型: {self.model.name}")
        if len(self.optim.target_weights) != len(TARGET_NAMES):
            raise ValueError("target_weights 维度必须与目标输出一致")
        # physics 残差里默认只对 x/y/z 三个坐标方向求导，因此这里强约束长度为 3。
        if len(self.physics.coord_scales) != 3:
            raise ValueError("physics.coord_scales 必须包含 3 个坐标尺度")
