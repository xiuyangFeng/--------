from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

from pipeline.config import GLOBAL_COND_NAMES

from .config import (
    DataConfig,
    ExperimentConfig,
    MetaConfig,
    ModelConfig,
    OptimConfig,
    RunConfig,
    SystemConfig,
)

COORD_FEATURES = ["x", "y", "z"]
GEOMETRY_SCALAR_FEATURES = ["Abscissa", "NormRadius", "Curvature"]
TANGENT_FEATURES = ["Tangent_X", "Tangent_Y", "Tangent_Z"]
GEOMETRY_FEATURES = GEOMETRY_SCALAR_FEATURES + TANGENT_FEATURES
IS_WALL_FEATURE = ["is_wall"]
TIME_FEATURES = ["t_norm"]
BC_FEATURES = [name for name in GLOBAL_COND_NAMES if name != "t_norm"]


@dataclass(frozen=True)
class FieldPlanItem:
    # 一个 plan item 对应一个可直接训练的 JSON 配置文件。
    exp_id: str
    experiment_name: str
    config: ExperimentConfig
    output_relpath: str

    def manifest_row(self) -> Dict[str, object]:
        return {
            "exp_id": self.exp_id,
            "experiment_name": self.experiment_name,
            "output_relpath": self.output_relpath,
            "model": self.config.model.name,
            "feature_set": self.config.meta.feature_set,
            "study_group": self.config.meta.study_group,
            "ablation_axis": self.config.meta.ablation_axis,
            "seed": self.config.system.seed,
        }


def base_optim_config() -> OptimConfig:
    # 第一阶段所有 baseline / ablation 默认共用一套优化器超参，避免实验变量混杂。
    return OptimConfig(
        epochs=200,
        lr=5e-4,
        weight_decay=1e-4,
        scheduler_factor=0.5,
        scheduler_patience=10,
        early_stopping_patience=30,
        target_weights=[1.0, 1.0, 1.0, 1.0],
        grad_clip_norm=1.0,
    )


def base_system_config(seed: int) -> SystemConfig:
    return SystemConfig(seed=seed, device="auto", deterministic=True)


def build_data_config(
    *,
    data_root: str,
    split_file: str,
    graphs_subdir: str,
    enabled_node_features: Sequence[str],
    enabled_global_features: Sequence[str],
    augment: bool,
    augment_config: Mapping[str, object] | None,
) -> DataConfig:
    # 这里故意把构造逻辑集中起来，方便后续全局调整 batch size / dataloader 策略。
    return DataConfig(
        data_root=data_root,
        split_file=split_file,
        graphs_subdir=graphs_subdir,
        preload=False,
        batch_size=2,
        num_workers=0,
        pin_memory=False,
        enabled_node_features=list(enabled_node_features),
        enabled_global_features=list(enabled_global_features),
        augment=augment,
        augment_config=dict(augment_config or {}),
    )


def build_run_config(experiment_name: str, output_root: str) -> RunConfig:
    return RunConfig(
        experiment_name=experiment_name,
        output_root=output_root,
        save_every=10,
        save_best_only=True,
    )


def build_model_config(model_name: str) -> ModelConfig:
    return ModelConfig(
        name=model_name,
        hidden_dim=128,
        num_layers=3,
        dropout=0.1,
        heads=4,
    )


def make_plan_item(
    *,
    exp_id: str,
    experiment_name: str,
    study_group: str,
    feature_set: str,
    question: str,
    ablation_axis: str,
    model_name: str,
    data_root: str,
    split_file: str,
    graphs_subdir: str,
    enabled_node_features: Sequence[str],
    enabled_global_features: Sequence[str],
    augment: bool,
    augment_config: Mapping[str, object] | None,
    seed: int,
    output_root: str,
    notes: str = "",
    tags: Sequence[str] | None = None,
) -> FieldPlanItem:
    # make_plan_item 是所有实验模板的公共出口。
    # 统一从这里构建 config，可以保证 meta / run / data 命名口径始终一致。
    config = ExperimentConfig(
        run=build_run_config(experiment_name=experiment_name, output_root=output_root),
        data=build_data_config(
            data_root=data_root,
            split_file=split_file,
            graphs_subdir=graphs_subdir,
            enabled_node_features=enabled_node_features,
            enabled_global_features=enabled_global_features,
            augment=augment,
            augment_config=augment_config,
        ),
        model=build_model_config(model_name),
        optim=base_optim_config(),
        system=base_system_config(seed=seed),
        meta=MetaConfig(
            task="field",
            exp_id=exp_id,
            stage="task_a",
            study_group=study_group,
            question=question,
            feature_set=feature_set,
            ablation_axis=ablation_axis,
            tags=list(tags or []),
            notes=notes,
            generated_from="training.make_field_plan",
        ),
    )
    rel_dir = Path(study_group) / f"{exp_id}_seed{seed}.json"
    return FieldPlanItem(
        exp_id=exp_id,
        experiment_name=experiment_name,
        config=config,
        output_relpath=str(rel_dir),
    )


def canonical_feature_set(
    *,
    include_geometry: bool,
    include_is_wall: bool,
    include_bc: bool,
    coords_only: bool = False,
) -> str:
    # feature_set 会进入实验记录和输出索引，所以这里保持简短且稳定。
    parts = ["coord", "t"]
    if include_bc:
        parts.append("bc")
    if include_geometry:
        parts.append("geom")
    if include_is_wall:
        parts.append("wall")
    if coords_only:
        parts.append("point")
    return "_".join(parts)


def main_aug_presets() -> Dict[str, Dict[str, object]]:
    # 第一版只暴露文档里要求的增强层级，不引入更激进的形变操作。
    return {
        "none": {},
        "rotate": {
            "rotation_prob": 0.5,
            "rotation_axes": "xyz",
            "translation_prob": 0.0,
            "translation_range": 0.0,
            "scale_prob": 0.0,
            "mirror_prob": 0.0,
        },
        "rotate_translate": {
            "rotation_prob": 0.5,
            "rotation_axes": "xyz",
            "translation_prob": 0.5,
            "translation_range": 0.1,
            "scale_prob": 0.0,
            "mirror_prob": 0.0,
        },
        "rotate_translate_scale": {
            "rotation_prob": 0.5,
            "rotation_axes": "xyz",
            "translation_prob": 0.5,
            "translation_range": 0.1,
            "scale_prob": 0.25,
            "scale_range": [0.95, 1.05],
            "mirror_prob": 0.0,
        },
    }


def build_task_a_baselines(
    *,
    data_root: str,
    split_file: str,
    graphs_subdir: str,
    output_root: str,
    seed: int,
) -> List[FieldPlanItem]:
    # 这 4 组直接对应 docs/01-任务/任务A/任务A实验清单.md 里的第一批必须跑通实验。
    main_augment = main_aug_presets()["rotate_translate"]
    return [
        make_plan_item(
            exp_id="A-Base-01",
            experiment_name="field_mlp_coord_t_bc",
            study_group="baseline",
            feature_set=canonical_feature_set(
                include_geometry=False,
                include_is_wall=False,
                include_bc=True,
                coords_only=True,
            ),
            question="建立无图结构下限，验证图建模是否有收益。",
            ablation_axis="baseline_model",
            model_name="mlp",
            data_root=data_root,
            split_file=split_file,
            graphs_subdir=graphs_subdir,
            enabled_node_features=COORD_FEATURES,
            enabled_global_features=TIME_FEATURES + BC_FEATURES,
            augment=False,
            augment_config={},
            seed=seed,
            output_root=output_root,
            notes="任务A基线1：Point-wise MLP。",
            tags=["baseline", "mlp", "coord+t+bc"],
        ),
        make_plan_item(
            exp_id="A-Base-02",
            experiment_name="field_graphsage_coord_t_bc_wall",
            study_group="baseline",
            feature_set=canonical_feature_set(
                include_geometry=False,
                include_is_wall=True,
                include_bc=True,
            ),
            question="验证基础图模型是否优于点级模型。",
            ablation_axis="baseline_model",
            model_name="graphsage",
            data_root=data_root,
            split_file=split_file,
            graphs_subdir=graphs_subdir,
            enabled_node_features=COORD_FEATURES + IS_WALL_FEATURE,
            enabled_global_features=TIME_FEATURES + BC_FEATURES,
            augment=True,
            augment_config=main_augment,
            seed=seed,
            output_root=output_root,
            notes="任务A基线2：GraphSAGE。",
            tags=["baseline", "graphsage", "coord+t+bc+wall"],
        ),
        make_plan_item(
            exp_id="A-Base-03",
            experiment_name="field_transformer_coord_t_bc_wall",
            study_group="baseline",
            feature_set=canonical_feature_set(
                include_geometry=False,
                include_is_wall=True,
                include_bc=True,
            ),
            question="控制 backbone，隔离显式几何特征的贡献。",
            ablation_axis="baseline_model",
            model_name="transformer",
            data_root=data_root,
            split_file=split_file,
            graphs_subdir=graphs_subdir,
            enabled_node_features=COORD_FEATURES + IS_WALL_FEATURE,
            enabled_global_features=TIME_FEATURES + BC_FEATURES,
            augment=True,
            augment_config=main_augment,
            seed=seed,
            output_root=output_root,
            notes="任务A基线3：Transformer without geometry。",
            tags=["baseline", "transformer", "coord+t+bc+wall"],
        ),
        make_plan_item(
            exp_id="A-Main-01",
            experiment_name="field_transformer_coord_t_bc_geom_wall",
            study_group="baseline",
            feature_set=canonical_feature_set(
                include_geometry=True,
                include_is_wall=True,
                include_bc=True,
            ),
            question="作为任务A第一阶段主模型。",
            ablation_axis="main_model",
            model_name="transformer",
            data_root=data_root,
            split_file=split_file,
            graphs_subdir=graphs_subdir,
            enabled_node_features=COORD_FEATURES + GEOMETRY_FEATURES + IS_WALL_FEATURE,
            enabled_global_features=TIME_FEATURES + BC_FEATURES,
            augment=True,
            augment_config=main_augment,
            seed=seed,
            output_root=output_root,
            notes="任务A主模型：Transformer with geometry。",
            tags=["main", "transformer", "coord+t+bc+geom+wall"],
        ),
    ]


def build_input_ablation_plan(
    *,
    data_root: str,
    split_file: str,
    graphs_subdir: str,
    output_root: str,
    seed: int,
) -> List[FieldPlanItem]:
    # 输入特征消融要回答的是：BC / geometry / is_wall 分别有没有贡献。
    main_augment = main_aug_presets()["rotate_translate"]
    entries = [
        (
            "A-Abl-01-01",
            "field_transformer_coord_t",
            COORD_FEATURES,
            TIME_FEATURES,
            "coord_t",
            "coords + t",
        ),
        (
            "A-Abl-01-02",
            "field_transformer_coord_t_bc",
            COORD_FEATURES,
            TIME_FEATURES + BC_FEATURES,
            "coord_t_bc",
            "coords + t + BC",
        ),
        (
            "A-Abl-01-03",
            "field_transformer_coord_t_bc_wall",
            COORD_FEATURES + IS_WALL_FEATURE,
            TIME_FEATURES + BC_FEATURES,
            "coord_t_bc_wall",
            "coords + t + BC + is_wall",
        ),
        (
            "A-Abl-01-04",
            "field_transformer_coord_t_bc_geom",
            COORD_FEATURES + GEOMETRY_FEATURES,
            TIME_FEATURES + BC_FEATURES,
            "coord_t_bc_geom",
            "coords + t + BC + geometry",
        ),
        (
            "A-Abl-01-05",
            "field_transformer_coord_t_bc_geom_wall",
            COORD_FEATURES + GEOMETRY_FEATURES + IS_WALL_FEATURE,
            TIME_FEATURES + BC_FEATURES,
            "coord_t_bc_geom_wall",
            "coords + t + BC + geometry + is_wall",
        ),
    ]
    items: List[FieldPlanItem] = []
    for exp_id, experiment_name, node_feats, global_feats, feature_set, label in entries:
        items.append(
            make_plan_item(
                exp_id=exp_id,
                experiment_name=experiment_name,
                study_group="ablation_input",
                feature_set=feature_set,
                question="边界条件、显式几何特征与 is_wall 分别带来什么贡献。",
                ablation_axis="input_features",
                model_name="transformer",
                data_root=data_root,
                split_file=split_file,
                graphs_subdir=graphs_subdir,
                enabled_node_features=node_feats,
                enabled_global_features=global_feats,
                augment=True,
                augment_config=main_augment,
                seed=seed,
                output_root=output_root,
                notes=f"输入特征层级消融：{label}。",
                tags=["ablation", "input_features", feature_set],
            )
        )
    return items


def build_geometry_ablation_plan(
    *,
    data_root: str,
    split_file: str,
    graphs_subdir: str,
    output_root: str,
    seed: int,
) -> List[FieldPlanItem]:
    # 几何分量消融固定 backbone 和其他输入，只改变显式几何子集。
    main_augment = main_aug_presets()["rotate_translate"]
    full = COORD_FEATURES + GEOMETRY_FEATURES + IS_WALL_FEATURE
    variants = [
        ("A-Abl-02-00", "field_transformer_geometry_full", full, "full_geometry", "全部显式几何特征"),
        (
            "A-Abl-02-01",
            "field_transformer_geometry_no_abscissa",
            [name for name in full if name != "Abscissa"],
            "no_abscissa",
            "去掉 Abscissa",
        ),
        (
            "A-Abl-02-02",
            "field_transformer_geometry_no_normradius",
            [name for name in full if name != "NormRadius"],
            "no_normradius",
            "去掉 NormRadius",
        ),
        (
            "A-Abl-02-03",
            "field_transformer_geometry_no_curvature",
            [name for name in full if name != "Curvature"],
            "no_curvature",
            "去掉 Curvature",
        ),
        (
            "A-Abl-02-04",
            "field_transformer_geometry_no_tangent",
            [name for name in full if name not in set(TANGENT_FEATURES)],
            "no_tangent",
            "去掉 Tangent",
        ),
    ]
    items: List[FieldPlanItem] = []
    for exp_id, experiment_name, node_feats, feature_set, label in variants:
        items.append(
            make_plan_item(
                exp_id=exp_id,
                experiment_name=experiment_name,
                study_group="ablation_geometry",
                feature_set=feature_set,
                question="哪类显式几何先验真正有效。",
                ablation_axis="geometry_components",
                model_name="transformer",
                data_root=data_root,
                split_file=split_file,
                graphs_subdir=graphs_subdir,
                enabled_node_features=node_feats,
                enabled_global_features=TIME_FEATURES + BC_FEATURES,
                augment=True,
                augment_config=main_augment,
                seed=seed,
                output_root=output_root,
                notes=f"显式几何分量消融：{label}。",
                tags=["ablation", "geometry", feature_set],
            )
        )
    return items


def build_augment_ablation_plan(
    *,
    data_root: str,
    split_file: str,
    graphs_subdir: str,
    output_root: str,
    seed: int,
) -> List[FieldPlanItem]:
    # 增强消融单独比较训练时的数据增强强度，不动 backbone 和输入特征。
    presets = main_aug_presets()
    variants = [
        ("A-Abl-04-01", "field_transformer_aug_none", False, presets["none"], "none"),
        ("A-Abl-04-02", "field_transformer_aug_rotate", True, presets["rotate"], "rotate"),
        (
            "A-Abl-04-03",
            "field_transformer_aug_rotate_translate",
            True,
            presets["rotate_translate"],
            "rotate_translate",
        ),
        (
            "A-Abl-04-04",
            "field_transformer_aug_rotate_translate_scale",
            True,
            presets["rotate_translate_scale"],
            "rotate_translate_scale",
        ),
    ]
    items: List[FieldPlanItem] = []
    for exp_id, experiment_name, augment, augment_config, feature_set in variants:
        items.append(
            make_plan_item(
                exp_id=exp_id,
                experiment_name=experiment_name,
                study_group="ablation_augment",
                feature_set=feature_set,
                question="在线增强是否提升患者级泛化。",
                ablation_axis="augmentation",
                model_name="transformer",
                data_root=data_root,
                split_file=split_file,
                graphs_subdir=graphs_subdir,
                enabled_node_features=COORD_FEATURES + GEOMETRY_FEATURES + IS_WALL_FEATURE,
                enabled_global_features=TIME_FEATURES + BC_FEATURES,
                augment=augment,
                augment_config=augment_config,
                seed=seed,
                output_root=output_root,
                notes=f"增强策略消融：{feature_set}。",
                tags=["ablation", "augmentation", feature_set],
            )
        )
    return items


def build_coord_ablation_plan(
    *,
    data_root: str,
    split_file: str,
    output_root: str,
    seed: int,
    coord_variants: Mapping[str, str],
) -> List[FieldPlanItem]:
    # 坐标归一化实验强依赖真实数据目录组织，所以只接受外部显式传入 variant -> subdir 映射。
    items: List[FieldPlanItem] = []
    for idx, (variant_name, graphs_subdir) in enumerate(coord_variants.items(), start=1):
        items.append(
            make_plan_item(
                exp_id=f"A-Abl-03-{idx:02d}",
                experiment_name=f"field_transformer_coordnorm_{variant_name}",
                study_group="ablation_coord",
                feature_set=variant_name,
                question="病例内坐标标准化是否有利于跨病例泛化。",
                ablation_axis="coord_normalization",
                model_name="transformer",
                data_root=data_root,
                split_file=split_file,
                graphs_subdir=graphs_subdir,
                enabled_node_features=COORD_FEATURES + GEOMETRY_FEATURES + IS_WALL_FEATURE,
                enabled_global_features=TIME_FEATURES + BC_FEATURES,
                augment=True,
                augment_config=main_aug_presets()["rotate_translate"],
                seed=seed,
                output_root=output_root,
                notes=f"坐标归一化消融：graphs_subdir={graphs_subdir}。",
                tags=["ablation", "coord_normalization", variant_name],
            )
        )
    return items


def build_task_a_plan(
    *,
    data_root: str,
    split_file: str,
    graphs_subdir: str,
    output_root: str,
    seeds: Iterable[int],
    groups: Iterable[str],
    coord_variants: Mapping[str, str] | None = None,
) -> List[FieldPlanItem]:
    # groups 控制这次批量生成哪些实验块；seeds 控制每个实验块重复多少随机种子。
    group_set = set(groups)
    items: List[FieldPlanItem] = []
    for seed in seeds:
        if "baseline" in group_set:
            items.extend(
                build_task_a_baselines(
                    data_root=data_root,
                    split_file=split_file,
                    graphs_subdir=graphs_subdir,
                    output_root=output_root,
                    seed=seed,
                )
            )
        if "input" in group_set:
            items.extend(
                build_input_ablation_plan(
                    data_root=data_root,
                    split_file=split_file,
                    graphs_subdir=graphs_subdir,
                    output_root=output_root,
                    seed=seed,
                )
            )
        if "geometry" in group_set:
            items.extend(
                build_geometry_ablation_plan(
                    data_root=data_root,
                    split_file=split_file,
                    graphs_subdir=graphs_subdir,
                    output_root=output_root,
                    seed=seed,
                )
            )
        if "augment" in group_set:
            items.extend(
                build_augment_ablation_plan(
                    data_root=data_root,
                    split_file=split_file,
                    graphs_subdir=graphs_subdir,
                    output_root=output_root,
                    seed=seed,
                )
            )
        if "coord" in group_set and coord_variants:
            items.extend(
                build_coord_ablation_plan(
                    data_root=data_root,
                    split_file=split_file,
                    output_root=output_root,
                    seed=seed,
                    coord_variants=coord_variants,
                )
            )
    return items
