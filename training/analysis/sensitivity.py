"""Sensitivity analysis configuration generator for key hyperparameters.

Generates experiment configs that systematically vary one pipeline or
model hyperparameter at a time, enabling the sensitivity tables/figures
required by the paper.

Covered hyperparameters:
  - KNN k (graph construction)
  - FPS ratio (point sampling)
  - Near-wall threshold (stratified sampling)
  - Target total points (sampling budget)
  - Batch size / hidden dim (training)
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Tuple


def knn_sensitivity_configs(
    base_config: Dict[str, Any],
    k_values: List[int] = None,
) -> List[Tuple[str, Dict[str, Any]]]:
    """Generate configs sweeping KNN neighbour count."""
    if k_values is None:
        k_values = [4, 6, 8, 12, 16]

    configs = []
    for k in k_values:
        cfg = deepcopy(base_config)
        cfg.setdefault("meta", {})["exp_id"] = f"sens-knn-k{k}"
        cfg["meta"]["study_group"] = "sensitivity_knn"
        cfg["meta"]["ablation_axis"] = "knn_k"
        cfg["meta"]["question"] = f"How does k={k} affect reconstruction quality?"
        cfg.setdefault("pipeline_overrides", {})["k_neighbors"] = k
        configs.append((f"knn_k{k}", cfg))
    return configs


def sampling_sensitivity_configs(
    base_config: Dict[str, Any],
    target_points: List[int] = None,
    fps_ratios: List[float] = None,
    near_wall_thresholds: List[float] = None,
) -> List[Tuple[str, Dict[str, Any]]]:
    """Generate configs sweeping sampling parameters."""
    configs = []

    if target_points is None:
        target_points = [10000, 20000, 40000, 60000]
    for n in target_points:
        cfg = deepcopy(base_config)
        cfg.setdefault("meta", {})["exp_id"] = f"sens-npts-{n}"
        cfg["meta"]["study_group"] = "sensitivity_sampling"
        cfg["meta"]["ablation_axis"] = "target_points"
        cfg.setdefault("pipeline_overrides", {})["target_total_points"] = n
        configs.append((f"npts_{n}", cfg))

    if fps_ratios is None:
        fps_ratios = [0.0, 0.1, 0.2, 0.3, 0.5]
    for r in fps_ratios:
        cfg = deepcopy(base_config)
        cfg.setdefault("meta", {})["exp_id"] = f"sens-fps-{r}"
        cfg["meta"]["study_group"] = "sensitivity_sampling"
        cfg["meta"]["ablation_axis"] = "fps_ratio"
        cfg.setdefault("pipeline_overrides", {})["hybrid_fps_ratio"] = r
        configs.append((f"fps_{r}", cfg))

    if near_wall_thresholds is None:
        near_wall_thresholds = [1.0, 1.5, 2.0, 3.0, 5.0]
    for t in near_wall_thresholds:
        cfg = deepcopy(base_config)
        cfg.setdefault("meta", {})["exp_id"] = f"sens-nearwall-{t}mm"
        cfg["meta"]["study_group"] = "sensitivity_sampling"
        cfg["meta"]["ablation_axis"] = "boundary_threshold"
        cfg.setdefault("pipeline_overrides", {})["boundary_threshold"] = t
        configs.append((f"nearwall_{t}mm", cfg))

    return configs


def build_sensitivity_plan(
    base_config: Dict[str, Any],
    seeds: List[int] = None,
) -> List[Tuple[str, Dict[str, Any]]]:
    """Combine all sensitivity sweeps into a single plan."""
    if seeds is None:
        seeds = [1, 2, 3]

    all_configs = []
    all_configs.extend(knn_sensitivity_configs(base_config))
    all_configs.extend(sampling_sensitivity_configs(base_config))

    expanded = []
    for name, cfg in all_configs:
        for seed in seeds:
            cfg_s = deepcopy(cfg)
            cfg_s.setdefault("system", {})["seed"] = seed
            expanded.append((f"{name}_seed{seed}", cfg_s))

    return expanded
