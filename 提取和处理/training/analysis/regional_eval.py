"""Regional evaluation for Task A field predictions.

Computes metrics (RMSE, MAE, R²) per spatial region so that the paper
can report performance broken down by:
  - wall vs interior nodes
  - high-curvature vs low-curvature regions
  - bifurcation vs trunk regions
  - near-wall boundary layer vs core flow

The thresholds below are configurable and should be reported alongside
any results table.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from pipeline.config import NODE_FEATURE_NAMES, TARGET_NAMES
from pipeline.dataset import load_graph_data

IS_WALL_IDX = NODE_FEATURE_NAMES.index("is_wall")
CURVATURE_IDX = NODE_FEATURE_NAMES.index("Curvature")
NORM_RADIUS_IDX = NODE_FEATURE_NAMES.index("NormRadius")
ABSCISSA_IDX = NODE_FEATURE_NAMES.index("Abscissa")

_WARNED_REGION_GEOMETRY_FALLBACK = False


def load_node_features_for_region_masks(payload: Dict[str, Any]) -> torch.Tensor:
    """供分区域评估使用的节点特征：优先从图资产读取未 mask 的完整 ``x``。

    训练时 ``FieldGraphDataset`` 会对 ``data.x`` 按列乘 ``feature_mask``，
    被关闭的几何列在导出预测里为 0，会导致 ``high_curvature`` / ``near_wall`` /
    ``bifurcation`` / ``trunk`` 等 mask 与模型无关地错误。区域口径应依赖
    ``graph_path`` 指向的固定图文件，与模型是否使用几何输入无关。
    """
    global _WARNED_REGION_GEOMETRY_FALLBACK
    y_true = payload["y_true"]
    if not isinstance(y_true, torch.Tensor):
        raise TypeError("payload['y_true'] 必须为 torch.Tensor")
    n = int(y_true.shape[0])
    graph_path = payload.get("graph_path")
    if graph_path:
        path = Path(str(graph_path))
        if path.is_file():
            data = load_graph_data(path)
            x = data.x.detach().cpu()
            if x.shape[0] == n and x.shape[1] == len(NODE_FEATURE_NAMES):
                return x
    if not _WARNED_REGION_GEOMETRY_FALLBACK:
        warnings.warn(
            "无法从 graph_path 加载与预测对齐的完整节点特征，回退使用导出文件中的 x；"
            "若该 run 关闭了几何列，分区域指标可能不可靠。",
            UserWarning,
            stacklevel=2,
        )
        _WARNED_REGION_GEOMETRY_FALLBACK = True
    x_fb = payload["x"]
    if not isinstance(x_fb, torch.Tensor):
        raise TypeError("payload['x'] 必须为 torch.Tensor")
    return x_fb.detach().cpu()


@dataclass
class RegionSpec:
    name: str
    description: str = ""


def _curvature_quantile_threshold(
    curvature: torch.Tensor,
    q: float,
    subsample_max: int = 1_000_000,
) -> torch.Tensor:
    """torch.quantile 对超长一维张量会报错；大样本时用步进子样本估计分位数（确定性、省内存）。"""
    n = curvature.numel()
    if n == 0:
        return torch.zeros((), device=curvature.device, dtype=curvature.dtype)
    if n <= subsample_max:
        return torch.quantile(curvature, q)
    step = max(1, n // subsample_max)
    idx = torch.arange(0, n, step, device=curvature.device)
    return torch.quantile(curvature[idx], q)


DEFAULT_REGIONS = {
    "all": RegionSpec("all", "All nodes"),
    "wall": RegionSpec("wall", "Wall nodes (is_wall == 1)"),
    "interior": RegionSpec("interior", "Interior nodes (is_wall == 0)"),
    "high_curvature": RegionSpec(
        "high_curvature",
        "Nodes where curvature > threshold (default: 75th percentile)",
    ),
    "low_curvature": RegionSpec("low_curvature", "Nodes where curvature <= threshold"),
    "near_wall": RegionSpec(
        "near_wall",
        "Interior nodes close to wall (NormRadius > 0.8)",
    ),
    "core_flow": RegionSpec(
        "core_flow",
        "Interior nodes far from wall (NormRadius <= 0.5)",
    ),
    "bifurcation": RegionSpec(
        "bifurcation",
        "Nodes near bifurcation zone (Abscissa in [0.6, 0.9])",
    ),
    "trunk": RegionSpec(
        "trunk",
        "Nodes in the main trunk (Abscissa < 0.6)",
    ),
}


def build_region_masks(
    node_features: torch.Tensor,
    curvature_quantile: float = 0.75,
    near_wall_threshold: float = 0.8,
    core_flow_threshold: float = 0.5,
    bifurcation_range: tuple = (0.6, 0.9),
) -> Dict[str, torch.Tensor]:
    """Return boolean masks for each predefined region."""
    n = node_features.size(0)
    is_wall = node_features[:, IS_WALL_IDX] > 0.5
    curvature = node_features[:, CURVATURE_IDX]
    norm_radius = node_features[:, NORM_RADIUS_IDX]
    abscissa = node_features[:, ABSCISSA_IDX]

    curv_thresh = _curvature_quantile_threshold(curvature, curvature_quantile)

    masks: Dict[str, torch.Tensor] = {
        "all": torch.ones(n, dtype=torch.bool, device=node_features.device),
        "wall": is_wall,
        "interior": ~is_wall,
        "high_curvature": curvature > curv_thresh,
        "low_curvature": curvature <= curv_thresh,
        "near_wall": (~is_wall) & (norm_radius > near_wall_threshold),
        "core_flow": (~is_wall) & (norm_radius <= core_flow_threshold),
        "bifurcation": (abscissa >= bifurcation_range[0]) & (abscissa <= bifurcation_range[1]),
        "trunk": abscissa < bifurcation_range[0],
    }
    return masks


def compute_regional_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    node_features: torch.Tensor,
    **mask_kwargs,
) -> Dict[str, Dict[str, float]]:
    """Compute RMSE / MAE / R² per region and per target variable.

    Returns a nested dict ``{region_name: {metric_name: value}}``.
    """
    masks = build_region_masks(node_features, **mask_kwargs)
    results: Dict[str, Dict[str, float]] = {}

    for region_name, mask in masks.items():
        if mask.sum() == 0:
            results[region_name] = {"n_nodes": 0}
            continue

        p = pred[mask]
        t = target[mask]
        diff = p - t
        metrics: Dict[str, float] = {"n_nodes": int(mask.sum().item())}

        metrics["rmse"] = torch.sqrt((diff ** 2).mean()).item()
        metrics["mae"] = diff.abs().mean().item()

        ss_res = ((t - p) ** 2).sum(dim=0)
        ss_tot = ((t - t.mean(dim=0, keepdim=True)) ** 2).sum(dim=0).clamp_min(1e-12)
        r2 = 1.0 - ss_res / ss_tot

        for idx, name in enumerate(TARGET_NAMES):
            d = diff[:, idx]
            metrics[f"rmse_{name}"] = torch.sqrt((d ** 2).mean()).item()
            metrics[f"mae_{name}"] = d.abs().mean().item()
            metrics[f"r2_{name}"] = r2[idx].item()

        vel_p = p[:, :3].norm(dim=1)
        vel_t = t[:, :3].norm(dim=1)
        metrics["rmse_vel_mag"] = torch.sqrt(((vel_p - vel_t) ** 2).mean()).item()
        metrics["mae_vel_mag"] = (vel_p - vel_t).abs().mean().item()

        results[region_name] = metrics

    return results


@dataclass
class RegionalMeter:
    """Accumulates predictions across batches for regional evaluation."""
    preds: List[torch.Tensor] = field(default_factory=list)
    targets: List[torch.Tensor] = field(default_factory=list)
    features: List[torch.Tensor] = field(default_factory=list)

    def update(self, pred: torch.Tensor, target: torch.Tensor, node_features: torch.Tensor):
        self.preds.append(pred.detach().cpu())
        self.targets.append(target.detach().cpu())
        self.features.append(node_features.detach().cpu())

    def compute(self, **mask_kwargs) -> Dict[str, Dict[str, float]]:
        pred = torch.cat(self.preds, dim=0)
        target = torch.cat(self.targets, dim=0)
        features = torch.cat(self.features, dim=0)
        return compute_regional_metrics(pred, target, features, **mask_kwargs)

    def reset(self):
        self.preds.clear()
        self.targets.clear()
        self.features.clear()
