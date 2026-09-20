"""Model factory for the core trainer and evaluator."""

from __future__ import annotations

from torch import nn

from .baseline_models import build_baseline_model
from .pointnext import PointNeXtSeg

BASELINE_MODELS = frozenset({"mlp", "pointnet", "pointnetpp"})
SUPPORTED_MODELS = BASELINE_MODELS | {"pointnext_s"}


def build_model(model_cfg, in_dim: int) -> nn.Module:
    """Build a configured model without coupling baseline code to PointNeXt."""
    if model_cfg.name in BASELINE_MODELS:
        return build_baseline_model(model_cfg, in_dim)
    if model_cfg.name != "pointnext_s":
        expected = "/".join(sorted(SUPPORTED_MODELS))
        raise ValueError(f"unknown model.name={model_cfg.name!r}; expected {expected}")

    stage_lengths = {
        len(model_cfg.sa_ratios),
        len(model_cfg.sa_radius),
        len(model_cfg.sa_nsample),
        len(model_cfg.sa_blocks),
    }
    if len(stage_lengths) != 1:
        raise ValueError("PointNeXt SA ratios/radii/nsample/blocks must have equal length")
    return PointNeXtSeg(
        in_dim=in_dim,
        width=model_cfg.width,
        sa_ratios=tuple(model_cfg.sa_ratios),
        sa_radius=tuple(model_cfg.sa_radius),
        sa_nsample=tuple(model_cfg.sa_nsample),
        sa_blocks=tuple(model_cfg.sa_blocks),
        invres_radius_scale=model_cfg.invres_radius_scale,
        fp_knn=model_cfg.fp_knn,
        head_hidden=model_cfg.head_hidden,
        out_dim=model_cfg.out_dim,
        dropout=model_cfg.dropout,
    )
