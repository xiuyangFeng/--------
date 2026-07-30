"""Training losses and validation-selection rules.

Keeping these pure computations outside the CLI makes the training protocol easy
to test and lets diagnostic experiments reuse exactly the same implementation.
"""

from __future__ import annotations

import torch

from . import metrics as M
from .config import TrainConfig


def fixed_quantile_scale(a: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """Map values to [0, 1] using frozen train-only quantiles."""
    return torch.clamp((a - lo) / (hi - lo + 1e-9), 0, 1)


def _batch_quantile_scale(a: torch.Tensor) -> torch.Tensor:
    lo, hi = torch.quantile(a, 0.02), torch.quantile(a, 0.98)
    return torch.clamp((a - lo) / (hi - lo + 1e-9), 0, 1)


def casebalanced_hotspot_bce(
    logits: torch.Tensor,
    target: torch.Tensor,
    batch_index: torch.Tensor,
    quantile: float,
) -> torch.Tensor:
    """Balanced per-case BCE for a case-relative high-WSS hotspot mask."""
    losses = []
    for case_index in torch.unique(batch_index, sorted=True):
        mask = batch_index == case_index
        case_target = target[mask].float()
        case_logits = logits[mask].float()
        threshold = torch.quantile(case_target.detach(), float(quantile))
        positive = case_target >= threshold
        negative = ~positive
        if not positive.any() or not negative.any():
            continue
        positive_loss = torch.nn.functional.softplus(-case_logits[positive]).mean()
        negative_loss = torch.nn.functional.softplus(case_logits[negative]).mean()
        losses.append(0.5 * (positive_loss + negative_loss))
    if not losses:
        raise ValueError("hotspot BCE requires both hotspot and non-hotspot points")
    return torch.stack(losses).mean()


def pinball_loss(
    prediction: torch.Tensor, target: torch.Tensor, quantile: float
) -> torch.Tensor:
    """Mean pinball loss for the requested upper conditional quantile."""
    residual = target.float() - prediction.float()
    q = float(quantile)
    return torch.maximum(q * residual, (q - 1.0) * residual).mean()


def compute_loss(
    pred: torch.Tensor,
    batch: dict,
    cfg: TrainConfig,
    device: str,
    target_stats: dict | None = None,
) -> torch.Tensor:
    """Compute the configured scalar-regression objective."""
    y = batch["y"].to(device)
    hotspot_logits = None
    if cfg.loss == "gaussian_nll":
        if pred.ndim != 2 or pred.shape[-1] != 2:
            raise ValueError("gaussian_nll requires model out_dim=2")
        mu = pred[:, 0]
        logvar = torch.clamp(pred[:, 1], cfg.nll_logvar_min, cfg.nll_logvar_max)
        per_point = 0.5 * (logvar + (y - mu) ** 2 / torch.exp(logvar))
        per_point = per_point + cfg.nll_logvar_reg * logvar.square()
        pred_for_raw_loss = mu
    else:
        if pred.ndim == 2:
            if pred.shape[-1] == 1:
                pred = pred.squeeze(-1)
            elif (
                pred.shape[-1] == 2
                and float(cfg.loss_hotspot_bce_lambda) > 0
            ):
                hotspot_logits = pred[:, 1]
                pred = pred[:, 0]
            else:
                raise ValueError(
                    "non-NLL prediction must have one regression channel, or "
                    "regression+hotspot channels when hotspot BCE is enabled"
                )
        if cfg.loss == "huber":
            per_point = torch.nn.functional.huber_loss(
                pred, y, delta=cfg.huber_delta, reduction="none"
            )
        else:
            per_point = (pred - y).square()
        pred_for_raw_loss = pred

    if cfg.loss_geom_weight or cfg.loss_weight_target:
        weights = torch.ones_like(per_point)
        use_fixed = bool(cfg.loss_weight_fixed_quantiles)
        if cfg.loss_geom_weight:
            curvature = batch["curv"].to(device)
            inverse_radius = batch["invr"].to(device)
            if use_fixed and cfg.curv_q02 is not None and cfg.curv_q98 is not None:
                curvature_weight = fixed_quantile_scale(
                    curvature, cfg.curv_q02, cfg.curv_q98
                )
            else:
                curvature_weight = _batch_quantile_scale(curvature)
            if use_fixed and cfg.invr_q02 is not None and cfg.invr_q98 is not None:
                radius_weight = fixed_quantile_scale(
                    inverse_radius, cfg.invr_q02, cfg.invr_q98
                )
            else:
                radius_weight = _batch_quantile_scale(inverse_radius)
            weights = (
                weights
                + cfg.loss_weight_curv * curvature_weight
                + cfg.loss_weight_invradius * radius_weight
            )
        if cfg.loss_weight_target:
            if use_fixed and cfg.y_norm_q02 is not None and cfg.y_norm_q98 is not None:
                target_weight = fixed_quantile_scale(y, cfg.y_norm_q02, cfg.y_norm_q98)
            else:
                target_weight = _batch_quantile_scale(y)
            weights = weights + cfg.loss_weight_target_alpha * target_weight
        loss = (per_point * weights).sum() / weights.sum()
    else:
        loss = per_point.mean()

    raw_loss_weight = float(cfg.loss_raw_huber_lambda or 0.0)
    if raw_loss_weight > 0 and target_stats is not None and cfg.loss != "gaussian_nll":
        y_raw = batch["y_raw"].to(device).float()
        mean = torch.as_tensor(
            target_stats["log"]["mean"], device=device, dtype=torch.float32
        )
        std = torch.as_tensor(
            target_stats["log"]["std"], device=device, dtype=torch.float32
        )
        eps = torch.as_tensor(target_stats["eps"], device=device, dtype=torch.float32)
        pred_raw = torch.exp(pred_for_raw_loss.float() * std + mean) - eps
        raw_p90 = float(
            getattr(cfg, "_raw_p90", None)
            or target_stats.get("raw_percentiles", {}).get("p90", 1.0)
        )
        scale = max(raw_p90, 1e-3)
        raw_huber = torch.nn.functional.huber_loss(
            pred_raw / scale,
            y_raw / scale,
            delta=cfg.raw_huber_delta,
            reduction="mean",
        )
        loss = loss + raw_loss_weight * raw_huber

    pinball_weight = float(cfg.loss_pinball_lambda or 0.0)
    if pinball_weight > 0:
        loss = loss + pinball_weight * pinball_loss(
            pred_for_raw_loss, y, cfg.pinball_quantile
        )

    hotspot_weight = float(cfg.loss_hotspot_bce_lambda or 0.0)
    if hotspot_weight > 0:
        if hotspot_logits is None:
            raise ValueError("hotspot BCE requires a second prediction channel")
        loss = loss + hotspot_weight * casebalanced_hotspot_bce(
            hotspot_logits,
            y,
            batch["batch"].to(device),
            cfg.hotspot_quantile,
        )
    return loss


def compute_selection_score(
    cfg: TrainConfig,
    aggregate: dict,
    field: dict,
    field_casebalanced: dict,
    calibration: dict,
    hotspot: dict,
) -> float:
    """Apply the checkpoint-selection rule recorded in the JSON config."""
    if cfg.selection_rule == "r4_composite_v1":
        return M.selection_score_r4_composite_v1(
            aggregate, field, calibration, hotspot
        )
    if cfg.selection_rule == "field_casebalanced":
        return float(field_casebalanced.get("r2", float("-inf")))
    if cfg.selection_rule == "field_raw":
        return float(field.get("r2", float("-inf")))
    if cfg.selection_rule == "casemean":
        return float(aggregate.get("r2_casemean", float("-inf")))
    if cfg.selection_rule == "train_loss":
        # Val-free protocol: train.py selects by -train_loss directly.
        return float("nan")
    if "casemean" in cfg.ckpt_metric:
        return float(aggregate.get("r2_casemean", float("-inf")))
    return float(field.get("r2", float("-inf")))
