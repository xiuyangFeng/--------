"""Training losses and validation-selection rules.

Keeping these pure computations outside the CLI makes the training protocol easy
to test and lets diagnostic experiments reuse exactly the same implementation.
"""

from __future__ import annotations

import math

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


def raw_space_mse(
    prediction: torch.Tensor, target: torch.Tensor, cfg: TrainConfig,
    target_stats: dict | None,
) -> torch.Tensor:
    """Pa squared error scaled by the frozen training p90; no clipping/fallback."""
    if prediction.ndim != 1 or target.ndim != 1 or prediction.shape != target.shape:
        raise ValueError("Pa-MSE requires matching one-dimensional scalar WSS tensors")
    if target_stats is None or not {"log", "eps"} <= target_stats.keys():
        raise ValueError("Pa-MSE requires global log_z target statistics")
    if target_stats.get("method", "log_z") != "log_z":
        raise ValueError("Pa-MSE requires global log_z target statistics")
    raw_p90 = getattr(cfg, "_raw_p90", None)
    if raw_p90 is None:
        raw_p90 = target_stats.get("raw_percentiles", {}).get("p90")
    if raw_p90 is None or not math.isfinite(float(raw_p90)) or float(raw_p90) <= 0:
        raise ValueError("Pa-MSE requires a finite positive train-only raw p90")
    mean, std = float(target_stats["log"]["mean"]), float(target_stats["log"]["std"])
    eps = float(target_stats["eps"])
    if not all(math.isfinite(x) for x in (mean, std, eps)) or std <= 0 or eps < 0:
        raise ValueError("Pa-MSE requires finite global log statistics, std > 0 and eps >= 0")
    with torch.autocast(device_type=prediction.device.type, enabled=False):
        recovered = torch.exp(prediction.float() * std + mean) - eps
        result = ((recovered - target.float()) / float(raw_p90)).square().mean()
    if not bool(torch.isfinite(result)):
        raise FloatingPointError("non-finite Pa-MSE (unclipped FP32 physical recovery)")
    return result


def joint_volume_loss(
    pred: torch.Tensor, target: torch.Tensor, batch_index: torch.Tensor,
    velocity_mask: torch.Tensor, pressure_mask: torch.Tensor,
    components: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Equal cases, equal tasks; velocity averages its three normalized components.

    Slice supervision before arithmetic so placeholder/unobserved labels cannot
    influence either the value or the gradient (including NaN placeholders).
    """
    if pred.ndim != 2 or pred.shape[1] != 4 or pred.shape != target.shape:
        raise ValueError("joint volume loss requires matching [N,4] ux/uy/uz/pressure tensors")
    for name, mask in (("velocity_mask", velocity_mask), ("pressure_mask", pressure_mask)):
        if mask.shape != (len(pred),) or mask.dtype != torch.bool:
            raise ValueError(f"{name} must be a boolean vector with one value per query")
    if batch_index.shape != (len(pred),) or len(pred) == 0:
        raise ValueError("joint volume loss requires a case index for every nonempty query")
    velocity_losses, pressure_losses = [], []
    for case_index in torch.unique(batch_index, sorted=True):
        case_rows = batch_index == case_index
        vrows, prows = case_rows & velocity_mask, case_rows & pressure_mask
        if not bool(vrows.any()) or not bool(prows.any()):
            raise ValueError("each joint case requires both velocity and pressure supervision")
        velocity_losses.append((pred[vrows, :3].float() - target[vrows, :3].float()).square().mean())
        pressure_losses.append((pred[prows, 3].float() - target[prows, 3].float()).square().mean())
    velocity = torch.stack(velocity_losses).mean()
    pressure = torch.stack(pressure_losses).mean()
    total = 0.5 * velocity + 0.5 * pressure
    if components is not None:
        components.update({"velocity_mse": velocity.detach(), "pressure_mse": pressure.detach(),
                           "velocity_weighted": (0.5 * velocity).detach(),
                           "pressure_weighted": (0.5 * pressure).detach(),
                           "base": total.detach(), "total": total.detach()})
    return total


def compute_loss(
    pred: torch.Tensor,
    batch: dict,
    cfg: TrainConfig,
    device: str,
    target_stats: dict | None = None,
    components: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Compute the configured scalar, velocity, or masked joint objective."""
    y = batch["y"].to(device)
    if "velocity_mask" in batch or "pressure_mask" in batch:
        if not {"velocity_mask", "pressure_mask", "batch"} <= batch.keys():
            raise ValueError("joint volume supervision requires both masks and case indices")
        if (cfg.loss != "mse" or cfg.loss_geom_weight or cfg.loss_weight_target
                or any(float(getattr(cfg, key, 0.0) or 0.0) != 0.0 for key in (
                    "loss_pinball_lambda", "loss_hotspot_bce_lambda", "loss_raw_huber_lambda",
                    "loss_raw_mse_lambda"))):
            raise ValueError("joint volume loss supports only equal-weight masked MSE")
        return joint_volume_loss(
            pred, y, batch["batch"].to(device), batch["velocity_mask"].to(device),
            batch["pressure_mask"].to(device), components,
        )
    if y.ndim == 2 and y.shape[-1] == 4:
        raise ValueError("joint volume targets require explicit supervision masks")
    hotspot_logits = None
    if cfg.loss == "gaussian_nll":
        if pred.ndim != 2 or pred.shape[-1] != 2:
            raise ValueError("gaussian_nll requires model out_dim=2")
        mu = pred[:, 0]
        logvar = torch.clamp(pred[:, 1], cfg.nll_logvar_min, cfg.nll_logvar_max)
        per_point = 0.5 * (logvar + (y - mu) ** 2 / torch.exp(logvar))
        per_point = per_point + cfg.nll_logvar_reg * logvar.square()
        pred_for_raw_loss = mu
    elif pred.ndim == 2 and y.ndim == 2 and pred.shape == y.shape:
        # vector target (velocity components): per-point mean over channels
        if cfg.loss == "huber":
            per_point = torch.nn.functional.huber_loss(
                pred, y, delta=cfg.huber_delta, reduction="none"
            ).mean(-1)
        else:
            per_point = (pred - y).square().mean(-1)
        pred_for_raw_loss = pred
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
    if components is not None:
        components["base"] = loss.detach()

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
        if components is not None:
            components["raw_huber"] = raw_huber.detach()
            components["raw_huber_weighted"] = (raw_loss_weight * raw_huber).detach()

    raw_mse_weight = float(getattr(cfg, "loss_raw_mse_lambda", 0.0) or 0.0)
    if raw_mse_weight > 0:
        if cfg.loss == "gaussian_nll":
            raise ValueError("Pa-MSE is not supported with gaussian_nll")
        raw_mse = raw_space_mse(
            pred_for_raw_loss, batch["y_raw"].to(device), cfg, target_stats,
        )
        loss = loss + raw_mse_weight * raw_mse
        if components is not None:
            components["raw_mse"] = raw_mse.detach()
            components["raw_mse_weighted"] = (raw_mse_weight * raw_mse).detach()

    pinball_weight = float(cfg.loss_pinball_lambda or 0.0)
    if pinball_weight > 0:
        pinball = pinball_loss(
            pred_for_raw_loss, y, cfg.pinball_quantile
        )
        loss = loss + pinball_weight * pinball
        if components is not None:
            components["pinball"] = pinball.detach()
            components["pinball_weighted"] = (pinball_weight * pinball).detach()

    hotspot_weight = float(cfg.loss_hotspot_bce_lambda or 0.0)
    if hotspot_weight > 0:
        if hotspot_logits is None:
            raise ValueError("hotspot BCE requires a second prediction channel")
        hotspot = casebalanced_hotspot_bce(
            hotspot_logits,
            y,
            batch["batch"].to(device),
            cfg.hotspot_quantile,
        )
        loss = loss + hotspot_weight * hotspot
        if components is not None:
            components["hotspot_bce"] = hotspot.detach()
            components["hotspot_bce_weighted"] = (hotspot_weight * hotspot).detach()
    if components is not None:
        components["total"] = loss.detach()
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
