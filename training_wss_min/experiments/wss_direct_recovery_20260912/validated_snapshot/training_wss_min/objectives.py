"""Training losses and validation-selection rules.

Keeping these pure computations outside the CLI makes the training protocol easy
to test and lets diagnostic experiments reuse exactly the same implementation.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
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


def _pair_inputs(prediction, target, batch_index, name):
    if prediction.ndim != 1 or target.shape != prediction.shape:
        raise ValueError(f"{name} requires matching 1-D prediction/target tensors")
    if batch_index is None or batch_index.shape != prediction.shape:
        raise ValueError(f"{name} requires one case index per query")
    if batch_index.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"{name} requires integer case indices")
    return batch_index.detach().cpu().numpy(), target.detach().float().cpu().numpy()


def _cap_pairs(pairs, max_pairs):
    # Spread the budget across candidates instead of taking only their prefix.
    if len(pairs) > max_pairs:
        pairs = pairs[np.linspace(0, len(pairs) - 1, max_pairs, dtype=np.int64)]
    return pairs


def _casebalanced_pair_loss(prediction, target, case_pairs, *, ranking, diagnostics):
    """Average edges per contributing case, then give those cases equal weight."""
    counts = [len(pairs) for pairs in case_pairs]
    missing = sum(count == 0 for count in counts)
    name = "pairwise_ranking" if ranking else "local_difference"
    if diagnostics is not None:
        for key, value in {
            "pair_count": sum(counts), "case_count": len(counts),
            "cases_with_pairs": len(counts) - missing, "cases_without_pairs": missing,
            "min_pairs_per_case": min(counts, default=0),
            "max_pairs_per_case": max(counts, default=0),
        }.items():
            diagnostics[f"{name}_{key}"] = prediction.new_tensor(value).detach()
    if missing:
        warnings.warn(
            f"{name}: {missing}/{len(counts)} cases have no eligible pairs; "
            "inspect pair-count diagnostics (empty cases do not contribute).",
            RuntimeWarning, stacklevel=3,
        )
    losses = []
    for pairs in case_pairs:
        if not len(pairs):
            continue
        indices = torch.as_tensor(pairs, device=prediction.device, dtype=torch.long)
        a, b = indices.unbind(dim=1)
        dp = prediction[a].float() - prediction[b].float()
        dy = target[a].float() - target[b].float()
        if ranking:
            losses.append(torch.nn.functional.softplus(-torch.sign(dy) * dp).mean())
        else:
            losses.append(torch.nn.functional.smooth_l1_loss(dp, dy, reduction="mean"))
    return torch.stack(losses).mean() if losses else prediction.sum() * 0.0


def local_difference_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    pos: torch.Tensor | None,
    batch_index: torch.Tensor | None,
    *,
    query_geometry: torch.Tensor | None = None,
    min_distance_mm: float = 2.0,
    max_distance_mm: float = 5.0,
    normal_cos_min: float = 0.5,
    normal_displacement_max: float = 0.5,
    margin: float = 0.0,
    max_pairs: int = 256,
    diagnostics: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Match local surface differences in physical millimetres.

    Raw geometry is [normal(3), tangent(3), radius_mm, coord_scale_mm].
    A KD-tree radius search finds true nearby candidates, including across
    spatial-bin boundaries. Normal compatibility and near-tangential chords
    reject common cross-wall shortcuts; this is an approximate surface graph,
    not exact mesh geodesics. At most 512 anchor queries bound search cost;
    the final budget is ``max_pairs`` for *each* case, after filtering.
    """
    ids, truth = _pair_inputs(prediction, target, batch_index, "local difference loss")
    n = prediction.numel()
    if pos is None or pos.shape != (n, 3):
        raise ValueError("local difference loss requires normalized query positions [N,3]")
    if query_geometry is None or query_geometry.shape != (n, 8):
        raise ValueError("local difference loss requires raw query_geometry [N,8]")
    if not (0 <= min_distance_mm < max_distance_mm < float("inf")):
        raise ValueError("local difference loss requires 0 <= min_mm < max_mm < inf")
    if max_pairs < 1 or not (-1 <= normal_cos_min <= 1) or not (0 <= normal_displacement_max <= 1):
        raise ValueError("invalid local difference pair budget or normal thresholds")
    # Metadata transfers are batch-wise; there is no Python loop over GPU voxels.
    geometry = query_geometry.detach().float().cpu().numpy().astype(np.float64)
    positions = pos.detach().float().cpu().numpy().astype(np.float64)
    if not np.isfinite(geometry).all() or not np.isfinite(positions).all():
        raise ValueError("local difference geometry/positions must be finite")
    norms = np.linalg.norm(geometry[:, :3], axis=1)
    if np.any(geometry[:, 7] <= 0) or np.any(norms < 1e-6):
        raise ValueError("local difference requires positive coordinate scales and nonzero normals")
    normals = geometry[:, :3] / norms[:, None]
    positions_mm = positions * geometry[:, 7:8]
    from scipy.spatial import cKDTree

    case_pairs = []
    for case in np.unique(ids):
        rows = np.flatnonzero(ids == case)
        if not np.allclose(geometry[rows, 7], geometry[rows[0], 7], rtol=1e-5, atol=1e-6):
            raise ValueError("local difference coordinate scale must be constant within each case")
        if len(rows) < 2:
            case_pairs.append(np.empty((0, 2), dtype=np.int64))
            continue
        points = positions_mm[rows]
        tree = cKDTree(points)
        anchors = np.linspace(0, len(rows) - 1, min(len(rows), 512), dtype=np.int64)
        neighbours = tree.query_ball_point(points[anchors], max_distance_mm, workers=1)
        a = np.repeat(anchors, [len(group) for group in neighbours])
        b = np.concatenate(neighbours).astype(np.int64, copy=False)
        pairs = np.unique(np.sort(np.column_stack((a, b)), axis=1), axis=0)
        delta = points[pairs[:, 0]] - points[pairs[:, 1]]
        distance = np.linalg.norm(delta, axis=1)
        na, nb = normals[rows[pairs[:, 0]]], normals[rows[pairs[:, 1]]]
        keep = ((distance >= min_distance_mm) & (distance <= max_distance_mm)
                & (distance > 1e-8) & ((na * nb).sum(axis=1) >= normal_cos_min)
                & (np.abs((delta * na).sum(axis=1)) <= normal_displacement_max * distance)
                & (np.abs((delta * nb).sum(axis=1)) <= normal_displacement_max * distance))
        if margin > 0:
            keep &= np.abs(truth[rows[pairs[:, 0]]] - truth[rows[pairs[:, 1]]]) >= margin
        case_pairs.append(rows[_cap_pairs(pairs[keep], max_pairs)])
    return _casebalanced_pair_loss(
        prediction, target, case_pairs, ranking=False, diagnostics=diagnostics,
    )


def pairwise_ranking_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    batch_index: torch.Tensor | None,
    *,
    margin: float = 0.05,
    max_pairs: int = 256,
    diagnostics: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Within-case pairwise logistic ranking objective.

    Dataset query order is randomly sampled, so consecutive non-overlapping
    pairs provide a deterministic random-pair sample without another RNG.
    Ties/small target differences are filtered before the per-case budget;
    cases receive equal weight irrespective of their eligible pair counts.
    """
    ids, truth = _pair_inputs(prediction, target, batch_index, "pairwise ranking loss")
    if margin < 0 or not math.isfinite(margin) or max_pairs < 1:
        raise ValueError("ranking requires a finite nonnegative margin and positive per-case budget")
    case_pairs = []
    for case in np.unique(ids):
        rows = np.flatnonzero(ids == case)
        pairs = rows[:len(rows) - len(rows) % 2].reshape(-1, 2)
        keep = np.abs(truth[pairs[:, 0]] - truth[pairs[:, 1]]) > margin
        case_pairs.append(_cap_pairs(pairs[keep], max_pairs))
    return _casebalanced_pair_loss(
        prediction, target, case_pairs, ranking=True, diagnostics=diagnostics,
    )


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

    # Optional local / ranking auxiliaries.  They are deliberately discovered
    # via getattr so older TrainConfig objects and JSON files retain identical
    # behaviour until the new fields are explicitly registered and non-zero.
    local_weight = float(getattr(cfg, "loss_local_diff_lambda", 0.0) or 0.0)
    if local_weight > 0:
        if pred_for_raw_loss.ndim != 1:
            raise ValueError("local difference loss currently supports scalar targets only")
        local = local_difference_loss(
            pred_for_raw_loss,
            y,
            batch.get("pos"),
            batch.get("batch"),
            query_geometry=batch.get("query_geometry"),
            min_distance_mm=float(getattr(cfg, "local_diff_min_distance_mm", 2.0)),
            max_distance_mm=float(getattr(cfg, "local_diff_max_distance_mm", 5.0)),
            normal_cos_min=float(getattr(cfg, "local_diff_normal_cos_min", 0.5)),
            normal_displacement_max=float(getattr(cfg, "local_diff_normal_displacement_max", 0.5)),
            margin=float(getattr(cfg, "local_diff_margin", 0.0) or 0.0),
            max_pairs=int(getattr(cfg, "local_diff_max_pairs", 256) or 256),
            diagnostics=components,
        )
        loss = loss + local_weight * local
        if components is not None:
            components["local_difference"] = local.detach()
            components["local_difference_weighted"] = (local_weight * local).detach()

    rank_weight = float(getattr(cfg, "loss_pairwise_rank_lambda", 0.0) or 0.0)
    if rank_weight > 0:
        if pred_for_raw_loss.ndim != 1:
            raise ValueError("pairwise ranking loss currently supports scalar targets only")
        ranking = pairwise_ranking_loss(
            pred_for_raw_loss, y, batch.get("batch"),
            margin=float(getattr(cfg, "pairwise_rank_margin", 0.05)),
            max_pairs=int(getattr(cfg, "pairwise_rank_max_pairs", 256) or 256),
            diagnostics=components,
        )
        loss = loss + rank_weight * ranking
        if components is not None:
            components["pairwise_ranking"] = ranking.detach()
            components["pairwise_ranking_weighted"] = (rank_weight * ranking).detach()
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
