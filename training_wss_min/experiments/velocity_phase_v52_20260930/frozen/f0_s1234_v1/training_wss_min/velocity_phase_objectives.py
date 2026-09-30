"""Velocity-only phase objectives; historical runners never import this module."""
from __future__ import annotations

import torch

from .joint_cycle import case_weighted_loss as original_z_loss


def case_phase_mean(values, weights, batch, multiplier=None):
    """[query,phase] means, normalized independently within each case/phase."""
    if values.ndim != 2 or weights.ndim != 1 or batch.ndim != 1:
        raise ValueError("expected values[N,F], weights[N], batch[N]")
    if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
        raise ValueError("invalid quadrature weights")
    result = []
    for index in range(int(batch.max().item()) + 1):
        selected = batch == index
        w = weights[selected].float()[:, None].expand_as(values[selected])
        if multiplier is not None:
            w = w * multiplier[selected].float()
        denominator = w.sum(0)
        # A masked phase without eligible points contributes zero, not a new denominator.
        numerator = (values[selected].float() * w).sum(0)
        result.append(torch.where(denominator > 0,
                                  numerator / denominator.clamp_min(1e-30),
                                  numerator * 0).mean())
    return torch.stack(result).mean()


def physical_velocity(pred_z, stats):
    spec = stats["targets"]["velocity"]
    return (pred_z.float() * pred_z.new_tensor(spec["std"], dtype=torch.float32)[None]
            + pred_z.new_tensor(spec["mean"], dtype=torch.float32)[None])


def direction_loss(pred_raw, truth_raw, weights, batch, floor=0.01, epsilon=0.01):
    if min(floor, epsilon) <= 0:
        raise ValueError("direction thresholds must be positive")
    predicted = pred_raw.float()
    truth = truth_raw.float()
    speed = torch.linalg.vector_norm(truth, dim=-1)
    pred_speed = torch.linalg.vector_norm(predicted, dim=-1)
    eligible = speed >= floor
    cosine = (predicted * truth).sum(-1) / (
        pred_speed.clamp_min(epsilon) * speed.clamp_min(epsilon))
    return case_phase_mean(1 - cosine.clamp(-1, 1), weights, batch, eligible)


@torch.no_grad()
def direction_coverage(truth_raw, weights, batch, floor=0.01):
    """Volume coverage and empty case/phase counts for the direction objective."""
    eligible = torch.linalg.vector_norm(truth_raw.float(), dim=-1) >= floor
    empty = 0
    cases = int(batch.max().item()) + 1
    for index in range(cases):
        selected = batch == index
        eligible_volume = (eligible[selected] * weights[selected, None]).sum(0)
        empty += int((eligible_volume <= 0).sum())
    return {"eligible_volume_fraction": float(case_phase_mean(eligible.float(), weights, batch)),
            "empty_case_phases": empty,
            "total_case_phases": cases * truth_raw.shape[1]}


def velocity_objective(pred_z, q, stats, arm, *, direction_lambda=0., q80=None,
                       direction_floor=0.01, direction_epsilon=0.01):
    error = (pred_z.float() - q["y"].float()).square().mean(-1)
    # Preserve the historical reduction order for every unchanged base objective.
    base = original_z_loss(pred_z, q["y"], q["weight"], q["batch"])
    parts = {"z_mse": base}
    total = base
    if arm == "D1":
        if not 1e-4 <= float(direction_lambda) <= 1:
            raise ValueError("D1 lambda must be calibrated and frozen in [1e-4,1]")
        auxiliary = direction_loss(physical_velocity(pred_z, stats), q["y_raw"],
                                   q["weight"], q["batch"], direction_floor, direction_epsilon)
        total = base + float(direction_lambda) * auxiliary
        parts.update(direction=auxiliary, direction_weighted=float(direction_lambda) * auxiliary)
    elif arm == "D2":
        if q80 is None or tuple(q80.shape) != (pred_z.shape[1],):
            raise ValueError("D2 requires 80 train-only q80 thresholds")
        tail = torch.linalg.vector_norm(q["y_raw"].float(), dim=-1) >= q80[None]
        total = case_phase_mean(error, q["weight"], q["batch"], 1. + tail.float())
        parts["tail_weighted_z_mse"] = total
    parts["total"] = total
    return total, parts


def gradient_l2(loss, parameters, *, retain_graph=False):
    gradients = torch.autograd.grad(loss, parameters, retain_graph=retain_graph,
                                    allow_unused=True)
    return torch.sqrt(sum((x.float().square().sum() for x in gradients if x is not None),
                          loss.new_zeros(()))).detach()
