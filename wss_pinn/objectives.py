from __future__ import annotations

import torch
import torch.nn.functional as F

from .physics.residuals import continuity_residual


def gauge_invariant_pressure_mse(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    prediction_centered = prediction - prediction.mean()
    target_centered = target - target.mean()
    return F.mse_loss(prediction_centered, target_centered)


def no_slip_loss(velocity_at_wall: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.square(velocity_at_wall))


def stage_losses(
    model,
    descriptor: torch.Tensor,
    wall_coords: torch.Tensor,
    wall_wss: torch.Tensor,
    interior_coords: torch.Tensor,
    velocity_target: torch.Tensor,
    pressure_target: torch.Tensor,
    weights: dict,
) -> dict[str, torch.Tensor]:
    requires_derivative = float(weights["continuity_weight"]) > 0
    interior_coords = interior_coords.requires_grad_(requires_derivative)
    wall_output = model(wall_coords, descriptor)
    interior_output = model(interior_coords, descriptor)
    losses = {
        "direct_wss": F.mse_loss(
            wall_output["log_wss_direct"], torch.log1p(torch.clamp(wall_wss, min=0))
        ),
        "velocity_data": F.mse_loss(interior_output["velocity"], velocity_target),
        "pressure_data": gauge_invariant_pressure_mse(
            interior_output["pressure"], pressure_target
        ),
        "continuity": torch.zeros((), device=wall_coords.device, dtype=wall_coords.dtype),
        "no_slip": torch.zeros((), device=wall_coords.device, dtype=wall_coords.dtype),
    }
    if requires_derivative:
        divergence = continuity_residual(
            interior_output["velocity"], interior_coords, create_graph=True
        )
        losses["continuity"] = torch.mean(torch.square(divergence))
    if float(weights["no_slip_weight"]) > 0:
        losses["no_slip"] = no_slip_loss(wall_output["velocity"])
    total = sum(
        losses[name] * float(weights[f"{name}_weight"])
        for name in ("direct_wss", "velocity_data", "pressure_data", "continuity", "no_slip")
    )
    return {**losses, "total": total}

