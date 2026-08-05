"""Data, boundary and PDE losses for the configured experiment route."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .physics import generalized_newtonian_residuals, no_slip_loss
from .physics.residuals import denormalize_fields


DATA_NAMES = ("u", "v", "w", "pressure")
MOMENTUM_NAMES = ("x", "y", "z")


def _data_losses(
    prediction: torch.Tensor,
    target: torch.Tensor,
    config,
) -> dict[str, torch.Tensor]:
    """Return the four supervised MSE terms and their configured sum."""
    raw = [
        F.mse_loss(prediction[:, index], target[:, index])
        for index in range(4)
    ]
    weights = [
        *[float(value) for value in config["loss"]["velocity_component_weights"]],
        float(config["loss"]["pressure_weight"]),
    ]
    # V3 registered an equal four-channel mean; V1/V2 registered a weighted sum.
    scale = 0.25 if config.route == "volume_uvwp_peak_qs_smooth_v3" else 1.0
    weighted = [value * weight * scale for value, weight in zip(raw, weights)]
    losses = {
        f"data_{name}_raw": value for name, value in zip(DATA_NAMES, raw)
    }
    losses.update(
        {
            f"data_{name}_weighted": value
            for name, value in zip(DATA_NAMES, weighted)
        }
    )
    losses["data_total"] = sum(weighted)
    return losses


def _pde_losses(
    model,
    encoded: dict[str, torch.Tensor],
    batch: dict[str, Any],
    *,
    field_stats: dict[str, Any],
    config,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    coords = batch["physics_coords"].detach().clone().requires_grad_(True)
    prediction = model.decode_query(encoded, coords, batch["physics_batch"])
    residuals = generalized_newtonian_residuals(
        prediction,
        coords,
        batch["physics_length_m"],
        field_stats=field_stats,
        physics_config=config["physics"],
    )
    continuity = torch.mean(torch.square(residuals["continuity_hat"]))
    momentum = [
        torch.mean(torch.square(residuals["momentum_hat"][:, index]))
        for index in range(3)
    ]
    return continuity, momentum


def _v1_physics_losses(
    model,
    encoded: dict[str, torch.Tensor],
    batch: dict[str, Any],
    *,
    field_stats: dict[str, Any],
    config,
) -> dict[str, torch.Tensor]:
    if config.mode != "pinn":
        return {}

    continuity, momentum = _pde_losses(
        model, encoded, batch, field_stats=field_stats, config=config
    )
    wall_prediction = model.decode_query(
        encoded, batch["wall_coords"], batch["wall_batch"]
    )
    no_slip, _ = no_slip_loss(
        wall_prediction,
        field_stats=field_stats,
        reference_velocity_m_s=float(
            config["physics"]["reference_velocity_m_s"]
        ),
    )
    continuity_weighted = continuity * float(
        config["loss"]["continuity_weight"]
    )
    momentum_weighted = [
        value * float(config["loss"]["momentum_weight"]) for value in momentum
    ]
    no_slip_weighted = no_slip * float(config["loss"]["no_slip_weight"])
    return {
        "continuity_raw": continuity,
        **{
            f"momentum_{name}_raw": value
            for name, value in zip(MOMENTUM_NAMES, momentum)
        },
        "no_slip_raw": no_slip,
        "wall_raw": no_slip,
        "continuity_weighted": continuity_weighted,
        **{
            f"momentum_{name}_weighted": value
            for name, value in zip(MOMENTUM_NAMES, momentum_weighted)
        },
        "no_slip_weighted": no_slip_weighted,
        "bc_group_raw": no_slip,
        "bc_weighted": no_slip_weighted,
        "pde_group_raw": continuity + sum(momentum),
        "pde_weighted": continuity_weighted + sum(momentum_weighted),
    }


def _v3_boundary_losses(
    model,
    encoded: dict[str, torch.Tensor],
    batch: dict[str, Any],
    *,
    field_stats: dict[str, Any],
    config,
) -> dict[str, torch.Tensor]:
    if config.mode == "data":
        return {}

    velocity_scale = float(config["physics"]["reference_velocity_m_s"])
    pressure_scale = float(config["physics"]["density_kg_m3"]) * velocity_scale**2

    wall_prediction = model.decode_query(
        encoded, batch["wall_coords"], batch["wall_batch"]
    )
    wall_velocity, _ = denormalize_fields(wall_prediction, field_stats)
    wall = torch.mean(torch.square(wall_velocity / velocity_scale))

    inlet_prediction = model.decode_query(
        encoded, batch["inlet_coords"], batch["inlet_batch"]
    )
    inlet_velocity, _ = denormalize_fields(inlet_prediction, field_stats)
    inlet = torch.mean(
        torch.square(
            (inlet_velocity - batch["inlet_velocity_m_s"]) / velocity_scale
        )
    )

    outlet_prediction = model.decode_query(
        encoded, batch["outlet_coords"], batch["outlet_batch"]
    )
    _, outlet_pressure = denormalize_fields(outlet_prediction, field_stats)
    outlet = torch.mean(
        torch.square(
            (outlet_pressure - batch["outlet_pressure_relative_pa"])
            / pressure_scale
        )
    )
    group = (wall + inlet + outlet) / 3.0
    return {
        "wall_raw": wall,
        "inlet_raw": inlet,
        "outlet_raw": outlet,
        "bc_group_raw": group,
        "bc_weighted": group * float(config["loss"]["lambda_bc"]),
    }


def _v3_pde_losses(
    model,
    encoded: dict[str, torch.Tensor],
    batch: dict[str, Any],
    *,
    field_stats: dict[str, Any],
    config,
) -> dict[str, torch.Tensor]:
    if config.mode != "data_bc_pde":
        return {}
    continuity, momentum = _pde_losses(
        model, encoded, batch, field_stats=field_stats, config=config
    )
    group = (continuity + sum(momentum) / 3.0) / 2.0
    return {
        "continuity_raw": continuity,
        **{
            f"momentum_{name}_raw": value
            for name, value in zip(MOMENTUM_NAMES, momentum)
        },
        "pde_group_raw": group,
        "pde_weighted": group * float(config["loss"]["lambda_pde"]),
    }


def compute_losses(
    model,
    batch: dict[str, Any],
    *,
    field_stats: dict[str, Any],
    config,
    epoch: int,
) -> dict[str, torch.Tensor]:
    """Compute all configured losses from one case batch."""
    encoded = model.encode_support(
        batch["support_coords"],
        batch["support_features"],
        batch["support_batch"],
        unit_ids=batch["case_ids"],
        epoch=int(epoch),
        global_seed=int(config["train"]["seed"]),
        evaluation=False,
    )
    prediction = model.decode_query(
        encoded, batch["query_coords"], batch["query_batch"]
    )
    zero = prediction.new_zeros(())
    losses = {
        name: zero
        for name in (
            "wall_raw",
            "inlet_raw",
            "outlet_raw",
            "bc_group_raw",
            "bc_weighted",
            "continuity_raw",
            "momentum_x_raw",
            "momentum_y_raw",
            "momentum_z_raw",
            "no_slip_raw",
            "continuity_weighted",
            "momentum_x_weighted",
            "momentum_y_weighted",
            "momentum_z_weighted",
            "no_slip_weighted",
            "pde_group_raw",
            "pde_weighted",
        )
    }
    losses.update(_data_losses(prediction, batch["query_target"], config))

    if config.route == "volume_uvwp_peak_qs_smooth_v3":
        losses.update(
            _v3_boundary_losses(
                model, encoded, batch, field_stats=field_stats, config=config
            )
        )
        losses.update(
            _v3_pde_losses(
                model, encoded, batch, field_stats=field_stats, config=config
            )
        )
    else:
        losses.update(
            _v1_physics_losses(
                model, encoded, batch, field_stats=field_stats, config=config
            )
        )

    losses["physics_total"] = losses["bc_weighted"] + losses["pde_weighted"]
    losses["total"] = losses["data_total"] + losses["physics_total"]
    denominator = losses["total"].detach().clamp_min(1e-30)
    losses["data_contribution_ratio"] = losses["data_total"] / denominator
    losses["bc_contribution_ratio"] = losses["bc_weighted"] / denominator
    losses["pde_contribution_ratio"] = losses["pde_weighted"] / denominator
    return losses
