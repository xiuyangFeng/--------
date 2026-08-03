"""Four-channel data supervision plus explicit physics constraints."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .physics import generalized_newtonian_residuals, no_slip_loss


def compute_losses(
    model,
    batch: dict[str, Any],
    *,
    field_stats: dict[str, Any],
    config,
    epoch: int,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Compute raw/weighted loss terms; physics is active from step one."""
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
    target = batch["query_target"]
    data_components = [
        F.mse_loss(prediction[:, component], target[:, component])
        for component in range(4)
    ]
    velocity_weights = [
        float(value) for value in config["loss"]["velocity_component_weights"]
    ]
    pressure_weight = float(config["loss"]["pressure_weight"])
    data_weighted_components = [
        data_components[index] * velocity_weights[index] for index in range(3)
    ] + [data_components[3] * pressure_weight]
    data_total = sum(data_weighted_components)

    zero = prediction.new_zeros(())
    losses: dict[str, torch.Tensor] = {
        "data_u_raw": data_components[0],
        "data_v_raw": data_components[1],
        "data_w_raw": data_components[2],
        "data_pressure_raw": data_components[3],
        "data_u_weighted": data_weighted_components[0],
        "data_v_weighted": data_weighted_components[1],
        "data_w_weighted": data_weighted_components[2],
        "data_pressure_weighted": data_weighted_components[3],
        "data_total": data_total,
        "continuity_raw": zero,
        "momentum_x_raw": zero,
        "momentum_y_raw": zero,
        "momentum_z_raw": zero,
        "no_slip_raw": zero,
        "continuity_weighted": zero,
        "momentum_x_weighted": zero,
        "momentum_y_weighted": zero,
        "momentum_z_weighted": zero,
        "no_slip_weighted": zero,
        "physics_total": zero,
    }
    diagnostics: dict[str, torch.Tensor] = {
        "prediction": prediction,
    }
    if config.mode == "pinn":
        physics_coords = batch["physics_coords"].detach().clone().requires_grad_(True)
        physics_prediction = model.decode_query(
            encoded, physics_coords, batch["physics_batch"]
        )
        residuals = generalized_newtonian_residuals(
            physics_prediction,
            physics_coords,
            batch["physics_length_m"],
            field_stats=field_stats,
            physics_config=config["physics"],
        )
        continuity = torch.mean(torch.square(residuals["continuity_hat"]))
        momentum_components = [
            torch.mean(torch.square(residuals["momentum_hat"][:, component]))
            for component in range(3)
        ]
        wall_prediction = model.decode_query(
            encoded, batch["wall_coords"], batch["wall_batch"]
        )
        no_slip, wall_speed = no_slip_loss(
            wall_prediction,
            field_stats=field_stats,
            reference_velocity_m_s=float(
                config["physics"]["reference_velocity_m_s"]
            ),
        )
        continuity_weighted = continuity * float(
            config["loss"]["continuity_weight"]
        )
        momentum_weight = float(config["loss"]["momentum_weight"])
        momentum_weighted = [value * momentum_weight for value in momentum_components]
        no_slip_weighted = no_slip * float(config["loss"]["no_slip_weight"])
        physics_total = continuity_weighted + sum(momentum_weighted) + no_slip_weighted
        losses.update(
            {
                "continuity_raw": continuity,
                "momentum_x_raw": momentum_components[0],
                "momentum_y_raw": momentum_components[1],
                "momentum_z_raw": momentum_components[2],
                "no_slip_raw": no_slip,
                "continuity_weighted": continuity_weighted,
                "momentum_x_weighted": momentum_weighted[0],
                "momentum_y_weighted": momentum_weighted[1],
                "momentum_z_weighted": momentum_weighted[2],
                "no_slip_weighted": no_slip_weighted,
                "physics_total": physics_total,
            }
        )
        diagnostics.update(
            {
                **residuals,
                "physics_prediction": physics_prediction,
                "physics_coords": physics_coords,
                "wall_prediction": wall_prediction,
                "wall_speed_m_s": wall_speed,
            }
        )
    losses["total"] = losses["data_total"] + losses["physics_total"]
    return losses, diagnostics
