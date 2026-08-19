"""V4 data, BC and strong-form PDE losses with per-case nondimensionalization."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from wss_pinn.physics.rheology import carreau_yasuda


def rcr_pressure_form_residual(
    pressure_pa: torch.Tensor,
    mass_flow_kg_s: torch.Tensor,
    dpressure_dt_pa_s: torch.Tensor,
    dmass_dt_kg_s2: torch.Tensor,
    R1: torch.Tensor,
    R2: torch.Tensor,
    C: torch.Tensor,
) -> torch.Tensor:
    """Pressure-form RCR ODE, algebraically equivalent to the UDF flow form."""
    return (
        R2 * C * dpressure_dt_pa_s
        + pressure_pa
        - (R1 + R2) * mass_flow_kg_s
        - R1 * R2 * C * dmass_dt_kg_s2
    )


def _gradient(value: torch.Tensor, variable: torch.Tensor) -> torch.Tensor:
    if not value.requires_grad:
        return torch.zeros_like(variable)
    result = torch.autograd.grad(
        value.sum(),
        variable,
        create_graph=True,
        retain_graph=True,
        allow_unused=True,
    )[0]
    return torch.zeros_like(variable) if result is None else result


def _stats_tensors(
    stats: dict[str, Any], reference: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        reference.new_tensor(stats["velocity_mean"]),
        reference.new_tensor(stats["velocity_std"]),
        reference.new_tensor(stats["pressure_mean"]).reshape(()),
        reference.new_tensor(stats["pressure_std"]).reshape(()),
    )


def denormalize(
    prediction: torch.Tensor, stats: dict[str, Any]
) -> tuple[torch.Tensor, torch.Tensor]:
    velocity_mean, velocity_std, pressure_mean, pressure_std = _stats_tensors(
        stats, prediction
    )
    velocity = prediction[:, :3] * velocity_std + velocity_mean
    pressure = prediction[:, 3] * pressure_std + pressure_mean
    return velocity, pressure


def _velocity_jacobian(
    velocity: torch.Tensor, coords: torch.Tensor, length_m: torch.Tensor
) -> torch.Tensor:
    gradient_hat = torch.stack(
        [_gradient(velocity[:, component], coords) for component in range(3)], dim=1
    )
    return gradient_hat / length_m[:, None, None]


def strong_form_residuals(
    prediction: torch.Tensor,
    coords: torch.Tensor,
    time_s: torch.Tensor | None,
    point_batch: torch.Tensor,
    batch: dict[str, Any],
    *,
    field_stats: dict[str, Any],
    physics_config: dict[str, Any],
) -> dict[str, torch.Tensor]:
    if not coords.requires_grad:
        raise ValueError("physics coordinates must require gradients")
    length = batch["length_m"][point_batch]
    U_c = batch["U_c_m_s"][point_batch]
    L_c = batch["L_c_m"][point_batch]
    density = float(physics_config["density_kg_m3"])
    velocity, pressure = denormalize(prediction, field_stats)
    jacobian = _velocity_jacobian(velocity, coords, length)
    continuity_phys = torch.diagonal(jacobian, dim1=1, dim2=2).sum(dim=1)
    continuity = continuity_phys / (U_c / L_c)

    strain = 0.5 * (jacobian + jacobian.transpose(1, 2))
    shear_epsilon = float(physics_config["shear_rate_epsilon_s_inv"])
    shear_rate = torch.sqrt(
        2.0 * torch.sum(strain * strain, dim=(1, 2)) + shear_epsilon**2
    )
    viscosity = carreau_yasuda(shear_rate, physics_config["rheology"])
    stress = 2.0 * viscosity[:, None, None] * strain
    stress_divergence_components = []
    for component in range(3):
        value = torch.zeros_like(continuity_phys)
        for direction in range(3):
            derivative_hat = _gradient(stress[:, component, direction], coords)
            value = value + derivative_hat[:, direction] / length
        stress_divergence_components.append(value)
    stress_divergence = torch.stack(stress_divergence_components, dim=1)
    convective = density * torch.einsum("nj,nij->ni", velocity, jacobian)
    pressure_gradient = _gradient(pressure, coords) / length[:, None]
    if time_s is None:
        transient = torch.zeros_like(convective)
        du_dt = torch.zeros_like(velocity)
    else:
        if not time_s.requires_grad:
            raise ValueError("transient physics time must require gradients")
        du_dt = torch.stack(
            [_gradient(velocity[:, component], time_s)[:, 0] for component in range(3)],
            dim=1,
        )
        transient = density * du_dt
    momentum_phys = transient + convective + pressure_gradient - stress_divergence
    momentum_scale = density * U_c.square() / L_c
    momentum = momentum_phys / momentum_scale[:, None]
    return {
        "velocity_m_s": velocity,
        "pressure_pa": pressure,
        "continuity": continuity,
        "continuity_phys_s_inv": continuity_phys,
        "momentum": momentum,
        "momentum_phys_pa_per_m": momentum_phys,
        "du_dt_m_s2": du_dt,
        "momentum_transient_pa_per_m": transient,
        "momentum_convective_pa_per_m": convective,
        "momentum_pressure_pa_per_m": pressure_gradient,
        "momentum_viscous_pa_per_m": stress_divergence,
        "shear_rate_s_inv": shear_rate,
        "viscosity_pa_s": viscosity,
    }


def _data_losses(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    names = ("u", "v", "w", "p")
    raw = [F.mse_loss(prediction[:, index], target[:, index]) for index in range(4)]
    return {
        **{f"data_{name}_raw": value for name, value in zip(names, raw)},
        **{f"data_{name}_weighted": value / 4.0 for name, value in zip(names, raw)},
        "data_total": sum(raw) / 4.0,
    }


def _boundary_losses(
    model: Any,
    encoded: dict[str, torch.Tensor],
    batch: dict[str, Any],
    config: Any,
    field_stats: dict[str, Any],
) -> dict[str, torch.Tensor]:
    zero = encoded["geometry_latent"].new_zeros(())
    if not bool(config["physics"]["bc_enabled"]):
        return {
            "no_slip_raw": zero,
            "inlet_bc_raw": zero,
            "rcr_bc_raw": zero,
            **{f"rcr_bc_outlet_{index}_raw": zero for index in range(4)},
            "bc_total": zero,
        }
    transient = str(config["experiment"]["temporal_mode"]) == "transient_81"
    wall_prediction = model.decode_query(
        encoded,
        batch["wall_coords"],
        batch["wall_batch"],
        batch["wall_time_s"] if transient else None,
    )
    wall_velocity, _ = denormalize(wall_prediction, field_stats)
    wall_scale = batch["U_c_m_s"][batch["wall_batch"]]
    no_slip = torch.mean(
        torch.sum(torch.square(wall_velocity / wall_scale[:, None]), dim=1)
    )

    inlet_prediction = model.decode_query(
        encoded,
        batch["inlet_coords"],
        batch["inlet_batch"],
        batch["inlet_time_s"] if transient else None,
    )
    inlet_velocity, _ = denormalize(inlet_prediction, field_stats)
    inlet_scale = batch["U_c_m_s"][batch["inlet_batch"]]
    inlet = torch.mean(
        torch.sum(
            torch.square(
                (inlet_velocity - batch["inlet_target_velocity_m_s"])
                / inlet_scale[:, None]
            ),
            dim=1,
        )
    )

    rcr_by_outlet = [zero for _ in range(4)]
    if transient:
        outlet_time = batch["outlet_time_s"].detach().clone().requires_grad_(True)
        outlet_prediction = model.decode_query(
            encoded,
            batch["outlet_coords"],
            batch["outlet_batch"],
            outlet_time,
        )
        outlet_velocity, outlet_pressure = denormalize(outlet_prediction, field_stats)
        density = float(config["physics"]["density_kg_m3"])
        for outlet_index in range(4):
            residuals = []
            for case_index in range(len(batch["case_ids"])):
                mask = (batch["outlet_batch"] == case_index) & (
                    batch["outlet_zone"] == outlet_index
                )
                area = batch["outlet_area_weights_m2"][mask]
                normal = batch["outlet_normals"][mask]
                area_sum = area.sum().clamp_min(1.0e-30)
                pressure_mean = torch.sum(area * outlet_pressure[mask]) / area_sum
                mass_flow = density * torch.sum(
                    area
                    * torch.sum(outlet_velocity[mask] * normal, dim=1)
                )
                dpressure_dt = _gradient(pressure_mean, outlet_time)[mask].sum()
                dmass_dt = _gradient(mass_flow, outlet_time)[mask].sum()
                R1 = batch["R1"][case_index, outlet_index]
                R2 = batch["R2"][case_index, outlet_index]
                C = batch["C"][case_index, outlet_index]
                # Pressure-form equivalent of the mass-flow RCR ODE:
                # R2*C*dP/dt + P - (R1+R2)m - R1*R2*C*dm/dt = 0.
                value = rcr_pressure_form_residual(
                    pressure_mean,
                    mass_flow,
                    dpressure_dt,
                    dmass_dt,
                    R1,
                    R2,
                    C,
                ) / batch["rcr_pressure_scale_pa"][case_index, outlet_index]
                residuals.append(value)
            rcr_by_outlet[outlet_index] = torch.mean(torch.square(torch.stack(residuals)))
        rcr = sum(rcr_by_outlet) / 4.0
        bc_total = (no_slip + inlet + rcr) / 3.0
    else:
        rcr = zero
        bc_total = (no_slip + inlet) / 2.0
    return {
        "no_slip_raw": no_slip,
        "inlet_bc_raw": inlet,
        "rcr_bc_raw": rcr,
        **{
            f"rcr_bc_outlet_{index}_raw": value
            for index, value in enumerate(rcr_by_outlet)
        },
        "bc_total": bc_total,
    }


def _pde_losses(
    model: Any,
    encoded: dict[str, torch.Tensor],
    batch: dict[str, Any],
    config: Any,
    field_stats: dict[str, Any],
) -> dict[str, torch.Tensor]:
    zero = encoded["geometry_latent"].new_zeros(())
    if not bool(config["physics"]["pde_enabled"]):
        return {
            "continuity_raw": zero,
            "momentum_x_raw": zero,
            "momentum_y_raw": zero,
            "momentum_z_raw": zero,
            "momentum_group_raw": zero,
            "pde_total": zero,
            "du_dt_rms": zero,
            "dv_dt_rms": zero,
            "dw_dt_rms": zero,
            "transient_term_rms": zero,
            "convective_term_rms": zero,
            "pressure_term_rms": zero,
            "viscous_term_rms": zero,
        }
    transient = str(config["experiment"]["temporal_mode"]) == "transient_81"
    coords = batch["physics_coords"].detach().clone().requires_grad_(True)
    time_s = None
    if transient:
        time_s = batch["physics_time_s"].detach().clone().requires_grad_(True)
    prediction = model.decode_query(
        encoded, coords, batch["physics_batch"], time_s
    )
    residual = strong_form_residuals(
        prediction,
        coords,
        time_s,
        batch["physics_batch"],
        batch,
        field_stats=field_stats,
        physics_config=config["physics"],
    )
    continuity = torch.mean(torch.square(residual["continuity"]))
    momentum = [
        torch.mean(torch.square(residual["momentum"][:, index]))
        for index in range(3)
    ]
    momentum_group = sum(momentum) / 3.0
    pde_total = (continuity + momentum_group) / 2.0

    def rms(name: str, component: int | None = None) -> torch.Tensor:
        value = residual[name]
        if component is not None:
            value = value[:, component]
        return torch.sqrt(torch.mean(torch.square(value)) + 1.0e-30)

    return {
        "continuity_raw": continuity,
        "momentum_x_raw": momentum[0],
        "momentum_y_raw": momentum[1],
        "momentum_z_raw": momentum[2],
        "momentum_group_raw": momentum_group,
        "pde_total": pde_total,
        "du_dt_rms": rms("du_dt_m_s2", 0),
        "dv_dt_rms": rms("du_dt_m_s2", 1),
        "dw_dt_rms": rms("du_dt_m_s2", 2),
        "transient_term_rms": rms("momentum_transient_pa_per_m"),
        "convective_term_rms": rms("momentum_convective_pa_per_m"),
        "pressure_term_rms": rms("momentum_pressure_pa_per_m"),
        "viscous_term_rms": rms("momentum_viscous_pa_per_m"),
    }


def compute_raw_losses(
    model: Any,
    batch: dict[str, Any],
    config: Any,
    *,
    field_stats: dict[str, Any],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    encoded = model.encode_support(
        batch["support_coords"],
        batch["support_features"],
        batch["support_batch"],
        batch["bc_vector"],
        batch["case_ids"],
        int(config["train"]["seed"]),
    )
    transient = str(config["experiment"]["temporal_mode"]) == "transient_81"
    prediction = model.decode_query(
        encoded,
        batch["query_coords"],
        batch["query_batch"],
        batch["query_time_s"] if transient else None,
    )
    losses = _data_losses(prediction, batch["query_target"])
    losses.update(_boundary_losses(model, encoded, batch, config, field_stats))
    losses.update(_pde_losses(model, encoded, batch, config, field_stats))
    return losses, encoded


def _flat_grad(
    loss: torch.Tensor, parameters: list[torch.nn.Parameter]
) -> torch.Tensor:
    if not loss.requires_grad:
        return torch.zeros(sum(parameter.numel() for parameter in parameters), device=parameters[0].device)
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )
    return torch.cat(
        [
            (torch.zeros_like(parameter) if gradient is None else gradient).reshape(-1)
            for parameter, gradient in zip(parameters, gradients)
        ]
    )


def gradient_diagnostics(
    model: Any, losses: dict[str, torch.Tensor]
) -> dict[str, float]:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    data_gradient = _flat_grad(losses["data_total"], parameters)
    pde_gradient = _flat_grad(losses["pde_total"], parameters)
    data_norm = torch.linalg.vector_norm(data_gradient)
    pde_norm = torch.linalg.vector_norm(pde_gradient)

    def cosine(name: str) -> float:
        other = _flat_grad(losses[name], parameters)
        denominator = data_norm * torch.linalg.vector_norm(other)
        if float(denominator.detach().cpu()) <= 1.0e-30:
            return 0.0
        return float(torch.dot(data_gradient, other).div(denominator).detach().cpu())

    return {
        "grad_norm_data": float(data_norm.detach().cpu()),
        "grad_norm_physics": float(pde_norm.detach().cpu()),
        "cos_grad_data_continuity": cosine("continuity_raw"),
        "cos_grad_data_momentum": cosine("momentum_group_raw"),
        "cos_grad_data_no_slip": cosine("no_slip_raw"),
        "cos_grad_data_inlet_bc": cosine("inlet_bc_raw"),
        "cos_grad_data_rcr_bc": cosine("rcr_bc_raw"),
    }
