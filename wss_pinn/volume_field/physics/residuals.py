"""Autograd residuals for steady incompressible non-Newtonian flow."""

from __future__ import annotations

from typing import Any

import torch

from .rheology import carreau_yasuda


def _gradient(value: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """Pointwise scalar gradient, returning exact zeros for constants."""
    if not value.requires_grad:
        return torch.zeros_like(coords)
    gradient = torch.autograd.grad(
        value.sum(),
        coords,
        create_graph=True,
        retain_graph=True,
        allow_unused=True,
    )[0]
    return torch.zeros_like(coords) if gradient is None else gradient


def velocity_jacobian(velocity: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    if velocity.ndim != 2 or velocity.shape[1] != 3:
        raise ValueError("velocity must have shape [N,3]")
    return torch.stack(
        [_gradient(velocity[:, component], coords) for component in range(3)],
        dim=1,
    )


def _field_stat_tensors(
    field_stats: dict[str, Any],
    reference: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    velocity_mean = reference.new_tensor(field_stats["velocity_m_s"]["mean"])
    velocity_std = reference.new_tensor(field_stats["velocity_m_s"]["std"])
    pressure_mean = reference.new_tensor(field_stats["pressure_relative_pa"]["mean"][0])
    pressure_std = reference.new_tensor(field_stats["pressure_relative_pa"]["std"][0])
    return velocity_mean, velocity_std, pressure_mean, pressure_std


def denormalize_fields(
    prediction: torch.Tensor,
    field_stats: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert normalized network output to registered SI velocity and relative Pa."""
    if prediction.ndim != 2 or prediction.shape[1] != 4:
        raise ValueError("prediction must have shape [N,4]")
    velocity_mean, velocity_std, pressure_mean, pressure_std = _field_stat_tensors(
        field_stats, prediction
    )
    velocity = prediction[:, :3] * velocity_std + velocity_mean
    pressure = prediction[:, 3] * pressure_std + pressure_mean
    return velocity, pressure


def generalized_newtonian_residuals(
    prediction: torch.Tensor,
    coords: torch.Tensor,
    length_m: torch.Tensor,
    *,
    field_stats: dict[str, Any],
    physics_config: dict[str, Any],
) -> dict[str, torch.Tensor]:
    """Return dimensionless residuals and SI diagnostics.

    ``coords`` are registered dimensionless coordinates ``x_hat`` and must be
    independent query leaves.  The first version is quasi-steady and therefore
    intentionally omits ``du/dt``.
    """
    if not coords.requires_grad:
        raise ValueError("physics query coordinates must require gradients")
    length = length_m.reshape(-1).to(device=coords.device, dtype=coords.dtype)
    if len(length) != len(coords) or bool(torch.any(length <= 0)):
        raise ValueError("length_m must be positive and match physics queries")
    density = float(physics_config["density_kg_m3"])
    velocity_reference = float(physics_config["reference_velocity_m_s"])
    pressure_reference = density * velocity_reference**2
    velocity_phys, pressure_phys = denormalize_fields(prediction, field_stats)
    velocity_hat = velocity_phys / velocity_reference
    pressure_hat = pressure_phys / pressure_reference

    gradient_hat = velocity_jacobian(velocity_hat, coords)
    continuity_hat = torch.diagonal(gradient_hat, dim1=1, dim2=2).sum(dim=1)
    strain_hat = 0.5 * (gradient_hat + gradient_hat.transpose(1, 2))
    strain_phys = strain_hat * (velocity_reference / length)[:, None, None]
    shear_eps = float(physics_config["shear_rate_epsilon_s_inv"])
    shear_rate = torch.sqrt(
        2.0 * torch.sum(strain_phys * strain_phys, dim=(1, 2)) + shear_eps**2
    )
    viscosity = carreau_yasuda(shear_rate, physics_config["rheology"])
    stress_phys = 2.0 * viscosity[:, None, None] * strain_phys

    convective_hat = torch.einsum("nj,nij->ni", velocity_hat, gradient_hat)
    pressure_gradient_hat = _gradient(pressure_hat, coords)
    stress_divergence_phys_components = []
    for component in range(3):
        divergence = torch.zeros_like(continuity_hat)
        for direction in range(3):
            derivative_hat = _gradient(stress_phys[:, component, direction], coords)
            divergence = divergence + derivative_hat[:, direction] / length
        stress_divergence_phys_components.append(divergence)
    stress_divergence_phys = torch.stack(
        stress_divergence_phys_components, dim=1
    )
    inertia_scale = density * velocity_reference**2 / length
    viscous_hat = stress_divergence_phys / inertia_scale[:, None]
    momentum_hat = convective_hat + pressure_gradient_hat - viscous_hat

    return {
        "velocity_phys_m_s": velocity_phys,
        "pressure_relative_pa": pressure_phys,
        "velocity_hat": velocity_hat,
        "pressure_hat": pressure_hat,
        "velocity_gradient_hat": gradient_hat,
        "strain_rate_phys_s_inv": strain_phys,
        "shear_rate_s_inv": shear_rate,
        "viscosity_pa_s": viscosity,
        "stress_phys_pa": stress_phys,
        "continuity_hat": continuity_hat,
        "continuity_phys_s_inv": continuity_hat * velocity_reference / length,
        "momentum_convective_hat": convective_hat,
        "momentum_pressure_hat": pressure_gradient_hat,
        "momentum_viscous_hat": viscous_hat,
        "momentum_hat": momentum_hat,
        "momentum_phys_pa_per_m": momentum_hat * inertia_scale[:, None],
    }


def no_slip_loss(
    prediction: torch.Tensor,
    *,
    field_stats: dict[str, Any],
    reference_velocity_m_s: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return dimensionless no-slip MSE and physical wall speed."""
    velocity, _ = denormalize_fields(prediction, field_stats)
    velocity_hat = velocity / float(reference_velocity_m_s)
    loss = torch.mean(torch.sum(velocity_hat * velocity_hat, dim=1))
    speed = torch.linalg.vector_norm(velocity, dim=1)
    return loss, speed
