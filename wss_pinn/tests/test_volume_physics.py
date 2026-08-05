from __future__ import annotations

import unittest

import torch

from wss_pinn.physics.residuals import generalized_newtonian_residuals
from wss_pinn.physics.rheology import carreau_yasuda


FIELD_STATS = {
    "velocity_m_s": {"mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]},
    "pressure_relative_pa": {"mean": [0.0], "std": [1060.0]},
}
PHYSICS = {
    "density_kg_m3": 1060.0,
    "reference_velocity_m_s": 1.0,
    "shear_rate_epsilon_s_inv": 1e-6,
    "rheology": {
        "mu_inf_pa_s": 0.0035,
        "mu_zero_pa_s": 0.16,
        "lambda_s": 8.2,
        "a": 0.64,
        "n": 0.2128,
    },
}


class VolumePhysicsTests(unittest.TestCase):
    def test_carreau_yasuda_limits_and_monotonicity(self):
        shear = torch.tensor([0.0, 0.1, 1.0, 1000.0], dtype=torch.float64, requires_grad=True)
        viscosity = carreau_yasuda(shear, PHYSICS["rheology"])
        self.assertAlmostEqual(float(viscosity[0]), 0.16, places=12)
        self.assertTrue(bool(torch.all(viscosity[1:] < viscosity[:-1])))
        viscosity.sum().backward()
        self.assertTrue(torch.isfinite(shear.grad).all())

    def test_couette_is_continuity_and_momentum_free(self):
        coords = torch.rand(32, 3, dtype=torch.float64, requires_grad=True)
        velocity = torch.stack(
            [2.0 * coords[:, 1], torch.zeros(32, dtype=torch.float64), torch.zeros(32, dtype=torch.float64)],
            dim=1,
        )
        prediction = torch.cat([velocity, torch.zeros(32, 1, dtype=torch.float64)], dim=1)
        residual = generalized_newtonian_residuals(
            prediction,
            coords,
            torch.ones(32, dtype=torch.float64),
            field_stats=FIELD_STATS,
            physics_config=PHYSICS,
        )
        self.assertLess(float(residual["continuity_hat"].abs().max()), 1e-10)
        self.assertLess(float(residual["momentum_hat"].abs().max()), 1e-9)
        self.assertAlmostEqual(
            float(residual["shear_rate_s_inv"].mean()), 2.0, places=6
        )

    def test_solid_rotation_pressure_sign_and_gauge_invariance(self):
        coords = (torch.rand(48, 3, dtype=torch.float64) - 0.5).requires_grad_(True)
        omega = 0.7
        velocity = torch.stack(
            [-omega * coords[:, 1], omega * coords[:, 0], torch.zeros(48, dtype=torch.float64)],
            dim=1,
        )
        pressure_hat = 0.5 * omega**2 * (
            torch.square(coords[:, 0]) + torch.square(coords[:, 1])
        )
        prediction = torch.cat([velocity, pressure_hat[:, None]], dim=1)
        residual = generalized_newtonian_residuals(
            prediction,
            coords,
            torch.ones(48, dtype=torch.float64),
            field_stats=FIELD_STATS,
            physics_config=PHYSICS,
        )
        shifted = prediction.clone()
        shifted[:, 3] = shifted[:, 3] + 123.0
        residual_shifted = generalized_newtonian_residuals(
            shifted,
            coords,
            torch.ones(48, dtype=torch.float64),
            field_stats=FIELD_STATS,
            physics_config=PHYSICS,
        )
        self.assertLess(float(residual["momentum_hat"].abs().max()), 1e-9)
        torch.testing.assert_close(
            residual["momentum_hat"], residual_shifted["momentum_hat"], rtol=1e-10, atol=1e-10
        )

    def test_variable_viscosity_stress_divergence_is_finite_and_nonzero(self):
        coords = (torch.rand(64, 3, dtype=torch.float64) + 0.2).requires_grad_(True)
        velocity = torch.stack(
            [torch.square(coords[:, 1]), torch.zeros(64, dtype=torch.float64), torch.zeros(64, dtype=torch.float64)],
            dim=1,
        )
        prediction = torch.cat([velocity, torch.zeros(64, 1, dtype=torch.float64)], dim=1)
        residual = generalized_newtonian_residuals(
            prediction,
            coords,
            torch.ones(64, dtype=torch.float64),
            field_stats=FIELD_STATS,
            physics_config=PHYSICS,
        )
        self.assertTrue(torch.isfinite(residual["momentum_viscous_hat"]).all())
        self.assertGreater(float(residual["momentum_viscous_hat"][:, 0].abs().mean()), 1e-6)


if __name__ == "__main__":
    unittest.main()
