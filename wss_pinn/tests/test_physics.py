from __future__ import annotations

import unittest

import numpy as np
import torch

from wss_pinn.objectives import gauge_invariant_pressure_mse, no_slip_loss
from wss_pinn.physics.residuals import continuity_residual
from wss_pinn.physics.rheology import (
    UDF_PARAMETERS,
    carreau_yasuda_numpy,
    carreau_yasuda_torch,
)


class PhysicsTests(unittest.TestCase):
    def test_carreau_yasuda_limits_positive_and_units(self):
        values = carreau_yasuda_numpy(np.asarray([0.0, 1.0, 1e9]))
        self.assertAlmostEqual(values[0], UDF_PARAMETERS["mu_zero_pa_s"], places=12)
        self.assertAlmostEqual(values[-1], UDF_PARAMETERS["mu_inf_pa_s"], places=5)
        self.assertTrue(np.all(values > 0))
        self.assertTrue(np.all(np.diff(values) < 0))

    def test_carreau_yasuda_finite_gradient(self):
        shear = torch.tensor([0.1, 1.0, 10.0], requires_grad=True)
        viscosity = carreau_yasuda_torch(shear)
        viscosity.sum().backward()
        self.assertTrue(torch.isfinite(shear.grad).all())

    def test_divergence_positive_and_negative_controls(self):
        coords = torch.randn(64, 3, requires_grad=True)
        good_velocity = torch.stack(
            [coords[:, 0], coords[:, 1], -2.0 * coords[:, 2]], dim=1
        )
        bad_velocity = coords
        good = continuity_residual(good_velocity, coords)
        bad = continuity_residual(bad_velocity, coords)
        self.assertLess(float(good.abs().max()), 1e-6)
        self.assertTrue(torch.allclose(bad, torch.full_like(bad, 3.0)))

    def test_no_slip_controls(self):
        zero = torch.zeros(32, 3)
        nonzero = torch.ones(32, 3)
        self.assertEqual(float(no_slip_loss(zero)), 0.0)
        self.assertGreater(float(no_slip_loss(nonzero)), 0.0)

    def test_pressure_gauge_invariance(self):
        target = torch.randn(128)
        prediction = target + 42.0
        self.assertLess(float(gauge_invariant_pressure_mse(prediction, target)), 1e-10)


if __name__ == "__main__":
    unittest.main()

