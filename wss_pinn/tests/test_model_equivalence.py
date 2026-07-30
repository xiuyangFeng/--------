from __future__ import annotations

import unittest

import torch

from wss_pinn.config import DEFAULTS
from wss_pinn.models import WSSPINNModel
from wss_pinn.objectives import stage_losses


class ModelEquivalenceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1234)
        self.model = WSSPINNModel(DEFAULTS["model"])
        self.descriptor = torch.randn(15)
        self.wall_coords = torch.randn(24, 3)
        self.wall_wss = torch.rand(24) * 4.0
        self.interior_coords = torch.randn(32, 3)
        self.velocity = torch.randn(32, 3)
        self.pressure = torch.randn(32)

    def weights(self):
        return {
            "direct_wss_weight": 1.0,
            "velocity_data_weight": 1.0,
            "pressure_data_weight": 0.25,
            "continuity_weight": 0.0,
            "no_slip_weight": 0.0,
        }

    def test_f1_zero_physics_equals_f0up_single_batch(self):
        control = stage_losses(
            self.model,
            self.descriptor,
            self.wall_coords,
            self.wall_wss,
            self.interior_coords.clone(),
            self.velocity,
            self.pressure,
            self.weights(),
        )
        f1_zero = stage_losses(
            self.model,
            self.descriptor,
            self.wall_coords,
            self.wall_wss,
            self.interior_coords.clone(),
            self.velocity,
            self.pressure,
            {**self.weights(), "continuity_weight": 0.0, "no_slip_weight": 0.0},
        )
        self.assertTrue(torch.equal(control["total"], f1_zero["total"]))

    def test_cpu_forward_backward_is_finite(self):
        losses = stage_losses(
            self.model,
            self.descriptor,
            self.wall_coords,
            self.wall_wss,
            self.interior_coords.clone(),
            self.velocity,
            self.pressure,
            self.weights(),
        )
        losses["total"].backward()
        self.assertTrue(torch.isfinite(losses["total"]))
        self.assertTrue(
            all(
                parameter.grad is None or torch.isfinite(parameter.grad).all()
                for parameter in self.model.parameters()
            )
        )


if __name__ == "__main__":
    unittest.main()

