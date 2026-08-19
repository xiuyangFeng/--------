from __future__ import annotations

import json
import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from wss_pinn.v4.config import DEFAULTS
from wss_pinn.v4.build import _case_complete
from wss_pinn.v4.controller import PDEWeightController
from wss_pinn.v4.models import build_model
from wss_pinn.v4.physics import rcr_pressure_form_residual, strong_form_residuals
from wss_pinn.v4.waveform import q_nom_numpy, q_nom_torch


class TestWaveform(unittest.TestCase):
    def test_documented_reset_jump_is_preserved(self):
        left = float(q_nom_numpy(0.0))
        right = float(q_nom_numpy(0.8 - 1.0e-9))
        self.assertAlmostEqual(left, 1.48424e-5, delta=2.0e-10)
        self.assertAlmostEqual(right, 1.43140e-5, delta=2.0e-9)
        self.assertGreater(abs(left - right), 1.0e-8)

    def test_waveform_remains_in_autograd_graph(self):
        time = torch.tensor([[0.21]], dtype=torch.float64, requires_grad=True)
        value = q_nom_torch(time)
        derivative = torch.autograd.grad(value.sum(), time)[0]
        self.assertTrue(torch.isfinite(derivative).all())
        self.assertGreater(abs(float(derivative)), 1.0e-8)


class TestStage0Resume(unittest.TestCase):
    def test_old_manifest_without_target_domain_mask_is_rebuilt(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            boundary = root / "boundary.npz"
            boundary.touch()
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "gate_result": "pass",
                        "transient": {
                            "max_bc_monitor_vs_udf_relative_difference": 0.0,
                        },
                        "files": {
                            "boundary_faces": {"path": str(boundary)},
                            "transient": {},
                        },
                    }
                ),
                encoding="utf-8",
            )
            self.assertFalse(_case_complete(manifest, build_transient=True))


class TestController(unittest.TestCase):
    def test_ratio_is_detached_and_pde_gradient_survives(self):
        config = deepcopy(DEFAULTS["loss_balancing"])
        config["mode"] = "ema_loss_ratio"
        config["update_interval_steps"] = 1
        controller = PDEWeightController(config)
        value = torch.tensor(0.25, requires_grad=True)
        data = (value - 1.0).square()
        pde = (value + 1.0).square()
        record = controller.update(data, pde, global_step=1, epoch=100)
        total = data + record.lambda_pde * pde
        gradient = torch.autograd.grad(total, value)[0]
        expected = 2.0 * (value - 1.0) + record.lambda_pde * 2.0 * (value + 1.0)
        self.assertAlmostEqual(float(gradient), float(expected), places=6)
        self.assertIsInstance(record.lambda_pde, float)


class TestModel(unittest.TestCase):
    def _config(self, backbone: str, transient: bool):
        config = deepcopy(DEFAULTS)
        config["experiment"]["backbone"] = backbone
        config["experiment"]["temporal_mode"] = (
            "transient_81" if transient else "steady_peak"
        )
        return config

    def test_backbones_are_capacity_matched(self):
        counts = []
        for backbone in ("pointnet", "pointnetpp"):
            model = build_model(self._config(backbone, False))
            counts.append(sum(parameter.numel() for parameter in model.parameters()))
        self.assertLessEqual(max(counts) / min(counts), 1.10)

    def test_transient_decoder_has_xyz_and_time_derivatives(self):
        torch.manual_seed(7)
        config = self._config("pointnet", True)
        model = build_model(config)
        support_coords = torch.randn(12, 3)
        support_features = torch.randn(12, 6)
        support_batch = torch.zeros(12, dtype=torch.long)
        encoded = model.encode_support(
            support_coords,
            support_features,
            support_batch,
            torch.randn(1, 18),
            ["synthetic/case"],
            1234,
        )
        coords = torch.randn(5, 3, requires_grad=True)
        time = torch.full((5, 1), 0.21, requires_grad=True)
        output = model.decode_query(encoded, coords, torch.zeros(5, dtype=torch.long), time)
        xyz_grad = torch.autograd.grad(output[:, 0].sum(), coords, create_graph=True)[0]
        time_grad = torch.autograd.grad(output[:, 0].sum(), time)[0]
        self.assertTrue(torch.isfinite(xyz_grad).all())
        self.assertTrue(torch.isfinite(time_grad).all())


class TestPhysics(unittest.TestCase):
    def test_constant_rcr_solution(self):
        R1 = torch.tensor(2.0e5)
        R2 = torch.tensor(8.0e5)
        C = torch.tensor(1.0e-6)
        mass = torch.tensor(0.05)
        pressure = (R1 + R2) * mass
        residual = rcr_pressure_form_residual(
            pressure,
            mass,
            torch.tensor(0.0),
            torch.tensor(0.0),
            R1,
            R2,
            C,
        )
        self.assertAlmostEqual(float(residual), 0.0, places=5)

    def test_transient_time_chain_rule_is_physical_seconds(self):
        coords = torch.tensor(
            [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]], requires_grad=True
        )
        time = torch.tensor([[0.2], [0.3]], requires_grad=True)
        velocity = torch.cat([coords[:, :1] * time, coords[:, 1:2], coords[:, 2:3]], dim=1)
        pressure = coords[:, 0]
        prediction = torch.cat([velocity, pressure[:, None]], dim=1)
        batch = {
            "length_m": torch.tensor([0.2]),
            "U_c_m_s": torch.tensor([0.5]),
            "L_c_m": torch.tensor([0.02]),
        }
        residual = strong_form_residuals(
            prediction,
            coords,
            time,
            torch.zeros(2, dtype=torch.long),
            batch,
            field_stats={
                "velocity_mean": np.zeros(3),
                "velocity_std": np.ones(3),
                "pressure_mean": np.zeros(1),
                "pressure_std": np.ones(1),
            },
            physics_config=DEFAULTS["physics"],
        )
        self.assertTrue(torch.allclose(residual["du_dt_m_s2"][:, 0], coords[:, 0]))
        self.assertTrue(torch.isfinite(residual["momentum"]).all())


if __name__ == "__main__":
    unittest.main()
