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
from wss_pinn.utils import sha256_file
from wss_pinn.v4.data import (
    CENTERLINE_V2_STAGING_ROUTE,
    V4Dataset,
    _geometry_transform,
)
from wss_pinn.v4.models import build_model
from wss_pinn.v4.physics import rcr_pressure_form_residual, strong_form_residuals
from wss_pinn.v4.evaluate import denormalize_prediction, vector_relative_l2
from wss_pinn.v4.geometry_v2 import (
    GEOMETRY_MODEL_NAMES,
    GEOMETRY_RAW_NAMES,
    registration_from_openings,
)
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


class TestCenterlineV2Geometry(unittest.TestCase):
    def test_opening_ids_define_right_handed_anatomical_frame(self):
        junction = np.zeros(3, dtype=np.float64)
        openings = {
            "inlet": np.asarray([0.0, 0.0, 10.0]),
            "out-le": np.asarray([4.0, 1.0, -8.0]),
            "out-li": np.asarray([3.0, -1.0, -7.0]),
            "out-ri": np.asarray([-3.0, 1.0, -7.0]),
            "out-re": np.asarray([-4.0, -1.0, -8.0]),
        }
        centroid, rotation, diagnostics = registration_from_openings(junction, openings)
        np.testing.assert_allclose(centroid, junction)
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0, places=12)
        self.assertGreater(diagnostics["inlet_z_mm"], 0.0)
        self.assertLess(diagnostics["outlet_z_max_mm"], 0.0)
        self.assertGreater(diagnostics["left_minus_right_x_mm"], 0.0)

    def test_v2_geometry_has_one_shared_transform_without_curvature_clip(self):
        raw = np.asarray([[0.25, 5.0, -20.0], [0.75, 7.0, 20.0]], dtype=np.float32)
        transformed = raw.copy()
        transformed[:, 2] = np.sign(transformed[:, 2]) * np.log1p(
            np.abs(transformed[:, 2])
        )
        stats = {
            "raw_names": list(GEOMETRY_RAW_NAMES),
            "names": list(GEOMETRY_MODEL_NAMES),
            "mean": transformed.mean(axis=0).tolist(),
            "std": np.where(transformed.std(axis=0) < 1e-6, 1.0, transformed.std(axis=0)).tolist(),
        }
        output = _geometry_transform(raw, stats)
        np.testing.assert_allclose(output[:, 2], [-1.0, 1.0], atol=1e-6)
        self.assertNotIn("curvature_clip_abs", stats)

    def test_training_loader_rejects_not_ready_staging(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            split_path = root / "split.json"
            stats_path = root / "stats.json"
            audit_path = root / "array_audit.json"
            manifest_path = root / "manifest.json"
            split_path.write_text(
                json.dumps({"train_cases": ["AG/fast/SYNTH"], "test_cases": []}),
                encoding="utf-8",
            )
            stats_path.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "route": CENTERLINE_V2_STAGING_ROUTE,
                        "train_case_ids": ["AG/fast/SYNTH"],
                        "geometry": {
                            "raw_names": list(GEOMETRY_RAW_NAMES),
                            "names": list(GEOMETRY_MODEL_NAMES),
                        },
                    }
                ),
                encoding="utf-8",
            )
            audit_path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "errors": [],
                        "case_count": 1,
                        "array_count": 0,
                        "case_manifest_sha256": {"AG/fast/SYNTH": "synthetic-case-sha"},
                    }
                ),
                encoding="utf-8",
            )
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "route": CENTERLINE_V2_STAGING_ROUTE,
                        "gate_result": "staging_pass",
                        "training_ready": False,
                        "split": {"path": str(split_path), "sha256": sha256_file(split_path)},
                        "stats": {"path": str(stats_path), "sha256": sha256_file(stats_path)},
                        "array_audit": {
                            "path": str(audit_path),
                            "sha256": sha256_file(audit_path),
                            "array_count": 0,
                        },
                        "cases": [
                            {
                                "canonical_id": "AG/fast/SYNTH",
                                "role": "train",
                                "manifest": str(root / "case.json"),
                                "manifest_sha256": "synthetic-case-sha",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            config = deepcopy(DEFAULTS)
            config["paths"]["manifest"] = str(manifest_path)
            config["paths"]["stats"] = str(stats_path)
            config["paths"]["split"] = str(split_path)
            for roles in (("train",), ["train"], ("train", "audit")):
                with self.subTest(roles=roles):
                    with self.assertRaisesRegex(ValueError, "not training-ready"):
                        V4Dataset(config, roles=roles)


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


class TestEvaluateMetrics(unittest.TestCase):
    def test_vector_relative_l2_is_the_registered_primary(self):
        truth = np.asarray([[3.0, 0.0, 4.0], [0.0, 0.0, 0.0]], dtype=np.float64)
        pred = np.asarray([[3.0, 0.0, 4.0], [0.0, 3.0, 4.0]], dtype=np.float64)
        # ||err||_2 = 5, ||truth||_2 = 5
        self.assertAlmostEqual(vector_relative_l2(pred, truth), 1.0)

    def test_denormalize_prediction_inverts_field_stats(self):
        stats = {
            "velocity_mean": np.asarray([1.0, -2.0, 0.5], dtype=np.float32),
            "velocity_std": np.asarray([2.0, 4.0, 0.5], dtype=np.float32),
            "pressure_mean": np.asarray([100.0], dtype=np.float32),
            "pressure_std": np.asarray([10.0], dtype=np.float32),
        }
        pred = np.asarray([[0.5, 0.25, -2.0, 1.5]], dtype=np.float32)
        velocity, pressure = denormalize_prediction(pred, stats)
        np.testing.assert_allclose(velocity, [[2.0, -1.0, -0.5]], atol=1e-6)
        np.testing.assert_allclose(pressure, [115.0], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
