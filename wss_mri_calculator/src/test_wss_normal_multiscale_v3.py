from __future__ import annotations

import unittest

import numpy as np
from scipy.spatial import cKDTree

from wss_normal_multiscale_v3 import (
    NormalMultiscaleV3Config,
    _depth_limit_mm,
    fit_wall_gradient_normal_multiscale,
)


class NormalTubeV3Test(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(23)
        eta_m = np.linspace(8e-5, 1.4e-3, 160)
        near_lateral_mm = rng.normal(scale=0.035, size=(len(eta_m), 2))
        near = np.column_stack(
            [near_lateral_mm[:, 0], near_lateral_mm[:, 1], eta_m * 1e3]
        )
        contaminant_lateral = rng.normal(loc=0.9, scale=0.04, size=len(eta_m))
        contaminant = np.column_stack(
            [contaminant_lateral, np.zeros(len(eta_m)), eta_m * 1e3]
        )
        self.interior_mm = np.vstack([near, contaminant])
        self.true_gradient = np.array([42.0, -12.0, 0.0])
        quadratic = np.array([4800.0, -1300.0, 0.0])
        near_velocity = (
            eta_m[:, None] * self.true_gradient
            + eta_m[:, None] ** 2 * quadratic
        )
        contaminant_velocity = eta_m[:, None] * np.array([5.0, 2.0, 0.0])
        self.velocity = np.vstack([near_velocity, contaminant_velocity])
        self.target_wall = np.zeros((1, 3), dtype=np.float64)
        self.target_normal = np.array([[0.0, 0.0, 1.0]])
        self.full_wall = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.20, 0.0, 0.0],
                [-0.20, 0.0, 0.0],
                [0.0, 0.20, 0.0],
                [0.0, -0.20, 0.0],
                [0.90, 0.0, 0.0],
                [1.10, 0.0, 0.0],
                [0.90, 0.20, 0.0],
            ],
            dtype=np.float64,
        )
        self.full_normals = np.tile(self.target_normal, (len(self.full_wall), 1))
        self.tree = cKDTree(self.interior_mm)

    def _config(self) -> NormalMultiscaleV3Config:
        return NormalMultiscaleV3Config.from_mapping(
            {
                "sampling": {
                    "query_neighbors": 256,
                    "depth_fractions": [0.45, 0.65, 0.82, 1.0],
                    "depth_bins": 8,
                    "points_per_bin": 2,
                    "min_samples": 8,
                    "tube_wall_spacing_multiplier": 0.6,
                    "tube_depth_slope": 0.1,
                    "tube_min_radius_mm": 0.08,
                    "tube_max_radius_fraction": 0.15,
                },
                "ownership": {
                    "enabled": True,
                    "max_anchor_spacing_multiplier": 2.0,
                    "max_anchor_separation_mm": 0.45,
                    "min_normal_cosine": 0.5,
                },
                "depth_prior": {
                    "min_mm": 0.4,
                    "max_mm": 1.4,
                    "max_radius_fraction": 0.7,
                    "fallback_radius_mm": 2.0,
                },
                "fit": {"model": "anisotropic_mls", "cv_tolerance": 1.25},
                "multiscale": {"enabled": False},
            }
        )

    def test_normal_tube_rejects_lateral_contamination(self) -> None:
        gradient, used, diagnostics = fit_wall_gradient_normal_multiscale(
            self.target_wall,
            self.target_normal,
            self.full_wall,
            self.full_normals,
            self.interior_mm,
            self.velocity,
            self.tree,
            local_radius_mm=np.array([2.0]),
            config=self._config(),
        )
        np.testing.assert_allclose(
            gradient[0], self.true_gradient, rtol=2e-3, atol=2e-3
        )
        self.assertGreaterEqual(int(used[0]), 8)
        self.assertGreater(int(diagnostics["ownership_rejected"][0]), 0)
        self.assertLess(float(diagnostics["lateral_p95_mm"][0]), 0.20)

    def test_reynolds_depth_decreases_with_edge_velocity(self) -> None:
        config = self._config()
        slow_depth, slow_re = _depth_limit_mm(5.0, 0.05, config)
        fast_depth, fast_re = _depth_limit_mm(5.0, 1.0, config)
        self.assertGreater(fast_re, slow_re)
        self.assertLess(fast_depth, slow_depth)

    def test_normal_ray_interpolation_recovers_smooth_profile(self) -> None:
        payload = {
            "sampling": {
                "query_neighbors": 256,
                "mode": "normal_ray_interpolation",
                "depth_fractions": [0.45, 0.65, 0.82, 1.0],
                "depth_bins": 8,
                "points_per_bin": 2,
                "min_samples": 8,
                "ray_stations": 12,
                "ray_interpolation_neighbors": 8,
                "ray_min_interpolation_neighbors": 4,
                "ray_start_fractions": [0.35],
                "ray_distance_floor_mm": 0.02,
            },
            "ownership": {
                "enabled": True,
                "max_anchor_spacing_multiplier": 2.0,
                "max_anchor_separation_mm": 0.45,
                "min_normal_cosine": 0.5,
            },
            "depth_prior": {
                "min_mm": 0.4,
                "max_mm": 1.4,
                "max_radius_fraction": 0.7,
                "fallback_radius_mm": 2.0,
            },
            "fit": {"model": "normal_profile", "cv_tolerance": 1.25},
            "multiscale": {"enabled": False},
        }
        gradient, used, diagnostics = fit_wall_gradient_normal_multiscale(
            self.target_wall,
            self.target_normal,
            self.full_wall,
            self.full_normals,
            self.interior_mm,
            self.velocity,
            self.tree,
            local_radius_mm=np.array([2.0]),
            config=NormalMultiscaleV3Config.from_mapping(payload),
        )
        np.testing.assert_allclose(
            gradient[0], self.true_gradient, rtol=3e-2, atol=3e-2
        )
        self.assertEqual(int(used[0]), 12)
        self.assertTrue(np.isfinite(diagnostics["lateral_p50_mm"][0]))

    def test_unknown_config_key_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            NormalMultiscaleV3Config.from_mapping(
                {"sampling": {"query_neighbours_typo": 128}}
            )

    def test_hybrid_uses_fallback_when_geometry_risk_is_below_gate(self) -> None:
        payload = {
            "sampling": {
                "query_neighbors": 256,
                "depth_fractions": [0.45, 0.65, 0.82, 1.0],
                "depth_bins": 8,
                "points_per_bin": 2,
                "min_samples": 8,
            },
            "ownership": {"enabled": True},
            "depth_prior": {
                "min_mm": 0.4,
                "max_mm": 1.4,
                "max_radius_fraction": 0.7,
                "fallback_radius_mm": 2.0,
            },
            "fit": {"model": "anisotropic_mls"},
            "multiscale": {"enabled": False},
            "hybrid": {"enabled": True, "lateral_over_eta_min": 100.0},
        }
        fallback = np.array([[7.0, 3.0, 0.0]])
        gradient, _, diagnostics = fit_wall_gradient_normal_multiscale(
            self.target_wall,
            self.target_normal,
            self.full_wall,
            self.full_normals,
            self.interior_mm,
            self.velocity,
            self.tree,
            local_radius_mm=np.array([2.0]),
            config=NormalMultiscaleV3Config.from_mapping(payload),
            fallback_gradient=fallback,
        )
        np.testing.assert_allclose(gradient, fallback)
        self.assertEqual(int(diagnostics["ray_used"][0]), 0)

    def test_hybrid_caps_positive_correction_and_preserves_direction(self) -> None:
        payload = {
            "sampling": {
                "query_neighbors": 256,
                "depth_fractions": [0.45, 0.65, 0.82, 1.0],
                "depth_bins": 8,
                "points_per_bin": 2,
                "min_samples": 8,
            },
            "ownership": {"enabled": True},
            "depth_prior": {
                "min_mm": 0.4,
                "max_mm": 1.4,
                "max_radius_fraction": 0.7,
                "fallback_radius_mm": 2.0,
            },
            "fit": {"model": "anisotropic_mls"},
            "multiscale": {"enabled": False},
            "hybrid": {
                "enabled": True,
                "lateral_over_eta_min": 0.01,
                "min_ray_to_fallback_norm": 1.0,
                "max_ray_to_fallback_norm": 1.2,
                "preserve_fallback_direction": True,
            },
        }
        fallback = np.array([[7.0, 3.0, 0.0]])
        gradient, _, diagnostics = fit_wall_gradient_normal_multiscale(
            self.target_wall,
            self.target_normal,
            self.full_wall,
            self.full_normals,
            self.interior_mm,
            self.velocity,
            self.tree,
            local_radius_mm=np.array([2.0]),
            config=NormalMultiscaleV3Config.from_mapping(payload),
            fallback_gradient=fallback,
        )
        np.testing.assert_allclose(gradient, 1.2 * fallback, rtol=1e-12)
        self.assertEqual(int(diagnostics["ray_used"][0]), 1)
        self.assertAlmostEqual(
            float(diagnostics["applied_correction"][0]), 1.2
        )


if __name__ == "__main__":
    unittest.main()
