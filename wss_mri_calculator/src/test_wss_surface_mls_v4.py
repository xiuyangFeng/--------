from __future__ import annotations

import unittest

import numpy as np
from scipy.spatial import cKDTree

from wss_surface_mls_v4 import SurfaceMLSV4Config, fit_wall_gradient_surface_mls
from wss_surface_mls_v4 import (
    SurfaceColumnV4Config,
    fit_wall_gradient_surface_columns,
)


class SurfaceMLSV4Test(unittest.TestCase):
    def test_recovers_planar_surface_gradient_with_tangential_variation(self) -> None:
        grid = np.linspace(-1.0, 1.0, 9)
        xx, yy = np.meshgrid(grid, grid, indexing="ij")
        wall = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)])
        normals = np.tile(np.array([0.0, 0.0, 1.0]), (len(wall), 1))
        depths_mm = np.array([0.12, 0.25, 0.45, 0.70, 1.0])
        interior = np.concatenate(
            [wall + depth * normals for depth in depths_mm], axis=0
        )
        anchor_x = np.tile(wall[:, 0], len(depths_mm))
        depth_m = np.repeat(depths_mm, len(wall)) * 1e-3
        target_gradient = 120.0
        local_gradient = target_gradient + 8.0 * anchor_x
        velocity_x = depth_m * local_gradient - 2.0e4 * depth_m**2
        velocity = np.column_stack(
            [velocity_x, np.zeros(len(interior)), np.zeros(len(interior))]
        )
        target = np.zeros((1, 3), dtype=np.float64)
        target_normal = np.array([[0.0, 0.0, 1.0]])
        fallback = np.array([[90.0, 0.0, 0.0]])
        gradient, diagnostics = fit_wall_gradient_surface_mls(
            target,
            target_normal,
            wall,
            normals,
            interior,
            velocity,
            cKDTree(interior),
            local_radius_mm=np.array([5.0]),
            fallback_gradient=fallback,
            config=SurfaceMLSV4Config(
                query_neighbors=len(interior),
                correction_max=1.5,
                surface_max_radius_mm=2.0,
            ),
        )
        self.assertAlmostEqual(float(gradient[0, 0]), target_gradient, delta=2.0)
        self.assertEqual(int(diagnostics["applied"][0]), 1)
        self.assertGreaterEqual(int(diagnostics["anchor_groups"][0]), 6)
        self.assertAlmostEqual(
            float(diagnostics["min_cell_depth_mm"][0]),
            float(depths_mm.min()),
            delta=0.03,
        )
        self.assertGreater(float(diagnostics["depth_span_mm"][0]), 0.2)
        self.assertTrue(np.isfinite(diagnostics["depth_coefficient_2_ratio"][0]))
        self.assertTrue(np.isfinite(diagnostics["fit_residual_nrmse"][0]))
        self.assertGreaterEqual(int(diagnostics["fit_count"][0]), 1)
        self.assertTrue(np.isfinite(diagnostics["near_secant_gradient_median"][0]))
        self.assertGreaterEqual(int(diagnostics["near_cell_count"][0]), 4)

    def test_uses_each_anchor_normal_on_a_curved_wall(self) -> None:
        radius_mm = 5.0
        angle = np.linspace(-0.22, 0.22, 23)
        axial = np.linspace(-1.0, 1.0, 7)
        aa, zz = np.meshgrid(angle, axial, indexing="ij")
        wall = np.column_stack(
            [
                radius_mm * np.cos(aa.ravel()),
                radius_mm * np.sin(aa.ravel()),
                zz.ravel(),
            ]
        )
        normals = np.column_stack(
            [
                -np.cos(aa.ravel()),
                -np.sin(aa.ravel()),
                np.zeros(aa.size),
            ]
        )
        depths_mm = np.array([0.15, 0.30, 0.55, 0.85, 1.15])
        interior = np.concatenate(
            [wall + depth * normals for depth in depths_mm], axis=0
        )
        depth_m = np.repeat(depths_mm, len(wall)) * 1e-3
        velocity_z = depth_m * 75.0 - 8.0e3 * depth_m**2
        velocity = np.column_stack(
            [np.zeros(len(interior)), np.zeros(len(interior)), velocity_z]
        )
        target = np.array([[radius_mm, 0.0, 0.0]])
        target_normal = np.array([[-1.0, 0.0, 0.0]])
        fallback = np.array([[0.0, 0.0, 60.0]])
        gradient, diagnostics = fit_wall_gradient_surface_mls(
            target,
            target_normal,
            wall,
            normals,
            interior,
            velocity,
            cKDTree(interior),
            local_radius_mm=np.array([radius_mm]),
            fallback_gradient=fallback,
            config=SurfaceMLSV4Config(
                query_neighbors=len(interior),
                correction_max=1.5,
                surface_max_radius_mm=2.0,
            ),
        )
        self.assertAlmostEqual(float(gradient[0, 2]), 75.0, delta=2.0)
        self.assertTrue(np.isfinite(diagnostics["raw_scalar_gradient"][0]))

    def test_invalid_configuration_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            SurfaceMLSV4Config(correction_min=1.5, correction_max=1.2).validate()

    def test_parallel_transport_preserves_rotated_tangent_magnitude(self) -> None:
        from wss_surface_mls_v4 import _parallel_transport_to_target

        angle = np.deg2rad(35.0)
        anchor_normal = np.array([[np.sin(angle), 0.0, np.cos(angle)]])
        target_normal = np.array([0.0, 0.0, 1.0])
        anchor_tangent = np.array([[np.cos(angle), 0.0, -np.sin(angle)]])
        transported = _parallel_transport_to_target(
            anchor_tangent, anchor_normal, target_normal
        )
        np.testing.assert_allclose(transported, [[1.0, 0.0, 0.0]], atol=1e-12)

    def test_two_level_columns_recover_planar_gradient(self) -> None:
        grid = np.linspace(-0.8, 0.8, 7)
        xx, yy = np.meshgrid(grid, grid, indexing="ij")
        wall = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)])
        normals = np.tile(np.array([0.0, 0.0, 1.0]), (len(wall), 1))
        depths_mm = np.array([0.10, 0.22, 0.38, 0.60, 0.90])
        interior = np.concatenate(
            [wall + depth * normals for depth in depths_mm], axis=0
        )
        anchor_x = np.tile(wall[:, 0], len(depths_mm))
        depth_m = np.repeat(depths_mm, len(wall)) * 1e-3
        local_gradient = 100.0 + 10.0 * anchor_x
        velocity_x = depth_m * local_gradient - 1.2e4 * depth_m**2
        velocity = np.column_stack(
            [velocity_x, np.zeros(len(interior)), np.zeros(len(interior))]
        )
        gradient, diagnostics = fit_wall_gradient_surface_columns(
            np.zeros((1, 3)),
            np.array([[0.0, 0.0, 1.0]]),
            wall,
            normals,
            interior,
            velocity,
            cKDTree(interior),
            local_radius_mm=np.array([5.0]),
            fallback_gradient=np.array([[80.0, 0.0, 0.0]]),
            config=SurfaceColumnV4Config(
                query_neighbors=len(interior),
                surface_max_radius_mm=2.0,
            ),
        )
        self.assertAlmostEqual(float(gradient[0, 0]), 100.0, delta=3.0)
        self.assertGreaterEqual(int(diagnostics["selected_columns"][0]), 6)


if __name__ == "__main__":
    unittest.main()
