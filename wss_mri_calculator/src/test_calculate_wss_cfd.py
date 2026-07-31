from __future__ import annotations

import unittest

import numpy as np
from scipy.spatial import cKDTree

from calculate_wss_cfd import fit_wall_gradient


class FitWallGradientTest(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(7)
        n = 96
        eta_m = np.linspace(8e-5, 1.2e-3, n)
        lateral_mm = rng.normal(scale=0.08, size=(n, 2))
        self.interior_mm = np.column_stack(
            [lateral_mm[:, 0], lateral_mm[:, 1], eta_m * 1e3]
        )
        self.true_gradient = np.array([32.0, -11.0, 0.0])
        quadratic = np.array([4200.0, -1800.0, 0.0])
        self.velocity = (
            eta_m[:, None] * self.true_gradient
            + eta_m[:, None] ** 2 * quadratic
        )
        self.wall_mm = np.zeros((1, 3), dtype=np.float64)
        self.normals = np.array([[0.0, 0.0, 1.0]])
        self.tree = cKDTree(self.interior_mm)

    def test_fixed_quadratic_recovers_wall_derivative(self) -> None:
        gradient, used, diagnostics = fit_wall_gradient(
            self.wall_mm,
            self.normals,
            self.interior_mm,
            self.velocity,
            self.tree,
            neighbors=64,
            degree=2,
            neighbor_mode="fixed",
        )
        np.testing.assert_allclose(gradient[0], self.true_gradient, rtol=1e-10, atol=1e-10)
        self.assertEqual(int(used[0]), 64)
        self.assertEqual(int(diagnostics["selected_neighbors"][0]), 64)
        self.assertLess(float(diagnostics["design_condition"][0]), 1e4)

    def test_adaptive_cv_uses_velocity_only_and_recovers_derivative(self) -> None:
        candidates = (16, 20, 24, 32, 40, 48, 64)
        gradient, used, diagnostics = fit_wall_gradient(
            self.wall_mm,
            self.normals,
            self.interior_mm,
            self.velocity,
            self.tree,
            degree=2,
            neighbor_mode="adaptive_cv",
            adaptive_neighbors=candidates,
            cv_tolerance=1.25,
        )
        np.testing.assert_allclose(gradient[0], self.true_gradient, rtol=1e-10, atol=1e-10)
        self.assertIn(int(diagnostics["selected_neighbors"][0]), candidates)
        self.assertGreaterEqual(int(used[0]), 5)
        self.assertTrue(np.isfinite(diagnostics["cv_score"][0]))

    def test_invalid_cv_tolerance_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            fit_wall_gradient(
                self.wall_mm,
                self.normals,
                self.interior_mm,
                self.velocity,
                self.tree,
                neighbor_mode="adaptive_cv",
                cv_tolerance=0.99,
            )


if __name__ == "__main__":
    unittest.main()
