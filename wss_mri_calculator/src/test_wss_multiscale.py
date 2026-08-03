from __future__ import annotations

import unittest

import numpy as np
from scipy.spatial import cKDTree

from wss_multiscale import fit_wall_gradient_multiscale


class MultiScaleWallGradientTest(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(13)
        eta_m = np.linspace(8e-5, 1.4e-3, 128)
        lateral_mm = rng.normal(scale=0.08, size=(len(eta_m), 2))
        self.interior_mm = np.column_stack(
            [lateral_mm[:, 0], lateral_mm[:, 1], eta_m * 1e3]
        )
        self.wall_mm = np.zeros((1, 3), dtype=np.float64)
        self.normals = np.array([[0.0, 0.0, 1.0]])
        self.true_gradient = np.array([36.0, -9.0, 0.0])
        self.tree = cKDTree(self.interior_mm)
        self.eta_m = eta_m

    def _fit(self, velocity: np.ndarray):
        return fit_wall_gradient_multiscale(
            self.wall_mm,
            self.normals,
            self.interior_mm,
            velocity,
            self.tree,
            adaptive_neighbors=(16, 20, 24, 28, 32, 40, 48, 64, 96),
            smooth_neighbors=1,
        )

    def test_exact_quadratic_keeps_unit_correction(self) -> None:
        quadratic = np.array([5200.0, -1400.0, 0.0])
        velocity = (
            self.eta_m[:, None] * self.true_gradient
            + self.eta_m[:, None] ** 2 * quadratic
        )
        gradient, used, diagnostics = self._fit(velocity)
        np.testing.assert_allclose(gradient[0], self.true_gradient, rtol=1e-9, atol=1e-9)
        self.assertGreaterEqual(int(used[0]), 5)
        self.assertAlmostEqual(float(diagnostics["correction"][0]), 1.0, places=8)

    def test_correction_is_positive_and_bounded(self) -> None:
        quadratic = np.array([7000.0, -2000.0, 0.0])
        cubic = np.array([-2.5e6, 7.0e5, 0.0])
        velocity = (
            self.eta_m[:, None] * self.true_gradient
            + self.eta_m[:, None] ** 2 * quadratic
            + self.eta_m[:, None] ** 3 * cubic
        )
        gradient, _, diagnostics = self._fit(velocity)
        correction = float(diagnostics["correction"][0])
        self.assertGreaterEqual(correction, 0.85)
        self.assertLessEqual(correction, 1.35)
        self.assertGreater(float(np.dot(gradient[0], self.true_gradient)), 0.0)

    def test_rejects_invalid_configuration(self) -> None:
        velocity = self.eta_m[:, None] * self.true_gradient
        with self.assertRaises(ValueError):
            fit_wall_gradient_multiscale(
                self.wall_mm,
                self.normals,
                self.interior_mm,
                velocity,
                self.tree,
                adaptive_neighbors=(16, 32),
            )


if __name__ == "__main__":
    unittest.main()
