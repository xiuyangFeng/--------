"""Regression tests for the explicit log(local_radius) input feature."""

from __future__ import annotations

import unittest

import numpy as np

from training_wss_min import config as C
from training_wss_min import dataset as D


class TestLogLocalRadiusFeature(unittest.TestCase):
    @staticmethod
    def _case(values: list[float]) -> dict:
        radius = np.asarray(values, dtype=np.float32)
        return {
            "cohort": "AG/fast",
            "local_radius": radius,
            "log_local_radius": np.log(radius).astype(np.float32),
        }

    def test_config_accepts_log_local_radius(self) -> None:
        cfg = C.ExpConfig.from_dict(
            {
                "data": {
                    "input_features": [
                        "x",
                        "y",
                        "z",
                        "abscissa_norm",
                        "local_radius",
                        "curvature",
                        "log_local_radius",
                    ]
                }
            }
        )
        self.assertEqual(cfg.data.input_features[-1], "log_local_radius")
        self.assertEqual(C.input_dim(cfg), 7)

    def test_stats_and_feature_values_use_natural_log(self) -> None:
        cases = [self._case([1.0, 2.0]), self._case([4.0, 8.0])]
        stats = D.compute_feature_stats(cases, ("log_local_radius",))
        expected = np.log(np.asarray([1.0, 2.0, 4.0, 8.0]))
        self.assertAlmostEqual(
            stats["log_local_radius"]["mean"],
            float(expected.mean()),
            places=6,
        )
        case = {
            **cases[0],
            "pos": np.zeros((2, 3), dtype=np.float32),
        }
        features = D.build_features(
            case,
            np.asarray([0, 1]),
            ("log_local_radius",),
            stats,
        )
        normalized = (
            np.log(np.asarray([1.0, 2.0]))
            - stats["log_local_radius"]["mean"]
        ) / stats["log_local_radius"]["std"]
        np.testing.assert_allclose(features[:, 0], normalized, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
