from __future__ import annotations

import unittest

import numpy as np

from wss_pinn.volume_field.data.builder import (
    _geometry,
    rotate_velocity_to_registered,
)
from wss_pinn.volume_field.data.dataset import VolumeFieldDataset, _take_strict_volume


class VolumeDataTests(unittest.TestCase):
    def test_velocity_rotation_preserves_speed_and_rotates_components(self):
        rotation = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        velocity = np.asarray([[1.0, 0.0, 0.0], [0.0, 2.0, 3.0]])
        registered = rotate_velocity_to_registered(velocity, rotation)
        np.testing.assert_allclose(registered, velocity @ rotation, atol=1e-7)
        np.testing.assert_allclose(
            np.linalg.norm(registered, axis=1), np.linalg.norm(velocity, axis=1), atol=1e-7
        )

    def test_geometry_uses_signed_log_curvature(self):
        output = _geometry(
            np.asarray([0.0, 1.0]),
            np.asarray([2.0, 3.0]),
            np.asarray([-9.0, 9.0]),
        )
        np.testing.assert_allclose(output[:, 2], [-np.log(10.0), np.log(10.0)])

    def test_strict_volume_sampler_excludes_boundary_and_support(self):
        wall = np.zeros(100, dtype=bool)
        wall[[2, 4, 6, 8]] = True
        rng = np.random.default_rng(1234)
        support = _take_strict_volume(rng, wall, 20)
        query = _take_strict_volume(rng, wall, 20, excluded=support)
        self.assertFalse(bool(wall[support].any()))
        self.assertFalse(bool(wall[query].any()))
        self.assertFalse(bool(set(support).intersection(query)))
        self.assertEqual(len(np.unique(support)), 20)
        self.assertEqual(len(np.unique(query)), 20)

    def test_same_query_mode_reuses_support_indices_without_labels_in_support(self):
        class FakeCase:
            case_id = "AG/fast/SAME_SYNTHETIC"
            coords = np.arange(360, dtype=np.float32).reshape(120, 3)
            geometry = np.ones((120, 3), dtype=np.float32)
            region = np.zeros(120, dtype=np.int8)
            is_wall = np.zeros(120, dtype=bool)
            velocity = np.ones((120, 3), dtype=np.float32)
            pressure = np.ones(120, dtype=np.float32)
            wall_coords = np.ones((40, 3), dtype=np.float32)
            length_m = 0.05

        dataset = VolumeFieldDataset.__new__(VolumeFieldDataset)
        dataset.cases = [FakeCase()]
        dataset.input_variant = "xyz"
        dataset.sampling = {
            "support_points": 32,
            "query_points": 32,
            "physics_points": 8,
            "wall_points": 8,
            "query_mode": "same",
        }
        dataset.seed = 1234
        dataset.epoch = 0
        dataset.velocity_mean = np.zeros(3, dtype=np.float32)
        dataset.velocity_std = np.ones(3, dtype=np.float32)
        dataset.pressure_mean = 0.0
        dataset.pressure_std = 1.0
        sample = dataset[0]
        np.testing.assert_array_equal(sample["support_coords"], sample["query_coords"])
        self.assertNotIn("support_target", sample)


if __name__ == "__main__":
    unittest.main()
