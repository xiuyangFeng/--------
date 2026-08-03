from __future__ import annotations

import copy
import unittest
from pathlib import Path

from wss_pinn.volume_field.config import VolumeExperimentConfig


ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = ROOT / "wss_pinn/configs/volume_uvwp_peak_v1"
SAME_CONFIG_ROOT = ROOT / "wss_pinn/configs/volume_uvwp_peak_same5k_e7500_v2"


class VolumeConfigTests(unittest.TestCase):
    def test_eight_matrix_configs_validate(self):
        paths = sorted(CONFIG_ROOT.glob("*.json"))
        configs = [path for path in paths if path.name != "matrix.json"]
        self.assertEqual(len(configs), 8)
        loaded = [VolumeExperimentConfig.from_json(path) for path in configs]
        self.assertEqual({item.mode for item in loaded}, {"data_only", "pinn"})
        self.assertEqual({item.architecture for item in loaded}, {"pointnet", "pointnetpp"})
        self.assertEqual({item.input_variant for item in loaded}, {"xyz", "xyz_geom"})

    def test_warm_start_is_rejected(self):
        base = VolumeExperimentConfig.from_json(
            CONFIG_ROOT / "pointnet_xyz_pinn.json"
        ).as_dict()
        base["train"]["init_checkpoint"] = "outputs/wss_pinn/fake.pt"
        with self.assertRaisesRegex(ValueError, "warm-start"):
            VolumeExperimentConfig(base)

    def test_same5k_matrix_configs_validate(self):
        paths = sorted(SAME_CONFIG_ROOT.glob("*.json"))
        configs = [path for path in paths if path.name != "matrix.json"]
        self.assertEqual(len(configs), 8)
        loaded = [VolumeExperimentConfig.from_json(path) for path in configs]
        self.assertEqual({item["sampling"]["query_mode"] for item in loaded}, {"same"})
        self.assertEqual({item["train"]["epochs"] for item in loaded}, {7500})
        self.assertEqual(
            {tuple(item["train"]["milestone_epochs"]) for item in loaded},
            {(400, 1000, 2500, 5000, 7500)},
        )

    def test_same_mode_requires_equal_point_counts(self):
        base = VolumeExperimentConfig.from_json(
            CONFIG_ROOT / "pointnet_xyz_data_only.json"
        ).as_dict()
        base["sampling"]["query_mode"] = "same"
        base["sampling"]["query_points"] = 4999
        with self.assertRaisesRegex(ValueError, "support_points == query_points"):
            VolumeExperimentConfig(base)

    def test_data_only_cannot_hide_nonzero_physics(self):
        base = VolumeExperimentConfig.from_json(
            CONFIG_ROOT / "pointnet_xyz_data_only.json"
        ).as_dict()
        base["loss"]["continuity_weight"] = 1.0
        with self.assertRaisesRegex(ValueError, "data_only"):
            VolumeExperimentConfig(base)


if __name__ == "__main__":
    unittest.main()
