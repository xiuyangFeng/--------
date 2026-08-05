from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from apply_v4_calibrator import _validate_feature_schema
from run_v4_experiment import run


ROOT = Path("/public/newhome/cy/Digital_twin/GNN/wss_mri_calculator")
TRAIN_CONFIG = ROOT / "experiments/pointcloud_surface_mls_v4/config_train36.json"
TEST_CONFIG = ROOT / "experiments/pointcloud_surface_mls_v4/config_test35_cache.json"


class V4CalibratorContractTest(unittest.TestCase):
    def test_matching_feature_schema_is_accepted(self) -> None:
        _validate_feature_schema(
            {"feature_names": ["physics", "surface_fit"]},
            ["physics", "surface_fit"],
        )

    def test_feature_schema_mismatch_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "feature schema mismatch"):
            _validate_feature_schema(
                {"feature_names": ["physics", "surface_fit"]},
                ["surface_fit", "physics"],
            )

    def test_test_role_requires_evaluation_only(self) -> None:
        payload = json.loads(TRAIN_CONFIG.read_text(encoding="utf-8"))
        payload["split"]["roles"] = ["test"]
        payload["evaluation_only"] = False
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "invalid_test_config.json"
            config_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "evaluation_only=true"):
                run(config_path)

    def test_repository_test_config_is_explicitly_evaluation_only(self) -> None:
        payload = json.loads(TEST_CONFIG.read_text(encoding="utf-8"))
        self.assertEqual(payload["split"]["roles"], ["test"])
        self.assertTrue(payload["evaluation_only"])


if __name__ == "__main__":
    unittest.main()
