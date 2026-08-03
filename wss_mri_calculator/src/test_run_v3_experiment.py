from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from run_v3_experiment import load_experiment_config


ROOT = Path("/public/newhome/cy/Digital_twin/GNN/wss_mri_calculator")
SMOKE_CONFIG = (
    ROOT
    / "experiments/pointcloud_normal_multiscale_v3/config_smoke_train3.json"
)


class V3ExperimentConfigTest(unittest.TestCase):
    def test_repository_smoke_config_is_valid(self) -> None:
        payload = load_experiment_config(SMOKE_CONFIG)
        self.assertEqual(payload["split"]["roles"], ["train"])
        self.assertFalse(payload["method_frozen"])

    def test_blind_test_is_rejected_before_freeze(self) -> None:
        payload = json.loads(SMOKE_CONFIG.read_text(encoding="utf-8"))
        payload["split"]["roles"] = ["test"]
        payload["method_frozen"] = False
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid_test_config.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "test role is forbidden"):
                load_experiment_config(path)

    def test_runtime_must_target_node04(self) -> None:
        payload = json.loads(SMOKE_CONFIG.read_text(encoding="utf-8"))
        payload["runtime"]["host"] = "master"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid_host_config.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "runtime.host must be node04"):
                load_experiment_config(path)


if __name__ == "__main__":
    unittest.main()
