from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from wss_pinn.config import DEFAULTS, ExperimentConfig
from wss_pinn.data.dataset import PhysicsDataset
from wss_pinn.data.raw_io import load_cases
from wss_pinn.data.sampling import build_three_slot_sampling
from wss_pinn.tools.preflight import science_gate
from wss_pinn.tools.prepare_full_training_configs import build_full_configs
from wss_pinn.tools.derive_pinn_split import derive_split
from wss_pinn.utils import ROOT, guard_write_path


def payload(stage: str) -> dict:
    value = copy.deepcopy(DEFAULTS)
    value["experiment"] = {"id": f"test-{stage}", "stage": stage}
    value["train"]["run_dir"] = str(ROOT / "outputs/wss_pinn/tests/test-run")
    if stage == "f0u":
        value["model"]["predict_pressure"] = False
    elif stage == "f0up":
        value["experiment"]["scientific_gate_control_run_dir"] = str(
            ROOT / "outputs/wss_pinn/tests/f0u-control"
        )
        value["model"]["predict_pressure"] = True
        value["loss"]["pressure_data_weight"] = 1.0
    elif stage == "f1":
        value["experiment"]["control_run_dir"] = str(
            ROOT / "outputs/wss_pinn/tests/f0up-control"
        )
        value["model"]["predict_pressure"] = True
        value["loss"]["pressure_data_weight"] = 1.0
        value["loss"]["continuity_weight"] = 0.1
        value["loss"]["no_slip_weight"] = 0.1
    return value


class ConfigSamplingTests(unittest.TestCase):
    def test_valid_stage_contracts(self):
        for stage in ("f0u", "f0up", "f1"):
            self.assertEqual(ExperimentConfig(payload(stage)).stage, stage)

    def test_f1_rejects_momentum(self):
        value = payload("f1")
        value["loss"]["momentum_weight"] = 0.1
        with self.assertRaisesRegex(ValueError, "momentum"):
            ExperimentConfig(value)

    def test_f1_requires_both_first_order_losses(self):
        value = payload("f1")
        value["loss"]["no_slip_weight"] = 0.0
        with self.assertRaisesRegex(ValueError, "F1"):
            ExperimentConfig(value)

    def test_f1_diagnostic_allows_one_first_order_loss(self):
        value = payload("f1")
        value["experiment"]["diagnostic_only"] = True
        value["loss"]["no_slip_weight"] = 0.0
        self.assertEqual(ExperimentConfig(value).stage, "f1")

    def test_init_and_resume_are_mutually_exclusive(self):
        value = payload("f1")
        value["train"]["init_checkpoint"] = str(ROOT / "outputs/wss_pinn/a.pt")
        value["train"]["resume"] = str(ROOT / "outputs/wss_pinn/b.pt")
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            ExperimentConfig(value)

    def test_write_guard_rejects_old_roots(self):
        with self.assertRaisesRegex(ValueError, "read-only"):
            guard_write_path(ROOT / "training_wss_min/runs/forbidden")

    def test_three_slots_reproducible_and_disjoint(self):
        int_type = np.asarray([1] * 20 + [0] * 30, dtype=np.int8)
        first = build_three_slot_sampling(40, int_type, 10, 8, 9, 1234)
        second = build_three_slot_sampling(40, int_type, 10, 8, 9, 1234)
        for slot in first:
            np.testing.assert_array_equal(first[slot], second[slot])
            self.assertGreater(len(first[slot]), 0)
        self.assertEqual(np.intersect1d(first["near_wall"], first["core"]).size, 0)

    def test_grouped_split_expands_roles_and_frozen_paths(self):
        payload = {
            "train_cases": ["AG/slow/CASE_A"],
            "test_cases": ["ILO/CASE_B-0/before"],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            rows = load_cases(path)
        self.assertEqual([row["role"] for row in rows], ["train", "test"])
        self.assertEqual(rows[0]["canonical_id"], "AG/slow/CASE_A")
        self.assertEqual(
            rows[1]["bundle_path"],
            str(ROOT / "data_wss_min/ILO/CASE_B-0/before/bundle.npz"),
        )

    def test_pinn_split_derivation_only_removes_requested_test_case(self):
        parent_path = (
            ROOT
            / "training/splits/split_AG_AAA_ILO_q2v_pool2025_train138_test36.json"
        )
        parent = json.loads(parent_path.read_text(encoding="utf-8"))
        result = derive_split(
            parent,
            parent_path=parent_path,
            remove_test_case="AAA/ruputer/SHI_YUN_XI",
            split_version="test-pinn-split",
            reason="test-only exclusion",
        )
        self.assertEqual(result["train_cases"], parent["train_cases"])
        self.assertEqual(result["counts"]["train"], 138)
        self.assertEqual(result["counts"]["test"], 35)
        self.assertNotIn(
            "AAA/ruputer/SHI_YUN_XI", result["test_cases"]
        )
        self.assertIn(
            "AAA/ruputer/SHI_YUN_XI", result["unused_cases"]
        )
        self.assertTrue(result["baseline_parent_immutable"])

    def test_case_batching_is_deterministic(self):
        class FakeCase:
            def __init__(self, name):
                self.name = name

            def sample(self, **kwargs):
                return {"name": self.name, "seed": kwargs["seed"]}

        dataset = PhysicsDataset.__new__(PhysicsDataset)
        dataset.cases = [FakeCase(str(index)) for index in range(10)]
        config = {
            "seed": 1234,
            "batch_cases": 4,
            "batch_wall": 1,
            "batch_near_wall": 1,
            "batch_core": 1,
        }
        first = dataset.epoch_samples(config, 7)
        second = dataset.epoch_samples(config, 7)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 4)

    def test_full_f1_gate_requires_selected_diagnostic_weights(self):
        selected_path = (
            ROOT
            / "wss_pinn/configs/f1_diagnostics_20260730/f1d_c1e4_n1.json"
        )
        selected = ExperimentConfig.from_json(selected_path)
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "matrix.json"
            report_path.write_text(
                json.dumps(
                    {
                        "gate_result": "pass",
                        "selected_config": str(selected_path),
                    }
                ),
                encoding="utf-8",
            )
            matching_payload = selected.as_dict()
            matching_payload["experiment"]["diagnostic_only"] = False
            matching_payload["experiment"]["scientific_gate_report"] = str(
                report_path
            )
            matching = ExperimentConfig(matching_payload)
            self.assertTrue(science_gate(matching)["passed"])

            mismatched_payload = matching.as_dict()
            mismatched_payload["loss"]["no_slip_weight"] = 10.0
            mismatched = ExperimentConfig(mismatched_payload)
            gate = science_gate(mismatched)
            self.assertFalse(gate["passed"])
            self.assertFalse(
                gate["checks"]["f1_diagnostic_matrix"][
                    "selected_weights_match"
                ]
            )

    def test_source_audit_gate_is_bound_to_config_split(self):
        selected_path = (
            ROOT
            / "wss_pinn/configs/f1_diagnostics_20260730/f1d_c1e4_n1.json"
        )
        selected = ExperimentConfig.from_json(selected_path)
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "source.json"
            report_path.write_text(
                json.dumps(
                    {
                        "gate_result": "pass",
                        "split": {"sha256": "wrong-split"},
                    }
                ),
                encoding="utf-8",
            )
            value = selected.as_dict()
            value["experiment"]["source_data_audit_report"] = str(report_path)
            gate = science_gate(ExperimentConfig(value))
            self.assertFalse(gate["passed"])
            self.assertFalse(
                gate["checks"]["source_data_audit"]["split_binding_match"]
            )

    def test_full_config_builder_keeps_f0up_gate_and_f1_pairing_separate(self):
        p1 = ExperimentConfig.from_json(
            ROOT / "wss_pinn/configs/p1_full_train138_test36_v1.json"
        )
        selected = ExperimentConfig.from_json(
            ROOT
            / "wss_pinn/configs/f1_diagnostics_20260730/f1d_c1e4_n10.json"
        )
        marker = ROOT / "outputs/wss_pinn/tests/fake-report.json"
        f0up, f1 = build_full_configs(
            p1,
            source_report_path=marker,
            sidecar_report_path=marker,
            matrix_report_path=marker,
            selected=selected,
            run_tag="test",
            batch_cases=3,
        )
        self.assertIn(
            "scientific_gate_control_run_dir", f0up["experiment"]
        )
        self.assertNotIn("control_run_dir", f0up["experiment"])
        self.assertEqual(
            f1["experiment"]["control_run_dir"], f0up["train"]["run_dir"]
        )
        self.assertEqual(f0up["model"], f1["model"])
        self.assertEqual(f0up["data"], f1["data"])
        self.assertEqual(f0up["sampling"], f1["sampling"])
        self.assertEqual(f0up["train"]["batch_cases"], 3)
        self.assertEqual(f1["loss"]["continuity_weight"], 1e-4)
        self.assertEqual(f1["loss"]["no_slip_weight"], 10.0)
        self.assertEqual(f1["loss"]["wss_physics_weight"], 0.0)
        self.assertEqual(f1["loss"]["momentum_weight"], 0.0)


if __name__ == "__main__":
    unittest.main()
