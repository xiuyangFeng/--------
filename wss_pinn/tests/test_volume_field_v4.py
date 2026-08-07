from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from wss_pinn.config import ExperimentConfig
from wss_pinn.data import VolumeFieldDataset
from wss_pinn.models import build_model
from wss_pinn.tools import b0_atlas_v4
from wss_pinn.tools.audit_boundary_field_v4 import _parse_peak_from_log
from wss_pinn.validation import aggregate_case_validation
from wss_pinn.volume_utils import tensor_state_sha256


ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = ROOT / "wss_pinn/configs/volume_uvwp_peak_field_v4"
CONFIRM_CONFIG_ROOT = ROOT / "wss_pinn/configs/volume_uvwp_peak_field_v4_multiseed_top2"


class FieldV4Tests(unittest.TestCase):
    def test_case_balanced_score_matches_manual_and_exposes_legacy_double_weight(self):
        cases = [
            {"case_id": f"case-{index}", "field_score": float(index + 1), "points": 5000}
            for index in range(15)
        ]
        report = aggregate_case_validation(cases, legacy_batch_cases=2)
        self.assertAlmostEqual(report["validation_field_score_cb"], 8.0)
        expected_legacy = sum([1.5, 3.5, 5.5, 7.5, 9.5, 11.5, 13.5, 15.0]) / 8.0
        self.assertAlmostEqual(
            report["validation_field_score_legacy_batch_weighted"], expected_legacy
        )
        self.assertEqual(report["legacy_overweighted_case_ids"], ["case-14"])
        self.assertAlmostEqual(report["legacy_weight_ratio_max_to_min"], 2.0)

    def test_four_configs_are_data_only_guard_test35_and_match_parameter_budget(self):
        paths = sorted(CONFIG_ROOT.glob("*_s1234.json"))
        self.assertEqual(len(paths), 4)
        configs = [ExperimentConfig.from_json(path) for path in paths]
        self.assertEqual({config.mode for config in configs}, {"data"})
        self.assertEqual(
            {tuple(config["data"]["eval_roles"]) for config in configs},
            {("val",)},
        )

    def test_four_configs_parameter_ratio_and_paired_initialization(self):
        configs = {
            path.name: ExperimentConfig.from_json(path)
            for path in sorted(CONFIG_ROOT.glob("*_s1234.json"))
        }
        counts = {}
        hashes = {}
        for name, config in configs.items():
            torch.manual_seed(1234)
            model = build_model(config)
            counts[name] = sum(parameter.numel() for parameter in model.parameters())
            hashes[name] = tensor_state_sha256(model.state_dict())
        self.assertLessEqual(max(counts.values()) / min(counts.values()), 1.10)
        self.assertEqual(hashes["g_raw_s1234.json"], hashes["l_raw_s1234.json"])
        self.assertEqual(hashes["g_pe_s1234.json"], hashes["l_pe_s1234.json"])

    def test_confirmation_matrix_changes_only_seed_and_selected_query_axis(self):
        paths = sorted(CONFIRM_CONFIG_ROOT.glob("g_*_s*.json"))
        self.assertEqual(len(paths), 4)
        configs = {
            path.stem: ExperimentConfig.from_json(path) for path in paths
        }
        self.assertEqual(
            {int(config["train"]["seed"]) for config in configs.values()},
            {2345, 3456},
        )
        self.assertEqual(
            {int(config["sampling"]["seed"]) for config in configs.values()},
            {1234},
        )
        self.assertEqual({config.mode for config in configs.values()}, {"data"})

        def normalized(config, *, seed=False, query=False):
            payload = config.as_dict()
            payload["experiment"].pop("id")
            payload["experiment"].pop("description")
            payload["paths"].pop("run_dir")
            if seed:
                payload["train"]["seed"] = "seed"
            if query:
                payload["model"]["field_v4"]["query_encoding"] = "query"
            return payload

        for encoding in ("raw", "pe"):
            self.assertEqual(
                normalized(configs[f"g_{encoding}_s2345"], seed=True),
                normalized(configs[f"g_{encoding}_s3456"], seed=True),
            )
        for seed in (2345, 3456):
            self.assertEqual(
                normalized(configs[f"g_raw_s{seed}"], query=True),
                normalized(configs[f"g_pe_s{seed}"], query=True),
            )

    def test_fixed_validation_queries_are_shared_and_test_role_is_rejected(self):
        config = ExperimentConfig.from_json(CONFIG_ROOT / "g_raw_s1234.json")
        dataset = VolumeFieldDataset(
            config["paths"]["sidecar_manifest"],
            config["paths"]["field_stats"],
            roles=["val"],
            input_variant=config.input_variant,
            sampling=config["sampling"],
            seed=1234,
        )
        dataset.set_epoch(0)
        first = dataset[0]
        dataset.set_epoch(37)
        second = dataset[0]
        np.testing.assert_array_equal(
            first["query_indices"], second["query_indices"]
        )
        with self.assertRaisesRegex(ValueError, "test35 guard"):
            VolumeFieldDataset(
                config["paths"]["sidecar_manifest"],
                config["paths"]["field_stats"],
                roles=["test"],
                input_variant=config.input_variant,
                sampling=config["sampling"],
                seed=1234,
            )

    def test_local_and_pe_queries_have_finite_first_and_second_derivatives(self):
        for name in ("l_raw_s1234.json", "g_pe_s1234.json", "l_pe_s1234.json"):
            config = ExperimentConfig.from_json(CONFIG_ROOT / name)
            torch.manual_seed(1234)
            model = build_model(config).double().eval()
            support = torch.rand(256, 3, dtype=torch.float64)
            features = torch.rand(256, 6, dtype=torch.float64)
            encoded = model.encode_support(
                support,
                features,
                torch.zeros(256, dtype=torch.long),
                unit_ids=["synthetic"],
                epoch=0,
                global_seed=1234,
                evaluation=True,
            )
            query = torch.rand(8, 3, dtype=torch.float64, requires_grad=True)
            output = model.decode_query(encoded, query, torch.zeros(8, dtype=torch.long))
            first = torch.autograd.grad(output[:, 0].sum(), query, create_graph=True)[0]
            second = torch.autograd.grad(first.square().sum(), query)[0]
            self.assertTrue(torch.isfinite(first).all())
            self.assertTrue(torch.isfinite(second).all())

    def test_b0_predictor_interface_has_no_truth_argument(self):
        parameters = set(inspect.signature(b0_atlas_v4.predict).parameters)
        self.assertEqual(parameters, {"force", "chunk_size"})
        query_manifest = json.loads(b0_atlas_v4.QUERY_MANIFEST.read_text())
        self.assertEqual(
            query_manifest["truth_access"],
            "none; coordinate files contain no u/v/w/p targets",
        )
        self.assertFalse(
            any(
                key in row["prediction_only_coords"]
                for row in query_manifest["cases"]
                for key in ("velocity", "pressure", "target", "truth")
            )
        )

    def test_outlet_log_parser_deduplicates_mpi_rows_and_aligns_peak_step(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "Fluent.out"
            lines = []
            for label, pressure, flow in (
                ("outle", 101.0, 1.0),
                ("outli", 102.0, 2.0),
                ("outri", 103.0, 3.0),
                ("outre", 104.0, 4.0),
            ):
                row = f"P_ave_{label}={pressure} Q_ave_{label}={flow}\n"
                lines.extend([row, row, row])
            lines.append("Flow time = 0.5s, time step = 100\n")
            path.write_text("".join(lines), encoding="utf-8")
            result = _parse_peak_from_log(path, 100)
        self.assertEqual(result["time_step"], 100)
        self.assertEqual(result["outlets"]["out-le"]["printed_rows"], 3)
        self.assertEqual(
            result["outlets"]["out-le"]["unique_rows_after_mpi_dedup"], 1
        )
        self.assertEqual(result["outlets"]["out-ri"]["pressure_absolute_pa"], 103.0)


if __name__ == "__main__":
    unittest.main()
