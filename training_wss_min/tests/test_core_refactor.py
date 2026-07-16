"""Regression tests for the cleaned core module boundaries and frozen baseline."""

from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path

import torch

from training_wss_min.baseline_models import PointNetRegressor
from training_wss_min.config import ExpConfig, ModelConfig
from training_wss_min.examples.pointnet_baseline_train import PointNetWSS
from training_wss_min.models import SUPPORTED_MODELS, build_model
from training_wss_min import dataset as D


REPO = Path(__file__).resolve().parents[2]
BASELINE_MANIFEST = (
    REPO / "training_wss_min/configs/sweeps/baseline_2x3_simple.txt"
)


class TestModelFactory(unittest.TestCase):
    def test_all_registered_models_construct(self):
        for name in sorted(SUPPORTED_MODELS):
            with self.subTest(name=name):
                model = build_model(ModelConfig(name=name), in_dim=3)
                self.assertGreater(sum(p.numel() for p in model.parameters()), 0)


class TestCanonicalV4Split(unittest.TestCase):
    def test_legacy_and_canonical_ids_are_supported(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "split.json"
            path.write_text(json.dumps({
                "train_cases": ["slow/OLD", "AAA/ruputer/NEW"],
                "val_cases": [], "test_cases": ["AG/fast/TEST"],
            }))
            self.assertEqual(D.load_split_cases(path, "train"), [
                ("AG/slow", "OLD"), ("AAA/ruputer", "NEW")])

    def test_partition_leakage_and_invalid_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "split.json"
            path.write_text(json.dumps({
                "train_cases": ["slow/SAME"], "val_cases": [],
                "test_cases": ["AG/slow/SAME"],
            }))
            with self.assertRaises(ValueError): D.load_split(path)
            path.write_text(json.dumps({"train_cases": ["ILO/X/Y"], "val_cases": [], "test_cases": []}))
            with self.assertRaises(ValueError): D.load_split(path)

    def test_pointnext_rejects_mismatched_stage_lengths(self):
        cfg = ModelConfig(name="pointnext_s", sa_blocks=(1,))
        with self.assertRaises(ValueError):
            build_model(cfg, in_dim=3)


class TestFrozenBaseline(unittest.TestCase):
    def test_manifest_has_six_parseable_configs(self):
        paths = [REPO / line for line in BASELINE_MANIFEST.read_text().splitlines() if line]
        self.assertEqual(len(paths), 6)
        configs = [ExpConfig.from_json(path) for path in paths]
        self.assertEqual(
            {cfg.model.name for cfg in configs},
            {"mlp", "pointnet", "pointnetpp"},
        )
        self.assertEqual(
            {tuple(cfg.data.input_features) for cfg in configs},
            {
                ("x", "y", "z"),
                ("x", "y", "z", "abscissa_norm", "local_radius", "curvature"),
            },
        )
        self.assertTrue(all(cfg.name.startswith("baseline_2x3_simple/outputs/") for cfg in configs))


class TestTeacherPointNetExample(unittest.TestCase):
    def test_example_stays_within_250_lines(self):
        source = REPO / "training_wss_min/examples/pointnet_baseline_train.py"
        text = source.read_text()
        self.assertLessEqual(len(text.splitlines()), 250)
        self.assertNotIn("from training_wss_min", text)
        self.assertNotIn("import training_wss_min", text)
        self.assertNotIn("torch_geometric", text)

    def test_model_matches_frozen_baseline_implementation(self):
        reference = PointNetRegressor(6, width=8, head_hidden=8)
        example = PointNetWSS(6, width=8, head_dim=8)
        example.load_state_dict(reference.state_dict())
        reference.eval()
        example.eval()
        torch.manual_seed(7)
        x = torch.randn(20, 6)
        pos = torch.randn(20, 3)
        batch = torch.tensor([0] * 8 + [1] * 12)
        torch.testing.assert_close(reference(pos, x, batch), example(pos, x, batch))


if __name__ == "__main__":
    unittest.main()
