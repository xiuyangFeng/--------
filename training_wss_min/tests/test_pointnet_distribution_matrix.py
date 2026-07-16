"""Protocol tests for the PointNet distribution/normalization experiment matrix."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from training_wss_min import config as C
from training_wss_min import dataset as D
from training_wss_min import metrics as M
from training_wss_min.baseline_models import PointNetRegressor
from training_wss_min.evaluate import checkpoint_filename, evaluate_partition, write_reports
from training_wss_min.models import build_model
from training_wss_min.tools.export_wss_postview import (
    NORMALIZED_SCALARS,
    SELF_MAX_SCALARS,
    selfmax_scalar_data,
)


REPO = Path(__file__).resolve().parents[2]
MATRIX_DIR = REPO / "training_wss_min/configs/pointnet_distribution_matrix"


def _case(n: int = 64) -> dict:
    pos = np.stack((np.linspace(-1, 1, n), np.zeros(n), np.zeros(n)), axis=1).astype(np.float32)
    return {
        "cohort": "AG/fast", "case": "SYNTH", "pos": pos,
        "y_raw": np.linspace(1, 10, n, dtype=np.float32),
        "y_norm": np.linspace(0.1, 1, n, dtype=np.float32),
        "abscissa_norm": np.linspace(0, 1, n, dtype=np.float32),
        "local_radius": np.ones(n, dtype=np.float32),
        "curvature": np.zeros(n, dtype=np.float32),
        "coord_scale": np.full(n, 20, dtype=np.float32),
        "radius_gradient": np.zeros(n, dtype=np.float32),
    }


class TestMatrixConfigs(unittest.TestCase):
    def test_matrix_configs_encode_declared_pairs_and_optional_interaction(self):
        configs = {p.stem: C.ExpConfig.from_json(p) for p in MATRIX_DIR.glob("*.json")}
        self.assertEqual(
            set(configs),
            {"e2_global", "e3_global", "e2_case", "e3_case", "e23_global"},
        )
        for name, cfg in configs.items():
            self.assertEqual(cfg.model.name, "pointnet")
            self.assertEqual(tuple(cfg.data.input_features),
                             ("x", "y", "z", "abscissa_norm", "local_radius", "curvature"))
            self.assertEqual(cfg.train.epochs, 400)
            self.assertEqual(cfg.train.selection_rule, "train_loss")
            self.assertEqual(cfg.train.early_stop_patience, 0)
            self.assertEqual(cfg.model.dropout, 0.0)
            self.assertEqual(cfg.data.target_normalization,
                             "case_max" if name.endswith("case") else "global_stats")
        self.assertEqual(configs["e2_global"].model.pointnet_local_channels, (256, 512))
        self.assertEqual(configs["e2_global"].model.pointnet_decoder_channels, (512, 256))
        self.assertEqual(configs["e3_global"].data.sampling, "random")
        self.assertEqual(configs["e3_global"].data.wall_n_points, 5000)
        self.assertTrue(configs["e3_global"].data.resample_each_epoch)
        self.assertEqual(configs["e23_global"].model.pointnet_local_channels, (256, 512))
        self.assertEqual(configs["e23_global"].model.pointnet_decoder_channels, (512, 256))
        self.assertEqual(configs["e23_global"].data.sampling, "random")
        self.assertEqual(configs["e23_global"].data.wall_n_points, 5000)
        self.assertTrue(configs["e23_global"].data.resample_each_epoch)


class TestTeacherExactPointNet(unittest.TestCase):
    def test_exact_channels_and_forward_shape(self):
        cfg = C.ModelConfig(
            name="pointnet", pointnet_local_channels=(256, 512),
            pointnet_decoder_channels=(512, 256), dropout=0.0,
        )
        model = build_model(cfg, in_dim=6)
        linears = [m for m in model.modules() if isinstance(m, torch.nn.Linear)]
        self.assertEqual([(m.in_features, m.out_features) for m in linears],
                         [(6, 256), (256, 512), (1024, 512), (512, 256), (256, 1)])
        x = torch.randn(20, 6)
        batch = torch.tensor([0] * 8 + [1] * 12)
        self.assertEqual(model(torch.randn(20, 3), x, batch).shape, (20,))

    def test_legacy_width_state_dict_shape_is_unchanged(self):
        old = PointNetRegressor(6, width=8, head_hidden=8)
        new = build_model(C.ModelConfig(name="pointnet", width=8, head_hidden=8), 6)
        new.load_state_dict(old.state_dict())


class TestCaseMaxProtocol(unittest.TestCase):
    def test_case_max_normalization_and_metadata(self):
        raw = np.array([0.0, 2.0, 10.0], dtype=np.float32)
        norm, meta = D.normalize_target(raw, {}, "case_max")
        np.testing.assert_allclose(norm, [0.0, 0.2, 1.0])
        self.assertEqual(meta["case_wss_max"], 10.0)
        self.assertFalse(meta["used_as_model_input"])
        self.assertFalse(meta["physical_recovery_enabled"])

    def test_zero_case_max_is_rejected(self):
        with self.assertRaises(ValueError):
            D.normalize_target(np.zeros(4), {}, "case_max")

    def test_random_sampler_is_without_replacement_and_epoch_resampled(self):
        case = _case(100)
        cfg = C.DataConfig(wall_n_points=50, sampling="random", resample_each_epoch=True)
        ds = D.WSSMinDataset([case], cfg, {}, training=True, base_seed=1234)
        ds.set_epoch(0); first = ds[0]["pos"].numpy()
        ds.set_epoch(1); second = ds[0]["pos"].numpy()
        self.assertEqual(len(np.unique(first, axis=0)), 50)
        self.assertFalse(np.array_equal(first, second))
        ds.set_epoch(0)
        np.testing.assert_array_equal(first, ds[0]["pos"].numpy())

    def test_random_sampler_rejects_unequal_point_count_fallback(self):
        cfg = C.DataConfig(wall_n_points=100, sampling="random")
        with self.assertRaisesRegex(ValueError, "same number of points"):
            D.WSSMinDataset([_case(64)], cfg, {}, training=True)


class TestNormalizedMetrics(unittest.TestCase):
    def test_hotspot_distances_are_bbox_normalized(self):
        pos = np.zeros((100, 3)); pos[:, 0] = np.linspace(0, 10, 100)
        true = np.linspace(0, 1, 100)
        pred = true[::-1]
        got = M.hotspot_localization_metrics(true, pred, pos)
        self.assertAlmostEqual(got["bbox_diag"], 10.0)
        self.assertAlmostEqual(got["peak_point_dist_over_bbox"], 1.0)
        self.assertGreater(got["hotspot_centroid_dist_over_bbox"], 0.5)

    def test_distribution_reports_unclipped_out_of_range_predictions(self):
        got = M.distribution_metrics(np.array([0.0, 0.5, 1.0]), np.array([-0.1, 0.5, 1.2]))
        self.assertAlmostEqual(got["pred_below_zero_fraction"], 1 / 3)
        self.assertAlmostEqual(got["pred_above_one_fraction"], 1 / 3)

    def test_checkpoint_names_are_isolated(self):
        self.assertEqual(checkpoint_filename("best"), "ckpt_best.pt")
        self.assertEqual(checkpoint_filename("last"), "ckpt_last.pt")
        with self.assertRaises(ValueError):
            checkpoint_filename("candidate")

    def test_normalized_eval_and_csv_report(self):
        class CoordModel(torch.nn.Module):
            def forward(self, pos, x, batch):
                del pos, batch
                return x[:, 0]

        case = _case(64)
        case["peak_step"] = 0
        case["target_normalization_meta"] = {
            "mode": "case_max", "case_wss_max": 10.0,
            "used_as_model_input": False,
        }
        cfg = C.ExpConfig(
            data=C.DataConfig(
                input_features=("x", "y", "z"), target_normalization="case_max"
            ),
            model=C.ModelConfig(name="pointnet"),
        )
        result = evaluate_partition(CoordModel(), [case], cfg, {}, {}, "cpu")
        self.assertEqual(result["metric_space"], "normalized_target")
        self.assertIn("hotspot_centroid_dist_over_bbox", next(iter(result["per_case"].values()))["hotspot"])
        with tempfile.TemporaryDirectory() as tmp:
            write_reports({"test": result}, Path(tmp))
            header = (Path(tmp) / "per_case_metrics.csv").read_text().splitlines()[0]
            self.assertIn("overall_mae", header)
            self.assertIn("overall_rmse", header)

    def test_postview_normalized_scalar_contract(self):
        required = {
            "true_norm", "pred_norm", "abs_err_norm",
            "true_highrisk", "pred_highrisk", "highrisk_overlap",
            "highrisk_missed", "highrisk_false_positive",
        }
        self.assertTrue(required.issubset(set(NORMALIZED_SCALARS)))

    def test_postview_selfmax_scalar_contract(self):
        true = np.array([1.0, 2.0, 4.0], dtype=np.float64)
        pred = np.array([-1.0, 1.0, 2.0], dtype=np.float64)
        scalars, meta = selfmax_scalar_data(true, pred)
        self.assertEqual(set(scalars), set(SELF_MAX_SCALARS))
        np.testing.assert_allclose(
            scalars["wss_cfd_over_cfd_max"], [0.25, 0.5, 1.0]
        )
        np.testing.assert_allclose(
            scalars["wss_pred_over_pred_max"], [-0.5, 0.5, 1.0]
        )
        self.assertAlmostEqual(meta["wss_pred_max_over_cfd_max"], 0.5)
        self.assertAlmostEqual(meta["pred_below_zero_fraction"], 1.0 / 3.0)

    def test_postview_selfmax_rejects_nonpositive_prediction_max(self):
        with self.assertRaisesRegex(ValueError, "Pred WSS max"):
            selfmax_scalar_data(np.array([1.0, 2.0]), np.array([-2.0, 0.0]))


if __name__ == "__main__":
    unittest.main()
