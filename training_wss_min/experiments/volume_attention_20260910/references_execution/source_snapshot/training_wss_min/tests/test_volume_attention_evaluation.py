"""Regression tests for joint row/channel routing and saved long-wave diagnostics."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from training_wss_min import config as C, dataset as D, metrics as M
from training_wss_min import evaluate as E
from training_wss_min.tools import volume_attention_diagnostics as V


def make_case(n=120, n_wall=30):
    rng = np.random.default_rng(13)
    raw = rng.normal(size=(n, 4))
    stats = {"method": "linear", "linear": {"mean": 20., "std": 100.},
             "velocity": {"mean": [0.1, -0.2, .3], "std": [.5, .6, .7]},
             "speed": {"mean": .8, "std": .4}}
    mean = np.array([.1, -.2, .3, 20.])
    std = np.array([.5, .6, .7, 100.])
    raw = raw * std + mean
    raw[:n_wall, :3] = 10000.  # Deliberately invalid wall velocities must never affect V metrics.
    case = {"unit_id": "AG/fast/SYNTH", "cohort": "AG/fast", "case": "SYNTH",
            "bundle_path": "/unused/AG/fast/SYNTH/bundle.npz", "n_wall": n_wall,
            "pos": rng.normal(size=(n, 3)).astype(np.float32),
            "local_radius": rng.uniform(.2, 2., size=n).astype(np.float32),
            "point_kind": np.r_[np.zeros(n_wall, dtype=np.int8), np.ones(n-n_wall, dtype=np.int8)],
            "support_pool": np.arange(n_wall), "query_pool": np.arange(n),
            "y_raw": raw, "y_norm": (raw - mean) / std}
    pred = case["y_norm"] + rng.normal(scale=.1, size=(n, 4))
    return case, pred, stats


class JointEvaluationTest(unittest.TestCase):
    def test_joint_matches_independent_task_metrics(self):
        case, pred, stats = make_case()
        cfg = C.ExpConfig()
        cfg.data.target = "velocity_pressure"
        joint = E._evaluate_joint([case], [pred], cfg, stats, save_dir=None, make_plots=False)
        model = mock.Mock()
        pressure_case = dict(case, y_raw=case["y_raw"][:, 3], y_norm=case["y_norm"][:, 3])
        cfg.data.target = "pressure_mixed"
        with mock.patch.object(E, "predict_case_norm", return_value=pred[:, 3]):
            pressure = E.evaluate_partition(model, [pressure_case], cfg, {}, stats, "cpu")
        velocity_case = dict(case, y_raw=case["y_raw"][:, :3], y_norm=case["y_norm"][:, :3],
                             query_pool=np.arange(case["n_wall"], len(pred)))
        cfg.data.target = "velocity"
        with mock.patch.object(E, "predict_case_norm", return_value=pred[case["n_wall"]:, :3]):
            velocity = E.evaluate_partition(model, [velocity_case], cfg, {}, stats, "cpu")
        for task, expected in (("pressure", pressure), ("velocity", velocity)):
            for space in ("field", "field_casebalanced"):
                for metric in ("r2", "mae", "rmse"):
                    self.assertAlmostEqual(joint["tasks"][task][space][metric], expected[space][metric], places=12)
        self.assertEqual(joint["tasks"]["pressure"]["query_groups"]["wall"]["n_points"], case["n_wall"])
        self.assertEqual(joint["tasks"]["velocity"]["field"]["n"], len(pred) - case["n_wall"])
        with tempfile.TemporaryDirectory() as temp:
            E.write_reports({"test": joint}, Path(temp))
            root = json.loads((Path(temp) / "metrics.json").read_text())
            self.assertEqual(root["test"]["tasks"]["pressure"]["metrics_path"], "pressure/metrics.json")
            self.assertTrue((Path(temp) / "velocity" / "per_case_metrics.csv").is_file())

    def test_full_union_is_encoded_once_and_chunked_without_losing_channels(self):
        case, _, _ = make_case(n=23, n_wall=5)
        cfg = C.ExpConfig()
        cfg.data.target = "velocity_pressure"
        cfg.eval.fixed_support = True
        cfg.eval.query_chunk_size = 7
        cfg.data.input_features = ("x", "y", "z")
        model = mock.Mock()
        model.encode_support.return_value = "encoded"
        model.decode_query.side_effect = lambda encoded, pos, x, batch: torch.column_stack([pos, pos.sum(1)])
        with mock.patch.object(D, "sample_support_indices", return_value=np.arange(5)), \
             mock.patch.object(D, "build_features", side_effect=lambda c, idx, *a, **k: c["pos"][idx]):
            pred = E.predict_case_norm(model, case, cfg.data.input_features, {}, "cpu", cfg,
                                       return_all_channels=True)
        self.assertEqual(model.encode_support.call_count, 1)
        self.assertEqual(model.decode_query.call_count, 4)
        self.assertEqual(pred.shape, (23, 4))
        np.testing.assert_array_equal(pred[:, :3], case["pos"])
        np.testing.assert_allclose(pred[:, 3], case["pos"].sum(1))

    def test_prediction_archive_preserves_joint_channels_and_case_rows(self):
        case, pred, stats = make_case()
        with tempfile.TemporaryDirectory() as temp:
            entry = E._save_volume_prediction(case, pred, "velocity_pressure", stats, Path(temp))
            with np.load(Path(temp) / entry["file"]) as saved:
                np.testing.assert_array_equal(saved["query_idx"], case["query_pool"])
                np.testing.assert_array_equal(saved["point_kind"], case["point_kind"])
                np.testing.assert_array_equal(saved["true_raw"], case["y_raw"].astype(np.float32))
                self.assertEqual(saved["pred_raw"].shape, (120, 4))
                np.testing.assert_allclose(saved["pred_raw"][:, 3], pred[:, 3] * 100 + 20, rtol=1e-6)
                self.assertEqual(saved["unit_id"].item(), case["unit_id"])

    def test_joint_cli_writes_task_reports_and_prediction_manifest(self):
        case, pred, stats = make_case()
        cfg = C.ExpConfig()
        cfg.data.target = "velocity_pressure"
        cfg.eval.fixed_support = True
        with tempfile.TemporaryDirectory() as temp:
            argv = ["evaluate", "--run-dir", temp, "--partitions", "test", "--allow-test",
                    "--save-predictions", "--no-plots", "--checkpoint", "last"]
            with mock.patch("sys.argv", argv), \
                 mock.patch.object(E, "load_model_from_run", return_value=(cfg, {}, torch.nn.Linear(1, 4), {})), \
                 mock.patch.object(E, "load_wss_stats_for_run", return_value=stats), \
                 mock.patch.object(D, "load_partition", return_value=[case]), \
                 mock.patch.object(E, "predict_case_norm", return_value=pred) as predict, \
                 mock.patch.object(torch.cuda, "is_available", return_value=False):
                E.main()
            self.assertTrue(predict.call_args.kwargs["return_all_channels"])
            evaluation = Path(temp) / "eval" / "ckpt_last"
            manifest = json.loads((evaluation / "predictions" / "manifest.json").read_text())
            self.assertEqual(manifest["checkpoint"], "last")
            entry = manifest["partitions"]["test"][0]
            self.assertTrue((evaluation / "predictions" / entry["file"]).is_file())
            velocity = json.loads((evaluation / "velocity" / "metrics.json").read_text())
            self.assertEqual(velocity["test"]["field"]["n"], 90)
            self.assertTrue(velocity["test"]["efficiency"]["shared_joint_inference"])


class LongwaveDiagnosticTest(unittest.TestCase):
    def test_branches_do_not_mix_and_constant_bias_is_removed(self):
        segment = np.array([0, 0, 1, 1])
        s = np.array([1., 6., 1., 6.])  # Same s coordinates, different physical branches.
        true = np.zeros(4)
        pred = np.array([2., 2., -2., -2.])
        rows, result = V.branch_profiles(segment, s, np.ones(4), true, pred)
        for row in rows:
            self.assertAlmostEqual(row["residual_sigma40_mm"], 2 if row["segment_id"] == 0 else -2)
        self.assertEqual(result["sigmas_mm"]["40"]["residual_mse"], 4)
        _, constant = V.branch_profiles(segment, s, np.ones(4), true, np.full(4, 3.))
        self.assertAlmostEqual(constant["case_bias"], 3)
        for sigma in V.SIGMAS_MM:
            self.assertAlmostEqual(constant["sigmas_mm"][str(sigma)]["residual_mse"], 9)
            self.assertAlmostEqual(constant["sigmas_mm"][str(sigma)]["debiased_residual_mse"], 0)

    def test_bins_use_cell_volume_and_smoothing_preserves_boundary_constants(self):
        rows, result = V.branch_profiles([0, 0, 0], [1., 2., 21.], [1., 3., 2.], [0., 4., 7.], [2., 6., 9.])
        self.assertEqual(rows[0]["true"], 3)
        self.assertEqual(rows[0]["volume_m3"], 4)
        self.assertEqual(result["n_bins"], 2)  # Missing bins are not padded with zeros.
        self.assertAlmostEqual(rows[-1]["residual_sigma20_mm"], 2)

    def test_streaming_scalar_metrics_match_existing_casebalanced_definition(self):
        true = [np.array([1., 2., 4.]), np.array([2., 5., 7., 9., 12.])]
        pred = [v * .8 + .3 for v in true]
        actual = V.summarize_scalar([V.scalar_moments(t, p) for t, p in zip(true, pred)])
        self.assertAlmostEqual(actual["r2_casebalanced"], M.casebalanced_field_metrics(true, pred)["r2"])
        self.assertAlmostEqual(actual["r2_pooled"], M.r2_score(np.concatenate(true), np.concatenate(pred)))

    def test_saved_joint_diagnostics_exclude_wall_and_write_both_tasks(self):
        case, pred, stats = make_case(n=40, n_wall=10)
        nvol = 30
        geometry = {"segment": np.repeat([0, 1], 15), "s_mm": np.tile(np.arange(15.) * 5, 2),
                    "volume": np.ones(nvol), "axial": np.tile([1., 0., 0.], (nvol, 1)),
                    "radial": np.tile([0., 1., 0.], (nvol, 1)), "circ": np.tile([0., 0., 1.], (nvol, 1)),
                    "valid": np.ones(nvol, dtype=bool)}
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            evaluation = run / "eval" / "ckpt_last"
            archive = evaluation / "predictions" / "test"
            entry = E._save_volume_prediction(case, pred, "velocity_pressure", stats, archive)
            manifest = {"target": "velocity_pressure", "partitions": {"test": [{**entry, "file": "test/" + entry["file"]}]}}
            (archive.parent / "manifest.json").write_text(json.dumps(manifest))
            cfg = C.ExpConfig()
            joint = E._evaluate_joint([case], [pred], cfg, stats, save_dir=None, make_plots=False)
            E.write_reports({"test": joint}, evaluation)
            with mock.patch.object(V, "local_geometry", return_value=geometry):
                result = V.diagnose(run, "last", make_plots=False)
            self.assertEqual(set(result["tasks"]), {"pressure", "velocity"})
            self.assertEqual(result["tasks"]["velocity"]["components"]["u"]["n_points"], nvol)
            self.assertEqual(result["tasks"]["pressure"]["longwave"]["summary"]["n_cases"], 1)
            self.assertTrue((evaluation / "diagnostics" / "summary.json").is_file())
            self.assertEqual(len(list((evaluation / "diagnostics" / "profiles").glob("*.csv"))), 2)
            np.testing.assert_allclose(result["tasks"]["velocity"]["vector_rmse_m_s"],
                                       joint["tasks"]["velocity"]["vector"]["vector_rmse_m_s"], rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
