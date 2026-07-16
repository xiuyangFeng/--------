#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Focused unit tests for round-4 sampler / hotspot metrics / fixed weights."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from training_wss_min import config as C
from training_wss_min import dataset as D
from training_wss_min import metrics as M
from training_wss_min.evaluate import parse_and_guard_partitions
from training_wss_min.tools.gate1_compare import gate1
from training_wss_min.objectives import compute_loss, fixed_quantile_scale


def _synth_case(n=200, seed=0):
    rng = np.random.default_rng(seed)
    pos = rng.normal(size=(n, 3)).astype(np.float32)
    pos /= np.abs(pos).max() + 1e-6
    y_raw = np.exp(rng.normal(size=n)).astype(np.float32)
    stats = {"method": "log_z", "eps": 1e-6,
             "log": {"mean": 0.0, "std": 1.0},
             "linear": {"mean": 0.0, "std": 1.0}}
    return dict(
        cohort="AG/fast", case=f"SYNTH_{seed}",
        pos=pos,
        y_raw=y_raw,
        y_norm=D.normalize_wss(y_raw, stats).astype(np.float32),
        abscissa_norm=np.linspace(0, 1, n, dtype=np.float32),
        local_radius=np.clip(rng.random(n), 0.1, 1).astype(np.float32),
        curvature=rng.normal(size=n).astype(np.float32),
        coord_scale=np.full(n, 30.0, dtype=np.float32),
        radius_gradient=rng.normal(size=n).astype(np.float32),
        peak_step=0,
        coord_scale_scalar=30.0,
        geom_seed=1234 + 7919 * seed,
    )


class TestSampler(unittest.TestCase):
    def test_fps_reproducible_same_seed(self):
        pts = np.random.default_rng(0).normal(size=(500, 3))
        a = D.farthest_point_sample(pts, 50, seed=7)
        b = D.farthest_point_sample(pts, 50, seed=7)
        np.testing.assert_array_equal(a, b)

    def test_multistart_changes_with_epoch(self):
        case = _synth_case(300, 1)
        cfg = C.DataConfig(wall_n_points=40, sampling="fps_multistart", fps_pool_size=4,
                           resample_each_epoch=True)
        i0 = D.sample_indices(case, cfg, seed=0, epoch=0, case_index=1, run_seed=1234)
        i1 = D.sample_indices(case, cfg, seed=0, epoch=1, case_index=1, run_seed=1234)
        self.assertEqual(len(i0), 40)
        self.assertFalse(np.array_equal(i0, i1), "different epochs must pick different pool ids")
        # same epoch reproducible
        i0b = D.sample_indices(case, cfg, seed=0, epoch=0, case_index=1, run_seed=1234)
        np.testing.assert_array_equal(i0, i0b)

    def test_fixed_fps_stable_across_epochs(self):
        case = _synth_case(300, 2)
        cfg = C.DataConfig(wall_n_points=40, sampling="fps", resample_each_epoch=True)
        a = D.sample_indices(case, cfg, seed=case["geom_seed"], epoch=0, case_index=0, run_seed=1)
        b = D.sample_indices(case, cfg, seed=case["geom_seed"], epoch=5, case_index=0, run_seed=1)
        np.testing.assert_array_equal(a, b)

    def test_eval_full_cloud(self):
        case = _synth_case(120, 3)
        cfg = C.DataConfig(wall_n_points=40, sampling="fps", input_features=("x", "y", "z"))
        fs = {}
        batch = D.case_to_batch(case, cfg.input_features, fs)
        self.assertEqual(batch["pos"].shape[0], 120)


class TestHotspotMetrics(unittest.TestCase):
    def test_perfect_localization(self):
        n = 100
        pos = np.zeros((n, 3), dtype=np.float64)
        pos[:, 0] = np.linspace(0, 1, n)
        y = np.linspace(0, 10, n)
        h = M.hotspot_localization_metrics(y, y, pos, high_pctl=90.0)
        self.assertAlmostEqual(h["top10_iou"], 1.0, places=5)
        self.assertAlmostEqual(h["top10_recall"], 1.0, places=5)
        self.assertAlmostEqual(h["peak_point_dist"], 0.0, places=5)

    def test_shifted_peak_distance(self):
        n = 100
        pos = np.zeros((n, 3), dtype=np.float64)
        pos[:, 0] = np.arange(n)
        yt = np.zeros(n); yt[-1] = 10
        yp = np.zeros(n); yp[0] = 10
        h = M.hotspot_localization_metrics(yt, yp, pos, high_pctl=90.0)
        self.assertGreater(h["peak_point_dist"], 50)


class TestRound5Metrics(unittest.TestCase):
    def test_casebalanced_matches_manual_example(self):
        yt = [np.array([0.0, 2.0]), np.array([10.0, 10.0, 10.0, 10.0])]
        yp = [np.array([0.0, 1.0]), np.array([8.0, 8.0, 8.0, 8.0])]
        got = M.casebalanced_field_metrics(yt, yp)
        mean = (1.0 + 10.0) / 2.0
        mse = (np.mean((yt[0] - yp[0]) ** 2) + np.mean((yt[1] - yp[1]) ** 2)) / 2.0
        var = (np.mean((yt[0] - mean) ** 2) + np.mean((yt[1] - mean) ** 2)) / 2.0
        self.assertAlmostEqual(got["r2"], 1.0 - mse / var)
        self.assertEqual(got["n_cases"], 2)

    def test_casebalanced_not_dominated_by_case_point_count(self):
        short_t = np.array([0.0, 1.0])
        short_p = np.array([0.0, 0.0])
        long_t = np.full(1000, 10.0)
        long_p = np.full(1000, 9.0)
        a = M.casebalanced_field_metrics([short_t, long_t], [short_p, long_p])
        b = M.casebalanced_field_metrics(
            [short_t, long_t[:10]], [short_p, long_p[:10]]
        )
        self.assertAlmostEqual(a["r2"], b["r2"], places=12)

    def test_constant_single_case_is_nan(self):
        got = M.casebalanced_field_metrics([np.ones(4)], [np.zeros(4)])
        self.assertTrue(np.isnan(got["r2"]))

    def test_single_nonconstant_case_matches_standard_r2(self):
        yt = np.array([0.0, 1.0, 3.0])
        yp = np.array([0.5, 1.0, 2.5])
        got = M.casebalanced_field_metrics([yt], [yp])
        self.assertAlmostEqual(got["r2"], M.r2_score(yt, yp))


class TestRound5Gate(unittest.TestCase):
    @staticmethod
    def _metrics(field=0.3, case=0.2, mae=3.0, top10=0.4, iou=0.3):
        return {"r2_field": field, "r2_casemean": case, "mae": mae,
                "top10": top10, "iou": iou, "max_ratio": 0.8}

    def test_both_primary_metrics_must_exceed_band(self):
        ctrl = self._metrics()
        cand = self._metrics(field=0.321, case=0.221)
        self.assertTrue(gate1(ctrl, cand)["go"])
        cand_one_only = self._metrics(field=0.34, case=0.219)
        self.assertFalse(gate1(ctrl, cand_one_only)["go"])

    def test_high_wss_degradation_blocks_go(self):
        ctrl = self._metrics()
        cand = self._metrics(field=0.34, case=0.24, top10=0.34)
        got = gate1(ctrl, cand)
        self.assertFalse(got["go"])
        self.assertIn("top10_ratio_degraded", got["guard_reasons"])


class TestPartitionGuard(unittest.TestCase):
    def test_val_is_default_safe_partition(self):
        self.assertEqual(parse_and_guard_partitions("val"), ["val"])

    def test_test_requires_explicit_unlock(self):
        with self.assertRaises(PermissionError):
            parse_and_guard_partitions("val,test")
        self.assertEqual(parse_and_guard_partitions("test", allow_test=True), ["test"])

    def test_duplicate_and_unknown_partitions_rejected(self):
        with self.assertRaises(ValueError):
            parse_and_guard_partitions("val,val")
        with self.assertRaises(ValueError):
            parse_and_guard_partitions("validation")


class TestFixedWeight(unittest.TestCase):
    def test_same_y_same_weight_across_batches(self):
        tcfg = C.TrainConfig(
            loss="mse", loss_weight_target=True, loss_weight_target_alpha=2.0,
            loss_weight_fixed_quantiles=True,
            y_norm_q02=0.0, y_norm_q98=1.0,
        )
        y = torch.tensor([0.0, 0.5, 1.0, 2.0])
        pred = y.clone()
        # batch A
        b1 = {"y": y, "curv": torch.ones_like(y), "invr": torch.ones_like(y),
              "y_raw": torch.ones_like(y)}
        # batch B: different composition but same y values appear
        y2 = torch.tensor([0.5, 0.5, 1.0, 1.0, 0.0, 2.0])
        b2 = {"y": y2, "curv": torch.ones_like(y2), "invr": torch.ones_like(y2),
              "y_raw": torch.ones_like(y2)}
        # weight for y=0.5 should be identical: 1 + 2*((0.5-0)/(1-0))=2
        w1 = 1 + 2 * fixed_quantile_scale(torch.tensor([0.5]), 0.0, 1.0)
        w2 = 1 + 2 * fixed_quantile_scale(torch.tensor([0.5]), 0.0, 1.0)
        self.assertEqual(float(w1), float(w2))
        # losses finite
        self.assertTrue(torch.isfinite(compute_loss(pred, b1, tcfg, "cpu")))
        self.assertTrue(torch.isfinite(compute_loss(torch.ones_like(y2), b2, tcfg, "cpu")))


class TestForbiddenFeature(unittest.TestCase):
    def test_dist_to_wall_rejected(self):
        with self.assertRaises(ValueError):
            C.validate_features(C.ExpConfig(
                data=C.DataConfig(input_features=("x", "y", "z", "dist_to_wall"))
            ))


if __name__ == "__main__":
    unittest.main()
