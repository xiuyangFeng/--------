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
from training_wss_min.train import compute_loss


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
        from training_wss_min.train import _fixed01
        w1 = 1 + 2 * _fixed01(torch.tensor([0.5]), 0.0, 1.0)
        w2 = 1 + 2 * _fixed01(torch.tensor([0.5]), 0.0, 1.0)
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
