#!/usr/bin/env python3

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from training_wss_min import config as C
from training_wss_min.baseline_models import _group
from training_wss_min.dataset import _coords_sha256, load_fps_cache


def group_sets(row: torch.Tensor, col: torch.Tensor, n_centers: int) -> list[set[int]]:
    row, col = row.cpu(), col.cpu()
    return [set(col[row == center].tolist()) for center in range(n_centers)]


class SAGroupingMatrixTests(unittest.TestCase):
    def setUp(self):
        gen = torch.Generator().manual_seed(17)
        self.pos = torch.rand((120, 3), generator=gen)
        self.center_idx = torch.arange(0, 120, 10)
        self.centers = self.pos[self.center_idx]
        self.batch = torch.zeros(120, dtype=torch.long)
        self.center_batch = torch.zeros(len(self.centers), dtype=torch.long)

    def call(self, mode: str, k: int):
        return _group(
            self.pos, self.batch, self.centers, self.center_batch,
            radius_value=2.0, nsample=k, grouping=mode,
            adaptive_candidate_centers=8, overlap_cap=1.0 / 3.0,
        )

    def test_raw_knn_has_exact_k_members_per_center(self):
        row, col = self.call("knn", 8)
        self.assertEqual(len(row), 8 * len(self.centers))
        self.assertTrue(torch.equal(torch.bincount(row), torch.full((12,), 8)))
        self.assertTrue(torch.all(self.batch[col] == self.center_batch[row]))

    def test_knn_cover_repairs_all_uncovered_sources(self):
        row, col = self.call("knn_cover", 8)
        self.assertEqual(torch.unique(col).numel(), len(self.pos))
        self.assertTrue(torch.all(self.batch[col] == self.center_batch[row]))

    def test_adaptive_cover_guarantees_coverage_and_overlap_cap(self):
        row, col = self.call("adaptive_cover", 16)
        groups = group_sets(row, col, len(self.centers))
        self.assertEqual(len(set().union(*groups)), len(self.pos))
        memberships = torch.bincount(col, minlength=len(self.pos))
        self.assertGreaterEqual(int(memberships.min()), 1)
        self.assertLessEqual(int(memberships.max()), 2)
        for first in range(len(groups)):
            for second in range(first + 1, len(groups)):
                shared = len(groups[first] & groups[second])
                if shared:
                    overlap = shared / min(len(groups[first]), len(groups[second]))
                    self.assertLessEqual(overlap, 1.0 / 3.0 + 1e-12)

    def test_adaptive_cover_is_deterministic(self):
        first = self.call("adaptive_cover", 16)
        second = self.call("adaptive_cover", 16)
        self.assertTrue(torch.equal(first[0], second[0]))
        self.assertTrue(torch.equal(first[1], second[1]))

    def test_config_accepts_per_stage_grouping_and_rejects_loose_cap(self):
        cfg = C.ExpConfig.from_dict({
            "model": {
                "name": "pointnetpp", "sa_radius": [0.05, 0.1, 0.2],
                "sa_nsample": [16, 16, 16], "sa_ratios": [0.25, 0.25, 0.25],
                "sa_blocks": [1, 1, 1],
                "sa_center_counts": [500, 125, 32],
                "sa_grouping": ["adaptive_cover", "ball", "ball"],
                "sa_overlap_cap": 1.0 / 3.0,
            },
        })
        self.assertEqual(cfg.model.sa_grouping[0], "adaptive_cover")
        payload = cfg.to_dict()
        payload["model"]["sa_overlap_cap"] = 0.34
        with self.assertRaisesRegex(ValueError, "sa_overlap_cap"):
            C.ExpConfig.from_dict(payload)


class FPSCacheTests(unittest.TestCase):
    def test_disk_cache_contract(self):
        rng = np.random.default_rng(3)
        pos = rng.normal(size=(20, 3)).astype(np.float32)
        case = {"pos": pos, "unit_id": "AG/fast/CASE"}
        with tempfile.TemporaryDirectory() as tmp:
            cfg = C.DataConfig(
                fps_cache_dir=tmp, fps_cache_required=True, fps_pool_size=2,
            )
            path = Path(tmp) / "AG/fast/CASE/fps_k5_pool2.npz"
            path.parent.mkdir(parents=True)
            np.savez_compressed(
                path, fixed=np.arange(5), pool=np.stack([np.arange(5), np.arange(5, 10)]),
                n_total=np.int64(20), k=np.int64(5), pool_size=np.int64(2),
                coords_sha256=np.asarray(_coords_sha256(pos)),
            )
            cached = load_fps_cache(case, cfg, 5)
            self.assertTrue(np.array_equal(cached["fixed"], np.arange(5)))
            self.assertEqual(cached["pool"].shape, (2, 5))


if __name__ == "__main__":
    unittest.main()
