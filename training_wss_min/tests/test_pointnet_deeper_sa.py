"""Protocol and geometry tests for E4-DEEP and the three-SA foundation."""

from __future__ import annotations

import unittest
from pathlib import Path

import torch

from training_wss_min import config as C
from training_wss_min.baseline_models import sample_and_group
from training_wss_min.models import build_model


REPO = Path(__file__).resolve().parents[2]
E2_CONFIG = REPO / "training_wss_min/configs/pointnet_distribution_matrix/e2_global.json"
E4_CONFIG = REPO / "training_wss_min/configs/pointnet_deeper/e4_deep_global.json"
SA3_CONFIG = REPO / "training_wss_min/configs/pointnetpp_sa_foundation/sa3_xyzgeom.json"


class TestDeeperConfig(unittest.TestCase):
    def test_e4_differs_from_e2_only_in_identity_and_channels(self):
        e2 = C.ExpConfig.from_json(E2_CONFIG)
        e4 = C.ExpConfig.from_json(E4_CONFIG)
        self.assertEqual(e4.data, e2.data)
        self.assertEqual(e4.train, e2.train)
        self.assertEqual(e4.model.name, "pointnet")
        self.assertEqual(e4.model.pointnet_local_channels, (64, 128, 256, 512))
        self.assertEqual(e4.model.pointnet_decoder_channels, (512, 256, 128, 64))
        for field in e2.model.__dataclass_fields__:
            if field not in {"pointnet_local_channels", "pointnet_decoder_channels"}:
                self.assertEqual(getattr(e4.model, field), getattr(e2.model, field), field)

    def test_exact_linear_stack_parameter_count_and_backward(self):
        cfg = C.ExpConfig.from_json(E4_CONFIG)
        model = build_model(cfg.model, C.input_dim(cfg))
        linears = [m for m in model.modules() if isinstance(m, torch.nn.Linear)]
        self.assertEqual(
            [(m.in_features, m.out_features) for m in linears],
            [(6, 64), (64, 128), (128, 256), (256, 512),
             (1024, 512), (512, 256), (256, 128), (128, 64), (64, 1)],
        )
        self.assertEqual(sum(p.numel() for p in model.parameters()), 874561)
        x = torch.randn(35, 6)
        batch = torch.tensor([0] * 17 + [1] * 18)
        out = model(torch.randn(35, 3), x, batch)
        self.assertEqual(out.shape, (35,))
        out.square().mean().backward()


class TestThreeSAFoundation(unittest.TestCase):
    def test_config_is_visualization_only_three_stage_contract(self):
        cfg = C.ExpConfig.from_json(SA3_CONFIG)
        self.assertEqual(cfg.model.name, "pointnetpp")
        self.assertEqual(cfg.data.wall_n_points, 2000)
        self.assertEqual(cfg.data.sampling, "fps")
        self.assertEqual(cfg.model.sa_ratios, (0.25, 0.25, 0.25))
        self.assertEqual(cfg.model.sa_radius, (0.05, 0.1, 0.2))
        self.assertEqual(cfg.model.sa_nsample, (16, 16, 16))
        self.assertEqual(cfg.train.epochs, 0)
        self.assertGreater(sum(p.numel() for p in build_model(cfg.model, 6).parameters()), 0)

    def test_three_stage_trace_is_deterministic_bounded_and_traceable(self):
        torch.manual_seed(7)
        initial = torch.rand(2000, 3) * 2 - 1

        def trace():
            pos = initial
            batch = torch.zeros(len(pos), dtype=torch.long)
            lineage = torch.arange(len(pos))
            records = []
            for ratio, radius_value in zip((0.25, 0.25, 0.25), (0.05, 0.1, 0.2)):
                idx, pos_q, batch_q, row, col = sample_and_group(
                    pos, batch, ratio=ratio, radius_value=radius_value,
                    nsample=16, random_start=False,
                )
                distance = torch.linalg.norm(pos[col] - pos_q[row], dim=1)
                sizes = torch.bincount(row, minlength=len(pos_q))
                self.assertTrue(torch.all(distance <= radius_value + 1e-6))
                self.assertLessEqual(int(sizes.max()), 16)
                lineage = lineage[idx]
                records.append((idx.clone(), row.clone(), col.clone(), lineage.clone()))
                pos, batch = pos_q, batch_q
            return records

        first = trace()
        second = trace()
        self.assertEqual([len(r[0]) for r in first], [500, 125, 32])
        for a, b in zip(first, second):
            for left, right in zip(a, b):
                torch.testing.assert_close(left, right)
        self.assertEqual(len(first[-1][3]), 32)
        self.assertTrue(torch.all((first[-1][3] >= 0) & (first[-1][3] < 2000)))


if __name__ == "__main__":
    unittest.main()
