from __future__ import annotations

import unittest

import torch

from training_wss_min.baseline_models import LocalInvResBlock
from training_wss_min.config import ExpConfig, ModelConfig
from training_wss_min.models import build_model


class PointNeXtRCoreTests(unittest.TestCase):
    def test_zero_residual_scale_is_exact_identity(self):
        torch.manual_seed(4)
        pos = torch.rand(18, 3)
        x = torch.randn(18, 8)
        batch = torch.zeros(18, dtype=torch.long)
        block = LocalInvResBlock(
            8, radius_value=2.0, nsample=18, expansion=2,
            residual_scale_init=0.0,
        ).eval()
        out = block(pos, x, batch)
        torch.testing.assert_close(out, x, rtol=0.0, atol=0.0)

    def test_pointnetpp_consumes_sa_blocks(self):
        common = dict(
            name="pointnetpp", width=8,
            sa_ratios=(0.25, 0.25, 0.25),
            sa_radius=(0.4, 0.8, 1.6),
            sa_nsample=(4, 4, 2),
            sa_center_counts=(8, 4, 2),
            sa_grouping=("knn_cover", "ball", "ball"),
            head_hidden=8,
        )
        base = build_model(ModelConfig(**common, sa_blocks=(0, 0, 0)), 6)
        pnxr = build_model(
            ModelConfig(
                **common, sa_blocks=(1, 1, 0),
                invres_expansion=2, residual_scale_init=1e-3,
            ),
            6,
        )
        self.assertEqual([len(sa.blocks) for sa in base.sa], [0, 0, 0])
        self.assertEqual([len(sa.blocks) for sa in pnxr.sa], [1, 1, 0])
        self.assertGreater(
            sum(p.numel() for p in pnxr.parameters()),
            sum(p.numel() for p in base.parameters()),
        )
        self.assertAlmostEqual(float(pnxr.sa[0].blocks[0].gamma_local), 1e-3, places=7)

    def test_pointnext_r_support_query_forward_backward(self):
        torch.manual_seed(5)
        cfg = ModelConfig(
            name="pointnetpp", width=8,
            sa_ratios=(0.25, 0.25, 0.25),
            sa_radius=(0.4, 0.8, 1.6),
            sa_nsample=(4, 4, 2),
            sa_blocks=(1, 1, 0),
            sa_center_counts=(8, 4, 2),
            sa_grouping=("knn_cover", "ball", "ball"),
            head_hidden=8,
            invres_expansion=2,
            residual_scale_init=1e-3,
        )
        model = build_model(cfg, 6).train()
        pos = torch.rand(48, 3)
        x = torch.rand(48, 6)
        batch = torch.cat([
            torch.zeros(24, dtype=torch.long),
            torch.ones(24, dtype=torch.long),
        ])
        out = model.forward_support_query(
            pos, x, batch, pos, x, batch,
            unit_ids=["case-a", "case-b"], epoch=0, global_seed=1234,
            evaluation=False,
        )
        self.assertEqual(tuple(out.shape), (48,))
        out.square().mean().backward()
        self.assertIsNotNone(model.sa[0].blocks[0].gamma_local.grad)

    def test_pointnetpp_rejects_sa_block_length_drift(self):
        with self.assertRaisesRegex(ValueError, "sa_blocks"):
            ExpConfig.from_dict({
                "data": {"input_features": ["x", "y", "z"]},
                "model": {
                    "name": "pointnetpp",
                    "sa_radius": [0.1, 0.2, 0.4],
                    "sa_ratios": [0.25, 0.25, 0.25],
                    "sa_nsample": [8, 8, 8],
                    "sa_blocks": [1, 0],
                },
            })


if __name__ == "__main__":
    unittest.main()
