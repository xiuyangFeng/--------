#!/usr/bin/env python3
"""Compatibility, configuration, and wiring tests for static SA EdgeConv."""

from __future__ import annotations

import unittest

import torch

from training_wss_min.baseline_models import (
    StaticEdgeConvCorrection,
    build_baseline_model,
)
from training_wss_min.config import ExpConfig, ModelConfig, validate_features


COMMON = {
    "name": "pointnetpp",
    "width": 8,
    "sa_ratios": (0.25, 0.25, 0.25),
    "sa_radius": (0.05, 0.1, 0.2),
    "sa_nsample": (8, 8, 8),
    "sa_blocks": (1, 1, 0),
    "sa_center_counts": (8, 4, 2),
}


class StaticEdgeConvTests(unittest.TestCase):
    def test_selected_stages_only_create_selected_modules(self):
        model = build_baseline_model(
            ModelConfig(**COMMON, edgeconv_stages=(1, 2)),
            in_dim=6,
        )
        self.assertIsInstance(model.sa[0].edgeconv, StaticEdgeConvCorrection)
        self.assertIsInstance(model.sa[1].edgeconv, StaticEdgeConvCorrection)
        self.assertIsNone(model.sa[2].edgeconv)

    def test_default_disabled_adds_no_checkpoint_keys(self):
        model = build_baseline_model(ModelConfig(**COMMON), in_dim=6)
        self.assertTrue(all(sa.edgeconv is None for sa in model.sa))
        self.assertFalse(any("edgeconv" in key for key in model.state_dict()))

    def test_edge_message_uses_existing_graph_and_has_finite_gradients(self):
        torch.manual_seed(17)
        module = StaticEdgeConvCorrection(
            in_ch=4, out_ch=8, radius_value=0.2,
            residual_scale_init=1e-3,
        )
        x = torch.randn(5, 4, requires_grad=True)
        centers = torch.tensor([0, 3])
        row = torch.tensor([0, 0, 1, 1])
        col = torch.tensor([0, 1, 3, 4])
        relative = torch.randn(4, 3)
        out = module(x, centers, row, col, relative)
        self.assertEqual(tuple(out.shape), (4, 8))
        self.assertEqual(module.last_edges, 4)
        out.sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue(
            all(
                parameter.grad is not None
                and torch.isfinite(parameter.grad).all()
                for parameter in module.parameters()
            )
        )


class StaticEdgeConvConfigTests(unittest.TestCase):
    def test_stage_numbers_are_one_based_and_bounded(self):
        cfg = ExpConfig(model=ModelConfig(**COMMON, edgeconv_stages=(0,)))
        with self.assertRaisesRegex(ValueError, "1-based"):
            validate_features(cfg)

    def test_duplicate_stages_are_rejected(self):
        cfg = ExpConfig(model=ModelConfig(**COMMON, edgeconv_stages=(1, 1)))
        with self.assertRaisesRegex(ValueError, "duplicates"):
            validate_features(cfg)

    def test_non_pointnetpp_model_is_rejected(self):
        cfg = ExpConfig(
            model=ModelConfig(name="pointnet", edgeconv_stages=(1,))
        )
        with self.assertRaisesRegex(ValueError, "EdgeConv"):
            validate_features(cfg)


if __name__ == "__main__":
    unittest.main()
