#!/usr/bin/env python3
"""Focused compatibility and wiring tests for local SA transformers."""

from __future__ import annotations

import unittest

import torch

from training_wss_min.baseline_models import (
    CoarseGlobalBlock,
    LocalNeighborhoodTransformer,
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


class LocalNeighborhoodTransformerTests(unittest.TestCase):
    def test_zero_residual_scale_reproduces_local_max_pool(self):
        torch.manual_seed(7)
        tokens = torch.randn(9, 8, requires_grad=True)
        groups = torch.tensor([2, 0, 1, 0, 2, 1, 0, 2, 1])
        module = LocalNeighborhoodTransformer(
            8, heads=2, ffn_ratio=2, residual_scale_init=0.0
        ).eval()

        got = module(tokens, groups, n_groups=3)
        expected = torch.stack(
            [tokens[groups == group].max(dim=0).values for group in range(3)]
        )
        torch.testing.assert_close(got, expected, rtol=0.0, atol=0.0)
        got.sum().backward()
        self.assertTrue(torch.isfinite(tokens.grad).all())
        self.assertEqual(module.last_group_sizes.tolist(), [3, 3, 3])

    def test_selected_stages_and_global_block_are_independent(self):
        model = build_baseline_model(
            ModelConfig(
                **COMMON,
                local_transformer_stages=(1, 2),
                local_transformer_heads=4,
                coarse_attention=True,
            ),
            in_dim=6,
        )
        self.assertIsInstance(
            model.sa[0].local_transformer, LocalNeighborhoodTransformer
        )
        self.assertIsInstance(
            model.sa[1].local_transformer, LocalNeighborhoodTransformer
        )
        self.assertIsNone(model.sa[2].local_transformer)
        self.assertIsInstance(model.coarse_global, CoarseGlobalBlock)

    def test_default_disabled_adds_no_checkpoint_keys(self):
        model = build_baseline_model(ModelConfig(**COMMON), in_dim=6)
        self.assertTrue(all(sa.local_transformer is None for sa in model.sa))
        self.assertFalse(
            any("local_transformer" in key for key in model.state_dict())
        )


class LocalTransformerConfigTests(unittest.TestCase):
    def test_stage_numbers_are_one_based_and_bounded(self):
        cfg = ExpConfig(
            model=ModelConfig(**COMMON, local_transformer_stages=(0,))
        )
        with self.assertRaisesRegex(ValueError, "1-based"):
            validate_features(cfg)

    def test_selected_stage_channels_must_divide_heads(self):
        cfg = ExpConfig(
            model=ModelConfig(
                **COMMON, local_transformer_stages=(1,),
                local_transformer_heads=3,
            )
        )
        with self.assertRaisesRegex(ValueError, "divisible"):
            validate_features(cfg)


if __name__ == "__main__":
    unittest.main()
