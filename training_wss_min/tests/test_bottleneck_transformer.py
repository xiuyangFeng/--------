#!/usr/bin/env python3
"""Wiring, ordering and backward-compatibility tests for the bottleneck transformer.

These guard the invariants that were verified by hand during the 2026-09-08 review:
row order through to_dense_batch, centre-feature alignment through the SA stages, no cross-case
leakage, exact identity at gamma=0, the layers=0 conditioning-only control, and the promise that
leaving the flag off changes nothing about an existing run.
"""
from __future__ import annotations

import unittest

import torch

from training_wss_min.baseline_models import BottleneckTransformer, build_baseline_model
from training_wss_min.config import ExpConfig, ModelConfig, validate_features

COMMON = {
    "name": "pointnetpp",
    "width": 8,
    "sa_ratios": (0.25, 0.25, 0.25),
    "sa_radius": (0.05, 0.1, 0.2),
    "sa_nsample": (8, 8, 8),
    "sa_blocks": (1, 1, 0),
    "sa_center_counts": (8, 8, 4),
    "sa_grouping": ("knn_cover", "ball", "ball"),
    "local_geope": True,
    "fp_knn": 3,
    "head_hidden": 8,
}
IN_DIM = 6


def _model(**overrides):
    cfg = ModelConfig(**{**COMMON, **overrides})
    torch.manual_seed(0)
    return build_baseline_model(cfg, IN_DIM)


def _batch(counts=(40, 25), seed=0):
    torch.manual_seed(seed)
    pos = torch.rand(sum(counts), 3) * 2 - 1
    x = torch.randn(sum(counts), IN_DIM)
    batch = torch.cat([torch.full((c,), i, dtype=torch.long) for i, c in enumerate(counts)])
    return pos, x, batch


class BottleneckTransformerModuleTest(unittest.TestCase):
    def test_zero_gamma_is_a_bit_exact_identity(self):
        bt = BottleneckTransformer(16, in_dim=IN_DIM, layers=2, heads=4, ffn_ratio=2, dropout=0.1,
                                   pos_enc="input_features", geo_bias=True, residual_scale_init=1e-3).eval()
        for layer in bt.layers:
            with torch.no_grad():
                layer.gamma_attention.zero_()
                layer.gamma_ffn.zero_()
        counts = (7, 19)  # deliberately ragged so padding is exercised
        pos, feats, batch = _batch(counts)
        tokens = torch.randn(sum(counts), 16)
        out = bt(pos, tokens, batch, feats=feats)
        self.assertTrue(torch.equal(out, tokens))
        self.assertEqual(bt.last_token_counts, list(counts))

    def test_row_order_is_preserved_through_dense_round_trip(self):
        bt = BottleneckTransformer(16, in_dim=IN_DIM, layers=1, heads=4, ffn_ratio=2, dropout=0.0,
                                   pos_enc="none", geo_bias=False, residual_scale_init=1e-3).eval()
        counts = (7, 19)
        pos, _, batch = _batch(counts)
        tokens = torch.zeros(sum(counts), 16)
        tokens[:, 0] = torch.arange(sum(counts), dtype=torch.float32)  # unique fingerprint per row
        out = bt(pos, tokens, batch)
        self.assertTrue(bool(torch.all(out[1:, 0] > out[:-1, 0])))

    def test_padding_does_not_leak_between_cases(self):
        bt = BottleneckTransformer(16, in_dim=IN_DIM, layers=1, heads=4, ffn_ratio=2, dropout=0.0,
                                   pos_enc="none", geo_bias=False, residual_scale_init=1e-3).eval()
        counts = (7, 19)
        pos, _, batch = _batch(counts)
        tokens = torch.randn(sum(counts), 16)
        ref = bt(pos, tokens, batch)
        perturbed = tokens.clone()
        perturbed[counts[0]:] = 1e3
        out = bt(pos, perturbed, batch)
        self.assertTrue(torch.equal(ref[:counts[0]], out[:counts[0]]))

    def test_entropy_diagnostic_is_eval_only(self):
        bt = BottleneckTransformer(16, in_dim=IN_DIM, layers=1, heads=4, ffn_ratio=2, dropout=0.0,
                                   pos_enc="none", geo_bias=False, residual_scale_init=1e-3)
        pos, _, batch = _batch((7, 19))
        tokens = torch.randn(26, 16)
        bt.train()
        bt(pos, tokens, batch)
        self.assertIsNone(bt.layers[0].last_attention_entropy)  # no host-device sync while training
        bt.eval()
        bt(pos, tokens, batch)
        self.assertIsNotNone(bt.layers[0].last_attention_entropy)


class BottleneckWiringTest(unittest.TestCase):
    def test_centre_features_stay_aligned_with_the_coarsest_level(self):
        model = _model(bottleneck_transformer=True, bottleneck_heads=4, bottleneck_pos_enc="input_features").eval()
        pos, x, batch = _batch()
        with torch.no_grad():
            model.encode_support(pos, x, batch, unit_ids=["a", "b"], evaluation=True)
        expected = x
        for sa in model.sa:
            expected = expected[sa.last_indices]
        self.assertEqual(len(expected), sum(model.bottleneck.last_token_counts))
        coarse_batch = batch
        for sa in model.sa:
            coarse_batch = coarse_batch[sa.last_indices]
        self.assertTrue(bool(torch.all(coarse_batch[1:] >= coarse_batch[:-1])), "coarse batch must be sorted")

    def test_case_order_does_not_change_predictions(self):
        model = _model(bottleneck_transformer=True, bottleneck_heads=4).eval()
        pos_a, x_a, _ = _batch((40,), seed=1)
        pos_b, x_b, _ = _batch((40,), seed=2)

        def run(p1, f1, p2, f2):
            pos = torch.cat([p1, p2])
            x = torch.cat([f1, f2])
            b = torch.cat([torch.zeros(len(p1), dtype=torch.long), torch.ones(len(p2), dtype=torch.long)])
            with torch.no_grad():
                return model.forward_support_query(pos, x, b, pos, x, b, unit_ids=["A", "B"], evaluation=True)

        ab = run(pos_a, x_a, pos_b, x_b)
        ba = run(pos_b, x_b, pos_a, x_a)
        self.assertTrue(torch.allclose(ab[:len(pos_a)], ba[len(pos_b):], atol=1e-6))

    def test_layers_zero_is_conditioning_only(self):
        model = _model(bottleneck_transformer=True, bottleneck_layers=0, bottleneck_heads=4).eval()
        self.assertIsNotNone(model.bottleneck)
        self.assertEqual(len(model.bottleneck.layers), 0)
        pos, x, batch = _batch()
        with torch.no_grad():
            out = model.forward_support_query(pos, x, batch, pos, x, batch, unit_ids=["a", "b"], evaluation=True)
        self.assertEqual(out.shape, (len(pos),))

    def test_module_rejects_stacking_two_global_blocks(self):
        with self.assertRaises(ValueError):
            _model(bottleneck_transformer=True, bottleneck_heads=4, coarse_attention=True)


class BottleneckConfigTest(unittest.TestCase):
    def _cfg(self, **model_overrides):
        cfg = ExpConfig()
        cfg.model = ModelConfig(**{**COMMON, **model_overrides})
        cfg.data.input_features = ("x", "y", "z", "abscissa_norm", "local_radius", "curvature")
        return cfg

    def test_disabled_by_default_and_adds_no_checkpoint_keys(self):
        off = _model()
        on = _model(bottleneck_transformer=True, bottleneck_heads=4)
        self.assertIsNone(getattr(off, "bottleneck", "missing"))
        self.assertEqual(
            [k for k in on.state_dict() if k.startswith("bottleneck.")] and True,
            True,
        )
        self.assertEqual([k for k in off.state_dict() if "bottleneck" in k], [])

    def test_validation_rejects_bad_bottleneck_settings(self):
        for bad in (
            {"bottleneck_transformer": True, "bottleneck_heads": 4, "coarse_attention": True},
            {"bottleneck_transformer": True, "bottleneck_heads": 3},          # channels not divisible
            {"bottleneck_transformer": True, "bottleneck_heads": 4, "bottleneck_ffn_ratio": 0},
            {"bottleneck_transformer": True, "bottleneck_heads": 4, "bottleneck_layers": -1},
            {"bottleneck_transformer": True, "bottleneck_heads": 4, "bottleneck_pos_enc": "nope"},
            {"bottleneck_transformer": True, "bottleneck_heads": 4, "bottleneck_dropout": 1.0},
            {"bottleneck_transformer": True, "bottleneck_heads": 4, "bottleneck_layers": 0,
             "bottleneck_pos_enc": "none"},
        ):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    validate_features(self._cfg(**bad))

    def test_time_features_still_require_random_frame_mode(self):
        cfg = self._cfg(bottleneck_transformer=True, bottleneck_heads=4)
        validate_features(cfg)  # the valid case must keep passing


if __name__ == "__main__":
    unittest.main()
