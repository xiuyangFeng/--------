"""Protect follow-up attribution: fixed feature channels, query-only k and capacity control."""
from __future__ import annotations

import copy
import json
import unittest

import torch
from torch_geometric.nn import knn_interpolate

from training_wss_min import config as C
from training_wss_min.baseline_models import (
    LocalWallBranch, PointwiseWallBranch, PointNetPlusPlusRegressor, build_baseline_model,
)
from training_wss_min.tools import prepare_v6_followup_matrix as matrix


def model(**kwargs):
    torch.manual_seed(1234)
    options = dict(in_dim=25, width=16, sa_ratios=(0.25, 0.25, 0.25),
                   sa_radius=(0.05, 0.1, 0.2), sa_nsample=(8, 8, 8),
                   sa_center_counts=(16, 8, 4), sa_blocks=(0, 0, 0),
                   fp_knn=3, head_hidden=16, out_dim=1,
                   local_branch=True, local_branch_radii=(0.015, 0.03, 0.06),
                   local_branch_nsample=(8, 8, 8), local_branch_feature_indices=(14, 15, 16))
    options.update(kwargs)
    return PointNetPlusPlusRegressor(**options).eval()


def cloud(n=64, seed=9):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n, 3, generator=g) * 0.1, torch.randn(n, 25, generator=g), torch.zeros(n, dtype=torch.long)


class FeatureMaskTests(unittest.TestCase):
    def test_mask_is_post_standardization_copy_with_zero_gradients(self):
        m = model(input_feature_mask_indices=(14, 15, 16))
        _, x, _ = cloud()
        x.requires_grad_()
        original = x.detach().clone()
        masked = m._mask_input_features(x)
        self.assertEqual(masked.shape, (64, 25))
        self.assertEqual(int(torch.count_nonzero(masked[:, 14:17])), 0)
        torch.testing.assert_close(x.detach(), original, rtol=0, atol=0)
        masked.sum().backward()
        self.assertEqual(int(torch.count_nonzero(x.grad[:, 14:17])), 0)
        self.assertTrue(bool((x.grad[:, :14] == 1).all()))

    def test_mask_covers_support_normal_differences_and_query_features(self):
        m = model(input_feature_mask_indices=(14, 15, 16), query_decoder="local_attn",
                  query_decoder_k=16, query_decoder_residual=True,
                  query_decoder_feature_indices=(4, 14, 15, 16))
        with torch.no_grad():
            torch.nn.init.normal_(m.local_wall_branch.fuse.weight, std=.2)
            torch.nn.init.normal_(m.local_query.res_out.weight, std=.2)
            torch.nn.init.normal_(m.local_query.correction[-1].weight, std=.2)
        pos, x, batch = cloud()
        qpos, qx, qbatch = cloud(32, seed=10)
        changed_x, changed_qx = x.clone(), qx.clone()
        changed_x[:, 14:17] += 50
        changed_qx[:, 14:17] -= 30
        seen_attributes = []
        hook = m.local_wall_branch.register_forward_pre_hook(
            lambda module, args, kwargs: seen_attributes.append(kwargs["geo_attr"].clone()), with_kwargs=True)
        with torch.no_grad():
            a = m.forward_support_query(pos, x, batch, qpos, qx, qbatch)
            b = m.forward_support_query(pos, changed_x, batch, qpos, changed_qx, qbatch)
        hook.remove()
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertTrue(all(int(torch.count_nonzero(a)) == 0 for a in seen_attributes))

    def test_disabled_mask_preserves_tensor_identity_and_state(self):
        a, b = model(), model(input_feature_mask_indices=(), query_interpolation_k=0,
                              local_branch_mode="neighborhood")
        _, x, _ = cloud()
        self.assertIs(a._mask_input_features(x), x)
        self.assertEqual(a.state_dict().keys(), b.state_dict().keys())
        for key in a.state_dict():
            torch.testing.assert_close(a.state_dict()[key], b.state_dict()[key], rtol=0, atol=0)


class QueryAndCapacityTests(unittest.TestCase):
    def test_query_k_changes_only_decode_not_encoded_support(self):
        a, b = model(), model(query_interpolation_k=16)
        pos, x, batch = cloud()
        qpos, qx, qbatch = cloud(32, seed=10)
        with torch.no_grad():
            ea, eb = a.encode_support(pos, x, batch), b.encode_support(pos, x, batch)
            for u, v in zip(ea, eb):
                torch.testing.assert_close(u, v, rtol=0, atol=0)
            self.assertEqual([m.k for m in a.fp], [m.k for m in b.fp])
            expected = b.head(knn_interpolate(eb[1], eb[0], qpos, eb[2], qbatch, k=16)).squeeze(-1)
            actual = b.decode_query(eb, qpos, qx, qbatch)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_pointwise_control_has_close_capacity_and_no_spatial_reads(self):
        local = LocalWallBranch(32, 32, (.015, .03, .06), (16, 16, 16), 32, attr_dim=3)
        pointwise = PointwiseWallBranch(32, 32, 34, 3).eval()
        self.assertEqual(sum(p.numel() for p in local.parameters()), 10496)
        self.assertEqual(sum(p.numel() for p in pointwise.parameters()), 10640)
        x = torch.randn(32, 32)
        torch.testing.assert_close(pointwise(None, x, None), torch.zeros_like(x), rtol=0, atol=0)
        with torch.no_grad():
            pointwise.fuse.weight.fill_(.01)
            a = pointwise(None, x, None)
            b = pointwise(torch.randn(32, 3), x, torch.arange(32), geo_attr=torch.randn(32, 3))
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertGreater(float(a.abs().max()), 0)

    def test_pointwise_zero_init_matches_parent_without_branch(self):
        parent = model(local_branch=False)
        child = model(local_branch_mode="pointwise")
        child.load_state_dict(parent.state_dict(), strict=False)
        pos, x, batch = cloud()
        with torch.no_grad():
            torch.testing.assert_close(parent(pos, x, batch), child(pos, x, batch), rtol=0, atol=0)


class MatrixContractTests(unittest.TestCase):
    def test_exactly_authorised_arms_and_unchanged_protocol(self):
        anchor, configs = matrix.payloads()
        self.assertEqual(set(configs), {f"D{i}" for i in range(1, 8)} |
                         {f"L{i}" for i in range(1, 4)} | {f"M{i}" for i in range(1, 6)})
        for aid, p in configs.items():
            with self.subTest(arm=aid):
                cfg = C.ExpConfig.from_dict(p)
                self.assertEqual(p["data"]["input_features"], anchor["data"]["input_features"])
                self.assertEqual(p["data"]["data_root"], anchor["data"]["data_root"])
                self.assertEqual(cfg.data.target, "wss")
                self.assertEqual((cfg.train.epochs, cfg.train.seed), (400, 1234))
                self.assertEqual((cfg.model.out_dim, cfg.model.fp_knn), (1, 3))
                if aid.startswith("L"):
                    self.assertEqual(p["data"], anchor["data"])
                    self.assertEqual(p["model"], anchor["model"])
        self.assertEqual(configs["M5"]["model"]["local_branch_radii"], [.03] * 3)
        self.assertEqual(matrix.build(check=True), 0)

    def test_mask_indices_reject_invalid_or_silently_ignored_inputs(self):
        anchor, _ = matrix.payloads()
        for indices in ([25], [-1], [7, 7], [1.5], [True]):
            p = copy.deepcopy(anchor)
            p["model"]["input_feature_mask_indices"] = indices
            with self.subTest(indices=indices), self.assertRaisesRegex(ValueError, "input_feature_mask_indices"):
                C.ExpConfig.from_dict(p)

    def test_incompatible_query_override_and_pointwise_mode_are_rejected(self):
        anchor, _ = matrix.payloads()
        p = copy.deepcopy(anchor)
        p["model"].update(query_interpolation_k=16, query_decoder="sep_kernel")
        with self.assertRaisesRegex(ValueError, "query_interpolation_k"):
            C.ExpConfig.from_dict(p)
        p = copy.deepcopy(anchor)
        p["model"].update(local_branch=False, local_branch_mode="pointwise")
        with self.assertRaisesRegex(ValueError, "local_branch"):
            C.ExpConfig.from_dict(p)


if __name__ == "__main__":
    unittest.main()
