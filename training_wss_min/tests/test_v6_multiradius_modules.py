"""Attribution, grouping and gradient contracts for the six MS model arms."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import torch

from training_wss_min.baseline_models import (
    BottleneckTransformerLayer, CapacityMatchedTokenFFN,
    MultiRadiusSetAbstraction, PointNetPlusPlusRegressor, multiradius_group,
)
from training_wss_min.config import ExpConfig


ROOT = Path(__file__).resolve().parents[2]


def options():
    return dict(in_dim=25, width=8, sa_ratios=(.25, .25, .25),
                sa_radius=(.05, .1, .2), sa_nsample=(8, 8, 8),
                sa_center_counts=(16, 8, 4), sa_blocks=(0, 0, 0),
                fp_knn=3, head_hidden=16, local_geope=True,
                local_branch=True, local_branch_feature_indices=(14, 15, 16),
                local_branch_nsample=(8, 8, 8), bottleneck_heads=8)


def model(arm=0):
    cfg = options()
    if arm:
        cfg.update(multiradius_values=(.2,) if arm <= 2 else (.2, .2, .2) if arm == 5 else (.1, .2, .4),
                   multiradius_bottleneck=arm in (2, 4, 5), multiradius_ffn=arm == 6)
    torch.manual_seed(1234)
    return PointNetPlusPlusRegressor(**cfg).eval()


def cloud(n=64):
    gen = torch.Generator().manual_seed(19)
    return torch.rand(n, 3, generator=gen) * .25, torch.randn(n, 25, generator=gen), torch.zeros(n, dtype=torch.long)


class GroupingTests(unittest.TestCase):
    def test_exact_distance_order_self_duplicate_and_case_isolation(self):
        pos = torch.tensor([[0., 0, 0], [0., 0, 0], [0., 0, 0], [0., 0, 0],
                            [.1, 0, 0], [.3, 0, 0], [0., 0, 0], [.08, 0, 0]])
        batch = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1])
        centers = torch.tensor([3, 4, 6])
        for cap in (2, 6):
            edges = multiradius_group(pos, batch, centers, (.05, .2, .4, .2), cap)
            for radius, (rows, cols) in zip((.05, .2, .4, .2), edges):
                self.assertTrue(torch.equal(batch[cols], batch[centers][rows]))
                for i, center in enumerate(centers.tolist()):
                    expected = [j for j in range(len(pos)) if batch[j] == batch[center]
                                and (j == center or float(((pos[j] - pos[center]) ** 2).sum()) <= radius ** 2)]
                    expected.sort(key=lambda j: (-1 if j == center else float(((pos[j] - pos[center]) ** 2).sum()), j))
                    self.assertEqual(cols[rows == i].tolist(), expected[:cap])
                    self.assertIn(center, cols[rows == i].tolist())
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(edges[1], edges[3])))
        self.assertNotEqual(edges[0][1].tolist(), edges[2][1].tolist())


class InitializationTests(unittest.TestCase):
    def test_disabled_matches_frozen_historical_source_in_state_output_and_rng(self):
        path = ROOT / "training_wss_min/experiments/v6_followup_20260909/source_snapshot_13205/training_wss_min/baseline_models.py"
        spec = importlib.util.spec_from_file_location("training_wss_min._historical_ms_reference", path)
        old = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old)
        torch.manual_seed(1234)
        parent = old.PointNetPlusPlusRegressor(**options()).eval()
        old_rng = torch.get_rng_state().clone()
        child = model()
        self.assertTrue(torch.equal(old_rng, torch.get_rng_state()))
        self.assertEqual(parent.state_dict().keys(), child.state_dict().keys())
        for key, value in parent.state_dict().items():
            torch.testing.assert_close(value, child.state_dict()[key], rtol=0, atol=0)
        with torch.no_grad():
            torch.testing.assert_close(parent(*cloud()), child(*cloud()), rtol=0, atol=0)

    def test_common_initialization_and_rng_are_exact_across_arms(self):
        parent = model()
        expected_rng = torch.get_rng_state().clone()
        for arm in range(1, 7):
            child = model(arm)
            self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
            self.assertIsInstance(child.sa[-1], MultiRadiusSetAbstraction)
            self.assertNotIn("sa.2.local.0.weight", child.state_dict())
            for key, value in parent.state_dict().items():
                if not key.startswith("sa.2."):
                    torch.testing.assert_close(value, child.state_dict()[key], rtol=0, atol=0)

    def test_paired_branch_weights_and_middle_projection(self):
        a, b = model(4), model(5)
        self.assertEqual(a.state_dict().keys(), b.state_dict().keys())
        for key, value in a.state_dict().items():
            torch.testing.assert_close(value, b.state_dict()[key], rtol=0, atol=0)
        c = b.sa[-1].channels
        torch.testing.assert_close(b.sa[-1].fuse.weight, torch.cat([torch.zeros(c, c), torch.eye(c), torch.zeros(c, c)], -1), rtol=0, atol=0)
        for single, multi in ((1, 3), (2, 4)):
            a, b = model(single), model(multi)
            self.assertIsNone(a.sa[-1].fuse)
            for component in ("branches", "contexts"):
                sa = getattr(a.sa[-1], component)[0].state_dict()
                sb = getattr(b.sa[-1], component)[1].state_dict()
                for key in sa:
                    torch.testing.assert_close(sa[key], sb[key], rtol=0, atol=0)
            with torch.no_grad():
                torch.testing.assert_close(a(*cloud()), b(*cloud()), rtol=1e-6, atol=1e-7)


class ForwardContracts(unittest.TestCase):
    def test_every_arm_consumes_correct_original_center_features_and_has_finite_gradients(self):
        pos, feats, batch = cloud()
        for arm in range(1, 7):
            m = model(arm).train()
            seen = []
            hooks = [context.register_forward_pre_hook(
                lambda module, args, kw: seen.append(kw["feats"].detach().clone()), with_kwargs=True)
                for context in m.sa[-1].contexts]
            pred = m.forward_support_query(pos, feats, batch, pos[:23] + .001, feats[:23], batch[:23], evaluation=True)
            self.assertEqual(pred.shape, (23,))
            self.assertTrue(bool(torch.isfinite(pred).all()))
            pred.square().mean().backward()
            self.assertTrue(all(bool(torch.isfinite(p.grad).all()) for p in m.parameters() if p.grad is not None))
            expected = feats
            for stage in m.sa:
                expected = expected[stage.last_indices]
            self.assertEqual(len(seen), len(m.sa[-1].radii))
            for value in seen:
                torch.testing.assert_close(value, expected, rtol=0, atol=0)
            for branch in m.sa[-1].branches:
                self.assertIsNotNone(branch.geo_pe)
            for hook in hooks:
                hook.remove()

    def test_fusion_unblocks_outer_branch_gradients_after_first_update(self):
        m = model(4)
        optimizer = torch.optim.SGD(m.parameters(), lr=.5)
        pos, x, batch = cloud()
        m(pos, x, batch).square().mean().backward()
        outer = m.sa[-1].branches[0].local[0].weight
        self.assertEqual(float(outer.grad.abs().sum()), 0)
        self.assertGreater(float(m.sa[-1].fuse.weight.grad[:, :m.sa[-1].channels].abs().sum()), 0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        m(pos, x, batch).square().mean().backward()
        self.assertGreater(float(outer.grad.abs().sum()), 0)

    def test_global_attention_does_not_cross_cases_and_ignores_padding(self):
        module = MultiRadiusSetAbstraction(8, 64, 25, (.1, .2, .4), 8, 5,
                                           "fps", 2, True, 3, "bt", 1, 8, 2, 0., True, .001).eval()
        gen = torch.Generator().manual_seed(29)
        pos = torch.rand(13, 3, generator=gen) * .25
        x = torch.randn(13, 8, generator=gen)
        f = torch.randn(13, 25, generator=gen)
        batch = torch.tensor([0] * 10 + [1] * 3)
        with torch.no_grad():
            _, all_x, all_b = module(pos, x, batch, center_feats=f, geo_attr=f[:, 3:6], evaluation=True)
            _, one_x, _ = module(pos[:10], x[:10], batch[:10], center_feats=f[:10], geo_attr=f[:10, 3:6], evaluation=True)
        torch.testing.assert_close(all_x[all_b == 0], one_x, rtol=1e-5, atol=1e-6)


class FFNAndConfigurationTests(unittest.TestCase):
    def test_capacity_control_matches_real_bt_and_cannot_mix_tokens(self):
        bt = BottleneckTransformerLayer(256, 8, 2, 0., True, .001)
        ff = CapacityMatchedTokenFFN(256, 8, 2, True, 0., .001)
        count = sum(p.numel() for p in bt.parameters())
        self.assertEqual(count, ff.target_parameter_count)
        self.assertLess(abs(sum(p.numel() for p in ff.parameters()) / count - 1), .01)
        self.assertAlmostEqual(float(ff.gamma_ffn), .001)
        self.assertFalse(any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in ff.modules()))
        for training in (True, False):
            ff.train(training)
            x = torch.randn(4, 256, requires_grad=True)
            ff(x)[0].sum().backward()
            self.assertEqual(int(torch.count_nonzero(x.grad[1:])), 0)

    def test_uniform_attention_entropy_averages_heads_and_valid_queries(self):
        layer = BottleneckTransformerLayer(16, 4, 2, 0., False, .001).eval()
        with torch.no_grad():
            layer.qkv.weight.zero_()
            layer.qkv.bias.zero_()
            valid = torch.tensor([[True, True, True], [True, False, False]])
            layer(torch.zeros(2, 3, 16), torch.zeros(2, 3, 3), valid)
        self.assertAlmostEqual(float(layer.last_attention_entropy), float(3 * torch.log(torch.tensor(3.)) / 4), places=6)

    def test_invalid_config_combinations_rejected(self):
        base = json.loads((ROOT / "training_wss_min/configs/v6_followup_20260909/M2_a5_independent_k3_s1234.json").read_text())
        cases = [dict(multiradius_bottleneck=True),
                 dict(multiradius_values=[0]), dict(multiradius_values=[float("nan")]),
                 dict(multiradius_values=[.2], multiradius_bottleneck=True, multiradius_ffn=True),
                 dict(multiradius_values=[.2], bottleneck_transformer=True),
                 dict(multiradius_values=[.2], bottleneck_pos_enc="none"),
                 dict(multiradius_values=[.2], sa_blocks=[1, 1, 1]),
                 dict(multiradius_values=[.2], bottleneck_heads=7),
                 dict(multiradius_values=[.2], name="mlp")]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.json"
            for changed in cases:
                cfg = json.loads(json.dumps(base))
                cfg["model"].update(changed)
                path.write_text(json.dumps(cfg))
                with self.subTest(changed=changed), self.assertRaises(ValueError):
                    ExpConfig.from_json(path)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
