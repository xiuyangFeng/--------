"""V6 single-frame matrix modules: parent parity at init, and the config contracts.

Every new capability (local wall branch, local query decoder, case-scale head,
pointwise Jensen back-transform, V6 point features) must be a strict no-op when
its config field is absent, so historical runs stay bit-for-bit reproducible.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import torch

from training_wss_min import config as C, dataset as D
from training_wss_min.baseline_models import PointNetPlusPlusRegressor

REPO = Path(__file__).resolve().parents[2]
VIEW_ROOT = REPO / "data_wss_v5/views/wss_min_view_v1"
GEOM_ROOT = REPO / "data_wss_v5/views/wss_min_geom_v2"


def build(in_dim=7, **kwargs):
    torch.manual_seed(0)
    defaults = dict(
        in_dim=in_dim, width=16, sa_ratios=(0.25, 0.25, 0.25), sa_radius=(0.05, 0.1, 0.2),
        sa_nsample=(16, 16, 16), fp_knn=3, head_hidden=16, out_dim=1,
        sa_center_counts=(32, 16, 8), sa_blocks=(1, 1, 0), local_geope=True,
        local_geope_feature_indices=(3, 4, 5),
    )
    defaults.update(kwargs)
    return PointNetPlusPlusRegressor(**defaults)


def cloud(n=256, graphs=2, in_dim=7, seed=1):
    generator = torch.Generator().manual_seed(seed)
    pos = torch.rand(n * graphs, 3, generator=generator) * 2 - 1
    x = torch.rand(n * graphs, in_dim, generator=generator)
    batch = torch.arange(graphs).repeat_interleave(n)
    return pos, x, batch


class LocalWallBranchTest(unittest.TestCase):
    def test_zero_initialised_branch_reproduces_parent(self):
        pos, x, batch = cloud()
        parent, child = build(), build(local_branch=True, local_branch_radii=(0.03, 0.06),
                                       local_branch_nsample=(8, 8), local_branch_channels=8)
        child.load_state_dict(parent.state_dict(), strict=False)
        parent.eval(), child.eval()
        with torch.no_grad():
            expected = parent(pos, x, batch)
            actual = child(pos, x, batch)
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)

    def test_branch_changes_output_once_trained(self):
        pos, x, batch = cloud()
        model = build(local_branch=True, local_branch_radii=(0.05,), local_branch_nsample=(8,))
        model.eval()
        with torch.no_grad():
            before = model(pos, x, batch)
            torch.nn.init.normal_(model.local_wall_branch.fuse.weight, std=0.1)
            after = model(pos, x, batch)
        self.assertGreater(float((after - before).abs().max()), 1e-4)

    def test_state_dict_unchanged_when_disabled(self):
        self.assertEqual(sorted(build().state_dict()), sorted(build(local_branch=False).state_dict()))
        self.assertIsNone(build().local_wall_branch)


class LocalQueryDecoderTest(unittest.TestCase):
    def test_initialisation_matches_inverse_square_interpolation(self):
        support_pos, support_x, support_batch = cloud(n=128, in_dim=7, seed=2)
        query_pos, query_x, query_batch = cloud(n=64, in_dim=7, seed=3)
        parent = build(fp_knn=8)
        child = build(fp_knn=8, query_decoder="local_attn", query_decoder_k=8)
        child.load_state_dict(parent.state_dict(), strict=False)
        parent.eval(), child.eval()
        with torch.no_grad():
            encoded_parent = parent.encode_support(support_pos, support_x, support_batch)
            encoded_child = child.encode_support(support_pos, support_x, support_batch)
            expected = parent.decode_query(encoded_parent, query_pos, query_x, query_batch)
            actual = child.decode_query(encoded_child, query_pos, query_x, query_batch)
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)

    def test_residual_head_starts_at_zero_and_uses_query_features(self):
        support_pos, support_x, support_batch = cloud(n=128, in_dim=7, seed=2)
        query_pos, query_x, query_batch = cloud(n=64, in_dim=7, seed=3)
        model = build(fp_knn=3, query_decoder="local_attn", query_decoder_k=16,
                      query_decoder_residual=True)
        model.eval()
        with torch.no_grad():
            encoded = model.encode_support(support_pos, support_x, support_batch)
            base = model.decode_query(encoded, query_pos, query_x, query_batch)
            changed = model.decode_query(encoded, query_pos, query_x + 1.0, query_batch)
            torch.testing.assert_close(changed, base, rtol=0, atol=1e-6)  # zero-init residual
            torch.nn.init.normal_(model.local_query.res_out.weight, std=0.5)
            moved = model.decode_query(encoded, query_pos, query_x + 1.0, query_batch)
        self.assertGreater(float((moved - base).abs().max()), 1e-4)


class CaseScaleHeadTest(unittest.TestCase):
    def test_zero_initialised_head_reproduces_parent(self):
        pos, x, batch = cloud()
        parent, child = build(), build(case_scale_head=True)
        child.load_state_dict(parent.state_dict(), strict=False)
        parent.eval(), child.eval()
        with torch.no_grad():
            torch.testing.assert_close(child(pos, x, batch), parent(pos, x, batch), rtol=0, atol=1e-6)

    def test_bias_is_constant_within_a_case(self):
        pos, x, batch = cloud()
        model = build(case_scale_head=True)
        model.eval()
        with torch.no_grad():
            before = model(pos, x, batch)
            torch.nn.init.normal_(model.case_scale.mlp[-1].weight, std=0.5)
            after = model(pos, x, batch)
        delta = (after - before).numpy()
        for graph in (0, 1):
            rows = (batch == graph).numpy()
            self.assertLess(float(np.ptp(delta[rows])), 1e-5)
        self.assertGreater(abs(float(delta[batch == 0][0] - delta[batch == 1][0])), 1e-6)


class JensenBackTransformTest(unittest.TestCase):
    STATS = {"method": "log_z", "eps": 1e-6, "log": {"mean": -0.5, "std": 1.3}}

    def test_matches_lognormal_mean(self):
        mu = np.array([0.0, 1.0, -1.0])
        logvar = np.array([-2.0, 0.0, 1.0])
        actual = D.denormalize_wss_lognormal_mean(mu, logvar, self.STATS)
        log_mean = mu * 1.3 - 0.5
        expected = np.exp(log_mean + 0.5 * np.exp(logvar) * 1.3 ** 2) - 1e-6
        np.testing.assert_allclose(actual, expected, rtol=1e-12)

    def test_reduces_to_exp_mu_when_variance_vanishes(self):
        mu = np.array([0.3, -0.7])
        actual = D.denormalize_wss_lognormal_mean(mu, np.full(2, -60.0), self.STATS, logvar_min=-60.0)
        np.testing.assert_allclose(actual, D.denormalize_wss(mu, self.STATS), rtol=1e-9)


class ConfigContractTest(unittest.TestCase):
    def base(self, **data):
        payload = {
            "name": "unit/v6", "data": {"input_features": ["x", "y", "z"], **data},
            "model": {"name": "pointnetpp", "sa_center_counts": [32, 16, 8],
                      "sa_radius": [0.05, 0.1, 0.2], "sa_nsample": [16, 16, 16],
                      "sa_blocks": [1, 1, 0]},
            "train": {}, "eval": {},
        }
        return payload

    def test_surface_features_require_the_sidecar_root(self):
        payload = self.base(input_features=["x", "y", "z", "curv_k1"])
        with self.assertRaisesRegex(ValueError, "point_features_root"):
            C.ExpConfig.from_dict(payload)
        payload["data"]["point_features_root"] = str(GEOM_ROOT)
        C.ExpConfig.from_dict(payload)

    def test_local_attn_requires_independent_query(self):
        payload = self.base()
        payload["model"]["query_decoder"] = "local_attn"
        with self.assertRaisesRegex(ValueError, "independent"):
            C.ExpConfig.from_dict(payload)
        payload["data"].update(query_mode="independent", query_n_points=128,
                               query_sampling="random", support_sampling="random")
        C.ExpConfig.from_dict(payload)

    def test_pointwise_jensen_requires_the_nll_head(self):
        payload = self.base()
        payload["eval"]["pointwise_jensen"] = True
        with self.assertRaisesRegex(ValueError, "gaussian NLL"):
            C.ExpConfig.from_dict(payload)
        payload["train"]["loss"] = "gaussian_nll"
        payload["model"]["out_dim"] = 2
        C.ExpConfig.from_dict(payload)

    def test_legacy_configs_do_not_gain_new_behaviour(self):
        cfg = C.ExpConfig.from_json(
            REPO / "training_wss_min/configs/v5_rerun_20260906/r4_lsa2_h2_logradius_v5feat_s1234.json"
        )
        self.assertFalse(cfg.model.local_branch)
        self.assertFalse(cfg.model.case_scale_head)
        self.assertFalse(cfg.eval.pointwise_jensen)
        self.assertIsNone(cfg.data.point_features_root)
        self.assertEqual(C.v6_point_features(cfg), ())


class MatrixConsistencyTest(unittest.TestCase):
    def test_configs_on_disk_match_the_generator(self):
        from training_wss_min.tools.prepare_v6_singleframe_matrix import build
        build(check=True)   # raises SystemExit if any arm config is stale

    def test_each_arm_moves_only_declared_fields(self):
        matrix = json.loads((REPO / "training_wss_min/configs/v6_singleframe_20260909/matrix.json").read_text())
        anchor = json.loads((REPO / "training_wss_min/configs/v5_rerun_20260906"
                             "/r4_lsa2_h2_logradius_v5feat_s1234.json").read_text())
        for arm in matrix["arms"]:
            payload = json.loads((REPO / "training_wss_min/configs/v6_singleframe_20260909" / arm["config"]).read_text())
            moved = {f"{section}.{key}"
                     for section in ("data", "model", "train", "eval")
                     for key, value in payload[section].items()
                     if anchor[section].get(key, value) != value or key not in anchor[section]}
            declared = set(arm["changes"]) | set(arm.get("declared_but_already_default", []))
            self.assertLessEqual(moved, declared, f"{arm['id']} moved an undeclared field")
            self.assertLessEqual(set(arm["changes"]), moved, f"{arm['id']} declared a change that did not happen")


@unittest.skipUnless(GEOM_ROOT.is_dir(), "V6 geometry sidecar not built")
class GeometrySidecarTest(unittest.TestCase):
    CASE = ("AAA/ruputer", "DING_JUN_FENG")

    def load(self, features):
        stats = D.load_wss_stats(VIEW_ROOT / "wss_global_stats_train138.json")
        return D.load_case(*self.CASE, stats, data_root=VIEW_ROOT,
                           required_frame_version="v5_atlas_frame_v1",
                           extra_point_features=features, point_features_root=GEOM_ROOT)

    def test_every_declared_surface_feature_is_loadable(self):
        case = self.load(C.V6_SURFACE_FEATURE_KEYS)
        for key in C.V6_SURFACE_FEATURE_KEYS:
            self.assertEqual(len(case[key]), len(case["pos"]), key)
            self.assertTrue(np.isfinite(case[key]).all(), key)
        self.assertLessEqual(float(np.abs(case["shape_index"]).max()), 1.0 + 1e-6)

    def test_curvature_rows_align_and_match_the_atlas_radius(self):
        case = self.load(("curv_k1", "curv_k2", "tn_dot"))
        self.assertEqual(len(case["curv_k1"]), len(case["pos"]))
        tube = case["local_radius"] > 1e-6
        product = np.median(case["curv_k1"][tube] * case["local_radius"][tube])
        self.assertGreater(product, 0.7)   # k1 ~ 1/R on a tube
        self.assertLess(product, 1.5)
        self.assertLess(float(np.median(np.abs(case["tn_dot"]))), 0.3)

    def test_semantic_one_hot_and_branch_arclength(self):
        case = self.load(C.V6_SEMANTIC_FEATURE_KEYS + ("s_local_frac",))
        stacked = np.stack([case[k] for k in C.V6_SEMANTIC_FEATURE_KEYS], axis=1)
        np.testing.assert_array_equal(stacked.sum(axis=1), np.ones(len(case["pos"]), dtype=np.float32))
        self.assertGreaterEqual(float(case["s_local_frac"].min()), 0.0)
        self.assertLessEqual(float(case["s_local_frac"].max()), 1.0 + 1e-6)

    def test_nothing_is_attached_without_a_request(self):
        case = self.load(())
        self.assertNotIn("curv_k1", case)
        self.assertNotIn("sem_trunk", case)


if __name__ == "__main__":
    unittest.main()
