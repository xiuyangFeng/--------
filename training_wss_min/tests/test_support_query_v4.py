"""Support/Query, sampler naming, fixed SA, and canonical PostView regressions."""

from __future__ import annotations

import json
import unittest
from unittest import mock
from pathlib import Path

import numpy as np
import torch

from training_wss_min import config as C, dataset as D
from training_wss_min.baseline_models import PointNetPlusPlusRegressor, PointNetRegressor
from training_wss_min.evaluate import _evaluate_space
from training_wss_min.tools.preflight_v4_jobs import requires_surface_area
from training_wss_min.tools.export_wss_postview import (
    _find_partition,
    _resolve_bundle_path,
    build_postview_hotspot_payload,
)


REPO = Path(__file__).resolve().parents[2]
SPLIT = REPO / "training/splits/split_AG_AAA_wss_min_v4_stratified_seed1234.json"
CONFIGS = tuple(sorted((REPO / "training_wss_min/configs/pointnet_v4").glob("*random5000*.json"))) + tuple(
    sorted((REPO / "training_wss_min/configs/pointnetpp_v4").glob("*fixed_same.json"))
) + tuple(sorted((REPO / "training_wss_min/configs/pointnetpp_v4").glob("*center_*.json")))


def synthetic_case(n=800):
    rng = np.random.default_rng(7)
    pos = rng.normal(size=(n, 3)).astype(np.float32)
    return {"cohort": "AG/fast", "case": "SYNTH", "unit_id": "AG/fast/SYNTH",
            "pos": pos, "y_norm": rng.normal(size=n).astype(np.float32),
            "y_raw": rng.random(n).astype(np.float32),
            "curvature": rng.normal(size=n).astype(np.float32),
            "local_radius": (rng.random(n) + .1).astype(np.float32)}


class TestConfigsAndSampling(unittest.TestCase):
    def test_twelve_frozen_configs_are_explicit_and_unique(self):
        self.assertEqual(len(CONFIGS), 12)
        runs = set()
        for path in CONFIGS:
            cfg = C.ExpConfig.from_json(path)
            self.assertTrue(cfg.eval.fixed_support)
            self.assertEqual(cfg.eval.query_chunk_size, 16384)
            self.assertIn(cfg.data.support_sampling, {"fps", "random", "area_random"})
            expected_mode = (
                "both_strict" if cfg.data.support_sampling == "area_random"
                else "legacy_vertex"
            )
            self.assertEqual(cfg.eval.surface_metric_mode, expected_mode)
            self.assertEqual(requires_surface_area(cfg), expected_mode == "both_strict")
            self.assertNotIn(cfg.name, runs); runs.add(cfg.name)

    def test_q3v_changes_only_center_sampler_from_q1v(self):
        root = REPO / "training_wss_min/configs/pointnetpp_v4"
        q1v = json.loads((root / "ag_aaa_v4_stratified_sa3_random5000_fpscenter_same.json").read_text())
        q3v = json.loads((root / "ag_aaa_v4_stratified_sa3_random5000_randomcenter_same.json").read_text())
        for payload in (q1v, q3v):
            payload.pop("name")
            payload.pop("notes")
            payload["model"].pop("sa_center_sampling")
        self.assertEqual(q1v, q3v)

    def test_eval_mode_is_backward_compatible_and_validated(self):
        self.assertEqual(C.ExpConfig.from_dict({}).eval.surface_metric_mode, "legacy_vertex")
        with self.assertRaisesRegex(ValueError, "surface_metric_mode"):
            C.ExpConfig.from_dict({"eval": {"surface_metric_mode": "uniform_area"}})

    def test_vertex_metrics_do_not_require_or_emit_fake_area_weights(self):
        case = synthetic_case(100)
        result = _evaluate_space(
            [case], [case["y_raw"]], [case["y_raw"]], save_dir=None,
            make_plots=False, plot_space="physical", include_area=False,
        )
        self.assertNotIn("surface_area_weights", case)
        self.assertNotIn("primary_physical_or_normalized_casebalanced_area", result)
        self.assertNotIn("area_hotspot", result)
        self.assertNotIn("area_overall", next(iter(result["per_case"].values())))

    def test_legacy_postview_does_not_load_or_emit_area_metrics(self):
        case = synthetic_case(100)
        with mock.patch(
            "training_wss_min.tools.export_wss_postview.S.area_weights_for_case",
            side_effect=AssertionError("legacy PostView must not load STL area"),
        ):
            payload = build_postview_hotspot_payload(
                case, case["y_raw"], case["y_raw"], "legacy_vertex"
            )
        self.assertEqual(payload["basis"], "legacy_vertex")
        self.assertIsNone(payload["area_hotspot"])
        self.assertNotIn("surface_area_weights", case)
        self.assertTrue(np.array_equal(
            payload["scalars"]["true_highrisk"],
            payload["scalars"]["legacy_vertex_true_highrisk"],
        ))

    def test_strict_postview_emits_area_metrics(self):
        case = synthetic_case(100)
        case["surface_area_weights"] = np.full(100, 0.01, dtype=np.float64)
        payload = build_postview_hotspot_payload(
            case, case["y_raw"], case["y_raw"], "both_strict"
        )
        self.assertEqual(payload["basis"], "surface_area")
        self.assertIsNotNone(payload["area_hotspot"])

    def test_unused_same_query_area_sampler_does_not_load_area(self):
        case = synthetic_case(800)
        cfg = C.DataConfig(
            wall_n_points=200, sampling="random", support_n_points=200,
            support_sampling="random", query_mode="same", query_n_points=200,
            query_sampling="area_random", input_features=("x", "y", "z"),
        )
        exp = C.ExpConfig(data=cfg)
        self.assertFalse(requires_surface_area(exp))
        with mock.patch(
            "training_wss_min.dataset.S.area_weights_for_case",
            side_effect=AssertionError("unused SAME query sampler must not load area"),
        ):
            ds = D.WSSMinDataset([case], cfg, {}, training=True, base_seed=1234)
            self.assertEqual(len(ds[0]["query_idx"]), 200)

    def test_same_and_independent_indices(self):
        case = synthetic_case()
        same = C.DataConfig(wall_n_points=200, sampling="random", support_n_points=200,
                            support_sampling="random", query_mode="same", query_n_points=200,
                            query_sampling="random", input_features=("x", "y", "z"))
        ds = D.WSSMinDataset([case.copy()], same, {}, training=True, base_seed=1234)
        a = ds[0]
        self.assertTrue(torch.equal(a["support_idx"], a["query_idx"]))
        sep = C.DataConfig(wall_n_points=200, sampling="random", support_n_points=200,
                           support_sampling="random", query_mode="independent", query_n_points=200,
                           query_sampling="random", input_features=("x", "y", "z"))
        ds2 = D.WSSMinDataset([case.copy()], sep, {}, training=True, base_seed=1234)
        b = ds2[0]
        self.assertIsNot(b["support_idx"], b["query_idx"])
        self.assertFalse(torch.equal(b["support_idx"], b["query_idx"]))
        self.assertEqual(len(torch.unique(b["support_idx"])), 200)
        ds2.set_epoch(1)
        self.assertFalse(torch.equal(b["support_idx"], ds2[0]["support_idx"]))

    def test_area_random_probabilities_are_distinct_and_reproducible(self):
        case = synthetic_case(6000)
        weights = np.linspace(1.0, 3.0, 6000, dtype=np.float64)
        case["surface_area_weights"] = weights / weights.sum()
        cfg = C.DataConfig(wall_n_points=5000, sampling="area_random")
        a = D.sample_indices(case, cfg, 1234)
        b = D.sample_indices(case, cfg, 1234)
        c = D.sample_indices(case, cfg, 2025)
        self.assertTrue(np.array_equal(a, b))
        self.assertFalse(np.array_equal(a, c))
        self.assertEqual(len(np.unique(a)), 5000)
        self.assertTrue(np.isfinite(case["surface_area_weights"]).all())
        self.assertAlmostEqual(float(case["surface_area_weights"].sum()), 1.0)

    def test_canonical_partitions_cover_three_cohorts(self):
        self.assertEqual(_find_partition(SPLIT, "AG/fast/LIU_LI_QUN"), "test")
        self.assertEqual(_find_partition(SPLIT, "AAA/ruputer/DING_JUN_FENG"), "test")
        self.assertEqual(_find_partition(SPLIT, "AAA/unruputer/CHEN_FU_YE"), "test")
        self.assertEqual(D.canonical_unit_id("fast/LIU_LI_QUN"), "AG/fast/LIU_LI_QUN")

    def test_postview_bundle_resolution_covers_ag_and_aaa(self):
        root = Path("/frozen/data_wss_min")
        expected = {
            "fast/LIU_LI_QUN": root / "AG/fast/LIU_LI_QUN/bundle.npz",
            "AAA/ruputer/DING_JUN_FENG": root / "AAA/ruputer/DING_JUN_FENG/bundle.npz",
            "AAA/unruputer/CHEN_FU_YE": root / "AAA/unruputer/CHEN_FU_YE/bundle.npz",
        }
        for label, path in expected.items():
            canonical, actual = _resolve_bundle_path(root, label)
            self.assertEqual(actual, path)
            self.assertEqual(canonical.split("/", 1)[0], path.relative_to(root).parts[0])


class TestModels(unittest.TestCase):
    def test_qad_config_requires_pointnetpp_fixed_support(self):
        with self.assertRaisesRegex(ValueError, "model.name='pointnetpp'"):
            C.ExpConfig.from_dict({
                "model": {"name": "pointnet", "query_decoder": "qad_lite"},
            })
        with self.assertRaisesRegex(ValueError, "fixed support/query"):
            C.ExpConfig.from_dict({
                "model": {"name": "pointnetpp", "query_decoder": "qad_lite"},
            })

    def test_pointnet_same_is_exact(self):
        torch.manual_seed(1)
        model = PointNetRegressor(6, 32, 64, local_channels=(32, 64),
                                  decoder_channels=(64, 32)).eval()
        pos, x = torch.randn(100, 3), torch.randn(100, 6)
        batch = torch.tensor([0] * 50 + [1] * 50)
        legacy = model(pos, x, batch)
        separated = model.forward_support_query(pos, x, batch, pos, x, batch)
        self.assertTrue(torch.equal(legacy, separated))

    def test_pointnetpp_fixed_counts_and_chunk_decode(self):
        torch.manual_seed(2)
        model = PointNetPlusPlusRegressor(
            6, 16, (.25, .25, .25), (.2, .4, .8), (16, 16, 16), 3, 32,
            sa_center_counts=(50, 20, 8), sa_center_sampling="random",
        ).eval()
        pos, x = torch.randn(400, 3), torch.randn(400, 6)
        batch = torch.tensor([0] * 200 + [1] * 200)
        encoded = model.encode_support(pos, x, batch, unit_ids=["a", "b"],
                                       global_seed=1234, evaluation=True)
        self.assertEqual(torch.bincount(encoded[2]).tolist(), [200, 200])
        before = model.support_encode_calls
        full = model.decode_query(encoded, pos, x, batch)
        chunks = torch.cat([
            model.decode_query(encoded, pos[:200], x[:200], torch.zeros(200, dtype=torch.long)),
            model.decode_query(encoded, pos[200:], x[200:], torch.ones(200, dtype=torch.long)),
        ])
        self.assertEqual(model.support_encode_calls, before)
        self.assertTrue(torch.allclose(full, chunks, atol=1e-5, rtol=1e-5))

    def test_pointnetpp_protocol_center_counts(self):
        torch.manual_seed(3)
        for center_sampling in ("fps", "random"):
            model = PointNetPlusPlusRegressor(
                6, 8, (.25, .25, .25), (.05, .10, .20), (16, 16, 16), 3, 16,
                sa_center_counts=(500, 125, 32), sa_center_sampling=center_sampling,
            ).eval()
            pos, x = torch.randn(1200, 3), torch.randn(1200, 6)
            batch = torch.tensor([0] * 600 + [1] * 600)
            model.encode_support(pos, x, batch, unit_ids=["a", "b"],
                                 global_seed=1234, evaluation=True)
            self.assertEqual([sa.last_center_counts for sa in model.sa],
                             [[500, 500], [125, 125], [32, 32]])

    def test_qad_zero_init_is_base_equivalent_and_uses_query_geometry(self):
        kwargs = dict(
            in_dim=6, width=16, sa_ratios=(.25, .25, .25),
            sa_radius=(.2, .4, .8), sa_nsample=(16, 16, 16),
            fp_knn=3, head_hidden=32, sa_center_counts=(50, 20, 8),
            sa_center_sampling="fps",
        )
        torch.manual_seed(11)
        base = PointNetPlusPlusRegressor(**kwargs).eval()
        torch.manual_seed(11)
        qad = PointNetPlusPlusRegressor(
            **kwargs, query_decoder="qad_lite", qad_hidden=16,
        ).eval()
        pos, x = torch.randn(240, 3), torch.randn(240, 6)
        batch = torch.tensor([0] * 120 + [1] * 120)
        base_encoded = base.encode_support(
            pos, x, batch, unit_ids=["a", "b"], global_seed=1234, evaluation=True,
        )
        qad_encoded = qad.encode_support(
            pos, x, batch, unit_ids=["a", "b"], global_seed=1234, evaluation=True,
        )
        base_out = base.decode_query(base_encoded, pos, x, batch)
        qad_out = qad.decode_query(qad_encoded, pos, x, batch)
        self.assertTrue(torch.equal(base_out, qad_out))

        with torch.no_grad():
            qad.qad_out.weight.fill_(0.05)
        changed_x = x.clone()
        changed_x[:, 3:] += 0.5
        changed = qad.decode_query(qad_encoded, pos, changed_x, batch)
        self.assertFalse(torch.allclose(qad_out, changed))

    def test_qad_full_query_matches_case_chunks(self):
        torch.manual_seed(12)
        model = PointNetPlusPlusRegressor(
            6, 16, (.25, .25, .25), (.2, .4, .8), (16, 16, 16), 3, 32,
            sa_center_counts=(50, 20, 8), sa_center_sampling="random",
            query_decoder="qad_lite", qad_hidden=16,
        ).eval()
        with torch.no_grad():
            model.qad_out.weight.fill_(0.03)
        pos, x = torch.randn(400, 3), torch.randn(400, 6)
        batch = torch.tensor([0] * 200 + [1] * 200)
        encoded = model.encode_support(
            pos, x, batch, unit_ids=["a", "b"], global_seed=1234, evaluation=True,
        )
        full = model.decode_query(encoded, pos, x, batch)
        chunks = torch.cat([
            model.decode_query(encoded, pos[:200], x[:200], torch.zeros(200, dtype=torch.long)),
            model.decode_query(encoded, pos[200:], x[200:], torch.ones(200, dtype=torch.long)),
        ])
        self.assertTrue(torch.allclose(full, chunks, atol=1e-5, rtol=1e-5))


if __name__ == "__main__":
    unittest.main()
