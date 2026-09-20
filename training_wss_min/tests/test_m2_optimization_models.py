"""M2 optimization: paired local modulation and fixed-3NN decoder semantics."""

from __future__ import annotations

import importlib.util
import inspect
import json
from pathlib import Path
import unittest

import torch
from torch_geometric.nn import knn_interpolate

from training_wss_min import config as C
from training_wss_min.baseline_models import (
    LocalQueryDecoder,
    LocalWallBranch,
    PointNetPlusPlusRegressor,
)
from training_wss_min.models import build_model


ROOT = Path(__file__).resolve().parents[2]
M2_CONFIG = ROOT / "training_wss_min/configs/v6_followup_20260909/M2_a5_independent_k3_s1234.json"
OLD_SOURCE = ROOT / "training_wss_min/experiments/v6_followup_20260909/source_snapshot_13205/training_wss_min/baseline_models.py"


def options(**changes):
    raw = json.loads(M2_CONFIG.read_text())["model"]
    accepted = inspect.signature(PointNetPlusPlusRegressor).parameters
    result = {key: value for key, value in raw.items() if key in accepted}
    result.update(in_dim=25, sa_center_counts=(16, 12, 8))
    result.update(changes)
    return result


def model(**changes):
    torch.manual_seed(1234)
    return PointNetPlusPlusRegressor(**options(**changes))


def cloud(n=48, graphs=2, features=25, seed=3):
    gen = torch.Generator().manual_seed(seed)
    # Cases intentionally overlap in xyz: batch isolation must use case IDs.
    pos = torch.rand(n * graphs, 3, generator=gen) * 0.2
    x = torch.randn(n * graphs, features, generator=gen)
    batch = torch.arange(graphs).repeat_interleave(n)
    return pos, x, batch


def wall_branch(mode="none"):
    torch.manual_seed(1234)
    return LocalWallBranch(32, 32, (0.015, 0.03, 0.06), (16, 16, 16), 32,
                           attr_dim=3, modulation=mode, modulation_hidden=16)


class M2ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_disabled_preserves_historical_weights_rng_and_query_output(self):
        spec = importlib.util.spec_from_file_location("training_wss_min._m2_frozen_reference", OLD_SOURCE)
        old = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old)
        torch.manual_seed(1234)
        parent = old.PointNetPlusPlusRegressor(**options()).eval()
        expected_rng = torch.get_rng_state().clone()
        child = model().eval()
        self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
        self.assertEqual(parent.state_dict().keys(), child.state_dict().keys())
        for key, value in parent.state_dict().items():
            torch.testing.assert_close(value, child.state_dict()[key], rtol=0, atol=0)
        support, query = cloud(), cloud(n=31, seed=7)
        with torch.no_grad():
            expected = parent.decode_query(parent.encode_support(*support), *query)
            actual = child.decode_query(child.encode_support(*support), *query)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_modulation_preserves_parent_initialization_and_rng(self):
        parent = model()
        expected_rng = torch.get_rng_state().clone()
        children = []
        for mode in ("gate", "additive"):
            child = model(local_branch_modulation=mode)
            self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
            for key, value in parent.state_dict().items():
                torch.testing.assert_close(value, child.state_dict()[key], rtol=0, atol=0)
            children.append(child)
        self.assertEqual(children[0].state_dict().keys(), children[1].state_dict().keys())
        for key, value in children[0].state_dict().items():
            torch.testing.assert_close(value, children[1].state_dict()[key], rtol=0, atol=0)
        base_count = sum(p.numel() for p in parent.parameters())
        for child in children:
            self.assertEqual(sum(p.numel() for p in child.parameters()) - base_count, 3184)

    def test_zero_modulation_preserves_an_already_active_wall_branch(self):
        parent = wall_branch().eval()
        with torch.no_grad():
            parent.fuse.weight.normal_(std=0.2)
            parent.fuse.bias.fill_(0.3)
        pos, x, batch = cloud(features=32)
        normals = torch.randn(len(pos), 3)
        with torch.no_grad():
            expected = parent(pos, x, batch, normals)
            for mode in ("gate", "additive"):
                child = wall_branch(mode).eval()
                missing, unexpected = child.load_state_dict(parent.state_dict(), strict=False)
                self.assertTrue(all(key.startswith("modulation.") for key in missing))
                self.assertEqual(unexpected, [])
                actual = child(pos, x, batch, normals)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                diag = child.last_modulation_diagnostics
                self.assertEqual(float(diag["modulation_residual_l2"]), 0.0)
                self.assertGreater(float(diag["fused_residual_l2"]), 0.0)
                if mode == "gate":
                    self.assertEqual(float(diag["gate_mean"]), 1.0)
                    self.assertEqual(float(diag["gate_std"]), 0.0)

    def test_active_modulation_changes_output_and_receives_gradients(self):
        pos, x, batch = cloud(features=32)
        normals = torch.randn(len(pos), 3)
        for mode in ("gate", "additive"):
            branch = wall_branch(mode).eval()
            with torch.no_grad():
                branch.fuse.weight.normal_(std=0.2)
                branch.modulation[-1].weight.normal_(std=0.1)
                branch.modulation[-1].bias.fill_(0.1)
            out = branch(pos, x, batch, normals)
            diag = branch.last_modulation_diagnostics
            self.assertGreater(float(diag["modulation_residual_l2"]), 0.0)
            self.assertGreater(float(diag["fused_modulation_delta_l2"]), 0.0)
            self.assertTrue(all(torch.isfinite(value) for value in diag.values()))
            if mode == "gate":
                self.assertGreater(float(diag["gate_std"]), 0.0)
                self.assertGreaterEqual(float(diag["gate_min"]), 0.0)
                self.assertLessEqual(float(diag["gate_max"]), 2.0)
            out.square().mean().backward()
            for parameter in branch.modulation.parameters():
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(float(parameter.grad.abs().sum()), 0.0)

    def test_modulation_stays_pointwise_and_does_not_cross_cases(self):
        pos, x, batch = cloud(n=32, features=32)
        normals = torch.randn(len(pos), 3)
        for mode in ("gate", "additive"):
            branch = wall_branch(mode).eval()
            with torch.no_grad():
                branch.fuse.weight.normal_(std=0.2)
                branch.modulation[-1].weight.normal_(std=0.1)
                together = branch(pos, x, batch, normals)
                separate = torch.cat([
                    branch(pos[batch == i], x[batch == i], torch.zeros(32, dtype=torch.long),
                           normals[batch == i])
                    for i in range(2)
                ])
            torch.testing.assert_close(together, separate, rtol=1e-6, atol=1e-6)

    def test_three_neighbor_decoder_initialization_including_coincident_query(self):
        support_pos, support_x, support_batch = cloud(features=32)
        query_pos, query_input, query_batch = cloud(n=29, seed=5)
        query_pos[0] = support_pos[0]
        support_input = cloud()[1]
        attrs = (support_input[:, [4, 14, 15, 16]], query_input[:, [4, 14, 15, 16]])
        expected = knn_interpolate(support_x, support_pos, query_pos, support_batch, query_batch, k=3)
        for residual in (False, True):
            decoder = LocalQueryDecoder(32, 25, 1, 3, 32, 2.0, residual=residual, attr_dim=4)
            actual, correction = decoder(support_x, support_pos, query_pos, support_batch, query_batch,
                                         support_input, query_input, attrs)
            torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
            if residual:
                torch.testing.assert_close(correction, torch.zeros_like(correction), rtol=0, atol=0)
            else:
                self.assertIsNone(correction)

    def test_three_neighbor_residual_decoder_is_chunk_invariant(self):
        candidate = model(query_decoder="local_attn", query_decoder_k=3,
                          query_decoder_residual=True,
                          query_decoder_feature_indices=(4, 14, 15, 16)).eval()
        with torch.no_grad():
            candidate.local_query.correction[-1].weight.normal_(std=0.1)
            candidate.local_query.res_out.weight.normal_(std=0.1)
            encoded = candidate.encode_support(*cloud(graphs=1))
            query = cloud(n=43, graphs=1, seed=7)
            together = candidate.decode_query(encoded, *query)
            chunks = torch.cat([
                candidate.decode_query(encoded, *(part[start:start + 13] for part in query))
                for start in range(0, 43, 13)
            ])
        torch.testing.assert_close(together, chunks, rtol=1e-5, atol=1e-6)
        self.assertEqual(candidate.support_encode_calls, 1)

    def test_extra_sa1_block_preserves_existing_drop_path_rates(self):
        parent, child = model(), model(sa_blocks=(2, 1, 0))
        self.assertEqual(parent.sa[0].blocks[0].drop_path_rate, child.sa[0].blocks[0].drop_path_rate)
        self.assertEqual(parent.sa[1].blocks[0].drop_path_rate, child.sa[1].blocks[0].drop_path_rate)
        self.assertAlmostEqual(child.sa[0].blocks[1].drop_path_rate, 0.05)
        self.assertEqual(sum(p.numel() for p in child.parameters()) - sum(p.numel() for p in parent.parameters()), 25474)

    def test_model_factory_passes_modulation_and_rejects_incompatible_branch(self):
        cfg = C.ModelConfig(**json.loads(M2_CONFIG.read_text())["model"])
        cfg.local_branch_modulation = "gate"
        cfg.local_branch_modulation_hidden = 16
        candidate = build_model(cfg, 25)
        self.assertEqual(candidate.local_wall_branch.modulation_mode, "gate")
        self.assertEqual(sum(p.numel() for p in candidate.local_wall_branch.modulation.parameters()), 3184)
        for changes in (
            {"local_branch_modulation": "invalid"},
            {"local_branch_modulation": "gate", "local_branch": False},
            {"local_branch_modulation": "additive", "local_branch_mode": "pointwise"},
            {"local_branch_modulation_hidden": 0},
        ):
            with self.assertRaises(ValueError):
                model(**changes)


if __name__ == "__main__":
    unittest.main()
