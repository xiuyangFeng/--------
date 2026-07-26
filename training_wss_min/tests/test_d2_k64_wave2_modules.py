from __future__ import annotations

import unittest

import torch
from torch_geometric.nn import knn_interpolate
from torch_geometric.utils import scatter

from training_wss_min.baseline_models import (
    CoarseGlobalBlock,
    ConservativeSEPKernel,
    LocalInvResBlock,
    PointNetPlusPlusRegressor,
    _drop_neighbors_preserve_one,
)
from training_wss_min.config import ExpConfig, ModelConfig
from training_wss_min.models import build_model


COMMON = dict(
    name="pointnetpp",
    width=8,
    sa_ratios=(0.25, 0.25, 0.25),
    sa_radius=(0.5, 1.0, 2.0),
    sa_nsample=(6, 4, 2),
    sa_blocks=(1, 1, 0),
    sa_center_counts=(10, 5, 2),
    sa_grouping=("knn_cover", "ball", "ball"),
    head_hidden=8,
)


def two_graph_cloud(n_per_graph: int = 24):
    torch.manual_seed(23)
    pos = torch.rand(2 * n_per_graph, 3)
    pos[n_per_graph:] += 3.0
    x = torch.rand(2 * n_per_graph, 6)
    batch = torch.cat([
        torch.zeros(n_per_graph, dtype=torch.long),
        torch.ones(n_per_graph, dtype=torch.long),
    ])
    return pos, x, batch


class LocalGeoPETests(unittest.TestCase):
    def test_local_geope_changes_parameters_and_receives_gradients(self):
        base = build_model(ModelConfig(**COMMON, local_geope=False), 6)
        geo = build_model(
            ModelConfig(
                **COMMON,
                local_geope=True,
                local_geope_feature_indices=(3, 4, 5),
            ),
            6,
        ).train()
        self.assertGreater(
            sum(p.numel() for p in geo.parameters()),
            sum(p.numel() for p in base.parameters()),
        )
        self.assertTrue(all(sa.geo_pe is not None for sa in geo.sa))
        pos, x, batch = two_graph_cloud()
        out = geo.forward_support_query(
            pos, x, batch, pos, x, batch,
            unit_ids=["a", "b"], epoch=0, global_seed=1234,
        )
        out.square().mean().backward()
        self.assertIsNotNone(geo.sa[0].geo_pe[-1].weight.grad)
        self.assertGreater(float(geo.sa[0].geo_pe[-1].weight.grad.abs().sum()), 0.0)

    def test_local_geope_feature_indices_are_validated(self):
        with self.assertRaisesRegex(ValueError, "local_geope_feature_indices"):
            ExpConfig.from_dict({
                "data": {"input_features": ["x", "y", "z"]},
                "model": {
                    "name": "pointnetpp",
                    "sa_radius": [0.1, 0.2, 0.4],
                    "sa_ratios": [0.25, 0.25, 0.25],
                    "sa_nsample": [8, 8, 8],
                    "sa_blocks": [0, 0, 0],
                    "local_geope": True,
                    "local_geope_feature_indices": [3, 4],
                },
            })

    def test_xyz_only_local_geope_uses_relative_position_and_distance(self):
        geo = build_model(
            ModelConfig(
                **COMMON, local_geope=True, local_geope_feature_indices=(),
            ),
            3,
        ).train()
        self.assertTrue(all(sa.geo_pe[0].in_features == 4 for sa in geo.sa))
        pos, _, batch = two_graph_cloud()
        x = torch.rand(pos.size(0), 3)
        out = geo.forward_support_query(
            pos, x, batch, pos, x, batch,
            unit_ids=["a", "b"], epoch=0, global_seed=1234,
        )
        out.square().mean().backward()
        self.assertGreater(float(geo.sa[0].geo_pe[-1].weight.grad.abs().sum()), 0.0)


class CoarseAttentionTests(unittest.TestCase):
    def test_attention_is_graph_local_and_normalized(self):
        torch.manual_seed(29)
        block = CoarseGlobalBlock(16, heads=4, residual_scale_init=1e-3).eval()
        pos = torch.rand(9, 3)
        x = torch.rand(9, 16)
        batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 1])
        out = block(pos, x, batch)
        self.assertEqual(out.shape, x.shape)
        self.assertEqual(len(block.last_attention_row_sums), 2)
        for sums in block.last_attention_row_sums:
            torch.testing.assert_close(
                sums, torch.ones_like(sums), rtol=1e-6, atol=1e-6
            )

    def test_pointnetpp_coarse_attention_is_wired(self):
        model = build_model(
            ModelConfig(**COMMON, coarse_attention=True, coarse_attention_heads=4),
            6,
        )
        self.assertIsInstance(model.coarse_global, CoarseGlobalBlock)
        pos, x, batch = two_graph_cloud()
        out = model.forward_support_query(
            pos, x, batch, pos, x, batch,
            unit_ids=["a", "b"], epoch=0, global_seed=1234,
        )
        self.assertEqual(tuple(out.shape), (48,))


class S3RegularizationTests(unittest.TestCase):
    def test_neighbor_drop_preserves_one_edge_per_center(self):
        row = torch.arange(5).repeat_interleave(4)
        col = torch.arange(20)
        pos_query = torch.arange(5, dtype=torch.float32).unsqueeze(-1).repeat(1, 3)
        pos_src = pos_query.repeat_interleave(4, dim=0)
        pos_src[:, 0] += torch.tensor([0.0, 0.1, 0.2, 0.3] * 5)
        kept_row, _ = _drop_neighbors_preserve_one(
            row, col, pos_src, pos_query, drop_rate=1.0, training=True
        )
        torch.testing.assert_close(
            torch.bincount(kept_row, minlength=5),
            torch.ones(5, dtype=torch.long),
        )

    def test_drop_path_is_graphwise_and_eval_is_identity_safe(self):
        torch.manual_seed(41)
        n_graphs, n_nodes, channels = 16, 4, 8
        batch = torch.arange(n_graphs).repeat_interleave(n_nodes)
        pos = torch.rand(n_graphs * n_nodes, 3)
        pos += batch.unsqueeze(-1) * 3.0
        x = torch.rand(n_graphs * n_nodes, channels)
        block = LocalInvResBlock(
            channels, radius_value=1.0, nsample=4,
            drop_path_rate=0.5, neighbor_drop_rate=0.0,
        ).train()
        out = block(pos, x, batch)
        graph_scale = block.last_drop_path_graph_scale.squeeze(-1)
        self.assertTrue(bool((graph_scale == 0).any()))
        self.assertTrue(bool((graph_scale > 1).any()))
        dropped = graph_scale == 0
        torch.testing.assert_close(out[dropped], x[dropped], rtol=0.0, atol=0.0)
        block.eval()
        eval_out_1 = block(pos, x, batch)
        eval_out_2 = block(pos, x, batch)
        torch.testing.assert_close(eval_out_1, eval_out_2, rtol=0.0, atol=0.0)

    def test_regularization_config_ranges_and_wiring(self):
        cfg = ModelConfig(
            **COMMON, local_geope=True,
            drop_path_rate=0.1, neighbor_drop_rate=0.05,
        )
        model = build_model(cfg, 6)
        rates = [
            block.drop_path_rate for sa in model.sa for block in sa.blocks
        ]
        self.assertEqual(len(rates), 2)
        self.assertAlmostEqual(rates[0], 0.0)
        self.assertAlmostEqual(rates[1], 0.1)
        self.assertTrue(all(sa.neighbor_drop_rate == 0.05 for sa in model.sa))
        with self.assertRaisesRegex(ValueError, "drop_path_rate"):
            ExpConfig.from_dict({
                "model": {
                    "name": "pointnetpp",
                    "sa_radius": [0.1, 0.2, 0.4],
                    "sa_ratios": [0.25, 0.25, 0.25],
                    "sa_nsample": [8, 8, 8],
                    "sa_blocks": [1, 1, 0],
                    "drop_path_rate": 1.0,
                },
            })


class ConservativeSEPTests(unittest.TestCase):
    def test_zero_correction_beta2_matches_pyg_interpolation(self):
        torch.manual_seed(31)
        support_pos = torch.rand(18, 3)
        support_x = torch.rand(18, 7)
        support_batch = torch.tensor([0] * 9 + [1] * 9)
        support_pos[9:] += 2.0
        query_pos = torch.rand(10, 3)
        query_batch = torch.tensor([0] * 5 + [1] * 5)
        query_pos[5:] += 2.0
        kernel = ConservativeSEPKernel(7, hidden=8, k=3, beta_init=2.0).eval()
        actual = kernel(
            support_x, support_pos, query_pos, support_batch, query_batch
        )
        expected = knn_interpolate(
            support_x, support_pos, query_pos, support_batch, query_batch, k=3
        )
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
        row, weights = kernel.last_assignment
        self.assertTrue(bool(torch.all(weights >= 0)))
        sums = scatter(weights, row, dim=0, dim_size=query_pos.size(0), reduce="sum")
        torch.testing.assert_close(sums, torch.ones_like(sums), rtol=1e-6, atol=1e-6)

    def test_sep_same_query_uses_exact_identity(self):
        model = PointNetPlusPlusRegressor(
            6, 8, (0.25, 0.25, 0.25), (0.5, 1.0, 2.0), (6, 4, 2),
            3, 8, sa_center_counts=(10, 5, 2),
            sa_grouping=("knn_cover", "ball", "ball"),
            sa_blocks=(1, 1, 0), query_decoder="sep_kernel",
            sep_hidden=8, sep_beta_init=2.0,
        ).eval()
        pos, x, batch = two_graph_cloud()
        encoded = model.encode_support(
            pos, x, batch, unit_ids=["a", "b"], evaluation=True,
            global_seed=1234,
        )
        decoded = model.decode_query(encoded, pos, x, batch)
        expected = model.head(encoded[1]).squeeze(-1)
        torch.testing.assert_close(decoded, expected, rtol=0.0, atol=0.0)

    def test_sep_config_requires_pointnetpp_fixed_support(self):
        with self.assertRaisesRegex(ValueError, "model.name='pointnetpp'"):
            ExpConfig.from_dict({
                "model": {"name": "pointnet", "query_decoder": "sep_kernel"},
            })
        with self.assertRaisesRegex(ValueError, "fixed support/query"):
            ExpConfig.from_dict({
                "model": {"name": "pointnetpp", "query_decoder": "sep_kernel"},
            })


if __name__ == "__main__":
    unittest.main()
