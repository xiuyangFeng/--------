from __future__ import annotations

import unittest

import torch

from wss_pinn.volume_field.config import VolumeExperimentConfig
from wss_pinn.volume_field.models import build_volume_model
from wss_pinn.volume_field.models.point_models import differentiable_knn_interpolate


def second_derivative_smoke(output: torch.Tensor, coords: torch.Tensor):
    first = torch.autograd.grad(output.sum(), coords, create_graph=True)[0]
    second = torch.autograd.grad(
        torch.square(first).sum(), coords, create_graph=False
    )[0]
    return first, second


class VolumeModelTests(unittest.TestCase):
    def test_differentiable_interpolation_has_first_and_second_coordinate_gradients(self):
        support_pos = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=torch.float64,
        )
        features = torch.tensor([[0.0], [1.0], [1.0], [1.0]], dtype=torch.float64)
        query = torch.tensor([[0.2, 0.2, 0.2], [0.3, 0.1, 0.2]], dtype=torch.float64, requires_grad=True)
        batch_s = torch.zeros(4, dtype=torch.long)
        batch_q = torch.zeros(2, dtype=torch.long)
        output = differentiable_knn_interpolate(
            features, support_pos, query, batch_s, batch_q, k=3
        )
        first, second = second_derivative_smoke(output, query)
        self.assertTrue(torch.isfinite(first).all())
        self.assertTrue(torch.isfinite(second).all())

    def test_pointnet_query_is_smooth_and_pointwise_batch_invariant(self):
        config = VolumeExperimentConfig.from_json(
            "wss_pinn/configs/volume_uvwp_peak_v1/pointnet_xyz_geom_pinn.json"
        )
        torch.manual_seed(1234)
        model = build_volume_model(config).double().eval()
        support_pos = torch.rand(64, 3, dtype=torch.float64)
        support_features = torch.cat(
            [support_pos, torch.rand(64, 3, dtype=torch.float64)], dim=1
        )
        batch_s = torch.zeros(64, dtype=torch.long)
        encoded = model.encode_support(support_pos, support_features, batch_s)
        query_one = torch.tensor([[0.31, 0.42, 0.53]], dtype=torch.float64, requires_grad=True)
        out_one = model.decode_query(encoded, query_one, torch.zeros(1, dtype=torch.long))
        first, second = second_derivative_smoke(out_one[:, 0], query_one)
        self.assertTrue(torch.isfinite(first).all())
        self.assertTrue(torch.isfinite(second).all())
        query_many = torch.cat(
            [query_one.detach(), torch.tensor([[0.8, 0.7, 0.6]], dtype=torch.float64)], dim=0
        )
        out_many = model.decode_query(encoded, query_many, torch.zeros(2, dtype=torch.long))
        torch.testing.assert_close(out_one.detach()[0], out_many.detach()[0], rtol=1e-10, atol=1e-10)

    def test_pointnetpp_query_has_finite_first_and_second_derivatives(self):
        config = VolumeExperimentConfig.from_json(
            "wss_pinn/configs/volume_uvwp_peak_v1/pointnetpp_xyz_pinn.json"
        )
        torch.manual_seed(1234)
        model = build_volume_model(config).double().eval()
        support = torch.rand(256, 3, dtype=torch.float64)
        encoded = model.encode_support(
            support,
            support,
            torch.zeros(256, dtype=torch.long),
            unit_ids=["synthetic"],
            epoch=0,
            global_seed=1234,
            evaluation=True,
        )
        query = torch.tensor(
            [[0.31, 0.42, 0.53], [0.70, 0.20, 0.60]],
            dtype=torch.float64,
            requires_grad=True,
        )
        output = model.decode_query(
            encoded, query, torch.zeros(2, dtype=torch.long)
        )
        first, second = second_derivative_smoke(output[:, 0], query)
        self.assertTrue(torch.isfinite(first).all())
        self.assertTrue(torch.isfinite(second).all())


if __name__ == "__main__":
    unittest.main()
