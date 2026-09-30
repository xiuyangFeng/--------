import numpy as np
import torch

from training_wss_min.joint_cycle import case_weighted_loss, case_metrics, weighted_fields


def test_duplicate_mesh_cells_preserve_physical_loss_and_metrics():
    y = torch.zeros(2, 80, 3)
    p = torch.tensor([1., 3.])[:, None, None].expand_as(y).clone().requires_grad_()
    weight = torch.tensor([3., 1.])
    batch = torch.zeros(2, dtype=torch.long)
    loss = case_weighted_loss(p, y, weight, batch)
    duplicated = case_weighted_loss(p[[0, 0, 1]], y[[0, 0, 1]],
                                   torch.tensor([1.5, 1.5, 1.]), torch.zeros(3, dtype=torch.long))
    assert torch.allclose(loss, duplicated)
    loss.backward()
    assert torch.isfinite(p.grad).all()


def test_vector_metric_is_rotation_invariant():
    rng = np.random.default_rng(7)
    y = rng.normal(size=(19, 80, 3)) + np.array([10., -2., 5.])
    p = y + rng.normal(size=y.shape) * .1
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    w = rng.uniform(.1, 1., 19)
    a, b = weighted_fields(y, p, w), weighted_fields(y @ rotation, p @ rotation, w)
    for key in ("rmse", "relative_l2", "r2"):
        np.testing.assert_allclose(a[key], b[key], rtol=1e-10)


def test_zero_prediction_cannot_improve_direction_by_disappearing():
    y = np.ones((2, 80, 3))
    p = y.copy()
    p[1] = 0
    metrics = case_metrics("velocity", y, p, np.ones(2), .01)
    assert metrics["cycle"]["direction_weight_coverage"] == 1
    np.testing.assert_allclose(metrics["cycle"]["direction_cosine"], .5)
