from __future__ import annotations

import pytest
import torch

from training_wss_min import config as C
from training_wss_min.objectives import compute_loss, local_difference_loss, pairwise_ranking_loss


def _batch():
    # Identical physical patches encoded at different per-case coordinate scales.
    physical = torch.tensor([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0],
                             [6.0, 0.0, 0.0], [9.0, 0.0, 0.0]] * 2)
    geometry = torch.tensor([[0., 0., 1., 1., 0., 0., 5., 100.]] * 4
                            + [[0., 0., 1., 1., 0., 0., 5., 300.]] * 4)
    target = torch.tensor([0., 1., 2., 3., 0., 2., 4., 6.])
    return {"pos": physical / geometry[:, 7:8], "query_geometry": geometry,
            "batch": torch.tensor([0] * 4 + [1] * 4), "y": target, "y_raw": target}


def _local(pred, batch, **kwargs):
    return local_difference_loss(pred, batch["y"], batch["pos"], batch["batch"],
                                 query_geometry=batch["query_geometry"], **kwargs)


def test_auxiliaries_are_bit_compatible_when_disabled():
    cfg = C.TrainConfig(loss="mse")
    pred = torch.linspace(0, 1, 8, requires_grad=True)
    batch = _batch()
    # Geometry is not accessed at all when the optional weights are zero.
    del batch["query_geometry"]
    expected = ((pred - batch["y"]) ** 2).mean()
    actual = compute_loss(pred, batch, cfg, "cpu")
    assert torch.equal(actual, expected)
    expected_grad, = torch.autograd.grad(expected, pred, retain_graph=True)
    actual_grad, = torch.autograd.grad(actual, pred)
    assert torch.equal(actual_grad, expected_grad)


def test_local_difference_has_physical_mm_units_and_correct_target_minimum():
    batch = _batch()
    pred = batch["y"] * 0.25
    parts = {}
    value = _local(pred, batch, diagnostics=parts)
    assert value > 0
    assert _local(batch["y"], batch) == 0
    assert _local(torch.zeros_like(pred), batch) > value
    assert parts["local_difference_pair_count"] == 6
    assert parts["local_difference_cases_without_pairs"] == 0
    batch["pos"] /= 7
    batch["query_geometry"][:, 7] *= 7
    assert _local(pred, batch) == pytest.approx(value.item())


def test_local_edges_cross_arbitrary_voxel_boundaries():
    batch = _batch()
    batch["pos"][1] = torch.tensor([0.048, 0., 0.])
    batch["pos"][2] = torch.tensor([0.078, 0., 0.])
    pred = batch["y"].clone()
    pred[2] += 0.5
    assert _local(pred, batch) > 0


@pytest.mark.parametrize("name", ["pos", "query_geometry", "batch"])
def test_local_difference_missing_geometry_fails_loudly(name):
    batch = _batch()
    batch[name] = None
    with pytest.raises(ValueError, match="requires"):
        _local(batch["y"], batch)


def test_local_difference_invalid_physical_geometry_rejected():
    batch = _batch()
    batch["query_geometry"][0, 7] = 0
    with pytest.raises(ValueError, match="positive coordinate scales"):
        _local(batch["y"], batch)


@pytest.mark.parametrize("opposite_normals", [True, False])
def test_cross_wall_edges_are_filtered_and_zero_pairs_are_diagnosed(opposite_normals):
    batch = {key: val[:2].clone() for key, val in _batch().items()}
    if opposite_normals:
        batch["query_geometry"][1, :3] *= -1
    else:
        batch["pos"][1] = torch.tensor([0., 0., 0.03])
    pred = torch.zeros(2, requires_grad=True)
    parts = {}
    with pytest.warns(RuntimeWarning, match="no eligible pairs"):
        loss = _local(pred, batch, diagnostics=parts)
    assert loss == 0
    assert parts["local_difference_cases_without_pairs"] == 1
    assert parts["local_difference_pair_count"] == 0
    loss.backward()
    assert torch.equal(pred.grad, torch.zeros_like(pred))


@pytest.mark.parametrize("kind", ["local", "ranking"])
def test_pair_losses_cover_every_case_with_per_case_cap_and_are_case_order_invariant(kind):
    batch = _batch()
    pred = (batch["y"] * 0.25).requires_grad_()
    parts = {}

    def loss_fn(p, data, diagnostics=None):
        if kind == "local":
            return _local(p, data, max_pairs=1, diagnostics=diagnostics)
        return pairwise_ranking_loss(p, data["y"], data["batch"], max_pairs=1,
                                     diagnostics=diagnostics)

    value = loss_fn(pred, batch, parts)
    prefix = "local_difference" if kind == "local" else "pairwise_ranking"
    assert parts[f"{prefix}_pair_count"] == 2
    assert parts[f"{prefix}_min_pairs_per_case"] == 1
    assert parts[f"{prefix}_max_pairs_per_case"] == 1
    value.backward()
    assert pred.grad[:4].abs().sum() > 0
    assert pred.grad[4:].abs().sum() > 0
    order = torch.tensor([4, 5, 6, 7, 0, 1, 2, 3])
    reordered = {key: val[order] for key, val in batch.items()}
    reordered["batch"] = 1 - reordered["batch"]
    assert loss_fn(pred.detach()[order], reordered) == pytest.approx(value.item())


@pytest.mark.parametrize("kind", ["local", "ranking"])
def test_losses_average_cases_not_pairs(kind):
    batch = {key: val[:6] for key, val in _batch().items()}
    pred = batch["y"] * 0.25

    def call(p, data):
        if kind == "local":
            return _local(p, data)
        return pairwise_ranking_loss(p, data["y"], data["batch"])

    solo = [call(pred[sl], {key: val[sl] for key, val in batch.items()})
            for sl in (slice(0, 4), slice(4, 6))]
    assert call(pred, batch) == pytest.approx(torch.stack(solo).mean().item())


def test_rank_correct_order_beats_constant_and_reversed_order():
    batch = _batch()
    target, ids = batch["y"], batch["batch"]
    good = pairwise_ranking_loss(target, target, ids)
    flat = pairwise_ranking_loss(torch.zeros_like(target), target, ids)
    bad = pairwise_ranking_loss(-target, target, ids)
    assert good < flat < bad


def test_rank_target_ties_are_explicitly_diagnosed():
    target = torch.zeros(8)
    pred = torch.zeros(8, requires_grad=True)
    parts = {}
    with pytest.warns(RuntimeWarning, match="2/2 cases have no eligible pairs"):
        loss = pairwise_ranking_loss(pred, target, _batch()["batch"], diagnostics=parts)
    assert loss == 0
    assert parts["pairwise_ranking_cases_without_pairs"] == 2
    loss.backward()
    assert torch.equal(pred.grad, torch.zeros_like(pred))


def test_rank_filters_ties_before_limiting_each_case():
    target = torch.tensor([0., 0., 0., 2., 0., 0., 0., 3.])
    parts = {}
    loss = pairwise_ranking_loss(torch.zeros(8), target, _batch()["batch"],
                                 max_pairs=1, diagnostics=parts)
    assert torch.isfinite(loss)
    assert parts["pairwise_ranking_min_pairs_per_case"] == 1


def test_compute_loss_adds_both_weighted_components_and_pair_diagnostics():
    cfg = C.TrainConfig(loss="mse")
    cfg.loss_local_diff_lambda = 0.05
    cfg.local_diff_min_distance_mm = 2.0
    cfg.local_diff_max_distance_mm = 5.0
    cfg.loss_pairwise_rank_lambda = 0.05
    cfg.pairwise_rank_margin = 0.05
    batch = _batch()
    pred = (batch["y"] * 0.25).view(-1, 1).requires_grad_()
    parts = {}
    loss = compute_loss(pred, batch, cfg, "cpu", components=parts)
    expected = (parts["base"] + parts["local_difference_weighted"]
                + parts["pairwise_ranking_weighted"])
    assert torch.equal(loss.detach(), expected)
    assert parts["local_difference_pair_count"] == 6
    assert parts["pairwise_ranking_pair_count"] == 4
    loss.backward()
    assert torch.isfinite(pred.grad).all()
