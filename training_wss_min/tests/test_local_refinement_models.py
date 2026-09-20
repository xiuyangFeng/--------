import torch
import pytest
from training_wss_min.baseline_models import LocalWallBranch, PointNetPlusPlusRegressor
from training_wss_min.local_refinement import (
    tangent_frame, DirectionalNeighborhoodRefinement, LocalPatchFiLMRefinement,
    MultiStatisticPool, FixedMmDualScale, geometry_neighborhood,
)


@pytest.fixture(autouse=True)
def limited_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def geometry(n, scale=100., radius=8.):
    result = torch.zeros(n, 8)
    result[:, 2] = 1.
    result[:, 3] = 1.
    result[:, 6] = radius
    result[:, 7] = scale
    return result


def small_model(**overrides):
    options = dict(in_dim=25, width=8, sa_ratios=(0.5, 0.5), sa_radius=(0.08, 0.15),
                   sa_nsample=(8, 8), sa_center_counts=(12, 6), fp_knn=3,
                   head_hidden=16, local_branch=True, local_branch_channels=8,
                   local_branch_feature_indices=(3, 4, 5))
    options.update(overrides)
    torch.manual_seed(1234)
    return PointNetPlusPlusRegressor(**options)


def model_inputs():
    generator = torch.Generator().manual_seed(11)
    pos = torch.rand(48, 3, generator=generator) * .1
    x = torch.randn(48, 25, generator=generator)
    batch = torch.arange(2).repeat_interleave(24)
    qpos = pos[::2].clone() + .001
    qx = x[::2].clone()
    qbatch = batch[::2].clone()
    patch = {"features": torch.randn(24, 16, 25, generator=generator),
             "relative_mm": torch.randn(24, 16, 3, generator=generator)}
    return (pos, x, batch), (qpos, qx, qbatch), geometry(48), patch


def cloud(n=12):
    torch.manual_seed(4)
    pos = torch.rand(n, 3)
    batch = torch.tensor([0] * (n // 2) + [1] * (n - n // 2))
    x = torch.randn(n, 8)
    tangent = torch.tensor([[1., 0., 0.]]).repeat(n, 1)
    normal = torch.tensor([[0., 0., 1.]]).repeat(n, 1)
    return pos, x, batch, tangent, normal


def test_tangent_frame_is_orthonormal():
    _, _, _, t, n = cloud()
    a, c, un = tangent_frame(t, n)
    assert torch.allclose(a.norm(dim=-1), torch.ones(a.size(0)), atol=1e-5)
    assert torch.allclose(c.norm(dim=-1), torch.ones(c.size(0)), atol=1e-5)
    assert torch.max((a * un).abs()) < 1e-5
    assert torch.max((c * un).abs()) < 1e-5


def test_directional_branch_zero_initialized_and_finite():
    pos, x, batch, t, n = cloud()
    m = DirectionalNeighborhoodRefinement(8, k=4)
    y = m(pos, x, batch, t, n)
    assert y.shape == x.shape and torch.isfinite(y).all()
    assert torch.allclose(y, torch.zeros_like(y))
    assert 0 < m.last_edge_keep_fraction <= 1


def test_query_patch_film_shape_and_zero_output():
    pos, x, batch, _, _ = cloud()
    qpos, qx, qb = pos[:7] + 0.01, x[:7], batch[:7]
    m = LocalPatchFiLMRefinement(8, 8, context_channels=8, out_dim=1, k=3)
    y = m(pos, x, batch, qpos, qx, qb, context=x)
    assert y.shape == (7, 1) and torch.isfinite(y).all()
    assert torch.allclose(y, torch.zeros_like(y))


def test_multistat_pool_and_fixed_mm_branch():
    pos, x, batch, _, _ = cloud()
    row = torch.arange(pos.size(0))
    p = MultiStatisticPool(8)
    y = p(x, row, pos.size(0))
    assert y.shape == x.shape and torch.isfinite(y).all()
    f = FixedMmDualScale(8)
    z = f(pos, x, batch, torch.tensor([100., 120.]))
    assert z.shape == x.shape and torch.allclose(z, torch.zeros_like(z))


@pytest.mark.parametrize("all_equal", [False, True])
def test_multistat_backward_is_finite_for_singletons_and_zero_variance(all_equal):
    values = (torch.ones(7, 8) if all_equal else torch.randn(7, 8)).requires_grad_()
    row = torch.tensor([0, 1, 1, 2, 2, 2, 3])
    pool = MultiStatisticPool(8)
    # Test with std path active as well as the parent-compatible initialization.
    for active in (False, True):
        pool.zero_grad()
        values.grad = None
        if active:
            with torch.no_grad():
                pool.proj.weight[:, 16:].fill_(0.1)
        pool(values, row, 4).square().mean().backward()
        assert torch.isfinite(values.grad).all()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in pool.parameters())


def test_directional_grouping_reserves_all_four_wall_directions_and_rejects_other_sheet():
    # The many close +axial points must not crowd out the other three sectors.
    plus = torch.stack((torch.linspace(.001, .009, 20), torch.zeros(20), torch.zeros(20)), -1)
    pos = torch.cat((torch.zeros(1, 3), plus,
                     torch.tensor([[-.02, 0, 0], [0, .02, 0], [0, -.02, 0], [0, 0, .001]])))
    geom = geometry(pos.size(0))
    geom[-1, :3] *= -1
    row, col, local = geometry_neighborhood(pos, torch.zeros(pos.size(0), dtype=torch.long),
                                          geom, .03, 16, directional=True)
    center_neighbors = col[row == 0].tolist()
    assert {21, 22, 23}.issubset(center_neighbors)
    assert 24 not in center_neighbors
    assert len(center_neighbors) == 16
    assert torch.allclose(local[row == 0, 2], torch.zeros((row == 0).sum()))


def test_directional_candidate_budget_respects_torch_cluster_cuda_limit(monkeypatch):
    from training_wss_min import local_refinement
    original = local_refinement.knn
    called = []

    def checked(*args, **kwargs):
        called.append(kwargs["k"])
        assert kwargs["k"] <= 100, "torch-cluster CUDA kNN rejects k > 100"
        return original(*args, **kwargs)

    monkeypatch.setattr(local_refinement, "knn", checked)
    pos = torch.rand(160, 3) * .01
    pos[:, 2] = 0
    row, _, _ = geometry_neighborhood(pos, torch.zeros(160, dtype=torch.long),
                                     geometry(160), .03, 16, directional=True)
    assert called == [100]
    assert torch.bincount(row).max() == 16


def test_physical_neighborhood_is_scale_invariant_and_uses_local_radius():
    pos_mm = torch.tensor([[0., 0, 0], [2., 0, 0], [4., 0, 0], [8., 0, 0]])
    batch = torch.zeros(4, dtype=torch.long)
    first = geometry_neighborhood(pos_mm / 100, batch, geometry(4, scale=100), 2.5, 16, physical=True)
    second = geometry_neighborhood(pos_mm / 250, batch, geometry(4, scale=250), 2.5, 16, physical=True)
    for a, b in zip(first, second):
        torch.testing.assert_close(a, b)
    assert set(first[1][first[0] == 0].tolist()) == {0, 1}
    radius_by_point = torch.tensor([5., 1., 1., 1.])
    row, col, _ = geometry_neighborhood(pos_mm / 100, batch, geometry(4), radius_by_point, 16, physical=True)
    assert set(col[row == 0].tolist()) == {0, 1, 2}
    assert set(col[row == 1].tolist()) == {1}


@pytest.mark.parametrize("options", [
    {"local_branch_directional": True},
    {"local_branch_pool": "multistat"},
    {"local_branch_radius_mode": "physical_dual"},
    {"query_patch_film": True},
    {"local_branch_directional": True, "query_patch_film": True},
])
def test_enabled_model_paths_have_finite_nonzero_gradients(options):
    support, query, geom, patch = model_inputs()
    model = small_model(**options).train()
    # Open zero-initialized gates to exercise the entire path in one backward.
    with torch.no_grad():
        model.local_wall_branch.fuse.weight.fill_(.02)
        if model.query_patch is not None:
            model.query_patch.out.weight.fill_(.02)
    prediction = model.forward_support_query(*support, *query,
                                             support_geometry=geom, query_patch=patch)
    assert prediction.shape == (24,)
    loss = (prediction - torch.linspace(-1, 1, 24)).square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.local_wall_branch.scales[0][0].weight.grad.abs().sum() > 0
    if model.query_patch is not None:
        assert model.query_patch.patch[0].weight.grad.abs().sum() > 0
        assert model.query_patch.to_gamma.weight.grad.abs().sum() > 0


def test_query_full_wall_patch_and_coarse_context_affect_prediction_without_chunk_dependence():
    support, query, geom, patch = model_inputs()
    model = small_model(query_patch_film=True).eval()
    with torch.no_grad():
        model.query_patch.out.weight.normal_()
        encoded = model.encode_support(*support, geometry=geom)
        full = model.decode_query(encoded, *query, patch=patch)
        chunks = []
        for start in range(0, query[0].size(0), 5):
            chunks.append(model.decode_query(encoded, *(v[start:start + 5] for v in query),
                                             patch={k: v[start:start + 5] for k, v in patch.items()}))
        torch.testing.assert_close(full, torch.cat(chunks), atol=2e-6, rtol=1e-5)
        changed_patch = dict(patch, features=patch["features"] + 3.)
        changed = model.decode_query(encoded, *query, patch=changed_patch)
        assert not torch.allclose(full, changed)
        changed_context = (*encoded[:5], encoded[5] + 2.)
        context_prediction = model.decode_query(changed_context, *query, patch=patch)
        assert not torch.allclose(full, context_prediction)
    with pytest.raises(ValueError, match="full-wall patch"):
        model.decode_query(encoded, *query)


def test_new_features_preserve_parent_weights_rng_and_initial_query_output():
    support, query, geom, patch = model_inputs()
    parent = small_model().eval()
    rng = torch.get_rng_state().clone()
    child = small_model(local_branch_pool="multistat", query_patch_film=True).eval()
    assert torch.equal(rng, torch.get_rng_state())
    for key, value in parent.state_dict().items():
        torch.testing.assert_close(value, child.state_dict()[key], atol=0, rtol=0)
    with torch.no_grad():
        expected = parent.forward_support_query(*support, *query)
        actual = child.forward_support_query(*support, *query, support_geometry=geom, query_patch=patch)
    torch.testing.assert_close(expected, actual, atol=0, rtol=0)


@pytest.mark.parametrize("options", [{"local_branch_directional": True}, {"local_branch_radius_mode": "physical_dual"}])
def test_enabled_geometric_models_require_raw_geometry(options):
    support, _, _, _ = model_inputs()
    model = small_model(**options).eval()
    with pytest.raises(ValueError, match="geometry"):
        model.encode_support(*support)
