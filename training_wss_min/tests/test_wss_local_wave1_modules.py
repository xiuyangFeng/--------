"""Wave-1 (wss_local_wave1_20260912) modules: exact no-op at init, wiring, losses, paired init."""
from __future__ import annotations

import copy
import json

import numpy as np
import pytest
import torch

from training_wss_min import config as C
from training_wss_min.baseline_models import PointNetPlusPlusRegressor
from training_wss_min.local_refinement import LocalPatchFiLMRefinement, MoEHead, SectionContext
from training_wss_min.objectives import compute_loss


@pytest.fixture(autouse=True)
def limited_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def geometry(n, scale=100.0, radius=8.0):
    g = torch.zeros(n, 8)
    g[:, 2] = 1.0   # normal +z
    g[:, 3] = 1.0   # tangent +x
    g[:, 6] = radius
    g[:, 7] = scale
    return g


def section(n, seed=3):
    gen = torch.Generator().manual_seed(seed)
    seg = torch.randint(0, 7, (n,), generator=gen).float()
    s = torch.rand(n, generator=gen) * 120.0
    return torch.stack((seg, s), dim=1)


def inputs(n_support=48, n_query=24, in_dim=25, k=16, seed=11):
    gen = torch.Generator().manual_seed(seed)
    pos = torch.rand(n_support, 3, generator=gen) * 0.1
    x = torch.randn(n_support, in_dim, generator=gen)
    batch = torch.arange(2).repeat_interleave(n_support // 2)
    qpos = pos[::2].clone() + 0.001
    qx = x[::2].clone()
    qbatch = batch[::2].clone()
    patch = {"features": torch.randn(n_query, k, in_dim, generator=gen),
             "relative_mm": torch.randn(n_query, k, 3, generator=gen)}
    return (pos, x, batch), (qpos, qx, qbatch), geometry(n_support), geometry(n_query), patch


def base_options(**overrides):
    options = dict(in_dim=25, width=8, sa_ratios=(0.5, 0.5), sa_radius=(0.08, 0.15),
                   sa_nsample=(8, 8), sa_center_counts=(12, 6), fp_knn=3, head_hidden=16,
                   local_branch=True, local_branch_channels=8, local_branch_feature_indices=(14, 15, 16),
                   local_branch_directional=True, query_patch_film=True, query_patch_k=16,
                   query_patch_hidden=16)
    options.update(overrides)
    return options


def build(**overrides):
    torch.manual_seed(1234)
    return PointNetPlusPlusRegressor(**base_options(**overrides))


def run(model, **extra):
    (pos, x, batch), (qpos, qx, qbatch), sg, qg, patch = inputs()
    model.eval()
    with torch.no_grad():
        return model.forward_support_query(pos, x, batch, qpos, qx, qbatch, support_geometry=sg,
                                           query_geometry=qg, query_patch=patch, **extra)


def test_parent_bit_identical_when_new_options_are_off():
    parent = build()
    same = build(query_patch_frame="atlas", query_patch_pool="mean", head_moe_experts=0,
                 direction_head=False, section_context=False)
    for (ka, a), (kb, b) in zip(parent.state_dict().items(), same.state_dict().items()):
        assert ka == kb and torch.equal(a, b)


@pytest.mark.parametrize("overrides", [
    dict(query_patch_pool="attention"),
    dict(query_patch_frame="tangent"),
    dict(query_patch_frame="tangent", query_patch_pool="attention", query_patch_k=16),
    dict(head_moe_experts=4),
    dict(section_context=True, section_context_bin_mm=4.0, section_context_max_bins=32,
         section_context_layers=1, section_context_heads=2, section_context_hidden=16),
])
def test_new_modules_are_exact_noops_at_init(overrides):
    parent = build()
    child = build(**overrides)
    parent_state = parent.state_dict()
    child_state = child.state_dict()
    # shared tensors identical (construction order preserved by fork_rng)
    for key, tensor in parent_state.items():
        assert key in child_state and torch.equal(child_state[key], tensor), key
    extra = {}
    if overrides.get("section_context"):
        extra = dict(support_section=section(48), query_section=section(24, seed=5))
    out_parent = run(parent)
    out_child = run(child, **extra)
    assert out_parent.shape == out_child.shape
    if overrides.get("query_patch_frame") == "tangent":
        # relative coordinates change but the zero output projection keeps the residual at 0
        assert torch.allclose(out_parent, out_child, atol=1e-6)
    else:
        assert torch.allclose(out_parent, out_child, atol=1e-6)


def test_moe_head_gate_uniform_at_init_and_trainable():
    head = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 1))
    moe = MoEHead(head, 25, 3, hidden=8)
    x = torch.randn(10, 8)
    inp = torch.randn(10, 25)
    assert torch.allclose(moe(x, inp), head(x), atol=1e-6)
    weights = torch.softmax(moe.gate(inp), dim=-1)
    assert torch.allclose(weights, torch.full_like(weights, 1 / 3))
    moe(x, inp).sum().backward()
    assert moe.gate[-1].weight.grad is not None


def test_direction_head_adds_channels_and_keeps_channel_zero():
    parent = build()
    child = build(direction_head=True, direction_head_hidden=8)
    out_parent = run(parent)
    out_child = run(child)
    assert out_parent.ndim == 1 and out_child.shape == (out_parent.numel(), 3)
    assert torch.allclose(out_child[:, 0], out_parent, atol=1e-6)


def test_direction_loss_uses_only_extra_channels():
    cfg = C.TrainConfig(loss="mse", loss_direction_lambda=0.5, loss_pinball_lambda=0.0)
    n = 12
    y = torch.randn(n)
    pred = torch.zeros(n, 3)
    pred[:, 0] = y
    pred[:, 1] = 1.0
    y_dir = torch.zeros(n, 2)
    y_dir[:, 0] = 1.0     # perfectly aligned axial direction
    y_dir[-2:] = 0.0      # two invalid rows must be masked
    batch = {"y": y, "y_raw": torch.exp(y), "y_dir": y_dir, "batch": torch.zeros(n, dtype=torch.long),
             "curv": torch.zeros(n), "invr": torch.ones(n)}
    components = {}
    loss = compute_loss(pred, batch, cfg, "cpu", None, components=components)
    assert float(loss) < 1e-6 and abs(float(components["direction_valid_fraction"]) - 10 / 12) < 1e-6
    pred_bad = pred.clone()
    pred_bad[:, 1] = -1.0
    loss_bad = compute_loss(pred_bad, batch, cfg, "cpu", None)
    assert abs(float(loss_bad) - 0.5 * 2.0) < 1e-6


def test_section_context_lookup_falls_back_to_nearest_occupied_bin():
    module = SectionContext(4, 3, 5, bin_mm=10.0, max_bins=4, layers=1, heads=1, hidden=8)
    x = torch.randn(6, 3)
    inp = torch.randn(6, 4)
    sec = torch.tensor([[0., 5.], [0., 5.], [0., 35.], [1., 0.], [1., 0.], [1., 12.]])
    batch = torch.zeros(6, dtype=torch.long)
    tokens, occupied = module(x, inp, sec, batch, 1)
    assert tokens.shape == (1, 28, 5) and occupied.view(7, 4)[0].tolist() == [True, False, False, True]
    assert torch.count_nonzero(tokens) == 0  # zero-init output
    with torch.no_grad():
        module.out.weight.fill_(0.1)
        module.out.bias.fill_(0.2)
    tokens, occupied = module(x, inp, sec, batch, 1)
    query = torch.tensor([[0., 15.], [0., 39.], [2., 3.]])   # bin1 (empty -> bin0), bin3 (occupied), unseen segment
    out = module.lookup(tokens, occupied, query, torch.zeros(3, dtype=torch.long))
    assert torch.allclose(out[0], tokens[0, 0]) and torch.allclose(out[1], tokens[0, 3])
    assert torch.count_nonzero(out[2]) == 0


def test_tangent_frame_patch_rotates_offsets_into_query_frame():
    module = LocalPatchFiLMRefinement(4, 4, context_channels=6, out_dim=1, k=2, hidden=8, frame="tangent")
    rel = torch.tensor([[[1., 0., 0.], [0., 1., 0.]]])
    g = torch.zeros(1, 8)
    g[0, :3] = torch.tensor([0., 0., 1.])   # normal +z
    g[0, 3:6] = torch.tensor([0., 1., 0.])  # tangent +y -> axial +y, circumferential = n x a = -x
    local = module._relative(rel, g)
    assert torch.allclose(local[0, 0], torch.tensor([0., -1., 0., 1.]))
    assert torch.allclose(local[0, 1], torch.tensor([1., 0., 0., 1.]))


def test_config_contracts_for_wave1():
    raw = json.loads((C.PROJECT_ROOT / "training_wss_min/configs/wss_direct_recovery_20260912/C1_s1234.json").read_text())
    C.ExpConfig.from_dict(raw)
    bad = copy.deepcopy(raw); bad["train"]["ema_decay"] = 1.0
    with pytest.raises(ValueError):
        C.ExpConfig.from_dict(bad)
    bad = copy.deepcopy(raw); bad["model"]["head_moe_experts"] = -1
    with pytest.raises(ValueError):
        C.ExpConfig.from_dict(bad)
    ok = copy.deepcopy(raw)
    ok["data"]["point_features_root"] = [raw["data"]["point_features_root"],
                                         str(C.PROJECT_ROOT / "data_wss_v5/views/wss_min_flowref_v1")]
    ok["data"]["input_features"] += ["bend_cos", "carina_cos", "log_tau0_murray", "atlas_prior_logwss"]
    cfg = C.ExpConfig.from_dict(ok)
    assert C.v6_point_features(cfg)[-4:] == ("bend_cos", "carina_cos", "log_tau0_murray", "atlas_prior_logwss")


def test_flowref_sidecar_rows_align_with_bundle():
    root = C.PROJECT_ROOT / "data_wss_v5/views/wss_min_flowref_v1"
    bundle_root = C.PROJECT_ROOT / "data_wss_v5/views/wss_min_view_v1"
    if not root.is_dir():
        pytest.skip("flow-reference sidecar not built")
    case = "AG/fast/WANG_CHUN_MING"
    with np.load(root / case / "features.npz") as z, np.load(bundle_root / case / "bundle.npz", allow_pickle=True) as b:
        assert np.array_equal(z["wall_node_id_cas"], b["wall_node_id_cas"])
        for name in C.V7_FLOW_FEATURE_KEYS:
            values = z[f"wall_{name}"]
            assert values.shape == (len(b["wall_node_id_cas"]),) and np.isfinite(values).all(), name
        assert np.abs(z["wall_bend_cos"]).max() <= 1.0 + 1e-6 and np.abs(z["wall_carina_cos"]).max() <= 1.0 + 1e-6
        assert (z["wall_bif_plane_cos"] >= 0).all() and (z["wall_up_min_r_ratio_5d"] <= 1.0 + 1e-5).all()
        assert bool(z["wall_prior_is_loo"]) is True


# ---- wave 2: residual target, volume Murray extension, expandable layouts ----
def test_residual_target_round_trip_matches_standard_logz():
    from training_wss_min import dataset as D
    stats = {"method": "log_z", "eps": 1e-6, "log": {"mean": 0.5, "std": 1.3},
             "offset": {"feature": "log_tau0_murray", "mean": 7.1, "std": 0.9}}
    rng = np.random.default_rng(0)
    wss = rng.lognormal(0.5, 1.0, 50)
    offset = rng.normal(-6.5, 1.0, 50)
    z_res = D.normalize_wss(wss, stats, offset=offset)
    back = D.residual_to_standard_logz(z_res, offset, stats)
    np.testing.assert_allclose(back, D.standard_logz(wss, stats), rtol=1e-10, atol=1e-10)
    with pytest.raises(ValueError):
        D.normalize_wss(wss, stats)


def test_expandable_layouts_cover_volume_decoder_and_bottleneck():
    from training_wss_min.paired_initialization import _expandable_keys, _expand_columns
    n_ref, n_new = 18, 20
    reference = {"stem.0.weight": torch.randn(32, n_ref), "qad_local.0.weight": torch.randn(32, n_ref + 4),
                 "bottleneck.pos.0.weight": torch.randn(256, n_ref), "sa.2.contexts.0.pos.0.weight": torch.randn(256, n_ref),
                 "head.0.weight": torch.randn(64, 32)}
    state = {"stem.0.weight": torch.zeros(32, n_new), "qad_local.0.weight": torch.zeros(32, n_new + 4),
             "bottleneck.pos.0.weight": torch.zeros(256, n_new), "sa.2.contexts.0.pos.0.weight": torch.zeros(256, n_new),
             "head.0.weight": torch.zeros(64, 32)}
    layouts = _expandable_keys(reference, state, n_ref, n_new)
    assert set(layouts) == {"stem.0.weight", "qad_local.0.weight", "bottleneck.pos.0.weight", "sa.2.contexts.0.pos.0.weight"}
    expanded = _expand_columns(reference["qad_local.0.weight"], state["qad_local.0.weight"].shape, layouts["qad_local.0.weight"])
    assert torch.equal(expanded[:, :n_ref], reference["qad_local.0.weight"][:, :n_ref])
    assert torch.equal(expanded[:, n_new:], reference["qad_local.0.weight"][:, n_ref:])
    assert torch.count_nonzero(expanded[:, n_ref:n_new]) == 0


def test_volume_murray_features_broadcast_by_segment():
    from training_wss_min.dataset import _attach_volume
    n_wall, n_vol = 6, 5
    case = {"pos": np.zeros((n_wall, 3), np.float32), "y_raw": np.ones(n_wall, np.float32), "y_norm": np.ones(n_wall, np.float32),
            "local_radius": np.full(n_wall, 8.0, np.float32), "coord_scale_scalar": 100.0,
            "_wall_segment_id": np.array([0, 0, 1, 1, 2, 2]),
            "log_q_branch_murray": np.log(np.array([1, 1, .5, .5, .3, .3], np.float32)),
            "log_tau0_murray": (np.log(np.array([1, 1, .5, .5, .3, .3])) - 3 * np.log(8.0)).astype(np.float32)}
    vol = {"vol_coords_norm": np.zeros((n_vol, 3), np.float32), "vol_radial_aligned": np.zeros((n_vol, 3), np.float32),
           "vol_dist_to_wall_mm": np.ones(n_vol, np.float32), "vol_pressure_rel_peak": np.zeros(n_vol, np.float32),
           "vol_segment_id": np.array([2, 1, 0, 2, 1]), "vol_local_radius": np.array([4., 5., 10., 4., 5.], np.float32),
           "p_ref_pa": 0.0}
    stats = {"linear": {"mean": 0.0, "std": 1.0}}
    _attach_volume(case, vol, "pressure_mixed", stats)
    q = case["log_q_branch_murray"]
    np.testing.assert_allclose(q[n_wall:], np.log([.3, .5, 1, .3, .5]), rtol=1e-6)
    tau = case["log_tau0_murray"]
    np.testing.assert_allclose(tau[n_wall:], np.log([.3, .5, 1, .3, .5]) - 3 * np.log([4., 5., 10., 4., 5.]), rtol=1e-6)
    assert len(q) == n_wall + n_vol


def test_wave2_config_contracts():
    raw = json.loads((C.PROJECT_ROOT / "training_wss_min/configs/wss_local_wave1_20260912/X5_s1234.json").read_text())
    ok = copy.deepcopy(raw); ok["data"]["target_log_offset_feature"] = "log_tau0_murray"
    C.ExpConfig.from_dict(ok)
    bad = copy.deepcopy(raw); bad["data"]["target_log_offset_feature"] = "bend_cos"   # not in input_features
    with pytest.raises(ValueError):
        C.ExpConfig.from_dict(bad)
    vol = json.loads((C.PROJECT_ROOT / "training_wss_min/configs/volume_attention_20260910/P02_s1234.json").read_text())
    vol_ok = copy.deepcopy(vol); vol_ok["data"]["input_features"] += ["log_q_branch_murray", "log_tau0_murray"]
    vol_ok["data"]["point_features_root"] = str(C.PROJECT_ROOT / "data_wss_v5/views/wss_min_flowref_v1")
    C.ExpConfig.from_dict(vol_ok)
    vol_bad = copy.deepcopy(vol_ok); vol_bad["data"]["input_features"].append("bend_cos")
    with pytest.raises(ValueError):
        C.ExpConfig.from_dict(vol_bad)
