"""Scientific controls for the isolated difficult-phase velocity arms."""
import copy

import pytest
import torch

from training_wss_min.joint_cycle_round2_model import build_joint_cycle_model, _count
from training_wss_min.velocity_phase_model import (
    AnchorOperator, VELOCITY_PHASE_ARMS, build_velocity_phase_model)


OPTIONS = dict(geometry=dict(width=8, sa_center_counts=(8, 4), sa_ratios=(.5, .5),
                            sa_radius=(2., 3.), sa_nsample=(6, 4)),
               head_hidden=16, round2_module="query_mean", round2_hidden=8,
               round2_heads=2, round2_neighbors=8, frame_hidden=12,
               graph_anchor_count=8, graph_neighbors=3, graph_hidden=12,
               graph_layers=2, graph_activation_checkpointing=True)
STATS = dict(dimensions=dict(support=6, velocity=5), bc_dims=2, phase_dims=4,
             targets=dict(velocity=dict(std=[[.01 + .001 * f, .04, .08] for f in range(8)])))


def build(arm, **options):
    return build_velocity_phase_model(
        dict(arm=arm, seed=23, variant="I", tasks=["velocity"], model=dict(OPTIONS, **options)), STATS)


@pytest.fixture
def cloud():
    torch.set_num_threads(1)
    torch.manual_seed(79)
    pos, x = torch.randn(40, 3) * .1, torch.randn(40, 6)
    batch = torch.arange(2).repeat_interleave(20)
    qpos, qx = torch.randn(6, 3) * .1, torch.randn(6, 5)
    frame = torch.eye(3).expand(6, 3, 3).clone()
    frame[0] = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    valid = torch.ones(6, 1)
    valid[1] = 0
    geom = torch.cat((frame.flatten(1), valid), -1)
    q = dict(pos=qpos, x=qx, batch=torch.arange(2).repeat_interleave(3),
             frame=frame, geom_extra=geom)
    apos, ax = torch.randn(16, 3) * .1, torch.randn(16, 5)
    target = torch.arange(16).repeat_interleave(3)
    source = torch.cat([torch.tensor([((i + j) % 8) + 8 * (i // 8) for j in (1, 2, 3)])
                        for i in range(16)])
    rel = apos[source] - apos[target]
    edge = torch.cat((rel, rel.norm(dim=-1, keepdim=True), torch.randn(len(rel), 1),
                      torch.randint(2, (len(rel), 2)).float()), -1)
    anchors = dict(pos=apos, x=ax, batch=torch.arange(2).repeat_interleave(8),
                   geom_extra=torch.cat((torch.eye(3).flatten().expand(16, 9), torch.ones(16, 1)), -1),
                   edge_index=torch.stack((target, source)), edge_attr=edge)
    return dict(pos=pos, x=x, batch=batch, q=q, phase=torch.randn(8, 4),
                bc=torch.randn(2, 2), anchors=anchors)


def run(model, cloud, **kwargs):
    return model(cloud["pos"], cloud["x"], cloud["batch"], {"velocity": cloud["q"]},
                 cloud["phase"], bc=cloud["bc"], anchors=cloud["anchors"], evaluation=True,
                 **kwargs)["velocity"]


@pytest.mark.parametrize("arm", VELOCITY_PHASE_ARMS)
def test_common_initialization_rng_and_exact_u0_function(cloud, arm):
    baseline = build_joint_cycle_model("I", 6, {"velocity": 5}, seed=23,
        active_tasks=["velocity"], model_options=dict(OPTIONS, bc_dim=2, phase_dim=4)).eval()
    before_rng = torch.random.get_rng_state().clone()
    model = build(arm).eval()
    assert torch.equal(before_rng, torch.random.get_rng_state())
    for name, value in baseline.state_dict().items():
        torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
    with torch.no_grad():
        expected = baseline(cloud["pos"], cloud["x"], cloud["batch"], {"velocity": cloud["q"]},
                            cloud["phase"], bc=cloud["bc"], evaluation=True)["velocity"]
        result = run(model, cloud)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    c = model.velocity_phase_contract
    assert c["total_parameters"] == _count(model)
    assert c["base_parameters"] == _count(baseline)
    assert c["added_parameters"] == _count(model) - _count(baseline)
    if arm.startswith("D"):
        assert c["added_parameters"] == 0


def test_physical_rotation_scale_and_invalid_frame_fallback(cloud):
    a0, a1 = build("A0"), build("A1")
    for name, param in a0.frame_residual.named_parameters():
        torch.testing.assert_close(dict(a1.frame_residual.named_parameters())[name], param, rtol=0, atol=0)
    r = torch.tensor([1., 2., -3.])[None, None].expand(6, 8, 3)
    phases = torch.arange(8)
    q = cloud["q"]
    xyz = a0.physical_residual_to_z(r, q["frame"], q["geom_extra"], phases)
    local = a1.physical_residual_to_z(r, q["frame"], q["geom_extra"], phases)
    std, scale = a0.velocity_target_std, a0.velocity_residual_scale
    torch.testing.assert_close(xyz * std, r * scale)
    torch.testing.assert_close(local[0] * std, torch.tensor([-2., 1., -3.]) * scale)
    torch.testing.assert_close(local[1], xyz[1], rtol=0, atol=0)
    # Permuted/subset phases must use their original XYZ scales.
    chosen = torch.tensor([6, 1, 4])
    chosen_z = a1.physical_residual_to_z(r[:, :3], q["frame"], q["geom_extra"], chosen)
    torch.testing.assert_close(chosen_z, local[:, chosen])


def test_family_active_parameter_matching_and_identical_a0_path():
    family = [build(arm) for arm in ("G00", "G01", "G10", "G11")]
    counts = []
    ref_frame = family[0].frame_residual.state_dict()
    ref_readout = family[0].anchor_readout.state_dict()
    for model in family:
        c = model.velocity_phase_contract
        assert c["anchor_parameters"] == _count(model.anchor_operator)
        assert c["relative_parameter_mismatch"] <= .05
        counts.append(c["graph_added_parameters"])
        assert c["graph_added_parameters"] == _count(model.anchor_operator) + _count(model.anchor_readout)
        assert c["graph_family_relative_parameter_spread"] <= .05
        for name, value in ref_frame.items():
            torch.testing.assert_close(model.frame_residual.state_dict()[name], value, rtol=0, atol=0)
        for name, value in ref_readout.items():
            torch.testing.assert_close(model.anchor_readout.state_dict()[name], value, rtol=0, atol=0)
        assert model.anchor_operator.condition.in_features == (6 if model.dynamic_anchors else 2)
    assert (max(counts) - min(counts)) / max(counts) <= .05
    for graph in (False, True):
        module = AnchorOperator(25, 4, 12, 8, graph=graph)
        assert _count(module) == AnchorOperator.parameter_count(25, 4, 12, 8, graph=graph)


@pytest.mark.parametrize("arm", ["G00", "G01", "G10", "G11"])
def test_phase_and_bc_factorial_control(cloud, arm):
    model = build(arm).eval()
    with torch.no_grad():
        encoded = model.encode(cloud["pos"], cloud["x"], cloud["batch"],
                               anchors=cloud["anchors"], evaluation=True)
        prepare = lambda p, b: model.prepare_phase(encoded, p, b)["anchor_features"]
        before = prepare(cloud["phase"], cloud["bc"])
        phase_changed = prepare(cloud["phase"] + 2, cloud["bc"])
        bc_changed = prepare(cloud["phase"], cloud["bc"] + 2)
        if model.dynamic_anchors:
            assert not torch.allclose(before, phase_changed)
        else:
            torch.testing.assert_close(before, phase_changed, rtol=0, atol=0)
        assert not torch.allclose(before, bc_changed)


@pytest.mark.parametrize("arm", ["A0", "A1", "G00", "G01", "G10", "G11"])
def test_nonzero_residual_gradients_and_query_phase_chunks(cloud, arm):
    model = build(arm).eval()
    run(model, cloud).square().mean().backward()
    assert model.frame_residual.output.weight.grad.abs().sum() > 0
    if model.has_anchors:
        assert model.anchor_readout.output.weight.grad.abs().sum() > 0
    with torch.no_grad():
        model.frame_residual.output.weight.normal_(std=.05)
        if model.has_anchors:
            model.anchor_readout.output.weight.normal_(std=.05)
    model.zero_grad(set_to_none=True)
    run(model, cloud).square().mean().backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name
        # Every extra parameter participates after the zero readout opens.
        if name.startswith(("frame_residual.", "anchor_operator.", "anchor_readout.")):
            assert param.grad.abs().sum() > 0, name
    with torch.no_grad():
        encoded = model.encode(cloud["pos"], cloud["x"], cloud["batch"],
                               anchors=cloud["anchors"], evaluation=True)
        state = model.prepare_phase(encoded, cloud["phase"], cloud["bc"], phase_chunk_size=8)
        short_state = model.prepare_phase(encoded, cloud["phase"], cloud["bc"], phase_chunk_size=3)
        if model.has_anchors:
            torch.testing.assert_close(state["anchor_features"], short_state["anchor_features"],
                                       rtol=1e-5, atol=1e-6)
        q = cloud["q"]
        def decode(s, chunk, state):
            return model.decode(encoded, "velocity", q["pos"][s], q["x"][s], q["batch"][s],
                                phase_state=state, phase_chunk_size=chunk,
                                query_frame=q["frame"][s], query_geom=q["geom_extra"][s])
        full = decode(slice(None), 8, state)
        parts = torch.cat([decode(s, 3, short_state) for s in (slice(0, 2), slice(2, 5), slice(5, None))])
        torch.testing.assert_close(parts, full, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("arm", ["G00", "G01", "G10", "G11"])
def test_edge_permutation_case_isolation_and_actual_neighbour_mixing(cloud, arm):
    model = build(arm).eval()
    a = cloud["anchors"]
    with torch.no_grad():
        encoded = model.encode(cloud["pos"], cloud["x"], cloud["batch"], anchors=a, evaluation=True)
        before = model.prepare_phase(encoded, cloud["phase"], cloud["bc"])["anchor_features"]
        perm = torch.randperm(a["edge_index"].shape[1])
        shuffled = dict(a, edge_index=a["edge_index"][:, perm], edge_attr=a["edge_attr"][perm])
        encoded2 = model.encode(cloud["pos"], cloud["x"], cloud["batch"], anchors=shuffled, evaluation=True)
        after = model.prepare_phase(encoded2, cloud["phase"], cloud["bc"])["anchor_features"]
        torch.testing.assert_close(before, after, rtol=1e-5, atol=1e-6)
        # Mutate one node feature after encoding to isolate operator mixing.
        encoded["anchors"]["node_features"] = encoded["anchors"]["node_features"].clone()
        encoded["anchors"]["node_features"][1] += 2
        changed = model.prepare_phase(encoded, cloud["phase"], cloud["bc"])["anchor_features"]
        torch.testing.assert_close(before[8:], changed[8:], rtol=0, atol=0)
        if model.graph_anchors:
            assert not torch.allclose(before[0], changed[0])
        else:
            torch.testing.assert_close(before[0], changed[0], rtol=0, atol=0)


def test_checkpoint_amp_and_phase_subset_validation(cloud):
    original = build("G11", graph_activation_checkpointing=False).train()
    with torch.no_grad():
        original.frame_residual.output.weight.normal_(std=.05)
        original.anchor_readout.output.weight.normal_(std=.05)
    checked = copy.deepcopy(original)
    checked.graph_activation_checkpointing = True
    for model in (original, checked):
        with torch.autocast("cpu", dtype=torch.bfloat16):
            result = run(model, cloud, phase_chunk_size=3)
            loss = result.square().mean()
        loss.backward()
        assert torch.isfinite(loss)
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    for (name, p), (name2, p2) in zip(original.named_parameters(), checked.named_parameters()):
        assert name == name2
        torch.testing.assert_close(p.grad, p2.grad, rtol=1e-4, atol=1e-6)
    encoded = checked.encode(cloud["pos"], cloud["x"], cloud["batch"], anchors=cloud["anchors"], evaluation=True)
    with pytest.raises(ValueError, match="phase_indices"):
        checked.prepare_phase(encoded, cloud["phase"][:3], cloud["bc"])
    subset = checked.prepare_phase(encoded, cloud["phase"][[5, 1, 3]], cloud["bc"],
                                   phase_indices=torch.tensor([5, 1, 3]))
    assert subset["anchor_features"].shape[1] == 3


@pytest.mark.parametrize("arm", ["G00", "G10"])
def test_static_anchor_computed_once_and_preserves_time_dependent_bc(cloud, arm):
    model = build(arm).eval()
    calls = []
    hook = model.anchor_operator.register_forward_pre_hook(lambda module, args: calls.append(args[1].shape[1]))
    encoded = model.encode(cloud["pos"], cloud["x"], cloud["batch"],
                           anchors=cloud["anchors"], evaluation=True)
    state = model.prepare_phase(encoded, cloud["phase"], cloud["bc"], phase_chunk_size=3)
    assert calls == [1]
    ctx = state["anchor_features"]
    torch.testing.assert_close(ctx, ctx[:, :1].expand_as(ctx), rtol=0, atol=0)
    # Compare expansion's accumulated gradient with explicit repeated forwards.
    ctx.square().mean().backward()
    cached_grads = {name: p.grad.clone() for name, p in model.anchor_operator.named_parameters()}
    model.zero_grad(set_to_none=True)
    a = encoded["anchors"]
    cond = cloud["bc"][:, None].expand(-1, 8, -1)
    explicit = model.anchor_operator(a["node_features"].detach(), cond, a["batch"],
                                     a["edge_index"], a["edge_attr"], a["edge_summary"])
    explicit.square().mean().backward()
    for name, p in model.anchor_operator.named_parameters():
        torch.testing.assert_close(p.grad, cached_grads[name], rtol=1e-5, atol=1e-6)
    calls.clear()
    varying = cloud["bc"][:, None].expand(-1, 8, -1).clone()
    varying[:, 4:] += 2
    state = model.prepare_phase(encoded, cloud["phase"], varying, phase_chunk_size=3)
    assert calls == [3, 3, 2]
    assert not torch.allclose(state["anchor_features"][:, 0], state["anchor_features"][:, 7])
    hook.remove()


def test_invalid_edges_and_missing_geometries_fail_explicitly(cloud):
    model = build("G11")
    with pytest.raises(ValueError, match="require deployment"):
        model.encode(cloud["pos"], cloud["x"], cloud["batch"], evaluation=True)
    bad = copy.deepcopy(cloud["anchors"])
    bad["edge_index"][1, 0] = 15
    with pytest.raises(ValueError, match="cross cases"):
        model.encode(cloud["pos"], cloud["x"], cloud["batch"], anchors=bad, evaluation=True)
    a0 = build("A0").eval()
    encoded = a0.encode(cloud["pos"], cloud["x"], cloud["batch"], evaluation=True)
    q = cloud["q"]
    with pytest.raises(ValueError, match="query_frame"):
        a0.decode(encoded, "velocity", q["pos"], q["x"], q["batch"], cloud["phase"], bc=cloud["bc"])
