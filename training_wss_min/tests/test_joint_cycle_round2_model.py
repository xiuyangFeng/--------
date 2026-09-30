"""Functional controls for second-round temporal and query residuals."""
import copy

import pytest
import torch

from training_wss_min.joint_cycle_model import build_joint_cycle_model as build_original
from training_wss_min.joint_cycle_round2_model import (
    TemporalResidual, build_joint_cycle_model, _count)


OPTIONS = {"geometry": {"width": 8, "sa_center_counts": (8, 4),
                        "sa_ratios": (.5, .5), "sa_radius": (2., 3.),
                        "sa_nsample": (6, 4)}, "head_hidden": 16,
           "bc_dim": 2, "round2_hidden": 8, "round2_heads": 2,
           "round2_neighbors": 16, "temporal_bottleneck": 8}
DIMS = {"velocity": 5, "pressure": 5, "wss": 7}
ARMS = [("temporal_ffn", "velocity"), ("temporal_tcn", "velocity"),
        ("temporal_ffn", "wss"), ("temporal_tcn", "wss"),
        ("query_mean", "velocity"), ("query_attention", "velocity"),
        ("wall_pointwise", "wss"), ("wall_patch", "wss")]


@pytest.fixture
def cloud():
    torch.set_num_threads(1)
    torch.manual_seed(79)
    pos, x = torch.randn(40, 3) * .1, torch.randn(40, 6)
    batch = torch.arange(2).repeat_interleave(20)
    queries = {task: {"pos": torch.randn(6, 3) * .1, "x": torch.randn(6, dim),
                      "batch": torch.arange(2).repeat_interleave(3)} for task, dim in DIMS.items()}
    phase, bc = torch.randn(80, 4), torch.randn(2, 2)
    return pos, x, batch, queries, phase, bc


def build(module, task, **options):
    return build_joint_cycle_model("I", 6, DIMS, seed=23, active_tasks=[task],
                                    model_options=dict(OPTIONS, round2_module=module, **options))


def residual(model, task):
    return (model.decoders[task].round2_residual if model.round2_module.startswith("temporal_")
            else model.round2_spatial[task])


@pytest.mark.parametrize("module,task", ARMS)
def test_zero_start_preserves_original_parameters_rng_and_function(cloud, module, task):
    pos, x, batch, queries, phase, bc = cloud
    original = build_original("I", 6, DIMS, seed=23, active_tasks=[task], model_options=OPTIONS).eval()
    rng = torch.random.get_rng_state().clone()
    model = build(module, task).eval()
    assert torch.equal(rng, torch.random.get_rng_state())
    for key, value in original.state_dict().items():
        torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
    assert model.round2_contract["residual_parameters"] == _count(residual(model, task))
    assert model.round2_contract["relative_parameter_mismatch"] <= .05
    with torch.no_grad():
        expected = original(pos, x, batch, {task: queries[task]}, phase, bc=bc,
                            phase_chunk_size=80, evaluation=True)[task]
        actual = model(pos, x, batch, {task: queries[task]}, phase, bc=bc,
                       phase_chunk_size=80, evaluation=True)[task]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("module,task", ARMS)
def test_active_branch_chunks_cases_bc_and_gradients(cloud, module, task):
    pos, x, batch, queries, phase, bc = cloud
    model = build(module, task).eval()
    q = queries[task]
    res = residual(model, task)
    # First step reaches the zero output projection; the next reaches the body.
    pred = model(pos, x, batch, {task: q}, phase, bc=bc, evaluation=True)[task]
    (pred - .4).square().mean().backward()
    assert res.out.weight.grad.abs().sum() > 0
    assert model.encoders[task].backbone.stem[0].weight.grad.abs().sum() > 0
    with torch.no_grad():
        res.out.weight.normal_(std=.1)
    model.zero_grad(set_to_none=True)
    pred = model(pos, x, batch, {task: q}, phase, bc=bc, evaluation=True)[task]
    (pred - .4).square().mean().backward()
    inner = [(name, p.grad) for name, p in res.named_parameters() if not name.startswith("out.")]
    assert all(grad is not None and torch.isfinite(grad).all() for _, grad in inner)
    assert all(grad.abs().sum() > 0 for _, grad in inner)
    with torch.no_grad():
        encoded = model.encode(pos, x, batch, evaluation=True)
        with pytest.raises(ValueError, match="requires explicit"):
            model.prepare_phase(encoded, phase)
        state = model.prepare_phase(encoded, phase, bc)
        full = model.decode(encoded, task, q["pos"], q["x"], q["batch"],
                            phase_state=state, phase_chunk_size=80)
        parts = [model.decode(encoded, task, q["pos"][s], q["x"][s], q["batch"][s],
                              phase_state=state, phase_chunk_size=7)
                 for s in (slice(0, 2), slice(2, 5), slice(5, None))]
        torch.testing.assert_close(torch.cat(parts), full, rtol=1e-5, atol=1e-6)
        x2, bc2 = x.clone(), bc.clone()
        x2[batch == 1] += 10
        bc2[1] += 5
        changed = model(pos, x2, batch, {task: q}, phase, bc=bc2, evaluation=True)[task]
        torch.testing.assert_close(changed[:3], pred[:3], rtol=0, atol=0)
        assert not torch.allclose(changed[3:], pred[3:])


def test_temporal_mixing_has_31_frame_receptive_field_and_pointwise_control():
    torch.manual_seed(14)
    x = torch.randn(2, 80, 16)
    changed = x.clone()
    changed[0, 40] += 2
    for temporal in (False, True):
        block = TemporalResidual(16, 4, temporal=temporal).eval()
        with torch.no_grad():
            block.out.weight.normal_(std=.1)
            before, after = block(x), block(changed)
        torch.testing.assert_close(before[1], after[1], rtol=0, atol=0)
        torch.testing.assert_close(before[0, :25], after[0, :25], rtol=0, atol=0)
        torch.testing.assert_close(before[0, 56:], after[0, 56:], rtol=0, atol=0)
        if temporal:
            assert not torch.allclose(before[0, 39], after[0, 39], atol=1e-7)
        else:
            torch.testing.assert_close(before[0, :40], after[0, :40], rtol=0, atol=0)
            torch.testing.assert_close(before[0, 41:], after[0, 41:], rtol=0, atol=0)


@pytest.mark.parametrize("module,task", [("query_mean", "velocity"),
                                         ("query_attention", "velocity"),
                                         ("wall_patch", "wss"), ("wall_pointwise", "wss")])
def test_actual_neighbour_context_and_case_local_edges(cloud, module, task):
    pos, x, batch, queries, phase, bc = cloud
    model = build(module, task).eval()
    q = queries[task]
    block = residual(model, task)
    with torch.no_grad():
        block.out.weight.normal_(std=.1)
        encoded = model.encode(pos, x, batch, evaluation=True)
        state = model.prepare_phase(encoded, phase, bc)
        before = model.decode(encoded, task, q["pos"], q["x"], q["batch"], phase_state=state)
        # Isolate newly supplied neighbour geometry from the base encoder.
        encoded["geometry"][task]["raw_x"] = x.clone()
        encoded["geometry"][task]["raw_x"][batch == 0] += 4
        after = model.decode(encoded, task, q["pos"], q["x"], q["batch"], phase_state=state)
    torch.testing.assert_close(before[3:], after[3:], rtol=0, atol=0)
    if module == "wall_pointwise":
        torch.testing.assert_close(before, after, rtol=0, atol=0)
    else:
        assert not torch.allclose(before[:3], after[:3], atol=1e-6)


def test_query_attention_weights_depend_on_query_and_are_normalized(cloud):
    pos, x, batch, queries, phase, bc = cloud
    model = build("query_attention", "velocity").eval()
    q = queries["velocity"]
    block = residual(model, "velocity")
    with torch.no_grad():
        geo = model.encode(pos, x, batch, evaluation=True)["geometry"]["velocity"]
        base = torch.zeros(6, 8)
        _, weights, row, col = block(base, q["pos"], q["x"], q["batch"], geo, return_attention=True)
        _, changed, row2, col2 = block(base, q["pos"], q["x"] + 2, q["batch"], geo, return_attention=True)
    assert torch.equal(row, row2) and torch.equal(col, col2)
    assert torch.equal(q["batch"][row], batch[col])
    assert not torch.allclose(weights, changed, atol=1e-6)
    for i in range(6):
        assert (row == i).sum() == 16
        torch.testing.assert_close(weights[row == i].sum(0), torch.ones(2))


@pytest.mark.parametrize("module,task", [("temporal_tcn", "velocity"), ("query_attention", "velocity"),
                                         ("wall_patch", "wss")])
def test_checkpoint_and_autocast_have_finite_equivalent_gradients(cloud, module, task):
    pos, x, batch, queries, phase, bc = cloud
    reference = build(module, task).train()
    with torch.no_grad():
        residual(reference, task).out.weight.normal_(std=.05)
    checked = copy.deepcopy(reference)
    checked.activation_checkpointing = True
    for model in (reference, checked):
        with torch.autocast("cpu", dtype=torch.bfloat16):
            out = model(pos, x, batch, {task: queries[task]}, phase, bc=bc, evaluation=True)[task]
            loss = out.float().square().mean()
        loss.backward()
        assert torch.isfinite(loss)
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    for (name, p), (name2, p2) in zip(reference.named_parameters(), checked.named_parameters()):
        assert name == name2
        if p.grad is not None:
            torch.testing.assert_close(p.grad, p2.grad, rtol=1e-4, atol=1e-6)


def test_none_path_remains_original_and_invalid_cross_task_is_rejected():
    original = build_original("J1", 6, DIMS, seed=23, active_tasks=["pressure"], model_options=OPTIONS)
    model = build_joint_cycle_model("J1", 6, DIMS, seed=23, active_tasks=["pressure"],
                                   model_options=dict(OPTIONS, round2_module="none"))
    assert type(model) is type(original)
    for key, value in original.state_dict().items():
        torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
    with pytest.raises(ValueError, match="mismatch"):
        build("query_attention", "wss")
