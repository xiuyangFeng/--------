"""Contracts for the new joint cycle experiment (small CPU point clouds)."""
import copy

import pytest
import torch

from training_wss_min.joint_cycle_model import TASK_CHANNELS, build_joint_cycle_model, _interpolate_fp32


OPTIONS = {"geometry": {"width": 8, "sa_center_counts": (12, 6),
                         "sa_ratios": (0.5, 0.5), "sa_radius": (2., 3.),
                         "sa_nsample": (6, 4)},
           "head_hidden": 16, "token_count": 8, "token_hidden": 16, "token_heads": 2}
DIMS = {"velocity": 5, "pressure": 5, "wss": 7}


@pytest.fixture
def cloud():
    torch.manual_seed(88)
    pos = torch.randn(64, 3) * .1
    x = torch.randn(64, 6)
    batch = torch.arange(2).repeat_interleave(32)
    queries = {task: {"pos": torch.randn(12, 3) * .1,
                      "x": torch.randn(12, dim),
                      "batch": torch.arange(2).repeat_interleave(6)} for task, dim in DIMS.items()}
    phase = torch.randn(5, 4)
    return pos, x, batch, queries, phase


def build(variant, **kwargs):
    return build_joint_cycle_model(variant, 6, DIMS, seed=23, model_options=OPTIONS, **kwargs)


def test_paired_initial_functions_including_split_independent_runs(cloud):
    pos, x, batch, queries, phase = cloud
    reference = build("J0").eval()
    with torch.no_grad():
        expected = reference(pos, x, batch, queries, phase, evaluation=True)
        for variant in ("I", "J1c", "J1"):
            candidate = build(variant).eval()
            actual = candidate(pos, x, batch, queries, phase, evaluation=True)
            for task in TASK_CHANNELS:
                assert actual[task].shape == (12, 5, TASK_CHANNELS[task])
                torch.testing.assert_close(actual[task], expected[task], rtol=0, atol=0)
        for task in TASK_CHANNELS:
            separate = build("I", active_tasks=[task]).eval()
            assert list(separate.encoders) == [task]
            assert list(separate.decoders) == [task]
            actual = separate(pos, x, batch, {task: queries[task]}, phase, evaluation=True)
            torch.testing.assert_close(actual[task], expected[task], rtol=0, atol=0)


def test_phase_and_query_chunks_preserve_predictions(cloud):
    pos, x, batch, queries, phase = cloud
    model = build("J1").eval()
    with torch.no_grad():
        model.interaction.out.weight.normal_(std=.1)
        encoded = model.encode(pos, x, batch, evaluation=True)
        state = model.prepare_phase(encoded, phase, phase_chunk_size=2)
        for task, q in queries.items():
            full = model.decode(encoded, task, q["pos"], q["x"], q["batch"],
                                phase_state=state, phase_chunk_size=5)
            parts = [model.decode(encoded, task, q["pos"][sl], q["x"][sl], q["batch"][sl],
                                  phase_state=state, phase_chunk_size=2)
                     for sl in (slice(0, 4), slice(4, 9), slice(9, None))]
            torch.testing.assert_close(torch.cat(parts), full, rtol=1e-5, atol=1e-6)


def test_phase_changes_neighbour_weights_and_fixed_condition_control(cloud):
    pos, x, batch, queries, phase = cloud
    dynamic = build("J1").eval()
    static = build("J1c").eval()
    with torch.no_grad():
        dynamic.interaction.out.weight.normal_(std=.1)
        static.load_state_dict(dynamic.state_dict())
        ed = dynamic.encode(pos, x, batch, evaluation=True)
        es = static.encode(pos, x, batch, evaluation=True)
        ds = dynamic.prepare_phase(ed, phase)["token_delta"]["shared"][0]
        ss = static.prepare_phase(es, phase)["token_delta"]["shared"][0]
        assert not torch.allclose(ds[0], ds[1], atol=1e-6)
        torch.testing.assert_close(ss[0], ss[1], rtol=0, atol=0)
        geo = ed["geometry"]["shared"]; ids = geo["token_ids"][0]
        _, attn = dynamic.interaction(geo["features"][ids], geo["pos"][ids], phase,
                                     return_attention=True)
        assert not torch.allclose(attn[0], attn[1], atol=1e-6)


def test_backward_reaches_geometry_tasks_and_spatial_branch_after_zero_start(cloud):
    pos, x, batch, queries, phase = cloud
    model = build("J1").train()
    outputs = model(pos, x, batch, queries, phase, evaluation=True)
    loss = sum((v - .3).square().mean() for v in outputs.values())
    loss.backward()
    assert model.interaction.out.weight.grad.abs().sum() > 0
    assert model.encoders["shared"].backbone.stem[0].weight.grad.abs().sum() > 0
    for task in TASK_CHANNELS:
        assert model.decoders[task].output.weight.grad.abs().sum() > 0
    with torch.no_grad():
        model.interaction.out.weight.add_(model.interaction.out.weight.grad, alpha=-.1)
    model.zero_grad(set_to_none=True)
    outputs = model(pos, x, batch, queries, phase, evaluation=True)
    sum((v - .3).square().mean() for v in outputs.values()).backward()
    assert model.interaction.condition.weight.grad.abs().sum() > 0
    assert model.interaction.qkv.weight.grad.abs().sum() > 0


def test_case_isolation_and_explicit_bc_interface(cloud):
    pos, x, batch, queries, phase = cloud
    options = copy.deepcopy(OPTIONS); options["bc_dim"] = 2
    model = build_joint_cycle_model("J1", 6, DIMS, seed=23, model_options=options).eval()
    bc = torch.randn(2, 2)
    with torch.no_grad():
        model.interaction.out.weight.normal_(std=.1)
        encoded = model.encode(pos, x, batch, evaluation=True)
        with pytest.raises(ValueError, match="requires explicit"):
            model.prepare_phase(encoded, phase)
        before = model(pos, x, batch, queries, phase, bc=bc, evaluation=True)
        changed = x.clone(); changed[batch == 1] += 100
        bc_after = bc.clone(); bc_after[1] += 100
        after = model(pos, changed, batch, queries, phase, bc=bc_after, evaluation=True)
        for task in TASK_CHANNELS:
            torch.testing.assert_close(before[task][:6], after[task][:6], rtol=0, atol=0)


def test_construction_does_not_advance_rng_and_independent_parameters():
    torch.manual_seed(123)
    state = torch.random.get_rng_state().clone()
    independent = build("I")
    assert torch.equal(state, torch.random.get_rng_state())
    u = independent.encoders["velocity"].backbone.stem[0].weight
    p = independent.encoders["pressure"].backbone.stem[0].weight
    assert u.data_ptr() != p.data_ptr()
    torch.testing.assert_close(u, p, rtol=0, atol=0)


def test_half_features_at_exact_points_have_finite_interpolation_and_gradients():
    pos = torch.tensor([[0., 0., 0.], [1e-5, 0., 0.], [2e-5, 0., 0.]])
    features = torch.tensor([[1., 3.], [2., 4.], [3., 5.]], dtype=torch.float16, requires_grad=True)
    batch = torch.zeros(3, dtype=torch.long)
    out = _interpolate_fp32(features, pos, pos, batch, batch, k=3)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, features.float(), rtol=1e-5, atol=1e-5)
    out.square().sum().backward()
    assert torch.isfinite(features.grad).all()


def test_checkpointed_phase_blocks_preserve_predictions_and_gradients(cloud):
    pos, x, batch, queries, phase = cloud
    reference = build("J1").train()
    candidate = copy.deepcopy(reference)
    candidate.activation_checkpointing = True
    with torch.no_grad():
        reference.interaction.out.weight.normal_(std=.1)
        candidate.load_state_dict(reference.state_dict())
    for model in (reference, candidate):
        outputs = model(pos, x, batch, queries, phase, evaluation=True, phase_chunk_size=2)
        sum(v.square().mean() for v in outputs.values()).backward()
        if model is reference:
            expected = {k: v.detach() for k, v in outputs.items()}
        else:
            for task in TASK_CHANNELS:
                torch.testing.assert_close(outputs[task], expected[task], rtol=0, atol=0)
    for (name, rp), (cn, cp) in zip(reference.named_parameters(), candidate.named_parameters()):
        assert name == cn
        if rp.grad is None:
            assert cp.grad is None
        else:
            torch.testing.assert_close(rp.grad, cp.grad, rtol=1e-5, atol=1e-7)


def test_cpu_autocast_all_three_heads_have_finite_loss_and_gradients(cloud):
    pos, x, batch, queries, phase = cloud
    model = build("J1").train()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        outputs = model(pos, x, batch, queries, phase, evaluation=True)
        loss = sum(v.float().square().mean() for v in outputs.values())
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
