"""Physical objective and paired fresh initialization invariants for the M2 sweep."""
from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from training_wss_min import config as C
from training_wss_min.models import build_model
from training_wss_min.objectives import compute_loss, raw_space_mse
from training_wss_min.paired_initialization import build_paired_model
from training_wss_min.runtime import seed_all


REFERENCE = C.PROJECT_ROOT / "training_wss_min/configs/v6_followup_20260909/M2_a5_independent_k3_s1234.json"
STATS = {"method": "log_z", "log": {"mean": 0.5, "std": 1.25}, "eps": 1e-6,
         "raw_percentiles": {"p90": 10.0}}


def paired_config(**model_changes):
    cfg = C.ExpConfig.from_json(REFERENCE)
    cfg.train.init_reference_config = str(REFERENCE)
    for key, value in model_changes.items():
        setattr(cfg.model, key, value)
    return cfg


@pytest.fixture(scope="module", autouse=True)
def small_thread_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_raw_mse_matches_physical_formula_and_gradient():
    pred = torch.tensor([-0.5, 0.2, 1.0], requires_grad=True)
    truth = torch.tensor([0.5, 5.0, 20.0])
    value = raw_space_mse(pred, truth, C.TrainConfig(), STATS)
    recovered = torch.exp(pred.detach() * 1.25 + 0.5) - 1e-6
    expected = ((recovered - truth) / 10.0).square().mean()
    torch.testing.assert_close(value, expected, rtol=0, atol=0)
    value.backward()
    expected_gradient = 2 * (recovered - truth) * (recovered + 1e-6) * 1.25 / 100 / 3
    torch.testing.assert_close(pred.grad, expected_gradient)


def test_physical_auxiliary_and_components_sum_to_total():
    cfg = C.TrainConfig(loss_raw_mse_lambda=.03, loss_pinball_lambda=.2)
    pred = torch.tensor([-0.2, 0.5, 1.5], requires_grad=True)
    batch = {"y": torch.tensor([-.1, .4, 2.0]), "y_raw": torch.tensor([1.0, 3.0, 30.0])}
    parts = {}
    result = compute_loss(pred, batch, cfg, "cpu", STATS, components=parts)
    expected = parts["base"] + parts["raw_mse_weighted"] + parts["pinball_weighted"]
    torch.testing.assert_close(result.detach(), expected, rtol=0, atol=0)
    assert parts["raw_mse_weighted"] == pytest.approx(float(parts["raw_mse"]) * .03)
    result.backward()
    assert torch.isfinite(pred.grad).all()
    assert all(not value.requires_grad for value in parts.values())


def test_physical_recovery_stays_fp32_in_autocast():
    pred = torch.tensor([0.25, 1.25], dtype=torch.bfloat16, requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = raw_space_mse(pred, torch.tensor([3., 9.]), C.TrainConfig(), STATS)
    assert loss.dtype == torch.float32
    loss.backward()
    assert torch.isfinite(pred.grad).all()


@pytest.mark.parametrize("p90", [None, 0, -1, float("nan"), float("inf")])
def test_pa_mse_never_silently_replaces_invalid_train_p90(p90):
    stats = copy.deepcopy(STATS)
    stats["raw_percentiles"]["p90"] = p90
    with pytest.raises(ValueError, match="positive train-only"):
        raw_space_mse(torch.ones(2), torch.ones(2), C.TrainConfig(), stats)


def test_train_p90_override_must_be_valid_and_is_used():
    cfg = C.TrainConfig()
    cfg._raw_p90 = 20.0
    actual = raw_space_mse(torch.zeros(2), torch.ones(2), cfg, STATS)
    expected = raw_space_mse(torch.zeros(2), torch.ones(2), C.TrainConfig(), STATS) / 4
    torch.testing.assert_close(actual, expected)
    cfg._raw_p90 = 0.0
    with pytest.raises(ValueError, match="positive train-only"):
        raw_space_mse(torch.zeros(2), torch.ones(2), cfg, STATS)


def test_nonfinite_recovery_is_fatal_and_vector_target_is_rejected():
    with pytest.raises(FloatingPointError, match="non-finite Pa-MSE"):
        raw_space_mse(torch.tensor([1000.]), torch.ones(1), C.TrainConfig(), STATS)
    with pytest.raises(ValueError, match="scalar WSS"):
        raw_space_mse(torch.ones(2, 3), torch.ones(2, 3), C.TrainConfig(), STATS)


@pytest.mark.parametrize("changes", [
    {}, {"loss_raw_huber_lambda": .03},
    {"loss_weight_target": True, "loss_weight_target_alpha": 1,
     "y_norm_q02": -2., "y_norm_q98": 2.},
])
def test_disabled_pa_mse_and_logging_preserve_historical_loss_and_gradient(changes):
    source = C.PROJECT_ROOT / "training_wss_min/experiments/v6_followup_20260909/source_snapshot_13205/training_wss_min/objectives.py"
    spec = importlib.util.spec_from_file_location("training_wss_min._m2_legacy_objectives", source)
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    cfg = C.TrainConfig(loss_pinball_lambda=.2, **changes)
    batch = {"y": torch.tensor([-.5, 0., 1.]), "y_raw": torch.tensor([.5, 2., 8.])}
    a = torch.tensor([-.2, .25, .8], requires_grad=True)
    b = a.detach().clone().requires_grad_()
    c = a.detach().clone().requires_grad_()
    old_loss = legacy.compute_loss(a, batch, cfg, "cpu", STATS)
    new_loss = compute_loss(b, batch, cfg, "cpu", STATS)
    logged_loss = compute_loss(c, batch, cfg, "cpu", STATS, components={})
    old_loss.backward(); new_loss.backward(); logged_loss.backward()
    torch.testing.assert_close(old_loss, new_loss, rtol=0, atol=0)
    torch.testing.assert_close(old_loss, logged_loss, rtol=0, atol=0)
    torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
    torch.testing.assert_close(a.grad, c.grad, rtol=0, atol=0)


@pytest.mark.parametrize("model_changes", [
    {}, {"local_branch_channels": 64}, {"sa_blocks": (2, 1, 0)},
    {"local_branch_modulation": "gate"},
    {"query_decoder": "local_attn", "query_decoder_k": 3},
])
def test_paired_model_copies_reference_parameters_buffers_and_postbuild_rng(model_changes):
    cfg = paired_config(**model_changes)
    seed_all(cfg.train.seed)
    ref_cfg = C.ExpConfig.from_json(REFERENCE)
    reference = build_model(ref_cfg.model, C.input_dim(ref_cfg)).state_dict()
    expected_cpu = torch.get_rng_state().clone()
    expected_numpy = np.random.get_state()
    seed_all(88213)  # pairing must not depend on unrelated prior construction
    candidate, evidence = build_paired_model(cfg)
    assert evidence["trained_checkpoint_loaded"] is False
    assert torch.equal(torch.get_rng_state(), expected_cpu)
    actual_numpy = np.random.get_state()
    assert actual_numpy[0] == expected_numpy[0]
    np.testing.assert_array_equal(actual_numpy[1], expected_numpy[1])
    assert actual_numpy[2:] == expected_numpy[2:]
    state = candidate.state_dict()
    for key, tensor in reference.items():
        if state[key].shape == tensor.shape:
            torch.testing.assert_close(state[key], tensor, rtol=0, atol=0)
        else:
            assert key.startswith("local_wall_branch.")


def test_gate_and_additive_share_exact_modulation_initialization():
    gate, a = build_paired_model(paired_config(local_branch_modulation="gate"))
    additive, b = build_paired_model(paired_config(local_branch_modulation="additive"))
    assert a["substream_seeds"] == b["substream_seeds"]
    for key, value in gate.state_dict().items():
        torch.testing.assert_close(value, additive.state_dict()[key], rtol=0, atol=0)
    assert torch.count_nonzero(gate.local_wall_branch.modulation[-1].weight) == 0


def test_decoder_residual_does_not_shift_shared_attention_initialization():
    args = dict(query_decoder="local_attn", query_decoder_k=3,
                query_decoder_feature_indices=(4, 14, 15, 16))
    plain, _ = build_paired_model(paired_config(**args))
    residual, _ = build_paired_model(paired_config(**args, query_decoder_residual=True))
    for key, value in plain.state_dict().items():
        torch.testing.assert_close(value, residual.state_dict()[key], rtol=0, atol=0)
    assert torch.count_nonzero(residual.local_query.res_out.weight) == 0


def test_new_module_substreams_are_independent_of_other_architecture_additions():
    query = dict(query_decoder="local_attn", query_decoder_k=3, query_decoder_residual=True)
    query_only, _ = build_paired_model(paired_config(**query))
    gate_only, _ = build_paired_model(paired_config(local_branch_modulation="gate"))
    deep_only, _ = build_paired_model(paired_config(sa_blocks=(2, 1, 0)))
    combined, _ = build_paired_model(paired_config(
        **query, local_branch_modulation="gate", sa_blocks=(2, 1, 0),
    ))
    for root, separate in [("local_query", query_only),
                           ("local_wall_branch.modulation", gate_only),
                           ("sa.0.blocks.1", deep_only)]:
        for key, value in separate.get_submodule(root).state_dict().items():
            torch.testing.assert_close(
                value, combined.get_submodule(root).state_dict()[key], rtol=0, atol=0,
            )


def test_unapproved_reference_shape_drift_and_checkpoint_loading_are_rejected():
    with pytest.raises(ValueError, match="unapproved shape changes"):
        build_paired_model(paired_config(width=64))
    cfg = paired_config()
    cfg.train.init_checkpoint_path = "unused_checkpoint.pt"
    with pytest.raises(ValueError, match="cannot load a trained checkpoint"):
        build_paired_model(cfg)


@pytest.mark.parametrize("raw_mse_weight", [0.0, 0.03])
def test_training_diagnostics_do_not_change_optimizer_updates(tmp_path, monkeypatch, raw_mse_weight):
    """A two-step synthetic CPU loop verifies the real CLI wiring and artifacts."""
    from training_wss_min import train

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(3, 1)

        def forward(self, pos, x, batch):
            return self.linear(x).squeeze(-1)

    class TinyDataset:
        def __init__(self, *args, **kwargs):
            pass

        def set_epoch(self, epoch):
            pass

    batch = {"pos": torch.zeros(4, 3), "x": torch.arange(12).reshape(4, 3).float() / 12,
             "batch": torch.tensor([0, 0, 1, 1]),
             "y": torch.tensor([-.2, .1, .5, 1.]), "y_raw": torch.tensor([1., 2., 3., 4.])}
    quantiles = {key: value for key, value in zip(
        ("y_norm_q02", "y_norm_q98", "curv_q02", "curv_q98", "invr_q02", "invr_q98", "raw_p90"),
        (-2, 2, 0, 1, 0, 1, 10),
    )}
    monkeypatch.setattr(C, "RUNS_ROOT", tmp_path / "runs")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(train.D, "load_wss_stats", lambda *a, **kw: copy.deepcopy(STATS))
    monkeypatch.setattr(train.D, "load_partition", lambda *a, **kw: [{"cohort": "toy", "case": "a"}])
    monkeypatch.setattr(train.D, "load_split_cases", lambda *a, **kw: [])
    monkeypatch.setattr(train.D, "compute_feature_stats", lambda *a, **kw: {})
    monkeypatch.setattr(train.D, "compute_train_weight_quantiles", lambda *a, **kw: quantiles)
    monkeypatch.setattr(train.D, "WSSMinDataset", TinyDataset)
    monkeypatch.setattr(train, "DataLoader", lambda *a, **kw: [batch])
    monkeypatch.setattr(train, "build_model", lambda *a, **kw: TinyModel())
    monkeypatch.setattr(train, "plot_history", lambda *a, **kw: None)
    outputs = []
    for logging_enabled in (False, True):
        cfg = C.ExpConfig(name=f"toy_{logging_enabled}")
        cfg.train.epochs = 2
        cfg.train.amp = False
        cfg.train.selection_rule = "train_loss"
        cfg.train.loss_pinball_lambda = .2
        cfg.train.loss_raw_mse_lambda = raw_mse_weight
        cfg.train.log_loss_components = logging_enabled
        cfg.data.num_workers = 0
        path = cfg.to_json(tmp_path / f"cfg_{logging_enabled}.json")
        monkeypatch.setattr(sys, "argv", ["train", "--config", str(path)])
        train.main()
        outputs.append(cfg.run_dir)
    states = [torch.load(root / "ckpt_last.pt", weights_only=False)["model"] for root in outputs]
    for key in states[0]:
        torch.testing.assert_close(states[0][key], states[1][key], rtol=0, atol=0)
    assert not (outputs[0] / "training_diagnostics.json").exists()
    report = json.loads((outputs[1] / "training_diagnostics.json").read_text())
    assert report["completed_epochs"] == 2
    assert report["amp_overflow_steps"] == 0
    assert report["peak_cuda_allocated_mb"] == 0
    histories = [[json.loads(line) for line in (root / "history.jsonl").read_text().splitlines()]
                 for root in outputs]
    for a, b in zip(*histories):
        assert a["train_loss"] == b["train_loss"]
        assert "loss_components" not in a
        assert b["loss_components"]["total"] == b["train_loss"]
        assert b["grad_norm_preclip_mean"] >= 0
