"""Contract tests for the temporal Transformer probe.

These tests are deliberately CPU-only.  They exercise the model's temporal
factorisation and checkpoint/config contracts without starting a training job.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from training_wss_min import time_transformer as TT


def _arch(kind: str = "transformer") -> dict:
    return {
        "temporal": kind,
        "d_model": 16,
        "nhead": 4,
        "num_layers": 2,
        "dim_feedforward": 32,
        "dropout": 0.0,
        "spatial_hidden": 24,
        "decoder_hidden": [24, 24],
        "global_pool": "mean",
    }


def _inputs(n: int = 7) -> torch.Tensor:
    torch.manual_seed(100)
    return torch.randn(1, n, 60)


def _phase(frames: int = 7) -> torch.Tensor:
    torch.manual_seed(101)
    return torch.randn(frames, 4)


def test_same_seed_constructs_identical_state_tensors_for_attention_and_control():
    torch.manual_seed(1234)
    first = TT.TemporalFieldModel(input_dim=60, arch=_arch("transformer"))
    torch.manual_seed(1234)
    second = TT.TemporalFieldModel(input_dim=60, arch=_arch("frame_mlp"))

    assert first.state_dict().keys() == second.state_dict().keys()
    for key, value in first.state_dict().items():
        assert torch.equal(value, second.state_dict()[key]), key


def test_transformer_temporal_context_responds_to_other_frame_perturbation():
    model = TT.TemporalFieldModel(input_dim=60, arch=_arch("transformer")).eval()
    torch.manual_seed(1)
    point_feat = torch.randn(2, 5, 16)
    phase = _phase(7)
    changed = phase.clone()
    changed[0] += 10.0

    with torch.no_grad():
        baseline = model.temporal_context(point_feat, phase)
        perturbed = model.temporal_context(point_feat, changed)

    # A true temporal attention block can propagate a token perturbation to a
    # different frame.  Frame zero itself is not the tested location.
    assert not torch.allclose(baseline[:, 1], perturbed[:, 1], atol=1e-7, rtol=1e-6)


def test_frame_mlp_is_point_to_point_in_time():
    model = TT.TemporalFieldModel(input_dim=60, arch=_arch("frame_mlp")).eval()
    torch.manual_seed(2)
    point_feat = torch.randn(2, 5, 16)
    phase = _phase(7)
    changed = phase.clone()
    changed[0] += 10.0

    with torch.no_grad():
        baseline = model.temporal_context(point_feat, phase)
        perturbed = model.temporal_context(point_feat, changed)

    # The no-attention control must leave every other time token unchanged.
    assert torch.equal(baseline[:, 1:], perturbed[:, 1:])
    assert not torch.equal(baseline[:, 0], perturbed[:, 0])


def test_forward_residual_is_exactly_zero_at_peak():
    model = TT.TemporalFieldModel(input_dim=60, arch=_arch("transformer")).eval()
    with torch.no_grad():
        # Test the anchor after the initially zero output layer has changed.
        model.decoder[-1].weight.normal_(0.0, 0.2)
        model.decoder[-1].bias.fill_(0.5)
    with torch.no_grad():
        output = model(_inputs(), _phase(), peak_index=3)

    peak = output[:, :, 3]
    assert torch.equal(peak, torch.zeros_like(peak))
    assert output.abs().max() > 0


def test_explicit_context_makes_query_chunk_decode_equivalent():
    model = TT.TemporalFieldModel(input_dim=60, arch=_arch("transformer")).eval()
    with torch.no_grad():
        model.decoder[-1].weight.normal_(0.0, 0.2)
        model.decoder[-1].bias.fill_(0.5)
    x = _inputs(9)
    phase = _phase(7)
    with torch.no_grad():
        point_feat = model.point(x)
        context = model.temporal_context(point_feat, phase)
        full = model.decode(point_feat, context, peak_index=2)
        first = model(x[:, :4], phase, peak_index=2, context=context)
        second = model(x[:, 4:], phase, peak_index=2, context=context)

    assert torch.allclose(first, full[:, :4], atol=1e-6, rtol=1e-5)
    assert torch.allclose(second, full[:, 4:], atol=1e-6, rtol=1e-5)


def test_real_checkpoint_state_roundtrip_if_available(tmp_path):
    """Load one produced checkpoint and verify a save/load round trip.

    The checkpoint is an experiment artifact rather than a test fixture.  Keep
    this test skippable for a clean source checkout where experiment outputs
    are intentionally absent.
    """

    root = Path(__file__).resolve().parents[2]
    checkpoint = root / "training_wss_min/runs/wss_time_transformer_v51_20260924/TT-warm_f0_s1234/ckpt_last.pt"
    if not checkpoint.is_file():
        pytest.skip("the real temporal-transformer checkpoint is not present")

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = payload["config"]
    inputs = cfg["inputs"]
    input_dim = sum(TT.INPUT_DIMS[key] for key in inputs)
    first = TT.TemporalFieldModel(input_dim=input_dim, arch=cfg["architecture"])
    first.load_state_dict(payload["state_dict"], strict=True)

    roundtrip = tmp_path / "ckpt.pt"
    torch.save({"state_dict": first.state_dict(), "config": cfg}, roundtrip)
    restored_payload = torch.load(roundtrip, map_location="cpu", weights_only=False)
    second = TT.TemporalFieldModel(input_dim=input_dim, arch=cfg["architecture"])
    second.load_state_dict(restored_payload["state_dict"], strict=True)
    first.eval()
    second.eval()

    # Use the checkpoint's actual input width; the model must not depend on
    # CUDA or on optimizer state for deterministic inference.
    x = torch.randn(1, 4, input_dim)
    phase = torch.randn(7, 4)
    with torch.no_grad():
        out_first = first(x, phase, peak_index=2)
        out_second = second(x, phase, peak_index=2)
    assert torch.equal(out_first, out_second)


def test_real_fold_contract_has_81_steps_peak_floor_and_matching_split(tmp_path):
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs/wss_time_transformer_v51_20260924/TT-warm_f0_s1234.json"
    )
    if not config_path.is_file():
        pytest.skip("temporal-transformer configs are not present")

    # Old v1 configurations are audit-only.  Reuse their immutable data paths
    # with the current schema without changing the saved experiment config.
    raw_cfg = json.loads(config_path.read_text())
    raw_cfg["schema"] = TT.SCHEMA
    current = tmp_path / "contract.json"
    current.write_text(json.dumps(raw_cfg))
    cfg = TT.load_config(current)
    fold = TT._load_fold(cfg)
    split = json.loads(Path(cfg["split_path"]).read_text())
    waveform = json.loads(Path(cfg["waveform_path"]).read_text())

    assert fold["peak_index"] == 21
    assert fold["phase"].shape == (81, 4)
    assert fold["mu"].shape == (81,)
    assert fold["sd"].shape == (81,)
    assert np.isfinite(fold["mu"]).all() and (fold["sd"] > 0).all()
    assert fold["floor"] == pytest.approx(0.05)
    assert set(fold["train_ids"]) == set(split["train_cases"])
    assert set(fold["test_ids"]) == set(split["test_cases"])
    assert len(waveform["steps"]) == 81
    assert np.array_equal(fold["steps"], waveform["steps"])


def test_input_keeps_canonical_60d_width_when_query_x_is_omitted():
    rng = np.random.default_rng(5)
    n = 6
    case = {
        "query_x": rng.normal(size=(n, 32)).astype(np.float32),
        "raw": rng.normal(size=(n, 27)).astype(np.float32),
        "peak_ln": rng.normal(size=n).astype(np.float32),
    }
    # Masking must happen after normalization, even when omitted channels have
    # nonzero means.  Otherwise a raw-only control could still use a constant
    # query_x feature determined by the donor statistics.
    mean = np.full(60, 2.0, dtype=np.float32)
    std = np.full(60, 3.0, dtype=np.float32)
    got = TT._input(case, np.arange(n), ["raw", "peak_ln"], mean, std)

    assert got.shape == (n, 60)
    assert np.array_equal(got[:, :32], np.zeros((n, 32), dtype=np.float32))
    assert np.allclose(got[:, 32:59], (case["raw"] - 2.0) / 3.0)
    assert np.allclose(got[:, 59], (case["peak_ln"] - 2.0) / 3.0)


def _synthetic_case(tmp_path, *, partition="test", step_offset=0, peak_offset=0):
    uid, n = "AAA/ruputer/SYNTHETIC", 6
    cache = tmp_path / "cache"
    bundle_dir = tmp_path / "view" / uid
    cache.mkdir(exist_ok=True)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    steps = np.arange(1120, 1281, 2, dtype=np.int64)
    # Several values are below the training floor, including exactly zero.
    tau = np.tile(np.array([0.0, 0.001, 0.01, 0.049, 0.1, 2.0]), (81, 1))
    np.savez(
        cache / f"{TT.TA.case_key(uid)}.npz",
        unit_id=np.array(uid),
        partition=np.array(partition),
        peak_index=np.array(21),
        peak_ln=np.linspace(-2, 2, n, dtype=np.float64),
        query_x=np.arange(n * 32, dtype=np.float32).reshape(n, 32),
        raw=np.arange(n * 27, dtype=np.float32).reshape(n, 27),
    )
    np.savez(
        bundle_dir / "bundle.npz",
        steps=steps + step_offset,
        peak_step=np.array(steps[21] + peak_offset),
        wall_wss=tau,
    )
    cfg = {
        "cache_dir": str(cache), "data_root": str(tmp_path / "view"),
        "seed": 1234, "train": {"train_pool_points": n, "context_points": 3},
    }
    fold = {
        "peak_index": 21, "steps": steps, "floor": 0.05, "eps": 1e-6,
        "manifest": {"cases": {uid: {"n_points": n}}},
    }
    return cfg, uid, fold, tau


def test_load_case_preserves_unclipped_raw_truth_for_evaluation(tmp_path):
    cfg, uid, fold, tau = _synthetic_case(tmp_path)
    case = TT._load_case(cfg, uid, fold, training=False)

    assert np.array_equal(case["tau_raw"], tau)
    assert (case["tau_raw"] < fold["floor"]).any()
    assert "ln_y" not in case
    assert np.array_equal(case["idx"], np.arange(tau.shape[1]))


def test_load_case_uses_floor_only_for_training_target(tmp_path):
    cfg, uid, fold, tau = _synthetic_case(tmp_path, partition="train")
    case = TT._load_case(cfg, uid, fold, training=True)

    expected = TT.TA.ln_floor(tau, fold["floor"], fold["eps"]).T.astype(np.float32)
    assert np.array_equal(case["ln_y"], expected)
    assert "tau_raw" not in case


@pytest.mark.parametrize("step_offset,peak_offset", [(2, 0), (0, 2)])
def test_load_case_rejects_physical_step_or_peak_mismatch(tmp_path, step_offset, peak_offset):
    cfg, uid, fold, _ = _synthetic_case(tmp_path, step_offset=step_offset, peak_offset=peak_offset)
    with pytest.raises(ValueError, match="physical frame steps or peak"):
        TT._load_case(cfg, uid, fold, training=False)


def test_load_case_rejects_wrong_partition(tmp_path):
    cfg, uid, fold, _ = _synthetic_case(tmp_path, partition="train")
    with pytest.raises(ValueError, match="cache partition"):
        TT._load_case(cfg, uid, fold, training=False)
