"""time_adapter_probe：锚定恒等、零初始化 = T-null、配置校验（纯 CPU，合成数据）。"""
import json

import numpy as np
import pytest
import torch

from training_wss_min import time_adapter_probe as TA


def test_anchor_is_exact_zero_at_peak_and_zero_init_is_tnull():
    torch.manual_seed(0)
    head = TA.TimeAdapterHead(5, 4, 16, 8, [16, 16], zero_init_output=False)
    geom, phase = torch.randn(33, 5), torch.randn(81, 4)
    delta = TA.anchored(head(geom, phase), 21)
    assert torch.count_nonzero(delta[:, 21]) == 0
    assert delta.abs().max() > 0
    zero = TA.TimeAdapterHead(5, 4, 16, 8, [16, 16], zero_init_output=True)
    assert torch.count_nonzero(TA.anchored(zero(geom, phase), 21)) == 0


def test_tnull_base_reproduces_peak_ln():
    rng = np.random.default_rng(1)
    mu, sd = rng.normal(size=81), rng.uniform(0.5, 1.5, size=81)
    peak_ln = rng.normal(size=200)
    base = TA.tnull_base(peak_ln, mu, sd, 21)
    assert base.shape == (200, 81)
    assert np.max(np.abs(base[:, 21] - peak_ln)) < 1e-12


def test_config_validation(tmp_path):
    good = {"schema": TA.PROBE_SCHEMA, "name": "x/y", "arm": "P-mlp", "fold": 0, "cache_dir": "c", "split_path": "s",
            "data_root": "d", "frame_stats_path": "f", "waveform_path": "w", "out_dir": "o"}
    p = tmp_path / "a.json"
    p.write_text(json.dumps(good))
    cfg = TA.load_probe_config(p)
    assert cfg["head"]["inputs"] == ["query_x", "patch_context", "peak_ln"] and cfg["train"]["steps"] == 4000
    for bad in ({"head": {"anchor": False}}, {"head": {"inputs": ["nope"]}}, {"unknown": 1}, {"train": {"selection": "best"}}):
        p.write_text(json.dumps({**good, **bad}))
        with pytest.raises((ValueError, KeyError)):
            TA.load_probe_config(p)


def test_input_builder_standardises_on_training_cases():
    rng = np.random.default_rng(2)
    cases = [{"peak_ln": rng.normal(size=n), "query_x": rng.normal(3, 2, size=(n, 4)), "patch_context": rng.normal(size=6),
              "raw": rng.normal(size=(n, 3))} for n in (50, 70)]
    b = TA.InputBuilder(["query_x", "patch_context", "peak_ln"], cases)
    block = np.concatenate([b.point_block(c) for c in cases])
    assert b.dim == 11 and block.shape == (120, 11)
    assert np.allclose(block[:, :4].mean(0), 0, atol=1e-5) and np.allclose(block[:, :4].std(0), 1, atol=1e-4)


def test_finetune_config_validation(tmp_path):
    from training_wss_min import time_adapter_finetune as FT
    good = {"schema": FT.SCHEMA, "name": "x/y", "arm": "TL-warm", "fold": 0, "donor_run": "r", "cache_dir": "c",
            "split_path": "s", "data_root": "d", "frame_stats_path": "f", "waveform_path": "w", "out_dir": "o"}
    p = tmp_path / "ft.json"
    p.write_text(json.dumps(good))
    cfg = FT.load_config(p)
    assert cfg["trainable_modules"] == ["fp", "local_wall_branch"] and cfg["head"]["inputs"] == ["query_x", "peak_ln"]
    for bad in ({"geometry_init": "scratch"}, {"head": {"inputs": ["patch_context"]}}, {"train": {"selection": "best"}}, {"extra": 1}):
        p.write_text(json.dumps({**good, **bad}))
        with pytest.raises((ValueError, KeyError)):
            FT.load_config(p)
