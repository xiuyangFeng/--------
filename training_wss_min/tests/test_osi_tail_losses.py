"""2026-09-24 OSI 尾部辅助项（wss_cycle_multi 的 OSI 通道）：默认关闭时损失逐位不变、尾部加权只改 OSI 通道、
软掩膜 BCE 阈值映射与 dataset 归一化同口径、配置校验。"""
import json

import numpy as np
import pytest
import torch

from training_wss_min import config as C
from training_wss_min import dataset as D
from training_wss_min import objectives as O


def _multi_stats():
    return {"method": "multi", "channels": ["wss", "tawss", "osi"], "eps": 1e-6, "floor": 0.05,
            "wss": {"method": "log_z", "eps": 1e-6, "floor": 0.0, "log": {"mean": 0.6, "std": 1.3}, "linear": {"mean": 4.0, "std": 7.0}},
            "tawss": {"method": "log_z", "eps": 1e-6, "floor": 0.05, "log": {"mean": -0.3, "std": 1.0}, "linear": {"mean": 1.3, "std": 1.9}},
            "osi": {"method": "logit_z", "eps": 0.0, "logit": {"mean": -1.45, "std": 1.8, "scale": 0.5, "clip": 1e-3}}}


def _batch(n=400, seed=0):
    rng = np.random.default_rng(seed)
    st = _multi_stats()
    y_raw = np.stack([rng.lognormal(0.5, 1.0, n), rng.lognormal(-0.3, 0.8, n), rng.uniform(0.0, 0.48, n)], axis=1).astype(np.float32)
    y = D.normalize_wss(y_raw, st).astype(np.float32)
    pred = torch.from_numpy(y + rng.normal(0, 0.4, y.shape).astype(np.float32))
    batch = {"y": torch.from_numpy(y), "y_raw": torch.from_numpy(y_raw), "batch": torch.zeros(n, dtype=torch.long)}
    return pred, batch, st


def _train_cfg(**kw):
    return C.TrainConfig(**kw)


def test_defaults_leave_multi_loss_unchanged():
    pred, batch, st = _batch()
    cfg = _train_cfg()
    assert cfg.loss_osi_tail_alpha == 0.0 and cfg.loss_osi_mask_lambda == 0.0
    loss = O.compute_loss(pred, batch, cfg, "cpu", st)
    ref = (pred - batch["y"]).square().mean(-1).mean()
    assert torch.equal(loss, ref)


def test_tail_weights_only_touch_osi_channel():
    pred, batch, st = _batch()
    cfg = _train_cfg(loss_osi_tail_alpha=2.0, osi_tail_threshold=0.2)
    parts = {}
    loss = O.compute_loss(pred, batch, cfg, "cpu", st, components=parts)
    sq = (pred - batch["y"]).square()
    w = 1.0 + 2.0 * (batch["y_raw"][:, 2] > 0.2).float(); w = w / w.mean()
    manual = torch.stack([sq[:, 0], sq[:, 1], sq[:, 2] * w], dim=1).mean(-1).mean()
    assert torch.allclose(loss, manual, rtol=1e-6)
    assert abs(float(parts["osi_tail_fraction"]) - float((batch["y_raw"][:, 2] > 0.2).float().mean())) < 1e-7
    assert torch.allclose(O.osi_tail_weights(batch["y_raw"][:, 2], 0.2, 2.0).mean(), torch.tensor(1.0))


def test_mask_threshold_matches_dataset_normalization():
    st = _multi_stats()
    for t in (0.1, 0.2, 0.3):
        z = O.osi_threshold_z(t, st["osi"])
        assert np.isclose(z, D.normalize_wss(np.array([t]), st["osi"])[0], atol=1e-12)
        assert np.isclose(D.denormalize_wss(np.array([z]), st["osi"])[0], t, atol=1e-9)
    lin = {"method": "linear", "linear": {"mean": 0.1, "std": 0.08}}
    assert np.isclose(O.osi_threshold_z(0.3, lin), (0.3 - 0.1) / 0.08)


def test_mask_bce_added_and_decreases_when_prediction_crosses_threshold():
    pred, batch, st = _batch()
    base = O.compute_loss(pred, batch, _train_cfg(), "cpu", st)
    cfg = _train_cfg(loss_osi_mask_lambda=0.5, osi_mask_thresholds=(0.2, 0.3), osi_mask_temperature=0.25)
    parts = {}
    loss = O.compute_loss(pred, batch, cfg, "cpu", st, components=parts)
    assert torch.allclose(loss, base + 0.5 * parts["osi_mask_bce"], rtol=1e-6)
    perfect = batch["y"].clone()
    far = perfect.clone()
    hi = batch["y_raw"][:, 2] > 0.3
    far[hi, 2] += 3.0; far[~hi, 2] -= 3.0                       # push every point away from the 0.3 boundary on the right side
    bce_near = O.osi_soft_mask_bce(perfect[:, 2], batch["y_raw"][:, 2], st["osi"], (0.3,), 0.25)
    bce_far = O.osi_soft_mask_bce(far[:, 2], batch["y_raw"][:, 2], st["osi"], (0.3,), 0.25)
    assert float(bce_far) < float(bce_near)


def test_config_validation(tmp_path):
    base = {"data": {"target": "wss_cycle_multi", "input_features": ["x", "y", "z"], "cycle_view_root": str(tmp_path)},
            "model": {"out_dim": 3}, "train": {"loss": "mse", "loss_osi_tail_alpha": 2.0, "loss_osi_mask_lambda": 0.1,
                                               "osi_mask_thresholds": [0.2, 0.3]}, "eval": {}}
    cfg = C.ExpConfig.from_dict(base)
    assert cfg.train.osi_mask_thresholds == (0.2, 0.3)
    wss = json.loads(json.dumps(base)); wss["data"] = {"target": "wss", "input_features": ["x", "y", "z"]}; wss["model"]["out_dim"] = 1
    with pytest.raises(ValueError, match="wss_cycle_multi"):
        C.ExpConfig.from_dict(wss)
    for key, value, msg in (("loss_osi_tail_alpha", -1.0, "non-negative"), ("osi_mask_thresholds", [0.6], "osi_mask_thresholds"),
                            ("osi_mask_temperature", 0.0, "temperature"), ("osi_tail_threshold", 0.0, "osi_tail_threshold")):
        bad = json.loads(json.dumps(base)); bad["train"][key] = value
        with pytest.raises(ValueError, match=msg):
            C.ExpConfig.from_dict(bad)
    # 旧配置（无新键）解析后默认关闭
    old = json.loads(json.dumps(base)); [old["train"].pop(k) for k in ("loss_osi_tail_alpha", "loss_osi_mask_lambda", "osi_mask_thresholds")]
    cfg = C.ExpConfig.from_dict(old)
    assert cfg.train.loss_osi_tail_alpha == 0.0 and cfg.train.loss_osi_mask_lambda == 0.0
