"""2026-09-25 M1 辅助通道（data.multi_aux_channels）：配置校验、切平面坐标架、平均向量分量与派生 OSI、五通道归一化往返。"""
import json

import numpy as np
import pytest

from training_wss_min import config as C
from training_wss_min import dataset as D
from wss_v5.views import wall_cycle_v1 as CY


def _base(tmp_path, aux=(), out_dim=3, target="wss_cycle_multi"):
    return {"data": {"target": target, "input_features": ["x", "y", "z"], "cycle_view_root": str(tmp_path),
                     "multi_aux_channels": list(aux)},
            "model": {"out_dim": out_dim}, "train": {"loss": "mse"}, "eval": {}}


def test_config_validation(tmp_path):
    cfg = C.ExpConfig.from_dict(_base(tmp_path, ("mean_axial", "mean_circ"), 5))
    assert cfg.data.multi_aux_channels == ("mean_axial", "mean_circ")
    assert C.ExpConfig.from_dict(_base(tmp_path, ("rev_frac",), 4)).model.out_dim == 4
    old = _base(tmp_path); old["data"].pop("multi_aux_channels")
    assert C.ExpConfig.from_dict(old).data.multi_aux_channels == ()
    for aux, out_dim, msg in ((("mean_axial",), 4, "together"), (("rev_frac",), 3, "out_dim"), (("bogus",), 4, "distinct names"),
                              (("rev_frac", "rev_frac"), 5, "distinct names")):
        with pytest.raises(ValueError, match=msg):
            C.ExpConfig.from_dict(_base(tmp_path, aux, out_dim))
    bad = _base(tmp_path, ("rev_frac",), 1, target="wss"); bad["data"].pop("cycle_view_root")
    with pytest.raises(ValueError, match="multi_aux_channels requires"):
        C.ExpConfig.from_dict(bad)


def _geometry(n=200, seed=0):
    rng = np.random.default_rng(seed)
    normal = rng.standard_normal((n, 3)); normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    tangent = rng.standard_normal((n, 3)); tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    tangent[0] = normal[0]                                   # degenerate row: tangent parallel to the normal
    return np.column_stack([normal, tangent, np.ones(n), np.ones(n)])


def test_tangent_plane_axes_orthonormal():
    g = _geometry()
    axial, circ = D.tangent_plane_axes(g)
    normal = g[:, :3]
    assert np.allclose(np.linalg.norm(axial, axis=1), 1) and np.allclose(np.linalg.norm(circ, axis=1), 1)
    assert np.allclose((axial * normal).sum(1), 0, atol=1e-9) and np.allclose((circ * normal).sum(1), 0, atol=1e-9)
    assert np.allclose((axial * circ).sum(1), 0, atol=1e-9)


def test_mean_vector_ratios_and_derived_osi_match_definition():
    g = _geometry(n=300, seed=1)
    axial, circ = D.tangent_plane_axes(g)
    rng = np.random.default_rng(2)
    t = np.linspace(0, 2 * np.pi, 80, endpoint=False)
    amp_a = 1.0 + 0.8 * np.sin(t)[:, None] + 0.3 * rng.standard_normal((80, 300))
    amp_c = 0.4 * np.cos(2 * t)[:, None] + 0.2 * rng.standard_normal((80, 300))
    vec = amp_a[:, :, None] * axial[None] + amp_c[:, :, None] * circ[None]     # purely tangential shear
    r = D.cycle_mean_vector_ratios(vec, g)
    assert r.shape == (300, 2) and np.all(np.linalg.norm(r, axis=1) <= 1 + 1e-9)
    osi_def = CY.cycle_labels(np.concatenate([vec, vec[:1]]).astype(np.float32))["arrays"]["wall_osi"]
    assert np.allclose(D.derived_osi(r[:, 0], r[:, 1]), osi_def, atol=1e-5)
    # constant axial flow → r = (1, 0), OSI 0; perfectly reversing half-cycle → OSI 0.5
    const = np.repeat(axial[None], 80, axis=0)
    rc = D.cycle_mean_vector_ratios(const, g)
    assert np.allclose(rc, np.column_stack([np.ones(300), np.zeros(300)]), atol=1e-9)
    flip = const.copy(); flip[40:] *= -1
    assert np.allclose(D.derived_osi(*D.cycle_mean_vector_ratios(flip, g).T), 0.5)
    with pytest.raises(ValueError):
        D.cycle_mean_vector_ratios(vec[:79], g)


def test_five_channel_multi_normalization_roundtrip():
    st = {"method": "multi", "channels": ["wss", "tawss", "osi", "mean_axial", "mean_circ"], "eps": 1e-6, "floor": 0.05,
          "wss": {"method": "log_z", "eps": 1e-6, "floor": 0.0, "log": {"mean": 0.6, "std": 1.3}, "linear": {"mean": 4.0, "std": 7.0}},
          "tawss": {"method": "log_z", "eps": 1e-6, "floor": 0.05, "log": {"mean": -0.3, "std": 1.0}, "linear": {"mean": 1.3, "std": 1.9}},
          "osi": {"method": "logit_z", "eps": 0.0, "logit": {"mean": -1.4, "std": 1.8, "scale": 0.5, "clip": 1e-3}},
          "mean_axial": {"method": "linear", "linear": {"mean": 0.6, "std": 0.4}},
          "mean_circ": {"method": "linear", "linear": {"mean": 0.0, "std": 0.2}}}
    y = np.array([[2.0, 0.8, 0.1, 0.7, -0.1], [10.0, 3.0, 0.35, -0.2, 0.05]])
    z = D.normalize_wss(y, st)
    assert z.shape == (2, 5) and np.allclose(z[:, 3], (y[:, 3] - 0.6) / 0.4)
    assert np.allclose(D.denormalize_wss(z, st), y, atol=1e-9)


def test_four_channel_multi_loss_is_not_mistaken_for_joint_volume():
    import torch
    from training_wss_min import objectives as O
    st = {"method": "multi", "channels": ["wss", "tawss", "osi", "rev_frac"]}
    y = torch.randn(50, 4); pred = torch.randn(50, 4)
    loss = O.compute_loss(pred, {"y": y, "batch": torch.zeros(50, dtype=torch.long)}, C.TrainConfig(), "cpu", st)
    assert torch.allclose(loss, (pred - y).square().mean())
    with pytest.raises(ValueError, match="joint volume"):
        O.compute_loss(pred, {"y": y}, C.TrainConfig(), "cpu", {"method": "linear"})
