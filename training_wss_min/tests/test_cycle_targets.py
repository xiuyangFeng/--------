"""周期积分量目标（tawss / osi）合同测试：标签公式、80 帧权、logit_z 归一化、阈值掩膜指标、配置校验（不读真实数据）。"""
import json
import math

import numpy as np
import pytest

from training_wss_min import config as C
from training_wss_min import dataset as D
from training_wss_min import metrics as M
from wss_v5.views import wall_cycle_v1 as CY


def _synthetic_vec(n=50, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 2.0 * np.pi, 81)                       # frame 80 = frame 0 + T
    base = rng.standard_normal((n, 3))
    amp = 1.0 + 0.8 * np.sin(t)[:, None, None]                  # (81, 1, 1)
    vec = amp * base[None]                                      # periodic magnitude, fixed direction
    flip = np.ones(81); flip[40:60] = -1.0                      # 20 frames reversed for half the points
    vec[:, : n // 2] *= flip[:, None, None]
    return vec


def test_cycle_labels_hand_computed():
    vec = _synthetic_vec()
    out = CY.cycle_labels(vec)["arrays"]
    mag = np.linalg.norm(vec, axis=2)
    # TAWSS uses frames 0..79 only; frame 80 duplicates phase 0 and must be excluded
    assert np.allclose(out["wall_tawss"], mag[:80].mean(0), atol=1e-6)
    assert not np.allclose(out["wall_tawss"], mag.mean(0), atol=1e-6) or np.allclose(mag[0], mag[80])
    mv = np.linalg.norm(vec[:80].mean(0), axis=1)
    osi = 0.5 * (1 - mv / mag[:80].mean(0))
    assert np.allclose(out["wall_osi"], osi, atol=1e-6)
    assert out["wall_osi"].min() >= 0.0 and out["wall_osi"].max() <= 0.5
    # points with no reversal → OSI 0, reversed points → OSI > 0 and rev_frac = 20/80
    assert np.allclose(out["wall_osi"][25:], 0.0, atol=1e-6)
    assert (out["wall_osi"][:25] > 0.05).all()
    assert np.allclose(out["wall_rev_frac"][:25], 0.25) and np.allclose(out["wall_rev_frac"][25:], 0.0)
    rrt = 1.0 / ((1 - 2 * out["wall_osi"]) * np.clip(out["wall_tawss"], 0.05, None))
    assert np.allclose(out["wall_rrt"], rrt, rtol=1e-5)


def test_cycle_labels_rejects_wrong_shape():
    with pytest.raises(ValueError):
        CY.cycle_labels(np.zeros((80, 5, 3)))


def test_logit_z_normalization_roundtrip():
    stats = {"method": "logit_z", "eps": 0.0, "logit": {"mean": -1.2, "std": 1.7, "scale": 0.5, "clip": 1e-3}}
    osi = np.array([0.0, 0.001, 0.05, 0.25, 0.49, 0.5])
    z = D.normalize_wss(osi, stats)
    back = D.denormalize_wss(z, stats)
    inner = (osi > 0.5 * 1e-3) & (osi < 0.5 * (1 - 1e-3))
    assert np.allclose(back[inner], osi[inner], atol=1e-9)
    assert (back >= 0).all() and (back <= 0.5).all()
    assert np.isfinite(z).all()


def test_linear_method_unchanged_for_osi_linear_stats():
    stats = {"method": "linear", "eps": 0.0, "linear": {"mean": 0.1, "std": 0.08}}
    v = np.array([0.0, 0.1, 0.3])
    assert np.allclose(D.denormalize_wss(D.normalize_wss(v, stats), stats), v)


def test_threshold_mask_metrics():
    yt = np.array([0.0, 0.2, 0.3, 0.05, 0.4])
    yp = np.array([0.15, 0.2, 0.05, 0.05, 0.4])
    r = M.threshold_mask_metrics(yt, yp, 0.1, above=True)
    # true > 0.1: idx 1,2,4 ; pred > 0.1: idx 0,1,4 → inter {1,4}=2, union {0,1,2,4}=4
    assert math.isclose(r["iou"], 0.5) and math.isclose(r["precision"], 2 / 3) and math.isclose(r["recall"], 2 / 3)
    assert math.isclose(r["true_frac"], 0.6) and math.isclose(r["pred_frac"], 0.6) and math.isclose(r["frac_abs_err"], 0.0)
    r2 = M.threshold_mask_metrics(yt, yp, 0.1, above=False)
    # true < 0.1: idx 0,3 ; pred < 0.1: idx 2,3 → inter {3}=1, union {0,2,3}=3
    assert math.isclose(r2["iou"], 1 / 3)
    empty = M.threshold_mask_metrics(np.array([0.0, 0.0]), np.array([0.0, 0.0]), 0.1, above=True)
    assert math.isnan(empty["iou"]) and math.isclose(empty["frac_abs_err"], 0.0)


def _base_cfg(tmp_path, target, method="log_z"):
    d = {"data": {"target": target, "input_features": ["x", "y", "z"], "cycle_view_root": str(tmp_path)},
         "model": {}, "train": {}, "eval": {}}
    return d


def test_config_cycle_target_requires_view_root_and_rejects_root_for_wss(tmp_path):
    # from_dict already runs validate_features, so construction itself must fail
    with pytest.raises(ValueError, match="cycle_view_root"):
        C.ExpConfig.from_dict({"data": {"target": "tawss", "input_features": ["x", "y", "z"]}, "model": {}, "train": {}, "eval": {}})
    cfg = C.ExpConfig.from_dict(_base_cfg(tmp_path, "osi"))
    C.validate_features(cfg)   # osi + root is fine
    with pytest.raises(ValueError, match="only meaningful"):
        C.ExpConfig.from_dict({"data": {"target": "wss", "input_features": ["x", "y", "z"], "cycle_view_root": str(tmp_path)},
                               "model": {}, "train": {}, "eval": {}})
    with pytest.raises(ValueError):
        C.ExpConfig.from_dict({**_base_cfg(tmp_path, "tawss"), "data": {**_base_cfg(tmp_path, "tawss")["data"], "timesteps": "random_frame",
                                                                        "waveform_path": "x.json", "input_features": ["x", "y", "z", "q_norm"]}})


def test_config_threshold_masks_parse_and_default_empty(tmp_path):
    d = _base_cfg(tmp_path, "osi")
    d["eval"] = {"threshold_masks_above": [0.1, 0.3], "threshold_masks_below": []}
    cfg = C.ExpConfig.from_dict(d)
    C.validate_features(cfg)
    assert cfg.eval.threshold_masks_above == (0.1, 0.3) and cfg.eval.threshold_masks_below == ()
    old = C.ExpConfig.from_dict({"data": {"input_features": ["x", "y", "z"]}, "model": {}, "train": {}, "eval": {}})
    assert old.data.cycle_view_root is None and old.eval.threshold_masks_above == () and old.eval.threshold_masks_below == ()


def test_load_cycle_label_checks_rows(tmp_path):
    root = tmp_path / "cyc"; (root / "AG/fast/CASE").mkdir(parents=True)
    ids = np.arange(10, 20, dtype=np.int64)
    np.savez(root / "AG/fast/CASE/cycle.npz", wall_node_id_cas=ids, wall_tawss=np.linspace(0.1, 2.0, 10).astype(np.float32),
             wall_osi=np.linspace(0.0, 0.5, 10).astype(np.float32))
    y = D.load_cycle_label(root, "AG/fast", "CASE", "tawss", ids)
    assert y.shape == (10,) and y.dtype == np.float32
    with pytest.raises(ValueError, match="rows"):
        D.load_cycle_label(root, "AG/fast", "CASE", "osi", ids[::-1])
    with pytest.raises(FileNotFoundError):
        D.load_cycle_label(root, "AG/fast", "NOPE", "osi", ids)
    with pytest.raises(ValueError, match="cycle_view_root"):
        D.load_cycle_label(None, "AG/fast", "CASE", "osi", ids)


def test_lin_ccc_and_dice():
    a = np.array([1.0, 2.0, 3.0, 4.0]); b = 2.0 * a
    assert math.isclose(M.lin_ccc(a, a), 1.0)
    assert M.lin_ccc(a, b) < 1.0 and M.lin_ccc(a, b) > 0.0     # perfect correlation but amplitude bias → CCC < 1
    assert math.isclose(M.lin_ccc(a, b), 2 * np.cov(a, b, bias=True)[0, 1] / (a.var() + b.var() + (a.mean() - b.mean()) ** 2))
    assert math.isclose(M.dice_masks(np.array([1, 1, 0, 0], bool), np.array([1, 0, 1, 0], bool)), 0.5)
    assert math.isnan(M.dice_masks(np.zeros(3, bool), np.zeros(3, bool)))


def test_cycle_agreement_case_and_aggregate():
    rng = np.random.default_rng(3)
    yt = np.exp(rng.normal(0.0, 1.0, 2000)); yp = yt * np.exp(rng.normal(0.0, 0.2, 2000))
    seg = np.repeat(np.arange(5), 400)
    r = M.cycle_agreement_case_metrics(yt, yp, log_space=True, floor=0.05, rel_tolerance=0.3, segment_ids=seg,
                                       min_segment_points=200, masks_above=(), masks_below=(0.4,))
    assert 0.9 < r["ccc"] <= 1.0 and r["ccc_space"] == "log" and 0.0 < r["rel_err_med"] < 0.3 and 0.5 < r["within_tol"] <= 1.0
    assert r["n_segments"] == 5 and all(v["n"] == 400 for v in r["segments"].values()) and r["seg_rel_err_max"] >= r["seg_rel_err_med"]
    assert "below_0.4" in r["masks"] and 0.0 <= r["masks"]["below_0.4"]["dice"] <= 1.0 and 0.0 <= r["top_dice"] <= 1.0
    # segments below the point threshold are skipped; negative ids skipped
    r2 = M.cycle_agreement_case_metrics(yt, yp, log_space=False, floor=0.0, segment_ids=np.where(seg == 0, -1, seg), min_segment_points=500)
    assert r2["n_segments"] == 0 and math.isnan(r2["seg_rel_err_med"]) and r2["ccc_space"] == "linear"
    with pytest.raises(ValueError):
        M.cycle_agreement_case_metrics(yt, yp, log_space=True, floor=0.05, segment_ids=seg[:10])
    agg = M.cycle_agreement_aggregate({"a": r, "b": r})
    assert agg["n_cases"] == 2 and math.isclose(agg["ccc"]["med"], r["ccc"]) and agg["case_mean"]["n"] == 2
    assert agg["masks"]["below_0.4"]["n_dice"] == 2 and "area_share" in agg["masks"]["below_0.4"]
    ba = M._bland_altman([1.0, 2.0, 3.0, 4.0], [1.1, 2.0, 3.3, 4.0], relative=False)
    assert math.isclose(ba["bias"], 0.1) and ba["loa_low"] < 0.1 < ba["loa_high"] and ba["n"] == 4
    ba_rel = M._bland_altman([1.0, 2.0], [1.1, 2.4], relative=True)
    assert math.isclose(ba_rel["bias"], 0.15) and ba_rel["relative"] is True


def test_cycle_agreement_config_defaults_and_validation(tmp_path):
    cfg = C.ExpConfig.from_dict(_base_cfg(tmp_path, "tawss"))
    assert cfg.eval.cycle_agreement is True and cfg.eval.cycle_rel_tolerance == 0.3 and cfg.eval.cycle_min_segment_points == 200
    d = _base_cfg(tmp_path, "tawss"); d["eval"] = {"cycle_rel_tolerance": 0.0}
    with pytest.raises(ValueError, match="cycle_rel_tolerance"):
        C.ExpConfig.from_dict(d)


def _multi_stats():
    return {"method": "multi", "channels": ["wss", "tawss", "osi"], "eps": 1e-6, "floor": 0.05,
            "wss": {"method": "log_z", "eps": 1e-6, "floor": 0.0, "log": {"mean": 0.6, "std": 1.3}, "linear": {"mean": 4.0, "std": 7.0}},
            "tawss": {"method": "log_z", "eps": 1e-6, "floor": 0.05, "log": {"mean": -0.3, "std": 1.0}, "linear": {"mean": 1.3, "std": 1.9}},
            "osi": {"method": "logit_z", "eps": 0.0, "logit": {"mean": -1.4, "std": 1.8, "scale": 0.5, "clip": 1e-3}}}


def test_multi_stats_roundtrip_and_shape_checks():
    st = _multi_stats()
    y = np.stack([np.array([0.5, 3.0, 20.0]), np.array([0.2, 1.0, 5.0]), np.array([0.02, 0.2, 0.45])], axis=1)
    z = D.normalize_wss(y, st)
    assert z.shape == (3, 3) and np.isfinite(z).all()
    assert np.allclose(z[:, 0], D.normalize_wss(y[:, 0], st["wss"])) and np.allclose(z[:, 2], D.normalize_wss(y[:, 2], st["osi"]))
    back = D.denormalize_wss(z, st)
    assert np.allclose(back, y, rtol=1e-6, atol=1e-6)
    with pytest.raises(ValueError):
        D.normalize_wss(y[:, :2], st)
    with pytest.raises(ValueError):
        D.denormalize_wss(z[:, 0], st)


def test_config_multi_target_validation(tmp_path):
    base = {"data": {"target": "wss_cycle_multi", "input_features": ["x", "y", "z"], "cycle_view_root": str(tmp_path)},
            "model": {"out_dim": 3}, "train": {"loss": "mse", "loss_pinball_lambda": 0.0}, "eval": {}}
    cfg = C.ExpConfig.from_dict(base)
    assert cfg.model.out_dim == 3 and cfg.data.target == C.MULTI_TARGET
    bad = json.loads(json.dumps(base)); bad["model"]["out_dim"] = 1
    with pytest.raises(ValueError, match="out_dim"):
        C.ExpConfig.from_dict(bad)
    bad = json.loads(json.dumps(base)); bad["train"]["loss_pinball_lambda"] = 0.2
    with pytest.raises(ValueError, match="wss_cycle_multi"):
        C.ExpConfig.from_dict(bad)
    bad = json.loads(json.dumps(base)); bad["data"].pop("cycle_view_root")
    with pytest.raises(ValueError, match="cycle_view_root"):
        C.ExpConfig.from_dict(bad)
