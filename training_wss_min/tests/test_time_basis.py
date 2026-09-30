"""时间基头 / 帧条件归一化 / EMA 选模 的合同测试（不读真实数据）。"""
import json
import numpy as np
import pytest

from training_wss_min import config as C
from training_wss_min import dataset as D
from training_wss_min import time_basis as TB


def _stats(T=81, floor=0.05):
    rng = np.random.default_rng(0)
    mu = np.linspace(-1.5, 0.6, T); sd = 1.0 + 0.2 * rng.random(T)
    return {"method": "log_z", "eps": 1e-6, "floor": floor, "log": {"mean": float(mu.mean()), "std": 1.3},
            "linear": {"mean": 4.0, "std": 7.0}, "frame": {"log_mean": mu.tolist(), "log_std": sd.tolist()}}


def _basis_file(tmp_path, stats, T=81):
    rng = np.random.default_rng(1)
    mu = np.asarray(stats["frame"]["log_mean"]); sd = np.asarray(stats["frame"]["log_std"])
    # 随机正交基，第 0 列固定为常向量（模拟去均值 PCA 的零特征向量在末尾）：取 QR 后剔除常向量分量
    A = rng.standard_normal((T, T)); A -= A.mean(axis=0, keepdims=True)
    Q, _ = np.linalg.qr(A)
    Q -= Q.mean(axis=0, keepdims=True)  # 保证与常向量正交
    Q, _ = np.linalg.qr(Q)
    Q -= Q.mean(axis=0, keepdims=True); Q, _ = np.linalg.qr(Q)
    path = tmp_path / "basis.npz"
    np.savez(path, mu=mu, sigma=sd, phi=Q, steps=np.arange(1120, 1281, 2), floor=stats["floor"], eps=stats["eps"], weights="uniform")
    return path


def test_frame_stats_view_matches_manual_normalisation():
    stats = _stats(); tau = np.array([0.01, 0.5, 3.0, 40.0])
    k = 7
    z = D.normalize_wss(tau, D.frame_stats_view(stats, k))
    expect = (np.log(np.clip(tau, 0.05, None) + 1e-6) - stats["frame"]["log_mean"][k]) / stats["frame"]["log_std"][k]
    assert np.allclose(z, expect)
    assert np.allclose(D.denormalize_wss(z, D.frame_stats_view(stats, k)), np.clip(tau, 0.05, None))
    with pytest.raises(ValueError):
        D.frame_stats_view({"method": "log_z", "eps": 1e-6, "log": {"mean": 0, "std": 1}}, 0)


def test_time_basis_project_reconstruct_roundtrip(tmp_path):
    stats = _stats(); path = _basis_file(tmp_path, stats)
    basis = TB.load_time_basis(path, 8)
    TB.check_stats(basis, stats)
    rng = np.random.default_rng(2); N = 50
    coef = rng.standard_normal((N, 9))
    z = TB.reconstruct(coef, basis)
    assert z.shape == (81, N)
    back = TB.project(z, basis)
    assert np.allclose(back, coef, atol=1e-10)      # 基正交且与常向量正交 ⇒ 投影精确还原
    # 截断投影 = 最小二乘（残差与基正交）
    z_full = z + rng.standard_normal((81, N)) * 0.1
    c2 = TB.project(z_full, basis); resid = z_full - TB.reconstruct(c2, basis)
    assert np.allclose(basis["phi_k"].T @ resid, 0, atol=1e-10) and np.allclose(resid.mean(axis=0), 0, atol=1e-10)
    with pytest.raises(ValueError):
        TB.check_stats(basis, {**stats, "frame": {"log_mean": stats["frame"]["log_mean"], "log_std": (np.asarray(stats["frame"]["log_std"]) + 1e-3).tolist()}})
    with pytest.raises(ValueError):
        TB.load_time_basis(path, 0)


def _cfg(**over):
    cfg = C.ExpConfig()
    cfg.data.input_features = ("x", "y", "z")
    cfg.train.selection_rule = "train_loss"; cfg.train.ckpt_metric = "train_loss"
    for k, v in over.items():
        obj, attr = k.split(".")
        setattr(getattr(cfg, obj), attr, v)
    return cfg


def test_default_config_still_validates_and_time_basis_rules():
    C.validate_features(_cfg())  # 旧行为：timesteps='peak'，无时间基
    good = _cfg(**{"data.timesteps": "time_basis", "data.time_basis_path": "/x/basis.npz", "data.target_normalization": "frame_stats",
                   "data.waveform_path": "/x/wf.json", "model.time_basis_k": 8, "model.out_dim": 9})
    C.validate_features(good)
    for bad in (
        {"data.timesteps": "time_basis", "data.time_basis_path": "/x", "data.target_normalization": "frame_stats", "model.time_basis_k": 8, "model.out_dim": 1},
        {"data.timesteps": "time_basis", "data.time_basis_path": "/x", "data.target_normalization": "global_stats", "model.time_basis_k": 8, "model.out_dim": 9},
        {"data.timesteps": "peak", "model.time_basis_k": 8, "model.out_dim": 9},
        {"data.timesteps": "peak", "data.target_normalization": "frame_stats"},
        {"data.timesteps": "time_basis", "data.time_basis_path": "/x", "data.target_normalization": "frame_stats", "model.time_basis_k": 8, "model.out_dim": 9,
         "train.loss_pinball_lambda": 0.2},
    ):
        with pytest.raises(ValueError):
            C.validate_features(_cfg(**bad))
    # random_frame + frame_stats 允许；random_frame + case_max 仍拒绝
    rf = _cfg(**{"data.timesteps": "random_frame", "data.waveform_path": "/x/wf.json", "data.target_normalization": "frame_stats",
                 "data.input_features": ("x", "y", "z", "q_norm", "dq_norm", "t_sin", "t_cos")})
    C.validate_features(rf)
    with pytest.raises(ValueError):
        C.validate_features(_cfg(**{"data.timesteps": "random_frame", "data.waveform_path": "/x/wf.json", "data.target_normalization": "case_max",
                                  "data.input_features": ("x", "y", "z", "q_norm")}))


def test_normalize_frames_frame_stats_vs_global():
    stats = _stats(); frames = np.abs(np.random.default_rng(3).standard_normal((81, 20))) * 5
    zf = D._normalize_frames(frames, stats, "frame_stats"); zg = D._normalize_frames(frames, stats, "global_stats")
    assert zf.shape == zg.shape == (81, 20)
    assert np.allclose(zg, D.normalize_wss(frames, stats))
    k = 30
    assert np.allclose(zf[k], D.normalize_wss(frames[k], D.frame_stats_view(stats, k)))


def test_subset_case_slices_multiframe_labels():
    n = 10; rows = np.array([1, 4, 7])
    case = {"pos": np.zeros((n, 3)), "y_norm": np.arange(n * 3).reshape(n, 3), "y_raw_frames": np.arange(81 * n).reshape(81, n),
            "y_norm_frames": np.arange(81 * n).reshape(81, n) * 0.5, "steps": np.arange(81), "unit_id": "AG/x/y"}
    sub = D.subset_case(case, rows)
    assert sub["y_norm"].shape == (3, 3) and sub["y_raw_frames"].shape == (81, 3) and sub["y_norm_frames"].shape == (81, 3)
    assert np.array_equal(sub["y_raw_frames"][:, 1], case["y_raw_frames"][:, 4]) and sub["steps"].shape == (81,)


def test_volume_time_metrics_pooled_nmae_range():
    from training_wss_min.volume_time import VolumeTimeMetrics, PEAK_INDEX
    q = np.linspace(0.2, 1.0, 81)
    rng = np.random.default_rng(0)
    yt1 = rng.normal(size=(81, 40))
    yt2 = rng.normal(size=(81, 25)) + 3.0
    yp1 = yt1 + 0.1
    yp2 = yt2.copy()
    yp2[PEAK_INDEX] = yt2[PEAK_INDEX]
    met = VolumeTimeMetrics(q, "pressure")
    met.add(yt1, yp1)
    met.add(yt2, yp2)
    st = np.concatenate([yt1, yt2], axis=1)
    sp = np.concatenate([yp1, yp2], axis=1)
    expect = np.mean(np.abs(st - sp), axis=1) / np.maximum(st.max(axis=1) - st.min(axis=1), 1e-12)
    got = np.asarray(met.summary()["frame_nmae_range"])
    assert np.allclose(got, expect)
    assert got[PEAK_INDEX] == pytest.approx(expect[PEAK_INDEX])
    vel = VolumeTimeMetrics(q, "velocity")
    yt = rng.normal(size=(81, 30, 3))
    yp = yt + 0.05
    vel.add(yt, yp)
    st = np.linalg.norm(yt, axis=2)
    sp = np.linalg.norm(yp, axis=2)
    expect_v = np.mean(np.abs(st - sp), axis=1) / np.maximum(st.max(axis=1) - st.min(axis=1), 1e-12)
    assert np.allclose(vel.summary()["frame_nmae_range"], expect_v)
