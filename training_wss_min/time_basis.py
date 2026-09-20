"""时间基输出头（timesteps='time_basis'）的合同：基文件读取、系数投影、帧重建。

WSS（标量）基文件由 ``experiments/wss_time_ecc_20260918/offline/d1_time_basis_ceiling.py`` 在**训练折**上生成：
  mu (81,)、sigma (81,)：逐帧 ln(max(τ, floor) + eps) 的均值/标准差（= 帧条件归一化 frame_stats）
  phi (81, 81)：按能量降序的正交时间基（列向量；对去逐点时间均值后的 z 做 PCA）
  steps (81,)、floor、eps、weights='uniform'
z(x,t) = (ln(max(τ,floor)+eps) − mu(t)) / sigma(t) ≈ b0(x) + Σ_{k=1..K} a_k(x) φ_k(t)
每个壁面点的监督目标 = [b0, a_1, …, a_K]（K+1 维，标准化空间），推理一次重建全部 81 帧。
基正交归一且与常向量正交 ⇒ 系数 MSE（K+1 通道均值）∝ 帧空间 z 的 MSE。

体场（2026-09-19）：目标空间换成**线性** z（压力 Pa、速度逐分量 m/s），mu/sigma 是逐帧线性均值/标准差，
基文件另带 ``channels``（压力 1、速度 3）与 ``target``。多通道时每个点的目标按**通道优先**摊平成
C·(K+1) 列：列 c·(K+1)+j = 第 c 个分量的第 j 个系数。C=1 时与历史标量布局逐位相同。
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict

import numpy as np

_CACHE: Dict[str, Dict] = {}


def load_time_basis(path: str | Path, k: int) -> Dict:
    key = (str(Path(path).resolve()), int(k))
    if key in _CACHE:
        return _CACHE[key]
    if int(k) < 1:
        raise ValueError("time_basis_k must be >= 1")
    with np.load(Path(path), allow_pickle=False) as z:
        mu = np.asarray(z["mu"], dtype=np.float64)
        sigma = np.asarray(z["sigma"], dtype=np.float64)
        phi = np.asarray(z["phi"], dtype=np.float64)
        steps = np.asarray(z["steps"], dtype=np.int64)
        floor = float(z["floor"]); eps = float(z["eps"])
        # 体场基（多通道、线性目标空间）带这两个键；WSS 标量基没有，按历史默认值读
        channels = int(z["channels"]) if "channels" in z.files else 1
        space = str(z["space"]) if "space" in z.files else "log"
    n = mu.shape[-1]
    if channels == 1:
        if mu.shape != (n,) or sigma.shape != (n,):
            raise ValueError(f"scalar time basis needs mu/sigma of shape (T,): {path}")
    elif mu.shape != (channels, n) or sigma.shape != (channels, n):
        raise ValueError(f"multi-channel time basis needs mu/sigma of shape (C, T): {path}")
    if phi.shape != (n, n) or steps.shape != (n,):
        raise ValueError(f"time basis shape mismatch in {path}")
    if int(k) > n - 1:
        raise ValueError(f"time_basis_k={k} exceeds available modes ({n - 1})")
    phi_k = np.ascontiguousarray(phi[:, : int(k)])
    gram = phi_k.T @ phi_k
    if not np.allclose(gram, np.eye(int(k)), atol=1e-8):
        raise ValueError("time basis columns are not orthonormal")
    if not np.allclose(phi_k.sum(axis=0), 0.0, atol=1e-6):
        raise ValueError("time basis columns must be orthogonal to the constant (b0) direction")
    basis = dict(mu=mu, sigma=sigma, phi_k=phi_k, steps=steps, floor=floor, eps=eps, k=int(k), n_frames=n,
                 channels=channels, space=space,
                 path=str(Path(path).resolve()), sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())
    _CACHE[key] = basis
    return basis


def out_dim(basis: Dict) -> int:
    """时间基头的输出通道数 = 目标分量数 × (K+1)。"""
    return int(basis["channels"]) * (int(basis["k"]) + 1)


def check_stats(basis: Dict, stats: Dict) -> None:
    """基的 mu/sigma 必须与 frame_stats 统计逐位一致（同一训练折）。

    WSS 是 log 空间的标量（frame.log_mean/log_std）；体场是线性空间，压力取
    frame.linear_mean/linear_std、速度取 frame.velocity_mean/velocity_std（(T, 3) → 转置成 (C, T)）。
    """
    frame = stats.get("frame")
    if frame is None:
        raise ValueError("time_basis requires per-frame statistics ('frame' block) in the stats file")
    space, channels = basis["space"], int(basis["channels"])
    if space == "log":
        mean = np.asarray(frame["log_mean"], dtype=np.float64)
        std = np.asarray(frame["log_std"], dtype=np.float64)
    elif channels == 1:
        mean = np.asarray(frame["linear_mean"], dtype=np.float64)
        std = np.asarray(frame["linear_std"], dtype=np.float64)
    else:
        mean = np.asarray(frame["velocity_mean"], dtype=np.float64).T
        std = np.asarray(frame["velocity_std"], dtype=np.float64).T
    if not (np.allclose(basis["mu"], mean, atol=1e-9) and np.allclose(basis["sigma"], std, atol=1e-9)):
        raise ValueError("time basis mu/sigma differ from the frame statistics: basis and stats must come from the same train partition")
    if space == "log" and (abs(float(stats.get("floor", 0.0)) - basis["floor"]) > 1e-12
                           or abs(float(stats["eps"]) - basis["eps"]) > 1e-15):
        raise ValueError("time basis floor/eps differ from the stats file")


def project(z_frames: np.ndarray, basis: Dict) -> np.ndarray:
    """标准化帧 → 系数。(T, N) → (N, K+1)；多通道 (T, N, C) → (N, C·(K+1))，列按通道优先摊平。"""
    z = np.asarray(z_frames, dtype=np.float64)
    if z.shape[0] != basis["n_frames"]:
        raise ValueError("frame count differs from the time basis")
    channels = int(basis["channels"])
    if channels == 1:
        if z.ndim != 2:
            raise ValueError("scalar time basis expects (T, N) frames")
        b0 = z.mean(axis=0)
        a = basis["phi_k"].T @ (z - b0[None, :])          # (K, N)
        return np.concatenate([b0[None, :], a], axis=0).T  # (N, K+1)
    if z.ndim != 3 or z.shape[2] != channels:
        raise ValueError(f"multi-channel time basis expects (T, N, {channels}) frames")
    parts = []
    for c in range(channels):
        zc = z[:, :, c]
        b0 = zc.mean(axis=0)
        a = basis["phi_k"].T @ (zc - b0[None, :])
        parts.append(np.concatenate([b0[None, :], a], axis=0).T)   # (N, K+1)
    return np.concatenate(parts, axis=1)                            # (N, C*(K+1))


def reconstruct_frame(coef: np.ndarray, basis: Dict, frame_index: int) -> np.ndarray:
    """只重建单帧：(N, C·(K+1)) → (N,)（标量）或 (N, C）。全点云评估时不必先展开 81 帧。"""
    c = np.asarray(coef, dtype=np.float64)
    width = int(basis["k"]) + 1
    channels = int(basis["channels"])
    if c.ndim != 2 or c.shape[1] != width * channels:
        raise ValueError(f"coefficients must be (N, C*(K+1)) with K={basis['k']}, C={channels}")
    phi = basis["phi_k"][int(frame_index)]                            # (K,)
    out = [c[:, i * width] + c[:, i * width + 1:(i + 1) * width] @ phi for i in range(channels)]
    return out[0] if channels == 1 else np.stack(out, axis=1)


def reconstruct(coef: np.ndarray, basis: Dict) -> np.ndarray:
    """系数 → 标准化帧。(N, K+1) → (T, N)；多通道 (N, C·(K+1)) → (T, N, C)。"""
    c = np.asarray(coef, dtype=np.float64)
    width = int(basis["k"]) + 1
    channels = int(basis["channels"])
    if c.ndim != 2 or c.shape[1] != width * channels:
        raise ValueError(f"coefficients must be (N, C*(K+1)) with K={basis['k']}, C={channels}")
    if channels == 1:
        return c[:, 0][None, :] + basis["phi_k"] @ c[:, 1:].T
    out = [c[:, i * width] [None, :] + basis["phi_k"] @ c[:, i * width + 1:(i + 1) * width].T
           for i in range(channels)]
    return np.stack(out, axis=2)                                     # (T, N, C)
