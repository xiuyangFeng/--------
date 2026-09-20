"""阶段 0-A2：体场时间基（训练折 PCA）+ 留出折真值投影的**指标级上限** → 定 K*（压力/速度各一套）。

口径与 WSS 线 `d1_time_basis_ceiling.py` 逐条对应，差别只在目标空间是线性的：

    z(x, t) = (y(x, t) − μ_f(t)) / σ_f(t)          压力 y = p_rel(Pa)；速度 y = 逐分量 m/s
    b0(x)   = mean_t z(x, t)                       时间基头的第 0 通道
    基       = 对 zc = z − b0 的 PCA（均匀帧权），Φ 列正交归一，符号取峰值帧分量为正

基本身用**全点云**拟合（由 A0 的逐例时间矩精确求得，不必重读 81 帧）；上限指标在 A0 缓存的
逐例子采样轨迹（2 万内部单元 / 4 千壁面节点）上算，因为峰时误差、峰谷幅值这类指标无法由二阶矩推出。

速度三分量**共用一套 Φ**（在三分量合并的中心化 Gram 上拟合），因此时间基头输出 3·(K+1) 通道；
报告里同时给出逐分量能量，便于判断共用基是否够用。

产出：
- `time_basis_<target>_<scope>.npz`：training_wss_min/time_basis.py 的加载合同（含 channels/space）
- `a2_time_basis.json`：逐折能量谱、逐 K 上限指标、三折均值与 K*

    python -u -m training_wss_min.experiments.volume_time_20260919.offline.a2_time_basis
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from training_wss_min.experiments.volume_time_20260919.offline import common as K  # noqa: E402
from training_wss_min.experiments.volume_time_20260919.offline.a1_frame_stats import load_moments  # noqa: E402

KS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 24]
# 预注册 K* 规则（与 WSS 线同形）：谷底四分位 R²_cb ≥ 0.95 且时均场 R²_cb ≥ 0.98（三折均值）
GATE_TROUGH, GATE_TMEAN = 0.95, 0.98


def fit_basis(scope: str, target: str, cache: dict):
    """返回 (mu, sigma, Phi, energy, energy_per_channel)。mu/sigma：压力 (81,)、速度 (3, 81)。"""
    mu, sd, _ = K.frame_stats(scope, target)
    names = ('pressure_interior', 'pressure_wall') if target == 'pressure' else \
            tuple(f'velocity_{c}' for c in K.VELOCITY_COMPONENTS)
    G = np.zeros((K.N_FRAMES, K.N_FRAMES))
    n = 0
    per_channel = {}
    for ci, name in enumerate(names):
        acc = K.ChannelMoments()
        for cid in K.SCOPES[scope]:
            acc.add(*cache[cid][name])
        m, s = (mu, sd) if target == 'pressure' else (mu[ci], sd[ci])
        g = acc.gram(m, s)
        G += g
        n += acc.n
        w_c, _ = K.time_basis_from_gram(g, acc.n)
        per_channel[name] = (np.cumsum(np.maximum(w_c, 0)) / max(np.maximum(w_c, 0).sum(), 1e-300)).tolist()
    w, V = K.time_basis_from_gram(G, n)
    energy = np.cumsum(np.maximum(w, 0)) / max(np.maximum(w, 0).sum(), 1e-300)
    channels = 1 if target == 'pressure' else 3
    np.savez(K.OUT / f'time_basis_{target}_{scope}.npz',
             mu=mu, sigma=sd, phi=V, eigvals=w, energy=energy,
             channels=np.int64(channels), space=np.asarray('linear'),
             floor=np.float64(0.0), eps=np.float64(0.0),
             steps=np.asarray(np.arange(1120, 1281, 2), dtype=np.int64),
             n_points=np.int64(n), weights=np.asarray('uniform'),
             target=np.asarray(target), scope=np.asarray(scope),
             cases=np.asarray(K.SCOPES[scope]))
    return mu, sd, V, energy, per_channel


def load_sub(cid: str, target: str) -> np.ndarray:
    """A0 缓存的子采样轨迹：压力 (81, n_int + n_wall)；速度 (81, n_int, 3)。"""
    with np.load(K.SCAN / cid / 'sub.npz', allow_pickle=False) as z:
        if target == 'pressure':
            return np.concatenate([z['pressure'].astype(np.float64), z['wall_pressure'].astype(np.float64)], axis=1)
        return z['velocity'].astype(np.float64)


def ceiling(mu, sd, V, held: list[str], target: str, q_norm) -> dict:
    """留出折真值投影到前 K 个基（加 b0）后重建，再按体场周期指标算上限。"""
    mets = {k: K.VolumeTimeMetrics(q_norm, target) for k in KS}
    for cid in held:
        y = load_sub(cid, target)
        if target == 'pressure':
            z = (y - mu[:, None]) / sd[:, None]
        else:
            z = (y - mu.T[:, None, :]) / sd.T[:, None, :]
        b0 = z.mean(axis=0)
        coef = np.tensordot(V.T, z - b0, axes=(1, 0))           # (81, ...)
        for k in KS:
            zr = b0 + np.tensordot(V[:, :k], coef[:k], axes=(1, 0))
            y_hat = mu[:, None] + sd[:, None] * zr if target == 'pressure' else \
                mu.T[:, None, :] + sd.T[:, None, :] * zr
            mets[k].add(y, y_hat)
    return {k: m.summary() for k, m in mets.items()}


def main() -> int:
    t0 = time.time()
    cache = {cid: load_moments(cid) for cid in K.TRAIN}
    q_norm = np.asarray(K.load_waveform()['q_norm'], np.float64)
    res = {'KS': KS, 'gate': {'trough_r2cb': GATE_TROUGH, 'tmean_r2cb': GATE_TMEAN}}

    for target in K.TARGETS:
        res[target] = {}
        for f in range(3):
            scope = f'fold{f}'
            mu, sd, V, energy, per_channel = fit_basis(scope, target, cache)
            cl = ceiling(mu, sd, V, K.FOLD_HELD[f], target, q_norm)
            res[target][scope] = {'energy': energy.tolist(), 'energy_per_channel': per_channel,
                                  'ceiling': {str(k): v for k, v in cl.items()}}
            print(f'[a2] {target} {scope}: 累计能量 K=1/2/3/5/8/12/16 = '
                  + '/'.join(f'{energy[k - 1]:.3f}' for k in (1, 2, 3, 5, 8, 12, 16)), flush=True)
            for k in KS:
                print(f'    K={k:2d}  {K.fmt_summary(cl[k])}', flush=True)
        mu, sd, V, energy, per_channel = fit_basis('train136', target, cache)
        cl = ceiling(mu, sd, V, K.TEST, target, q_norm)
        res[target]['train136'] = {'energy': energy.tolist(), 'energy_per_channel': per_channel,
                                   'ceiling_test34': {str(k): v for k, v in cl.items()}}
        print(f'[a2] {target} train136: 累计能量 K=1/2/3/5/8/12/16 = '
              + '/'.join(f'{energy[k - 1]:.3f}' for k in (1, 2, 3, 5, 8, 12, 16)), flush=True)

        keys = ('cycle_r2cb', 'peak_r2cb', 'trough_r2cb', 'tmean_r2cb', 'min_frame_r2cb',
                'peak_time_err_med_frames', 'peak_time_err_p90_frames', 'ts_corr', 'amp_err_med')
        mean = {k: {key: float(np.mean([res[target][f'fold{f}']['ceiling'][str(k)][key] for f in range(3)]))
                    for key in keys} for k in KS}
        print(f'\n=== {target} 三折均值上限 ===\n   K   cycle   peak   trough  tmean   min     pk-t med/p90   ts-corr  amp-err')
        for k in KS:
            m = mean[k]
            print(f'  {k:2d}  {m["cycle_r2cb"]:.4f}  {m["peak_r2cb"]:.4f}  {m["trough_r2cb"]:.4f}  '
                  f'{m["tmean_r2cb"]:.4f}  {m["min_frame_r2cb"]:.4f}  '
                  f'{m["peak_time_err_med_frames"]:.1f}/{m["peak_time_err_p90_frames"]:.1f}        '
                  f'{m["ts_corr"]:.3f}   {m["amp_err_med"]:.4f}', flush=True)
        k_star = next((k for k in KS if mean[k]['trough_r2cb'] >= GATE_TROUGH
                       and mean[k]['tmean_r2cb'] >= GATE_TMEAN), None)
        res[target]['fold_mean'] = {str(k): mean[k] for k in KS}
        res[target]['k_star'] = k_star
        print(f'\nK*({target}) = {k_star}  '
              f'（规则：谷底四分位 R²cb ≥ {GATE_TROUGH} 且时均场 R²cb ≥ {GATE_TMEAN}，三折均值）\n', flush=True)

    K.write_json(K.OUT / 'a2_time_basis.json', res)
    print(f'[a2] done in {time.time() - t0:.0f}s -> a2_time_basis.json', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
