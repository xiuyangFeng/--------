"""阶段 0-A3：T-null 自由基线 —— 「峰值帧真值 × 训练折平均时间形状」能拿到多少周期分数。

任何融入 t 的体场模型，**周期指标必须显著超过 T-null**，否则它学到的只是入口波形驱动的公共
时间形状（172 例逐位相同），而不是几何相关的时空型态。WSS 线 T0 就是在这一关通过、在峰值帧
硬门禁上失败的；体场线先把这条尺子立起来。

T-null 构造（线性目标空间，逐帧标准化口径与训练一致）：

    z_null(x, t) = z_true(x, PEAK) · r(t),   r(t) = 训练折 corr 意义下的最优公共缩放
                                             ≡ Σ_x z(x,t) z(x,PEAK) / Σ_x z(x,PEAK)²（训练折全点云）

r(t) 由 A0 的时间矩精确算出（Gram 的第 PEAK 行 / 对角元），无需重读原始帧；留出折上按体场周期
指标评分。同时给出两个参照：
- `peak_freeze`：r(t) ≡ 1（把峰值帧真值当作整周期预测），衡量"完全不建模时间"的代价；
- `frame_mean`：预测恒为逐帧均值（z ≡ 0），即 R²_cb 的零基线，用来确认指标实现没有偏置。

    python -u -m training_wss_min.experiments.volume_time_20260919.offline.a3_tnull
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from training_wss_min.experiments.volume_time_20260919.offline import common as K  # noqa: E402
from training_wss_min.experiments.volume_time_20260919.offline.a1_frame_stats import load_moments  # noqa: E402
from training_wss_min.experiments.volume_time_20260919.offline.a2_time_basis import load_sub  # noqa: E402


def common_shape(scope: str, target: str, cache: dict) -> np.ndarray:
    """训练折公共时间形状 r(t)：压力 (81,)；速度 (3, 81)（逐分量各一条）。"""
    mu, sd, _ = K.frame_stats(scope, target)
    names = ('pressure_interior', 'pressure_wall') if target == 'pressure' else \
            tuple(f'velocity_{c}' for c in K.VELOCITY_COMPONENTS)
    if target == 'pressure':
        G = np.zeros((K.N_FRAMES, K.N_FRAMES))
        for name in names:
            acc = K.ChannelMoments()
            for cid in K.SCOPES[scope]:
                acc.add(*cache[cid][name])
            G += acc.gram(mu, sd)
        return G[K.PEAK_INDEX] / G[K.PEAK_INDEX, K.PEAK_INDEX]
    rows = []
    for ci, name in enumerate(names):
        acc = K.ChannelMoments()
        for cid in K.SCOPES[scope]:
            acc.add(*cache[cid][name])
        G = acc.gram(mu[ci], sd[ci])
        rows.append(G[K.PEAK_INDEX] / G[K.PEAK_INDEX, K.PEAK_INDEX])
    return np.stack(rows)


def evaluate(scope: str, held: list[str], target: str, r: np.ndarray, q_norm) -> dict:
    mu, sd, _ = K.frame_stats(scope, target)
    arms = {'tnull': r, 'peak_freeze': np.ones_like(r), 'frame_mean': np.zeros_like(r)}
    mets = {name: K.VolumeTimeMetrics(q_norm, target) for name in arms}
    for cid in held:
        y = load_sub(cid, target)
        if target == 'pressure':
            z = (y - mu[:, None]) / sd[:, None]
            zp = z[K.PEAK_INDEX]
            for name, shape in arms.items():
                mets[name].add(y, mu[:, None] + sd[:, None] * (shape[:, None] * zp[None, :]))
        else:
            z = (y - mu.T[:, None, :]) / sd.T[:, None, :]
            zp = z[K.PEAK_INDEX]
            for name, shape in arms.items():
                zr = shape.T[:, None, :] * zp[None, :, :]
                mets[name].add(y, mu.T[:, None, :] + sd.T[:, None, :] * zr)
    return {name: m.summary() for name, m in mets.items()}


def main() -> int:
    t0 = time.time()
    cache = {cid: load_moments(cid) for cid in K.TRAIN}
    q_norm = np.asarray(K.load_waveform()['q_norm'], np.float64)
    res = {}
    for target in K.TARGETS:
        res[target] = {}
        for f in range(3):
            scope = f'fold{f}'
            r = common_shape(scope, target, cache)
            out = evaluate(scope, K.FOLD_HELD[f], target, r, q_norm)
            res[target][scope] = {'r_t': r.tolist(), 'arms': out}
            print(f'[a3] {target} {scope}  r(t) 范围 {np.min(r):+.3f}..{np.max(r):+.3f}', flush=True)
            for name in ('tnull', 'peak_freeze', 'frame_mean'):
                print(f'    {name:12s} {K.fmt_summary(out[name])}', flush=True)
        keys = ('cycle_r2cb', 'peak_r2cb', 'trough_r2cb', 'tmean_r2cb', 'ts_corr',
                'peak_time_err_med_frames', 'amp_err_med')
        res[target]['fold_mean'] = {
            name: {k: float(np.mean([res[target][f'fold{f}']['arms'][name][k] for f in range(3)])) for k in keys}
            for name in ('tnull', 'peak_freeze', 'frame_mean')}
        print(f'\n=== {target} 三折均值 ===\n  arm           cycle   peak   trough  tmean   ts-corr  pk-t  amp-err')
        for name, m in res[target]['fold_mean'].items():
            print(f'  {name:12s}  {m["cycle_r2cb"]:.4f}  {m["peak_r2cb"]:.4f}  {m["trough_r2cb"]:.4f}  '
                  f'{m["tmean_r2cb"]:.4f}  {m["ts_corr"]:.3f}   {m["peak_time_err_med_frames"]:.1f}   '
                  f'{m["amp_err_med"]:.4f}', flush=True)
        print('', flush=True)
    K.write_json(K.OUT / 'a3_tnull.json', res)
    print(f'[a3] done in {time.time() - t0:.0f}s -> a3_tnull.json', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
