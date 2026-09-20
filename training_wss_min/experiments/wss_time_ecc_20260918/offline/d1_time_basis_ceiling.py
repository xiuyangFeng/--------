"""D1：训练折时间基（PCA）+ 留出折真值投影的**指标级上限**，定 K*。
z(x,t) = (ln max(τ,0.05)+eps − μ_f(t)) / σ_f(t)；基在去逐点时间均值 b0(x) 后的 z 上拟合（均匀帧权）。
上限 = 把留出折真值投影到前 K 个基上重建，再按 §8 口径算指标；K* = 最小 K 使谷底四分位 R²_cb ≥ 0.95 且 TAWSS ≥ 0.98（三折均值）。"""
import sys
import numpy as np
from common import *

KS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 24]
limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
wf = load_waveform(); q_norm = np.asarray(wf['q_norm'])


def z_of(lg, mu, sd):
    return (lg - mu[:, None]) / sd[:, None]


def fit_basis(scope, train_cases):
    mu, sd, _ = load_frame_stats(scope)
    C = np.zeros((N_FRAMES, N_FRAMES)); n = 0; b0_sum = 0.0; b0_sq = 0.0
    for cid in train_cases:
        d = load_bundle(cid)
        z = z_of(ln_pa(d['tau']), mu, sd)
        b0 = z.mean(axis=0); zc = z - b0
        C += zc @ zc.T; n += z.shape[1]; b0_sum += b0.sum(); b0_sq += (b0 ** 2).sum()
    C /= n
    w, V = np.linalg.eigh(C); order = np.argsort(w)[::-1]; w, V = w[order], V[:, order]
    for k in range(V.shape[1]):                      # 符号约定：峰值帧分量为正
        if V[PEAK_INDEX, k] < 0: V[:, k] *= -1
    energy = np.cumsum(w) / w.sum()
    np.savez(OUT / f'time_basis_{scope}.npz', mu=mu, sigma=sd, phi=V, eigvals=w, energy=energy,
             n_points=n, b0_mean=b0_sum / n, b0_std=np.sqrt(max(b0_sq / n - (b0_sum / n) ** 2, 0)),
             cases=np.array(train_cases), steps=np.arange(1120, 1281, 2), floor=FLOOR, eps=EPS, weights='uniform')
    return mu, sd, V, energy


def ceiling(mu, sd, V, held):
    mets = {K: TimeMetrics(q_norm) for K in KS}
    for cid in held:
        d = load_bundle(cid)
        lg = ln_pa(d['tau']); z = z_of(lg, mu, sd); b0 = z.mean(axis=0); zc = z - b0
        coef = V.T @ zc                                # (81, N)
        for K in KS:
            zr = b0 + V[:, :K] @ coef[:K]
            mets[K].add(lg, mu[:, None] + sd[:, None] * zr)
    return {K: m.summary() for K, m in mets.items()}


res = {}
for f in range(3):
    tr, held = FOLD_TRAIN[f][:limit], FOLD_HELD[f][:limit]
    mu, sd, V, energy = fit_basis(f'fold{f}', tr)
    print(f'fold{f}: 训练 {len(tr)} 例；累计能量 K=1/2/3/5/8/12 = ' + '/'.join(f'{energy[k-1]:.3f}' for k in (1, 2, 3, 5, 8, 12)), flush=True)
    res[f'fold{f}'] = dict(energy=[float(v) for v in energy], ceiling=ceiling(mu, sd, V, held))
    for K in KS:
        print(f'   K={K:2d}  ' + fmt_summary(res[f'fold{f}']['ceiling'][K]), flush=True)
mu, sd, V, energy = fit_basis('train136', TRAIN[:limit])
res['train136'] = dict(energy=[float(v) for v in energy], ceiling_test34=ceiling(mu, sd, V, TEST[:limit]))
print('train136 基：累计能量 K=1/2/3/5/8/12 = ' + '/'.join(f'{energy[k-1]:.3f}' for k in (1, 2, 3, 5, 8, 12)))
for K in KS:
    print(f'   test34 K={K:2d}  ' + fmt_summary(res['train136']['ceiling_test34'][K]))

print('\n=== 三折均值上限 ===')
keys = ('cycle_r2cb_pa', 'cycle_r2cb_ln', 'peak_r2cb_pa', 'trough_r2cb_pa', 'tawss_r2cb', 'peak_time_err_med_frames', 'peak_time_err_p90_frames', 'ts_corr_ln', 'amp_err_ln_med')
mean = {K: {k: float(np.mean([res[f'fold{f}']['ceiling'][K][k] for f in range(3)])) for k in keys} for K in KS}
print('  K   cycle_Pa  cycle_ln  peak_Pa  trough_Pa  TAWSS   pk-t med/p90   ts-corr  amp-err')
for K in KS:
    m = mean[K]
    print(f'  {K:2d}  {m["cycle_r2cb_pa"]:.4f}   {m["cycle_r2cb_ln"]:.4f}   {m["peak_r2cb_pa"]:.4f}  {m["trough_r2cb_pa"]:.4f}    {m["tawss_r2cb"]:.4f}  {m["peak_time_err_med_frames"]:.1f}/{m["peak_time_err_p90_frames"]:.1f}       {m["ts_corr_ln"]:.3f}   {m["amp_err_ln_med"]:.3f}')
k_star = next((K for K in KS if mean[K]['trough_r2cb_pa'] >= 0.95 and mean[K]['tawss_r2cb'] >= 0.98), None)
print(f'\nK* (预注册规则：谷底四分位 R²cb ≥ 0.95 且 TAWSS ≥ 0.98) = {k_star}')
res['fold_mean'] = {str(K): mean[K] for K in KS}; res['k_star'] = k_star; res['KS'] = KS
write_json(OUT / 'd1_ceiling.json', res)
print('wrote d1_ceiling.json')
