"""D5：物理级交互门（只用 CFD 真值，不用模型；170 例）。
问：截面几何偏心（幅值 A1_geom = 截面 r(θ) 的 m=1 形变，方向 φ1_geom；以及 sidecar e_geom）能否解释
    周期各相位（加速/峰值/减速/谷底）里壁面 WSS 的周向不对称（ln τ 的 m=1 幅值 A1_τ 与相位 φ1_τ）与 OSI。
门：减速或谷底相位 R²(A1_τ ~ A1_geom, 病例去均值) ≥ 0.05 且方向对齐（|cos Δφ| 的加权均值）在折间稳定 → 开 I1；< 0.02 → 不开。"""
import numpy as np
from scipy.stats import spearmanr
from common import *
import os
_L = int(os.environ['TECC_LIMIT']) if os.environ.get('TECC_LIMIT') else None
TRAIN, TEST = TRAIN[:_L], TEST[:_L]

wf = load_waveform(); q = np.asarray(wf['q_norm']); dq = np.asarray(wf['dq_norm'])
peak_f = np.flatnonzero(q >= np.quantile(q, .8)); trough_f = np.sort(np.argsort(q)[:20])
rest = np.setdiff1d(np.arange(N_FRAMES), np.union1d(peak_f, trough_f))
groups = dict(accel=rest[dq[rest] > 0], peak=peak_f, decel=rest[dq[rest] < 0], trough=trough_f)
print('相位分组帧数：', {k: int(len(v)) for k, v in groups.items()})

recs = []
for i, cid in enumerate(TRAIN + TEST):
    d = load_bundle(cid, vec=True)
    lg = ln_pa(d['tau']); v, m, r, rm = load_ecc(cid)
    vec = d['vec'].astype(np.float64); tau = d['tau'].astype(np.float64)
    osi = 0.5 * (1 - np.linalg.norm(vec.mean(0), axis=1) / np.maximum(tau.mean(0), 1e-12))
    inv, counts = bins_of(d)
    r_mm = d['rho'] * d['rad']
    for b in np.flatnonzero(counts >= MIN_PTS):
        s = inv == b; th = d['theta'][s]
        a_g, p_g, _ = m1_fit(th, r_mm[s]); c0 = float(np.mean(r_mm[s]))
        if c0 <= 1e-6:
            continue
        a_t, p_t, sh = m1_fit(th, lg[:, s])                     # (81,) each
        recs.append(dict(cid=cid, cohort=cid.split('/')[0], fold=FOLD_OF.get(cid, 3), n=int(counts[b]),
                         A1_geom=a_g / c0, phi_geom=p_g, e_geom=float(np.mean(v[s][m[s]])) if m[s].any() else np.nan,
                         round_=float(np.mean(r[s][rm[s]])) if rm[s].any() else np.nan, R=c0,
                         A1_tau=a_t, phi_tau=p_t, share=sh, osi=float(np.mean(osi[s])), tawss=float(np.mean(tau[:, s]))))
    if (i + 1) % 20 == 0:
        print(f'  {i+1}/170', flush=True)
print(f'区间数 {len(recs)}（{len(set(x["cid"] for x in recs))} 例）')

cid_arr = np.array([x['cid'] for x in recs]); fold_arr = np.array([x['fold'] for x in recs])
A1g = np.array([x['A1_geom'] for x in recs]); eg = np.array([x['e_geom'] for x in recs]); phig = np.array([x['phi_geom'] for x in recs])
A1t = np.array([x['A1_tau'] for x in recs]); phit = np.array([x['phi_tau'] for x in recs]); share = np.array([x['share'] for x in recs])
osi = np.array([x['osi'] for x in recs]); w = np.array([x['n'] for x in recs], float)


def demean_by_case(x):
    out = x.copy()
    for c in np.unique(cid_arr):
        s = cid_arr == c; out[s] = x[s] - np.nanmean(x[s])
    return out


def r2_lin(x, y, wts):
    ok = np.isfinite(x) & np.isfinite(y)
    X = np.column_stack([x[ok], np.ones(ok.sum())]); W = np.sqrt(wts[ok])
    coef, *_ = np.linalg.lstsq(X * W[:, None], y[ok] * W, rcond=None)
    res = y[ok] - X @ coef
    return float(1 - np.average(res ** 2, weights=wts[ok]) / np.average((y[ok] - np.average(y[ok], weights=wts[ok])) ** 2, weights=wts[ok])), float(coef[0])


okg = np.isfinite(eg)
print(f'\n区间几何：A1_geom 中位 {np.median(A1g):.4f} p90 {np.quantile(A1g,.9):.4f}；sidecar e_geom 中位 {np.nanmedian(eg):.4f}；'
      f'A1_geom vs e_geom Spearman {spearmanr(A1g[okg], eg[okg]).statistic:+.3f}')
def _wavg(x, wt):
    ok = np.isfinite(x); return float(np.average(x[ok], weights=wt[ok]))


print('WSS m=1 份额（真值，区间均值）按相位：' + '，'.join(f'{k} {_wavg(np.nanmean(share[:, f], axis=1), w):.3f}' for k, f in groups.items()))
out = dict(groups={k: [int(i) for i in f] for k, f in groups.items()}, n_bins=len(recs))
print('\n相位组     R²(A1τ~A1geom)  斜率    R²(A1τ~e_geom)  Spearman(A1τ,A1geom)  对齐|cosΔφ|  R(2Δφ)  |Δφ|<45°/>135°  分折|cosΔφ| f0/f1/f2/test')
for k, f in groups.items():
    at = A1t[:, f].mean(1)
    ph = np.arctan2(np.sin(phit[:, f]).mean(1), np.cos(phit[:, f]).mean(1))
    dphi = circ_diff(ph, phig)
    x1, y1 = demean_by_case(A1g), demean_by_case(at)
    r2a, slope = r2_lin(x1, y1, w); r2e, _ = r2_lin(demean_by_case(eg), y1, w)
    sp = spearmanr(A1g, at).statistic
    strong = A1g >= 0.02
    ww = w[strong] * A1g[strong]
    cos_al = float(np.average(np.abs(np.cos(dphi[strong])), weights=ww))
    R2 = float(np.hypot(np.average(np.cos(2 * dphi[strong]), weights=ww), np.average(np.sin(2 * dphi[strong]), weights=ww)))
    f45 = float(np.average(np.abs(dphi[strong]) < np.pi / 4, weights=ww)); f135 = float(np.average(np.abs(dphi[strong]) > 3 * np.pi / 4, weights=ww))
    per_fold = []
    for fo in range(4):
        s = strong & (fold_arr == fo)
        per_fold.append(float(np.average(np.abs(np.cos(dphi[s])), weights=w[s] * A1g[s])) if s.any() else float('nan'))
    out[k] = dict(r2_A1geom=r2a, slope=slope, r2_egeom=r2e, spearman=float(sp), align_abscos=cos_al, align_R2=R2, frac_lt45=f45, frac_gt135=f135, per_fold_abscos=per_fold)
    print(f'{k:8s}     {r2a:7.4f}       {slope:+.3f}     {r2e:7.4f}          {sp:+.3f}                 {cos_al:.3f}       {R2:.3f}     {f45:.2f} / {f135:.2f}        ' + '/'.join(f'{v:.2f}' for v in per_fold))
print('（随机方向的 |cosΔφ| 期望 = 0.637，R(2Δφ) 期望 ≈ 0；|Δφ|<45° 与 >135° 各期望 0.25）')
sp_o = spearmanr(demean_by_case(A1g), demean_by_case(osi)).statistic
sp_oe = spearmanr(demean_by_case(eg)[okg], demean_by_case(osi)[okg]).statistic
r2o, _ = r2_lin(demean_by_case(A1g), demean_by_case(osi), w)
print(f'\nOSI（区间均值，病例去均值）vs A1_geom：Spearman {sp_o:+.3f}，R² {r2o:.4f}；vs e_geom Spearman {sp_oe:+.3f}')
out['osi'] = dict(spearman_A1geom=float(sp_o), r2_A1geom=r2o, spearman_egeom=float(sp_oe))
best = max(out['decel']['r2_A1geom'], out['trough']['r2_A1geom'], out['decel']['r2_egeom'], out['trough']['r2_egeom'])
align_vals = [v for v in out['decel']['per_fold_abscos'] + out['trough']['per_fold_abscos'] if np.isfinite(v)]
align_ok = all(v >= 0.70 for v in align_vals)
verdict = 'PASS' if (best >= 0.05 and align_ok) else ('FAIL' if best < 0.02 else 'BORDERLINE')
out['gate'] = dict(best_r2_decel_or_trough=best, align_stable=bool(align_ok), verdict=verdict,
                   rule='减速/谷底 R² ≥ 0.05 且分折 |cosΔφ| ≥ 0.70 → PASS；R² < 0.02 → FAIL')
print(f'\n⇒ D5 门：{verdict}（减速/谷底最好 R² = {best:.4f}，方向对齐分折稳定 = {align_ok}）')
write_json(OUT / 'd5_physics_gate.json', out)
np.savez(OUT / 'd5_bins.npz', cid=cid_arr, fold=fold_arr, n=w, A1_geom=A1g, e_geom=eg, phi_geom=phig, A1_tau=A1t, phi_tau=phit, share=share, osi=osi,
         tawss=np.array([x['tawss'] for x in recs]), R=np.array([x['R'] for x in recs]), roundness=np.array([x['round_'] for x in recs]))
print('wrote d5_physics_gate.json, d5_bins.npz')
