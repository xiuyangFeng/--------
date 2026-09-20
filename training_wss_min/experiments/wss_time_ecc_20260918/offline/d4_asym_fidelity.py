"""D4：输出端不对称保真度（峰值帧，折外）。逐 (分支 × 4 mm) 区间：
  ln 真值与 ln 预测各自拟合 m=1（与 m=2）周向模态 → 幅值比 A_pred/A_true、相位差、m=1 方差份额（真值 vs 预测）；
  e_τ = ‖Σ τ_i u_i‖ / Σ τ_i（u_i = (cosθ, sinθ)，WSS 加权方向偏心）→ 比 e_τ^pred / e_τ^true。
按病例中位 e_geom 三档分层。"""
import numpy as np
from common import *
import os
_L = int(os.environ['TECC_LIMIT']) if os.environ.get('TECC_LIMIT') else None
TRAIN, TEST = TRAIN[:_L], TEST[:_L]


def m_fit(theta, y, m):
    A = np.column_stack([np.ones(len(theta)), np.cos(m * theta), np.sin(m * theta)])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    amp = float(np.hypot(coef[1], coef[2])); pha = float(np.arctan2(coef[2], coef[1]))
    v = y.var(); share = float(1 - (y - A @ coef).var() / v) if v > 0 else np.nan
    return amp, pha, share


def e_tau(theta, tau):
    u = np.column_stack([np.cos(theta), np.sin(theta)])
    return float(np.linalg.norm((tau[:, None] * u).sum(0)) / max(tau.sum(), 1e-12))


rows = []
for partition, cases in (('train', TRAIN), ('test', TEST)):
    for cid in cases:
        d = load_bundle(cid)
        lt = ln_pa(d['tau'][PEAK_INDEX]); lp = load_oof_peak_ln(cid, partition, d['ids'])
        tt, tp = np.exp(lt) - EPS, np.exp(lp) - EPS
        v, m, _, _ = load_ecc(cid); ecc_med = float(np.median(v[m]))
        inv, counts = bins_of(d)
        ar1, ar2, dph, sh_t, sh_p, et, w = [], [], [], [], [], [], []
        for b in np.flatnonzero(counts >= MIN_PTS):
            s = inv == b; th = d['theta'][s]
            a_t, p_t, s_t = m_fit(th, lt[s], 1); a_p, p_p, s_p = m_fit(th, lp[s], 1)
            a2_t, _, _ = m_fit(th, lt[s], 2); a2_p, _, _ = m_fit(th, lp[s], 2)
            if a_t < 1e-3:
                continue
            ar1.append(a_p / a_t); ar2.append(a2_p / max(a2_t, 1e-6)); dph.append(abs(float(circ_diff(p_p, p_t))))
            sh_t.append(s_t); sh_p.append(s_p); et.append(e_tau(th, tp[s]) / max(e_tau(th, tt[s]), 1e-6)); w.append(int(counts[b]))
        w = np.array(w, float); ar1 = np.array(ar1); dph = np.array(dph)
        rows.append(dict(cid=cid, partition=partition, cohort=cid.split('/')[0], ecc_med=ecc_med, n_bins=int(len(w)),
                         m1_amp_ratio_wmed=float(np.exp(np.average(np.log(ar1), weights=w))), m1_amp_ratio_med=float(np.median(ar1)),
                         m1_amp_ratio_frac_lt1=float(np.average(ar1 < 1, weights=w)),
                         m2_amp_ratio_wmed=float(np.exp(np.average(np.log(np.array(ar2)), weights=w))),
                         m1_phase_err_med_deg=float(np.degrees(np.median(dph))), m1_phase_err_frac_lt45=float(np.average(dph < np.pi / 4, weights=w)),
                         m1_share_true=float(np.average(sh_t, weights=w)), m1_share_pred=float(np.average(sh_p, weights=w)),
                         e_tau_ratio_wmed=float(np.exp(np.average(np.log(np.maximum(et, 1e-6)), weights=w)))))
res = {}
for partition in ('train', 'test'):
    sub = [x for x in rows if x['partition'] == partition]
    tag = 'train136 折外' if partition == 'train' else 'test34 五 seed 集成'
    print(f'\n=== D4 不对称保真度 · {tag}（{len(sub)} 例，逐例中位）===')
    out = {}
    for k, lab in (('m1_amp_ratio_wmed', 'm=1 幅值比 pred/true（区间加权几何均值）'), ('m1_amp_ratio_frac_lt1', 'm=1 幅值比 < 1 的区间比例'),
                   ('m2_amp_ratio_wmed', 'm=2 幅值比'), ('m1_phase_err_med_deg', 'm=1 相位误差中位（°）'), ('m1_phase_err_frac_lt45', '相位误差 < 45° 比例'),
                   ('m1_share_true', 'm=1 方差份额 · 真值'), ('m1_share_pred', 'm=1 方差份额 · 预测'), ('e_tau_ratio_wmed', 'e_τ 比 pred/true')):
        vals = np.array([x[k] for x in sub]); out[k] = dict(median=float(np.median(vals)), p10=float(np.quantile(vals, .1)), p90=float(np.quantile(vals, .9)))
        print(f'  {lab:40s} {np.median(vals):7.3f}   [p10 {np.quantile(vals,.1):6.3f}, p90 {np.quantile(vals,.9):6.3f}]')
    x = np.array([x['ecc_med'] for x in sub]); order = np.argsort(x); k = len(order) // 3
    ter = {}
    for name, idx in (('low', order[:k]), ('mid', order[k:2 * k]), ('high', order[2 * k:])):
        ter[name] = {kk: float(np.median([sub[i][kk] for i in idx])) for kk in ('m1_amp_ratio_wmed', 'm1_phase_err_med_deg', 'e_tau_ratio_wmed', 'm1_share_true')}
    out['by_ecc_tercile'] = ter
    print('  按病例中位 e_geom 三档（低/中/高）：m=1 幅值比 ' + ' / '.join(f'{ter[n]["m1_amp_ratio_wmed"]:.3f}' for n in ('low', 'mid', 'high'))
          + '；相位误差 ' + ' / '.join(f'{ter[n]["m1_phase_err_med_deg"]:.0f}°' for n in ('low', 'mid', 'high'))
          + '；真值 m=1 份额 ' + ' / '.join(f'{ter[n]["m1_share_true"]:.3f}' for n in ('low', 'mid', 'high')))
    res[partition] = out
write_json(OUT / 'd4_asym_fidelity.json', dict(summary=res, rows=rows))
print('wrote d4_asym_fidelity.json')
