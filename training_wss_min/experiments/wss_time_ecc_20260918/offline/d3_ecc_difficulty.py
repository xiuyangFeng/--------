"""D3：截面偏心作为**病例难度标识**——n=136 折外复核（+ test34 五 seed 集成对照）。
逐例：折外峰值帧物理 R²、MAE；e_geom 中位/p95/>0.2 比例；混杂量：中位半径、最大/中位半径、主干长（max abscissa）、峰值帧 ln WSS 空间 sd。
门：折外 |Spearman| ≥ 0.3 且三队列同号。"""
import numpy as np
from scipy.stats import spearmanr
from common import *
import os
_L = int(os.environ['TECC_LIMIT']) if os.environ.get('TECC_LIMIT') else None
TRAIN, TEST = TRAIN[:_L], TEST[:_L]


def rank(x):
    return np.argsort(np.argsort(x)).astype(float)


def partial_spearman(x, y, Z):
    X = np.column_stack([rank(z) for z in Z] + [np.ones(len(x))])
    rx = rank(x) - X @ np.linalg.lstsq(X, rank(x), rcond=None)[0]
    ry = rank(y) - X @ np.linalg.lstsq(X, rank(y), rcond=None)[0]
    return spearmanr(rx, ry)


rows = []
for partition, cases in (('train', TRAIN), ('test', TEST)):
    for cid in cases:
        d = load_bundle(cid)
        true = d['tau'][PEAK_INDEX].astype(np.float64); lp = load_oof_peak_ln(cid, partition, d['ids'])
        pred = np.exp(lp) - EPS
        v, m, r, rm = load_ecc(cid)
        rows.append(dict(cid=cid, partition=partition, cohort=cid.split('/')[0], fold=FOLD_OF.get(cid, -1),
                         r2=r2_score(true, pred), mae=float(np.mean(np.abs(true - pred))), r2_ln=r2_score(ln_pa(true), lp),
                         ecc_med=float(np.median(v[m])), ecc_p95=float(np.quantile(v[m], .95)), ecc_frac02=float(np.mean(v[m] > 0.2)),
                         ecc_valid=float(m.mean()), round_med=float(np.median(r[rm])) if rm.any() else np.nan,
                         rad_med=float(np.median(d['rad'])), rad_maxmed=float(d['rad'].max() / np.median(d['rad'])),
                         length=float(d['s_abs'].max()), ln_sd=float(ln_pa(true).std()), n=int(len(true))))
res = {}
for partition in ('train', 'test'):
    sub = [x for x in rows if x['partition'] == partition]
    r2 = np.array([x['r2'] for x in sub]); coh = np.array([x['cohort'] for x in sub])
    tag = 'train136 折外（cascade，seed 1234）' if partition == 'train' else 'test34 五 seed 集成'
    print(f'\n=== D3 · {tag}（n={len(sub)}）逐例 R² 均值 {r2.mean():.3f} ===')
    out = {}
    for key, lab in (('ecc_med', '中位偏心'), ('ecc_p95', 'p95 偏心'), ('ecc_frac02', '偏心>0.2 比例'), ('round_med', '中位圆度'),
                     ('rad_med', '中位半径'), ('rad_maxmed', '最大/中位半径'), ('length', '主干长'), ('ln_sd', 'ln WSS 空间 sd')):
        x = np.array([x[key] for x in sub], float); ok = np.isfinite(x)
        sr = spearmanr(x[ok], r2[ok]); out[key] = dict(spearman=float(sr.statistic), p=float(sr.pvalue))
        print(f'  {lab:14s} Spearman {sr.statistic:+.3f} (p={sr.pvalue:.3f})')
    x = np.array([x['ecc_med'] for x in sub])
    xc, yc = x.copy(), r2.copy()
    for c in set(coh):
        s = coh == c; xc[s] -= x[s].mean(); yc[s] -= r2[s].mean()
    sr = spearmanr(xc, yc); out['ecc_med_cohort_demeaned'] = dict(spearman=float(sr.statistic), p=float(sr.pvalue))
    print(f'  中位偏心 · 队列去均值      Spearman {sr.statistic:+.3f} (p={sr.pvalue:.3f})')
    Z = [np.array([x[k] for x in sub], float) for k in ('rad_med', 'rad_maxmed', 'length', 'ln_sd')]
    pr = partial_spearman(x, r2, Z); out['ecc_med_partial'] = dict(spearman=float(pr.statistic), p=float(pr.pvalue))
    print(f'  中位偏心 · 偏相关（控制半径/膨大比/长度/ln sd） {pr.statistic:+.3f} (p={pr.pvalue:.3f})')
    per = {}
    for c in ('AAA', 'AG', 'ILO'):
        s = coh == c; sr = spearmanr(x[s], r2[s]); per[c] = dict(n=int(s.sum()), spearman=float(sr.statistic), p=float(sr.pvalue), r2_mean=float(r2[s].mean()))
        print(f'  {c:4s} n={s.sum():3d} 组内 Spearman {sr.statistic:+.3f} (p={sr.pvalue:.3f})  R² {r2[s].mean():.3f}')
    out['by_cohort'] = per
    order = np.argsort(x); k = len(order) // 3
    mae = np.array([x['mae'] for x in sub])
    ter = [(float(r2[idx].mean()), float(mae[idx].mean())) for idx in (order[:k], order[k:2 * k], order[2 * k:])]
    out['terciles_r2_mae'] = ter
    print(f'  按中位偏心三档：R² 低 {ter[0][0]:.3f} / 中 {ter[1][0]:.3f} / 高 {ter[2][0]:.3f}；MAE {ter[0][1]:.3f} / {ter[1][1]:.3f} / {ter[2][1]:.3f} Pa')
    if partition == 'train':
        signs = [float(np.sign(per[c]['spearman'])) for c in per]
        gate = abs(out['ecc_med']['spearman']) >= 0.3 and len(set(signs)) == 1
        out['gate'] = dict(passed=bool(gate), rule='|Spearman| ≥ 0.3 且三队列同号（折外 n=136）')
        print(f'  ⇒ D3 门：{"通过" if gate else "不通过"}（|ρ|={abs(out["ecc_med"]["spearman"]):.3f}，队列符号 {signs}）')
        fl = np.array([x['fold'] for x in sub]); out['by_fold'] = {}
        for f in range(3):
            s = fl == f; sr = spearmanr(x[s], r2[s]); out['by_fold'][f] = float(sr.statistic)
            print(f'     fold{f} n={s.sum()} Spearman {sr.statistic:+.3f}')
    res[partition] = out
write_json(OUT / 'd3_ecc_difficulty.json', dict(summary=res, rows=rows))
print('wrote d3_ecc_difficulty.json')
