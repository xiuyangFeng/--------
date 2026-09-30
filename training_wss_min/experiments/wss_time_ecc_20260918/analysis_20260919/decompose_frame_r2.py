"""逐帧 R² 的相关/幅值/偏置分解（只读已有 metrics.json）。
R²(NSE) = 2·α·r − α² − β_n²,  r = Pearson 相关, α = σ_pred/σ_true, β_n = (μ_pred − μ_true)/σ_true  (Gupta et al. 2009)
存的字段：r2_linear_fit = r², linear_fit_slope = r·α（pred = a·true + b 的 a）→ α = a / r, β_n² = 2αr − α² − R²。
KGE = 1 − sqrt((r−1)² + (α−1)² + (β−1)²)，β = μ_pred/μ_true 这里无法从存量字段还原，改报 β_n²。
"""
import json, csv, sys, math
import numpy as np
ROOT = '/public/newhome/cy/Digital_twin/GNN/training_wss_min'
SRC = f'{ROOT}/experiments/wss_time_ecc_20260918/analysis_20260919/frame_r2_source.csv'
ARMS = ['T0', 'TB8', 'TB16']; FOLDS = [0, 1, 2]; CKPT = sys.argv[1] if len(sys.argv) > 1 else 'best'
phase = {}
with open(SRC) as f:
    for row in csv.DictReader(f):
        s = int(row['step']); phase[s] = ('trough' if row['in_trough'] == '1' else 'peak' if row['in_peak'] == '1'
                                         else 'accel' if row['in_systole'] == '1' else 'decel')
        phase.setdefault('_q', {})[s] = float(row['q_norm'])
rows = []
for arm in ARMS:
    per_fold = []
    for fo in FOLDS:
        m = json.load(open(f'{ROOT}/runs/wss_time_ecc_20260918/{arm}_f{fo}_s1234/eval/ckpt_{CKPT}/metrics.json'))['test']
        fr = m['frames']
        for s_str, e in fr.items():
            s = int(s_str); ph = e['physical']; cb = ph['field_casebalanced']; fd = ph['field']
            r2 = cb['r2']; r = math.sqrt(max(cb['r2_linear_fit'], 0.0)); a = cb['linear_fit_slope']
            alpha = a / r if r > 1e-9 else float('nan')
            beta2 = 2 * alpha * r - alpha ** 2 - r2 if np.isfinite(alpha) else float('nan')
            ccc = 2 * r * alpha / (1 + alpha ** 2 + max(beta2, 0.0)) if np.isfinite(alpha) else float('nan')
            rows.append(dict(arm=arm, fold=fo, step=s, phase=phase[s], q_norm=phase['_q'][s],
                             r2cb=r2, r=r, r2_fit=r ** 2, alpha=alpha, beta_n2=beta2, ccc=ccc, slope=a,
                             spearman=ph['hotspot']['spearman_all_casemean'],
                             r2_casemean=ph['aggregate']['r2_casemean'],
                             nmae_range=fd['nmae_range'], nrmse_mean=fd['nrmse_mean'],
                             nmae_casemean=ph['aggregate']['nmae_casemean'],
                             r2cb_norm=e['normalized']['field_casebalanced']['r2'],
                             top10_ratio=ph['calibration']['top10_pred_true_ratio']))
out = f'{ROOT}/experiments/wss_time_ecc_20260918/analysis_20260919/frame_r2_decomposition_{CKPT}.csv'
with open(out, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print('wrote', out, len(rows), 'rows')
# 三折均值 → 相位汇总
keys = ['r2cb', 'r', 'r2_fit', 'alpha', 'beta_n2', 'ccc', 'spearman', 'r2_casemean', 'nmae_range', 'nrmse_mean', 'nmae_casemean', 'r2cb_norm', 'top10_ratio']
def agg(sel):
    return {k: float(np.nanmean([x[k] for x in sel])) for k in keys}
print(f'\n== ckpt={CKPT}  三折均值 × 相位（accel/peak/decel/trough/全周期）==')
hdr = f"{'arm':5s} {'phase':7s} {'n':>3s} " + ' '.join(f'{k:>10s}' for k in keys); print(hdr)
for arm in ARMS:
    for ph_ in ['accel', 'peak', 'decel', 'trough', 'cycle']:
        sel = [x for x in rows if x['arm'] == arm and (ph_ == 'cycle' or x['phase'] == ph_)]
        a = agg(sel); n = len({x['step'] for x in sel})
        print(f'{arm:5s} {ph_:7s} {n:3d} ' + ' '.join(f'{a[k]:10.3f}' for k in keys))
# 逐帧（每 5 帧）T0
print(f'\n== T0 {CKPT} 三折均值，每 5 帧 ==')
steps = sorted({x['step'] for x in rows})[::5]
for k in ['q_norm', 'r2cb', 'r', 'alpha', 'beta_n2', 'ccc', 'spearman', 'nmae_range', 'nrmse_mean', 'r2cb_norm']:
    vals = [float(np.nanmean([x[k] for x in rows if x['arm'] == 'T0' and x['step'] == s])) for s in steps]
    print(f'{k:10s} ' + ' '.join(f'{v:6.3f}' for v in vals))
print('steps      ' + ' '.join(f'{s:6d}' for s in steps))
