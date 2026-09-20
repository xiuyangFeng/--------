"""Inference-protocol comparison for the deployment ensemble (no training): how to combine X5's five seeds.

Protocols (all deployment-legal, no parameter fitted on test34):
  log_mean        mean of the seeds in log_z, then exp                      (current deployment form)
  pa_mean         mean of the seeds in Pa
  log_median      median of the seeds in log_z
  log_mean_jensen log-mean + 0.5 * inter-seed variance (per point, log space): lognormal-mean style correction
  pa_max/…        not used (would bias)
Also: X5X11 three seeds, and the 8-model union.
    python training_wss_min/experiments/wss_local_wave5_20260913/offline/ensemble_protocol.py
"""
import json, sys
import numpy as np
from pathlib import Path
sys.path.insert(0, '/public/newhome/cy/Digital_twin/GNN')
from training_wss_min import evaluate as E
M = E.M
ROOT = Path('/public/newhome/cy/Digital_twin/GNN'); RUNS = ROOT / 'training_wss_min/runs'
X5 = {'1234': RUNS / 'wss_local_wave1_20260912/X5_s1234', '7': RUNS / 'wss_local_wave1b_20260912/X5_s7', '2025': RUNS / 'wss_local_wave1b_20260912/X5_s2025',
      '11': RUNS / 'wss_local_wave4_20260913/X5_s11', '2026': RUNS / 'wss_local_wave4_20260913/X5_s2026'}
X5X11 = {s: RUNS / f'wss_local_wave2_20260912/X5X11_s{s}' for s in ('1234', '7', '2025')}
PRED = 'eval/ckpt_best/predictions/test'
OUT = Path(__file__).resolve().parent
units = sorted(str(p.parent.relative_to(X5['1234'] / PRED)) for p in (X5['1234'] / PRED).glob('*/*/*/predictions.npz')); assert len(units) == 34

def load(run):
    out = {}
    for u in units:
        with np.load(run / PRED / u / 'predictions.npz') as z:
            out[u] = {k: z[k].astype(np.float64) for k in ('pred_norm', 'true_norm', 'pred_pa', 'true_pa')}
    return out

d0 = load(X5['1234']); A = np.polyfit(d0[units[0]]['pred_norm'], np.log(d0[units[0]]['pred_pa']), 1); std, mean = A
to_pa = lambda n: np.exp(n * std + mean)
truth_pa = [d0[u]['true_pa'] for u in units]; truth_n = [d0[u]['true_norm'] for u in units]

def cb(t, p): return float(M.casebalanced_field_metrics(t, p)['r2'])
def tails(t, p):
    tt, pp = np.concatenate(t), np.concatenate(p); q = np.quantile(tt, 0.9); m = tt >= q
    ious = [float(((a >= np.percentile(a, 90)) & (b >= np.percentile(b, 90))).sum() / ((a >= np.percentile(a, 90)) | (b >= np.percentile(b, 90))).sum()) for a, b in zip(t, p)]
    pc = [1 - ((a - b) ** 2).sum() / ((a - a.mean()) ** 2).sum() for a, b in zip(t, p)]
    return dict(high_r2=float(1 - ((tt[m] - pp[m]) ** 2).sum() / ((tt[m] - tt[m].mean()) ** 2).sum()), top10_ratio=float(pp[m].mean() / tt[m].mean()),
                p99_ratio=float(np.quantile(pp, 0.99) / np.quantile(tt, 0.99)), iou=float(np.mean(ious)), mae=float(np.mean(np.abs(tt - pp))),
                case_mean=float(np.mean(pc)), case_p10=float(np.quantile(pc, 0.1)))

def protocols(pool):
    seeds = list(pool); rows = {}
    N = [np.stack([pool[s][u]['pred_norm'] for s in seeds]) for u in units]   # (S, n)
    P = [np.stack([pool[s][u]['pred_pa'] for s in seeds]) for u in units]
    rows['log_mean'] = [to_pa(x.mean(0)) for x in N]
    rows['pa_mean'] = [x.mean(0) for x in P]
    rows['log_median'] = [to_pa(np.median(x, 0)) for x in N]
    rows['log_mean_jensen_seedvar'] = [to_pa(x.mean(0) + 0.5 * x.var(0, ddof=1) * std) for x in N]  # var in log_z units -> ln units needs *std: exp(mu + 0.5*sigma_ln^2), sigma_ln^2 = var_z*std^2, in z units add 0.5*var_z*std
    rows['log_mean_plus_seedsd'] = [to_pa(x.mean(0) + x.std(0, ddof=1)) for x in N]
    rows['log_max'] = [to_pa(x.max(0)) for x in N]
    return rows, N

def report(name, pool):
    rows, N = protocols(pool)
    out = {}
    for s in pool:
        pa = [pool[s][u]['pred_pa'] for u in units]; out[f'single_s{s}'] = dict(pa_r2_cb=cb(truth_pa, pa), norm_r2_cb=cb(truth_n, [pool[s][u]['pred_norm'] for u in units]), **tails(truth_pa, pa))
    for k, pa in rows.items():
        out[k] = dict(pa_r2_cb=cb(truth_pa, pa), norm_r2_cb=cb(truth_n, [np.log(np.clip(x, 1e-9, None)) / std - mean / std for x in pa]), **tails(truth_pa, pa))
    spread = np.concatenate([x.std(0, ddof=1) for x in N]) * std
    out['_seed_spread_ln_units'] = dict(median=float(np.median(spread)), p90=float(np.quantile(spread, 0.9)), mean_sq=float(np.mean(spread ** 2)))
    print(f'=== {name} ({len(pool)} models)')
    for k, v in out.items():
        if k.startswith('_'): print('  seed spread (ln units):', v); continue
        print(f"  {k:26s} Pa R²_cb {v['pa_r2_cb']:.4f} | norm {v['norm_r2_cb']:.4f} | case mean {v['case_mean']:.3f} P10 {v['case_p10']:.3f} | MAE {v['mae']:.3f} | high-WSS R² {v['high_r2']:.3f} | top10 {v['top10_ratio']:.3f} | p99 {v['p99_ratio']:.3f} | IoU {v['iou']:.3f}")
    return out

x5 = {s: load(r) for s, r in X5.items()}
x5x11 = {s: load(r) for s, r in X5X11.items() if (r / PRED).is_dir()}
res = {'X5_5seed': report('X5 five seeds', x5)}
if x5x11:
    res['X5X11_3seed'] = report('X5+X11 three seeds', x5x11)
    union = {**{f'x5_{s}': v for s, v in x5.items()}, **{f'x5x11_{s}': v for s, v in x5x11.items()}}
    res['union_8'] = report('X5 ∪ X5X11 (8 models)', union)
(OUT / 'ensemble_protocol.json').write_text(json.dumps(res, indent=1))
print('wrote', OUT / 'ensemble_protocol.json')
