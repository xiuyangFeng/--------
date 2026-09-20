"""cv3_v51 out-of-fold read of X5D_v51: per-fold held-out Pa/normalised R2_cb, pooled over the 136 train cases (each once)."""
import json, sys, numpy as np
from pathlib import Path
sys.path.insert(0, '/public/newhome/cy/Digital_twin/GNN')
from training_wss_min import evaluate as E
M = E.M
ROOT = Path('/public/newhome/cy/Digital_twin/GNN'); RUNS = ROOT / 'training_wss_min/runs/wss_v51_wave2a_20260916'; P = 'eval/ckpt_best/predictions/test'
def cb(t, p): return M.casebalanced_field_metrics(t, p)['r2']
pooled_t, pooled_p, pooled_tn, pooled_pn, coh = [], [], [], [], []
out = {}
for k in range(3):
    run = RUNS / f'X5D_v51_f{k}_s1234'; units = sorted(str(p.parent.relative_to(run / P)) for p in (run / P).glob('*/*/*/predictions.npz'))
    t, p, tn, pn = [], [], [], []
    for u in units:
        with np.load(run / P / u / 'predictions.npz') as z:
            t.append(z['true_pa'].astype(float)); p.append(z['pred_pa'].astype(float)); tn.append(z['true_norm'].astype(float)); pn.append(z['pred_norm'].astype(float))
    pooled_t += t; pooled_p += p; pooled_tn += tn; pooled_pn += pn; coh += [u.split('/')[0] for u in units]
    out[f'fold{k}'] = dict(n=len(units), pa_r2_cb=cb(t, p), norm_r2_cb=cb(tn, pn)); print(f"fold{k}: 留出 {len(units)} 例 Pa R2_cb {out[f'fold{k}']['pa_r2_cb']:.4f} 归一化 {out[f'fold{k}']['norm_r2_cb']:.4f}")
pc = np.array([1 - ((a - b) ** 2).sum() / ((a - a.mean()) ** 2).sum() for a, b in zip(pooled_t, pooled_p)])
out['pooled'] = dict(n=len(pooled_t), pa_r2_cb=cb(pooled_t, pooled_p), norm_r2_cb=cb(pooled_tn, pooled_pn), case_r2_mean=float(pc.mean()), case_r2_p10=float(np.quantile(pc, .1)),
                     per_cohort={c: cb([x for x, cc in zip(pooled_t, coh) if cc == c], [x for x, cc in zip(pooled_p, coh) if cc == c]) for c in ('AG', 'AAA', 'ILO')})
print(f"合并 {len(pooled_t)} 例折外: Pa R2_cb {out['pooled']['pa_r2_cb']:.4f} 归一化 {out['pooled']['norm_r2_cb']:.4f} | 逐例均值/p10 {out['pooled']['case_r2_mean']:.3f}/{out['pooled']['case_r2_p10']:.3f} | AG/AAA/ILO " + '/'.join(f'{v:.3f}' for v in out['pooled']['per_cohort'].values()))
Path(__file__).with_suffix('.json').write_text(json.dumps(out, indent=1))
