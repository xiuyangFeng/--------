"""v5.1 wave 1 offline analysis: five-seed X5D_v51 ensemble on the new test34, paired deltas of cap / dual / noise vs the
same-seed X5D_v51, per-cohort and per-case wins, three-seed ensembles per family. Reads saved best-checkpoint predictions."""
import json, sys
import numpy as np
from pathlib import Path
sys.path.insert(0, '/public/newhome/cy/Digital_twin/GNN')
from training_wss_min import evaluate as E
M = E.M
ROOT = Path('/public/newhome/cy/Digital_twin/GNN'); RUNS = ROOT / 'training_wss_min/runs/wss_v51_wave1_20260916'; OUT = Path(__file__).resolve().parent
P = 'eval/ckpt_best/predictions/test'
units = sorted(str(p.parent.relative_to(RUNS / 'X5D_v51_s1234' / P)) for p in (RUNS / 'X5D_v51_s1234' / P).glob('*/*/*/predictions.npz')); assert len(units) == 34
def load(arm):
    out = {}
    for u in units:
        with np.load(RUNS / arm / P / u / 'predictions.npz') as z: out[u] = {k: z[k].astype(np.float64) for k in ('pred_norm', 'true_norm', 'pred_pa', 'true_pa')}
    return out
def cb(t, p): return M.casebalanced_field_metrics(t, p)['r2']
def r2(t, p): return 1 - ((t - p) ** 2).sum() / ((t - t.mean()) ** 2).sum()
def tail(t_list, p_list):
    t = np.concatenate(t_list); p = np.concatenate(p_list); q = np.quantile(t, .9); m = t >= q
    return dict(high_r2=float(1 - ((t[m] - p[m]) ** 2).sum() / ((t[m] - t[m].mean()) ** 2).sum()), top10_ratio=float(p[m].mean() / t[m].mean()))
data = {a: load(a) for a in [f'X5D_v51_s{s}' for s in (1234, 7, 2025, 11, 2026)] + [f'{f}_s{s}' for f in ('X5Dcap', 'X5Ddual', 'X5Dnoise') for s in (1234, 7, 2025)]}
truth = [data['X5D_v51_s1234'][u]['true_pa'] for u in units]; truth_n = [data['X5D_v51_s1234'][u]['true_norm'] for u in units]
coh = [u.split('/')[0] for u in units]; res = {}
def ens(arms, space='pa'):
    if space == 'pa': return [np.mean([data[a][u]['pred_pa'] for a in arms], axis=0) for u in units]
    return [np.exp(np.mean([np.log(data[a][u]['pred_pa']) for a in arms], axis=0)) for u in units]
def summarize(name, preds):
    pc = np.array([r2(t, p) for t, p in zip(truth, preds)])
    d = dict(pa_r2_cb=cb(truth, preds), case_r2_mean=float(pc.mean()), case_r2_p10=float(np.quantile(pc, .1)), **tail(truth, preds),
             per_cohort={c: cb([truth[i] for i in range(34) if coh[i] == c], [preds[i] for i in range(34) if coh[i] == c]) for c in ('AG', 'AAA', 'ILO')})
    res[name] = d; print(f'{name:34s} Pa R2_cb {d["pa_r2_cb"]:.4f} | case mean/p10 {d["case_r2_mean"]:.3f}/{d["case_r2_p10"]:.3f} | high-WSS R2 {d["high_r2"]:.3f} top10 {d["top10_ratio"]:.3f} | AG/AAA/ILO ' + '/'.join(f'{v:.3f}' for v in d['per_cohort'].values()))
    return d
print('=== X5D_v51 ensembles on the corrected test34')
base5 = [f'X5D_v51_s{s}' for s in (1234, 7, 2025, 11, 2026)]; base3 = base5[:3]
for a in base5: summarize(a, [data[a][u]['pred_pa'] for u in units])
summarize('X5D_v51 5-seed (Pa mean)', ens(base5, 'pa')); summarize('X5D_v51 5-seed (log mean)', ens(base5, 'log')); summarize('X5D_v51 3-seed (Pa mean)', ens(base3, 'pa'))
print('\n=== paired deltas vs same-seed X5D_v51 (best ckpt)')
for fam in ('X5Dcap', 'X5Ddual', 'X5Dnoise'):
    rows = []
    for s in (1234, 7, 2025):
        a, b = f'{fam}_s{s}', f'X5D_v51_s{s}'
        pa, pb = [data[a][u]['pred_pa'] for u in units], [data[b][u]['pred_pa'] for u in units]
        na, nb = [data[a][u]['pred_norm'] for u in units], [data[b][u]['pred_norm'] for u in units]
        d_pa = cb(truth, pa) - cb(truth, pb); d_n = cb(truth_n, na) - cb(truth_n, nb)
        wins = int(sum(r2(t, x) > r2(t, y) for t, x, y in zip(truth, pa, pb)))
        d_coh = {c: cb([truth[i] for i in range(34) if coh[i] == c], [pa[i] for i in range(34) if coh[i] == c]) - cb([truth[i] for i in range(34) if coh[i] == c], [pb[i] for i in range(34) if coh[i] == c]) for c in ('AG', 'AAA', 'ILO')}
        ta, tb = tail(truth, pa), tail(truth, pb)
        rows.append(dict(seed=s, d_pa=d_pa, d_norm=d_n, wins=wins, d_coh=d_coh, d_high=ta['high_r2'] - tb['high_r2'], d_top10=ta['top10_ratio'] - tb['top10_ratio']))
        print(f'  {a:16s} Δ物理 {d_pa:+.4f} Δ归一化 {d_n:+.4f} 逐例胜 {wins}/34 | AG/AAA/ILO ' + '/'.join(f'{v:+.3f}' for v in d_coh.values()) + f' | Δhigh-WSS R2 {ta["high_r2"]-tb["high_r2"]:+.3f} Δtop10 {ta["top10_ratio"]-tb["top10_ratio"]:+.3f}')
    print(f'  {fam} 三 seed 均值: Δ物理 {np.mean([r["d_pa"] for r in rows]):+.4f} (sd {np.std([r["d_pa"] for r in rows]):.4f}) Δ归一化 {np.mean([r["d_norm"] for r in rows]):+.4f} (sd {np.std([r["d_norm"] for r in rows]):.4f})')
    res[f'{fam}_paired'] = rows
    e_fam = summarize(f'{fam} 3-seed (Pa mean)', ens([f'{fam}_s{s}' for s in (1234, 7, 2025)], 'pa'))
    print(f'  {fam} 3-seed ensemble vs base 3-seed: Δ {e_fam["pa_r2_cb"] - res["X5D_v51 3-seed (Pa mean)"]["pa_r2_cb"]:+.4f}')
(OUT / 'analyze_wave1.json').write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float))
