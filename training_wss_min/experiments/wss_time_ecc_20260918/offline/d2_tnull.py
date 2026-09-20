"""D2：自由基线 T-null（不训练）。
A：ln τ̂(x,t) = ln τ̂_peak(x) + μ_f(t) − μ_f(peak)          （峰值场 × 共享时间形状，ln 平移）
B：z_peak = (ln τ̂_peak − μ_f(peak)) / σ_f(peak)；ln τ̂ = μ_f(t) + σ_f(t)·z_peak   （σ 缩放版）
C：同 A 但用**真值**峰值场（诊断分解：时间形状误差本身有多大；仅残差分析，不是模型）
train136 用 cascade 折外峰值预测 + 该病例所在折的训练折统计；test34 用五 seed 集成 + train136 统计。"""
import sys
import numpy as np
from common import *

limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
wf = load_waveform(); q_norm = np.asarray(wf['q_norm'])
stats = {s: load_frame_stats(s)[:2] for s in ('train136', 'fold0', 'fold1', 'fold2')}
res = {}
groups = [('train', TRAIN[:limit]), ('test', TEST[:limit])] + [(f'fold{f}', FOLD_HELD[f][:limit]) for f in range(3)]
for partition, cases in groups:
    mets = {v: TimeMetrics(q_norm) for v in ('A_shift', 'B_scale', 'C_oracle_peak')}
    for cid in cases:
        d = load_bundle(cid)
        mu, sd = stats['train136' if partition == 'test' else f'fold{FOLD_OF[cid]}']
        lg = ln_pa(d['tau']); lp = load_oof_peak_ln(cid, 'test' if partition == 'test' else 'train', d['ids'])
        mets['A_shift'].add(lg, lp[None, :] + (mu - mu[PEAK_INDEX])[:, None])
        mets['B_scale'].add(lg, mu[:, None] + sd[:, None] * ((lp - mu[PEAK_INDEX]) / sd[PEAK_INDEX])[None, :])
        mets['C_oracle_peak'].add(lg, lg[PEAK_INDEX][None, :] + (mu - mu[PEAK_INDEX])[:, None])
    res[partition] = {v: m.summary() for v, m in mets.items()}
    tag = {'train': 'train136 折外', 'test': 'test34 五 seed 集成'}.get(partition, f'{partition} 留出折（折外）')
    print(f'\n=== T-null · {tag}（{len(cases)} 例）===')
    for v in mets:
        print(f'  {v:14s} ' + fmt_summary(res[partition][v]))
write_json(OUT / 'd2_tnull.json', res)
print('wrote d2_tnull.json')
