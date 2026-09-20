"""不训练的验证：按沿程通道有效性逐点在"底座 / 沿程臂"之间切换，看能否拿掉缺失点上的代价。

三 seed Pa 均值集成；门控只用部署侧可得的 mask，不用真值。
"""
import json, sys
import numpy as np
from pathlib import Path
sys.path.insert(0, '/public/newhome/cy/Digital_twin/GNN')
from training_wss_min import evaluate as E
from training_wss_min import longitudinal_geometry as L
M = E.M
ROOT = Path('/public/newhome/cy/Digital_twin/GNN'); RUNS = ROOT / 'training_wss_min/runs'
BASE = RUNS / 'wss_v51_wave1_20260916'; LONG = RUNS / 'wss_x5d_longitudinal_20260917'
SIDE = ROOT / 'training_wss_min/experiments/wss_x5d_longitudinal_20260917/frozen_longitudinal_features'
OUT = Path(__file__).resolve().parent
P = 'eval/ckpt_best/predictions/test'; S = (1234, 7, 2025)
V = list(L.VALUE_KEYS); REF = [v for v in V if v.startswith('geom_ref_')]
units = sorted(str(p.parent.relative_to(BASE / 'X5D_v51_s1234' / P))
               for p in (BASE / 'X5D_v51_s1234' / P).glob('*/*/*/predictions.npz'))
truth, pb, pl, gate32, gate8 = [], [], [], [], []
for u in units:
    b, l = [], []
    for s in S:
        with np.load(BASE / f'X5D_v51_s{s}' / P / u / 'predictions.npz') as z:
            b.append(z['pred_pa']); rows = z['row_index']; t = z['true_pa'].astype(np.float64)
        with np.load(LONG / f'X5D_long_s{s}' / P / u / 'predictions.npz') as z:
            l.append(z['pred_pa'])
    with np.load(SIDE / u / 'features.npz', allow_pickle=False) as z:
        m32 = np.ones(len(rows), bool); m8 = np.ones(len(rows), bool)
        for v in V:
            ok = z[f'wall_{v}_valid'][rows].astype(bool)
            m32 &= ok
            if v in REF:
                m8 &= ok
    truth.append(t); pb.append(np.mean(b, 0)); pl.append(np.mean(l, 0)); gate32.append(m32); gate8.append(m8)
coh = [u.split('/')[0] for u in units]


def rep(name, preds):
    r = M.casebalanced_field_metrics(truth, preds)['r2']
    per = {c: M.casebalanced_field_metrics([truth[i] for i in range(34) if coh[i] == c],
                                           [preds[i] for i in range(34) if coh[i] == c])['r2'] for c in ('AG', 'AAA', 'ILO')}
    tt = np.concatenate(truth); pp = np.concatenate(preds); q = np.quantile(tt, .9); mk = tt >= q
    hi = 1 - ((tt[mk] - pp[mk]) ** 2).sum() / ((tt[mk] - tt[mk].mean()) ** 2).sum()
    print(f'  {name:36s} Pa R2_cb {r:.4f}  high-WSS {hi:.3f}  AG/AAA/ILO ' + '/'.join(f'{v:.3f}' for v in per.values()))
    return {'pa_r2_cb': float(r), 'high_wss_r2': float(hi), 'per_cohort': {k: float(v) for k, v in per.items()}}


res = {}
print('test34，三 seed Pa 均值集成')
res['base'] = rep('底座 X5D_v51', pb)
res['long'] = rep('沿程臂 X5D_long', pl)
res['mean'] = rep('两者等权平均（对照）', [(a + c) / 2 for a, c in zip(pb, pl)])
res['gate32'] = rep('门控：32 通道全有效处用沿程臂', [np.where(g, c, a) for a, c, g in zip(pb, pl, gate32)])
res['gate8'] = rep('门控：ref 8 全有效处用沿程臂', [np.where(g, c, a) for a, c, g in zip(pb, pl, gate8)])
for w in (0.25, 0.5, 0.75):
    res[f'soft{w}'] = rep(f'软门控：有效处 {1-w:.2f}底座+{w:.2f}沿程，缺失处纯底座',
                          [np.where(g, (1 - w) * a + w * c, a) for a, c, g in zip(pb, pl, gate32)])
(OUT / 'gate_by_validity.json').write_text(json.dumps(res, indent=1, ensure_ascii=False))
print('\nwrote', OUT / 'gate_by_validity.json')
