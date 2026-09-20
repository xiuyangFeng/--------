"""把各臂放到同一组分桶上比（分桶由通道有效性定义，不随臂变），否则各臂"缺失区"点集不同不可比。"""
import json, sys
import numpy as np
from pathlib import Path
sys.path.insert(0, '/public/newhome/cy/Digital_twin/GNN')
from training_wss_min import longitudinal_geometry as L
ROOT = Path('/public/newhome/cy/Digital_twin/GNN'); RUNS = ROOT / 'training_wss_min/runs'
SIDE = ROOT / 'training_wss_min/experiments/wss_x5d_longitudinal_20260917/frozen_longitudinal_features'
P = 'eval/ckpt_best/predictions/test'
REF = [v for v in L.VALUE_KEYS if v.startswith('geom_ref_')]
ARMS = [('全32列', RUNS / 'wss_x5d_longitudinal_20260917/X5D_long_s1234'),
        ('L8', RUNS / 'wss_x5d_long_prune_20260917/L8_s1234'),
        ('L5', RUNS / 'wss_x5d_long_prune_20260917/L5_s1234'),
        ('L2', RUNS / 'wss_x5d_long_prune_20260917/L2_s1234'),
        ('L1', RUNS / 'wss_x5d_long_prune_20260917/L1_s1234')]
BASE = RUNS / 'wss_v51_wave1_20260916/X5D_v51_s1234'
units = sorted(str(p.parent.relative_to(BASE / P)) for p in (BASE / P).glob('*/*/*/predictions.npz'))
eb, e = [], {n: [] for n, _ in ARMS}
m_all, m_ref, m_round = [], [], []
for u in units:
    with np.load(BASE / P / u / 'predictions.npz') as z:
        rows, t = z['row_index'], z['true_pa'].astype(np.float64); pb = z['pred_pa']
    lt = np.log(np.maximum(t, 1e-6))
    eb.append((lt - np.log(np.maximum(pb, 1e-6))) ** 2)
    for n, run in ARMS:
        with np.load(run / P / u / 'predictions.npz') as z:
            e[n].append((lt - np.log(np.maximum(z['pred_pa'], 1e-6))) ** 2)
    with np.load(SIDE / u / 'features.npz', allow_pickle=False) as z:
        a = np.ones(len(rows), bool); r = np.ones(len(rows), bool)
        for v in L.VALUE_KEYS: a &= z[f'wall_{v}_valid'][rows].astype(bool)
        for v in REF: r &= z[f'wall_{v}_valid'][rows].astype(bool)
        m_round.append(z['wall_geom_ref_roundness_valid'][rows].astype(bool))
    m_all.append(a); m_ref.append(r)
eb = np.concatenate(eb); e = {n: np.concatenate(v) for n, v in e.items()}
m_all = np.concatenate(m_all); m_ref = np.concatenate(m_ref); m_round = np.concatenate(m_round)
B = [('A 全 32 通道有效', m_all),
     ('B ref8 有效但 pc 缺', m_ref & ~m_all),
     ('C 只有 roundness 有效', m_round & ~m_ref),
     ('D 连 roundness 也缺', ~m_round),
     ('全部', np.ones(len(eb), bool))]
print(f'{"分桶":24s} {"点占比":>7s} ' + ' '.join(f'{n:>9s}' for n, _ in ARMS))
for nm, sel in B:
    vals = [(eb[sel].mean() - e[n][sel].mean()) / eb[sel].mean() * 100 for n, _ in ARMS]
    print(f'{nm:24s} {sel.mean():7.3f} ' + ' '.join(f'{v:+8.2f}%' for v in vals))
