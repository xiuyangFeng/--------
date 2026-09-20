import sys
import numpy as np
from pathlib import Path
sys.path.insert(0,'/public/newhome/cy/Digital_twin/GNN')
from training_wss_min import longitudinal_geometry as L
ROOT=Path('/public/newhome/cy/Digital_twin/GNN'); RUNS=ROOT/'training_wss_min/runs'
BASE=RUNS/'wss_v51_wave1_20260916'; LONG=RUNS/'wss_x5d_longitudinal_20260917'
SIDE=ROOT/'training_wss_min/experiments/wss_x5d_longitudinal_20260917/frozen_longitudinal_features'
P='eval/ckpt_best/predictions/test'; S=(1234,7,2025)
V=list(L.VALUE_KEYS); REF=[v for v in V if v.startswith('geom_ref_')]; PC=[v for v in V if v.startswith('geom_pc_')]
units=sorted(str(p.parent.relative_to(BASE/'X5D_v51_s1234'/P)) for p in (BASE/'X5D_v51_s1234'/P).glob('*/*/*/predictions.npz'))
eb=[];el=[];mr=[];mp=[];tt=[]
for u in units:
    b,l=[],[]
    for s in S:
        with np.load(BASE/f'X5D_v51_s{s}'/P/u/'predictions.npz') as z: b.append(z['pred_pa']); rows=z['row_index']; t=z['true_pa'].astype(np.float64)
        with np.load(LONG/f'X5D_long_s{s}'/P/u/'predictions.npz') as z: l.append(z['pred_pa'])
    lt=np.log(np.maximum(t,1e-6)); lb=np.log(np.maximum(np.mean(b,0),1e-6)); ll=np.log(np.maximum(np.mean(l,0),1e-6))
    with np.load(SIDE/u/'features.npz',allow_pickle=False) as z:
        r_ok=np.ones(len(rows),bool); p_ok=np.ones(len(rows),bool)
        for v in REF: r_ok&=z[f'wall_{v}_valid'][rows].astype(bool)
        for v in PC:  p_ok&=z[f'wall_{v}_valid'][rows].astype(bool)
    eb.append((lt-lb)**2); el.append((lt-ll)**2); mr.append(r_ok); mp.append(p_ok); tt.append(t)
eb=np.concatenate(eb);el=np.concatenate(el);mr=np.concatenate(mr);mp=np.concatenate(mp);tt=np.concatenate(tt)
print(f'{"分桶":30s} {"点占比":>7s} {"底座 MSE":>9s} {"沿程 MSE":>9s} {"相对改善":>9s}')
for nm,sel in (('ref 有效 且 pc 有效',mr&mp),('ref 有效 但 pc 缺失',mr&~mp),('ref 缺失',~mr)):
    a,b_=eb[sel].mean(),el[sel].mean()
    print(f'{nm:30s} {sel.mean():7.3f} {a:9.4f} {b_:9.4f} {(a-b_)/a*100:8.2f}%')
print()
print('再按 WSS 高低拆 ref 缺失区（看代价来自哪）')
q=np.quantile(tt,.9)
for nm,sel in (('ref 缺失 且 WSS<P90',(~mr)&(tt<q)),('ref 缺失 且 WSS>=P90',(~mr)&(tt>=q))):
    a,b_=eb[sel].mean(),el[sel].mean()
    print(f'  {nm:28s} {sel.mean():7.3f} {a:9.4f} {b_:9.4f} {(a-b_)/a*100:8.2f}%')
