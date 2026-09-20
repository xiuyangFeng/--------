import json,sys
import numpy as np
from pathlib import Path
sys.path.insert(0,'/public/newhome/cy/Digital_twin/GNN')
from training_wss_min import longitudinal_geometry as L
ROOT=Path('/public/newhome/cy/Digital_twin/GNN'); RUNS=ROOT/'training_wss_min/runs'
BASE=RUNS/'wss_v51_wave1_20260916'; LONG=RUNS/'wss_x5d_longitudinal_20260917'
SIDE=ROOT/'training_wss_min/experiments/wss_x5d_longitudinal_20260917/frozen_longitudinal_features'
P='eval/ckpt_best/predictions/test'; S=(1234,7,2025)
units=sorted(str(p.parent.relative_to(BASE/'X5D_v51_s1234'/P)) for p in (BASE/'X5D_v51_s1234'/P).glob('*/*/*/predictions.npz'))
eb=[];el=[];vv=[];tt=[];cc=[]
for k,u in enumerate(units):
    pb=[];pl=[]
    for s in S:
        with np.load(BASE/f'X5D_v51_s{s}'/P/u/'predictions.npz') as z: pb.append(z['pred_pa']); rows=z['row_index']; true=z['true_pa'].astype(np.float64)
        with np.load(LONG/f'X5D_long_s{s}'/P/u/'predictions.npz') as z: pl.append(z['pred_pa'])
    lb=np.log(np.maximum(np.mean(pb,0),1e-6)); ll=np.log(np.maximum(np.mean(pl,0),1e-6)); lt=np.log(np.maximum(true,1e-6))
    with np.load(SIDE/u/'features.npz',allow_pickle=False) as z:
        m=np.stack([z[f'wall_{n}_valid'][rows].astype(bool) for n in L.VALUE_KEYS],1)
    eb.append((lt-lb)**2); el.append((lt-ll)**2); vv.append(m.all(1)); tt.append(true); cc.append(np.full(len(lt),k))
eb=np.concatenate(eb);el=np.concatenate(el);vv=np.concatenate(vv);tt=np.concatenate(tt);cc=np.concatenate(cc)
print(f'总点 {len(eb)}；32 通道全有效 {vv.mean():.3f}')
print(f'{"分组":22s} {"点占比":>7s} {"底座 MSE(ln)":>13s} {"沿程 MSE(ln)":>13s} {"相对改善":>9s}')
for nm,sel in (('全部',np.ones(len(eb),bool)),('沿程通道全有效',vv),('有缺失通道',~vv)):
    a,b=eb[sel].mean(),el[sel].mean()
    print(f'{nm:22s} {sel.mean():7.3f} {a:13.4f} {b:13.4f} {(a-b)/a*100:8.2f}%')
print()
q=np.quantile(tt,[.5,.9,.99])
for nm,sel in (('WSS 下 50%',tt<q[0]),('50–90%',(tt>=q[0])&(tt<q[1])),('90–99%',(tt>=q[1])&(tt<q[2])),('top 1%',tt>=q[2])):
    a,b=eb[sel].mean(),el[sel].mean()
    print(f'{nm:22s} {sel.mean():7.3f} {a:13.4f} {b:13.4f} {(a-b)/a*100:8.2f}%')
