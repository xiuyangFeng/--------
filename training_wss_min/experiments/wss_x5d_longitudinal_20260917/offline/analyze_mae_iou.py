import json, sys
import numpy as np
from pathlib import Path
sys.path.insert(0,'/public/newhome/cy/Digital_twin/GNN')
R=Path('/public/newhome/cy/Digital_twin/GNN/training_wss_min/runs')
NEW=R/'wss_x5d_longitudinal_20260917'; BASE=R/'wss_v51_wave1_20260916'
P='eval/ckpt_best/predictions/test'
units=sorted(str(p.parent.relative_to(BASE/'X5D_v51_s1234'/P)) for p in (BASE/'X5D_v51_s1234'/P).glob('*/*/*/predictions.npz'))
def load(run):
    out={}
    for u in units:
        with np.load(run/P/u/'predictions.npz') as z: out[u]={k:z[k].astype(np.float64) for k in ('pred_pa','true_pa')}
    return out
S=(1234,7,2025)
d={f'L{s}':load(NEW/f'X5D_long_s{s}') for s in S}; d.update({f'B{s}':load(BASE/f'X5D_v51_s{s}') for s in S})
d.update({f'B{s}':load(BASE/f'X5D_v51_s{s}') for s in (11,2026)})
def mae(pr): return float(np.mean([np.mean(np.abs(d['B1234'][u]['true_pa']-pr[i])) for i,u in enumerate(units)]))
def iou(pr):
    v=[]
    for i,u in enumerate(units):
        t=d['B1234'][u]['true_pa']; p=pr[i]
        mt=t>=np.quantile(t,.9); mp=p>=np.quantile(p,.9)
        v.append((mt&mp).sum()/(mt|mp).sum())
    return float(np.mean(v))
print('per-seed paired (best ckpt):')
rm=[];ri=[]
for s in S:
    pl=[d[f'L{s}'][u]['pred_pa'] for u in units]; pb=[d[f'B{s}'][u]['pred_pa'] for u in units]
    dm=mae(pl)-mae(pb); di=iou(pl)-iou(pb); rm.append(dm); ri.append(di)
    print(f'  s{s}: MAE {mae(pb):.4f} -> {mae(pl):.4f} (Δ {dm:+.4f} Pa)  top10 IoU {iou(pb):.4f} -> {iou(pl):.4f} (Δ {di:+.4f})')
print(f'  三 seed 均值 ΔMAE {np.mean(rm):+.4f} Pa ({sum(x<0 for x in rm)}/3 降)  ΔIoU {np.mean(ri):+.4f} ({sum(x>0 for x in ri)}/3 升)')
def ens(names): return [np.mean([d[n][u]['pred_pa'] for n in names],axis=0) for u in units]
L3=ens([f'L{s}' for s in S]); B3=ens([f'B{s}' for s in S]); B5=ens([f'B{s}' for s in (1234,7,2025,11,2026)])
for n,e in (('X5D_long 3-seed',L3),('X5D_v51 3-seed',B3),('X5D_v51 5-seed',B5)):
    print(f'  {n:18s} MAE {mae(e):.4f} Pa  top10 IoU {iou(e):.4f}')
