# 帧 0 与帧 80 同相位（相隔一个周期）的真值一致性 = 舒张平台相位的标签可重复性（只读）
import sys, numpy as np
sys.path.insert(0, '/public/newhome/cy/Digital_twin/GNN/training_wss_min/experiments/wss_time_ecc_20260918/offline')
from common import TRAIN, ln_pa, VIEW
from scipy.stats import spearmanr
rows=[]
for cid in TRAIN:
    with np.load(VIEW/cid/'bundle.npz', allow_pickle=False) as z:
        tau=z['wall_wss']
        a, b = tau[0].astype(np.float64), tau[80].astype(np.float64)
        mid = tau[40].astype(np.float64)
    la, lb = ln_pa(a), ln_pa(b)
    r2_ln = 1-((la-lb)**2).mean()/la.var()
    r2_pa = 1-((a-b)**2).mean()/a.var()
    pr = np.corrcoef(la, lb)[0,1]
    sp = spearmanr(a, b).correlation
    rel = np.linalg.norm(a-b)/np.linalg.norm(a)
    rows.append((cid.split('/')[0], r2_ln, r2_pa, pr, sp, rel, a.mean(), b.mean()))
import collections
arr=np.array([r[1:] for r in rows])
names=['R2_ln','R2_pa','pearson_ln','spearman','relL2_pa','mean0','mean80']
def summ(x): return f'med {np.median(x):.3f}  p10 {np.quantile(x,.1):.3f}  p90 {np.quantile(x,.9):.3f}  mean {x.mean():.3f}'
print('n', len(rows))
for i,n in enumerate(names[:5]): print(n.ljust(11), summ(arr[:,i]))
print('均值比 frame80/frame0:', summ(arr[:,6]/arr[:,5]))
for c in sorted(set(r[0] for r in rows)):
    m=np.array([r[0]==c for r in rows])
    print(c, int(m.sum()), 'R2_ln med', round(float(np.median(arr[m,0])),3), 'spearman med', round(float(np.median(arr[m,3])),3), 'relL2 med', round(float(np.median(arr[m,4])),3))
print('R2_ln < 0.9 的病例数:', int((arr[:,0]<0.9).sum()), ' < 0.7:', int((arr[:,0]<0.7).sum()))
