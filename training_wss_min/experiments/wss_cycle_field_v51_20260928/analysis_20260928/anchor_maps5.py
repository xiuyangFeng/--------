# 5 张相位窗图（每段一张）的真值上限：非中心化、系数只随相位变（训练折池化拟合 → 留出折），两种图定义
import sys, numpy as np
sys.path.insert(0, '/public/newhome/cy/Digital_twin/GNN/training_wss_min/experiments/wss_time_ecc_20260918/offline')
from common import TRAIN, FOLD_OF, ln_pa, VIEW
SEG={"plateau":list(range(0,5))+list(range(58,80)),"accel":list(range(10,17)),"peak":list(range(17,27)),
     "decel":list(range(27,43)),"trough":list(range(5,10))+list(range(43,58))}
ORDER=['peak','accel','decel','plateau','trough']
TRO=np.sort(np.argsort(np.load if False else np.array(__import__('json').load(open('/public/newhome/cy/Digital_twin/GNN/training_wss_min/experiments/wss_time_ecc_20260918/offline/protocol_inlet_waveform_v51.json'))['q_norm']))[:20])
cases=[]
for cid in TRAIN:
    with np.load(VIEW/cid/'bundle.npz', allow_pickle=False) as z:
        tau=z['wall_wss'].astype(np.float64)       # 81,N
    L=ln_pa(tau)
    mA=np.stack([ln_pa(tau[SEG[s]].mean(0)) for s in ORDER])   # ln(窗均 τ)
    mB=np.stack([L[SEG[s]].mean(0) for s in ORDER])            # 窗均 ln τ
    cases.append((cid,tau,L,mA,mB))
def r2cb_pa(taus, preds):
    # 病例等权逐帧 R²_cb（Pa），帧 0–80 的周期均值、谷底 20 帧均值
    m1=np.array([t.mean(1) for t in taus]); m2=np.array([(t**2).mean(1) for t in taus])
    mse=np.array([((t-p)**2).mean(1) for t,p in zip(taus,preds)])
    g=m1.mean(0); r=1-mse.mean(0)/(m2-2*g*m1+g**2).mean(0)
    return r
for tag,idx in (('ln(窗均τ)',3),('窗均lnτ',4)):
    preds={}
    for f in range(3):
        tr=[c for c in cases if FOLD_OF[c[0]]!=f]
        XtX=np.zeros((6,6)); XtY=np.zeros((6,81))
        for c in tr:
            X=np.vstack([np.ones(c[idx].shape[1]),c[idx]]).T; w=1/X.shape[0]
            XtX+=w*X.T@X; XtY+=w*X.T@c[2].T
        B=np.linalg.solve(XtX,XtY)
        for c in cases:
            if FOLD_OF[c[0]]==f:
                X=np.vstack([np.ones(c[idx].shape[1]),c[idx]]).T
                preds[c[0]]=np.exp(X@B).T              # 81,N  Pa
    # 逐折 R²_cb 再三折平均（与主表口径一致）
    rows=[]
    for f in range(3):
        cs=[c for c in cases if FOLD_OF[c[0]]==f]
        r=r2cb_pa([c[1] for c in cs],[preds[c[0]] for c in cs]); rows.append(r)
    r=np.mean(rows,0)
    lnr2=[]
    for c in cases:
        Lp=np.log(np.clip(preds[c[0]],0.05,None)+1e-6); Y=c[2]
        lnr2.append(1-((Lp-Y)**2).mean(1)/Y.var(1))
    lnr2=np.median(np.array(lnr2),0)
    print(f'{tag}: Pa R²_cb 全周期 {r[:80].mean():.3f}  谷底20帧 {r[TRO].mean():.3f}  峰值帧 {r[21]:.3f} | ln R² 段均(病例中位):',
          {s:round(float(lnr2[SEG[s]].mean()),3) for s in SEG})
