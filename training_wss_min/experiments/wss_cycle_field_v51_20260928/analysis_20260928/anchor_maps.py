# 真值"锚图"能解释多少逐帧 ln τ（只读 oracle）：每帧 ln τ(x,t) ≈ a_t + Σ_k b_kt·map_k(x)
# (i) 逐例逐帧最小二乘（最乐观）；(ii) 系数只随相位变、跨病例共享（训练折拟合、留出折评估，接近可部署结构）
import sys, numpy as np
sys.path.insert(0, '/public/newhome/cy/Digital_twin/GNN/training_wss_min/experiments/wss_time_ecc_20260918/offline')
from common import TRAIN, FOLD_OF, ln_pa, VIEW
SEG={"plateau":list(range(0,5))+list(range(58,80)),"accel":list(range(10,17)),"peak":list(range(17,27)),
     "decel":list(range(27,43)),"trough":list(range(5,10))+list(range(43,58))}
cases=[]
for cid in TRAIN:
    with np.load(VIEW/cid/'bundle.npz', allow_pickle=False) as z:
        tau=z['wall_wss'][:80].astype(np.float64)
    L=ln_pa(tau)                                  # 80,N
    maps={'P':L[21],'M':ln_pa(tau.mean(0)),'D':ln_pa(tau[SEG['plateau']].mean(0)),
          'T':ln_pa(tau[SEG['trough']].mean(0)),'E':ln_pa(tau[SEG['decel']].mean(0))}
    cases.append((cid,L,maps))
SETS={'峰值 P':['P'],'TAWSS M':['M'],'舒张平台 D':['D'],'P+M':['P','M'],'P+D':['P','D'],'P+D+E':['P','D','E'],'P+D+E+T':['P','D','E','T']}
def design(maps,keys):
    X=np.stack([maps[k] for k in keys],1); X=X-X.mean(0); return X
print('(i) 逐例逐帧最小二乘 oracle，ln R²（病例中位的段均值）')
print('set'.ljust(14),' '.join(s.rjust(8) for s in SEG))
for name,keys in SETS.items():
    r2=np.zeros((len(cases),80))
    for i,(cid,L,maps) in enumerate(cases):
        X=design(maps,keys); Y=L-L.mean(1,keepdims=True)          # 80,N
        B,_,_,_=np.linalg.lstsq(X,Y.T,rcond=None); R=Y.T-X@B
        r2[i]=1-(R**2).mean(0)/Y.var(1)
    med=np.median(r2,0)
    print(name.ljust(14),' '.join(f'{med[SEG[s]].mean():8.3f}' for s in SEG))
print('\n(ii) 系数只随相位变、跨病例共享：训练折拟合 → 留出折评估（ln R² 病例中位段均值）')
print('set'.ljust(14),' '.join(s.rjust(8) for s in SEG))
for name,keys in SETS.items():
    r2=np.zeros((len(cases),80))
    for f in range(3):
        tr=[c for c in cases if FOLD_OF[c[0]]!=f]; te=[(i,c) for i,c in enumerate(cases) if FOLD_OF[c[0]]==f]
        # 按病例中心化后池化（每例等权：按点数开方缩放）
        XtX=np.zeros((len(keys),len(keys))); XtY=np.zeros((len(keys),80))
        for cid,L,maps in tr:
            X=design(maps,keys); Y=L-L.mean(1,keepdims=True); w=1/X.shape[0]
            XtX+=w*X.T@X; XtY+=w*X.T@Y.T
        B=np.linalg.solve(XtX,XtY)
        for i,(cid,L,maps) in te:
            X=design(maps,keys); Y=L-L.mean(1,keepdims=True); R=Y.T-X@B
            r2[i]=1-(R**2).mean(0)/Y.var(1)
    med=np.median(r2,0)
    print(name.ljust(14),' '.join(f'{med[SEG[s]].mean():8.3f}' for s in SEG))
# 各锚图之间的相关（逐例 Pearson 中位）
import itertools
print('\n锚图两两相关（逐例 ln Pearson 中位）')
for a,b in itertools.combinations(['P','M','D','E','T'],2):
    print(a,b, round(float(np.median([np.corrcoef(m[a],m[b])[0,1] for _,_,m in cases])),3))
