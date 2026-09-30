import numpy as np, glob, os, json
R='training_wss_min/runs'
def per_case_metrics(t,p):
    e=p-t; ae=np.abs(e)
    var=np.sum((t-t.mean())**2)
    return dict(r2=1-np.sum(e**2)/var if var>0 else np.nan,
                mae=ae.mean(), nmae_max=ae.mean()/np.abs(t).max(), nmae_range=ae.mean()/(t.max()-t.min()),
                nmae_mean=ae.mean()/np.abs(t).mean(), rel_l2=np.linalg.norm(e)/np.linalg.norm(t),
                nrmse_range=np.sqrt(np.mean(e**2))/(t.max()-t.min()))
def summarize(name, cases):
    keys=list(next(iter(cases.values())).keys())
    print(f'== {name}  n_cases={len(cases)}')
    for k in keys:
        v=np.array([c[k] for c in cases.values()])
        print(f'  {k:12s} mean {v.mean():.4f}  sd {v.std(ddof=1):.4f}  median {np.median(v):.4f}  min {v.min():.4f}  max {v.max():.4f}')
def pooled(name,T,P):
    t=np.concatenate(T); p=np.concatenate(P); m=per_case_metrics(t,p)
    print(f'  pooled: '+'  '.join(f'{k}={v:.4f}' for k,v in m.items())+f'  n={len(t)}')
    # case-balanced R2 (ratio of means)
    mse=np.mean([np.mean((pp-tt)**2) for tt,pp in zip(T,P)]); gm=t.mean()
    var=np.mean([np.mean((tt-gm)**2) for tt in T]); print(f'  R2_cb(ratio of means, global mean)={1-mse/var:.4f}')

# ---- WSS 5-seed ensemble
seeds=['1234','7','2025','11','2026']
runs=[f'{R}/wss_v51_wave1_20260916/X5D_v51_s{s}' for s in seeds]
runs=[r for r in runs if os.path.isdir(r)]
print('WSS runs found:',[os.path.basename(r) for r in runs])
units=sorted(glob.glob(runs[0]+'/eval/ckpt_best/predictions/test/*/*/*/predictions.npz'))
units=[u.split('/predictions/test/')[1].rsplit('/',1)[0] for u in units]
cases={}; T=[]; P=[]
for u in units:
    ts=[];ps=[]
    for r in runs:
        z=np.load(f'{r}/eval/ckpt_best/predictions/test/{u}/predictions.npz'); ts.append(z['true_pa'].astype(float)); ps.append(z['pred_pa'].astype(float))
    assert all(np.allclose(ts[0],x) for x in ts)
    t=ts[0]; p=np.mean(ps,axis=0); cases[u]=per_case_metrics(t,p); T.append(t); P.append(p)
summarize('WSS X5D_v51 5-seed Pa-mean, test34 (Pa)',cases); pooled('WSS',T,P)
# high-WSS region (per-case top10 truth) NMAE_max
# ---- pressure PF6 and velocity VF6 (3 seeds)
def vol_runs(prefix):
    out=[]
    for s,wave in [('1234','wss_local_wave2_20260912'),('7','wss_local_wave3_20260913'),('2025','wss_local_wave3_20260913')]:
        d=f'{R}/{wave}/{prefix}_s{s}'
        if os.path.isdir(d): out.append(d)
    return out
for prefix in ['PF6','VF6']:
    runs=vol_runs(prefix); print(prefix,'runs:',runs)
    units=sorted(glob.glob(runs[0]+'/eval/ckpt_best/predictions/test/*/*/*/predictions.npz'))
    units=[u.split('/predictions/test/')[1].rsplit('/',1)[0] for u in units]
    cases={}; T=[]; P=[]; casesw={}; Tw=[]; Pw=[]; casesv={}; 
    for u in units:
        ts=[];ps=[];kinds=None
        for r in runs:
            z=np.load(f'{r}/eval/ckpt_best/predictions/test/{u}/predictions.npz'); ts.append(z['true_raw'].astype(float)); ps.append(z['pred_raw'].astype(float)); kinds=z['point_kind']
        t=ts[0]; p=np.mean(ps,axis=0)
        if prefix=='VF6':
            # vector rel L2 and |u| metrics on interior points
            tm=np.linalg.norm(t,axis=1); pm=np.linalg.norm(p,axis=1)
            m=per_case_metrics(tm,pm); m['vec_rel_l2']=np.linalg.norm(p-t)/np.linalg.norm(t); m['vec_mae']=np.abs(p-t).mean()
            cases[u]=m; T.append(tm); P.append(pm)
        else:
            cases[u]=per_case_metrics(t,p); T.append(t); P.append(p)
            wall=kinds==0 if (kinds==0).any() else None
            interior=kinds!=0
            casesw[u]=per_case_metrics(t[interior],p[interior]); Tw.append(t[interior]); Pw.append(p[interior])
    summarize(f'{prefix} {len(runs)}-seed mean, test34 (raw units; pressure Pa relative to case mean; velocity |u| m/s)',cases); pooled(prefix,T,P)
    if prefix=='PF6':
        summarize('PF6 interior-only (point_kind!=0)',casesw); pooled('PF6 interior',Tw,Pw)
        print('  point_kind values example:',np.unique(kinds,return_counts=True))
