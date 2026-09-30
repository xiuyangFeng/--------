from pathlib import Path
import json, csv, numpy as np
from scipy.stats import spearmanr
ROOT=Path('/public/newhome/cy/Digital_twin/GNN')
RUNS=ROOT/'training_wss_min/runs'
OUT=Path(__file__).resolve().parent
SEEDS=(1234,7,2025)
P='eval/ckpt_best/predictions/test'
rows=[]
percase={}

def load_manifest(base):
    j=json.load(open(base/'manifest.json'))
    return {e.get('unit_id',str(Path(e['file']).parent)): base/e['file'] for e in j['cases']}

def metrics(t,p):
    t=np.asarray(t,np.float64).reshape(-1); p=np.asarray(p,np.float64).reshape(-1)
    assert t.shape==p.shape and np.isfinite(t).all() and np.isfinite(p).all()
    t2=np.mean(t*t);p2=np.mean(p*p);mse=np.mean((t-p)**2)
    return dict(n_points=len(t),true_mean=t.mean(),true_second_moment=t2,mse=mse,mae=np.mean(np.abs(t-p)),approx_disp=np.sqrt(mse/t2),cosine_spatial=np.mean(t*p)/np.sqrt(t2*p2),spearman=spearmanr(t,p).correlation)

def aggregate(per):
    vals=list(per.values()); n=len(vals)
    mu=np.mean([x['true_mean'] for x in vals]); m2=np.mean([x['true_second_moment'] for x in vals]); mse=np.mean([x['mse'] for x in vals])
    ans=dict(n_cases=n,n_points=sum(x['n_points'] for x in vals),r2_cb=1-mse/(m2-mu*mu))
    for key in ('mae','approx_disp','cosine_spatial','spearman'):
        v=np.array([x[key] for x in vals]);ans[key+'_cb' if key=='mae' else key+'_casemean']=float(np.mean(v));ans[key+'_casesd']=float(np.std(v,ddof=1))
    return ans

def calc(label,target,groups,protocol,subset=None):
    maps=[]
    for bases in groups:
        d={}
        for base in bases:
            f=load_manifest(base);assert not set(d)&set(f);d.update(f)
        maps.append(d)
    assert all(set(m)==set(maps[0]) for m in maps)
    cases=sorted(maps[0]);per={}
    for cid in cases:
        if subset is not None and cid not in subset:continue
        preds=[];tru=None;ri=None
        for m in maps:
            with np.load(m[cid]) as z:
                t=z['true_pa'].astype(np.float64);p=z['pred_pa'].astype(np.float64)
                if tru is None:tru=t;ri=z['row_index']
                else:assert np.array_equal(tru,t) and np.array_equal(ri,z['row_index'])
                preds.append(p)
        pred=np.mean(preds,axis=0)
        if target=='osi':pred=np.clip(pred,0,.5)
        per[cid]=metrics(tru,pred)
    row=dict(label=label,target=target,protocol=protocol,n_seed_ensemble=len(groups),sources=[[str(b) for b in g] for g in groups],**aggregate(per))
    rows.append(row);percase[label+'|'+target]=per
    print(json.dumps({k:v for k,v in row.items() if k!='sources'}),flush=True)
    return per

cv51={}
for target,arm,ch in [('tawss','A1',None),('osi','O1',None),('osi','O2',None),('tawss','M1','tawss'),('osi','M1','osi')]:
    if arm=='M1': bases=[RUNS/'wss_cycle_m1_20260921'/('M1_f%d_s1234'%k)/P/ch for k in range(3)]
    else:bases=[RUNS/'wss_cycle_20260920'/('%s_f%d_s1234'%(arm,k))/P for k in range(3)]
    cv51[arm+'|'+target]=calc('v5.1 '+arm+' cv3 s1234',target,[bases],'v5.1 cv3 OOF train136; 3 folds merged; single seed1234')
lib136=set(cv51['M1|osi'])
for target,arm,ch in [('tawss','A1',None),('osi','O1',None),('osi','O2',None),('tawss','M1','tawss'),('osi','M1','osi')]:
    if arm=='M1':groups=[[RUNS/'wss_cycle_m1_stage3_20260922'/('M1_s%d'%s)/P/ch] for s in SEEDS]
    else:groups=[[RUNS/'wss_cycle_stage3_20260921'/('%s_s%d'%(arm,s))/P] for s in SEEDS]
    calc('v5.1 '+arm+' test34 ens3',target,groups,'v5.1 train136 -> test34; seed1234/7/2025 physical mean ensemble')
for target in ('tawss','osi'):
    groups=[[RUNS/'wss_cycle_m1_v52_20260925'/('M1cap_v52cv_f%d_s%d'%(k,s))/P/target for k in range(5)] for s in SEEDS]
    for s,g in zip(SEEDS,groups):
        per=calc('v5.2 M1cap CV5 s%d'%s,target,[g],'v5.2 CV5 OOF261; 5 folds merged; single seed')
        row=dict(label='v5.2 M1cap CV5 s%d common136'%s,target=target,protocol='v5.2 CV5 OOF restricted to same136 as v5.1 cv3',n_seed_ensemble=1,sources=[[str(x) for x in g]],**aggregate({c:d for c,d in per.items() if c in lib136}))
        rows.append(row)
        print(json.dumps({k:v for k,v in row.items() if k!='sources'}),flush=True)
    per=calc('v5.2 M1cap CV5 ens3',target,groups,'v5.2 CV5 OOF261; 5 folds merged; seed1234/7/2025 physical mean ensemble')
    rows.append(dict(label='v5.2 M1cap CV5 ens3 common136',target=target,protocol='v5.2 CV5 ensemble OOF restricted to same136 as v5.1 cv3',n_seed_ensemble=3,sources=[[str(x) for x in g] for g in groups],**aggregate({c:d for c,d in per.items() if c in lib136})))
# Three independent training seeds: mean and sample sd of the final OOF metrics (not fold means)
for target in ('tawss','osi'):
    for suffix in ('',' common136'):
        sel=[r for r in rows if r['target']==target and r['label'] in ['v5.2 M1cap CV5 s%d%s'%(s,suffix) for s in SEEDS]]
        out=dict(label='v5.2 M1cap CV5 seedmean3'+suffix,target=target,protocol=sel[0]['protocol']+'; mean/sample sd over three training seeds',n_cases=sel[0]['n_cases'],n_seeds=3)
        for k in ('r2_cb','mae_cb','approx_disp_casemean','cosine_spatial_casemean','spearman_casemean'):
            out[k]=float(np.mean([r[k] for r in sel]));out[k+'_seedsd']=float(np.std([r[k] for r in sel],ddof=1))
        rows.append(out)
        print(json.dumps(out),flush=True)
meta={'checkpoint':'existing ckpt_best selected by configured train_loss rule, no new training or inference','aggregation':'r2_cb=1-mean_case(mean_point(error^2))/[mean_case(mean_point(y^2))-mean_case(mean_point(y))^2]; MAE mean_case(mean_point(abs(error)))','approx_disp':'mean_case(||pred-true||2 / ||true||2), matching paper scalar derived-map metric','cosine':'mean_case(dot(pred,true)/(||pred||2 ||true||2)); cosine of the complete spatial scalar field, NOT pointwise 3D-vector direction cosine','spearman':'mean of within-case Spearman rank correlations','case_sd':'sample sd between cases, seed sd is distinct','data_contract':'TAWSS and OSI are direct heads; labels computed from 80 vector WSS phases0..79; TAWSS Pa and OSI dimensionless; ensembles average physical predictions at matching rows'}
with open(OUT/'cycle_metrics.json','w') as f:json.dump(dict(meta=meta,rows=rows),f,indent=2)
with open(OUT/'cycle_percase.json','w') as f:json.dump(percase,f)
cols=['label','target','protocol','n_cases','r2_cb','mae_cb','approx_disp_casemean','cosine_spatial_casemean','spearman_casemean','r2_cb_seedsd','mae_cb_seedsd','approx_disp_casemean_seedsd','cosine_spatial_casemean_seedsd','spearman_casemean_seedsd']
with open(OUT/'cycle_metrics.tsv','w') as f:
    w=csv.DictWriter(f,fieldnames=cols,delimiter='\t',extrasaction='ignore');w.writeheader();w.writerows(rows)
print('OUTPUT',str(OUT/'cycle_metrics.json'),flush=True)
