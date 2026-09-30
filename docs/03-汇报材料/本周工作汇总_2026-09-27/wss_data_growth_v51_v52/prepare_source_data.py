from pathlib import Path
import json,csv,numpy as np
ROOT=Path('/public/newhome/cy/Digital_twin/GNN')
OUT=Path(__file__).resolve().parent
SEEDS=(1234,7,2025)
LC=ROOT/'training_wss_min/experiments/wss_learning_curve_20260920/offline/analyze_learning_curve.json'
hist=json.loads(LC.read_text()); rows=[]; cases_out=[]; sources=[]
def metric(vals):
    a=np.asarray(vals,dtype=np.float64)
    mu=a[:,0].mean(); var=a[:,1].mean()-mu*mu
    return dict(r2_cb=float(1-a[:,2].mean()/var),mae=float(a[:,3].mean()),n_eval=len(a))
def old_run(frac,f,s):
    if frac==100 and s==1234:return ROOT/f'training_wss_min/runs/wss_v51_wave2a_20260916/X5D_v51_f{f}_s{s}'
    return ROOT/f'training_wss_min/runs/wss_learning_curve_20260920/LC{frac}_f{f}_s{s}'
common=None
for frac in (25,50,75,100):
 for s in SEEDS:
  merged={};nt=[]
  for f in range(3):
   run=old_run(frac,f,s);mp=run/'eval/ckpt_best/metrics.json';j=json.loads(mp.read_text())['test']
   cfg=json.loads((run/'config.json').read_text());split=json.loads(Path(cfg['data']['split_path']).read_text());nt.append(len(split['train_cases']))
   pc=j['per_case'];maes={cid:float(v['overall']['mae']) for cid,v in pc.items()}
   assert np.isclose(np.mean(list(maes.values())),j['field_casebalanced']['mae'],atol=1e-10,rtol=0)
   assert not set(merged)&set(maes);merged.update(maes);sources.append(str(mp))
  assert len(merged)==136
  if common is None:common=set(merged)
  assert set(merged)==common
  h=hist['pooled'][f'LC{frac}_s{s}'];assert h['complete']
  rows.append(dict(version='V5.1',recipe='X5D',fraction=frac,seed=s,n_train_mean=float(np.mean(nt)),n_train_min=min(nt),n_train_max=max(nt),n_eval=136,r2_cb=h['r2cb'],mae=float(np.mean(list(merged.values()))),r2_jet=h['r2cb_jet'],r2_nonjet=h['r2cb_nonjet'],r2_ilo=h['cohorts']['ILO'],protocol='CV3',eval_set='common136'))
  for cid,v in merged.items():cases_out.append(dict(version='V5.1',recipe='X5D',fraction=frac,seed=s,case_id=cid,mae=v))
print('V5.1 learning curve: 12 rows, each with 136 cases',flush=True)
full52=[];validation=[];train52=None
for recipe in ('X5D','X5Dcap'):
 for s in SEEDS:
  pc={};nt=[]
  for f in range(5):
   run=ROOT/f'training_wss_min/runs/wss_v52_20260923/{recipe}_v52cv_f{f}_s{s}'
   cfg=json.loads((run/'config.json').read_text());split=json.loads(Path(cfg['data']['split_path']).read_text());nt.append(len(split['train_cases']))
   p=run/'eval/ckpt_best/predictions/test';man=json.loads((p/'manifest.json').read_text());fold=[]
   assert set(split['test_cases'])=={r['unit_id'] for r in man['cases']}
   for c in man['cases']:
    cid=c['unit_id'];assert cid not in pc
    with np.load(p/c['file']) as z:
     t=z['true_pa'].astype(np.float64);pr=z['pred_pa'].astype(np.float64)
    assert t.shape==pr.shape and np.isfinite(t).all() and np.isfinite(pr).all()
    v=[float(t.mean()),float(np.mean(t*t)),float(np.mean((pr-t)**2)),float(np.mean(np.abs(pr-t)))]
    pc[cid]=v;fold.append(v)
    cases_out.append(dict(version='V5.2',recipe=recipe,fraction=100,seed=s,case_id=cid,mae=v[3],true_mean=v[0],true_second_moment=v[1],mse=v[2],in_common136=cid in common))
   expected=json.loads((run/'eval/ckpt_best/metrics.json').read_text())['test']['field_casebalanced'];got=metric(fold)
   for key,ek in [('r2_cb','r2'),('mae','mae')]:
    err=abs(got[key]-expected[ek]);assert err<1e-10;validation.append(err)
   sources.append(str(p/'manifest.json'))
  assert len(pc)==261 and common<=set(pc)
  train52=nt
  base=dict(version='V5.2',recipe=recipe,fraction=100,seed=s,n_train_mean=float(np.mean(nt)),n_train_min=min(nt),n_train_max=max(nt),protocol='CV5')
  rows.append(dict(**base,eval_set='common136',**metric([pc[c] for c in sorted(common)])))
  full52.append(dict(**base,eval_set='full261',**metric(list(pc.values()))))
  print(recipe,s,'same136',rows[-1]['r2_cb'],rows[-1]['mae'],flush=True)
summary=[]
for version,recipe,frac in [('V5.1','X5D',f) for f in (25,50,75,100)]+[('V5.2',r,100) for r in ('X5D','X5Dcap')]:
 rr=[r for r in rows if r['version']==version and r['recipe']==recipe and r['fraction']==frac];assert len(rr)==3
 sm={k:rr[0][k] for k in ('version','recipe','fraction','n_train_mean','n_train_min','n_train_max','n_eval','protocol','eval_set')}
 sm['n_seeds']=3
 for k in ('r2_cb','mae','r2_jet','r2_nonjet','r2_ilo'):
  if all(k in r for r in rr):sm[k]=float(np.mean([r[k] for r in rr]));sm[k+'_sd']=float(np.std([r[k] for r in rr],ddof=1))
 summary.append(sm)
report=dict(seed_rows=rows,summary=summary,full261=full52,common_case_ids=sorted(common),validation_max_abs=max(validation),sources=sources,metadata=dict(metric_space='physical Pa',aggregation='case-balanced',seeds=list(SEEDS),checkpoint='ckpt_best by configured training-loss selection',ensemble=False,v51_total_units=170,v52_total_units=261,v52_fold_train_counts=train52,comparison='same 136 case IDs; CV3/CV5 and geometry revision differ; not pure data intervention'))
(OUT/'source_data.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
for name,rs in [('source_data.csv',rows),('summary.csv',summary),('case_metrics.csv',cases_out),('v52_full261.csv',full52)]:
 keys=list(dict.fromkeys(k for r in rs for k in r))
 with (OUT/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rs)
print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
