"""Read-only audit of existing velocity-phase artifacts; no model construction/inference."""
import datetime, hashlib, json, math, os, platform, time
from pathlib import Path
import numpy as np
import torch

ROOT=Path('/public/newhome/cy/Digital_twin/GNN')
EXP=ROOT/'training_wss_min/experiments/velocity_phase_v52_20260930'
OUT=EXP/'analysis_20260930'
SUB=json.loads((EXP/'submission.json').read_text())
FREEZE=Path(SUB['freeze'])
U0=ROOT/'training_wss_min/runs/joint_cycle_round2_v52_20260929/U0_f0_s1234'
ARMS=list(SUB['run_paths'])
errors=[]
def read(p): return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''): h.update(b)
 return h.hexdigest()
def check(condition,message):
 if not condition: errors.append(message)
 return bool(condition)
def finite_tree(x,path=''):
 if isinstance(x,float): return [] if math.isfinite(x) else [path]
 if isinstance(x,dict): return sum((finite_tree(v,path+'.'+k) for k,v in x.items()),[])
 if isinstance(x,list): return sum((finite_tree(v,path+'['+str(i)+']') for i,v in enumerate(x)),[])
 return []
def audit_hashes(mapping,name):
 bad=[p for p,h in mapping.items() if not Path(p).is_file() or sha(p)!=h]
 check(not bad,name+': changed hashes '+str(bad));return {'count':len(mapping),'changed':bad,'status':'passed' if not bad else 'failed'}
def diff(a,b,p=''):
 result={}
 for k in sorted(set(a)|set(b)):
  name=f'{p}.{k}' if p else k
  if isinstance(a.get(k),dict) and isinstance(b.get(k),dict): result.update(diff(a[k],b[k],name))
  elif a.get(k)!=b.get(k): result[name]={'reference':a.get(k),'candidate':b.get(k)}
 return result
started=time.time()
result={'schema':'velocity_phase_completion_audit_v1','created_at':datetime.datetime.now().astimezone().isoformat(),'host':platform.node(),'slurm_job_id':os.getenv('SLURM_JOB_ID'),'scope':'existing artifacts only; no model construction, training, or inference','runs':{},'hash_audits':{}}
result['hash_audits']['protected_legacy']=audit_hashes(read(EXP/'compatibility_before.json')['protected_sha256'],'protected')
result['hash_audits']['frozen']=audit_hashes(read(FREEZE/'fingerprints.json'),'freeze')
result['hash_audits']['source']=audit_hashes({str(FREEZE/k):v for k,v in read(FREEZE/'source_manifest.json').items()},'source')
for name in ['geometry_gate','calibration_gate','cache_validation']:
 gate=read(FREEZE/(name+'.json'));check(gate['status']=='passed',name+' status');result['hash_audits'][name]=audit_hashes(gate['hashes'],name)
smoke=read(FREEZE/'smoke_gate.json')
check(smoke['status']=='passed' and {x['arm'] for x in smoke['arms']}==set(ARMS),'smoke gate arms/status')
for name in ['geometry_gate','calibration_gate']:
 check(smoke[name+'_sha256']==sha(FREEZE/(name+'.json')),'smoke '+name+' hash')
cfg0=read(Path(SUB['config_paths'][0]));split=read(cfg0['data']['split_path']);train=split['train_cases'];test=split['test_cases'];cache=Path(cfg0['data']['cache_root'])
check(len(train)==206 and len(test)==55 and not set(train)&set(test),'split sizes/disjointness')
def patient(cid): return '/'.join(cid.split('/')[:2]) if cid.startswith('ILO/') else cid
patient_overlap=sorted(set(map(patient,train))&set(map(patient,test)))
check(not patient_overlap,'ILO patient overlap')
result['data_contract']={'train_units':len(train),'development_holdout_units':len(test),'train_patient_groups':len(set(map(patient,train))),'holdout_patient_groups':len(set(map(patient,test))),'patient_overlap':patient_overlap,'split_sha256':sha(cfg0['data']['split_path']),'stats_sha256':sha(cfg0['data']['stats_path']),'U0_reference':str(U0)}
tail=read(cfg0['data']['tail_stats_path']);cal=read(cfg0['train']['direction_lambda_path'])
check(tail['train_ids']==train and tail['heldout_labels_read'] is False and len(tail['q80_m_s'])==80,'tail train-only contract')
probe_ids=set(sum((r['unit_ids'] for r in cal['records']),[]))
check(probe_ids<=set(train) and cal['partition']=='train' and cal['n_batches']==32 and cal['optimizer_updates']==0,'D1 train-only calibration')
result['calibration']={'direction_lambda':cal['direction_lambda'],'train_probe_batches':cal['n_batches'],'probe_units':len(probe_ids),'heldout_probe_units':sorted(probe_ids&set(test)),'tail_frames':len(tail['q80_m_s'])}
u0cfg=read(U0/'config.json');u0pc=[json.loads(s) for s in (U0/'per_case.jsonl').read_text().splitlines()]
check([x['unit_id'] for x in u0pc]==test,'U0 unit order')
allcfg={}
for arm in ARMS:
 print('audit_run_start',arm,flush=True)
 run=Path(SUB['run_paths'][arm]);cfg=read(run/'config.json');allcfg[arm]=cfg;prov=read(run/'provenance.json');met=read(run/'metrics.json');done=read(run/'training_complete.json');execution=read(run/'execution.json')
 history=[json.loads(s) for s in (run/'history.jsonl').read_text().splitlines()]
 pc=[json.loads(s) for s in (run/'per_case.jsonl').read_text().splitlines()]
 runerrors=len(errors)
 check(len(history)==150 and [x['epoch'] for x in history]==list(range(1,151)),arm+' epoch continuity')
 check([x['step'] for x in history]==list(range(52,7801,52)),arm+' step continuity')
 nf=finite_tree(history)+finite_tree(pc)+finite_tree(met);check(not nf,arm+' nonfinite metrics/history '+str(nf[:5]))
 check(done['epochs']==150 and done['steps']==7800 and done['smoke'] is False,arm+' training complete')
 check(execution['status']=='completed' and execution['returncode']==0,arm+' execution')
 check(cfg==read(prov['config_path']),arm+' run config semantics differs frozen')
 check(prov['config_sha256']==sha(prov['config_path']),arm+' config sha')
 check(prov['stats_sha256']==sha(cfg['data']['stats_path'])==cfg['data']['stats_sha256'],arm+' stats sha')
 check(cfg['data']['split_sha256']==sha(cfg['data']['split_path']),arm+' split sha')
 check(prov['geometry_audit']['train_ids']==train and prov['geometry_audit']['test_ids']==test,arm+' geometry split')
 check(prov['geometry_audit']['original_cache_mutated'] is False,arm+' cache mutation declared')
 check(met['selection']=='last' and met['n_units']==55 and met['phase_count']==80 and met['partition']=='test' and met['status']=='complete',arm+' evaluation contract')
 check(cfg['train']['selection']=='last' and cfg['seed']==1234 and cfg['train']['epochs']==150,arm+' selection budget seed')
 check([x['unit_id'] for x in pc]==test,arm+' evaluated units')
 check(all(len(x['tasks']['velocity']['per_frame'])==80 for x in pc),arm+' metrics phases')
 code_hash=audit_hashes(prov['code'],arm+' run source');check(code_hash['count']==10,arm+' run source count')
 checkpoint=torch.load(run/'ckpt_last.pt',map_location='cpu',weights_only=False)
 check(checkpoint['epoch']==149 and checkpoint['step']==7800 and checkpoint['config_sha256']==prov['config_sha256'],arm+' checkpoint identity')
 modelfinite=all(bool(torch.isfinite(v).all()) for v in checkpoint['model'].values() if v.is_floating_point());check(modelfinite,arm+' checkpoint nonfinite')
 ckptinfo={k:checkpoint[k] for k in ['schema','epoch','step','config_sha256']};ckptinfo['model_finite']=modelfinite;del checkpoint
 preds=sorted((run/'predictions').glob('*.npz'));check(len(preds)==55,arm+' prediction count')
 prediction_info=[]
 for index,unit in enumerate(test):
  predpath=run/'predictions'/f'{index:03d}.npz'
  with np.load(predpath,allow_pickle=False) as p, np.load(U0/'predictions'/f'{index:03d}.npz',allow_pickle=False) as u:
   a=p['velocity_prediction']; rows=p['velocity_rows']; valid=bool(np.isfinite(a).all());
   check(str(p['unit_id'])==unit and str(u['unit_id'])==unit,arm+' prediction unit '+unit)
   check(a.shape==(len(rows),80,3) and valid,arm+' prediction shape/finite '+unit)
   same_u0=np.array_equal(rows,u['velocity_rows']);same_cache=np.array_equal(rows,np.load(cache/'cases'/unit/'eval_volume_rows.npy'))
   check(same_u0 and same_cache,arm+' prediction row mismatch '+unit)
   prediction_info.append({'unit_id':unit,'shape':list(a.shape),'finite':valid,'same_rows_U0':same_u0,'same_rows_cache':same_cache,'rows_sha256':hashlib.sha256(rows.tobytes()).hexdigest()})
  print('audit_prediction_complete',arm,index+1,unit,flush=True)
 result['runs'][arm]={'status':'passed' if len(errors)==runerrors else 'failed','run':str(run),'slurm_job_id':execution['job_id'],'array_task_id':execution['array_task_id'],'history_rows':len(history),'last_epoch':history[-1]['epoch'],'last_step':history[-1]['step'],'all_history_metrics_finite':not nf,'checkpoint':ckptinfo,'evaluation_units':len(pc),'prediction_files':len(preds),'predictions':prediction_info,'parameter_count':met['parameter_count'],'config_sha256':prov['config_sha256'],'config_differences_from_U0':diff(u0cfg,cfg),'train_seconds':done['seconds'],'last_common_z_mse':history[-1]['objective']['z_mse'],'D1_empty_case_phase_total':sum(x.get('direction_coverage',{}).get('empty_case_phases',0) for x in history) if arm=='D1' else None}
 print('audit_run_complete',arm,result['runs'][arm]['status'],flush=True)
result['pair_config_differences']={a+'_vs_'+b:diff(allcfg[b],allcfg[a]) for a,b in [('A1','A0'),('G01','G00'),('G10','G00'),('G11','G10'),('G11','G01')]}
result['interpretation_limits']=['Single seed 1234 on a repeatedly exposed development holdout; not an independent confirmatory test.','U0 is a historical round2 reference on identical split/statistics/query rows and budget; no contemporaneous unchanged U0 retraining.','A1 versus A0 isolates residual coordinate choice at identical parameter count; A0 versus U0 also adds frame inputs and 22659 parameters.','G arms add both anchor information and capacity versus A0; graph versus pointwise and phase versus static must use within-G contrasts.','G active additional parameter mismatch is approximately 0.96%, not exact equality.','Wall-clock training/inference observations share cluster hardware; they are not a controlled throughput benchmark.']
result['errors']=errors;result['status']='passed' if not errors else 'failed';result['seconds']=time.time()-started
(OUT/'completion_audit.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
print(json.dumps({'status':result['status'],'errors':errors,'seconds':result['seconds'],'runs':{k:v['status'] for k,v in result['runs'].items()}},ensure_ascii=False),flush=True)
raise SystemExit(0 if not errors else 1)
