"""Verify actual Adam state step counts from existing last checkpoints, without inference."""
import collections,datetime,json,os,subprocess
from pathlib import Path
import torch
ROOT=Path('/public/newhome/cy/Digital_twin/GNN');EXP=ROOT/'training_wss_min/experiments/velocity_phase_v52_20260930';OUT=EXP/'analysis_20260930'
s=json.loads((EXP/'submission.json').read_text());result={'checked_at':datetime.datetime.now().astimezone().isoformat(),'job_id':os.getenv('SLURM_JOB_ID'),'runs':{},'status':'passed'}
for arm,p in s['run_paths'].items():
 c=torch.load(Path(p)/'ckpt_last.pt',map_location='cpu',weights_only=False)
 counts=collections.Counter(int(x['step']) for x in c['optimizer']['state'].values() if 'step' in x)
 good=set(counts)=={7800} and c['scheduler']['last_epoch']==7800
 if not good:result['status']='failed'
 result['runs'][arm]={'optimizer_step_counts':dict(counts),'scheduler_last_epoch':c['scheduler']['last_epoch'],'scaler_state':c['scaler'],'status':'passed' if good else 'failed'}
 print(arm,result['runs'][arm],flush=True)
(OUT/'optimizer_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(result['status'],flush=True)
raise SystemExit(0 if result['status']=='passed' else 1)
