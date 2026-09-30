"""Resource/AMP probe on one TRAIN case only, before full-cache completion."""
import hashlib
import json
import os
from pathlib import Path
import time

import torch

from training_wss_min.joint_cycle import case_weighted_loss, move
from training_wss_min.joint_cycle_data import JointCycleDataset, collate_joint_cycle
from training_wss_min.joint_cycle_model import build_joint_cycle_model

ROOT = Path('/public/newhome/cy/Digital_twin/GNN')
EXP = ROOT/'training_wss_min/experiments/joint_cycle_v52_20260929'
CACHE = EXP/'data_audit_smoke'
torch.set_num_threads(4)
torch.manual_seed(1234)
ds = JointCycleDataset(CACHE, case_ids=['AAA/ruputer/DING_JUN_FENG'],
                      support_n=5000, query_n=512, allow_partial_stats=True)
batch = move(collate_joint_cycle([ds[0] for _ in range(4)]), 'cuda')
stats = ds.stats
model = build_joint_cycle_model('J1', 27, {'velocity':20,'pressure':20,'wss':27},
    seed=1234, model_options={
        'geometry_config_path': str(ROOT/'training_wss_min/configs/wss_v52_phys_20260926/X5Dcap_asym2_s1234.json'),
        'geometry':{'local_branch_directional':False},'token_count':128,'token_hidden':64,
        'token_heads':4,'head_hidden':128,'bc_dim':stats['bc_dims'],
        'activation_checkpointing':False}).cuda()
optimizer = torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
scaler = torch.amp.GradScaler('cuda',init_scale=1024.)
rows=[]
for step in range(3):
    torch.cuda.synchronize(); start=time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.float16):
        e=model.encode(**{f'support_{k}':batch['support'][k] for k in ('pos','x','batch')},
            unit_ids=batch['unit_ids'],epoch=0,global_seed=1234,evaluation=False)
        state=model.prepare_phase(e,batch['phase'],batch['bc'],phase_chunk_size=8)
        losses={}
        for task,q in batch['queries'].items():
            p=model.decode(e,task,q['pos'],q['x'],q['batch'],phase_state=state,phase_chunk_size=8)
            losses[task]=case_weighted_loss(p,q['y'],q['weight'],q['batch'])
        loss=torch.stack(list(losses.values())).mean()
    scaler.scale(loss).backward();scaler.unscale_(optimizer)
    gn=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    previous=scaler.get_scale();scaler.step(optimizer);scaler.update()
    assert scaler.get_scale()>=previous
    torch.cuda.synchronize()
    row={'step':step,'seconds':time.perf_counter()-start,'losses':{k:float(v) for k,v in losses.items()},
         'grad_norm':float(gn),'peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,
         'peak_reserved_mib':torch.cuda.max_memory_reserved()/1024**2}
    print(json.dumps(row),flush=True);rows.append(row)
result={'passed':True,'only_train_case':'AAA/ruputer/DING_JUN_FENG','repeated_cases_for_memory':4,
        'seed':1234,'independent_experiment_result':False,'slurm_job_id':os.environ.get('SLURM_JOB_ID'),
        'steps':rows,'code_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (ROOT/'training_wss_min').glob('joint_cycle*.py')}}
(EXP/f'gpu_probe_{os.environ["SLURM_JOB_ID"]}.json').write_text(json.dumps(result,indent=2)+'\n')
