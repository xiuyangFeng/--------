import hashlib,json,os,subprocess,sys,time
from pathlib import Path
root=Path(__file__).resolve().parent
for name,expected in json.loads((root/'fingerprints.json').read_text()).items():
    if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=expected:
        raise RuntimeError('Frozen experiment input changed: '+name)
configs=json.loads((root/'configs.json').read_text())
mode=sys.argv[1]
if mode not in ('smoke','train'):raise ValueError(mode)
if mode=='train':
    gate=json.loads((root/'smoke_gate.json').read_text())
    if gate['status']!='passed' or len(gate['arms'])!=10:
        raise RuntimeError('All ten GPU smoke arms must pass before training')
selected=configs if mode=='smoke' else [configs[int(os.environ['SLURM_ARRAY_TASK_ID'])]]
passed=[]
for config in selected:
    cfg=json.loads(Path(config).read_text())
    cmd=[sys.executable,'-u','-m','training_wss_min.joint_cycle_round2','--config',config]
    run=Path(cfg['out_dir'])
    if mode=='smoke':
        cmd.append('--smoke')
        run=Path(cfg['experiment_dir'])/'smoke'/(os.environ['SLURM_JOB_ID']+'_'+cfg['arm'])
    else:
        if run.exists() and any(run.iterdir()):raise FileExistsError(str(run))
        run.mkdir(parents=True,exist_ok=True)
        (run/'execution.json').write_text(json.dumps({'status':'running','job_id':os.environ.get('SLURM_JOB_ID'),
          'array_task_id':os.environ.get('SLURM_ARRAY_TASK_ID'),'started_at':time.time(),'freeze':str(root)}))
    print('LAUNCH',json.dumps(cmd),flush=True)
    done=subprocess.run(cmd,cwd=root)
    if mode=='train':
        (run/'execution.json').write_text(json.dumps({'status':'completed' if done.returncode==0 else 'failed',
          'job_id':os.environ.get('SLURM_JOB_ID'),'array_task_id':os.environ.get('SLURM_ARRAY_TASK_ID'),
          'returncode':done.returncode,'ended_at':time.time(),'freeze':str(root)}))
    done.check_returncode()
    if mode=='smoke':
        metric=json.loads((run/'metrics.json').read_text())
        if (metric['status']!='smoke_passed' or metric['partition']!='train' or metric['phase_count']!=80
            or metric['seed']!=1234 or metric['arm']!=cfg['arm']):
            raise RuntimeError('Invalid smoke result: '+cfg['arm'])
        passed.append({'arm':cfg['arm'],'run':str(run),'config_sha256':hashlib.sha256(Path(config).read_bytes()).hexdigest(),
                       'parameter_count':metric['parameter_count']})
if mode=='smoke':
    if len(passed)!=10:raise RuntimeError('Incomplete smoke gate')
    (root/'smoke_gate.json').write_text(json.dumps({'status':'passed','job_id':os.environ['SLURM_JOB_ID'],
      'completed_at':time.time(),'arms':passed},indent=2)+'\n')
