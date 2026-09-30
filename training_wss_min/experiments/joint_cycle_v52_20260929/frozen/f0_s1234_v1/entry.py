import hashlib,json,os,subprocess,sys,time
from pathlib import Path
root=Path(__file__).resolve().parent
for name,expected in json.loads((root/'fingerprints.json').read_text()).items():
    if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=expected:
        raise RuntimeError('Frozen experiment input changed: '+name)
configs=json.loads((root/'configs.json').read_text())
mode=sys.argv[1]
selected=configs if mode=='smoke' else [configs[int(os.environ['SLURM_ARRAY_TASK_ID'])]]
for config in selected:
    cmd=[sys.executable,'-u','-m','training_wss_min.joint_cycle','--config',config]
    if mode=='smoke':cmd.append('--smoke')
    print('LAUNCH',json.dumps(cmd),flush=True)
    cfg=json.loads(Path(config).read_text())
    run=Path(cfg['out_dir'])
    if mode=='train':
        run.mkdir(parents=True,exist_ok=True)
        if (run/'metrics.json').exists():raise FileExistsError(str(run))
        (run/'execution.json').write_text(json.dumps({'status':'running','job_id':os.environ.get('SLURM_JOB_ID'),'started_at':time.time()}))
    done=subprocess.run(cmd,cwd=root)
    if mode=='train':
        (run/'execution.json').write_text(json.dumps({'status':'completed' if done.returncode==0 else 'failed','job_id':os.environ.get('SLURM_JOB_ID'),'returncode':done.returncode,'ended_at':time.time()}))
    done.check_returncode()
