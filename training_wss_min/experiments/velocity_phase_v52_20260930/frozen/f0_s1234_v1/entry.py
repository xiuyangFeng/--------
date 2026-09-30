import hashlib,json,os,subprocess,sys,time
from pathlib import Path
root=Path(__file__).resolve().parent
def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)
def verify_hashes(values):
    for path,expected in values.items():
        if sha(path)!=expected:raise RuntimeError('Immutable experiment input changed: '+path)
verify_hashes(read(root/'fingerprints.json'))
configs=read(root/'configs.json');cfg0=read(configs[0]);mode=sys.argv[1]
from training_wss_min.velocity_phase_data import validate_geometry_manifest
if mode=='geometry':
    target=Path(cfg0['data']['geometry_root'])
    if target.exists() and any(target.iterdir()):raise FileExistsError('refuse geometry overwrite '+str(target))
    cmd=[sys.executable,'-u','-m','training_wss_min.velocity_phase_data','prepare',
         '--cache-root',cfg0['data']['cache_root'],'--geometry-root',str(target),'--workers','4']
    print('LAUNCH',json.dumps(cmd),flush=True);subprocess.run(cmd,cwd=root,check=True)
    validate_geometry_manifest(target,Path(cfg0['data']['cache_root']),require_complete=True,verify_files=True)
    paths=[p for p in sorted(target.rglob('*')) if p.is_file()]
    write(root/'geometry_gate.json',{'status':'passed','job_id':os.environ.get('SLURM_JOB_ID'),
        'completed_at':time.time(),'hashes':{str(p):sha(p) for p in paths}})
    sys.exit(0)
geometry=read(root/'geometry_gate.json')
if geometry['status']!='passed':raise RuntimeError('geometry gate not passed')
verify_hashes(geometry['hashes'])
if mode=='smoke':
    target=Path(cfg0['train']['direction_lambda_path'])
    if target.exists():raise FileExistsError('refuse lambda calibration overwrite '+str(target))
    cmd=[sys.executable,'-u','-m','training_wss_min.velocity_phase','--config',configs[0],'--probe-direction-lambda']
    print('LAUNCH',json.dumps(cmd),flush=True);subprocess.run(cmd,cwd=root,check=True)
    probe=read(target)
    if (probe.get('status')!='passed' or probe.get('partition')!='train' or probe.get('seed')!=1234
        or probe.get('n_batches')!=32 or not 1e-4<=probe.get('direction_lambda',0)<=1
        or probe.get('config_sha256')!=sha(configs[0]) or probe.get('stats_sha256')!=cfg0['data']['stats_sha256']):
        raise ValueError('invalid training-only direction calibration')
    write(root/'calibration_gate.json',{'status':'passed','hashes':{str(target):sha(target)},'job_id':os.environ.get('SLURM_JOB_ID')})
elif mode=='train':
    gate=read(root/'smoke_gate.json')
    if gate['status']!='passed' or {a['arm'] for a in gate['arms']}!={read(c)['arm'] for c in configs}:
        raise RuntimeError('all eight GPU smoke arms must pass')
else:raise ValueError('unknown mode '+mode)
verify_hashes(read(root/'calibration_gate.json')['hashes'])
selected=configs if mode=='smoke' else [configs[int(os.environ['SLURM_ARRAY_TASK_ID'])]]
passed=[]
for filename in selected:
    cfg=read(filename);run=Path(cfg['out_dir'])
    cmd=[sys.executable,'-u','-m','training_wss_min.velocity_phase','--config',filename]
    if mode=='smoke':
        cmd.append('--smoke');run=Path(cfg['experiment_dir'])/'smoke'/(os.environ['SLURM_JOB_ID']+'_'+cfg['arm'])
    else:
        if run.exists() and any(run.iterdir()):raise FileExistsError('refuse formal run overwrite '+str(run))
        run.mkdir(parents=True,exist_ok=True)
        write(run/'execution.json',{'status':'running','job_id':os.environ.get('SLURM_JOB_ID'),
            'array_task_id':os.environ.get('SLURM_ARRAY_TASK_ID'),'started_at':time.time(),'freeze':str(root)})
    print('LAUNCH',json.dumps(cmd),flush=True)
    done=subprocess.run(cmd,cwd=root)
    if mode=='train':write(run/'execution.json',{'status':'completed' if done.returncode==0 else 'failed',
        'job_id':os.environ.get('SLURM_JOB_ID'),'array_task_id':os.environ.get('SLURM_ARRAY_TASK_ID'),
        'returncode':done.returncode,'ended_at':time.time(),'freeze':str(root)})
    done.check_returncode()
    if mode=='smoke':
        metric=read(run/'metrics.json')
        if (metric.get('status')!='smoke_passed' or metric.get('partition')!='train' or metric.get('phase_count')!=80
            or metric.get('seed')!=1234 or metric.get('arm')!=cfg['arm']):raise ValueError('invalid smoke '+cfg['arm'])
        passed.append({'arm':cfg['arm'],'run':str(run),'config_sha256':sha(filename),
                       'parameter_count':metric['parameter_count'],'peak_cuda_memory_bytes':metric.get('peak_cuda_memory_bytes')})
if mode=='smoke':write(root/'smoke_gate.json',{'status':'passed','job_id':os.environ['SLURM_JOB_ID'],
    'completed_at':time.time(),'arms':passed,'geometry_gate_sha256':sha(root/'geometry_gate.json'),
    'calibration_gate_sha256':sha(root/'calibration_gate.json')})
