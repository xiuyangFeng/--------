#!/usr/bin/env python3
"""Freeze eight velocity experiments, then enqueue them without changing old jobs.

Preparation copies the complete static repository import closure. Formal jobs
run only from that snapshot, with PYTHONPATH/PYTHONHOME cleared. New geometry and
training-only calibration are audited as subsequent immutable stage artifacts.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
NAME = "velocity_phase_v52_20260930"
EXP = ROOT / "training_wss_min/experiments" / NAME
CONFIGS = ROOT / "training_wss_min/configs" / NAME
OLD_EXP = ROOT / "training_wss_min/experiments/joint_cycle_v52_20260929"
PYTHON = "/public/newhome/cy/.conda/envs/GNN/bin/python"
SBATCH = "/public/slurm/bin/sbatch"
ARMS = ("D1", "D2", "A0", "A1", "G00", "G01", "G10", "G11")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, data):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def check_tag(tag):
    if not re.fullmatch(r"[a-z0-9_]+", tag):
        raise ValueError("tag requires lowercase letters, digits, underscores")
    return tag


def import_closure(entries):
    """Resolve static intra-repository imports, including imports inside functions."""
    pending, seen = [ROOT / p for p in entries], set()

    def resolve(parts):
        stem = ROOT.joinpath(*parts)
        for target in (stem.with_suffix(".py"), stem / "__init__.py"):
            if target.is_file():
                pending.append(target)
                break

    while pending:
        path = pending.pop().resolve()
        if path in seen:
            continue
        if not path.is_file() or ROOT not in path.parents:
            raise FileNotFoundError(path)
        seen.add(path)
        relative = path.relative_to(ROOT)
        package = list(relative.parts[:-1])
        for n in range(1, len(package) + 1):
            init = ROOT.joinpath(*package[:n], "__init__.py")
            if init.is_file() and init not in seen:
                pending.append(init)
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    resolve(alias.name.split("."))
            elif isinstance(node, ast.ImportFrom):
                prefix = package[:len(package) - node.level + 1] if node.level else []
                module = prefix + (node.module.split(".") if node.module else [])
                resolve(module)
                for alias in node.names:
                    if alias.name != "*":
                        resolve(module + alias.name.split("."))
    return sorted(seen)


def verify_cache(matrix):
    data = read_json(CONFIGS / matrix["arms"][0]["config"])["data"]
    cache = Path(data["cache_root"])
    split_path, stats_path = Path(data["split_path"]), Path(data["stats_path"])
    split, stats = read_json(split_path), read_json(stats_path)
    manifest, audit = read_json(cache / "manifest.json"), read_json(cache / "audit.json")
    original = read_json(OLD_EXP / "data_audit_summary.json")
    if digest(split_path) != matrix["split_sha256"] or digest(stats_path) != matrix["stats_sha256"]:
        raise ValueError("registered split/statistics checksum changed")
    train, test = set(split["train_cases"]), set(split["test_cases"])
    if len(train) != 206 or len(test) != 55 or train & test:
        raise ValueError("expected unchanged disjoint 206/55 research partition")
    if not manifest["complete"] or set(manifest["case_ids"]) != train | test:
        raise ValueError("expected unchanged 261-unit cache, not full-data cache")
    if audit["errors"] or original["status"] != "complete" or original["patient_group_intersection"] != 0:
        raise ValueError("original cache and patient-group audits must pass")
    if (not stats["complete_train_partition"] or set(stats["train_ids"]) != train
            or stats["split_sha256"] != matrix["split_sha256"]):
        raise ValueError("normalization must use only complete206 train partition")
    for cid in sorted(train | test):
        directory = cache / "cases" / cid
        files = ["support_pos.npy", "support_x.npy", "phase.npy", "bc.npy", "metadata.json"]
        files += [f"{mode}_volume_{field}.npy" for mode in ("train", "eval")
                  for field in ("pos", "x", "rows", "weight")]
        files += [f"{mode}_velocity.npy" for mode in ("train", "eval")]
        if any(not (directory / f).is_file() for f in files):
            raise FileNotFoundError(f"incomplete existing cache {cid}")
    paths = (split_path, stats_path, cache / "manifest.json", cache / "audit.json",
             OLD_EXP / "data_audit_summary.json")
    return {"status": "passed", "cache_root": str(cache), "train_count": 206,
            "development_holdout_count": 55, "case_count": 261,
            "rebuild_performed": False, "hashes": {str(p): digest(p) for p in paths}}


PARAMETER_ENTRY = '''import json
from pathlib import Path
import torch
from training_wss_min.velocity_phase import build,validate_config
root=Path(__file__).resolve().parent
torch.set_num_threads(2)
arms={}
for filename in json.loads((root/'configs.json').read_text()):
    cfg=json.loads(Path(filename).read_text()); validate_config(cfg)
    model=build(cfg,json.loads(Path(cfg['data']['stats_path']).read_text()))
    contract=getattr(model,'velocity_contract',None)
    if contract is None: contract=getattr(model,'velocity_phase_contract',None)
    if not isinstance(contract,dict): raise ValueError('missing velocity model parameter contract')
    total=sum(p.numel() for p in model.parameters() if p.requires_grad)
    added=contract.get('added_parameters')
    if not isinstance(added,int): raise ValueError('missing active added-parameter count')
    arms[cfg['arm']]={'total_parameters':total,'added_parameters':added,
                      'graph_added_parameters':contract.get('graph_added_parameters'),'module_contract':contract}
    del model
pairs=[]
for group in [('A0','A1'),('G00','G01','G10','G11')]:
    for i,a in enumerate(group):
        for b in group[i+1:]:
            key='graph_added_parameters' if a.startswith('G') else 'added_parameters'
            x,y=arms[a][key],arms[b][key]
            if not isinstance(x,int) or not isinstance(y,int):raise ValueError('missing active '+key)
            delta=abs(x-y)/max(x,y,1)
            if delta>.05:raise ValueError('active parameter mismatch '+a+'/'+b)
            pairs.append({'arms':[a,b],'relative_difference':delta,'passed':True})
out={'status':'passed','arms':arms,'pairs':pairs,'scope':'CPU construction and active parameter gate; GPU smoke pending'}
(root/'parameter_validation.json').write_text(json.dumps(out,indent=2)+'\\n')
print(json.dumps(out,indent=2))
'''


ENTRY = '''import hashlib,json,os,subprocess,sys,time
from pathlib import Path
root=Path(__file__).resolve().parent
def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(data,indent=2)+'\\n');tmp.replace(path)
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
'''


def prepare(tag):
    freeze = EXP / "frozen" / check_tag(tag)
    if freeze.exists():
        raise FileExistsError(f"refusing snapshot overwrite {freeze}")
    matrix = read_json(CONFIGS / "matrix.json")
    if tuple(a["arm"] for a in matrix["arms"]) != ARMS:
        raise ValueError("matrix must contain the eight registered ordered arms")
    validation = verify_cache(matrix)
    for item in matrix["arms"]:
        source = CONFIGS / item["config"]
        if digest(source) != item["config_sha256"]:
            raise ValueError(f"registered config checksum changed {source}")
        cfg = read_json(source)
        if (cfg["schema"] != "velocity_phase_v52_v1" or cfg["seed"] != 1234 or cfg["data"]["fold"] != 0
                or cfg["train"]["epochs"] != 150 or cfg["train"]["selection"] != "last"
                or cfg["data"]["split_sha256"] != matrix["split_sha256"]
                or cfg["data"]["stats_sha256"] != matrix["stats_sha256"]):
            raise ValueError(f"fixed research contract changed for {cfg['arm']}")
        run = Path(cfg["out_dir"])
        if run.exists() and any(run.iterdir()):
            raise FileExistsError(f"formal run already exists {run}")
    sources = import_closure(["training_wss_min/velocity_phase.py",
                              "training_wss_min/velocity_phase_data.py"])
    freeze.mkdir(parents=True)
    for folder in ("configs", "provenance"):
        (freeze / folder).mkdir()
    (EXP / "logs").mkdir(exist_ok=True)
    for source in sources:
        target = freeze / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    shutil.copy2(Path(__file__), freeze / "submit_snapshot.py")
    for filename in ("geometry_x5d.json", "PREREG.md"):
        shutil.copy2(CONFIGS / filename, freeze / "configs" / filename)
    for filename in validation["hashes"]:
        shutil.copy2(filename, freeze / "provenance" / Path(filename).name)
    write_json(freeze / "cache_validation.json", validation)
    write_json(freeze / "source_manifest.json", {str(p.relative_to(ROOT)): digest(p) for p in sources})
    frozen_configs, frozen_arms = [], []
    for item in matrix["arms"]:
        cfg = read_json(CONFIGS / item["config"])
        cfg["model"]["geometry_config_path"] = str(freeze / "configs/geometry_x5d.json")
        target = freeze / "configs" / item["config"]
        write_json(target, cfg)
        frozen_configs.append(str(target))
        frozen_arms.append({**item, "config": str(target), "config_sha256": digest(target)})
    write_json(freeze / "configs.json", frozen_configs)
    (freeze / "parameter_entry.py").write_text(PARAMETER_ENTRY)
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME", "GNN_JOINT_SPLIT_PATH")}
    env.update(GNN_JOINT_SOURCE_ROOT=str(ROOT), OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONNOUSERSITE="1")
    result = subprocess.run([PYTHON, "-u", "parameter_entry.py"], cwd=freeze, env=env, capture_output=True, text=True)
    (freeze / "parameter_validation.log").write_text(result.stdout)
    (freeze / "parameter_validation.err").write_text(result.stderr)
    result.check_returncode()
    matrix = {**matrix, "arms": frozen_arms, "geometry_config": str(freeze / "configs/geometry_x5d.json"),
              "parameter_validation": read_json(freeze / "parameter_validation.json")}
    write_json(freeze / "configs/matrix.json", matrix)
    (freeze / "entry.py").write_text(ENTRY)
    environment = f'''set -euo pipefail
unset PYTHONPATH PYTHONHOME GNN_JOINT_SPLIT_PATH
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export GNN_JOINT_SOURCE_ROOT={shlex.quote(str(ROOT))}
cd {shlex.quote(str(freeze))}
date -Is
hostname
id
'''
    cpu = "#!/bin/bash\n#SBATCH --partition=CPU\n#SBATCH --nodelist=node03\n#SBATCH --cpus-per-task=8\n#SBATCH --mem=32G\n"
    gpu = "#!/bin/bash\n#SBATCH --partition=GPU\n#SBATCH --gres=gpu:4090:1\n#SBATCH --cpus-per-task=8\n#SBATCH --mem=32G\n"
    (freeze / "geometry.slurm").write_text(cpu + environment + f"{shlex.quote(PYTHON)} -u entry.py geometry\n")
    for mode in ("smoke", "train"):
        (freeze / f"{mode}.slurm").write_text(gpu + environment +
            "nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total --format=csv\n" +
            f"{shlex.quote(PYTHON)} -u entry.py {mode}\n")
    for original, target in (("report_velocity_phase_v52.py", "report.py"),
                              ("report_joint_cycle_v52.py", "report_joint_cycle_v52.py"),
                              ("report_joint_cycle_round2_v52.py", "report_joint_cycle_round2_v52.py")):
        shutil.copy2(ROOT / "training_wss_min/tools" / original, freeze / target)
    (freeze / "report.slurm").write_text(cpu + environment +
        f"{shlex.quote(PYTHON)} -u report.py --config-dir {shlex.quote(str(freeze / 'configs'))} --experiment-dir {shlex.quote(str(EXP))}\n")
    hashes = {str(p): digest(p) for p in sorted(freeze.rglob("*")) if p.is_file() and "__pycache__" not in p.parts}
    hashes.update(validation["hashes"])
    write_json(freeze / "fingerprints.json", hashes)
    record = {"schema": "velocity_phase_v52_submission_v1", "status": "prepared",
              "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "freeze": str(freeze),
              "config_paths": frozen_configs, "run_paths": {a["arm"]: a["out_dir"] for a in frozen_arms},
              "seed": 1234, "fold": 0, "epochs": 150, "train_count": 206, "development_holdout_count": 55,
              "gpu_models": 8, "max_parallel": 4, "source_file_count": len(sources),
              "parameter_validation": str(freeze / "parameter_validation.json"),
              "cache_validation": str(freeze / "cache_validation.json"),
              "geometry_gate": str(freeze / "geometry_gate.json"),
              "calibration_gate": str(freeze / "calibration_gate.json"),
              "smoke_gate": str(freeze / "smoke_gate.json")}
    write_json(freeze / "submission.json", record)
    return freeze, record


def submit_prepared(tag):
    freeze = EXP / "frozen" / check_tag(tag)
    for path, expected in read_json(freeze / "fingerprints.json").items():
        if digest(path) != expected:
            raise RuntimeError(f"frozen input changed: {path}")
    record = read_json(freeze / "submission.json")
    if record["status"] != "prepared":
        raise ValueError("snapshot already submitted/attempted; refuse duplicate submission")
    for path in record["run_paths"].values():
        if Path(path).exists() and any(Path(path).iterdir()):
            raise FileExistsError(path)
    with (freeze / "submission.lock").open("x") as handle:
        handle.write(str(os.getpid()) + "\n")

    def save():
        write_json(freeze / "submission.json", record)
        write_json(EXP / "submission.json", record)

    def submit(script, name, duration, dependency=None, array=None):
        command = [SBATCH, "--parsable", "--kill-on-invalid-dep=yes", f"--job-name={name}", f"--time={duration}",
                   "--output=" + str(EXP / ("logs/train_%A_%a.log" if array else f"logs/{script}_%j.log")),
                   "--error=" + str(EXP / ("logs/train_%A_%a.err" if array else f"logs/{script}_%j.err"))]
        if dependency:
            command.append("--dependency=" + dependency)
        if array:
            command.append("--array=" + array)
        command.append(str(freeze / f"{script}.slurm"))
        result = subprocess.run(command, text=True, capture_output=True, check=True)
        job = result.stdout.strip().split(";")[0]
        if not job.isdigit():
            raise RuntimeError(f"unrecognized sbatch response {result.stdout!r}")
        return job

    try:
        record["geometry_job"] = submit("geometry", "vphase52_geometry", "08:00:00")
        record["status"] = "geometry_submitted"; save()
        record["smoke_job"] = submit("smoke", "vphase52_gate", "04:00:00", "afterok:" + record["geometry_job"])
        record["status"] = "smoke_submitted"; save()
        record["batch1_job"] = submit("train", "vphase52_batch1", "24:00:00", "afterok:" + record["smoke_job"], "0-3%4")
        record["status"] = "batch1_submitted"; save()
        record["batch2_job"] = submit("train", "vphase52_batch2", "24:00:00",
            "afterany:" + record["batch1_job"] + ",afterok:" + record["smoke_job"], "4-7%4")
        record["array_jobs"] = [record["batch1_job"], record["batch2_job"]]
        record["status"] = "training_queued"; save()
        record["report_job"] = submit("report", "vphase52_report", "00:30:00",
            "afterany:" + ":".join(record[k] for k in ("geometry_job", "smoke_job", "batch1_job", "batch2_job")))
        record["status"] = "submitted"; save()
    except Exception as exc:
        record.update(status="submission_failed", error=repr(exc)); save()
        raise
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", metavar="TAG")
    mode.add_argument("--submit-prepared", metavar="TAG")
    args = parser.parse_args()
    record = prepare(args.prepare_only)[1] if args.prepare_only else submit_prepared(args.submit_prepared)
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
