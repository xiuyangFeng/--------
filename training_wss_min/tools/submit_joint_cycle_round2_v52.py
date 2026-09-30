#!/usr/bin/env python3
"""Prepare immutable ten-arm snapshots, then submit one-seed GPU jobs.

The existing 261-case cache is verified and reused. No preparation job or
training is launched by --prepare-only. --submit-prepared TAG submits the
already reviewed snapshot without recopying mutable workspace files.
"""
from __future__ import annotations

import argparse
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
EXP = ROOT / "training_wss_min/experiments/joint_cycle_round2_v52_20260929"
CONFIGS = ROOT / "training_wss_min/configs/joint_cycle_round2_v52_20260929"
OLD_EXP = ROOT / "training_wss_min/experiments/joint_cycle_v52_20260929"
PYTHON = "/public/newhome/cy/.conda/envs/GNN/bin/python"
SBATCH = "/public/slurm/bin/sbatch"
ARMS = ("UT0", "UT1", "WT0", "WT1", "U0", "U1", "W0", "W1", "P1", "J2")
SOURCE_FILES = ("__init__.py", "joint_cycle_round2.py", "joint_cycle_round2_model.py",
                "joint_cycle_pcgrad.py", "joint_cycle_round2_metrics.py",
                "joint_cycle.py", "joint_cycle_model.py", "joint_cycle_data.py",
                "baseline_models.py", "local_refinement.py", "surface.py")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def check_tag(tag):
    if not re.fullmatch(r"[a-z0-9_]+", tag):
        raise ValueError("tag must contain only lowercase letters, digits, underscores")
    return tag


def verify_cache(matrix):
    """Check provenance and small metadata; never reread/rebuild CFD arrays."""
    first = read_json(CONFIGS / matrix["arms"][0]["config"])
    data = first["data"]
    cache, split_path, stats_path = (Path(data[k]) for k in ("cache_root", "split_path", "stats_path"))
    split, stats = read_json(split_path), read_json(stats_path)
    manifest, audit = read_json(cache / "manifest.json"), read_json(cache / "audit.json")
    final = read_json(OLD_EXP / "data_audit_summary.json")
    if digest(split_path) != matrix["split_sha256"] or digest(stats_path) != matrix["stats_sha256"]:
        raise ValueError("registered split/statistics checksum mismatch")
    expected_train, expected_test = set(split["train_cases"]), set(split["test_cases"])
    if len(expected_train) != 206 or len(expected_test) != 55 or expected_train & expected_test:
        raise ValueError("incorrect data-unit partition")
    if not manifest["complete"] or set(manifest["case_ids"]) != expected_train | expected_test:
        raise ValueError("cache does not cover all 261 registered units")
    if audit["errors"] or final["status"] != "complete" or final["patient_group_intersection"] != 0:
        raise ValueError("original source/patient grouping audit did not pass")
    if (not stats["complete_train_partition"] or set(stats["train_ids"]) != expected_train
            or stats["split_sha256"] != matrix["split_sha256"]):
        raise ValueError("training-only normalization contract mismatch")
    files = [split_path, stats_path, cache / "manifest.json", cache / "audit.json",
             OLD_EXP / "data_audit_summary.json"]
    required = ["support_pos.npy", "support_x.npy", "phase.npy", "bc.npy", "metadata.json"]
    for mode in ("train", "eval"):
        required += [f"{mode}_{domain}_{field}.npy" for domain in ("volume", "wall")
                     for field in ("pos", "x", "weight", "rows")]
        required += [f"{mode}_{task}.npy" for task in ("velocity", "pressure", "wss")]
    for case_id in sorted(expected_train | expected_test):
        case = cache / "cases" / case_id
        if any(not (case / name).is_file() for name in required):
            raise FileNotFoundError(f"incomplete audited cache: {case_id}")
        if read_json(case / "metadata.json")["canonical_id"] != case_id:
            raise ValueError(f"cached case identity mismatch: {case_id}")
    return {"status": "passed", "cache_root": str(cache), "case_count": 261,
            "train_count": 206, "development_holdout_count": 55,
            "patient_group_intersection": 0, "source_data_job": 16147,
            "rebuild_performed": False, "hashes": {str(p): digest(p) for p in files}}


PARAMETER_ENTRY = '''import json
from pathlib import Path
import torch
from training_wss_min.joint_cycle_round2 import build, validate_config
root=Path(__file__).resolve().parent
torch.set_num_threads(2)
arms={}
for path in json.loads((root/'configs.json').read_text()):
    cfg=json.loads(Path(path).read_text())
    validate_config(cfg)
    model=build(cfg,json.loads(Path(cfg['data']['stats_path']).read_text()))
    contract=getattr(model,'round2_contract',{'module':'none','residual_parameters':0})
    arms[cfg['arm']]={'total_trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
                      'added_parameters':contract['residual_parameters'],'module_contract':contract}
    del model
pairs=[]
for control,treatment in [('UT0','UT1'),('WT0','WT1'),('U0','U1'),('W0','W1')]:
    n0,n1=arms[control]['added_parameters'],arms[treatment]['added_parameters']
    relative=abs(n0-n1)/max(n0,n1)
    if relative>0.05:raise ValueError('Added parameter mismatch: '+control+'/'+treatment)
    pairs.append({'control':control,'treatment':treatment,'control_added_parameters':n0,
                  'treatment_added_parameters':n1,'relative_difference':relative,'passed':True})
out={'status':'passed','device':'cpu','seed':1234,'arms':arms,'pairs':pairs,
     'scope':'parameter/config construction only; GPU smoke still required'}
(root/'parameter_validation.json').write_text(json.dumps(out,indent=2)+'\\n')
print(json.dumps(out,indent=2))
'''


ENTRY = '''import hashlib,json,os,subprocess,sys,time
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
      'completed_at':time.time(),'arms':passed},indent=2)+'\\n')
'''


def prepare(tag):
    freeze = EXP / "frozen" / check_tag(tag)
    if freeze.exists():
        raise FileExistsError(f"refusing to overwrite snapshot {freeze}")
    matrix = read_json(CONFIGS / "matrix.json")
    if tuple(a["arm"] for a in matrix["arms"]) != ARMS:
        raise ValueError("registered matrix must contain exactly ten ordered arms")
    cache_validation = verify_cache(matrix)
    for arm in matrix["arms"]:
        cfg = read_json(CONFIGS / arm["config"])
        if (cfg["seed"] != 1234 or cfg["train"]["epochs"] != 150
                or cfg["data"]["fold"] != 0 or cfg["train"]["selection"] != "last"):
            raise ValueError("additional seeds/folds or changed budgets are not authorized")
        if (cfg["data"]["stats_sha256"] != matrix["stats_sha256"]
                or cfg["data"]["split_sha256"] != matrix["split_sha256"]):
            raise ValueError("inconsistent data fingerprints")
        out = Path(cfg["out_dir"])
        if out.exists() and any(out.iterdir()):
            raise FileExistsError(f"refusing to replace existing formal run {out}")
    (freeze / "training_wss_min").mkdir(parents=True)
    (freeze / "configs").mkdir()
    (freeze / "provenance").mkdir()
    (EXP / "logs").mkdir(exist_ok=True)
    for name in SOURCE_FILES:
        shutil.copy2(ROOT / "training_wss_min" / name, freeze / "training_wss_min" / name)
    shutil.copy2(Path(__file__), freeze / "submit_snapshot.py")
    shutil.copy2(CONFIGS / "geometry_x5d.json", freeze / "configs/geometry_x5d.json")
    shutil.copy2(CONFIGS / "PREREG.md", freeze / "configs/PREREG.md")
    for original in cache_validation["hashes"]:
        shutil.copy2(original, freeze / "provenance" / Path(original).name)
    write_json(freeze / "cache_validation.json", cache_validation)
    frozen_configs, frozen_arms = [], []
    for arm in matrix["arms"]:
        original = CONFIGS / arm["config"]
        cfg = read_json(original)
        cfg["model"]["geometry_config_path"] = str(freeze / "configs/geometry_x5d.json")
        target = freeze / "configs" / original.name
        write_json(target, cfg)
        frozen_configs.append(str(target))
        frozen_arms.append({**arm, "config": str(target), "config_sha256": digest(target)})
    write_json(freeze / "configs.json", frozen_configs)
    (freeze / "parameter_entry.py").write_text(PARAMETER_ENTRY)
    env = {**os.environ, "GNN_JOINT_SOURCE_ROOT": str(ROOT), "OMP_NUM_THREADS": "2",
           "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    completed = subprocess.run([PYTHON, "-u", "parameter_entry.py"], cwd=freeze, env=env,
                               text=True, capture_output=True)
    (freeze / "parameter_validation.log").write_text(completed.stdout)
    (freeze / "parameter_validation.err").write_text(completed.stderr)
    completed.check_returncode()
    params = read_json(freeze / "parameter_validation.json")
    for arm in frozen_arms:
        arm["parameter_validation"] = params["arms"][arm["arm"]]
    matrix = {**matrix, "arms": frozen_arms,
              "parameter_matching": {**matrix["parameter_matching"], "status": "passed_cpu",
                                     "gpu_smoke": "pending", "actual_pairs": params["pairs"]}}
    write_json(freeze / "configs/matrix.json", matrix)
    (freeze / "entry.py").write_text(ENTRY)
    # Slurm controls visible devices; do not override CUDA_VISIBLE_DEVICES.
    common = f'''#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
set -euo pipefail
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export GNN_JOINT_SOURCE_ROOT={shlex.quote(str(ROOT))}
unset GNN_JOINT_SPLIT_PATH
cd {shlex.quote(str(freeze))}
date -Is
hostname
id
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total --format=csv
'''
    (freeze / "smoke.slurm").write_text(common + f"{shlex.quote(PYTHON)} -u entry.py smoke\n")
    (freeze / "train.slurm").write_text(common + f"{shlex.quote(PYTHON)} -u entry.py train\n")
    shutil.copy2(ROOT / "training_wss_min/tools/report_joint_cycle_round2_v52.py", freeze / "report.py")
    shutil.copy2(ROOT / "training_wss_min/tools/report_joint_cycle_v52.py", freeze / "report_joint_cycle_v52.py")
    (freeze / "report.slurm").write_text(f'''#!/bin/bash
#SBATCH --partition=CPU
#SBATCH --nodelist=node03
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
set -euo pipefail
export GNN_JOINT_SOURCE_ROOT={shlex.quote(str(ROOT))}
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd {shlex.quote(str(freeze))}
{shlex.quote(PYTHON)} -u report.py --config-dir {shlex.quote(str(freeze / 'configs'))} --experiment-dir {shlex.quote(str(EXP))}
''')
    fingerprints = {str(p): digest(p) for p in sorted(freeze.rglob("*"))
                    if p.is_file() and "__pycache__" not in p.parts}
    fingerprints.update(cache_validation["hashes"])
    write_json(freeze / "fingerprints.json", fingerprints)
    record = {"schema": "joint_cycle_round2_v52_submission_v1",
              "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "freeze": str(freeze),
              "config_paths": frozen_configs, "run_paths": {a["arm"]: a["out_dir"] for a in frozen_arms},
              "seed": 1234, "fold": 0, "epochs": 150, "gpu_models": 10, "max_parallel": 4,
              "parameter_validation": str(freeze / "parameter_validation.json"),
              "cache_validation": str(freeze / "cache_validation.json"),
              "status": "prepared", "dependency_chain": "audited existing cache -> ten-arm GPU smoke -> array 0-9%4 -> CPU report"}
    write_json(freeze / "submission.json", record)
    return freeze, record


def verify_prepared(freeze):
    for name, expected in read_json(freeze / "fingerprints.json").items():
        if digest(name) != expected:
            raise RuntimeError(f"Frozen input changed: {name}")
    record = read_json(freeze / "submission.json")
    if record["status"] != "prepared" or any(k in record for k in ("smoke_job", "array_job", "report_job")):
        raise ValueError("snapshot already submitted or partially submitted; do not duplicate jobs")
    for name in record["run_paths"].values():
        run = Path(name)
        if run.exists() and any(run.iterdir()):
            raise FileExistsError(f"formal run already exists: {run}")
    return record


def submit_prepared(freeze):
    record = verify_prepared(freeze)
    lock = freeze / "submission.lock"
    with lock.open("x") as handle:
        handle.write(str(os.getpid()) + "\n")

    def submit(extra, script):
        completed = subprocess.run([SBATCH, "--parsable", *extra, str(freeze / script)],
                                   text=True, capture_output=True, check=True)
        job = completed.stdout.strip().split(";")[0]
        if not job.isdigit():
            raise RuntimeError(f"invalid sbatch result: {completed.stdout!r}")
        return job

    def save():
        write_json(freeze / "submission.json", record)
        write_json(EXP / "submission.json", record)

    try:
        record["smoke_job"] = submit(["--job-name=joint52r2_smoke", "--time=02:00:00",
            "--output=" + str(EXP / "logs/smoke_%j.log"),
            "--error=" + str(EXP / "logs/smoke_%j.err")], "smoke.slurm")
        record["status"] = "smoke_submitted"
        save()
        record["array_job"] = submit(["--job-name=joint52r2_s1234", "--time=24:00:00", "--array=0-9%4",
            "--dependency=afterok:" + record["smoke_job"],
            "--output=" + str(EXP / "logs/train_%A_%a.log"),
            "--error=" + str(EXP / "logs/train_%A_%a.err")], "train.slurm")
        record["status"] = "submitted"
        save()
        record["report_job"] = submit(["--job-name=joint52r2_report", "--time=00:30:00",
            "--dependency=afterany:" + record["array_job"],
            "--output=" + str(EXP / "logs/report_%j.log"),
            "--error=" + str(EXP / "logs/report_%j.err")], "report.slurm")
        save()
    except Exception as exc:
        record.update(status="submission_failed", error=repr(exc))
        save()
        raise
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default=None)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--submit-prepared", metavar="TAG")
    args = parser.parse_args()
    if args.submit_prepared:
        if args.tag:
            parser.error("--tag cannot be combined with --submit-prepared")
        record = submit_prepared(EXP / "frozen" / check_tag(args.submit_prepared))
    else:
        freeze, record = prepare(args.tag or time.strftime("%Y%m%d_%H%M%S"))
        if not args.prepare_only:
            record = submit_prepared(freeze)
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
