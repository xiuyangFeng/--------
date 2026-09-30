#!/usr/bin/env python3
"""Freeze this six-run single-seed experiment and submit gated Slurm jobs.

Only dedicated new files are copied; existing source/runs are never overwritten.
GPU training depends on data preparation AND a real-data six-arm smoke job.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "training_wss_min/experiments/joint_cycle_v52_20260929"
CONFIGS = ROOT / "training_wss_min/configs/joint_cycle_v52_20260929"
PYTHON = "/public/newhome/cy/.conda/envs/GNN/bin/python"
SBATCH = "/public/slurm/bin/sbatch"
SOURCE_FILES = ("__init__.py", "joint_cycle.py", "joint_cycle_model.py", "joint_cycle_data.py",
                "baseline_models.py", "local_refinement.py", "surface.py")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-job", required=True, help="Successful CPU data/cache Slurm job dependency")
    parser.add_argument("--tag", default=time.strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if not args.data_job.isdigit() or any(c not in "0123456789_abcdefghijklmnopqrstuvwxyz" for c in args.tag):
        raise ValueError("invalid job id/tag")
    freeze = EXP / "frozen" / args.tag
    if freeze.exists():
        raise FileExistsError(f"refusing to overwrite snapshot {freeze}")
    (freeze / "training_wss_min").mkdir(parents=True)
    (freeze / "configs").mkdir()
    (EXP / "logs").mkdir(exist_ok=True)
    for name in SOURCE_FILES:
        shutil.copy2(ROOT / "training_wss_min" / name, freeze / "training_wss_min" / name)
    shutil.copy2(CONFIGS / "geometry_x5d.json", freeze / "configs/geometry_x5d.json")
    matrix = json.loads((CONFIGS / "matrix.json").read_text())
    arms = matrix["arms"]
    if len(arms) != 6:
        raise ValueError("registered matrix must contain exactly six single-seed runs")
    frozen_configs = []
    frozen_arms = []
    for arm in arms:
        original = Path(arm["config"])
        if not original.is_absolute():
            original = CONFIGS / original
        cfg = json.loads(original.read_text())
        if cfg["seed"] != 1234:
            raise ValueError("additional seeds are not authorised in this matrix")
        cfg["model"]["geometry_config_path"] = str(freeze / "configs/geometry_x5d.json")
        target = freeze / "configs" / original.name
        write_json(target, cfg)
        frozen_configs.append(str(target))
        frozen_arms.append({**arm, "config": str(target), "config_sha256": digest(target)})
    write_json(freeze / "configs.json", frozen_configs)
    write_json(freeze / "configs/matrix.json", {**matrix, "arms": frozen_arms})
    shutil.copy2(CONFIGS / "PREREG.md", freeze / "configs/PREREG.md")
    split = ROOT / "data_wss_v5/views_v5_2_full_20260923/wss_min_view_v1/cv5_v52/fold0.json"
    fingerprints = {str(p): digest(p) for p in sorted(freeze.rglob("*")) if p.is_file()}
    fingerprints[str(split)] = digest(split)
    fingerprints[str(CONFIGS / "PREREG.md")] = digest(CONFIGS / "PREREG.md")
    write_json(freeze / "fingerprints.json", fingerprints)
    entry = '''import hashlib,json,os,subprocess,sys,time
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
'''
    (freeze / "entry.py").write_text(entry)
    # No override of CUDA_VISIBLE_DEVICES: Slurm owns the GPU assignment.
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
export GNN_JOINT_SOURCE_ROOT={ROOT}
cd {freeze}
date -Is
hostname
id
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total --format=csv
'''
    smoke_script = freeze / "smoke.slurm"
    train_script = freeze / "train.slurm"
    smoke_script.write_text(common + f"{PYTHON} -u entry.py smoke\n")
    train_script.write_text(common + f"{PYTHON} -u entry.py train\n")
    report_tool = ROOT / "training_wss_min/tools/report_joint_cycle_v52.py"
    shutil.copy2(report_tool, freeze / "report.py")
    report_script = freeze / "report.slurm"
    report_script.write_text(f'''#!/bin/bash
#SBATCH --partition=CPU
#SBATCH --nodelist=node03
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
set -euo pipefail
export GNN_JOINT_SOURCE_ROOT={ROOT}
cd {freeze}
{PYTHON} -u report.py --config-dir {freeze / 'configs'} --experiment-dir {EXP}
''')
    for path in (freeze / "entry.py", smoke_script, train_script, freeze / "report.py", report_script):
        fingerprints[str(path)] = digest(path)
    write_json(freeze / "fingerprints.json", fingerprints)
    record = {"schema": "joint_cycle_v52_submission_v1", "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
              "data_job": args.data_job, "freeze": str(freeze), "config_paths": frozen_configs,
              "seed": 1234, "fold": 0, "gpu_models": 6, "max_parallel": 4,
              "status": "prepared", "dependency_chain": "CPU preparation -> six-arm GPU smoke -> six GPU array tasks"}
    write_json(freeze / "submission.json", record)
    if args.prepare_only:
        print(json.dumps(record, ensure_ascii=False, indent=2))
        return
    def submit(extra, script):
        completed = subprocess.run([SBATCH, "--parsable", *extra, str(script)],
                                   text=True, capture_output=True, check=True)
        job = completed.stdout.strip().split(";")[0]
        if not job.isdigit():
            raise RuntimeError(f"invalid sbatch result: {completed.stdout!r}")
        return job
    smoke_job = submit(["--job-name=joint52_smoke", "--time=02:00:00",
                        "--dependency=afterok:" + args.data_job,
                        "--output=" + str(EXP / "logs/smoke_%j.log"),
                        "--error=" + str(EXP / "logs/smoke_%j.err")], smoke_script)
    record.update(smoke_job=smoke_job, status="smoke_submitted")
    write_json(freeze / "submission.json", record)
    array_job = submit(["--job-name=joint52_s1234", "--time=24:00:00", "--array=0-5%4",
                        "--dependency=afterok:" + smoke_job,
                        "--output=" + str(EXP / "logs/train_%A_%a.log"),
                        "--error=" + str(EXP / "logs/train_%A_%a.err")], train_script)
    record.update(array_job=array_job, status="submitted")
    write_json(freeze / "submission.json", record)
    write_json(EXP / "submission.json", record)
    report_job = submit(["--job-name=joint52_report", "--time=00:10:00",
                         "--dependency=afterany:" + array_job,
                         "--output=" + str(EXP / "logs/report_%j.log"),
                         "--error=" + str(EXP / "logs/report_%j.err")], report_script)
    record["report_job"] = report_job
    write_json(freeze / "submission.json", record)
    write_json(EXP / "submission.json", record)
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
