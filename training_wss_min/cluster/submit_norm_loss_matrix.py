#!/usr/bin/env python3
"""Submit the Q2V normalization-口径 / tail-loss 5-arm matrix.

流程：先在 node03（CPU）跑 prepare 生成平衡口径统计 + 5 个 arm 配置 + manifest，
GPU array 训练作业以 ``afterok`` 依赖 prepare。A0 复用已训练的
q2v_ilo_d3_pool2025_refit，不重复提交。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SBATCH = Path("/public/slurm/bin/sbatch")
PREPARE_SCRIPT = REPO / "training_wss_min/cluster/prepare_norm_loss_matrix_cpu.slurm"
TRAIN_SCRIPT = REPO / "training_wss_min/cluster/run_norm_loss_matrix.slurm"
OUTPUT = REPO / "training_wss_min/preflight/norm_loss_matrix_submission.json"


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sbatch(*args: str) -> str:
    result = subprocess.run([str(SBATCH), "--parsable", *args], cwd=REPO,
                            text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "sbatch failed")
    return result.stdout.strip().split(";", 1)[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--concurrency", type=int, default=4,
                        help="array %%N concurrency cap")
    args = parser.parse_args()
    if not SBATCH.is_file():
        raise FileNotFoundError(SBATCH)
    for script in (PREPARE_SCRIPT, TRAIN_SCRIPT):
        if not script.is_file():
            raise FileNotFoundError(script)
    if OUTPUT.exists() and not args.dry_run:
        raise FileExistsError(f"refusing duplicate submission manifest: {OUTPUT}")

    array_spec = f"0-4%{max(1, args.concurrency)}"
    payload = {
        "schema_version": 1,
        "created_at": now(),
        "status": "dry_run" if args.dry_run else "submitting",
        "protocol": "q2v_pool2025_norm_loss_single_seed_5arm",
        "prepare_script": str(PREPARE_SCRIPT.resolve()), "prepare_script_sha256": sha(PREPARE_SCRIPT),
        "train_script": str(TRAIN_SCRIPT.resolve()), "train_script_sha256": sha(TRAIN_SCRIPT),
        "array": array_spec,
        "jobs": [],
    }
    if args.dry_run:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    try:
        prep_job = sbatch("--job-name", "normloss_prep", str(PREPARE_SCRIPT))
        payload["jobs"].append({"role": "cpu_prepare_node03", "job_id": prep_job,
                                "script": str(PREPARE_SCRIPT.resolve())})
        dump(OUTPUT, payload)
        train_job = sbatch("--job-name", "normloss",
                           f"--dependency=afterok:{prep_job}",
                           f"--array={array_spec}", str(TRAIN_SCRIPT))
        payload["jobs"].append({"role": "gpu_training_array", "job_id": train_job,
                                "array": array_spec, "dependency": f"afterok:{prep_job}",
                                "script": str(TRAIN_SCRIPT.resolve())})
        payload["status"] = "submitted"
        payload["submitted_at"] = now()
        dump(OUTPUT, payload)
    except Exception as exc:
        payload["status"] = "partial_failed" if payload["jobs"] else "failed"
        payload["error"] = f"{type(exc).__name__}: {exc}"
        dump(OUTPUT, payload)
        raise
    print(json.dumps({"status": payload["status"], "jobs": payload["jobs"],
                      "manifest": str(OUTPUT)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
