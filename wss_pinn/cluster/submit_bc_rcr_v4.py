"""Submit Stage-0 build -> GPU preflight -> 48-run array with four-way concurrency."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from wss_pinn.utils import (
    ROOT,
    atomic_write_json,
    git_state,
    guard_write_path,
    sha256_file,
    utc_now,
)
from wss_pinn.v4.config import CONFIG_ROOT, load_config


SBATCH = Path("/public/slurm/bin/sbatch")
SCANCEL = Path("/public/slurm/bin/scancel")
SQUEUE = Path("/public/slurm/bin/squeue")
BUILD = ROOT / "wss_pinn/cluster/build_bc_rcr_v4.slurm"
PREFLIGHT = ROOT / "wss_pinn/cluster/preflight_bc_rcr_v4.slurm"
TRAIN = ROOT / "wss_pinn/cluster/run_bc_rcr_v4.slurm"
OUTPUT = ROOT / "outputs/wss_pinn/volume_uvwp_bc_rcr_v4/submission.json"


def _submit(arguments: list[str]) -> str:
    process = subprocess.run(
        [str(SBATCH), "--parsable", *arguments],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return process.stdout.strip().split(";", 1)[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--max-concurrent", type=int, default=4)
    parser.add_argument("--build-workers", type=int, default=8)
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args()
    if int(args.max_concurrent) != 4:
        raise ValueError("the authorized V4 submission is frozen to four concurrent GPUs")
    matrix_path = CONFIG_ROOT / "matrix.json"
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    names = list(matrix["config_order"])
    if len(names) != 48:
        raise ValueError("formal V4 matrix must contain 48 runs")
    configs = [(CONFIG_ROOT / name).resolve() for name in names]
    for path in configs:
        load_config(path, require_assets=False)
    config_list = guard_write_path(CONFIG_ROOT / "matrix_configs.txt")
    expected_list = "\n".join(str(path) for path in configs) + "\n"
    if config_list.read_text(encoding="utf-8") != expected_list:
        raise ValueError("matrix config list drift")
    output = Path(args.output).resolve()
    payload = {
        "schema_version": 1,
        "created_at": utc_now(),
        "route": matrix["route"],
        "status": "dry_run" if not args.submit else "submitting",
        "formal_training_submitted": False,
        "conditional_on_gates": True,
        "matrix": {"path": str(matrix_path), "sha256": sha256_file(matrix_path)},
        "config_list": {"path": str(config_list), "sha256": sha256_file(config_list)},
        "configs": [
            {"path": str(path), "sha256": sha256_file(path)} for path in configs
        ],
        "slurm": {
            "partition": "GPU",
            "gpu_type": "4090",
            "array": "0-47%4",
            "max_concurrent_gpus": 4,
            "evaluation_after_training": False,
        },
        "scripts": {
            "build": {"path": str(BUILD), "sha256": sha256_file(BUILD)},
            "preflight": {"path": str(PREFLIGHT), "sha256": sha256_file(PREFLIGHT)},
            "train": {"path": str(TRAIN), "sha256": sha256_file(TRAIN)},
        },
        "git": git_state(),
        "jobs": [],
    }
    submitted: list[str] = []
    try:
        if args.submit:
            build_id = _submit(
                [
                    f"--export=ALL,V4_BUILD_WORKERS={int(args.build_workers)}",
                    str(BUILD),
                ]
            )
            submitted.append(build_id)
            preflight_id = _submit(
                [f"--dependency=afterok:{build_id}", str(PREFLIGHT)]
            )
            submitted.append(preflight_id)
            train_id = _submit(
                [
                    f"--dependency=afterok:{preflight_id}",
                    "--array=0-47%4",
                    f"--export=ALL,V4_CONFIG_LIST={config_list}",
                    str(TRAIN),
                ]
            )
            submitted.append(train_id)
            queue = subprocess.run(
                [
                    str(SQUEUE),
                    "-j",
                    ",".join(submitted),
                    "-o",
                    "%.18i %.9P %.28j %.8T %.10M %.6D %R",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            ).stdout
            payload.update(
                {
                    "status": "submitted",
                    "formal_training_submitted": True,
                    "jobs": [
                        {"role": "stage0_build_and_gate", "job_id": build_id},
                        {
                            "role": "gpu_preflight",
                            "job_id": preflight_id,
                            "dependency": f"afterok:{build_id}",
                        },
                        {
                            "role": "matrix_training_48_runs_max4",
                            "job_id": train_id,
                            "dependency": f"afterok:{preflight_id}",
                        },
                    ],
                    "squeue_snapshot": queue,
                }
            )
    except Exception:
        for job_id in submitted:
            subprocess.run([str(SCANCEL), job_id], cwd=ROOT, check=False)
        raise
    atomic_write_json(output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
