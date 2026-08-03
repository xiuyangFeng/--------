"""Prepare or explicitly submit the eight-arm GPU matrix.

The default is dry-run.  Formal submission additionally requires a passing
sidecar Gate, a passing preflight report and ``--submit``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from wss_pinn.utils import ROOT, atomic_write_json, sha256_file, utc_now


CONFIG_ROOT = ROOT / "wss_pinn/configs/volume_uvwp_peak_v1"
CONFIG_LIST = CONFIG_ROOT / "matrix_configs.txt"
PREFLIGHT = ROOT / "outputs/wss_pinn/audits/volume_uvwp_peak_v1_preflight/report.json"
DATA_GATE = ROOT / "outputs/wss_pinn/audits/volume_uvwp_peak_v1_train138_test35/report.json"
SLURM_PREFLIGHT = ROOT / "wss_pinn/volume_field/cluster/preflight.slurm"
SLURM_RUN = ROOT / "wss_pinn/volume_field/cluster/run_experiment.slurm"
OUTPUT = ROOT / "outputs/wss_pinn/volume_uvwp_peak_v1/submission.json"


def _gate(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("gate_result") != "pass":
        raise RuntimeError(f"Gate did not pass: {path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--config-root", default=str(CONFIG_ROOT))
    parser.add_argument("--config-list", default=str(CONFIG_LIST))
    parser.add_argument("--preflight-report", default=str(PREFLIGHT))
    parser.add_argument("--data-gate", default=str(DATA_GATE))
    parser.add_argument("--output", default=str(OUTPUT))
    parser.add_argument("--max-concurrent", type=int, default=4)
    args = parser.parse_args()
    config_root = Path(args.config_root).resolve()
    config_list = Path(args.config_list).resolve()
    preflight_report = Path(args.preflight_report).resolve()
    data_gate_path = Path(args.data_gate).resolve()
    output_path = Path(args.output).resolve()
    configs = [
        line.strip()
        for line in config_list.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not configs:
        raise ValueError("config list is empty")
    if int(args.max_concurrent) <= 0:
        raise ValueError("max-concurrent must be positive")
    payload = {
        "schema_version": 1,
        "created_at": utc_now(),
        "status": "dry_run" if not args.submit else "submitting",
        "formal_training_submitted": False,
        "config_root": str(config_root),
        "config_list": {"path": str(config_list), "sha256": sha256_file(config_list)},
        "slurm_time_limit": "0 (no requested limit; partition policy may still apply)",
        "configs": configs,
        "jobs": [],
    }
    if args.submit:
        _gate(preflight_report)
        _gate(data_gate_path)
        preflight_job = subprocess.run(
            [
                "sbatch",
                "--parsable",
                (
                    "--export=ALL,"
                    f"VOLUME_CONFIG_LIST={config_list},"
                    f"VOLUME_CONFIG_ROOT={config_root},"
                    f"VOLUME_PREFLIGHT_OUTPUT={preflight_report}"
                ),
                str(SLURM_PREFLIGHT),
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
        train_job = subprocess.run(
            [
                "sbatch",
                "--parsable",
                f"--dependency=afterok:{preflight_job}",
                f"--array=0-{len(configs) - 1}%{int(args.max_concurrent)}",
                f"--export=ALL,VOLUME_CONFIG_LIST={config_list}",
                str(SLURM_RUN),
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
        payload.update(
            {
                "status": "submitted",
                "formal_training_submitted": True,
                "preflight_gate_sha256": sha256_file(preflight_report),
                "data_gate_sha256": sha256_file(data_gate_path),
                "jobs": [
                    {"role": "gpu_preflight", "job_id": preflight_job},
                    {"role": "eight_arm_training", "job_id": train_job},
                ],
            }
        )
    output = atomic_write_json(output_path, payload)
    print(json.dumps({"status": payload["status"], "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
