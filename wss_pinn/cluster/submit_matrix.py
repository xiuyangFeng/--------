"""Prepare or explicitly submit a config matrix.

The default is dry-run.  Formal submission additionally requires a passing
data Gate, a passing preflight report and ``--submit``.  ``matrix.json`` is the
default source of experiment configs; text lists remain an optional compatibility
override for completed runs.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from wss_pinn.config import ExperimentConfig
from wss_pinn.utils import (
    ROOT,
    atomic_write_json,
    guard_write_path,
    sha256_file,
    utc_now,
)


CONFIG_ROOT = ROOT / "wss_pinn/configs/volume_uvwp_peak_v1"
SLURM_PREFLIGHT = ROOT / "wss_pinn/cluster/preflight.slurm"
SLURM_RUN = ROOT / "wss_pinn/cluster/run_experiment.slurm"


def _gate(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("gate_result") != "pass":
        raise RuntimeError(f"Gate did not pass: {path}")
    return payload


def _matrix_configs(config_root: Path) -> tuple[Path, dict, list[Path]]:
    matrix_path = config_root / "matrix.json"
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    groups = matrix.get("pairs") or matrix.get("groups")
    if not groups:
        raise ValueError(f"matrix has no pairs/groups: {matrix_path}")
    names = [name for group in groups for name in group]
    if len(names) != len(set(names)):
        raise ValueError("matrix config names must be unique")
    paths = [(config_root / name).resolve() for name in names]
    configs = [ExperimentConfig.from_json(path) for path in paths]
    if any(config.route != matrix.get("route") for config in configs):
        raise ValueError("matrix route does not match its experiment configs")
    return matrix_path, matrix, paths


def _config_path(value: str) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def _submission_path(cli_value: str | None, submission: dict, key: str) -> Path:
    value = cli_value or submission.get(key)
    if not value:
        raise ValueError(f"matrix submission.{key} is required")
    return _config_path(str(value))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--config-root", default=str(CONFIG_ROOT))
    parser.add_argument("--config-list")
    parser.add_argument("--preflight-report")
    parser.add_argument("--data-gate")
    parser.add_argument("--output")
    parser.add_argument("--max-concurrent", type=int, default=4)
    args = parser.parse_args()
    config_root = Path(args.config_root).resolve()
    matrix_path, matrix, matrix_config_paths = _matrix_configs(config_root)
    submission = matrix.get("submission", {})
    preflight_report = _submission_path(
        args.preflight_report, submission, "preflight_report"
    )
    data_gate_path = _submission_path(args.data_gate, submission, "data_gate")
    output_path = _submission_path(args.output, submission, "output")
    if args.submit and submission.get("enabled", True) is not True:
        raise RuntimeError(submission.get("reason", "standard matrix submission disabled"))
    if args.config_list:
        config_list = Path(args.config_list).resolve()
        configs = [
            line.strip()
            for line in config_list.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        configs = [str(path) for path in matrix_config_paths]
        config_list = guard_write_path(output_path.parent / "submission_configs.txt")
        config_list.parent.mkdir(parents=True, exist_ok=True)
        config_list.write_text("\n".join(configs) + "\n", encoding="utf-8")
    if [_config_path(value) for value in configs] != matrix_config_paths:
        raise ValueError("config list must match matrix.json order exactly")
    if int(args.max_concurrent) <= 0:
        raise ValueError("max-concurrent must be positive")
    payload = {
        "schema_version": 1,
        "created_at": utc_now(),
        "status": "dry_run" if not args.submit else "submitting",
        "formal_training_submitted": False,
        "config_root": str(config_root),
        "matrix": {"path": str(matrix_path), "sha256": sha256_file(matrix_path)},
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
                    {"role": "matrix_training", "job_id": train_job},
                ],
            }
        )
    output = atomic_write_json(output_path, payload)
    print(json.dumps({"status": payload["status"], "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
