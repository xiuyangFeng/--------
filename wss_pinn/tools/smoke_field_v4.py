"""CPU/GPU dry-run evidence writer for the field-v4 matrix."""

from __future__ import annotations

import argparse
import json
import resource
from pathlib import Path

import torch

from wss_pinn.config import ExperimentConfig
from wss_pinn.train import run
from wss_pinn.utils import ROOT, atomic_write_json, git_state, sha256_file, utc_now


CONFIG_ROOT = ROOT / "wss_pinn/configs/volume_uvwp_peak_field_v4"


def execute(*, device: str, configs: list[str], output: Path) -> dict:
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft < 65535:
        target = 65535 if hard == resource.RLIM_INFINITY else min(65535, hard)
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
    reports = []
    for name in configs:
        path = CONFIG_ROOT / name
        config = ExperimentConfig.from_json(path)
        result = run(config, dry_run=True, device_override=device)
        reports.append(
            {
                "config": name,
                "config_sha256": sha256_file(path),
                "status": result["status"],
                "initialization_state_sha256": result[
                    "initialization_state_sha256"
                ],
                "validation_field_score_cb": result[
                    "best_validation_field_score_cb"
                ],
                "all_finite": result["last_record"]["all_finite"],
            }
        )
    report = {
        "schema_version": 1,
        "created_at": utc_now(),
        "status": "completed",
        "gate_result": (
            "pass"
            if all(row["status"] == "dry_run_completed" and row["all_finite"] for row in reports)
            else "fail"
        ),
        "route": "volume_uvwp_peak_field_v4",
        "device": device,
        "cuda_device": (
            torch.cuda.get_device_name(0) if device.startswith("cuda") else None
        ),
        "open_file_soft_limit": resource.getrlimit(resource.RLIMIT_NOFILE)[0],
        "configs": reports,
        "formal_training_submitted": False,
        "git": git_state(),
    }
    atomic_write_json(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument(
        "--configs",
        nargs="+",
        default=[
            "g_raw_s1234.json",
            "l_raw_s1234.json",
            "g_pe_s1234.json",
            "l_pe_s1234.json",
        ],
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = execute(
        device=args.device, configs=args.configs, output=Path(args.output)
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
