from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from wss_pinn.config import ExperimentConfig
from wss_pinn.utils import (
    ROOT,
    atomic_write_json,
    guard_write_path,
    sha256_file,
    utc_now,
)


DEFAULT_F0U_CAPABILITY_CONTROL = (
    ROOT / "outputs/wss_pinn/runs/PINN-F0U-overfit3-v2ext-s1234-20260730"
)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _require_gate(path: Path, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    report = _read(path)
    if report.get("gate_result") != "pass":
        raise RuntimeError(f"{name} Gate did not pass: {path}")
    return report


def validate_report_bindings(
    p1: ExperimentConfig,
    *,
    source_report_path: Path,
    sidecar_report_path: Path,
    matrix_report_path: Path,
) -> dict[str, Any]:
    split_path = Path(p1["data"]["split_path"])
    manifest_path = Path(p1["sampling"]["manifest_path"])
    if not split_path.is_file():
        raise FileNotFoundError(split_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    split_sha = sha256_file(split_path)
    manifest_sha = sha256_file(manifest_path)

    source = _require_gate(source_report_path, "source-data audit")
    if source.get("split", {}).get("sha256") != split_sha:
        raise RuntimeError("source-data audit is not bound to the P1 split")

    sidecar = _require_gate(sidecar_report_path, "sidecar audit")
    if sidecar.get("split", {}).get("sha256") != split_sha:
        raise RuntimeError("sidecar audit is not bound to the P1 split")
    if sidecar.get("aggregate_manifest", {}).get("sha256") != manifest_sha:
        raise RuntimeError("sidecar audit is not bound to the P1 manifest")

    matrix = _require_gate(matrix_report_path, "F1 diagnostic matrix")
    selected_path = matrix.get("selected_config")
    if not selected_path:
        raise RuntimeError("F1 matrix has no selected_config")
    selected = Path(selected_path)
    if not selected.is_absolute():
        selected = ROOT / selected
    if not selected.is_file():
        raise FileNotFoundError(selected)
    selected_config = ExperimentConfig.from_json(selected)
    if selected_config.stage != "f1":
        raise ValueError("selected F1 matrix config is not stage=f1")
    return {
        "split_sha256": split_sha,
        "manifest_sha256": manifest_sha,
        "source_report_sha256": sha256_file(source_report_path),
        "sidecar_report_sha256": sha256_file(sidecar_report_path),
        "matrix_report_sha256": sha256_file(matrix_report_path),
        "selected_config": str(selected.resolve()),
        "selected_config_sha256": sha256_file(selected),
        "selected": selected_config,
    }


def build_full_configs(
    p1: ExperimentConfig,
    *,
    source_report_path: Path,
    sidecar_report_path: Path,
    matrix_report_path: Path,
    selected: ExperimentConfig,
    run_tag: str,
    batch_cases: int = 3,
    f0u_capability_control: Path = DEFAULT_F0U_CAPABILITY_CONTROL,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if p1.stage != "p1":
        raise ValueError("full training preparation requires a P1 config")
    if selected.stage != "f1":
        raise ValueError("selected config must be stage=f1")
    if bool(selected["experiment"].get("diagnostic_only")) is not True:
        raise ValueError("selected config must be the audited diagnostic arm")
    if int(batch_cases) <= 0:
        raise ValueError("batch_cases must be positive for full-data training")

    dataset_version = Path(p1["paths"]["sidecar_root"]).name.replace("_", "-")
    seed = int(selected["train"]["seed"])
    f0up_id = f"PINN-F0UP-{dataset_version}-s{seed}-{run_tag}"
    f1_id = f"PINN-F1-{dataset_version}-s{seed}-{run_tag}"
    f0up_run = ROOT / "outputs/wss_pinn/runs" / f0up_id
    f1_run = ROOT / "outputs/wss_pinn/runs" / f1_id

    common_train = deepcopy(selected["train"])
    common_train.update(
        {
            "batch_cases": int(batch_cases),
            "init_checkpoint": None,
            "resume": None,
        }
    )
    common = {
        "paths": deepcopy(p1["paths"]),
        "data": deepcopy(p1["data"]),
        "sampling": deepcopy(p1["sampling"]),
        "model": deepcopy(selected["model"]),
        "cluster": deepcopy(selected["cluster"]),
    }
    shared_loss = deepcopy(selected["loss"])
    shared_loss["wss_physics_weight"] = 0.0
    shared_loss["momentum_weight"] = 0.0

    f0up = {
        **deepcopy(common),
        "experiment": {
            "id": f0up_id,
            "stage": "f0up",
            "scientific_gate_control_run_dir": str(
                f0u_capability_control.resolve()
            ),
            "source_data_audit_report": str(source_report_path.resolve()),
            "sidecar_audit_report": str(sidecar_report_path.resolve()),
            "scientific_gate": (
                "pilot F0-U velocity capability plus split-bound source and "
                "sidecar audits; full-data paired data-only control"
            ),
        },
        "loss": {
            **deepcopy(shared_loss),
            "continuity_weight": 0.0,
            "no_slip_weight": 0.0,
        },
        "train": {**deepcopy(common_train), "run_dir": str(f0up_run)},
    }
    f1 = {
        **deepcopy(common),
        "experiment": {
            "id": f1_id,
            "stage": "f1",
            "diagnostic_only": False,
            "control_run_dir": str(f0up_run),
            "source_data_audit_report": str(source_report_path.resolve()),
            "sidecar_audit_report": str(sidecar_report_path.resolve()),
            "scientific_gate_report": str(matrix_report_path.resolve()),
            "scientific_gate": (
                "paired full-data F0-UP plus split-bound source/sidecar "
                "audits and selected F1 diagnostic weights"
            ),
        },
        "loss": deepcopy(shared_loss),
        "train": {**deepcopy(common_train), "run_dir": str(f1_run)},
    }
    ExperimentConfig(f0up)
    ExperimentConfig(f1)
    return f0up, f1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p1-config", required=True)
    parser.add_argument("--source-audit-report", required=True)
    parser.add_argument("--sidecar-audit-report", required=True)
    parser.add_argument("--f1-matrix-report", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--batch-cases", type=int, default=3)
    parser.add_argument(
        "--f0u-capability-control",
        default=str(DEFAULT_F0U_CAPABILITY_CONTROL),
    )
    args = parser.parse_args()

    p1 = ExperimentConfig.from_json(args.p1_config)
    source_report = Path(args.source_audit_report).resolve()
    sidecar_report = Path(args.sidecar_audit_report).resolve()
    matrix_report = Path(args.f1_matrix_report).resolve()
    bindings = validate_report_bindings(
        p1,
        source_report_path=source_report,
        sidecar_report_path=sidecar_report,
        matrix_report_path=matrix_report,
    )
    f0up, f1 = build_full_configs(
        p1,
        source_report_path=source_report,
        sidecar_report_path=sidecar_report,
        matrix_report_path=matrix_report,
        selected=bindings.pop("selected"),
        run_tag=args.run_tag,
        batch_cases=args.batch_cases,
        f0u_capability_control=Path(args.f0u_capability_control),
    )
    output_dir = guard_write_path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"config output is not empty: {output_dir}")
    f0up_path = atomic_write_json(output_dir / "f0up_full.json", f0up)
    f1_path = atomic_write_json(output_dir / "f1_full.json", f1)
    prepared = atomic_write_json(
        output_dir / "prepared.json",
        {
            "schema_version": 1,
            "created_at": utc_now(),
            "status": "code_ready",
            "p1_config": str(p1.source),
            "p1_config_sha256": sha256_file(p1.source),
            "bindings": bindings,
            "f0up_config": str(f0up_path),
            "f0up_config_sha256": sha256_file(f0up_path),
            "f1_config": str(f1_path),
            "f1_config_sha256": sha256_file(f1_path),
            "submission_order": [
                "submit f0up_full.json and require completed evaluation Gate",
                "submit f1_full.json only after the paired F0-UP Gate passes",
            ],
        },
    )
    print(
        json.dumps(
            {
                "status": "code_ready",
                "f0up_config": str(f0up_path),
                "f1_config": str(f1_path),
                "prepared": str(prepared),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
