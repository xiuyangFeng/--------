from __future__ import annotations

import argparse
import json
from pathlib import Path

from wss_pinn.config import ExperimentConfig
from wss_pinn.data.dataset import PhysicsDataset
from wss_pinn.utils import ROOT, atomic_write_json, sha256_file, utc_now


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _root_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _paired_f1_control_contract(
    config: ExperimentConfig, control_dir: Path
) -> dict:
    resolved_path = control_dir / "resolved_config.json"
    detail = {
        "path": str(resolved_path.resolve()),
        "exists": resolved_path.is_file(),
        "passed": False,
    }
    if not resolved_path.is_file():
        return detail
    control = ExperimentConfig(_json(resolved_path)).as_dict()
    current = config.as_dict()
    checks = {
        "control_stage_f0up": control.get("experiment", {}).get("stage") == "f0up",
        "data_match": control.get("data") == current.get("data"),
        "sampling_match": control.get("sampling") == current.get("sampling"),
        "model_match": control.get("model") == current.get("model"),
    }
    train_fields = (
        "seed",
        "epochs",
        "learning_rate",
        "weight_decay",
        "precision",
        "batch_wall",
        "batch_near_wall",
        "batch_core",
        "batch_cases",
        "checkpoint_every",
        "log_every",
        "init_checkpoint",
        "resume",
    )
    checks["train_protocol_match"] = all(
        control.get("train", {}).get(key) == current.get("train", {}).get(key)
        for key in train_fields
    )
    shared_loss_fields = (
        "direct_wss_weight",
        "velocity_data_weight",
        "pressure_data_weight",
        "wss_physics_weight",
        "momentum_weight",
    )
    checks["shared_loss_match"] = all(
        control.get("loss", {}).get(key) == current.get("loss", {}).get(key)
        for key in shared_loss_fields
    )
    checks["control_first_order_losses_zero"] = all(
        float(control.get("loss", {}).get(key, 0.0)) == 0.0
        for key in ("continuity_weight", "no_slip_weight")
    )
    detail.update({"checks": checks, "passed": all(checks.values())})
    return detail


def science_gate(config: ExperimentConfig) -> dict:
    if config.stage == "f0u":
        requirements = {
            "p0a": ROOT / "outputs/wss_pinn/audits/p0a/summary.json",
            "p0c": ROOT / "outputs/wss_pinn/audits/p0c/summary.json",
            "p1": Path(config["sampling"]["manifest_path"]),
        }
        checks = {}
        for name, path in requirements.items():
            exists = path.exists()
            passed = exists
            if exists and name in {"p0a", "p0c"}:
                passed = _json(path).get("gate_result") == "pass"
            if exists and name == "p1":
                passed = _json(path).get("status") == "completed"
            checks[name] = {"path": str(path), "exists": exists, "passed": passed}
        passed = all(item["passed"] for item in checks.values())
        return {"passed": passed, "checks": checks, "rule": "P0-A/P0-C/P1 pass"}

    control = config["experiment"].get(
        "scientific_gate_control_run_dir"
    ) or config["experiment"].get("control_run_dir")
    if not control:
        return {
            "passed": False,
            "checks": {},
            "rule": "scientific gate control run is required",
        }
    control_dir = _root_path(control)
    report_path = control_dir / "evaluation_best.json"
    checks = {
        "control_evaluation": {
            "path": str(report_path.resolve()),
            "exists": report_path.exists(),
            "passed": False,
        }
    }
    if report_path.exists():
        report = _json(report_path)
        velocity_pass = all(
            row["velocity"]["r2"] >= 0.95
            and all(
                metric["r2"] >= 0.95
                for metric in row["velocity"].get("components", {}).values()
            )
            for row in report["cases"]
        )
        pressure_pass = all(
            row["pressure_gauge_invariant"]["r2"] >= 0.95 for row in report["cases"]
        )
        required = velocity_pass if config.stage == "f0up" else velocity_pass and pressure_pass
        checks["control_evaluation"].update(
            {
                "passed": required,
                "velocity_r2_ge_0_95": velocity_pass,
                "pressure_r2_ge_0_95": pressure_pass,
            }
        )
    if config.stage == "f1":
        paired_control_dir = _root_path(config["experiment"]["control_run_dir"])
        checks["paired_f0up_contract"] = _paired_f1_control_contract(
            config, paired_control_dir
        )
    split_path = Path(config["data"]["split_path"])
    split_sha256 = sha256_file(split_path) if split_path.is_file() else None
    manifest_path = Path(config["sampling"]["manifest_path"])
    manifest_sha256 = (
        sha256_file(manifest_path) if manifest_path.is_file() else None
    )
    extra_report_fields = (
        ("source_data_audit_report", "source_data_audit"),
        ("sidecar_audit_report", "sidecar_audit"),
        ("scientific_gate_report", "f1_diagnostic_matrix"),
    )
    for field, name in extra_report_fields:
        value = config["experiment"].get(field)
        if not value:
            continue
        path = _root_path(value)
        exists = path.is_file()
        passed = False
        detail = {}
        if exists:
            report = _json(path)
            passed = report.get("gate_result") == "pass"
            detail["gate_result"] = report.get("gate_result")
            if name == "source_data_audit":
                report_split_sha = report.get("split", {}).get("sha256")
                split_match = (
                    split_sha256 is not None
                    and report_split_sha == split_sha256
                )
                detail.update(
                    {
                        "report_split_sha256": report_split_sha,
                        "config_split_sha256": split_sha256,
                        "split_binding_match": split_match,
                    }
                )
                passed = passed and split_match
            if name == "sidecar_audit":
                report_split_sha = report.get("split", {}).get("sha256")
                report_manifest_sha = report.get(
                    "aggregate_manifest", {}
                ).get("sha256")
                split_match = (
                    split_sha256 is not None
                    and report_split_sha == split_sha256
                )
                manifest_match = (
                    manifest_sha256 is not None
                    and report_manifest_sha == manifest_sha256
                )
                detail.update(
                    {
                        "report_split_sha256": report_split_sha,
                        "config_split_sha256": split_sha256,
                        "split_binding_match": split_match,
                        "report_manifest_sha256": report_manifest_sha,
                        "config_manifest_sha256": manifest_sha256,
                        "manifest_binding_match": manifest_match,
                    }
                )
                passed = passed and split_match and manifest_match
            if name == "f1_diagnostic_matrix" and passed:
                selected_path = report.get("selected_config")
                selected = _root_path(selected_path) if selected_path else None
                selected_exists = bool(selected) and selected.is_file()
                selected_hash = (
                    sha256_file(selected) if selected_exists else None
                )
                selected_config = (
                    ExperimentConfig.from_json(selected)
                    if selected_exists
                    else None
                )
                weights_match = bool(selected_config) and all(
                    float(config["loss"][key])
                    == float(selected_config["loss"][key])
                    for key in ("continuity_weight", "no_slip_weight")
                )
                detail.update(
                    {
                        "selected_config": selected_path,
                        "selected_config_exists": selected_exists,
                        "selected_config_sha256": selected_hash,
                        "selected_weights_match": weights_match,
                    }
                )
                passed = passed and selected_exists and weights_match
        checks[name] = {
            "path": str(path.resolve()),
            "exists": exists,
            "passed": passed,
            **detail,
        }
    passed = all(item["passed"] for item in checks.values())
    return {
        "passed": passed,
        "checks": checks,
        "rule": (
            "F0-U completed with velocity R2>=0.95"
            if config.stage == "f0up"
            else (
                "F0-UP control plus every declared source/sidecar/F1 matrix Gate; "
                "full F1 weights must match the selected diagnostic config"
            )
        ),
    }


def run_preflight(
    config: ExperimentConfig, *, require_science_gate: bool, output: Path | None = None
) -> dict:
    dataset = PhysicsDataset(config["sampling"]["manifest_path"], verify=True)
    counts = [
        {
            "case_id": case.case_id,
            "wall": int(len(case.static["wall_coords"])),
            "near_wall": int(len(case.static["near_wall_coords"])),
            "core": int(len(case.static["core_coords"])),
        }
        for case in dataset.cases
    ]
    sampling = config["sampling"]
    slot_pass = all(
        row["wall"] >= min(int(sampling["wall_points"]), row["wall"])
        and row["near_wall"] > 0
        and row["core"] > 0
        for row in counts
    )
    gate = science_gate(config)
    passed = slot_pass and (gate["passed"] or not require_science_gate)
    payload = {
        "schema_version": 1,
        "created_at": utc_now(),
        "status": "passed" if passed else "failed",
        "stage": config.stage,
        "experiment_id": config.experiment_id,
        "config_source": str(config.source),
        "config_source_sha256": sha256_file(config.source) if config.source else None,
        "resolved_config_sha256": config.resolved_sha256,
        "sampling_manifest": config["sampling"]["manifest_path"],
        "sampling_manifest_sha256": sha256_file(config["sampling"]["manifest_path"]),
        "slot_counts": counts,
        "slot_contract_passed": slot_pass,
        "science_gate_required": require_science_gate,
        "science_gate": gate,
    }
    if output is not None:
        atomic_write_json(output, payload)
    if not passed:
        raise RuntimeError(json.dumps(payload, ensure_ascii=False))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--require-science-gate", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    config = ExperimentConfig.from_json(args.config)
    output = Path(args.output) if args.output else None
    result = run_preflight(
        config, require_science_gate=args.require_science_gate, output=output
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
