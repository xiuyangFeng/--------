from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from wss_pinn.config import ExperimentConfig
from wss_pinn.utils import ROOT, atomic_write_json, guard_write_path, sha256_file, utc_now


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _minimum_velocity_r2(report: dict[str, Any]) -> float:
    values = []
    for case in report["cases"]:
        values.append(float(case["velocity"]["r2"]))
        values.extend(
            float(metric["r2"])
            for metric in case["velocity"].get("components", {}).values()
        )
    return min(values)


def _minimum_pressure_r2(report: dict[str, Any]) -> float:
    return min(
        float(case["pressure_gauge_invariant"]["r2"])
        for case in report["cases"]
    )


def _case_map(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(case["case_id"]): case for case in report["cases"]}


def _mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def analyze_checkpoint(
    report: dict[str, Any],
    control: dict[str, Any],
    *,
    require_continuity: bool,
    require_no_slip: bool,
) -> dict[str, Any]:
    control_cases = _case_map(control)
    continuity_ratios = []
    no_slip_ratios = []
    for case in report["cases"]:
        parent = control_cases[case["case_id"]]
        continuity_ratios.append(
            float(case["continuity_rms_dimensionless"])
            / max(float(parent["continuity_rms_dimensionless"]), 1e-12)
        )
        no_slip_ratios.append(
            float(case["no_slip_rms_dimensionless"])
            / max(float(parent["no_slip_rms_dimensionless"]), 1e-12)
        )
    delta = {
        key: float(report["case_balanced"][key])
        - float(control["case_balanced"][key])
        for key in ("r2", "mae", "high_wss_nrmse", "top10_iou")
    }
    checks = {
        "continuity_mean_ratio_le_0_10": (
            not require_continuity or _mean(continuity_ratios) <= 0.10
        ),
        "continuity_each_ratio_le_0_20": (
            not require_continuity or max(continuity_ratios) <= 0.20
        ),
        "no_slip_mean_ratio_le_0_95": (
            not require_no_slip or _mean(no_slip_ratios) <= 0.95
        ),
        "no_slip_each_ratio_le_1_00": (
            not require_no_slip or max(no_slip_ratios) <= 1.00
        ),
        "velocity_min_r2_ge_0_95": _minimum_velocity_r2(report) >= 0.95,
        "pressure_min_r2_ge_0_95": _minimum_pressure_r2(report) >= 0.95,
        "wss_delta_r2_ge_minus_0_01": delta["r2"] >= -0.01,
        "wss_delta_mae_le_0_05_pa": delta["mae"] <= 0.05,
        "wss_delta_high_nrmse_le_0_01": delta["high_wss_nrmse"] <= 0.01,
        "wss_delta_iou_ge_minus_0_05": delta["top10_iou"] >= -0.05,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "continuity_ratio_mean": _mean(continuity_ratios),
        "continuity_ratio_max": max(continuity_ratios),
        "no_slip_ratio_mean": _mean(no_slip_ratios),
        "no_slip_ratio_max": max(no_slip_ratios),
        "velocity_r2_min": _minimum_velocity_r2(report),
        "pressure_r2_min": _minimum_pressure_r2(report),
        "delta_wss": delta,
    }


def analyze_config(path: Path) -> dict[str, Any]:
    config = ExperimentConfig.from_json(path)
    run_dir = config.run_dir
    control_dir = Path(config["experiment"]["control_run_dir"])
    if not control_dir.is_absolute():
        control_dir = ROOT / control_dir
    continuity = float(config["loss"]["continuity_weight"]) > 0
    no_slip = float(config["loss"]["no_slip_weight"]) > 0
    checkpoints = {}
    for name in ("best", "last"):
        report_path = run_dir / f"evaluation_{name}.json"
        control_path = control_dir / f"evaluation_{name}.json"
        report = _read(report_path)
        control = _read(control_path)
        checkpoints[name] = analyze_checkpoint(
            report,
            control,
            require_continuity=continuity,
            require_no_slip=no_slip,
        )
    joint = continuity and no_slip
    robust_pass = joint and all(row["passed"] for row in checkpoints.values())
    return {
        "experiment_id": config.experiment_id,
        "config": str(path.resolve()),
        "config_sha256": sha256_file(path),
        "resolved_config_sha256": config.resolved_sha256,
        "run_dir": str(run_dir),
        "continuity_weight": float(config["loss"]["continuity_weight"]),
        "no_slip_weight": float(config["loss"]["no_slip_weight"]),
        "joint_candidate": joint,
        "robust_gate_passed": robust_pass,
        "checkpoints": checkpoints,
    }


def _flatten(row: dict[str, Any]) -> dict[str, Any]:
    output = {
        key: value
        for key, value in row.items()
        if key not in {"checkpoints"}
    }
    for checkpoint, metrics in row["checkpoints"].items():
        for key, value in metrics.items():
            if key == "checks":
                for check, passed in value.items():
                    output[f"{checkpoint}_check_{check}"] = passed
            elif key == "delta_wss":
                for metric, delta in value.items():
                    output[f"{checkpoint}_delta_wss_{metric}"] = delta
            else:
                output[f"{checkpoint}_{key}"] = value
    return output


def _atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    target = guard_write_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    flattened = [_flatten(row) for row in rows]
    fields = sorted({key for row in flattened for key in row})
    fd, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(flattened)
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    rows = [analyze_config(Path(path)) for path in args.configs]
    passing = [row for row in rows if row["robust_gate_passed"]]
    selected = None
    if passing:
        selected = max(
            passing,
            key=lambda row: (
                row["checkpoints"]["best"]["velocity_r2_min"],
                -row["checkpoints"]["best"]["continuity_ratio_mean"],
                -row["checkpoints"]["best"]["no_slip_ratio_mean"],
            ),
        )
    payload = {
        "schema_version": 1,
        "created_at": utc_now(),
        "status": "completed",
        "gate_result": "pass" if selected else "fail",
        "gate_contract": {
            "joint_best_and_last_required": True,
            "continuity_mean_ratio_max": 0.10,
            "continuity_each_ratio_max": 0.20,
            "no_slip_mean_ratio_max": 0.95,
            "no_slip_each_ratio_max": 1.00,
            "velocity_min_r2": 0.95,
            "pressure_min_r2": 0.95,
            "wss_delta_r2_min": -0.01,
            "wss_delta_mae_max_pa": 0.05,
            "wss_delta_high_nrmse_max": 0.01,
            "wss_delta_top10_iou_min": -0.05,
        },
        "selected_experiment_id": (
            selected["experiment_id"] if selected else None
        ),
        "selected_config": selected["config"] if selected else None,
        "rows": rows,
    }
    output_dir = guard_write_path(args.output_dir)
    report = atomic_write_json(output_dir / "report.json", payload)
    _atomic_csv(output_dir / "matrix.csv", rows)
    print(
        json.dumps(
            {
                "status": "completed",
                "gate_result": payload["gate_result"],
                "selected_experiment_id": payload["selected_experiment_id"],
                "report": str(report),
            },
            ensure_ascii=False,
        )
    )
    if not selected:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
