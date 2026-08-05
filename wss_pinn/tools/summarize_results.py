"""Summarize completed peak-volume uvwp experiment matrices.

The script is intentionally read-only with respect to run directories. It reads
the completed evaluation/training logs and writes reproducible tables, a
manifest, and publication-friendly static figures under the route summary
directory.  Legacy V1 eight-arm inputs retain their original paired summary;
the V3 six-arm route additionally reports all three validation-selected
checkpoints, BC/PDE/geom increments, boundary diagnostics, and convergence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PAIRS = (
    {
        "key": "pointnet_xyz",
        "label": "PointNet / xyz",
        "short_label": "PN\nxyz",
        "architecture": "PointNet",
        "input": "xyz",
        "data_run": "VF-PN-XYZ-DATA-s1234-v1",
        "pinn_run": "VF-PN-XYZ-PINN-s1234-v1",
    },
    {
        "key": "pointnet_xyz_geom",
        "label": "PointNet / xyz+geom",
        "short_label": "PN\nxyz+geom",
        "architecture": "PointNet",
        "input": "xyz+geom",
        "data_run": "VF-PN-XYZG-DATA-s1234-v1",
        "pinn_run": "VF-PN-XYZG-PINN-s1234-v1",
    },
    {
        "key": "pointnetpp_xyz",
        "label": "PointNet++ / xyz",
        "short_label": "PN++\nxyz",
        "architecture": "PointNet++",
        "input": "xyz",
        "data_run": "VF-PNPP-XYZ-DATA-s1234-v1",
        "pinn_run": "VF-PNPP-XYZ-PINN-s1234-v1",
    },
    {
        "key": "pointnetpp_xyz_geom",
        "label": "PointNet++ / xyz+geom",
        "short_label": "PN++\nxyz+geom",
        "architecture": "PointNet++",
        "input": "xyz+geom",
        "data_run": "VF-PNPP-XYZG-DATA-s1234-v1",
        "pinn_run": "VF-PNPP-XYZG-PINN-s1234-v1",
    },
)

METRICS = (
    ("velocity_u_r2", "u R2", "r2"),
    ("velocity_v_r2", "v R2", "r2"),
    ("velocity_w_r2", "w R2", "r2"),
    ("speed_r2", "speed R2", "r2"),
    ("pressure_r2", "pressure R2", "r2"),
    ("continuity_rms_dimensionless", "continuity RMS", "lower_better"),
    ("momentum_rms_pa_per_m", "momentum RMS (Pa/m)", "lower_better"),
    ("wall_speed_rms_m_s", "wall speed RMS (m/s)", "lower_better"),
)

R2_KEYS = tuple(item[0] for item in METRICS if item[2] == "r2")
PHYSICS_KEYS = tuple(item[0] for item in METRICS if item[2] == "lower_better")
METRIC_LABELS = {key: label for key, label, _ in METRICS}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _rolling_mean(values: np.ndarray, window: int = 10) -> np.ndarray:
    if values.size < window:
        return values.copy()
    kernel = np.ones(window, dtype=np.float64) / window
    smoothed = np.convolve(values, kernel, mode="valid")
    prefix = np.full(window - 1, np.nan, dtype=np.float64)
    return np.concatenate([prefix, smoothed])


def _window_stats(records: list[dict[str, Any]], window: int = 50) -> dict[str, Any]:
    fields = {
        "data_total": "mean_data_total",
        "physics_total": "mean_physics_total",
        "continuity": "mean_continuity_raw",
        "momentum_sum": None,
        "no_slip": "mean_no_slip_raw",
        "total": "mean_total",
    }
    first = records[: min(window, len(records))]
    last = records[-min(window, len(records)) :]

    def values(block: list[dict[str, Any]], field: str, source: str | None) -> np.ndarray:
        if field == "momentum_sum":
            return np.asarray(
                [
                    row["mean_momentum_x_raw"]
                    + row["mean_momentum_y_raw"]
                    + row["mean_momentum_z_raw"]
                    for row in block
                ],
                dtype=np.float64,
            )
        assert source is not None
        return np.asarray([row[source] for row in block], dtype=np.float64)

    output: dict[str, Any] = {"window_epochs": min(window, len(records))}
    for field, source in fields.items():
        first_mean = float(np.mean(values(first, field, source)))
        last_mean = float(np.mean(values(last, field, source)))
        relative_change = None
        if abs(first_mean) > 1e-15:
            relative_change = 100.0 * (last_mean - first_mean) / abs(first_mean)
        output[field] = {
            "first_mean": first_mean,
            "last_mean": last_mean,
            "relative_change_percent": relative_change,
        }
    return output


def collect(runs_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    paired_rows: list[dict[str, Any]] = []
    paired_json: dict[str, Any] = {
        "checkpoint": "best_total",
        "evaluation_role": "test35",
        "pairs": [],
    }
    convergence: dict[str, Any] = {"window_epochs": 50, "runs": {}}

    for pair in PAIRS:
        pair_result: dict[str, Any] = {**pair, "metrics": {}}
        run_payloads: dict[str, dict[str, Any]] = {}
        for mode, run_name in (("data_only", pair["data_run"]), ("pinn", pair["pinn_run"])):
            run_dir = runs_root / run_name
            evaluation = _read_json(run_dir / "evaluation_best_total.json")
            summary = _read_json(run_dir / "training_summary.json")
            environment = _read_json(run_dir / "environment.json")
            history = _read_jsonl(run_dir / "epoch_progress.jsonl")
            if summary.get("status") != "completed":
                raise RuntimeError(f"run is not completed: {run_name}")
            if len(history) != int(summary["epochs_executed"]):
                raise RuntimeError(
                    f"epoch log mismatch for {run_name}: "
                    f"{len(history)} != {summary['epochs_executed']}"
                )
            started = _parse_time(environment["created_at"])
            finished = _parse_time(summary["created_at"])
            duration_seconds = (finished - started).total_seconds()
            run_payloads[mode] = {
                "metrics": evaluation["case_balanced"],
                "summary": summary,
                "history": history,
                "duration_seconds": duration_seconds,
            }
            convergence["runs"][run_name] = {
                "mode": mode,
                "architecture": pair["architecture"],
                "input": pair["input"],
                "epochs": int(summary["epochs_executed"]),
                "global_steps": int(summary["global_steps"]),
                "duration_seconds": duration_seconds,
                "window_summary": _window_stats(history, window=50),
            }

        data_metrics = run_payloads["data_only"]["metrics"]
        pinn_metrics = run_payloads["pinn"]["metrics"]
        for metric_key, metric_label, direction in METRICS:
            data_value = float(data_metrics[metric_key])
            pinn_value = float(pinn_metrics[metric_key])
            delta = pinn_value - data_value
            reduction = None
            if direction == "lower_better" and abs(data_value) > 1e-15:
                reduction = 100.0 * (data_value - pinn_value) / abs(data_value)
            metric_result = {
                "data_only": data_value,
                "pinn": pinn_value,
                "pinn_minus_data_only": delta,
                "relative_reduction_percent": reduction,
            }
            pair_result["metrics"][metric_key] = metric_result
            paired_rows.append(
                {
                    "pair": pair["key"],
                    "architecture": pair["architecture"],
                    "input": pair["input"],
                    "metric": metric_key,
                    "metric_label": metric_label,
                    "direction": direction,
                    **metric_result,
                }
            )
        pair_result["run_summary"] = {
            mode: {
                "experiment_id": pair[f"{'data' if mode == 'data_only' else 'pinn'}_run"],
                "epochs": int(payload["summary"]["epochs_executed"]),
                "global_steps": int(payload["summary"]["global_steps"]),
                "duration_seconds": payload["duration_seconds"],
            }
            for mode, payload in run_payloads.items()
        }
        paired_json["pairs"].append(pair_result)

    return paired_rows, paired_json, convergence


def _write_tables(
    output_dir: Path,
    paired_rows: list[dict[str, Any]],
    paired_json: dict[str, Any],
    convergence: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "paired_metrics_best_total.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired_rows[0]))
        writer.writeheader()
        writer.writerows(paired_rows)
    (output_dir / "paired_metrics_best_total.json").write_text(
        json.dumps(paired_json, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "convergence_summary.json").write_text(
        json.dumps(convergence, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _annotated_heatmap(ax, values: np.ndarray, row_labels: list[str], col_labels: list[str]) -> None:
    limit = max(float(np.max(np.abs(values))), 0.05)
    image = ax.imshow(values, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(np.arange(len(col_labels)), labels=col_labels)
    ax.set_yticks(np.arange(len(row_labels)), labels=row_labels)
    ax.set_title("R2 delta: PINN - data-only")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            color = "white" if abs(value) > 0.55 * limit else "black"
            ax.text(column, row, f"{value:+.3f}", ha="center", va="center", color=color, fontsize=9)
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="R2 difference")


def plot_paired_metrics(paired_json: dict[str, Any], output_dir: Path) -> None:
    pair_results = paired_json["pairs"]
    r2 = np.asarray(
        [
            [pair["metrics"][key]["pinn_minus_data_only"] for key in R2_KEYS]
            for pair in pair_results
        ],
        dtype=np.float64,
    )
    reductions = np.asarray(
        [
            [pair["metrics"][key]["relative_reduction_percent"] for key in PHYSICS_KEYS]
            for pair in pair_results
        ],
        dtype=np.float64,
    )
    labels = [pair["short_label"] for pair in pair_results]

    fig, (left, right) = plt.subplots(1, 2, figsize=(15.5, 5.6), constrained_layout=True)
    _annotated_heatmap(left, r2, labels, [METRIC_LABELS[key] for key in R2_KEYS])

    x = np.arange(len(labels), dtype=np.float64)
    width = 0.24
    colors = ("#2f6f9f", "#d37b35", "#4e8d63")
    for index, key in enumerate(PHYSICS_KEYS):
        bars = right.bar(
            x + (index - 1) * width,
            reductions[:, index],
            width,
            label=METRIC_LABELS[key],
            color=colors[index],
        )
        right.bar_label(bars, labels=[f"{value:.0f}%" for value in reductions[:, index]], padding=2, fontsize=8)
    right.axhline(0.0, color="0.25", linewidth=0.8)
    right.set_xticks(x, labels=labels)
    right.set_ylabel("Reduction versus data-only (%)")
    right.set_title("Physics inconsistency reduction (higher is better)")
    right.set_ylim(0.0, max(100.0, float(np.max(reductions)) + 12.0))
    right.grid(axis="y", color="0.88", linewidth=0.8)
    right.legend(frameon=False, loc="upper right")

    fig.suptitle("Peak-volume uvwp paired results — best_total checkpoint on test35", fontsize=14)
    for suffix in ("png", "svg"):
        fig.savefig(output_dir / f"paired_metrics_best_total.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_convergence(runs_root: Path, output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(18.5, 8.2), constrained_layout=True, sharex="col")
    for column, pair in enumerate(PAIRS):
        data_history = _read_jsonl(runs_root / pair["data_run"] / "epoch_progress.jsonl")
        pinn_history = _read_jsonl(runs_root / pair["pinn_run"] / "epoch_progress.jsonl")
        epochs = np.asarray([row["epoch"] + 1 for row in data_history], dtype=np.int64)

        top = axes[0, column]
        for label, history, color in (
            ("data-only", data_history, "#5b6770"),
            ("PINN", pinn_history, "#2f6f9f"),
        ):
            values = np.asarray([row["mean_data_total"] for row in history], dtype=np.float64)
            top.plot(epochs, values, color=color, alpha=0.18, linewidth=0.7)
            top.plot(epochs, _rolling_mean(values), color=color, linewidth=1.8, label=label)
        top.set_yscale("log")
        top.set_title(pair["label"])
        top.grid(color="0.90", linewidth=0.7)
        if column == 0:
            top.set_ylabel("Data loss (log scale)")
        top.legend(frameon=False, fontsize=8)

        bottom = axes[1, column]
        physics_series = (
            ("physics total", "mean_physics_total", "#2f6f9f", "-"),
            ("continuity", "mean_continuity_raw", "#4e8d63", "--"),
            ("no-slip", "mean_no_slip_raw", "#8b5b91", ":"),
        )
        for label, field, color, linestyle in physics_series:
            values = np.asarray([row[field] for row in pinn_history], dtype=np.float64)
            bottom.plot(
                epochs,
                _rolling_mean(values),
                color=color,
                linestyle=linestyle,
                linewidth=1.6,
                label=label,
            )
        momentum = np.asarray(
            [
                row["mean_momentum_x_raw"]
                + row["mean_momentum_y_raw"]
                + row["mean_momentum_z_raw"]
                for row in pinn_history
            ],
            dtype=np.float64,
        )
        bottom.plot(
            epochs,
            _rolling_mean(momentum),
            color="#d37b35",
            linestyle="-.",
            linewidth=1.6,
            label="momentum sum",
        )
        bottom.set_yscale("log")
        bottom.set_xlabel("Epoch")
        bottom.grid(color="0.90", linewidth=0.7)
        if column == 0:
            bottom.set_ylabel("PINN physics loss (log scale)")
        bottom.legend(frameon=False, fontsize=7.5)

    fig.suptitle("Training convergence (raw epoch means + 10-epoch moving average)", fontsize=14)
    for suffix in ("png", "svg"):
        fig.savefig(output_dir / f"convergence_curves.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_manifest(
    runs_root: Path,
    output_dir: Path,
    paired_json: dict[str, Any],
) -> None:
    sources: list[dict[str, str]] = []
    for pair in PAIRS:
        for run_name in (pair["data_run"], pair["pinn_run"]):
            for filename in ("evaluation_best_total.json", "training_summary.json", "epoch_progress.jsonl"):
                path = runs_root / run_name / filename
                sources.append({"path": str(path.resolve()), "sha256": _sha256(path)})
    all_pressure_improved = all(
        pair["metrics"]["pressure_r2"]["pinn_minus_data_only"] > 0
        for pair in paired_json["pairs"]
    )
    manifest = {
        "schema_version": 1,
        "route": "volume_uvwp_peak_v1",
        "generated_at": datetime.now().astimezone().isoformat(),
        "checkpoint": "best_total",
        "evaluation_role": "test35",
        "matrix_job": "11128_[0-7%4]",
        "status": "8/8 completed and summarized",
        "sources": sources,
        "outputs": [
            "paired_metrics_best_total.csv",
            "paired_metrics_best_total.json",
            "convergence_summary.json",
            "paired_metrics_best_total.png",
            "paired_metrics_best_total.svg",
            "convergence_curves.png",
            "convergence_curves.svg",
        ],
        "conclusions": {
            "pressure_r2_improved_in_all_pairs": all_pressure_improved,
            "physics_consistency": "continuity, momentum, and wall-speed RMS improved in every pair",
            "velocity_generalization": "mixed; only PointNet xyz+geom improved all u/v/w/speed metrics",
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


V3_ARMS = (
    {"arm": "E1", "run": "QSF-XYZ-DATA-s1234-v3", "input": "xyz", "mode": "DATA"},
    {"arm": "E2", "run": "QSF-XYZ-BC-s1234-v3", "input": "xyz", "mode": "DATA+BC"},
    {"arm": "E3", "run": "QSF-XYZ-BCPDE-s1234-v3", "input": "xyz", "mode": "DATA+BC+PDE"},
    {"arm": "E4", "run": "QSF-XYZG-DATA-s1234-v3", "input": "xyz+geom", "mode": "DATA"},
    {"arm": "E5", "run": "QSF-XYZG-BC-s1234-v3", "input": "xyz+geom", "mode": "DATA+BC"},
    {"arm": "E6", "run": "QSF-XYZG-BCPDE-s1234-v3", "input": "xyz+geom", "mode": "DATA+BC+PDE"},
)
V3_CHECKPOINTS = (
    "best_validation_data",
    "best_validation_total",
    "last",
)
V3_INCREMENT_PAIRS = (
    ("BC_xyz", "E2", "E1", "BC increment: E2-E1"),
    ("PDE_on_BC_xyz", "E3", "E2", "PDE on BC: E3-E2"),
    ("full_physics_xyz", "E3", "E1", "BC+PDE increment: E3-E1"),
    ("BC_xyz_geom", "E5", "E4", "BC increment: E5-E4"),
    ("PDE_on_BC_xyz_geom", "E6", "E5", "PDE on BC: E6-E5"),
    ("full_physics_xyz_geom", "E6", "E4", "BC+PDE increment: E6-E4"),
    ("geom_DATA", "E4", "E1", "geom increment: E4-E1"),
    ("geom_BC", "E5", "E2", "geom increment: E5-E2"),
    ("geom_BCPDE", "E6", "E3", "geom increment: E6-E3"),
)
V3_DELTA_FIELDS = (
    "speed_case_balanced_r2",
    "speed_pooled_r2",
    "speed_case_balanced_mae_m_s",
    "speed_case_balanced_rmse_m_s",
    "near_wall_speed_case_balanced_r2",
    "near_wall_speed_case_balanced_mae_m_s",
    "near_wall_speed_case_balanced_rmse_m_s",
    "core_speed_case_balanced_r2",
    "core_speed_case_balanced_mae_m_s",
    "core_speed_case_balanced_rmse_m_s",
    "pressure_case_balanced_r2",
    "pressure_pooled_r2",
    "wall_speed_rms_m_s",
    "inlet_velocity_vector_rmse_m_s",
    "inlet_flow_absolute_relative_error",
    "outlet_pressure_mae_pa",
    "outlet_mass_balance_absolute_relative_error",
    "continuity_rms_dimensionless",
    "momentum_rms_pa_per_m",
)


def _v3_evaluation_path(run_dir: Path, checkpoint: str) -> Path:
    return run_dir / f"evaluation_{checkpoint}.json"


def _v3_flatten_evaluation(
    arm: dict[str, str], checkpoint: str, evaluation: dict[str, Any]
) -> dict[str, Any]:
    case_balanced = evaluation["case_balanced_regression"]
    pooled = evaluation["pooled_regression"]
    row: dict[str, Any] = {
        **arm,
        "checkpoint": checkpoint,
        "cases": len(evaluation["cases"]),
    }
    for metric, unit in (("u", "m_s"), ("v", "m_s"), ("w", "m_s"), ("speed", "m_s"), ("pressure", "pa")):
        prefix = "pressure" if metric == "pressure" else metric
        row[f"{prefix}_case_balanced_r2"] = case_balanced[metric]["r2"]
        row[f"{prefix}_case_balanced_mae_{unit}"] = case_balanced[metric]["mae"]
        row[f"{prefix}_case_balanced_rmse_{unit}"] = case_balanced[metric]["rmse"]
        row[f"{prefix}_pooled_r2"] = pooled[metric]["r2"]
        row[f"{prefix}_pooled_mae_{unit}"] = pooled[metric]["mae"]
        row[f"{prefix}_pooled_rmse_{unit}"] = pooled[metric]["rmse"]
    for region in ("near_wall", "core"):
        metric = f"{region}_speed"
        row[f"{metric}_case_balanced_r2"] = case_balanced[metric]["r2"]
        row[f"{metric}_case_balanced_mae_m_s"] = case_balanced[metric]["mae"]
        row[f"{metric}_case_balanced_rmse_m_s"] = case_balanced[metric]["rmse"]
        row[f"{metric}_pooled_r2"] = pooled[metric]["r2"]
        row[f"{metric}_pooled_mae_m_s"] = pooled[metric]["mae"]
        row[f"{metric}_pooled_rmse_m_s"] = pooled[metric]["rmse"]
    row.update(evaluation["case_balanced"])
    row.update(
        {
            "speed_positive_r2_cases": sum(
                case["metrics"]["speed"]["r2"] > 0.0
                for case in evaluation["cases"]
            ),
            "speed_case_r2_median": float(
                np.median(
                    [case["metrics"]["speed"]["r2"] for case in evaluation["cases"]]
                )
            ),
            "speed_worst_case": min(
                evaluation["cases"], key=lambda case: case["metrics"]["speed"]["r2"]
            )["case_id"],
            "speed_worst_case_r2": min(
                case["metrics"]["speed"]["r2"] for case in evaluation["cases"]
            ),
            **evaluation["collapse_diagnostic"],
        }
    )
    return row


def _v3_case_rows(
    arm: dict[str, str], evaluation: dict[str, Any]
) -> list[dict[str, Any]]:
    rows = []
    for case in evaluation["cases"]:
        boundary = case["boundary"]
        rows.append(
            {
                **arm,
                "case_id": case["case_id"],
                "evaluated_points": case["evaluated_points"],
                "speed_r2": case["metrics"]["speed"]["r2"],
                "speed_mae_m_s": case["metrics"]["speed"]["mae"],
                "speed_rmse_m_s": case["metrics"]["speed"]["rmse"],
                "speed_prediction_variance_ratio": case["metrics"]["speed"]["prediction_to_truth_variance_ratio"],
                "near_wall_speed_r2": case["metrics"]["near_wall_speed"]["r2"],
                "near_wall_speed_mae_m_s": case["metrics"]["near_wall_speed"]["mae"],
                "near_wall_speed_rmse_m_s": case["metrics"]["near_wall_speed"]["rmse"],
                "core_speed_r2": case["metrics"]["core_speed"]["r2"],
                "core_speed_mae_m_s": case["metrics"]["core_speed"]["mae"],
                "core_speed_rmse_m_s": case["metrics"]["core_speed"]["rmse"],
                "pressure_r2": case["metrics"]["pressure"]["r2"],
                "pressure_mae_pa": case["metrics"]["pressure"]["mae"],
                "pressure_gauge_error_pa": case["pressure_gauge_diagnostic_pa"]["prediction_minus_truth_mean"],
                "wall_speed_rms_m_s": case["wall_speed_m_s"]["rms"],
                "wall_speed_p95_m_s": case["wall_speed_m_s"]["p95"],
                "wall_speed_max_m_s": case["wall_speed_m_s"]["max"],
                "inlet_velocity_vector_rmse_m_s": boundary["inlet"]["velocity"]["vector_rmse_m_s"],
                "inlet_flow_absolute_relative_error": boundary["inlet"]["flow_absolute_relative_error"],
                "outlet_pressure_mae_pa": boundary["outlet"]["pressure_mae_pa"],
                "outlet_pressure_rmse_pa": boundary["outlet"]["pressure_rmse_pa"],
                "outlet_mass_balance_absolute_relative_error": boundary["outlet"]["mass_balance_absolute_relative_error"],
                "continuity_rms_dimensionless": case["physics"]["continuity_rms_dimensionless"],
                "momentum_x_rms_dimensionless": case["physics"]["momentum_rms_dimensionless"][0],
                "momentum_y_rms_dimensionless": case["physics"]["momentum_rms_dimensionless"][1],
                "momentum_z_rms_dimensionless": case["physics"]["momentum_rms_dimensionless"][2],
                "momentum_rms_pa_per_m": case["physics"]["momentum_rms_pa_per_m"],
            }
        )
    return rows


def _late_relative_change(records: list[dict[str, Any]], field: str) -> float:
    count = max(2, int(np.ceil(0.10 * len(records))))
    late = records[-count:]
    midpoint = len(late) // 2
    first = float(np.mean([row[field] for row in late[:midpoint]]))
    second = float(np.mean([row[field] for row in late[midpoint:]]))
    return 100.0 * (second - first) / max(abs(first), 1e-30)


def _v3_convergence(run_dir: Path, arm: dict[str, str]) -> dict[str, Any]:
    records = _read_jsonl(run_dir / "epoch_progress.jsonl")
    summary = _read_json(run_dir / "training_summary.json")
    if summary.get("status") != "completed" or len(records) != 2500:
        raise RuntimeError(f"incomplete V3 run: {arm['run']}")
    late_count = max(2, int(np.ceil(0.10 * len(records))))
    late = records[-late_count:]
    bc_dominance = []
    pde_dominance = []
    bc_shares = []
    pde_shares = []
    for record in late:
        bc_values = np.asarray(
            [record["validation_wall_raw"], record["validation_inlet_raw"], record["validation_outlet_raw"]],
            dtype=np.float64,
        )
        if float(bc_values.sum()) > 0.0:
            shares = bc_values / bc_values.sum()
            bc_shares.append(shares)
            bc_dominance.append(float(np.max(shares)))
        pde_values = np.asarray(
            [
                record["validation_continuity_raw"],
                record["validation_momentum_x_raw"] / 3.0,
                record["validation_momentum_y_raw"] / 3.0,
                record["validation_momentum_z_raw"] / 3.0,
            ],
            dtype=np.float64,
        )
        if float(pde_values.sum()) > 0.0:
            shares = pde_values / pde_values.sum()
            pde_shares.append(shares)
            pde_dominance.append(float(np.max(shares)))
    fields = (
        "validation_data_total",
        "validation_bc_group_raw",
        "validation_pde_group_raw",
        "validation_continuity_raw",
        "validation_momentum_x_raw",
        "validation_momentum_y_raw",
        "validation_momentum_z_raw",
        "validation_total",
    )
    trends = {field: _late_relative_change(records, field) for field in fields}
    applicable = [trends["validation_data_total"]]
    if arm["mode"] != "DATA":
        applicable.append(trends["validation_bc_group_raw"])
    if arm["mode"] == "DATA+BC+PDE":
        applicable.append(trends["validation_pde_group_raw"])
    return {
        **arm,
        "status": summary["status"],
        "epochs": summary["epochs_executed"],
        "global_steps": summary["global_steps"],
        "best_validation_data_epoch": int(
            min(records, key=lambda row: row["validation_data_total"])["epoch"]
        ),
        "best_validation_total_epoch": int(
            min(records, key=lambda row: row["validation_total"])["epoch"]
        ),
        "last_10_percent_epochs": late_count,
        "last_10_percent_relative_change_percent": trends,
        "validation_still_declining_over_1pct": any(value < -1.0 for value in applicable),
        "late_gradient_clipping_fraction_mean": float(
            np.mean([row["gradient_clipping_fraction"] for row in late])
        ),
        "late_gradient_clipping_fraction_p95": float(
            np.quantile([row["gradient_clipping_fraction"] for row in late], 0.95)
        ),
        "late_total_loss_contribution_ratio_mean": {
            "data": float(np.mean([row["validation_data_contribution_ratio"] for row in late])),
            "bc": float(np.mean([row["validation_bc_contribution_ratio"] for row in late])),
            "pde": float(np.mean([row["validation_pde_contribution_ratio"] for row in late])),
        },
        "late_bc_subterm_share_mean": (
            dict(zip(("wall", "inlet", "outlet"), np.mean(bc_shares, axis=0).tolist()))
            if bc_shares
            else {"wall": 0.0, "inlet": 0.0, "outlet": 0.0}
        ),
        "late_pde_subterm_share_mean": (
            dict(zip(("continuity", "mx", "my", "mz"), np.mean(pde_shares, axis=0).tolist()))
            if pde_shares
            else {"continuity": 0.0, "mx": 0.0, "my": 0.0, "mz": 0.0}
        ),
        "bc_subterm_over_90pct_epoch_fraction": (
            float(np.mean(np.asarray(bc_dominance) > 0.90)) if bc_dominance else 0.0
        ),
        "pde_subterm_over_90pct_epoch_fraction": (
            float(np.mean(np.asarray(pde_dominance) > 0.90)) if pde_dominance else 0.0
        ),
        "bc_single_subterm_long_term_dominance": bool(
            bc_dominance and np.mean(np.asarray(bc_dominance) > 0.90) > 0.50
        ),
        "pde_single_subterm_long_term_dominance": bool(
            pde_dominance and np.mean(np.asarray(pde_dominance) > 0.90) > 0.50
        ),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_v3(primary_rows: list[dict[str, Any]], runs_root: Path, output_dir: Path) -> None:
    labels = [row["arm"] for row in primary_rows]
    x = np.arange(len(labels), dtype=np.float64)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0), constrained_layout=True)
    width = 0.36
    axes[0].bar(
        x - width / 2,
        [row["speed_case_balanced_r2"] for row in primary_rows],
        width,
        label="case-balanced",
        color="#2f6f9f",
    )
    axes[0].bar(
        x + width / 2,
        [row["speed_pooled_r2"] for row in primary_rows],
        width,
        label="pooled",
        color="#d37b35",
    )
    axes[0].axhline(0.0, color="0.25", linewidth=0.8)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("speed R²")
    axes[0].set_title("Test35 speed accuracy")
    axes[0].grid(axis="y", color="0.9")
    axes[0].legend(frameon=False)
    axes[1].bar(
        x - width / 2,
        [row["near_wall_speed_case_balanced_r2"] for row in primary_rows],
        width,
        label="near-wall",
        color="#4e8d63",
    )
    axes[1].bar(
        x + width / 2,
        [row["core_speed_case_balanced_r2"] for row in primary_rows],
        width,
        label="core",
        color="#8b5b91",
    )
    axes[1].axhline(0.0, color="0.25", linewidth=0.8)
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("case-balanced speed R²")
    axes[1].set_title("Near-wall versus core")
    axes[1].grid(axis="y", color="0.9")
    axes[1].legend(frameon=False)
    fig.suptitle("QS smooth-field six-arm matrix — best_validation_data")
    for suffix in ("png", "svg"):
        fig.savefig(output_dir / f"six_arm_primary_metrics.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(16.0, 8.2), constrained_layout=True)
    for axis, arm in zip(axes.flat, V3_ARMS):
        history = _read_jsonl(runs_root / arm["run"] / "epoch_progress.jsonl")
        epochs = np.asarray([row["epoch"] + 1 for row in history])
        validation_data = np.asarray([row["validation_data_total"] for row in history])
        axis.plot(epochs, _rolling_mean(validation_data), color="#2f6f9f", label="val data")
        if arm["mode"] != "DATA":
            bc = np.asarray([row["validation_bc_group_raw"] for row in history])
            axis.plot(epochs, _rolling_mean(bc), color="#4e8d63", label="val BC")
        if arm["mode"] == "DATA+BC+PDE":
            pde = np.asarray([row["validation_pde_group_raw"] for row in history])
            axis.plot(epochs, _rolling_mean(pde), color="#d37b35", label="val PDE")
        axis.set_yscale("log")
        axis.set_title(f"{arm['arm']} {arm['input']} {arm['mode']}")
        axis.set_xlabel("Epoch")
        axis.grid(color="0.9")
        axis.legend(frameon=False, fontsize=8)
    fig.suptitle("Validation convergence (10-epoch moving average)")
    for suffix in ("png", "svg"):
        fig.savefig(output_dir / f"convergence_curves.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def summarize_v3(runs_root: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    evaluations: dict[tuple[str, str], dict[str, Any]] = {}
    for checkpoint in V3_CHECKPOINTS:
        for arm in V3_ARMS:
            evaluation_path = _v3_evaluation_path(runs_root / arm["run"], checkpoint)
            evaluation = _read_json(evaluation_path)
            if evaluation.get("status") != "completed" or len(evaluation.get("cases", [])) != 35:
                raise RuntimeError(f"incomplete V3 evaluation: {evaluation_path}")
            evaluations[(arm["arm"], checkpoint)] = evaluation
            all_rows.append(_v3_flatten_evaluation(arm, checkpoint, evaluation))
            if checkpoint == "best_validation_data":
                case_rows.extend(_v3_case_rows(arm, evaluation))

    primary_rows = [row for row in all_rows if row["checkpoint"] == "best_validation_data"]
    primary_by_arm = {row["arm"]: row for row in primary_rows}
    delta_rows = []
    delta_json = []
    for key, treatment, control, label in V3_INCREMENT_PAIRS:
        metrics = {}
        for field in V3_DELTA_FIELDS:
            treatment_value = float(primary_by_arm[treatment][field])
            control_value = float(primary_by_arm[control][field])
            metrics[field] = {
                "treatment": treatment_value,
                "control": control_value,
                "delta": treatment_value - control_value,
            }
            delta_rows.append(
                {
                    "comparison": key,
                    "label": label,
                    "treatment": treatment,
                    "control": control,
                    "metric": field,
                    **metrics[field],
                }
            )
        delta_json.append(
            {
                "comparison": key,
                "label": label,
                "treatment": treatment,
                "control": control,
                "metrics": metrics,
            }
        )

    convergence_rows = [
        _v3_convergence(runs_root / arm["run"], arm) for arm in V3_ARMS
    ]
    convergence = {
        "schema_version": 1,
        "route": "volume_uvwp_peak_qs_smooth_v3",
        "extension_rule": (
            "extend all six arms only if any applicable validation data/BC/PDE "
            "group declines by more than 1% over the last 10%"
        ),
        "any_arm_requires_matrix_extension": any(
            row["validation_still_declining_over_1pct"] for row in convergence_rows
        ),
        "runs": convergence_rows,
    }

    _write_csv(output_dir / "six_arm_metrics_all_checkpoints.csv", all_rows)
    _write_csv(output_dir / "six_arm_metrics_best_validation_data.csv", primary_rows)
    _write_csv(output_dir / "case_metrics_best_validation_data.csv", case_rows)
    _write_csv(output_dir / "incremental_effects_best_validation_data.csv", delta_rows)
    (output_dir / "six_arm_metrics_all_checkpoints.json").write_text(
        json.dumps(all_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "six_arm_metrics_best_validation_data.json").write_text(
        json.dumps(primary_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "incremental_effects_best_validation_data.json").write_text(
        json.dumps(delta_json, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "convergence_summary.json").write_text(
        json.dumps(convergence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _plot_v3(primary_rows, runs_root, output_dir)

    sources = []
    for arm in V3_ARMS:
        run_dir = runs_root / arm["run"]
        for filename in ("training_summary.json", "epoch_progress.jsonl"):
            path = run_dir / filename
            sources.append({"path": str(path.resolve()), "sha256": _sha256(path)})
        for checkpoint in V3_CHECKPOINTS:
            path = _v3_evaluation_path(run_dir, checkpoint)
            sources.append({"path": str(path.resolve()), "sha256": _sha256(path)})
    manifest = {
        "schema_version": 1,
        "route": "volume_uvwp_peak_qs_smooth_v3",
        "generated_at": datetime.now().astimezone().isoformat(),
        "status": "6/6 training completed; 18/18 checkpoint evaluations summarized",
        "primary_checkpoint": "best_validation_data",
        "sensitivity_checkpoints": ["best_validation_total", "last"],
        "evaluation_role": "test35",
        "evaluation_protocol": "fixed 5k support to full strict volume",
        "scientific_positioning": "Fluent transient-peak labels + Carreau-Yasuda quasi-steady PINN regularization",
        "outlet_flow_limitation": (
            "Exact outlet pressure targets are available. Outlet flow is a PCA-normal "
            "prediction/mass-balance diagnostic because exact outlet flow targets and "
            "face area weights were not frozen in the V3 boundary asset."
        ),
        "sources": sources,
        "outputs": sorted(path.name for path in output_dir.iterdir() if path.is_file()),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("outputs/wss_pinn/volume_uvwp_peak_v1/runs"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/wss_pinn/volume_uvwp_peak_v1/summary"),
    )
    args = parser.parse_args()
    runs_root = args.runs_root.resolve()
    output_dir = args.output_dir.resolve()
    if (runs_root / "QSF-XYZ-DATA-s1234-v3").is_dir():
        summarize_v3(runs_root, output_dir)
        print(json.dumps({"status": "completed", "output_dir": str(output_dir)}, indent=2))
        return
    paired_rows, paired_json, convergence = collect(runs_root)
    _write_tables(output_dir, paired_rows, paired_json, convergence)
    plot_paired_metrics(paired_json, output_dir)
    plot_convergence(runs_root, output_dir)
    write_manifest(runs_root, output_dir, paired_json)
    print(json.dumps({"status": "completed", "output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
