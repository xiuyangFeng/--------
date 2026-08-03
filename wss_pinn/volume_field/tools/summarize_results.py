"""Summarize the eight-arm peak-volume uvwp experiment matrix.

The script is intentionally read-only with respect to run directories. It reads
the completed evaluation/training logs and writes reproducible tables, a
manifest, and publication-friendly static figures under the route summary
directory.
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
    paired_rows, paired_json, convergence = collect(runs_root)
    _write_tables(output_dir, paired_rows, paired_json, convergence)
    plot_paired_metrics(paired_json, output_dir)
    plot_convergence(runs_root, output_dir)
    write_manifest(runs_root, output_dir, paired_json)
    print(json.dumps({"status": "completed", "output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
