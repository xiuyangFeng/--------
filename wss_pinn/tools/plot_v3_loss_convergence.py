"""Create teacher-facing V3 data/physics loss convergence figures.

The plots are generated only from the frozen ``epoch_progress.jsonl`` logs.
Raw epoch values are retained as faint lines and a 25-epoch trailing mean is
drawn on top.  The final 10% of training is shaded so plateau/overfit behaviour
can be judged without using test35.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


ARMS = (
    {"arm": "E1", "run": "QSF-XYZ-DATA-s1234-v3", "input": "xyz", "mode": "DATA"},
    {"arm": "E2", "run": "QSF-XYZ-BC-s1234-v3", "input": "xyz", "mode": "DATA+BC"},
    {"arm": "E3", "run": "QSF-XYZ-BCPDE-s1234-v3", "input": "xyz", "mode": "DATA+BC+PDE"},
    {"arm": "E4", "run": "QSF-XYZG-DATA-s1234-v3", "input": "xyz+geom", "mode": "DATA"},
    {"arm": "E5", "run": "QSF-XYZG-BC-s1234-v3", "input": "xyz+geom", "mode": "DATA+BC"},
    {"arm": "E6", "run": "QSF-XYZG-BCPDE-s1234-v3", "input": "xyz+geom", "mode": "DATA+BC+PDE"},
)

COLORS = {
    "data_train": "#2f6f9f",
    "data_validation": "#d37b35",
    "physics_train": "#4e8d63",
    "physics_validation": "#a64b4b",
    "total": "#5b6770",
    "wall": "#2f6f9f",
    "inlet": "#4e8d63",
    "outlet": "#d37b35",
    "group": "#8b5b91",
    "continuity": "#2f6f9f",
    "mx": "#4e8d63",
    "my": "#d37b35",
    "mz": "#8b5b91",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rolling_mean(values: np.ndarray, window: int = 25) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    output = np.full(values.shape, np.nan, dtype=np.float64)
    if not len(values):
        return output
    cumulative = np.concatenate([[0.0], np.cumsum(values, dtype=np.float64)])
    for end in range(1, len(values) + 1):
        start = max(0, end - int(window))
        output[end - 1] = (cumulative[end] - cumulative[start]) / (end - start)
    return output


def _late_change(records: list[dict[str, Any]], field: str) -> float:
    count = max(2, int(np.ceil(0.10 * len(records))))
    block = records[-count:]
    midpoint = len(block) // 2
    first = float(np.mean([row[field] for row in block[:midpoint]]))
    second = float(np.mean([row[field] for row in block[midpoint:]]))
    return 100.0 * (second - first) / max(abs(first), 1e-30)


def _configure_axis(ax, *, title: str, late_only: bool = False) -> None:
    ax.set_title(title, fontsize=10, fontweight="normal")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Normalized loss")
    ax.grid(True, which="both", color="0.90", linewidth=0.65)
    ax.set_yscale("log")
    if late_only:
        ax.set_xlim(2251, 2500)
    else:
        ax.set_xlim(1, 2500)
        ax.axvspan(2251, 2500, color="0.88", alpha=0.35, label="last 10%")


def _plot_loss(
    ax,
    epochs: np.ndarray,
    values: np.ndarray,
    *,
    label: str,
    color: str,
    linestyle: str = "-",
    raw: bool = True,
) -> None:
    values = np.asarray(values, dtype=np.float64)
    positive = values > 0.0
    if not bool(np.any(positive)):
        return
    if raw:
        ax.plot(
            epochs[positive],
            values[positive],
            color=color,
            alpha=0.13,
            linewidth=0.55,
            linestyle=linestyle,
        )
    smooth = _rolling_mean(values)
    smooth_positive = smooth > 0.0
    ax.plot(
        epochs[smooth_positive],
        smooth[smooth_positive],
        color=color,
        linewidth=1.8,
        linestyle=linestyle,
        label=label,
    )


def _mark_best(
    ax,
    epochs: np.ndarray,
    values: np.ndarray,
    *,
    color: str,
    label: str = "best val data",
) -> None:
    index = int(np.argmin(values))
    ax.axvline(epochs[index], color=color, linewidth=0.9, linestyle=":")
    ax.scatter([epochs[index]], [values[index]], color=color, s=19, zorder=5)
    ax.annotate(
        f"{label} {values[index]:.3g} @ {epochs[index]}",
        (epochs[index], values[index]),
        xytext=(6, 7),
        textcoords="offset points",
        fontsize=8,
        color=color,
    )


def _load_runs(runs_root: Path) -> dict[str, dict[str, Any]]:
    output = {}
    for arm in ARMS:
        run_dir = runs_root / arm["run"]
        records = _read_jsonl(run_dir / "epoch_progress.jsonl")
        if len(records) != 2500 or records[-1].get("epoch") != 2499:
            raise RuntimeError(f"incomplete epoch log: {run_dir}")
        output[arm["arm"]] = {"meta": arm, "records": records, "run_dir": run_dir}
    return output


def _overview_figure(runs: dict[str, dict[str, Any]]):
    fig, axes = plt.subplots(3, 2, figsize=(14.5, 13.0), constrained_layout=True)
    for ax, arm in zip(axes.flat, ARMS):
        records = runs[arm["arm"]]["records"]
        epochs = np.arange(1, len(records) + 1)
        data_train = np.asarray([row["mean_data_total"] for row in records])
        data_validation = np.asarray([row["validation_data_total"] for row in records])
        physics_train = np.asarray([row["mean_physics_total"] for row in records])
        physics_validation = np.asarray(
            [row["validation_physics_total"] for row in records]
        )
        _configure_axis(
            ax, title=f"{arm['arm']}  {arm['input']}  {arm['mode']}"
        )
        _plot_loss(
            ax,
            epochs,
            data_train,
            label="train data",
            color=COLORS["data_train"],
        )
        _plot_loss(
            ax,
            epochs,
            data_validation,
            label="validation data",
            color=COLORS["data_validation"],
        )
        if arm["mode"] != "DATA":
            _plot_loss(
                ax,
                epochs,
                physics_train,
                label="train physics",
                color=COLORS["physics_train"],
                linestyle="--",
            )
            _plot_loss(
                ax,
                epochs,
                physics_validation,
                label="validation physics",
                color=COLORS["physics_validation"],
                linestyle="--",
            )
        _mark_best(ax, epochs, data_validation, color=COLORS["data_validation"])
        ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.suptitle(
        "V3 six-arm convergence: data loss and physics loss\n"
        "raw epochs (faint) + 25-epoch trailing mean; shaded region = final 10%",
        fontsize=14,
        fontweight="normal",
    )
    return fig


def _arm_figure(arm: dict[str, str], records: list[dict[str, Any]]):
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.8), constrained_layout=True)
    epochs = np.arange(1, len(records) + 1)
    data_train = np.asarray([row["mean_data_total"] for row in records])
    data_validation = np.asarray([row["validation_data_total"] for row in records])
    physics_train = np.asarray([row["mean_physics_total"] for row in records])
    physics_validation = np.asarray(
        [row["validation_physics_total"] for row in records]
    )
    total_train = np.asarray([row["mean_total"] for row in records])
    total_validation = np.asarray([row["validation_total"] for row in records])

    for ax, late_only in ((axes[0, 0], False), (axes[1, 0], True)):
        _configure_axis(
            ax,
            title="Data loss — final 10%" if late_only else "Data loss — all epochs",
            late_only=late_only,
        )
        _plot_loss(
            ax, epochs, data_train, label="train data", color=COLORS["data_train"]
        )
        _plot_loss(
            ax,
            epochs,
            data_validation,
            label="validation data",
            color=COLORS["data_validation"],
        )
        if not late_only:
            _mark_best(ax, epochs, data_validation, color=COLORS["data_validation"])
        ax.legend(frameon=False, fontsize=8)

    for ax, late_only in ((axes[0, 1], False), (axes[1, 1], True)):
        _configure_axis(
            ax,
            title=(
                "Physics / total loss — final 10%"
                if late_only
                else "Physics / total loss — all epochs"
            ),
            late_only=late_only,
        )
        if arm["mode"] == "DATA":
            ax.set_yscale("linear")
            ax.set_ylim(-0.05, 1.05)
            ax.text(
                0.5,
                0.52,
                "Physics loss disabled\nL_phy = 0; L_total = L_data",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=11,
            )
            ax.plot([], [], color=COLORS["total"], label="total = data")
        else:
            _plot_loss(
                ax,
                epochs,
                physics_train,
                label="train physics",
                color=COLORS["physics_train"],
            )
            _plot_loss(
                ax,
                epochs,
                physics_validation,
                label="validation physics",
                color=COLORS["physics_validation"],
            )
            _plot_loss(
                ax,
                epochs,
                total_train,
                label="train total",
                color=COLORS["total"],
                linestyle=":",
                raw=False,
            )
            _plot_loss(
                ax,
                epochs,
                total_validation,
                label="validation total",
                color=COLORS["data_validation"],
                linestyle=":",
                raw=False,
            )
            if not late_only:
                _mark_best(
                    ax,
                    epochs,
                    total_validation,
                    color=COLORS["physics_validation"],
                    label="best val total",
                )
        ax.legend(frameon=False, fontsize=8)

    formula = {
        "DATA": "L = L_data",
        "DATA+BC": "L = L_data + L_BC",
        "DATA+BC+PDE": "L = L_data + L_BC + L_PDE",
    }[arm["mode"]]
    fig.suptitle(
        f"{arm['arm']}  {arm['run']}\n{formula}",
        fontsize=14,
        fontweight="normal",
    )
    return fig


def _bc_component_figure(runs: dict[str, dict[str, Any]]):
    arms = [arm for arm in ARMS if arm["mode"] != "DATA"]
    fig, axes = plt.subplots(4, 2, figsize=(15.5, 16.5), constrained_layout=True)
    fields = (
        ("wall", "mean_wall_raw", "validation_wall_raw"),
        ("inlet", "mean_inlet_raw", "validation_inlet_raw"),
        ("outlet", "mean_outlet_raw", "validation_outlet_raw"),
        ("BC group", "mean_bc_group_raw", "validation_bc_group_raw"),
    )
    for row_index, arm in enumerate(arms):
        records = runs[arm["arm"]]["records"]
        epochs = np.arange(1, len(records) + 1)
        for column, scope in enumerate(("train", "validation")):
            ax = axes[row_index, column]
            _configure_axis(
                ax, title=f"{arm['arm']} {arm['input']} {arm['mode']} — {scope} BC"
            )
            for label, train_field, validation_field in fields:
                field = train_field if scope == "train" else validation_field
                key = "group" if label == "BC group" else label
                _plot_loss(
                    ax,
                    epochs,
                    np.asarray([record[field] for record in records]),
                    label=label,
                    color=COLORS[key],
                )
            ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.suptitle(
        "Boundary-condition loss components\n"
        "L_BC = (L_wall + L_inlet + L_outlet) / 3",
        fontsize=14,
        fontweight="normal",
    )
    return fig


def _pde_component_figure(runs: dict[str, dict[str, Any]]):
    arms = [arm for arm in ARMS if arm["mode"] == "DATA+BC+PDE"]
    fig, axes = plt.subplots(2, 2, figsize=(15.0, 9.0), constrained_layout=True)
    fields = (
        ("continuity", "mean_continuity_raw", "validation_continuity_raw"),
        ("mx", "mean_momentum_x_raw", "validation_momentum_x_raw"),
        ("my", "mean_momentum_y_raw", "validation_momentum_y_raw"),
        ("mz", "mean_momentum_z_raw", "validation_momentum_z_raw"),
        ("PDE group", "mean_pde_group_raw", "validation_pde_group_raw"),
    )
    for row_index, arm in enumerate(arms):
        records = runs[arm["arm"]]["records"]
        epochs = np.arange(1, len(records) + 1)
        for column, scope in enumerate(("train", "validation")):
            ax = axes[row_index, column]
            _configure_axis(
                ax, title=f"{arm['arm']} {arm['input']} — {scope} PDE"
            )
            for label, train_field, validation_field in fields:
                field = train_field if scope == "train" else validation_field
                key = "group" if label == "PDE group" else label
                _plot_loss(
                    ax,
                    epochs,
                    np.asarray([record[field] for record in records]),
                    label=label,
                    color=COLORS[key],
                )
            ax.legend(frameon=False, fontsize=8, ncol=3)
    fig.suptitle(
        "Quasi-steady PDE loss components\n"
        "L_PDE = (L_cont + (L_mx + L_my + L_mz) / 3) / 2",
        fontsize=14,
        fontweight="normal",
    )
    return fig


def _summary_rows(runs: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for arm in ARMS:
        records = runs[arm["arm"]]["records"]
        data_validation = np.asarray([row["validation_data_total"] for row in records])
        physics_validation = np.asarray(
            [row["validation_physics_total"] for row in records]
        )
        best_data_index = int(np.argmin(data_validation))
        if arm["mode"] == "DATA":
            best_physics_epoch = "disabled"
            best_physics_loss = "0"
            last_physics_loss = "0"
            physics_last_to_best = "n/a"
            late_physics_change = "n/a"
        else:
            best_physics_index = int(np.argmin(physics_validation))
            best_physics_epoch = best_physics_index + 1
            best_physics_loss = float(physics_validation[best_physics_index])
            last_physics_loss = float(physics_validation[-1])
            physics_last_to_best = float(
                physics_validation[-1] / physics_validation[best_physics_index]
            )
            late_physics_change = _late_change(records, "validation_physics_total")
        late = records[-250:]
        rows.append(
            {
                **arm,
                "best_validation_data_epoch": best_data_index + 1,
                "best_validation_data_loss": float(data_validation[best_data_index]),
                "last_validation_data_loss": float(data_validation[-1]),
                "data_last_to_best_ratio": float(
                    data_validation[-1] / data_validation[best_data_index]
                ),
                "last_10pct_validation_data_change_percent": _late_change(
                    records, "validation_data_total"
                ),
                "best_validation_physics_epoch": best_physics_epoch,
                "best_validation_physics_loss": best_physics_loss,
                "last_validation_physics_loss": last_physics_loss,
                "physics_last_to_best_ratio": physics_last_to_best,
                "last_10pct_validation_physics_change_percent": late_physics_change,
                "late_gradient_clipping_fraction_mean": float(
                    np.mean([record["gradient_clipping_fraction"] for record in late])
                ),
                "convergence_interpretation": (
                    "optimization plateaued; validation optimum was early and later worsened"
                    if arm["mode"] == "DATA"
                    else "data/physics optimization plateaued; validation data and physics worsened after early optima"
                ),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("outputs/wss_pinn/volume_uvwp_peak_qs_smooth_v3/runs"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/wss_pinn/volume_uvwp_peak_qs_smooth_v3/summary/loss_convergence"
        ),
    )
    args = parser.parse_args()
    runs_root = args.runs_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = _load_runs(runs_root)

    pdf_path = output_dir / "teacher_loss_convergence.pdf"
    generated: list[Path] = []
    with PdfPages(pdf_path) as pdf:
        overview = _overview_figure(runs)
        for suffix in ("png", "svg"):
            path = output_dir / f"teacher_loss_overview.{suffix}"
            overview.savefig(path, dpi=220, bbox_inches="tight")
            generated.append(path)
        pdf.savefig(overview, bbox_inches="tight")
        plt.close(overview)

        for arm in ARMS:
            figure = _arm_figure(arm, runs[arm["arm"]]["records"])
            for suffix in ("png", "svg"):
                path = output_dir / f"{arm['arm'].lower()}_loss_convergence.{suffix}"
                figure.savefig(path, dpi=220, bbox_inches="tight")
                generated.append(path)
            pdf.savefig(figure, bbox_inches="tight")
            plt.close(figure)

        bc_figure = _bc_component_figure(runs)
        for suffix in ("png", "svg"):
            path = output_dir / f"teacher_bc_loss_components.{suffix}"
            bc_figure.savefig(path, dpi=220, bbox_inches="tight")
            generated.append(path)
        pdf.savefig(bc_figure, bbox_inches="tight")
        plt.close(bc_figure)

        pde_figure = _pde_component_figure(runs)
        for suffix in ("png", "svg"):
            path = output_dir / f"teacher_pde_loss_components.{suffix}"
            pde_figure.savefig(path, dpi=220, bbox_inches="tight")
            generated.append(path)
        pdf.savefig(pde_figure, bbox_inches="tight")
        plt.close(pde_figure)
    generated.append(pdf_path)

    rows = _summary_rows(runs)
    csv_path = output_dir / "teacher_loss_convergence_summary.csv"
    json_path = output_dir / "teacher_loss_convergence_summary.json"
    _write_csv(csv_path, rows)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
    generated.extend([csv_path, json_path])

    sources = []
    for arm in ARMS:
        path = runs_root / arm["run"] / "epoch_progress.jsonl"
        sources.append({"path": str(path), "sha256": _sha256(path)})
    manifest = {
        "schema_version": 1,
        "route": "volume_uvwp_peak_qs_smooth_v3",
        "smoothing": "25-epoch trailing mean over raw epoch logs",
        "late_window": "last 10% = epochs 2251-2500",
        "data_loss": "(L_u + L_v + L_w + L_p) / 4",
        "bc_loss": "(L_wall + L_inlet + L_outlet) / 3",
        "pde_loss": "(L_cont + (L_mx + L_my + L_mz) / 3) / 2",
        "physics_loss": "lambda_BC * L_BC + lambda_PDE * L_PDE",
        "sources": sources,
        "outputs": [path.name for path in generated],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {"status": "completed", "output_dir": str(output_dir), "pdf": str(pdf_path)},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
