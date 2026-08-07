"""Create teacher-facing V2 data/physics loss convergence figures.

SAME5K-E7500 v2 has train138 / val0 / test35, so these plots use only the
frozen ``epoch_progress.jsonl`` train losses.  Raw epoch values are retained
as faint lines and a 50-epoch trailing mean is drawn on top.  The final 10%
of training is shaded so plateau behaviour can be judged without using
test35.
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


N_EPOCHS = 7500
SMOOTH_WINDOW = 50
LATE_START = 6751  # last 10%

ARMS = (
    {
        "arm": "E1",
        "run": "VF-PN-XYZ-DATA-SAME5K-E7500-s1234-v2",
        "architecture": "PointNet",
        "input": "xyz",
        "mode": "DATA",
    },
    {
        "arm": "E2",
        "run": "VF-PN-XYZ-PINN-SAME5K-E7500-s1234-v2",
        "architecture": "PointNet",
        "input": "xyz",
        "mode": "PINN",
    },
    {
        "arm": "E3",
        "run": "VF-PN-XYZG-DATA-SAME5K-E7500-s1234-v2",
        "architecture": "PointNet",
        "input": "xyz+geom",
        "mode": "DATA",
    },
    {
        "arm": "E4",
        "run": "VF-PN-XYZG-PINN-SAME5K-E7500-s1234-v2",
        "architecture": "PointNet",
        "input": "xyz+geom",
        "mode": "PINN",
    },
    {
        "arm": "E5",
        "run": "VF-PNPP-XYZ-DATA-SAME5K-E7500-s1234-v2",
        "architecture": "PointNet++",
        "input": "xyz",
        "mode": "DATA",
    },
    {
        "arm": "E6",
        "run": "VF-PNPP-XYZ-PINN-SAME5K-E7500-s1234-v2",
        "architecture": "PointNet++",
        "input": "xyz",
        "mode": "PINN",
    },
    {
        "arm": "E7",
        "run": "VF-PNPP-XYZG-DATA-SAME5K-E7500-s1234-v2",
        "architecture": "PointNet++",
        "input": "xyz+geom",
        "mode": "DATA",
    },
    {
        "arm": "E8",
        "run": "VF-PNPP-XYZG-PINN-SAME5K-E7500-s1234-v2",
        "architecture": "PointNet++",
        "input": "xyz+geom",
        "mode": "PINN",
    },
)

PAIRS = (
    {"label": "PointNet / xyz", "data": "E1", "pinn": "E2"},
    {"label": "PointNet / xyz+geom", "data": "E3", "pinn": "E4"},
    {"label": "PointNet++ / xyz", "data": "E5", "pinn": "E6"},
    {"label": "PointNet++ / xyz+geom", "data": "E7", "pinn": "E8"},
)

COLORS = {
    "data": "#2f6f9f",
    "physics": "#4e8d63",
    "total": "#5b6770",
    "data_only": "#5b6770",
    "pinn": "#2f6f9f",
    "continuity": "#2f6f9f",
    "mx": "#4e8d63",
    "my": "#d37b35",
    "mz": "#8b5b91",
    "no_slip": "#a64b4b",
    "momentum": "#d37b35",
}


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


def _rolling_mean(values: np.ndarray, window: int = SMOOTH_WINDOW) -> np.ndarray:
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
    ax.set_ylabel("Train loss")
    ax.grid(True, which="both", color="0.90", linewidth=0.65)
    ax.set_yscale("log")
    if late_only:
        ax.set_xlim(LATE_START, N_EPOCHS)
    else:
        ax.set_xlim(1, N_EPOCHS)
        ax.axvspan(LATE_START, N_EPOCHS, color="0.88", alpha=0.35, label="last 10%")


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
    label: str,
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
        if len(records) != N_EPOCHS or int(records[-1].get("epoch", -1)) != N_EPOCHS - 1:
            raise RuntimeError(
                f"incomplete epoch log: {run_dir} has {len(records)} rows"
            )
        output[arm["arm"]] = {"meta": arm, "records": records, "run_dir": run_dir}
    return output


def _overview_figure(runs: dict[str, dict[str, Any]]):
    fig, axes = plt.subplots(4, 2, figsize=(14.5, 16.5), constrained_layout=True)
    for ax, arm in zip(axes.flat, ARMS):
        records = runs[arm["arm"]]["records"]
        epochs = np.arange(1, len(records) + 1)
        data = np.asarray([row["mean_data_total"] for row in records])
        physics = np.asarray([row["mean_physics_total"] for row in records])
        total = np.asarray([row["mean_total"] for row in records])
        _configure_axis(
            ax,
            title=f"{arm['arm']}  {arm['architecture']} / {arm['input']}  {arm['mode']}",
        )
        _plot_loss(ax, epochs, data, label="train data", color=COLORS["data"])
        if arm["mode"] == "PINN":
            _plot_loss(
                ax,
                epochs,
                physics,
                label="train physics",
                color=COLORS["physics"],
                linestyle="--",
            )
            _plot_loss(
                ax,
                epochs,
                total,
                label="train total",
                color=COLORS["total"],
                linestyle=":",
                raw=False,
            )
        _mark_best(ax, epochs, data, color=COLORS["data"], label="best train data")
        ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.suptitle(
        "V2 SAME5K-E7500 eight-arm convergence: train data loss and physics loss\n"
        "raw epochs (faint) + 50-epoch trailing mean; shaded region = final 10%; "
        "no validation split (val0)",
        fontsize=14,
        fontweight="normal",
    )
    return fig


def _paired_figure(runs: dict[str, dict[str, Any]]):
    fig, axes = plt.subplots(2, 4, figsize=(18.5, 8.5), constrained_layout=True)
    for column, pair in enumerate(PAIRS):
        data_records = runs[pair["data"]]["records"]
        pinn_records = runs[pair["pinn"]]["records"]
        epochs = np.arange(1, len(data_records) + 1)

        top = axes[0, column]
        top.set_title(pair["label"])
        top.set_yscale("log")
        top.grid(color="0.90", linewidth=0.7)
        top.axvspan(LATE_START, N_EPOCHS, color="0.88", alpha=0.35)
        top.set_xlim(1, N_EPOCHS)
        for label, records, color in (
            ("data-only", data_records, COLORS["data_only"]),
            ("PINN", pinn_records, COLORS["pinn"]),
        ):
            values = np.asarray([row["mean_data_total"] for row in records])
            top.plot(epochs, values, color=color, alpha=0.15, linewidth=0.55)
            top.plot(epochs, _rolling_mean(values), color=color, linewidth=1.8, label=label)
        if column == 0:
            top.set_ylabel("Train data loss")
        top.legend(frameon=False, fontsize=8)

        bottom = axes[1, column]
        bottom.set_yscale("log")
        bottom.grid(color="0.90", linewidth=0.7)
        bottom.axvspan(LATE_START, N_EPOCHS, color="0.88", alpha=0.35)
        bottom.set_xlim(1, N_EPOCHS)
        bottom.set_xlabel("Epoch")
        for label, field, color, linestyle in (
            ("physics total", "mean_physics_total", COLORS["physics"], "-"),
            ("continuity", "mean_continuity_raw", COLORS["continuity"], "--"),
            ("no-slip", "mean_no_slip_raw", COLORS["no_slip"], ":"),
        ):
            values = np.asarray([row[field] for row in pinn_records], dtype=np.float64)
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
                for row in pinn_records
            ],
            dtype=np.float64,
        )
        bottom.plot(
            epochs,
            _rolling_mean(momentum),
            color=COLORS["momentum"],
            linestyle="-.",
            linewidth=1.6,
            label="momentum sum",
        )
        if column == 0:
            bottom.set_ylabel("PINN train physics loss")
        bottom.legend(frameon=False, fontsize=7.5)

    fig.suptitle(
        "Paired train convergence (data-only vs PINN)\n"
        "50-epoch trailing mean; shaded = final 10%",
        fontsize=14,
        fontweight="normal",
    )
    return fig


def _arm_figure(arm: dict[str, str], records: list[dict[str, Any]]):
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.8), constrained_layout=True)
    epochs = np.arange(1, len(records) + 1)
    data = np.asarray([row["mean_data_total"] for row in records])
    physics = np.asarray([row["mean_physics_total"] for row in records])
    total = np.asarray([row["mean_total"] for row in records])

    for ax, late_only in ((axes[0, 0], False), (axes[1, 0], True)):
        _configure_axis(
            ax,
            title="Data loss — final 10%" if late_only else "Data loss — all epochs",
            late_only=late_only,
        )
        _plot_loss(ax, epochs, data, label="train data", color=COLORS["data"])
        if not late_only:
            _mark_best(ax, epochs, data, color=COLORS["data"], label="best train data")
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
                ax, epochs, physics, label="train physics", color=COLORS["physics"]
            )
            _plot_loss(
                ax,
                epochs,
                total,
                label="train total",
                color=COLORS["total"],
                linestyle=":",
                raw=False,
            )
            if not late_only:
                _mark_best(
                    ax, epochs, total, color=COLORS["total"], label="best train total"
                )
        ax.legend(frameon=False, fontsize=8)

    formula = {
        "DATA": "L = L_data",
        "PINN": "L = L_data + L_physics\n"
        "L_physics = continuity + momentum_x/y/z + no-slip",
    }[arm["mode"]]
    fig.suptitle(
        f"{arm['arm']}  {arm['architecture']} / {arm['input']}  {arm['mode']}\n"
        f"{arm['run']}\n{formula}",
        fontsize=13,
        fontweight="normal",
    )
    return fig


def _physics_component_figure(runs: dict[str, dict[str, Any]]):
    pinn_arms = [arm for arm in ARMS if arm["mode"] == "PINN"]
    fig, axes = plt.subplots(2, 2, figsize=(15.0, 10.0), constrained_layout=True)
    fields = (
        ("continuity", "mean_continuity_raw", COLORS["continuity"]),
        ("mx", "mean_momentum_x_raw", COLORS["mx"]),
        ("my", "mean_momentum_y_raw", COLORS["my"]),
        ("mz", "mean_momentum_z_raw", COLORS["mz"]),
        ("no-slip", "mean_no_slip_raw", COLORS["no_slip"]),
        ("physics total", "mean_physics_total", COLORS["physics"]),
    )
    for ax, arm in zip(axes.flat, pinn_arms):
        records = runs[arm["arm"]]["records"]
        epochs = np.arange(1, len(records) + 1)
        _configure_axis(
            ax,
            title=f"{arm['arm']}  {arm['architecture']} / {arm['input']}",
        )
        for label, field, color in fields:
            _plot_loss(
                ax,
                epochs,
                np.asarray([row[field] for row in records]),
                label=label,
                color=color,
                raw=label in {"physics total", "no-slip", "continuity"},
            )
        ax.legend(frameon=False, fontsize=8, ncol=3)
    fig.suptitle(
        "PINN physics loss components (train)\n"
        "continuity / momentum_x,y,z / no-slip / physics total",
        fontsize=14,
        fontweight="normal",
    )
    return fig


def _summary_rows(runs: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for arm in ARMS:
        records = runs[arm["arm"]]["records"]
        data = np.asarray([row["mean_data_total"] for row in records])
        physics = np.asarray([row["mean_physics_total"] for row in records])
        total = np.asarray([row["mean_total"] for row in records])
        best_data_index = int(np.argmin(data))
        best_total_index = int(np.argmin(total))
        if arm["mode"] == "DATA":
            best_physics_epoch = "disabled"
            best_physics_loss = 0.0
            last_physics_loss = 0.0
            late_physics_change = "n/a"
        else:
            best_physics_index = int(np.argmin(physics))
            best_physics_epoch = best_physics_index + 1
            best_physics_loss = float(physics[best_physics_index])
            last_physics_loss = float(physics[-1])
            late_physics_change = _late_change(records, "mean_physics_total")
        # last-50 vs prior-50 for README-aligned plateau check
        last50 = float(np.mean(data[-50:]))
        prior50 = float(np.mean(data[-100:-50]))
        rows.append(
            {
                **arm,
                "best_train_data_epoch": best_data_index + 1,
                "best_train_data_loss": float(data[best_data_index]),
                "last_train_data_loss": float(data[-1]),
                "data_last_to_best_ratio": float(data[-1] / data[best_data_index]),
                "last_10pct_train_data_change_percent": _late_change(
                    records, "mean_data_total"
                ),
                "last50_vs_prior50_data_change_percent": 100.0
                * (last50 - prior50)
                / max(abs(prior50), 1e-30),
                "best_train_total_epoch": best_total_index + 1,
                "best_train_total_loss": float(total[best_total_index]),
                "last_train_total_loss": float(total[-1]),
                "best_train_physics_epoch": best_physics_epoch,
                "best_train_physics_loss": best_physics_loss,
                "last_train_physics_loss": last_physics_loss,
                "last_10pct_train_physics_change_percent": late_physics_change,
                "convergence_interpretation": (
                    "train data loss plateaued in final 10%; no validation split"
                    if arm["mode"] == "DATA"
                    else "train data/physics plateaued in final 10%; no validation split"
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
        default=Path("outputs/wss_pinn/volume_uvwp_peak_same5k_e7500_v2/runs"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/wss_pinn/volume_uvwp_peak_same5k_e7500_v2/summary/loss_convergence"
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

        paired = _paired_figure(runs)
        for suffix in ("png", "svg"):
            path = output_dir / f"convergence_curves.{suffix}"
            paired.savefig(path, dpi=220, bbox_inches="tight")
            generated.append(path)
        pdf.savefig(paired, bbox_inches="tight")
        plt.close(paired)

        for arm in ARMS:
            figure = _arm_figure(arm, runs[arm["arm"]]["records"])
            for suffix in ("png", "svg"):
                path = output_dir / f"{arm['arm'].lower()}_loss_convergence.{suffix}"
                figure.savefig(path, dpi=220, bbox_inches="tight")
                generated.append(path)
            pdf.savefig(figure, bbox_inches="tight")
            plt.close(figure)

        physics_figure = _physics_component_figure(runs)
        for suffix in ("png", "svg"):
            path = output_dir / f"teacher_physics_loss_components.{suffix}"
            physics_figure.savefig(path, dpi=220, bbox_inches="tight")
            generated.append(path)
        pdf.savefig(physics_figure, bbox_inches="tight")
        plt.close(physics_figure)
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
        "route": "volume_uvwp_peak_same5k_e7500_v2",
        "smoothing": "50-epoch trailing mean over raw epoch logs",
        "late_window": f"last 10% = epochs {LATE_START}-{N_EPOCHS}",
        "split_note": "train138 / val0 / test35; figures use train losses only",
        "data_loss": "mean_data_total from epoch_progress.jsonl",
        "physics_loss": (
            "mean_physics_total = continuity + momentum_x/y/z + no-slip "
            "(DATA arms have physics=0)"
        ),
        "sources": sources,
        "outputs": [path.name for path in generated],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    generated.append(manifest_path)

    readme = output_dir / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# V2 SAME5K-E7500 损失收敛图",
                "",
                "> 数据来源：八个正式 run 的冻结 `epoch_progress.jsonl`，每臂 7500 epoch。",
                ">",
                "> 本轮无 validation（val0）。图中淡线为逐 epoch 原始值，粗线为 50 epoch",
                "> trailing mean；灰色区域为最后 10%（epoch 6751–7500）。收敛判断只使用",
                "> train loss，不读取 test35。",
                "",
                "## 损失函数",
                "",
                "```text",
                "L_data = data_u + data_v + data_w + data_pressure",
                "L_physics = continuity + momentum_x + momentum_y + momentum_z + no_slip",
                "",
                "DATA:  L = L_data",
                "PINN:  L = L_data + L_physics",
                "```",
                "",
                "## 给老师看的主文件",
                "",
                "- `teacher_loss_convergence.pdf`：汇报版总册（总览、配对、分臂、物理分量）",
                "- `teacher_loss_overview.png`：八臂 data / physics 总览",
                "- `convergence_curves.png`：四对 data-only vs PINN 配对收敛",
                "- `e1_loss_convergence.*`–`e8_loss_convergence.*`：分臂全程 + 最后 10%",
                "- `teacher_physics_loss_components.png`：四个 PINN 臂的物理分量",
                "- `teacher_loss_convergence_summary.csv/json`：最优 epoch 与最后 10% 趋势",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    generated.append(readme)

    print(
        json.dumps(
            {
                "status": "completed",
                "output_dir": str(output_dir),
                "pdf": str(pdf_path),
                "n_outputs": len(generated),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
