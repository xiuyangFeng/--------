#!/usr/bin/env python3
"""P0-1 (A-Opt-01): 与 A-Main-01 对比可视化。数据来源：各 run 的 summary.json。"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs" / "field" / "plots"

MAIN_RUNS = [
    ROOT / "outputs/field/field_transformer_coord_t_bc_geom_wall_split_AG_v1_seed1_20260322_064925/summary.json",
    ROOT / "outputs/field/field_transformer_coord_t_bc_geom_wall_split_AG_v1_seed2_20260322_141028/summary.json",
    ROOT / "outputs/field/field_transformer_coord_t_bc_geom_wall_split_AG_v1_seed3_20260323_004454/summary.json",
]
OPT_RUNS = [
    ROOT / "outputs/field/field_transformer_coord_t_bc_geom_wall_tw22205_split_AG_v1_seed1_20260325_135407/summary.json",
    ROOT / "outputs/field/field_transformer_coord_t_bc_geom_wall_tw22205_split_AG_v1_seed2_20260325_135428/summary.json",
    ROOT / "outputs/field/field_transformer_coord_t_bc_geom_wall_tw22205_split_AG_v1_seed3_20260325_232853/summary.json",
]

# English labels for reliable matplotlib rendering across environments
METRIC_KEYS = [
    ("rmse", "RMSE (u,v,w,p)"),
    ("rmse_vel_mag", r"RMSE $\|\mathbf{v}\|$"),
    ("rmse_u", "RMSE u"),
    ("rmse_v", "RMSE v"),
    ("rmse_w", "RMSE w"),
    ("rmse_p", "RMSE p"),
    ("r2_u", r"$R^2$ u"),
    ("r2_v", r"$R^2$ v"),
    ("r2_w", r"$R^2$ w"),
    ("r2_p", r"$R^2$ p"),
]


def load_metrics(paths: list[Path]) -> dict[str, np.ndarray]:
    rows = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        rows.append(d["test_metrics"])
    keys = {k for r in rows for k in r}
    return {k: np.array([r[k] for r in rows], dtype=np.float64) for k in keys}


def plot_bar_comparison(
    main_m: dict[str, np.ndarray],
    opt_m: dict[str, np.ndarray],
    out_path: Path,
) -> None:
    labels = [lbl for _, lbl in METRIC_KEYS]
    keys = [k for k, _ in METRIC_KEYS]
    x = np.arange(len(keys))
    w = 0.35
    main_mean = np.array([main_m[k].mean() for k in keys])
    main_std = np.array([main_m[k].std(ddof=1) if len(main_m[k]) > 1 else 0 for k in keys])
    opt_mean = np.array([opt_m[k].mean() for k in keys])
    opt_std = np.array([opt_m[k].std(ddof=1) if len(opt_m[k]) > 1 else 0 for k in keys])

    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.bar(x - w / 2, main_mean, w, yerr=main_std, label="A-Main-01", capsize=3, color="#4C72B0", ecolor="#333")
    ax.bar(x + w / 2, opt_mean, w, yerr=opt_std, label="A-Opt-01 (P0-1)", capsize=3, color="#DD8452", ecolor="#333")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=22, ha="right")
    ax.set_ylabel("Metric value")
    ax.set_title(
        "P0-1 A-Opt-01: target_weights=[2,2,2,0.5] vs A-Main-01\n"
        "Test set: mean ± std over 3 seeds"
    )
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_val_loss_curves(out_path: Path) -> None:
    main_hist = [
        p.parent / "history.csv" for p in MAIN_RUNS
    ]
    opt_hist = [
        p.parent / "history.csv" for p in OPT_RUNS
    ]
    fig, ax = plt.subplots(figsize=(9, 5))
    colors_main = ["#4C72B0", "#6BA3D6", "#8FB8E8"]
    colors_opt = ["#DD8452", "#E8A87C", "#F4C4A8"]
    for i, csv_p in enumerate(main_hist):
        epochs, vals = [], []
        with open(csv_p, newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                epochs.append(int(row["epoch"]))
                vals.append(float(row["val_loss"]))
        ax.plot(epochs, vals, color=colors_main[i], alpha=0.85, lw=1.4, label=f"Main seed {i+1}")
    for i, csv_p in enumerate(opt_hist):
        epochs, vals = [], []
        with open(csv_p, newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                epochs.append(int(row["epoch"]))
                vals.append(float(row["val_loss"]))
        ax.plot(
            epochs,
            vals,
            color=colors_opt[i],
            alpha=0.9,
            lw=1.4,
            linestyle="--",
            label=f"Opt-01 seed {i+1}",
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("val_loss (weighted training objective)")
    ax.set_title(
        "Validation loss curves\n"
        "Note: val_loss scales differ between runs because target_weights differ; do not compare magnitudes across colors."
    )
    ax.legend(ncol=2, fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_summary_csv(main_m: dict[str, np.ndarray], opt_m: dict[str, np.ndarray], out_path: Path) -> None:
    lines = ["metric,A-Main-01_mean,A-Main-01_std,A-Opt-01_mean,A-Opt-01_std,delta_mean,opt01_minus_main"]
    for key, _ in METRIC_KEYS:
        mm, ms = float(main_m[key].mean()), float(main_m[key].std(ddof=1) if len(main_m[key]) > 1 else 0)
        om, os_ = float(opt_m[key].mean()), float(opt_m[key].std(ddof=1) if len(opt_m[key]) > 1 else 0)
        delta = om - mm
        lines.append(f"{key},{mm:.6f},{ms:.6f},{om:.6f},{os_:.6f},{delta:.6f},{delta:+.6f}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    main_m = load_metrics(MAIN_RUNS)
    opt_m = load_metrics(OPT_RUNS)
    plot_bar_comparison(main_m, opt_m, OUT_DIR / "fig_P0-1_A-Opt-01_vs_A-Main-01_metrics.png")
    plot_val_loss_curves(OUT_DIR / "fig_P0-1_A-Opt-01_vs_A-Main-01_val_loss_curves.png")
    write_summary_csv(main_m, opt_m, OUT_DIR / "fig_P0-1_A-Opt-01_summary.csv")
    print("Wrote:", OUT_DIR / "fig_P0-1_A-Opt-01_vs_A-Main-01_metrics.png")
    print("Wrote:", OUT_DIR / "fig_P0-1_A-Opt-01_vs_A-Main-01_val_loss_curves.png")
    print("Wrote:", OUT_DIR / "fig_P0-1_A-Opt-01_summary.csv")


if __name__ == "__main__":
    main()
