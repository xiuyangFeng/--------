#!/usr/bin/env python3
"""Pred-vs-CFD scatter with the y = a·x + b fit for postview case packages (same-point wall CSV, all wall nodes).

Convention matches ``training_wss_min.metrics.linear_fit_metrics``: x = CFD truth, y = prediction,
``y_pred = a * y_true + b``; "R² fit" is the regression R² (= Pearson r²), "R² raw" is 1 - SSE/SST on the identity.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


def fit(x, y):
    a, b = np.polyfit(x, y, 1)
    r2_raw = 1.0 - np.sum((y - x) ** 2) / np.sum((x - x.mean()) ** 2)
    r2_fit = 1.0 - np.sum((y - (a * x + b)) ** 2) / np.sum((y - y.mean()) ** 2)
    return float(a), float(b), float(r2_raw), float(r2_fit)


def point_density(x, y, *, max_kde=6000, seed=0, grid_size=None):
    """KDE colors only; optional grid interpolation keeps large clouds practical."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x), size=min(max_kde, len(x)), replace=False)
    kde = gaussian_kde(np.vstack([x[idx], y[idx]]))
    if grid_size is None:
        return kde(np.vstack([x, y]))
    from scipy.interpolate import RegularGridInterpolator
    gx = np.linspace(x.min(), x.max(), grid_size)
    gy = np.linspace(y.min(), y.max(), grid_size)
    xx, yy = np.meshgrid(gx, gy, indexing="ij")
    density_grid = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    return RegularGridInterpolator((gx, gy), density_grid)(np.column_stack([x, y]))


def panel(ax, x, y, *, title, unit, max_kde=6000, seed=0,
          field_name="WSS", point_label="wall nodes", density=None, point_size=4):
    dens = point_density(x, y, max_kde=max_kde, seed=seed) if density is None else density
    if np.shape(dens) != np.shape(x):
        raise ValueError("density must have one color value per point")
    order = np.argsort(dens)
    ax.scatter(x[order], y[order], c=dens[order], s=point_size, cmap="viridis", alpha=0.8, linewidths=0, rasterized=True)
    a, b, r2_raw, r2_fit = fit(x, y)
    lo = min(x.min(), y.min()); hi = max(x.max(), y.max())
    pad = 0.03 * (hi - lo); lo -= pad; hi += pad
    grid = np.linspace(lo, hi, 2)
    ax.plot(grid, grid, color="0.35", lw=1.0, ls="--", label="y = x")
    ax.plot(grid, a * grid + b, color="crimson", lw=1.6, label=f"y = {a:.3f}·x {b:+.3f}")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.set_xlabel(f"CFD {field_name} ({unit})"); ax.set_ylabel(f"Pred {field_name} ({unit})")
    ax.set_title(title, fontsize=10)
    ax.text(0.03, 0.97, f"n = {len(x):,} {point_label}\nR² raw = {r2_raw:.3f}\nR² fit = {r2_fit:.3f}\nslope a = {a:.3f}, b = {b:.3f} {unit}",
            transform=ax.transAxes, va="top", ha="left", fontsize=9, bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
    ax.legend(loc="lower right", fontsize=8, frameon=True)
    ax.grid(alpha=0.25)
    return {"n": int(len(x)), "slope": a, "intercept": b, "r2_raw": r2_raw, "r2_fit": r2_fit}


def case_figure(case_dir: Path, label: str, out_png: Path) -> dict:
    short = label.split("/")[-1]
    df = pd.read_csv(case_dir / "_export" / f"{short}__wall.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4))
    rep = {
        "physical_pa": panel(axes[0], df["wss_cfd"].to_numpy(float), df["wss_pred"].to_numpy(float), title=f"{label} · physical WSS (Pa)", unit="Pa"),
        "normalized_log_z": panel(axes[1], df["true_norm"].to_numpy(float), df["pred_norm"].to_numpy(float), title=f"{label} · normalized target (log_z)", unit="log_z"),
    }
    fig.suptitle(f"{label} · peak_step=1162 · R4 s1234 best · pred = a·CFD + b（全部壁面节点，同点）", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=170); plt.close(fig)
    return rep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-dir", required=True)
    ap.add_argument("--cases", required=True, help="comma-separated labels in ranking order, e.g. AG/fast/LOU_YANG,AG/fast/SUN_ZHI_YU")
    ap.add_argument("--tags", default="best,worst")
    args = ap.parse_args()
    batch = Path(args.batch_dir)
    labels = [c.strip() for c in args.cases.split(",")]
    tags = [t.strip() for t in args.tags.split(",")]
    reports = {}
    fig, axes = plt.subplots(len(labels), 2, figsize=(11, 5.2 * len(labels)))
    axes = np.atleast_2d(axes)
    for i, (label, tag) in enumerate(zip(labels, tags)):
        case_dir = batch / (label.replace("/", "__") + "__peak_wss")
        reports[label] = case_figure(case_dir, label, case_dir / "plots" / "fig_wss_scatter_fit.png")
        short = label.split("/")[-1]
        df = pd.read_csv(case_dir / "_export" / f"{short}__wall.csv")
        panel(axes[i, 0], df["wss_cfd"].to_numpy(float), df["wss_pred"].to_numpy(float), title=f"[{tag}] {label} · physical (Pa)", unit="Pa")
        panel(axes[i, 1], df["true_norm"].to_numpy(float), df["pred_norm"].to_numpy(float), title=f"[{tag}] {label} · normalized (log_z)", unit="log_z")
        (case_dir / "plots" / "fig_wss_scatter_fit_report.json").write_text(json.dumps(reports[label], indent=2), encoding="utf-8")
        m = json.loads((case_dir / "manifest_bundle.json").read_text(encoding="utf-8"))
        m["files"]["scatter_fit"] = "plots/fig_wss_scatter_fit.png"; m["linear_fit_same_point"] = reports[label]
        (case_dir / "manifest_bundle.json").write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
    fig.suptitle("R4 s1234 best · pred = a·CFD + b · 同点全部壁面节点 · 上：最好例，下：最差例", fontsize=12)
    fig.tight_layout()
    out = batch / "fig_scatter_fit_best_vs_worst.png"
    fig.savefig(out, dpi=170); plt.close(fig)
    (batch / "scatter_fit_report.json").write_text(json.dumps(reports, indent=2, ensure_ascii=False), encoding="utf-8")
    for label, rep in reports.items():
        p, n = rep["physical_pa"], rep["normalized_log_z"]
        print(f"{label}: n={p['n']} | Pa: a={p['slope']:.3f} b={p['intercept']:.3f} R2raw={p['r2_raw']:.3f} R2fit={p['r2_fit']:.3f} | log_z: a={n['slope']:.3f} b={n['intercept']:.3f} R2raw={n['r2_raw']:.3f} R2fit={n['r2_fit']:.3f}")
    print("combined ->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
