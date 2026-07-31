"""
全量壁面点：pred WSS vs truth WSS 分析图组
========================================
默认 --sample-count 0（全部 CFD 壁面节点），输出：
  - scatter_full.png          全量散点
  - analysis_panels.png       多面板诊断（残差/Bland-Altman/空间/方向/分箱）
  - metrics.json              指标
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

_VIZ = Path(__file__).resolve().parent
_SRC = _VIZ.parent / "src"
_REPO = _VIZ.parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_SRC))

from calculate_wss_cfd import run_case


def save_full_scatter(truth: np.ndarray, pred: np.ndarray, report: dict, path: Path) -> None:
    valid = np.isfinite(truth) & np.isfinite(pred)
    y, p = truth[valid], pred[valid]
    alpha = float(report["alpha"])
    lim = float(max(y.max(), p.max(), 1e-6)) * 1.05

    fig, ax = plt.subplots(figsize=(6.8, 6.5), dpi=150)
    # density-friendly: hexbin for full wall
    hb = ax.hexbin(y, p, gridsize=80, cmap="viridis", mincnt=1, bins="log")
    cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("log10(count)")
    ax.plot([0, lim], [0, lim], "w--", lw=1.4, label="y = x")
    ax.plot(
        [0, lim],
        [0, lim / max(alpha, 1e-12)],
        color="#ff7f0e",
        lw=1.6,
        label=f"y = x/{alpha:.3f}  (truth≈α·pred)",
    )
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("CFD truth WSS (Pa)")
    ax.set_ylabel("Velocity→WSS pred (Pa)")
    ax.set_title(
        f"{report['case']}  step={report['step']}  FULL wall n={report['n_valid']}\n"
        f"raw R²={report['raw_r2']:.3f}  scaled R²={report['scaled_r2']:.3f}  "
        f"Spearman={report['spearman']:.3f}  α={alpha:.3f}",
        fontsize=10,
    )
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.25)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_analysis_panels(
    truth: np.ndarray,
    pred: np.ndarray,
    truth_vec: np.ndarray,
    pred_vec: np.ndarray,
    coords_mm: np.ndarray,
    report: dict,
    path: Path,
) -> None:
    valid = np.isfinite(truth) & np.isfinite(pred) & np.isfinite(pred_vec).all(axis=1)
    y, p = truth[valid], pred[valid]
    tv, pv = truth_vec[valid], pred_vec[valid]
    xyz = coords_mm[valid]
    err = p - y
    abs_err = np.abs(err)
    rel = abs_err / np.maximum(np.abs(y), 1e-6)
    alpha = float(report["alpha"])
    cos = np.einsum("ij,ij->i", pv, tv) / np.maximum(
        np.linalg.norm(pv, axis=1) * np.linalg.norm(tv, axis=1), 1e-12
    )

    centered = xyz - xyz.mean(0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    uv = centered @ vt[:2].T

    fig = plt.figure(figsize=(14.5, 10.5), dpi=150)
    fig.suptitle(
        f"{report['case']}  peak step={report['step']}  FULL n={len(y)}  "
        f"rawR²={report['raw_r2']:.3f}  scaledR²={report['scaled_r2']:.3f}  α={alpha:.3f}",
        fontsize=11,
    )

    # 1 scatter points (subsample overlay on hex already separate)
    ax = fig.add_subplot(2, 3, 1)
    lim = float(max(y.max(), p.max(), 1e-6)) * 1.05
    ax.scatter(y, p, s=4, alpha=0.15, c="#1f77b4", edgecolors="none")
    ax.plot([0, lim], [0, lim], "k--", lw=1)
    ax.plot([0, lim], [0, lim / max(alpha, 1e-12)], color="#d62728", lw=1.2)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("truth WSS (Pa)")
    ax.set_ylabel("pred WSS (Pa)")
    ax.set_title("1. Scatter (all wall nodes)")
    ax.grid(True, alpha=0.25)

    # 2 residual vs truth
    ax = fig.add_subplot(2, 3, 2)
    ax.scatter(y, err, s=4, alpha=0.15, c="#ff7f0e", edgecolors="none")
    ax.axhline(0, color="k", lw=1)
    ax.set_xlabel("truth WSS (Pa)")
    ax.set_ylabel("pred − truth (Pa)")
    ax.set_title(
        f"2. Residual vs truth\n"
        f"bias={err.mean():.3f}Pa  |err| p50={np.median(abs_err):.3f} p95={np.quantile(abs_err,0.95):.3f}"
    )
    ax.grid(True, alpha=0.25)

    # 3 Bland-Altman
    ax = fig.add_subplot(2, 3, 3)
    mean_yp = 0.5 * (y + p)
    ax.scatter(mean_yp, err, s=4, alpha=0.15, c="#2ca02c", edgecolors="none")
    mu, sd = float(err.mean()), float(err.std())
    ax.axhline(mu, color="#d62728", lw=1.2, label=f"mean={mu:.3f}")
    ax.axhline(mu + 1.96 * sd, color="k", ls="--", lw=1, label=f"±1.96σ")
    ax.axhline(mu - 1.96 * sd, color="k", ls="--", lw=1)
    ax.set_xlabel("(truth+pred)/2 (Pa)")
    ax.set_ylabel("pred − truth (Pa)")
    ax.set_title("3. Bland–Altman")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    # 4 spatial |error|
    ax = fig.add_subplot(2, 3, 4)
    sc = ax.scatter(uv[:, 0], uv[:, 1], c=abs_err, s=4, cmap="magma", alpha=0.85)
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="|err| Pa")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("PCA-u (mm)")
    ax.set_ylabel("PCA-v (mm)")
    ax.set_title("4. Spatial |pred−truth|")
    ax.grid(True, alpha=0.2)

    # 5 direction cosine
    ax = fig.add_subplot(2, 3, 5)
    ax.hist(cos, bins=60, color="#9467bd", alpha=0.9)
    ax.axvline(np.median(cos), color="#d62728", lw=1.2, label=f"p50={np.median(cos):.4f}")
    ax.set_xlabel("direction cosine (pred·truth)")
    ax.set_ylabel("count")
    ax.set_title("5. WSS vector direction agreement")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    # 6 binned relative error by truth quantile
    ax = fig.add_subplot(2, 3, 6)
    edges = np.quantile(y, np.linspace(0, 1, 11))
    edges = np.unique(edges)
    centers, med_rel, med_abs = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (y >= lo) & (y <= hi if hi == edges[-1] else y < hi)
        if m.sum() < 20:
            continue
        centers.append(0.5 * (lo + hi))
        med_rel.append(float(np.median(rel[m])))
        med_abs.append(float(np.median(abs_err[m])))
    ax.plot(centers, med_rel, "o-", color="#1f77b4", label="median |err|/truth")
    ax.set_xlabel("truth WSS bin center (Pa)")
    ax.set_ylabel("median relative |err|")
    ax.set_title("6. Relative error by truth decile")
    ax.grid(True, alpha=0.25)
    ax2 = ax.twinx()
    ax2.plot(centers, med_abs, "s--", color="#d62728", alpha=0.8, label="median |err| Pa")
    ax2.set_ylabel("median |err| (Pa)", color="#d62728")
    # combine legends
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper right")

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case-dir",
        type=str,
        default="/public/newhome/cy/Digital_twin/GNN/data_new/AG/slow/LIU_JIN_LIANG",
    )
    parser.add_argument("--sample-count", type=int, default=0, help="0=全部壁面点")
    parser.add_argument("--neighbors", type=int, default=64)
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--normals", choices=["pca", "stl"], default="pca")
    parser.add_argument("--viscosity", choices=["carreau", "newton"], default="carreau")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="输出目录；默认写到 explore/01_pred_vs_truth/<case>/",
    )
    args = parser.parse_args()

    case_dir = Path(args.case_dir).resolve()
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = (
            Path("/public/newhome/cy/Digital_twin/GNN/outputs/wss_pinn/audits/cfd_velocity_wss_explore")
            / "01_pred_vs_truth"
            / case_dir.name
        )

    # need arrays + coords: run_case returns arrays but not coords; re-run lightly via return_arrays
    # extend: call run_case then reload coords for spatial plot
    import data_loader_cfd as dl
    from calculate_wss_cfd import resolve_step

    report = run_case(
        case_dir,
        None,
        args.sample_count,
        args.neighbors,
        args.degree,
        args.viscosity,
        False,
        args.seed,
        args.normals,
        prefer_peak=True,
        return_arrays=True,
    )
    arrays = report.pop("_arrays")
    step, _ = resolve_step(case_dir, None, prefer_peak=True)
    wall = dl.load_wall(dl.step_path(case_dir, step, "ascii"))
    n_wall = len(wall["coords_mm"])
    if 0 < args.sample_count < n_wall:
        rng = np.random.default_rng(args.seed)
        idx = np.sort(rng.choice(n_wall, args.sample_count, replace=False))
    else:
        idx = np.arange(n_wall)
    coords = wall["coords_mm"][idx]

    out_dir.mkdir(parents=True, exist_ok=True)
    scatter_path = out_dir / "scatter_full_hexbin.png"
    panels_path = out_dir / "analysis_panels.png"
    # also classic scatter without hex for teachers who prefer points
    from calculate_wss_cfd import save_scatter_plot

    point_scatter = out_dir / "scatter_full_points.png"
    save_scatter_plot(arrays["truth_mag"], arrays["pred_mag"], report, point_scatter)
    save_full_scatter(arrays["truth_mag"], arrays["pred_mag"], report, scatter_path)
    save_analysis_panels(
        arrays["truth_mag"],
        arrays["pred_mag"],
        arrays["truth_vec"],
        arrays["pred_vec"],
        coords,
        report,
        panels_path,
    )

    report_out = dict(report)
    report_out["n_points_plotted"] = int(np.isfinite(arrays["truth_mag"]).sum())
    report_out["figures"] = {
        "scatter_full_points": str(point_scatter),
        "scatter_full_hexbin": str(scatter_path),
        "analysis_panels": str(panels_path),
    }
    json_path = out_dir / "metrics.json"
    json_path.write_text(json.dumps(report_out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report_out, indent=2, ensure_ascii=False))
    print(f"\n输出目录: {out_dir}")


if __name__ == "__main__":
    main()
