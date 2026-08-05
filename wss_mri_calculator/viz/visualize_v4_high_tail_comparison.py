"""Compare frozen V4 final with the selected high-tail candidate for one case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np


VIZ = Path(__file__).resolve().parent
PACKAGE = VIZ.parent
ROOT = PACKAGE.parent
DEFAULT_CASE = "ILO/YU_XIANG_SHENG-1/before"
DEFAULT_PREDICTIONS = (
    ROOT
    / "outputs/wss_mri_calculator/pointcloud_surface_mls_v4/"
    "test35_high_tail_v2_peak30_predictions.npz"
)
DEFAULT_RESULTS = (
    ROOT
    / "outputs/wss_mri_calculator/pointcloud_surface_mls_v4/"
    "test35_high_tail_v2_peak30.json"
)
DEFAULT_OUT = (
    ROOT
    / "outputs/wss_mri_calculator/pointcloud_surface_mls_v4/00_quicklook/"
    "12_high_tail_comparison_YU_XIANG_SHENG.png"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default=DEFAULT_CASE)
    parser.add_argument("--predictions", default=str(DEFAULT_PREDICTIONS))
    parser.add_argument("--results", default=str(DEFAULT_RESULTS))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--zoom-radius-mm", type=float, default=15.0)
    args = parser.parse_args()

    with np.load(args.predictions, allow_pickle=False) as source:
        mask = np.asarray(source["point_case"]) == args.case
        wall = np.asarray(source["wall_mm"][mask], dtype=np.float64)
        truth = np.asarray(source["truth_mag"][mask], dtype=np.float64)
        current = np.linalg.norm(
            np.asarray(source["current_wss_vec"][mask], dtype=np.float64), axis=1
        )
        tail = np.linalg.norm(
            np.asarray(source["high_tail_wss_vec"][mask], dtype=np.float64), axis=1
        )
    results = json.loads(Path(args.results).read_text(encoding="utf-8"))
    row = next(item for item in results["cases"] if item["canonical_id"] == args.case)
    current_report = row["methods"]["current_safe"]
    tail_report = row["methods"]["high_tail_w10_s80"]

    threshold = float(np.quantile(truth, 0.90))
    high = truth >= threshold
    centered = wall - np.mean(wall, axis=0, keepdims=True)
    _, _, axes = np.linalg.svd(centered, full_matrices=False)
    uv = centered @ axes[:2].T
    peak = int(np.argmax(truth))
    peak_uv = uv[peak]
    zoom_radius = float(args.zoom_radius_mm)
    zoom = (
        (np.abs(uv[:, 0] - peak_uv[0]) <= zoom_radius)
        & (np.abs(uv[:, 1] - peak_uv[1]) <= zoom_radius)
    )
    zoom_high = zoom & high
    shared_max = float(max(np.max(truth[high]), np.max(current[high]), np.max(tail[high])))
    norm = LogNorm(vmin=max(threshold, 1e-6), vmax=shared_max)
    improvement = np.abs(truth - current) - np.abs(truth - tail)
    gain_limit = float(max(np.quantile(np.abs(improvement[high]), 0.98), 1e-6))

    fig, panels = plt.subplots(2, 4, figsize=(19.0, 9.8), dpi=180)
    background = "#c9cdd2"
    current_color = "#167d8d"
    tail_color = "#c84b31"
    shared = None
    for ax, title, values in (
        (panels[0, 0], "A. CFD truth high-WSS", truth),
        (panels[0, 1], "B. Current V4 final", current),
        (panels[0, 2], "C. High-tail candidate", tail),
    ):
        ax.scatter(uv[:, 0], uv[:, 1], s=8, color=background, alpha=0.28, edgecolors="none")
        shared = ax.scatter(
            uv[high, 0], uv[high, 1], c=values[high], s=35,
            cmap="turbo", norm=norm, edgecolors="none",
        )
        ax.scatter([peak_uv[0]], [peak_uv[1]], marker="*", s=180, facecolor="none", edgecolor="black", linewidth=1.0)
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_axis_off()
    ax = panels[0, 3]
    ax.scatter(uv[:, 0], uv[:, 1], s=8, color=background, alpha=0.28, edgecolors="none")
    gain_image = ax.scatter(
        uv[high, 0], uv[high, 1], c=improvement[high], s=35,
        cmap="RdBu_r", vmin=-gain_limit, vmax=gain_limit, edgecolors="none",
    )
    ax.scatter([peak_uv[0]], [peak_uv[1]], marker="*", s=180, facecolor="none", edgecolor="black", linewidth=1.0)
    ax.set_title("D. Absolute-error improvement\nred = high-tail candidate is better")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_axis_off()
    if shared is not None:
        fig.colorbar(shared, ax=panels[0, :3], shrink=0.77, label="WSS (Pa, log scale)")
    fig.colorbar(gain_image, ax=panels[0, 3], shrink=0.77, label="|current error| − |tail error| (Pa)")

    for ax, title, values in (
        (panels[1, 0], "E. Peak-region current V4 final", current),
        (panels[1, 1], "F. Peak-region high-tail candidate", tail),
    ):
        ax.scatter(uv[zoom, 0], uv[zoom, 1], s=18, color=background, alpha=0.32, edgecolors="none")
        ax.scatter(
            uv[zoom_high, 0], uv[zoom_high, 1], c=values[zoom_high], s=55,
            cmap="turbo", norm=norm, edgecolors="white", linewidths=0.25,
        )
        ax.scatter([peak_uv[0]], [peak_uv[1]], marker="*", s=200, facecolor="none", edgecolor="black", linewidth=1.1)
        ax.set_xlim(peak_uv[0] - zoom_radius, peak_uv[0] + zoom_radius)
        ax.set_ylim(peak_uv[1] - zoom_radius, peak_uv[1] + zoom_radius)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title)
        ax.set_xlabel("wall PCA-1 (mm)")
        ax.set_ylabel("wall PCA-2 (mm)")
        ax.grid(alpha=0.18)

    ax = panels[1, 2]
    limit = float(max(np.max(truth[high]), np.max(current[high]), np.max(tail[high])) * 1.04)
    ax.scatter(truth[high], current[high], s=30, color=current_color, alpha=0.55, edgecolors="none", label=f"current: high R²={current_report['high_r2']:.3f}")
    ax.scatter(truth[high], tail[high], s=33, color=tail_color, alpha=0.72, edgecolors="none", label=f"high-tail: high R²={tail_report['high_r2']:.3f}")
    ax.plot([0, limit], [0, limit], "k--", lw=1.0)
    ax.scatter([truth[peak]], [tail[peak]], marker="*", s=180, color="#ff9d00", edgecolors="black", linewidths=0.8, label=f"peak: {truth[peak]:.1f} → {tail[peak]:.1f} Pa")
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"G. Pointwise high-WSS accuracy (q90={threshold:.2f} Pa)")
    ax.set_xlabel("CFD truth WSS (Pa)")
    ax.set_ylabel("predicted WSS (Pa)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(alpha=0.20)

    ax = panels[1, 3]
    order = np.argsort(truth[high])
    rank = np.arange(1, int(np.sum(high)) + 1)
    ax.plot(rank, truth[high][order], color="#222222", lw=1.7, label="CFD truth")
    ax.plot(rank, current[high][order], color=current_color, lw=1.3, label="current V4 final")
    ax.plot(rank, tail[high][order], color=tail_color, lw=1.5, label="high-tail candidate")
    ax.set_title(
        "H. High-WSS tail\n"
        f"peak underestimation: {100*current_report['peak_underestimate_fraction']:.1f}% → "
        f"{100*tail_report['peak_underestimate_fraction']:.1f}%"
    )
    ax.set_xlabel("truth-sorted high-WSS point rank")
    ax.set_ylabel("WSS (Pa)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.20)

    fig.suptitle(
        f"High-tail V4 comparison — {args.case}\n"
        f"overall R² {current_report['raw_r2']:.3f} → {tail_report['raw_r2']:.3f}; "
        f"high-WSS R² {current_report['high_r2']:.3f} → {tail_report['high_r2']:.3f}; "
        f"high-WSS NRMSE {current_report['high_nrmse']:.3f} → {tail_report['high_nrmse']:.3f}",
        fontsize=13.2,
    )
    fig.subplots_adjust(left=0.035, right=0.97, bottom=0.07, top=0.88, hspace=0.30, wspace=0.23)
    output = Path(args.out).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
