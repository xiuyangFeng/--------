"""Visualize high-WSS accuracy for one frozen Surface-MLS V4 case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from scipy.stats import spearmanr


VIZ = Path(__file__).resolve().parent
PACKAGE = VIZ.parent
ROOT = PACKAGE.parent

DEFAULT_CASE = "ILO/YU_XIANG_SHENG-1/before"
DEFAULT_CACHE = (
    ROOT
    / "outputs/wss_mri_calculator/pointcloud_surface_mls_v4/"
    "point_cache_test35_s1200/ILO__YU_XIANG_SHENG-1__before.npz"
)
DEFAULT_PREDICTIONS = (
    ROOT
    / "outputs/wss_mri_calculator/pointcloud_surface_mls_v4/"
    "test35_calibrated_predictions_s1200.npz"
)
DEFAULT_OUT = (
    ROOT
    / "outputs/wss_mri_calculator/pointcloud_surface_mls_v4/00_quicklook/"
    "11_high_wss_accuracy_YU_XIANG_SHENG.png"
)


def metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    variance = float(np.sum((truth - np.mean(truth)) ** 2))
    error = prediction - truth
    rmse = float(np.sqrt(np.mean(error**2)))
    return {
        "r2": 1.0 - float(np.sum(error**2)) / max(variance, 1e-30),
        "mae_pa": float(np.mean(np.abs(error))),
        "rmse_pa": rmse,
        "nrmse": rmse / max(float(np.mean(np.abs(truth))), 1e-12),
        "bias_pa": float(np.mean(error)),
        "spearman": float(spearmanr(truth, prediction).statistic),
        "predicted_max_pa": float(np.max(prediction)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default=DEFAULT_CASE)
    parser.add_argument("--cache", default=str(DEFAULT_CACHE))
    parser.add_argument("--predictions", default=str(DEFAULT_PREDICTIONS))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--high-quantile", type=float, default=0.90)
    parser.add_argument("--zoom-radius-mm", type=float, default=15.0)
    args = parser.parse_args()

    cache_path = Path(args.cache).resolve()
    predictions_path = Path(args.predictions).resolve()
    output_path = Path(args.out).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with np.load(cache_path, allow_pickle=False) as source:
        cached_case = str(source["canonical_id"].item())
        if cached_case != args.case:
            raise ValueError(f"cache case {cached_case!r} does not match {args.case!r}")
        step = int(source["step"])
        wall_mm = np.asarray(source["wall_mm"], dtype=np.float64)
        truth = np.asarray(source["truth_mag"], dtype=np.float64)
        sample_index = np.asarray(source["sample_index"], dtype=np.int64)

    with np.load(predictions_path, allow_pickle=False) as source:
        case_mask = np.asarray(source["point_case"]) == args.case
        prediction_wall = np.asarray(source["wall_mm"][case_mask], dtype=np.float64)
        if not np.allclose(wall_mm, prediction_wall):
            raise RuntimeError("cache and calibrated-prediction wall order mismatch")
        base = np.linalg.norm(
            np.asarray(source["base_wss_vec"][case_mask], dtype=np.float64), axis=1
        )
        calibrated = np.linalg.norm(
            np.asarray(source["calibrated_wss_vec"][case_mask], dtype=np.float64), axis=1
        )

    threshold = float(np.quantile(truth, args.high_quantile))
    high = truth >= threshold
    high_truth = truth[high]
    high_base = base[high]
    high_calibrated = calibrated[high]
    base_metrics = metrics(high_truth, high_base)
    calibrated_metrics = metrics(high_truth, high_calibrated)

    base_top = base >= np.quantile(base, args.high_quantile)
    calibrated_top = calibrated >= np.quantile(calibrated, args.high_quantile)
    base_intersection = int(np.sum(high & base_top))
    calibrated_intersection = int(np.sum(high & calibrated_top))
    base_union = int(np.sum(high | base_top))
    calibrated_union = int(np.sum(high | calibrated_top))
    base_jaccard = base_intersection / max(base_union, 1)
    calibrated_jaccard = calibrated_intersection / max(calibrated_union, 1)

    centered = wall_mm - wall_mm.mean(axis=0, keepdims=True)
    _, _, pca_axes = np.linalg.svd(centered, full_matrices=False)
    wall_uv = centered @ pca_axes[:2].T
    peak_row = int(np.argmax(truth))
    peak_uv = wall_uv[peak_row]
    zoom_radius = float(args.zoom_radius_mm)
    zoom = (
        (np.abs(wall_uv[:, 0] - peak_uv[0]) <= zoom_radius)
        & (np.abs(wall_uv[:, 1] - peak_uv[1]) <= zoom_radius)
    )
    zoom_high = zoom & high

    positive_min = max(threshold, 1e-6)
    shared_max = float(max(np.max(truth[high]), np.max(base[high]), np.max(calibrated[high])))
    norm = LogNorm(vmin=positive_min, vmax=shared_max)
    error = calibrated - truth
    error_limit = float(max(np.quantile(np.abs(error[high]), 0.98), 1e-6))

    fig, axes = plt.subplots(2, 4, figsize=(19.0, 9.8), dpi=180)
    background_color = "#c9cdd2"
    base_color = "#167d8d"
    calibrated_color = "#c84b31"

    top_series = (
        ("A. CFD truth: high-WSS locations", truth),
        ("B. Physics V4 on the same locations", base),
        ("C. Final calibrated V4", calibrated),
    )
    shared_image = None
    for ax, (title, values) in zip(axes[0, :3], top_series):
        ax.scatter(
            wall_uv[:, 0], wall_uv[:, 1], s=8, color=background_color,
            alpha=0.28, edgecolors="none",
        )
        shared_image = ax.scatter(
            wall_uv[high, 0], wall_uv[high, 1], c=values[high], s=34,
            cmap="turbo", norm=norm, edgecolors="none",
        )
        ax.scatter(
            [peak_uv[0]], [peak_uv[1]], marker="*", s=180,
            facecolor="none", edgecolor="black", linewidth=1.0,
        )
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_axis_off()

    ax = axes[0, 3]
    ax.scatter(
        wall_uv[:, 0], wall_uv[:, 1], s=8, color=background_color,
        alpha=0.28, edgecolors="none",
    )
    error_image = ax.scatter(
        wall_uv[high, 0], wall_uv[high, 1], c=error[high], s=35,
        cmap="RdBu_r", vmin=-error_limit, vmax=error_limit, edgecolors="none",
    )
    ax.scatter(
        [peak_uv[0]], [peak_uv[1]], marker="*", s=180,
        facecolor="none", edgecolor="black", linewidth=1.0,
    )
    ax.set_title("D. Final signed error\nblue = underprediction")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_axis_off()
    if shared_image is not None:
        fig.colorbar(
            shared_image, ax=axes[0, :3], shrink=0.77,
            label="WSS at truth-defined high-WSS points (Pa, log scale)",
        )
    fig.colorbar(error_image, ax=axes[0, 3], shrink=0.77, label="prediction − truth (Pa)")

    for ax, title, values in (
        (axes[1, 0], "E. Peak-region CFD truth", truth),
        (axes[1, 1], "F. Peak-region final V4", calibrated),
    ):
        ax.scatter(
            wall_uv[zoom, 0], wall_uv[zoom, 1], s=18,
            color=background_color, alpha=0.32, edgecolors="none",
        )
        ax.scatter(
            wall_uv[zoom_high, 0], wall_uv[zoom_high, 1],
            c=values[zoom_high], s=55, cmap="turbo", norm=norm,
            edgecolors="white", linewidths=0.25,
        )
        ax.scatter(
            [peak_uv[0]], [peak_uv[1]], marker="*", s=200,
            facecolor="none", edgecolor="black", linewidth=1.1,
        )
        ax.set_xlim(peak_uv[0] - zoom_radius, peak_uv[0] + zoom_radius)
        ax.set_ylim(peak_uv[1] - zoom_radius, peak_uv[1] + zoom_radius)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title)
        ax.set_xlabel("wall PCA-1 (mm)")
        ax.set_ylabel("wall PCA-2 (mm)")
        ax.grid(alpha=0.18)

    ax = axes[1, 2]
    limit = float(max(np.max(high_truth), np.max(high_base), np.max(high_calibrated)) * 1.04)
    ax.scatter(
        high_truth, high_base, s=28, color=base_color, alpha=0.52,
        edgecolors="none", label=(
            f"Physics V4: R²={base_metrics['r2']:.3f}, "
            f"NRMSE={base_metrics['nrmse']:.3f}"
        ),
    )
    ax.scatter(
        high_truth, high_calibrated, s=32, color=calibrated_color,
        alpha=0.72, edgecolors="none", label=(
            f"Final V4: R²={calibrated_metrics['r2']:.3f}, "
            f"NRMSE={calibrated_metrics['nrmse']:.3f}"
        ),
    )
    ax.plot([0, limit], [0, limit], "k--", lw=1.0)
    ax.scatter(
        [truth[peak_row]], [calibrated[peak_row]], marker="*", s=180,
        color="#ff9d00", edgecolors="black", linewidths=0.8,
        label=(f"peak: {truth[peak_row]:.1f} → {calibrated[peak_row]:.1f} Pa"),
    )
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"G. Pointwise accuracy above q90={threshold:.2f} Pa")
    ax.set_xlabel("CFD truth WSS (Pa)")
    ax.set_ylabel("predicted WSS (Pa)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(alpha=0.20)

    ax = axes[1, 3]
    order = np.argsort(high_truth)
    rank = np.arange(1, int(np.sum(high)) + 1)
    ax.plot(rank, high_truth[order], color="#222222", lw=1.7, label="CFD truth")
    ax.plot(rank, high_base[order], color=base_color, lw=1.3, label="Physics V4")
    ax.plot(rank, high_calibrated[order], color=calibrated_color, lw=1.5, label="Final V4")
    ax.scatter(
        [rank[-1]], [high_truth[order][-1]], marker="*", s=130,
        color="#ff9d00", edgecolors="black", linewidths=0.7,
    )
    ax.set_title(
        "H. High-WSS tail\n"
        f"top-10% spatial overlap: {calibrated_intersection}/120; Jaccard={calibrated_jaccard:.3f}"
    )
    ax.set_xlabel("truth-sorted high-WSS point rank")
    ax.set_ylabel("WSS (Pa)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.20)

    fig.suptitle(
        f"High-WSS accuracy audit — {args.case}, peak {step}\n"
        f"project definition: truth top 10% ({int(np.sum(high))}/{len(truth)} points), "
        f"threshold={threshold:.2f} Pa; final high-WSS R²={calibrated_metrics['r2']:.3f}, "
        f"MAE={calibrated_metrics['mae_pa']:.2f} Pa, NRMSE={calibrated_metrics['nrmse']:.3f}",
        fontsize=13.2,
    )
    fig.subplots_adjust(
        left=0.035, right=0.97, bottom=0.07, top=0.88, hspace=0.30, wspace=0.23
    )
    fig.savefig(output_path)
    plt.close(fig)

    report = {
        "case": args.case,
        "step": step,
        "n_sampled_points": int(len(truth)),
        "high_wss_definition": f"truth >= case q{int(args.high_quantile * 100)}",
        "high_wss_threshold_pa": threshold,
        "n_high_wss_points": int(np.sum(high)),
        "truth_high_wss_mean_pa": float(np.mean(high_truth)),
        "truth_high_wss_max_pa": float(np.max(high_truth)),
        "peak_sample_row": peak_row,
        "peak_full_wall_index": int(sample_index[peak_row]),
        "physics_v4": {
            **base_metrics,
            "top10_intersection_count": base_intersection,
            "top10_jaccard": base_jaccard,
        },
        "final_v4": {
            **calibrated_metrics,
            "top10_intersection_count": calibrated_intersection,
            "top10_jaccard": calibrated_jaccard,
        },
        "figure": str(output_path),
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(output_path)


if __name__ == "__main__":
    main()
