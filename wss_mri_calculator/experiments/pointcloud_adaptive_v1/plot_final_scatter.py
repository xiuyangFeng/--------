from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr


ROOT = Path("/public/newhome/cy/Digital_twin/GNN")
SRC = ROOT / "wss_mri_calculator/src"
sys.path.insert(0, str(SRC))

from batch_validate_cfd import DEFAULT_SPLIT, load_split_cases  # noqa: E402
from calculate_wss_cfd import (  # noqa: E402
    DEFAULT_ADAPTIVE_NEIGHBORS,
    DEFAULT_CV_TOLERANCE,
    TRAIN_FROZEN_GLOBAL_SCALE,
    run_case,
)


def metrics(truth: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(truth) & np.isfinite(pred)
    y = truth[valid]
    p = pred[valid]
    variance = float(np.sum((y - y.mean()) ** 2))
    return {
        "n_points": int(len(y)),
        "pooled_raw_r2": 1.0 - float(np.sum((y - p) ** 2)) / max(variance, 1e-30),
        "pooled_mae_pa": float(np.mean(np.abs(y - p))),
        "pooled_nrmse": float(
            np.sqrt(np.mean((y - p) ** 2)) / max(np.mean(np.abs(y)), 1e-12)
        ),
        "pooled_spearman": float(spearmanr(y, p).statistic),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-count", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-display-points", type=int, default=50000)
    parser.add_argument("--output", default="fig_final_model_scatter_raw_vs_scaled.png")
    parser.add_argument("--metrics-out", default="final_scatter_metrics.json")
    args = parser.parse_args()

    cases = load_split_cases(DEFAULT_SPLIT, ROOT / "data_new", {"test"})
    truth_parts: list[np.ndarray] = []
    raw_parts: list[np.ndarray] = []

    for index, case in enumerate(cases, start=1):
        report = run_case(
            Path(case["case_dir"]),
            None,
            args.sample_count,
            64,
            2,
            "carreau",
            False,
            args.seed,
            "pca",
            prefer_peak=True,
            return_arrays=True,
            neighbor_mode="adaptive_cv",
            adaptive_neighbors=DEFAULT_ADAPTIVE_NEIGHBORS,
            cv_tolerance=DEFAULT_CV_TOLERANCE,
            prediction_scale=1.0,
        )
        arrays = report["_arrays"]
        truth_parts.append(np.asarray(arrays["truth_mag"], dtype=np.float64))
        raw_parts.append(np.asarray(arrays["pred_mag"], dtype=np.float64))
        print(f"[{index:02d}/{len(cases):02d}] {case['canonical_id']}")

    truth = np.concatenate(truth_parts)
    raw = np.concatenate(raw_parts)
    scaled = raw * TRAIN_FROZEN_GLOBAL_SCALE
    valid = np.isfinite(truth) & np.isfinite(raw)
    truth, raw, scaled = truth[valid], raw[valid], scaled[valid]

    rng = np.random.default_rng(args.seed)
    if len(truth) > args.max_display_points:
        display_index = np.sort(
            rng.choice(len(truth), args.max_display_points, replace=False)
        )
    else:
        display_index = rng.permutation(len(truth))

    display_truth = truth[display_index]
    display_raw = raw[display_index]
    display_scaled = scaled[display_index]
    limit = float(np.quantile(np.concatenate([truth, scaled]), 0.9975)) * 1.05

    stage_d = json.loads(
        (Path(__file__).parent / "results/stage_d_test35_global_scale1222.json").read_text(
            encoding="utf-8"
        )
    )
    calibrated_name = next(
        name for name in stage_d["aggregate"]["methods"] if "_scale" in name
    )
    case_mean_raw = stage_d["aggregate"]["methods"]["adaptive_cv_tol1p25"]["raw_r2"]["mean"]
    case_mean_scaled = stage_d["aggregate"]["methods"][calibrated_name]["raw_r2"]["mean"]

    raw_metrics = metrics(truth, raw)
    scaled_metrics = metrics(truth, scaled)
    raw_metrics["case_mean_raw_r2"] = float(case_mean_raw)
    scaled_metrics["case_mean_raw_r2"] = float(case_mean_scaled)

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 6.1), dpi=180, sharex=True, sharey=True)
    panels = [
        (
            axes[0],
            display_raw,
            "Without scalar correction",
            raw_metrics,
            "#2878B5",
        ),
        (
            axes[1],
            display_scaled,
            f"With train-only scalar × {TRAIN_FROZEN_GLOBAL_SCALE:.4f}",
            scaled_metrics,
            "#E07A1F",
        ),
    ]
    for ax, pred, title, report, color in panels:
        ax.scatter(
            display_truth,
            pred,
            s=5,
            alpha=0.16,
            color=color,
            edgecolors="none",
            rasterized=True,
        )
        ax.plot([0, limit], [0, limit], "k--", lw=1.2, label="prediction = CFD truth")
        ax.set_xlim(0, limit)
        ax.set_ylim(0, limit)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("CFD WSS truth (Pa)")
        ax.grid(alpha=0.22)
        ax.legend(loc="upper left", fontsize=8)
        annotation = (
            f"test35 · {report['n_points']:,} paired points\n"
            f"case-mean R² = {report['case_mean_raw_r2']:.3f}\n"
            f"pooled R² = {report['pooled_raw_r2']:.3f}\n"
            f"pooled MAE = {report['pooled_mae_pa']:.3f} Pa\n"
            f"Spearman = {report['pooled_spearman']:.3f}"
        )
        ax.text(
            0.97,
            0.04,
            annotation,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.86},
        )

    axes[0].set_ylabel("Velocity→WSS prediction (Pa)")
    fig.suptitle(
        "Frozen point-cloud WSS model vs CFD truth\n"
        "blind test35, peak step, adaptive-CV neighborhood",
        fontsize=14,
    )
    fig.text(
        0.5,
        0.015,
        "Axes capped at the 99.75th percentile for readability; metrics use all finite points.",
        ha="center",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout(rect=[0, 0.035, 1, 0.92])

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)

    payload = {
        "scope": "blind test35",
        "sample_count_per_case": args.sample_count,
        "n_cases": len(cases),
        "prediction_scale": TRAIN_FROZEN_GLOBAL_SCALE,
        "adaptive_neighbors": list(DEFAULT_ADAPTIVE_NEIGHBORS),
        "cv_tolerance": DEFAULT_CV_TOLERANCE,
        "without_scalar": raw_metrics,
        "with_train_scalar": scaled_metrics,
        "display_points": int(len(display_index)),
        "axis_limit_pa": limit,
        "figure": str(output),
    }
    metrics_out = Path(args.metrics_out)
    metrics_out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(output)


if __name__ == "__main__":
    main()
