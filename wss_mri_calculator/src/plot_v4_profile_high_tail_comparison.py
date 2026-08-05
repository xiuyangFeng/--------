#!/usr/bin/env python3
"""Compare the current V4 high-tail model with the near-wall-profile candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-json", type=Path, required=True)
    parser.add_argument("--current-predictions", type=Path, required=True)
    parser.add_argument("--profile-json", type=Path, required=True)
    parser.add_argument("--profile-predictions", type=Path, required=True)
    parser.add_argument("--case", default="ILO/YU_XIANG_SHENG-1/before")
    parser.add_argument("--png-out", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    return parser.parse_args()


def magnitude(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return np.linalg.norm(array, axis=1) if array.ndim == 2 else array


def case_rows(payload: dict) -> dict[str, dict]:
    return {
        row["canonical_id"]: row["methods"]["high_tail_w10_s80"]
        for row in payload["cases"]
    }


def short_case(case_id: str) -> str:
    parts = case_id.split("/")
    return parts[-2] if parts[0] == "ILO" else parts[-1]


def main() -> None:
    args = parse_args()
    current_report = json.loads(args.current_json.read_text())
    profile_report = json.loads(args.profile_json.read_text())
    current_npz = np.load(args.current_predictions, allow_pickle=False)
    profile_npz = np.load(args.profile_predictions, allow_pickle=False)
    groups = np.asarray(current_npz["point_case"])
    if not np.array_equal(groups, np.asarray(profile_npz["point_case"])):
        raise RuntimeError("prediction point order differs")
    truth = np.asarray(current_npz["truth_mag"], dtype=np.float64)
    current = magnitude(current_npz["high_tail_wss_vec"])
    profile = magnitude(profile_npz["high_tail_wss_vec"])
    wall = np.asarray(current_npz["wall_mm"], dtype=np.float64)

    current_cases = case_rows(current_report)
    profile_cases = case_rows(profile_report)
    ids = sorted(current_cases)
    current_high_r2 = np.array([current_cases[c]["high_r2"] for c in ids])
    profile_high_r2 = np.array([profile_cases[c]["high_r2"] for c in ids])
    current_peak = 100 * np.array(
        [current_cases[c]["peak_underestimate_fraction"] for c in ids]
    )
    profile_peak = 100 * np.array(
        [profile_cases[c]["peak_underestimate_fraction"] for c in ids]
    )

    mask = groups == args.case
    local_truth = truth[mask]
    local_current = current[mask]
    local_profile = profile[mask]
    local_wall = wall[mask]
    high = local_truth >= np.quantile(local_truth, 0.90)
    centered = local_wall - np.mean(local_wall, axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    projection = centered @ vh[:2].T
    high_values = local_truth[high]
    color_min = max(float(np.min(high_values)), 1e-3)
    color_max = float(
        max(np.max(local_truth[high]), np.max(local_current[high]), np.max(local_profile[high]))
    )
    log_norm = LogNorm(color_min, color_max)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    spatial_axes = axes[0]
    for ax, values, title in (
        (spatial_axes[0], local_truth, "A. CFD truth at truth-defined high-WSS points"),
        (spatial_axes[1], local_current, "B. Current high-tail V4"),
        (spatial_axes[2], local_profile, "C. Near-wall profile candidate"),
    ):
        ax.scatter(
            projection[:, 0], projection[:, 1], s=7, color="0.80", alpha=0.32, edgecolors="none"
        )
        image = ax.scatter(
            projection[high, 0],
            projection[high, 1],
            c=values[high],
            cmap="turbo",
            norm=log_norm,
            s=25,
            edgecolors="none",
        )
        peak_index = int(np.argmax(local_truth))
        ax.scatter(
            projection[peak_index, 0],
            projection[peak_index, 1],
            marker="*",
            s=170,
            color="#f6ad00",
            edgecolor="black",
            linewidth=0.9,
            zorder=5,
        )
        ax.set_title(title)
        ax.set_xlabel("wall PCA-1 (mm)")
        ax.set_ylabel("wall PCA-2 (mm)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.15)
    fig.colorbar(image, ax=spatial_axes, label="WSS (Pa, log scale)", shrink=0.85)

    ax_r2, ax_peak, ax_tail = axes[1]
    lo_r2 = min(0.0, float(min(current_high_r2.min(), profile_high_r2.min())))
    ax_r2.plot([lo_r2, 1.0], [lo_r2, 1.0], color="0.25", linewidth=1.1)
    ax_r2.axhline(0.9, color="#2f855a", linestyle="--", linewidth=1.0)
    ax_r2.axvline(0.9, color="#2f855a", linestyle="--", linewidth=1.0)
    ax_r2.scatter(current_high_r2, profile_high_r2, s=48, alpha=0.78, color="#2b6cb0")
    selected_index = ids.index(args.case)
    ax_r2.scatter(
        [current_high_r2[selected_index]],
        [profile_high_r2[selected_index]],
        marker="*",
        s=210,
        color="#f6ad00",
        edgecolor="black",
        zorder=5,
    )
    improvement = profile_high_r2 - current_high_r2
    for idx in np.argsort(np.abs(improvement))[-4:]:
        ax_r2.annotate(
            short_case(ids[idx]),
            (current_high_r2[idx], profile_high_r2[idx]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    ax_r2.set_xlim(lo_r2, 1.0)
    ax_r2.set_ylim(lo_r2, 1.0)
    ax_r2.set_xlabel("current case high-WSS R²")
    ax_r2.set_ylabel("profile candidate case high-WSS R²")
    ax_r2.set_title(
        "D. Per-case high-WSS R²\n"
        f"mean {current_high_r2.mean():.3f} → {profile_high_r2.mean():.3f}"
    )
    ax_r2.grid(True, alpha=0.2)

    peak_limit = max(65.0, float(max(current_peak.max(), profile_peak.max())) * 1.06)
    ax_peak.plot([0, peak_limit], [0, peak_limit], color="0.25", linewidth=1.1)
    ax_peak.axhline(10, color="#2f855a", linestyle="--", linewidth=1.0)
    ax_peak.axvline(10, color="#2f855a", linestyle="--", linewidth=1.0)
    ax_peak.scatter(current_peak, profile_peak, s=48, alpha=0.78, color="#c05621")
    ax_peak.scatter(
        [current_peak[selected_index]],
        [profile_peak[selected_index]],
        marker="*",
        s=210,
        color="#f6ad00",
        edgecolor="black",
        zorder=5,
    )
    for idx in np.argsort(profile_peak)[-3:]:
        ax_peak.annotate(
            short_case(ids[idx]),
            (current_peak[idx], profile_peak[idx]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    ax_peak.set_xlim(0, peak_limit)
    ax_peak.set_ylim(0, peak_limit)
    ax_peak.set_xlabel("current peak underestimation (%)")
    ax_peak.set_ylabel("profile candidate peak underestimation (%)")
    ax_peak.set_title(
        "E. Per-case extreme-peak underestimation\n"
        f"mean {current_peak.mean():.1f}% → {profile_peak.mean():.1f}%"
    )
    ax_peak.grid(True, alpha=0.2)

    order = np.argsort(local_truth[high])
    ax_tail.plot(local_truth[high][order], color="0.12", linewidth=1.8, label="CFD truth")
    ax_tail.plot(local_current[high][order], color="#c05621", linewidth=1.4, label="current")
    ax_tail.plot(local_profile[high][order], color="#2b6cb0", linewidth=1.4, label="profile")
    ax_tail.set_xlabel("truth-sorted high-WSS point rank")
    ax_tail.set_ylabel("WSS (Pa)")
    old_case = current_cases[args.case]
    new_case = profile_cases[args.case]
    ax_tail.set_title(
        "F. Representative case high-WSS tail\n"
        f"R² {old_case['high_r2']:.3f} → {new_case['high_r2']:.3f}; "
        f"peak under {100 * old_case['peak_underestimate_fraction']:.1f}% → "
        f"{100 * new_case['peak_underestimate_fraction']:.1f}%"
    )
    ax_tail.grid(True, alpha=0.2)
    ax_tail.legend(frameon=False)

    old_aggregate = current_report["aggregate_case_balanced"]["high_tail_w10_s80"]
    new_aggregate = profile_report["aggregate_case_balanced"]["high_tail_w10_s80"]
    old_pooled = current_report["pooled_case_q90_high_wss"]["high_tail_w10_s80"]
    new_pooled = profile_report["pooled_case_q90_high_wss"]["high_tail_w10_s80"]
    fig.suptitle(
        "Near-wall profile features vs current high-tail V4 on test35\n"
        f"overall R² {old_aggregate['raw_r2']['mean']:.3f}→{new_aggregate['raw_r2']['mean']:.3f}; "
        f"pooled high-WSS R² {old_pooled['r2']:.3f}→{new_pooled['r2']:.3f}; "
        f"mean peak under {100 * old_aggregate['peak_underestimate_fraction']['mean']:.1f}%"
        f"→{100 * new_aggregate['peak_underestimate_fraction']['mean']:.1f}%",
        fontsize=15,
    )
    args.png_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.png_out, dpi=180, bbox_inches="tight")
    plt.close(fig)

    result = {
        "case": args.case,
        "current": {
            "overall_r2_mean": old_aggregate["raw_r2"]["mean"],
            "pooled_high_r2": old_pooled["r2"],
            "case_balanced_high_r2_mean": old_aggregate["high_r2"]["mean"],
            "peak_underestimate_mean": old_aggregate["peak_underestimate_fraction"]["mean"],
            "detail_case": old_case,
        },
        "profile_candidate": {
            "overall_r2_mean": new_aggregate["raw_r2"]["mean"],
            "pooled_high_r2": new_pooled["r2"],
            "case_balanced_high_r2_mean": new_aggregate["high_r2"]["mean"],
            "peak_underestimate_mean": new_aggregate["peak_underestimate_fraction"]["mean"],
            "detail_case": new_case,
        },
        "figure": str(args.png_out.resolve()),
    }
    args.json_out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
