#!/usr/bin/env python3
"""Plot pooled and per-case high-WSS accuracy for the frozen V4 final model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COHORT_COLORS = {
    "AAA": "#2b6cb0",
    "AG": "#dd6b20",
    "ILO": "#2f855a",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-json", type=Path, required=True)
    parser.add_argument("--predictions-npz", type=Path, required=True)
    parser.add_argument("--png-out", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--method", default="current_safe")
    return parser.parse_args()


def short_case(case_id: str) -> str:
    parts = case_id.split("/")
    if parts[0] == "ILO":
        return f"ILO/{parts[1].replace('-0', '').replace('-1', '')}"
    return f"{parts[0]}/{parts[-1]}"


def main() -> None:
    args = parse_args()
    metrics = json.loads(args.metrics_json.read_text())
    method = args.method
    cases = []
    for case in metrics["cases"]:
        row = dict(case["methods"][method])
        row["canonical_id"] = case["canonical_id"]
        row["cohort"] = case["canonical_id"].split("/", 1)[0]
        cases.append(row)

    pred = np.load(args.predictions_npz, allow_pickle=False)
    truth = np.asarray(pred["truth_mag"], dtype=float)
    pred_mag = np.linalg.norm(np.asarray(pred["current_wss_vec"], dtype=float), axis=1)
    point_case = np.asarray(pred["point_case"])
    point_cohort = np.asarray(pred["point_cohort"])
    high_mask = np.zeros(truth.shape, dtype=bool)
    for case_id in np.unique(point_case):
        case_mask = point_case == case_id
        threshold = np.quantile(truth[case_mask], 0.9)
        high_mask |= case_mask & (truth >= threshold)

    pooled = metrics["pooled_case_q90_high_wss"][method]
    aggregate = metrics["aggregate_case_balanced"][method]
    order = np.argsort([row["high_r2"] for row in cases])
    ordered = [cases[i] for i in order]
    high_r2 = np.array([row["high_r2"] for row in ordered])
    peak_under = 100.0 * np.array(
        [row["peak_underestimate_fraction"] for row in ordered]
    )
    jaccard = np.array([row["top10_jaccard"] for row in ordered])
    cohort = np.array([row["cohort"] for row in ordered])
    case_ids = [row["canonical_id"] for row in ordered]
    rank = np.arange(1, len(ordered) + 1)

    n_r2_pass = int(np.sum(high_r2 >= 0.9))
    n_peak_pass = int(np.sum(peak_under <= 10.0))
    n_both_pass = int(np.sum((high_r2 >= 0.9) & (peak_under <= 10.0)))

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), constrained_layout=True)
    ax_scatter, ax_r2, ax_peak, ax_gate = axes.ravel()

    # A. Pooled pointwise accuracy on each case's CFD-defined top 10% wall points.
    positive = high_mask & (truth > 0) & (pred_mag > 0)
    lo = max(0.05, float(min(truth[positive].min(), pred_mag[positive].min())))
    hi = float(max(truth[positive].max(), pred_mag[positive].max())) * 1.08
    grid = np.geomspace(lo, hi, 200)
    ax_scatter.fill_between(grid, 0.9 * grid, grid / 0.9, color="0.85", alpha=0.45)
    for name in ("AAA", "AG", "ILO"):
        mask = positive & (point_cohort == name)
        ax_scatter.scatter(
            truth[mask],
            pred_mag[mask],
            s=10,
            alpha=0.34,
            edgecolors="none",
            color=COHORT_COLORS[name],
            label=f"{name} ({int(mask.sum())})",
        )
    ax_scatter.plot(grid, grid, color="0.15", linewidth=1.3, label="prediction = truth")
    ax_scatter.plot(grid, 0.9 * grid, color="0.4", linewidth=0.9, linestyle="--")
    ax_scatter.plot(grid, grid / 0.9, color="0.4", linewidth=0.9, linestyle="--")
    ax_scatter.set_xscale("log")
    ax_scatter.set_yscale("log")
    ax_scatter.set_xlim(lo, hi)
    ax_scatter.set_ylim(lo, hi)
    ax_scatter.set_aspect("equal", adjustable="box")
    ax_scatter.set_xlabel("CFD truth WSS (Pa, log scale)")
    ax_scatter.set_ylabel("V4 final WSS (Pa, log scale)")
    ax_scatter.set_title(
        "A. Pooled high-WSS points (4,200 points)\n"
        f"R²={pooled['r2']:.3f}; mean bias={pooled['bias_pa']:+.2f} Pa; "
        f"mean underestimate={100 * pooled['mean_underestimate_fraction']:.1f}%"
    )
    ax_scatter.grid(True, which="both", alpha=0.18)
    ax_scatter.legend(loc="upper left", frameon=False)

    # B. Per-case high-WSS R2, sorted from the hardest to the easiest case.
    colors = [COHORT_COLORS[name] for name in cohort]
    ax_r2.axhspan(0.9, 1.02, color="#38a169", alpha=0.10)
    ax_r2.axhline(0.9, color="#2f855a", linewidth=1.2, linestyle="--")
    ax_r2.vlines(rank, 0.25, high_r2, color=colors, alpha=0.5, linewidth=1.1)
    ax_r2.scatter(rank, high_r2, c=colors, s=42, edgecolor="white", linewidth=0.5)
    ax_r2.set_xlim(0, len(ordered) + 1)
    ax_r2.set_ylim(0.25, 1.02)
    ax_r2.set_xlabel("case rank (hardest → easiest by high-WSS R²)")
    ax_r2.set_ylabel("case high-WSS R²")
    ax_r2.set_title(
        "B. Per-case high-WSS R²\n"
        f"mean={aggregate['high_r2']['mean']:.3f}; median={aggregate['high_r2']['p50']:.3f}; "
        f"{n_r2_pass}/{len(ordered)} cases ≥ 0.90"
    )
    ax_r2.grid(True, axis="y", alpha=0.22)
    for idx in range(min(4, len(ordered))):
        ax_r2.annotate(
            short_case(case_ids[idx]),
            (rank[idx], high_r2[idx]),
            xytext=(3, 7 + 11 * idx),
            textcoords="offset points",
            fontsize=8,
            arrowprops={"arrowstyle": "-", "lw": 0.6, "color": "0.35"},
        )

    # C. Peak underestimation in the same case order as panel B.
    ax_peak.axhspan(0, 10, color="#38a169", alpha=0.10)
    ax_peak.axhline(10, color="#2f855a", linewidth=1.2, linestyle="--")
    peak_colors = np.where(peak_under <= 10.0, "#2f855a", "#c53030")
    ax_peak.vlines(rank, 0, peak_under, color=peak_colors, alpha=0.55, linewidth=1.1)
    ax_peak.scatter(rank, peak_under, c=peak_colors, s=42, edgecolor="white", linewidth=0.5)
    ax_peak.set_xlim(0, len(ordered) + 1)
    ax_peak.set_ylim(0, max(68.0, float(peak_under.max()) * 1.08))
    ax_peak.set_xlabel("same case rank as panel B")
    ax_peak.set_ylabel("peak underestimation (%)")
    ax_peak.set_title(
        "C. Per-case extreme peak underestimation\n"
        f"mean={aggregate['peak_underestimate_fraction']['mean'] * 100:.1f}%; "
        f"median={aggregate['peak_underestimate_fraction']['p50'] * 100:.1f}%; "
        f"{n_peak_pass}/{len(ordered)} cases ≤ 10%"
    )
    ax_peak.grid(True, axis="y", alpha=0.22)
    worst_peak = np.argsort(peak_under)[-3:][::-1]
    for offset, idx in enumerate(worst_peak):
        ax_peak.annotate(
            short_case(case_ids[idx]),
            (rank[idx], peak_under[idx]),
            xytext=(4, -14 - 12 * offset),
            textcoords="offset points",
            fontsize=8,
            arrowprops={"arrowstyle": "-", "lw": 0.6, "color": "0.35"},
        )

    # D. Joint target gate and spatial overlap.
    ax_gate.add_patch(
        plt.Rectangle(
            (0.9, 0),
            0.12,
            10,
            facecolor="#38a169",
            alpha=0.13,
            edgecolor="none",
            zorder=0,
        )
    )
    ax_gate.axvline(0.9, color="#2f855a", linewidth=1.2, linestyle="--")
    ax_gate.axhline(10, color="#2f855a", linewidth=1.2, linestyle="--")
    sizes = 55 + 220 * (jaccard - jaccard.min()) / max(np.ptp(jaccard), 1e-12)
    for name in ("AAA", "AG", "ILO"):
        mask = cohort == name
        ax_gate.scatter(
            high_r2[mask],
            peak_under[mask],
            s=sizes[mask],
            alpha=0.78,
            color=COHORT_COLORS[name],
            edgecolor="white",
            linewidth=0.8,
            label=name,
        )
    selected_id = "ILO/YU_XIANG_SHENG-1/before"
    selected_idx = case_ids.index(selected_id)
    ax_gate.scatter(
        [high_r2[selected_idx]],
        [peak_under[selected_idx]],
        marker="*",
        s=280,
        color="#f6ad00",
        edgecolor="black",
        linewidth=0.9,
        zorder=5,
    )
    ax_gate.annotate(
        "YU_XIANG_SHENG\n(detail figure)",
        (high_r2[selected_idx], peak_under[selected_idx]),
        xytext=(10, 10),
        textcoords="offset points",
        fontsize=8,
        arrowprops={"arrowstyle": "-", "lw": 0.7, "color": "0.35"},
    )
    for idx in np.argsort(high_r2)[:2]:
        ax_gate.annotate(
            short_case(case_ids[idx]),
            (high_r2[idx], peak_under[idx]),
            xytext=(7, -18),
            textcoords="offset points",
            fontsize=8,
            arrowprops={"arrowstyle": "-", "lw": 0.6, "color": "0.35"},
        )
    ax_gate.set_xlim(0.25, 1.02)
    ax_gate.set_ylim(0, max(68.0, float(peak_under.max()) * 1.08))
    ax_gate.set_xlabel("case high-WSS R²")
    ax_gate.set_ylabel("peak underestimation (%)")
    ax_gate.set_title(
        "D. Joint target gate (high R² alone is not enough; peak must also be ≤10%)\n"
        f"{n_both_pass}/{len(ordered)} cases meet both targets; marker size = top-10% spatial overlap"
    )
    ax_gate.grid(True, alpha=0.22)
    ax_gate.legend(loc="upper right", frameon=False)

    fig.suptitle(
        "Frozen V4 final: high-WSS accuracy audit on test35\n"
        "High WSS is defined independently in each case as the CFD-truth top 10% wall points",
        fontsize=15,
    )
    args.png_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.png_out, dpi=180, bbox_inches="tight")
    plt.close(fig)

    report = {
        "method": method,
        "definition": "per-case CFD truth top 10% wall points (q90)",
        "n_cases": len(ordered),
        "n_high_wss_points": int(high_mask.sum()),
        "pooled_high_wss": pooled,
        "case_balanced": {
            "high_r2_mean": aggregate["high_r2"]["mean"],
            "high_r2_median": aggregate["high_r2"]["p50"],
            "peak_underestimate_mean": aggregate["peak_underestimate_fraction"]["mean"],
            "peak_underestimate_median": aggregate["peak_underestimate_fraction"]["p50"],
            "top10_jaccard_mean": aggregate["top10_jaccard"]["mean"],
            "cases_high_r2_at_least_0p9": n_r2_pass,
            "cases_peak_underestimate_at_most_10pct": n_peak_pass,
            "cases_meeting_both": n_both_pass,
        },
        "detail_case": {
            "canonical_id": selected_id,
            **{k: v for k, v in ordered[selected_idx].items() if k != "canonical_id"},
        },
        "cases_sorted_by_high_r2": [
            {
                "canonical_id": row["canonical_id"],
                "cohort": row["cohort"],
                "high_r2": row["high_r2"],
                "peak_underestimate_fraction": row["peak_underestimate_fraction"],
                "top10_jaccard": row["top10_jaccard"],
                "high_mae_pa": row["high_mae_pa"],
                "high_nrmse": row["high_nrmse"],
            }
            for row in ordered
        ],
        "figure": str(args.png_out.resolve()),
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
