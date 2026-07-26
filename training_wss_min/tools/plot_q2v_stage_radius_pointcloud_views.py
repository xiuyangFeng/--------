#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Point-cloud global + local views for Q2V SA2/SA3 and ball-radius audit.

The figures intentionally use the *actual source support of each SA layer*:
SA1 groups the 5,000 sampled vessel points, SA2 groups the 500 SA1 centres,
and SA3 groups the 125 SA2 centres.  This avoids visually implying that every
layer groups directly from the original vessel point cloud.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from training_wss_min.tools.build_q2v_sa_overlap_audit_package import (
    PACKAGE,
    TRACE_ROOT,
    global_nearest_pair,
    load_stage,
    pair_detail,
    radius_reference_pair,
)


BALL_SETTINGS = (
    ("0.6×", 0.6, "ball_r60"),
    ("0.8×", 0.8, "ball_r80"),
    ("1.0× baseline", 1.0, "ball_r100"),
    ("1.2×", 1.2, "ball_r120"),
    ("1.5×", 1.5, "ball_r150"),
)


def source_support(data: dict) -> np.ndarray:
    """Return the true source point set that was eligible for this SA layer."""
    if data["stage"] == 1:
        return data["full_xyz"]

    centre_rows: list[dict[str, str]] = []
    with (data["trace_root"] / "sa_centers_by_stage.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(row["sa_stage"]) == data["stage"] - 1:
                centre_rows.append(row)
    return np.asarray(
        [[float(row["x_mm"]), float(row["y_mm"]), float(row["z_mm"])] for row in centre_rows], dtype=float
    )


def xyz(data: dict, point_ids: set[int]) -> np.ndarray:
    return np.asarray([data["point_xyz"][key] for key in sorted(point_ids)], dtype=float).reshape((-1, 3))


def draw(ax, data: dict, detail: dict, pair: tuple[int, int], *, local: bool, support_label: str) -> None:
    from matplotlib.patches import Circle

    center_a, center_b = data["centres"][pair[0]], data["centres"][pair[1]]
    midpoint = (center_a + center_b) / 2
    support = source_support(data)
    if local:
        context = support[np.linalg.norm(support - midpoint, axis=1) <= data["radius_mm"] * 1.45]
        point_size, alpha = 21, 0.48
    else:
        context = support
        point_size, alpha = (9.0 if len(context) > 1000 else 22), 0.55
    ax.scatter(context[:, 0], context[:, 2], c="#D0D0D0", s=point_size, alpha=alpha, linewidths=0,
               label=support_label, zorder=1)
    for key, colour, label, size in (
        ("only_a", "#1976D2", "A only", 78 if local else 70),
        ("only_b", "#FBC02D", "B only", 78 if local else 70),
        ("shared", "#8E24AA", "shared", 94 if local else 84),
    ):
        points = xyz(data, detail[key])
        if len(points):
            ax.scatter(points[:, 0], points[:, 2], c=colour, s=size, edgecolors="white", linewidths=0.45,
                       label=f"{label} ({len(points)})", zorder=3)
    ax.scatter(center_a[0], center_a[2], marker="*", s=270, c="#0D47A1", edgecolors="black", linewidths=0.55,
               label=f"center A ({pair[0]})", zorder=6)
    ax.scatter(center_b[0], center_b[2], marker="*", s=270, c="#F9A825", edgecolors="black", linewidths=0.55,
               label=f"center B ({pair[1]})", zorder=6)
    ax.plot([center_a[0], center_b[0]], [center_a[2], center_b[2]], c="#4A4A4A", lw=1.0, ls="--", zorder=2)
    ax.add_patch(Circle((center_a[0], center_a[2]), data["radius_mm"], fill=False, ls="--", lw=1.35, ec="#0D47A1"))
    ax.add_patch(Circle((center_b[0], center_b[2]), data["radius_mm"], fill=False, ls="--", lw=1.35, ec="#F9A825"))
    if local:
        half = max(float(np.ptp(context[:, (0, 2)], axis=0).max()) / 2, data["radius_mm"] * 1.18, 7.0)
        ax.set_xlim(midpoint[0] - half, midpoint[0] + half)
        ax.set_ylim(midpoint[2] - half, midpoint[2] + half)
        ax.set_title("Local enlargement", fontsize=10.5)
    else:
        pad = 4.0
        ax.set_xlim(support[:, 0].min() - pad, support[:, 0].max() + pad)
        ax.set_ylim(support[:, 2].min() - pad, support[:, 2].max() + pad)
        ax.set_title("Global point-cloud context", fontsize=10.5)
    ax.set_aspect("equal")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Z (mm)")


def plot_pair(data: dict, pair: tuple[int, int], *, output: Path, title: str, support_label: str) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    detail = pair_detail(data, data["members"], pair)
    fig, axes = plt.subplots(1, 2, figsize=(17.2, 7.4), gridspec_kw={"width_ratios": (1.15, 1)})
    draw(axes[0], data, detail, pair, local=False, support_label=support_label)
    draw(axes[1], data, detail, pair, local=True, support_label=support_label)
    axes[0].legend(loc="upper left", fontsize=7.5, frameon=True)
    axes[1].legend(loc="upper right", fontsize=7.5, frameon=True)
    fig.suptitle(title, fontsize=14, y=0.985)
    fig.text(
        0.5, 0.94,
        f"nearby FPS centres {pair[0]}/{pair[1]} | r={data['radius_mm']:.2f} mm | "
        f"shared={detail['shared_points']}/{min(detail['group_a_points'], detail['group_b_points'])} "
        f"= {detail['overlap_rate']:.1%}; centre distance={detail['center_distance_mm']:.2f} mm",
        ha="center", va="center", fontsize=11.5,
    )
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.09, top=0.885, wspace=0.13)
    fig.savefig(output, dpi=240)
    plt.close(fig)
    return detail


def main() -> None:
    parser = argparse.ArgumentParser(description="plot Q2V stage and radius grouping views")
    parser.add_argument("--output-dir", type=Path, default=PACKAGE)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    baseline_trace = TRACE_ROOT / "ball_r100"
    stage_specs = (
        (2, "09_SA2_ball_完整点云与局部放大.png", "Q2V baseline SA2 ball query", "true SA2 support: 500 SA1 centres"),
        (3, "10_SA3_ball_完整点云与局部放大.png", "Q2V baseline SA3 ball query", "true SA3 support: 125 SA2 centres"),
    )
    for stage, filename, title, support_label in stage_specs:
        data = load_stage(baseline_trace, stage)
        pair = global_nearest_pair(data)
        plot_pair(data, pair, output=args.output_dir / filename, title=title, support_label=support_label)
        print(args.output_dir / filename)

    ball_data = {multiplier: load_stage(TRACE_ROOT / dirname, 1) for _, multiplier, dirname in BALL_SETTINGS}
    pair = radius_reference_pair(ball_data)
    csv_rows = []
    for sequence, (label, multiplier, _) in enumerate(BALL_SETTINGS, start=11):
        data = ball_data[multiplier]
        filename = f"{sequence:02d}_SA1_ball_r{multiplier:.1f}x_完整点云与局部放大.png"
        detail = plot_pair(
            data, pair, output=args.output_dir / filename,
            title=f"Q2V SA1 ball-radius counterfactual ({label}; same nearest pair)",
            support_label="true SA1 support: vertex-random5000",
        )
        csv_rows.append({
            "radius_multiplier": multiplier, "center_a": pair[0], "center_b": pair[1],
            "radius_mm": data["radius_mm"], "center_distance_mm": detail["center_distance_mm"],
            "group_a_points": detail["group_a_points"], "group_b_points": detail["group_b_points"],
            "shared_points": detail["shared_points"], "overlap_rate": detail["overlap_rate"],
            "figure": filename,
        })
        print(args.output_dir / filename)
    with (args.output_dir / "SA1_ball半径_同一最近pair_全局局部明细.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)


if __name__ == "__main__":
    main()
