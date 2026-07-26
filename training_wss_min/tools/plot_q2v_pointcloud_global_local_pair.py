#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Make matched point-cloud global+local views for baseline ball SA1 and KNN-8."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from training_wss_min.tools.build_q2v_sa_overlap_audit_package import (
    PACKAGE,
    TRACE_ROOT,
    knn_members,
    load_stage,
    pair_detail,
)


TRACE = TRACE_ROOT / "ball_r100"
PAIR = (283, 488)  # genuine nearest-neighbour relation, ball overlap = 5/16


def _xyz(data: dict, ids: set[int]) -> np.ndarray:
    return np.asarray([data["point_xyz"][point] for point in sorted(ids)], dtype=float).reshape((-1, 3))


def _draw_points(ax, data: dict, members: dict[int, set[int]], detail: dict, pair: tuple[int, int], *,
                 local: bool, circle_note: str) -> None:
    from matplotlib.patches import Circle

    ca, cb = data["centres"][pair[0]], data["centres"][pair[1]]
    midpoint = (ca + cb) / 2
    if local:
        context = data["full_xyz"][np.linalg.norm(data["full_xyz"] - midpoint, axis=1) <= data["radius_mm"] * 1.45]
        size, alpha = 10, 0.42
    else:
        context = data["full_xyz"]
        size, alpha = 8.5, 0.50
    ax.scatter(context[:, 0], context[:, 2], c="#D0D0D0", s=size, alpha=alpha, linewidths=0,
               label="full Q2V support (5000)", zorder=1)
    for key, colour, label, point_size in (
        ("only_a", "#1976D2", "A only", 68 if local else 62),
        ("only_b", "#FBC02D", "B only", 68 if local else 62),
        ("shared", "#8E24AA", "shared", 82 if local else 76),
    ):
        xyz = _xyz(data, detail[key])
        if len(xyz):
            ax.scatter(xyz[:, 0], xyz[:, 2], c=colour, s=point_size, edgecolors="white", linewidths=0.45,
                       label=f"{label} ({len(xyz)})", zorder=3)
    ax.scatter(ca[0], ca[2], marker="*", s=250, c="#0D47A1", edgecolors="black", linewidths=0.55,
               label=f"center A ({pair[0]})", zorder=6)
    ax.scatter(cb[0], cb[2], marker="*", s=250, c="#F9A825", edgecolors="black", linewidths=0.55,
               label=f"center B ({pair[1]})", zorder=6)
    ax.plot([ca[0], cb[0]], [ca[2], cb[2]], c="#4A4A4A", lw=1.0, ls="--", zorder=2)
    # For KNN this is explicitly an equal-radius spatial reference, never a KNN boundary.
    ax.add_patch(Circle((ca[0], ca[2]), data["radius_mm"], fill=False, ls="--", lw=1.25, ec="#0D47A1"))
    ax.add_patch(Circle((cb[0], cb[2]), data["radius_mm"], fill=False, ls="--", lw=1.25, ec="#F9A825"))
    if local:
        half = max(float(np.ptp(context[:, (0, 2)], axis=0).max()) / 2, 7.0)
        ax.set_xlim(midpoint[0] - half, midpoint[0] + half)
        ax.set_ylim(midpoint[2] - half, midpoint[2] + half)
    else:
        pad = 4.0
        ax.set_xlim(data["full_xyz"][:, 0].min() - pad, data["full_xyz"][:, 0].max() + pad)
        ax.set_ylim(data["full_xyz"][:, 2].min() - pad, data["full_xyz"][:, 2].max() + pad)
    ax.set_aspect("equal")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Z (mm)")
    ax.set_title(circle_note, fontsize=10.1)


def plot_one(data: dict, members: dict[int, set[int]], *, grouping_name: str, circle_note: str, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    detail = pair_detail(data, members, PAIR)
    figure, axes = plt.subplots(1, 2, figsize=(17.2, 7.4), gridspec_kw={"width_ratios": (1.15, 1)},
                                constrained_layout=True)
    _draw_points(axes[0], data, members, detail, PAIR, local=False,
                 circle_note=f"Global point-cloud context | {circle_note}")
    _draw_points(axes[1], data, members, detail, PAIR, local=True,
                 circle_note=f"Local enlargement | {circle_note}")
    axes[0].legend(loc="upper left", fontsize=7.5, frameon=True)
    axes[1].legend(loc="upper right", fontsize=7.5, frameon=True)
    figure.suptitle(
        f"Q2V-10477 SA1 | {grouping_name} | same nearby centres {PAIR[0]}/{PAIR[1]}\n"
        f"shared={detail['shared_points']}/{min(detail['group_a_points'], detail['group_b_points'])} "
        f"= {detail['overlap_rate']:.1%}; centre distance={detail['center_distance_mm']:.2f} mm",
        fontsize=14,
    )
    figure.savefig(output, dpi=240)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="plot matched Q2V point-cloud global+local groups")
    parser.add_argument("--output-dir", type=Path, default=PACKAGE)
    args = parser.parse_args()
    data = load_stage(TRACE, 1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_one(
        data, data["members"], grouping_name="baseline ball query (SA1)",
        circle_note=f"blue/yellow dashed circles = actual ball radius r={data['radius_mm']:.1f} mm",
        output=args.output_dir / "07_SA1_ball_完整点云与局部放大.png",
    )
    plot_one(
        data, knn_members(data, 8), grouping_name="KNN-8 counterfactual",
        circle_note=f"blue/yellow dashed circles = baseline ball-radius reference r={data['radius_mm']:.1f} mm (not KNN boundary)",
        output=args.output_dir / "08_SA1_KNN8_完整点云与局部放大.png",
    )
    print(args.output_dir / "07_SA1_ball_完整点云与局部放大.png")
    print(args.output_dir / "08_SA1_KNN8_完整点云与局部放大.png")


if __name__ == "__main__":
    main()
