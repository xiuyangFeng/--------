#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Visual comparison of SA1 typical neighbouring-group overlap across audits."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from training_wss_min.tools.plot_typical_nearest_group_overlap import _load_stage, _typical_nearest_pair


PROJECT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = PROJECT / "例子/06_PointNet++_SA三层采样与分组"
DEFAULT_OUTPUT = EXAMPLE_ROOT / "09_SA1_采样方式与半径_典型相邻group重叠对比.png"


def _xyz(point_ids: set[int], point_xyz: dict[int, np.ndarray]) -> np.ndarray:
    return np.asarray([point_xyz[item] for item in sorted(point_ids)], dtype=np.float64).reshape((-1, 3))


def build_grid(output: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    rows = (
        ("FPS-2000 support", "Q0-10475_FPS2000_FPS-center_fast_RAN_QING_BO_ParaView",
         "CF_FPS2000_r60_fast_RAN_QING_BO_ParaView"),
        ("FPS-5000 support", "CF_FPS5000_r100_fast_RAN_QING_BO_ParaView",
         "CF_FPS5000_r60_fast_RAN_QING_BO_ParaView"),
        ("vertex-random5000 support", "Q2V-10477_vertex-random5000_FPS-center_fast_RAN_QING_BO_ParaView",
         "CF_random5000_r60_fast_RAN_QING_BO_ParaView"),
    )
    columns = ("reference radii: 0.05 / 0.10 / 0.20", "60% radii: 0.03 / 0.06 / 0.12")
    fig, axes = plt.subplots(3, 2, figsize=(12.4, 14.8), constrained_layout=True)
    report_rows: list[dict[str, int | float | str]] = []
    for row_index, (sampling_label, reference_dir, smaller_radius_dir) in enumerate(rows):
        for column_index, (radius_label, directory) in enumerate(zip(columns, (reference_dir, smaller_radius_dir))):
            ax = axes[row_index, column_index]
            data = _load_stage(EXAMPLE_ROOT / directory, 1)
            (a, b), rate, jaccard, center_distance = _typical_nearest_pair(data)
            ma, mb = data["members"][a], data["members"][b]
            common, only_a, only_b = ma & mb, ma - mb, mb - ma
            xyz_a, xyz_b, xyz_common = (_xyz(values, data["point_xyz"])
                                         for values in (only_a, only_b, common))
            ca, cb = data["centers"][a], data["centers"][b]
            ax.scatter(data["full_xyz"][:, 0], data["full_xyz"][:, 2], s=7.5, c="#D2D2D2",
                       alpha=0.42, linewidths=0, label="full support point cloud", zorder=1)
            if len(xyz_a):
                ax.scatter(xyz_a[:, 0], xyz_a[:, 2], s=55, c="#1976D2", edgecolors="white",
                           linewidths=0.35, label=f"A only ({len(only_a)})", zorder=3)
            if len(xyz_b):
                ax.scatter(xyz_b[:, 0], xyz_b[:, 2], s=55, c="#F57C00", edgecolors="white",
                           linewidths=0.35, label=f"B only ({len(only_b)})", zorder=3)
            ax.scatter(xyz_common[:, 0], xyz_common[:, 2], s=66, c="#8E24AA", edgecolors="white",
                       linewidths=0.4, label=f"shared ({len(common)})", zorder=4)
            ax.scatter([ca[0]], [ca[2]], marker="*", s=220, c="#0D47A1", edgecolors="black",
                       linewidths=0.55, label="center A", zorder=6)
            ax.scatter([cb[0]], [cb[2]], marker="*", s=220, c="#E65100", edgecolors="black",
                       linewidths=0.55, label="center B", zorder=6)
            ax.add_patch(Circle((ca[0], ca[2]), data["radius"], fill=False, linestyle="--",
                                linewidth=1.25, edgecolor="#0D47A1"))
            ax.add_patch(Circle((cb[0], cb[2]), data["radius"], fill=False, linestyle="--",
                                linewidth=1.25, edgecolor="#E65100"))
            pad = 5.0
            ax.set_xlim(data["full_xyz"][:, 0].min() - pad, data["full_xyz"][:, 0].max() + pad)
            ax.set_ylim(data["full_xyz"][:, 2].min() - pad, data["full_xyz"][:, 2].max() + pad)
            ax.set_aspect("equal")
            ax.set_xlabel("X (mm)")
            ax.set_ylabel("Z (mm)")
            status = "PASS <= 1/3" if rate <= 1 / 3 else "HIGH > 1/3"
            ax.set_title(
                f"{sampling_label} | {radius_label}\n"
                f"typical nearest pair: shared/smaller group={rate:.1%} ({status})\n"
                f"shared={len(common)}/{min(len(ma), len(mb))}; centre distance={center_distance:.1f} mm",
                fontsize=9.5,
            )
            if column_index == 0:
                ax.legend(loc="upper right", fontsize=7.0, frameon=True)
            report_rows.append({
                "sampling": sampling_label, "radius_setting": radius_label, "stage": 1,
                "selection": "nearest pair closest to median overlap", "center_a_id": a, "center_b_id": b,
                "group_a_points": len(ma), "group_b_points": len(mb), "shared_points": len(common),
                "shared_fraction_of_smaller_group": rate, "jaccard": jaccard,
                "center_distance_mm": center_distance,
            })
    fig.suptitle(
        "SA1: sampling and ball-query-radius counterfactuals (no model retraining)\n"
        "Each panel is a typical nearest-centre pair; purple = shared members, blue/orange = unique members",
        fontsize=14,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=240)
    plt.close(fig)
    with output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report_rows[0]))
        writer.writeheader()
        writer.writerows(report_rows)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="plot SA1 sampling/radius counterfactual overlap grid")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(build_grid(args.output))


if __name__ == "__main__":
    main()
