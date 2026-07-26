#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Show representative (median), not worst-case, neighbouring SA group overlap."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


PROJECT = Path(__file__).resolve().parents[2]
ROOT_Q0 = PROJECT / (
    "例子/06_PointNet++_SA三层采样与分组/"
    "Q0-10475_FPS2000_FPS-center_fast_RAN_QING_BO_ParaView"
)
ROOT_Q2V = PROJECT / (
    "例子/06_PointNet++_SA三层采样与分组/"
    "Q2V-10477_vertex-random5000_FPS-center_fast_RAN_QING_BO_ParaView"
)
DEFAULT_OUTPUT = PROJECT / (
    "例子/06_PointNet++_SA三层采样与分组/"
    "08_Q0与Q2V_典型最近邻group重叠对比.png"
)


def _load_stage(root: Path, stage: int) -> dict:
    with (root / "sa_centers_marked.csv").open(newline="", encoding="utf-8") as handle:
        marked = list(csv.DictReader(handle))
    point_xyz = {
        int(row["full_wall_index"]): np.asarray(
            [float(row["x_mm"]), float(row["y_mm"]), float(row["z_mm"])], dtype=np.float64
        ) for row in marked
    }
    full_xyz = np.asarray(list(point_xyz.values()), dtype=np.float64)
    centers: dict[int, np.ndarray] = {}
    with (root / "sa_centers_by_stage.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(row["sa_stage"]) == stage:
                centers[int(row["center_id"])] = np.asarray(
                    [float(row["x_mm"]), float(row["y_mm"]), float(row["z_mm"])], dtype=np.float64
                )
    members: dict[int, set[int]] = defaultdict(set)
    with (root / f"sa{stage}_assignments.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            members[int(row["center_id"])].add(int(row["source_full_wall_index"]))
    with (root / "manifest_sa3.json").open(encoding="utf-8") as handle:
        radius = float(json.load(handle)["stages"][stage - 1]["radius_mm_for_this_case"])
    return {"full_xyz": full_xyz, "point_xyz": point_xyz, "centers": centers,
            "members": members, "radius": radius}


def _typical_nearest_pair(data: dict) -> tuple[tuple[int, int], float, float, float]:
    center_ids = np.asarray(sorted(data["centers"]), dtype=np.int64)
    xyz = np.asarray([data["centers"][int(center_id)] for center_id in center_ids])
    distances, neighbours = cKDTree(xyz).query(xyz, k=2)
    pairs = sorted({tuple(sorted((int(center_ids[i]), int(center_ids[j]))))
                    for i, j in enumerate(neighbours[:, 1])})
    # Teacher-facing rate: shared members divided by the smaller group size.
    overlap = np.asarray([
        len(data["members"][a] & data["members"][b]) /
        min(len(data["members"][a]), len(data["members"][b]))
        for a, b in pairs
    ])
    median = float(np.median(overlap))
    index = int(np.argmin(np.abs(overlap - median)))
    a, b = pairs[index]
    shared = data["members"][a] & data["members"][b]
    union = data["members"][a] | data["members"][b]
    center_distance = float(np.linalg.norm(data["centers"][a] - data["centers"][b]))
    return (a, b), float(overlap[index]), len(shared) / len(union), center_distance


def build_figure(root_q0: Path, root_q2v: Path, output: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle


    configs = (("Q0-10475: FPS-2000", root_q0),
               ("Q2V-10477: random-5000", root_q2v))
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 10.2), constrained_layout=True)
    report_rows: list[dict[str, int | float | str]] = []
    for row_index, (label, root) in enumerate(configs):
        for stage, ax in enumerate(axes[row_index], start=1):
            data = _load_stage(root, stage)
            (a, b), shared_rate, jaccard, center_distance = _typical_nearest_pair(data)
            members_a, members_b = data["members"][a], data["members"][b]
            shared, only_a, only_b = members_a & members_b, members_a - members_b, members_b - members_a
            point_xyz = data["point_xyz"]
            to_xyz = lambda point_ids: np.asarray(
                [point_xyz[point_id] for point_id in sorted(point_ids)], dtype=np.float64
            ).reshape((-1, 3))
            xyz_a, xyz_b, xyz_shared = to_xyz(only_a), to_xyz(only_b), to_xyz(shared)
            ca, cb = data["centers"][a], data["centers"][b]
            ax.scatter(data["full_xyz"][:, 0], data["full_xyz"][:, 2], s=7.5, c="#D2D2D2",
                       alpha=0.42, linewidths=0, label="full support point cloud", zorder=1)
            if len(xyz_a):
                ax.scatter(xyz_a[:, 0], xyz_a[:, 2], s=55, c="#1976D2", edgecolors="white",
                           linewidths=0.35, label=f"A only ({len(only_a)})", zorder=3)
            if len(xyz_b):
                ax.scatter(xyz_b[:, 0], xyz_b[:, 2], s=55, c="#F57C00", edgecolors="white",
                           linewidths=0.35, label=f"B only ({len(only_b)})", zorder=3)
            ax.scatter(xyz_shared[:, 0], xyz_shared[:, 2], s=66, c="#8E24AA", edgecolors="white",
                       linewidths=0.4, label=f"shared members ({len(shared)})", zorder=4)
            ax.scatter([ca[0]], [ca[2]], marker="*", s=220, c="#0D47A1", edgecolors="black",
                       linewidths=0.55, label=f"center A ({a})", zorder=6)
            ax.scatter([cb[0]], [cb[2]], marker="*", s=220, c="#E65100", edgecolors="black",
                       linewidths=0.55, label=f"center B ({b})", zorder=6)
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
            result = "PASS <= 1/3" if shared_rate <= 1 / 3 else "HIGH > 1/3"
            ax.set_title(
                f"{label} | SA{stage}\nTypical nearest pair: shared / smaller group = {shared_rate:.1%} ({result})\n"
                f"shared={len(shared)}/{min(len(members_a), len(members_b))}; center distance={center_distance:.1f} mm",
                fontsize=10,
            )
            if stage == 1:
                ax.legend(loc="upper right", fontsize=7.1, frameon=True)
            report_rows.append({
                "experiment": label, "stage": stage, "selection": "nearest pair closest to median overlap",
                "center_a_id": a, "center_b_id": b, "group_a_points": len(members_a),
                "group_b_points": len(members_b), "shared_points": len(shared),
                "shared_fraction_of_smaller_group": shared_rate, "jaccard": jaccard,
                "center_distance_mm": center_distance,
            })
    fig.suptitle(
        "Typical neighbouring-center group overlap (one pair closest to the layer median; not the worst case)\n"
        "blue/orange = unique members; purple = shared members; star = centre; dashed circle = ball-query radius",
        fontsize=15,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=240)
    plt.close(fig)
    report = output.with_suffix(".csv")
    with report.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report_rows[0]))
        writer.writeheader()
        writer.writerows(report_rows)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="plot typical nearest PointNet++ group overlap")
    parser.add_argument("--q0-root", type=Path, default=ROOT_Q0)
    parser.add_argument("--q2v-root", type=Path, default=ROOT_Q2V)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(build_figure(args.q0_root, args.q2v_root, args.output))


if __name__ == "__main__":
    main()
