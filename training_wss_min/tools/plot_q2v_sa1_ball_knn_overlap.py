#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit Q2V SA1 ball-query and KNN-16 group overlap without training.

The historical ``07`` figure intentionally chose a strongly overlapping pair;
it is not a nearest-centre audit.  This tool makes that distinction explicit,
then compares exact traced ball-query members with a deterministic KNN-16
counterfactual on the *same* absolute-nearest FPS-centre pair.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from training_wss_min.tools.plot_typical_nearest_group_overlap import _load_stage


PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / (
    "例子/06_PointNet++_SA三层采样与分组/"
    "Q2V-10477_vertex-random5000_FPS-center_fast_RAN_QING_BO_ParaView"
)
DEFAULT_OUTPUT = ROOT / "08_SA1_ball与KNN_最近center及局部重叠对比.png"


def _overlap(members_a: set[int], members_b: set[int]) -> dict[str, object]:
    shared = members_a & members_b
    only_a, only_b = members_a - members_b, members_b - members_a
    smaller = min(len(members_a), len(members_b))
    union = members_a | members_b
    return {
        "shared": shared,
        "only_a": only_a,
        "only_b": only_b,
        "group_a_points": len(members_a),
        "group_b_points": len(members_b),
        "shared_points": len(shared),
        "shared_fraction_of_smaller_group": len(shared) / smaller,
        "jaccard": len(shared) / len(union),
    }


def _all_centre_pairs(data: dict) -> list[tuple[float, int, int]]:
    center_ids = np.asarray(sorted(data["centers"]), dtype=np.int64)
    xyz = np.asarray([data["centers"][int(key)] for key in center_ids], dtype=float)
    pairs: list[tuple[float, int, int]] = []
    for first in range(len(center_ids)):
        distances = np.linalg.norm(xyz[first + 1:] - xyz[first], axis=1)
        for offset, distance in enumerate(distances, start=first + 1):
            pairs.append((float(distance), int(center_ids[first]), int(center_ids[offset])))
    return sorted(pairs)


def _knn16_members(data: dict) -> dict[int, set[int]]:
    """Return 16 Euclidean-nearest SA1 source points for every fixed centre."""
    source_ids = np.asarray(sorted(data["point_xyz"]), dtype=np.int64)
    source_xyz = np.asarray([data["point_xyz"][int(key)] for key in source_ids], dtype=float)
    center_ids = np.asarray(sorted(data["centers"]), dtype=np.int64)
    center_xyz = np.asarray([data["centers"][int(key)] for key in center_ids], dtype=float)
    _, nearest = cKDTree(source_xyz).query(center_xyz, k=16)
    return {
        int(center_id): {int(source_ids[index]) for index in nearest[row]}
        for row, center_id in enumerate(center_ids)
    }


def _historical_max_jaccard_pair(data: dict) -> tuple[int, int]:
    """Mirror the selection criterion used by the old `07` illustrative figure."""
    pairs = _all_centre_pairs(data)
    candidates = []
    for distance, a, b in pairs:
        result = _overlap(data["members"][a], data["members"][b])
        if result["shared_points"]:
            candidates.append((
                float(result["jaccard"]), int(result["shared_points"]), -distance, a, b,
            ))
    if not candidates:
        raise RuntimeError("no overlapping traced ball-query pair")
    _, _, _, a, b = max(candidates)
    return a, b


def _threshold_pair(data: dict, pairs: list[tuple[float, int, int]]) -> tuple[int, int]:
    """Nearby, non-minimum-distance pair with the teacher's 5/16 target overlap."""
    for distance, a, b in pairs:
        if (a, b) == (pairs[0][1], pairs[0][2]):
            continue
        result = _overlap(data["members"][a], data["members"][b])
        if (result["group_a_points"], result["group_b_points"], result["shared_points"]) == (16, 16, 5):
            return a, b
    raise RuntimeError("no 5/16 nearby ball-query pair found")


def _points(data: dict, point_ids: set[int]) -> np.ndarray:
    return np.asarray([data["point_xyz"][key] for key in sorted(point_ids)], dtype=float).reshape((-1, 3))


def _local_context(data: dict, center_a: np.ndarray, center_b: np.ndarray, *, scale: float) -> np.ndarray:
    midpoint = (center_a + center_b) / 2
    distance = np.linalg.norm(data["full_xyz"] - midpoint, axis=1)
    return data["full_xyz"][distance <= scale]


def _draw_panel(ax, data: dict, *, title: str, selection: str, pair: tuple[int, int],
                members: dict[int, set[int]], show_radius: bool) -> dict[str, object]:
    from matplotlib.patches import Circle

    a, b = pair
    ca, cb = data["centers"][a], data["centers"][b]
    result = _overlap(members[a], members[b])
    local_scale = max(data["radius"] * 1.45 if show_radius else 13.5,
                      np.linalg.norm(ca - cb) * 1.35)
    context = _local_context(data, ca, cb, scale=local_scale)

    # The local crop removes the visual crowding of a whole-vessel projection.
    ax.scatter(context[:, 0], context[:, 2], c="#CFCFCF", s=8,
               alpha=0.40, linewidths=0, label="local support")
    style = (("only_a", "#1976D2", "A only"), ("only_b", "#F57C00", "B only"),
             ("shared", "#8E24AA", "shared"))
    for key, colour, label in style:
        xyz = _points(data, result[key])
        if len(xyz):
            ax.scatter(xyz[:, 0], xyz[:, 2], c=colour, s=66 if key != "shared" else 78,
                       edgecolors="white", linewidths=0.45,
                       label=f"{label} ({len(result[key])})")
    ax.scatter(ca[0], ca[2], marker="*", s=235, c="#0D47A1", edgecolors="black", linewidths=0.55,
               label=f"center A ({a})")
    ax.scatter(cb[0], cb[2], marker="*", s=235, c="#E65100", edgecolors="black", linewidths=0.55,
               label=f"center B ({b})")
    ax.plot([ca[0], cb[0]], [ca[2], cb[2]], c="#505050", lw=1.0, ls="--")
    if show_radius:
        ax.add_patch(Circle((ca[0], ca[2]), data["radius"], fill=False, linestyle="--",
                            linewidth=1.15, edgecolor="#0D47A1"))
        ax.add_patch(Circle((cb[0], cb[2]), data["radius"], fill=False, linestyle="--",
                            linewidth=1.15, edgecolor="#E65100"))
    midpoint = (ca + cb) / 2
    ranges = np.ptp(context[:, (0, 2)], axis=0)
    half = max(float(ranges.max()) / 2, 7.0)
    ax.set_xlim(midpoint[0] - half, midpoint[0] + half)
    ax.set_ylim(midpoint[2] - half, midpoint[2] + half)
    ax.set_aspect("equal")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Z (mm)")
    radius_note = f"; ball r={data['radius']:.1f} mm" if show_radius else "; exactly 16 Euclidean nearest points"
    ax.set_title(
        f"{title}\n{selection}\n"
        f"shared/smaller = {result['shared_fraction_of_smaller_group']:.1%} "
        f"({result['shared_points']}/{min(result['group_a_points'], result['group_b_points'])}); "
        f"centres = {np.linalg.norm(ca - cb):.2f} mm{radius_note}",
        fontsize=9.2,
    )
    return {
        "panel": title,
        "selection": selection,
        "grouping": "exact traced ball-query (r=0.05, max 16)" if show_radius else "counterfactual KNN-16",
        "center_a_id": a,
        "center_b_id": b,
        "center_distance_mm": float(np.linalg.norm(ca - cb)),
        "ball_radius_mm": float(data["radius"]) if show_radius else "",
        **{key: value for key, value in result.items() if key not in {"shared", "only_a", "only_b"}},
    }


def build_figure(output: Path) -> tuple[Path, Path, Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = _load_stage(ROOT, 1)
    all_pairs = _all_centre_pairs(data)
    nearest_pair = all_pairs[0][1:]
    historical_pair = _historical_max_jaccard_pair(data)
    threshold_pair = _threshold_pair(data, all_pairs)
    knn_members = _knn16_members(data)

    figure = plt.figure(figsize=(15.2, 12.8))
    axes = [figure.add_subplot(2, 2, index + 1) for index in range(4)]
    figure.subplots_adjust(left=0.055, right=0.985, bottom=0.055, top=0.875,
                            wspace=0.20, hspace=0.36)
    reports = [
        _draw_panel(axes[0], data, title="A. Historical high-overlap ball pair", pair=historical_pair,
                    members=data["members"], show_radius=True,
                    selection="chosen by maximum Jaccard; not the nearest-centre pair"),
        _draw_panel(axes[1], data, title="B. Nearby non-minimum ball pair", pair=threshold_pair,
                    members=data["members"], show_radius=True,
                    selection="selected for teacher reference: exactly 5/16 shared (≤ 1/3)"),
        _draw_panel(axes[2], data, title="C. Absolute-nearest centres: traced ball", pair=nearest_pair,
                    members=data["members"], show_radius=True,
                    selection="global minimum centre distance among all 500 SA1 FPS centres"),
        _draw_panel(axes[3], data, title="D. Same absolute-nearest centres: KNN-16", pair=nearest_pair,
                    members=knn_members, show_radius=False,
                    selection="same centres as C; grouping rule only is replaced by KNN-16"),
    ]
    axes[0].legend(loc="upper left", bbox_to_anchor=(-0.02, 1.02), fontsize=7.1, frameon=True)
    figure.suptitle(
        "Q2V-10477 | SA1 local group-overlap audit — exact ball query vs KNN-16 counterfactual\n"
        "Each panel is a local enlarged X–Z view: purple = shared source points; blue/orange = unique points; stars = FPS centres",
        fontsize=14,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=240)
    plt.close(figure)

    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(reports[0]))
        writer.writeheader()
        writer.writerows(reports)
    json_path = output.with_suffix(".json")
    json_path.write_text(json.dumps({
        "purpose": "SA1 local group-overlap visual audit only; no training and no model inference",
        "experiment": "Q2V-10477", "case": "fast/RAN_QING_BO",
        "support": "fixed vertex-random5000 evaluation support", "centres": "fixed FPS 500",
        "ball_query": "exact assignment trace from the original PointNet++ evaluation, radius=0.05 normalized / 9.715 mm, max_num_neighbors=16",
        "knn_counterfactual": "same fixed centres and support; each centre takes its 16 Euclidean-nearest SA1 source points",
        "overlap_definition": "number of shared unique source-point IDs divided by the smaller group size",
        "important_selection_note": "The historical 07 pair is maximum-Jaccard illustrative pair, not the global-nearest pair.",
        "panels": reports,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output, csv_path, json_path


def main() -> None:
    parser = argparse.ArgumentParser(description="plot Q2V SA1 ball-query and KNN overlap audit")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(*build_figure(args.output), sep="\n")


if __name__ == "__main__":
    main()
