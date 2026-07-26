#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare Q1V SA1 neighbour-group overlap under radius-only changes.

All three traces use the same Q1V case, random-5000 fixed support and
deterministic FPS centres.  One nearest-centre pair whose *three-radius mean*
is closest to the teacher's 1/3 reference is selected once and reused in every
panel, so the displayed difference is only the ball-query radius rather than a
different choice of centres.
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
ROOT = PROJECT / "例子/06_PointNet++_SA三层采样与分组"
DEFAULT_OUTPUT = ROOT / "10_Q1V_SA1_半径0.8x_1.2x_1.5x_同一相邻group重叠对比.png"

SETTINGS = (
    ("0.8× radius", (0.04, 0.08, 0.16), ROOT / "Q1V-10476_r80_vertex-random5000_SAME_fast_RAN_QING_BO_ParaView"),
    ("1.2× radius", (0.06, 0.12, 0.24), ROOT / "Q1V-10476_r120_vertex-random5000_SAME_fast_RAN_QING_BO_ParaView"),
    ("1.5× radius", (0.075, 0.15, 0.30), ROOT / "Q1V-10476_r150_vertex-random5000_SAME_fast_RAN_QING_BO_ParaView"),
)


def _nearest_pairs(data: dict) -> tuple[list[tuple[int, int]], dict[tuple[int, int], float]]:
    center_ids = np.asarray(sorted(data["centers"]), dtype=np.int64)
    xyz = np.asarray([data["centers"][int(center_id)] for center_id in center_ids], dtype=np.float64)
    distances, neighbours = cKDTree(xyz).query(xyz, k=2)
    pairs = sorted({tuple(sorted((int(center_ids[i]), int(center_ids[j]))))
                    for i, j in enumerate(neighbours[:, 1])})
    distance_by_pair = {
        pair: float(np.linalg.norm(data["centers"][pair[0]] - data["centers"][pair[1]]))
        for pair in pairs
    }
    return pairs, distance_by_pair


def _select_threshold_reference_pair(data_by_setting: list[tuple[str, tuple[float, float, float], Path, dict]]) -> tuple[int, int]:
    """Choose one nonzero pair near the stated 1/3 overlap reference.

    This selection is intentionally independent of a single radius.  The full
    nearest-pair distribution is also reported, so the selected panel remains a
    teacher-facing illustration rather than a claim about every pair.
    """
    pairs, distances = _nearest_pairs(data_by_setting[0][3])
    candidates = []
    for pair in pairs:
        rates = [_overlap(data, pair)["shared_fraction_of_smaller_group"]
                 for _, _, _, data in data_by_setting]
        if min(rates) > 0:
            candidates.append((abs(float(np.mean(rates)) - 1 / 3), distances[pair], pair))
    if not candidates:
        raise RuntimeError("no nearest-centre pair has nonzero overlap at every radius")
    return min(candidates)[2]


def _overlap(data: dict, pair: tuple[int, int]) -> dict[str, object]:
    a, b = pair
    members_a, members_b = data["members"][a], data["members"][b]
    shared = members_a & members_b
    smaller = min(len(members_a), len(members_b))
    union = members_a | members_b
    return {
        "center_a_id": a,
        "center_b_id": b,
        "group_a_points": len(members_a),
        "group_b_points": len(members_b),
        "shared_points": len(shared),
        "shared_fraction_of_smaller_group": len(shared) / smaller,
        "jaccard": len(shared) / len(union),
        "center_distance_mm": float(np.linalg.norm(data["centers"][a] - data["centers"][b])),
        "only_a": members_a - members_b,
        "only_b": members_b - members_a,
        "shared": shared,
    }


def _all_nearest_pair_summary(data: dict) -> dict[str, float | int]:
    pairs, _ = _nearest_pairs(data)
    rates = np.asarray([_overlap(data, pair)["shared_fraction_of_smaller_group"] for pair in pairs], dtype=float)
    jaccards = np.asarray([_overlap(data, pair)["jaccard"] for pair in pairs], dtype=float)
    return {
        "nearest_pair_count": len(pairs),
        "nearest_pair_overlap_mean": float(rates.mean()),
        "nearest_pair_overlap_median": float(np.median(rates)),
        "nearest_pair_overlap_p90": float(np.quantile(rates, 0.90)),
        "nearest_pair_fraction_at_or_below_one_third": float((rates <= 1 / 3).mean()),
        "nearest_pair_jaccard_median": float(np.median(jaccards)),
    }


def _xyz(point_ids: set[int], point_xyz: dict[int, np.ndarray]) -> np.ndarray:
    return np.asarray([point_xyz[point_id] for point_id in sorted(point_ids)], dtype=float).reshape((-1, 3))


def _draw_panel(ax, data: dict, setting: str, pair: tuple[int, int], result: dict, *, add_legend: bool) -> None:
    from matplotlib.patches import Circle

    a, b = pair
    ca, cb = data["centers"][a], data["centers"][b]
    xyz_a = _xyz(result["only_a"], data["point_xyz"])
    xyz_b = _xyz(result["only_b"], data["point_xyz"])
    xyz_shared = _xyz(result["shared"], data["point_xyz"])
    ax.scatter(data["full_xyz"][:, 0], data["full_xyz"][:, 2], s=6.5, c="#D2D2D2", alpha=0.42,
               linewidths=0, label="fixed Q1V support (5000)", zorder=1)
    if len(xyz_a):
        ax.scatter(xyz_a[:, 0], xyz_a[:, 2], s=56, c="#1976D2", edgecolors="white", linewidths=0.35,
                   label=f"A only ({len(result['only_a'])})", zorder=3)
    if len(xyz_b):
        ax.scatter(xyz_b[:, 0], xyz_b[:, 2], s=56, c="#F57C00", edgecolors="white", linewidths=0.35,
                   label=f"B only ({len(result['only_b'])})", zorder=3)
    ax.scatter(xyz_shared[:, 0], xyz_shared[:, 2], s=68, c="#8E24AA", edgecolors="white", linewidths=0.4,
               label=f"shared ({result['shared_points']})", zorder=4)
    ax.scatter([ca[0]], [ca[2]], marker="*", s=240, c="#0D47A1", edgecolors="black", linewidths=0.55,
               label=f"center A ({a})", zorder=6)
    ax.scatter([cb[0]], [cb[2]], marker="*", s=240, c="#E65100", edgecolors="black", linewidths=0.55,
               label=f"center B ({b})", zorder=6)
    ax.add_patch(Circle((ca[0], ca[2]), data["radius"], fill=False, linestyle="--", linewidth=1.3,
                        edgecolor="#0D47A1"))
    ax.add_patch(Circle((cb[0], cb[2]), data["radius"], fill=False, linestyle="--", linewidth=1.3,
                        edgecolor="#E65100"))
    pad = 5.0
    ax.set_xlim(data["full_xyz"][:, 0].min() - pad, data["full_xyz"][:, 0].max() + pad)
    ax.set_ylim(data["full_xyz"][:, 2].min() - pad, data["full_xyz"][:, 2].max() + pad)
    ax.set_aspect("equal")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Z (mm)")
    teacher_rule = "≤ 1/3" if result["shared_fraction_of_smaller_group"] <= 1 / 3 else "> 1/3"
    ax.set_title(
        f"Q1V SA1 | {setting} | r={data['radius']:.1f} mm\n"
        f"same centres {a}–{b}: shared/smaller = {result['shared_fraction_of_smaller_group']:.1%} ({teacher_rule})\n"
        f"shared={result['shared_points']}/{min(result['group_a_points'], result['group_b_points'])}; "
        f"Jaccard={result['jaccard']:.1%}; centres={result['center_distance_mm']:.1f} mm",
        fontsize=9.4,
    )
    if add_legend:
        ax.legend(loc="upper right", fontsize=7.0, frameon=True)


def build_figure(output: Path) -> tuple[Path, Path, Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data_by_setting = [(label, radii, root, _load_stage(root, 1)) for label, radii, root in SETTINGS]
    reference = data_by_setting[0][3]
    pair = _select_threshold_reference_pair(data_by_setting)
    reference_centres = np.asarray([reference["centers"][key] for key in sorted(reference["centers"])])
    for _, _, _, data in data_by_setting[1:]:
        centres = np.asarray([data["centers"][key] for key in sorted(data["centers"])])
        if not np.array_equal(reference_centres, centres):
            raise RuntimeError("radius-only comparison requires exactly identical SA1 centres")

    results = []
    fig, axes = plt.subplots(1, 3, figsize=(18.4, 6.6), constrained_layout=True)
    for index, (label, radii, root, data) in enumerate(data_by_setting):
        result = _overlap(data, pair)
        summary = _all_nearest_pair_summary(data)
        results.append({
            "radius_multiplier": label.split()[0], "sa1_radius_norm": radii[0], "sa1_radius_mm": data["radius"],
            "source_directory": str(root),
            "fixed_pair_selection": "one nearest-centre pair with nonzero overlap at all radii and mean overlap closest to 1/3",
            **{key: value for key, value in result.items() if key not in {"only_a", "only_b", "shared"}},
            **summary,
        })
        _draw_panel(axes[index], data, label, pair, result, add_legend=index == 0)
        # A single-panel copy makes it easy to show one radius at a time in ParaView discussion.
        one_fig, one_ax = plt.subplots(figsize=(7.2, 6.8), constrained_layout=True)
        _draw_panel(one_ax, data, label, pair, result, add_legend=True)
        one_fig.savefig(root / "07_SA相邻group重叠示例.png", dpi=240)
        plt.close(one_fig)

    fig.suptitle(
        "Q1V SA1 radius-only neighbour-group overlap (same fixed support and same two FPS centres)\n"
        "purple = shared group members; blue/orange = unique members; stars = centres; circles = ball-query radii",
        fontsize=14,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=240)
    plt.close(fig)
    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    json_path = output.with_suffix(".json")
    json_path.write_text(json.dumps({
        "purpose": "Q1V SA1 radius-only fixed-pair neighbour-group overlap audit; no training or model inference",
        "case": "AG/fast/RAN_QING_BO", "support": "Q1V fixed vertex-random5000 support",
        "centre_sampling": "deterministic FPS; identical 500 SA1 centres in all three panels",
        "display_pair": {"center_a_id": pair[0], "center_b_id": pair[1]},
        "display_pair_selection": "nonzero at all three radii; mean shared/smaller closest to 1/3",
        "overlap_definition": "shared members divided by the smaller of the two groups",
        "teacher_reference_rule": "shared/smaller-group ≤ 1/3 is the requested low-overlap reference",
        "results": results,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output, csv_path, json_path


def main() -> None:
    parser = argparse.ArgumentParser(description="plot Q1V SA1 radius-only neighbour-group overlap")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(*build_figure(args.output), sep="\n")


if __name__ == "__main__":
    main()
