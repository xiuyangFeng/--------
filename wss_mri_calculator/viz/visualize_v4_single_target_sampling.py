"""Visualize the exact internal-cell sampling for one Surface-MLS V4 wall target."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
import numpy as np
from scipy.spatial import cKDTree


VIZ = Path(__file__).resolve().parent
PACKAGE = VIZ.parent
SRC = PACKAGE / "src"
ROOT = PACKAGE.parent
sys.path.insert(0, str(SRC))

import data_loader_cfd as dl  # noqa: E402
from calculate_wss_cfd import pca_wall_normals  # noqa: E402


DEFAULT_CACHE = (
    ROOT
    / "outputs/wss_mri_calculator/pointcloud_surface_mls_v4/"
    "point_cache_test35_s1200/ILO__YU_XIANG_SHENG-1__before.npz"
)
DEFAULT_OUT = (
    ROOT
    / "outputs/wss_mri_calculator/pointcloud_surface_mls_v4/00_quicklook/"
    "10_single_wall_point_internal_sampling.png"
)


def tangent_axis(normal: np.ndarray, preferred: np.ndarray | None) -> np.ndarray:
    if preferred is not None:
        tangent = preferred - float(preferred @ normal) * normal
        magnitude = float(np.linalg.norm(tangent))
        if magnitude > 1e-10:
            return tangent / magnitude
    seed = np.zeros(3, dtype=np.float64)
    seed[int(np.argmin(np.abs(normal)))] = 1.0
    tangent = np.cross(normal, seed)
    return tangent / max(float(np.linalg.norm(tangent)), 1e-12)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default=str(DEFAULT_CACHE))
    parser.add_argument("--wall-index", type=int, default=10382)
    parser.add_argument(
        "--support-prefix",
        choices=("surface", "depth3"),
        default="surface",
        help="Use the base V4 support or the depth-3 Profile-Secant support.",
    )
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    cache_path = Path(args.cache).resolve()
    output_path = Path(args.out).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with np.load(cache_path, allow_pickle=False) as source:
        canonical_id = str(source["canonical_id"].item())
        step = int(source["step"])
        sample_index = np.asarray(source["sample_index"], dtype=np.int64)
        matches = np.flatnonzero(sample_index == int(args.wall_index))
        if not len(matches):
            raise ValueError(f"wall index {args.wall_index} is not in the cached sample")
        row = int(matches[0])
        target = np.asarray(source["wall_mm"][row], dtype=np.float64)
        normal = np.asarray(source["normal"][row], dtype=np.float64)
        truth_wss = float(source["truth_mag"][row])
        prefix = args.support_prefix + "__"
        chosen_fraction = float(source[prefix + "selected_fraction"][row])
        depth_limit = float(source[prefix + "depth_limit_mm"][row])
        surface_radius = float(source[prefix + "surface_radius_mm"][row])
        cached_selected = int(source[prefix + "selected_samples"][row])
        cached_anchor_groups = int(source[prefix + "anchor_groups"][row])

    case_dir = ROOT / "data_new" / canonical_id
    wall = dl.load_wall(dl.step_path(case_dir, step, "ascii"))
    interior = dl.load_interior(dl.step_path(case_dir, step, "ascii_in"))
    wall_mm = np.asarray(wall["coords_mm"], dtype=np.float64)
    interior_mm = np.asarray(interior["coords_mm"], dtype=np.float64)
    wall_tree = cKDTree(wall_mm)
    interior_tree = cKDTree(interior_mm)
    full_normal = pca_wall_normals(wall_mm, interior_mm, interior_tree)

    wall_distance, _ = wall_tree.query(wall_mm, k=min(8, len(wall_mm)))
    wall_spacing = np.median(wall_distance[:, 1:], axis=1)
    spacing = float(wall_spacing[int(args.wall_index)])

    _, candidate_index = interior_tree.query(target, k=min(384, len(interior_mm)))
    candidate_index = np.atleast_1d(candidate_index).astype(np.int64)
    candidate = interior_mm[candidate_index]
    nearest_wall_distance, anchor_index = wall_tree.query(candidate, k=1)
    anchor = wall_mm[anchor_index]
    anchor_normal = full_normal[anchor_index]

    opposite_distance_all, opposite_index_all = wall_tree.query(
        target, k=min(1024, len(wall_mm))
    )
    opposite_dot = full_normal[opposite_index_all] @ normal
    opposite_mask = (
        (opposite_dot < -0.60)
        & (opposite_distance_all > max(3.0 * spacing, 1.0))
    )
    if opposite_mask.any():
        opposite_candidates = np.flatnonzero(opposite_mask)
        closest = int(opposite_candidates[np.argmin(opposite_distance_all[opposite_mask])])
        opposite_index = int(opposite_index_all[closest])
        opposite_distance = float(opposite_distance_all[closest])
        preferred = wall_mm[opposite_index] - target
    else:
        opposite_index = -1
        opposite_distance = float("nan")
        preferred = None

    tangent_1 = tangent_axis(normal, preferred)
    tangent_2 = np.cross(normal, tangent_1)
    tangent_2 /= max(float(np.linalg.norm(tangent_2)), 1e-12)

    anchor_to_cell = candidate - anchor
    local_depth = np.einsum("ij,ij->i", anchor_to_cell, anchor_normal)
    owner_cosine = anchor_normal @ normal
    anchor_delta = anchor - target
    s1 = anchor_delta @ tangent_1
    s2 = anchor_delta @ tangent_2
    surface_distance = np.sqrt(s1**2 + s2**2)
    signed_surface_distance = np.where(s1 < 0, -surface_distance, surface_distance)
    fit_depth = chosen_fraction * depth_limit * 1.15
    in_fit_prism = (
        (local_depth > 1e-4)
        & (local_depth <= fit_depth)
        & (surface_distance <= surface_radius)
    )
    retained = in_fit_prism & (owner_cosine >= 0.50)
    opposite_guarded = in_fit_prism & (owner_cosine < -0.25)
    ambiguous_guarded = in_fit_prism & (owner_cosine >= -0.25) & (owner_cosine < 0.50)
    outside_prism = ~in_fit_prism

    target_distance = np.linalg.norm(candidate - target, axis=1)
    anchor_separation = np.linalg.norm(anchor - target, axis=1)
    penetration_risk = (
        in_fit_prism
        & (owner_cosine < -0.25)
        & (anchor_separation > max(3.0 * spacing, 1.5))
        & (nearest_wall_distance < 0.8 * target_distance)
    )
    postfilter_penetration_risk = penetration_risk & retained

    centered_wall = wall_mm - wall_mm.mean(axis=0, keepdims=True)
    _, _, global_axes = np.linalg.svd(centered_wall, full_matrices=False)
    wall_uv = centered_wall @ global_axes[:2].T
    target_uv = (target - wall_mm.mean(axis=0)) @ global_axes[:2].T

    view_radius = max(2.6 * surface_radius, 1.25 * opposite_distance, 3.5)
    local_wall_index = np.asarray(
        wall_tree.query_ball_point(target, 1.5 * view_radius), dtype=np.int64
    )
    local_wall_delta = wall_mm[local_wall_index] - target
    wall_t = local_wall_delta @ tangent_1
    wall_eta = local_wall_delta @ normal
    local_wall_dot = full_normal[local_wall_index] @ normal
    same_wall = local_wall_dot >= 0.50
    opposite_wall = local_wall_dot < -0.60

    candidate_delta = candidate - target
    candidate_t = candidate_delta @ tangent_1
    candidate_eta = candidate_delta @ normal
    anchor_t = anchor_delta @ tangent_1
    anchor_eta = anchor_delta @ normal

    selected_rank = np.flatnonzero(retained)
    if len(selected_rank) > 80:
        connector_rank = selected_rank[
            np.linspace(0, len(selected_rank) - 1, 80, dtype=int)
        ]
    else:
        connector_rank = selected_rank
    physical_segments = np.stack(
        [
            np.column_stack([anchor_t[connector_rank], anchor_eta[connector_rank]]),
            np.column_stack([candidate_t[connector_rank], candidate_eta[connector_rank]]),
        ],
        axis=1,
    )

    fig, axes = plt.subplots(
        1, 3, figsize=(18.0, 5.8), dpi=190,
        gridspec_kw={"width_ratios": [1.04, 1.06, 1.12]},
    )
    opposite_distance_label = (
        f"{opposite_distance:.2f} mm"
        if np.isfinite(opposite_distance)
        else "not detected in the local search"
    )

    ax = axes[0]
    ax.scatter(
        wall_uv[:, 0], wall_uv[:, 1], s=1.5, color="#c7c7c7",
        alpha=0.38, edgecolors="none",
    )
    ax.scatter(
        [target_uv[0]], [target_uv[1]], s=190, marker="*", color="#ff9d00",
        edgecolors="black", linewidths=0.9, zorder=8, label="target wall point",
    )
    ax.add_patch(
        Rectangle(
            (target_uv[0] - 20.0, target_uv[1] - 20.0), 40.0, 40.0,
            fill=False, linestyle="--", linewidth=1.3, edgecolor="#ff9d00",
        )
    )
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(f"A. Target location on the full wall\nwall index={args.wall_index}")
    ax.set_xlabel("wall PCA-1 (mm)")
    ax.set_ylabel("wall PCA-2 (mm)")
    ax.grid(alpha=0.18)

    ax = axes[1]
    ax.scatter(
        wall_t[same_wall], wall_eta[same_wall], s=8, color="#afb5bc",
        alpha=0.50, edgecolors="none", label="same-facing wall",
    )
    if opposite_wall.any():
        ax.scatter(
            wall_t[opposite_wall], wall_eta[opposite_wall], s=19,
            facecolors="none", edgecolors="#e31a1c", linewidths=0.9,
            alpha=0.82, label="opposite-facing wall",
        )
    ax.scatter(
        candidate_t[outside_prism], candidate_eta[outside_prism], s=22,
        color="#c9cdd2", alpha=0.46, edgecolors="none",
        label="queried but outside depth/radius",
    )
    if len(physical_segments):
        ax.add_collection(
            LineCollection(
                physical_segments, colors="#167d8d", linewidths=0.65,
                alpha=0.43, label="cell-to-nearest-wall anchor",
            )
        )
    ax.scatter(
        anchor_t[retained], anchor_eta[retained], s=9, color="#222222",
        alpha=0.34, edgecolors="none", label="nearest wall anchors",
    )
    ax.scatter(
        candidate_t[retained], candidate_eta[retained], s=38,
        color="#7b2cbf", alpha=0.93, edgecolors="none",
        label="retained internal cells",
    )
    if ambiguous_guarded.any():
        ax.scatter(
            candidate_t[ambiguous_guarded], candidate_eta[ambiguous_guarded],
            s=48, marker="x", color="#ff9d00", linewidths=1.1,
            label="orientation rejected",
        )
    if opposite_guarded.any():
        ax.scatter(
            candidate_t[opposite_guarded], candidate_eta[opposite_guarded],
            s=62, marker="x", color="#e31a1c", linewidths=1.25,
            label="opposite-wall candidate rejected",
        )
    if penetration_risk.any():
        ax.scatter(
            candidate_t[penetration_risk], candidate_eta[penetration_risk],
            s=105, marker="X", color="#e31a1c", edgecolors="black",
            linewidths=0.7, label="cross-wall risk caught by guard",
        )
    ax.scatter(
        [0], [0], s=190, marker="*", color="#ff9d00", edgecolors="black",
        linewidths=0.9, zorder=8,
    )
    ax.axhline(0, color="black", lw=0.8, alpha=0.65)
    ax.annotate(
        "target inward normal", xy=(0, 0.82 * view_radius), xytext=(0, 0),
        arrowprops={"arrowstyle": "->", "color": "#2878b5", "lw": 1.4},
        ha="center", va="bottom", fontsize=8, color="#2878b5",
    )
    ax.set_xlim(-view_radius, view_radius)
    ax.set_ylim(-0.32 * view_radius, view_radius)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(
        f"B. Physical local view of all 384 queried cells\n"
        f"closest opposite wall={opposite_distance_label}"
    )
    ax.set_xlabel("local tangent coordinate (mm)")
    ax.set_ylabel("target-normal coordinate (mm)")
    ax.grid(alpha=0.20)

    ax = axes[2]
    ax.add_patch(
        Rectangle(
            (-surface_radius, 0), 2 * surface_radius, fit_depth,
            facecolor="#7b2cbf", edgecolor="#333333", linewidth=1.1,
            linestyle="--", alpha=0.07, label="depth/radius selection window",
        )
    )
    ax.scatter(
        signed_surface_distance[outside_prism], local_depth[outside_prism],
        s=24, color="#c9cdd2", alpha=0.48, edgecolors="none",
    )
    ax.scatter(
        signed_surface_distance[retained], np.zeros(int(retained.sum())),
        s=10, color="#222222", alpha=0.34, edgecolors="none",
    )
    ax.scatter(
        signed_surface_distance[retained], local_depth[retained], s=40,
        color="#7b2cbf", alpha=0.94, edgecolors="none",
    )
    if ambiguous_guarded.any():
        ax.scatter(
            signed_surface_distance[ambiguous_guarded], local_depth[ambiguous_guarded],
            s=50, marker="x", color="#ff9d00", linewidths=1.1,
        )
    if opposite_guarded.any():
        ax.scatter(
            signed_surface_distance[opposite_guarded], local_depth[opposite_guarded],
            s=64, marker="x", color="#e31a1c", linewidths=1.25,
        )
    if penetration_risk.any():
        ax.scatter(
            signed_surface_distance[penetration_risk], local_depth[penetration_risk],
            s=108, marker="X", color="#e31a1c", edgecolors="black",
            linewidths=0.7,
        )
    ax.scatter(
        [0], [0], s=190, marker="*", color="#ff9d00", edgecolors="black",
        linewidths=0.9, zorder=8,
    )
    ax.axhline(0, color="black", lw=0.8, alpha=0.65)
    ax.axhline(fit_depth, color="#333333", ls="--", lw=1.0)
    ax.axvline(-surface_radius, color="#333333", ls="--", lw=1.0)
    ax.axvline(surface_radius, color="#333333", ls="--", lw=1.0)
    ax.set_xlim(-1.18 * max(surface_radius, np.max(np.abs(signed_surface_distance))),
                1.18 * max(surface_radius, np.max(np.abs(signed_surface_distance))))
    upper_depth = max(float(np.quantile(local_depth[np.isfinite(local_depth)], 0.98)), fit_depth)
    ax.set_ylim(-0.18 * upper_depth, 1.12 * upper_depth)
    ax.set_title(
        "C. Selection coordinates used by the local profile fit\n"
        f"radius≤{surface_radius:.2f} mm; depth≤{fit_depth:.2f} mm\n"
        "anchor-normal cosine≥0.50"
    )
    ax.set_xlabel("signed distance of each cell's wall anchor from target (mm)")
    ax.set_ylabel("cell depth from its own nearest wall anchor (mm)")
    ax.grid(alpha=0.20)

    legend_pairs = [
        pair
        for ax in axes
        for pair in zip(*ax.get_legend_handles_labels())
        if pair[1]
    ]
    unique = {label: handle for handle, label in legend_pairs}
    fig.legend(
        unique.values(), unique.keys(), loc="lower center", ncol=5,
        fontsize=7.8, frameon=True, bbox_to_anchor=(0.5, 0.012),
    )
    fig.suptitle(
        f"One-wall-point internal sampling for "
        f"{'Profile-Secant V3 depth-3 support' if args.support_prefix == 'depth3' else 'Surface-MLS V4'}"
        f" — {canonical_id}, peak {step}\n"
        f"384 Euclidean-nearest interior cells → retained={int(retained.sum())}; "
        f"orientation rejected={int(ambiguous_guarded.sum())}; "
        f"opposite-wall rejected={int(opposite_guarded.sum())}; "
        f"cross-wall risk caught={int(penetration_risk.sum())}; "
        f"post-filter retained risk={int(postfilter_penetration_risk.sum())}",
        fontsize=13.2,
    )
    fig.subplots_adjust(left=0.05, right=0.99, bottom=0.15, top=0.82, wspace=0.23)
    fig.savefig(output_path)
    plt.close(fig)

    report = {
        "case": canonical_id,
        "support_prefix": args.support_prefix,
        "support_name": (
            "Profile-Secant V3 depth-3 support"
            if args.support_prefix == "depth3"
            else "Surface-MLS V4 base support"
        ),
        "step": step,
        "sample_row": row,
        "wall_index": int(args.wall_index),
        "truth_wss_pa": truth_wss,
        "candidate_query": "384 Euclidean-nearest interior cell centers",
        "candidate_count": int(len(candidate)),
        "chosen_depth_fraction": chosen_fraction,
        "depth_limit_mm": depth_limit,
        "support_depth_overshoot": 1.15,
        "final_fit_depth_mm": fit_depth,
        "surface_radius_mm": surface_radius,
        "owner_normal_cosine_min": 0.50,
        "retained_count": int(retained.sum()),
        "cached_selected_count": cached_selected,
        "retained_anchor_groups": int(len(np.unique(anchor_index[retained]))),
        "cached_anchor_groups": cached_anchor_groups,
        "outside_depth_or_radius_count": int(outside_prism.sum()),
        "orientation_rejected_count": int(ambiguous_guarded.sum()),
        "opposite_wall_rejected_count": int(opposite_guarded.sum()),
        "prefilter_penetration_risk_count_caught_by_orientation_guard": int(
            penetration_risk.sum()
        ),
        "postfilter_penetration_risk_count_retained": int(
            postfilter_penetration_risk.sum()
        ),
        "closest_opposite_wall_distance_mm": (
            float(opposite_distance) if np.isfinite(opposite_distance) else None
        ),
        "figure": str(output_path),
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(output_path)


if __name__ == "__main__":
    main()
