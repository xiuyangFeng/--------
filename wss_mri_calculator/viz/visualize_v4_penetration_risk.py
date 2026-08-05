"""Audit closest-surface MLS V4 supports for opposite-wall penetration risk."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, Rectangle
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
    "09_surface_mls_penetration_wide20mm.png"
)


def _tangent_axis(normal: np.ndarray, preferred: np.ndarray | None) -> np.ndarray:
    if preferred is not None:
        tangent = preferred - float(preferred @ normal) * normal
        magnitude = float(np.linalg.norm(tangent))
        if magnitude > 1e-10:
            return tangent / magnitude
    seed = np.zeros(3, dtype=np.float64)
    seed[int(np.argmin(np.abs(normal)))] = 1.0
    tangent = np.cross(normal, seed)
    return tangent / max(float(np.linalg.norm(tangent)), 1e-12)


def _canonical_case_dir(canonical_id: str) -> Path:
    return ROOT / "data_new" / canonical_id


def _load_cache(path: Path) -> dict[str, np.ndarray | str | int]:
    with np.load(path, allow_pickle=False) as source:
        return {
            key: (source[key].item() if source[key].ndim == 0 else np.asarray(source[key]))
            for key in source.files
        }


def _choose_focus(
    retained_envelope_mm: np.ndarray,
    opposite_distance_mm: np.ndarray,
    truth_wss_pa: np.ndarray,
    prefilter_risk_count: np.ndarray,
) -> list[tuple[str, int]]:
    finite = np.isfinite(opposite_distance_mm)
    valid_ratio = finite & np.isfinite(retained_envelope_mm)
    envelope_ratio = retained_envelope_mm / opposite_distance_mm
    first = int(np.nanargmax(np.where(valid_ratio, envelope_ratio, np.nan)))
    if np.max(prefilter_risk_count) > 0:
        second = int(np.argmax(prefilter_risk_count))
        second_label = "most cross-wall candidates caught by guard"
    else:
        second = int(np.nanargmin(np.where(valid_ratio, opposite_distance_mm, np.nan)))
        second_label = "closest opposite-facing wall"
    narrow_cut = float(np.nanquantile(opposite_distance_mm[valid_ratio], 0.15))
    narrow = valid_ratio & (opposite_distance_mm <= narrow_cut)
    third = int(np.nanargmax(np.where(narrow, truth_wss_pa, np.nan)))

    chosen: list[tuple[str, int]] = []
    for label, index in (
        ("largest retained-envelope / opposite-wall ratio", first),
        (second_label, second),
        ("high-WSS target near a narrow wall", third),
    ):
        if index not in [item[1] for item in chosen]:
            chosen.append((label, index))
    if len(chosen) < 3:
        order = np.argsort(np.where(valid_ratio, envelope_ratio, -np.inf))[::-1]
        for index in order:
            if not valid_ratio[int(index)]:
                continue
            if int(index) not in [item[1] for item in chosen]:
                chosen.append(("additional narrow-wall target", int(index)))
            if len(chosen) == 3:
                break
    return chosen[:3]


def audit(
    cache_path: Path,
    zoom_radius_mm: float,
    support_prefix: str,
) -> tuple[dict[str, Any], plt.Figure]:
    cache = _load_cache(cache_path)
    canonical_id = str(cache["canonical_id"])
    step = int(cache["step"])
    case_dir = _canonical_case_dir(canonical_id)

    wall = dl.load_wall(dl.step_path(case_dir, step, "ascii"))
    interior = dl.load_interior(dl.step_path(case_dir, step, "ascii_in"))
    wall_mm = np.asarray(wall["coords_mm"], dtype=np.float64)
    interior_mm = np.asarray(interior["coords_mm"], dtype=np.float64)
    target_mm = np.asarray(cache["wall_mm"], dtype=np.float64)
    target_normal = np.asarray(cache["normal"], dtype=np.float64)
    sample_index = np.asarray(cache["sample_index"], dtype=np.int64)
    truth_wss = np.asarray(cache["truth_mag"], dtype=np.float64)
    prefix = support_prefix + "__"
    chosen_fraction = np.asarray(cache[prefix + "selected_fraction"], dtype=np.float64)
    depth_limit = np.asarray(cache[prefix + "depth_limit_mm"], dtype=np.float64)
    surface_radius = np.asarray(cache[prefix + "surface_radius_mm"], dtype=np.float64)

    wall_tree = cKDTree(wall_mm)
    interior_tree = cKDTree(interior_mm)
    full_normal = pca_wall_normals(wall_mm, interior_mm, interior_tree)
    wall_neighbor_distance, _ = wall_tree.query(wall_mm, k=min(8, len(wall_mm)))
    if wall_neighbor_distance.ndim == 1:
        wall_spacing = np.maximum(wall_neighbor_distance, 1e-3)
    else:
        wall_spacing = np.median(wall_neighbor_distance[:, 1:], axis=1)

    _, candidate_index = interior_tree.query(
        target_mm, k=min(384, len(interior_mm))
    )
    if candidate_index.ndim == 1:
        candidate_index = candidate_index[:, None]
    candidate_mm = interior_mm[candidate_index]
    _, anchor_index_flat = wall_tree.query(candidate_mm.reshape(-1, 3), k=1)
    anchor_index = anchor_index_flat.reshape(candidate_index.shape)
    anchor_mm = wall_mm[anchor_index]
    anchor_normal = full_normal[anchor_index]

    anchor_to_cell = candidate_mm - anchor_mm
    local_depth = np.einsum("nki,nki->nk", anchor_to_cell, anchor_normal)
    owner_cosine = np.einsum("nki,ni->nk", anchor_normal, target_normal)
    anchor_delta = anchor_mm - target_mm[:, None, :]
    anchor_eta = np.einsum("nki,ni->nk", anchor_delta, target_normal)
    surface_distance = np.sqrt(
        np.maximum(np.sum(anchor_delta**2, axis=2) - anchor_eta**2, 0.0)
    )
    fit_depth = chosen_fraction * depth_limit * 1.15
    in_fit_prism = (
        (local_depth > 1e-4)
        & (local_depth <= fit_depth[:, None])
        & (surface_distance <= surface_radius[:, None])
    )
    retained = in_fit_prism & (owner_cosine >= 0.50)
    opposite_guarded = in_fit_prism & (owner_cosine < -0.25)
    ambiguous_guarded = in_fit_prism & (owner_cosine >= -0.25) & (owner_cosine < 0.50)

    target_distance = np.linalg.norm(candidate_mm - target_mm[:, None, :], axis=2)
    nearest_wall_distance = np.linalg.norm(anchor_to_cell, axis=2)
    anchor_separation = np.linalg.norm(anchor_delta, axis=2)
    nonlocal_threshold = np.maximum(3.0 * wall_spacing[sample_index], 1.5)
    prefilter_penetration_risk = (
        in_fit_prism
        & (owner_cosine < -0.25)
        & (anchor_separation > nonlocal_threshold[:, None])
        & (nearest_wall_distance < 0.8 * target_distance)
    )
    postfilter_penetration_risk = prefilter_penetration_risk & retained

    retained_count = retained.sum(axis=1).astype(np.int32)
    retained_envelope = np.max(
        np.where(retained, target_distance, -np.inf), axis=1
    )
    retained_envelope[~np.isfinite(retained_envelope)] = np.nan

    opposite_k = min(1024, len(wall_mm))
    opposite_distance_all, opposite_index_all = wall_tree.query(
        target_mm, k=opposite_k
    )
    if opposite_distance_all.ndim == 1:
        opposite_distance_all = opposite_distance_all[:, None]
        opposite_index_all = opposite_index_all[:, None]
    opposite_dot = np.einsum(
        "nki,ni->nk", full_normal[opposite_index_all], target_normal
    )
    opposite_mask = (
        (opposite_dot < -0.60)
        & (
            opposite_distance_all
            > np.maximum(3.0 * wall_spacing[sample_index, None], 1.0)
        )
    )
    masked_opposite = np.where(opposite_mask, opposite_distance_all, np.inf)
    opposite_column = np.argmin(masked_opposite, axis=1)
    opposite_distance = masked_opposite[np.arange(len(target_mm)), opposite_column]
    opposite_index = opposite_index_all[np.arange(len(target_mm)), opposite_column]
    no_opposite = ~np.isfinite(opposite_distance)
    opposite_distance[no_opposite] = np.nan
    opposite_index[no_opposite] = -1

    focus = _choose_focus(
        retained_envelope,
        opposite_distance,
        truth_wss,
        prefilter_penetration_risk.sum(axis=1),
    )

    centered_wall = wall_mm - wall_mm.mean(axis=0, keepdims=True)
    _, _, global_axes = np.linalg.svd(centered_wall, full_matrices=False)
    wall_uv = centered_wall @ global_axes[:2].T
    candidate_uv = (candidate_mm - wall_mm.mean(axis=0, keepdims=True)) @ global_axes[:2].T
    anchor_uv = (anchor_mm - wall_mm.mean(axis=0, keepdims=True)) @ global_axes[:2].T
    target_uv = (target_mm - wall_mm.mean(axis=0, keepdims=True)) @ global_axes[:2].T

    fig, axes = plt.subplots(3, 3, figsize=(18.5, 13.2), dpi=180)
    focus_colors = ("#d62728", "#ff9d00", "#2ca02c")
    focus_report: list[dict[str, Any]] = []

    for column, ((label, row), color) in enumerate(zip(focus, focus_colors)):
        selected = retained[row]
        guarded_opposite = opposite_guarded[row]
        guarded_ambiguous = ambiguous_guarded[row]
        risky = prefilter_penetration_risk[row]
        target = target_mm[row]
        target_plot = target_uv[row]
        selected_rank = np.flatnonzero(selected)
        if len(selected_rank) > 70:
            selected_line_rank = selected_rank[
                np.linspace(0, len(selected_rank) - 1, 70, dtype=int)
            ]
        else:
            selected_line_rank = selected_rank

        ax = axes[0, column]
        ax.scatter(
            wall_uv[:, 0], wall_uv[:, 1], s=1.4, color="#c7c7c7",
            alpha=0.34, edgecolors="none",
        )
        ax.scatter(
            candidate_uv[row, selected, 0], candidate_uv[row, selected, 1],
            s=22, color="#7b2cbf", alpha=0.90, edgecolors="none",
        )
        if guarded_opposite.any():
            ax.scatter(
                candidate_uv[row, guarded_opposite, 0],
                candidate_uv[row, guarded_opposite, 1],
                s=45, marker="x", color="#e31a1c", linewidths=1.0,
            )
        if risky.any():
            ax.scatter(
                candidate_uv[row, risky, 0], candidate_uv[row, risky, 1],
                s=85, marker="X", color="#e31a1c", edgecolors="black", linewidths=0.6,
            )
        ax.scatter(
            [target_plot[0]], [target_plot[1]], s=165, marker="*", color=color,
            edgecolors="black", linewidths=0.8, zorder=8,
        )
        ax.add_patch(
            Rectangle(
                (target_plot[0] - zoom_radius_mm, target_plot[1] - zoom_radius_mm),
                2 * zoom_radius_mm, 2 * zoom_radius_mm, fill=False,
                linestyle="--", linewidth=1.2, edgecolor=color,
            )
        )
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(
            f"{chr(65 + column)}. {label}\n"
            f"wall={int(sample_index[row])}, retained={int(selected.sum())}, caught risk={int(risky.sum())}"
        )
        ax.set_xlabel("wall PCA-1 (mm)")
        if column == 0:
            ax.set_ylabel("wall PCA-2 (mm)")
        ax.grid(alpha=0.18)

        ax = axes[1, column]
        within_zoom = (
            (np.abs(wall_uv[:, 0] - target_plot[0]) <= zoom_radius_mm)
            & (np.abs(wall_uv[:, 1] - target_plot[1]) <= zoom_radius_mm)
        )
        local_wall_index = np.flatnonzero(within_zoom)
        local_dot = full_normal[local_wall_index] @ target_normal[row]
        same_wall = local_dot >= 0.50
        opposite_wall = local_dot < -0.60
        other_wall = ~(same_wall | opposite_wall)
        ax.scatter(
            wall_uv[local_wall_index[other_wall], 0],
            wall_uv[local_wall_index[other_wall], 1],
            s=7, color="#d7d7d7", alpha=0.30, edgecolors="none",
        )
        ax.scatter(
            wall_uv[local_wall_index[same_wall], 0],
            wall_uv[local_wall_index[same_wall], 1],
            s=8, color="#aeb4bb", alpha=0.52, edgecolors="none", label="same-facing wall",
        )
        if opposite_wall.any():
            ax.scatter(
                wall_uv[local_wall_index[opposite_wall], 0],
                wall_uv[local_wall_index[opposite_wall], 1],
                s=20, facecolors="none", edgecolors="#e31a1c", linewidths=0.9,
                alpha=0.82, label="opposite-facing wall",
            )
        segments = np.stack(
            [anchor_uv[row, selected_line_rank], candidate_uv[row, selected_line_rank]],
            axis=1,
        )
        if len(segments):
            ax.add_collection(
                LineCollection(segments, colors="#167d8d", linewidths=0.55, alpha=0.40)
            )
        ax.scatter(
            anchor_uv[row, selected, 0], anchor_uv[row, selected, 1],
            s=8, color="#222222", alpha=0.32, edgecolors="none", label="nearest wall anchors",
        )
        ax.scatter(
            candidate_uv[row, selected, 0], candidate_uv[row, selected, 1],
            s=32, color="#7b2cbf", alpha=0.92, edgecolors="none", label="retained V4 cells",
        )
        if guarded_ambiguous.any():
            ax.scatter(
                candidate_uv[row, guarded_ambiguous, 0],
                candidate_uv[row, guarded_ambiguous, 1],
                s=42, marker="x", color="#ff9d00", linewidths=1.0,
                label="orientation guard rejected",
            )
        if guarded_opposite.any():
            ax.scatter(
                candidate_uv[row, guarded_opposite, 0],
                candidate_uv[row, guarded_opposite, 1],
                s=58, marker="x", color="#e31a1c", linewidths=1.2,
                label="opposite-wall candidate rejected",
            )
        if risky.any():
            ax.scatter(
                candidate_uv[row, risky, 0], candidate_uv[row, risky, 1],
                s=95, marker="X", color="#e31a1c", edgecolors="black", linewidths=0.7,
                label="cross-wall risk caught by guard",
            )
        ax.scatter(
            [target_plot[0]], [target_plot[1]], s=180, marker="*", color=color,
            edgecolors="black", linewidths=0.9, zorder=8, label="target wall point",
        )
        ax.add_patch(
            Circle(
                (target_plot[0], target_plot[1]), float(surface_radius[row]),
                fill=False, linestyle="--", linewidth=1.4, edgecolor=color,
                label=f"surface radius={surface_radius[row]:.2f} mm",
            )
        )
        ax.set_xlim(target_plot[0] - zoom_radius_mm, target_plot[0] + zoom_radius_mm)
        ax.set_ylim(target_plot[1] - zoom_radius_mm, target_plot[1] + zoom_radius_mm)
        ax.set_aspect("equal", adjustable="box")
        envelope_ratio = retained_envelope[row] / opposite_distance[row]
        ax.set_title(
            f"Wide wall context (±{zoom_radius_mm:g} mm)\n"
            f"dopp={opposite_distance[row]:.2f} mm, retained envelope={retained_envelope[row]:.2f} mm, "
            f"ratio={envelope_ratio:.2f}"
        )
        ax.set_xlabel("wall PCA-1 (mm)")
        if column == 0:
            ax.set_ylabel("wall PCA-2 (mm)")
        ax.grid(alpha=0.20)
        ax = axes[2, column]
        opposite_id = int(opposite_index[row])
        preferred = (
            wall_mm[opposite_id] - target if opposite_id >= 0 else None
        )
        tangent = _tangent_axis(target_normal[row], preferred)
        section_radius = max(
            2.5 * float(surface_radius[row]),
            1.25 * float(opposite_distance[row]),
            3.0,
        )
        section_wall_id = np.asarray(
            wall_tree.query_ball_point(target, 1.45 * section_radius), dtype=np.int64
        )
        section_wall_delta = wall_mm[section_wall_id] - target
        section_wall_t = section_wall_delta @ tangent
        section_wall_eta = section_wall_delta @ target_normal[row]
        section_dot = full_normal[section_wall_id] @ target_normal[row]
        section_same = section_dot >= 0.50
        section_opposite = section_dot < -0.60
        cell_delta = candidate_mm[row] - target
        cell_t = cell_delta @ tangent
        cell_eta = cell_delta @ target_normal[row]
        anchor_local_delta = anchor_mm[row] - target
        anchor_t = anchor_local_delta @ tangent
        anchor_local_eta = anchor_local_delta @ target_normal[row]
        local_segments = np.stack(
            [
                np.column_stack(
                    [anchor_t[selected_line_rank], anchor_local_eta[selected_line_rank]]
                ),
                np.column_stack([cell_t[selected_line_rank], cell_eta[selected_line_rank]]),
            ],
            axis=1,
        )
        ax.scatter(
            section_wall_t[section_same], section_wall_eta[section_same], s=8,
            color="#aeb4bb", alpha=0.50, edgecolors="none",
        )
        if section_opposite.any():
            ax.scatter(
                section_wall_t[section_opposite], section_wall_eta[section_opposite],
                s=19, facecolors="none", edgecolors="#e31a1c", linewidths=0.9,
                alpha=0.82,
            )
        if len(local_segments):
            ax.add_collection(
                LineCollection(local_segments, colors="#167d8d", linewidths=0.6, alpha=0.42)
            )
        ax.scatter(
            anchor_t[selected], anchor_local_eta[selected], s=9,
            color="#222222", alpha=0.34, edgecolors="none",
        )
        ax.scatter(
            cell_t[selected], cell_eta[selected], s=34,
            color="#7b2cbf", alpha=0.92, edgecolors="none",
        )
        if guarded_ambiguous.any():
            ax.scatter(
                cell_t[guarded_ambiguous], cell_eta[guarded_ambiguous], s=42,
                marker="x", color="#ff9d00", linewidths=1.0,
            )
        if guarded_opposite.any():
            ax.scatter(
                cell_t[guarded_opposite], cell_eta[guarded_opposite], s=60,
                marker="x", color="#e31a1c", linewidths=1.2,
            )
        if risky.any():
            ax.scatter(
                cell_t[risky], cell_eta[risky], s=100, marker="X",
                color="#e31a1c", edgecolors="black", linewidths=0.7,
            )
        ax.scatter(
            [0], [0], s=180, marker="*", color=color, edgecolors="black",
            linewidths=0.9, zorder=8,
        )
        ax.axhline(0, color="black", lw=0.8, alpha=0.65)
        ax.annotate(
            "target inward normal", xy=(0, 0.82 * section_radius), xytext=(0, 0),
            arrowprops={"arrowstyle": "->", "color": "#2878b5", "lw": 1.4},
            ha="center", va="bottom", fontsize=8, color="#2878b5",
        )
        ax.set_xlim(-section_radius, section_radius)
        ax.set_ylim(-0.32 * section_radius, section_radius)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(
            f"Local tangent-normal section\n"
            f"fit depth={fit_depth[row]:.2f} mm; opposite candidates guarded={int(guarded_opposite.sum())}"
        )
        ax.set_xlabel("local tangent coordinate (mm)")
        if column == 0:
            ax.set_ylabel("target-normal coordinate (mm)")
        ax.grid(alpha=0.20)

        focus_report.append(
            {
                "label": label,
                "sample_row": int(row),
                "wall_index": int(sample_index[row]),
                "truth_wss_pa": float(truth_wss[row]),
                "chosen_depth_fraction": float(chosen_fraction[row]),
                "fit_depth_with_overshoot_mm": float(fit_depth[row]),
                "surface_radius_mm": float(surface_radius[row]),
                "retained_support_points": int(selected.sum()),
                "opposite_candidates_guarded": int(guarded_opposite.sum()),
                "ambiguous_orientation_candidates_guarded": int(guarded_ambiguous.sum()),
                "prefilter_penetration_risk_points_caught": int(risky.sum()),
                "postfilter_penetration_risk_points_retained": int(
                    postfilter_penetration_risk[row].sum()
                ),
                "retained_envelope_mm": float(retained_envelope[row]),
                "opposite_wall_distance_mm": float(opposite_distance[row]),
                "retained_envelope_over_opposite_distance": float(envelope_ratio),
            }
        )

    fig.suptitle(
        f"{'Profile-Secant V3 depth-3 support' if support_prefix == 'depth3' else 'Closest-surface MLS V4'} "
        f"penetration audit — {canonical_id}, peak {step}\n"
        f"wall targets={len(target_mm):,}; retained support occurrences={int(retained.sum()):,}; "
        f"opposite candidates rejected={int(opposite_guarded.sum()):,}; "
        f"cross-wall risks caught={int(prefilter_penetration_risk.sum()):,}; "
        f"post-filter retained risk={int(postfilter_penetration_risk.sum()):,}",
        fontsize=13.2,
    )
    legend_pairs = [
        pair
        for ax in axes.flat
        for pair in zip(*ax.get_legend_handles_labels())
        if pair[1]
    ]
    if legend_pairs:
        unique = {label: handle for handle, label in legend_pairs}
        fig.legend(
            unique.values(), unique.keys(), loc="lower center", ncol=5,
            fontsize=8, frameon=True, bbox_to_anchor=(0.5, 0.012),
        )
    fig.subplots_adjust(
        left=0.05, right=0.99, bottom=0.075, top=0.90, hspace=0.34, wspace=0.20
    )

    finite_opposite = np.isfinite(opposite_distance)
    report = {
        "case": canonical_id,
        "step": step,
        "cache": str(cache_path.resolve()),
        "support_prefix": support_prefix,
        "support_name": (
            "Profile-Secant V3 depth-3 support"
            if support_prefix == "depth3"
            else "Surface-MLS V4 base support"
        ),
        "scope": (
            f"All {len(target_mm):,} wall targets stored in this cache; "
            "counts are target-support occurrences, not unique interior cells."
        ),
        "n_full_wall_points": int(len(wall_mm)),
        "n_interior_cells": int(len(interior_mm)),
        "n_sampled_targets": int(len(target_mm)),
        "retained_support_occurrences": int(retained.sum()),
        "opposite_candidates_rejected_by_orientation_guard": int(opposite_guarded.sum()),
        "ambiguous_candidates_rejected_by_orientation_guard": int(ambiguous_guarded.sum()),
        "targets_with_opposite_candidates_rejected": int(
            np.count_nonzero(opposite_guarded.sum(axis=1))
        ),
        "prefilter_penetration_risk_count_caught_by_orientation_guard": int(
            prefilter_penetration_risk.sum()
        ),
        "postfilter_penetration_risk_count_retained": int(
            postfilter_penetration_risk.sum()
        ),
        "prefilter_penetration_risk_fraction_of_prism_candidates": float(
            prefilter_penetration_risk.sum() / max(int(in_fit_prism.sum()), 1)
        ),
        "retained_count_p05_p50_p95": [
            float(value) for value in np.quantile(retained_count, [0.05, 0.50, 0.95])
        ],
        "opposite_wall_distance_p05_p50_p95_mm": [
            float(value)
            for value in np.quantile(opposite_distance[finite_opposite], [0.05, 0.50, 0.95])
        ],
        "focus": focus_report,
        "selection_definition": (
            "Among the 384 nearest interior cells, retain local_depth>0, "
            "local_depth<=1.15*chosen_fraction*depth_limit, closest-wall-anchor "
            "surface distance<=surface_radius, and anchor-normal dot target-normal>=0.5."
        ),
        "risk_definition": (
            "Before the orientation guard, a cell is closer to a nonlocal/opposite wall patch: "
            "anchor-normal dot target-normal<-0.25, anchor separation>max(3*local "
            "wall spacing,1.5mm), and nearest-wall distance<0.8*distance to target. "
            "The retained post-filter count applies the owner-normal>=0.5 guard as well."
        ),
        "limitation": (
            "This Euclidean closest-wall audit detects opposite-facing cross-wall mixing. "
            "Without wall topology it cannot prove geodesic branch ownership for "
            "same-facing, spatially adjacent surfaces."
        ),
    }
    return report, fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default=str(DEFAULT_CACHE))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--zoom-radius-mm", type=float, default=20.0)
    parser.add_argument(
        "--support-prefix",
        choices=("surface", "depth3"),
        default="surface",
        help="Use the base V4 support or the depth-3 Profile-Secant support.",
    )
    args = parser.parse_args()

    cache_path = Path(args.cache).resolve()
    output_path = Path(args.out).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report, figure = audit(
        cache_path,
        float(args.zoom_radius_mm),
        args.support_prefix,
    )
    figure.savefig(output_path)
    plt.close(figure)
    json_path = output_path.with_suffix(".json")
    report["figure"] = str(output_path)
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(output_path)


if __name__ == "__main__":
    main()
