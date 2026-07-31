"""在完整壁面投影上叠加 WSS 自适应近邻，并提供同投影局部放大。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np
from scipy.spatial import cKDTree


VIZ = Path(__file__).resolve().parent
SRC = VIZ.parent / "src"
REPO = VIZ.parents[1]
sys.path.insert(0, str(SRC))

import data_loader_cfd as dl  # noqa: E402
from calculate_wss_cfd import pca_wall_normals, resolve_step  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case-dir",
        default="/public/newhome/cy/Digital_twin/GNN/data_new/AG/fast/RAN_QING_BO",
    )
    parser.add_argument("--audit-json", default="")
    parser.add_argument("--out-dir", default="")
    parser.add_argument(
        "--zoom-radius-mm",
        type=float,
        default=20.0,
        help="Lower-panel physical half-width in mm (default: 20, i.e. a 40 mm field of view).",
    )
    args = parser.parse_args()
    if args.zoom_radius_mm <= 0:
        parser.error("--zoom-radius-mm must be positive")

    case_dir = Path(args.case_dir).resolve()
    step, _ = resolve_step(case_dir, None, prefer_peak=True)
    default_out = (
        REPO
        / "outputs/wss_pinn/audits/cfd_velocity_wss_explore/04_sampling_diagnostics"
        / case_dir.name
    )
    audit_json = (
        Path(args.audit_json)
        if args.audit_json
        else default_out / f"{case_dir.name}_peak{step}_penetration_check.json"
    )
    audit = json.loads(audit_json.read_text(encoding="utf-8"))

    wall = dl.load_wall(dl.step_path(case_dir, step, "ascii"))
    interior = dl.load_interior(dl.step_path(case_dir, step, "ascii_in"))
    wall_mm = wall["coords_mm"]
    interior_mm = interior["coords_mm"]
    wall_tree = cKDTree(wall_mm)
    interior_tree = cKDTree(interior_mm)
    normals = pca_wall_normals(wall_mm, interior_mm, interior_tree)

    centered = wall_mm - wall_mm.mean(axis=0)
    _, _, projection_axes = np.linalg.svd(centered, full_matrices=False)
    wall_uv = centered @ projection_axes[:2].T
    wall_min = wall_uv.min(axis=0)
    wall_max = wall_uv.max(axis=0)
    padding = 0.04 * float(np.max(wall_max - wall_min))

    focus = audit["focus"]
    colors = ["#d62728", "#ff7f0e", "#2ca02c"]
    fig, axes = plt.subplots(2, 3, figsize=(15.6, 10.2), dpi=180)

    legend_handles = None
    legend_labels = None
    overlay_report = []
    for column, (entry, color) in enumerate(zip(focus, colors)):
        wi = int(entry["wall_index"])
        k = int(entry["selected_k"])
        target = wall_mm[wi]
        target_uv = wall_uv[wi]
        distance, candidate_id = interior_tree.query(target, k=64)
        candidate_xyz = interior_mm[candidate_id]
        candidate_uv = (candidate_xyz - wall_mm.mean(axis=0)) @ projection_axes[:2].T
        delta = candidate_xyz - target
        eta = delta @ normals[wi]
        chosen = np.arange(64) < k
        selected = chosen & (eta > 1e-4)
        dropped = chosen & ~selected
        not_selected = ~chosen

        selected_id = candidate_id[selected]
        selected_target_distance = np.linalg.norm(delta[selected], axis=1)
        nearest_wall_distance, nearest_wall_id = wall_tree.query(interior_mm[selected_id], k=1)
        nearest_dot = normals[nearest_wall_id] @ normals[wi]
        anchor_separation = np.linalg.norm(wall_mm[nearest_wall_id] - target, axis=1)
        spacing_distance, _ = wall_tree.query(target, k=8)
        spacing = float(np.median(spacing_distance[1:]))
        selected_risk = (
            (nearest_dot < -0.25)
            & (anchor_separation > max(3.0 * spacing, 1.5))
            & (nearest_wall_distance < 0.8 * selected_target_distance)
        )
        selected_rank = np.flatnonzero(selected)
        safe_rank = selected_rank[~selected_risk]
        risk_rank = selected_rank[selected_risk]

        k_radius = float(distance[max(k - 1, 0)])
        # Keep enough vessel context to recognize branches while ensuring the
        # actual adaptive-neighbor circle is never cropped by the lower panel.
        zoom_radius = max(float(args.zoom_radius_mm), 5.0 * k_radius)
        local_wall_id = np.asarray(wall_tree.query_ball_point(target, zoom_radius), dtype=int)
        local_uv = wall_uv[local_wall_id]
        local_dot = normals[local_wall_id] @ normals[wi]
        same_wall = local_dot >= -0.25
        opposite_wall = local_dot < -0.8

        # 上排：完整壁面轮廓 + 实际近邻位置。
        ax = axes[0, column]
        ax.scatter(wall_uv[:, 0], wall_uv[:, 1], s=2, color="#cfcfcf", alpha=0.36, edgecolors="none")
        ax.scatter(
            candidate_uv[not_selected, 0], candidate_uv[not_selected, 1],
            s=27, facecolors="none", edgecolors="#ff7f0e", linewidths=0.9,
            alpha=0.8, label="Kmax candidate, not selected",
        )
        if dropped.any():
            ax.scatter(
                candidate_uv[dropped, 0], candidate_uv[dropped, 1], s=38,
                marker="x", color="#666666", linewidths=1.1, label="discarded: eta<=0",
            )
        ax.scatter(
            candidate_uv[safe_rank, 0], candidate_uv[safe_rank, 1], s=39,
            color="#7b2cbf", alpha=0.92, edgecolors="none", label="selected for WSS gradient",
        )
        if len(risk_rank):
            ax.scatter(
                candidate_uv[risk_rank, 0], candidate_uv[risk_rank, 1], s=75,
                marker="X", color="#e31a1c", edgecolors="k", linewidths=0.6,
                label="penetration risk",
            )
        ax.scatter(
            [target_uv[0]], [target_uv[1]], s=155, marker="*", color=color,
            edgecolors="k", linewidths=0.8, zorder=8, label="target wall point",
        )
        ax.add_patch(
            Rectangle(
                (target_uv[0] - zoom_radius, target_uv[1] - zoom_radius),
                2 * zoom_radius, 2 * zoom_radius, fill=False, linestyle="--",
                linewidth=1.2, edgecolor=color, label="lower-panel zoom region",
            )
        )
        ax.set_xlim(wall_min[0] - padding, wall_max[0] + padding)
        ax.set_ylim(wall_min[1] - padding, wall_max[1] + padding)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(
            f"{chr(65+column)}. {entry['label']}\n"
            f"wall={wi}, K={k}, selected={int(selected.sum())}, risk={int(selected_risk.sum())}"
        )
        ax.set_xlabel("wall PCA-1 (mm)")
        if column == 0:
            ax.set_ylabel("wall PCA-2 (mm)")
        ax.grid(alpha=0.18)

        # 下排：保持相同 PCA 投影，只放大目标附近，直观看点是否落到对侧壁。
        ax = axes[1, column]
        ax.scatter(
            local_uv[same_wall, 0], local_uv[same_wall, 1], s=8,
            color="#bdbdbd", alpha=0.48, edgecolors="none", label="local wall",
        )
        if opposite_wall.any():
            ax.scatter(
                local_uv[opposite_wall, 0], local_uv[opposite_wall, 1], s=18,
                facecolors="none", edgecolors="#d62728", linewidths=0.9,
                alpha=0.85, label="opposite-facing wall",
            )
        ax.scatter(
            candidate_uv[not_selected, 0], candidate_uv[not_selected, 1], s=34,
            facecolors="none", edgecolors="#ff7f0e", linewidths=1.0,
            alpha=0.85, label="Kmax candidate, not selected",
        )
        if dropped.any():
            ax.scatter(
                candidate_uv[dropped, 0], candidate_uv[dropped, 1], s=44,
                marker="x", color="#666666", linewidths=1.2, label="discarded: eta<=0",
            )
        ax.scatter(
            candidate_uv[safe_rank, 0], candidate_uv[safe_rank, 1], s=48,
            color="#7b2cbf", alpha=0.94, edgecolors="none", label="selected for WSS gradient",
        )
        if len(risk_rank):
            ax.scatter(
                candidate_uv[risk_rank, 0], candidate_uv[risk_rank, 1], s=90,
                marker="X", color="#e31a1c", edgecolors="k", linewidths=0.7,
                label="penetration risk",
            )
        ax.scatter(
            [target_uv[0]], [target_uv[1]], s=180, marker="*", color=color,
            edgecolors="k", linewidths=0.9, zorder=8, label="target wall point",
        )
        ax.add_patch(
            Circle(
                (target_uv[0], target_uv[1]), k_radius, fill=False,
                linestyle="--", linewidth=1.4, edgecolor=color,
                label=f"adaptive K radius = {k_radius:.2f} mm",
            )
        )
        ax.set_xlim(target_uv[0] - zoom_radius, target_uv[0] + zoom_radius)
        ax.set_ylim(target_uv[1] - zoom_radius, target_uv[1] + zoom_radius)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(
            f"Wider wall context (±{zoom_radius:.0f} mm) · rK={k_radius:.2f} mm\n"
            f"opposite-wall distance={entry['opposite_wall_distance_mm']:.2f} mm, "
            f"rK/dopp={entry['k_radius_over_opposite_distance']:.2f}"
        )
        ax.set_xlabel("wall PCA-1 (mm)")
        if column == 0:
            ax.set_ylabel("wall PCA-2 (mm)")
        ax.grid(alpha=0.2)
        if column == 0:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

        overlay_report.append(
            {
                "wall_index": wi,
                "label": entry["label"],
                "selected_k": k,
                "selected_points": int(selected.sum()),
                "discarded_eta_nonpositive": int(dropped.sum()),
                "penetration_risk_points": int(selected_risk.sum()),
                "k_radius_mm": k_radius,
                "zoom_radius_mm": zoom_radius,
            }
        )

    fig.suptitle(
        f"WSS adaptive-neighbor penetration check on the full wall projection — {case_dir.name}, peak {step}\n"
        "purple = selected gradient points; red X = detected penetration risk",
        fontsize=13,
    )
    if legend_handles is not None:
        unique = dict(zip(legend_labels, legend_handles))
        fig.legend(
            unique.values(), unique.keys(), loc="lower center", ncol=4,
            fontsize=8, frameon=True, bbox_to_anchor=(0.5, 0.015),
        )
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.10, top=0.88, hspace=0.25, wspace=0.18)

    out_dir = Path(args.out_dir) if args.out_dir else default_out
    out_dir.mkdir(parents=True, exist_ok=True)
    zoom_tag = f"{args.zoom_radius_mm:g}".replace(".", "p")
    stem = f"{case_dir.name}_peak{step}_penetration_fullwall_overlay_wide{zoom_tag}mm"
    figure_path = out_dir / f"{stem}.png"
    json_path = out_dir / f"{stem}.json"
    fig.savefig(figure_path)
    plt.close(fig)
    json_path.write_text(
        json.dumps(
            {
                "case": audit["case"],
                "step": step,
                "source_audit": str(audit_json),
                "global_penetration_risk_count": audit["penetration_risk_count"],
                "requested_zoom_radius_mm": float(args.zoom_radius_mm),
                "focus": overlay_report,
                "figure": str(figure_path),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(figure_path)


if __name__ == "__main__":
    main()
