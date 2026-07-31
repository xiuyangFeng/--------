"""纯点云 velocity→WSS 近邻是否穿模的空间审计图。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
from scipy.spatial import cKDTree


VIZ = Path(__file__).resolve().parent
SRC = VIZ.parent / "src"
REPO = VIZ.parents[1]
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(VIZ))

import data_loader_cfd as dl  # noqa: E402
from calculate_wss_cfd import (  # noqa: E402
    DEFAULT_ADAPTIVE_NEIGHBORS,
    DEFAULT_CV_TOLERANCE,
    fit_wall_gradient,
    pca_wall_normals,
    resolve_step,
)
from diagnose_wss_sampling import load_local_radius_mm  # noqa: E402


def tangent_axis(normal: np.ndarray, preferred: np.ndarray | None = None) -> np.ndarray:
    if preferred is not None:
        tangent = preferred - float(preferred @ normal) * normal
        magnitude = np.linalg.norm(tangent)
        if magnitude > 1e-8:
            return tangent / magnitude
    seed = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.8 else np.array([0.0, 1.0, 0.0])
    tangent = np.cross(normal, seed)
    return tangent / np.linalg.norm(tangent)


def opposite_wall_info(
    wall_index: int,
    wall_mm: np.ndarray,
    normals: np.ndarray,
    wall_tree: cKDTree,
    spacing_mm: float,
    search_k: int = 512,
) -> tuple[float, int | None]:
    distance, index = wall_tree.query(wall_mm[wall_index], k=min(search_k, len(wall_mm)))
    dot = normals[index] @ normals[wall_index]
    mask = (dot < -0.8) & (distance > max(3.0 * spacing_mm, 1.0))
    if not mask.any():
        return float("nan"), None
    location = int(np.flatnonzero(mask)[np.argmin(distance[mask])])
    return float(distance[location]), int(index[location])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case-dir",
        default="/public/newhome/cy/Digital_twin/GNN/data_new/AG/fast/RAN_QING_BO",
    )
    parser.add_argument("--audit-sample", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--out-dir", default="")
    args = parser.parse_args()

    case_dir = Path(args.case_dir).resolve()
    step, step_source = resolve_step(case_dir, None, prefer_peak=True)
    wall = dl.load_wall(dl.step_path(case_dir, step, "ascii"))
    interior = dl.load_interior(dl.step_path(case_dir, step, "ascii_in"))
    wall_mm = wall["coords_mm"]
    interior_mm = interior["coords_mm"]
    wall_tree = cKDTree(wall_mm)
    interior_tree = cKDTree(interior_mm)
    normals = pca_wall_normals(wall_mm, interior_mm, interior_tree)
    local_radius = load_local_radius_mm(case_dir, wall_mm)

    # 对全部壁面点运行冻结 adaptive-CV，取到每点真正使用的 K。
    _, used_count, diagnostics = fit_wall_gradient(
        wall_mm,
        normals,
        interior_mm,
        interior["velocity"],
        interior_tree,
        neighbor_mode="adaptive_cv",
        adaptive_neighbors=DEFAULT_ADAPTIVE_NEIGHBORS,
        cv_tolerance=DEFAULT_CV_TOLERANCE,
    )
    selected_k = diagnostics["selected_neighbors"].astype(np.int32)
    max_k = max(DEFAULT_ADAPTIVE_NEIGHBORS)
    interior_distance, interior_index = interior_tree.query(wall_mm, k=max_k)
    delta = interior_mm[interior_index] - wall_mm[:, None, :]
    eta = np.einsum("nki,ni->nk", delta, normals)
    rank = np.arange(max_k)[None, :]
    inside_chosen_k = rank < selected_k[:, None]
    selected_mask = inside_chosen_k & (eta > 1e-4)
    dropped_negative = inside_chosen_k & ~selected_mask

    # 最近壁面锚点审计：若选中内部点明显更靠近反向、非局部壁面，则记为穿模风险。
    selected_interior_id = interior_index[selected_mask]
    selected_target_id = np.broadcast_to(
        np.arange(len(wall_mm))[:, None], selected_mask.shape
    )[selected_mask]
    selected_delta = delta[selected_mask]
    target_distance = np.linalg.norm(selected_delta, axis=1)
    nearest_wall_distance, nearest_wall_index = wall_tree.query(
        interior_mm[selected_interior_id], k=1
    )
    nearest_normal_dot = np.einsum(
        "ij,ij->i", normals[nearest_wall_index], normals[selected_target_id]
    )
    anchor_separation = np.linalg.norm(
        wall_mm[nearest_wall_index] - wall_mm[selected_target_id], axis=1
    )
    wall_spacing, _ = wall_tree.query(wall_mm, k=8)
    wall_spacing = np.median(wall_spacing[:, 1:], axis=1)
    cross_risk = (
        (nearest_normal_dot < -0.25)
        & (anchor_separation > np.maximum(3.0 * wall_spacing[selected_target_id], 1.5))
        & (nearest_wall_distance < 0.8 * target_distance)
    )

    # 每个 target 的穿模风险计数，供焦点和全局统计使用。
    risk_count = np.bincount(
        selected_target_id, weights=cross_risk.astype(np.int32), minlength=len(wall_mm)
    ).astype(np.int32)
    selected_count = np.bincount(selected_target_id, minlength=len(wall_mm)).astype(np.int32)
    eta_max = np.max(np.where(selected_mask, eta, -np.inf), axis=1)
    eta_max[~np.isfinite(eta_max)] = np.nan
    depth_ratio = (
        eta_max / np.maximum(local_radius, 1e-6)
        if local_radius is not None
        else np.full(len(wall_mm), np.nan)
    )

    # 在确定性子样本上搜索“查询球最接近对侧壁”的位置。
    rng = np.random.default_rng(args.seed)
    audit_index = np.sort(
        rng.choice(len(wall_mm), min(args.audit_sample, len(wall_mm)), replace=False)
    )
    audit_wall_distance, audit_wall_index = wall_tree.query(wall_mm[audit_index], k=512)
    audit_dot = np.einsum(
        "nki,ni->nk", normals[audit_wall_index], normals[audit_index]
    )
    audit_spacing = wall_spacing[audit_index]
    audit_opposite_mask = (
        (audit_dot < -0.8)
        & (audit_wall_distance > np.maximum(3.0 * audit_spacing[:, None], 1.0))
    )
    audit_opposite_distance = np.min(
        np.where(audit_opposite_mask, audit_wall_distance, np.inf), axis=1
    )
    audit_k_radius = np.asarray(
        [interior_distance[wi, max(int(selected_k[wi]) - 1, 0)] for wi in audit_index]
    )
    proximity_ratio = audit_k_radius / audit_opposite_distance

    focus_depth = int(np.nanargmax(depth_ratio)) if np.isfinite(depth_ratio).any() else int(audit_index[0])
    finite_opposite = np.isfinite(audit_opposite_distance)
    focus_proximity = int(
        audit_index[np.nanargmax(np.where(finite_opposite, proximity_ratio, np.nan))]
    )
    close_high = finite_opposite & (audit_opposite_distance < 5.0)
    focus_high_wss = int(
        audit_index[np.argmax(np.where(close_high, wall["wss_mag"][audit_index], -np.inf))]
    )
    focus_list: list[int] = []
    focus_label_list: list[str] = []
    for wi, label in (
        (focus_depth, "largest depth/radius"),
        (focus_proximity, "closest query/opposite-wall"),
        (focus_high_wss, "high-WSS near narrow wall"),
    ):
        if wi not in focus_list:
            focus_list.append(int(wi))
            focus_label_list.append(label)
    if len(focus_list) < 3:
        extras = audit_index[np.argsort(wall["wss_mag"][audit_index])[::-1]]
        for wi in extras:
            if int(wi) not in focus_list:
                focus_list.append(int(wi))
                focus_label_list.append("fallback high-WSS")
            if len(focus_list) == 3:
                break
    focus = np.asarray(focus_list[:3], dtype=int)
    focus_labels = focus_label_list[:3]

    # 全局 PCA 投影，风格与用户给出的 group-overlap 示例一致。
    centered = wall_mm - wall_mm.mean(axis=0)
    _, _, global_axes = np.linalg.svd(centered, full_matrices=False)
    global_uv = centered @ global_axes[:2].T

    fig = plt.figure(figsize=(18, 5.8), dpi=180)
    grid = fig.add_gridspec(1, 4, width_ratios=[1.08, 1, 1, 1], wspace=0.28)
    ax = fig.add_subplot(grid[0, 0])
    ax.scatter(global_uv[:, 0], global_uv[:, 1], s=2, color="#cfcfcf", alpha=0.42, edgecolors="none")
    focus_colors = ["#d62728", "#ff7f0e", "#2ca02c"]
    for color, label, wi in zip(focus_colors, focus_labels, focus):
        ax.scatter(
            global_uv[wi, 0], global_uv[wi, 1], s=130, marker="*", color=color,
            edgecolors="k", linewidths=0.7, zorder=5, label=f"{label} (wall {wi})",
        )
        ax.add_patch(
            Circle(
                (global_uv[wi, 0], global_uv[wi, 1]),
                radius=float(interior_distance[wi, max(int(selected_k[wi]) - 1, 0)]),
                fill=False, linestyle="--", linewidth=1.2, edgecolor=color,
            )
        )
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title("A. Full wall point cloud\n(dashed circle = projected adaptive K radius)")
    ax.set_xlabel("wall PCA-1 (mm)")
    ax.set_ylabel("wall PCA-2 (mm)")
    ax.legend(fontsize=7, loc="best")
    ax.grid(alpha=0.18)

    focus_report = []
    for panel, (color, label, wi) in enumerate(zip(focus_colors, focus_labels, focus), start=1):
        ax = fig.add_subplot(grid[0, panel])
        spacing = float(wall_spacing[wi])
        opposite_distance, opposite_index = opposite_wall_info(
            wi, wall_mm, normals, wall_tree, spacing
        )
        preferred = (
            wall_mm[opposite_index] - wall_mm[wi] if opposite_index is not None else None
        )
        tangent = tangent_axis(normals[wi], preferred)
        k = int(selected_k[wi])
        k_radius = float(interior_distance[wi, max(k - 1, 0)])
        view_radius = max(2.3 * k_radius, 1.15 * opposite_distance if np.isfinite(opposite_distance) else 4.0)
        local_wall_id = np.asarray(wall_tree.query_ball_point(wall_mm[wi], view_radius), dtype=int)
        local_wall_delta = wall_mm[local_wall_id] - wall_mm[wi]
        wall_t = local_wall_delta @ tangent
        wall_eta = local_wall_delta @ normals[wi]
        wall_dot = normals[local_wall_id] @ normals[wi]
        same_wall = wall_dot >= -0.25
        opposite_wall = wall_dot < -0.8

        candidate_id = interior_index[wi]
        candidate_delta = delta[wi]
        candidate_t = candidate_delta @ tangent
        candidate_eta = eta[wi]
        chosen = np.arange(max_k) < k
        used = chosen & (candidate_eta > 1e-4)
        dropped = chosen & ~used
        not_chosen = ~chosen

        # 对本焦点逐点映射全局 cross_risk 判据。
        used_ids = candidate_id[used]
        used_target_distance = np.linalg.norm(candidate_delta[used], axis=1)
        used_nearest_distance, used_nearest_wall = wall_tree.query(interior_mm[used_ids], k=1)
        used_dot = normals[used_nearest_wall] @ normals[wi]
        used_anchor_sep = np.linalg.norm(wall_mm[used_nearest_wall] - wall_mm[wi], axis=1)
        used_cross = (
            (used_dot < -0.25)
            & (used_anchor_sep > max(3.0 * spacing, 1.5))
            & (used_nearest_distance < 0.8 * used_target_distance)
        )

        ax.scatter(wall_t[same_wall], wall_eta[same_wall], s=7, color="#bdbdbd", alpha=0.5, label="local wall")
        if opposite_wall.any():
            ax.scatter(
                wall_t[opposite_wall], wall_eta[opposite_wall], s=13, facecolors="none",
                edgecolors="#d62728", linewidths=0.8, alpha=0.8, label="opposite-facing wall",
            )
        ax.scatter(
            candidate_t[not_chosen], candidate_eta[not_chosen], s=24, facecolors="none",
            edgecolors="#ff7f0e", linewidths=0.8, alpha=0.7, label="Kmax candidate, not selected",
        )
        if dropped.any():
            ax.scatter(
                candidate_t[dropped], candidate_eta[dropped], s=34, marker="x",
                color="#666666", linewidths=1.0, label="discarded: eta<=0",
            )
        safe_used = np.flatnonzero(used)[~used_cross]
        risky_used = np.flatnonzero(used)[used_cross]
        ax.scatter(
            candidate_t[safe_used], candidate_eta[safe_used], s=34, color="#7b2cbf",
            alpha=0.88, edgecolors="none", label="selected for WSS gradient",
        )
        if len(risky_used):
            ax.scatter(
                candidate_t[risky_used], candidate_eta[risky_used], s=65, color="#e31a1c",
                marker="X", edgecolors="k", linewidths=0.5, label="penetration risk",
            )
        ax.scatter([0], [0], s=135, marker="*", color=color, edgecolors="k", linewidths=0.7, zorder=7, label="target wall point")
        ax.add_patch(Circle((0, 0), k_radius, fill=False, linestyle="--", linewidth=1.2, edgecolor=color))
        ax.axhline(0, color="k", linewidth=0.8, alpha=0.65)
        ax.annotate(
            "inward normal",
            xy=(0, min(k_radius, 0.8 * view_radius)), xytext=(0, 0),
            arrowprops={"arrowstyle": "->", "color": "#1f77b4", "lw": 1.5},
            ha="center", va="bottom", fontsize=8, color="#1f77b4",
        )
        ax.set_xlim(-view_radius, view_radius)
        ax.set_ylim(-0.25 * view_radius, view_radius)
        ax.set_aspect("equal", adjustable="box")
        ratio_text = (
            f"rK/dopp={k_radius/opposite_distance:.2f}" if np.isfinite(opposite_distance) else "dopp=N/A"
        )
        ax.set_title(
            f"{chr(65+panel)}. {label}\nwall={wi}, K={k}, rK={k_radius:.2f}mm, {ratio_text}\n"
            f"selected={int(used.sum())}, penetration-risk={int(used_cross.sum())}"
        )
        ax.set_xlabel("local tangent coordinate (mm)")
        if panel == 1:
            ax.set_ylabel("inward normal distance eta (mm)")
        ax.grid(alpha=0.2)
        if panel == 3:
            ax.legend(fontsize=6.6, loc="upper right")

        focus_report.append(
            {
                "wall_index": int(wi),
                "label": label,
                "truth_wss_pa": float(wall["wss_mag"][wi]),
                "selected_k": k,
                "selected_count_eta_positive": int(used.sum()),
                "discarded_eta_nonpositive": int(dropped.sum()),
                "penetration_risk_count": int(used_cross.sum()),
                "k_radius_mm": k_radius,
                "opposite_wall_distance_mm": opposite_distance,
                "k_radius_over_opposite_distance": (
                    float(k_radius / opposite_distance) if np.isfinite(opposite_distance) else None
                ),
                "eta_max_mm": float(np.max(candidate_eta[used])) if used.any() else None,
                "local_radius_mm": (
                    float(local_radius[wi]) if local_radius is not None else None
                ),
            }
        )

    total_selected = int(selected_mask.sum())
    total_dropped = int(dropped_negative.sum())
    global_risk = int(cross_risk.sum())
    fig.suptitle(
        f"Adaptive point-cloud WSS neighbor penetration audit — {case_dir.name}, peak {step}\n"
        f"all wall targets={len(wall_mm):,}; selected interior points={total_selected:,}; "
        f"eta<=0 discarded={total_dropped:,}; penetration-risk={global_risk:,} "
        f"({global_risk/max(total_selected,1):.3%})",
        fontsize=12,
    )
    fig.subplots_adjust(left=0.05, right=0.99, bottom=0.12, top=0.80, wspace=0.30)

    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else REPO
        / "outputs/wss_pinn/audits/cfd_velocity_wss_explore/04_sampling_diagnostics"
        / case_dir.name
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{case_dir.name}_peak{step}_penetration_check"
    figure_path = out_dir / f"{stem}.png"
    json_path = out_dir / f"{stem}.json"
    fig.savefig(figure_path)
    plt.close(fig)

    report = {
        "case": str(case_dir.relative_to(REPO / "data_new")),
        "step": int(step),
        "step_source": step_source,
        "method": {
            "neighbor_mode": "adaptive_cv",
            "candidate_k": list(DEFAULT_ADAPTIVE_NEIGHBORS),
            "cv_tolerance": DEFAULT_CV_TOLERANCE,
        },
        "n_wall_targets": int(len(wall_mm)),
        "n_interior": int(len(interior_mm)),
        "selected_interior_points": total_selected,
        "discarded_eta_nonpositive": total_dropped,
        "penetration_risk_count": global_risk,
        "penetration_risk_fraction": float(global_risk / max(total_selected, 1)),
        "selected_k_p05_p50_p95": [
            float(value) for value in np.quantile(selected_k[selected_k > 0], [0.05, 0.5, 0.95])
        ],
        "eta_max_p50_p95_max_mm": [
            float(value) for value in np.nanquantile(eta_max, [0.5, 0.95, 1.0])
        ],
        "depth_over_local_radius_p50_p95_max": (
            [float(value) for value in np.nanquantile(depth_ratio, [0.5, 0.95, 1.0])]
            if local_radius is not None
            else None
        ),
        "focus": focus_report,
        "risk_definition": (
            "selected interior point is nearer to an opposite/nonlocal wall patch: "
            "normal_dot<-0.25, anchor separation>max(3*local wall spacing,1.5mm), "
            "and nearest-wall distance<0.8*distance to target wall point"
        ),
        "figure": str(figure_path),
    }
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(figure_path)


if __name__ == "__main__":
    main()
