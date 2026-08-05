"""Generate the frozen V4 sampling, calibration, and accuracy quicklook atlas."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy.spatial import cKDTree


VIZ = Path(__file__).resolve().parent
PACKAGE = VIZ.parent
SRC = PACKAGE / "src"
ROOT = PACKAGE.parent
sys.path.insert(0, str(SRC))

import data_loader_cfd as dl  # noqa: E402
from calculate_wss_cfd import pca_wall_normals, wss_from_gradient  # noqa: E402


DEFAULT_BASE_RESULT = (
    ROOT
    / "outputs/wss_mri_calculator/pointcloud_surface_mls_v4/test35_base_s1200.json"
)
DEFAULT_CAL_RESULT = (
    ROOT
    / "outputs/wss_mri_calculator/pointcloud_surface_mls_v4/test35_calibrated_s1200.json"
)
DEFAULT_PREDICTIONS = (
    ROOT
    / "outputs/wss_mri_calculator/pointcloud_surface_mls_v4/"
    "test35_calibrated_predictions_s1200.npz"
)
DEFAULT_CACHE_DIR = (
    ROOT
    / "outputs/wss_mri_calculator/pointcloud_surface_mls_v4/point_cache_test35_s1200"
)
DEFAULT_OUT = (
    ROOT / "outputs/wss_mri_calculator/pointcloud_surface_mls_v4/00_quicklook"
)

COLORS = {
    "v3": "#6b7280",
    "base": "#167d8d",
    "cal": "#c84b31",
    "AG": "#4c78a8",
    "AAA": "#f58518",
    "ILO": "#54a24b",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def cache_path(cache_dir: Path, canonical_id: str) -> Path:
    return cache_dir / (canonical_id.replace("/", "__") + ".npz")


def select_focus_cases(
    base_result: dict[str, Any], calibrated_result: dict[str, Any]
) -> list[tuple[str, str]]:
    base_by_id = {row["canonical_id"]: row for row in base_result["cases"]}
    rows = calibrated_result["cases"]

    def delta(row: dict[str, Any]) -> float:
        v3 = base_by_id[row["canonical_id"]]["methods"]["normal_multiscale_v3"]
        return float(row["methods"]["v4_calibrated"]["raw_r2"] - v3["raw_r2"])

    best = max(rows, key=delta)
    guard = min(rows, key=delta)
    median_delta = float(np.median([delta(row) for row in rows]))
    median = min(
        [row for row in rows if row is not best and row is not guard],
        key=lambda row: abs(delta(row) - median_delta),
    )
    return [
        ("best_gain", best["canonical_id"]),
        ("median_gain", median["canonical_id"]),
        ("guard_case", guard["canonical_id"]),
    ]


def load_case_predictions(
    canonical_id: str,
    cache_dir: Path,
    merged: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    mask = merged["point_case"] == canonical_id
    with np.load(cache_path(cache_dir, canonical_id), allow_pickle=False) as source:
        gradient_v3 = np.asarray(source["gradient_v3"], dtype=np.float64)
        sample_index = np.asarray(source["sample_index"], dtype=np.int64)
        surface_applied = np.asarray(source["surface__applied"], dtype=np.int8)
        surface_ratio = np.asarray(source["surface__raw_scalar_gradient"], dtype=np.float64) / np.maximum(
            np.linalg.norm(np.asarray(source["gradient_v1"], dtype=np.float64), axis=1),
            1e-12,
        )
        depth_limit = np.asarray(source["surface__depth_limit_mm"], dtype=np.float64)
        surface_radius = np.asarray(source["surface__surface_radius_mm"], dtype=np.float64)
        selected_fraction = np.asarray(source["surface__selected_fraction"], dtype=np.float64)
        local_radius = np.asarray(source["local_radius_mm"], dtype=np.float64)
    return {
        "wall_mm": merged["wall_mm"][mask],
        "truth": merged["truth_mag"][mask],
        "v3": np.linalg.norm(wss_from_gradient(gradient_v3, "carreau"), axis=1),
        "base": np.linalg.norm(merged["base_wss_vec"][mask], axis=1),
        "cal": np.linalg.norm(merged["calibrated_wss_vec"][mask], axis=1),
        "ratio": merged["calibration_ratio"][mask],
        "sample_index": sample_index,
        "surface_applied": surface_applied,
        "surface_ratio": surface_ratio,
        "depth_limit": depth_limit,
        "surface_radius": surface_radius,
        "selected_fraction": selected_fraction,
        "local_radius": local_radius,
    }


def save_accuracy_summary(
    base_result: dict[str, Any], calibrated_result: dict[str, Any], path: Path
) -> None:
    base_by_id = {row["canonical_id"]: row for row in base_result["cases"]}
    rows = calibrated_result["cases"]
    v3 = np.asarray(
        [base_by_id[row["canonical_id"]]["methods"]["normal_multiscale_v3"]["raw_r2"] for row in rows]
    )
    base = np.asarray([row["methods"]["v4_base"]["raw_r2"] for row in rows])
    cal = np.asarray([row["methods"]["v4_calibrated"]["raw_r2"] for row in rows])
    cohorts = np.asarray([row["cohort"] for row in rows])
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 10.4), dpi=170)

    ax = axes[0, 0]
    lower = float(min(v3.min(), cal.min()) - 0.01)
    upper = float(max(v3.max(), cal.max()) + 0.01)
    for cohort in sorted(set(cohorts)):
        mask = cohorts == cohort
        ax.scatter(
            v3[mask], cal[mask], s=50, color=COLORS[cohort], alpha=0.85,
            edgecolors="white", linewidths=0.5, label=cohort,
        )
    ax.plot([lower, upper], [lower, upper], "k--", lw=1)
    ax.set(xlim=(lower, upper), ylim=(lower, upper), xlabel="Frozen V3 raw R²", ylabel="Calibrated V4 raw R²")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("A. Test35 paired accuracy")
    ax.legend(frameon=False)
    ax.grid(alpha=0.22)

    ax = axes[0, 1]
    order = np.argsort(cal - v3)
    rank = np.arange(1, len(rows) + 1)
    ax.plot(rank, (base - v3)[order], "o-", ms=4, lw=1.1, color=COLORS["base"], label="Physics V4 − V3")
    ax.plot(rank, (cal - v3)[order], "o-", ms=4, lw=1.2, color=COLORS["cal"], label="Calibrated V4 − V3")
    ax.axhline(0, color="black", lw=0.9)
    ax.set(xlabel="Cases ordered by final gain", ylabel="Δ raw R²", title="B. Every test case improves")
    ax.legend(frameon=False)
    ax.grid(alpha=0.22)

    ax = axes[1, 0]
    aggregate = calibrated_result["aggregate"]
    names = ["Raw R²", "Scaled R²", "Spearman"]
    base_values = [
        aggregate["v4_base"]["raw_r2"]["mean"],
        aggregate["v4_base"]["scaled_r2"]["mean"],
        aggregate["v4_base"]["spearman"]["mean"],
    ]
    cal_values = [
        aggregate["v4_calibrated"]["raw_r2"]["mean"],
        aggregate["v4_calibrated"]["scaled_r2"]["mean"],
        aggregate["v4_calibrated"]["spearman"]["mean"],
    ]
    x = np.arange(len(names))
    width = 0.36
    ax.bar(x - width / 2, base_values, width, color=COLORS["base"], label="Physics V4")
    ax.bar(x + width / 2, cal_values, width, color=COLORS["cal"], label="Calibrated V4")
    for xpos, values in ((x - width / 2, base_values), (x + width / 2, cal_values)):
        for px, value in zip(xpos, values):
            ax.text(px, value + 0.002, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x, names)
    ax.set_ylim(0.88, 1.0)
    ax.set_title("C. Mean distribution accuracy")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.22)

    ax = axes[1, 1]
    names = ["MAE (Pa)", "NRMSE", "High-WSS NRMSE"]
    base_values = [
        aggregate["v4_base"]["mae_pa"]["mean"],
        aggregate["v4_base"]["nrmse"]["mean"],
        aggregate["v4_base"]["high_wss_nrmse"]["mean"],
    ]
    cal_values = [
        aggregate["v4_calibrated"]["mae_pa"]["mean"],
        aggregate["v4_calibrated"]["nrmse"]["mean"],
        aggregate["v4_calibrated"]["high_wss_nrmse"]["mean"],
    ]
    x = np.arange(len(names))
    ax.bar(x - width / 2, base_values, width, color=COLORS["base"])
    ax.bar(x + width / 2, cal_values, width, color=COLORS["cal"])
    for xpos, values in ((x - width / 2, base_values), (x + width / 2, cal_values)):
        for px, value in zip(xpos, values):
            ax.text(px, value + 0.012, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x, names)
    ax.set_title("D. Absolute and high-shear error")
    ax.grid(axis="y", alpha=0.22)

    fig.suptitle("Surface-MLS V4 + frozen diagnostic calibration: test35", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path)
    plt.close(fig)


def save_case_scatter(
    label: str,
    canonical_id: str,
    case: dict[str, np.ndarray],
    base_result: dict[str, Any],
    calibrated_result: dict[str, Any],
    path: Path,
) -> None:
    base_row = next(row for row in base_result["cases"] if row["canonical_id"] == canonical_id)
    cal_row = next(row for row in calibrated_result["cases"] if row["canonical_id"] == canonical_id)
    truth = case["truth"]
    limit = float(np.quantile(np.r_[truth, case["v3"], case["base"], case["cal"]], 0.995) * 1.06)
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.9), dpi=170, sharex=True, sharey=True)
    series = [
        ("Frozen normal-ray V3", "v3", COLORS["v3"], base_row["methods"]["normal_multiscale_v3"]),
        ("Closest-surface physics V4", "base", COLORS["base"], cal_row["methods"]["v4_base"]),
        ("Frozen calibrated V4", "cal", COLORS["cal"], cal_row["methods"]["v4_calibrated"]),
    ]
    for ax, (title, key, color, report) in zip(axes, series):
        ax.scatter(truth, case[key], s=9, alpha=0.32, color=color, edgecolors="none")
        ax.plot([0, limit], [0, limit], "k--", lw=1)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(0, limit)
        ax.set_ylim(0, limit)
        ax.set_title(f"{title}\nraw R²={report['raw_r2']:.3f}, scaled={report['scaled_r2']:.3f}")
        ax.grid(alpha=0.2)
        ax.set_xlabel("CFD WSS truth (Pa)")
    axes[0].set_ylabel("Predicted WSS (Pa)")
    fig.suptitle(f"{label.replace('_', ' ').title()}: {canonical_id}", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path)
    plt.close(fig)


def save_spatial_map(canonical_id: str, case: dict[str, np.ndarray], path: Path) -> None:
    coords = case["wall_mm"]
    centered = coords - coords.mean(axis=0, keepdims=True)
    _, _, axes = np.linalg.svd(centered, full_matrices=False)
    uv = centered @ axes[:2].T
    truth = case["truth"]
    vmax = float(np.quantile(np.r_[truth, case["base"], case["cal"]], 0.98))
    error_gain = np.abs(truth - case["base"]) - np.abs(truth - case["cal"])
    gain_limit = float(np.quantile(np.abs(error_gain), 0.98))
    fig, panels = plt.subplots(1, 4, figsize=(17.0, 4.5), dpi=170)
    for ax, title, values in zip(
        panels[:3],
        ("CFD truth", "Physics V4", "Calibrated V4"),
        (truth, case["base"], case["cal"]),
    ):
        image = ax.scatter(uv[:, 0], uv[:, 1], c=values, s=11, cmap="turbo", vmin=0, vmax=vmax)
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="box")
        ax.set_axis_off()
    fig.colorbar(image, ax=panels[:3], shrink=0.76, label="WSS (Pa)")
    image = panels[3].scatter(
        uv[:, 0], uv[:, 1], c=error_gain, s=11, cmap="RdBu_r",
        vmin=-gain_limit, vmax=gain_limit,
    )
    panels[3].set_title("Absolute-error reduction")
    panels[3].set_aspect("equal", adjustable="box")
    panels[3].set_axis_off()
    fig.colorbar(image, ax=panels[3], shrink=0.76, label="|error base| − |error final| (Pa)")
    fig.suptitle(f"Spatial WSS reconstruction: {canonical_id}", fontsize=14)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.85, bottom=0.03, wspace=0.08)
    fig.savefig(path)
    plt.close(fig)


def tangent_axis(normal: np.ndarray, delta: np.ndarray) -> np.ndarray:
    lateral = delta - np.outer(delta @ normal, normal)
    if len(lateral) and np.linalg.norm(lateral) > 1e-10:
        _, _, axes = np.linalg.svd(lateral, full_matrices=False)
        tangent = axes[0] - float(axes[0] @ normal) * normal
        if np.linalg.norm(tangent) > 1e-10:
            return tangent / np.linalg.norm(tangent)
    seed = np.zeros(3)
    seed[int(np.argmin(np.abs(normal)))] = 1.0
    tangent = np.cross(normal, seed)
    return tangent / max(float(np.linalg.norm(tangent)), 1e-12)


def save_sampling_audit(
    canonical_id: str, case: dict[str, np.ndarray], cache_dir: Path, path: Path
) -> None:
    case_dir = ROOT / "data_new" / canonical_id
    with np.load(cache_path(cache_dir, canonical_id), allow_pickle=False) as source:
        step = int(source["step"])
        normals_sample = np.asarray(source["normal"], dtype=np.float64)
    wall = dl.load_wall(dl.step_path(case_dir, step, "ascii"))
    interior = dl.load_interior(dl.step_path(case_dir, step, "ascii_in"))
    wall_mm = np.asarray(wall["coords_mm"], dtype=np.float64)
    interior_mm = np.asarray(interior["coords_mm"], dtype=np.float64)
    interior_tree = cKDTree(interior_mm)
    full_normals = pca_wall_normals(wall_mm, interior_mm, interior_tree)
    valid = (
        (case["surface_applied"] > 0)
        & np.isfinite(case["surface_ratio"])
        & np.isfinite(case["selected_fraction"])
    )
    score = np.where(valid, case["ratio"] * case["surface_ratio"], -np.inf)
    row = int(np.argmax(score))
    target = case["wall_mm"][row]
    normal = normals_sample[row]
    _, index = interior_tree.query(target, k=min(384, len(interior_mm)))
    candidate = interior_mm[index]
    wall_tree = cKDTree(wall_mm)
    _, anchor = wall_tree.query(candidate, k=1)
    anchor_point = wall_mm[anchor]
    anchor_normal = full_normals[anchor]
    anchor_delta = anchor_point - target
    tangent = tangent_axis(normal, anchor_delta)
    signed_surface = anchor_delta @ tangent
    target_eta = (candidate - target) @ normal
    anchor_eta = anchor_delta @ normal
    local_depth = np.einsum("ij,ij->i", candidate - anchor_point, anchor_normal)
    owner_cosine = anchor_normal @ normal
    surface_distance = np.sqrt(
        np.maximum(np.sum(anchor_delta**2, axis=1) - anchor_eta**2, 0.0)
    )
    bandwidth = float(case["surface_radius"][row])
    fit_depth = float(case["depth_limit"][row] * case["selected_fraction"][row] * 1.15)
    retained = (
        (local_depth > 1e-4)
        & (local_depth <= fit_depth)
        & (surface_distance <= bandwidth)
        & (owner_cosine >= 0.5)
    )
    owner_rejected = (local_depth > 1e-4) & (owner_cosine < 0.5)
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.6), dpi=170)
    ax = axes[0]
    ax.scatter(signed_surface, target_eta, s=18, color="#b6bbc3", alpha=0.5, label="queried cells")
    ax.scatter(signed_surface[owner_rejected], target_eta[owner_rejected], s=24, color="#d95f59", alpha=0.75, label="ownership rejected")
    ax.scatter(signed_surface[retained], target_eta[retained], s=28, color=COLORS["base"], alpha=0.9, label="retained")
    ax.scatter(signed_surface, anchor_eta, s=8, color="#222222", alpha=0.25)
    ax.scatter([0], [0], marker="*", s=150, color="#f2b134", edgecolors="black", linewidths=0.5, label="target wall")
    ax.axvline(-bandwidth, color="black", ls="--", lw=0.9)
    ax.axvline(bandwidth, color="black", ls="--", lw=0.9)
    ax.set(xlabel="Signed tangent displacement (mm)", ylabel="Depth from target tangent plane (mm)", title="A. Curved wall in target-plane coordinates")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.2)

    ax = axes[1]
    ax.scatter(signed_surface, local_depth, s=18, color="#b6bbc3", alpha=0.5)
    ax.scatter(signed_surface[owner_rejected], local_depth[owner_rejected], s=24, color="#d95f59", alpha=0.75)
    ax.scatter(signed_surface[retained], local_depth[retained], s=28, color=COLORS["base"], alpha=0.9)
    ax.axhline(0, color="black", lw=1)
    ax.axhline(fit_depth, color="black", ls="--", lw=0.9, label=f"fit depth {fit_depth:.2f} mm")
    ax.axvline(-bandwidth, color="black", ls="--", lw=0.9)
    ax.axvline(bandwidth, color="black", ls="--", lw=0.9, label=f"surface radius {bandwidth:.2f} mm")
    ax.set(xlabel="Wall-anchor tangent displacement (mm)", ylabel="Depth from each cell's own wall anchor (mm)", title="B. Closest-surface coordinates used by V4")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.2)
    fig.suptitle(
        f"Closest-surface sampling audit: {canonical_id}\n"
        f"surface ratio={case['surface_ratio'][row]:.2f}, final calibration={case['ratio'][row]:.2f}",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(path)
    plt.close(fig)


def load_diagnostic_arrays(cache_dir: Path) -> dict[str, np.ndarray]:
    blocks = []
    for path in sorted(cache_dir.glob("*.npz")):
        with np.load(path, allow_pickle=False) as source:
            gradient_v1 = np.asarray(source["gradient_v1"], dtype=np.float64)
            raw_surface = np.asarray(source["surface__raw_scalar_gradient"], dtype=np.float64)
            blocks.append(
                {
                    "surface_ratio": raw_surface / np.maximum(np.linalg.norm(gradient_v1, axis=1), 1e-12),
                    "radius": np.asarray(source["local_radius_mm"], dtype=np.float64),
                    "re": np.asarray(source["surface__local_reynolds"], dtype=np.float64),
                    "combined": np.asarray(source["combined_correction"], dtype=np.float64),
                }
            )
    return {key: np.concatenate([block[key] for block in blocks]) for key in blocks[0]}


def save_calibration_diagnostics(
    merged: dict[str, np.ndarray], diagnostic: dict[str, np.ndarray], path: Path
) -> None:
    truth = merged["truth_mag"]
    base = np.linalg.norm(merged["base_wss_vec"], axis=1)
    cal = np.linalg.norm(merged["calibrated_wss_vec"], axis=1)
    ratio = merged["calibration_ratio"]
    surface_ratio = diagnostic["surface_ratio"]
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 10.2), dpi=170)

    ax = axes[0, 0]
    image = ax.hexbin(base, ratio, gridsize=48, mincnt=1, cmap="viridis", bins="log")
    ax.set(xlabel="Physics V4 WSS (Pa)", ylabel="Frozen calibration ratio", title="A. Magnitude-dependent residual correction")
    fig.colorbar(image, ax=ax, label="log point count")
    ax.grid(alpha=0.15)

    ax = axes[0, 1]
    valid = np.isfinite(surface_ratio)
    image = ax.hexbin(surface_ratio[valid], ratio[valid], gridsize=48, mincnt=1, cmap="magma", bins="log", extent=(0.5, 2.2, 0.8, 1.6))
    ax.set(xlim=(0.5, 2.2), ylim=(0.8, 1.6), xlabel="Direct surface gradient / V1 gradient", ylabel="Frozen calibration ratio", title="B. Surface-fit evidence drives local correction")
    fig.colorbar(image, ax=ax, label="log point count")
    ax.grid(alpha=0.15)

    ax = axes[1, 0]
    quantile = np.quantile(truth, np.linspace(0, 1, 11))
    center = []
    before = []
    after = []
    for low, high in zip(quantile[:-1], quantile[1:]):
        mask = (truth >= low) & (truth <= high)
        center.append(float(np.mean(truth[mask])))
        before.append(float(np.sqrt(np.mean((truth[mask] - base[mask]) ** 2))))
        after.append(float(np.sqrt(np.mean((truth[mask] - cal[mask]) ** 2))))
    ax.plot(center, before, "o-", color=COLORS["base"], label="Physics V4")
    ax.plot(center, after, "o-", color=COLORS["cal"], label="Calibrated V4")
    ax.set(xlabel="Truth WSS decile mean (Pa)", ylabel="RMSE (Pa)", title="C. Error reduction across the WSS range")
    ax.legend(frameon=False)
    ax.grid(alpha=0.22)

    ax = axes[1, 1]
    cohorts = merged["point_cohort"]
    data = [ratio[cohorts == cohort] for cohort in ("AG", "AAA", "ILO")]
    violin = ax.violinplot(data, showmeans=True, showextrema=False)
    for body, cohort in zip(violin["bodies"], ("AG", "AAA", "ILO")):
        body.set_facecolor(COLORS[cohort])
        body.set_alpha(0.65)
    ax.set_xticks([1, 2, 3], ["AG", "AAA", "ILO"])
    ax.set(ylabel="Calibration ratio", title="D. Cohort-stable correction distribution")
    ax.grid(axis="y", alpha=0.22)
    fig.suptitle("Frozen diagnostic residual calibration", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path)
    plt.close(fig)


def save_error_diagnostics(
    base_result: dict[str, Any], calibrated_result: dict[str, Any], path: Path
) -> None:
    base_by_id = {row["canonical_id"]: row for row in base_result["cases"]}
    rows = calibrated_result["cases"]
    v3 = np.asarray([base_by_id[row["canonical_id"]]["methods"]["normal_multiscale_v3"]["raw_r2"] for row in rows])
    base = np.asarray([row["methods"]["v4_base"]["raw_r2"] for row in rows])
    cal = np.asarray([row["methods"]["v4_calibrated"]["raw_r2"] for row in rows])
    cohorts = np.asarray([row["cohort"] for row in rows])
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 4.9), dpi=170)
    ax = axes[0]
    for i, cohort in enumerate(("AG", "AAA", "ILO"), start=1):
        values = cal[cohorts == cohort] - base[cohorts == cohort]
        jitter = np.linspace(-0.08, 0.08, len(values))
        ax.scatter(i + jitter, values, color=COLORS[cohort], s=36, alpha=0.8)
        ax.hlines(np.median(values), i - 0.2, i + 0.2, color="black", lw=2)
    ax.axhline(0, color="black", lw=0.9)
    ax.set_xticks([1, 2, 3], ["AG", "AAA", "ILO"])
    ax.set(ylabel="Δ raw R²: calibrated − physics V4", title="A. Calibration gain by cohort")
    ax.grid(axis="y", alpha=0.22)

    ax = axes[1]
    order = np.argsort(v3)
    ax.plot(v3[order], "o-", color=COLORS["v3"], ms=4, label="V3")
    ax.plot(base[order], "o-", color=COLORS["base"], ms=4, label="Physics V4")
    ax.plot(cal[order], "o-", color=COLORS["cal"], ms=4, label="Final V4")
    ax.set(xlabel="Cases ordered by V3", ylabel="Raw R²", title="B. Tail lifting without trading top cases")
    ax.legend(frameon=False)
    ax.grid(alpha=0.22)

    ax = axes[2]
    metrics = ("raw_r2", "scaled_r2")
    labels = ["raw p05", "raw min", "scaled p05", "scaled min"]
    base_values = [
        calibrated_result["aggregate"]["v4_base"][metric][stat]
        for metric in metrics for stat in ("p05", "min")
    ]
    cal_values = [
        calibrated_result["aggregate"]["v4_calibrated"][metric][stat]
        for metric in metrics for stat in ("p05", "min")
    ]
    x = np.arange(len(labels))
    width = 0.36
    ax.bar(x - width / 2, base_values, width, color=COLORS["base"], label="Physics V4")
    ax.bar(x + width / 2, cal_values, width, color=COLORS["cal"], label="Final V4")
    ax.set_xticks(x, labels, rotation=18)
    ax.set_ylim(0.8, 1.0)
    ax.set_title("C. Lower-tail robustness")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.22)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_contact_sheet(images: list[Path], path: Path) -> None:
    thumbs = []
    for source in images:
        image = Image.open(source).convert("RGB")
        image.thumbnail((760, 520))
        canvas = Image.new("RGB", (780, 570), "white")
        canvas.paste(image, ((780 - image.width) // 2, 30))
        draw = ImageDraw.Draw(canvas)
        draw.text((20, 10), source.name, fill="black")
        thumbs.append(canvas)
    rows = (len(thumbs) + 1) // 2
    sheet = Image.new("RGB", (1560, 570 * rows), "#e9edf1")
    for index, image in enumerate(thumbs):
        sheet.paste(ImageOps.expand(image, border=2, fill="#c4cad1"), ((index % 2) * 780, (index // 2) * 570))
    sheet.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-result", default=str(DEFAULT_BASE_RESULT))
    parser.add_argument("--calibrated-result", default=str(DEFAULT_CAL_RESULT))
    parser.add_argument("--predictions", default=str(DEFAULT_PREDICTIONS))
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    base_result = load_json(Path(args.base_result))
    calibrated_result = load_json(Path(args.calibrated_result))
    cache_dir = Path(args.cache_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with np.load(args.predictions, allow_pickle=False) as source:
        merged = {key: np.asarray(source[key]) for key in source.files}

    generated = []
    summary_path = out / "01_test35_accuracy_summary.png"
    save_accuracy_summary(base_result, calibrated_result, summary_path)
    generated.append(summary_path)

    focus = select_focus_cases(base_result, calibrated_result)
    focus_cases = {}
    for number, (label, canonical_id) in enumerate(focus, start=2):
        case = load_case_predictions(canonical_id, cache_dir, merged)
        focus_cases[label] = {"canonical_id": canonical_id, "case": case}
        target = out / f"{number:02d}_{label}_scatter.png"
        save_case_scatter(label, canonical_id, case, base_result, calibrated_result, target)
        generated.append(target)

    best = focus_cases["best_gain"]
    spatial_path = out / "05_best_gain_spatial_map.png"
    save_spatial_map(best["canonical_id"], best["case"], spatial_path)
    generated.append(spatial_path)

    sampling_path = out / "06_closest_surface_sampling.png"
    save_sampling_audit(best["canonical_id"], best["case"], cache_dir, sampling_path)
    generated.append(sampling_path)

    diagnostic = load_diagnostic_arrays(cache_dir)
    calibration_path = out / "07_calibration_diagnostics.png"
    save_calibration_diagnostics(merged, diagnostic, calibration_path)
    generated.append(calibration_path)

    error_path = out / "08_error_and_tail_diagnostics.png"
    save_error_diagnostics(base_result, calibrated_result, error_path)
    generated.append(error_path)

    contact = out / "00_overview_contact_sheet.png"
    save_contact_sheet(generated, contact)
    audit = {
        "base_result": str(Path(args.base_result).resolve()),
        "calibrated_result": str(Path(args.calibrated_result).resolve()),
        "predictions": str(Path(args.predictions).resolve()),
        "focus_cases": {label: item["canonical_id"] for label, item in focus_cases.items()},
        "generated": [path.name for path in [contact, *generated]],
        "test35_metrics": calibrated_result["aggregate"],
        "paired": calibrated_result["paired"],
    }
    (out / "audit_summary.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out / "README.md").write_text(
        "# Surface-MLS V4 quicklook\n\n"
        "This atlas summarizes frozen test35 accuracy, closest-surface sampling, "
        "diagnostic calibration, spatial reconstruction, and lower-tail behavior.\n\n"
        + "\n".join(f"- `{path.name}`" for path in [contact, *generated])
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
