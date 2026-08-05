"""Generate a frozen-V3 accuracy, sampling, and penetration quicklook atlas.

The script reproduces the blind-test sample points with the frozen V1/V2/V3
methods, then writes a compact set of teacher-facing PNG audits.  It never
changes the frozen method or uses the blind result for parameter selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree


VIZ = Path(__file__).resolve().parent
PACKAGE = VIZ.parent
SRC = PACKAGE / "src"
ROOT = PACKAGE.parent
sys.path.insert(0, str(SRC))

import data_loader_cfd as dl  # noqa: E402
from calculate_wss_cfd import (  # noqa: E402
    evaluate,
    fit_wall_gradient,
    pca_wall_normals,
    wss_from_gradient,
)
from run_v3_experiment import _resolve_radius  # noqa: E402
from wss_multiscale import fit_wall_gradient_multiscale  # noqa: E402
from wss_normal_multiscale_v3 import (  # noqa: E402
    NormalMultiscaleV3Config,
    fit_wall_gradient_normal_multiscale,
)


DEFAULT_RESULT = (
    ROOT
    / "outputs/wss_mri_calculator/pointcloud_normal_multiscale_v3"
    / "stage_c_blind_test35_s1200.json"
)
DEFAULT_CONFIG = (
    PACKAGE
    / "experiments/pointcloud_normal_multiscale_v3/config_stage_c_blind_test35.json"
)
DEFAULT_OUT = (
    ROOT
    / "outputs/wss_mri_calculator/pointcloud_normal_multiscale_v3/00_quicklook"
)
V1_CONFIG = PACKAGE / "experiments/pointcloud_adaptive_v1/config_frozen_v1.json"
V2_CONFIG = PACKAGE / "experiments/pointcloud_multiscale_v2/config_frozen_v2.json"

METHODS = ("adaptive_cv_v1", "multiscale_v2", "normal_multiscale_v3")
METHOD_LABELS = {
    "adaptive_cv_v1": "Adaptive-CV V1",
    "multiscale_v2": "Multiscale V2",
    "normal_multiscale_v3": "Normal-ray V3",
}
METHOD_COLORS = {
    "adaptive_cv_v1": "#5b6470",
    "multiscale_v2": "#e28e2c",
    "normal_multiscale_v3": "#167d8d",
}
COHORT_COLORS = {"AG": "#4c78a8", "AAA": "#f58518", "ILO": "#54a24b"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def select_focus_cases(result: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    usable = [row for row in result["cases"] if row.get("truth_usable", True)]

    def delta(row: dict[str, Any], baseline: str) -> float:
        methods = row["methods"]
        return float(
            methods["normal_multiscale_v3"]["raw_r2"]
            - methods[baseline]["raw_r2"]
        )

    best = max(usable, key=lambda row: delta(row, "adaptive_cv_v1"))
    guard = min(usable, key=lambda row: delta(row, "multiscale_v2"))
    v1_delta = np.asarray([delta(row, "adaptive_cv_v1") for row in usable])
    median_delta = float(np.median(v1_delta))
    candidates = [row for row in usable if row is not best and row is not guard]
    median = min(
        candidates,
        key=lambda row: abs(delta(row, "adaptive_cv_v1") - median_delta),
    )
    return [("best_gain", best), ("median_gain", median), ("guard_case", guard)]


def tangent_axis(normal: np.ndarray, delta: np.ndarray | None = None) -> np.ndarray:
    if delta is not None and len(delta):
        lateral = delta - np.outer(delta @ normal, normal)
        if np.linalg.norm(lateral) > 1e-10:
            _, _, axes = np.linalg.svd(lateral, full_matrices=False)
            tangent = axes[0] - float(axes[0] @ normal) * normal
            norm = float(np.linalg.norm(tangent))
            if norm > 1e-10:
                return tangent / norm
    seed = np.zeros(3, dtype=np.float64)
    seed[int(np.argmin(np.abs(normal)))] = 1.0
    tangent = np.cross(normal, seed)
    return tangent / max(float(np.linalg.norm(tangent)), 1e-12)


def opposite_wall(
    target: np.ndarray,
    normal: np.ndarray,
    wall_mm: np.ndarray,
    wall_normals: np.ndarray,
    wall_tree: cKDTree,
    spacing_mm: float,
) -> tuple[float, int | None]:
    distance, index = wall_tree.query(target, k=min(512, len(wall_mm)))
    dot = wall_normals[index] @ normal
    mask = (dot < -0.6) & (distance > max(3.0 * spacing_mm, 1.0))
    if not mask.any():
        return float("nan"), None
    local = int(np.flatnonzero(mask)[np.argmin(distance[mask])])
    return float(distance[local]), int(index[local])


def compute_case(
    row: dict[str, Any],
    experiment_config: dict[str, Any],
) -> dict[str, Any]:
    case_dir = ROOT / "data_new" / row["canonical_id"]
    if not case_dir.is_dir():
        raise FileNotFoundError(case_dir)
    step = int(row["step"])
    wall = dl.load_wall(dl.step_path(case_dir, step, "ascii"))
    interior = dl.load_interior(dl.step_path(case_dir, step, "ascii_in"))
    wall_mm = np.asarray(wall["coords_mm"], dtype=np.float64)
    interior_mm = np.asarray(interior["coords_mm"], dtype=np.float64)
    interior_tree = cKDTree(interior_mm)
    full_normals = pca_wall_normals(wall_mm, interior_mm, interior_tree)

    execution = experiment_config["execution"]
    sample_count = int(execution["sample_count"])
    seed = int(execution["seed"])
    if 0 < sample_count < len(wall_mm):
        rng = np.random.default_rng(seed)
        sample_index = np.sort(rng.choice(len(wall_mm), sample_count, replace=False))
    else:
        sample_index = np.arange(len(wall_mm))
    sample_wall = wall_mm[sample_index]
    sample_normals = full_normals[sample_index]
    truth_mag = np.asarray(wall["wss_mag"], dtype=np.float64)[sample_index]
    truth_vec = np.asarray(wall["wss_vec"], dtype=np.float64)[sample_index]

    wall_tree = cKDTree(wall_mm)
    geometry = experiment_config.get("geometry", {})
    local_radius, radius_source = _resolve_radius(
        str(geometry.get("radius_source", "bundle_then_wall")),
        case_dir,
        sample_wall,
        sample_normals,
        wall_mm,
        full_normals,
        wall_tree,
        geometry,
    )

    v1 = load_json(V1_CONFIG)
    gradient_v1, _, diag_v1 = fit_wall_gradient(
        sample_wall,
        sample_normals,
        interior_mm,
        interior["velocity"],
        interior_tree,
        degree=int(v1["degree"]),
        neighbor_mode="adaptive_cv",
        adaptive_neighbors=tuple(int(value) for value in v1["adaptive_neighbors"]),
        cv_tolerance=float(v1["cv_tolerance"]),
    )
    pred_vec_v1 = wss_from_gradient(gradient_v1, "carreau")

    v2 = load_json(V2_CONFIG)
    gradient_v2, _, diag_v2 = fit_wall_gradient_multiscale(
        sample_wall,
        sample_normals,
        interior_mm,
        interior["velocity"],
        interior_tree,
        adaptive_neighbors=tuple(int(value) for value in v2["adaptive_neighbors"]),
        degree=int(v2["degree"]),
        base_cv_tolerance=float(v2["base_cv_tolerance"]),
        extrapolation_cv_tolerance=float(v2["extrapolation_cv_tolerance"]),
        extrapolation_power=float(v2["extrapolation_power"]),
        smooth_neighbors=int(v2["smooth_neighbors"]),
        global_blend=float(v2["global_blend"]),
        correction_min=float(v2["correction_min"]),
        correction_max=float(v2["correction_max"]),
        correction_shrink=float(v2["correction_shrink"]),
        max_trend_correlation=float(v2["max_trend_correlation"]),
        min_direction_cosine=float(v2["min_direction_cosine"]),
    )
    pred_vec_v2 = wss_from_gradient(gradient_v2, "carreau")

    v3_config = NormalMultiscaleV3Config.from_mapping(experiment_config["method"])
    gradient_v3, _, diag_v3 = fit_wall_gradient_normal_multiscale(
        sample_wall,
        sample_normals,
        wall_mm,
        full_normals,
        interior_mm,
        interior["velocity"],
        interior_tree,
        local_radius_mm=local_radius,
        config=v3_config,
        fallback_gradient=gradient_v1,
    )
    pred_vec_v3 = wss_from_gradient(gradient_v3, "carreau")

    prediction_vectors = {
        "adaptive_cv_v1": pred_vec_v1,
        "multiscale_v2": pred_vec_v2,
        "normal_multiscale_v3": pred_vec_v3,
    }
    reports = {
        method: evaluate(truth_mag, truth_vec, vector)
        for method, vector in prediction_vectors.items()
    }
    for method in METHODS:
        expected = float(row["methods"][method]["raw_r2"])
        observed = float(reports[method]["raw_r2"])
        if not np.isclose(expected, observed, rtol=0.0, atol=2e-10):
            raise RuntimeError(
                f"{row['canonical_id']} {method} reproduction mismatch: "
                f"expected {expected}, observed {observed}"
            )

    return {
        "row": row,
        "case_dir": case_dir,
        "step": step,
        "wall": wall,
        "interior": interior,
        "wall_mm": wall_mm,
        "interior_mm": interior_mm,
        "interior_tree": interior_tree,
        "wall_tree": wall_tree,
        "full_normals": full_normals,
        "sample_index": sample_index,
        "sample_wall": sample_wall,
        "sample_normals": sample_normals,
        "truth_mag": truth_mag,
        "truth_vec": truth_vec,
        "pred_vectors": prediction_vectors,
        "pred_mag": {
            method: np.linalg.norm(vector, axis=1)
            for method, vector in prediction_vectors.items()
        },
        "reports": reports,
        "gradient_v1": gradient_v1,
        "diag_v1": diag_v1,
        "diag_v2": diag_v2,
        "diag_v3": diag_v3,
        "local_radius": local_radius,
        "radius_source": radius_source,
        "v3_config": v3_config,
    }


def save_blind_summary(result: dict[str, Any], path: Path) -> None:
    rows = [row for row in result["cases"] if row.get("truth_usable", True)]
    v1 = np.asarray([row["methods"]["adaptive_cv_v1"]["raw_r2"] for row in rows])
    v2 = np.asarray([row["methods"]["multiscale_v2"]["raw_r2"] for row in rows])
    v3 = np.asarray([row["methods"]["normal_multiscale_v3"]["raw_r2"] for row in rows])
    cohorts = np.asarray([row["cohort"] for row in rows])

    fig, axes = plt.subplots(2, 2, figsize=(13.8, 10.4), dpi=170)
    ax = axes[0, 0]
    lower = float(min(v1.min(), v3.min()) - 0.01)
    upper = float(max(v1.max(), v3.max()) + 0.01)
    for cohort in sorted(set(cohorts)):
        mask = cohorts == cohort
        ax.scatter(
            v1[mask], v3[mask], s=48, color=COHORT_COLORS.get(cohort, "#777777"),
            alpha=0.85, edgecolors="white", linewidths=0.5, label=cohort,
        )
    ax.plot([lower, upper], [lower, upper], "k--", lw=1.1)
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("V1 raw R²")
    ax.set_ylabel("V3 raw R²")
    ax.set_title("A. Paired blind-test accuracy by case")
    ax.legend(frameon=False)
    ax.grid(alpha=0.22)

    ax = axes[0, 1]
    order = np.argsort(v3 - v1)
    rank = np.arange(1, len(rows) + 1)
    ax.plot(rank, (v3 - v1)[order], "o-", ms=4, lw=1.2, color=METHOD_COLORS["normal_multiscale_v3"], label="V3 − V1")
    ax.plot(rank, (v3 - v2)[order], "s--", ms=3.5, lw=1.0, color=METHOD_COLORS["multiscale_v2"], label="V3 − V2")
    ax.axhline(0, color="k", lw=1)
    ax.set_xlabel("Cases ordered by V3 − V1")
    ax.set_ylabel("Δ raw R²")
    ax.set_title("B. Case-wise gain and guard-tail behavior")
    ax.legend(frameon=False)
    ax.grid(alpha=0.22)

    ax = axes[1, 0]
    values = [v1, v2, v3]
    box = ax.boxplot(values, patch_artist=True, widths=0.55, showfliers=False)
    for patch, method in zip(box["boxes"], METHODS):
        patch.set_facecolor(METHOD_COLORS[method])
        patch.set_alpha(0.68)
    rng = np.random.default_rng(7)
    for index, (method, data) in enumerate(zip(METHODS, values), start=1):
        jitter = rng.normal(index, 0.045, len(data))
        ax.scatter(jitter, data, s=12, color=METHOD_COLORS[method], alpha=0.45, edgecolors="none")
    ax.set_xticks([1, 2, 3], ["V1", "V2", "V3"])
    ax.set_ylabel("raw R²")
    ax.set_title("C. Blind-test raw R² distribution")
    ax.grid(axis="y", alpha=0.22)

    ax = axes[1, 1]
    stats = ("mean", "p05", "min")
    x = np.arange(len(stats))
    width = 0.24
    for offset, method in enumerate(METHODS):
        raw = result["aggregate"][method]["raw_r2"]
        ax.bar(
            x + (offset - 1) * width,
            [raw[stat] for stat in stats],
            width,
            color=METHOD_COLORS[method],
            alpha=0.86,
            label=METHOD_LABELS[method],
        )
    ax.set_xticks(x, ["mean", "p05", "min"])
    ax.set_ylim(0.72, 0.92)
    ax.set_ylabel("raw R²")
    ax.set_title("D. Mean and lower-tail accuracy")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.22)

    paired_v1 = result["aggregate"]["v3_paired"]["adaptive_cv_v1"]
    paired_v2 = result["aggregate"]["v3_paired"]["multiscale_v2"]
    fig.suptitle(
        "Frozen normal-ray V3 — blind test35 × 1200\n"
        f"V3 mean raw R²={v3.mean():.4f}; V3−V1={np.mean(v3-v1):+.4f} "
        f"({paired_v1['wins']}/{paired_v1['n']} wins); "
        f"V3−V2={np.mean(v3-v2):+.4f} ({paired_v2['wins']}/{paired_v2['n']} wins)",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path)
    plt.close(fig)


def save_case_scatter(data: dict[str, Any], role: str, path: Path) -> None:
    truth = data["truth_mag"]
    max_value = max(
        float(np.nanmax(truth)),
        *(float(np.nanmax(data["pred_mag"][method])) for method in METHODS),
    )
    limit = 1.07 * max(max_value, 1e-6)
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.2), dpi=175, sharex=True, sharey=True)
    for ax, method in zip(axes, METHODS):
        pred = data["pred_mag"][method]
        valid = np.isfinite(truth) & np.isfinite(pred)
        hb = ax.hexbin(
            truth[valid], pred[valid], gridsize=54, mincnt=1, bins="log",
            cmap="viridis", extent=(0, limit, 0, limit),
        )
        ax.plot([0, limit], [0, limit], "w--", lw=1.2)
        report = data["reports"][method]
        ax.set_title(
            f"{METHOD_LABELS[method]}\n"
            f"raw R²={report['raw_r2']:.4f}  MAE={report['mae_pa']:.3f} Pa  α={report['alpha']:.3f}"
        )
        ax.set_xlabel("CFD truth WSS (Pa)")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.18)
        fig.colorbar(hb, ax=ax, fraction=0.045, pad=0.025, label="log count")
    axes[0].set_ylabel("Predicted WSS (Pa)")
    row = data["row"]
    d1 = row["methods"]["normal_multiscale_v3"]["raw_r2"] - row["methods"]["adaptive_cv_v1"]["raw_r2"]
    d2 = row["methods"]["normal_multiscale_v3"]["raw_r2"] - row["methods"]["multiscale_v2"]["raw_r2"]
    fig.suptitle(
        f"{role.replace('_', ' ').title()} — {row['canonical_id']}  peak {data['step']}  n={len(truth)}\n"
        f"V3−V1={d1:+.5f}; V3−V2={d2:+.5f}; frozen blind-test sample reproduced exactly",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(path)
    plt.close(fig)


def ray_support_for_target(data: dict[str, Any], row_index: int) -> dict[str, Any]:
    config: NormalMultiscaleV3Config = data["v3_config"]
    point = data["sample_wall"][row_index]
    normal = data["sample_normals"][row_index]
    depth = float(
        data["diag_v3"]["selected_scale_fraction"][row_index]
        * data["diag_v3"]["depth_limit_mm"][row_index]
    )
    start_fraction = float(config.ray_start_fractions[0])
    stations = np.linspace(start_fraction * depth, depth, config.ray_stations)
    query = point + stations[:, None] * normal
    query_distance, query_index = data["interior_tree"].query(
        query, k=min(config.ray_interpolation_neighbors, len(data["interior_mm"]))
    )
    if query_index.ndim == 1:
        query_index = query_index[:, None]
        query_distance = query_distance[:, None]
    support = data["interior_mm"][query_index]
    delta = support - point
    eta = np.einsum("nki,i->nk", delta, normal)
    lateral = np.sqrt(np.maximum(np.sum(delta**2, axis=2) - eta**2, 0.0))
    valid_depth = (eta > 1e-4) & (eta <= 1.25 * depth)

    spacing_distance, _ = data["wall_tree"].query(point, k=min(8, len(data["wall_mm"])))
    spacing = float(np.median(spacing_distance[1:])) if np.ndim(spacing_distance) else float(spacing_distance)
    nearest_wall_distance, anchor = data["wall_tree"].query(support.reshape(-1, 3), k=1)
    nearest_wall_distance = nearest_wall_distance.reshape(query_index.shape)
    anchor = anchor.reshape(query_index.shape)
    anchor_separation = np.linalg.norm(data["wall_mm"][anchor] - point, axis=2)
    normal_cosine = np.einsum("nki,i->nk", data["full_normals"][anchor], normal)
    max_separation = max(
        config.ownership_max_anchor_separation_mm,
        config.ownership_max_anchor_spacing_multiplier * spacing,
    )
    owner_ok = (
        (anchor_separation <= max_separation)
        & (normal_cosine >= config.ownership_min_normal_cosine)
    )
    accepted = valid_depth & owner_ok
    rejected = valid_depth & ~owner_ok

    target_distance = np.linalg.norm(delta, axis=2)
    post_filter_risk = (
        accepted
        & (normal_cosine < -0.25)
        & (anchor_separation > max(3.0 * spacing, 1.5))
        & (nearest_wall_distance < 0.8 * target_distance)
    )
    station_retained = np.sum(accepted, axis=1) >= config.ray_min_interpolation_neighbors

    selected_k = int(data["diag_v1"]["selected_neighbors"][row_index])
    max_k = max(load_json(V1_CONFIG)["adaptive_neighbors"])
    _, v1_index = data["interior_tree"].query(point, k=max_k)
    v1_delta = data["interior_mm"][v1_index] - point
    v1_eta = v1_delta @ normal
    v1_used = (np.arange(max_k) < selected_k) & (v1_eta > 1e-4)

    return {
        "point": point,
        "normal": normal,
        "depth_mm": depth,
        "stations_mm": stations,
        "station_retained": station_retained,
        "query_index": query_index,
        "support_delta": delta,
        "support_eta": eta,
        "support_lateral": lateral,
        "accepted": accepted,
        "rejected": rejected,
        "post_filter_risk": post_filter_risk,
        "anchor": anchor,
        "normal_cosine": normal_cosine,
        "anchor_separation": anchor_separation,
        "nearest_wall_distance": nearest_wall_distance,
        "spacing_mm": spacing,
        "v1_index": v1_index,
        "v1_delta": v1_delta,
        "v1_eta": v1_eta,
        "v1_used": v1_used,
        "selected_k": selected_k,
    }


def local_projection(
    data: dict[str, Any],
    support: dict[str, Any],
    view_radius: float,
) -> dict[str, np.ndarray]:
    point = support["point"]
    normal = support["normal"]
    flat_delta = support["support_delta"].reshape(-1, 3)
    opposite_distance, opposite_index = opposite_wall(
        point,
        normal,
        data["wall_mm"],
        data["full_normals"],
        data["wall_tree"],
        support["spacing_mm"],
    )
    preferred = None
    if opposite_index is not None:
        preferred = data["wall_mm"][opposite_index] - point
    tangent = tangent_axis(normal, preferred[None, :] if preferred is not None else flat_delta)
    local_wall_index = np.asarray(data["wall_tree"].query_ball_point(point, view_radius), dtype=int)
    wall_delta = data["wall_mm"][local_wall_index] - point
    return {
        "tangent": tangent,
        "wall_index": local_wall_index,
        "wall_t": wall_delta @ tangent,
        "wall_eta": wall_delta @ normal,
        "wall_dot": data["full_normals"][local_wall_index] @ normal,
        "support_t": np.einsum("nki,i->nk", support["support_delta"], tangent),
        "v1_t": support["v1_delta"] @ tangent,
        "opposite_distance": np.asarray(opposite_distance),
    }


def choose_sampling_focus(data: dict[str, Any]) -> list[tuple[str, int]]:
    truth = data["truth_mag"]
    p1 = data["pred_mag"]["adaptive_cv_v1"]
    p3 = data["pred_mag"]["normal_multiscale_v3"]
    ray = np.asarray(data["diag_v3"]["ray_used"], dtype=bool)
    finite = ray & np.isfinite(p1) & np.isfinite(p3) & np.isfinite(truth)
    if not finite.any():
        finite = np.isfinite(p3) & np.isfinite(truth)
    repair = np.abs(p1 - truth) - np.abs(p3 - truth)
    first = int(np.nanargmax(np.where(finite, repair, np.nan)))
    second = int(np.nanargmax(np.where(finite, truth, np.nan)))
    ownership = np.asarray(data["diag_v3"]["ownership_rejected"], dtype=float)
    third = int(np.nanargmax(ownership))
    selected: list[tuple[str, int]] = []
    for label, index in (
        ("largest V1 error repair", first),
        ("high-WSS ray target", second),
        ("ownership-gated target", third),
    ):
        if index not in [item[1] for item in selected]:
            selected.append((label, index))
    if len(selected) < 3:
        for index in np.argsort(repair)[::-1]:
            if int(index) not in [item[1] for item in selected]:
                selected.append(("additional repaired target", int(index)))
            if len(selected) == 3:
                break
    return selected[:3]


def save_sampling_figure(data: dict[str, Any], path: Path) -> dict[str, Any]:
    focus = choose_sampling_focus(data)
    centered = data["wall_mm"] - data["wall_mm"].mean(axis=0)
    _, _, axes = np.linalg.svd(centered, full_matrices=False)
    wall_uv = centered @ axes[:2].T
    colors = ("#d62728", "#ff9d00", "#2ca02c")
    fig = plt.figure(figsize=(18.0, 5.8), dpi=180)
    grid = fig.add_gridspec(1, 4, width_ratios=(1.08, 1, 1, 1), wspace=0.28)

    ax = fig.add_subplot(grid[0, 0])
    ax.scatter(wall_uv[:, 0], wall_uv[:, 1], s=2, color="#c7c7c7", alpha=0.42, edgecolors="none")
    for (label, index), color in zip(focus, colors):
        wall_index = int(data["sample_index"][index])
        ax.scatter(
            wall_uv[wall_index, 0], wall_uv[wall_index, 1], s=140, marker="*",
            color=color, edgecolors="k", linewidths=0.7, label=label,
        )
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title("A. Full wall; stars are audited V3 targets")
    ax.set_xlabel("wall PCA-1 (mm)")
    ax.set_ylabel("wall PCA-2 (mm)")
    ax.legend(fontsize=7, frameon=False)
    ax.grid(alpha=0.18)

    report_focus = []
    for panel, ((label, index), color) in enumerate(zip(focus, colors), start=1):
        support = ray_support_for_target(data, index)
        radius = float(data["local_radius"][index]) if data["local_radius"] is not None else 5.0
        view_radius = max(2.4 * support["depth_mm"], min(1.25 * radius, 8.0), 1.5)
        projection = local_projection(data, support, view_radius)
        ax = fig.add_subplot(grid[0, panel])
        same = projection["wall_dot"] >= -0.25
        opposite = projection["wall_dot"] < -0.6
        ax.scatter(projection["wall_t"][same], projection["wall_eta"][same], s=8, color="#b9b9b9", alpha=0.48, edgecolors="none", label="local wall")
        if opposite.any():
            ax.scatter(projection["wall_t"][opposite], projection["wall_eta"][opposite], s=18, facecolors="none", edgecolors="#d62728", linewidths=0.9, label="opposite-facing wall")

        v1_used = support["v1_used"]
        ax.scatter(
            projection["v1_t"][v1_used], support["v1_eta"][v1_used], s=34,
            facecolors="none", edgecolors="#7b2cbf", linewidths=0.8,
            alpha=0.75, label=f"V1 selected KNN (K={support['selected_k']})",
        )
        support_t = projection["support_t"]
        accepted = support["accepted"]
        rejected = support["rejected"]
        if rejected.any():
            ax.scatter(support_t[rejected], support["support_eta"][rejected], s=46, marker="x", color="#e31a1c", linewidths=1.1, label="V3 ownership rejected")
        ax.scatter(support_t[accepted], support["support_eta"][accepted], s=25, color="#2ca25f", alpha=0.72, edgecolors="none", label="V3 IDW support")
        retained_station = support["station_retained"]
        ax.scatter(
            np.zeros(int(retained_station.sum())), support["stations_mm"][retained_station],
            s=54, marker="D", color="#2878b5", edgecolors="white", linewidths=0.5,
            label="normal-ray station",
        )
        ax.plot([0, 0], [0, support["depth_mm"]], color="#2878b5", lw=1.4)
        ax.scatter([0], [0], s=150, marker="*", color=color, edgecolors="k", linewidths=0.7, zorder=8, label="target wall point")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xlim(-view_radius, view_radius)
        ax.set_ylim(-0.25 * view_radius, view_radius)
        ax.set_aspect("equal", adjustable="box")
        pred1 = data["pred_mag"]["adaptive_cv_v1"][index]
        pred3 = data["pred_mag"]["normal_multiscale_v3"][index]
        ax.set_title(
            f"{chr(65+panel)}. {label}\n"
            f"truth={data['truth_mag'][index]:.2f} Pa, V1={pred1:.2f}, V3={pred3:.2f}\n"
            f"depth={support['depth_mm']:.2f} mm, Re={data['diag_v3']['local_reynolds'][index]:.0f}, "
            f"ray_used={int(data['diag_v3']['ray_used'][index])}"
        )
        ax.set_xlabel("local tangent coordinate (mm)")
        if panel == 1:
            ax.set_ylabel("inward normal distance η (mm)")
        ax.grid(alpha=0.2)
        if panel == 3:
            ax.legend(fontsize=6.2, loc="upper right")

        report_focus.append(
            {
                "label": label,
                "sample_row": int(index),
                "wall_index": int(data["sample_index"][index]),
                "truth_wss_pa": float(data["truth_mag"][index]),
                "v1_wss_pa": float(pred1),
                "v3_wss_pa": float(pred3),
                "ray_used": int(data["diag_v3"]["ray_used"][index]),
                "applied_correction": float(data["diag_v3"]["applied_correction"][index]),
                "depth_mm": float(support["depth_mm"]),
                "local_reynolds": float(data["diag_v3"]["local_reynolds"][index]),
                "accepted_support_occurrences": int(support["accepted"].sum()),
                "ownership_rejected_occurrences": int(support["rejected"].sum()),
                "retained_stations": int(support["station_retained"].sum()),
            }
        )

    fig.suptitle(
        f"Frozen V3 normal-ray sampling — {data['row']['canonical_id']}, peak {data['step']}\n"
        "diamonds lie on the physical normal; nearby irregular cell centres are used only as IDW support",
        fontsize=12.5,
    )
    fig.subplots_adjust(left=0.045, right=0.99, bottom=0.12, top=0.80, wspace=0.28)
    fig.savefig(path)
    plt.close(fig)
    return {"case": data["row"]["canonical_id"], "focus": report_focus}


def audit_all_ray_supports(data: dict[str, Any]) -> dict[str, Any]:
    config: NormalMultiscaleV3Config = data["v3_config"]
    n = len(data["sample_wall"])
    depth = (
        np.asarray(data["diag_v3"]["selected_scale_fraction"], dtype=np.float64)
        * np.asarray(data["diag_v3"]["depth_limit_mm"], dtype=np.float64)
    )
    start = float(config.ray_start_fractions[0])
    station_fraction = np.linspace(start, 1.0, config.ray_stations)
    stations = depth[:, None] * station_fraction[None, :]
    query = data["sample_wall"][:, None, :] + stations[:, :, None] * data["sample_normals"][:, None, :]
    query_distance, query_index = data["interior_tree"].query(
        query.reshape(-1, 3), k=min(config.ray_interpolation_neighbors, len(data["interior_mm"]))
    )
    query_distance = query_distance.reshape(n, config.ray_stations, -1)
    query_index = query_index.reshape(n, config.ray_stations, -1)
    support = data["interior_mm"][query_index]
    delta = support - data["sample_wall"][:, None, None, :]
    eta = np.einsum("nski,ni->nsk", delta, data["sample_normals"])
    valid_depth = (eta > 1e-4) & (eta <= 1.25 * depth[:, None, None])

    spacing_distance, _ = data["wall_tree"].query(data["sample_wall"], k=min(8, len(data["wall_mm"])))
    spacing = np.median(spacing_distance[:, 1:], axis=1)
    nearest_wall_distance, anchor = data["wall_tree"].query(support.reshape(-1, 3), k=1)
    nearest_wall_distance = nearest_wall_distance.reshape(query_index.shape)
    anchor = anchor.reshape(query_index.shape)
    anchor_separation = np.linalg.norm(
        data["wall_mm"][anchor] - data["sample_wall"][:, None, None, :], axis=3
    )
    normal_cosine = np.einsum(
        "nski,ni->nsk", data["full_normals"][anchor], data["sample_normals"]
    )
    max_separation = np.maximum(
        config.ownership_max_anchor_separation_mm,
        config.ownership_max_anchor_spacing_multiplier * spacing,
    )
    owner_ok = (
        (anchor_separation <= max_separation[:, None, None])
        & (normal_cosine >= config.ownership_min_normal_cosine)
    )
    accepted = valid_depth & owner_ok
    rejected = valid_depth & ~owner_ok
    target_distance = np.linalg.norm(delta, axis=3)
    post_filter_risk = (
        accepted
        & (normal_cosine < -0.25)
        & (anchor_separation > np.maximum(3.0 * spacing, 1.5)[:, None, None])
        & (nearest_wall_distance < 0.8 * target_distance)
    )
    return {
        "depth": depth,
        "stations": stations,
        "query_distance": query_distance,
        "query_index": query_index,
        "delta": delta,
        "eta": eta,
        "valid_depth": valid_depth,
        "accepted": accepted,
        "rejected": rejected,
        "post_filter_risk": post_filter_risk,
        "anchor": anchor,
        "normal_cosine": normal_cosine,
        "anchor_separation": anchor_separation,
        "nearest_wall_distance": nearest_wall_distance,
        "spacing": spacing,
        "accepted_count": accepted.sum(axis=(1, 2)),
        "rejected_count": rejected.sum(axis=(1, 2)),
        "risk_count": post_filter_risk.sum(axis=(1, 2)),
    }


def choose_penetration_focus(data: dict[str, Any], audit: dict[str, Any]) -> list[tuple[str, int]]:
    rejected = np.asarray(audit["rejected_count"])
    radius = np.asarray(data["local_radius"], dtype=np.float64)
    truth = data["truth_mag"]
    first = int(np.argmax(rejected))
    candidate = rejected >= np.quantile(rejected, 0.80)
    second = int(np.nanargmin(np.where(candidate, radius, np.nan)))
    third = int(np.nanargmax(np.where(candidate, truth, np.nan)))
    selected: list[tuple[str, int]] = []
    for label, index in (
        ("max ownership rejection", first),
        ("narrow-wall target", second),
        ("high-WSS guarded target", third),
    ):
        if index not in [item[1] for item in selected]:
            selected.append((label, index))
    if len(selected) < 3:
        for index in np.argsort(rejected)[::-1]:
            if int(index) not in [item[1] for item in selected]:
                selected.append(("additional guarded target", int(index)))
            if len(selected) == 3:
                break
    return selected[:3]


def save_penetration_figure(data: dict[str, Any], path: Path) -> dict[str, Any]:
    audit = audit_all_ray_supports(data)
    focus = choose_penetration_focus(data, audit)
    centered = data["wall_mm"] - data["wall_mm"].mean(axis=0)
    _, _, axes = np.linalg.svd(centered, full_matrices=False)
    wall_uv = centered @ axes[:2].T
    colors = ("#d62728", "#ff9d00", "#2ca02c")
    fig = plt.figure(figsize=(18.0, 5.8), dpi=180)
    grid = fig.add_gridspec(1, 4, width_ratios=(1.08, 1, 1, 1), wspace=0.28)

    ax = fig.add_subplot(grid[0, 0])
    ax.scatter(wall_uv[:, 0], wall_uv[:, 1], s=2, color="#c7c7c7", alpha=0.42, edgecolors="none")
    for (label, index), color in zip(focus, colors):
        wall_index = int(data["sample_index"][index])
        ax.scatter(wall_uv[wall_index, 0], wall_uv[wall_index, 1], s=140, marker="*", color=color, edgecolors="k", linewidths=0.7, label=label)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title("A. Audited targets on full wall")
    ax.set_xlabel("wall PCA-1 (mm)")
    ax.set_ylabel("wall PCA-2 (mm)")
    ax.legend(fontsize=7, frameon=False)
    ax.grid(alpha=0.18)

    focus_report = []
    for panel, ((label, index), color) in enumerate(zip(focus, colors), start=1):
        support = ray_support_for_target(data, index)
        support["support_delta"] = audit["delta"][index]
        support["support_eta"] = audit["eta"][index]
        support["accepted"] = audit["accepted"][index]
        support["rejected"] = audit["rejected"][index]
        support["post_filter_risk"] = audit["post_filter_risk"][index]
        support["stations_mm"] = audit["stations"][index]
        radius = float(data["local_radius"][index])
        opposite_distance, _ = opposite_wall(
            support["point"], support["normal"], data["wall_mm"], data["full_normals"],
            data["wall_tree"], float(audit["spacing"][index]),
        )
        view_radius = max(
            2.5 * float(audit["depth"][index]),
            1.15 * opposite_distance if np.isfinite(opposite_distance) else 0.0,
            min(1.1 * radius, 8.0),
            1.5,
        )
        projection = local_projection(data, support, view_radius)
        ax = fig.add_subplot(grid[0, panel])
        same = projection["wall_dot"] >= -0.25
        opposite = projection["wall_dot"] < -0.6
        ax.scatter(projection["wall_t"][same], projection["wall_eta"][same], s=8, color="#b9b9b9", alpha=0.48, edgecolors="none", label="local wall")
        if opposite.any():
            ax.scatter(projection["wall_t"][opposite], projection["wall_eta"][opposite], s=18, facecolors="none", edgecolors="#d62728", linewidths=0.9, label="opposite-facing wall")
        v1_used = support["v1_used"]
        ax.scatter(projection["v1_t"][v1_used], support["v1_eta"][v1_used], s=34, facecolors="none", edgecolors="#7b2cbf", linewidths=0.8, alpha=0.70, label="V1 selected KNN")
        support_t = projection["support_t"]
        if support["rejected"].any():
            ax.scatter(support_t[support["rejected"]], support["support_eta"][support["rejected"]], s=52, marker="x", color="#e31a1c", linewidths=1.2, label="rejected before IDW")
        ax.scatter(support_t[support["accepted"]], support["support_eta"][support["accepted"]], s=24, color="#2ca25f", alpha=0.70, edgecolors="none", label="accepted IDW support")
        if support["post_filter_risk"].any():
            ax.scatter(support_t[support["post_filter_risk"]], support["support_eta"][support["post_filter_risk"]], s=80, marker="X", color="#8b0000", edgecolors="k", linewidths=0.5, label="post-filter penetration risk")
        ax.scatter(np.zeros(len(support["stations_mm"])), support["stations_mm"], s=48, marker="D", color="#2878b5", edgecolors="white", linewidths=0.5, label="normal-ray stations")
        ax.plot([0, 0], [0, float(audit["depth"][index])], color="#2878b5", lw=1.4)
        ax.scatter([0], [0], s=150, marker="*", color=color, edgecolors="k", linewidths=0.7, zorder=8, label="target wall point")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xlim(-view_radius, view_radius)
        ax.set_ylim(-0.25 * view_radius, view_radius)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(
            f"{chr(65+panel)}. {label}\n"
            f"pre-filter={int(audit['valid_depth'][index].sum())}, "
            f"rejected={int(audit['rejected_count'][index])}, accepted={int(audit['accepted_count'][index])}\n"
            f"post-filter risk={int(audit['risk_count'][index])}, ray_used={int(data['diag_v3']['ray_used'][index])}"
        )
        ax.set_xlabel("local tangent coordinate (mm)")
        if panel == 1:
            ax.set_ylabel("inward normal distance η (mm)")
        ax.grid(alpha=0.2)
        if panel == 3:
            ax.legend(fontsize=5.9, loc="upper right")

        focus_report.append(
            {
                "label": label,
                "sample_row": int(index),
                "wall_index": int(data["sample_index"][index]),
                "truth_wss_pa": float(data["truth_mag"][index]),
                "local_radius_mm": radius,
                "pre_filter_support_occurrences": int(audit["valid_depth"][index].sum()),
                "ownership_rejected_occurrences": int(audit["rejected_count"][index]),
                "accepted_support_occurrences": int(audit["accepted_count"][index]),
                "post_filter_penetration_risk": int(audit["risk_count"][index]),
                "ray_used": int(data["diag_v3"]["ray_used"][index]),
            }
        )

    total_valid = int(audit["valid_depth"].sum())
    total_rejected = int(audit["rejected"].sum())
    total_accepted = int(audit["accepted"].sum())
    total_risk = int(audit["post_filter_risk"].sum())
    fig.suptitle(
        f"Frozen V3 ownership / penetration audit — {data['row']['canonical_id']}, peak {data['step']}\n"
        f"selected-scale station-support occurrences: pre-filter={total_valid:,}; ownership rejected={total_rejected:,}; "
        f"accepted={total_accepted:,}; post-filter penetration-risk={total_risk:,}",
        fontsize=12.3,
    )
    fig.subplots_adjust(left=0.045, right=0.99, bottom=0.12, top=0.80, wspace=0.28)
    fig.savefig(path)
    plt.close(fig)
    return {
        "case": data["row"]["canonical_id"],
        "n_targets": len(data["sample_wall"]),
        "pre_filter_support_occurrences": total_valid,
        "ownership_rejected_occurrences": total_rejected,
        "accepted_support_occurrences": total_accepted,
        "post_filter_penetration_risk": total_risk,
        "scope": "12 normal-ray stations at each target's selected fitted scale; counts are support occurrences, not unique cells",
        "focus": focus_report,
        "risk_definition": (
            "accepted support is closer to a nonlocal/opposite wall anchor: normal_dot<-0.25, "
            "anchor separation>max(3*wall spacing,1.5mm), and nearest-wall distance<0.8*target distance"
        ),
    }


def save_depth_diagnostics(data: dict[str, Any], path: Path) -> None:
    truth = data["truth_mag"]
    p1 = data["pred_mag"]["adaptive_cv_v1"]
    p3 = data["pred_mag"]["normal_multiscale_v3"]
    valid = np.isfinite(truth) & np.isfinite(p1) & np.isfinite(p3)
    improvement = np.abs(p1 - truth) - np.abs(p3 - truth)
    depth = np.asarray(data["diag_v3"]["depth_limit_mm"], dtype=float)
    reynolds = np.asarray(data["diag_v3"]["local_reynolds"], dtype=float)
    radius = np.asarray(data["local_radius"], dtype=float)
    ratio = np.asarray(data["diag_v3"]["lateral_p50_mm"], dtype=float) / np.maximum(
        np.asarray(data["diag_v3"]["eta_p50_mm"], dtype=float), 1e-6
    )
    ownership = np.asarray(data["diag_v3"]["ownership_rejected"], dtype=float)
    correction = np.asarray(data["diag_v3"]["applied_correction"], dtype=float)
    ray_used = np.asarray(data["diag_v3"]["ray_used"], dtype=bool)

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.5), dpi=170)
    ax = axes[0, 0]
    ax.scatter(truth[valid], np.abs(p1[valid] - truth[valid]), s=12, color=METHOD_COLORS["adaptive_cv_v1"], alpha=0.30, edgecolors="none", label="V1")
    ax.scatter(truth[valid], np.abs(p3[valid] - truth[valid]), s=12, color=METHOD_COLORS["normal_multiscale_v3"], alpha=0.30, edgecolors="none", label="V3")
    ax.set_xlabel("CFD truth WSS (Pa)")
    ax.set_ylabel("absolute error (Pa)")
    ax.set_title("A. Error across WSS magnitude")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)

    ax = axes[0, 1]
    scatter = ax.scatter(radius, depth, c=np.log10(np.maximum(reynolds, 1.0)), s=18, cmap="viridis", alpha=0.70, edgecolors="none")
    radius_axis = np.linspace(0.0, max(float(np.nanmax(radius)), 1.0), 200)
    ax.plot(
        radius_axis,
        np.minimum(data["v3_config"].depth_max_mm, 0.7 * radius_axis),
        "k--",
        lw=1,
        label="min(2.5 mm, 0.7 × radius)",
    )
    ax.set_xlabel("local radius (mm)")
    ax.set_ylabel("Re-based depth limit (mm)")
    ax.set_title("B. Physical depth prior")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.2)
    fig.colorbar(scatter, ax=ax, fraction=0.045, pad=0.03, label="log10(local Re)")

    ax = axes[0, 2]
    ax.hist(reynolds[np.isfinite(reynolds)], bins=42, color="#4c78a8", alpha=0.85)
    ax.axvline(np.nanmedian(reynolds), color="#d62728", lw=1.3, label=f"p50={np.nanmedian(reynolds):.0f}")
    ax.set_xlabel("local Reynolds number")
    ax.set_ylabel("wall target count")
    ax.set_title("C. Local Re distribution")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)

    ax = axes[1, 0]
    ax.scatter(ratio[~ray_used], improvement[~ray_used], s=15, color="#a5a5a5", alpha=0.45, edgecolors="none", label="fallback kept")
    ax.scatter(ratio[ray_used], improvement[ray_used], s=18, color=METHOD_COLORS["normal_multiscale_v3"], alpha=0.65, edgecolors="none", label="ray used")
    ax.axvline(data["v3_config"].hybrid_lateral_over_eta_min, color="#d62728", ls="--", lw=1.2, label="gate=1.4")
    ax.axhline(0, color="k", lw=0.9)
    ax.set_xlabel("IDW support lateral / η")
    ax.set_ylabel("|V1 error| − |V3 error| (Pa)")
    ax.set_title("D. Geometry-risk gate and error repair")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.2)

    ax = axes[1, 1]
    log_ownership = np.log1p(ownership)
    ax.scatter(log_ownership[~ray_used], correction[~ray_used], s=15, color="#a5a5a5", alpha=0.45, edgecolors="none", label="fallback kept")
    ax.scatter(log_ownership[ray_used], correction[ray_used], s=18, color=METHOD_COLORS["normal_multiscale_v3"], alpha=0.65, edgecolors="none", label="ray used")
    ax.axvline(np.log1p(data["v3_config"].hybrid_max_ownership_rejected), color="#d62728", ls="--", lw=1.2, label="ownership gate=20")
    ax.set_xlabel("log1p(ownership-rejected support count)")
    ax.set_ylabel("applied V1 magnitude correction")
    ax.set_title("E. Ownership gate prevents unsafe takeover")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.2)

    ax = axes[1, 2]
    bins = np.linspace(0.995, 1.205, 30)
    ax.hist(correction[ray_used], bins=bins, color=METHOD_COLORS["normal_multiscale_v3"], alpha=0.85, label="ray-used targets")
    ax.axvline(1.0, color="k", lw=1)
    ax.axvline(1.2, color="#d62728", ls="--", lw=1.2, label="frozen cap=1.2")
    ax.set_xlabel("applied correction")
    ax.set_ylabel("wall target count")
    ax.set_title(f"F. Capped correction; ray used={ray_used.mean():.1%}")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.2)

    fig.suptitle(
        f"Frozen V3 depth/Re/gating diagnostics — {data['row']['canonical_id']}, peak {data['step']}",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path)
    plt.close(fig)


def save_contact_sheet(image_paths: list[Path], path: Path) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(18, 22), dpi=120)
    for ax, image_path in zip(axes.flat, image_paths):
        image = plt.imread(image_path)
        ax.imshow(image)
        ax.set_title(image_path.stem.replace("_", " "), fontsize=10)
        ax.axis("off")
    for ax in axes.flat[len(image_paths):]:
        ax.axis("off")
    fig.suptitle("Frozen normal-ray V3 quicklook atlas", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(path)
    plt.close(fig)


def write_readme(
    out_dir: Path,
    focus_rows: list[tuple[str, dict[str, Any]]],
    penetration: dict[str, Any],
) -> None:
    case_lines = "\n".join(
        f"- `{role}`：`{row['canonical_id']}`"
        for role, row in focus_rows
    )
    text = f"""# Frozen V3 quicklook

本目录只做冻结 V3 的结果可视化，不修改方法参数。点级散点使用与 blind test35 完全
一致的病例、随机种子和每例 1200 个壁面点重新计算，并检查 raw R² 与冻结结果逐病例一致。

## 推荐看图顺序

| 图片 | 内容 | 重点 |
| --- | --- | --- |
| `00_overview_contact_sheet.png` | 一页图册 | 快速总览 |
| `01_blind_accuracy_summary.png` | test35 病例级精度 | V3 相对 V1/V2 的均值和尾部改善 |
| `02_best_gain_scatter.png` | 最大增益病例点级散点 | V3 是否修复幅值低估 |
| `03_median_gain_scatter.png` | 中位增益病例点级散点 | 典型收益是否稳定 |
| `04_guard_case_scatter.png` | V3 相对 V2 最差病例 | 查看保护门控的边界情况 |
| `05_normal_ray_sampling.png` | 法线站点与不规则点云 IDW 支撑 | 对应老师提出的“画法线、把最近点投到线上” |
| `06_penetration_ownership_audit.png` | 邻壁归属和穿模审计 | 红叉在 IDW 前被拒绝；检查过滤后是否仍穿模 |
| `07_depth_re_error_diagnostics.png` | Re 深度先验与 hybrid 门控 | 深度、横向混合、ownership 和 1.2 修正上限 |

代表病例：

{case_lines}

穿模审计病例 `{penetration['case']}`：在每个目标最终选中尺度的 12 个法线站点上，
预过滤支撑出现
`{penetration['pre_filter_support_occurrences']:,}` 次，ownership 拒绝
`{penetration['ownership_rejected_occurrences']:,}` 次，接受
`{penetration['accepted_support_occurrences']:,}` 次；按图中三维非局部/对侧壁判据，
过滤后 penetration-risk 为 `{penetration['post_filter_penetration_risk']}`。

这里统计的是“站点—支撑点出现次数”，同一内部单元可支撑多个相邻站点；它与 V3
运行结果中跨全部候选尺度累计的 `ownership_rejected_mean` 不是同一个分母。

完整机器可读记录见 `audit_summary.json`。
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    result_path = args.result.resolve()
    config_path = args.config.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    result = load_json(result_path)
    experiment_config = load_json(config_path)
    focus_rows = select_focus_cases(result)

    figures: list[Path] = []
    summary_path = out_dir / "01_blind_accuracy_summary.png"
    save_blind_summary(result, summary_path)
    figures.append(summary_path)

    reproduced: dict[str, Any] = {}
    sampling_report: dict[str, Any] | None = None
    penetration_report: dict[str, Any] | None = None
    for index, (role, row) in enumerate(focus_rows, start=2):
        data = compute_case(row, experiment_config)
        scatter_path = out_dir / f"{index:02d}_{role}_scatter.png"
        save_case_scatter(data, role, scatter_path)
        figures.append(scatter_path)
        reproduced[role] = {
            "canonical_id": row["canonical_id"],
            "step": data["step"],
            "reports": data["reports"],
            "frozen_result_metrics": row["methods"],
            "radius_source": data["radius_source"],
        }
        if role == "best_gain":
            sampling_path = out_dir / "05_normal_ray_sampling.png"
            sampling_report = save_sampling_figure(data, sampling_path)
            figures.append(sampling_path)
            diagnostics_path = out_dir / "07_depth_re_error_diagnostics.png"
            save_depth_diagnostics(data, diagnostics_path)
        if role == "guard_case":
            penetration_path = out_dir / "06_penetration_ownership_audit.png"
            penetration_report = save_penetration_figure(data, penetration_path)
            figures.append(penetration_path)

    diagnostics_path = out_dir / "07_depth_re_error_diagnostics.png"
    if diagnostics_path.exists():
        figures.append(diagnostics_path)
    if sampling_report is None or penetration_report is None:
        raise RuntimeError("sampling or penetration report was not generated")

    contact_path = out_dir / "00_overview_contact_sheet.png"
    figures = sorted(figures, key=lambda item: item.name)
    save_contact_sheet(figures, contact_path)
    frozen_method_path = (
        PACKAGE / "experiments/pointcloud_normal_multiscale_v3/config_frozen_v3.json"
    )
    audit = {
        "schema_version": 1,
        "method": "pointcloud_normal_multiscale_v3",
        "experiment_config": str(config_path),
        "experiment_config_sha256": sha256_file(config_path),
        "frozen_method_config": str(frozen_method_path),
        "frozen_method_sha256": sha256_file(frozen_method_path),
        "blind_result": str(result_path),
        "blind_result_sha256": sha256_file(result_path),
        "no_post_test_tuning": True,
        "focus_cases": reproduced,
        "sampling": sampling_report,
        "penetration": penetration_report,
        "figures": [str(contact_path), *(str(path) for path in figures)],
    }
    (out_dir / "audit_summary.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_readme(out_dir, focus_rows, penetration_report)
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    print(f"quicklook: {out_dir}")


if __name__ == "__main__":
    main()
