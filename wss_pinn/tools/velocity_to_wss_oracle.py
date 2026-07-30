from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from wss_pinn.config import ExperimentConfig
from wss_pinn.data.raw_io import load_cases, read_interior, step_file
from wss_pinn.physics.wall_shear import (
    map_stl_normals_to_wall,
    wall_shear_from_normal_gradient,
)
from wss_pinn.utils import atomic_write_json, utc_now


def metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(truth) & np.isfinite(prediction)
    y = truth[valid]
    p = prediction[valid]
    if len(y) < 3:
        return {"count": float(len(y)), "r2": float("nan"), "mae_pa": float("nan")}
    denominator = float(np.sum(np.square(y - y.mean())))
    r2 = 1.0 - float(np.sum(np.square(y - p))) / max(denominator, 1e-12)
    rmse = float(np.sqrt(np.mean(np.square(y - p))))
    high = y >= np.quantile(y, 0.9)
    return {
        "count": float(len(y)),
        "r2": r2,
        "mae_pa": float(np.mean(np.abs(y - p))),
        "rmse_pa": rmse,
        "nrmse": rmse / max(float(np.mean(np.abs(y))), 1e-12),
        "high_wss_nrmse": float(np.sqrt(np.mean(np.square(y[high] - p[high]))))
        / max(float(np.mean(np.abs(y[high]))), 1e-12),
    }


def fit_many(
    tree: cKDTree,
    wall_points_mm: np.ndarray,
    normals: np.ndarray,
    interior_coords_mm: np.ndarray,
    velocity_m_s: np.ndarray,
    *,
    neighbors: int,
    degree: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    k = min(int(neighbors), len(interior_coords_mm))
    distances, neighbor_indices = tree.query(wall_points_mm, k=k)
    gradients = np.full((len(wall_points_mm), 3), np.nan, dtype=np.float64)
    conditions = np.full(len(wall_points_mm), np.inf, dtype=np.float64)
    used_counts = np.zeros(len(wall_points_mm), dtype=np.int32)
    for row, (wall, normal, dist, index) in enumerate(
        zip(wall_points_mm, normals, distances, neighbor_indices)
    ):
        delta = interior_coords_mm[index] - wall
        signed = delta @ normal
        positive = signed > 0
        negative = signed < 0
        mask = positive if positive.sum() >= negative.sum() else negative
        if mask.sum() < 8:
            mask = np.ones_like(signed, dtype=bool)
        d_m = np.abs(signed[mask]) * 1e-3
        tangent = velocity_m_s[index][mask]
        tangent = tangent - np.outer(tangent @ normal, normal)
        keep = d_m > 1e-7
        d_m = d_m[keep]
        tangent = tangent[keep]
        if len(d_m) < 8:
            continue
        design = np.column_stack(
            [d_m**power for power in range(1, int(degree) + 1)]
        )
        bandwidth = max(float(np.median(d_m)), 1e-8)
        weights = np.exp(-np.square(d_m / (2.0 * bandwidth)))
        weighted = design * np.sqrt(weights[:, None])
        target = tangent * np.sqrt(weights[:, None])
        try:
            coefficient, *_ = np.linalg.lstsq(weighted, target, rcond=1e-10)
        except np.linalg.LinAlgError:
            continue
        gradients[row] = coefficient[0]
        conditions[row] = np.linalg.cond(weighted.T @ weighted)
        used_counts[row] = len(d_m)
    return gradients, conditions, used_counts


def audit_case(case: dict, sample_count: int, seed: int) -> dict:
    bundle_path = Path(case["bundle_path"])
    raw_case_dir = Path(case["raw_case_dir"])
    with np.load(bundle_path, allow_pickle=False) as source:
        peak = int(source["peak_step"])
        step_index = int(np.flatnonzero(source["steps"] == peak)[0])
        wall_coords_norm = np.asarray(source["wall_coords_norm"])
        wall_coords_raw = np.asarray(source["wall_coords_raw"])
        wall_wss = np.asarray(source["wall_wss"])[step_index]
        wall_wss_vector = np.asarray(source["wall_wss_vec"])[step_index]
        int_coords_norm = np.asarray(source["int_coords_norm"])
        coord_scale_mm = float(source["coord_scale"])
        rotation = np.asarray(source["transform_rotation"])
        stl_path = Path(str(source["original_stl_path"].item()))
        stl_scale = float(source["original_stl_scale_to_mm"])
    raw = read_interior(step_file(raw_case_dir, peak, "ascii_in"))
    if len(raw["velocity"]) != len(int_coords_norm):
        raise ValueError("raw/bundle interior row mismatch; P0-A alignment is required")
    velocity = raw["velocity"] @ rotation
    interior_mm = int_coords_norm.astype(np.float64) * coord_scale_mm
    rng = np.random.default_rng(seed)
    wall_indices = np.sort(
        rng.choice(len(wall_coords_norm), min(sample_count, len(wall_coords_norm)), replace=False)
    )
    normals, map_distance = map_stl_normals_to_wall(
        wall_coords_raw[wall_indices], stl_path, stl_scale, rotation=rotation
    )
    wall_points_mm = wall_coords_norm[wall_indices].astype(np.float64) * coord_scale_mm
    tree = cKDTree(interior_mm)
    method_reports = {}
    vectors = {}
    for name, neighbors, degree in (
        ("linear_k32", 32, 1),
        ("linear_k64", 64, 1),
        ("quadratic_k64", 64, 2),
    ):
        gradient, condition, used = fit_many(
            tree,
            wall_points_mm,
            normals,
            interior_mm,
            velocity,
            neighbors=neighbors,
            degree=degree,
        )
        shear_vector, viscosity = wall_shear_from_normal_gradient(gradient)
        magnitude = np.linalg.norm(shear_vector, axis=1)
        truth = wall_wss[wall_indices]
        truth_vector = wall_wss_vector[wall_indices]
        valid_direction = (
            np.isfinite(shear_vector).all(axis=1)
            & (magnitude > 1e-12)
            & (np.linalg.norm(truth_vector, axis=1) > 1e-12)
        )
        cosine = np.sum(shear_vector[valid_direction] * truth_vector[valid_direction], axis=1)
        cosine /= (
            magnitude[valid_direction] * np.linalg.norm(truth_vector[valid_direction], axis=1)
        )
        report = metrics(truth, magnitude)
        report.update(
            {
                "finite_fraction": float(np.isfinite(magnitude).mean()),
                "condition_p95": float(np.nanquantile(condition, 0.95)),
                "neighbors_used_p50": float(np.median(used)),
                "viscosity_pa_s_p50": float(np.nanmedian(viscosity)),
                "direction_cosine_mean": float(np.nanmean(cosine)),
                "direction_abs_cosine_mean": float(np.nanmean(np.abs(cosine))),
            }
        )
        method_reports[name] = report
        vectors[name] = magnitude
    best_name = max(
        method_reports,
        key=lambda name: (
            method_reports[name]["r2"]
            if np.isfinite(method_reports[name]["r2"])
            else -np.inf
        ),
    )
    best = method_reports[best_name]
    shell_stability = float(
        np.nanmedian(
            np.abs(vectors["linear_k32"] - vectors["linear_k64"])
            / np.maximum(np.abs(vectors["linear_k64"]), 1e-6)
        )
    )
    passed = bool(
        best["finite_fraction"] >= 0.99
        and best["r2"] >= 0.5
        and best["high_wss_nrmse"] <= 0.5
        and shell_stability <= 0.35
    )
    return {
        "case_id": case["case_id"],
        "cohort": case["cohort"],
        "status": "completed",
        "gate_result": "pass" if passed else "fail",
        "sample_count": int(len(wall_indices)),
        "normal_mapping_distance_mm_p95": float(np.quantile(map_distance, 0.95)),
        "methods": method_reports,
        "best_method": best_name,
        "multi_shell_relative_difference_p50": shell_stability,
        "gate": {
            "finite_fraction_min": 0.99,
            "r2_min": 0.5,
            "high_wss_nrmse_max": 0.5,
            "multi_shell_relative_difference_p50_max": 0.35,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--sample-count", type=int, default=256)
    args = parser.parse_args()
    config = ExperimentConfig.from_json(args.config)
    reports = [
        audit_case(
            case,
            sample_count=args.sample_count,
            seed=int(config["sampling"]["seed"]) + index,
        )
        for index, case in enumerate(load_cases(config["data"]["split_path"]))
    ]
    passed = all(row["gate_result"] == "pass" for row in reports)
    payload = {
        "schema_version": 1,
        "created_at": utc_now(),
        "stage": "P0-B",
        "status": "completed",
        "gate_result": "pass" if passed else "fail",
        "f2_status": "code_ready" if passed else "blocked",
        "does_not_block": ["P1", "F0-U", "F0-UP", "F1"],
        "cases": reports,
        "conclusion": (
            "P0-B gates only WSS-physics/F2. F1 does not use this operator."
        ),
    }
    output = atomic_write_json(
        Path(config["paths"]["output_root"]) / "audits/p0b/summary.json", payload
    )
    print(f"P0-B {payload['gate_result']} F2={payload['f2_status']} {output}")


if __name__ == "__main__":
    main()

