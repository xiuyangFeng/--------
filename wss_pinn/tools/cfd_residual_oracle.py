from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from wss_pinn.config import ExperimentConfig
from wss_pinn.data.raw_io import load_cases, read_interior, step_file
from wss_pinn.physics.residuals import local_linear_velocity_gradients
from wss_pinn.utils import atomic_write_json, utc_now


def summarize_gradient(gradient: np.ndarray, condition: np.ndarray) -> dict:
    divergence = np.trace(gradient, axis1=1, axis2=2)
    gradient_norm = np.linalg.norm(gradient, axis=(1, 2))
    relative = np.abs(divergence) / np.maximum(gradient_norm, 1e-8)
    finite = np.isfinite(divergence) & np.isfinite(relative) & np.isfinite(condition)
    return {
        "finite_fraction": float(finite.mean()),
        "divergence_abs_p50": float(np.nanmedian(np.abs(divergence))),
        "divergence_abs_p95": float(np.nanquantile(np.abs(divergence), 0.95)),
        "relative_divergence_p50": float(np.nanmedian(relative)),
        "relative_divergence_p95": float(np.nanquantile(relative, 0.95)),
        "condition_p50": float(np.nanmedian(condition)),
        "condition_p99": float(np.nanquantile(condition, 0.99)),
    }


def analytic_check() -> dict:
    rng = np.random.default_rng(1234)
    coords = rng.uniform(-1, 1, size=(4000, 3))
    query = rng.choice(len(coords), 256, replace=False)
    velocity_good = np.column_stack([coords[:, 0], coords[:, 1], -2 * coords[:, 2]])
    velocity_bad = coords.copy()
    grad_good, _ = local_linear_velocity_gradients(
        coords, velocity_good, query, neighbors=32
    )
    grad_bad, _ = local_linear_velocity_gradients(
        coords, velocity_bad, query, neighbors=32
    )
    good = np.trace(grad_good, axis1=1, axis2=2)
    bad = np.trace(grad_bad, axis1=1, axis2=2)
    passed = float(np.max(np.abs(good))) < 1e-8 and float(np.median(bad)) > 2.9
    return {
        "status": "pass" if passed else "fail",
        "divergence_free_max_abs": float(np.max(np.abs(good))),
        "positive_control_median": float(np.median(bad)),
    }


def audit_case(case: dict, sample_count: int, seed: int) -> dict:
    bundle_path = Path(case["bundle_path"])
    raw_case_dir = Path(case["raw_case_dir"])
    with np.load(bundle_path, allow_pickle=False) as source:
        peak = int(source["peak_step"])
        coords = np.asarray(source["int_coords_norm"], dtype=np.float64)
        int_type = np.asarray(source["int_type"])
        rotation = np.asarray(source["transform_rotation"])
    raw = read_interior(step_file(raw_case_dir, peak, "ascii_in"))
    if len(raw["velocity"]) != len(coords):
        raise ValueError("raw/bundle interior row mismatch")
    velocity = raw["velocity"] @ rotation
    velocity_scale = max(
        float(np.quantile(np.linalg.norm(velocity, axis=1), 0.95)), 1e-6
    )
    velocity_nd = velocity / velocity_scale
    rng = np.random.default_rng(seed)
    near = np.flatnonzero(int_type == 1)
    # Frozen upstream encoding is core/interior=0, near-wall=1.
    core = np.flatnonzero(int_type == 0)
    each = max(1, sample_count // 2)
    query = np.concatenate(
        [
            rng.choice(near, min(each, len(near)), replace=False),
            rng.choice(core, min(each, len(core)), replace=False),
        ]
    )
    reports = {}
    divergences = {}
    gradient_norms = {}
    for neighbors in (24, 48):
        gradient, condition = local_linear_velocity_gradients(
            coords, velocity_nd, query, neighbors=neighbors
        )
        reports[f"local_linear_k{neighbors}"] = summarize_gradient(gradient, condition)
        divergences[neighbors] = np.trace(gradient, axis1=1, axis2=2)
        gradient_norms[neighbors] = np.linalg.norm(gradient, axis=(1, 2))
    cross = float(
        np.nanmedian(
            np.abs(divergences[24] - divergences[48])
            / np.maximum(
                0.5 * (gradient_norms[24] + gradient_norms[48]), 1e-8
            )
        )
    )
    selected = reports["local_linear_k48"]
    passed = bool(
        selected["finite_fraction"] >= 0.99
        and selected["relative_divergence_p95"] <= 0.5
        and selected["condition_p99"] <= 1e10
        and cross <= 0.25
    )
    return {
        "case_id": case["case_id"],
        "cohort": case["cohort"],
        "status": "completed",
        "gate_result": "pass" if passed else "fail",
        "sample_count": int(len(query)),
        "velocity_scale_m_s": velocity_scale,
        "coordinate_scale": "frozen per-case dimensionless registered coordinate",
        "residual_scale": "div(u/U)/d(x/L), dimensionless",
        "methods": reports,
        "neighbor_cross_gradient_normalized_difference_p50": cross,
        "gate_revision": (
            "The first pilot used |div24-div48|/mean(|div|), which is singular "
            "near the incompressible target div=0. It was replaced before the "
            "decisive rerun by |div24-div48|/mean(||grad u||)."
        ),
        "gate": {
            "finite_fraction_min": 0.99,
            "relative_divergence_p95_max": 0.5,
            "condition_p99_max": 1e10,
            "neighbor_cross_gradient_normalized_difference_p50_max": 0.25,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--sample-count", type=int, default=512)
    args = parser.parse_args()
    config = ExperimentConfig.from_json(args.config)
    analytic = analytic_check()
    reports = [
        audit_case(
            case,
            sample_count=args.sample_count,
            seed=int(config["sampling"]["seed"]) + index,
        )
        for index, case in enumerate(load_cases(config["data"]["split_path"]))
    ]
    passed = analytic["status"] == "pass" and all(
        row["gate_result"] == "pass" for row in reports
    )
    payload = {
        "schema_version": 1,
        "created_at": utc_now(),
        "stage": "P0-C",
        "scope": "continuity only; momentum remains outside F1 and weight=0",
        "status": "completed",
        "gate_result": "pass" if passed else "fail",
        "analytic_field_check": analytic,
        "cases": reports,
        "f1_status": "unblocked" if passed else "blocked",
    }
    output = atomic_write_json(
        Path(config["paths"]["output_root"]) / "audits/p0c/summary.json", payload
    )
    print(f"P0-C {payload['gate_result']} F1={payload['f1_status']} {output}")


if __name__ == "__main__":
    main()
