"""Train-only study of guarded quadratic/cubic Surface-MLS fusion."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from batch_validate_cfd import load_split_cases, summarize
from calculate_wss_cfd import (
    evaluate,
    fit_wall_gradient,
    pca_wall_normals,
    resolve_step,
    wss_from_gradient,
)
import data_loader_cfd as dl
from run_v3_experiment import _resolve_radius
from wss_normal_multiscale_v3 import (
    NormalMultiscaleV3Config,
    fit_wall_gradient_normal_multiscale,
)
from wss_surface_mls_v4 import SurfaceMLSV4Config, fit_wall_gradient_surface_mls


ROOT = Path("/public/newhome/cy/Digital_twin/GNN")
SPLIT = (
    ROOT
    / "wss_pinn/configs/splits/"
    "split_WSS_PINN_AG_AAA_ILO_q2v_pool2025_train138_test35_exclude_SHI_YUN_XI_v1.json"
)
V1_CONFIG = (
    ROOT
    / "wss_mri_calculator/experiments/pointcloud_adaptive_v1/config_frozen_v1.json"
)
V3_CONFIG = (
    ROOT
    / "wss_mri_calculator/experiments/pointcloud_normal_multiscale_v3/config_frozen_v3.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit-per-cohort", type=int, default=12)
    parser.add_argument("--sample-count", type=int, default=400)
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--point-cache-dir", default="")
    args = parser.parse_args()

    cases = load_split_cases(SPLIT, ROOT / "data_new", {"train"})
    if args.limit_per_cohort > 0:
        kept = []
        counts: dict[str, int] = {}
        for case in cases:
            if counts.get(case["cohort"], 0) < args.limit_per_cohort:
                kept.append(case)
                counts[case["cohort"]] = counts.get(case["cohort"], 0) + 1
        cases = kept

    v1_payload = json.loads(V1_CONFIG.read_text(encoding="utf-8"))
    v3_payload = json.loads(V3_CONFIG.read_text(encoding="utf-8"))
    v3_config = NormalMultiscaleV3Config.from_mapping(v3_payload["method"])
    base_config = SurfaceMLSV4Config(correction_max=1.5)
    depth3_config = SurfaceMLSV4Config(
        correction_max=1.5,
        parallel_transport=True,
        group_balance_power=0.5,
        depth_degree=3,
    )
    guard_settings = [
        (cv_factor, blend)
        for cv_factor in (1.0, 1.25, 1.5)
        for blend in (0.25, 0.50, 1.00)
    ]
    method_names = [
        "v3",
        "base",
        "depth3",
        "max_base_depth3",
        *[
            f"guard_cv{cv_factor:g}_blend{blend:g}"
            for cv_factor, blend in guard_settings
        ],
    ]
    cache_dir = Path(args.point_cache_dir).resolve() if args.point_cache_dir else None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    started = time.time()
    for number, case in enumerate(cases, start=1):
        case_dir = Path(case["case_dir"])
        step, _ = resolve_step(case_dir, None, prefer_peak=True)
        wall = dl.load_wall(dl.step_path(case_dir, step, "ascii"))
        interior = dl.load_interior(dl.step_path(case_dir, step, "ascii_in"))
        interior_tree = cKDTree(interior["coords_mm"])
        full_normals = pca_wall_normals(
            wall["coords_mm"], interior["coords_mm"], interior_tree
        )
        sample_index = np.sort(
            np.random.default_rng(0).choice(
                len(wall["coords_mm"]),
                min(args.sample_count, len(wall["coords_mm"])),
                replace=False,
            )
        )
        wall_sample = wall["coords_mm"][sample_index]
        normal_sample = full_normals[sample_index]
        local_radius, _ = _resolve_radius(
            "bundle_then_wall",
            case_dir,
            wall_sample,
            normal_sample,
            wall["coords_mm"],
            full_normals,
            cKDTree(wall["coords_mm"]),
            {},
        )
        v1_gradient, _, _ = fit_wall_gradient(
            wall_sample,
            normal_sample,
            interior["coords_mm"],
            interior["velocity"],
            interior_tree,
            degree=int(v1_payload["degree"]),
            neighbor_mode="adaptive_cv",
            adaptive_neighbors=tuple(v1_payload["adaptive_neighbors"]),
            cv_tolerance=float(v1_payload["cv_tolerance"]),
        )
        v3_gradient, _, _ = fit_wall_gradient_normal_multiscale(
            wall_sample,
            normal_sample,
            wall["coords_mm"],
            full_normals,
            interior["coords_mm"],
            interior["velocity"],
            interior_tree,
            local_radius_mm=local_radius,
            config=v3_config,
            fallback_gradient=v1_gradient,
        )
        _, base_diag = fit_wall_gradient_surface_mls(
            wall_sample,
            normal_sample,
            wall["coords_mm"],
            full_normals,
            interior["coords_mm"],
            interior["velocity"],
            interior_tree,
            local_radius_mm=local_radius,
            fallback_gradient=v1_gradient,
            config=base_config,
        )
        _, depth3_diag = fit_wall_gradient_surface_mls(
            wall_sample,
            normal_sample,
            wall["coords_mm"],
            full_normals,
            interior["coords_mm"],
            interior["velocity"],
            interior_tree,
            local_radius_mm=local_radius,
            fallback_gradient=v1_gradient,
            config=depth3_config,
        )

        v1_norm = np.linalg.norm(v1_gradient, axis=1)
        v3_correction = np.linalg.norm(v3_gradient, axis=1) / np.maximum(v1_norm, 1e-12)
        base_correction = np.maximum(v3_correction, base_diag["correction"])
        depth3_correction = np.maximum(v3_correction, depth3_diag["correction"])
        corrections: dict[str, np.ndarray] = {
            "v3": v3_correction,
            "base": base_correction,
            "depth3": depth3_correction,
            "max_base_depth3": np.maximum(base_correction, depth3_correction),
        }
        depth3_gain = np.maximum(depth3_correction - base_correction, 0.0)
        finite_scores = np.isfinite(base_diag["cv_score"]) & np.isfinite(
            depth3_diag["cv_score"]
        )
        stable_common = (
            finite_scores
            & (depth3_diag["condition"] <= 1e5)
            & (depth3_diag["selected_samples"] >= 24)
            & (depth3_diag["anchor_groups"] >= 8)
        )
        for cv_factor, blend in guard_settings:
            allowed = stable_common & (
                depth3_diag["cv_score"]
                <= cv_factor * np.maximum(base_diag["cv_score"], 1e-8)
            )
            corrections[f"guard_cv{cv_factor:g}_blend{blend:g}"] = (
                base_correction + blend * depth3_gain * allowed
            )

        truth_mag = wall["wss_mag"][sample_index]
        truth_vec = wall["wss_vec"][sample_index]
        methods = {}
        predictions = {}
        for name in method_names:
            gradient = v1_gradient * corrections[name][:, None]
            vector = wss_from_gradient(gradient, "carreau")
            predictions[name] = np.linalg.norm(vector, axis=1)
            methods[name] = evaluate(truth_mag, truth_vec, vector)
        rows.append(
            {
                "canonical_id": case["canonical_id"],
                "cohort": case["cohort"],
                "methods": methods,
            }
        )
        if cache_dir is not None:
            path = cache_dir / (case["canonical_id"].replace("/", "__") + ".npz")
            with path.open("wb") as handle:
                np.savez_compressed(
                    handle,
                    canonical_id=np.asarray(case["canonical_id"]),
                    cohort=np.asarray(case["cohort"]),
                    sample_index=sample_index,
                    wall_mm=wall_sample.astype(np.float32),
                    truth_mag=truth_mag.astype(np.float32),
                    truth_vec=truth_vec.astype(np.float32),
                    v1_gradient=v1_gradient.astype(np.float32),
                    v3_correction=v3_correction.astype(np.float32),
                    base_correction=base_correction.astype(np.float32),
                    depth3_correction=depth3_correction.astype(np.float32),
                    base_cv=np.asarray(base_diag["cv_score"], dtype=np.float32),
                    depth3_cv=np.asarray(depth3_diag["cv_score"], dtype=np.float32),
                    base_condition=np.asarray(base_diag["condition"], dtype=np.float32),
                    depth3_condition=np.asarray(depth3_diag["condition"], dtype=np.float32),
                    base_samples=np.asarray(base_diag["selected_samples"], dtype=np.int32),
                    depth3_samples=np.asarray(depth3_diag["selected_samples"], dtype=np.int32),
                    base_anchor_groups=np.asarray(base_diag["anchor_groups"], dtype=np.int32),
                    depth3_anchor_groups=np.asarray(depth3_diag["anchor_groups"], dtype=np.int32),
                    **{
                        f"pred__{name}": value.astype(np.float32)
                        for name, value in predictions.items()
                    },
                )
        print(
            f"[{number:02d}/{len(cases):02d}] {case['canonical_id']} "
            f"base={methods['base']['raw_r2']:.4f} "
            f"d3={methods['depth3']['raw_r2']:.4f} "
            f"max={methods['max_base_depth3']['raw_r2']:.4f}",
            flush=True,
        )

    aggregate = {}
    for name in method_names:
        aggregate[name] = {
            metric: summarize([row["methods"][name][metric] for row in rows])
            for metric in (
                "raw_r2",
                "scaled_r2",
                "spearman",
                "alpha",
                "mae_pa",
                "nrmse",
                "high_wss_nrmse",
            )
        }
        if name != "base":
            delta = np.asarray(
                [
                    row["methods"][name]["raw_r2"]
                    - row["methods"]["base"]["raw_r2"]
                    for row in rows
                ]
            )
            aggregate[name]["vs_base"] = {
                "mean": float(np.mean(delta)),
                "wins": int(np.sum(delta > 0)),
                "min": float(np.min(delta)),
            }
    payload = {
        "status": "train_only_guarded_depth3_study",
        "sample_count": args.sample_count,
        "limit_per_cohort": args.limit_per_cohort,
        "base_config": base_config.__dict__,
        "depth3_config": depth3_config.__dict__,
        "guard_settings": [
            {"cv_factor": cv_factor, "blend": blend}
            for cv_factor, blend in guard_settings
        ],
        "aggregate": aggregate,
        "rows": rows,
        "elapsed_seconds": time.time() - started,
    }
    output = Path(args.json_out).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(aggregate, indent=2, ensure_ascii=False), flush=True)
    print(output)


if __name__ == "__main__":
    main()
