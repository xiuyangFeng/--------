"""Train-only sweep of closest-surface MLS fitting variants."""

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
from wss_surface_mls_v4 import (
    SurfaceColumnV4Config,
    SurfaceMLSV4Config,
    fit_wall_gradient_surface_columns,
    fit_wall_gradient_surface_mls,
)


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


VARIANTS: dict[str, dict[str, object]] = {
    "base": {},
    "transport": {"parallel_transport": True},
    "balance05": {"group_balance_power": 0.5},
    "balance10": {"group_balance_power": 1.0},
    "transport_balance05": {
        "parallel_transport": True,
        "group_balance_power": 0.5,
    },
    "transport_balance10": {
        "parallel_transport": True,
        "group_balance_power": 1.0,
    },
    "transport_balance05_depth3": {
        "parallel_transport": True,
        "group_balance_power": 0.5,
        "depth_degree": 3,
    },
}

COLUMN_VARIANTS: dict[str, dict[str, object]] = {
    "columns_linear_surface": {},
    "columns_constant_surface": {"surface_gradient_degree": 0},
    "columns_depth1": {"column_depth_degree": 1},
    "columns_depth3": {"column_depth_degree": 3},
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit-per-cohort", type=int, default=1)
    parser.add_argument("--sample-count", type=int, default=240)
    parser.add_argument("--json-out", required=True)
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
    variant_configs = {
        name: SurfaceMLSV4Config(correction_max=1.5, **settings)
        for name, settings in VARIANTS.items()
    }
    column_configs = {
        name: SurfaceColumnV4Config(correction_max=1.5, **settings)
        for name, settings in COLUMN_VARIANTS.items()
    }
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
        if 0 < args.sample_count < len(wall["coords_mm"]):
            sample_index = np.sort(
                np.random.default_rng(0).choice(
                    len(wall["coords_mm"]), args.sample_count, replace=False
                )
            )
        else:
            sample_index = np.arange(len(wall["coords_mm"]))
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
        v1_norm = np.linalg.norm(v1_gradient, axis=1)
        v3_correction = np.linalg.norm(v3_gradient, axis=1) / np.maximum(
            v1_norm, 1e-12
        )
        truth_mag = wall["wss_mag"][sample_index]
        truth_vec = wall["wss_vec"][sample_index]
        methods = {
            "v3": evaluate(
                truth_mag, truth_vec, wss_from_gradient(v3_gradient, "carreau")
            )
        }
        for name, config in variant_configs.items():
            _, diagnostics = fit_wall_gradient_surface_mls(
                wall_sample,
                normal_sample,
                wall["coords_mm"],
                full_normals,
                interior["coords_mm"],
                interior["velocity"],
                interior_tree,
                local_radius_mm=local_radius,
                fallback_gradient=v1_gradient,
                config=config,
            )
            correction = np.maximum(v3_correction, diagnostics["correction"])
            gradient = v1_gradient * correction[:, None]
            report = evaluate(
                truth_mag, truth_vec, wss_from_gradient(gradient, "carreau")
            )
            report["fit_coverage"] = float(
                np.mean(np.isfinite(diagnostics["raw_scalar_gradient"]))
            )
            report["applied_fraction"] = float(np.mean(diagnostics["applied"] > 0))
            methods[name] = report
        for name, config in column_configs.items():
            _, diagnostics = fit_wall_gradient_surface_columns(
                wall_sample,
                normal_sample,
                wall["coords_mm"],
                full_normals,
                interior["coords_mm"],
                interior["velocity"],
                interior_tree,
                local_radius_mm=local_radius,
                fallback_gradient=v1_gradient,
                config=config,
            )
            correction = np.maximum(v3_correction, diagnostics["correction"])
            gradient = v1_gradient * correction[:, None]
            report = evaluate(
                truth_mag, truth_vec, wss_from_gradient(gradient, "carreau")
            )
            report["fit_coverage"] = float(
                np.mean(np.isfinite(diagnostics["raw_scalar_gradient"]))
            )
            report["applied_fraction"] = float(np.mean(diagnostics["applied"] > 0))
            methods[name] = report
        rows.append(
            {
                "canonical_id": case["canonical_id"],
                "cohort": case["cohort"],
                "methods": methods,
            }
        )
        candidate_names = (*VARIANTS, *COLUMN_VARIANTS)
        best = max(candidate_names, key=lambda name: methods[name]["raw_r2"])
        print(
            f"[{number:03d}/{len(cases):03d}] {case['canonical_id']} "
            f"v3={methods['v3']['raw_r2']:.4f} "
            f"base={methods['base']['raw_r2']:.4f} "
            f"best={best}:{methods[best]['raw_r2']:.4f}",
            flush=True,
        )

    aggregate = {}
    for name in ("v3", *VARIANTS, *COLUMN_VARIANTS):
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
        if name != "v3":
            delta = np.asarray(
                [
                    row["methods"][name]["raw_r2"]
                    - row["methods"]["base"]["raw_r2"]
                    for row in rows
                ]
            )
            aggregate[name]["vs_base"] = {
                "mean": float(delta.mean()),
                "wins": int(np.sum(delta > 0)),
                "min": float(delta.min()),
            }
    output = {
        "status": "train_only_fit_variant_screen",
        "sample_count": args.sample_count,
        "limit_per_cohort": args.limit_per_cohort,
        "variants": {
            **{name: config.__dict__ for name, config in variant_configs.items()},
            **{name: config.__dict__ for name, config in column_configs.items()},
        },
        "aggregate": aggregate,
        "rows": rows,
        "elapsed_seconds": time.time() - started,
    }
    path = Path(args.json_out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(aggregate, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
