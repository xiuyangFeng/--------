"""Train-only evaluation of closest-surface MLS as an add-on to frozen V3.

The V3 source/config/result remain immutable.  V4 combines two independent
positive magnitude estimates while preserving the frozen V1 direction:

    correction_v4 = max(correction_v3, correction_surface_mls)

No CFD WSS truth is used by either local estimator or by the fusion rule.
Truth is read only after prediction for paired evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from batch_validate_cfd import load_split_cases, summarize
from calculate_wss_cfd import (
    check_truth_quality,
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
    SurfaceMLSV4Config,
    fit_wall_gradient_surface_mls,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    os.replace(temporary, path)


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [row for row in rows if row["truth_usable"]]
    methods = ("adaptive_cv_v1", "normal_multiscale_v3", "surface_mls_v4")
    metrics = (
        "raw_r2",
        "scaled_r2",
        "spearman",
        "alpha",
        "mae_pa",
        "nrmse",
        "high_wss_nrmse",
        "direction_cosine_p50",
        "coverage",
    )
    result = {
        name: {
            metric: summarize([row["methods"][name][metric] for row in usable])
            for metric in metrics
        }
        for name in methods
    }
    delta = np.asarray(
        [
            row["methods"]["surface_mls_v4"]["raw_r2"]
            - row["methods"]["normal_multiscale_v3"]["raw_r2"]
            for row in usable
        ],
        dtype=np.float64,
    )
    result["v4_vs_v3"] = {
        "raw_r2_delta": summarize(delta.tolist()),
        "wins": int(np.sum(delta > 0)),
        "losses": int(np.sum(delta < 0)),
        "n": int(len(delta)),
    }
    by_cohort: dict[str, Any] = {}
    for cohort in sorted({row["cohort"] for row in usable}):
        cohort_delta = np.asarray(
            [
                row["methods"]["surface_mls_v4"]["raw_r2"]
                - row["methods"]["normal_multiscale_v3"]["raw_r2"]
                for row in usable
                if row["cohort"] == cohort
            ],
            dtype=np.float64,
        )
        by_cohort[cohort] = {
            "raw_r2_delta": summarize(cohort_delta.tolist()),
            "wins": int(np.sum(cohort_delta > 0)),
            "losses": int(np.sum(cohort_delta < 0)),
            "n": int(len(cohort_delta)),
        }
    result["v4_vs_v3_by_cohort"] = by_cohort
    return result


def _surface_summary(diagnostics: dict[str, np.ndarray]) -> dict[str, float]:
    def quantile(key: str, q: float) -> float:
        value = np.asarray(diagnostics[key], dtype=np.float64)
        value = value[np.isfinite(value)]
        return float(np.quantile(value, q)) if len(value) else float("nan")

    return {
        "surface_fit_coverage": float(
            np.mean(np.isfinite(diagnostics["raw_scalar_gradient"]))
        ),
        "surface_applied_fraction": float(np.mean(diagnostics["applied"] > 0)),
        "surface_correction_p50": quantile("correction", 0.50),
        "surface_correction_p95": quantile("correction", 0.95),
        "surface_correction_max": quantile("correction", 1.00),
        "surface_cv_score_p50": quantile("cv_score", 0.50),
        "surface_condition_p95": quantile("condition", 0.95),
        "surface_samples_p50": quantile("selected_samples", 0.50),
        "surface_effective_samples_p50": quantile("effective_samples", 0.50),
        "surface_anchor_groups_p50": quantile("anchor_groups", 0.50),
        "surface_depth_limit_mm_p50": quantile("depth_limit_mm", 0.50),
        "surface_radius_mm_p50": quantile("surface_radius_mm", 0.50),
    }


def run(config_path: Path) -> dict[str, Any]:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    roles = {str(value) for value in payload["split"]["roles"]}
    evaluation_only = bool(payload.get("evaluation_only", False))
    if roles != {"train"} and not evaluation_only:
        raise ValueError(
            "unfrozen V4 experiments are restricted to train unless "
            "evaluation_only=true"
        )
    if "test" in roles and not evaluation_only:
        raise ValueError("test requires evaluation_only=true")
    split_path = Path(payload["split"]["path"]).resolve()
    expected_split_sha = str(payload["split"]["expected_sha256"])
    actual_split_sha = _sha256(split_path)
    if actual_split_sha != expected_split_sha:
        raise RuntimeError("split SHA256 mismatch")
    data_root = Path(payload["split"]["data_root"]).resolve()
    v1_path = Path(payload["frozen_v1_config"]).resolve()
    v3_path = Path(payload["frozen_v3_config"]).resolve()
    v1_payload = json.loads(v1_path.read_text(encoding="utf-8"))
    v3_payload = json.loads(v3_path.read_text(encoding="utf-8"))
    v3_config = NormalMultiscaleV3Config.from_mapping(v3_payload["method"])
    v4_mapping = dict(payload.get("method", {}))
    if "depth_fractions" in v4_mapping:
        v4_mapping["depth_fractions"] = tuple(v4_mapping["depth_fractions"])
    v4_config = SurfaceMLSV4Config(**v4_mapping)
    v4_config.validate()

    execution = dict(payload["execution"])
    cases = load_split_cases(split_path, data_root, roles)
    limit = int(execution.get("limit_per_cohort", 0))
    if limit > 0:
        kept: list[dict[str, Any]] = []
        counts: dict[str, int] = {}
        for case in cases:
            cohort = case["cohort"]
            if counts.get(cohort, 0) < limit:
                kept.append(case)
                counts[cohort] = counts.get(cohort, 0) + 1
        cases = kept
    sample_count = int(execution.get("sample_count", 600))
    seed = int(execution.get("seed", 0))
    prefer_peak = bool(execution.get("prefer_peak", True))
    output_path = Path(payload["output"]["json_path"]).resolve()
    point_cache_dir_value = payload["output"].get("point_cache_dir")
    point_cache_dir = (
        Path(point_cache_dir_value).resolve() if point_cache_dir_value else None
    )
    geometry = dict(payload.get("geometry", {}))
    radius_source = str(geometry.get("radius_source", "bundle_then_wall"))

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    started = time.time()

    def snapshot() -> dict[str, Any]:
        result = {
            "experiment": payload["experiment_name"],
            "status": payload.get("status", "train_candidate"),
            "method_frozen": False,
            "evaluation_only": evaluation_only,
            "split": {
                "path": str(split_path),
                "sha256": actual_split_sha,
                "roles": sorted(roles),
            },
            "config": {
                "path": str(config_path.resolve()),
                "sha256": _sha256(config_path),
                "frozen_v1_config": str(v1_path),
                "frozen_v1_sha256": _sha256(v1_path),
                "frozen_v3_config": str(v3_path),
                "frozen_v3_sha256": _sha256(v3_path),
                "surface_mls_v4": asdict(v4_config),
            },
            "execution": {
                **execution,
                "n_requested": len(cases),
                "n_completed": len(rows),
            },
            "aggregate": _aggregate(rows) if rows else {},
            "cases": rows,
            "failures": failures,
            "elapsed_seconds": time.time() - started,
        }
        return result

    print(
        f"experiment={payload['experiment_name']} train_cases={len(cases)} "
        f"sample_count={sample_count}",
        flush=True,
    )
    for case_number, case in enumerate(cases, start=1):
        case_started = time.time()
        try:
            case_dir = Path(case["case_dir"])
            step, step_source = resolve_step(case_dir, None, prefer_peak=prefer_peak)
            wall = dl.load_wall(dl.step_path(case_dir, step, "ascii"))
            interior = dl.load_interior(dl.step_path(case_dir, step, "ascii_in"))
            warnings = check_truth_quality(wall, interior)
            interior_tree = cKDTree(interior["coords_mm"])
            full_normals = pca_wall_normals(
                wall["coords_mm"], interior["coords_mm"], interior_tree
            )
            if 0 < sample_count < len(wall["coords_mm"]):
                sample_index = np.sort(
                    np.random.default_rng(seed).choice(
                        len(wall["coords_mm"]), sample_count, replace=False
                    )
                )
            else:
                sample_index = np.arange(len(wall["coords_mm"]))
            wall_sample = wall["coords_mm"][sample_index]
            normal_sample = full_normals[sample_index]
            local_radius, resolved_radius_source = _resolve_radius(
                radius_source,
                case_dir,
                wall_sample,
                normal_sample,
                wall["coords_mm"],
                full_normals,
                cKDTree(wall["coords_mm"]),
                geometry,
            )
            v1_gradient, _, _ = fit_wall_gradient(
                wall_sample,
                normal_sample,
                interior["coords_mm"],
                interior["velocity"],
                interior_tree,
                degree=int(v1_payload["degree"]),
                neighbor_mode="adaptive_cv",
                adaptive_neighbors=tuple(
                    int(value) for value in v1_payload["adaptive_neighbors"]
                ),
                cv_tolerance=float(v1_payload["cv_tolerance"]),
            )
            v3_gradient, _, v3_diagnostics = fit_wall_gradient_normal_multiscale(
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
            _, surface_diagnostics = fit_wall_gradient_surface_mls(
                wall_sample,
                normal_sample,
                wall["coords_mm"],
                full_normals,
                interior["coords_mm"],
                interior["velocity"],
                interior_tree,
                local_radius_mm=local_radius,
                fallback_gradient=v1_gradient,
                config=v4_config,
            )
            v1_norm = np.linalg.norm(v1_gradient, axis=1)
            v3_correction = np.linalg.norm(v3_gradient, axis=1) / np.maximum(
                v1_norm, 1e-12
            )
            combined_correction = np.maximum(
                v3_correction, surface_diagnostics["correction"]
            )
            v4_gradient = v1_gradient * combined_correction[:, None]

            if point_cache_dir is not None:
                cache_arrays: dict[str, np.ndarray] = {
                    "canonical_id": np.asarray(case["canonical_id"]),
                    "cohort": np.asarray(case["cohort"]),
                    "step": np.asarray(step, dtype=np.int32),
                    "sample_index": sample_index.astype(np.int64),
                    "wall_mm": np.asarray(wall_sample, dtype=np.float32),
                    "normal": np.asarray(normal_sample, dtype=np.float32),
                    "truth_mag": np.asarray(
                        wall["wss_mag"][sample_index], dtype=np.float32
                    ),
                    "truth_vec": np.asarray(
                        wall["wss_vec"][sample_index], dtype=np.float32
                    ),
                    "local_radius_mm": np.asarray(local_radius, dtype=np.float32),
                    "gradient_v1": np.asarray(v1_gradient, dtype=np.float32),
                    "gradient_v3": np.asarray(v3_gradient, dtype=np.float32),
                    "gradient_v4": np.asarray(v4_gradient, dtype=np.float32),
                    "v3_correction": np.asarray(v3_correction, dtype=np.float32),
                    "combined_correction": np.asarray(
                        combined_correction, dtype=np.float32
                    ),
                }
                for name, values in v3_diagnostics.items():
                    cache_arrays[f"v3__{name}"] = np.asarray(values)
                for name, values in surface_diagnostics.items():
                    cache_arrays[f"surface__{name}"] = np.asarray(values)
                cache_path = point_cache_dir / (
                    case["canonical_id"].replace("/", "__") + ".npz"
                )
                _atomic_npz(cache_path, cache_arrays)

            truth_mag = wall["wss_mag"][sample_index]
            truth_vec = wall["wss_vec"][sample_index]
            methods = {
                "adaptive_cv_v1": evaluate(
                    truth_mag, truth_vec, wss_from_gradient(v1_gradient, "carreau")
                ),
                "normal_multiscale_v3": evaluate(
                    truth_mag, truth_vec, wss_from_gradient(v3_gradient, "carreau")
                ),
                "surface_mls_v4": evaluate(
                    truth_mag, truth_vec, wss_from_gradient(v4_gradient, "carreau")
                ),
            }
            methods["surface_mls_v4"].update(_surface_summary(surface_diagnostics))
            methods["surface_mls_v4"].update(
                {
                    "v3_correction_p50": float(np.nanmedian(v3_correction)),
                    "combined_correction_p50": float(
                        np.nanmedian(combined_correction)
                    ),
                    "combined_correction_p95": float(
                        np.nanquantile(combined_correction, 0.95)
                    ),
                    "combined_correction_max": float(
                        np.nanmax(combined_correction)
                    ),
                }
            )
            truth_normal_cosine = np.abs(
                np.einsum("ij,ij->i", truth_vec, normal_sample)
            ) / np.maximum(np.linalg.norm(truth_vec, axis=1), 1e-12)
            rows.append(
                {
                    "canonical_id": case["canonical_id"],
                    "cohort": case["cohort"],
                    "role": case["role"],
                    "step": int(step),
                    "step_source": step_source,
                    "n_wall_total": int(len(wall["coords_mm"])),
                    "n_interior": int(len(interior["coords_mm"])),
                    "n_sampled": int(len(sample_index)),
                    "radius_source": resolved_radius_source,
                    "truth_usable": not warnings,
                    "data_warnings": warnings,
                    "truth_normal_abs_cosine_p50": float(
                        np.nanmedian(truth_normal_cosine)
                    ),
                    "truth_normal_abs_cosine_p95": float(
                        np.nanquantile(truth_normal_cosine, 0.95)
                    ),
                    "methods": methods,
                    "elapsed_seconds": time.time() - case_started,
                }
            )
            delta = (
                methods["surface_mls_v4"]["raw_r2"]
                - methods["normal_multiscale_v3"]["raw_r2"]
            )
            print(
                f"[{case_number:03d}/{len(cases):03d}] {case['canonical_id']} "
                f"v3={methods['normal_multiscale_v3']['raw_r2']:.4f} "
                f"v4={methods['surface_mls_v4']['raw_r2']:.4f} "
                f"delta={delta:+.4f}",
                flush=True,
            )
        except Exception as error:
            failures.append(
                {
                    "canonical_id": case["canonical_id"],
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            traceback.print_exc(limit=2)
        _atomic_json(output_path, snapshot())
    result = snapshot()
    _atomic_json(output_path, result)
    print(json.dumps(result["aggregate"], indent=2, ensure_ascii=False), flush=True)
    print(f"wrote {output_path}", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(Path(args.config).resolve())


if __name__ == "__main__":
    main()
