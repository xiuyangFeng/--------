"""Configuration-driven paired validation for point-cloud WSS V3.

This entry point keeps the frozen V1/V2 implementations as read-only
comparators, loads each case once, and evaluates the geometry-aware V3 method
on the same wall points.  Hyperparameter selection must use train roles only;
the configuration validator refuses test roles unless a method is explicitly
marked frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping

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
from wss_multiscale import fit_wall_gradient_multiscale
from wss_normal_multiscale_v3 import (
    NormalMultiscaleV3Config,
    estimate_local_radius_from_wall,
    fit_wall_gradient_normal_multiscale,
)


ROOT = Path("/public/newhome/cy/Digital_twin/GNN")
DEFAULT_SPLIT_SHA256 = (
    "964d7021f2d12baadd630e7b936456a4e62294fa72c5ada4fc70abe4d9361f2b"
)
SUPPORTED_COMPARATORS = {"adaptive_cv_v1", "multiscale_v2", "normal_multiscale_v3"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _known_keys(mapping: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ValueError(f"unknown {label} keys: {unknown}")


def _path(value: str, base: Path) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else (base / candidate).resolve()


def load_experiment_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _known_keys(
        payload,
        {
            "schema_version",
            "experiment_name",
            "status",
            "description",
            "method_frozen",
            "frozen_method",
            "split",
            "execution",
            "geometry",
            "comparators",
            "method",
            "output",
            "runtime",
        },
        "top-level",
    )
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError("schema_version must be 1")
    name = str(payload.get("experiment_name", ""))
    if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]*", name):
        raise ValueError("experiment_name must be lowercase filesystem-safe text")
    split = dict(payload.get("split", {}))
    execution = dict(payload.get("execution", {}))
    geometry = dict(payload.get("geometry", {}))
    output = dict(payload.get("output", {}))
    runtime = dict(payload.get("runtime", {}))
    _known_keys(
        split,
        {"path", "expected_sha256", "data_root", "roles"},
        "split",
    )
    _known_keys(
        execution,
        {
            "sample_count",
            "seed",
            "prefer_peak",
            "limit_per_cohort",
            "num_shards",
            "shard_index",
        },
        "execution",
    )
    _known_keys(
        geometry,
        {
            "radius_source",
            "wall_radius_search_neighbors",
            "wall_radius_opposite_normal_cosine",
        },
        "geometry",
    )
    _known_keys(output, {"json_path"}, "output")
    _known_keys(
        runtime,
        {
            "host",
            "conda_python",
            "gpu",
            "max_gpu_memory_used_mb_before_launch",
            "poll_seconds",
        },
        "runtime",
    )
    required_split = {"path", "data_root", "roles"}
    if not required_split.issubset(split):
        raise ValueError(f"split missing keys: {sorted(required_split - set(split))}")
    roles = {str(item) for item in split["roles"]}
    if not roles or not roles <= {"train", "test", "val"}:
        raise ValueError("split.roles must contain train/test/val")
    if "test" in roles and not bool(payload.get("method_frozen", False)):
        raise ValueError("test role is forbidden until method_frozen=true")
    if bool(payload.get("method_frozen", False)):
        frozen_method = dict(payload.get("frozen_method", {}))
        _known_keys(
            frozen_method, {"path", "expected_sha256"}, "frozen_method"
        )
        if not {"path", "expected_sha256"} <= set(frozen_method):
            raise ValueError("method_frozen=true requires frozen_method path and SHA256")
        frozen_path = _path(str(frozen_method["path"]), path.parent)
        actual_frozen_sha = sha256_file(frozen_path)
        if actual_frozen_sha != str(frozen_method["expected_sha256"]):
            raise ValueError("frozen method SHA256 mismatch")
        frozen_payload = json.loads(frozen_path.read_text(encoding="utf-8"))
        if frozen_payload.get("method") != payload.get("method"):
            raise ValueError("experiment method differs from frozen method")
    sample_count = int(execution.get("sample_count", 1200))
    if sample_count < 0:
        raise ValueError("execution.sample_count must be >= 0")
    num_shards = int(execution.get("num_shards", 1))
    shard_index = int(execution.get("shard_index", 0))
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("invalid shard configuration")
    radius_source = str(geometry.get("radius_source", "bundle_then_wall"))
    if radius_source not in {"bundle", "wall", "bundle_then_wall", "fixed"}:
        raise ValueError("unsupported geometry.radius_source")
    comparators = payload.get("comparators", [])
    if not isinstance(comparators, list) or not comparators:
        raise ValueError("comparators must be a non-empty list")
    seen: set[str] = set()
    for comparator in comparators:
        _known_keys(comparator, {"name", "config"}, "comparator")
        method_name = str(comparator.get("name", ""))
        if method_name not in SUPPORTED_COMPARATORS:
            raise ValueError(f"unsupported comparator: {method_name}")
        if method_name in seen:
            raise ValueError(f"duplicate comparator: {method_name}")
        seen.add(method_name)
        if method_name != "normal_multiscale_v3" and "config" not in comparator:
            raise ValueError(f"{method_name} requires a frozen config path")
    if "normal_multiscale_v3" not in seen:
        raise ValueError("comparators must include normal_multiscale_v3")
    method_config = NormalMultiscaleV3Config.from_mapping(payload.get("method", {}))
    comparator_names = [str(item["name"]) for item in comparators]
    if method_config.hybrid_enabled:
        if "adaptive_cv_v1" not in comparator_names:
            raise ValueError("hybrid V3 requires adaptive_cv_v1 comparator")
        if comparator_names.index("adaptive_cv_v1") > comparator_names.index(
            "normal_multiscale_v3"
        ):
            raise ValueError("adaptive_cv_v1 must precede hybrid V3")
    if not output.get("json_path"):
        raise ValueError("output.json_path is required")
    if runtime:
        if runtime.get("host") != "node04":
            raise ValueError("V3 runtime.host must be node04")
        if not str(runtime.get("conda_python", "")).startswith("/public/"):
            raise ValueError("runtime.conda_python must be an absolute /public path")
        gpu = runtime.get("gpu", "auto")
        if gpu not in {"auto", 0, 1, "0", "1"}:
            raise ValueError("runtime.gpu must be auto, 0, or 1")
    return payload


def _load_bundle_radius(case_dir: Path, wall_mm: np.ndarray) -> np.ndarray | None:
    parts = case_dir.resolve().parts
    if "data_new" not in parts:
        return None
    index = parts.index("data_new")
    bundle = (
        Path(*parts[:index])
        / "data_wss_min"
        / Path(*parts[index + 1 :])
        / "bundle.npz"
    )
    if not bundle.is_file():
        return None
    with np.load(bundle, allow_pickle=False) as source:
        if not {"wall_coords_raw", "wall_local_radius"} <= set(source.files):
            return None
        raw = np.asarray(source["wall_coords_raw"], dtype=np.float64)
        radius = np.asarray(source["wall_local_radius"], dtype=np.float64)
    if np.nanmedian(np.linalg.norm(raw, axis=1)) < 20:
        raw = raw * 1e3
    _, nearest = cKDTree(raw).query(wall_mm, k=1)
    return radius[nearest]


def _resolve_radius(
    source: str,
    case_dir: Path,
    sample_wall_mm: np.ndarray,
    sample_normals: np.ndarray,
    full_wall_mm: np.ndarray,
    full_normals: np.ndarray,
    wall_tree: cKDTree,
    geometry: Mapping[str, Any],
) -> tuple[np.ndarray | None, str]:
    if source in {"bundle", "bundle_then_wall"}:
        radius = _load_bundle_radius(case_dir, sample_wall_mm)
        if radius is not None:
            return radius, "bundle_centerline_radius"
        if source == "bundle":
            raise FileNotFoundError(f"bundle local radius unavailable: {case_dir}")
    if source in {"wall", "bundle_then_wall"}:
        radius = estimate_local_radius_from_wall(
            sample_wall_mm,
            sample_normals,
            full_wall_mm,
            full_normals,
            wall_tree,
            search_neighbors=int(geometry.get("wall_radius_search_neighbors", 512)),
            opposite_normal_cosine=float(
                geometry.get("wall_radius_opposite_normal_cosine", -0.6)
            ),
        )
        return radius, "wall_opposite_distance"
    return None, "fixed_fallback_radius"


def _diagnostic_summary(diagnostics: Mapping[str, np.ndarray]) -> dict[str, float]:
    def quantile(key: str, q: float) -> float:
        values = np.asarray(diagnostics[key], dtype=np.float64)
        finite = values[np.isfinite(values)]
        return float(np.quantile(finite, q)) if len(finite) else float("nan")

    used = np.asarray(diagnostics["selected_samples"], dtype=np.float64)
    return {
        "selected_samples_p50": quantile("selected_samples", 0.50),
        "selected_scale_fraction_p50": quantile(
            "selected_scale_fraction", 0.50
        ),
        "depth_limit_mm_p50": quantile("depth_limit_mm", 0.50),
        "depth_limit_mm_p95": quantile("depth_limit_mm", 0.95),
        "local_reynolds_p50": quantile("local_reynolds", 0.50),
        "edge_velocity_m_s_p50": quantile("edge_velocity_m_s", 0.50),
        "eta_p50_mm": quantile("eta_p50_mm", 0.50),
        "eta_max_mm_p95": quantile("eta_max_mm", 0.95),
        "lateral_p50_mm": quantile("lateral_p50_mm", 0.50),
        "lateral_p95_mm": quantile("lateral_p95_mm", 0.50),
        "lateral_over_eta_p50": float(
            quantile("lateral_p50_mm", 0.50)
            / max(quantile("eta_p50_mm", 0.50), 1e-12)
        ),
        "ownership_rejected_mean": float(
            np.mean(diagnostics["ownership_rejected"])
        ),
        "tube_rejected_mean": float(np.mean(diagnostics["tube_rejected"])),
        "fallback_tube_expanded_fraction": float(
            np.mean(diagnostics["fallback_tube_expanded"] > 0)
        ),
        "multiscale_correction_p50": quantile("correction", 0.50),
        "ray_fit_coverage": float(np.mean(used > 0)),
        "ray_used_fraction": float(np.mean(diagnostics["ray_used"] > 0)),
        "ray_to_fallback_norm_p50": quantile(
            "ray_to_fallback_norm", 0.50
        ),
        "applied_correction_p50": quantile("applied_correction", 0.50),
        "applied_correction_p95": quantile("applied_correction", 0.95),
        "applied_correction_max": quantile("applied_correction", 1.00),
    }


def _load_frozen_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _method_report(
    name: str,
    comparator: Mapping[str, Any],
    config_base: Path,
    wall_sample: np.ndarray,
    normal_sample: np.ndarray,
    full_wall: np.ndarray,
    full_normals: np.ndarray,
    interior: dict[str, np.ndarray],
    interior_tree: cKDTree,
    local_radius: np.ndarray | None,
    v3_config: NormalMultiscaleV3Config,
    truth_mag: np.ndarray,
    truth_vec: np.ndarray,
    fallback_gradient: np.ndarray | None,
) -> tuple[dict[str, Any], np.ndarray]:
    if name == "adaptive_cv_v1":
        frozen = _load_frozen_config(_path(str(comparator["config"]), config_base))
        gradient, _, diagnostics = fit_wall_gradient(
            wall_sample,
            normal_sample,
            interior["coords_mm"],
            interior["velocity"],
            interior_tree,
            degree=int(frozen["degree"]),
            neighbor_mode="adaptive_cv",
            adaptive_neighbors=tuple(int(v) for v in frozen["adaptive_neighbors"]),
            cv_tolerance=float(frozen["cv_tolerance"]),
        )
        report = evaluate(truth_mag, truth_vec, wss_from_gradient(gradient, "carreau"))
        selected = diagnostics["selected_neighbors"]
        selected = selected[selected > 0]
        report.update(
            {
                "selected_neighbors_p50": float(
                    np.median(selected)
                ) if len(selected) else 0.0,
                "frozen_config": str(_path(str(comparator["config"]), config_base)),
            }
        )
        return report, gradient
    if name == "multiscale_v2":
        frozen = _load_frozen_config(_path(str(comparator["config"]), config_base))
        gradient, _, diagnostics = fit_wall_gradient_multiscale(
            wall_sample,
            normal_sample,
            interior["coords_mm"],
            interior["velocity"],
            interior_tree,
            adaptive_neighbors=tuple(int(v) for v in frozen["adaptive_neighbors"]),
            degree=int(frozen["degree"]),
            base_cv_tolerance=float(frozen["base_cv_tolerance"]),
            extrapolation_cv_tolerance=float(
                frozen["extrapolation_cv_tolerance"]
            ),
            extrapolation_power=float(frozen["extrapolation_power"]),
            smooth_neighbors=int(frozen["smooth_neighbors"]),
            global_blend=float(frozen["global_blend"]),
            correction_min=float(frozen["correction_min"]),
            correction_max=float(frozen["correction_max"]),
            correction_shrink=float(frozen["correction_shrink"]),
            max_trend_correlation=float(frozen["max_trend_correlation"]),
            min_direction_cosine=float(frozen["min_direction_cosine"]),
        )
        report = evaluate(truth_mag, truth_vec, wss_from_gradient(gradient, "carreau"))
        report.update(
            {
                "multiscale_correction_p50": float(
                    np.nanmedian(diagnostics["correction"])
                ),
                "frozen_config": str(_path(str(comparator["config"]), config_base)),
            }
        )
        return report, gradient
    gradient, _, diagnostics = fit_wall_gradient_normal_multiscale(
        wall_sample,
        normal_sample,
        full_wall,
        full_normals,
        interior["coords_mm"],
        interior["velocity"],
        interior_tree,
        local_radius_mm=local_radius,
        config=v3_config,
        fallback_gradient=fallback_gradient,
    )
    report = evaluate(truth_mag, truth_vec, wss_from_gradient(gradient, "carreau"))
    report.update(_diagnostic_summary(diagnostics))
    return report, gradient


def aggregate(rows: list[dict[str, Any]], method_names: list[str]) -> dict[str, Any]:
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
    usable = [row for row in rows if row["truth_usable"]]
    result: dict[str, Any] = {}
    for name in method_names:
        result[name] = {
            metric: summarize([row["methods"][name][metric] for row in usable])
            for metric in metrics
        }
    if "normal_multiscale_v3" in method_names:
        paired: dict[str, Any] = {}
        for baseline in method_names:
            if baseline == "normal_multiscale_v3":
                continue
            delta = [
                row["methods"]["normal_multiscale_v3"]["raw_r2"]
                - row["methods"][baseline]["raw_r2"]
                for row in usable
            ]
            paired[baseline] = {
                "raw_r2_delta": summarize(delta),
                "wins": int(np.sum(np.asarray(delta) > 0)),
                "n": len(delta),
            }
        result["v3_paired"] = paired
    return result


def run(config_path: Path, *, validate_only: bool = False) -> dict[str, Any]:
    payload = load_experiment_config(config_path)
    base = config_path.parent
    split_config = payload["split"]
    split_path = _path(str(split_config["path"]), base)
    data_root = _path(str(split_config["data_root"]), base)
    expected_sha = str(split_config.get("expected_sha256", DEFAULT_SPLIT_SHA256))
    actual_sha = sha256_file(split_path)
    if actual_sha != expected_sha:
        raise RuntimeError(
            f"split SHA mismatch: got {actual_sha}, expected {expected_sha}"
        )
    config_sha = sha256_file(config_path)
    v3_config = NormalMultiscaleV3Config.from_mapping(payload["method"])
    output_path = _path(str(payload["output"]["json_path"]), base)
    resolved = {
        "experiment_name": payload["experiment_name"],
        "config_path": str(config_path.resolve()),
        "config_sha256": config_sha,
        "split_path": str(split_path),
        "split_sha256": actual_sha,
        "data_root": str(data_root),
        "output_path": str(output_path),
        "method": v3_config.to_dict(),
        "runtime": payload.get("runtime", {}),
    }
    if validate_only:
        print(json.dumps(resolved, indent=2, ensure_ascii=False))
        return {"validation": resolved}

    roles = {str(role) for role in split_config["roles"]}
    cases = load_split_cases(split_path, data_root, roles)
    execution = payload.get("execution", {})
    limit = int(execution.get("limit_per_cohort", 0))
    if limit > 0:
        kept: list[dict[str, Any]] = []
        counts: dict[str, int] = {}
        for case in cases:
            if counts.get(case["cohort"], 0) < limit:
                kept.append(case)
                counts[case["cohort"]] = counts.get(case["cohort"], 0) + 1
        cases = kept
    num_shards = int(execution.get("num_shards", 1))
    shard_index = int(execution.get("shard_index", 0))
    cases = [case for index, case in enumerate(cases) if index % num_shards == shard_index]
    sample_count = int(execution.get("sample_count", 1200))
    seed = int(execution.get("seed", 0))
    prefer_peak = bool(execution.get("prefer_peak", True))
    comparators = payload["comparators"]
    method_names = [str(item["name"]) for item in comparators]
    geometry = payload.get("geometry", {})
    radius_source = str(geometry.get("radius_source", "bundle_then_wall"))
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    started = time.time()
    print(
        f"experiment={payload['experiment_name']} cases={len(cases)} "
        f"roles={sorted(roles)} sample_count={sample_count} methods={method_names}",
        flush=True,
    )
    for case_number, case in enumerate(cases, start=1):
        case_started = time.time()
        print(
            f"[{case_number:03d}/{len(cases):03d}] START {case['canonical_id']}",
            flush=True,
        )
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
            n_wall = len(wall["coords_mm"])
            if 0 < sample_count < n_wall:
                rng = np.random.default_rng(seed)
                sample_index = np.sort(
                    rng.choice(n_wall, sample_count, replace=False)
                )
            else:
                sample_index = np.arange(n_wall)
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
            truth_mag = wall["wss_mag"][sample_index]
            truth_vec = wall["wss_vec"][sample_index]
            methods: dict[str, Any] = {}
            method_gradients: dict[str, np.ndarray] = {}
            for comparator in comparators:
                name = str(comparator["name"])
                methods[name], method_gradients[name] = _method_report(
                    name,
                    comparator,
                    base,
                    wall_sample,
                    normal_sample,
                    wall["coords_mm"],
                    full_normals,
                    interior,
                    interior_tree,
                    local_radius,
                    v3_config,
                    truth_mag,
                    truth_vec,
                    method_gradients.get("adaptive_cv_v1"),
                )
            elapsed = time.time() - case_started
            row = {
                "canonical_id": case["canonical_id"],
                "cohort": case["cohort"],
                "role": case["role"],
                "step": int(step),
                "step_source": step_source,
                "n_wall_total": n_wall,
                "n_interior": len(interior["coords_mm"]),
                "n_sampled": len(sample_index),
                "radius_source": resolved_radius_source,
                "truth_usable": not warnings,
                "data_warnings": warnings,
                "methods": methods,
                "elapsed_seconds": elapsed,
            }
            rows.append(row)
            score_text = " ".join(
                f"{name}={methods[name]['raw_r2']:.3f}" for name in method_names
            )
            print(
                f"[{case_number:03d}/{len(cases):03d}] DONE  "
                f"{case['canonical_id']} {score_text} elapsed={elapsed:.1f}s",
                flush=True,
            )
        except Exception as error:
            failures.append(
                {
                    "canonical_id": case["canonical_id"],
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            print(
                f"[{case_number:03d}/{len(cases):03d}] FAILED "
                f"{case['canonical_id']}: {type(error).__name__}: {error}",
                file=sys.stderr,
                flush=True,
            )
            traceback.print_exc(limit=2)
    result = {
        "experiment": payload["experiment_name"],
        "status": payload.get("status", "unspecified"),
        "description": payload.get("description", ""),
        "method_frozen": bool(payload.get("method_frozen", False)),
        "config": resolved,
        "execution": {
            **execution,
            "roles": sorted(roles),
            "n_requested": len(cases),
        },
        "aggregate": aggregate(rows, method_names),
        "cases": rows,
        "failures": failures,
        "elapsed_seconds": time.time() - started,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"wrote {output_path}", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate schema, paths and frozen split hash without loading cases",
    )
    args = parser.parse_args()
    run(Path(args.config).resolve(), validate_only=args.validate_only)


if __name__ == "__main__":
    main()
