"""Paired validation of adaptive-CV v1 and multi-scale point-cloud WSS v2."""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from batch_validate_cfd import (
    DATA_ROOT,
    DEFAULT_SPLIT,
    EXPECTED_SPLIT_SHA256,
    load_split_cases,
    sha256_file,
    summarize,
)
from calculate_wss_cfd import (
    DEFAULT_ADAPTIVE_NEIGHBORS,
    DEFAULT_CV_TOLERANCE,
    check_truth_quality,
    evaluate,
    pca_wall_normals,
    resolve_step,
    wss_from_gradient,
)
import data_loader_cfd as dl
from wss_multiscale import fit_wall_gradient_multiscale


def parse_int_list(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def method_summary(rows: list[dict], name: str) -> dict:
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
    usable = [row["methods"][name] for row in rows if row["truth_usable"]]
    return {metric: summarize([report[metric] for report in usable]) for metric in metrics}


def aggregate(rows: list[dict]) -> dict:
    usable = [row for row in rows if row["truth_usable"]]
    if not usable:
        return {}
    method_names = list(usable[0]["methods"])
    payload = {name: method_summary(rows, name) for name in method_names}
    paired: dict[str, dict] = {}
    for name in method_names:
        if name == "adaptive_cv_v1":
            continue
        delta = np.asarray(
            [
                row["methods"][name]["raw_r2"]
                - row["methods"]["adaptive_cv_v1"]["raw_r2"]
                for row in usable
            ],
            dtype=np.float64,
        )
        paired[name] = {
            "raw_r2_delta": summarize(delta.tolist()),
            "raw_r2_win_fraction": float(np.mean(delta > 0)) if len(delta) else float("nan"),
            "raw_r2_wins": int(np.sum(delta > 0)),
            "n_pairs": int(len(delta)),
        }
    payload["paired_vs_adaptive_cv_v1"] = paired
    # Preserve the original key for existing v2 raw result readers.
    if "multiscale_v2" in paired:
        payload["paired"] = paired["multiscale_v2"]
    return payload


def correction_summary(diagnostics: dict) -> dict:
    correction = np.asarray(diagnostics["correction"], dtype=np.float64)
    raw = np.asarray(diagnostics["raw_correction"], dtype=np.float64)
    stable = np.asarray(diagnostics["stable_scale_count"], dtype=np.float64)
    return {
        "correction_p05": float(np.nanquantile(correction, 0.05)),
        "correction_p50": float(np.nanmedian(correction)),
        "correction_p95": float(np.nanquantile(correction, 0.95)),
        "raw_correction_p50": float(np.nanmedian(raw)),
        "case_correction": float(diagnostics["case_correction"]),
        "stable_scale_count_p50": float(np.nanmedian(stable)),
        "extrapolated_fraction": float(np.mean(np.abs(raw - 1.0) > 1e-12)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=str(DATA_ROOT))
    parser.add_argument("--split", default=str(DEFAULT_SPLIT))
    parser.add_argument("--roles", default="train,test")
    parser.add_argument("--limit-per-cohort", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--sample-count", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-peak", action="store_true")
    parser.add_argument(
        "--adaptive-neighbors",
        default=",".join(str(value) for value in DEFAULT_ADAPTIVE_NEIGHBORS),
    )
    parser.add_argument("--base-cv-tolerance", type=float, default=DEFAULT_CV_TOLERANCE)
    parser.add_argument("--extrapolation-cv-tolerance", type=float, default=1.5)
    parser.add_argument("--extrapolation-power", type=float, default=1.0)
    parser.add_argument("--smooth-neighbors", type=int, default=24)
    parser.add_argument("--global-blend", type=float, default=0.0)
    parser.add_argument("--correction-min", type=float, default=1.0)
    parser.add_argument("--correction-max", type=float, default=1.35)
    parser.add_argument("--correction-shrink", type=float, default=0.5)
    parser.add_argument("--max-trend-correlation", type=float, default=0.0)
    parser.add_argument("--min-direction-cosine", type=float, default=0.98)
    parser.add_argument("--calibrated-scale", type=float, default=0.0)
    parser.add_argument("--v1-calibrated-scale", type=float, default=0.0)
    parser.add_argument("--json-out", required=True)
    args = parser.parse_args()

    split_path = Path(args.split).resolve()
    split_sha = sha256_file(split_path)
    if split_path == DEFAULT_SPLIT.resolve() and split_sha != EXPECTED_SPLIT_SHA256:
        raise RuntimeError(
            f"default split SHA mismatch: got {split_sha}, expected {EXPECTED_SPLIT_SHA256}"
        )
    roles = {item.strip() for item in args.roles.split(",") if item.strip()}
    cases = load_split_cases(split_path, Path(args.data_root), roles)
    if args.limit_per_cohort > 0:
        kept: list[dict] = []
        counts: dict[str, int] = {}
        for case in cases:
            cohort = case["cohort"]
            if counts.get(cohort, 0) < args.limit_per_cohort:
                kept.append(case)
                counts[cohort] = counts.get(cohort, 0) + 1
        cases = kept
    if args.num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard_index must be in [0, num_shards)")
    cases = [
        case
        for index, case in enumerate(cases)
        if index % args.num_shards == args.shard_index
    ]

    adaptive_neighbors = parse_int_list(args.adaptive_neighbors)
    rows: list[dict] = []
    failures: list[dict] = []
    started = time.time()
    print(f"cases={len(cases)} roles={sorted(roles)} sample_count={args.sample_count}")

    for case_index, case in enumerate(cases, start=1):
        case_started = time.time()
        try:
            case_dir = Path(case["case_dir"])
            step, step_source = resolve_step(
                case_dir, None, prefer_peak=not args.no_peak
            )
            wall = dl.load_wall(dl.step_path(case_dir, step, "ascii"))
            interior = dl.load_interior(dl.step_path(case_dir, step, "ascii_in"))
            warnings = check_truth_quality(wall, interior)
            tree = cKDTree(interior["coords_mm"])
            normals = pca_wall_normals(
                wall["coords_mm"], interior["coords_mm"], tree
            )

            n_wall = len(wall["coords_mm"])
            if 0 < args.sample_count < n_wall:
                rng = np.random.default_rng(args.seed)
                sample_index = np.sort(
                    rng.choice(n_wall, args.sample_count, replace=False)
                )
            else:
                sample_index = np.arange(n_wall)
            wall_sample = wall["coords_mm"][sample_index]
            normal_sample = normals[sample_index]

            v2_gradient, _, v2_diagnostics = fit_wall_gradient_multiscale(
                wall_sample,
                normal_sample,
                interior["coords_mm"],
                interior["velocity"],
                tree,
                adaptive_neighbors=adaptive_neighbors,
                degree=2,
                base_cv_tolerance=args.base_cv_tolerance,
                extrapolation_cv_tolerance=args.extrapolation_cv_tolerance,
                extrapolation_power=args.extrapolation_power,
                smooth_neighbors=args.smooth_neighbors,
                global_blend=args.global_blend,
                correction_min=args.correction_min,
                correction_max=args.correction_max,
                correction_shrink=args.correction_shrink,
                max_trend_correlation=args.max_trend_correlation,
                min_direction_cosine=args.min_direction_cosine,
            )
            # The v2 diagnostics retain the exact adaptive-CV base gradient,
            # avoiding a second identical profile-fit pass during validation.
            base_gradient = np.asarray(
                v2_diagnostics["base_gradient"], dtype=np.float64
            )

            truth_mag = wall["wss_mag"][sample_index]
            truth_vec = wall["wss_vec"][sample_index]
            base_wss = wss_from_gradient(base_gradient, "carreau")
            base_report = evaluate(truth_mag, truth_vec, base_wss)
            v2_wss = wss_from_gradient(v2_gradient, "carreau")
            v2_report = evaluate(truth_mag, truth_vec, v2_wss)
            v2_report.update(correction_summary(v2_diagnostics))
            methods = {
                "adaptive_cv_v1": base_report,
                "multiscale_v2": v2_report,
            }
            if args.v1_calibrated_scale > 0:
                methods["adaptive_cv_v1_calibrated"] = evaluate(
                    truth_mag, truth_vec, base_wss * args.v1_calibrated_scale
                )
            if args.calibrated_scale > 0:
                calibrated_report = evaluate(
                    truth_mag, truth_vec, v2_wss * args.calibrated_scale
                )
                calibrated_report.update(correction_summary(v2_diagnostics))
                methods["multiscale_v2_calibrated"] = calibrated_report

            row = {
                "canonical_id": case["canonical_id"],
                "cohort": case["cohort"],
                "role": case["role"],
                "step": step,
                "step_source": step_source,
                "n_wall_total": n_wall,
                "n_interior": len(interior["coords_mm"]),
                "n_sampled": int(len(sample_index)),
                "truth_usable": not warnings,
                "data_warnings": warnings,
                "methods": methods,
                "elapsed_seconds": time.time() - case_started,
            }
            rows.append(row)
            calibrated_text = (
                f" calibrated={methods['multiscale_v2_calibrated']['raw_r2']:.3f}"
                if "multiscale_v2_calibrated" in methods
                else ""
            )
            print(
                f"[{case_index:03d}/{len(cases):03d}] {case['canonical_id']} "
                f"v1={base_report['raw_r2']:.3f} v2={v2_report['raw_r2']:.3f} "
                f"corr={v2_report['correction_p50']:.3f}{calibrated_text}"
            )
        except Exception as error:
            failures.append(
                {
                    "case": case["canonical_id"],
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            print(
                f"[{case_index:03d}/{len(cases):03d}] {case['canonical_id']} FAILED",
                file=sys.stderr,
            )
            traceback.print_exc(limit=1)

    payload = {
        "experiment": "pointcloud_multiscale_extrapolation_v2",
        "split": {
            "path": str(split_path),
            "sha256": split_sha,
            "roles": sorted(roles),
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
        },
        "config": {
            "sample_count": args.sample_count,
            "seed": args.seed,
            "prefer_peak": not args.no_peak,
            "adaptive_neighbors": list(adaptive_neighbors),
            "degree": 2,
            "base_cv_tolerance": args.base_cv_tolerance,
            "extrapolation_cv_tolerance": args.extrapolation_cv_tolerance,
            "extrapolation_power": args.extrapolation_power,
            "smooth_neighbors": args.smooth_neighbors,
            "global_blend": args.global_blend,
            "correction_min": args.correction_min,
            "correction_max": args.correction_max,
            "correction_shrink": args.correction_shrink,
            "max_trend_correlation": args.max_trend_correlation,
            "min_direction_cosine": args.min_direction_cosine,
            "calibrated_scale": args.calibrated_scale,
            "v1_calibrated_scale": args.v1_calibrated_scale,
            "viscosity": "carreau",
            "normals": "pca",
        },
        "n_requested": len(cases),
        "n_completed": len(rows),
        "n_usable": int(sum(row["truth_usable"] for row in rows)),
        "elapsed_seconds": time.time() - started,
        "aggregate": aggregate(rows),
        "aggregate_by_role": {
            role: aggregate([row for row in rows if row["role"] == role])
            for role in sorted({row["role"] for row in rows})
        },
        "aggregate_by_cohort": {
            cohort: aggregate([row for row in rows if row["cohort"] == cohort])
            for cohort in sorted({row["cohort"] for row in rows})
        },
        "cases": rows,
        "failures": failures,
    }
    output = Path(args.json_out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(payload["aggregate"], indent=2, ensure_ascii=False))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
