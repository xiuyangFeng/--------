"""同一病例加载一次，成对比较 fixed-K 与纯点云 adaptive-CV velocity→WSS。

该脚本用于实验跟踪，不使用 CFD 网格拓扑，也不使用病例真值选择局部邻域。
所有方法共享同一病例、时间步、壁面抽样点、PCA 法向和 Carreau–Yasuda 模型。
"""

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
    fit_wall_gradient,
    pca_wall_normals,
    resolve_step,
    wss_from_gradient,
)
import data_loader_cfd as dl


def parse_int_list(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def parse_float_list(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def tolerance_label(value: float) -> str:
    return str(value).replace(".", "p")


def diagnostic_summary(used: np.ndarray, diagnostics: dict[str, np.ndarray]) -> dict:
    selected = diagnostics["selected_neighbors"]
    selected = selected[selected > 0]
    conditions = diagnostics["design_condition"]
    scores = diagnostics["cv_score"]
    values, counts = np.unique(selected, return_counts=True) if len(selected) else ([], [])
    return {
        "samples_used_p50": float(np.median(used[used > 0])) if (used > 0).any() else 0.0,
        "selected_neighbors_p05": float(np.quantile(selected, 0.05)) if len(selected) else 0.0,
        "selected_neighbors_p50": float(np.median(selected)) if len(selected) else 0.0,
        "selected_neighbors_p95": float(np.quantile(selected, 0.95)) if len(selected) else 0.0,
        "fit_cv_score_p50": float(np.nanmedian(scores)),
        "fit_condition_p95": float(np.nanquantile(conditions, 0.95)),
        "selected_neighbors_histogram": {
            str(int(value)): int(count) for value, count in zip(values, counts)
        },
    }


def build_methods(
    fixed_neighbors: tuple[int, ...],
    cv_tolerances: tuple[float, ...],
    adaptive_neighbors: tuple[int, ...],
    calibrated_scale: float,
) -> list[dict]:
    methods = [
        {
            "name": f"fixed_k{neighbors}",
            "neighbor_mode": "fixed",
            "neighbors": neighbors,
            "adaptive_neighbors": adaptive_neighbors,
            "cv_tolerance": 1.0,
            "prediction_scale": 1.0,
        }
        for neighbors in fixed_neighbors
    ]
    methods.extend(
        {
            "name": f"adaptive_cv_tol{tolerance_label(tolerance)}",
            "neighbor_mode": "adaptive_cv",
            "neighbors": max(adaptive_neighbors),
            "adaptive_neighbors": adaptive_neighbors,
            "cv_tolerance": tolerance,
            "prediction_scale": 1.0,
        }
        for tolerance in cv_tolerances
    )
    if calibrated_scale > 0:
        if not cv_tolerances:
            raise ValueError("calibrated_scale requires at least one cv_tolerance")
        tolerance = cv_tolerances[0]
        methods.append(
            {
                "name": (
                    f"adaptive_cv_tol{tolerance_label(tolerance)}_"
                    f"scale{tolerance_label(calibrated_scale)}"
                ),
                "neighbor_mode": "adaptive_cv",
                "neighbors": max(adaptive_neighbors),
                "adaptive_neighbors": adaptive_neighbors,
                "cv_tolerance": tolerance,
                "prediction_scale": calibrated_scale,
            }
        )
    if not methods:
        raise ValueError("at least one fixed-neighbor or cv-tolerance method is required")
    return methods


def aggregate(rows: list[dict], methods: list[dict]) -> dict:
    metric_names = (
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
    method_summary: dict[str, dict] = {}
    for method in methods:
        name = method["name"]
        valid = [row["methods"][name] for row in rows if row["truth_usable"]]
        method_summary[name] = {
            metric: summarize([report[metric] for report in valid]) for metric in metric_names
        }

    paired: dict[str, dict] = {}
    baseline = methods[0]["name"]
    for method in methods[1:]:
        name = method["name"]
        usable = [row for row in rows if row["truth_usable"]]
        delta = [
            row["methods"][name]["raw_r2"] - row["methods"][baseline]["raw_r2"]
            for row in usable
        ]
        paired[name] = {
            "baseline": baseline,
            "raw_r2_delta": summarize(delta),
            "raw_r2_win_fraction": float(np.mean(np.asarray(delta) > 0)) if delta else float("nan"),
        }
    return {"methods": method_summary, "paired_vs_first_method": paired}


def grouped_aggregate(rows: list[dict], methods: list[dict], group_key: str) -> dict:
    return {
        str(group): aggregate(
            [row for row in rows if row.get(group_key) == group], methods
        )
        for group in sorted({row.get(group_key) for row in rows})
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=str(DATA_ROOT))
    parser.add_argument("--split", default=str(DEFAULT_SPLIT))
    parser.add_argument("--roles", default="train,test")
    parser.add_argument("--limit-per-cohort", type=int, default=0)
    parser.add_argument("--sample-count", type=int, default=1200, help="0=每例全部壁面点")
    parser.add_argument("--fixed-neighbors", default="64")
    parser.add_argument("--cv-tolerances", default=str(DEFAULT_CV_TOLERANCE))
    parser.add_argument(
        "--adaptive-neighbors",
        default=",".join(str(value) for value in DEFAULT_ADAPTIVE_NEIGHBORS),
    )
    parser.add_argument(
        "--calibrated-scale",
        type=float,
        default=0.0,
        help=">0 时增加一个 train-only 全局幅值校准的 adaptive 方法",
    )
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-peak", action="store_true")
    parser.add_argument("--json-out", required=True)
    args = parser.parse_args()

    fixed_neighbors = parse_int_list(args.fixed_neighbors)
    cv_tolerances = parse_float_list(args.cv_tolerances)
    adaptive_neighbors = parse_int_list(args.adaptive_neighbors)
    methods = build_methods(
        fixed_neighbors, cv_tolerances, adaptive_neighbors, args.calibrated_scale
    )

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

    print(f"cases={len(cases)} roles={sorted(roles)} sample_count={args.sample_count}")
    print("methods=" + ", ".join(method["name"] for method in methods))
    rows: list[dict] = []
    failures: list[dict] = []
    started = time.time()

    for case_index, case in enumerate(cases, start=1):
        case_started = time.time()
        try:
            case_dir = Path(case["case_dir"])
            step, step_source = resolve_step(case_dir, None, prefer_peak=not args.no_peak)
            wall = dl.load_wall(dl.step_path(case_dir, step, "ascii"))
            interior = dl.load_interior(dl.step_path(case_dir, step, "ascii_in"))
            warnings = check_truth_quality(wall, interior)
            tree = cKDTree(interior["coords_mm"])
            normals = pca_wall_normals(wall["coords_mm"], interior["coords_mm"], tree)

            n_wall = len(wall["coords_mm"])
            if 0 < args.sample_count < n_wall:
                rng = np.random.default_rng(args.seed)
                sample_index = np.sort(rng.choice(n_wall, args.sample_count, replace=False))
            else:
                sample_index = np.arange(n_wall)

            truth_mag = wall["wss_mag"][sample_index]
            truth_vec = wall["wss_vec"][sample_index]
            reports: dict[str, dict] = {}
            for method in methods:
                gradient, used, diagnostics = fit_wall_gradient(
                    wall["coords_mm"][sample_index],
                    normals[sample_index],
                    interior["coords_mm"],
                    interior["velocity"],
                    tree,
                    neighbors=method["neighbors"],
                    degree=args.degree,
                    bandwidth_weight=False,
                    neighbor_mode=method["neighbor_mode"],
                    adaptive_neighbors=method["adaptive_neighbors"],
                    cv_tolerance=method["cv_tolerance"],
                )
                pred = wss_from_gradient(gradient, "carreau") * method["prediction_scale"]
                report = evaluate(truth_mag, truth_vec, pred)
                report.update(diagnostic_summary(used, diagnostics))
                reports[method["name"]] = report

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
                "methods": reports,
                "elapsed_seconds": time.time() - case_started,
            }
            rows.append(row)
            metrics = " ".join(
                f"{name}={reports[name]['raw_r2']:.3f}" for name in reports
            )
            print(f"[{case_index:03d}/{len(cases):03d}] {case['canonical_id']} {metrics}")
        except Exception as error:
            failures.append(
                {"case": case["canonical_id"], "error": f"{type(error).__name__}: {error}"}
            )
            print(f"[{case_index:03d}/{len(cases):03d}] {case['canonical_id']} FAILED", file=sys.stderr)
            traceback.print_exc(limit=1)

    usable = [row for row in rows if row["truth_usable"]]
    payload = {
        "experiment": "pointcloud_adaptive_neighbor_comparison",
        "split": {
            "path": str(split_path),
            "sha256": split_sha,
            "expected_sha256_default": EXPECTED_SPLIT_SHA256,
            "roles": sorted(roles),
        },
        "config": {
            "sample_count": args.sample_count,
            "degree": args.degree,
            "viscosity": "carreau",
            "normals": "pca",
            "prefer_peak": not args.no_peak,
            "seed": args.seed,
            "methods": methods,
        },
        "n_requested": len(cases),
        "n_completed": len(rows),
        "n_usable": len(usable),
        "elapsed_seconds": time.time() - started,
        "aggregate": aggregate(rows, methods),
        "aggregate_by_role": grouped_aggregate(rows, methods, "role"),
        "aggregate_by_cohort": grouped_aggregate(rows, methods, "cohort"),
        "cases": rows,
        "failures": failures,
    }
    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload["aggregate"], indent=2, ensure_ascii=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
