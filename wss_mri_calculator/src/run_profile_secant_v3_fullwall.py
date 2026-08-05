#!/usr/bin/env python3
"""Run frozen Profile-Secant V3 on every wall node of the frozen test35 split.

The script is a resumable orchestration layer.  It does not modify the frozen
V4 implementation or refit the calibrator.  It only:

1. balances the 35 test cases into independent full-wall shards;
2. runs the existing V4 cache builder with ``sample_count=0``;
3. enriches every cache with the depth-3 profile/secant diagnostics;
4. applies the frozen Profile-Secant V3 calibrator to the merged caches.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "wss_mri_calculator/src"
COMPAT_RUNNER = SRC / "run_frozen_wss_compat.py"
EXPERIMENT = ROOT / "wss_mri_calculator/experiments/pointcloud_surface_mls_v4"
DEFAULT_CONFIG = EXPERIMENT / "config_test35_cache.json"
DEFAULT_SAMPLE_RESULT = (
    ROOT
    / "outputs/wss_mri_calculator/pointcloud_surface_mls_v4/test35_base_s1200.json"
)
DEFAULT_MODEL = EXPERIMENT / "calibrator_profile_secant_high_tail_anchor10_v3.joblib"
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/wss_mri_calculator/pointcloud_surface_mls_v4/"
    "profile_secant_v3_fullwall"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def expected_cache_names(cases: list[str]) -> set[str]:
    return {case.replace("/", "__") + ".npz" for case in cases}


def cache_complete(directory: Path, cases: list[str]) -> bool:
    return expected_cache_names(cases) == {path.name for path in directory.glob("*.npz")}


def balanced_shards(cases: list[str], weights: dict[str, int], count: int) -> list[list[str]]:
    shards: list[list[str]] = [[] for _ in range(count)]
    totals = [0] * count
    for case in sorted(cases, key=lambda item: (-weights[item], item)):
        target = min(range(count), key=lambda index: (totals[index], index))
        shards[target].append(case)
        totals[target] += weights[case]
    return shards


def prepare(
    base_config_path: Path,
    sample_result_path: Path,
    output_root: Path,
    num_shards: int,
) -> list[dict[str, Any]]:
    base_config = json.loads(base_config_path.read_text(encoding="utf-8"))
    original_split_path = Path(base_config["split"]["path"]).resolve()
    original_split = json.loads(original_split_path.read_text(encoding="utf-8"))
    sample_result = json.loads(sample_result_path.read_text(encoding="utf-8"))
    weights = {
        row["canonical_id"]: int(row["n_wall_total"])
        for row in sample_result["cases"]
    }
    cases = [str(case) for case in original_split["test_cases"]]
    missing = sorted(set(cases) - set(weights))
    if missing:
        raise RuntimeError(f"sample result is missing wall counts for: {missing}")

    shards = balanced_shards(cases, weights, num_shards)
    run_dir = output_root / "inference/run"
    base_cache_root = output_root / "inference/base_cache_shards"
    final_cache = output_root / "inference/point_cache_test35_profile_secant_fullwall"
    final_cache.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    for index, shard_cases in enumerate(shards):
        shard_name = f"shard_{index:02d}"
        split_payload = copy.deepcopy(original_split)
        split_payload["test_cases"] = shard_cases
        split_payload["train_cases"] = []
        split_payload["val_cases"] = []
        split_payload["counts"] = {"test": len(shard_cases), "train": 0, "val": 0}
        split_payload["expected_counts"] = {
            "test": len(shard_cases),
            "train": 0,
            "val": 0,
        }
        split_payload["derived_fullwall_evaluation"] = {
            "source_split": str(original_split_path),
            "source_split_sha256": sha256(original_split_path),
            "shard_index": index,
            "num_shards": num_shards,
            "selection": "all frozen test35 cases, balanced by n_wall_total",
        }
        split_path = run_dir / f"{shard_name}_split.json"
        write_json(split_path, split_payload)

        base_cache = base_cache_root / shard_name
        config_payload = copy.deepcopy(base_config)
        config_payload["experiment_name"] = f"profile_secant_v3_fullwall_{shard_name}"
        config_payload["status"] = "frozen_profile_secant_v3_fullwall_evaluation"
        config_payload["split"]["path"] = str(split_path.resolve())
        config_payload["split"]["expected_sha256"] = sha256(split_path)
        config_payload["execution"]["sample_count"] = 0
        config_payload["execution"]["limit_per_cohort"] = 0
        config_payload["output"]["json_path"] = str(
            (run_dir / f"{shard_name}_base.json").resolve()
        )
        config_payload["output"]["point_cache_dir"] = str(base_cache.resolve())
        config_path = run_dir / f"{shard_name}_config.json"
        write_json(config_path, config_payload)

        records.append(
            {
                "index": index,
                "name": shard_name,
                "cases": shard_cases,
                "n_wall_expected": sum(weights[case] for case in shard_cases),
                "split": split_path,
                "config": config_path,
                "base_json": run_dir / f"{shard_name}_base.json",
                "base_cache": base_cache,
                "depth3_json": run_dir / f"{shard_name}_depth3_profile_secant.json",
                "log_base": run_dir / f"{shard_name}_base.log",
                "log_depth3": run_dir / f"{shard_name}_depth3.log",
                "final_cache": final_cache,
            }
        )

    manifest = {
        "status": "prepared",
        "source_config": str(base_config_path),
        "source_config_sha256": sha256(base_config_path),
        "source_split": str(original_split_path),
        "source_split_sha256": sha256(original_split_path),
        "sample_result_used_only_for_shard_balancing": str(sample_result_path),
        "sample_result_sha256": sha256(sample_result_path),
        "sample_count": 0,
        "n_cases": len(cases),
        "n_wall_expected": sum(weights.values()),
        "num_shards": num_shards,
        "shards": [
            {
                "name": record["name"],
                "cases": record["cases"],
                "n_wall_expected": record["n_wall_expected"],
                "config": str(record["config"]),
            }
            for record in records
        ],
    }
    write_json(run_dir / "prepared_manifest.json", manifest)
    return records


def run_command(command: list[str], log_path: Path, environment: dict[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        log.write("command: " + " ".join(command) + "\n")
        log.flush()
        completed = subprocess.run(
            command,
            cwd=SRC,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            env=environment,
            check=False,
        )
        log.write(f"\nreturncode={completed.returncode}\n")
        log.write(f"elapsed_seconds={time.time() - started:.3f}\n")
    if completed.returncode:
        raise RuntimeError(f"command failed ({completed.returncode}); see {log_path}")


def run_parallel(jobs: list[tuple[list[str], Path]], workers: int) -> None:
    environment = os.environ.copy()
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        environment[name] = "1"
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(run_command, command, log, environment): log
            for command, log in jobs
        }
        for future in as_completed(futures):
            log = futures[future]
            future.result()
            print(f"completed: {log}", flush=True)


def merge_stage_json(records: list[dict[str, Any]], key: str, output: Path) -> None:
    sources = [Path(record[key]) for record in records]
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in sources]
    if key == "base_json":
        rows = [row for payload in payloads for row in payload["cases"]]
        failures = [failure for payload in payloads for failure in payload["failures"]]
        merged = {
            "status": "fullwall_base_cache_shards_complete",
            "sample_count": 0,
            "n_cases": len(rows),
            "n_points": sum(int(row["n_sampled"]) for row in rows),
            "source_shards": [str(path) for path in sources],
            "cases": sorted(rows, key=lambda row: row["canonical_id"]),
            "failures": failures,
        }
    else:
        rows = [row for payload in payloads for row in payload["rows"]]
        merged = {
            "status": "fullwall_depth3_profile_secant_cache_shards_complete",
            "sample_count": 0,
            "n_cases": len(rows),
            "source_shards": [str(path) for path in sources],
            "rows": sorted(rows, key=lambda row: row["canonical_id"]),
        }
    write_json(output, merged)


def verify_fullwall_cache(records: list[dict[str, Any]], final_cache: Path) -> dict[str, Any]:
    expected_cases = [case for record in records for case in record["cases"]]
    if not cache_complete(final_cache, expected_cases):
        actual = {path.name for path in final_cache.glob("*.npz")}
        missing = sorted(expected_cache_names(expected_cases) - actual)
        extra = sorted(actual - expected_cache_names(expected_cases))
        raise RuntimeError(f"full-wall cache mismatch; missing={missing}, extra={extra}")

    import numpy as np

    case_rows = []
    required = {
        "gradient_v4_base",
        "combined_correction_base",
        "depth3__fit_count",
        "depth3__secant_gradient_p90",
        "depth3__near_cell_count",
    }
    for path in sorted(final_cache.glob("*.npz")):
        with np.load(path, allow_pickle=False) as source:
            missing_keys = sorted(required - set(source.files))
            if missing_keys:
                raise RuntimeError(f"{path} is missing keys: {missing_keys}")
            sample_index = np.asarray(source["sample_index"], dtype=np.int64)
            n_points = int(len(source["truth_mag"]))
            is_fullwall = np.array_equal(sample_index, np.arange(n_points))
            case_rows.append(
                {
                    "canonical_id": str(source["canonical_id"]),
                    "n_points": n_points,
                    "sample_index_is_full_arange": bool(is_fullwall),
                    "cache": str(path),
                    "sha256": sha256(path),
                }
            )
    if not all(row["sample_index_is_full_arange"] for row in case_rows):
        raise RuntimeError("at least one cache is not a full-wall arange cache")
    return {
        "n_cases": len(case_rows),
        "n_points": sum(row["n_points"] for row in case_rows),
        "all_sample_index_is_full_arange": True,
        "cases": case_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--sample-result", type=Path, default=DEFAULT_SAMPLE_RESULT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--stage",
        choices=("all", "prepare", "base", "depth3", "calibrate", "verify"),
        default="all",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.num_shards < 1 or args.workers < 1:
        raise ValueError("num-shards and workers must be positive")

    output_root = args.output_root.resolve()
    records = prepare(
        args.base_config.resolve(),
        args.sample_result.resolve(),
        output_root,
        args.num_shards,
    )
    if args.stage == "prepare":
        return

    if args.stage in {"all", "base"}:
        jobs = []
        for record in records:
            if (
                not args.force
                and Path(record["base_json"]).is_file()
                and cache_complete(Path(record["base_cache"]), record["cases"])
            ):
                continue
            jobs.append(
                (
                    [
                        sys.executable,
                        str(COMPAT_RUNNER),
                        str(SRC / "run_v4_experiment.py"),
                        "--config",
                        str(record["config"]),
                    ],
                    Path(record["log_base"]),
                )
            )
        if jobs:
            run_parallel(jobs, min(args.workers, len(jobs)))
        merge_stage_json(
            records,
            "base_json",
            output_root / "inference/test35_base_fullwall_merged.json",
        )
        if args.stage == "base":
            return

    final_cache = Path(records[0]["final_cache"])
    if args.stage in {"all", "depth3"}:
        jobs = []
        for record in records:
            expected = expected_cache_names(record["cases"])
            existing = {path.name for path in final_cache.glob("*.npz")}
            if not args.force and Path(record["depth3_json"]).is_file() and expected <= existing:
                continue
            jobs.append(
                (
                    [
                        sys.executable,
                        str(COMPAT_RUNNER),
                        str(SRC / "build_v4_depth3_fusion_cache.py"),
                        "--input-cache-dir",
                        str(record["base_cache"]),
                        "--output-cache-dir",
                        str(final_cache),
                        "--json-out",
                        str(record["depth3_json"]),
                    ],
                    Path(record["log_depth3"]),
                )
            )
        if jobs:
            run_parallel(jobs, min(args.workers, len(jobs)))
        merge_stage_json(
            records,
            "depth3_json",
            output_root / "inference/test35_depth3_profile_secant_fullwall_merged.json",
        )
        if args.stage == "depth3":
            return

    verification = verify_fullwall_cache(records, final_cache)
    verify_path = output_root / "inference/fullwall_cache_audit.json"
    write_json(verify_path, verification)
    if args.stage == "verify":
        print(json.dumps(verification, indent=2, ensure_ascii=False))
        return

    if args.stage in {"all", "calibrate"}:
        result = output_root / "inference/test35_profile_secant_v3_fullwall.json"
        predictions = (
            output_root / "inference/test35_profile_secant_v3_fullwall_predictions.npz"
        )
        log = output_root / "inference/run/calibrate.log"
        if args.force or not (result.is_file() and predictions.is_file()):
            environment = os.environ.copy()
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            ):
                environment[name] = "1"
            run_command(
                [
                    sys.executable,
                    str(COMPAT_RUNNER),
                    str(SRC / "apply_v4_high_tail_calibrator.py"),
                    "--cache-dir",
                    str(final_cache),
                    "--model",
                    str(args.model.resolve()),
                    "--json-out",
                    str(result),
                    "--predictions-out",
                    str(predictions),
                ],
                log,
                environment,
            )
        audit = {
            "status": "complete",
            "model": "Profile-Secant V3",
            "sample_count": 0,
            "model_path": str(args.model.resolve()),
            "model_sha256": sha256(args.model.resolve()),
            "result_json": str(result),
            "result_sha256": sha256(result),
            "predictions": str(predictions),
            "predictions_sha256": sha256(predictions),
            "cache_audit": str(verify_path),
            "n_cases": verification["n_cases"],
            "n_points": verification["n_points"],
        }
        write_json(output_root / "inference/inference_manifest.json", audit)
        print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
