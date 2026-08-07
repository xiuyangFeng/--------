"""Train-only case-blind atlas baseline for field-v4.

Build/predict/evaluate are deliberately separate interfaces.  The predictor
accepts only a frozen atlas asset and coordinate-only validation queries; only
the evaluator opens validation truth after predictions are on disk.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from wss_pinn.data.dataset import VolumeFieldDataset, _take_strict_volume
from wss_pinn.metrics import RegressionAccumulator
from wss_pinn.utils import (
    ROOT,
    atomic_write_json,
    git_state,
    guard_write_path,
    sha256_file,
    utc_now,
)


ROUTE = "volume_uvwp_peak_field_v4"
DATA_ROOT = ROOT / "data_wss_pinn/volume_uvwp_peak_field_v4_train123_val15"
DATA_MANIFEST = DATA_ROOT / "manifest.json"
FIELD_STATS = ROOT / "data_wss_pinn/volume_uvwp_peak_qs_smooth_v3_train123_val15_test35/field_stats.json"
QUERY_MANIFEST = DATA_ROOT / "validation_queries/manifest.json"
ATLAS_ROOT = DATA_ROOT / "b0_atlas"
ATLAS_NPZ = ATLAS_ROOT / "atlas.npz"
ATLAS_MANIFEST = ATLAS_ROOT / "manifest.json"
PREDICTION_ROOT = ROOT / "outputs/wss_pinn/volume_uvwp_peak_field_v4/b0/predictions"
PREDICTION_MANIFEST = PREDICTION_ROOT / "manifest.json"
EVALUATION_JSON = ROOT / "outputs/wss_pinn/volume_uvwp_peak_field_v4/b0/evaluation_val15.json"
EVALUATION_CSV = ROOT / "outputs/wss_pinn/volume_uvwp_peak_field_v4/b0/case_metrics_val15.csv"
SPLIT_SHA256 = "c80cb65ad95f7d76fadaff82ea02c35c27ad24ab2474e1050bf981eff97493f9"


def _case_seed(case_id: str, seed: int) -> int:
    value = int.from_bytes(hashlib.sha256(case_id.encode()).digest()[:8], "little")
    return (int(seed) + value) % (2**63 - 1)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    target = guard_write_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _normalized_fields(case, dataset: VolumeFieldDataset, indices: np.ndarray) -> np.ndarray:
    velocity = np.asarray(case.velocity[indices], dtype=np.float32)
    pressure = np.asarray(case.pressure[indices], dtype=np.float32)
    return np.column_stack(
        [
            (velocity - dataset.velocity_mean) / dataset.velocity_std,
            (pressure - dataset.pressure_mean) / dataset.pressure_std,
        ]
    ).astype(np.float32)


def _predict_neighbors(
    fields: np.ndarray,
    distances: np.ndarray,
    indices: np.ndarray,
    *,
    k: int,
    bandwidth: float,
) -> np.ndarray:
    d = np.asarray(distances[:, : int(k)], dtype=np.float64)
    idx = np.asarray(indices[:, : int(k)], dtype=np.int64)
    weights = np.exp(-np.square(d) / (2.0 * float(bandwidth) ** 2))
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-30)
    return np.einsum("nk,nkc->nc", weights, fields[idx], optimize=True)


def _internal_cv(
    dataset: VolumeFieldDataset,
    *,
    seed: int,
    folds: int,
    atlas_points_per_case: int,
    query_points_per_case: int,
    k_candidates: list[int],
    bandwidth_candidates: list[float],
) -> dict[str, Any]:
    ordered = sorted(
        dataset.cases,
        key=lambda case: hashlib.sha256(f"{seed}|{case.case_id}".encode()).hexdigest(),
    )
    scores = {
        (k, bandwidth): []
        for k in k_candidates
        for bandwidth in bandwidth_candidates
    }
    fold_reports = []
    for fold in range(int(folds)):
        held = [case for index, case in enumerate(ordered) if index % folds == fold]
        fit = [case for index, case in enumerate(ordered) if index % folds != fold]
        fit_coords = []
        fit_fields = []
        for case in fit:
            rng = np.random.default_rng(_case_seed(case.case_id, seed + fold * 1009))
            idx = _take_strict_volume(
                rng, case.is_wall, min(int(atlas_points_per_case), int((~case.is_wall).sum()))
            )
            fit_coords.append(np.asarray(case.coords[idx], dtype=np.float32))
            fit_fields.append(_normalized_fields(case, dataset, idx))
        coords = np.concatenate(fit_coords)
        fields = np.concatenate(fit_fields)
        tree = cKDTree(coords)
        fold_case_scores = {key: [] for key in scores}
        for case in held:
            rng = np.random.default_rng(_case_seed(case.case_id, seed + 50_000 + fold))
            query_idx = _take_strict_volume(
                rng,
                case.is_wall,
                min(int(query_points_per_case), int((~case.is_wall).sum())),
            )
            truth = _normalized_fields(case, dataset, query_idx)
            distances, neighbors = tree.query(
                np.asarray(case.coords[query_idx], dtype=np.float32),
                k=max(k_candidates),
                workers=-1,
            )
            for key in fold_case_scores:
                k, bandwidth = key
                prediction = _predict_neighbors(
                    fields, distances, neighbors, k=k, bandwidth=bandwidth
                )
                prediction[:, 3] -= float(np.mean(prediction[:, 3]))
                fold_case_scores[key].append(float(np.mean(np.square(prediction - truth))))
        for key, values in fold_case_scores.items():
            scores[key].extend(values)
        fold_reports.append(
            {
                "fold": fold,
                "fit_cases": [case.case_id for case in fit],
                "held_cases": [case.case_id for case in held],
            }
        )
    rows = [
        {
            "k": key[0],
            "bandwidth_normalized": key[1],
            "validation_field_score_cb": float(np.mean(values)),
            "case_count": len(values),
        }
        for key, values in scores.items()
    ]
    rows.sort(key=lambda row: (row["validation_field_score_cb"], row["k"], row["bandwidth_normalized"]))
    return {"candidates": rows, "selected": rows[0], "folds": fold_reports}


def build_atlas(
    *,
    seed: int = 1234,
    points_per_case: int = 4096,
    force: bool = False,
) -> dict[str, Any]:
    if ATLAS_MANIFEST.exists() and not force:
        raise FileExistsError(f"refusing to overwrite B0 atlas: {ATLAS_MANIFEST}")
    dataset = VolumeFieldDataset(
        DATA_MANIFEST,
        FIELD_STATS,
        roles=["train"],
        input_variant="xyz_geom",
        sampling={
            "support_points": 1,
            "query_points": 1,
            "physics_points": 1,
            "wall_points": 1,
            "query_mode": "independent",
        },
        seed=seed,
    )
    cv = _internal_cv(
        dataset,
        seed=seed,
        folds=5,
        atlas_points_per_case=1024,
        query_points_per_case=128,
        k_candidates=[16, 64, 256],
        bandwidth_candidates=[0.02, 0.05, 0.10],
    )
    coords = []
    fields = []
    case_index = []
    case_ids = []
    for index, case in enumerate(dataset.cases):
        rng = np.random.default_rng(_case_seed(case.case_id, seed))
        selected = _take_strict_volume(
            rng, case.is_wall, min(int(points_per_case), int((~case.is_wall).sum()))
        )
        coords.append(np.asarray(case.coords[selected], dtype=np.float32))
        fields.append(_normalized_fields(case, dataset, selected))
        case_index.append(np.full(len(selected), index, dtype=np.int16))
        case_ids.append(case.case_id)
    _atomic_npz(
        ATLAS_NPZ,
        coords=np.concatenate(coords),
        fields_standardized=np.concatenate(fields),
        train_case_index=np.concatenate(case_index),
    )
    manifest = {
        "schema_version": 1,
        "created_at": utc_now(),
        "status": "completed",
        "route": ROUTE,
        "baseline": "B0 train-only case-blind registered-coordinate Gaussian atlas",
        "seed": int(seed),
        "train_case_count": len(case_ids),
        "train_case_ids": case_ids,
        "sampling": {
            "points_per_case": int(points_per_case),
            "method": "uniform strict-volume without replacement",
        },
        "internal_case_cv": cv,
        "selected_k": int(cv["selected"]["k"]),
        "selected_bandwidth_normalized": float(cv["selected"]["bandwidth_normalized"]),
        "pressure_gauge": "prediction-only query mean centered per case",
        "split_sha256": SPLIT_SHA256,
        "field_stats": {"path": str(FIELD_STATS), "sha256": sha256_file(FIELD_STATS)},
        "source_manifest": {"path": str(DATA_MANIFEST), "sha256": sha256_file(DATA_MANIFEST)},
        "atlas": {"path": str(ATLAS_NPZ), "sha256": sha256_file(ATLAS_NPZ)},
        "git": git_state(),
    }
    atomic_write_json(ATLAS_MANIFEST, manifest)
    return manifest


def predict(*, force: bool = False, chunk_size: int = 5000) -> dict[str, Any]:
    if PREDICTION_MANIFEST.exists() and not force:
        raise FileExistsError(f"refusing to overwrite B0 predictions: {PREDICTION_MANIFEST}")
    atlas_manifest = json.loads(ATLAS_MANIFEST.read_text(encoding="utf-8"))
    query_manifest = json.loads(QUERY_MANIFEST.read_text(encoding="utf-8"))
    if (
        atlas_manifest.get("split_sha256") != SPLIT_SHA256
        or query_manifest.get("truth_access")
        != "none; coordinate files contain no u/v/w/p targets"
    ):
        raise ValueError("B0 predictor input contract drift")
    if sha256_file(ATLAS_NPZ) != atlas_manifest["atlas"]["sha256"]:
        raise ValueError("B0 atlas hash drift")
    stats = json.loads(FIELD_STATS.read_text(encoding="utf-8"))
    velocity_mean = np.asarray(stats["velocity_m_s"]["mean"], dtype=np.float64)
    velocity_std = np.asarray(stats["velocity_m_s"]["std"], dtype=np.float64)
    pressure_mean = float(stats["pressure_relative_pa"]["mean"][0])
    pressure_std = float(stats["pressure_relative_pa"]["std"][0])
    with np.load(ATLAS_NPZ, allow_pickle=False) as atlas:
        coords = np.asarray(atlas["coords"], dtype=np.float32)
        fields = np.asarray(atlas["fields_standardized"], dtype=np.float32)
    tree = cKDTree(coords)
    k = int(atlas_manifest["selected_k"])
    bandwidth = float(atlas_manifest["selected_bandwidth_normalized"])
    rows = []
    for query_row in query_manifest["cases"]:
        coords_path = Path(query_row["prediction_only_coords"]["path"])
        if sha256_file(coords_path) != query_row["prediction_only_coords"]["sha256"]:
            raise ValueError(f"B0 query coordinate hash drift: {coords_path}")
        query = np.load(coords_path, mmap_mode="r")
        normalized = []
        for start in range(0, len(query), int(chunk_size)):
            distances, neighbors = tree.query(
                np.asarray(query[start : start + chunk_size], dtype=np.float32),
                k=k,
                workers=-1,
            )
            normalized.append(
                _predict_neighbors(fields, distances, neighbors, k=k, bandwidth=bandwidth)
            )
        prediction = np.concatenate(normalized)
        velocity = prediction[:, :3] * velocity_std + velocity_mean
        pressure = prediction[:, 3] * pressure_std + pressure_mean
        pressure -= float(np.mean(pressure))
        output_path = PREDICTION_ROOT / query_row["case_id"] / "prediction.npz"
        _atomic_npz(
            output_path,
            velocity_m_s=velocity.astype(np.float32),
            pressure_relative_pa=pressure.astype(np.float32),
        )
        rows.append(
            {
                "case_id": query_row["case_id"],
                "query_sha256": query_row["query_sha256"],
                "prediction": {"path": str(output_path), "sha256": sha256_file(output_path)},
            }
        )
    report = {
        "schema_version": 1,
        "created_at": utc_now(),
        "status": "completed",
        "route": ROUTE,
        "baseline": "B0",
        "predictor_truth_access": False,
        "predictor_inputs": ["frozen train-only atlas", "coordinate-only val15 queries"],
        "atlas_manifest": {"path": str(ATLAS_MANIFEST), "sha256": sha256_file(ATLAS_MANIFEST)},
        "query_manifest": {"path": str(QUERY_MANIFEST), "sha256": sha256_file(QUERY_MANIFEST)},
        "cases": rows,
    }
    atomic_write_json(PREDICTION_MANIFEST, report)
    return report


def _accumulators() -> dict[str, RegressionAccumulator]:
    return {name: RegressionAccumulator() for name in ("u", "v", "w", "speed", "pressure")}


def _metrics(accumulators: dict[str, RegressionAccumulator]) -> dict[str, Any]:
    return {name: value.result() for name, value in accumulators.items()}


def evaluate_predictions() -> dict[str, Any]:
    data_manifest = json.loads(DATA_MANIFEST.read_text(encoding="utf-8"))
    prediction_manifest = json.loads(PREDICTION_MANIFEST.read_text(encoding="utf-8"))
    stats = json.loads(FIELD_STATS.read_text(encoding="utf-8"))
    velocity_mean = np.asarray(stats["velocity_m_s"]["mean"], dtype=np.float64)
    velocity_std = np.asarray(stats["velocity_m_s"]["std"], dtype=np.float64)
    pressure_mean = float(stats["pressure_relative_pa"]["mean"][0])
    pressure_std = float(stats["pressure_relative_pa"]["std"][0])
    prediction_rows = {row["case_id"]: row for row in prediction_manifest["cases"]}
    pooled = _accumulators()
    reports = []
    for row in data_manifest["cases"]:
        if row["role"] != "val":
            continue
        case_id = row["canonical_id"]
        contract = row["fixed_validation_query"]
        indices = np.load(contract["indices"]["path"], mmap_mode="r")
        volume = json.loads(Path(row["source_volume_manifest"]).read_text(encoding="utf-8"))
        velocity_truth = np.asarray(
            np.load(volume["files"]["velocity_m_s"]["path"], mmap_mode="r")[indices],
            dtype=np.float64,
        )
        pressure_truth = np.asarray(
            np.load(volume["files"]["pressure_relative_pa"]["path"], mmap_mode="r")[indices],
            dtype=np.float64,
        )
        pred_row = prediction_rows[case_id]
        if pred_row["query_sha256"] != contract["query_sha256"]:
            raise ValueError(f"B0 prediction/query mismatch: {case_id}")
        with np.load(pred_row["prediction"]["path"], allow_pickle=False) as values:
            velocity_pred = np.asarray(values["velocity_m_s"], dtype=np.float64)
            pressure_pred = np.asarray(values["pressure_relative_pa"], dtype=np.float64)
        case_acc = _accumulators()
        truth_values = {
            "u": velocity_truth[:, 0],
            "v": velocity_truth[:, 1],
            "w": velocity_truth[:, 2],
            "speed": np.linalg.norm(velocity_truth, axis=1),
            "pressure": pressure_truth,
        }
        pred_values = {
            "u": velocity_pred[:, 0],
            "v": velocity_pred[:, 1],
            "w": velocity_pred[:, 2],
            "speed": np.linalg.norm(velocity_pred, axis=1),
            "pressure": pressure_pred,
        }
        for name in truth_values:
            case_acc[name].update(truth_values[name], pred_values[name])
            pooled[name].update(truth_values[name], pred_values[name])
        truth_std = np.column_stack(
            [
                (velocity_truth - velocity_mean) / velocity_std,
                (pressure_truth - pressure_mean) / pressure_std,
            ]
        )
        pred_std = np.column_stack(
            [
                (velocity_pred - velocity_mean) / velocity_std,
                (pressure_pred - pressure_mean) / pressure_std,
            ]
        )
        reports.append(
            {
                "case_id": case_id,
                "cohort": row["cohort"],
                "points": int(len(indices)),
                "validation_field_score": float(np.mean(np.square(pred_std - truth_std))),
                "metrics": _metrics(case_acc),
            }
        )
    if len(reports) != 15:
        raise ValueError("B0 evaluator must cover exactly val15")
    metric_names = ("u", "v", "w", "speed", "pressure")
    report = {
        "schema_version": 1,
        "created_at": utc_now(),
        "status": "completed",
        "route": ROUTE,
        "baseline": "B0",
        "predictor_evaluator_interface_isolated": True,
        "test35_cases_read": 0,
        "validation_field_score_cb": float(np.mean([row["validation_field_score"] for row in reports])),
        "case_balanced_regression": {
            name: {
                statistic: float(np.mean([row["metrics"][name][statistic] for row in reports]))
                for statistic in ("r2", "mae", "rmse")
            }
            for name in metric_names
        },
        "pooled_regression": _metrics(pooled),
        "cases": reports,
        "atlas_manifest_sha256": sha256_file(ATLAS_MANIFEST),
        "prediction_manifest_sha256": sha256_file(PREDICTION_MANIFEST),
        "split_sha256": SPLIT_SHA256,
    }
    atomic_write_json(EVALUATION_JSON, report)
    target = guard_write_path(EVALUATION_CSV)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["case_id", "cohort", "validation_field_score"] + [
            f"{name}_{stat}"
            for name in metric_names
            for stat in ("r2", "mae", "rmse")
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in reports:
            flat = {
                "case_id": row["case_id"],
                "cohort": row["cohort"],
                "validation_field_score": row["validation_field_score"],
            }
            flat.update(
                {
                    f"{name}_{stat}": row["metrics"][name][stat]
                    for name in metric_names
                    for stat in ("r2", "mae", "rmse")
                }
            )
            writer.writerow(flat)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "predict", "evaluate", "all"))
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--points-per-case", type=int, default=4096)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    result = None
    if args.command in {"build", "all"}:
        result = build_atlas(seed=args.seed, points_per_case=args.points_per_case, force=args.force)
    if args.command in {"predict", "all"}:
        result = predict(force=args.force)
    if args.command in {"evaluate", "all"}:
        result = evaluate_predictions()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
