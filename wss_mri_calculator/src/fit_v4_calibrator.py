"""Fit and serialize the frozen V4 diagnostic residual calibrator."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from calculate_wss_cfd import wss_from_gradient
from calibrate_v4_oof import (
    EPS,
    build_case_scale_features,
    build_features,
    load_cache,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--model-out", required=True)
    parser.add_argument("--manifest-out", required=True)
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir).resolve()
    data = load_cache(cache_dir)
    x, feature_names = build_features(data)
    truth = np.asarray(data["truth_mag"], dtype=np.float64)
    gradient = np.asarray(data["gradient_v4"], dtype=np.float64)
    base_mag = np.linalg.norm(wss_from_gradient(gradient, "carreau"), axis=1)
    target = np.log(np.maximum(truth, EPS) / np.maximum(base_mag, EPS))
    target = np.clip(target, np.log(0.5), np.log(2.5))

    parameters = {
        "loss": "squared_error",
        "learning_rate": 0.05,
        "max_iter": 350,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 120,
        "l2_regularization": 5.0,
        "random_state": args.seed,
    }
    model = HistGradientBoostingRegressor(**parameters)
    model.fit(x, target)
    global_alpha = float(
        np.dot(truth, base_mag) / max(float(np.dot(base_mag, base_mag)), EPS)
    )
    point_case = np.asarray(data["point_case"])
    case_names, case_x = build_case_scale_features(x, feature_names, point_case)
    case_alpha = []
    for case in case_names:
        mask = point_case == case
        case_alpha.append(
            float(
                np.dot(truth[mask], base_mag[mask])
                / max(float(np.dot(base_mag[mask], base_mag[mask])), EPS)
            )
        )
    case_model = make_pipeline(
        StandardScaler(), RidgeCV(alphas=(0.1, 1.0, 10.0, 100.0, 1000.0))
    )
    case_model.fit(case_x, np.log(np.asarray(case_alpha)))
    package = {
        "schema_version": 1,
        "method_name": "surface_mls_v4_hist_calibrated",
        "model": model,
        "case_model": case_model,
        "feature_names": feature_names,
        "raw_ratio_clip": (0.7, 1.8),
        "global_alpha": global_alpha,
        "case_alpha_clip": (1.00, 1.40),
        "local_strength": {
            "minimum": 0.25,
            "full_at_case_alpha": 1.08,
            "fallback_to_physics_at_lower_bound": True,
        },
        "final_ratio_clip": (0.85, 1.60),
        "parameters": parameters,
    }
    model_out = Path(args.model_out).resolve()
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(package, model_out, compress=3)

    cache_files = sorted(cache_dir.glob("*.npz"))
    manifest = {
        "schema_version": 1,
        "method_name": "surface_mls_v4_hist_calibrated",
        "selection_roles": ["train"],
        "n_cases": int(len(np.unique(np.asarray(data["point_case"])))),
        "n_points": int(len(target)),
        "feature_names": feature_names,
        "raw_ratio_clip": [0.7, 1.8],
        "global_alpha": global_alpha,
        "case_alpha_clip": [1.00, 1.40],
        "local_strength": {
            "minimum": 0.25,
            "full_at_case_alpha": 1.08,
            "fallback_to_physics_at_lower_bound": True,
        },
        "final_ratio_clip": [0.85, 1.60],
        "parameters": parameters,
        "cache_dir": str(cache_dir),
        "cache_file_count": len(cache_files),
        "model_path": str(model_out),
        "model_sha256": sha256_file(model_out),
    }
    manifest_out = Path(args.manifest_out).resolve()
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
