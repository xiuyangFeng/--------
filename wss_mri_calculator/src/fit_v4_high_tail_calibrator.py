"""Fit and serialize the selected train138 high-WSS-tail V4 calibrator."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from calculate_wss_cfd import wss_from_gradient
from calibrate_v4_high_tail_oof import (
    fit_case_scale_models,
    smooth_gate,
    within_case_rank,
)
from calibrate_v4_oof import EPS, build_features, load_cache


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
    parser.add_argument(
        "--method-name", default="surface_mls_v4_high_tail_w10_s80"
    )
    parser.add_argument(
        "--grouped-oof-evidence", default="high_tail_oof_train138_v1.json"
    )
    parser.add_argument(
        "--fixed-holdout-evidence", default="high_tail_dev73_holdout65_v1.json"
    )
    parser.add_argument("--physics-evidence", default="")
    parser.add_argument("--profile-features", action="store_true")
    parser.add_argument("--peak-anchor-strength", type=float, default=0.30)
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir).resolve()
    data = load_cache(cache_dir)
    if args.profile_features:
        from v4_profile_features import base_v4_view, build_profile_features

        x, feature_names = build_profile_features(data)
        prediction_data = base_v4_view(data)
    else:
        x, feature_names = build_features(data)
        prediction_data = data
    truth = np.asarray(data["truth_mag"], dtype=np.float64)
    groups = np.asarray(data["point_case"])
    base_vector = wss_from_gradient(
        np.asarray(prediction_data["gradient_v4"], dtype=np.float64), "carreau"
    )
    base_mag = np.linalg.norm(base_vector, axis=1)
    target = np.log(np.maximum(truth, EPS) / np.maximum(base_mag, EPS))
    target = np.clip(target, np.log(0.5), np.log(2.5))
    truth_rank = within_case_rank(truth, groups)

    parameters = {
        "loss": "squared_error",
        "learning_rate": 0.05,
        "max_iter": 350,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 120,
        "l2_regularization": 5.0,
    }
    current_model = HistGradientBoostingRegressor(
        **parameters, random_state=args.seed
    )
    current_model.fit(x, target)

    tail_emphasis = smooth_gate(truth_rank, 0.80)
    underprediction = np.clip(
        target / max(float(np.log(2.5)), EPS), 0.0, 1.0
    )
    sample_weight = 1.0 + 10.0 * tail_emphasis + 5.0 * tail_emphasis * underprediction
    tail_model = HistGradientBoostingRegressor(
        **parameters, random_state=args.seed + 1000
    )
    tail_model.fit(x, target, sample_weight=sample_weight)

    overall_case_model, tail_case_model, peak_case_model = fit_case_scale_models(
        x, feature_names, groups, truth, base_mag
    )
    package = {
        "schema_version": 2,
        "method_name": args.method_name,
        "selection_evidence": {
            "grouped_oof": args.grouped_oof_evidence,
            "fixed_holdout": args.fixed_holdout_evidence,
            "physics_kernel": args.physics_evidence or None,
            "selection_roles": ["train"],
        },
        "current_model": current_model,
        "tail_model": tail_model,
        "overall_case_model": overall_case_model,
        "tail_case_model": tail_case_model,
        "peak_case_model": peak_case_model,
        "feature_names": feature_names,
        "current_raw_ratio_clip": (0.7, 1.8),
        "tail_raw_ratio_clip": (0.7, 2.2),
        "case_alpha_clip": (1.00, 1.40),
        "tail_alpha_clip": (1.00, 1.80),
        "peak_alpha_clip": (1.00, 2.20),
        "current_final_ratio_clip": (0.85, 1.60),
        "tail_final_ratio_clip": (0.85, 2.20),
        "current_local_strength": {
            "minimum": 0.25,
            "full_at_case_alpha": 1.08,
            "fallback_to_physics_at_lower_bound": True,
        },
        "tail_gate_start": 0.80,
        "tail_weight": 10.0,
        "underprediction_tail_weight": 5.0,
        "peak_anchor_strength": args.peak_anchor_strength,
        "parameters": parameters,
        "seed": args.seed,
        "feature_variant": "depth3_profile" if args.profile_features else "base",
    }
    model_out = Path(args.model_out).resolve()
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(package, model_out, compress=3)

    manifest = {
        "schema_version": 2,
        "method_name": package["method_name"],
        "selection_roles": ["train"],
        "n_cases": int(len(np.unique(groups))),
        "n_points": int(len(groups)),
        "feature_names": feature_names,
        "feature_variant": package["feature_variant"],
        "tail_gate_start": package["tail_gate_start"],
        "peak_anchor_strength": package["peak_anchor_strength"],
        "tail_weight": package["tail_weight"],
        "underprediction_tail_weight": package["underprediction_tail_weight"],
        "ratio_clips": {
            "current_raw": list(package["current_raw_ratio_clip"]),
            "tail_raw": list(package["tail_raw_ratio_clip"]),
            "current_final": list(package["current_final_ratio_clip"]),
            "tail_final": list(package["tail_final_ratio_clip"]),
        },
        "selection_evidence": package["selection_evidence"],
        "cache_dir": str(cache_dir),
        "cache_file_count": len(list(cache_dir.glob("*.npz"))),
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
