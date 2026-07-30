#!/usr/bin/env python3
"""Prepare a same-seed H2 control and log(local_radius) input arm."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from training_wss_min import dataset as D
from training_wss_min.config import ExpConfig


ROOT = Path(__file__).resolve().parents[2]
PARENT_CONFIG = (
    ROOT
    / "training_wss_min/configs/pointnetpp_regp10_lsa2_objective_ind_20260728"
    / "lsa2_same_h2_pinball_q90_lam020_s1234.json"
)
PARENT_RUN = (
    ROOT
    / "training_wss_min/runs/pointnetpp_regp10_lsa2_objective_ind/outputs"
    / "lsa2_same_h2_pinball_q90_lam020_s1234"
)
CONTROL_SPLIT = (
    ROOT
    / "training/splits/split_AG_AAA_ILO_q2v_pool2025_control106_test36.json"
)
BASE_FEATURE_STATS = (
    ROOT
    / "data_wss_min/fold_stats/q2v_ilo_20260718"
    / "q2v_pool2025_control106_feature_stats.json"
)
AUGMENTED_FEATURE_STATS = (
    ROOT
    / "data_wss_min/fold_stats/q2v_ilo_20260718"
    / "q2v_pool2025_control106_feature_stats_logradius.json"
)
CONFIG_DIR = (
    ROOT
    / "training_wss_min/configs/pointnetpp_lsa2_h2_logradius_20260729"
)
PREFLIGHT = ROOT / "training_wss_min/preflight"
MANIFEST = PREFLIGHT / "lsa2_h2_logradius_matrix_20260729_prepared.json"
STATIC_AUDIT = PREFLIGHT / "lsa2_h2_logradius_matrix_20260729_static_audit.json"
RUN_PREFIX = "pointnetpp_lsa2_h2_logradius/outputs"

ARMS = (
    ("lsa2_h2_same_repro_s1234", False),
    ("lsa2_h2_logradius_s1234", True),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def changed_paths(control: object, treatment: object, prefix: str = "") -> set[str]:
    if isinstance(control, dict) and isinstance(treatment, dict):
        output: set[str] = set()
        for key in set(control) | set(treatment):
            path = f"{prefix}.{key}" if prefix else key
            if key not in control or key not in treatment:
                output.add(path)
            else:
                output.update(changed_paths(control[key], treatment[key], path))
        return output
    return set() if control == treatment else {prefix}


def make_augmented_feature_stats(parent: dict) -> dict:
    base = json.loads(BASE_FEATURE_STATS.read_text(encoding="utf-8"))
    expected_base = {"abscissa_norm", "local_radius", "curvature"}
    if set(base) != expected_base:
        raise RuntimeError(
            f"unexpected parent feature-stat keys: {sorted(base)}"
        )
    target_stats = D.load_wss_stats(parent["data"]["wss_stats_path"])
    cases = D.load_partition(
        str(CONTROL_SPLIT),
        "train",
        target_stats,
        strict=True,
        target=parent["data"]["target"],
        target_normalization=parent["data"]["target_normalization"],
        data_root=parent["data"]["data_root"],
        required_frame_version=parent["data"]["required_frame_version"],
        case_features_path=None,
    )
    if len(cases) != 106:
        raise RuntimeError(f"expected 106 control cases, got {len(cases)}")
    radius = np.concatenate(
        [np.asarray(case["local_radius"], dtype=np.float64) for case in cases]
    )
    log_radius = np.concatenate(
        [np.asarray(case["log_local_radius"], dtype=np.float64) for case in cases]
    )
    if (
        not np.isfinite(radius).all()
        or not np.isfinite(log_radius).all()
        or np.any(radius <= 0)
    ):
        raise RuntimeError("control106 local_radius is non-positive or non-finite")
    expected_log = np.log(radius)
    if not np.allclose(log_radius, expected_log, rtol=1e-6, atol=1e-6):
        raise RuntimeError("log_local_radius does not equal natural-log radius")
    derived = D.compute_feature_stats(cases, ("log_local_radius",))
    augmented = deepcopy(base)
    augmented["log_local_radius"] = derived["log_local_radius"]
    dump(AUGMENTED_FEATURE_STATS, augmented)
    return {
        "n_cases": len(cases),
        "n_points": int(radius.size),
        "local_radius_min": float(radius.min()),
        "local_radius_max": float(radius.max()),
        "log_local_radius_min": float(log_radius.min()),
        "log_local_radius_max": float(log_radius.max()),
        "log_local_radius_mean": augmented["log_local_radius"]["mean"],
        "log_local_radius_std": augmented["log_local_radius"]["std"],
    }


def anchor_metrics() -> dict:
    metrics_path = PARENT_RUN / "eval/ckpt_best/metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))["test"]
    return {
        "metrics": str(metrics_path.resolve()),
        "metrics_sha256": sha256(metrics_path),
        "checkpoint": str((PARENT_RUN / "ckpt_best.pt").resolve()),
        "checkpoint_sha256": sha256(PARENT_RUN / "ckpt_best.pt"),
        "physical_r2_casebalanced": metrics["field_casebalanced"]["r2"],
        "normalized_r2_casebalanced": metrics["normalized"][
            "field_casebalanced"
        ]["r2"],
        "physical_mae_casebalanced": metrics["field_casebalanced"]["mae"],
        "high_wss_nrmse_range": metrics["regional_field"]["high_wss"][
            "nrmse_range"
        ],
        "top10_iou": metrics["hotspot"]["top10_iou_casemean"],
    }


def make_configs(parent: dict) -> list[dict]:
    rows = []
    for index, (experiment_id, add_log_radius) in enumerate(ARMS):
        cfg = deepcopy(parent)
        cfg["name"] = f"{RUN_PREFIX}/{experiment_id}"
        if add_log_radius:
            cfg["data"]["input_features"] = [
                *cfg["data"]["input_features"],
                "log_local_radius",
            ]
            cfg["data"]["feature_stats_path"] = str(
                AUGMENTED_FEATURE_STATS.resolve()
            )
            cfg["notes"] = (
                "REG-P10-LSA2 SAME H2 q90 pinball lambda0.20 anchor plus one "
                "extra train-only-standardized natural-log(local_radius) input "
                "column; raw local_radius and LocalGeoPE indices stay unchanged. "
                "Mixed 138/0/36 reused-development protocol; seed1234 only; "
                "no RCR, area weighting, H1 combination, or multi-seed."
            )
        else:
            cfg["notes"] = (
                "Exact same-seed concurrent reproduction of the REG-P10-LSA2 "
                "SAME H2 q90 pinball lambda0.20 anchor; no architecture, feature, "
                "loss, split, or normalization change. Mixed 138/0/36 reused-"
                "development protocol; seed1234 only; no multi-seed."
            )
        path = CONFIG_DIR / f"{experiment_id}.json"
        dump(path, cfg)
        parsed = ExpConfig.from_json(path)
        changed = changed_paths(parent, cfg)
        allowed = {"name", "notes"}
        if add_log_radius:
            allowed.update(
                {
                    "data.input_features",
                    "data.feature_stats_path",
                }
            )
        if changed != allowed:
            raise RuntimeError(
                f"{experiment_id}: unexpected config changes "
                f"{sorted(changed ^ allowed)}"
            )
        if tuple(parsed.model.local_geope_feature_indices) != (3, 4, 5):
            raise RuntimeError("LocalGeoPE indices must remain on raw geometry")
        rows.append(
            {
                "index": index,
                "experiment_id": experiment_id,
                "role": "logradius_treatment" if add_log_radius else "concurrent_control",
                "config": str(path.resolve()),
                "config_sha256": sha256(path),
                "run_dir": str(parsed.run_dir.resolve()),
                "input_features": list(parsed.data.input_features),
                "input_dim": len(parsed.data.input_features),
                "feature_stats": str(
                    Path(parsed.data.feature_stats_path).resolve()
                ),
                "feature_stats_sha256": sha256(
                    Path(parsed.data.feature_stats_path)
                ),
                "changed_paths_from_frozen_anchor": sorted(changed),
            }
        )
    return rows


def main() -> None:
    parent = json.loads(PARENT_CONFIG.read_text(encoding="utf-8"))
    parsed_parent = ExpConfig.from_dict(parent)
    if (
        tuple(parsed_parent.data.input_features)
        != ("x", "y", "z", "abscissa_norm", "local_radius", "curvature")
        or parsed_parent.data.query_mode != "same"
        or parsed_parent.train.seed != 1234
        or float(parsed_parent.train.loss_pinball_lambda) != 0.20
        or float(parsed_parent.train.pinball_quantile) != 0.90
        or tuple(parsed_parent.model.local_transformer_stages) != (2,)
        or float(parsed_parent.model.drop_path_rate) != 0.10
        or parsed_parent.data.case_features_path is not None
    ):
        raise RuntimeError("parent is not the frozen SAME-H2 LSA2 anchor")
    split = json.loads(Path(parsed_parent.data.split_path).read_text(encoding="utf-8"))
    counts = (
        len(split.get("train_cases", [])),
        len(split.get("val_cases", [])),
        len(split.get("test_cases", [])),
    )
    if counts != (138, 0, 36):
        raise RuntimeError(f"expected mixed 138/0/36, got {counts}")
    if Path(parsed_parent.data.feature_stats_path).resolve() != BASE_FEATURE_STATS.resolve():
        raise RuntimeError("parent feature-stat source drift")
    stats_summary = make_augmented_feature_stats(parent)
    rows = make_configs(parent)
    anchor = anchor_metrics()
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "prepared",
        "protocol": (
            "REG-P10-LSA2 SAME H2 anchor; exact same-seed concurrent control "
            "versus one extra log(local_radius) input; mixed 138/0/36; "
            "seed1234; legacy_vertex; no RCR/area/H1/multi-seed"
        ),
        "source_anchor_config": str(PARENT_CONFIG.resolve()),
        "source_anchor_config_sha256": sha256(PARENT_CONFIG),
        "source_anchor_run": str(PARENT_RUN.resolve()),
        "source_anchor_best": anchor,
        "control_feature_stats": str(BASE_FEATURE_STATS.resolve()),
        "control_feature_stats_sha256": sha256(BASE_FEATURE_STATS),
        "augmented_feature_stats": str(AUGMENTED_FEATURE_STATS.resolve()),
        "augmented_feature_stats_sha256": sha256(AUGMENTED_FEATURE_STATS),
        "feature_stats_source_split": str(CONTROL_SPLIT.resolve()),
        "feature_stats_source_split_sha256": sha256(CONTROL_SPLIT),
        "feature_stats_summary": stats_summary,
        "configs": rows,
        "registered_comparison": (
            "lsa2_h2_logradius_s1234 minus concurrent "
            "lsa2_h2_same_repro_s1234"
        ),
        "registered_gate": {
            "primary": (
                "delta physical R2_casebalanced >= +0.012 OR "
                "delta high-WSS nRMSE_range <= -0.002"
            ),
            "guards": (
                "delta normalized R2_casebalanced >= -0.01; "
                "delta physical MAE_casebalanced <= +0.05 Pa; "
                "delta top10 IoU >= -0.005"
            ),
            "policy": (
                "Use the concurrent same-seed H2 reproduction for the formal "
                "decision. The historical H2 result is reference-only because "
                "CUDA training-trajectory drift has already been observed."
            ),
        },
        "interpretation_limits": [
            "Single seed1234 engineering screen; no multi-seed confirmation.",
            "test36 is a repeatedly reused development set, not an independent final test set.",
            "Checkpoint selection remains train-loss because the frozen split has no validation partition.",
            "The new feature is natural log of physical-mm local_radius, standardized using the same control106 train-only source as the frozen parent features.",
        ],
    }
    dump(MANIFEST, manifest)
    audit = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "manifest": str(MANIFEST.resolve()),
        "manifest_sha256": sha256(MANIFEST),
        "checks": {
            "split_counts": counts,
            "experiment_ids": [row["experiment_id"] for row in rows],
            "roles": [row["role"] for row in rows],
            "input_dims": [row["input_dim"] for row in rows],
            "input_features": [row["input_features"] for row in rows],
            "feature_stats_summary": stats_summary,
            "same_h2_anchor_contract": True,
            "local_geope_indices_unchanged": [3, 4, 5],
            "surface_metric_mode": "legacy_vertex",
            "no_rcr_area_h1_or_multiseed": True,
        },
    }
    dump(STATIC_AUDIT, audit)
    print(
        json.dumps(
            {
                "status": "passed",
                "configs": len(rows),
                "manifest": str(MANIFEST),
                "static_audit": str(STATIC_AUDIT),
                "log_radius_stats": stats_summary,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
