#!/usr/bin/env python3
"""Freeze REG-P10-LSA2 as anchor and prepare SAME/IND × MSE/H1/H2."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from training_wss_min.config import ExpConfig


ROOT = Path(__file__).resolve().parents[2]
BASE = (
    ROOT
    / "training_wss_min/configs/pointnetpp_regp10_transformer_20260726"
    / "regp10_localtf_sa2_s1234.json"
)
HISTORICAL_RUN = (
    ROOT
    / "training_wss_min/runs/pointnetpp_regp10_transformer/outputs"
    / "localtf_sa2_s1234"
)
SPLIT = ROOT / "training/splits/split_AG_AAA_ILO_q2v_pool2025_train138_test36.json"
CONFIG_DIR = (
    ROOT
    / "training_wss_min/configs/"
    "pointnetpp_regp10_lsa2_objective_ind_20260728"
)
PREFLIGHT = ROOT / "training_wss_min/preflight"
MANIFEST = (
    PREFLIGHT / "regp10_lsa2_objective_ind_matrix_20260728_prepared.json"
)
STATIC_AUDIT = (
    PREFLIGHT / "regp10_lsa2_objective_ind_matrix_20260728_static_audit.json"
)
RUN_PREFIX = "pointnetpp_regp10_lsa2_objective_ind/outputs"

ARMS = (
    ("lsa2_same_mse_s1234", "same", "mse"),
    ("lsa2_same_h1_bce_q90_lam020_s1234", "same", "h1"),
    ("lsa2_same_h2_pinball_q90_lam020_s1234", "same", "h2"),
    ("lsa2_ind_mse_s1234", "independent", "mse"),
    ("lsa2_ind_h1_bce_q90_lam020_s1234", "independent", "h1"),
    ("lsa2_ind_h2_pinball_q90_lam020_s1234", "independent", "h2"),
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
        output = set()
        for key in set(control) | set(treatment):
            path = f"{prefix}.{key}" if prefix else key
            if key not in control or key not in treatment:
                output.add(path)
            else:
                output.update(
                    changed_paths(control[key], treatment[key], path)
                )
        return output
    return set() if control == treatment else {prefix}


def explicit_anchor() -> dict:
    anchor = json.loads(BASE.read_text(encoding="utf-8"))
    anchor["data"]["case_features_path"] = None
    anchor["model"].update(
        {
            "drop_path_rate": 0.10,
            "neighbor_drop_rate": 0.0,
            "local_transformer_stages": [2],
            "local_transformer_heads": 4,
            "local_transformer_ffn_ratio": 2,
            "local_transformer_dropout": 0.0,
            "edgeconv_stages": [],
            "coarse_attention": False,
        }
    )
    anchor["train"].update(
        {
            "loss_hotspot_bce_lambda": 0.0,
            "hotspot_quantile": 0.90,
            "loss_pinball_lambda": 0.0,
            "pinball_quantile": 0.90,
        }
    )
    return anchor


def make_configs(anchor: dict) -> list[dict]:
    rows = []
    for index, (experiment_id, query_mode, objective) in enumerate(ARMS):
        cfg = deepcopy(anchor)
        cfg["name"] = f"{RUN_PREFIX}/{experiment_id}"
        cfg["data"]["query_mode"] = query_mode
        cfg["data"]["query_n_points"] = 5000
        cfg["data"]["query_sampling"] = "random"
        if objective == "h1":
            cfg["model"]["out_dim"] = 2
            cfg["train"]["loss_hotspot_bce_lambda"] = 0.20
        elif objective == "h2":
            cfg["model"]["out_dim"] = 1
            cfg["train"]["loss_pinball_lambda"] = 0.20
        else:
            cfg["model"]["out_dim"] = 1
        cfg["notes"] = (
            "REG-P10-LSA2 formal development anchor; "
            f"query_mode={query_mode}; objective={objective}; "
            "no RCR/case features, no area weighting, no multi-seed. "
            "Mixed 138/0/36 reused-development protocol; seed1234; "
            "random5000 support/query; 400 epochs; train-loss selection; "
            "legacy_vertex."
        )
        path = CONFIG_DIR / f"{experiment_id}.json"
        dump(path, cfg)
        parsed = ExpConfig.from_json(path)
        actual = changed_paths(anchor, cfg)
        allowed = {"name", "notes"}
        if query_mode == "independent":
            allowed.add("data.query_mode")
        if objective == "h1":
            allowed.update(
                {
                    "model.out_dim",
                    "train.loss_hotspot_bce_lambda",
                }
            )
        if objective == "h2":
            allowed.add("train.loss_pinball_lambda")
        if actual != allowed:
            raise RuntimeError(
                f"{experiment_id}: unexpected config changes "
                f"{sorted(actual ^ allowed)}"
            )
        rows.append(
            {
                "index": index,
                "experiment_id": experiment_id,
                "query_mode": query_mode,
                "objective": objective,
                "config": str(path.resolve()),
                "config_sha256": sha256(path),
                "run_dir": str(parsed.run_dir.resolve()),
                "out_dim": parsed.model.out_dim,
                "hotspot_bce_lambda": parsed.train.loss_hotspot_bce_lambda,
                "pinball_lambda": parsed.train.loss_pinball_lambda,
                "changed_paths_from_explicit_anchor": sorted(actual),
            }
        )
    return rows


def main() -> None:
    anchor = explicit_anchor()
    parsed_anchor = ExpConfig.from_dict(anchor)
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    counts = (
        len(split.get("train_cases", [])),
        len(split.get("val_cases", [])),
        len(split.get("test_cases", [])),
    )
    if counts != (138, 0, 36):
        raise RuntimeError(f"expected mixed 138/0/36, got {counts}")
    if (
        parsed_anchor.data.case_features_path is not None
        or tuple(parsed_anchor.model.local_transformer_stages) != (2,)
        or float(parsed_anchor.model.drop_path_rate) != 0.10
        or parsed_anchor.model.coarse_attention
        or tuple(parsed_anchor.model.edgeconv_stages)
    ):
        raise RuntimeError("base is not the exact non-RCR REG-P10-LSA2 anchor")
    if not (HISTORICAL_RUN / "eval/ckpt_best/metrics.json").is_file():
        raise RuntimeError("historical REG-P10-LSA2 result is missing")
    historical = json.loads(
        (HISTORICAL_RUN / "eval/ckpt_best/metrics.json").read_text(
            encoding="utf-8"
        )
    )["test"]
    rows = make_configs(anchor)
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "prepared",
        "protocol": (
            "REG-P10-LSA2 formal development anchor; SAME/IND x MSE/H1/H2; "
            "mixed 138/0/36; seed1234; random5000; legacy_vertex; no RCR"
        ),
        "source_anchor_config": str(BASE.resolve()),
        "source_anchor_config_sha256": sha256(BASE),
        "historical_anchor_run": str(HISTORICAL_RUN.resolve()),
        "historical_anchor_best": {
            "physical_r2_casebalanced": historical["field_casebalanced"]["r2"],
            "normalized_r2_casebalanced": historical["normalized"][
                "field_casebalanced"
            ]["r2"],
            "physical_mae_casebalanced": historical["field_casebalanced"]["mae"],
            "high_wss_nrmse_range": historical["regional_field"]["high_wss"][
                "nrmse_range"
            ],
            "top10_iou": historical["hotspot"]["top10_iou_casemean"],
        },
        "explicit_anchor_contract": {
            "drop_path_rate": 0.10,
            "local_transformer_stages": [2],
            "local_transformer_heads": 4,
            "local_transformer_ffn_ratio": 2,
            "local_transformer_dropout": 0.0,
            "query_decoder": "interpolate",
            "case_features_path": None,
        },
        "split": str(SPLIT.resolve()),
        "split_sha256": sha256(SPLIT),
        "configs": rows,
        "registered_comparisons": {
            "same_h1": "same H1 minus concurrent same MSE",
            "same_h2": "same H2 minus concurrent same MSE",
            "ind_main": "IND MSE minus concurrent same MSE",
            "ind_h1": "IND H1 minus concurrent IND MSE",
            "ind_h2": "IND H2 minus concurrent IND MSE",
            "ind_within_objective": [
                "IND H1 minus same H1",
                "IND H2 minus same H2",
            ],
        },
        "registered_gate": {
            "H1_primary": "delta top10 IoU >= +0.02",
            "H2_primary": "delta high-WSS nRMSE_range <= -0.002",
            "objective_shared_guards": (
                "delta normalized R2_casebalanced >= -0.01; "
                "delta physical MAE_casebalanced <= +0.05 Pa"
            ),
            "IND_primary": "delta physical R2_casebalanced >= +0.012",
            "IND_guards": (
                "delta normalized R2_casebalanced >= -0.01; "
                "delta physical MAE_casebalanced <= +0.03 Pa; "
                "delta high-WSS nRMSE_range <= +0.001; "
                "delta top10 IoU >= -0.005"
            ),
            "promotion_policy": (
                "Use only concurrent SAME/IND controls for decisions. "
                "Do not compare a new arm directly to historical O0. "
                "Do not combine H1 and H2 in this matrix."
            ),
        },
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
            "n_configs": len(rows),
            "experiment_ids": [row["experiment_id"] for row in rows],
            "query_modes": [row["query_mode"] for row in rows],
            "objectives": [row["objective"] for row in rows],
            "out_dims": [row["out_dim"] for row in rows],
            "hotspot_bce_lambdas": [
                row["hotspot_bce_lambda"] for row in rows
            ],
            "pinball_lambdas": [row["pinball_lambda"] for row in rows],
            "no_rcr_or_case_features": all(
                ExpConfig.from_json(row["config"]).data.case_features_path is None
                for row in rows
            ),
            "all_regp10_lsa2": all(
                tuple(
                    ExpConfig.from_json(row["config"]).model.local_transformer_stages
                )
                == (2,)
                and float(
                    ExpConfig.from_json(row["config"]).model.drop_path_rate
                )
                == 0.10
                for row in rows
            ),
            "surface_metric_mode": "legacy_vertex",
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
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
