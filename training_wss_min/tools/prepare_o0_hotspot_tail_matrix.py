#!/usr/bin/env python3
"""Prepare two single-variable high-WSS objectives on the frozen O0 parent."""

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
    / "training_wss_min/configs/pointnetpp_rcr_oracle_20260728"
    / "o0_geometry_s1234.json"
)
SPLIT = ROOT / "training/splits/split_AG_AAA_ILO_q2v_pool2025_train138_test36.json"
CONFIG_DIR = (
    ROOT / "training_wss_min/configs/pointnetpp_o0_hotspot_tail_20260728"
)
PREFLIGHT = ROOT / "training_wss_min/preflight"
MANIFEST = PREFLIGHT / "o0_hotspot_tail_matrix_20260728_prepared.json"
STATIC_AUDIT = PREFLIGHT / "o0_hotspot_tail_matrix_20260728_static_audit.json"


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
        paths = set()
        for key in set(control) | set(treatment):
            path = f"{prefix}.{key}" if prefix else key
            if key not in control or key not in treatment:
                paths.add(path)
            else:
                paths.update(changed_paths(control[key], treatment[key], path))
        return paths
    return set() if control == treatment else {prefix}


def make_configs() -> list[dict]:
    base = json.loads(BASE.read_text(encoding="utf-8"))
    variants = (
        {
            "experiment_id": "h1_hotspot_bce_q90_lam020_s1234",
            "notes": (
                "H1 single-variable hotspot-localization arm: O0 plus a case-relative "
                "top10 balanced BCE auxiliary output; no RCR, sampling, or backbone change."
            ),
            "model": {"out_dim": 2},
            "train": {
                "loss_hotspot_bce_lambda": 0.20,
                "hotspot_quantile": 0.90,
                "loss_pinball_lambda": 0.0,
                "pinball_quantile": 0.90,
            },
            "allowed_changes": {
                "name",
                "notes",
                "model.out_dim",
                "train.loss_hotspot_bce_lambda",
                "train.hotspot_quantile",
                "train.loss_pinball_lambda",
                "train.pinball_quantile",
            },
        },
        {
            "experiment_id": "h2_pinball_q90_lam020_s1234",
            "notes": (
                "H2 single-variable high-WSS amplitude arm: O0 plus q=0.90 pinball "
                "auxiliary loss; no RCR, sampling, backbone, or output-head change."
            ),
            "model": {"out_dim": 1},
            "train": {
                "loss_hotspot_bce_lambda": 0.0,
                "hotspot_quantile": 0.90,
                "loss_pinball_lambda": 0.20,
                "pinball_quantile": 0.90,
            },
            "allowed_changes": {
                "name",
                "notes",
                "train.loss_hotspot_bce_lambda",
                "train.hotspot_quantile",
                "train.loss_pinball_lambda",
                "train.pinball_quantile",
            },
        },
    )
    rows = []
    for index, variant in enumerate(variants):
        cfg = deepcopy(base)
        experiment_id = variant["experiment_id"]
        cfg["name"] = f"pointnetpp_o0_hotspot_tail/outputs/{experiment_id}"
        cfg["notes"] = (
            variant["notes"]
            + " Frozen mixed 138/0/36 reused-development protocol; seed1234; "
            "400 epochs; train-loss checkpoint selection; legacy_vertex."
        )
        cfg["model"].update(variant["model"])
        cfg["train"].update(variant["train"])
        path = CONFIG_DIR / f"{experiment_id}.json"
        dump(path, cfg)
        parsed = ExpConfig.from_json(path)
        actual_changes = changed_paths(base, cfg)
        if actual_changes != variant["allowed_changes"]:
            raise RuntimeError(
                f"{experiment_id}: unexpected config changes "
                f"{sorted(actual_changes ^ variant['allowed_changes'])}"
            )
        rows.append(
            {
                "index": index,
                "experiment_id": experiment_id,
                "config": str(path.resolve()),
                "config_sha256": sha256(path),
                "run_dir": str(parsed.run_dir.resolve()),
                "out_dim": parsed.model.out_dim,
                "hotspot_bce_lambda": parsed.train.loss_hotspot_bce_lambda,
                "hotspot_quantile": parsed.train.hotspot_quantile,
                "pinball_lambda": parsed.train.loss_pinball_lambda,
                "pinball_quantile": parsed.train.pinball_quantile,
                "changed_paths": sorted(actual_changes),
            }
        )
    return rows


def main() -> None:
    base = json.loads(BASE.read_text(encoding="utf-8"))
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    counts = (
        len(split.get("train_cases", [])),
        len(split.get("val_cases", [])),
        len(split.get("test_cases", [])),
    )
    if counts != (138, 0, 36):
        raise RuntimeError(f"expected mixed 138/0/36, got {counts}")
    if base["data"].get("case_features_path") is not None:
        raise RuntimeError("O0 parent must not use case-level/RCR features")
    rows = make_configs()
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "prepared",
        "protocol": (
            "O0 geometry-only S3 PointNeXt-R + LocalGeoPE; mixed 138/0/36; "
            "seed1234; legacy_vertex; single-variable hotspot/tail objectives"
        ),
        "base_config": str(BASE.resolve()),
        "base_config_sha256": sha256(BASE),
        "split": str(SPLIT.resolve()),
        "split_sha256": sha256(SPLIT),
        "configs": rows,
        "registered_gate": {
            "daily_metrics": [
                "normalized R2_casebalanced",
                "physical MAE_casebalanced",
                "physical high-WSS nRMSE_range",
                "physical top10 IoU_casebalanced",
            ],
            "H1_primary": "delta top10 IoU >= +0.02",
            "H2_primary": "delta high-WSS nRMSE_range <= -0.002",
            "shared_guards": (
                "delta normalized R2_casebalanced >= -0.01; "
                "delta physical MAE_casebalanced <= +0.05"
            ),
            "combination_policy": (
                "Do not combine H1/H2 unless each passes its own primary and shared guards."
            ),
        },
    }
    dump(MANIFEST, payload)
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
            "out_dims": [row["out_dim"] for row in rows],
            "hotspot_bce_lambdas": [
                row["hotspot_bce_lambda"] for row in rows
            ],
            "pinball_lambdas": [row["pinball_lambda"] for row in rows],
            "no_case_features_or_rcr": all(
                ExpConfig.from_json(row["config"]).data.case_features_path is None
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
