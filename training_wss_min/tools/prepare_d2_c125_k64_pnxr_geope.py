#!/usr/bin/env python3
"""Prepare the fixed-split D2 c125×k64 PointNeXt-R + LocalGeoPE control."""
from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from training_wss_min.config import ExpConfig


ROOT = Path(__file__).resolve().parents[2]
PARENT = ROOT / "training_wss_min/configs/pointnetpp_sa1_scale_single_seed_20260721/d2_rand5000_c125_k64.json"
CONFIG = ROOT / "training_wss_min/configs/pointnetpp_d2_c125_k64_pnxr_geope_20260726/d2_c125_k64_pnxr_geope.json"
MANIFEST = ROOT / "training_wss_min/preflight/d2_c125_k64_pnxr_geope_20260726_prepared.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if CONFIG.exists() or MANIFEST.exists():
        raise FileExistsError("refusing to replace D2 c125×k64 PNXR-GeoPE artifacts")
    parent = json.loads(PARENT.read_text(encoding="utf-8"))
    child = copy.deepcopy(parent)
    child["name"] = "pointnetpp_d2_c125_k64_pnxr_geope/outputs/d2_c125_k64_pnxr_geope"
    child["notes"] = (
        "Strict D2 c125×k64 architecture control: fixed AG/AAA split 106/0/27, "
        "random5000/SAME and all D2 SA settings frozen; only PointNeXt-R (1,1,0) "
        "and LocalGeoPE with abscissa/radius/curvature deltas are enabled."
    )
    child["model"]["sa_blocks"] = [1, 1, 0]
    child["model"]["local_geope"] = True
    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    CONFIG.write_text(json.dumps(child, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    cfg = ExpConfig.from_json(CONFIG)
    if tuple(cfg.model.sa_blocks) != (1, 1, 0) or not cfg.model.local_geope:
        raise RuntimeError("PointNeXt-R + LocalGeoPE architecture contract failed")
    if tuple(cfg.model.local_geope_feature_indices) != (3, 4, 5):
        raise RuntimeError("D2 LocalGeoPE must retain the three geometry attributes")
    if cfg.run_dir.exists():
        raise FileExistsError(f"run already exists: {cfg.run_dir}")

    row = {
        "experiment_id": "d2_c125_k64_pnxr_geope",
        "arm": "D2_c125_k64",
        "variant": "pointnext_r_local_geope",
        "seed": 1234,
        "config": str(CONFIG.resolve()),
        "sha256": sha(CONFIG),
        "parent_config": str(PARENT.resolve()),
        "parent_config_sha256": sha(PARENT),
        "run_dir": str(cfg.run_dir.resolve()),
        "split": str(Path(cfg.data.split_path).resolve()),
        "split_sha256": sha(Path(cfg.data.split_path)),
        "target_stats": str(Path(cfg.data.wss_stats_path).resolve()),
        "target_stats_sha256": sha(Path(cfg.data.wss_stats_path)),
        "feature_stats": None if cfg.data.feature_stats_path is None else str(Path(cfg.data.feature_stats_path).resolve()),
        "feature_stats_sha256": None if cfg.data.feature_stats_path is None else sha(Path(cfg.data.feature_stats_path)),
        "only_allowed_differences": ["name", "notes", "model.sa_blocks", "model.local_geope"],
    }
    MANIFEST.write_text(json.dumps({
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "prepared",
        "protocol": "D2_c125_k64_fixed_106_0_27_PointNeXtR_LocalGeoPE_single_seed_20260726",
        "parent": str(PARENT.resolve()),
        "parent_sha256": sha(PARENT),
        "frozen_contract": {
            "split": "AG/AAA 106/0/27",
            "sampling": "random5000/SAME",
            "sa_centers": [125, 125, 32],
            "sa_nsample": [64, 16, 16],
            "input_features": ["x", "y", "z", "abscissa_norm", "local_radius", "curvature"],
            "train": "seed1234; 400 epochs; train-loss selection",
        },
        "configs": [row],
        "expected_new_jobs": 1,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "prepared", "manifest": str(MANIFEST), "config": str(CONFIG)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
