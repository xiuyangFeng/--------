#!/usr/bin/env python3
"""Prepare the xyz-only PointNeXt-R LocalGeoPE control."""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from training_wss_min.config import ExpConfig


ROOT = Path(__file__).resolve().parents[2]
PARENT = ROOT / "training_wss_min/configs/pointnetpp_xyz_input_ablation_20260724/s2_d2k64_pnxr_mixed_xyz.json"
CONFIG = ROOT / "training_wss_min/configs/pointnetpp_xyz_geope_20260725/s2_d2k64_pnxr_mixed_xyz_geope.json"
MANIFEST = ROOT / "training_wss_min/preflight/xyz_local_geope_20260725_prepared.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if CONFIG.exists() or MANIFEST.exists():
        raise FileExistsError("refusing to replace xyz-LocalGeoPE artifacts")
    parent = json.loads(PARENT.read_text(encoding="utf-8"))
    child = copy.deepcopy(parent)
    child["name"] = "pointnetpp_xyz_geope/outputs/s2_d2k64_pnxr_mixed_xyz_geope"
    child["notes"] = (
        "S2 xyz-only plus XYZ-LocalGeoPE: normalized relative xyz and distance only; "
        "no unavailable semantic geometry deltas."
    )
    child["model"]["local_geope"] = True
    child["model"]["local_geope_feature_indices"] = []
    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    CONFIG.write_text(json.dumps(child, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    cfg = ExpConfig.from_json(CONFIG)
    if tuple(cfg.data.input_features) != ("x", "y", "z") or not cfg.model.local_geope or cfg.model.local_geope_feature_indices:
        raise RuntimeError("xyz-only LocalGeoPE contract failed")
    if cfg.run_dir.exists():
        raise FileExistsError(f"run already exists: {cfg.run_dir}")
    row = {
        "experiment_id": "s2_d2k64_pnxr_mixed_xyz_geope", "arm": "S2", "variant": "xyz_local_geope",
        "seed": 1234, "config": str(CONFIG.resolve()), "sha256": sha(CONFIG),
        "parent_config": str(PARENT.resolve()), "parent_config_sha256": sha(PARENT),
        "run_dir": str(cfg.run_dir.resolve()), "split": str(Path(cfg.data.split_path).resolve()), "split_sha256": sha(Path(cfg.data.split_path)),
        "target_stats": str(Path(cfg.data.wss_stats_path).resolve()), "target_stats_sha256": sha(Path(cfg.data.wss_stats_path)),
        "feature_stats": str(Path(cfg.data.feature_stats_path).resolve()), "feature_stats_sha256": sha(Path(cfg.data.feature_stats_path)),
        "only_allowed_differences": ["name", "notes", "model.local_geope", "model.local_geope_feature_indices"],
    }
    MANIFEST.write_text(json.dumps({
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(), "status": "prepared",
        "protocol": "xyz_only_local_geope_single_seed_20260725",
        "formula": "edge=[(xyz_j-xyz_i)/r, ||(xyz_j-xyz_i)/r||]; semantic deltas disabled because input_features=xyz",
        "parent": str(PARENT.resolve()), "parent_sha256": sha(PARENT), "configs": [row], "expected_new_jobs": 1,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "prepared", "manifest": str(MANIFEST), "config": str(CONFIG)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
