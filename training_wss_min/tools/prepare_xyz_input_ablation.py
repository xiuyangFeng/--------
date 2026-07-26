#!/usr/bin/env python3
"""Prepare three xyz-only controls from their frozen D2-K64 parents."""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from training_wss_min.config import ExpConfig


REPO = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO / "training_wss_min/configs/pointnetpp_xyz_input_ablation_20260724"
MANIFEST = REPO / "training_wss_min/preflight/xyz_input_ablation_20260724_prepared.json"
XYZ = ["x", "y", "z"]
PARENTS = (
    (
        "d2_c125_k64_xyz",
        "D2 c125×k64 historical control: only input_features changes from xyz+geom to xyz.",
        REPO / "training_wss_min/configs/pointnetpp_sa1_scale_single_seed_20260721/d2_rand5000_c125_k64.json",
    ),
    (
        "m1_d2k64_mixed138_test36_xyz",
        "M1/S0 D2-K64 mixed138/test36 control: only input_features changes from xyz+geom to xyz.",
        REPO / "training_wss_min/configs/pointnetpp_d2_k64_ilo_structure_20260723/m1_d2k64_mixed138_test36.json",
    ),
    (
        "s2_d2k64_pnxr_mixed_xyz",
        "S2 D2-K64 + PointNeXt-R mixed138/test36 residual control: only input_features changes from xyz+geom to xyz.",
        REPO / "training_wss_min/configs/pointnetpp_d2_k64_ilo_structure_20260723/s2_d2k64_pnxr_mixed.json",
    ),
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    rows = []
    for experiment_id, note, parent_path in PARENTS:
        if not parent_path.is_file():
            raise FileNotFoundError(parent_path)
        parent = json.loads(parent_path.read_text(encoding="utf-8"))
        payload = copy.deepcopy(parent)
        payload["name"] = f"pointnetpp_xyz_input_ablation/outputs/{experiment_id}"
        payload["notes"] = note
        payload["data"]["input_features"] = XYZ
        config_path = CONFIG_DIR / f"{experiment_id}.json"
        dump(config_path, payload)
        cfg = ExpConfig.from_json(config_path)
        if tuple(cfg.data.input_features) != tuple(XYZ):
            raise RuntimeError(f"xyz features were not retained: {config_path}")
        if cfg.run_dir.exists():
            raise FileExistsError(f"refusing to overwrite existing run: {cfg.run_dir}")
        rows.append({
            "experiment_id": experiment_id,
            "parent_config": str(parent_path.resolve()),
            "parent_sha256": sha(parent_path),
            "config": str(config_path.resolve()),
            "sha256": sha(config_path),
            "run_dir": str(cfg.run_dir.resolve()),
            "seed": cfg.train.seed,
            "input_features": list(cfg.data.input_features),
        })

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "prepared",
        "protocol": "d2_k64_xyz_only_three_parent_controls",
        "contract": "Each child differs from its frozen parent only in name, notes, and data.input_features=[x,y,z].",
        "configs": rows,
        "expected_new_jobs": 3,
    }
    dump(MANIFEST, manifest)
    print(json.dumps({"status": "prepared", "manifest": str(MANIFEST), "configs": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
