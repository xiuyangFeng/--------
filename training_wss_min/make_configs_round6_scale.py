#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate the pre-registered Round-6 A/B/D scale-diagnostic configs.

Only ``data.input_features``, run name, note and model seed may differ.  Every
other field is copied from the frozen Round-4 B1/dev1 protocol so the feature
comparison is attributable to the inputs rather than training drift.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
CONFIG_ROOT = ROOT / "training_wss_min" / "configs"
OUT_DIR = CONFIG_ROOT / "round6"
BASE = CONFIG_ROOT / "r4_dev1_b1_tgtw_fixedq_s1234.json"
SEEDS = (1234, 7, 2025)
GROUPS = {
    "A_xyz": {
        "features": ["x", "y", "z"],
        "description": "A: per-case normalized xyz only",
    },
    "B_xyzscale": {
        "features": ["x", "y", "z", "coord_scale"],
        "description": "B: per-case normalized xyz plus case physical scale",
    },
    "D_xyzgeom": {
        "features": ["x", "y", "z", "abscissa_norm", "local_radius", "curvature"],
        "description": "D: per-case normalized xyz plus engineered geometry control",
    },
}


def main() -> None:
    base = json.loads(BASE.read_text(encoding="utf-8"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    for group, spec in GROUPS.items():
        for seed in SEEDS:
            cfg = copy.deepcopy(base)
            name = f"r6_scale_{group}_s{seed}"
            cfg["name"] = name
            cfg["notes"] = (
                "r6 scale diagnostic; " + spec["description"]
                + f"; frozen B1/dev1/FPS2000 protocol; val-only; seed={seed}"
            )
            cfg["data"]["input_features"] = spec["features"]
            cfg["train"]["seed"] = seed
            path = OUT_DIR / f"{name}.json"
            path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            manifest.append(path.relative_to(ROOT).as_posix())

    manifest_path = CONFIG_ROOT / "sweep_round6_scale_abd.txt"
    manifest_path.write_text("\n".join(manifest) + "\n", encoding="utf-8")

    protocol = {
        "protocol_id": "r6_scale_abd_v1",
        "status": "PRE_REGISTERED",
        "purpose": "Diagnose whether per-case coordinate normalization makes xyz-only comparison unfair by removing physical scale.",
        "groups": GROUPS,
        "seeds": list(SEEDS),
        "control_config": str(BASE.relative_to(ROOT)),
        "invariants": {
            "split": base["data"]["split_path"],
            "wss_stats": base["data"]["wss_stats_path"],
            "partition": "val",
            "wall_n_points": base["data"]["wall_n_points"],
            "sampling": base["data"]["sampling"],
            "timesteps": base["data"]["timesteps"],
            "model": base["model"],
            "train_except_seed": {k: v for k, v in base["train"].items() if k != "seed"},
        },
        "primary_metrics": ["r2_field_raw", "r2_casemean"],
        "guardrails": ["top10_pred_true_ratio", "top10_iou", "negative_r2_count"],
        "decision_rules": {
            "scale_signal": "B-A > 0.02 on both primary metrics using three-seed means",
            "geometry_increment": "D-B on both primary metrics; report mean and seed variability without single-seed ranking",
            "test_policy": "No legacy test16 access during this diagnostic",
        },
    }
    (OUT_DIR / "scale_diagnostic_protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"generated {len(manifest)} configs -> {manifest_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
