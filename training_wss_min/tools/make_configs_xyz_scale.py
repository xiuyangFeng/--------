#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate XYZ scale-diagnostic configs (A/B/D feature groups).

Only ``data.input_features``, run name, note and model seed may differ.  Every
other field is copied from the frozen B1/dev1 protocol so the feature
comparison is attributable to the inputs rather than training drift.

Historical ``name`` values stay ``r6_scale_*`` so existing runs/ stay linked;
filenames are semantic under ``configs/xyz_scale_diag/``.
"""

from __future__ import annotations

import copy
import json

from training_wss_min import config as C
from training_wss_min.tools.config_paths import (
    CONFIG_ROOT,
    SWEEPS_DIR,
    config_path,
    repo_config_path,
)

ROOT = C.PROJECT_ROOT
OUT_DIR = CONFIG_ROOT / "xyz_scale_diag"
BASE = config_path("r4_dev1_b1_tgtw_fixedq_s1234")
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
    "E_xyzscalegeom": {
        "features": ["x", "y", "z", "coord_scale",
                     "abscissa_norm", "local_radius", "curvature"],
        "description": "E: per-case normalized xyz plus case physical scale plus engineered geometry (nested-factor cell for interaction E-D-B+A)",
    },
}
# 只需要重新提交的增量组（A/B/D 已完成三 seed；E 补齐 2x2 因子）。
INCREMENTAL_GROUPS = ("E_xyzscalegeom",)


def main() -> None:
    base = json.loads(BASE.read_text(encoding="utf-8"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SWEEPS_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    for group, spec in GROUPS.items():
        for seed in SEEDS:
            cfg = copy.deepcopy(base)
            # Keep historical name for runs/ compatibility.
            name = f"r6_scale_{group}_s{seed}"
            cfg["name"] = name
            cfg["notes"] = (
                "xyz scale diagnostic; " + spec["description"]
                + f"; frozen B1/dev1/FPS2000 protocol; val-only; seed={seed}"
            )
            cfg["data"]["input_features"] = spec["features"]
            cfg["train"]["seed"] = seed
            path = config_path(name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            manifest.append(repo_config_path(name))

    manifest_path = SWEEPS_DIR / "xyz_scale_abd.txt"
    manifest_path.write_text("\n".join(manifest) + "\n", encoding="utf-8")

    # 增量 manifest：只列尚未跑过的组，避免重复提交已完成的 A/B/D。
    incr = [
        repo_config_path(f"r6_scale_{g}_s{s}")
        for g in INCREMENTAL_GROUPS
        for s in SEEDS
    ]
    incr_path = SWEEPS_DIR / "xyz_scale_e.txt"
    incr_path.write_text("\n".join(incr) + "\n", encoding="utf-8")

    protocol = {
        "protocol_id": "xyz_scale_abde_v1",
        "status": "PRE_REGISTERED",
        "purpose": "Diagnose scale vs geometry as a 2x2 nested-factor design; A/B/D/E give the strict attributions B-A, E-B, E-D and interaction E-D-B+A.",
        "factor_design": {
            "A": {"scale": 0, "geom": 0},
            "B": {"scale": 1, "geom": 0},
            "D": {"scale": 0, "geom": 1},
            "E": {"scale": 1, "geom": 1},
            "contrasts": {
                "scale_main_effect": "B-A",
                "scale_given_geom": "E-D",
                "geom_main_effect": "D-A",
                "geom_given_scale": "E-B",
                "interaction": "E-D-B+A",
            },
        },
        "groups": GROUPS,
        "seeds": list(SEEDS),
        "control_config": repo_config_path("r4_dev1_b1_tgtw_fixedq_s1234"),
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
        "primary_metrics": ["r2_field_casebalanced", "r2_casemean", "r2_field_raw"],
        "guardrails": ["top10_pred_true_ratio", "top10_iou", "negative_r2_count"],
        "decision_rules": {
            "scale_signal": "B-A > 0.02 on both primary metrics using three-seed means",
            "geometry_increment": "D-B on both primary metrics; report mean and seed variability without single-seed ranking",
            "interaction": "report E-D-B+A with paired seed differences; do not claim scale+geom synergy unless interaction is positive beyond seed noise",
            "test_policy": "No legacy test16 access during this diagnostic",
        },
    }
    (OUT_DIR / "scale_diagnostic_protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"generated {len(manifest)} configs -> {manifest_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
