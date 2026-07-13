#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate Track A (wall pressure) multitarget configs.

§14 of the round-6 plan: same frozen minimal protocol (FPS-2000 / PointNeXt-S /
dev1 / B1 schedule) but predict wall *gauge* pressure instead of WSS, over the
``xyz`` vs ``xyz+geom`` input matrix.  Only target, stats path, input features,
loss weighting, name/note and seed differ from the B1 control.

Differences from the WSS control that are intentional and pre-registered:
  * ``data.target = "pressure"`` -> dataset reads wall_pressure, per-case demean.
  * ``data.wss_stats_path`` -> gauge-pressure linear stats (build first via
    ``python -m training_wss_min.tools.make_pressure_stats``).
  * ``train.loss_weight_target = False`` -> plain MSE; the WSS tail-weighting
    (alpha=2) is WSS-specific and off for pressure.
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
OUT_DIR = CONFIG_ROOT / "multitarget"
BASE = config_path("r4_dev1_b1_tgtw_fixedq_s1234")
PRESSURE_STATS = str(
    C.PROJECT_ROOT / "data_wss_min" / "fold_stats" / "pressure_gauge_stats_v2_dev1.json"
)
SEEDS = (1234, 7, 2025)
GROUPS = {
    "press_wall_xyz": {
        "features": ["x", "y", "z"],
        "description": "pressure(wall gauge); per-case normalized xyz only",
    },
    "press_wall_xyzgeom": {
        "features": ["x", "y", "z", "abscissa_norm", "local_radius", "curvature"],
        "description": "pressure(wall gauge); xyz plus engineered geometry",
    },
}


def main() -> None:
    base = json.loads(BASE.read_text(encoding="utf-8"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SWEEPS_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    for group, spec in GROUPS.items():
        for seed in SEEDS:
            cfg = copy.deepcopy(base)
            name = f"r6_{group}_s{seed}"
            cfg["name"] = name
            cfg["notes"] = (
                "r6 multitarget Track A; " + spec["description"]
                + "; gauge=per-case demeaned wall pressure; plain MSE (no target-weight); "
                + f"frozen B1/dev1/FPS2000 protocol; val-only; seed={seed}"
            )
            cfg["data"]["input_features"] = spec["features"]
            cfg["data"]["target"] = "pressure"
            cfg["data"]["wss_stats_path"] = PRESSURE_STATS
            cfg["train"]["seed"] = seed
            # Turn off WSS tail-weighting; pressure uses plain MSE.
            cfg["train"]["loss_weight_target"] = False
            path = config_path(name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            manifest.append(repo_config_path(name))

    manifest_path = SWEEPS_DIR / "pressure_wall.txt"
    manifest_path.write_text("\n".join(manifest) + "\n", encoding="utf-8")

    protocol = {
        "protocol_id": "multitarget_pressure_wall_v1",
        "status": "PRE_REGISTERED",
        "purpose": "Track A of round-6 §14: predict wall gauge pressure under the "
                   "frozen minimal protocol, xyz vs xyz+geom.",
        "target": "pressure",
        "target_definition": "per-case demeaned wall_pressure at peak step (gauge, Pa)",
        "stats": PRESSURE_STATS,
        "groups": GROUPS,
        "seeds": list(SEEDS),
        "control_config": repo_config_path("r4_dev1_b1_tgtw_fixedq_s1234"),
        "invariants": {
            "split": base["data"]["split_path"],
            "partition": "val",
            "wall_n_points": base["data"]["wall_n_points"],
            "sampling": base["data"]["sampling"],
            "timesteps": base["data"]["timesteps"],
            "model": base["model"],
            "loss": "mse (loss_weight_target=false)",
        },
        "primary_metrics": ["r2_field_raw", "r2_casemean"],
        "notes": "gauge pressure can be negative; eval does not clip. WSS "
                 "tail-weighting off. Compare xyz vs xyz+geom like scale A/D.",
    }
    (OUT_DIR / "pressure_wall_protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"generated {len(manifest)} configs -> {manifest_path.relative_to(ROOT)}")
    for m in manifest:
        print("  " + m)


if __name__ == "__main__":
    main()
