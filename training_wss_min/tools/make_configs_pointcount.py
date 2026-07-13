#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""点数—精度曲线配置（B1 配方下 xyz+基础几何）。

仅横扫 wall_n_points；2000/4000 复用已有 B1/B3，本脚本只生成新增四点。
"""

from __future__ import annotations

from training_wss_min import config as C
from training_wss_min.tools.config_paths import SWEEPS_DIR, config_path, repo_config_path
from training_wss_min.tools.make_configs_protocol_gates import (
    DEV1_SPLIT,
    DEV1_STATS,
    FEATURES,
    _base,
)

NEW_POINTS = (1000, 1500, 3000, 6000)
ALL_POINTS = (1000, 1500, 2000, 3000, 4000, 6000)
CONFIRM_SEEDS = (7, 2025)


def _pc(n: int, seed: int) -> C.ExpConfig:
    c = _base(
        f"r4_dev1_pc_xyzgeom_fps{n}_s{seed}",
        f"r4 PC: B1 recipe fixed FPS {n}; xyz+geom; seed={seed}",
        seed,
    )
    c.data.wall_n_points = n
    c.data.sampling = "fps"
    c.data.input_features = FEATURES
    return c


def make_new() -> list[C.ExpConfig]:
    return [_pc(n, 1234) for n in NEW_POINTS]


def make_multiseed_confirm() -> list[C.ExpConfig]:
    """仅生成 1000 点的 seed7/2025；2000 复用已有 B1 配置。"""
    return [_pc(1000, seed) for seed in CONFIRM_SEEDS]


def main():
    cfgs = make_new()
    for c in cfgs:
        C.validate_features(c)
        path = config_path(c.name)
        path.parent.mkdir(parents=True, exist_ok=True)
        print("wrote", c.to_json(path).name)

    confirm = make_multiseed_confirm()
    for c in confirm:
        C.validate_features(c)
        path = config_path(c.name)
        path.parent.mkdir(parents=True, exist_ok=True)
        print("wrote", c.to_json(path).name)

    SWEEPS_DIR.mkdir(parents=True, exist_ok=True)
    manifest = SWEEPS_DIR / "pointcount_s1234.txt"
    manifest.write_text("\n".join(repo_config_path(c.name) for c in cfgs) + "\n")

    map_path = SWEEPS_DIR / "pointcount_map_s1234.txt"
    reuse = {
        2000: "r4_dev1_b1_tgtw_fixedq_s1234",
        4000: "r4_dev1_b3_fps4000_s1234",
    }
    map_lines = []
    for n in ALL_POINTS:
        name = reuse.get(n, f"r4_dev1_pc_xyzgeom_fps{n}_s1234")
        map_lines.append(f"{n}\t{name}\t{repo_config_path(name)}")
    map_path.write_text("\n".join(map_lines) + "\n")

    confirm_manifest = SWEEPS_DIR / "pointcount_multiseed.txt"
    confirm_lines = [repo_config_path(c.name) for c in confirm] + [
        repo_config_path("r4_dev1_b1_tgtw_fixedq_s7"),
        repo_config_path("r4_dev1_b1_tgtw_fixedq_s2025"),
    ]
    confirm_manifest.write_text("\n".join(confirm_lines) + "\n")

    print(f"\n{len(cfgs)} curve configs -> {manifest}")
    print(f"6-point map -> {map_path}")
    print(f"{len(confirm_lines)} multiseed confirm -> {confirm_manifest}")
    print(f"split={DEV1_SPLIT}")
    print(f"stats={DEV1_STATS}")


if __name__ == "__main__":
    main()
