#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第四轮补充：B1 配方下 xyz+基础几何 点数—精度曲线配置。

仅横扫 wall_n_points；2000/4000 复用已有 B1/B3，本脚本只生成新增四点。
"""

from __future__ import annotations

from . import config as C
from .make_configs_round4 import FEATURES, DEV1_SPLIT, DEV1_STATS, _base

CONFIG_DIR = C.PROJECT_ROOT / "training_wss_min" / "configs"

# 首轮只新增这些；2000=B1、4000=B3 已存在
NEW_POINTS = (1000, 1500, 3000, 6000)
ALL_POINTS = (1000, 1500, 2000, 3000, 4000, 6000)
# 第二阶段：峰值 1000 vs 锚点 2000 的多 seed 确认
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
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    cfgs = make_new()
    for c in cfgs:
        C.validate_features(c)
        print("wrote", c.to_json(CONFIG_DIR / f"{c.name}.json").name)

    confirm = make_multiseed_confirm()
    for c in confirm:
        C.validate_features(c)
        print("wrote", c.to_json(CONFIG_DIR / f"{c.name}.json").name)

    manifest = CONFIG_DIR / "sweep_round4_pointcount_s1234.txt"
    lines = [f"training_wss_min/configs/{c.name}.json" for c in cfgs]
    manifest.write_text("\n".join(lines) + "\n")

    # 完整 6 点对照表（含复用锚点），便于汇总脚本引用
    map_path = CONFIG_DIR / "sweep_round4_pointcount_map_s1234.txt"
    reuse = {
        2000: "r4_dev1_b1_tgtw_fixedq_s1234",
        4000: "r4_dev1_b3_fps4000_s1234",
    }
    map_lines = []
    for n in ALL_POINTS:
        name = reuse.get(n, f"r4_dev1_pc_xyzgeom_fps{n}_s1234")
        map_lines.append(f"{n}\t{name}\ttraining_wss_min/configs/{name}.json")
    map_path.write_text("\n".join(map_lines) + "\n")

    # 多 seed 确认清单：1000 新跑 + 2000 用 B1
    confirm_manifest = CONFIG_DIR / "sweep_round4_pointcount_multiseed.txt"
    confirm_lines = [
        f"training_wss_min/configs/{c.name}.json" for c in confirm
    ] + [
        "training_wss_min/configs/r4_dev1_b1_tgtw_fixedq_s7.json",
        "training_wss_min/configs/r4_dev1_b1_tgtw_fixedq_s2025.json",
    ]
    confirm_manifest.write_text("\n".join(confirm_lines) + "\n")

    print(f"\n{len(cfgs)} curve configs -> {manifest}")
    print(f"6-point map -> {map_path}")
    print(f"{len(confirm_lines)} multiseed confirm -> {confirm_manifest}")
    print(f"split={DEV1_SPLIT}")
    print(f"stats={DEV1_STATS}")


if __name__ == "__main__":
    main()
