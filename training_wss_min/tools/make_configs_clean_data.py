#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第三轮 clean-data 主矩阵配置。

前置条件：
- split_AG_wss_min_v1 已将 slow/ZHANG_HUAN_LI 移入 excluded；
- data_wss_min/wss_global_stats.json 已按 clean train peak-only 重算；
- included 77 例 bundle 已通过 nodenumber 对齐守卫和 QA gate。
"""

from __future__ import annotations

from training_wss_min import config as C
from training_wss_min.tools.config_paths import SWEEPS_DIR, config_path, repo_config_path

FEATURES = ("x", "y", "z", "abscissa_norm", "local_radius", "curvature")
SEEDS = (1234, 7, 2025)


def _base(name: str, notes: str, seed: int) -> C.ExpConfig:
    c = C.ExpConfig(name=name, notes=notes)
    c.train.epochs = 240
    c.train.batch_cases = 8
    c.train.eval_every = 10
    c.train.seed = seed
    c.data.num_workers = 4
    c.data.wall_n_points = 2000
    c.data.sampling = "fps"
    c.data.resample_each_epoch = True
    c.data.rot_aug = False
    c.data.input_features = FEATURES
    c.data.curvature_transform = "signed_log1p"
    return c


def make_all() -> list[C.ExpConfig]:
    out: list[C.ExpConfig] = []
    for seed in SEEDS:
        out.append(_base(
            f"r3_clean_xyzgeom_mse_s{seed}",
            f"round3 clean peak stats; xyz+geom(no dist_to_wall); mse; seed={seed}",
            seed,
        ))
    for seed in SEEDS:
        c = _base(
            f"r3_clean_xyzgeom_tgtw_s{seed}",
            f"round3 clean peak stats; xyz+geom(no dist_to_wall); target-weight alpha=2; seed={seed}",
            seed,
        )
        c.train.loss_weight_target = True
        c.train.loss_weight_target_alpha = 2.0
        out.append(c)
    return out


def main():
    cfgs = make_all()
    for c in cfgs:
        path = config_path(c.name)
        path.parent.mkdir(parents=True, exist_ok=True)
        print("wrote", c.to_json(path).name)
    SWEEPS_DIR.mkdir(parents=True, exist_ok=True)
    manifest = SWEEPS_DIR / "clean_data_v1.txt"
    manifest.write_text("\n".join(repo_config_path(c.name) for c in cfgs) + "\n")
    print(f"\n{len(cfgs)} configs -> {manifest}")


if __name__ == "__main__":
    main()
