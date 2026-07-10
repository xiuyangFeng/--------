#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第四轮 Stage B/C 配置生成器（主开发划分 = v2_dev1）。"""

from __future__ import annotations

from pathlib import Path

from . import config as C

CONFIG_DIR = C.PROJECT_ROOT / "training_wss_min" / "configs"
FEATURES = ("x", "y", "z", "abscissa_norm", "local_radius", "curvature")
SEEDS = (1234, 7, 2025)
DEV1_SPLIT = str(C.PROJECT_ROOT / "training" / "splits" / "split_AG_wss_min_v2_dev1.json")
DEV1_STATS = str(C.DATA_ROOT / "fold_stats" / "wss_stats_v2_dev1.json")


def _base(name: str, notes: str, seed: int) -> C.ExpConfig:
    c = C.ExpConfig(name=name, notes=notes)
    c.data.split_path = DEV1_SPLIT
    c.data.wss_stats_path = DEV1_STATS
    c.data.num_workers = 4
    c.data.persistent_workers = False
    c.data.wall_n_points = 2000
    c.data.sampling = "fps"
    c.data.resample_each_epoch = True
    c.data.rot_aug = False
    c.data.input_features = FEATURES
    c.data.curvature_transform = "signed_log1p"
    c.train.epochs = 160
    c.train.batch_cases = 8
    c.train.eval_every = 10
    c.train.seed = seed
    c.train.selection_rule = "r4_composite_v1"
    c.train.ckpt_top_k = 3
    c.train.early_stop_patience = 6
    c.train.min_epoch = 40
    c.train.loss_weight_target = True
    c.train.loss_weight_target_alpha = 2.0
    c.train.loss_weight_fixed_quantiles = True
    c.train.loss_raw_huber_lambda = 0.0
    return c


def make_all() -> list[C.ExpConfig]:
    out: list[C.ExpConfig] = []
    # B0: 第三轮 tgtw 行为对照 —— batch 分位（固定阈值关闭）
    for seed in SEEDS:
        c = _base(
            f"r4_dev1_b0_tgtw_batchq_s{seed}",
            f"r4 B0 control: tgtw a=2 batch-quantile; v2_dev1; seed={seed}",
            seed,
        )
        c.train.loss_weight_fixed_quantiles = False
        out.append(c)
    # B1: 固定 train-global 阈值
    for seed in SEEDS:
        c = _base(
            f"r4_dev1_b1_tgtw_fixedq_s{seed}",
            f"r4 B1: tgtw a=2 fixed train quantiles; v2_dev1; seed={seed}",
            seed,
        )
        c.train.loss_weight_fixed_quantiles = True
        out.append(c)
    # B2: multi-start FPS（默认搭在 B1 配方上；提交时按胜者替换）
    for seed in SEEDS:
        c = _base(
            f"r4_dev1_b2_fpsms_s{seed}",
            f"r4 B2: fps_multistart pool=8 on B1 recipe; seed={seed}",
            seed,
        )
        c.data.sampling = "fps_multistart"
        c.data.fps_pool_size = 8
        out.append(c)
    # B3: fixed FPS 4000
    for seed in SEEDS:
        c = _base(
            f"r4_dev1_b3_fps4000_s{seed}",
            f"r4 B3: fixed FPS 4000 on B1 recipe; seed={seed}",
            seed,
        )
        c.data.wall_n_points = 4000
        c.data.sampling = "fps"
        out.append(c)
    # C1: raw Huber λ=0.05
    for seed in SEEDS:
        c = _base(
            f"r4_dev1_c1_rawhuber_s{seed}",
            f"r4 C1: log-MSE + raw Huber λ=0.05 on B1; seed={seed}",
            seed,
        )
        c.train.loss_raw_huber_lambda = 0.05
        out.append(c)
    # C2: alpha sweep
    for alpha in (1.0, 4.0):
        c = _base(
            f"r4_dev1_c2_alpha{int(alpha)}_s1234",
            f"r4 C2: fixedq tgtw alpha={alpha}; seed=1234",
            1234,
        )
        c.train.loss_weight_target_alpha = alpha
        out.append(c)
    # C4: coord_scale
    c = _base(
        "r4_dev1_c4_coordscale_s1234",
        "r4 C4: add coord_scale feature; seed=1234",
        1234,
    )
    c.data.input_features = FEATURES + ("coord_scale",)
    out.append(c)
    # C5: radius_gradient
    c = _base(
        "r4_dev1_c5_radiusgrad_s1234",
        "r4 C5: add radius_gradient; seed=1234",
        1234,
    )
    c.data.input_features = FEATURES + ("radius_gradient",)
    out.append(c)
    return out


def main():
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    cfgs = make_all()
    for c in cfgs:
        C.validate_features(c)
        print("wrote", c.to_json(CONFIG_DIR / f"{c.name}.json").name)
    # 首批只提交 B0 seed1234
    manifest_b0 = CONFIG_DIR / "sweep_round4_b0_s1234.txt"
    manifest_b0.write_text(
        "training_wss_min/configs/r4_dev1_b0_tgtw_batchq_s1234.json\n"
    )
    manifest_all = CONFIG_DIR / "sweep_round4_all.txt"
    manifest_all.write_text(
        "\n".join(f"training_wss_min/configs/{c.name}.json" for c in cfgs) + "\n"
    )
    print(f"\n{len(cfgs)} configs; first submit: {manifest_b0}")


if __name__ == "__main__":
    main()
