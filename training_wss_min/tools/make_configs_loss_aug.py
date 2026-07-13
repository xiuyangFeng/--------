#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第二轮 sweep 配置：以 xyz+几何 为默认，专打第一轮暴露的问题。

第一轮结论：
- 几何特征是压倒性杠杆（xyz+几何 ≈ 纯几何 ≫ 纯 xyz），故第二轮默认 xyz+几何。
- 狭窄区/高 WSS 区 R² 全负（peak 欠拟合）→ 用几何加权 loss / 目标幅值加权 loss 直接补。
- 单种子方差大（w1500 异常）→ 对最优配置跑多种子量方差。
- val/test gap 大 → 加旋转增广（几何/标签旋转不变，只增广 xyz 朝向）。

共 9 个 config，均 xyz+几何 / fps2000 / 峰值 / 标量 / 400 epoch。
"""
from __future__ import annotations

from training_wss_min import config as C
from training_wss_min.tools.config_paths import SWEEPS_DIR, config_path, repo_config_path

XYZ_GEOM = ("x", "y", "z", "dist_to_wall", "abscissa_norm", "local_radius", "curvature")


def _base(name: str, notes: str, seed: int = 1234) -> C.ExpConfig:
    c = C.ExpConfig(name=name, notes=notes)
    c.train.epochs = 400
    c.train.batch_cases = 8
    c.train.eval_every = 20
    c.train.seed = seed
    c.data.num_workers = 4
    c.data.wall_n_points = 2000
    c.data.sampling = "fps"
    c.data.input_features = XYZ_GEOM
    return c


def make_all() -> list[C.ExpConfig]:
    out: list[C.ExpConfig] = []

    # --- loss 消融（xyz+几何, fps2000, seed 1234）---
    out.append(_base("r2_xyzgeom_mse", "xyz+几何 参考(纯 MSE)"))

    c = _base("r2_xyzgeom_huber", "xyz+几何 + Huber(抗离群)")
    c.train.loss = "huber"; c.train.huber_delta = 1.0
    out.append(c)

    c = _base("r2_xyzgeom_geomwloss", "xyz+几何 + 几何加权 loss(狭窄/高曲率更重)")
    c.train.loss_geom_weight = True
    out.append(c)

    c = _base("r2_xyzgeom_tgtwloss", "xyz+几何 + 目标幅值加权 loss(直接补高 WSS 区)")
    c.train.loss_weight_target = True; c.train.loss_weight_target_alpha = 2.0
    out.append(c)

    c = _base("r2_xyzgeom_geomw_tgtw", "xyz+几何 + 几何加权 + 目标加权 loss")
    c.train.loss_geom_weight = True
    c.train.loss_weight_target = True; c.train.loss_weight_target_alpha = 2.0
    out.append(c)

    # --- 采样 × 目标加权（狭窄处更密 + peak 更重）---
    c = _base("r2_xyzgeom_geomwsamp_tgtw", "几何加权采样 + 目标加权 loss")
    c.data.sampling = "geom_weighted"
    c.data.geom_weight_curv = 1.0; c.data.geom_weight_invradius = 1.0
    c.train.loss_weight_target = True; c.train.loss_weight_target_alpha = 2.0
    out.append(c)

    # --- 旋转增广（缩小 val/test gap）---
    c = _base("r2_xyzgeom_rotaug", "xyz+几何 + 训练期随机旋转增广")
    c.data.rot_aug = True
    out.append(c)

    # --- 多种子（对目标加权配置量方差；seed 1234 版即 r2_xyzgeom_tgtwloss）---
    for sd in (2025, 7):
        c = _base(f"r2_xyzgeom_tgtw_s{sd}", f"目标加权 loss, seed={sd}(方差)", seed=sd)
        c.train.loss_weight_target = True; c.train.loss_weight_target_alpha = 2.0
        out.append(c)

    return out


def main():
    cfgs = make_all()
    for c in cfgs:
        path = config_path(c.name)
        path.parent.mkdir(parents=True, exist_ok=True)
        print("wrote", c.to_json(path).name)
    SWEEPS_DIR.mkdir(parents=True, exist_ok=True)
    manifest = SWEEPS_DIR / "loss_aug_v1.txt"
    manifest.write_text("\n".join(repo_config_path(c.name) for c in cfgs) + "\n")
    print(f"\n{len(cfgs)} configs -> {manifest}")


if __name__ == "__main__":
    main()
