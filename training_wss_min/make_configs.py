#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成第一版 baseline sweep 的配置 JSON。

三条正交的问题（都在标量 WSS + PointNeXt-残差 + 峰值收缩期上跑）：
  1. 点数-精度曲线：训练点数 {1000,1500,2000,3000,6000}，FPS，输入 xyz。
     评估恒在完整点云上 —— 回答"训练能省到多稀，全场精度还稳"。
  2. 特征消融：xyz / xyz+几何 / 纯几何(旋转不变)。纯几何一组能量出
     "配准坐标框架到底值多少分"（旋转不变特征对配准免疫）。
  3. 采样策略：FPS / random / 几何加权（狭窄+高曲率处更密）@2000 点。

w2000+xyz+fps 是三条曲线共用的锚点(canonical baseline)，只生成一次。
"""

from __future__ import annotations

from . import config as C

CONFIG_DIR = C.PROJECT_ROOT / "training_wss_min" / "configs"

XYZ = ("x", "y", "z")
GEOM = ("dist_to_wall", "abscissa_norm", "local_radius", "curvature")
XYZ_GEOM = XYZ + GEOM


def _base(name: str, notes: str) -> C.ExpConfig:
    c = C.ExpConfig(name=name, notes=notes)
    c.train.epochs = 400
    c.train.batch_cases = 8
    c.train.eval_every = 20
    c.data.num_workers = 4
    return c


def make_all() -> list[C.ExpConfig]:
    out: list[C.ExpConfig] = []

    # --- 1. 点数-精度曲线（xyz, fps）---
    for n in (1000, 1500, 2000, 3000, 6000):
        c = _base(f"pc_xyz_fps_w{n}_peak",
                  f"point-count sweep n={n}, xyz, fps, peak, scalar")
        c.data.wall_n_points = n
        c.data.sampling = "fps"
        c.data.input_features = XYZ
        out.append(c)

    # --- 2. 特征消融（fps 2000）; xyz 版即上面的 w2000，不重复 ---
    c = _base("feat_xyzgeom_fps_w2000_peak", "xyz + 几何特征, fps2000, peak")
    c.data.wall_n_points = 2000; c.data.sampling = "fps"; c.data.input_features = XYZ_GEOM
    out.append(c)

    c = _base("feat_geomonly_fps_w2000_peak",
              "纯旋转不变几何特征(无 xyz), fps2000 —— 量配准框架价值")
    c.data.wall_n_points = 2000; c.data.sampling = "fps"; c.data.input_features = GEOM
    out.append(c)

    # --- 3. 采样策略（xyz 2000）; fps 版即上面的 w2000 ---
    c = _base("samp_random_w2000_peak", "random 采样, xyz2000, peak")
    c.data.wall_n_points = 2000; c.data.sampling = "random"; c.data.input_features = XYZ
    out.append(c)

    c = _base("samp_geomw_w2000_peak", "几何加权采样(狭窄+高曲率更密), xyz2000, peak")
    c.data.wall_n_points = 2000; c.data.sampling = "geom_weighted"; c.data.input_features = XYZ
    c.data.geom_weight_curv = 1.0; c.data.geom_weight_invradius = 1.0
    out.append(c)

    return out


def main():
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    cfgs = make_all()
    for c in cfgs:
        p = c.to_json(CONFIG_DIR / f"{c.name}.json")
        print("wrote", p.name)
    # 生成一个清单，供 submit 脚本读取
    manifest = CONFIG_DIR / "sweep_baseline_v1.txt"
    manifest.write_text("\n".join(f"training_wss_min/configs/{c.name}.json" for c in cfgs) + "\n")
    print(f"\n{len(cfgs)} configs -> {manifest}")


if __name__ == "__main__":
    main()
