#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 baseline_sweep 配置（点数 / 特征 / 采样正交扫）。

三条正交的问题（都在标量 WSS + PointNeXt-残差 + 峰值收缩期上跑）：
  1. 点数-精度曲线：训练点数 {1000,1500,2000,3000,6000}，FPS，输入 xyz。
  2. 特征消融：xyz / xyz+几何 / 纯几何(旋转不变)。
  3. 采样策略：FPS / random / 几何加权（狭窄+高曲率处更密）@2000 点。

w2000+xyz+fps 是三条曲线共用的锚点(canonical baseline)，只生成一次。
"""

from __future__ import annotations

from training_wss_min import config as C
from training_wss_min.tools.config_paths import SWEEPS_DIR, config_path, repo_config_path

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

    for n in (1000, 1500, 2000, 3000, 6000):
        c = _base(f"pc_xyz_fps_w{n}_peak",
                  f"point-count sweep n={n}, xyz, fps, peak, scalar")
        c.data.wall_n_points = n
        c.data.sampling = "fps"
        c.data.input_features = XYZ
        out.append(c)

    c = _base("feat_xyzgeom_fps_w2000_peak", "xyz + 几何特征, fps2000, peak")
    c.data.wall_n_points = 2000
    c.data.sampling = "fps"
    c.data.input_features = XYZ_GEOM
    out.append(c)

    c = _base("feat_geomonly_fps_w2000_peak",
              "纯旋转不变几何特征(无 xyz), fps2000 —— 量配准框架价值")
    c.data.wall_n_points = 2000
    c.data.sampling = "fps"
    c.data.input_features = GEOM
    out.append(c)

    c = _base("samp_random_w2000_peak", "random 采样, xyz2000, peak")
    c.data.wall_n_points = 2000
    c.data.sampling = "random"
    c.data.input_features = XYZ
    out.append(c)

    c = _base("samp_geomw_w2000_peak", "几何加权采样(狭窄+高曲率更密), xyz2000, peak")
    c.data.wall_n_points = 2000
    c.data.sampling = "geom_weighted"
    c.data.input_features = XYZ
    c.data.geom_weight_curv = 1.0
    c.data.geom_weight_invradius = 1.0
    out.append(c)

    return out


def main():
    cfgs = make_all()
    for c in cfgs:
        path = config_path(c.name)
        path.parent.mkdir(parents=True, exist_ok=True)
        print("wrote", c.to_json(path).name)
    SWEEPS_DIR.mkdir(parents=True, exist_ok=True)
    manifest = SWEEPS_DIR / "baseline_v1.txt"
    manifest.write_text("\n".join(repo_config_path(c.name) for c in cfgs) + "\n")
    print(f"\n{len(cfgs)} configs -> {manifest}")


if __name__ == "__main__":
    main()
