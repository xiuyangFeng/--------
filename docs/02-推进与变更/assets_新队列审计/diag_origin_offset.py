#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""诊断 6 个原点错位病例：壁面网格 vs centerline 坐标系关系（只读）。

判断塌缩根因是「centerline 与壁面纯平移偏移（可修）」还是「二者几何不匹配（须弃）」：
  - 原生单位下各自 bbox/质心/对角线；
  - 质心偏移向量与模长；
  - 去掉质心偏移后，centerline 每点到壁面最近点的距离分布（小=形状吻合仅平移；大=不匹配）；
  - flow_divider 原点相对壁面 bbox 的位置（是否落在壁面外）。
"""
from __future__ import annotations
import sys, re
from pathlib import Path
import numpy as np

ROOT = Path("/public/newhome/cy/Digital_twin/GNN")
sys.path.insert(0, str(ROOT))
from pipeline_wss_min import raw_io, config as C
from pipeline_wss_min.preprocess import _resolve_unit_factor
from pipeline_wss_min.registration import _flow_divider_origin, _bifurcation_origin

CASES = [
    "ILO/WANG_LI_MIN-0/before", "ILO/WANG_LI_MIN-0/after",
    "ILO/ZHANG_MAO_JIN-0/before", "ILO/ZHANG_MAO_JIN-0/after",
    "AAA/ruputer/ZHOU_KE_XUN", "AAA/unruputer/LIU_WEN_QI",
    "ILO/AN_GUANG_JIE-0/after",   # 正常参照
]


def pfx(cd):
    for f in sorted((cd / "ascii").iterdir()):
        m = re.match(r"(.+)-(\d+)$", f.name)
        if m:
            return m.group(1)


def nn_dist(src, dst, cap=4000):
    """src 每点到 dst 最近点距离（dst 下采样到 cap）。"""
    if len(dst) > cap:
        idx = np.random.default_rng(0).choice(len(dst), cap, replace=False)
        dst = dst[idx]
    d = np.empty(len(src))
    for i in range(0, len(src), 512):
        blk = src[i:i+512]
        dd = np.sqrt(((blk[:, None, :] - dst[None, :, :]) ** 2).sum(-1))
        d[i:i+512] = dd.min(1)
    return d


def box(p):
    return p.min(0), p.max(0), p.mean(0), float(np.linalg.norm(p.max(0) - p.min(0)))


for u in CASES:
    cd = ROOT / "data_new" / u
    p = pfx(cd)
    steps = raw_io.list_timesteps(cd)
    wall = raw_io.read_wall_geometry(cd, p, steps[0])   # 原生单位
    cl = raw_io.read_centerline(cd)
    clc = cl["coords"]                                   # centerline 原生单位
    f, anom, ext = _resolve_unit_factor(wall, cl, C.DEFAULT.unit)
    wlo, whi, wctr, wdiag = box(wall)
    llo, lhi, lctr, ldiag = box(clc)
    off = lctr - wctr
    # centerline 去掉质心偏移后，与壁面(乘 factor? 注意 centerline 已是 mm，wall 原生)
    # 单位对齐：wall*factor -> mm 与 centerline(mm) 同尺度
    wall_mm = wall * f
    wlo2, whi2, wctr2, wdiag2 = box(wall_mm)
    off_mm = lctr - wctr2
    cl_shift = clc - off_mm            # 把 centerline 平移到壁面质心
    dmin = nn_dist(cl_shift, wall_mm)
    # flow_divider 原点是否落在壁面 bbox 内（mm）
    fd = _flow_divider_origin(cl, C.DEFAULT.registration)
    bf = _bifurcation_origin(cl, C.DEFAULT.registration)
    origin = fd if fd is not None else bf
    inside = None
    if origin is not None:
        inside = bool(np.all(origin >= wlo2 - 5) and np.all(origin <= whi2 + 5))

    print(f"\n{'='*70}\n{u}   unit_factor={f:.4g}  extent_mismatch={ext}")
    print(f"  壁面(mm)     bbox_center={np.round(wctr2,1)} diag={wdiag2:.1f}")
    print(f"  centerline(mm) bbox_center={np.round(lctr,1)} diag={ldiag:.1f}")
    print(f"  质心偏移(mm) = {np.round(off_mm,1)}  |offset|={np.linalg.norm(off_mm):.1f}  "
          f"(壁面对角线的 {100*np.linalg.norm(off_mm)/wdiag2:.0f}%)")
    print(f"  平移对齐后 centerline->壁面 最近点距离(mm): "
          f"median={np.median(dmin):.2f} p90={np.percentile(dmin,90):.2f} max={dmin.max():.2f}")
    print(f"  flow_divider原点={'找到' if fd is not None else '无'} "
          f"bifurcation原点={'找到' if bf is not None else '无'}  原点落在壁面bbox内(未平移)={inside}")
    if origin is not None:
        print(f"    原点(mm)={np.round(origin,1)}  距壁面质心={np.linalg.norm(origin-wctr2):.1f}mm")
