#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""数值化两类可视化异常（只读）：
  1) 离群点塌缩：outlier_ratio = max|aligned| / p99|aligned|，大 => 有远离群点，
     会把 per-case 归一化(max|.|) 撑爆、真实解剖被压成一点。
  2) 主轴歪斜：aligned 点云 PCA 第一主轴与 Z(主轴目标) 的夹角，大 => 没对到竖直。
"""
from __future__ import annotations
import sys, re
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path("/public/newhome/cy/Digital_twin/GNN")
sys.path.insert(0, str(ROOT))
from pipeline_wss_min import raw_io, config as C
from pipeline_wss_min.preprocess import _resolve_unit_factor
from pipeline_wss_min.registration import compute_transform

SCR = Path("/tmp/claude-1006/-public-newhome-cy-Digital-twin-GNN/"
           "c63e71da-acd6-4ca8-b8c0-20257aea34aa/scratchpad")
TI = pd.read_csv(SCR / "probe_registration_tiered.csv")
UNITS = TI[TI["status"] == "ok"]["unit_id"].tolist()


def prefix(cd):
    for f in sorted((cd / "ascii").iterdir()):
        m = re.match(r"(.+)-(\d+)$", f.name)
        if m: return m.group(1)


def metrics(u):
    cd = ROOT / "data_new" / u
    pfx = prefix(cd)
    steps = raw_io.list_timesteps(cd)
    wall = raw_io.read_wall_geometry(cd, pfx, steps[0])
    cl = raw_io.read_centerline(cd)
    f, _, _ = _resolve_unit_factor(wall, cl, C.DEFAULT.unit)
    T = compute_transform(wall * f, wall * f, cl, C.DEFAULT.registration)
    a = (wall * f - T.centroid) @ T.rotation
    r = np.linalg.norm(a, axis=1)
    p99 = np.percentile(r, 99)
    outlier_ratio = float(r.max() / p99) if p99 > 0 else float("inf")
    # PCA 主轴与 Z 夹角
    c = a - a.mean(0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    v = vt[0]
    tilt_deg = float(np.degrees(np.arccos(min(1.0, abs(v[2])))))
    return outlier_ratio, tilt_deg


rows = []
for i, u in enumerate(UNITS, 1):
    try:
        orat, tilt = metrics(u)
        rows.append({"unit_id": u, "outlier_ratio": round(orat, 2), "tilt_deg": round(tilt, 1)})
    except Exception as e:
        rows.append({"unit_id": u, "outlier_ratio": "err", "tilt_deg": str(e)[:40]})
    if i % 30 == 0: print(f"  {i}/{len(UNITS)}", flush=True)

df = pd.DataFrame(rows)
df.to_csv(SCR / "viz_numeric_flags.csv", index=False)
num = pd.to_numeric(df["outlier_ratio"], errors="coerce")
tilt = pd.to_numeric(df["tilt_deg"], errors="coerce")
print("\n=== 离群点塌缩 (outlier_ratio > 3；正常应 ~1.1-1.5) ===")
for _, r in df[num > 3].sort_values("outlier_ratio", ascending=False).iterrows():
    print(f"  {r['unit_id']:<32} outlier_ratio={r['outlier_ratio']}")
print(f"\n  合计 outlier_ratio>3: {(num>3).sum()}   >2: {(num>2).sum()}")
print("\n=== 主轴歪斜 (tilt_deg > 25；正常应 <10) ===")
for _, r in df[tilt > 25].sort_values("tilt_deg", ascending=False).iterrows():
    print(f"  {r['unit_id']:<32} tilt_deg={r['tilt_deg']}")
print(f"\n  合计 tilt>25: {(tilt>25).sum()}   >15: {(tilt>15).sum()}")
print(f"\n-> {SCR/'viz_numeric_flags.csv'}")
