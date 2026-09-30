"""独立抽查：交付包里的数值 vs Fluent 原始 ASCII 导出（不经过入库 bundle）。
随机抽 6 例（每队列 2 例）× 2 帧（峰值 1162、1240），按坐标逐点配对后比较压力、WSS 标量/三分量、速度三分量。
输出 <包>/verification/raw_export_spotcheck.csv。只读。"""
import csv
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path("/public/newhome/cy/Digital_twin/GNN")
PKG = ROOT / "outputs/synth59_teacher_package_2026-09-28"
rows = list(csv.DictReader(open(PKG / "manifest.csv")))
rng = np.random.default_rng(20260928)
pick = []
for coh in ("AG", "AAA", "ILO"):
    ids = sorted(r["case_id"] for r in rows if r["cohort"] == coh)
    pick += list(rng.choice(ids, 2, replace=False))
out = []
for cid in pick:
    raw = ROOT / "data_new" / cid
    h5 = next((PKG / cid).glob("*.h5"))
    with h5py.File(h5, "r") as f:
        steps = list(f["time/step"][...])
        for step in (1162, 1240):
            k = steps.index(step)
            for kind, sub, grp, cols in (("wall", "ascii", "wall", {"pressure": "pressure_pa", "wall-shear": "wss_scalar_pa"}),
                                         ("volume", "ascii_in", "volume", {"pressure": "pressure_pa"})):
                fp = next(p for p in (raw / sub).iterdir() if p.name.endswith(f"-{step}"))
                df = pd.read_csv(fp, skipinitialspace=True)
                df.columns = [c.strip() for c in df.columns]
                xyz_raw = df[["x-coordinate", "y-coordinate", "z-coordinate"]].to_numpy() * 1000.0
                xyz = f[f"{grp}/xyz_mm"][...]
                d, j = cKDTree(xyz_raw).query(xyz)
                rec = {"case_id": cid, "step": step, "kind": kind, "n_points": len(xyz), "raw_rows": len(df),
                       "match_max_dist_mm": float(d.max()), "match_unique": len(np.unique(j)) == len(xyz)}
                for rc, key in cols.items():
                    a = f[f"{grp}/{key}"][k]; b = df[rc].to_numpy()[j]
                    rec[f"{key}_max_abs_diff"] = float(np.abs(a - b).max()); rec[f"{key}_max_rel_diff"] = float((np.abs(a - b) / np.maximum(np.abs(b), 1e-6)).max())
                if kind == "wall":
                    a = f["wall/wss_vector_pa"][k]; b = df[["x-wall-shear", "y-wall-shear", "z-wall-shear"]].to_numpy()[j]
                    rec["wss_vector_pa_max_abs_diff"] = float(np.abs(a - b).max())
                else:
                    a = f["volume/velocity_m_s"][k]; b = df[["x-velocity", "y-velocity", "z-velocity"]].to_numpy()[j]
                    rec["velocity_m_s_max_abs_diff"] = float(np.abs(a - b).max())
                out.append(rec); print(rec, flush=True)
keys = sorted({k for r in out for k in r}, key=lambda k: (k not in ("case_id", "step", "kind"), k))
with open(PKG / "verification/raw_export_spotcheck.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(out)
