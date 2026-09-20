"""只读：沿程 eccentricity 通道的分布（它是否退化）+ 与逐例 R² 的关系。"""
import numpy as np, json
from pathlib import Path
ROOT = Path("training_wss_min/experiments/wss_x5d_longitudinal_20260917/frozen_longitudinal_features")
files = sorted(ROOT.rglob("*.npz"))
print("npz 文件数", len(files))
if files:
    z = np.load(files[0])
    print("keys 示例:", [k for k in z.files if "ecc" in k or "round" in k][:8])
rows = []
for p in files:
    z = np.load(p)
    v = z["wall_geom_ref_eccentricity"]; m = z["wall_geom_ref_eccentricity_valid"].astype(bool)
    r = z["wall_geom_ref_roundness"]; rm = z["wall_geom_ref_roundness_valid"].astype(bool)
    cid = str(p.relative_to(ROOT)).replace("/", "__").replace(".npz", "")
    if m.sum() == 0: continue
    rows.append((cid, float(np.median(v[m])), float(np.percentile(v[m], 95)), float(v[m].max()),
                 float(m.mean()), float(np.median(r[rm])) if rm.sum() else np.nan))
rows.sort(key=lambda r: -r[2])
arr = np.array([[r[1], r[2], r[3], r[4], r[5]] for r in rows], dtype=float)
print(f"\n病例数 {len(rows)}")
for i, lab in enumerate(["逐例中位偏心", "逐例 p95 偏心", "逐例 max 偏心", "有效点比例", "逐例中位圆度"]):
    c = arr[:, i]; c = c[np.isfinite(c)]
    print(f"{lab:14s} 队列中位 {np.median(c):.4f}  p10 {np.percentile(c,10):.4f}  p90 {np.percentile(c,90):.4f}  max {c.max():.4f}")
print("\np95 偏心最大的 8 例：")
for r in rows[:8]: print(f"  {r[0]:45s} med {r[1]:.3f}  p95 {r[2]:.3f}  max {r[3]:.3f}")
print("\np95 偏心最小的 5 例：")
for r in rows[-5:]: print(f"  {r[0]:45s} med {r[1]:.3f}  p95 {r[2]:.3f}  max {r[3]:.3f}")
