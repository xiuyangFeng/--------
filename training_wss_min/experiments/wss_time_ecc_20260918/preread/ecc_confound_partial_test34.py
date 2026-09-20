import json, numpy as np, h5py
from pathlib import Path
from scipy.stats import spearmanr
RUNS = Path("training_wss_min/runs/wss_v51_wave1_20260916")
ROOT = Path("training_wss_min/experiments/wss_x5d_longitudinal_20260917/frozen_longitudinal_features")
CASES = Path("data_wss_v5/anatomy_pointcloud_v5_1_20260916/cases")
per = {}
for s in ["s1234", "s7", "s2025"]:
    d = json.load(open(RUNS / f"X5D_v51_{s}/eval/ckpt_best/metrics.json"))["test"]
    for cid, e in d["per_case"].items(): per.setdefault(cid, []).append(float(e["overall"]["r2"]))
cases = {c: float(np.mean(v)) for c, v in per.items() if len(v) == 3}
rows = []
for cid, r2 in cases.items():
    z = np.load(ROOT / cid / "features.npz")
    v = z["wall_geom_ref_eccentricity"]; m = z["wall_geom_ref_eccentricity_valid"].astype(bool)
    h5 = CASES / cid.replace("/", "__") / "case.h5"
    with h5py.File(h5, "r") as f:
        valid = f["wall_static/valid"][:]
        rad = f["wall_static/radius_mm"][valid]
        tau = f["wall_temporal/wss_scalar_pa"][21, valid]
        s_max = float(f["wall_static/s_from_root_mm"][valid].max())
    rows.append((cid, r2, float(np.median(v[m])), float(np.median(rad)), float(rad.max()/max(np.median(rad),1e-6)),
                 s_max, float(np.log(np.maximum(tau,0.05)).std())))
a = np.array([r[1:] for r in rows])
lab = ["R²", "中位偏心", "中位半径 mm", "最大/中位半径", "主干长 mm", "ln WSS 空间 sd"]
print("与逐例 R² 的 Spearman：")
for j in range(1, a.shape[1]):
    sr = spearmanr(a[:, j], a[:, 0]); print(f"  {lab[j]:16s} {sr.statistic:+.3f} (p={sr.pvalue:.3f})")
print("\n中位偏心与各混杂量的 Spearman：")
for j in range(2, a.shape[1]):
    sr = spearmanr(a[:, j], a[:, 1]); print(f"  {lab[j]:16s} {sr.statistic:+.3f} (p={sr.pvalue:.3f})")
# 偏相关：把混杂量秩回归掉后再看
def rank(x): return np.argsort(np.argsort(x)).astype(float)
import numpy.linalg as la
X = np.column_stack([rank(a[:, j]) for j in range(2, a.shape[1])] + [np.ones(len(a))])
def resid(y): return y - X @ la.lstsq(X, y, rcond=None)[0]
sr = spearmanr(resid(rank(a[:, 1])), resid(rank(a[:, 0])))
print(f"\n控制住半径/形态/长度/WSS 离散度后，偏心 vs R² 偏相关：{sr.statistic:+.3f} (p={sr.pvalue:.3f}, n={len(a)})")
