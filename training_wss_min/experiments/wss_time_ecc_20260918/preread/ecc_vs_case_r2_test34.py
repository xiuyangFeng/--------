import json, numpy as np
from pathlib import Path
from scipy.stats import spearmanr, pearsonr
RUNS = Path("training_wss_min/runs/wss_v51_wave1_20260916")
ROOT = Path("training_wss_min/experiments/wss_x5d_longitudinal_20260917/frozen_longitudinal_features")
per = {}
for s in ["s1234", "s7", "s2025"]:
    d = json.load(open(RUNS / f"X5D_v51_{s}/eval/ckpt_best/metrics.json"))["test"]
    if s == "s1234": print("metric_space =", d.get("metric_space"))
    for cid, e in d["per_case"].items():
        per.setdefault(cid, []).append(float(e["overall"]["r2"]))
cases = {c: float(np.mean(v)) for c, v in per.items() if len(v) == 3}
ecc = {}
for p in ROOT.rglob("*.npz"):
    z = np.load(p)
    v = z["wall_geom_ref_eccentricity"]; m = z["wall_geom_ref_eccentricity_valid"].astype(bool)
    if m.sum() == 0: continue
    key = str(p.parent.relative_to(ROOT))
    ecc[key] = (float(np.median(v[m])), float(np.percentile(v[m], 95)), float(np.mean(v[m] > 0.2)))
matched = [(c, cases[c], ecc[c]) for c in cases if c in ecc]
print("test 例数", len(cases), "匹配", len(matched))
if len(matched) < 5:
    print("cases:", list(cases)[:3]); print("ecc:", list(ecc)[:3]); raise SystemExit
r2 = np.array([m[1] for m in matched])
med = np.array([m[2][0] for m in matched]); p95 = np.array([m[2][1] for m in matched]); frac = np.array([m[2][2] for m in matched])
print(f"逐例 R² 三 seed 均值：{r2.mean():.3f} [{r2.min():.3f}, {r2.max():.3f}]")
for lab, x in [("中位偏心", med), ("p95 偏心", p95), ("偏心>0.2 点比例", frac)]:
    sr = spearmanr(x, r2); pr = pearsonr(x, r2)
    print(f"{lab:16s} Spearman {sr.statistic:+.3f} (p={sr.pvalue:.3f})   Pearson {pr[0]:+.3f} (p={pr[1]:.3f})")
order = np.argsort(p95); k = len(order)//3
lo, hi = order[:k], order[-k:]
print(f"\n按 p95 偏心分档：低 {r2[lo].mean():.3f} (n={k})  高 {r2[hi].mean():.3f} (n={k})  差 {r2[hi].mean()-r2[lo].mean():+.3f}")
print("\n偏心最高 6 例：")
for i in order[::-1][:6]: print(f"  {matched[i][0]:40s} p95 {p95[i]:.3f}  R² {r2[i]:.3f}")
print("偏心最低 6 例：")
for i in order[:6]: print(f"  {matched[i][0]:40s} p95 {p95[i]:.3f}  R² {r2[i]:.3f}")

print("\n=== 分队列（检查是否只是队列混杂）===")
coh = np.array([m[0].split("/")[0] for m in matched])
for c in ["AAA", "AG", "ILO"]:
    s = coh == c
    if s.sum() >= 4:
        sr = spearmanr(med[s], r2[s])
        print(f"{c:4s} n={s.sum():2d}  中位偏心 {np.median(med[s]):.4f}  R² {r2[s].mean():.3f}  组内 Spearman {sr.statistic:+.3f} (p={sr.pvalue:.3f})")
# 队列去均值后的偏相关
r2c = r2.copy(); medc = med.copy()
for c in set(coh):
    s = coh == c
    r2c[s] -= r2[s].mean(); medc[s] -= med[s].mean()
sr = spearmanr(medc, r2c); pr = pearsonr(medc, r2c)
print(f"队列去均值后：Spearman {sr.statistic:+.3f} (p={sr.pvalue:.3f})  Pearson {pr[0]:+.3f} (p={pr[1]:.3f})")
