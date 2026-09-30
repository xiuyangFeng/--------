"""Storyboard of the four presentation stages for one case (matplotlib only: the cluster has no X/OSMesa)."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.font_manager as fm, glob
for _p in glob.glob("/usr/share/fonts/opentype/noto/*CJK*"):
    fm.fontManager.addfont(_p)
plt.rcParams["font.family"] = ["Noto Sans CJK JP", "Noto Serif CJK JP", "DejaVu Sans"]; plt.rcParams["axes.unicode_minus"] = False
OUT = Path(__file__).parent
tag = sys.argv[1]
z = np.load(OUT / f"{tag}.field.npz", allow_pickle=True); rep = json.load(open(OUT / f"{tag}.metrics.json"))
pts, wss, sd = z["pts"], z["wss_pa"], z["seed_sd_pa"]; seg = z["segment_id"]; s_root = z["s_from_root_mm"]; theta = z["theta_rad"]
tab, cols = z["atlas_table"], list(z["atlas_columns"]); names = json.loads(str(z["branch_names"])); col = lambda c: tab[:, cols.index(c)]
V = z["stl_vertices"]
# view: PCA of the point cloud -> first two axes (largest extent) as the projection plane
c = pts.mean(0); u, s_, vt = np.linalg.svd(pts - c, full_matrices=False); proj = lambda p: (p - c) @ vt[:2].T
bwr = LinearSegmentedColormap.from_list("gnn", ["#1f4e9c", "#f7f7f7", "#c0392b"])
vmax = float(rep["wss_field_pa"]["p99"])
fig, ax = plt.subplots(2, 3, figsize=(18, 11), constrained_layout=True); ax = ax.ravel()
# 1 raw STL
q = proj(V); ax[0].scatter(q[:, 0], q[:, 1], s=0.3, c="#8a8f98"); ax[0].set_title(f"① 原始 STL（{len(V):,} 顶点，面积 {rep['input_check']['stl_area_mm2']/100:.0f} cm²）")
# 2 centerline with radius, branch labels
cx = proj(np.stack([col("x_mm"), col("y_mm"), col("z_mm")], 1)); r = col("radius_mm"); sid = col("segment_id").astype(int)
sc = ax[1].scatter(cx[:, 0], cx[:, 1], s=(r * 2.2) ** 2, c=r, cmap="viridis", alpha=0.6); plt.colorbar(sc, ax=ax[1], label="内切半径 mm")
for k, nm in names.items():
    mk = sid == int(k)
    if mk.any(): m = cx[mk].mean(0); ax[1].annotate(nm, m, fontsize=10, ha="center", bbox=dict(boxstyle="round", fc="w", alpha=.8))
caps = rep["input_check"]["caps_mm"]; ax[1].set_title("② 中心线 + 半径 + 分支命名（盖面半径 " + ", ".join(f"{k}:{v['radius_mm']:.1f}" for k, v in caps.items()) + " mm）", fontsize=10)
# 3 model input cloud coloured by a geometric feature (distance to junction)
q = proj(pts); sc = ax[2].scatter(q[:, 0], q[:, 1], s=0.4, c=np.minimum(s_root, s_root.max()), cmap="magma"); plt.colorbar(sc, ax=ax[2], label="沿程弧长 s (mm)")
ax[2].set_title(f"③ 输入点云 {len(pts):,} 点 @ {rep['input_check']['median_spacing_mm']:.2f} mm（27 维特征之一：弧长）")
# 4 WSS heatmap
sc = ax[3].scatter(q[:, 0], q[:, 1], s=0.4, c=wss, cmap=bwr, vmin=0, vmax=vmax); plt.colorbar(sc, ax=ax[3], label="WSS (Pa)")
pk = rep["peak_location"]; pq = proj(np.array([pk["xyz_mm"]], dtype=np.float32))[0]; ax[3].plot(pq[0], pq[1], "k*", ms=14, mec="yellow")
ax[3].set_title(f"④ 峰值帧 WSS：p99 {rep['wss_field_pa']['p99']:.2f} Pa，最大 {pk['value_pa']:.1f} Pa @ {pk['branch']} s={pk['s_from_inlet_mm']:.0f} mm")
# 5 unrolled (s, theta) map
sc = ax[4].scatter(s_root, np.degrees(theta), s=0.5, c=wss, cmap=bwr, vmin=0, vmax=vmax); ax[4].set_xlabel("沿程弧长 s (mm)"); ax[4].set_ylabel("周向角 θ (°)")
for k, nm in names.items():
    mk = seg == int(k)
    if mk.sum() > 50: ax[4].text(float(np.median(s_root[mk])), 185, nm, fontsize=8, ha="center")
ax[4].set_ylim(-180, 200); ax[4].set_title("⑤ 展开图 (s, θ)：整段壁面一眼看全，热点位置一目了然")
# 6 per-branch bars + uncertainty
br = rep["per_branch"]; labels = list(br); p99 = [br[b]["wss_p99_pa"] for b in labels]; mean = [br[b]["wss_mean_pa"] for b in labels]; usd = [br[b]["seed_sd_median_pa"] for b in labels]
x = np.arange(len(labels)); ax[5].bar(x - 0.2, mean, 0.4, label="均值", color="#7f9cc4"); ax[5].bar(x + 0.2, p99, 0.4, yerr=usd, label="p99（误差棒 = 五 seed 离散度中位）", color="#c0392b")
ax[5].set_xticks(x); ax[5].set_xticklabels(labels, fontsize=9); ax[5].set_ylabel("WSS (Pa)"); ax[5].legend(fontsize=8)
pp = rep["population_percentile_train136"]; ax[5].set_title(f"⑥ 分支表 + 人群位置：p99 在 136 例中处第 {pp['p99']:.0f} 百分位（人群中位 {pp['reference_p99_pa_median']:.1f} Pa）", fontsize=10)
for a in ax[:4]: a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
fig.suptitle(f"{rep['unit_id']}  ·  部署链路四段展示 + 报告页（无 CFD 对照）  ·  发布包 {rep['release']}  ·  推理 {rep['device']}", fontsize=13)
fig.savefig(OUT / f"{tag}.storyboard.png", dpi=110); print("saved", OUT / f"{tag}.storyboard.png")
