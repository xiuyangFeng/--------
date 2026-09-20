#!/usr/bin/env python3
"""Two diagnostic plots for VF6→WSS vs X5. Reads diagnosis JSON only."""
from pathlib import Path
import json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

for p in (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
):
    if Path(p).exists():
        font_manager.fontManager.addfont(p)
plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False

OUT = Path("/public/newhome/cy/Digital_twin/GNN/training_wss_min/experiments/vf6_velocity_to_wss_20260916/analysis_20260916")
d = json.loads((OUT / "vel_vs_direct_wss_diagnosis.json").read_text())
shells = d["per_case_shells"]
ids = list(shells)
cohort = np.array([shells[i]["cohort"] for i in ids])
speed = np.array([shells[i]["speed_r2_mean_csv"] for i in ids])
wss = np.array([shells[i]["legacy_wss_r2_mean"] for i in ids])
x5 = np.array([shells[i]["x5_direct_r2_mean"] for i in ids])
r05 = np.array([shells[i]["bins"]["0-0.5mm"]["speed_r2"] for i in ids])
colors = {"AAA": "#d95f02", "AG": "#1b9e77", "ILO": "#7570b3"}

fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.6), dpi=160)
for ax, xs, xlab in (
    (axes[0], speed, "体内速度幅值 R²（三 seed 均值）"),
    (axes[1], r05, "距壁 0–0.5 mm 速度幅值 R²（s1234）"),
):
    for name, color in colors.items():
        m = cohort == name
        ax.scatter(xs[m], wss[m], c=color, s=36, label=name, edgecolors="k", linewidths=0.3, zorder=3)
    lo, hi = -0.3, 1.0
    ax.plot([lo, hi], [lo, hi], color="0.6", ls="--", lw=1)
    ax.set_xlim(0.35, 0.98)
    ax.set_ylim(-0.32, 0.92)
    ax.set_xlabel(xlab)
    ax.set_ylabel("VF6 派生 WSS R²（三 seed 均值）")
    ax.grid(True, alpha=0.3)
axes[0].legend(frameon=False, loc="lower right")
axes[0].set_title("整体速度好，不能推出派生 WSS 好")
axes[1].set_title("最内层 0–0.5 mm 与派生 WSS 更同向")
fig.tight_layout()
fig.savefig(OUT / "speed_vs_derived_wss_scatter.png")
plt.close(fig)

summary = d["shell_summary"]
names = ["0-0.5mm", "0.5-1mm", "1-2mm", "2-4mm", ">4mm"]
r2m = [summary[n]["speed_r2_casemean"] for n in names]
frac = [summary[n]["fraction_casemean"] for n in names]
x = np.arange(len(names))
fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=160)
bars = ax.bar(x, r2m, color="#4c78a8", width=0.62)
ax.axhline(0.795, color="#1b9e77", ls="--", lw=1.2, label="VF6 全场速度 R²_cb = 0.795")
ax.axhline(0.547, color="#d95f02", ls="--", lw=1.2, label="VF6 派生 WSS R²_cb = 0.547")
ax.axhline(0.717, color="#7570b3", ls=":", lw=1.2, label="X5 直接 WSS R²_cb = 0.717")
for i, (r, f) in enumerate(zip(r2m, frac)):
    ax.text(i, r + 0.012, f"{r:.2f}\n({100*f:.0f}%点)", ha="center", va="bottom", fontsize=8)
ax.set_xticks(x, ["0–0.5", "0.5–1", "1–2", "2–4", ">4"])
ax.set_xlabel("距壁 (mm)")
ax.set_ylabel("速度幅值 R²（34 例均值）")
ax.set_ylim(0.45, 0.95)
ax.legend(frameon=False, fontsize=8)
ax.set_title("VF6 点态速度按距壁分层（s1234）")
ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "vf6_speed_r2_by_wall_distance.png")
plt.close(fig)
print("wrote", OUT / "speed_vs_derived_wss_scatter.png")
print("wrote", OUT / "vf6_speed_r2_by_wall_distance.png")
