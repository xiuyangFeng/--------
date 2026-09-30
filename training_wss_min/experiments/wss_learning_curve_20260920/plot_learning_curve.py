"""wss_learning_curve_20260920：折外合并 136 例物理 R²_cb 随训练比例。

读 offline/analyze_learning_curve.json（ckpt best；jet = 真值 p99 > 40 Pa）。
图写到本实验目录根。
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

EXP = Path(__file__).resolve().parent
SRC = EXP / "offline" / "analyze_learning_curve.json"
OUT = EXP / "learning_curve_r2cb.png"

_cjk = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
font_manager.fontManager.addfont(str(_cjk))
plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(_cjk)).get_name()
plt.rcParams.update({"axes.unicode_minus": False, "font.size": 10})

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
payload = json.loads(SRC.read_text())
pooled = payload["pooled"]
fractions = (25, 50, 75, 100)
seeds = (7, 1234, 2025)
seed_style = {7: "o", 1234: "s", 2025: "D"}

x = np.array([payload["per_fraction"][str(f)]["n_train"] for f in fractions], dtype=float)
mean = np.array([payload["per_fraction"][str(f)]["r2cb"] for f in fractions])
sd = np.array([payload["per_fraction"][str(f)]["r2cb_sd"] for f in fractions])
jet = np.array([payload["per_fraction"][str(f)]["jet"] for f in fractions])
nonjet = np.array([payload["per_fraction"][str(f)]["nonjet"] for f in fractions])

fig, ax = plt.subplots(figsize=(7.2, 4.6), facecolor=SURF)
ax.set_facecolor(SURF)
ax.fill_between(x, mean - sd, mean + sd, color="#2a78d6", alpha=0.16, lw=0, zorder=1)
ax.plot(x, mean, color="#2a78d6", lw=2.0, marker="o", ms=6, zorder=3, label="全部 136 例（3 seed 均值）")
ax.plot(x, jet, color="#eb6834", lw=1.6, ls="--", marker="^", ms=5.5, zorder=3, label="jet（真值 p99 > 40 Pa）")
ax.plot(x, nonjet, color="#1baf7a", lw=1.6, ls="--", marker="v", ms=5.5, zorder=3, label="非 jet")
for seed in seeds:
    ys = [pooled[f"LC{f}_s{seed}"]["r2cb"] for f in fractions]
    ax.plot(x, ys, color=INK2, lw=0, marker=seed_style[seed], ms=4.0, alpha=0.9, zorder=4, label=f"seed {seed}")

ax.set_xticks(x)
ax.set_xticklabels([f"{f}%\n(n≈{n:.0f})" for f, n in zip(fractions, x)])
ax.set_xlim(16, 98)
ax.set_ylim(0.55, 0.80)
ax.set_xlabel("训练比例（三折平均训练例数）")
ax.set_ylabel("折外合并物理 R²_cb")
ax.set_title("X5D_v51 配方学习曲线 · ckpt best · Pa 空间", loc="left", color=INK, pad=10)
ax.tick_params(colors=INK)
for spine in ax.spines.values():
    spine.set_color("#d5d4d0")
ax.yaxis.grid(True, color=GRID, lw=0.8)
ax.set_axisbelow(True)
ax.legend(frameon=False, ncol=2, loc="lower right", fontsize=8.5)
fig.tight_layout()
fig.savefig(OUT, dpi=160, facecolor=SURF)
print(OUT)
