"""Publication-style redraw of the WSS learning curve.

The source values and uncertainty definition are unchanged.  This version
uses training fraction on an evenly spaced x-axis, a restrained palette,
direct end labels, and a compact seed key.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PIL import Image

EXP = Path(__file__).resolve().parent
SRC = EXP / "offline" / "analyze_learning_curve.json"
OUT = EXP / "learning_curve_r2cb_optimized.png"
PDF = EXP / "learning_curve_r2cb_optimized.pdf"
SVG = EXP / "learning_curve_r2cb_optimized.svg"

_cjk = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
_font_prop = font_manager.FontProperties(fname=str(_cjk))
# Matplotlib 3.1 does not expose ``fontManager.addfont``; registering the
# bundled CJK font through its public FontEntry list keeps the script portable.
if not any(getattr(entry, "fname", None) == str(_cjk) for entry in font_manager.fontManager.ttflist):
    font_manager.fontManager.ttflist.append(
        font_manager.FontEntry(fname=str(_cjk), name=_font_prop.get_name())
    )
FONT = _font_prop.get_name()

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": [FONT, "DejaVu Sans", "sans-serif"],
        "font.size": 9,
        "axes.unicode_minus": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

payload = json.loads(SRC.read_text())
pooled = payload["pooled"]
fractions = np.array([25, 50, 75, 100], dtype=float)
seeds = (7, 1234, 2025)
seed_style = {7: "o", 1234: "s", 2025: "D"}

mean = np.array([payload["per_fraction"][str(int(f))]["r2cb"] for f in fractions])
sd = np.array([payload["per_fraction"][str(int(f))]["r2cb_sd"] for f in fractions])
jet = np.array([payload["per_fraction"][str(int(f))]["jet"] for f in fractions])
nonjet = np.array([payload["per_fraction"][str(int(f))]["nonjet"] for f in fractions])
n_train = np.array([payload["per_fraction"][str(int(f))]["n_train"] for f in fractions])

# A restrained, color-vision-friendly palette.
INK = "#24313A"
MUTED = "#68747C"
GRID = "#D9E0E5"
BLUE = "#2F6FB3"
ORANGE = "#D96B3B"
TEAL = "#209A86"
SEED = "#6F7880"

fig, ax = plt.subplots(figsize=(7.2, 4.8), facecolor="white")
fig.subplots_adjust(left=0.12, right=0.83, bottom=0.19, top=0.81)
ax.set_facecolor("white")

# Main aggregate and its three-seed SD band.
ax.fill_between(
    fractions,
    mean - sd,
    mean + sd,
    color=BLUE,
    alpha=0.15,
    lw=0,
    zorder=1,
)
ax.plot(
    fractions,
    mean,
    color=BLUE,
    lw=2.8,
    marker="o",
    ms=7.2,
    mec="white",
    mew=1.2,
    zorder=4,
)
ax.plot(
    fractions,
    jet,
    color=ORANGE,
    lw=2.0,
    ls=(0, (5, 3)),
    marker="^",
    ms=7.0,
    mec="white",
    mew=0.9,
    zorder=3,
)
ax.plot(
    fractions,
    nonjet,
    color=TEAL,
    lw=2.0,
    ls=(0, (5, 3)),
    marker="v",
    ms=7.0,
    mec="white",
    mew=0.9,
    zorder=3,
)

# Individual seed results are deliberately neutral so they support, rather
# than compete with, the aggregate curve.
for seed in seeds:
    ys = np.array([pooled[f"LC{int(f)}_s{seed}"]["r2cb"] for f in fractions])
    ax.plot(
        fractions,
        ys,
        color=SEED,
        lw=0,
        marker=seed_style[seed],
        ms=5.2,
        mfc="white",
        mec=SEED,
        mew=1.0,
        alpha=0.95,
        zorder=5,
    )

ax.set_xlim(18, 112)
ax.set_ylim(0.55, 0.80)
ax.set_xticks(fractions)
ax.set_xticklabels([f"{int(f)}%\n(n≈{n:.0f})" for f, n in zip(fractions, n_train)])
ax.set_yticks(np.arange(0.55, 0.801, 0.05))
ax.set_xlabel("Training fraction (mean train cases across 3 folds)", labelpad=9, color=INK)
ax.set_ylabel("Pooled out-of-fold physical R²cb", labelpad=9, color=INK)
ax.tick_params(axis="both", colors=INK, length=4, width=0.8)
ax.grid(axis="y", color=GRID, lw=0.8)
ax.set_axisbelow(True)
for spine in (ax.spines["left"], ax.spines["bottom"]):
    spine.set_color("#AEB8BF")
    spine.set_linewidth(0.9)

# Direct labels reduce eye travel and keep the legend focused on uncertainty
# and replicate notation.
for y, label, color in (
    (nonjet[-1], "non-jet", TEAL),
    (mean[-1], "All 136 cases", BLUE),
    (jet[-1], "jet", ORANGE),
):
    ax.text(
        102.0,
        y,
        label,
        color=color,
        fontsize=9,
        fontweight="bold",
        va="center",
        ha="left",
        clip_on=False,
    )

# Compact seed/uncertainty key in the quiet upper margin of the axes.
legend_handles = [
    Patch(facecolor=BLUE, edgecolor="none", alpha=0.15, label="Blue band: SD across 3 seeds"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor=SEED, label="seed 7", markersize=5.3),
    Line2D([0], [0], marker="s", color="none", markerfacecolor="white", markeredgecolor=SEED, label="seed 1234", markersize=5.3),
    Line2D([0], [0], marker="D", color="none", markerfacecolor="white", markeredgecolor=SEED, label="seed 2025", markersize=5.0),
]
ax.legend(
    handles=legend_handles,
    loc="upper left",
    bbox_to_anchor=(0.0, 0.99),
    ncol=4,
    frameon=False,
    handletextpad=0.35,
    columnspacing=1.0,
    borderaxespad=0,
    fontsize=8.1,
)

fig.text(
    0.12,
    0.925,
    "Learning curve: training fraction and physical R²cb",
    ha="left",
    va="bottom",
    fontsize=15,
    fontweight="bold",
    color=INK,
)
fig.text(
    0.12,
    0.888,
    "X5D_v51 · ckpt best · Pa space · 3-fold cross-validation",
    ha="left",
    va="bottom",
    fontsize=9.5,
    color=MUTED,
)
fig.savefig(OUT, dpi=600, facecolor="white", bbox_inches="tight")
Image.open(OUT).save(
    str(EXP / "learning_curve_r2cb_optimized.tiff"),
    compression="tiff_lzw",
    dpi=(600, 600),
)
fig.savefig(PDF, facecolor="white", bbox_inches="tight")
fig.savefig(SVG, facecolor="white", bbox_inches="tight")
print(OUT)
