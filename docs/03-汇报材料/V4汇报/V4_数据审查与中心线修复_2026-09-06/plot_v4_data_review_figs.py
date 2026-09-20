#!/usr/bin/env python3
"""把 anatomy-only 审查图收进汇报目录，并生成压力/WSS 重算对照图与三例网格壁面中心线拼图。

数字来自
docs/02-推进与变更/WSS_PINN/WSS_PINN_V4_anatomy-only预处理准备与173例数据核查_2026-09-03.md
§13 / §23 / §26 / §27。只读复制，不改 outputs。
"""
from __future__ import annotations

import glob
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from PIL import Image

ROOT = Path("/public/newhome/cy/Digital_twin/GNN")
OUT = Path(__file__).resolve().parent
ATLAS = ROOT / "outputs/wss_pinn/volume_uvwp_bc_rcr_v4_anatomy_prep_20260903/audits/atlas_review"
MESHWALL = ROOT / "outputs/centerline_v2_meshwall_20260904/cases"
COMPARE = ROOT / "outputs/centerline_v2_full_173_20260828/compare_before_after"

for f in glob.glob("/usr/share/fonts/opentype/noto/NotoSansCJK-*.ttc") + glob.glob(
    "/usr/share/fonts/truetype/noto/NotoSansCJK*.ttc"
):
    try:
        font_manager.fontManager.addfont(f)
    except Exception:
        pass
_avail = {f.name for f in font_manager.fontManager.ttflist}
CJK = next((n for n in ["Noto Sans CJK SC", "Noto Sans CJK JP", "Noto Sans CJK TC"] if n in _avail), None)
plt.rcParams["font.family"] = [CJK, "DejaVu Sans"] if CJK else ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "dejavusans"
plt.rcParams["figure.dpi"] = 120

SURF = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e6e5e0"
BEFORE = "#C4443C"
AFTER = "#2E8B57"
DATA = "#3B7DD8"
ACCENT = "#0E6E8C"


def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.grid(True, axis="y", color=GRID, lw=0.7)
    ax.tick_params(colors=INK2, labelsize=9)


COPIES = {
    "YANG_YU_QING-1_before_vs_after_iso.png": COMPARE / "YANG_YU_QING-1_before_vs_after_iso.png",
    "XIE_JIN_QUAN_before_vs_after_iso.png": COMPARE / "XIE_JIN_QUAN_before_vs_after_iso.png",
    "stl_vs_cfd_wall_three_mismatch_cases.png": ATLAS / "stl_vs_cfd_wall_three_mismatch_cases.png",
    "AAA__ruputer__LI_LAO_PING_adaptive_overlay_20260905.png": ATLAS
    / "AAA__ruputer__LI_LAO_PING_adaptive_overlay_20260905.png",
    "LI_LAO_PING_sac_curvature_profile_adaptive_20260905.png": ATLAS
    / "LI_LAO_PING_sac_curvature_profile_adaptive_20260905.png",
    "end_zone_profiles_20260906.png": ATLAS / "end_zone_profiles_20260906.png",
}

MESHWALL_PANELS = [
    ("ZHOU_KE_XUN", MESHWALL / "AAA/ruputer/ZHOU_KE_XUN/review_iso.png", "从 .cas 壁面重提 · pass"),
    ("LIU_WEN_QI", MESHWALL / "AAA/unruputer/LIU_WEN_QI/review_iso.png", "从 .cas 壁面重提 · pass"),
    ("YU_XIANG_SHENG-1", MESHWALL / "ILO/YU_XIANG_SHENG-1/before/review_iso.png", "从 .cas 壁面重提 · pass"),
]


def copy_sources() -> None:
    for name, src in COPIES.items():
        if not src.is_file():
            raise FileNotFoundError(src)
        dst = OUT / name
        shutil.copy2(src, dst)
        print(f"copied {name}  {dst.stat().st_size / 1024:.0f} KB")


def plot_meshwall_three() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 5.4))
    fig.patch.set_facecolor("white")
    for ax, (name, path, note) in zip(axes, MESHWALL_PANELS):
        img = Image.open(path)
        ax.imshow(img)
        ax.set_title(f"{name}\n{note}", fontsize=11, color=INK, pad=6)
        ax.axis("off")
    fig.suptitle("三例 STL≠CFD 壁面：从网格解剖壁面重提中心线后回到管腔内", fontsize=13, color=INK, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = OUT / "fig_meshwall_three_iso.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.name}")


def plot_pressure_wss() -> None:
    wall_fix = [
        ("LIU_ZONG_YANG", -13536, -0.013),
        ("ZHANG_ZHI_JUN", np.nan, -0.016),
        ("MENG_GUANG_QIN", -12235, -0.005),
        ("WANG_FU_SHUN", -11658, 0.029),
        ("ZHOU_KE_XUN", 7576, -0.006),
        ("ZUO_DAO_SHENG", -4053, -0.008),
        ("CHEN_SHU_LIN", -10131, -0.017),
        ("LIU_YUE_DONG", -12821, -0.005),
        ("LI_SHENG_WEN", -12840, -0.077),
        ("ZHANG_YONG_SHENG", -12439, -0.033),
    ]
    dlayer = [
        ("DING_JUN_FENG", (3.25, 75.5, 263.2), (3.27, 75.0, 262.4)),
        ("GUO_BAO_CHUN", (0.51, 10.0, 59.2), (0.51, 10.0, 59.0)),
        ("WANG_JIN_MING-0", (2.64, 106.6, 294.7), (2.63, 106.9, 294.8)),
        ("SUN_XU_XIA-1", (4.14, 79.5, 273.5), (4.16, 80.1, 275.4)),
    ]
    # ZHANG_ZHI_JUN 修复前是帧格式混杂，用 12 kPa 量级占位会误导；单独标注
    names = [r[0] for r in wall_fix]
    before = np.array([abs(r[1]) if np.isfinite(r[1]) else np.nan for r in wall_fix])
    after = np.array([abs(r[2]) for r in wall_fix])

    # Publication-style, color-blind-safe palette.  The pressure comparison is
    # a grouped bar chart; WSS is split into three linear-scale facets so the
    # 20/60-iteration values and their small differences remain readable.
    before_color = "#D55E00"
    after_color = "#0072B2"
    iter20_color = "#9BB8D3"
    iter60_color = "#276A9B"
    grid_major = "#DDE1E6"
    grid_minor = "#EFF1F3"
    text_color = "#20242A"

    fig = plt.figure(figsize=(15.4, 9.1), facecolor="white")
    outer = fig.add_gridspec(
        2, 1, height_ratios=[1.22, 1.0],
        left=0.062, right=0.988, top=0.965, bottom=0.085, hspace=0.50,
    )
    bottom = outer[1].subgridspec(1, 3, wspace=0.22)

    def polish_axis(axis, *, minor_grid: bool = False) -> None:
        axis.set_facecolor("white")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color("#AEB4BC")
            axis.spines[side].set_linewidth(0.8)
        axis.grid(True, axis="y", which="major", color=grid_major, lw=0.85)
        if minor_grid:
            axis.grid(True, axis="y", which="minor", color=grid_minor, lw=0.45)
        axis.tick_params(axis="both", colors="#4B535D", labelsize=9, width=0.7)
        axis.set_axisbelow(True)

    def split_case_label(name: str) -> str:
        parts = name.split("_")
        return "_".join(parts[:-1]) + "\n" + parts[-1]

    # (a) Grouped bars retain the familiar before/after comparison while a log
    # scale keeps both kilopascal and sub-pascal values visible.
    ax = fig.add_subplot(outer[0])
    polish_axis(ax, minor_grid=True)
    x = np.arange(len(names), dtype=float)
    width = 0.36
    floor = 3e-3
    mask = np.isfinite(before)
    ax.bar(x[mask] - width / 2, before[mask] - floor, width, bottom=floor,
           color=before_color, edgecolor="white", linewidth=0.65,
           label="Before recomputation", zorder=3)
    ax.bar(x + width / 2, after - floor, width, bottom=floor,
           color=after_color, edgecolor="white", linewidth=0.65,
           label="After recomputation", zorder=3)
    missing_x = x[~mask] - width / 2
    for xi in missing_x:
        ax.text(xi, floor * 1.30, "N/A", ha="center", va="bottom",
                fontsize=7.8, color="#7A818A")
    ax.axhline(0.1, color="#7A818A", lw=0.9, ls=(0, (4, 3)), zorder=2)
    ax.text(len(names) - 0.48, 0.112, "0.1 Pa", ha="right", va="bottom",
            fontsize=8.2, color="#6B727B")
    ax.set_yscale("log")
    ax.set_ylim(floor, 3e4)
    ax.set_xlim(-0.55, len(names) - 0.45)
    ax.set_xticks(x)
    ax.set_xticklabels([split_case_label(name) for name in names], fontsize=8.5)
    ax.set_ylabel(r"Wall–volume $|\Delta p|$ (Pa, log scale)",
                  fontsize=11, color=text_color)
    ax.set_title("(a)  Wall–volume pressure mismatch before and after recomputation",
                 loc="left", fontsize=13, fontweight="semibold",
                 color=text_color, pad=12)
    ax.legend(frameon=False, fontsize=9.5, loc="upper right", ncol=2,
              handlelength=1.5, columnspacing=1.25, borderaxespad=0.25)

    # (b) Three facets avoid mixing p50, p99, and maximum on one axis.  Values
    # are printed above every bar because the convergence differences are small.
    metric_specs = [
        (r"$p_{50}$", 0, 2),
        (r"$p_{99}$", 1, 1),
        ("Maximum", 2, 1),
    ]
    axes_b = [fig.add_subplot(bottom[0, i]) for i in range(3)]
    case_x = np.arange(len(dlayer), dtype=float)
    b_width = 0.36
    for panel_index, (axis, (metric_name, metric_index, decimals)) in enumerate(zip(axes_b, metric_specs)):
        polish_axis(axis)
        values20 = np.asarray([row[1][metric_index] for row in dlayer], dtype=float)
        values60 = np.asarray([row[2][metric_index] for row in dlayer], dtype=float)
        bars20 = axis.bar(case_x - b_width / 2, values20, b_width,
                          color=iter20_color, edgecolor="white", linewidth=0.65,
                          label="20 iterations", zorder=3)
        bars60 = axis.bar(case_x + b_width / 2, values60, b_width,
                          color=iter60_color, edgecolor="white", linewidth=0.65,
                          label="60 iterations", zorder=3)
        ymax = float(max(values20.max(), values60.max())) * 1.24
        axis.set_ylim(0, ymax)
        axis.set_xlim(-0.55, len(dlayer) - 0.45)
        axis.set_xticks(case_x)
        axis.set_xticklabels([split_case_label(row[0]) for row in dlayer], fontsize=8.0)
        axis.set_title(metric_name, fontsize=11.5, fontweight="semibold",
                       color=text_color, pad=8)
        if panel_index == 0:
            axis.set_ylabel("Peak-systolic WSS (Pa)", fontsize=10.5, color=text_color)
        fmt = f"{{:.{decimals}f}}"
        label_pad = ymax * 0.012
        for bars, values in ((bars20, values20), (bars60, values60)):
            for bar, value in zip(bars, values):
                axis.text(bar.get_x() + bar.get_width() / 2,
                          bar.get_height() + label_pad, fmt.format(value),
                          ha="center", va="bottom", fontsize=7.7,
                          color="#3D444D")

    fig.text(0.062, 0.445, "(b)  Peak-systolic WSS at 20 and 60 iterations",
             ha="left", va="bottom", fontsize=13, fontweight="semibold",
             color=text_color)
    handles, labels = axes_b[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9.5, ncol=2,
               loc="center right", bbox_to_anchor=(0.988, 0.456),
               handlelength=1.5, columnspacing=1.25, borderaxespad=0.0)

    out = OUT / "fig_pressure_wss_recalc.png"
    fig.savefig(out, dpi=220, bbox_inches="tight", pad_inches=0.12, facecolor="white")
    plt.close(fig)
    print(f"wrote {out.name}")


def main() -> None:
    copy_sources()
    plot_meshwall_three()
    plot_pressure_wss()


if __name__ == "__main__":
    main()
