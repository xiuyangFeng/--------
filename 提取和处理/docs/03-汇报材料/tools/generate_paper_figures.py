#!/usr/bin/env python3
"""
Publication-quality figures for V3 WSS experiments.

Scientific style:
  - White background, no top/right spines
  - Colorblind-friendly palette (based on Seaborn colorblind)
  - Clean typography, no legend frame
  - v3d uses two-panel layout (avoids dual y-axis)

Outputs (overwrites existing files):
  - v3p_wss_comparison.png
  - v3p_wss_components.png
  - v3d_per_domain_metrics.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "docs" / "03-汇报材料" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RUNS = {
    "Main-PW": ROOT
    / "outputs/field/field_v3_pointnext_localpool_main01_geom_pw_wall13000_near2000_split_AG_v1_seed1_20260508_001936",
    "AsymW-a": ROOT
    / "outputs/field/field_v3_pointnext_localpool_main01_geom_pw_asymw_a_wall13000_near2000_split_AG_v1_seed1_20260522_124946",
    "WssDO-a": ROOT
    / "outputs/field/field_v3_pointnext_localpool_main01_geom_pw_wssdo_a_wall13000_near2000_split_AG_v1_seed1_20260522_131813",
    "WSS-a-PW": ROOT
    / "outputs/field/field_v3_pointnext_localpool_wss01a_geom_pw_lambda005_wall13000_near2000_split_AG_v1_seed1_20260507_001902",
}

V3D_PROBE_P = ROOT / (
    "outputs/field/field_v3d_pointnext_localpool_probe_p01_geom_wall13000_near2000_"
    "split_data_new_v3_v3_seed1_20260521_103843/eval_by_domain_test/metrics_by_domain.json"
)
V3D_PROBE_WSS = ROOT / (
    "outputs/field/field_v3d_pointnext_localpool_probe_wss01_geom_wall13000_near2000_"
    "split_data_new_v3_v3_seed1_20260521_101738/eval_by_domain_test/metrics_by_domain.json"
)

# ── Color palette ─────────────────────────────────────────────────────────
# Seaborn colorblind-friendly palette + scientific accent
C = {
    "baseline": "#9ECAE1",   # light steel-blue  (neutral baseline)
    "blue_mid": "#4292C6",   # medium blue
    "blue_dk":  "#2171B5",   # dark blue
    "navy":     "#08519C",   # navy – best/highlight
    "orange":   "#FD8D3C",   # warm orange – warm comparison
    "orange_dk":"#D94701",   # dark orange
    "green":    "#31A354",   # improvement indicator
    "red":      "#E41A1C",   # reference / warning
    "purple":   "#7B2D8B",   # supplemental
    "text":     "#1A1A1A",   # body text
    "annot":    "#3A3A3A",   # annotation text
    "spine":    "#BBBBBB",   # axis spine
    "grid":     "#EFEFEF",   # background grid
    "caption":  "#777777",   # caption / subtitle
}

FIG_SINGLE = (7.5, 4.8)
FIG_DOUBLE = (11.2, 4.8)
DPI = 300


# ── Style helpers ─────────────────────────────────────────────────────────

def setup_style() -> str:
    from matplotlib import font_manager

    candidates = ("Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Micro Hei", "SimHei")
    available = {f.name for f in font_manager.fontManager.ttflist}
    font = "DejaVu Sans"
    for fam in candidates:
        if fam in available:
            font = fam
            break

    plt.rcParams.update(
        {
            "font.family":          "sans-serif",
            "font.sans-serif":      [font, "Arial", "DejaVu Sans"],
            "font.size":            12,
            "axes.titlesize":       14,
            "axes.titleweight":     "bold",
            "axes.labelsize":       12.5,
            "axes.labelcolor":      C["text"],
            "axes.edgecolor":       C["spine"],
            "axes.facecolor":       "white",
            "axes.spines.top":      False,
            "axes.spines.right":    False,
            "figure.facecolor":     "white",
            "text.color":           C["text"],
            "xtick.color":          C["text"],
            "ytick.color":          C["text"],
            "xtick.labelsize":      11,
            "ytick.labelsize":      11,
            "xtick.major.size":     3.5,
            "ytick.major.size":     3.5,
            "xtick.major.width":    0.8,
            "ytick.major.width":    0.8,
            "xtick.major.pad":      4,
            "ytick.major.pad":      4,
            "legend.frameon":       False,
            "legend.fontsize":      11,
            "axes.unicode_minus":   False,
        }
    )
    return font


def _polish_ax(ax) -> None:
    """Apply consistent scientific spine/grid style to one axes."""
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(C["spine"])
        ax.spines[spine].set_linewidth(0.8)
    ax.tick_params(axis="both", length=3.5, width=0.8, color=C["spine"])
    ax.yaxis.grid(True, color=C["grid"], linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def save_fig(fig, name: str) -> None:
    path = OUT_DIR / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved  {path.name}")


def best_wss_metrics(run_dir: Path) -> dict:
    with open(run_dir / "summary.json", encoding="utf-8") as f:
        s = json.load(f)
    return s.get("test_metrics_best_wss") or s["test_metrics"]


# ── Figure 1: WSS overall comparison ─────────────────────────────────────

def fig_wss_comparison() -> None:
    keys   = ["Main-PW", "WSS-a-PW", "AsymW-a", "WssDO-a"]
    labels = ["Main-PW", "WSS-a-PW", "AsymW-a", "WssDO-a"]
    vals   = [best_wss_metrics(RUNS[k])["wss_r2_wss"] for k in keys]
    best   = int(np.argmax(vals))

    bar_colors = [C["baseline"], C["blue_mid"], C["blue_dk"], C["orange"]]
    bar_colors[best] = C["navy"]

    fig, ax = plt.subplots(figsize=FIG_SINGLE)

    bars = ax.bar(
        labels, vals,
        color=bar_colors,
        width=0.52,
        zorder=3,
        linewidth=0,
    )
    # Best bar: white outline for subtle highlight
    bars[best].set_edgecolor(C["navy"])
    bars[best].set_linewidth(1.5)

    # Value labels above bars
    y_pad = (max(vals) - min(vals)) * 0.06
    for bar, v, col in zip(bars, vals, bar_colors):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + y_pad * 0.35,
            f"{v:.3f}",
            ha="center", va="bottom",
            fontsize=10.5, color=C["annot"], fontweight="semibold",
        )

    # Reference dashed line
    ref = 0.40
    ax.axhline(ref, color=C["red"], lw=1.2, ls="--", alpha=0.70, zorder=2)
    ax.text(
        len(labels) - 0.58, ref + (max(vals) - min(vals)) * 0.02,
        f"empirical ceiling ≈ {ref:.2f}",
        color=C["red"], fontsize=9.5, va="bottom",
    )

    ax.set_ylim(0.325, max(vals) + y_pad * 2.2)
    ax.set_ylabel(r"WSS  $R^2$  (test, best-WSS ckpt)")
    ax.set_title("V3P — WSS Performance Comparison", pad=12)
    ax.text(
        0.5, -0.13,
        "split_AG_v1 · seed = 1",
        transform=ax.transAxes,
        ha="center", fontsize=10, color=C["caption"], style="italic",
    )
    _polish_ax(ax)
    fig.tight_layout(pad=1.8)
    save_fig(fig, "v3p_wss_comparison.png")


# ── Figure 2: WSS component-wise comparison ───────────────────────────────

def fig_wss_components() -> None:
    exps   = ["Main-PW", "AsymW-a"]
    comps  = ["wss_r2_wss_z", "wss_r2_wss_x", "wss_r2_wss_y"]
    clabels = ["wss_z", "wss_x", "wss_y"]
    colors  = [C["baseline"], C["navy"]]

    x     = np.arange(len(clabels))
    width = 0.35
    gap   = 0.06

    fig, ax = plt.subplots(figsize=FIG_SINGLE)

    metrics = [best_wss_metrics(RUNS[e]) for e in exps]

    for i, (exp, m, col) in enumerate(zip(exps, metrics, colors)):
        vals   = [m[c] for c in comps]
        offset = (i - 0.5) * (width + gap / 2)
        bars   = ax.bar(
            x + offset, vals,
            width=width, label=exp,
            color=col,
            linewidth=0,
            zorder=3,
        )
        for bar, v in zip(bars, vals):
            v_off = 0.004 if v >= 0 else -0.004
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + v_off,
                f"{v:.3f}",
                ha="center",
                va="bottom" if v >= 0 else "top",
                fontsize=9.5, color=C["annot"], fontweight="medium",
            )

    # Delta annotation between paired bars
    m0, m1 = metrics
    y_top_global = max(m0[c] for c in comps) if max(m0[c] for c in comps) > max(m1[c] for c in comps) \
        else max(m1[c] for c in comps)

    for ci, comp in enumerate(comps):
        delta = m1[comp] - m0[comp]
        if abs(delta) < 5e-4:
            continue
        y_ann = max(m0[comp], m1[comp]) + 0.024
        color = C["green"] if delta > 0 else C["red"]
        sign  = "+" if delta > 0 else ""
        ax.text(
            ci, y_ann,
            f"$\\Delta$ = {sign}{delta:.3f}",
            ha="center", va="bottom",
            fontsize=8.5, color=color, fontweight="bold",
        )

    ax.axhline(0, color=C["spine"], lw=0.8, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(clabels)
    ax.set_ylabel(r"$R^2$  (test, best-WSS ckpt)")
    ax.set_title("V3P — WSS Component-wise $R^2$", pad=12)
    ax.text(
        0.5, -0.13,
        "Main-PW vs. AsymW-a  ·  split_AG_v1 · seed = 1",
        transform=ax.transAxes,
        ha="center", fontsize=10, color=C["caption"], style="italic",
    )
    ax.legend(loc="upper right", handlelength=1.4)
    _polish_ax(ax)
    fig.tight_layout(pad=1.8)
    save_fig(fig, "v3p_wss_components.png")


# ── Figure 3: V3D per-domain, two-panel ───────────────────────────────────

def fig_v3d_per_domain() -> None:
    with open(V3D_PROBE_P,   encoding="utf-8") as f:
        p_data = json.load(f)
    with open(V3D_PROBE_WSS, encoding="utf-8") as f:
        w_data = json.load(f)

    domains = ["AAA", "AG", "ILO"]
    p_vals  = [p_data["by_domain"][d]["metrics"]["r2_p"]           for d in domains]
    w_vals  = [w_data["by_domain"][d]["metrics"]["wss_r2_wss"]     for d in domains]

    x     = np.arange(len(domains))
    bar_w = 0.50

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG_DOUBLE)

    # ── Panel (a): Pressure R² ──
    bars1 = ax1.bar(
        x, p_vals, bar_w,
        color=C["blue_dk"],
        linewidth=0, zorder=3,
    )
    for bar, v in zip(bars1, p_vals):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            v - 0.0055,
            f"{v:.3f}",
            ha="center", va="top",
            fontsize=10.5, color="white", fontweight="bold",
        )
    ax1.set_xticks(x)
    ax1.set_xticklabels(domains)
    ax1.set_ylim(0.86, 1.008)
    ax1.set_ylabel(r"Pressure  $R^2$")
    ax1.set_title("(a)  Probe-P — Pressure $R^2$", pad=10)
    _polish_ax(ax1)

    # ── Panel (b): WSS R² ──
    bars2 = ax2.bar(
        x, w_vals, bar_w,
        color=C["orange"],
        linewidth=0, zorder=3,
    )
    y_pad_w = (max(w_vals) - min(w_vals)) * 0.12
    for bar, v in zip(bars2, w_vals):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            v + y_pad_w * 0.35,
            f"{v:.3f}",
            ha="center", va="bottom",
            fontsize=10.5, color=C["annot"], fontweight="bold",
        )
    ax2.set_xticks(x)
    ax2.set_xticklabels(domains)
    ax2.set_ylim(0.14, max(w_vals) + y_pad_w * 2.0)
    ax2.set_ylabel(r"WSS  $R^2$")
    ax2.set_title("(b)  Probe-WSS — WSS $R^2$", pad=10)
    _polish_ax(ax2)

    fig.suptitle(
        "V3D Post-4901 — Per-Domain Test Metrics",
        fontsize=15, fontweight="bold", y=1.03,
    )
    fig.text(
        0.5, -0.05,
        "split_data_new_v3_v3 · seed = 1",
        ha="center", fontsize=10, color=C["caption"], style="italic",
    )
    fig.tight_layout(pad=2.0, w_pad=3.5)
    save_fig(fig, "v3d_per_domain_metrics.png")


# ── Entry point ───────────────────────────────────────────────────────────

def main() -> None:
    font = setup_style()
    print(f"font: {font}  |  dpi: {DPI}")
    fig_wss_comparison()
    fig_wss_components()
    fig_v3d_per_domain()
    print(f"\nAll figures saved to  {OUT_DIR}")


if __name__ == "__main__":
    main()
