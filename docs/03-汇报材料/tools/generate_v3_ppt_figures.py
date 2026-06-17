#!/usr/bin/env python3
"""Generate V3 PPT supplement figures from outputs/ and docs.

Auto-generated figures (run via main):
  - v3p_wss_comparison.png             (5-bar w/ AsymW 3-seed mean±std + scatter)
  - v3p_wss_components.png             (z/x/y · Main / AsymW mean±std / AsymW+WssDO)
  - v3p_asymw_seed_consistency.png     (3-seed consistency: bar + best_wss_ep scatter)
  - v3p_asymw_seed_val_wss_history.png (AsymW-a 3-seed val wss_r2_wss curves + best_wss_ep)
  - v3p_val_wss_components_history.png (1×3 val curves w/ best_wss_ep vline)
  - v3d_per_domain_metrics.png         (P / WSS two subplots, case counts in xtick)

User-generated externally (see PPT补充计划.md §7):
  - v3_path_map.png
  - v3_todo_priority.png

Style: 300 DPI · 16:9 (12.8x7.2) · facecolor #fafafa · 浅米色 axes · 隐藏顶/右脊
        suptitle + subtitle + 右下角小灰字签名 "V3 · 2026-05-24"
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "docs" / "03-汇报材料" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Run directories
# ---------------------------------------------------------------------------

RUNS = {
    "Main-PW": ROOT
    / "outputs/field/field_v3_pointnext_localpool_main01_geom_pw_wall13000_near2000_split_AG_v1_seed1_20260508_001936",
    "WSS-a-PW": ROOT
    / "outputs/field/field_v3_pointnext_localpool_wss01a_geom_pw_lambda005_wall13000_near2000_split_AG_v1_seed1_20260507_001902",
    "AsymW-a": ROOT
    / "outputs/field/field_v3_pointnext_localpool_main01_geom_pw_asymw_a_wall13000_near2000_split_AG_v1_seed1_20260522_124946",
    "AsymW-a-seed2": ROOT
    / "outputs/field/field_v3_pointnext_localpool_main01_geom_pw_asymw_a_wall13000_near2000_split_AG_v1_seed2_20260523_124511",
    "AsymW-a-seed3": ROOT
    / "outputs/field/field_v3_pointnext_localpool_main01_geom_pw_asymw_a_wall13000_near2000_split_AG_v1_seed3_20260523_124511",
    "AsymW+WssDO-a": ROOT
    / "outputs/field/field_v3_pointnext_localpool_main01_geom_pw_asymw_wssdo_a_wall13000_near2000_split_AG_v1_seed1_20260523_124511",
    "WssDO-a": ROOT
    / "outputs/field/field_v3_pointnext_localpool_main01_geom_pw_wssdo_a_wall13000_near2000_split_AG_v1_seed1_20260522_131813",
}

V3D_PROBE_P = ROOT / (
    "outputs/field/field_v3d_pointnext_localpool_probe_p01_geom_wall13000_near2000_"
    "split_data_new_v3_v3_seed1_20260521_103843/eval_by_domain_test/metrics_by_domain.json"
)
V3D_PROBE_WSS = ROOT / (
    "outputs/field/field_v3d_pointnext_localpool_probe_wss01_geom_wall13000_near2000_"
    "split_data_new_v3_v3_seed1_20260521_101738/eval_by_domain_test/metrics_by_domain.json"
)

# ---------------------------------------------------------------------------
# Style constants (16:9 PPT-friendly, 300 DPI)
# ---------------------------------------------------------------------------

FIG_W, FIG_H = 12.8, 7.2
DPI = 300
SIGNATURE = f"V3 · {date.today().isoformat()}"
SOURCE_NOTE = "split_AG_v1 · best_wss_model · summary.json + history.csv"

PALETTE = {
    "bg_figure": "#fafafa",
    "bg_axes": "#fdfcf7",
    "text": "#1f2937",
    "muted": "#6b7280",
    "faint": "#9ca3af",
    "grid": "#dcdcdc",
    "spine": "#cccccc",
    "baseline": "#94a3b8",
    "wss_a": "#1f77b4",
    "asymw": "#0f766e",
    "asymw_combo": "#0d9488",
    "wssdo": "#d97706",
    "highlight": "#0b4f9c",
    "ref_line": "#b91c1c",
    "wss_x_color": "#dc2626",
    "wss_y_color": "#7c3aed",
    "wss_z_color": "#059669",
    "bottleneck_band": "#fdecea",
    "zlearn_band": "#e7f5ec",
    "domain_p": "#2563eb",
    "domain_wss": "#ea580c",
}

# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------


def setup_style() -> str:
    from matplotlib import font_manager

    candidates = (
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "Source Han Sans CN",
        "Noto Sans CJK JP",
        "WenQuanYi Micro Hei",
        "SimHei",
    )
    available = {f.name for f in font_manager.fontManager.ttflist}
    font = "DejaVu Sans"
    for fam in candidates:
        if fam in available:
            font = fam
            break
    plt.rcParams["font.sans-serif"] = [font, "WenQuanYi Micro Hei", "DejaVu Sans"]
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.labelsize": 12,
            "axes.labelcolor": PALETTE["text"],
            "axes.edgecolor": PALETTE["spine"],
            "axes.facecolor": PALETTE["bg_axes"],
            "axes.linewidth": 0.8,
            "figure.facecolor": PALETTE["bg_figure"],
            "savefig.facecolor": PALETTE["bg_figure"],
            "text.color": PALETTE["text"],
            "xtick.color": PALETTE["muted"],
            "ytick.color": PALETTE["muted"],
            "legend.framealpha": 0.92,
            "legend.edgecolor": PALETTE["grid"],
            "axes.unicode_minus": False,
        }
    )
    return font


def style_axes(ax, *, y_grid: bool = True, x_grid: bool = False) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(PALETTE["spine"])
    ax.spines["bottom"].set_color(PALETTE["spine"])
    if y_grid:
        ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.8, alpha=0.4)
    if x_grid:
        ax.grid(axis="x", color=PALETTE["grid"], linewidth=0.8, alpha=0.4)
    ax.set_axisbelow(True)


def add_signature(fig, source: str | None = None) -> None:
    """Right-bottom muted signature + optional source note in left-bottom."""
    fig.text(
        0.995,
        0.005,
        SIGNATURE,
        ha="right",
        va="bottom",
        fontsize=8,
        color=PALETTE["faint"],
        style="italic",
    )
    if source:
        fig.text(
            0.005,
            0.005,
            source,
            ha="left",
            va="bottom",
            fontsize=8,
            color=PALETTE["faint"],
            style="italic",
        )


def add_titles(fig, title: str, subtitle: str | None = None) -> None:
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.98)
    if subtitle:
        fig.text(
            0.5,
            0.935,
            subtitle,
            ha="center",
            va="top",
            fontsize=10.5,
            color=PALETTE["muted"],
        )


def _domain_bar_ylim(
    vals: list[float],
    ref_y: float | None,
    *,
    low_pad: float,
    high_pad: float,
) -> tuple[float, float]:
    y_min = min(vals)
    y_max = max(vals)
    if ref_y is not None:
        y_min = min(y_min, ref_y)
        y_max = max(y_max, ref_y)
    return y_min - low_pad, y_max + high_pad


def _annotate_domain_bars(
    ax,
    bars,
    vals: list[float],
    ref_y: float | None,
    *,
    label_pad: float,
) -> None:
    line_clearance = 0.0028
    for b, v in zip(bars, vals):
        y_text = v + label_pad
        if ref_y is not None and abs(y_text - ref_y) < line_clearance:
            y_text = ref_y + line_clearance if v <= ref_y else v + label_pad * 1.6
        ax.text(
            b.get_x() + b.get_width() / 2,
            y_text,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            color=PALETTE["text"],
            fontweight="medium",
            zorder=5,
        )


def save_fig(fig, name: str) -> None:
    path = OUT_DIR / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  saved {path.name}  ({path.stat().st_size / 1024:.0f} KB)")


def best_wss_metrics(run_dir: Path) -> dict:
    with open(run_dir / "summary.json", encoding="utf-8") as f:
        s = json.load(f)
    return s.get("test_metrics_best_wss") or s["test_metrics"]


def best_wss_epoch(run_dir: Path) -> int | None:
    with open(run_dir / "summary.json", encoding="utf-8") as f:
        s = json.load(f)
    return s.get("best_wss_epoch")


# ---------------------------------------------------------------------------
# Figure 1: V3P 主线 WSS 对比（5 柱 · 三 seed 误差棒 + 散点）
# ---------------------------------------------------------------------------


def fig_wss_comparison() -> None:
    asymw_runs = ["AsymW-a", "AsymW-a-seed2", "AsymW-a-seed3"]
    asymw_vals = [best_wss_metrics(RUNS[r])["wss_r2_wss"] for r in asymw_runs]
    asymw_mean = float(np.mean(asymw_vals))
    asymw_std = float(np.std(asymw_vals, ddof=0))

    labels = ["Main-PW", "WSS-a-PW", "AsymW-a\n(三 seed)", "AsymW+\nWssDO-a", "WssDO-a"]
    vals = [
        best_wss_metrics(RUNS["Main-PW"])["wss_r2_wss"],
        best_wss_metrics(RUNS["WSS-a-PW"])["wss_r2_wss"],
        asymw_mean,
        best_wss_metrics(RUNS["AsymW+WssDO-a"])["wss_r2_wss"],
        best_wss_metrics(RUNS["WssDO-a"])["wss_r2_wss"],
    ]
    errs = [0, 0, asymw_std, 0, 0]

    colors = [
        PALETTE["baseline"],
        PALETTE["wss_a"],
        PALETTE["asymw"],
        PALETTE["asymw_combo"],
        PALETTE["wssdo"],
    ]
    best_idx = int(np.argmax(vals))

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    x_pos = np.arange(len(labels))
    bars = ax.bar(
        x_pos,
        vals,
        color=colors,
        edgecolor="white",
        linewidth=1.4,
        width=0.6,
        zorder=3,
    )
    bars[2].set_edgecolor("#0b3a37")
    bars[2].set_linewidth(1.6)
    bars[best_idx].set_edgecolor("white")
    bars[best_idx].set_linewidth(2.4)
    bars[best_idx].set_hatch("")
    bars[best_idx].set_zorder(4)

    ax.errorbar(
        x_pos[2],
        asymw_mean,
        yerr=asymw_std,
        fmt="none",
        ecolor="#0b3a37",
        elinewidth=1.6,
        capsize=7,
        capthick=1.6,
        zorder=5,
    )

    rng = np.random.default_rng(7)
    jitter = rng.uniform(-0.10, 0.10, size=len(asymw_vals))
    ax.scatter(
        x_pos[2] + jitter,
        asymw_vals,
        s=42,
        facecolor="white",
        edgecolor="#0b3a37",
        linewidth=1.4,
        zorder=6,
        label="AsymW-a · 各 seed",
    )

    ax.axhline(
        0.40,
        color=PALETTE["ref_line"],
        ls="--",
        lw=1.4,
        alpha=0.80,
        label="~0.40 经验上限",
        zorder=2,
    )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.32, 0.435)
    ax.set_ylabel("test  wss_r2_wss  (best_wss_model)")
    ax.set_xlabel("")

    for i, (bar, v) in enumerate(zip(bars, vals)):
        if i == 2:
            txt = f"{v:.3f} ± {asymw_std:.3f}\n(n=3)"
            y = v + asymw_std + 0.006
        else:
            txt = f"{v:.3f}"
            y = v + 0.005
        weight = "bold" if i == best_idx else "medium"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            txt,
            ha="center",
            va="bottom",
            fontsize=10,
            color=PALETTE["text"],
            fontweight=weight,
        )

    # Best-value crown marker
    ax.text(
        bars[best_idx].get_x() + bars[best_idx].get_width() / 2,
        0.430,
        "★ 当前最佳",
        ha="center",
        va="top",
        fontsize=9,
        color=PALETTE["highlight"],
        fontweight="bold",
    )

    leg = ax.legend(loc="lower right", fontsize=9.5, framealpha=0.92)
    leg.get_frame().set_edgecolor(PALETTE["grid"])
    style_axes(ax)

    add_titles(
        fig,
        "V3P 主线 WSS 对比（best_wss_model）",
        "split_AG_v1 · 主报 best_wss_model · AsymW-a 三 seed 均值±std · n=3",
    )
    add_signature(
        fig,
        source="data: outputs/field/.../summary.json · test_metrics_best_wss",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.92))
    save_fig(fig, "v3p_wss_comparison.png")


# ---------------------------------------------------------------------------
# Figure 2: WSS 分量对比（z / x / y）· Main / AsymW(±std) / AsymW+WssDO
# ---------------------------------------------------------------------------


def fig_wss_components() -> None:
    asymw_runs = ["AsymW-a", "AsymW-a-seed2", "AsymW-a-seed3"]
    comps = ["wss_r2_wss_z", "wss_r2_wss_x", "wss_r2_wss_y"]
    comp_labels = ["wss_z（轴向）", "wss_x（横向）", "wss_y（横向）"]
    x = np.arange(len(comp_labels))
    width = 0.27

    main_metrics = best_wss_metrics(RUNS["Main-PW"])
    combo_metrics = best_wss_metrics(RUNS["AsymW+WssDO-a"])

    asymw_vals = {c: [best_wss_metrics(RUNS[r])[c] for r in asymw_runs] for c in comps}
    asymw_mean = {c: float(np.mean(asymw_vals[c])) for c in comps}
    asymw_std = {c: float(np.std(asymw_vals[c], ddof=0)) for c in comps}

    main_vals = [main_metrics[c] for c in comps]
    asymw_m_vals = [asymw_mean[c] for c in comps]
    asymw_s_vals = [asymw_std[c] for c in comps]
    combo_vals = [combo_metrics[c] for c in comps]

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    # Background bands: z（绿）= 可学；x/y（红）= 横向瓶颈
    ax.axvspan(-0.5, 0.5, color=PALETTE["zlearn_band"], alpha=0.55, zorder=0)
    ax.axvspan(0.5, 2.5, color=PALETTE["bottleneck_band"], alpha=0.45, zorder=0)

    bars_main = ax.bar(
        x - width,
        main_vals,
        width,
        label="Main-PW (seed=1)",
        color=PALETTE["baseline"],
        edgecolor="white",
        linewidth=1.0,
        zorder=3,
    )
    bars_asymw = ax.bar(
        x,
        asymw_m_vals,
        width,
        yerr=asymw_s_vals,
        ecolor="#0b3a37",
        capsize=4,
        label="AsymW-a (mean±std, n=3)",
        color=PALETTE["asymw"],
        edgecolor="white",
        linewidth=1.0,
        zorder=3,
    )
    bars_combo = ax.bar(
        x + width,
        combo_vals,
        width,
        label="AsymW+WssDO-a (seed=1)",
        color=PALETTE["asymw_combo"],
        edgecolor="white",
        linewidth=1.0,
        zorder=3,
    )

    def label_bars(bars, vals, *, extra_offset: float = 0.0):
        for bar, v in zip(bars, vals):
            y_off = 0.012 if v >= 0 else -0.020
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + y_off + extra_offset,
                f"{v:.3f}",
                ha="center",
                va="bottom" if v >= 0 else "top",
                fontsize=8.5,
                color=PALETTE["text"],
            )

    label_bars(bars_main, main_vals)
    label_bars(bars_asymw, asymw_m_vals, extra_offset=0.012)
    label_bars(bars_combo, combo_vals)

    # Δ vs Main-PW for wss_z
    z_delta_asymw = asymw_mean["wss_r2_wss_z"] - main_metrics["wss_r2_wss_z"]
    ax.annotate(
        f"Δ vs Main = +{z_delta_asymw:.3f}",
        xy=(0, asymw_mean["wss_r2_wss_z"] + 0.04),
        ha="center",
        fontsize=9.5,
        color=PALETTE["asymw"],
        fontweight="bold",
    )

    ax.text(
        1.5,
        0.085,
        "≈0 · 横向分量瓶颈（全局坐标 / 缺局部表示）",
        ha="center",
        fontsize=10,
        color="#7f1d1d",
        fontweight="medium",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(comp_labels)
    ax.set_ylabel("test R²（best_wss_model）")
    ax.set_ylim(-0.04, 0.46)
    ax.axhline(0, color=PALETTE["muted"], lw=0.9, zorder=2)

    leg = ax.legend(loc="upper right", fontsize=9.5)
    leg.get_frame().set_edgecolor(PALETTE["grid"])
    style_axes(ax)

    add_titles(
        fig,
        "WSS 分量对比：Main-PW vs AsymW-a vs AsymW+WssDO-a",
        "split_AG_v1 · best_wss ckpt · AsymW-a 三 seed mean ± std",
    )
    add_signature(fig, source="背景：绿=z 可学；红=x/y 横向瓶颈")
    fig.tight_layout(rect=(0, 0.03, 1, 0.92))
    save_fig(fig, "v3p_wss_components.png")


# ---------------------------------------------------------------------------
# Figure 3 (NEW): AsymW-a 三 seed 一致性
# ---------------------------------------------------------------------------


def fig_asymw_seed_consistency() -> None:
    seeds = [1, 2, 3]
    jobs = ["4957", "4999", "5000"]
    runs = ["AsymW-a", "AsymW-a-seed2", "AsymW-a-seed3"]
    vals = [best_wss_metrics(RUNS[r])["wss_r2_wss"] for r in runs]
    eps = [best_wss_epoch(RUNS[r]) for r in runs]
    mean = float(np.mean(vals))
    std = float(np.std(vals, ddof=0))

    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, FIG_H * 0.78))
    ax_l, ax_r = axes

    # Left: horizontal bars w/ mean line
    seed_labels = [f"seed {s}\n(job {j})" for s, j in zip(seeds, jobs)]
    y_pos = np.arange(len(seeds))
    colors_l = [PALETTE["asymw"]] * 3
    bars = ax_l.barh(
        y_pos,
        vals,
        color=colors_l,
        edgecolor="white",
        linewidth=1.2,
        height=0.55,
        zorder=3,
    )
    ax_l.axvline(
        mean,
        color=PALETTE["ref_line"],
        ls="-",
        lw=2.0,
        alpha=0.85,
        zorder=4,
        label=f"mean = {mean:.3f}",
    )
    ax_l.axvspan(mean - std, mean + std, color=PALETTE["ref_line"], alpha=0.10, zorder=2)
    ax_l.axvline(
        0.40,
        color=PALETTE["muted"],
        ls="--",
        lw=1.0,
        alpha=0.7,
        zorder=2,
        label="0.40 经验上限",
    )

    for bar, v in zip(bars, vals):
        ax_l.text(
            v + 0.003,
            bar.get_y() + bar.get_height() / 2,
            f"{v:.3f}",
            va="center",
            ha="left",
            fontsize=10,
            color=PALETTE["text"],
            fontweight="medium",
        )

    ax_l.set_yticks(y_pos)
    ax_l.set_yticklabels(seed_labels)
    ax_l.invert_yaxis()
    ax_l.set_xlim(0.36, 0.42)
    ax_l.set_xlabel("test  wss_r2_wss")
    ax_l.set_title(
        f"三 seed wss_r2_wss\n(mean ± std = {mean:.3f} ± {std:.3f})",
        fontsize=12,
        pad=10,
    )
    ax_l.legend(loc="lower right", fontsize=9)
    style_axes(ax_l, y_grid=False, x_grid=True)

    # Right: best_wss_ep scatter
    ax_r.set_facecolor(PALETTE["bg_axes"])
    seed_x = np.array(seeds, dtype=float)
    sc = ax_r.scatter(
        seed_x,
        eps,
        s=160,
        c=vals,
        cmap="viridis",
        edgecolor="#0b3a37",
        linewidth=1.4,
        zorder=4,
    )
    for sx, ep, v in zip(seed_x, eps, vals):
        ax_r.annotate(
            f"ep {ep}\n{v:.3f}",
            (sx, ep),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=9.5,
            color=PALETTE["text"],
        )

    ax_r.plot(seed_x, eps, color=PALETTE["faint"], lw=1.0, ls=":", zorder=2)
    ax_r.set_xticks(seed_x)
    ax_r.set_xticklabels([f"seed {s}" for s in seeds])
    ax_r.set_xlabel("seed")
    ax_r.set_ylabel("best_wss_epoch\n(val WSS 峰值选模)")
    ax_r.set_xlim(0.5, 3.5)
    ax_r.set_ylim(0, max(eps) * 1.25)
    ax_r.set_title(
        "best_wss 选模 epoch（val 峰值方差大 · test 标量稳）",
        fontsize=12,
        pad=10,
    )
    cbar = fig.colorbar(sc, ax=ax_r, pad=0.02, fraction=0.046)
    cbar.set_label("wss_r2_wss", fontsize=9, color=PALETTE["muted"])
    cbar.ax.tick_params(labelsize=8)
    style_axes(ax_r)

    add_titles(
        fig,
        "AsymW-a 三 seed 一致性（V3P · split_AG_v1）",
        "wss_weights=[1, 0.05, 0.05, 0.90] · best_wss_model 按 val_wss_r2_wss 选模（非早停 epoch）",
    )
    add_signature(fig, source="data: outputs/field/...asymw_a.../summary.json")
    fig.tight_layout(rect=(0, 0.03, 1, 0.91))
    save_fig(fig, "v3p_asymw_seed_consistency.png")


def fig_asymw_seed_val_wss_history() -> None:
    """AsymW-a 三 seed 验证集标量 WSS R² 曲线，标注各 seed 的 best_wss_epoch。"""
    panels = [
        (1, "4957", "AsymW-a", PALETTE["asymw"]),
        (2, "4999", "AsymW-a-seed2", "#2563eb"),
        (3, "5000", "AsymW-a-seed3", "#7c3aed"),
    ]
    smooth_window = 5

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H * 0.72))
    max_epoch = 0

    for seed, job, run_key, color in panels:
        run_dir = RUNS[run_key]
        csv_path = run_dir / "history.csv"
        if not csv_path.is_file():
            continue
        df = pd.read_csv(csv_path)
        max_epoch = max(max_epoch, int(df["epoch"].max()))
        raw = df["val_wss_r2_wss"].clip(lower=0.25, upper=0.55)
        smoothed = raw.rolling(window=smooth_window, min_periods=1).mean()
        band_min = raw.rolling(window=smooth_window, min_periods=1).min()
        band_max = raw.rolling(window=smooth_window, min_periods=1).max()
        label = f"seed {seed} (job {job})"
        ax.fill_between(df["epoch"], band_min, band_max, color=color, alpha=0.12, zorder=2)
        ax.plot(
            df["epoch"],
            smoothed,
            color=color,
            lw=2.2,
            alpha=0.95,
            label=label,
            zorder=4,
        )
        ep = best_wss_epoch(run_dir)
        if ep is not None:
            peak = float(df.loc[df["epoch"] == ep, "val_wss_r2_wss"].iloc[0])
            ax.axvline(ep, color=color, ls="--", lw=1.3, alpha=0.75, zorder=3)
            ax.scatter([ep], [peak], s=70, color=color, edgecolor="white", linewidth=1.2, zorder=5)
            ax.annotate(
                f"ep {ep}\nval={peak:.3f}",
                (ep, peak),
                xytext=(10 if seed != 3 else -52, 12 if seed != 2 else -28),
                textcoords="offset points",
                fontsize=9,
                color=color,
                fontweight="medium",
                arrowprops=dict(arrowstyle="-", color=color, lw=0.8, alpha=0.6),
            )

    ax.set_xlabel("epoch")
    ax.set_ylabel("val  wss_r2_wss")
    ax.set_title("AsymW-a 三 seed 验证集 WSS 曲线", pad=12, fontsize=14)
    ax.set_xlim(0, max_epoch + 5)
    ax.set_ylim(0.28, 0.53)
    leg = ax.legend(loc="lower right", fontsize=9.5, framealpha=0.92)
    leg.get_frame().set_edgecolor(PALETTE["grid"])
    style_axes(ax)

    add_signature(fig, source="history.csv · jobs 4957/4999/5000 · rolling w=5")
    fig.tight_layout(rect=(0, 0.03, 1, 0.98))
    save_fig(fig, "v3p_asymw_seed_val_wss_history.png")


# ---------------------------------------------------------------------------
# Figure 4: val WSS 分量 R² 曲线 1×3 (Main / AsymW seed1 / AsymW+WssDO)
# ---------------------------------------------------------------------------


def fig_val_wss_history() -> None:
    panels = [
        ("Main-PW (seed=1)", RUNS["Main-PW"]),
        ("AsymW-a (seed=1)", RUNS["AsymW-a"]),
        ("AsymW+WssDO-a (seed=1)", RUNS["AsymW+WssDO-a"]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(FIG_W, FIG_H * 0.78), sharey=True)
    cols = [
        ("val_wss_r2_wss_x", "wss_x", PALETTE["wss_x_color"]),
        ("val_wss_r2_wss_y", "wss_y", PALETTE["wss_y_color"]),
        ("val_wss_r2_wss_z", "wss_z", PALETTE["wss_z_color"]),
    ]

    smooth_window = 5

    for ax, (name, run_dir) in zip(axes, panels):
        csv_path = run_dir / "history.csv"
        if not csv_path.is_file():
            ax.text(0.5, 0.5, f"缺少 {csv_path.name}", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(name, pad=10)
            style_axes(ax)
            continue
        df = pd.read_csv(csv_path)
        for col, lab, color in cols:
            if col not in df.columns:
                continue
            raw = df[col].clip(lower=-0.6, upper=1.05)
            smoothed = raw.rolling(window=smooth_window, min_periods=1).mean()
            ax.fill_between(
                df["epoch"],
                raw.rolling(window=smooth_window, min_periods=1).min(),
                raw.rolling(window=smooth_window, min_periods=1).max(),
                color=color,
                alpha=0.10,
                zorder=2,
            )
            ax.plot(
                df["epoch"],
                smoothed,
                color=color,
                lw=2.0,
                alpha=0.95,
                label=lab,
                zorder=4,
            )
        # best_wss_ep vertical line
        ep = best_wss_epoch(run_dir)
        if ep is not None:
            ax.axvline(ep, color=PALETTE["faint"], ls="--", lw=1.2, alpha=0.9, zorder=3)
            ymin, ymax = ax.get_ylim()
            ax.text(
                ep,
                0.95,
                f"best_wss\nep {ep}",
                ha="center",
                va="top",
                fontsize=8.5,
                color=PALETTE["muted"],
                transform=ax.get_xaxis_transform(),
            )

        ax.axhline(0, color=PALETTE["muted"], lw=0.8)
        ax.set_xlabel("epoch")
        if ax is axes[0]:
            ax.set_ylabel("val R² (rolling mean, w=5)")
        ax.set_title(name, pad=10)
        ax.set_ylim(-0.55, 1.05)
        leg = ax.legend(loc="lower right", fontsize=9, ncol=1)
        leg.get_frame().set_edgecolor(PALETTE["grid"])
        style_axes(ax)

    add_titles(
        fig,
        "验证集 WSS 分量 R² 曲线（过拟合诊断）",
        "1×3 · split_AG_v1 · 浅色填充 = window=5 min/max · 主线 = window=5 mean",
    )
    add_signature(fig, source="data: history.csv · best_wss ep 已标注（虚线）")
    fig.tight_layout(rect=(0, 0.03, 1, 0.92))
    save_fig(fig, "v3p_val_wss_components_history.png")


# ---------------------------------------------------------------------------
# Figure 5: V3D 分域 P / WSS（左右两子图，避免双 y 轴误读）
# ---------------------------------------------------------------------------


def fig_v3d_per_domain() -> None:
    with open(V3D_PROBE_P, encoding="utf-8") as f:
        p_data = json.load(f)
    with open(V3D_PROBE_WSS, encoding="utf-8") as f:
        w_data = json.load(f)

    domains = ["AAA", "AG", "ILO"]
    case_counts = [p_data["by_domain"][d]["num_cases"] for d in domains]
    p_vals = [p_data["by_domain"][d]["metrics"]["r2_p"] for d in domains]
    w_vals = [w_data["by_domain"][d]["metrics"]["wss_r2_wss"] for d in domains]

    p_global = p_data["global"]["metrics"]["r2_p"]
    w_global = w_data["global"]["metrics"]["wss_r2_wss"]

    fig, (ax_p, ax_w) = plt.subplots(1, 2, figsize=(FIG_W, FIG_H * 0.86))
    x = np.arange(len(domains))

    # P subplot
    bars_p = ax_p.bar(
        x,
        p_vals,
        width=0.55,
        color=PALETTE["domain_p"],
        edgecolor="white",
        linewidth=1.3,
        zorder=3,
    )
    ax_p.axhline(
        p_global,
        color=PALETTE["ref_line"],
        ls="--",
        lw=1.5,
        alpha=0.85,
        label=f"全局 r2_p = {p_global:.3f}",
        zorder=2,
    )
    ax_p.set_xticks(x)
    ax_p.set_xticklabels([f"{d}\n(n={n})" for d, n in zip(domains, case_counts)])
    p_ymin, p_ymax = _domain_bar_ylim(p_vals, p_global, low_pad=0.008, high_pad=0.014)
    ax_p.set_ylim(p_ymin, p_ymax)
    ax_p.set_ylabel("Probe-P  r2_p")
    ax_p.set_title("Probe-P · 分域 test", pad=10)
    _annotate_domain_bars(ax_p, bars_p, p_vals, p_global, label_pad=0.0035)
    leg = ax_p.legend(loc="upper right", fontsize=9)
    leg.get_frame().set_edgecolor(PALETTE["grid"])
    style_axes(ax_p)

    # WSS subplot
    bars_w = ax_w.bar(
        x,
        w_vals,
        width=0.55,
        color=PALETTE["domain_wss"],
        edgecolor="white",
        linewidth=1.3,
        zorder=3,
    )
    ax_w.axhline(
        w_global,
        color=PALETTE["ref_line"],
        ls="--",
        lw=1.5,
        alpha=0.85,
        label=f"全局 wss_r2_wss = {w_global:.3f}",
        zorder=2,
    )
    ax_w.set_xticks(x)
    ax_w.set_xticklabels([f"{d}\n(n={n})" for d, n in zip(domains, case_counts)])
    w_ymin, w_ymax = _domain_bar_ylim(w_vals, w_global, low_pad=0.014, high_pad=0.022)
    ax_w.set_ylim(w_ymin, w_ymax)
    ax_w.set_ylabel("Probe-WSS  wss_r2_wss")
    ax_w.set_title("Probe-WSS · 分域 test", pad=10)
    _annotate_domain_bars(ax_w, bars_w, w_vals, w_global, label_pad=0.0045)
    leg2 = ax_w.legend(loc="upper right", fontsize=9)
    leg2.get_frame().set_edgecolor(PALETTE["grid"])
    style_axes(ax_w)

    add_titles(
        fig,
        "V3D post-4901 分域 test 指标",
        "split_data_new_v3_v3 · post-4901 · seed=1 · 总样本 53（AAA 13 · AG 18 · ILO 22）",
    )
    add_signature(fig, source="data: eval_by_domain_test/metrics_by_domain.json")
    fig.tight_layout(rect=(0, 0.03, 1, 0.91))
    save_fig(fig, "v3d_per_domain_metrics.png")


# ---------------------------------------------------------------------------
# Optional: matplotlib placeholders (deprecated — use external tools instead)
# See PPT补充计划.md §7 for draw.io / Figma / Mermaid prompts.
# ---------------------------------------------------------------------------


def fig_path_map_optional() -> None:
    """Deprecated placeholder. Prefer user-generated v3_path_map.png."""
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.text(5, 6.5, "当前带宽：V3P ~0.40 / V3D ~0.24", ha="center", fontsize=13, fontweight="bold")
    boxes = [
        (0.3, 4.2, 2.8, 1.4, "路径 A (P0)\nWSS 局部坐标 / magnitude\nTODO-5,9,10,11", "#dcfce7"),
        (3.6, 4.2, 2.8, 1.4, "路径 B (P1)\n近壁几何与梯度\nTODO-12,17,18", "#dbeafe"),
        (6.9, 4.2, 2.8, 1.4, "路径 C (P1–P2)\n三域数据与归一化\nTODO-1,3,4,19", "#fef3c7"),
        (2.0, 2.0, 2.8, 1.3, "路径 D (P1 并行)\n病例级 / top-k 评估\nTODO-8,20", "#e0e7ff"),
        (5.2, 2.0, 2.8, 1.3, "路径 E (暂缓)\n架构/MTL/发散\nTODO-7,13–16,21–26", "#f3f4f6"),
    ]
    for x, y, w, h, text, color in boxes:
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.02", facecolor=color, edgecolor="#374151", lw=1.2
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9)
    ax.text(
        5,
        1.0,
        "推荐顺序：阶段0 TODO-6/7 ✅ 闭合 → 阶段1 并行 TODO-8 + TODO-5 + TODO-19",
        ha="center",
        fontsize=10,
    )
    add_signature(fig)
    fig.tight_layout()
    save_fig(fig, "v3_path_map.png")


def fig_todo_priority_optional() -> None:
    """Deprecated placeholder. Prefer user-generated v3_todo_priority.png."""
    rows = [
        ("TODO-5", "WSS 局部坐标系", "A", "P0", "未开始（下一跳）"),
        ("TODO-8", "病例级/高 WSS 评估", "D", "P1", "未开始"),
        ("TODO-19", "三域 per-domain 归一化", "C", "P1", "未开始"),
        ("TODO-6", "非对称权重 AsymW", "E→A", "P2", "✅ 三 seed 闭合，建议升级母版"),
        ("TODO-7", "WSS head Dropout", "E", "P2", "维持暂缓（组合未抬升）"),
        ("TODO-1", "数据扩充 257→150+", "C", "P0", "进行中"),
        ("TODO-12", "近壁几何特征", "B", "P1", "未开始"),
        ("TODO-21–26", "发散储备", "E", "P3", "未立项"),
    ]
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H * 0.55))
    ax.axis("off")
    table = ax.table(
        cellText=[[r[0], r[1], r[2], r[3], r[4]] for r in rows],
        colLabels=["ID", "标题", "路径", "优先级", "状态"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#1e40af")
            cell.set_text_props(color="white", fontweight="bold")
        elif col == 3 and row > 0:
            pri = rows[row - 1][3]
            if pri == "P0":
                cell.set_facecolor("#fef2f2")
            elif pri == "P1":
                cell.set_facecolor("#fffbeb")
    ax.set_title("V3 后续优化待办（节选，2026-05-24）", pad=12, fontsize=12)
    add_signature(fig)
    fig.tight_layout()
    save_fig(fig, "v3_todo_priority.png")


def main() -> None:
    font = setup_style()
    print(f"font: {font}  ·  dpi: {DPI}  ·  size: {FIG_W}x{FIG_H}")
    fig_wss_comparison()
    fig_wss_components()
    fig_asymw_seed_consistency()
    fig_asymw_seed_val_wss_history()
    fig_val_wss_history()
    fig_v3d_per_domain()
    print(f"figures saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
