"""Visualization utilities for paper figures.

Provides functions to generate the key plots required for publication:
  - Predicted vs ground-truth scatter plots
  - Error heatmaps on vessel surfaces
  - Regional error bar charts
  - Training loss curves
  - Ablation summary charts
  - Bland-Altman plots (for Task B)
"""
from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
except ImportError:
    plt = None
    mticker = None

from pipeline.config import TARGET_NAMES

# ── Publication-ready style ───────────────────────────────────────────────────
_PAPER_RC: dict = {
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "legend.fontsize": 9,
    "legend.framealpha": 0.88,
    "legend.edgecolor": "0.8",
    "lines.linewidth": 1.8,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.constrained_layout.use": False,
}

# Colorblind-friendly palette for known model families
_MODEL_COLORS: Dict[str, str] = {
    "MLP":             "#636363",   # gray
    "GraphSAGE":       "#3182bd",   # blue
    "Transformer":     "#e6550d",   # orange
    "Transformer+Geom": "#31a354",  # green (main model, highlight)
}
_SEED_LINESTYLES: List[str] = ["-", "--", ":"]

# Target variable display names (u, v, w, p)
_TARGET_DISPLAY = ["u (m/s)", "v (m/s)", "w (m/s)", "p (Pa)"]


def _require_mpl():
    if plt is None:
        raise ImportError("matplotlib is required for visualization")


@contextmanager
def _paper_style():
    """Context manager: temporarily apply paper-ready rcParams."""
    _require_mpl()
    with plt.rc_context(_PAPER_RC):
        yield


# ── Core plot functions ───────────────────────────────────────────────────────

def scatter_pred_vs_true(
    pred: np.ndarray,
    target: np.ndarray,
    target_names: List[str] = None,
    save_path: Optional[str] = None,
    title: str = "Predicted vs Ground Truth",
):
    """2×2 scatter / hexbin plot for u, v, w, p with R² and RMSE annotations."""
    if target_names is None:
        target_names = TARGET_NAMES

    display_names = _TARGET_DISPLAY if len(target_names) == 4 else target_names

    with _paper_style():
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        for idx, (ax, name) in enumerate(zip(axes.flat, display_names)):
            p_vals = pred[:, idx]
            t_vals = target[:, idx]

            lo = min(t_vals.min(), p_vals.min())
            hi = max(t_vals.max(), p_vals.max())
            pad = (hi - lo) * 0.04

            # Hexbin for density visualization
            hb = ax.hexbin(
                t_vals, p_vals,
                gridsize=70,
                cmap="Blues",
                linewidths=0.1,
                mincnt=1,
            )
            fig.colorbar(hb, ax=ax, label="count", fraction=0.046, pad=0.04)

            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
                    "r--", linewidth=1.2, zorder=5, label="ideal")
            ax.set_xlim(lo - pad, hi + pad)
            ax.set_ylim(lo - pad, hi + pad)
            ax.set_xlabel(f"CFD {name}")
            ax.set_ylabel(f"Predicted {name}")
            ax.set_title(name)
            ax.set_aspect("equal")

            # R² and RMSE annotation
            ss_res = float(np.sum((t_vals - p_vals) ** 2))
            ss_tot = float(np.sum((t_vals - t_vals.mean()) ** 2))
            r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
            rmse = float(np.sqrt(np.mean((t_vals - p_vals) ** 2)))
            ax.text(
                0.05, 0.95,
                f"$R^2$={r2:.3f}\nRMSE={rmse:.3f}",
                transform=ax.transAxes,
                va="top",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="0.7", alpha=0.85),
            )

        fig.suptitle(title, fontsize=14, y=1.01)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_regional_bar(
    regional_metrics: Dict[str, Dict[str, float]],
    metric_key: str = "rmse_vel_mag",
    save_path: Optional[str] = None,
    title: str = "Regional Error Comparison",
):
    """Bar chart of a single metric across regions with value labels."""
    regions, values = [], []
    for region, metrics in regional_metrics.items():
        if metric_key in metrics:
            regions.append(region)
            values.append(metrics[metric_key])

    with _paper_style():
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ["#3182bd" if v == min(values) else "#e6550d" if v == max(values) else "#9ecae1"
                  for v in values]
        bars = ax.bar(regions, values, color=colors, edgecolor="white", linewidth=0.8)

        # Value labels on bars
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.01,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=9,
            )

        ax.set_ylabel(metric_key, fontsize=12)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=35)
        ax.set_ylim(0, max(values) * 1.18)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    title: str = "Training Curves",
):
    """Plot train/val loss and key validation metrics over epochs.

    Adds a vertical marker at the best (minimum) val_loss epoch.
    """
    with _paper_style():
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

        # ── Panel 1: loss ──────────────────────────────────────────────────
        ax = axes[0]
        if "train_loss" in history:
            ax.plot(history["train_loss"], label="train", color="#3182bd",
                    linewidth=1.6, alpha=0.9)
        if "val_loss" in history:
            val_arr = np.array(history["val_loss"])
            ax.plot(val_arr, label="val", color="#e6550d",
                    linewidth=1.6, alpha=0.9)

            # Best epoch marker
            best_idx = int(np.argmin(val_arr))
            best_val = float(val_arr[best_idx])
            ax.axvline(best_idx, color="gray", linestyle=":", linewidth=1.0,
                       alpha=0.8)
            ax.scatter([best_idx], [best_val], color="#e6550d", s=40,
                       zorder=6, clip_on=False)
            ax.text(
                best_idx + len(val_arr) * 0.02,
                best_val,
                f"  best={best_val:.3f}\n  @ep{best_idx + 1}",
                fontsize=8, va="center", color="gray",
            )

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Loss (log scale)")
        ax.set_yscale("log")
        ax.legend(loc="upper right")

        # ── Panel 2: validation metrics ────────────────────────────────────
        ax2 = axes[1]
        _metric_styles = [
            ("val_rmse",         "RMSE (all)",    "#636363", "-"),
            ("val_rmse_vel_mag", "RMSE |v|",      "#3182bd", "--"),
            ("val_rmse_p",       "RMSE p",        "#e6550d", ":"),
        ]
        plotted = False
        for key, label, color, ls in _metric_styles:
            if key in history:
                ax2.plot(history[key], label=label, color=color,
                         linestyle=ls, linewidth=1.6)
                plotted = True
        if plotted:
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("RMSE")
            ax2.set_title("Validation Metrics")
            ax2.legend()

        fig.suptitle(title, fontsize=13)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_ablation_summary(
    experiment_names: List[str],
    metric_means: List[float],
    metric_stds: Optional[List[float]] = None,
    metric_name: str = "RMSE |v|",
    highlight_idx: Optional[int] = None,
    save_path: Optional[str] = None,
    title: str = "Ablation Summary",
):
    """Horizontal bar chart with error bars for ablation comparison.

    The bar with the lowest mean is highlighted in green; all others use blue.
    """
    with _paper_style():
        fig, ax = plt.subplots(figsize=(9, max(3.5, len(experiment_names) * 0.65)))
        y_pos = np.arange(len(experiment_names))
        xerr = metric_stds if metric_stds else None

        best_idx_auto = int(np.argmin(metric_means))
        colors = [
            "#31a354" if i == best_idx_auto else
            "#fd8d3c" if (highlight_idx is not None and i == highlight_idx) else
            "#9ecae1"
            for i in range(len(experiment_names))
        ]

        bars = ax.barh(y_pos, metric_means, xerr=xerr,
                       color=colors, capsize=4, edgecolor="white", linewidth=0.8,
                       error_kw=dict(ecolor="0.4", lw=1.2))
        ax.set_yticks(y_pos)
        ax.set_yticklabels(experiment_names)
        ax.set_xlabel(metric_name)
        ax.set_title(title)
        ax.invert_yaxis()

        # Value labels at end of each bar
        x_max = max(metric_means) + (max(metric_stds) if metric_stds else 0)
        for i, (bar, val) in enumerate(zip(bars, metric_means)):
            std_str = f" ±{metric_stds[i]:.3f}" if metric_stds else ""
            ax.text(
                bar.get_width() + x_max * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}{std_str}",
                va="center", fontsize=9,
            )
        ax.set_xlim(0, x_max * 1.28)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return fig


def bland_altman(
    cfd_values: np.ndarray,
    ai_values: np.ndarray,
    indicator_name: str = "TAWSS",
    save_path: Optional[str] = None,
):
    """Bland-Altman plot for Task B indicator agreement."""
    mean_vals = (cfd_values + ai_values) / 2
    diff_vals = ai_values - cfd_values
    md = diff_vals.mean()
    sd = diff_vals.std()

    with _paper_style():
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(mean_vals, diff_vals, alpha=0.5, s=10, color="#3182bd",
                   rasterized=True)
        ax.axhline(md, color="#e6550d", linestyle="-", linewidth=1.5,
                   label=f"Mean diff = {md:.4f}")
        ax.axhline(md + 1.96 * sd, color="gray", linestyle="--",
                   label=f"+1.96 SD = {md + 1.96 * sd:.4f}")
        ax.axhline(md - 1.96 * sd, color="gray", linestyle="--",
                   label=f"-1.96 SD = {md - 1.96 * sd:.4f}")
        ax.set_xlabel(f"Mean of CFD and AI {indicator_name}")
        ax.set_ylabel(f"Difference (AI − CFD) {indicator_name}")
        ax.set_title(f"Bland-Altman: {indicator_name}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return fig


# ── Additional analysis plots ─────────────────────────────────────────────────

def plot_error_distribution(
    pred: np.ndarray,
    target: np.ndarray,
    target_names: List[str] = None,
    save_path: Optional[str] = None,
    title: str = "Error Distribution",
):
    """Histogram of per-variable prediction errors."""
    if target_names is None:
        target_names = TARGET_NAMES
    display_names = _TARGET_DISPLAY if len(target_names) == 4 else target_names

    with _paper_style():
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        for idx, (ax, name) in enumerate(zip(axes.flat, display_names)):
            err = pred[:, idx] - target[:, idx]
            ax.hist(err, bins=100, density=True, alpha=0.75,
                    color="#3182bd", edgecolor="white", linewidth=0.3)
            ax.axvline(0, color="#e6550d", linestyle="--", linewidth=1.0)
            ax.set_xlabel(f"Error ({name})")
            ax.set_ylabel("Density")
            ax.set_title(f"{name}: mean={err.mean():.4f}, std={err.std():.4f}")
        fig.suptitle(title, fontsize=14)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_error_cdf(
    pred: np.ndarray,
    target: np.ndarray,
    target_names: List[str] = None,
    save_path: Optional[str] = None,
    title: str = "Error CDF",
):
    """Cumulative distribution of absolute errors per variable."""
    if target_names is None:
        target_names = TARGET_NAMES
    display_names = _TARGET_DISPLAY if len(target_names) == 4 else target_names

    _colors = ["#636363", "#3182bd", "#e6550d", "#31a354"]

    with _paper_style():
        fig, ax = plt.subplots(figsize=(8, 5))
        for idx, name in enumerate(display_names):
            abs_err = np.abs(pred[:, idx] - target[:, idx])
            sorted_err = np.sort(abs_err)
            cdf = np.arange(1, len(sorted_err) + 1) / len(sorted_err)
            ax.plot(sorted_err, cdf, label=name, color=_colors[idx % len(_colors)],
                    linewidth=1.8)

        vel_err = np.linalg.norm(pred[:, :3] - target[:, :3], axis=1)
        sorted_vel = np.sort(vel_err)
        ax.plot(sorted_vel, np.arange(1, len(sorted_vel) + 1) / len(sorted_vel),
                label="|v| (mag)", linestyle="--", color="purple", linewidth=1.8)

        ax.set_xlabel("Absolute Error")
        ax.set_ylabel("Cumulative Fraction")
        ax.set_title(title)
        ax.legend()
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_per_case_boxplot(
    per_case_metrics: Dict[str, Dict[str, float]],
    metric_keys: List[str] = None,
    save_path: Optional[str] = None,
    title: str = "Per-Case Metric Distribution",
):
    """Boxplot showing metric distribution across test cases."""
    if metric_keys is None:
        metric_keys = ["rmse_vel_mag", "rmse_p", "rmse"]

    n_metrics = len(metric_keys)
    with _paper_style():
        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))
        if n_metrics == 1:
            axes = [axes]

        for ax, key in zip(axes, metric_keys):
            values = [m[key] for m in per_case_metrics.values() if key in m]
            bp = ax.boxplot(
                values, vert=True,
                patch_artist=True,
                medianprops=dict(color="#e6550d", linewidth=2),
                boxprops=dict(facecolor="#c6dbef", alpha=0.8),
                whiskerprops=dict(linewidth=1.2),
                capprops=dict(linewidth=1.2),
                flierprops=dict(marker="o", markersize=3, alpha=0.4),
            )
            ax.set_ylabel(key)
            ax.set_title(f"{key}\n(n={len(values)} cases)")
            ax.set_xticks([])

        fig.suptitle(title, fontsize=14)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_multi_model_curves(
    model_histories: Dict[str, Dict[str, List[float]]],
    metric_key: str = "val_loss",
    save_path: Optional[str] = None,
    title: str = "Model Comparison",
    log_scale: bool = True,
    group_map: Optional[Dict[str, str]] = None,
    group_colors: Optional[Dict[str, str]] = None,
):
    """Overlay training curves from multiple models on a single plot.

    Parameters
    ----------
    group_map : dict, optional
        Maps each label in model_histories to a group name (e.g., model family).
        Labels in the same group share a color; different seeds get different
        line styles.  If None, matplotlib's default color cycle is used.
    group_colors : dict, optional
        Maps group name → hex color.  Falls back to ``_MODEL_COLORS`` for
        known model names; unknown groups get automatic colors.
    """
    with _paper_style():
        fig, ax = plt.subplots(figsize=(9, 5))

        if group_map is not None:
            _effective_colors = dict(_MODEL_COLORS)
            if group_colors:
                _effective_colors.update(group_colors)

            # Assign auto-colors for unknown groups
            auto_palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
            unknown_groups = sorted({g for g in group_map.values()
                                     if g not in _effective_colors})
            for i, grp in enumerate(unknown_groups):
                _effective_colors[grp] = auto_palette[i % len(auto_palette)]

            # Group labels → {group: [label, ...]}
            group_to_labels: Dict[str, List[str]] = defaultdict(list)
            for label in model_histories:
                group_to_labels[group_map.get(label, label)].append(label)

            # Plot group by group, seed by seed
            for group_name in sorted(group_to_labels):
                color = _effective_colors.get(group_name)
                for i, label in enumerate(sorted(group_to_labels[group_name])):
                    ls = _SEED_LINESTYLES[i % len(_SEED_LINESTYLES)]
                    hist = model_histories[label]
                    if metric_key not in hist:
                        continue
                    # Only show group name in legend for the first seed
                    legend_label = group_name if i == 0 else "_nolegend_"
                    ax.plot(hist[metric_key], label=legend_label,
                            color=color, linestyle=ls, linewidth=1.8, alpha=0.88)
        else:
            for label, history in model_histories.items():
                if metric_key in history:
                    ax.plot(history[metric_key], label=label, linewidth=1.8)

        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric_key)
        ax.set_title(title)
        if log_scale:
            ax.set_yscale("log")
        ax.legend(loc="upper right")
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_efficiency_bars(
    rows: List[Dict[str, object]],
    save_path: Optional[str] = None,
    title: str = "Inference Efficiency (single snapshot)",
):
    """Grouped metrics: parameters, mean latency (ms), peak GPU memory (MB).

    Each row should include keys: ``label``, ``total_params``, ``mean_ms``,
    ``peak_memory_mb`` (all numeric except ``label``).

    若行内含有 ``mean_ms_std`` / ``peak_memory_mb_std``（例如跨 seed 汇总），
    则对应子图显示误差棒；参数量一般无波动，不画误差棒。
    """
    if not rows:
        raise ValueError("rows 不能为空")
    labels = [str(r["label"]) for r in rows]
    params_m = [float(r["total_params"]) / 1e6 for r in rows]
    mean_ms = [float(r["mean_ms"]) for r in rows]
    peak_mb = [float(r["peak_memory_mb"]) for r in rows]
    err_ms = [float(r.get("mean_ms_std") or 0) for r in rows]
    err_peak = [float(r.get("peak_memory_mb_std") or 0) for r in rows]
    use_err = any(v > 0 for v in err_ms) or any(v > 0 for v in err_peak)

    with _paper_style():
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
        x = np.arange(len(labels))
        w = 0.65
        err_kw = dict(ecolor="0.35", capsize=3, linewidth=1.0)

        ax = axes[0]
        ax.bar(x, params_m, width=w, color="#3182bd", edgecolor="white", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=28, ha="right")
        ax.set_ylabel("Parameters (M)")
        ax.set_title("Model size")

        ax2 = axes[1]
        ax2.bar(
            x, mean_ms, width=w, color="#e6550d", edgecolor="white", linewidth=0.8,
            yerr=err_ms if use_err else None,
            error_kw=err_kw if use_err else None,
        )
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, rotation=28, ha="right")
        ax2.set_ylabel("Time (ms)")
        ax2.set_title("Latency / snapshot")

        ax3 = axes[2]
        ax3.bar(
            x, peak_mb, width=w, color="#31a354", edgecolor="white", linewidth=0.8,
            yerr=err_peak if use_err else None,
            error_kw=err_kw if use_err else None,
        )
        ax3.set_xticks(x)
        ax3.set_xticklabels(labels, rotation=28, ha="right")
        ax3.set_ylabel("Memory (MB)")
        ax3.set_title("Peak GPU memory")

        fig.suptitle(title, fontsize=13, fontweight="bold")
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_efficiency_bars_per_seed(
    grouped: List[List[Dict[str, object]]],
    save_path: Optional[str] = None,
    title: str = "Latency per seed (single snapshot)",
    metric_key: str = "mean_ms",
    ylabel: str = "Time (ms)",
):
    """分组柱状图：每个实验一组，组内为各 seed 并列柱。

    ``grouped`` 为按实验分组的行列表，每组内行须含 ``seed`` 与 ``metric_key``。
    """
    if not grouped:
        raise ValueError("grouped 不能为空")
    _require_mpl()
    n_exp = len(grouped)
    seeds_sets = [sorted({int(r["seed"]) for r in g}) for g in grouped]
    all_seeds = sorted(set(s for ss in seeds_sets for s in ss))
    n_s = len(all_seeds)
    colors = ["#6baed6", "#fd8d3c", "#74c476", "#9e9ac8", "#fdae6b"][:max(3, n_s)]

    with _paper_style():
        fig, ax = plt.subplots(figsize=(max(10, n_exp * 2.2), 5))
        width = 0.8 / max(1, n_s)
        x0 = np.arange(n_exp)

        for si, seed in enumerate(all_seeds):
            vals = []
            for g in grouped:
                row = next((r for r in g if int(r["seed"]) == seed), None)
                if row is None:
                    vals.append(float("nan"))
                else:
                    vals.append(float(row[metric_key]))
            offset = (si - (n_s - 1) / 2) * width
            ax.bar(
                x0 + offset,
                vals,
                width=width * 0.95,
                label=f"seed {seed}",
                color=colors[si % len(colors)],
                edgecolor="white",
                linewidth=0.6,
            )

        exp_labels = []
        for g in grouped:
            if not g:
                exp_labels.append("?")
            else:
                exp_labels.append(str(g[0].get("short_label", g[0]["label"])).replace("\n", " "))

        ax.set_xticks(x0)
        ax.set_xticklabels(exp_labels, rotation=22, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_pareto_accuracy_latency(
    rows: List[Dict[str, object]],
    save_path: Optional[str] = None,
    title: str = "Accuracy vs inference speed (Pareto)",
    rmse_key: str = "rmse_vel_mag",
    latency_key: str = "mean_ms",
):
    """Scatter: x = latency (ms), y = RMSE |v| (lower-left is better).

    若行内含 ``mean_ms_std`` / ``rmse_vel_mag_std``，则绘制水平/垂直误差棒。
    """
    if not rows:
        raise ValueError("rows 不能为空")
    with _paper_style():
        fig, ax = plt.subplots(figsize=(8, 5.5))
        for r in rows:
            lx = float(r[latency_key])
            ry = float(r[rmse_key])
            xe = float(r.get("mean_ms_std") or 0)
            ye = float(r.get(f"{rmse_key}_std") or r.get("rmse_vel_mag_std") or 0)
            use_err = xe > 0 or ye > 0
            if use_err:
                ax.errorbar(
                    lx, ry,
                    xerr=xe, yerr=ye,
                    fmt="o",
                    ms=7,
                    capsize=3,
                    color="#3182bd",
                    ecolor="0.35",
                    elinewidth=1.0,
                )
            else:
                ax.scatter(lx, ry, s=120, alpha=0.85, edgecolors="white", linewidths=0.8)
            ax.annotate(
                str(r["label"]).split("\n")[0],
                (lx, ry),
                textcoords="offset points",
                xytext=(6, 4),
                fontsize=9,
            )
        ax.set_xlabel("Inference time per snapshot (ms)")
        ax.set_ylabel(f"{rmse_key} (test)")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_pareto_per_seed_points(
    grouped: List[List[Dict[str, object]]],
    save_path: Optional[str] = None,
    title: str = "RMSE |v| vs latency — all seeds",
    rmse_key: str = "rmse_vel_mag",
    latency_key: str = "mean_ms",
):
    """同一实验用同色，各 seed 为散点，便于观察重复性。"""
    if not grouped:
        raise ValueError("grouped 不能为空")
    auto = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    with _paper_style():
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for gi, g in enumerate(grouped):
            color = auto[gi % len(auto)]
            lab = str(g[0].get("short_label", g[0]["label"])).replace("\n", " ") if g else f"g{gi}"
            for j, r in enumerate(sorted(g, key=lambda x: int(x["seed"]))):
                lx = float(r[latency_key])
                ry = float(r[rmse_key])
                ax.scatter(
                    lx, ry, s=90, alpha=0.88, c=color, edgecolors="white",
                    linewidths=0.6, label=lab if j == 0 else "_nolegend_",
                )
                ax.annotate(
                    f"s{int(r['seed'])}",
                    (lx, ry),
                    textcoords="offset points",
                    xytext=(4, 3),
                    fontsize=7,
                    color="0.2",
                )
        ax.legend(loc="best", fontsize=8)

        ax.set_xlabel("Inference time per snapshot (ms)")
        ax.set_ylabel(f"{rmse_key} (test)")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return fig


# ── Multi-model comparison plots ──────────────────────────────────────────────

def plot_multimodel_per_case_boxplot(
    models_data: Dict[str, Dict[str, Dict[str, float]]],
    metric_keys: List[str] = None,
    save_path: Optional[str] = None,
    title: str = "Per-Case Metric Comparison",
):
    """Side-by-side boxplots comparing per-case metrics across multiple models.

    Parameters
    ----------
    models_data : dict
        ``{model_label: {case_name: {metric_key: value}}}``
        Values should be per-case metrics already averaged across seeds.
    """
    if metric_keys is None:
        metric_keys = ["rmse_vel_mag", "rmse_p"]

    model_names = list(models_data.keys())
    n_models = len(model_names)
    n_metrics = len(metric_keys)

    auto_palette = plt.rcParams["axes.prop_cycle"].by_key()["color"] if plt else []
    colors = [
        _MODEL_COLORS.get(m, auto_palette[i % len(auto_palette)] if auto_palette else "#9ecae1")
        for i, m in enumerate(model_names)
    ]

    with _paper_style():
        fig, axes = plt.subplots(1, n_metrics, figsize=(max(6, 2.5 * n_models) * n_metrics, 5))
        if n_metrics == 1:
            axes = [axes]

        for ax, mkey in zip(axes, metric_keys):
            data_per_model = [
                [v[mkey] for v in models_data[mname].values() if mkey in v]
                for mname in model_names
            ]
            positions = list(range(1, n_models + 1))
            bp = ax.boxplot(
                data_per_model, positions=positions, vert=True,
                patch_artist=True, widths=0.55,
                medianprops=dict(color="black", linewidth=2),
                whiskerprops=dict(linewidth=1.2),
                capprops=dict(linewidth=1.2),
                flierprops=dict(marker="o", markersize=3, alpha=0.4),
            )
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.78)

            ax.set_xticks(positions)
            ax.set_xticklabels(model_names, rotation=15, ha="right")
            ax.set_ylabel(mkey)
            n_cases = max((len(models_data[m]) for m in model_names), default=0)
            ax.set_title(f"{mkey}  (n={n_cases} cases)")

        fig.suptitle(title, fontsize=14, y=1.02)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_multimodel_regional_bar(
    models_regional: Dict[str, Dict[str, Dict[str, float]]],
    metric_key: str = "rmse_vel_mag",
    regions: List[str] = None,
    save_path: Optional[str] = None,
    title: str = "Regional Error Comparison",
):
    """Grouped bar chart: x-axis = regions, one bar-group per model.

    Parameters
    ----------
    models_regional : dict
        ``{model_label: {region: {metric_key: value}}}``
        Values should be averaged across seeds.
    regions : list, optional
        Subset and order of regions to display.  Defaults to a canonical
        ordering that includes all available regions.
    """
    _preferred_order = [
        "all", "wall", "interior", "high_curvature",
        "near_wall", "bifurcation", "trunk", "low_curvature", "core_flow",
    ]
    model_names = list(models_regional.keys())
    n_models = len(model_names)

    if regions is None:
        all_regions: set = set()
        for rd in models_regional.values():
            all_regions.update(rd.keys())
        regions = [r for r in _preferred_order if r in all_regions]
        regions += sorted(all_regions - set(regions))

    n_regions = len(regions)
    x = np.arange(n_regions)
    width = 0.8 / max(n_models, 1)
    offsets = np.linspace(
        -(n_models - 1) / 2 * width,
        (n_models - 1) / 2 * width,
        n_models,
    )

    auto_palette = plt.rcParams["axes.prop_cycle"].by_key()["color"] if plt else []
    colors = [
        _MODEL_COLORS.get(m, auto_palette[i % len(auto_palette)] if auto_palette else "#9ecae1")
        for i, m in enumerate(model_names)
    ]

    with _paper_style():
        fig, ax = plt.subplots(figsize=(max(10, 2.2 * n_regions), 5.5))

        for i, (mname, color) in enumerate(zip(model_names, colors)):
            regional = models_regional[mname]
            vals = [
                regional.get(r, {}).get(metric_key, float("nan"))
                for r in regions
            ]
            bars = ax.bar(
                x + offsets[i], vals, width * 0.90,
                label=mname, color=color, alpha=0.82,
                edgecolor="white", linewidth=0.8,
            )
            for bar, v in zip(bars, vals):
                if not np.isnan(v):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() * 1.012,
                        f"{v:.3f}",
                        ha="center", va="bottom", fontsize=7, rotation=0,
                    )

        ax.set_xticks(x)
        ax.set_xticklabels(regions, rotation=30, ha="right")
        ax.set_ylabel(metric_key)
        ax.set_title(title)
        ax.legend(loc="upper right")
        finite_vals = [
            v
            for rd in models_regional.values()
            for r in regions
            for v in [rd.get(r, {}).get(metric_key, float("nan"))]
            if not np.isnan(v)
        ]
        if finite_vals:
            ax.set_ylim(0, max(finite_vals) * 1.22)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_multimodel_scatter(
    models_predictions: Dict[str, Tuple[np.ndarray, np.ndarray]],
    variable: str = "vel_mag",
    max_points: int = 100_000,
    seed: int = 42,
    save_path: Optional[str] = None,
    title: str = "Scatter Comparison",
):
    """Grid of scatter/hexbin plots comparing multiple models on one variable.

    Parameters
    ----------
    models_predictions : dict
        ``{model_label: (pred, true)}`` where each array has shape ``(N, 4)``
        with columns ``[u, v, w, p]``.
    variable : str
        One of ``"u"``, ``"v"``, ``"w"``, ``"p"``, or ``"vel_mag"``.
    """
    _VAR_IDX = {"u": 0, "v": 1, "w": 2, "p": 3}
    _VAR_LABEL = {
        "u": "u (m/s)", "v": "v (m/s)", "w": "w (m/s)",
        "p": "p (Pa)", "vel_mag": "|v| (m/s)",
    }

    model_names = list(models_predictions.keys())
    n_models = len(model_names)
    ncols = min(n_models, 4)
    nrows = (n_models + ncols - 1) // ncols
    rng = np.random.default_rng(seed)
    var_label = _VAR_LABEL.get(variable, variable)

    with _paper_style():
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(5 * ncols, 5 * nrows),
            squeeze=False,
        )
        for idx, (mname, (pred, true)) in enumerate(models_predictions.items()):
            ax = axes[idx // ncols][idx % ncols]

            if variable == "vel_mag":
                t_vals = np.linalg.norm(true[:, :3], axis=1)
                p_vals = np.linalg.norm(pred[:, :3], axis=1)
            elif variable in _VAR_IDX:
                col = _VAR_IDX[variable]
                t_vals = true[:, col]
                p_vals = pred[:, col]
            else:
                raise ValueError(f"Unknown variable: {variable!r}")

            if max_points > 0 and len(t_vals) > max_points:
                chosen = rng.choice(len(t_vals), max_points, replace=False)
                t_vals, p_vals = t_vals[chosen], p_vals[chosen]

            lo = min(float(t_vals.min()), float(p_vals.min()))
            hi = max(float(t_vals.max()), float(p_vals.max()))
            pad = (hi - lo) * 0.04

            hb = ax.hexbin(t_vals, p_vals, gridsize=60, cmap="Blues",
                            linewidths=0.1, mincnt=1)
            fig.colorbar(hb, ax=ax, label="count", fraction=0.046, pad=0.04)
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
                    "r--", linewidth=1.2, zorder=5)
            ax.set_xlim(lo - pad, hi + pad)
            ax.set_ylim(lo - pad, hi + pad)
            ax.set_xlabel(f"CFD {var_label}")
            ax.set_ylabel(f"Predicted {var_label}")
            ax.set_title(mname)
            ax.set_aspect("equal")

            ss_res = float(np.sum((t_vals - p_vals) ** 2))
            ss_tot = float(np.sum((t_vals - t_vals.mean()) ** 2))
            r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
            rmse = float(np.sqrt(np.mean((t_vals - p_vals) ** 2)))
            ax.text(
                0.05, 0.95,
                f"$R^2$={r2:.3f}\nRMSE={rmse:.3f}",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="0.7", alpha=0.85),
            )

        for idx in range(n_models, nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)

        fig.suptitle(f"{title} — {var_label}", fontsize=14)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return fig
