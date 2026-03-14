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

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from pipeline.config import TARGET_NAMES


def _require_mpl():
    if plt is None:
        raise ImportError("matplotlib is required for visualization")


def scatter_pred_vs_true(
    pred: np.ndarray,
    target: np.ndarray,
    target_names: List[str] = None,
    save_path: Optional[str] = None,
    title: str = "Predicted vs Ground Truth",
):
    """2x2 scatter plot for u, v, w, p."""
    _require_mpl()
    if target_names is None:
        target_names = TARGET_NAMES

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for idx, (ax, name) in enumerate(zip(axes.flat, target_names)):
        p, t = pred[:, idx], target[:, idx]
        ax.scatter(t, p, alpha=0.1, s=1, rasterized=True)
        lims = [min(t.min(), p.min()), max(t.max(), p.max())]
        ax.plot(lims, lims, "r--", linewidth=1)
        ax.set_xlabel(f"CFD {name}")
        ax.set_ylabel(f"Predicted {name}")
        ax.set_title(name)
        ax.set_aspect("equal")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_regional_bar(
    regional_metrics: Dict[str, Dict[str, float]],
    metric_key: str = "rmse_vel_mag",
    save_path: Optional[str] = None,
    title: str = "Regional Error Comparison",
):
    """Bar chart of a single metric across regions."""
    _require_mpl()
    regions = []
    values = []
    for region, metrics in regional_metrics.items():
        if metric_key in metrics:
            regions.append(region)
            values.append(metrics[metric_key])

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(regions, values)
    ax.set_ylabel(metric_key)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    title: str = "Training Curves",
):
    """Plot train/val loss and key metrics over epochs."""
    _require_mpl()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    if "train_loss" in history:
        axes[0].plot(history["train_loss"], label="train")
    if "val_loss" in history:
        axes[0].plot(history["val_loss"], label="val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].set_yscale("log")

    for key in ["val_rmse", "val_rmse_vel_mag", "val_rmse_p"]:
        if key in history:
            axes[1].plot(history[key], label=key.replace("val_", ""))
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("RMSE")
    axes[1].set_title("Validation Metrics")
    axes[1].legend()

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
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
    """Horizontal bar chart with error bars for ablation comparison."""
    _require_mpl()
    fig, ax = plt.subplots(figsize=(8, max(4, len(experiment_names) * 0.5)))
    y_pos = np.arange(len(experiment_names))
    xerr = metric_stds if metric_stds else None
    colors = ["#2196F3"] * len(experiment_names)
    if highlight_idx is not None:
        colors[highlight_idx] = "#FF5722"

    ax.barh(y_pos, metric_means, xerr=xerr, color=colors, capsize=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(experiment_names)
    ax.set_xlabel(metric_name)
    ax.set_title(title)
    ax.invert_yaxis()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return fig


def bland_altman(
    cfd_values: np.ndarray,
    ai_values: np.ndarray,
    indicator_name: str = "TAWSS",
    save_path: Optional[str] = None,
):
    """Bland-Altman plot for Task B indicator agreement."""
    _require_mpl()
    mean_vals = (cfd_values + ai_values) / 2
    diff_vals = ai_values - cfd_values
    md = diff_vals.mean()
    sd = diff_vals.std()

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(mean_vals, diff_vals, alpha=0.5, s=10, rasterized=True)
    ax.axhline(md, color="red", linestyle="-", label=f"Mean diff = {md:.4f}")
    ax.axhline(md + 1.96 * sd, color="gray", linestyle="--", label=f"+1.96 SD = {md + 1.96 * sd:.4f}")
    ax.axhline(md - 1.96 * sd, color="gray", linestyle="--", label=f"-1.96 SD = {md - 1.96 * sd:.4f}")
    ax.set_xlabel(f"Mean of CFD and AI {indicator_name}")
    ax.set_ylabel(f"Difference (AI - CFD) {indicator_name}")
    ax.set_title(f"Bland-Altman: {indicator_name}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return fig
