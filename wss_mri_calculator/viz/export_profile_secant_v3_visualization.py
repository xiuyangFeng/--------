#!/usr/bin/env python3
"""Export Profile-Secant V3-only quicklook figures and best/worst ParaView bundles."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULT = (
    ROOT
    / "outputs/wss_mri_calculator/pointcloud_surface_mls_v4/"
    "test35_profile_secant_high_tail_anchor10_v3.json"
)
DEFAULT_PREDICTIONS = (
    ROOT
    / "outputs/wss_mri_calculator/pointcloud_surface_mls_v4/"
    "test35_profile_secant_high_tail_anchor10_v3_predictions.npz"
)
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/wss_mri_calculator/pointcloud_surface_mls_v4/profile_secant_v3"
)
METHOD_KEY = "high_tail_w10_s80"
CMAP_CANDIDATES = (
    ROOT / "tools/cfdpost_cloud_export/GNN_blue_white_red.xml",
    ROOT
    / "例子/04_最佳权重_最好最差病例_postview/slow__XU_YI_CAI__peak_wss/"
    "GNN_blue_white_red.xml",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def magnitude(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return np.linalg.norm(array, axis=1) if array.ndim == 2 else array


def r2_score(truth: np.ndarray, prediction: np.ndarray) -> float:
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    denominator = float(np.sum((truth - np.mean(truth)) ** 2))
    if denominator <= 0:
        return float("nan")
    return float(1.0 - np.sum((truth - prediction) ** 2) / denominator)


def short_case(case_id: str) -> str:
    parts = case_id.split("/")
    if parts[0] == "ILO":
        return f"{parts[1]}_{parts[2]}"
    return parts[-1]


def safe_case_dirname(rank: str, case_id: str) -> str:
    return rank + "__" + "__".join(case_id.split("/"))


def load_case_rows(report: dict) -> dict[str, dict]:
    return {
        row["canonical_id"]: row["methods"][METHOD_KEY]
        for row in report["cases"]
    }


def case_arrays(source: dict[str, np.ndarray], case_id: str) -> dict[str, np.ndarray]:
    mask = np.asarray(source["point_case"]) == case_id
    if not np.any(mask):
        raise RuntimeError(f"prediction archive does not contain {case_id}")
    vector = np.asarray(source["high_tail_wss_vec"][mask], dtype=np.float64)
    truth = np.asarray(source["truth_mag"][mask], dtype=np.float64)
    return {
        "wall": np.asarray(source["wall_mm"][mask], dtype=np.float64),
        "truth": truth,
        "pred": magnitude(vector),
        "pred_vec": vector,
    }


def project_wall(wall: np.ndarray) -> np.ndarray:
    centered = wall - np.mean(wall, axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    return centered @ vh[:2].T


def spatial_panel(
    ax: plt.Axes,
    uv: np.ndarray,
    values: np.ndarray,
    title: str,
    *,
    norm: LogNorm | None = None,
    cmap: str = "turbo",
    background: np.ndarray | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
):
    if background is not None:
        ax.scatter(
            uv[:, 0], uv[:, 1], s=6, color="0.82", alpha=0.28, edgecolors="none"
        )
        indices = background
    else:
        indices = np.ones(len(values), dtype=bool)
    image = ax.scatter(
        uv[indices, 0],
        uv[indices, 1],
        c=values[indices],
        s=24 if background is not None else 13,
        cmap=cmap,
        norm=norm,
        vmin=vmin,
        vmax=vmax,
        edgecolors="none",
    )
    ax.set_title(title)
    ax.set_xlabel("wall PCA-1 (mm)")
    ax.set_ylabel("wall PCA-2 (mm)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.12)
    return image


def plot_case(
    case_id: str,
    metrics: dict,
    arrays: dict[str, np.ndarray],
    output: Path,
    label: str,
) -> None:
    truth = arrays["truth"]
    pred = arrays["pred"]
    uv = project_wall(arrays["wall"])
    high = truth >= np.quantile(truth, 0.90)
    positive = np.concatenate([truth[truth > 0], pred[pred > 0]])
    log_norm = LogNorm(
        vmin=max(float(np.percentile(positive, 1)), 1e-4),
        vmax=float(max(np.max(truth), np.max(pred))),
    )
    abs_error = np.abs(pred - truth)
    err_max = float(max(np.percentile(abs_error, 99), 1e-6))

    fig, axes = plt.subplots(2, 3, figsize=(17.5, 10.2), constrained_layout=True)
    image = spatial_panel(axes[0, 0], uv, truth, "A. CFD truth", norm=log_norm)
    spatial_panel(axes[0, 1], uv, pred, "B. Profile-Secant V3", norm=log_norm)
    err_image = spatial_panel(
        axes[0, 2], uv, abs_error, "C. Absolute error", cmap="magma", vmin=0, vmax=err_max
    )
    fig.colorbar(image, ax=axes[0, :2], label="WSS (Pa, log scale)", shrink=0.83)
    fig.colorbar(err_image, ax=axes[0, 2], label="absolute error (Pa)", shrink=0.83)

    high_image = spatial_panel(
        axes[1, 0],
        uv,
        pred,
        "D. Profile-Secant V3 at truth-defined top 10%",
        norm=log_norm,
        background=high,
    )
    fig.colorbar(high_image, ax=axes[1, 0], label="WSS (Pa, log scale)", shrink=0.83)

    ax = axes[1, 1]
    limit = float(max(np.max(truth[high]), np.max(pred[high])) * 1.04)
    ax.scatter(truth[high], pred[high], s=30, alpha=0.72, color="#2266aa", edgecolors="none")
    ax.plot([0, limit], [0, limit], "k--", lw=1.0)
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"E. High-WSS point accuracy (R²={metrics['high_r2']:.3f})")
    ax.set_xlabel("CFD truth WSS (Pa)")
    ax.set_ylabel("Profile-Secant V3 WSS (Pa)")
    ax.grid(alpha=0.20)

    ax = axes[1, 2]
    order = np.argsort(truth[high])
    rank = np.arange(1, int(np.sum(high)) + 1)
    ax.plot(rank, truth[high][order], color="0.15", lw=1.8, label="CFD truth")
    ax.plot(rank, pred[high][order], color="#2266aa", lw=1.5, label="Profile-Secant V3")
    ax.set_title(
        "F. Truth-sorted high-WSS tail\n"
        f"peak underestimation={100 * metrics['peak_underestimate_fraction']:.1f}%"
    )
    ax.set_xlabel("high-WSS point rank")
    ax.set_ylabel("WSS (Pa)")
    ax.grid(alpha=0.20)
    ax.legend(frameon=False)

    fig.suptitle(
        f"Profile-Secant V3 only — {label} case: {case_id}\n"
        f"overall R²={metrics['raw_r2']:.3f}; high-WSS R²={metrics['high_r2']:.3f}; "
        f"high-WSS NRMSE={metrics['high_nrmse']:.3f}",
        fontsize=14,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_test35_summary(
    case_metrics: dict[str, dict], report: dict, best_id: str, worst_id: str, output: Path
) -> None:
    ids = sorted(case_metrics)
    cohorts = {"AAA": "#3b78b4", "AG": "#df6b22", "ILO": "#4a9b76"}
    colors = [cohorts[case_id.split("/")[0]] for case_id in ids]
    overall = np.array([case_metrics[c]["raw_r2"] for c in ids])
    high_r2 = np.array([case_metrics[c]["high_r2"] for c in ids])
    high_nrmse = np.array([case_metrics[c]["high_nrmse"] for c in ids])
    peak_under = 100 * np.array(
        [case_metrics[c]["peak_underestimate_fraction"] for c in ids]
    )

    fig, axes = plt.subplots(2, 2, figsize=(16.5, 10.2), constrained_layout=True)
    panels = (
        (axes[0, 0], overall, "A. Per-case overall R²", None, False),
        (axes[0, 1], high_r2, "B. Per-case high-WSS R²", 0.90, False),
        (axes[1, 0], high_nrmse, "C. Per-case high-WSS NRMSE", None, True),
        (axes[1, 1], peak_under, "D. Per-case peak underestimation (%)", 10.0, True),
    )
    for ax, values, title, threshold, smaller_better in panels:
        order = np.argsort(values)
        ranked_ids = [ids[i] for i in order]
        ranked_colors = [colors[i] for i in order]
        ax.scatter(np.arange(1, len(ids) + 1), values[order], c=ranked_colors, s=48)
        if threshold is not None:
            ax.axhline(threshold, color="#2f855a", linestyle="--", lw=1.2)
        for selected, marker in ((best_id, "*"), (worst_id, "X")):
            idx = ranked_ids.index(selected)
            ax.scatter(
                idx + 1,
                values[ids.index(selected)],
                marker=marker,
                s=170,
                facecolor="#f5a000" if marker == "*" else "#c53030",
                edgecolor="black",
                linewidth=0.7,
                zorder=5,
            )
        direction = "lower is better" if smaller_better else "higher is better"
        ax.set_title(f"{title}\n{direction}")
        ax.set_xlabel("case rank")
        ax.grid(alpha=0.20)

    aggregate = report["aggregate_case_balanced"][METHOD_KEY]
    pooled = report["pooled_case_q90_high_wss"][METHOD_KEY]
    fig.suptitle(
        "Profile-Secant V3 only — test35 summary\n"
        f"overall R² mean={aggregate['raw_r2']['mean']:.3f}; "
        f"pooled high-WSS R²={pooled['r2']:.3f}; "
        f"case high-WSS R² mean={aggregate['high_r2']['mean']:.3f}; "
        f"peak underestimation mean={100 * aggregate['peak_underestimate_fraction']['mean']:.1f}%",
        fontsize=14,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_best_worst_tail(
    best_id: str,
    worst_id: str,
    case_metrics: dict[str, dict],
    source: dict[str, np.ndarray],
    output: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.5), constrained_layout=True)
    for ax, case_id, label in ((axes[0], best_id, "Best"), (axes[1], worst_id, "Worst")):
        arrays = case_arrays(source, case_id)
        truth, pred = arrays["truth"], arrays["pred"]
        high = truth >= np.quantile(truth, 0.90)
        order = np.argsort(truth[high])
        rank = np.arange(1, int(np.sum(high)) + 1)
        ax.plot(rank, truth[high][order], color="0.15", lw=1.9, label="CFD truth")
        ax.plot(rank, pred[high][order], color="#2266aa", lw=1.5, label="Profile-Secant V3")
        metrics = case_metrics[case_id]
        ax.set_title(
            f"{label}: {case_id}\n"
            f"high-WSS R²={metrics['high_r2']:.3f}; "
            f"NRMSE={metrics['high_nrmse']:.3f}; "
            f"peak under={100 * metrics['peak_underestimate_fraction']:.1f}%"
        )
        ax.set_xlabel("truth-sorted high-WSS point rank")
        ax.set_ylabel("WSS (Pa)")
        ax.grid(alpha=0.20)
        ax.legend(frameon=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_same_point_scatter(
    source: dict[str, np.ndarray], output: Path, scope_label: str
) -> dict:
    groups = np.asarray(source["point_case"])
    truth = np.asarray(source["truth_mag"], dtype=np.float64)
    pred = magnitude(np.asarray(source["high_tail_wss_vec"], dtype=np.float64))
    high = np.zeros(len(truth), dtype=bool)
    for case_id in np.unique(groups):
        mask = groups == case_id
        high[mask] = truth[mask] >= np.quantile(truth[mask], 0.90)

    positive = (truth > 0) & (pred > 0) & np.isfinite(truth) & np.isfinite(pred)
    high_positive = high & positive
    lower = max(float(np.quantile(np.concatenate([truth[positive], pred[positive]]), 0.001)), 1e-5)
    upper = float(max(np.quantile(truth[positive], 0.9999), np.quantile(pred[positive], 0.9999)))
    upper = max(upper, lower * 10.0)

    fig, axes = plt.subplots(1, 2, figsize=(14.8, 6.4), constrained_layout=True)
    panels = (
        (axes[0], positive, f"A. All {scope_label} same-point predictions"),
        (axes[1], high_positive, "B. Truth-defined per-case top 10% WSS"),
    )
    image = None
    for ax, mask, title in panels:
        image = ax.hexbin(
            truth[mask],
            pred[mask],
            gridsize=160,
            bins="log",
            mincnt=1,
            xscale="log",
            yscale="log",
            cmap="viridis",
            extent=(math.log10(lower), math.log10(upper), math.log10(lower), math.log10(upper)),
        )
        ax.plot([lower, upper], [lower, upper], "--", color="white", lw=1.2)
        ax.set_xlim(lower, upper)
        ax.set_ylim(lower, upper)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{title}\nN={int(mask.sum()):,}; pooled R²={r2_score(truth[mask], pred[mask]):.4f}")
        ax.set_xlabel("CFD truth WSS (Pa, log scale)")
        ax.set_ylabel("Profile-Secant V3 WSS (Pa, log scale)")
        ax.grid(alpha=0.16)
    if image is not None:
        fig.colorbar(image, ax=axes, label="log10(points per hexagon)", shrink=0.82)
    fig.suptitle(
        f"Profile-Secant V3 {scope_label} same-point scatter — test35, {len(truth):,} wall nodes",
        fontsize=14,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return {
        "n_points": int(len(truth)),
        "n_high_wss_points": int(high.sum()),
        "pooled_overall_r2": r2_score(truth, pred),
        "pooled_high_wss_r2_same_points": r2_score(truth[high], pred[high]),
    }


def find_stl(case_dir: Path) -> Path:
    stls = sorted(case_dir.glob("*.stl")) + sorted(case_dir.glob("*.STL"))
    if not stls:
        raise FileNotFoundError(f"no STL found in {case_dir}")
    return stls[0]


def write_case_csv(arrays: dict[str, np.ndarray], path: Path) -> None:
    truth = arrays["truth"]
    pred = arrays["pred"]
    pred_vec = arrays["pred_vec"]
    high = (truth >= np.quantile(truth, 0.90)).astype(np.float64)
    case_max = max(float(np.max(truth)), 1e-12)
    header = (
        "x,y,z,wss_cfd,wss_pred,wss_profile_secant_v3,err_wss,abs_err_wss,"
        "high_wss_truth_mask,wss_cfd_over_cfd_max,wss_pred_over_cfd_max,"
        "err_wss_over_cfd_max,abs_err_wss_over_cfd_max,"
        "profile_secant_v3_x,profile_secant_v3_y,profile_secant_v3_z"
    )
    rows = np.column_stack(
        [
            arrays["wall"],
            truth,
            pred,
            pred,
            pred - truth,
            np.abs(pred - truth),
            high,
            truth / case_max,
            pred / case_max,
            (pred - truth) / case_max,
            np.abs(pred - truth) / case_max,
            pred_vec,
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(header + "\n")
        np.savetxt(handle, rows, delimiter=",", fmt="%.8g")


def best_projection_plane(wall: np.ndarray) -> str:
    spans = np.ptp(wall, axis=0)
    pair = tuple(sorted(np.argsort(spans)[-2:].tolist()))
    return {(0, 1): "xy", (0, 2): "xz", (1, 2): "yz"}[pair]


def write_bundle_readme(
    output: Path,
    case_id: str,
    rank: str,
    metrics: dict,
    main_vtp: Path,
    csv_path: Path,
    stl: Path,
    mapping: dict,
    source_points: int,
    fullwall: bool,
) -> None:
    point_scope = (
        f"病例全部 {source_points:,} 个 CFD 壁面节点（sample_count=0）"
        if fullwall
        else f"{source_points:,} 个冻结壁面采样点"
    )
    text = f"""# Profile-Secant V3 后处理打开说明

## 病例与选择规则

- 病例：`{case_id}`
- 标记：`{rank}`
- 选择指标：test35 逐病例 high-WSS R²（Profile-Secant V3）
- 同点评估点：{point_scope}
- VTP 用途：空间可视化；正式指标仍以同点 CSV / JSON 为准

## 同点指标

- overall R²：`{metrics['raw_r2']:.6f}`
- scaled R²：`{metrics['scaled_r2']:.6f}`
- high-WSS R²：`{metrics['high_r2']:.6f}`
- high-WSS NRMSE：`{metrics['high_nrmse']:.6f}`
- peak underestimation：`{100 * metrics['peak_underestimate_fraction']:.3f}%`

## 主文件

| 文件 | 用途 |
| --- | --- |
| `{main_vtp.name}` | ParaView 直接打开 |
| `{csv_path.relative_to(output)}` | {point_scope}的同点数据 |
| `plots/fig_wss_triptych.png` | CFD / Profile-Secant V3 / 绝对误差预览 |
| `GNN_blue_white_red.xml` | ParaView 色标 |
| `{stl.name}` | 源 STL 副本 |

## VTP 标量

| 名称 | 含义 |
| --- | --- |
| `wss_cfd` | CFD WSS 真值，Pa |
| `wss_pred` | Profile-Secant V3 WSS；为兼容 postview 的字段名 |
| `wss_profile_secant_v3` | 与 `wss_pred` 相同的显式模型字段 |
| `err_wss` / `abs_err_wss` | Profile-Secant V3 − CFD / 绝对误差 |
| `high_wss_truth_mask` | 该病例全量/当前同点范围内 CFD top10% 高 WSS 掩码 |
| `profile_secant_v3_x/y/z` | Profile-Secant V3 WSS 矢量三个分量 |
| `map_dist` / `map_valid` | 采样点映射到 STL 的距离与有效标记 |

映射参数：Gaussian radius={mapping['params']['radius']} mm，
max_dist={mapping['params']['max_dist']} mm，fallback={mapping['params']['fallback']}。
该映射只用于展示，不在 VTP 上重新计算 R²。
"""
    (output / "README_后处理打开说明.md").write_text(text, encoding="utf-8")


def export_vtp_bundle(
    case_id: str,
    rank: str,
    metrics: dict,
    arrays: dict[str, np.ndarray],
    output_root: Path,
    fullwall: bool,
) -> dict:
    case_dir = ROOT / "data_new" / case_id
    if not case_dir.is_dir():
        raise FileNotFoundError(f"case directory not found: {case_dir}")
    stl = find_stl(case_dir)
    output = output_root / "06_postview_vtp" / safe_case_dirname(rank, case_id)
    export_dir = output / "_export"
    surface_dir = output / "surface_gaussian"
    plot_dir = output / "plots"
    for path in (export_dir, surface_dir, plot_dir):
        path.mkdir(parents=True, exist_ok=True)

    case_name = short_case(case_id)
    csv_path = export_dir / f"{case_name}__profile_secant_v3__wall.csv"
    write_case_csv(arrays, csv_path)
    stl_copy = output / stl.name
    shutil.copy2(stl, stl_copy)

    mapping_report = surface_dir / f"{case_name}__mapping_report_wall.json"
    scalar_names = (
        "wss_cfd,wss_pred,wss_profile_secant_v3,err_wss,abs_err_wss,"
        "high_wss_truth_mask,wss_cfd_over_cfd_max,wss_pred_over_cfd_max,"
        "err_wss_over_cfd_max,abs_err_wss_over_cfd_max,"
        "profile_secant_v3_x,profile_secant_v3_y,profile_secant_v3_z"
    )
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/cfdpost_cloud_export/map_to_stl_surface.py"),
            "--csv",
            str(csv_path),
            "--stl",
            str(stl),
            "--output-dir",
            str(surface_dir),
            "--method",
            "gaussian",
            "--radius",
            "3.0",
            "--sharpness",
            "2.0",
            "--max-dist",
            "8.0",
            "--fallback",
            "nearest",
            "--scalars",
            scalar_names,
            "--report-json",
            str(mapping_report),
        ],
        check=True,
    )
    mapped_vtp = surface_dir / f"{case_name}__profile_secant_v3__stl_mapped_wall.vtp"
    main_vtp = output / f"{case_name}__profile_secant_v3_surface_wall.vtp"
    shutil.copy2(mapped_vtp, main_vtp)

    for cmap in CMAP_CANDIDATES:
        if cmap.exists():
            shutil.copy2(cmap, output / "GNN_blue_white_red.xml")
            break

    triptych = plot_dir / "fig_wss_triptych.png"
    triptych_report = plot_dir / "fig_wss_triptych_report.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/cfdpost_cloud_export/plot_stl_mapped_triptych.py"),
            "--vtp",
            str(main_vtp),
            "--render",
            "surface",
            "--variable",
            "wss",
            "--plane",
            best_projection_plane(arrays["wall"]),
            "--title",
            f"Profile-Secant V3 — {rank} case — {case_id}",
            "--output",
            str(triptych),
            "--field-cmap",
            "GNN_BWR",
            "--err-cmap",
            "GNN_BWR",
            "--pred-label",
            "Profile-Secant V3",
            "--report-json",
            str(triptych_report),
        ],
        check=True,
    )

    mapping = json.loads(mapping_report.read_text(encoding="utf-8"))
    source_points = int(len(arrays["truth"]))
    write_bundle_readme(
        output,
        case_id,
        rank,
        metrics,
        main_vtp,
        csv_path,
        stl_copy,
        mapping,
        source_points,
        fullwall,
    )
    manifest = {
        "model": "Profile-Secant V3",
        "case": case_id,
        "rank": rank,
        "selection_metric": "per-case high-WSS R2",
        "source_points": source_points,
        "fullwall": fullwall,
        "metrics_same_point": metrics,
        "mapping": mapping,
        "files": {
            "main_vtp": str(main_vtp.resolve()),
            "mapped_vtp": str(mapped_vtp.resolve()),
            "source_csv": str(csv_path.resolve()),
            "triptych": str(triptych.resolve()),
            "readme": str((output / "README_后处理打开说明.md").resolve()),
        },
        "sha256": {
            "main_vtp": sha256(main_vtp),
            "source_csv": sha256(csv_path),
        },
    }
    (output / "manifest_bundle.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def make_contact_sheet(images: list[Path], output: Path) -> None:
    from PIL import Image, ImageDraw, ImageOps

    tile_width, tile_height, label_height = 720, 430, 34
    columns = 3
    rows = int(math.ceil(len(images) / columns))
    canvas = Image.new("RGB", (columns * tile_width, rows * (tile_height + label_height)), "white")
    draw = ImageDraw.Draw(canvas)
    for index, path in enumerate(images):
        with Image.open(path) as source:
            fitted = ImageOps.contain(source.convert("RGB"), (tile_width, tile_height))
        col, row = index % columns, index // columns
        x = col * tile_width + (tile_width - fitted.width) // 2
        y = row * (tile_height + label_height) + (tile_height - fitted.height) // 2
        canvas.paste(fitted, (x, y))
        draw.text(
            (col * tile_width + 10, row * (tile_height + label_height) + tile_height + 7),
            path.stem,
            fill="black",
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, optimize=True)


def write_quicklook_readme(
    quicklook: Path,
    best_id: str,
    worst_id: str,
    bundle_manifests: dict[str, dict],
    n_points: int,
    fullwall: bool,
    include_support_audit: bool,
) -> None:
    scope = (
        f"test35 全部病例的全部 {n_points:,} 个 CFD 壁面节点（sample_count=0）"
        if fullwall
        else f"test35 共 {n_points:,} 个冻结采样点"
    )
    support_lines = ""
    if include_support_audit:
        support_lines = """- `08_best_case_internal_sampling.png`：最佳病例高 WSS 壁面点的内部点筛选
- `09_worst_case_internal_sampling.png`：最差病例高 WSS 壁面点的内部点筛选
- `10_best_case_penetration_risk.png`：最佳病例全壁面穿模风险审计
- `11_worst_case_penetration_risk.png`：最差病例全壁面穿模风险审计
"""
    text = f"""# Profile-Secant V3 纯模型快速图集

本目录只展示 Profile-Secant V3，不混入 V4 final 曲线或空间标量。病例选择以
test35 逐病例 high-WSS R² 为准：

- 最佳病例：`{best_id}`
- 最差病例：`{worst_id}`
- 推理范围：{scope}

## 图件

- `00_overview_contact_sheet.png`：全部主要图件总览
- `01_profile_secant_v3_test35_summary.png`：Profile-Secant V3 的 test35 逐病例指标
- `02_profile_secant_v3_same_point_scatter.png`：全部同点与逐病例 top10% 高 WSS 散点密度图
- `03_best_case_profile_secant_v3.png`：最佳病例空间分布、误差与高 WSS 尾部
- `04_worst_case_profile_secant_v3.png`：最差病例空间分布、误差与高 WSS 尾部
- `05_best_worst_high_wss_tail.png`：最佳/最差病例高 WSS 尾部曲线
- `06_best_case_vtp_triptych.png`：最佳病例 VTP 面片预览
- `07_worst_case_vtp_triptych.png`：最差病例 VTP 面片预览
{support_lines}- `audit_summary.json`：病例选择、指标、产物路径和哈希

## ParaView VTP

- 最佳病例：`../06_postview_vtp/{safe_case_dirname('best', best_id)}/`
- 最差病例：`../06_postview_vtp/{safe_case_dirname('worst', worst_id)}/`

每个目录均包含主 `.vtp`、源 STL、同点 CSV、映射报告、三联预览、色标和打开说明。
VTP 由该病例的 Profile-Secant V3 同点预测映射到 STL，只用于空间展示；
正式 R² 仍读取源 JSON/CSV。
"""
    (quicklook / "README.md").write_text(text, encoding="utf-8")


def profile_cache_path(cache_dir: Path, case_id: str) -> Path:
    path = cache_dir / (case_id.replace("/", "__") + ".npz")
    if not path.is_file():
        raise FileNotFoundError(f"Profile-Secant cache not found: {path}")
    return path


def choose_high_wss_wall_index(cache_path: Path) -> int:
    with np.load(cache_path, allow_pickle=False) as source:
        truth = np.asarray(source["truth_mag"], dtype=np.float64)
        sample_index = np.asarray(source["sample_index"], dtype=np.int64)
        valid = (
            np.isfinite(truth)
            & np.isfinite(np.asarray(source["depth3__selected_fraction"], dtype=np.float64))
            & (np.asarray(source["depth3__selected_samples"]) >= 16)
        )
        if not np.any(valid):
            valid = np.isfinite(truth)
        row = int(np.nanargmax(np.where(valid, truth, np.nan)))
        return int(sample_index[row])


def run_support_audits(
    cache_dir: Path,
    quicklook: Path,
    best_id: str,
    worst_id: str,
) -> tuple[list[Path], dict[str, dict]]:
    results: dict[str, dict] = {}
    images: list[Path] = []
    compat = ROOT / "wss_mri_calculator/src/run_frozen_wss_compat.py"
    sampling_script = ROOT / "wss_mri_calculator/viz/visualize_v4_single_target_sampling.py"
    penetration_script = ROOT / "wss_mri_calculator/viz/visualize_v4_penetration_risk.py"
    for rank, case_id, sample_number, penetration_number in (
        ("best", best_id, 8, 10),
        ("worst", worst_id, 9, 11),
    ):
        cache_path = profile_cache_path(cache_dir, case_id)
        wall_index = choose_high_wss_wall_index(cache_path)
        sampling_png = quicklook / f"{sample_number:02d}_{rank}_case_internal_sampling.png"
        penetration_png = quicklook / f"{penetration_number:02d}_{rank}_case_penetration_risk.png"
        subprocess.run(
            [
                sys.executable,
                str(compat),
                str(sampling_script),
                "--cache",
                str(cache_path),
                "--wall-index",
                str(wall_index),
                "--support-prefix",
                "depth3",
                "--out",
                str(sampling_png),
            ],
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                str(compat),
                str(penetration_script),
                "--cache",
                str(cache_path),
                "--support-prefix",
                "depth3",
                "--zoom-radius-mm",
                "20",
                "--out",
                str(penetration_png),
            ],
            check=True,
        )
        images.extend([sampling_png, penetration_png])
        sampling_json = sampling_png.with_suffix(".json")
        penetration_json = penetration_png.with_suffix(".json")
        results[rank] = {
            "case": case_id,
            "cache": str(cache_path),
            "high_wss_wall_index": wall_index,
            "sampling_figure": str(sampling_png),
            "sampling_report": json.loads(sampling_json.read_text(encoding="utf-8")),
            "penetration_figure": str(penetration_png),
            "penetration_report": json.loads(
                penetration_json.read_text(encoding="utf-8")
            ),
        }
    return images, results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-json", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Profile-Secant cache directory; enables depth-3 sampling and penetration audits.",
    )
    parser.add_argument(
        "--fullwall",
        action="store_true",
        help="Mark outputs as sample_count=0 full-wall inference.",
    )
    args = parser.parse_args()

    result_json = args.result_json.resolve()
    predictions = args.predictions.resolve()
    output_root = args.output_root.resolve()
    cache_dir = args.cache_dir.resolve() if args.cache_dir else None
    quicklook = output_root / "00_quicklook"
    quicklook.mkdir(parents=True, exist_ok=True)

    report = json.loads(result_json.read_text(encoding="utf-8"))
    case_metrics = load_case_rows(report)
    best_id = max(case_metrics, key=lambda c: case_metrics[c]["high_r2"])
    worst_id = min(case_metrics, key=lambda c: case_metrics[c]["high_r2"])

    with np.load(predictions, allow_pickle=False) as archive:
        source = {key: np.asarray(archive[key]) for key in archive.files}

    summary_png = quicklook / "01_profile_secant_v3_test35_summary.png"
    scatter_png = quicklook / "02_profile_secant_v3_same_point_scatter.png"
    best_png = quicklook / "03_best_case_profile_secant_v3.png"
    worst_png = quicklook / "04_worst_case_profile_secant_v3.png"
    tail_png = quicklook / "05_best_worst_high_wss_tail.png"
    plot_test35_summary(case_metrics, report, best_id, worst_id, summary_png)
    scatter_metrics = plot_same_point_scatter(
        source,
        scatter_png,
        "full-wall" if args.fullwall else "sampled",
    )
    plot_case(best_id, case_metrics[best_id], case_arrays(source, best_id), best_png, "best")
    plot_case(worst_id, case_metrics[worst_id], case_arrays(source, worst_id), worst_png, "worst")
    plot_best_worst_tail(best_id, worst_id, case_metrics, source, tail_png)

    bundles = {
        "best": export_vtp_bundle(
            best_id,
            "best",
            case_metrics[best_id],
            case_arrays(source, best_id),
            output_root,
            args.fullwall,
        ),
        "worst": export_vtp_bundle(
            worst_id,
            "worst",
            case_metrics[worst_id],
            case_arrays(source, worst_id),
            output_root,
            args.fullwall,
        ),
    }
    best_triptych = Path(bundles["best"]["files"]["triptych"])
    worst_triptych = Path(bundles["worst"]["files"]["triptych"])
    best_triptych_quick = quicklook / "06_best_case_vtp_triptych.png"
    worst_triptych_quick = quicklook / "07_worst_case_vtp_triptych.png"
    shutil.copy2(best_triptych, best_triptych_quick)
    shutil.copy2(worst_triptych, worst_triptych_quick)

    images = [
        summary_png,
        scatter_png,
        best_png,
        worst_png,
        tail_png,
        best_triptych_quick,
        worst_triptych_quick,
    ]
    support_audits: dict[str, dict] = {}
    if cache_dir is not None:
        support_images, support_audits = run_support_audits(
            cache_dir,
            quicklook,
            best_id,
            worst_id,
        )
        images.extend(support_images)
    make_contact_sheet(sorted(images), quicklook / "00_overview_contact_sheet.png")

    aggregate = report["aggregate_case_balanced"][METHOD_KEY]
    pooled = report["pooled_case_q90_high_wss"][METHOD_KEY]
    audit = {
        "model": "Profile-Secant V3",
        "selection_metric": "per-case high-WSS R2",
        "source_result": str(result_json),
        "source_predictions": str(predictions),
        "source_cache_dir": str(cache_dir) if cache_dir else None,
        "fullwall": bool(args.fullwall),
        "n_points": scatter_metrics["n_points"],
        "source_sha256": {
            "result_json": sha256(result_json),
            "predictions": sha256(predictions),
        },
        "test35": {
            "overall_r2_mean": aggregate["raw_r2"]["mean"],
            "pooled_high_wss_r2": pooled["r2"],
            "case_high_wss_r2_mean": aggregate["high_r2"]["mean"],
            "high_wss_nrmse_mean": aggregate["high_nrmse"]["mean"],
            "peak_underestimate_mean": aggregate["peak_underestimate_fraction"]["mean"],
            **scatter_metrics,
        },
        "best_case": {"case": best_id, "metrics": case_metrics[best_id]},
        "worst_case": {"case": worst_id, "metrics": case_metrics[worst_id]},
        "bundles": bundles,
        "support_audits": support_audits,
    }
    (quicklook / "audit_summary.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_quicklook_readme(
        quicklook,
        best_id,
        worst_id,
        bundles,
        scatter_metrics["n_points"],
        args.fullwall,
        cache_dir is not None,
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
