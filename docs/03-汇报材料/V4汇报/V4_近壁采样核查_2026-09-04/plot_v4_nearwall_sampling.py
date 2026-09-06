#!/usr/bin/env python3
"""V4 near-wall definition and PointNet/PointNet++ sampling audit.

The historical trained V4 matrix and the current Centerline-V2 staging use
different near-wall contracts.  This script keeps the following objects
separate in both the numbers and the figures:

1. historical ``int_type`` (nominal 1.5 mm, then capped at 40,000 points);
2. a physically corrected 1.5 mm layer in the current Centerline-V2 staging;
3. the historical V4 epoch-0 5,000-point support sample;
4. PointNet++'s additional 128 FPS centers and 32-NN groups.

Run from the repository root with the GNN environment, for example::

    /public/newhome/cy/.conda/envs/GNN/bin/python \
      docs/03-汇报材料/V4汇报/V4_近壁采样核查_2026-09-04/plot_v4_nearwall_sampling.py
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from scipy.spatial import cKDTree


REPORT_DIR = Path(__file__).resolve().parent
ROOT = REPORT_DIR.parents[3]
CURRENT_ROOT = (
    ROOT
    / "data_wss_pinn/volume_uvwp_bc_rcr_v4_centerline_v2_rawfull_v2_staging_train138_test35"
)
LEGACY_ROOT = ROOT / "data_wss_min"
ATLAS_ROOT = (
    ROOT
    / "outputs/wss_pinn/volume_uvwp_bc_rcr_v4_anatomy_prep_20260903/atlas"
)
CASE_ID = "AG/fast/RAN_QING_BO"
SEED = 1234
EPOCH = 0
THRESHOLD_MM = 1.5
SUPPORT_COUNT = 5000
QUERY_COUNT = 5000
PNPP_CENTROIDS = 128
PNPP_NEIGHBORS = 32

NAVY = "#28536B"
BLUE = "#4C78A8"
ORANGE = "#F28E2B"
RED = "#C44E52"
PURPLE = "#7A5195"
GREEN = "#4C956C"
GRAY = "#B8C0C8"
DARK = "#263238"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def setup_plotting() -> None:
    candidates = [
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"),
    ]
    for path in candidates:
        if path.exists():
            font_manager.fontManager.addfont(str(path))
            family = font_manager.FontProperties(fname=str(path)).get_name()
            plt.rcParams["font.sans-serif"] = [family, "DejaVu Sans"]
            break
    plt.rcParams.update(
        {
            "axes.unicode_minus": False,
            "figure.dpi": 140,
            "savefig.dpi": 240,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
        }
    )


def rng_seed(global_seed: int, epoch: int, case_id: str, stream: str) -> int:
    digest = hashlib.sha256(
        f"{global_seed}|{epoch}|{case_id}|{stream}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "little")


def take_indices(
    population: np.ndarray,
    count: int,
    *,
    global_seed: int,
    epoch: int,
    case_id: str,
    stream: str,
) -> np.ndarray:
    rng = np.random.default_rng(rng_seed(global_seed, epoch, case_id, stream))
    replace = int(count) > len(population)
    return np.asarray(rng.choice(population, size=count, replace=replace), dtype=np.int64)


def deterministic_fps(
    points: np.ndarray, count: int, *, case_id: str, seed: int
) -> np.ndarray:
    """NumPy equivalent of ``wss_pinn.v4.models._deterministic_fps``."""

    points = np.asarray(points, dtype=np.float32)
    count = min(int(count), len(points))
    digest = hashlib.sha256(f"{seed}|{case_id}".encode("utf-8")).digest()
    current = int.from_bytes(digest[:8], "little") % len(points)
    selected = np.empty(count, dtype=np.int64)
    distance_sq = np.full(len(points), np.inf, dtype=np.float32)
    for offset in range(count):
        selected[offset] = current
        delta = points - points[current]
        distance_sq = np.minimum(distance_sq, np.sum(delta * delta, axis=-1))
        current = int(np.argmax(distance_sq))
    return selected


def atlas_radius(case_id: str) -> np.ndarray:
    path = ATLAS_ROOT / "cases" / case_id / "atlas.npz"
    with np.load(path, allow_pickle=False) as payload:
        table = np.asarray(payload["table"])
        columns = [str(value) for value in payload["columns"].tolist()]
    unique = table[:, columns.index("is_child_duplicate")] < 0.5
    return np.asarray(table[unique, columns.index("radius_mm")], dtype=np.float64)


def audit_all_cases() -> tuple[list[dict[str, Any]], np.ndarray]:
    manifest = read_json(CURRENT_ROOT / "manifest.json")
    rows: list[dict[str, Any]] = []
    all_radius: list[np.ndarray] = []

    for record in manifest["cases"]:
        case = read_json(Path(record["manifest"]))
        case_id = str(case["canonical_id"])
        steady = case["files"]["steady"]
        stored_distance = np.load(
            steady["distance_to_wall_mm"]["path"], mmap_mode="r"
        )
        factor = float(
            case["provenance"]["centerline_v2"]["fluent_to_mm_unit_factor"]
        )
        rescale = 1000.0 / factor
        true_threshold_in_stored_frame = THRESHOLD_MM / rescale
        current_true_n = int(np.count_nonzero(stored_distance <= true_threshold_in_stored_frame))
        current_stored_n = int(np.count_nonzero(stored_distance <= THRESHOLD_MM))
        n_points = int(len(stored_distance))

        volume_manifest = read_json(
            Path(case["provenance"]["source_volume_manifest"]["path"])
        )
        is_wall = np.asarray(
            np.load(
                volume_manifest["files"]["interior_is_wall"]["path"],
                mmap_mode="r",
            ),
            dtype=bool,
        )
        legacy_path = Path(
            case["provenance"]["source_legacy_bundle"]["path"]
        )
        with np.load(legacy_path, allow_pickle=False) as legacy:
            legacy_tag = np.asarray(legacy["int_type"]) == 1
            legacy_distance_true = (
                np.asarray(legacy["int_dist_to_wall"], dtype=np.float64) * rescale
            )
        strict = ~is_wall
        legacy_tag_strict = legacy_tag & strict
        legacy_true_candidate = (legacy_distance_true <= THRESHOLD_MM) & strict
        if np.any(legacy_tag_strict):
            legacy_effective_mm = float(np.max(legacy_distance_true[legacy_tag_strict]))
        else:
            legacy_effective_mm = float("nan")

        radius = atlas_radius(case_id)
        all_radius.append(radius)
        radius_q = np.quantile(radius, [0.0, 0.05, 0.5, 0.95, 1.0])
        rows.append(
            {
                "case_id": case_id,
                "role": str(record["role"]),
                "unit_factor": factor,
                "true_mm_rescale": rescale,
                "n_strict": n_points,
                "current_true_1p5_n": current_true_n,
                "current_true_1p5_fraction": current_true_n / n_points,
                "current_stored_region_n": current_stored_n,
                "current_stored_region_fraction": current_stored_n / n_points,
                "current_stored_1p5_true_thickness_mm": THRESHOLD_MM * rescale,
                "legacy_true_1p5_candidate_n": int(np.count_nonzero(legacy_true_candidate)),
                "legacy_true_1p5_candidate_fraction": float(np.mean(legacy_true_candidate[strict])),
                "legacy_tagged_n": int(np.count_nonzero(legacy_tag_strict)),
                "legacy_tagged_fraction": float(np.mean(legacy_tag_strict[strict])),
                "legacy_tag_effective_max_true_mm": legacy_effective_mm,
                "atlas_radius_min_mm": float(radius_q[0]),
                "atlas_radius_p05_mm": float(radius_q[1]),
                "atlas_radius_p50_mm": float(radius_q[2]),
                "atlas_radius_p95_mm": float(radius_q[3]),
                "atlas_radius_max_mm": float(radius_q[4]),
            }
        )
    return rows, np.concatenate(all_radius)


def write_case_summary(rows: list[dict[str, Any]]) -> None:
    path = REPORT_DIR / "v4_nearwall_case_summary.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def case_arrays() -> dict[str, Any]:
    path = CURRENT_ROOT / "cases" / CASE_ID / "manifest.json"
    case = read_json(path)
    steady = case["files"]["steady"]
    factor = float(case["provenance"]["centerline_v2"]["fluent_to_mm_unit_factor"])
    rescale = 1000.0 / factor
    coord_scale_true_mm = float(case["registration"]["coord_scale_mm"]) * rescale
    coords_true_mm = (
        np.asarray(np.load(steady["coords"]["path"], mmap_mode="r"), dtype=np.float32)
        * coord_scale_true_mm
    )
    distance_true_mm = (
        np.asarray(
            np.load(steady["distance_to_wall_mm"]["path"], mmap_mode="r"),
            dtype=np.float32,
        )
        * rescale
    )
    current_near = distance_true_mm <= THRESHOLD_MM

    volume_manifest = read_json(
        Path(case["provenance"]["source_volume_manifest"]["path"])
    )
    is_wall = np.asarray(
        np.load(volume_manifest["files"]["interior_is_wall"]["path"], mmap_mode="r"),
        dtype=bool,
    )
    historical_coords = np.asarray(
        np.load(volume_manifest["files"]["interior_coords"]["path"], mmap_mode="r"),
        dtype=np.float32,
    )
    legacy_path = Path(case["provenance"]["source_legacy_bundle"]["path"]
    )
    with np.load(legacy_path, allow_pickle=False) as legacy:
        legacy_tag = np.asarray(legacy["int_type"]) == 1
        legacy_distance_true_mm = (
            np.asarray(legacy["int_dist_to_wall"], dtype=np.float32) * rescale
        )
    strict = np.flatnonzero(~is_wall)

    support_index = take_indices(
        strict,
        SUPPORT_COUNT,
        global_seed=SEED,
        epoch=EPOCH,
        case_id=CASE_ID,
        stream="support",
    )
    query_index = take_indices(
        strict,
        QUERY_COUNT,
        global_seed=SEED,
        epoch=EPOCH,
        case_id=CASE_ID,
        stream="query",
    )
    support_historical_coords = historical_coords[support_index]
    fps_local = deterministic_fps(
        support_historical_coords,
        PNPP_CENTROIDS,
        case_id=CASE_ID,
        seed=SEED,
    )
    fps_global = support_index[fps_local]
    knn_local = cKDTree(support_historical_coords).query(
        support_historical_coords[fps_local], k=PNPP_NEIGHBORS
    )[1]
    knn_union_local = np.unique(knn_local.reshape(-1))
    knn_union_global = support_index[knn_union_local]
    membership = np.bincount(knn_local.reshape(-1), minlength=len(support_index))

    radius = atlas_radius(CASE_ID)
    return {
        "case": case,
        "factor": factor,
        "rescale": rescale,
        "coords_true_mm": coords_true_mm,
        "distance_true_mm": distance_true_mm,
        "current_near": current_near,
        "legacy_tag": legacy_tag,
        "legacy_distance_true_mm": legacy_distance_true_mm,
        "strict": strict,
        "support_index": support_index,
        "query_index": query_index,
        "fps_local": fps_local,
        "fps_global": fps_global,
        "knn_local": knn_local,
        "knn_union_local": knn_union_local,
        "knn_union_global": knn_union_global,
        "membership": membership,
        "radius": radius,
    }


def write_sampling_csv(data: dict[str, Any]) -> None:
    support = data["support_index"]
    fps_flag = np.zeros(len(support), dtype=np.int8)
    fps_flag[data["fps_local"]] = 1
    path = REPORT_DIR / "RAN_QING_BO_epoch0_support5000.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "support_order",
                "source_point_index",
                "x_true_mm",
                "y_true_mm",
                "z_true_mm",
                "distance_to_wall_true_mm",
                "current_true_near_wall_1p5",
                "legacy_near_wall_tag",
                "legacy_distance_to_wall_true_mm",
                "pnpp_fps_center",
                "pnpp_knn_membership_count",
            ]
        )
        coords = data["coords_true_mm"][support]
        for order, point_index in enumerate(support):
            writer.writerow(
                [
                    order,
                    int(point_index),
                    *[float(value) for value in coords[order]],
                    float(data["distance_true_mm"][point_index]),
                    int(data["current_near"][point_index]),
                    int(data["legacy_tag"][point_index]),
                    float(data["legacy_distance_true_mm"][point_index]),
                    int(fps_flag[order]),
                    int(data["membership"][order]),
                ]
            )


def percentage(values: np.ndarray) -> float:
    return 100.0 * float(np.mean(values))


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.01,
        0.99,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        fontweight="bold",
    )


def scatter_projection(
    ax: plt.Axes,
    coords: np.ndarray,
    near: np.ndarray,
    *,
    axes: tuple[int, int],
    size: float,
    alpha: float,
    label: bool = True,
) -> None:
    core = ~near
    ax.scatter(
        coords[core, axes[0]],
        coords[core, axes[1]],
        s=size,
        c=BLUE,
        alpha=alpha,
        linewidths=0,
        rasterized=True,
        label="> 1.5 mm" if label else None,
    )
    ax.scatter(
        coords[near, axes[0]],
        coords[near, axes[1]],
        s=size,
        c=ORANGE,
        alpha=alpha,
        linewidths=0,
        rasterized=True,
        label="≤ 1.5 mm" if label else None,
    )
    ax.set_aspect("equal", adjustable="box")


def plot_overview(
    rows: list[dict[str, Any]], radius_all: np.ndarray, data: dict[str, Any]
) -> None:
    case_row = next(row for row in rows if row["case_id"] == CASE_ID)
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.3), constrained_layout=True)
    fig.suptitle("BC/RCR V4 近壁 1.5 mm：血管尺度、staging 覆盖与旧标签截断", fontsize=15)

    ax = axes[0, 0]
    bins = np.linspace(0.0, min(float(np.quantile(radius_all, 0.995)), 36.0), 55)
    ax.hist(radius_all, bins=bins, color=NAVY, alpha=0.86, edgecolor="none")
    ax.axvline(THRESHOLD_MM, color=RED, linewidth=2, label="1.5 mm 近壁厚度")
    q = np.quantile(radius_all, [0.05, 0.5, 0.95])
    for value, text, align in zip(q, ["P5", "P50", "P95"], ["right", "center", "left"]):
        ax.axvline(value, color=DARK, linewidth=1, linestyle="--", alpha=0.7)
        ax.text(value, ax.get_ylim()[1] * 0.90, f"{text} {value:.2f}", ha=align, va="top")
    ax.set_xlabel("真实局部管腔半径 (mm)")
    ax.set_ylabel("中心线 atlas 样本数")
    ax.set_title("数据没有“血管壁组织厚度”；可比较的是管腔半径")
    ax.legend(frameon=False, loc="upper right")
    add_panel_label(ax, "a")

    ax = axes[0, 1]
    ordered = sorted(rows, key=lambda item: item["current_true_1p5_fraction"])
    x = np.arange(len(ordered))
    current = np.array([row["current_true_1p5_fraction"] for row in ordered]) * 100.0
    legacy = np.array([row["legacy_tagged_fraction"] for row in ordered]) * 100.0
    ax.plot(x, current, color=ORANGE, linewidth=2.2, label="Centerline-V2 真实 ≤1.5 mm")
    ax.fill_between(x, 0.0, current, color=ORANGE, alpha=0.14)
    ax.plot(x, legacy, color=RED, linewidth=1.7, label="旧 V4 实际 int_type（40k cap）")
    ax.axhline(100.0, color=DARK, linewidth=1, linestyle=":")
    ax.text(len(x) - 1, 99.0, "100% = 全部点", ha="right", va="top")
    ax.set_xlim(0, len(x) - 1)
    ax.set_ylim(0, 105)
    ax.set_xlabel("173 例（按当前 1.5 mm 覆盖率排序）")
    ax.set_ylabel("严格体域 cell 占比 (%)")
    ax.set_title(
        f"当前为 {current.min():.1f}%–{current.max():.1f}%，中位 {np.median(current):.1f}%；无一例全选"
    )
    ax.legend(frameon=False, loc="lower right")
    add_panel_label(ax, "b")

    ax = axes[1, 0]
    coords = data["coords_true_mm"]
    near = data["current_near"]
    z_center = 0.0
    slice_mask = np.abs(coords[:, 2] - z_center) <= 1.5
    scatter_projection(
        ax,
        coords[slice_mask],
        near[slice_mask],
        axes=(0, 1),
        size=2.3,
        alpha=0.65,
    )
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_title(f"RAN_QING_BO 截面：Z=0±1.5 mm（n={int(slice_mask.sum()):,}）")
    ax.legend(frameon=False, loc="best", markerscale=3)
    add_panel_label(ax, "c")

    ax = axes[1, 1]
    labels = [
        "旧距离中\n真实≤1.5 mm",
        "旧 V4 实际\nint_type",
        "Centerline-V2 staging\n真实≤1.5 mm",
    ]
    values = np.array(
        [
            case_row["legacy_true_1p5_candidate_fraction"],
            case_row["legacy_tagged_fraction"],
            case_row["current_true_1p5_fraction"],
        ]
    ) * 100.0
    bars = ax.bar(labels, values, color=[GRAY, RED, ORANGE], width=0.62)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1.2, f"{value:.2f}%", ha="center")
    ax.set_ylim(0, max(70.0, float(values.max() + 9.0)))
    ax.set_ylabel("严格体域 cell 占比 (%)")
    ax.set_title(
        "RAN_QING_BO：旧 cap 把名义 1.5 mm 层变成约 "
        f"{case_row['legacy_tag_effective_max_true_mm']:.3f} mm"
    )
    ax.text(
        0.02,
        0.97,
        f"当前 staging 存储的“1.5”为伪 mm，\n"
        f"对本例实际是 {case_row['current_stored_1p5_true_thickness_mm']:.3f} mm；"
        f"修正后按真实 1.5 mm 重算。",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
    )
    add_panel_label(ax, "d")

    fig.savefig(REPORT_DIR / "fig1_全队列近壁厚度与病例覆盖.png", bbox_inches="tight")
    plt.close(fig)


def plot_pointnet(data: dict[str, Any]) -> None:
    coords = data["coords_true_mm"]
    near = data["current_near"]
    support = data["support_index"]
    support_coords = coords[support]
    support_near = near[support]
    support_legacy = data["legacy_tag"][support]
    rng = np.random.default_rng(20260904)
    display = rng.choice(data["strict"], size=min(70000, len(data["strict"])), replace=False)

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.3), constrained_layout=True)
    fig.suptitle(
        "旧 pre-Centerline-V2 matched-v1.2 PointNet 采样："
        "V4-SP-PN-DATA-s1234 · RAN_QING_BO · epoch 0",
        fontsize=15,
    )

    for ax, proj, labels, panel in [
        (axes[0, 0], (0, 2), ("X (mm)", "Z (mm)"), "a"),
        (axes[0, 1], (1, 2), ("Y (mm)", "Z (mm)"), "b"),
    ]:
        ax.scatter(
            coords[display, proj[0]],
            coords[display, proj[1]],
            s=0.25,
            c=GRAY,
            alpha=0.17,
            linewidths=0,
            rasterized=True,
            label="完整体域（显示抽样）",
        )
        scatter_projection(
            ax,
            support_coords,
            support_near,
            axes=proj,
            size=3.0,
            alpha=0.72,
            label=True,
        )
        tagged = support_coords[support_legacy]
        ax.scatter(
            tagged[:, proj[0]],
            tagged[:, proj[1]],
            s=18,
            facecolors="none",
            edgecolors=RED,
            linewidths=0.65,
            label="旧 V4 int_type" if panel == "a" else None,
        )
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_title("全体域均匀随机 support=5000，不读取 near-wall 标签")
        add_panel_label(ax, panel)
    axes[0, 0].legend(frameon=False, loc="best", markerscale=2.5)

    ax = axes[1, 0]
    slice_mask = np.abs(coords[:, 2]) <= 1.5
    slice_support = np.abs(support_coords[:, 2]) <= 1.5
    scatter_projection(
        ax,
        coords[slice_mask],
        near[slice_mask],
        axes=(0, 1),
        size=2.0,
        alpha=0.35,
        label=False,
    )
    ax.scatter(
        support_coords[slice_support, 0],
        support_coords[slice_support, 1],
        s=22,
        facecolors="none",
        edgecolors=DARK,
        linewidths=0.8,
        label="PointNet support",
    )
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_title("Z=0±1.5 mm 截面：内核仍清晰存在，未全选")
    ax.legend(frameon=False, loc="best")
    add_panel_label(ax, "c")

    ax = axes[1, 1]
    legacy_true_near = data["legacy_distance_true_mm"][support] <= THRESHOLD_MM
    labels = ["当前真实\n≤1.5 mm", "旧距离中\n真实≤1.5 mm", "旧 V4\nint_type"]
    values = np.array(
        [percentage(support_near), percentage(legacy_true_near), percentage(support_legacy)]
    )
    bars = ax.bar(labels, values, color=[ORANGE, GRAY, RED], width=0.62)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1.0, f"{value:.2f}%", ha="center")
    ax.set_ylim(0, 70)
    ax.set_ylabel("support 占比 (%)")
    ax.set_title(
        f"同一个 5000 点样本：当前 1.5 mm 层 {int(support_near.sum())} 点，"
        f"旧标签仅 {int(support_legacy.sum())} 点"
    )
    ax.text(
        0.02,
        0.97,
        "PointNet 对 5000 点做逐点 MLP + global max；\n没有 FPS、距离分层或法线序列。",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
    )
    add_panel_label(ax, "d")

    fig.savefig(REPORT_DIR / "fig2_PointNet_RAN_QING_BO_epoch0采样.png", bbox_inches="tight")
    plt.close(fig)


def plot_pointnetpp(data: dict[str, Any]) -> None:
    coords = data["coords_true_mm"]
    near = data["current_near"]
    support = data["support_index"]
    fps = data["fps_global"]
    union = data["knn_union_global"]
    support_coords = coords[support]
    rng = np.random.default_rng(20260904)
    display = rng.choice(data["strict"], size=min(70000, len(data["strict"])), replace=False)

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.3), constrained_layout=True)
    fig.suptitle(
        "旧 pre-Centerline-V2 matched-v1.2 PointNet++："
        "同一 5000 support 上的 FPS-128 + 32-NN",
        fontsize=15,
    )

    for ax, proj, labels, panel in [
        (axes[0, 0], (0, 2), ("X (mm)", "Z (mm)"), "a"),
        (axes[0, 1], (1, 2), ("Y (mm)", "Z (mm)"), "b"),
    ]:
        ax.scatter(
            coords[display, proj[0]],
            coords[display, proj[1]],
            s=0.25,
            c=GRAY,
            alpha=0.14,
            linewidths=0,
            rasterized=True,
        )
        ax.scatter(
            support_coords[:, proj[0]],
            support_coords[:, proj[1]],
            s=2.0,
            c=NAVY,
            alpha=0.34,
            linewidths=0,
            rasterized=True,
            label="shared support=5000" if panel == "a" else None,
        )
        fps_near = near[fps]
        ax.scatter(
            coords[fps[~fps_near], proj[0]],
            coords[fps[~fps_near], proj[1]],
            s=28,
            c=BLUE,
            edgecolors=DARK,
            linewidths=0.35,
            label="FPS core" if panel == "a" else None,
        )
        ax.scatter(
            coords[fps[fps_near], proj[0]],
            coords[fps[fps_near], proj[1]],
            s=30,
            c=ORANGE,
            edgecolors=DARK,
            linewidths=0.35,
            label="FPS near-wall" if panel == "a" else None,
        )
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_aspect("equal", adjustable="box")
        ax.set_title("FPS 倾向选空间边界，但这不等于沿法线取速度剖面")
        add_panel_label(ax, panel)
    axes[0, 0].legend(frameon=False, loc="best", markerscale=1.4)

    ax = axes[1, 0]
    bins = np.linspace(0.0, 6.0, 45)
    ax.hist(
        data["distance_true_mm"][support],
        bins=bins,
        density=True,
        histtype="step",
        linewidth=2,
        color=NAVY,
        label="support=5000",
    )
    ax.hist(
        data["distance_true_mm"][union],
        bins=bins,
        density=True,
        histtype="step",
        linewidth=2,
        color=GREEN,
        label=f"32-NN unique={len(union)}",
    )
    ax.hist(
        data["distance_true_mm"][fps],
        bins=bins,
        density=True,
        histtype="step",
        linewidth=2.2,
        color=PURPLE,
        label="FPS centers=128",
    )
    ax.axvline(THRESHOLD_MM, color=RED, linewidth=1.8, linestyle="--", label="1.5 mm")
    ax.set_xlabel("到物理壁面距离 (mm)")
    ax.set_ylabel("密度")
    ax.set_title("FPS 中心近壁偏多；32-NN 并集的分布又接近 support")
    ax.legend(frameon=False, loc="upper right")
    add_panel_label(ax, "c")

    ax = axes[1, 1]
    stages = ["support\n5000", "FPS center\n128", f"32-NN union\n{len(union)}"]
    current_values = np.array(
        [
            percentage(near[support]),
            percentage(near[fps]),
            percentage(near[union]),
        ]
    )
    legacy_values = np.array(
        [
            percentage(data["legacy_tag"][support]),
            percentage(data["legacy_tag"][fps]),
            percentage(data["legacy_tag"][union]),
        ]
    )
    x = np.arange(3)
    width = 0.34
    bars_a = ax.bar(x - width / 2, current_values, width, color=ORANGE, label="当前真实 ≤1.5 mm")
    bars_b = ax.bar(x + width / 2, legacy_values, width, color=RED, label="旧 V4 int_type")
    for bars in [bars_a, bars_b]:
        for bar in bars:
            value = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, value + 1.0, f"{value:.1f}%", ha="center", fontsize=8.5)
    ax.set_xticks(x, stages)
    ax.set_ylim(0, 100)
    ax.set_ylabel("点占比 (%)")
    ax.set_title(
        "旧 matched-v1.2：单层 FPS128-k32；"
        "formal B：三层 D2 c125-k128"
    )
    ax.legend(frameon=False, loc="upper right")
    add_panel_label(ax, "d")

    fig.savefig(REPORT_DIR / "fig3_PointNetPP_RAN_QING_BO_epoch0采样.png", bbox_inches="tight")
    plt.close(fig)


def write_vtp(data: dict[str, Any]) -> bool:
    """Write a compact ParaView point cloud when VTK is available."""

    try:
        import vtk
        from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
    except ImportError:
        return False

    rng = np.random.default_rng(20260904)
    background = rng.choice(
        data["strict"], size=min(50000, len(data["strict"])), replace=False
    )
    indices = np.unique(np.concatenate([background, data["support_index"]]))
    support_lookup = {int(value): idx for idx, value in enumerate(data["support_index"])}
    is_support = np.array([int(value) in support_lookup for value in indices], dtype=np.int8)
    is_fps_set = set(int(value) for value in data["fps_global"])
    is_fps = np.array([int(value) in is_fps_set for value in indices], dtype=np.int8)
    membership = np.zeros(len(indices), dtype=np.int16)
    for row, point_index in enumerate(indices):
        local = support_lookup.get(int(point_index))
        if local is not None:
            membership[row] = int(data["membership"][local])

    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(data["coords_true_mm"][indices].astype(np.float32), deep=True))
    poly = vtk.vtkPolyData()
    poly.SetPoints(points)
    cells = np.empty(2 * len(indices), dtype=np.int64)
    cells[0::2] = 1
    cells[1::2] = np.arange(len(indices), dtype=np.int64)
    cell_array = vtk.vtkCellArray()
    cell_array.SetCells(len(indices), numpy_to_vtkIdTypeArray(cells, deep=True))
    poly.SetVerts(cell_array)

    arrays = {
        "source_point_index": indices.astype(np.int64),
        "distance_to_wall_true_mm": data["distance_true_mm"][indices].astype(np.float32),
        "near_wall_true_1p5": data["current_near"][indices].astype(np.int8),
        "legacy_near_wall_tag": data["legacy_tag"][indices].astype(np.int8),
        "is_support_epoch0": is_support,
        "is_pnpp_fps_center": is_fps,
        "pnpp_knn_membership_count": membership,
    }
    for name, values in arrays.items():
        vtk_array = numpy_to_vtk(np.ascontiguousarray(values), deep=True)
        vtk_array.SetName(name)
        poly.GetPointData().AddArray(vtk_array)

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(REPORT_DIR / "RAN_QING_BO_V4_sampling_audit.vtp"))
    writer.SetInputData(poly)
    writer.SetDataModeToBinary()
    writer.SetCompressorTypeToZLib()
    return bool(writer.Write())


def write_summary(
    rows: list[dict[str, Any]], radius_all: np.ndarray, data: dict[str, Any], vtp_written: bool
) -> None:
    fractions = np.array([row["current_true_1p5_fraction"] for row in rows])
    test_fractions = np.array(
        [row["current_true_1p5_fraction"] for row in rows if row["role"] == "test"]
    )
    legacy_fractions = np.array([row["legacy_tagged_fraction"] for row in rows])
    total = sum(int(row["n_strict"]) for row in rows)
    total_near = sum(int(row["current_true_1p5_n"]) for row in rows)
    case_row = next(row for row in rows if row["case_id"] == CASE_ID)
    support = data["support_index"]
    query = data["query_index"]
    fps = data["fps_global"]
    union = data["knn_union_global"]

    summary = {
        "schema_version": 1,
        "case_id": CASE_ID,
        "interpretation": {
            "wall_tissue_thickness_available": False,
            "available_geometry": "fluid-lumen points and local lumen radius",
            "near_wall_definition": "interior distance to physical wall",
        },
        "unit_contract": {
            "formula": "true_mm = stored_legacy_or_staging_length * 1000 / fluent_to_mm_unit_factor",
            "case_unit_factor": data["factor"],
            "case_true_mm_rescale": data["rescale"],
            "case_stored_threshold_true_mm": THRESHOLD_MM * data["rescale"],
        },
        "atlas_radius_true_mm_all_173": {
            "n_samples": int(len(radius_all)),
            "min_p05_p50_p95_max": [
                float(value)
                for value in np.quantile(radius_all, [0.0, 0.05, 0.5, 0.95, 1.0])
            ],
        },
        "current_centerline_v2_staging_untrained": {
            "n_cases": len(rows),
            "pooled_strict_points": int(total),
            "pooled_true_1p5_points": int(total_near),
            "pooled_true_1p5_fraction": total_near / total,
            "per_case_fraction_min_p50_max": [
                float(fractions.min()),
                float(np.median(fractions)),
                float(fractions.max()),
            ],
            "test35_fraction_min_p50_max": [
                float(test_fractions.min()),
                float(np.median(test_fractions)),
                float(test_fractions.max()),
            ],
            "n_cases_all_points_selected": int(np.count_nonzero(fractions >= 1.0)),
        },
        "legacy_trained_v4": {
            "per_case_tag_fraction_min_p50_max": [
                float(legacy_fractions.min()),
                float(np.median(legacy_fractions)),
                float(legacy_fractions.max()),
            ],
            "near_wall_is_training_sampler": False,
            "near_wall_is_evaluation_region_only": True,
        },
        "representative_case": {
            **case_row,
            "atlas_radius_true_mm_min_p05_p25_p50_p75_p95_max": [
                float(value)
                for value in np.quantile(data["radius"], [0, 0.05, 0.25, 0.5, 0.75, 0.95, 1])
            ],
        },
        "epoch0_sampling": {
            "support": {
                "n": len(support),
                "current_true_1p5_n": int(np.count_nonzero(data["current_near"][support])),
                "legacy_true_1p5_n": int(
                    np.count_nonzero(data["legacy_distance_true_mm"][support] <= THRESHOLD_MM)
                ),
                "legacy_tagged_n": int(np.count_nonzero(data["legacy_tag"][support])),
            },
            "query": {
                "n": len(query),
                "current_true_1p5_n": int(np.count_nonzero(data["current_near"][query])),
                "legacy_true_1p5_n": int(
                    np.count_nonzero(data["legacy_distance_true_mm"][query] <= THRESHOLD_MM)
                ),
                "legacy_tagged_n": int(np.count_nonzero(data["legacy_tag"][query])),
            },
            "pointnet": "uses all 5000 support points; point MLP + global max",
            "pointnetpp": {
                "fps_centers": len(fps),
                "neighbors_per_center": PNPP_NEIGHBORS,
                "knn_unique_union": len(union),
                "fps_current_true_1p5_n": int(np.count_nonzero(data["current_near"][fps])),
                "fps_legacy_tagged_n": int(np.count_nonzero(data["legacy_tag"][fps])),
                "knn_union_current_true_1p5_n": int(
                    np.count_nonzero(data["current_near"][union])
                ),
                "knn_union_legacy_tagged_n": int(np.count_nonzero(data["legacy_tag"][union])),
            },
            "base_support_identical_between_backbones": True,
        },
        "outputs": {
            "vtp_written": bool(vtp_written),
            "figures": [
                "fig1_全队列近壁厚度与病例覆盖.png",
                "fig2_PointNet_RAN_QING_BO_epoch0采样.png",
                "fig3_PointNetPP_RAN_QING_BO_epoch0采样.png",
            ],
        },
    }
    (REPORT_DIR / "audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    setup_plotting()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows, radius_all = audit_all_cases()
    data = case_arrays()
    write_case_summary(rows)
    write_sampling_csv(data)
    plot_overview(rows, radius_all, data)
    plot_pointnet(data)
    plot_pointnetpp(data)
    vtp_written = write_vtp(data)
    write_summary(rows, radius_all, data, vtp_written)
    print(
        json.dumps(
            {
                "report_dir": str(REPORT_DIR),
                "cases": len(rows),
                "representative_case": CASE_ID,
                "vtp_written": vtp_written,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
