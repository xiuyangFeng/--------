#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export an exact, deterministic three-stage PointNet++ SA trace.

The tool visualizes model-space FPS centers and ball-query memberships without
running or training a network.  Grouping is computed in normalized coordinates;
rendered VTP/PNG coordinates are converted back to the registered millimetre
frame for anatomical readability.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from training_wss_min import config as C
from training_wss_min import dataset as D
from training_wss_min.baseline_models import sample_and_group
from training_wss_min.tools.visualize_sampling import _bbox_diag, _load_stl, _write_vtp


DEFAULT_CONFIG = C.PROJECT_ROOT / "training_wss_min/configs/pointnetpp_sa_foundation/sa3_xyzgeom.json"
DEFAULT_CASE = "fast/RAN_QING_BO"
DEFAULT_OUTPUT = C.PROJECT_ROOT / "例子/06_PointNet++_SA三层采样与分组"


def _digest(*arrays: np.ndarray) -> str:
    h = hashlib.sha256()
    for arr in arrays:
        h.update(np.ascontiguousarray(arr).view(np.uint8))
    return h.hexdigest()


def _write_edges(center_xyz: np.ndarray, member_xyz: np.ndarray, output: Path,
                 center_ids: np.ndarray, source_indices: np.ndarray,
                 distances: np.ndarray) -> None:
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk

    points_np = np.empty((len(center_xyz) * 2, 3), dtype=np.float64)
    points_np[0::2] = center_xyz
    points_np[1::2] = member_xyz
    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(points_np, deep=True))

    lines = vtk.vtkCellArray()
    for i in range(len(center_xyz)):
        lines.InsertNextCell(2)
        lines.InsertCellPoint(2 * i)
        lines.InsertCellPoint(2 * i + 1)

    poly = vtk.vtkPolyData()
    poly.SetPoints(points)
    poly.SetLines(lines)
    for name, values in {
        "center_id": center_ids,
        "source_index": source_indices,
        "distance_norm": distances,
    }.items():
        arr = numpy_to_vtk(np.ascontiguousarray(values), deep=True)
        arr.SetName(name)
        poly.GetCellData().AddArray(arr)

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(output))
    writer.SetInputData(poly)
    if writer.Write() != 1:
        raise RuntimeError(f"VTP edge write failed: {output}")


def _plot_overview(stages: list[dict], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    views = ((0, 1, "X-Y"), (0, 2, "X-Z"), (1, 2, "Y-Z"))
    fig, axes = plt.subplots(len(stages), 3, figsize=(15, 4.7 * len(stages)),
                             constrained_layout=True)
    axes = np.atleast_2d(axes)
    for row_axes, stage in zip(axes, stages):
        src = stage["source_xyz_mm"]
        centers = stage["center_xyz_mm"]
        memberships = stage["membership_count"]
        for ax, (a, b, label) in zip(row_axes, views):
            ax.scatter(src[:, a], src[:, b], c=memberships, cmap="viridis",
                       vmin=0, vmax=max(1, int(memberships.max())), s=6,
                       alpha=0.78, linewidths=0)
            ax.scatter(centers[:, a], centers[:, b], c="#E31A1C", s=16,
                       marker="o", edgecolors="white", linewidths=0.25)
            ax.set_aspect("equal")
            ax.set_xlabel(f"{label[0]} (mm)")
            ax.set_ylabel(f"{label[2]} (mm)")
            ax.set_title(
                f"SA{stage['stage']} {label}: {len(src)} source / "
                f"{len(centers)} centers\ncolor=group membership count; red=center"
            )
    fig.suptitle("PointNet++ SA centers and grouped-point overlap", fontsize=16)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def _plot_representative_groups(stages: list[dict], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    n_groups = 6
    colors = plt.get_cmap("tab10").colors
    fig, axes = plt.subplots(len(stages), n_groups,
                             figsize=(3.2 * n_groups, 3.5 * len(stages)),
                             constrained_layout=True)
    axes = np.atleast_2d(axes)
    for row_axes, stage in zip(axes, stages):
        centers_norm = stage["center_xyz_norm"]
        pick = D.farthest_point_sample(
            centers_norm, min(n_groups, len(centers_norm)), seed=700 + stage["stage"]
        )
        for panel, (ax, center_id) in enumerate(zip(row_axes, pick)):
            mask = stage["row"] == center_id
            members = stage["source_xyz_mm"][stage["col"][mask]]
            center = stage["center_xyz_mm"][center_id]
            color = colors[panel % len(colors)]
            ax.scatter(members[:, 0], members[:, 2], s=36, c=[color], alpha=0.85,
                       edgecolors="white", linewidths=0.35)
            ax.scatter(center[0], center[2], s=110, c="#E31A1C", marker="*",
                       edgecolors="black", linewidths=0.5, zorder=3)
            ax.add_patch(Circle((center[0], center[2]), stage["radius_mm"], fill=False,
                                linestyle="--", linewidth=0.9, color="#555555"))
            pad = stage["radius_mm"] * 1.15
            ax.set_xlim(center[0] - pad, center[0] + pad)
            ax.set_ylim(center[2] - pad, center[2] + pad)
            ax.set_aspect("equal")
            ax.set_title(f"SA{stage['stage']} group {center_id}\nn={mask.sum()}")
            if panel == 0:
                ax.set_ylabel("Z (mm)")
            ax.set_xlabel("X (mm)")
    fig.suptitle("Representative exact ball-query groups (star=center, dashed=radius)",
                 fontsize=16)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def _stage_summary(stage: dict) -> dict:
    sizes = stage["group_size"]
    memberships = stage["membership_count"]
    return {
        "stage": stage["stage"],
        "source_points": int(len(stage["source_xyz_norm"])),
        "center_points": int(len(stage["center_xyz_norm"])),
        "ratio": stage["ratio"],
        "radius_norm": stage["radius_norm"],
        "radius_mm_for_this_case": stage["radius_mm"],
        "nsample": stage["nsample"],
        "assignments": int(len(stage["row"])),
        "group_size": {
            "min": int(sizes.min()), "median": float(np.median(sizes)),
            "mean": float(sizes.mean()), "max": int(sizes.max()),
        },
        "source_coverage_fraction": float(np.mean(memberships > 0)),
        "membership_overlap": {
            "mean": float(memberships.mean()), "max": int(memberships.max()),
        },
        "max_assignment_distance_norm": float(stage["distance_norm"].max()),
        "trace_sha256": stage["trace_sha256"],
    }


def export_sa_visualization(config_path: Path, case_label: str, output_dir: Path) -> Path:
    cfg = C.ExpConfig.from_json(config_path)
    if cfg.model.name != "pointnetpp":
        raise ValueError("SA visualization requires model.name='pointnetpp'")
    if not (len(cfg.model.sa_ratios) == len(cfg.model.sa_radius) ==
            len(cfg.model.sa_nsample) == 3):
        raise ValueError("foundation visualization requires exactly three SA stages")
    if cfg.data.sampling != "fps" or cfg.data.wall_n_points != 2000:
        raise ValueError("foundation visualization is frozen to FPS-2000 input")

    stats = D.load_wss_stats(cfg.data.wss_stats_path)
    cases = D.load_partition(
        cfg.data.split_path, "train", stats, strict=True, target=cfg.data.target,
        target_normalization=cfg.data.target_normalization,
    )
    matches = [
        (i, case) for i, case in enumerate(cases)
        if f"{case['cohort'].removeprefix('AG/')}/{case['case']}" == case_label
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one train case matching {case_label!r}, got {len(matches)}")
    case_index, case = matches[0]
    seed = int(cfg.train.seed + 7919 * case_index)
    full_wall_idx = D.sample_indices(
        case, cfg.data, seed, epoch=0, case_index=case_index, run_seed=cfg.train.seed,
    )
    pos = torch.from_numpy(np.ascontiguousarray(case["pos"][full_wall_idx]))
    batch = torch.zeros(len(pos), dtype=torch.long)
    scale = float(case["coord_scale_scalar"])
    lineage = np.asarray(full_wall_idx, dtype=np.int64)

    stages: list[dict] = []
    for stage_no, (ratio, radius_value, nsample) in enumerate(zip(
        cfg.model.sa_ratios, cfg.model.sa_radius, cfg.model.sa_nsample
    ), start=1):
        idx_t, pos_q_t, batch_q_t, row_t, col_t = sample_and_group(
            pos, batch, ratio=float(ratio), radius_value=float(radius_value),
            nsample=int(nsample), random_start=False,
        )
        source_norm = pos.numpy()
        center_norm = pos_q_t.numpy()
        idx = idx_t.numpy().astype(np.int64)
        row = row_t.numpy().astype(np.int64)
        col = col_t.numpy().astype(np.int64)
        distance_norm = np.linalg.norm(source_norm[col] - center_norm[row], axis=1)
        group_size = np.bincount(row, minlength=len(center_norm)).astype(np.int64)
        membership_count = np.bincount(col, minlength=len(source_norm)).astype(np.int64)
        stage = {
            "stage": stage_no, "ratio": float(ratio),
            "radius_norm": float(radius_value), "radius_mm": float(radius_value) * scale,
            "nsample": int(nsample), "source_xyz_norm": source_norm,
            "center_xyz_norm": center_norm, "source_xyz_mm": source_norm * scale,
            "center_xyz_mm": center_norm * scale, "idx": idx, "row": row, "col": col,
            "distance_norm": distance_norm, "group_size": group_size,
            "membership_count": membership_count, "source_lineage": lineage.copy(),
            "center_lineage": lineage[idx],
            "trace_sha256": _digest(idx, row, col),
        }
        stages.append(stage)
        pos, batch, lineage = pos_q_t, batch_q_t, lineage[idx]

    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = Path(case["bundle_path"])
    with np.load(bundle, allow_pickle=True) as data:
        wall_raw_xyz = data["wall_coords_raw"].astype(np.float64)
        centroid = data["transform_centroid"].astype(np.float64)
        rotation = data["transform_rotation"].astype(np.float64)
    stl = C.PROJECT_ROOT / "data_new" / case["cohort"] / case["case"] / f"{case['case']}.stl"
    stl_raw_xyz, triangles = _load_stl(stl)
    stl_scale = _bbox_diag(wall_raw_xyz) / max(_bbox_diag(stl_raw_xyz), 1e-12)
    stl_xyz = (stl_raw_xyz * stl_scale - centroid) @ rotation
    _write_vtp(stl_xyz, output_dir / "vessel_surface.vtp", triangles=triangles)

    for stage in stages:
        prefix = f"sa{stage['stage']}"
        _write_vtp(
            stage["source_xyz_mm"], output_dir / f"{prefix}_source_points.vtp",
            scalars={
                "full_wall_index": stage["source_lineage"],
                "group_membership_count": stage["membership_count"],
                "is_grouped": (stage["membership_count"] > 0).astype(np.int8),
            },
        )
        _write_vtp(
            stage["center_xyz_mm"], output_dir / f"{prefix}_centers.vtp",
            scalars={
                "center_id": np.arange(len(stage["center_xyz_mm"]), dtype=np.int64),
                "full_wall_index": stage["center_lineage"],
                "group_size": stage["group_size"],
            },
        )
        _write_edges(
            stage["center_xyz_mm"][stage["row"]],
            stage["source_xyz_mm"][stage["col"]],
            output_dir / f"{prefix}_group_edges.vtp",
            stage["row"], stage["source_lineage"][stage["col"]], stage["distance_norm"],
        )
        with (output_dir / f"{prefix}_assignments.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "stage", "center_id", "center_full_wall_index", "source_index",
                "source_full_wall_index", "distance_norm", "group_size",
            ])
            for center_id, source_id, distance in zip(
                stage["row"], stage["col"], stage["distance_norm"]
            ):
                writer.writerow([
                    stage["stage"], int(center_id),
                    int(stage["center_lineage"][center_id]), int(source_id),
                    int(stage["source_lineage"][source_id]), float(distance),
                    int(stage["group_size"][center_id]),
                ])

    _plot_overview(stages, output_dir / "01_SA三层中心点与group覆盖总览.png")
    _plot_representative_groups(stages, output_dir / "02_SA三层代表性group局部图.png")
    manifest = {
        "purpose": "PointNet++ three-stage SA visualization foundation; no training",
        "config": str(config_path), "case": case_label, "partition": "train",
        "input_sampling": "deterministic project FPS-2000",
        "sa_fps_mode": "eval-mode deterministic FPS (random_start=False)",
        "seed": int(cfg.train.seed), "case_index_in_train_split": case_index,
        "source_bundle": str(bundle), "source_stl": str(stl),
        "grouping_coordinate_system": "per-case normalized model coordinates",
        "render_coordinate_system": "registered rigid frame in millimetres",
        "stages": [_stage_summary(stage) for stage in stages],
    }
    (output_dir / "manifest_sa3.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "README_ParaView.md").write_text(
        "# PointNet++ 三层 SA 中心点与分组\n\n"
        f"- 病例：`{case_label}`；输入为固定 FPS-2000。\n"
        "- SA1/SA2/SA3 中心数：500 / 125 / 32。\n"
        "- 分组在模型归一化坐标中计算；VTP 按病例 `coord_scale` 还原到配准毫米坐标。\n"
        "- `sa*_source_points.vtp` 按 `group_membership_count` 着色；"
        "`sa*_centers.vtp` 以红色大点显示；`sa*_group_edges.vtp` 显示真实中心—成员连线。\n"
        "- PNG 总览使用全部 assignment；代表性 group 图只选择 6 个空间分散中心以避免遮挡。\n"
        "- 该配置仅用于进入 PointNet++ fine-tune 前的结构审计，本轮未训练 PointNet++。\n",
        encoding="utf-8",
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="导出 PointNet++ 三层 SA 中心点与 ball-query 分组")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--case", default=DEFAULT_CASE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(export_sa_visualization(args.config, args.case, args.output_dir))


if __name__ == "__main__":
    main()
