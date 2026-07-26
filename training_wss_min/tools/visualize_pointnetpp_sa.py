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


# The trained PNPP-SA3-AG+AAA-stratified experiment (Slurm job 9169).
DEFAULT_CONFIG = C.PROJECT_ROOT / "training_wss_min/configs/pointnetpp_v4/ag_aaa_v4_stratified_sa3_e2_global_fps2000.json"
DEFAULT_CASE = "fast/RAN_QING_BO"
DEFAULT_OUTPUT = C.PROJECT_ROOT / "例子/06_PointNet++_SA三层采样与分组/PNPP-SA3-AG+AAA-stratified_fast_RAN_QING_BO_ParaView"


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


def _representative_center_ids(stage: dict, n_groups: int = 6) -> np.ndarray:
    """Select the same spatially dispersed groups in every representative plot."""
    return D.farthest_point_sample(
        stage["center_xyz_norm"], min(n_groups, len(stage["center_xyz_norm"])),
        seed=700 + stage["stage"],
    )


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
        pick = _representative_center_ids(stage, n_groups)
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
            label = chr(ord("A") + panel)
            ax.set_title(f"{label}: SA{stage['stage']} group {center_id}\nn={mask.sum()}")
            if panel == 0:
                ax.set_ylabel("Z (mm)")
            ax.set_xlabel("X (mm)")
    fig.suptitle("Representative exact ball-query groups (star=center, dashed=radius)",
                 fontsize=16)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def _plot_global_local_correspondence(stages: list[dict], output: Path) -> None:
    """Show exactly where the representative local groups sit on the full vessel."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    n_groups = 6
    colors = plt.get_cmap("tab10").colors
    labels = tuple(chr(ord("A") + i) for i in range(n_groups))
    fig = plt.figure(figsize=(23.0, 10.8), constrained_layout=True)
    grid = fig.add_gridspec(len(stages), n_groups + 1, width_ratios=[1.6] + [1.0] * n_groups)
    full_geometry = stages[0]["source_xyz_mm"]

    for stage_row, stage in enumerate(stages):
        pick = _representative_center_ids(stage, n_groups)
        global_ax = fig.add_subplot(grid[stage_row, 0], projection="3d")
        global_ax.scatter(
            full_geometry[:, 0], full_geometry[:, 1], full_geometry[:, 2],
            c="#C7C7C7", s=2.5, alpha=0.32, linewidths=0,
        )
        for panel, (label, center_id) in enumerate(zip(labels, pick)):
            mask = stage["row"] == center_id
            members = stage["source_xyz_mm"][stage["col"][mask]]
            center = stage["center_xyz_mm"][center_id]
            color = colors[panel % len(colors)]
            global_ax.scatter(members[:, 0], members[:, 1], members[:, 2],
                              c=[color], s=17, alpha=0.95, depthshade=False)
            global_ax.scatter([center[0]], [center[1]], [center[2]], c="#E31A1C",
                              marker="*", s=105, edgecolors="black", linewidths=0.45,
                              depthshade=False)
            global_ax.text(center[0], center[1], center[2], f" {label}", fontsize=9,
                           weight="bold", color="black")
        global_ax.view_init(elev=18, azim=-62)
        global_ax.set_box_aspect(np.ptp(full_geometry, axis=0))
        global_ax.set_axis_off()
        global_ax.set_title(
            f"SA{stage['stage']} global location\ngrey=full FPS-2000; A–F=selected groups",
            fontsize=10,
        )

        for panel, (label, center_id) in enumerate(zip(labels, pick)):
            ax = fig.add_subplot(grid[stage_row, panel + 1])
            mask = stage["row"] == center_id
            members = stage["source_xyz_mm"][stage["col"][mask]]
            center = stage["center_xyz_mm"][center_id]
            color = colors[panel % len(colors)]
            ax.scatter(members[:, 0], members[:, 2], s=32, c=[color], alpha=0.88,
                       edgecolors="white", linewidths=0.3)
            ax.scatter(center[0], center[2], s=100, c="#E31A1C", marker="*",
                       edgecolors="black", linewidths=0.5, zorder=3)
            ax.add_patch(Circle((center[0], center[2]), stage["radius_mm"], fill=False,
                                linestyle="--", linewidth=0.9, color="#555555"))
            pad = stage["radius_mm"] * 1.15
            ax.set_xlim(center[0] - pad, center[0] + pad)
            ax.set_ylim(center[2] - pad, center[2] + pad)
            ax.set_aspect("equal")
            ax.set_title(f"{label}: group {center_id}\nn={mask.sum()}", fontsize=10)
            ax.set_xlabel("X (mm)", fontsize=8)
            if panel == 0:
                ax.set_ylabel("Z (mm)", fontsize=8)
            ax.tick_params(labelsize=7)

    fig.suptitle(
        "Representative PointNet++ SA groups: global anatomical location → local ball-query members\n"
        "Same A–F label/color links each local group to its location on the full vessel; star=center, dashed=radius",
        fontsize=15,
    )
    fig.savefig(output, dpi=200)
    plt.close(fig)


def _load_centerline_branch_proxy(centerline_path: Path, centroid: np.ndarray,
                                  rotation: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load centerline segments and branch IDs for a geometric grouping audit.

    ``BranchId`` is used only as a nearest-centreline QA proxy: it identifies
    whether a Euclidean ball-query contains points associated with different
    centreline branches.  It is not a learned model feature or a training label.
    """
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(centerline_path))
    reader.Update()
    poly = reader.GetOutput()
    branch_array = poly.GetPointData().GetArray("BranchId")
    if branch_array is None:
        raise ValueError(f"centerline has no BranchId array: {centerline_path}")
    points_raw = vtk_to_numpy(poly.GetPoints().GetData()).astype(np.float64)
    points_mm = (points_raw - centroid) @ rotation
    branch_ids = vtk_to_numpy(branch_array).astype(np.int64)

    segments: list[tuple[int, int]] = []
    lines = poly.GetLines()
    lines.InitTraversal()
    ids = vtk.vtkIdList()
    while lines.GetNextCell(ids):
        segments.extend((ids.GetId(i), ids.GetId(i + 1)) for i in range(ids.GetNumberOfIds() - 1))
    return points_mm, branch_ids, np.asarray(segments, dtype=np.int64)


def _select_grouping_audit_examples(stage: dict) -> list[dict]:
    """Pick two branch-pure groups and the strongest topology-shortcut candidate."""
    from scipy.sparse.csgraph import dijkstra

    source_branch = stage["source_branch_id"]
    rows, cols = stage["row"], stage["col"]
    summaries: list[dict] = []
    for center_id in range(len(stage["center_xyz_mm"])):
        member_idx = cols[rows == center_id]
        branch_ids, counts = np.unique(source_branch[member_idx], return_counts=True)
        euclidean = np.linalg.norm(
            stage["source_xyz_mm"][member_idx] - stage["center_xyz_mm"][center_id], axis=1
        )
        geodesic = dijkstra(
            stage["centerline_graph"], indices=stage["center_centerline_id"][center_id]
        )[stage["source_centerline_id"][member_idx]]
        topology_ratio = geodesic / np.maximum(euclidean, 1e-3)
        topology_ratio[euclidean <= 1e-3] = 1.0
        summaries.append({
            "center_id": center_id,
            "member_idx": member_idx,
            "branch_ids": branch_ids,
            "counts": counts,
            "purity": float(counts.max() / counts.sum()),
            "topology_ratio": topology_ratio,
            "max_topology_ratio": float(topology_ratio.max()),
        })
    risk = max(summaries, key=lambda item: item["max_topology_ratio"])

    selected: list[dict] = []
    risk_center = stage["center_xyz_mm"][risk["center_id"]]
    for branch_id in np.unique(source_branch):
        pure = [
            item for item in summaries
            if len(item["branch_ids"]) == 1 and int(item["branch_ids"][0]) == int(branch_id)
            and len(item["member_idx"]) >= max(8, stage["nsample"] - 2)
        ]
        if pure:
            # A separated, well-populated example makes the comparison readable.
            chosen = max(
                pure,
                key=lambda item: np.linalg.norm(
                    stage["center_xyz_mm"][item["center_id"]] - risk_center
                ),
            )
            selected.append({**chosen, "kind": "pure", "branch_id": int(branch_id)})
        if len(selected) == 2:
            break
    if len(selected) < 2:
        raise RuntimeError(f"SA{stage['stage']} lacks two branch-pure audit examples")
    selected.append({**risk, "kind": "risk", "branch_id": None})
    return selected


def _plot_global_grouping_audit(stages: list[dict], output: Path,
                                centerline_mm: np.ndarray, centerline_segments: np.ndarray) -> None:
    """Render the teacher-requested whole-vessel view and expose cross-branch risk."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle

    full_geometry = stages[0]["source_xyz_mm"]
    fig, axes = plt.subplots(1, len(stages), figsize=(17.4, 6.8))
    for ax, stage in zip(np.atleast_1d(axes), stages):
        ax.scatter(full_geometry[:, 0], full_geometry[:, 2], c="#D2D2D2", s=4, alpha=0.42,
                   linewidths=0, zorder=1)
        for start, end in centerline_segments:
            segment = centerline_mm[[start, end]]
            ax.plot(segment[:, 0], segment[:, 2], c="#4A4A4A", lw=0.45, alpha=0.58, zorder=2)

        examples = _select_grouping_audit_examples(stage)
        safe_colors = ("#009E73", "#0072B2")
        for example, safe_color in zip(examples[:2], safe_colors):
            center = stage["center_xyz_mm"][example["center_id"]]
            members = stage["source_xyz_mm"][example["member_idx"]]
            ax.scatter(members[:, 0], members[:, 2], c=safe_color, s=22, alpha=0.95,
                       edgecolors="white", linewidths=0.3, zorder=4)
            ax.scatter(center[0], center[2], marker="x", s=105, c=safe_color, linewidths=2.1,
                       zorder=6)
            ax.add_patch(Circle((center[0], center[2]), stage["radius_mm"], fill=False,
                                edgecolor=safe_color, linestyle="--", linewidth=1.2, zorder=3))

        risk = examples[2]
        center = stage["center_xyz_mm"][risk["center_id"]]
        member_idx = risk["member_idx"]
        members = stage["source_xyz_mm"][member_idx]
        ax.scatter(members[:, 0], members[:, 2], c="#E69F00", s=28, alpha=0.98,
                   edgecolors="white", linewidths=0.35, zorder=5)
        ax.scatter(center[0], center[2], marker="x", s=120, c="#E69F00", linewidths=2.4, zorder=7)
        ax.add_patch(Circle((center[0], center[2]), stage["radius_mm"], fill=False,
                            edgecolor="#E69F00", linestyle="--", linewidth=1.45, zorder=3))

        ax.set_title(
            f"SA{stage['stage']}: {len(stage['source_xyz_mm'])} → "
            f"{len(stage['center_xyz_mm'])} centers",
            fontsize=10,
        )
        ax.set_aspect("equal")
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Z (mm)")

    legend = [
        Line2D([], [], marker="o", linestyle="", markerfacecolor="#D2D2D2", markeredgecolor="none",
               label="vessel"),
        Line2D([], [], marker="o", linestyle="", markerfacecolor="#009E73", markeredgecolor="white",
               label="grouped points"),
        Line2D([], [], marker="x", color="#009E73", linestyle="", markersize=10, markeredgewidth=2,
               label="center"),
        Line2D([], [], color="#009E73", linestyle="--", label="ball-query radius"),
    ]
    fig.subplots_adjust(left=0.045, right=0.99, bottom=0.15, top=0.84, wspace=0.34)
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, 0.035), ncol=4,
               fontsize=9, frameon=False, columnspacing=2.2, handletextpad=0.6)
    fig.suptitle(
        "PointNet++ SA: centers and grouped points on the full vessel",
        fontsize=14,
    )
    fig.savefig(output, dpi=220, bbox_inches="tight")
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


def _plot_all_centers_and_group_overlap(stages: list[dict], output: Path) -> None:
    """Make the teacher-facing whole-vessel view of every group point and centre.

    In each SA panel every source point which belongs to one or more ball-query
    groups is plotted exactly once and coloured by how many groups contain it.
    The overlaid crosses are all FPS centres of that SA layer.  The fourth
    panel removes coincident centre copies and colours a point by its deepest
    selected SA stage, making nested centre reuse legible.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    base = stages[0]
    full_idx = base["source_lineage"]
    row_for_full_idx = {int(full_index): row for row, full_index in enumerate(full_idx)}
    deepest = np.zeros(len(full_idx), dtype=np.int8)
    for stage in stages:
        for full_index in stage["center_lineage"]:
            deepest[row_for_full_idx[int(full_index)]] = stage["stage"]

    fig, axes = plt.subplots(1, 4, figsize=(22, 7.2), constrained_layout=True)
    center_colors = ("#1976D2", "#F57C00", "#D32F2F")
    for ax, stage, color in zip(axes[:3], stages, center_colors):
        source = stage["source_xyz_mm"]
        centers = stage["center_xyz_mm"]
        sc = ax.scatter(
            source[:, 0], source[:, 2], c=stage["membership_count"], cmap="viridis",
            vmin=0, vmax=max(1, int(stage["membership_count"].max())),
            s=10, alpha=0.78, linewidths=0, zorder=1,
        )
        ax.scatter(
            centers[:, 0], centers[:, 2], c=color, marker="x", s=38,
            linewidths=1.25, alpha=0.96, zorder=3,
        )
        colorbar = fig.colorbar(sc, ax=ax, pad=0.015, shrink=0.7)
        colorbar.set_label("number of groups containing this point", fontsize=8)
        colorbar.ax.tick_params(labelsize=7)
        ax.set_title(
            f"SA{stage['stage']}: all group points + all centres\n"
            f"{len(source)} source points → {len(centers)} centres; "
            f"mean overlap={stage['membership_count'].mean():.2f}, "
            f"max={stage['membership_count'].max()}",
            fontsize=10,
        )
        ax.set_aspect("equal")
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Z (mm)")

    ax = axes[3]
    full = base["source_xyz_mm"]
    ax.scatter(full[:, 0], full[:, 2], c="#D0D0D0", s=8, alpha=0.52, linewidths=0, zorder=1)
    for stage, color, label, marker, size in (
        (1, "#1976D2", "SA1 only (375)", "o", 17),
        (2, "#F57C00", "SA1+SA2 only (93)", "s", 26),
        (3, "#D32F2F", "SA1+SA2+SA3 (32)", "*", 62),
    ):
        points = full[deepest == stage]
        ax.scatter(
            points[:, 0], points[:, 2], c=color, marker=marker, s=size,
            edgecolors="white", linewidths=0.28, alpha=0.96, label=label, zorder=stage + 1,
        )
    ax.set_title(
        "All three centre layers, with coincident copies merged\n"
        "colour/marker = deepest selected SA stage",
        fontsize=10,
    )
    ax.set_aspect("equal")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Z (mm)")
    ax.legend(loc="lower center", fontsize=8, frameon=True)

    fig.suptitle(
        "PointNet++ SA3: complete group coverage and nested FPS centre selection (fast/RAN_QING_BO)",
        fontsize=15,
    )
    fig.savefig(output, dpi=240)
    plt.close(fig)


def _write_paraview_multiblock(output_dir: Path) -> None:
    """Give ParaView a one-file entry point for the seven useful VTP layers."""
    datasets = (
        ("vessel surface", "vessel_surface.vtp"),
        ("SA1 group points (FPS-2000)", "sa1_source_points.vtp"),
        ("SA1 centres (500)", "sa1_centers.vtp"),
        ("SA2 group points (SA1 centres)", "sa2_source_points.vtp"),
        ("SA2 centres (125)", "sa2_centers.vtp"),
        ("SA3 group points (SA2 centres)", "sa3_source_points.vtp"),
        ("SA3 centres (32)", "sa3_centers.vtp"),
    )
    nodes = "\n".join(
        f'    <DataSet index="{index}" name="{name}" file="{filename}"/>'
        for index, (name, filename) in enumerate(datasets)
    )
    (output_dir / "SA3_全部中心与分组点.vtm").write_text(
        "<?xml version=\"1.0\"?>\n"
        "<VTKFile type=\"vtkMultiBlockDataSet\" version=\"1.0\" byte_order=\"LittleEndian\">\n"
        "  <vtkMultiBlockDataSet>\n"
        f"{nodes}\n"
        "  </vtkMultiBlockDataSet>\n"
        "</VTKFile>\n",
        encoding="utf-8",
    )


def _write_center_mark_csv(stages: list[dict], output_dir: Path) -> dict:
    """Write compact ParaView-friendly centre labels for the nested SA trace.

    ``sa_centers_marked.csv`` has precisely one row for each FPS-2000 input
    point.  A point selected at multiple stages is deliberately represented
    once, with its deepest selected stage (1/2/3).  Thus a categorical colour
    map reveals nesting without coincident glyphs hiding one another.
    """
    base = stages[0]
    full_indices = base["source_lineage"]
    full_xyz_mm = base["source_xyz_mm"]
    n_points = len(full_indices)
    index_to_row = {int(full_index): row for row, full_index in enumerate(full_indices)}
    stage_flags = np.zeros((n_points, 3), dtype=np.int8)
    stage_ids = np.full((n_points, 3), -1, dtype=np.int64)

    for stage in stages:
        stage_column = stage["stage"] - 1
        for center_id, full_index in enumerate(stage["center_lineage"]):
            row = index_to_row[int(full_index)]
            stage_flags[row, stage_column] = 1
            stage_ids[row, stage_column] = center_id

    deepest_stage = np.where(
        stage_flags[:, 2] == 1, 3,
        np.where(stage_flags[:, 1] == 1, 2, np.where(stage_flags[:, 0] == 1, 1, 0)),
    ).astype(np.int8)
    labels = ("input_only", "SA1_only", "SA2_only", "SA3")
    glyph_radius_mm = np.choose(deepest_stage, [0.25, 0.75, 1.20, 1.80])

    with (output_dir / "sa_centers_marked.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "x_mm", "y_mm", "z_mm", "full_wall_index", "deepest_sa_stage",
            "display_label", "is_sa1_center", "is_sa2_center", "is_sa3_center",
            "sa1_center_id", "sa2_center_id", "sa3_center_id", "glyph_radius_mm",
        ])
        for row, xyz in enumerate(full_xyz_mm):
            stage = int(deepest_stage[row])
            writer.writerow([
                *map(float, xyz), int(full_indices[row]), stage, labels[stage],
                *map(int, stage_flags[row]), *map(int, stage_ids[row]),
                float(glyph_radius_mm[row]),
            ])

    # This second CSV intentionally retains the three nested records.  It is
    # useful when the teacher wants to toggle a single stage with Threshold.
    with (output_dir / "sa_centers_by_stage.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "x_mm", "y_mm", "z_mm", "sa_stage", "center_id", "full_wall_index",
            "glyph_radius_mm",
        ])
        for stage in stages:
            radius = (0.75, 1.20, 1.80)[stage["stage"] - 1]
            for center_id, (xyz, full_index) in enumerate(
                zip(stage["center_xyz_mm"], stage["center_lineage"])
            ):
                writer.writerow([
                    *map(float, xyz), stage["stage"], center_id, int(full_index), radius,
                ])

    return {
        "input_only": int(np.sum(deepest_stage == 0)),
        "SA1_only": int(np.sum(deepest_stage == 1)),
        "SA2_only": int(np.sum(deepest_stage == 2)),
        "SA3": int(np.sum(deepest_stage == 3)),
        "sa_centers_by_stage_rows": int(sum(len(stage["center_lineage"]) for stage in stages)),
    }


def export_sa_visualization(config_path: Path, case_label: str, output_dir: Path,
                            partition: str = "auto") -> Path:
    cfg = C.ExpConfig.from_json(config_path)
    if cfg.model.name != "pointnetpp":
        raise ValueError("SA visualization requires model.name='pointnetpp'")
    if not (len(cfg.model.sa_ratios) == len(cfg.model.sa_radius) ==
            len(cfg.model.sa_nsample) == 3):
        raise ValueError("SA visualization requires exactly three SA stages")
    if cfg.data.sampling != "fps" or cfg.data.wall_n_points != 2000:
        raise ValueError("SA visualization is frozen to FPS-2000 input")

    stats = D.load_wss_stats(cfg.data.wss_stats_path)
    if partition not in {"auto", "train", "test"}:
        raise ValueError("partition must be one of: auto, train, test")
    partitions = ("train", "test") if partition == "auto" else (partition,)
    matches: list[tuple[str, int, dict]] = []
    for selected_partition in partitions:
        cases = D.load_partition(
            cfg.data.split_path, selected_partition, stats, strict=True, target=cfg.data.target,
            target_normalization=cfg.data.target_normalization,
        )
        matches.extend(
            (selected_partition, i, case) for i, case in enumerate(cases)
            if f"{case['cohort'].removeprefix('AG/')}/{case['case']}" == case_label
        )
    if len(matches) != 1:
        raise ValueError(f"expected one {partition} case matching {case_label!r}, got {len(matches)}")
    selected_partition, case_index, case = matches[0]
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
    centerline = C.PROJECT_ROOT / "data_new" / case["cohort"] / case["case"] / "centerline/centerline.vtp"
    centerline_mm, centerline_branch_ids, centerline_segments = _load_centerline_branch_proxy(
        centerline, centroid, rotation
    )
    from scipy.spatial import cKDTree
    from scipy.sparse import coo_matrix
    branch_tree = cKDTree(centerline_mm)
    segment_lengths = np.linalg.norm(
        centerline_mm[centerline_segments[:, 0]] - centerline_mm[centerline_segments[:, 1]], axis=1
    )
    centerline_graph = coo_matrix(
        (
            np.concatenate((segment_lengths, segment_lengths)),
            (
                np.concatenate((centerline_segments[:, 0], centerline_segments[:, 1])),
                np.concatenate((centerline_segments[:, 1], centerline_segments[:, 0])),
            ),
        ),
        shape=(len(centerline_mm), len(centerline_mm)),
    ).tocsr()
    for stage in stages:
        _, nearest_centerline = branch_tree.query(stage["source_xyz_mm"])
        stage["source_branch_id"] = centerline_branch_ids[nearest_centerline]
        stage["source_centerline_id"] = nearest_centerline
        stage["center_centerline_id"] = branch_tree.query(stage["center_xyz_mm"])[1]
        stage["centerline_graph"] = centerline_graph
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

    center_mark_counts = _write_center_mark_csv(stages, output_dir)
    _write_paraview_multiblock(output_dir)

    _plot_overview(stages, output_dir / "01_SA三层中心点与group覆盖总览.png")
    _plot_representative_groups(stages, output_dir / "02_SA三层代表性group局部图.png")
    _plot_global_local_correspondence(stages, output_dir / "03_SA代表group全局-局部对应图.png")
    _plot_global_grouping_audit(
        stages, output_dir / "04_SA三层全局group与跨支风险审计.png",
        centerline_mm, centerline_segments,
    )
    _plot_all_centers_and_group_overlap(
        stages, output_dir / "05_SA全部中心与分组点重合总览.png",
    )
    manifest = {
        "purpose": "Exact PointNet++ three-stage SA trace from the selected experiment config; no training is run by this exporter",
        "config": str(config_path), "case": case_label, "partition": selected_partition,
        "input_sampling": "deterministic project FPS-2000",
        "sa_fps_mode": "eval-mode deterministic FPS (random_start=False)",
        "seed": int(cfg.train.seed), "case_index_in_partition": case_index,
        "source_bundle": str(bundle), "source_stl": str(stl),
        "grouping_coordinate_system": "per-case normalized model coordinates",
        "render_coordinate_system": "registered rigid frame in millimetres",
        "stages": [_stage_summary(stage) for stage in stages],
        "paraview_center_mark_csv": {
            "file": "sa_centers_marked.csv",
            "meaning": "one row per FPS-2000 input point; deepest_sa_stage is the non-overlapping visual label",
            "counts": center_mark_counts,
        },
    }
    (output_dir / "manifest_sa3.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "README_ParaView.md").write_text(
        "# PointNet++ 三层 SA 中心点与分组\n\n"
        f"- 病例：`{case_label}`（`{selected_partition}`）；输入为固定 FPS-2000。\n"
        "- SA1/SA2/SA3 中心数：500 / 125 / 32。\n"
        "- **老师查看中心点重合时请优先打开 `sa_centers_marked.csv`：它有且只有 2000 行，每个输入点只保留一行。**\n"
        "  - `deepest_sa_stage=0`：未被任何 SA 层选为中心；`1`：仅 SA1；`2`：SA1+SA2；`3`：SA1+SA2+SA3。\n"
        "  - 因而 SA2、SA3 继承上层中心所导致的坐标重合已被合并为一个标签，不会被相同位置的点遮住。\n"
        "  - ParaView：CSV Reader → **Table To Points**（X=`x_mm`、Y=`y_mm`、Z=`z_mm`）→ Color By=`deepest_sa_stage`；\n"
        "    建议用 `Glyph`（Sphere）并选 `glyph_radius_mm` 作 Scale Array，或用点渲染。\n"
        "- `sa_centers_by_stage.csv` 保留 500+125+32=657 条逐层记录；在 ParaView 对 `sa_stage` 用 Threshold，\n"
        "  可单独开关 SA1、SA2、SA3。\n"
        "- **不想处理 CSV 时，直接打开 `SA3_全部中心与分组点.vtm`，点击 Apply 后按 `R`（Reset Camera）。**\n"
        "  它一次载入血管、每层全部 group 点和每层全部 center；在 Pipeline Browser 展开后可单独点亮/关闭各层。\n"
        "  建议显示方式：group point 用 Point Gaussian（Opacity≈0.25，Point Size≈3）；center 用 Point Gaussian\n"
        "  （Point Size：SA1≈6、SA2≈10、SA3≈16；颜色依次蓝/橙/红）。\n"
        "- `05_SA全部中心与分组点重合总览.png` 是可直接给老师看的静态总览：前三栏按 group membership count\n"
        "  显示同一点被多少 ball-query group 覆盖，最后一栏把三层同坐标中心合并后显示最深层标签。\n"
        "- 分组在模型归一化坐标中计算；VTP 按病例 `coord_scale` 还原到配准毫米坐标。\n"
        "- `sa*_source_points.vtp` 按 `group_membership_count` 着色；"
        "`sa*_centers.vtp` 以红色大点显示；`sa*_group_edges.vtp` 显示真实中心—成员连线。\n"
        "- PNG 总览使用全部 assignment；代表性 group 图只选择 6 个空间分散中心以避免遮挡。\n"
        "- `03_SA代表group全局-局部对应图.png` 用相同 A–F 标签/颜色连接全局解剖位置和局部成员；"
        "左侧是完整血管的三维视图，右侧为 X–Z 局部放大。\n"
        "- `04_SA三层全局group与跨支风险审计.png` 在完整血管上画出每层两个 branch-pure group 和"
        "一个中心线测地距离/直线距离比最高的候选 group；供人工判断是否发生跨支捷径。\n"
        "- 本工具只复算并导出已选实验配置的采样/分组轨迹，不会训练或改动 PNPP-SA3 实验结果。\n",
        encoding="utf-8",
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="导出 PointNet++ 三层 SA 中心点与 ball-query 分组")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--case", default=DEFAULT_CASE)
    parser.add_argument("--partition", choices=("auto", "train", "test"), default="auto",
                        help="病例所在划分；auto 会在 train/test 中查找")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(export_sa_visualization(args.config, args.case, args.output_dir, args.partition))


if __name__ == "__main__":
    main()
