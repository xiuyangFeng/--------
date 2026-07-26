#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build ParaView convenience assets from an already exported SA trace.

This utility deliberately reads the small VTP files already produced by
``visualize_pointnetpp_sa``.  It can therefore regenerate the teacher-facing
overview without reloading a dataset or running PointNet++ sampling.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


DEFAULT_OUTPUT = Path(
    "例子/06_PointNet++_SA三层采样与分组/"
    "PNPP-SA3-AG+AAA-stratified_fast_RAN_QING_BO_ParaView"
)


def _read_vtp(path: Path, scalar_names: tuple[str, ...]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    poly = reader.GetOutput()
    if poly.GetNumberOfPoints() == 0:
        raise RuntimeError(f"no points in {path}")
    points = vtk_to_numpy(poly.GetPoints().GetData()).astype(np.float64)
    values: dict[str, np.ndarray] = {}
    for scalar_name in scalar_names:
        scalar = poly.GetPointData().GetArray(scalar_name)
        if scalar is None:
            raise RuntimeError(f"missing {scalar_name!r} in {path}")
        values[scalar_name] = vtk_to_numpy(scalar)
    return points, values


def _write_multiblock(output_dir: Path) -> Path:
    datasets = (
        ("vessel surface", "vessel_surface.vtp"),
        ("SA1 group points (FPS-2000)", "sa1_source_points.vtp"),
        ("SA1 centres (500)", "sa1_centers.vtp"),
        ("SA2 group points (SA1 centres)", "sa2_source_points.vtp"),
        ("SA2 centres (125)", "sa2_centers.vtp"),
        ("SA3 group points (SA2 centres)", "sa3_source_points.vtp"),
        ("SA3 centres (32)", "sa3_centers.vtp"),
    )
    lines = "\n".join(
        f'    <DataSet index="{i}" name="{name}" file="{filename}"/>'
        for i, (name, filename) in enumerate(datasets)
    )
    output = output_dir / "SA3_全部中心与分组点.vtm"
    output.write_text(
        "<?xml version=\"1.0\"?>\n"
        "<VTKFile type=\"vtkMultiBlockDataSet\" version=\"1.0\" byte_order=\"LittleEndian\">\n"
        "  <vtkMultiBlockDataSet>\n"
        f"{lines}\n"
        "  </vtkMultiBlockDataSet>\n"
        "</VTKFile>\n",
        encoding="utf-8",
    )
    return output


def _write_pointcloud_only_multiblock(output_dir: Path) -> Path:
    """One ParaView entry point containing only points, never the vessel surface."""
    datasets = (
        ("SA1 all group points (2000)", "sa1_source_points.vtp"),
        ("SA1 centre ∩ group (500 centres)", "sa1_centers.vtp"),
        ("SA2 all group points (500)", "sa2_source_points.vtp"),
        ("SA2 centre ∩ group (125 centres)", "sa2_centers.vtp"),
        ("SA3 all group points (125)", "sa3_source_points.vtp"),
        ("SA3 centre ∩ group (32 centres)", "sa3_centers.vtp"),
    )
    lines = "\n".join(
        f'    <DataSet index="{i}" name="{name}" file="{filename}"/>'
        for i, (name, filename) in enumerate(datasets)
    )
    output = output_dir / "SA3_纯点云_中心与group重叠.vtm"
    output.write_text(
        "<?xml version=\"1.0\"?>\n"
        "<VTKFile type=\"vtkMultiBlockDataSet\" version=\"1.0\" byte_order=\"LittleEndian\">\n"
        "  <vtkMultiBlockDataSet>\n"
        f"{lines}\n"
        "  </vtkMultiBlockDataSet>\n"
        "</VTKFile>\n",
        encoding="utf-8",
    )
    return output


def _load_csv_trace(output_dir: Path) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Return full points, deepest labels, source masks, centre masks and memberships."""
    with (output_dir / "sa_centers_marked.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    full_index = np.asarray([int(row["full_wall_index"]) for row in rows], dtype=np.int64)
    full = np.asarray(
        [[float(row["x_mm"]), float(row["y_mm"]), float(row["z_mm"])] for row in rows],
        dtype=np.float64,
    )
    deepest = np.asarray([int(row["deepest_sa_stage"]) for row in rows], dtype=np.int8)
    centre_masks = [
        np.asarray([int(row[f"is_sa{stage}_center"]) == 1 for row in rows], dtype=bool)
        for stage in (1, 2, 3)
    ]
    source_masks = [np.ones(len(rows), dtype=bool), centre_masks[0], centre_masks[1]]
    memberships: list[np.ndarray] = []
    for stage in (1, 2, 3):
        counts: dict[int, int] = {}
        with (output_dir / f"sa{stage}_assignments.csv").open(newline="", encoding="utf-8") as f:
            for assignment in csv.DictReader(f):
                source_index = int(assignment["source_full_wall_index"])
                counts[source_index] = counts.get(source_index, 0) + 1
        memberships.append(np.asarray([counts.get(int(value), 0) for value in full_index], dtype=np.int64))
    return full, deepest, source_masks, centre_masks, memberships


def _write_pointcloud_only_view_and_report(output_dir: Path) -> tuple[Path, Path]:
    """Write the requested pure-point visual and a table of real overlap counts."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    full, deepest, source_masks, centre_masks, memberships = _load_csv_trace(output_dir)
    fig, axes = plt.subplots(1, 4, figsize=(19, 6.6), constrained_layout=True)
    group_colour, overlap_colours = "#55BFC0", ("#1976D2", "#F57C00", "#D32F2F")
    report_rows: list[dict[str, int | float | str]] = []

    for stage, (ax, source_mask, centre_mask, membership, colour) in enumerate(
        zip(axes[:3], source_masks, centre_masks, memberships, overlap_colours), start=1
    ):
        grouped = source_mask & (membership > 0)
        overlap = grouped & centre_mask
        group_only = grouped & ~centre_mask
        multi_group = source_mask & (membership > 1)
        ax.scatter(full[group_only, 0], full[group_only, 2], c=group_colour, s=11,
                   alpha=0.72, linewidths=0, label=f"group-only ({group_only.sum()})")
        ax.scatter(full[overlap, 0], full[overlap, 2], c=colour, marker="o", s=24,
                   alpha=0.96, edgecolors="white", linewidths=0.30,
                   label=f"center ∩ group ({overlap.sum()})", zorder=2)
        ax.set_title(
            f"SA{stage}: pure point cloud\n"
            f"group={grouped.sum()}, center∩group={overlap.sum()}", fontsize=11,
        )
        ax.set_aspect("equal")
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Z (mm)")
        ax.legend(loc="lower center", fontsize=8, frameon=True)
        report_rows.append({
            "view": f"SA{stage}",
            "source_points": int(source_mask.sum()),
            "unique_group_points": int(grouped.sum()),
            "centers": int(centre_mask.sum()),
            "center_group_overlap": int(overlap.sum()),
            "group_only": int(group_only.sum()),
            "multi_group_points": int(multi_group.sum()),
            "assignments": int(membership[source_mask].sum()),
            "mean_groups_per_source_point": float(membership[source_mask].mean()),
            "max_groups_per_source_point": int(membership[source_mask].max()),
        })

    ax = axes[3]
    group_union = memberships[0] > 0
    centre_union = np.logical_or.reduce(centre_masks)
    group_only = group_union & ~centre_union
    ax.scatter(full[group_only, 0], full[group_only, 2], c=group_colour, s=11,
               alpha=0.72, linewidths=0, label=f"group-only ({group_only.sum()})")
    for stage, colour, marker, size, label in (
        (1, "#1976D2", "o", 22, "SA1 only"),
        (2, "#F57C00", "s", 28, "SA1+SA2"),
        (3, "#D32F2F", "*", 68, "SA1+SA2+SA3"),
    ):
        points = full[deepest == stage]
        ax.scatter(points[:, 0], points[:, 2], c=colour, marker=marker, s=size,
                   alpha=0.97, edgecolors="white", linewidths=0.30,
                   label=f"{label} ({len(points)})", zorder=stage + 1)
    ax.set_title("All SA layers: unique point-cloud labels\nall 500 centres are group points", fontsize=11)
    ax.set_aspect("equal")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Z (mm)")
    ax.legend(loc="lower center", fontsize=8, frameon=True)
    fig.suptitle(
        "PointNet++ SA3: all group points and centre∩group overlap (no vessel surface)", fontsize=15,
    )
    figure_path = output_dir / "06_纯点云_全部center与group重叠.png"
    fig.savefig(figure_path, dpi=240)
    plt.close(fig)

    report_rows.append({
        "view": "all_SA_layers_unique_union",
        "source_points": int(len(full)),
        "unique_group_points": int(group_union.sum()),
        "centers": int(centre_union.sum()),
        "center_group_overlap": int((group_union & centre_union).sum()),
        "group_only": int(group_only.sum()),
        "multi_group_points": int(np.sum(memberships[0] > 1)),
        "assignments": int(sum(membership.sum() for membership in memberships)),
        "mean_groups_per_source_point": float("nan"),
        "max_groups_per_source_point": int(max(membership.max() for membership in memberships)),
    })
    report_path = output_dir / "center_group_overlap_report.csv"
    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(report_rows[0]))
        writer.writeheader()
        writer.writerows(report_rows)
    return figure_path, report_path


def _write_adjacent_group_overlap_example(output_dir: Path) -> tuple[Path, Path]:
    """Select and draw one strongly overlapping nearby group pair per SA stage."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    full, _, _, _, _ = _load_csv_trace(output_dir)
    with (output_dir / "sa_centers_marked.csv").open(newline="", encoding="utf-8") as f:
        marked_rows = list(csv.DictReader(f))
    point_xyz = {
        int(row["full_wall_index"]): full[row_index]
        for row_index, row in enumerate(marked_rows)
    }
    with (output_dir / "manifest_sa3.json").open(encoding="utf-8") as f:
        radii = {int(item["stage"]): float(item["radius_mm_for_this_case"])
                 for item in json.load(f)["stages"]}

    centres_by_stage: dict[int, dict[int, np.ndarray]] = defaultdict(dict)
    with (output_dir / "sa_centers_by_stage.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            stage, center_id = int(row["sa_stage"]), int(row["center_id"])
            centres_by_stage[stage][center_id] = np.asarray(
                [float(row["x_mm"]), float(row["y_mm"]), float(row["z_mm"])], dtype=np.float64
            )

    fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.6), constrained_layout=True)
    report_rows: list[dict[str, int | float]] = []
    for stage, ax in enumerate(axes, start=1):
        members: dict[int, set[int]] = defaultdict(set)
        point_groups: dict[int, list[int]] = defaultdict(list)
        with (output_dir / f"sa{stage}_assignments.csv").open(newline="", encoding="utf-8") as f:
            for assignment in csv.DictReader(f):
                center_id = int(assignment["center_id"])
                point_id = int(assignment["source_full_wall_index"])
                members[center_id].add(point_id)
                point_groups[point_id].append(center_id)

        shared_by_pair: Counter[tuple[int, int]] = Counter()
        for group_ids in point_groups.values():
            ids = sorted(set(group_ids))
            for index, center_a in enumerate(ids):
                for center_b in ids[index + 1:]:
                    shared_by_pair[center_a, center_b] += 1
        centres = centres_by_stage[stage]
        pair = max(
            shared_by_pair,
            key=lambda item: (
                shared_by_pair[item] / len(members[item[0]] | members[item[1]]),
                shared_by_pair[item],
                -np.linalg.norm(centres[item[0]] - centres[item[1]]),
            ),
        )
        center_a, center_b = pair
        points_a, points_b = members[center_a], members[center_b]
        shared = points_a & points_b
        only_a, only_b = points_a - points_b, points_b - points_a
        xyz = lambda ids: np.asarray(
            [point_xyz[index] for index in sorted(ids)], dtype=np.float64
        ).reshape((-1, 3))
        xyz_a, xyz_b, xyz_shared = xyz(only_a), xyz(only_b), xyz(shared)
        # Same whole-vessel context as figure 04: grey points are the complete
        # FPS-2000 wall input, colour is reserved for the selected adjacent pair.
        ax.scatter(full[:, 0], full[:, 2], c="#D2D2D2", s=8, alpha=0.42,
                   linewidths=0, label="full wall point cloud", zorder=1)
        if len(xyz_a):
            ax.scatter(xyz_a[:, 0], xyz_a[:, 2], c="#1976D2", s=56, alpha=0.95,
                       edgecolors="white", linewidths=0.35, label=f"A only ({len(only_a)})")
        if len(xyz_b):
            ax.scatter(xyz_b[:, 0], xyz_b[:, 2], c="#F57C00", s=56, alpha=0.95,
                       edgecolors="white", linewidths=0.35, label=f"B only ({len(only_b)})")
        ax.scatter(xyz_shared[:, 0], xyz_shared[:, 2], c="#8E24AA", s=68, alpha=0.98,
                   edgecolors="white", linewidths=0.4, label=f"A ∩ B ({len(shared)})", zorder=3)
        ca, cb = centres[center_a], centres[center_b]
        ax.scatter([ca[0]], [ca[2]], c="#0D47A1", marker="*", s=220, edgecolors="black",
                   linewidths=0.55, label=f"centre A ({center_a})", zorder=5)
        ax.scatter([cb[0]], [cb[2]], c="#E65100", marker="*", s=220, edgecolors="black",
                   linewidths=0.55, label=f"centre B ({center_b})", zorder=5)
        ax.add_patch(Circle((ca[0], ca[2]), radii[stage], fill=False, linestyle="--",
                            linewidth=1.25, edgecolor="#0D47A1", zorder=2))
        ax.add_patch(Circle((cb[0], cb[2]), radii[stage], fill=False, linestyle="--",
                            linewidth=1.25, edgecolor="#E65100", zorder=2))
        pad = 5.0
        ax.set_xlim(full[:, 0].min() - pad, full[:, 0].max() + pad)
        ax.set_ylim(full[:, 2].min() - pad, full[:, 2].max() + pad)
        ax.set_aspect("equal")
        center_distance = float(np.linalg.norm(ca - cb))
        jaccard = len(shared) / len(points_a | points_b)
        ax.set_title(
            f"SA{stage}: nearby group pair\n"
            f"shared={len(shared)}/{len(points_a | points_b)}; Jaccard={jaccard:.3f}; "
            f"centre distance={center_distance:.1f} mm",
            fontsize=10,
        )
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Z (mm)")
        ax.legend(loc="upper right", fontsize=7.2, frameon=True)
        report_rows.append({
            "stage": stage,
            "center_a_id": center_a,
            "center_b_id": center_b,
            "center_distance_mm": center_distance,
            "ball_query_radius_mm": radii[stage],
            "group_a_points": len(points_a),
            "group_b_points": len(points_b),
            "shared_points": len(shared),
            "a_only_points": len(only_a),
            "b_only_points": len(only_b),
            "jaccard_overlap": jaccard,
            "shared_fraction_of_a": len(shared) / len(points_a),
            "shared_fraction_of_b": len(shared) / len(points_b),
        })

    fig.suptitle(
        "PointNet++ SA3: nearby group-pair overlap on the full wall point cloud "
        "(purple = exact shared members; dashed circles = projected ball-query radius)",
        fontsize=14,
    )
    figure_path = output_dir / "07_SA相邻group重叠示例.png"
    fig.savefig(figure_path, dpi=240)
    plt.close(fig)
    report_path = output_dir / "adjacent_group_overlap_report.csv"
    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(report_rows[0]))
        writer.writeheader()
        writer.writerows(report_rows)
    return figure_path, report_path


def _write_overview(output_dir: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sources: list[np.ndarray] = []
    memberships: list[np.ndarray] = []
    centre_indices: list[np.ndarray] = []
    for stage in (1, 2, 3):
        source, source_data = _read_vtp(
            output_dir / f"sa{stage}_source_points.vtp",
            ("full_wall_index", "group_membership_count"),
        )
        _, centre_data = _read_vtp(
            output_dir / f"sa{stage}_centers.vtp", ("full_wall_index",)
        )
        sources.append(source)
        memberships.append(source_data["group_membership_count"].astype(np.int64))
        centre_indices.append(centre_data["full_wall_index"].astype(np.int64))

    full = sources[0]
    full_index = _read_vtp(
        output_dir / "sa1_source_points.vtp", ("full_wall_index",)
    )[1]["full_wall_index"].astype(np.int64)
    index_to_row = {int(value): row for row, value in enumerate(full_index)}
    deepest = np.zeros(len(full), dtype=np.int8)
    for stage, indices in enumerate(centre_indices, start=1):
        for value in indices:
            deepest[index_to_row[int(value)]] = stage

    fig, axes = plt.subplots(1, 4, figsize=(22, 7.2), constrained_layout=True)
    centre_colors = ("#1976D2", "#F57C00", "#D32F2F")
    for stage, (ax, source, membership, indices, colour) in enumerate(
        zip(axes[:3], sources, memberships, centre_indices, centre_colors), start=1
    ):
        rows = np.asarray([index_to_row[int(value)] for value in indices], dtype=np.int64)
        sc = ax.scatter(
            source[:, 0], source[:, 2], c=membership, cmap="viridis", s=10,
            alpha=0.78, linewidths=0, vmin=0, vmax=max(1, int(membership.max())), zorder=1,
        )
        # Each stage's source is a subset of the original cloud.  Centres are
        # in that source set, so draw them directly from its coordinates.
        source_row_for_idx = {
            int(value): row for row, value in enumerate(
                _read_vtp(output_dir / f"sa{stage}_source_points.vtp", ("full_wall_index",))[1]["full_wall_index"]
            )
        }
        centre_rows = np.asarray([source_row_for_idx[int(value)] for value in indices], dtype=np.int64)
        ax.scatter(source[centre_rows, 0], source[centre_rows, 2], c=colour, marker="x", s=38,
                   linewidths=1.25, alpha=0.96, zorder=3)
        bar = fig.colorbar(sc, ax=ax, pad=0.015, shrink=0.70)
        bar.set_label("number of groups containing this point", fontsize=8)
        bar.ax.tick_params(labelsize=7)
        ax.set_title(
            f"SA{stage}: all group points + all centres\n"
            f"{len(source)} source points → {len(indices)} centres; "
            f"mean overlap={membership.mean():.2f}, max={membership.max()}", fontsize=10,
        )
        ax.set_aspect("equal")
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Z (mm)")

    ax = axes[3]
    ax.scatter(full[:, 0], full[:, 2], c="#D0D0D0", s=8, alpha=0.52, linewidths=0, zorder=1)
    for stage, colour, label, marker, size in (
        (1, "#1976D2", "SA1 only (375)", "o", 17),
        (2, "#F57C00", "SA1+SA2 only (93)", "s", 26),
        (3, "#D32F2F", "SA1+SA2+SA3 (32)", "*", 62),
    ):
        points = full[deepest == stage]
        ax.scatter(points[:, 0], points[:, 2], c=colour, marker=marker, s=size,
                   edgecolors="white", linewidths=0.28, alpha=0.96, label=label, zorder=stage + 1)
    ax.set_title("All centre layers, coincident copies merged\ncolour/marker = deepest selected SA stage", fontsize=10)
    ax.set_aspect("equal")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Z (mm)")
    ax.legend(loc="lower center", fontsize=8, frameon=True)
    fig.suptitle(
        "PointNet++ SA3: complete group coverage and nested FPS centre selection (fast/RAN_QING_BO)",
        fontsize=15,
    )
    output = output_dir / "05_SA全部中心与分组点重合总览.png"
    fig.savefig(output, dpi=240)
    plt.close(fig)
    return output


def _write_overview_from_csv(output_dir: Path) -> Path:
    """CSV-only variant, kept independent of VTK readers for easy regeneration."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with (output_dir / "sa_centers_marked.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    full_index = np.asarray([int(row["full_wall_index"]) for row in rows], dtype=np.int64)
    full = np.asarray(
        [[float(row["x_mm"]), float(row["y_mm"]), float(row["z_mm"])] for row in rows],
        dtype=np.float64,
    )
    index_to_row = {int(value): row for row, value in enumerate(full_index)}
    deepest = np.asarray([int(row["deepest_sa_stage"]) for row in rows], dtype=np.int8)
    centre_masks = [
        np.asarray([int(row[f"is_sa{stage}_center"]) == 1 for row in rows], dtype=bool)
        for stage in (1, 2, 3)
    ]
    source_masks = [np.ones(len(rows), dtype=bool), centre_masks[0], centre_masks[1]]
    memberships: list[np.ndarray] = []
    for stage in (1, 2, 3):
        counts: dict[int, int] = {}
        with (output_dir / f"sa{stage}_assignments.csv").open(newline="", encoding="utf-8") as f:
            for assignment in csv.DictReader(f):
                source_index = int(assignment["source_full_wall_index"])
                counts[source_index] = counts.get(source_index, 0) + 1
        memberships.append(np.asarray([counts.get(int(value), 0) for value in full_index], dtype=np.int64))

    fig, axes = plt.subplots(1, 4, figsize=(22, 7.2), constrained_layout=True)
    centre_colors = ("#1976D2", "#F57C00", "#D32F2F")
    for stage, (ax, source_mask, centre_mask, membership, colour) in enumerate(
        zip(axes[:3], source_masks, centre_masks, memberships, centre_colors), start=1
    ):
        sc = ax.scatter(
            full[source_mask, 0], full[source_mask, 2], c=membership[source_mask], cmap="viridis",
            s=10, alpha=0.78, linewidths=0, vmin=0,
            vmax=max(1, int(membership[source_mask].max())), zorder=1,
        )
        ax.scatter(full[centre_mask, 0], full[centre_mask, 2], c=colour, marker="x", s=38,
                   linewidths=1.25, alpha=0.96, zorder=3)
        bar = fig.colorbar(sc, ax=ax, pad=0.015, shrink=0.70)
        bar.set_label("number of groups containing this point", fontsize=8)
        bar.ax.tick_params(labelsize=7)
        ax.set_title(
            f"SA{stage}: all group points + all centres\n"
            f"{source_mask.sum()} source points → {centre_mask.sum()} centres; "
            f"mean overlap={membership[source_mask].mean():.2f}, "
            f"max={membership[source_mask].max()}", fontsize=10,
        )
        ax.set_aspect("equal")
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Z (mm)")

    ax = axes[3]
    ax.scatter(full[:, 0], full[:, 2], c="#D0D0D0", s=8, alpha=0.52, linewidths=0, zorder=1)
    for stage, colour, label, marker, size in (
        (1, "#1976D2", "SA1 only (375)", "o", 17),
        (2, "#F57C00", "SA1+SA2 only (93)", "s", 26),
        (3, "#D32F2F", "SA1+SA2+SA3 (32)", "*", 62),
    ):
        points = full[deepest == stage]
        ax.scatter(points[:, 0], points[:, 2], c=colour, marker=marker, s=size,
                   edgecolors="white", linewidths=0.28, alpha=0.96, label=label, zorder=stage + 1)
    ax.set_title("All centre layers, coincident copies merged\ncolour/marker = deepest selected SA stage", fontsize=10)
    ax.set_aspect("equal")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Z (mm)")
    ax.legend(loc="lower center", fontsize=8, frameon=True)
    fig.suptitle(
        "PointNet++ SA3: complete group coverage and nested FPS centre selection (fast/RAN_QING_BO)",
        fontsize=15,
    )
    output = output_dir / "05_SA全部中心与分组点重合总览.png"
    fig.savefig(output, dpi=240)
    plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="从已有 SA VTP 生成 ParaView 一键入口和全量总览")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    vtm = _write_multiblock(args.output_dir)
    png = _write_overview_from_csv(args.output_dir)
    point_vtm = _write_pointcloud_only_multiblock(args.output_dir)
    point_png, report = _write_pointcloud_only_view_and_report(args.output_dir)
    adjacent_png, adjacent_report = _write_adjacent_group_overlap_example(args.output_dir)
    print(vtm)
    print(png)
    print(point_vtm)
    print(point_png)
    print(report)
    print(adjacent_png)
    print(adjacent_report)


if __name__ == "__main__":
    main()
