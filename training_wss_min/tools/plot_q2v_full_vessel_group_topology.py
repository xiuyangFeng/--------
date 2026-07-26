#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Full-vessel plus local SA1 group view with a surface-geodesic risk screen.

The selected pair is a true SA1 nearest-neighbour relation with 5/16 shared
members (the teacher's <=1/3 reference).  The screen compares a mesh-surface
shortest-path distance with the Euclidean distance used by ball query.  It is a
topology-risk flag, not an anatomical branch label.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from training_wss_min.tools.build_q2v_sa_overlap_audit_package import (
    PACKAGE,
    TRACE_ROOT,
    load_stage,
    nearest_pairs,
    pair_detail,
)


TRACE = TRACE_ROOT / "ball_r100"
DEFAULT_OUTPUT = PACKAGE / "06_SA1_完整血管_相邻group与局部放大_跨支风险审计.png"
RISK_RATIO = 2.5


def select_pair(data: dict) -> tuple[int, int]:
    """Pick the closest nearest-neighbour pair with exactly 5 shared of 16."""
    candidates = []
    for pair in nearest_pairs(data):
        detail = pair_detail(data, data["members"], pair)
        if (detail["group_a_points"], detail["group_b_points"], detail["shared_points"]) == (16, 16, 5):
            candidates.append((detail["center_distance_mm"], pair))
    if not candidates:
        raise RuntimeError("no SA1 nearest-neighbour pair with exactly 5/16 shared members")
    return min(candidates)[1]


def read_surface(path: Path) -> tuple[np.ndarray, np.ndarray]:
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    surface = reader.GetOutput()
    vertices = vtk_to_numpy(surface.GetPoints().GetData()).astype(float)
    raw = vtk_to_numpy(surface.GetPolys().GetData()).astype(np.int64)
    triangles = []
    index = 0
    while index < len(raw):
        size = int(raw[index])
        if size == 3:
            triangles.append(raw[index + 1:index + 4])
        index += size + 1
    return vertices, np.asarray(triangles, dtype=np.int64)


def surface_graph(vertices: np.ndarray, triangles: np.ndarray):
    edges = np.concatenate((triangles[:, (0, 1)], triangles[:, (1, 2)], triangles[:, (2, 0)]), axis=0)
    reverse = edges[:, ::-1]
    edges = np.concatenate((edges, reverse), axis=0)
    weights = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    return coo_matrix((weights, (edges[:, 0], edges[:, 1])), shape=(len(vertices), len(vertices))).tocsr()


def topology_rows(data: dict, pair: tuple[int, int], vertices: np.ndarray, graph) -> tuple[list[dict], dict]:
    detail = pair_detail(data, data["members"], pair)
    tree = cKDTree(vertices)
    centre_vertices = {centre: int(tree.query(data["centres"][centre])[1]) for centre in pair}
    geodesics = {centre: dijkstra(graph, directed=False, indices=vertex) for centre, vertex in centre_vertices.items()}
    rows: list[dict] = []
    for point_id in sorted(data["members"][pair[0]] | data["members"][pair[1]]):
        point = data["point_xyz"][point_id]
        point_vertex = int(tree.query(point)[1])
        membership = "shared" if point_id in detail["shared"] else "A only" if point_id in detail["only_a"] else "B only"
        for label, centre in (("A", pair[0]), ("B", pair[1])):
            if point_id not in data["members"][centre]:
                continue
            euclidean = float(np.linalg.norm(point - data["centres"][centre]))
            geodesic = float(geodesics[centre][point_vertex])
            ratio = 1.0 if euclidean < 1e-8 else geodesic / euclidean
            rows.append({
                "center_label": label, "center_id": centre, "source_full_wall_index": point_id,
                "membership": membership, "euclidean_ball_distance_mm": euclidean,
                "surface_geodesic_distance_mm": geodesic, "geodesic_over_euclidean": ratio,
                "topology_risk_flag_ratio_ge_2_5": int(ratio >= RISK_RATIO),
            })
    finite = [row["geodesic_over_euclidean"] for row in rows if np.isfinite(row["geodesic_over_euclidean"])]
    summary = {
        "pair": {"center_a_id": pair[0], "center_b_id": pair[1]},
        "ball_radius_mm": data["radius_mm"], "group_overlap": detail["overlap_rate"],
        "member_edges": len(rows), "risk_flag_edges": sum(row["topology_risk_flag_ratio_ge_2_5"] for row in rows),
        "max_geodesic_over_euclidean": float(max(finite)),
        "median_geodesic_over_euclidean": float(np.median(finite)),
        "interpretation": "A flag means the member is much farther along this surface mesh than in Euclidean space; it is a topology-shortcut screening signal, not a confirmed anatomical branch label.",
    }
    return rows, summary


def draw_figure(data: dict, pair: tuple[int, int], vertices: np.ndarray, triangles: np.ndarray, topology_rows: list[dict],
                topology: dict, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    detail = pair_detail(data, data["members"], pair)
    ca, cb = data["centres"][pair[0]], data["centres"][pair[1]]
    fig = plt.figure(figsize=(16.8, 7.5))
    left = fig.add_subplot(1, 2, 1, projection="3d")
    right = fig.add_subplot(1, 2, 2)

    # Full surface establishes the anatomical location.  A transparent mesh keeps highlighted points visible.
    mesh = Poly3DCollection(vertices[triangles], facecolor="#C9C9C9", edgecolor="none", alpha=0.48)
    left.add_collection3d(mesh)
    for ids, colour, label, size in ((detail["only_a"], "#1976D2", "A only", 76),
                                     (detail["only_b"], "#F57C00", "B only", 76),
                                     (detail["shared"], "#8E24AA", "shared", 94)):
        xyz = np.asarray([data["point_xyz"][point] for point in sorted(ids)], dtype=float)
        left.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=colour, s=size, edgecolors="white", linewidths=0.45,
                     depthshade=False, label=f"{label} ({len(ids)})")
    left.scatter(*ca, marker="*", s=340, c="#0D47A1", edgecolors="black", linewidths=0.6,
                 depthshade=False, label=f"center A ({pair[0]})")
    left.scatter(*cb, marker="*", s=340, c="#E65100", edgecolors="black", linewidths=0.6,
                 depthshade=False, label=f"center B ({pair[1]})")
    left.plot([ca[0], cb[0]], [ca[1], cb[1]], [ca[2], cb[2]], c="#444444", ls="--", lw=1.0)
    midpoint = (ca + cb) / 2
    ranges = np.ptp(vertices, axis=0)
    half = float(ranges.max()) / 2
    full_mid = (vertices.max(axis=0) + vertices.min(axis=0)) / 2
    left.set_xlim(full_mid[0] - half, full_mid[0] + half)
    left.set_ylim(full_mid[1] - half, full_mid[1] + half)
    left.set_zlim(full_mid[2] - half, full_mid[2] + half)
    left.set_box_aspect((1, 1, 1))
    left.view_init(elev=17, azim=-68)
    left.set_axis_off()
    flagged_ids = sorted({row["source_full_wall_index"] for row in topology_rows if row["topology_risk_flag_ratio_ge_2_5"]})
    if flagged_ids:
        flagged_xyz = np.asarray([data["point_xyz"][point] for point in flagged_ids], dtype=float)
        left.scatter(flagged_xyz[:, 0], flagged_xyz[:, 1], flagged_xyz[:, 2], marker="x", s=105, c="#C62828",
                     linewidths=1.8, depthshade=False, label=f"topology-screen flag ({len(flagged_ids)})", zorder=8)
    left.legend(loc="upper left", fontsize=7.3, frameon=True)
    left.set_title("Complete vessel surface | selected nearby SA1 groups", fontsize=12)

    local_scale = data["radius_mm"] * 1.45
    local = data["full_xyz"][np.linalg.norm(data["full_xyz"] - midpoint, axis=1) <= local_scale]
    right.scatter(local[:, 0], local[:, 2], c="#D0D0D0", s=9, alpha=0.42, linewidths=0, label="local support")
    for ids, colour, label, size in ((detail["only_a"], "#1976D2", "A only", 62),
                                     (detail["only_b"], "#F57C00", "B only", 62),
                                     (detail["shared"], "#8E24AA", "shared", 76)):
        xyz = np.asarray([data["point_xyz"][point] for point in sorted(ids)], dtype=float)
        right.scatter(xyz[:, 0], xyz[:, 2], c=colour, s=size, edgecolors="white", linewidths=0.45,
                      label=f"{label} ({len(ids)})", zorder=3)
    right.scatter(ca[0], ca[2], marker="*", s=245, c="#0D47A1", edgecolors="black", linewidths=0.55,
                  label=f"center A ({pair[0]})", zorder=5)
    right.scatter(cb[0], cb[2], marker="*", s=245, c="#E65100", edgecolors="black", linewidths=0.55,
                  label=f"center B ({pair[1]})", zorder=5)
    right.plot([ca[0], cb[0]], [ca[2], cb[2]], c="#444444", ls="--", lw=1.0)
    if flagged_ids:
        right.scatter(flagged_xyz[:, 0], flagged_xyz[:, 2], marker="x", s=115, c="#C62828", linewidths=1.8,
                      label=f"topology-screen flag ({len(flagged_ids)})", zorder=8)
    right.add_patch(Circle((ca[0], ca[2]), data["radius_mm"], fill=False, ec="#0D47A1", ls="--", lw=1.2))
    right.add_patch(Circle((cb[0], cb[2]), data["radius_mm"], fill=False, ec="#E65100", ls="--", lw=1.2))
    half = max(float(np.ptp(local[:, (0, 2)], axis=0).max()) / 2, 7.0)
    right.set_xlim(midpoint[0] - half, midpoint[0] + half)
    right.set_ylim(midpoint[2] - half, midpoint[2] + half)
    right.set_aspect("equal")
    right.set_xlabel("X (mm)")
    right.set_ylabel("Z (mm)")
    right.legend(loc="upper right", fontsize=7.4, frameon=True)
    right.set_title(
        f"Local enlargement | shared={detail['shared_points']}/16 = {detail['overlap_rate']:.1%}; "
        f"r={data['radius_mm']:.1f} mm\n"
        f"surface-topology screen: {topology['risk_flag_edges']}/{topology['member_edges']} member edges flagged; "
        f"max geodesic/Euclidean={topology['max_geodesic_over_euclidean']:.2f}", fontsize=10.4,
    )
    fig.suptitle(
        "Q2V-10477 | baseline SA1 ball query: full-vessel location and local group members\n"
        "blue/orange = unique group members; purple = shared; dashed circles = Euclidean ball-query radius", fontsize=14,
    )
    fig.subplots_adjust(left=0.03, right=0.985, bottom=0.06, top=0.86, wspace=0.10)
    fig.savefig(output, dpi=240)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="plot full-vessel SA1 pair and topology risk")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    data = load_stage(TRACE, 1)
    pair = select_pair(data)
    vertices, triangles = read_surface(TRACE / "vessel_surface.vtp")
    graph = surface_graph(vertices, triangles)
    rows, summary = topology_rows(data, pair, vertices, graph)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    draw_figure(data, pair, vertices, triangles, rows, summary, args.output)
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_path = args.output.with_suffix(".json")
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    print(csv_path)
    print(json_path)


if __name__ == "__main__":
    main()
