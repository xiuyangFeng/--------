"""Fluent case topology needed by the V5 bundle (thin wrapper over wss_pinn.v4.fluent_topology)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from wss_pinn.v4.fluent_topology import (
    FluentMesh,
    InterfaceZone,
    anatomy_topology,
    anatomy_wall_faces,
    read_fluent_mesh,
)

VELOCITY_INLET_BC_TYPES = (10, 4, 20)


@dataclass
class MeshData:
    mesh: FluentMesh
    centroids_m: np.ndarray  # (N+1, 3), index 0 NaN
    volumes_m3: np.ndarray   # (N+1,)
    cell_zone: np.ndarray    # (N+1,)
    anatomy_zone_id: int
    topology: dict[str, Any]
    wall: dict[str, np.ndarray]
    wall_node_ids: np.ndarray        # sorted unique 1-based node ids of the anatomy wall
    wall_node_coords_m: np.ndarray   # (Nw, 3)
    wall_node_normals_out: np.ndarray  # (Nw, 3) area-weighted, pointing out of the fluid
    wall_node_area_m2: np.ndarray    # (Nw,) equal share of incident face areas
    wall_node_rim: np.ndarray        # (Nw,) bool: node also belongs to an interface cut
    interfaces: list[InterfaceZone]
    inlet_bc_face_area_m2: float
    anatomy_volume_m3: float
    anatomy_wall_area_m2: float

    def node_row(self, node_ids: np.ndarray) -> np.ndarray:
        """Map 1-based node ids -> wall row index (-1 when not an anatomy wall node)."""
        lookup = -np.ones(self.mesh.node_count + 1, dtype=np.int64)
        lookup[self.wall_node_ids] = np.arange(len(self.wall_node_ids))
        return lookup[np.asarray(node_ids, dtype=np.int64)]


def _node_share_areas_and_normals(mesh: FluentMesh, wall: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    offsets = wall["face_offsets"]
    nodes = wall["face_nodes"]
    counts = np.diff(offsets)
    area_vec = wall["area_vectors_m2"]
    area = np.linalg.norm(area_vec, axis=1)
    face_index = np.repeat(np.arange(len(counts)), counts)
    n_nodes = mesh.node_count + 1
    share = np.bincount(nodes, weights=(area / counts)[face_index], minlength=n_nodes)
    normal = np.zeros((n_nodes, 3), dtype=np.float64)
    for axis in range(3):
        normal[:, axis] = np.bincount(nodes, weights=area_vec[face_index, axis], minlength=n_nodes)
    ids = np.unique(nodes)
    vectors = normal[ids]
    norm = np.linalg.norm(vectors, axis=1)
    return ids, share[ids], vectors / np.maximum(norm, 1.0e-300)[:, None]


def inlet_bc_face_area(mesh: FluentMesh) -> float:
    total = 0.0
    for bc_type in VELOCITY_INLET_BC_TYPES:
        for section in mesh.face_sections_of_type(bc_type):
            if not section.attached:
                continue
            _, area = mesh.face_geometry(section)
            total += float(np.linalg.norm(area, axis=1).sum())
    return total


def load_mesh(cas_path) -> MeshData:
    mesh = read_fluent_mesh(cas_path)
    centroids, volumes = mesh.cell_centroids_and_volumes()
    cell_zone = mesh.cell_zone_array()
    anatomy_zone_id = int(mesh.fluid_zone_by_name()["blood"])
    topology = anatomy_topology(mesh)
    wall = anatomy_wall_faces(mesh, topology)
    ids, areas, normals = _node_share_areas_and_normals(mesh, wall)
    interfaces = list(topology["interfaces"])
    rim_nodes = set()
    for interface in interfaces:
        rim_nodes.update(interface.node_ids.tolist())
    rim = np.fromiter((int(i) in rim_nodes for i in ids), dtype=bool, count=len(ids))
    anatomy_cells = cell_zone[1:] == anatomy_zone_id
    return MeshData(
        mesh=mesh,
        centroids_m=centroids,
        volumes_m3=volumes,
        cell_zone=cell_zone,
        anatomy_zone_id=anatomy_zone_id,
        topology=topology,
        wall=wall,
        wall_node_ids=ids,
        wall_node_coords_m=mesh.nodes_m[ids],
        wall_node_normals_out=normals,
        wall_node_area_m2=areas,
        wall_node_rim=rim,
        interfaces=interfaces,
        inlet_bc_face_area_m2=inlet_bc_face_area(mesh),
        anatomy_volume_m3=float(np.nansum(volumes[1:][anatomy_cells])),
        anatomy_wall_area_m2=float(np.linalg.norm(wall["area_vectors_m2"], axis=1).sum()),
    )


def wall_triangles(md: MeshData) -> tuple[np.ndarray, np.ndarray]:
    """Fan-triangulate the anatomy wall faces.

    Returns ``(triangles, face_of_triangle)`` with triangle vertices as wall
    row indices (into ``md.wall_node_ids`` order).
    """
    offsets = md.wall["face_offsets"]
    nodes = md.node_row(md.wall["face_nodes"])
    counts = np.diff(offsets)
    tris = []
    owner = []
    max_n = int(counts.max())
    for n in range(3, max_n + 1):
        faces = np.flatnonzero(counts == n)
        if not len(faces):
            continue
        starts = offsets[:-1][faces]
        poly = nodes[starts[:, None] + np.arange(n)[None, :]]  # (F, n)
        for k in range(1, n - 1):
            tris.append(np.stack([poly[:, 0], poly[:, k], poly[:, k + 1]], axis=1))
            owner.append(faces)
    triangles = np.concatenate(tris, axis=0)
    face_of_triangle = np.concatenate(owner, axis=0)
    if np.any(triangles < 0):
        raise ValueError("wall face references a node outside the anatomy wall node set")
    return triangles, face_of_triangle


def point_triangle_distance(points: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Exact Euclidean distance from each point to its own triangle (vectorised, Ericson 5.1.5)."""
    ab = b - a
    ac = c - a
    ap = points - a
    d1 = np.einsum("ij,ij->i", ab, ap)
    d2 = np.einsum("ij,ij->i", ac, ap)
    bp = points - b
    d3 = np.einsum("ij,ij->i", ab, bp)
    d4 = np.einsum("ij,ij->i", ac, bp)
    cp = points - c
    d5 = np.einsum("ij,ij->i", ab, cp)
    d6 = np.einsum("ij,ij->i", ac, cp)
    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4
    closest = np.empty_like(points)
    done = np.zeros(len(points), dtype=bool)

    def assign(mask, value):
        m = mask & ~done
        closest[m] = value[m] if value.ndim == 2 else value
        done[m] = True

    assign((d1 <= 0) & (d2 <= 0), a)
    assign((d3 >= 0) & (d4 <= d3), b)
    assign((d6 >= 0) & (d5 <= d6), c)
    with np.errstate(divide="ignore", invalid="ignore"):
        v = d1 / (d1 - d3)
        assign((vc <= 0) & (d1 >= 0) & (d3 <= 0), a + v[:, None] * ab)
        w = d2 / (d2 - d6)
        assign((vb <= 0) & (d2 >= 0) & (d6 <= 0), a + w[:, None] * ac)
        w2 = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        assign((va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0), b + w2[:, None] * (c - b))
        denom = 1.0 / np.maximum(va + vb + vc, 1.0e-300)
        vv = vb * denom
        ww = vc * denom
        assign(~done, a + vv[:, None] * ab + ww[:, None] * ac)
    return np.linalg.norm(points - closest, axis=1)


def distance_to_wall(points_m: np.ndarray, md: MeshData, k: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """Exact distance from points to the anatomy wall surface (nearest of k candidate triangles)."""
    from scipy.spatial import cKDTree

    tris, _ = wall_triangles(md)
    xyz = md.wall_node_coords_m
    tri_centroid = xyz[tris].mean(axis=1)
    tree = cKDTree(tri_centroid)
    k = min(k, len(tris))
    _, cand = tree.query(points_m, k=k)
    cand = np.atleast_2d(cand).reshape(len(points_m), k)
    best = np.full(len(points_m), np.inf)
    best_tri = np.zeros(len(points_m), dtype=np.int64)
    for j in range(k):
        t = tris[cand[:, j]]
        d = point_triangle_distance(points_m, xyz[t[:, 0]], xyz[t[:, 1]], xyz[t[:, 2]])
        better = d < best
        best[better] = d[better]
        best_tri[better] = cand[better, j]
    return best, best_tri
