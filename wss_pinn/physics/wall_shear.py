from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from .rheology import carreau_yasuda_numpy


def _binary_stl_triangles(path: Path) -> np.ndarray:
    size = path.stat().st_size
    with path.open("rb") as handle:
        header = handle.read(80)
        count_raw = handle.read(4)
        if len(count_raw) != 4:
            raise ValueError("truncated STL")
        count = struct.unpack("<I", count_raw)[0]
        if 84 + count * 50 != size:
            raise ValueError("not a binary STL")
        triangles = np.empty((count, 3, 3), dtype=np.float64)
        for index in range(count):
            record = handle.read(50)
            values = struct.unpack("<12fH", record)
            triangles[index] = np.asarray(values[3:12]).reshape(3, 3)
    return triangles


def _ascii_stl_triangles(path: Path) -> np.ndarray:
    vertices: list[list[float]] = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            fields = line.strip().split()
            if len(fields) == 4 and fields[0].lower() == "vertex":
                vertices.append([float(value) for value in fields[1:]])
    if len(vertices) == 0 or len(vertices) % 3:
        raise ValueError("invalid ASCII STL")
    return np.asarray(vertices, dtype=np.float64).reshape(-1, 3, 3)


def read_stl_triangles(path: str | Path) -> np.ndarray:
    source = Path(path)
    try:
        return _binary_stl_triangles(source)
    except ValueError:
        return _ascii_stl_triangles(source)


def stl_face_geometry(
    path: str | Path, scale_to_mm: float
) -> tuple[np.ndarray, np.ndarray]:
    triangles = read_stl_triangles(path) * float(scale_to_mm)
    edge_a = triangles[:, 1] - triangles[:, 0]
    edge_b = triangles[:, 2] - triangles[:, 0]
    normals = np.cross(edge_a, edge_b)
    magnitude = np.linalg.norm(normals, axis=1)
    valid = magnitude > 1e-12
    centers = triangles[valid].mean(axis=1)
    normals = normals[valid] / magnitude[valid, None]
    if not len(centers):
        raise ValueError("STL contains no non-degenerate faces")
    return centers, normals


def map_stl_normals_to_wall(
    wall_coords_mm: np.ndarray,
    stl_path: str | Path,
    scale_to_mm: float,
    rotation: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    from scipy.spatial import cKDTree

    centers, normals = stl_face_geometry(stl_path, scale_to_mm)
    distance, index = cKDTree(centers).query(np.asarray(wall_coords_mm), k=1)
    mapped = normals[index]
    if rotation is not None:
        mapped = mapped @ np.asarray(rotation, dtype=np.float64)
    mapped /= np.maximum(np.linalg.norm(mapped, axis=1, keepdims=True), 1e-12)
    return mapped.astype(np.float32), np.asarray(distance, dtype=np.float32)


def wall_shear_from_normal_gradient(
    tangential_gradient_s_inv: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    gradient = np.asarray(tangential_gradient_s_inv, dtype=np.float64)
    shear_rate = np.linalg.norm(gradient, axis=-1)
    viscosity = carreau_yasuda_numpy(shear_rate)
    return gradient * viscosity[..., None], viscosity


def fit_normal_velocity_gradient(
    wall_point_mm: np.ndarray,
    wall_normal: np.ndarray,
    interior_coords_mm: np.ndarray,
    interior_velocity_m_s: np.ndarray,
    *,
    neighbors: int = 64,
    degree: int = 1,
) -> tuple[np.ndarray, dict[str, float]]:
    """Fit no-slip tangential velocity against normal distance."""
    from scipy.spatial import cKDTree

    xyz = np.asarray(interior_coords_mm, dtype=np.float64)
    velocity = np.asarray(interior_velocity_m_s, dtype=np.float64)
    normal = np.asarray(wall_normal, dtype=np.float64)
    normal /= max(np.linalg.norm(normal), 1e-12)
    k = min(max(int(neighbors), 12), len(xyz))
    distance, index = cKDTree(xyz).query(np.asarray(wall_point_mm), k=k)
    delta = xyz[index] - wall_point_mm
    signed = delta @ normal
    positive = signed > 0
    negative = signed < 0
    use_positive = positive.sum() >= negative.sum()
    mask = positive if use_positive else negative
    if mask.sum() < max(8, degree + 3):
        mask = np.ones_like(signed, dtype=bool)
    d_mm = np.abs(signed[mask])
    tangent = velocity[index][mask] - np.outer(velocity[index][mask] @ normal, normal)
    keep = d_mm > 1e-5
    d_m = d_mm[keep] * 1e-3
    tangent = tangent[keep]
    if len(d_m) < max(6, degree + 2):
        return np.full(3, np.nan), {"n_used": float(len(d_m)), "condition": np.inf}
    columns = [d_m**power for power in range(1, int(degree) + 1)]
    design = np.column_stack(columns)
    bandwidth = max(float(np.median(d_m)), 1e-8)
    weights = np.exp(-np.square(d_m / (2.0 * bandwidth)))
    weighted = design * np.sqrt(weights[:, None])
    target = tangent * np.sqrt(weights[:, None])
    coefficients, *_ = np.linalg.lstsq(weighted, target, rcond=1e-10)
    return coefficients[0], {
        "n_used": float(len(d_m)),
        "condition": float(np.linalg.cond(weighted.T @ weighted)),
        "distance_p50_mm": float(np.median(d_mm[keep])),
        "distance_p95_mm": float(np.quantile(d_mm[keep], 0.95)),
    }

