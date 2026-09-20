#!/usr/bin/env python3
"""Surface topology utilities shared by sampling, metrics, and PostView."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


def stable_seed(*parts: object) -> int:
    payload = "|".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def load_stl(path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    path = Path(path)
    size = path.stat().st_size
    with path.open("rb") as f:
        header = f.read(84)
    n_faces = int.from_bytes(header[80:84], "little") if len(header) == 84 else -1
    if n_faces >= 0 and 84 + 50 * n_faces == size:
        dtype = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)),
                          ("attr", "<u2")])
        records = np.memmap(path, dtype=dtype, mode="r", offset=84, shape=(n_faces,))
        raw = np.asarray(records["vertices"], dtype=np.float64).reshape(-1, 3)
    else:
        raw = []
        with path.open("rt", errors="ignore") as f:
            for line in f:
                if line.lstrip().startswith("vertex "):
                    raw.append([float(v) for v in line.split()[1:4]])
        raw = np.asarray(raw, dtype=np.float64)
        if len(raw) % 3:
            raise RuntimeError(f"invalid ASCII STL vertices: {path}")
        n_faces = len(raw) // 3
    vertices, inverse = np.unique(raw, axis=0, return_inverse=True)
    faces = inverse.reshape(n_faces, 3).astype(np.int64)
    if not len(vertices) or not len(faces):
        raise RuntimeError(f"invalid STL surface: {path}")
    return vertices, faces


def area_weights_for_case(case: Dict, *, strict: bool = True) -> Tuple[np.ndarray, Dict]:
    """Transfer one-third triangle areas from the original STL to CFD wall points."""
    from scipy.spatial import cKDTree

    wall = np.asarray(case["wall_coords_raw"], dtype=np.float64)
    path = Path(case["original_stl_path"])
    vertices, faces = load_stl(path)
    vertices *= float(case["original_stl_scale_to_mm"])
    tri_area = 0.5 * np.linalg.norm(
        np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]],
                 vertices[faces[:, 2]] - vertices[faces[:, 0]]), axis=1
    )
    if not np.isfinite(tri_area).all() or np.any(tri_area < 0) or tri_area.sum() <= 0:
        raise ValueError(f"invalid triangle areas: {case['unit_id']}")
    vertex_area = np.zeros(len(vertices), dtype=np.float64)
    np.add.at(vertex_area, faces.reshape(-1), np.repeat(tri_area / 3.0, 3))

    wall_tree = cKDTree(wall)
    stl_to_wall_dist, nearest_wall = wall_tree.query(vertices, k=1)
    wall_area = np.zeros(len(wall), dtype=np.float64)
    np.add.at(wall_area, nearest_wall, vertex_area)
    stl_tree = cKDTree(vertices)
    wall_to_stl_dist = stl_tree.query(wall, k=1)[0]

    bbox_diag = float(np.linalg.norm(wall.max(axis=0) - wall.min(axis=0)))
    total_before = float(vertex_area.sum())
    total_after = float(wall_area.sum())
    positive = int(np.count_nonzero(wall_area > 0))
    report = {
        "unit_id": case["unit_id"],
        "stl_path": str(path),
        "n_wall": int(len(wall)),
        "n_stl_vertices": int(len(vertices)),
        "n_triangles": int(len(faces)),
        "n_positive_wall_weights": positive,
        "surface_area_mm2": total_before,
        "area_conservation_relerr": abs(total_after - total_before) / total_before,
        "bbox_diag_mm": bbox_diag,
        "original_stl_scale_to_mm": float(case["original_stl_scale_to_mm"]),
        "original_stl_match_score": float(case.get("original_stl_match_score", np.nan)),
        "wall_crop_applied": bool(case.get("wall_crop_applied", False)),
        "wall_crop_frac": float(case.get("wall_crop_frac", 0.0)),
        "stl_to_wall_p50_mm": float(np.percentile(stl_to_wall_dist, 50)),
        "stl_to_wall_p95_mm": float(np.percentile(stl_to_wall_dist, 95)),
        "stl_to_wall_max_mm": float(stl_to_wall_dist.max()),
        "wall_to_stl_p95_mm": float(np.percentile(wall_to_stl_dist, 95)),
        "wall_to_stl_max_mm": float(wall_to_stl_dist.max()),
    }
    p95_rel = max(report["stl_to_wall_p95_mm"], report["wall_to_stl_p95_mm"]) / max(bbox_diag, 1e-12)
    max_rel = max(report["stl_to_wall_max_mm"], report["wall_to_stl_max_mm"]) / max(bbox_diag, 1e-12)
    report.update({"bidirectional_p95_over_bbox": p95_rel,
                   "bidirectional_max_over_bbox": max_rel})
    failures = []
    if not np.isfinite(wall_area).all() or np.any(wall_area < 0):
        failures.append("nonfinite_or_negative_weights")
    if positive < 5000:
        failures.append("fewer_than_5000_positive_weights")
    if report["area_conservation_relerr"] > 1e-10:
        failures.append("area_not_conserved")
    if p95_rel > 0.02:
        failures.append("mapping_p95_over_2pct_bbox")
    if max_rel > 0.10:
        failures.append("mapping_max_over_10pct_bbox")
    report["failures"] = failures
    report["status"] = "passed" if not failures else "failed"
    if strict and failures:
        raise RuntimeError(f"surface-area mapping failed for {case['unit_id']}: {failures}")
    wall_area /= wall_area.sum()
    return wall_area, report
