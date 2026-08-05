#!/usr/bin/env python3
"""Run a frozen WSS calculator script with its NumPy rheology API restored.

The active WSS-PINN refactor currently exposes only the differentiable Torch
API, while the frozen MRI calculator imports ``carreau_yasuda_numpy``.  This
wrapper restores that historical, UDF-audited NumPy symbol in memory without
editing either route's source files.
"""

from __future__ import annotations

import runpy
from pathlib import Path
import struct
import sys
import types

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from wss_pinn.physics import rheology


UDF_PARAMETERS = {
    "mu_inf_pa_s": 0.0035,
    "mu_zero_pa_s": 0.16,
    "lambda_s": 8.2,
    "a": 0.64,
    "n": 0.2128,
}


def carreau_yasuda_numpy(
    shear_rate_s_inv: np.ndarray | float,
    *,
    mu_inf_pa_s: float = UDF_PARAMETERS["mu_inf_pa_s"],
    mu_zero_pa_s: float = UDF_PARAMETERS["mu_zero_pa_s"],
    lambda_s: float = UDF_PARAMETERS["lambda_s"],
    a: float = UDF_PARAMETERS["a"],
    n: float = UDF_PARAMETERS["n"],
) -> np.ndarray:
    gamma = np.asarray(shear_rate_s_inv, dtype=np.float64)
    if np.any(gamma < 0):
        raise ValueError("shear rate must be non-negative")
    return mu_inf_pa_s + (mu_zero_pa_s - mu_inf_pa_s) * np.power(
        1.0 + np.power(lambda_s * gamma, a), (n - 1.0) / a
    )


def _binary_stl_triangles(path: Path) -> np.ndarray:
    size = path.stat().st_size
    with path.open("rb") as handle:
        handle.read(80)
        count_raw = handle.read(4)
        if len(count_raw) != 4:
            raise ValueError("truncated STL")
        count = struct.unpack("<I", count_raw)[0]
        if 84 + count * 50 != size:
            raise ValueError("not a binary STL")
        triangles = np.empty((count, 3, 3), dtype=np.float64)
        for index in range(count):
            values = struct.unpack("<12fH", handle.read(50))
            triangles[index] = np.asarray(values[3:12]).reshape(3, 3)
    return triangles


def _ascii_stl_triangles(path: Path) -> np.ndarray:
    vertices: list[list[float]] = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            fields = line.strip().split()
            if len(fields) == 4 and fields[0].lower() == "vertex":
                vertices.append([float(value) for value in fields[1:]])
    if not vertices or len(vertices) % 3:
        raise ValueError("invalid ASCII STL")
    return np.asarray(vertices, dtype=np.float64).reshape(-1, 3, 3)


def stl_face_geometry(
    path: str | Path, scale_to_mm: float
) -> tuple[np.ndarray, np.ndarray]:
    source = Path(path)
    try:
        triangles = _binary_stl_triangles(source)
    except ValueError:
        triangles = _ascii_stl_triangles(source)
    triangles *= float(scale_to_mm)
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


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: run_frozen_wss_compat.py SCRIPT [ARG ...]")
    script = Path(sys.argv[1]).resolve()
    rheology.carreau_yasuda_numpy = carreau_yasuda_numpy
    rheology.UDF_PARAMETERS = UDF_PARAMETERS
    wall_shear = types.ModuleType("wss_pinn.physics.wall_shear")
    wall_shear.stl_face_geometry = stl_face_geometry
    sys.modules["wss_pinn.physics.wall_shear"] = wall_shear
    sys.argv = [str(script), *sys.argv[2:]]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
