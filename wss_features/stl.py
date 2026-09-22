"""STL reading and writing (binary and ASCII), with vertex de-duplication.

``load_stl`` is a verbatim copy of ``training_wss_min.surface.load_stl``; the
returned ``(vertices, faces)`` pair is bit-identical to the training loader.
"""
from __future__ import annotations

import struct
from pathlib import Path
from typing import Tuple

import numpy as np


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


def write_binary_stl(path: str | Path, vertices: np.ndarray, faces: np.ndarray, header: str = "wss_deploy") -> None:
    """Write a binary STL with recomputed facet normals (little-endian, 80-byte header)."""
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    tri = vertices[faces].astype(np.float32)
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
    rec = np.zeros(len(faces), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
    rec["n"], rec["v"] = normal, tri
    with Path(path).open("wb") as stream:
        stream.write(header.encode("ascii")[:80].ljust(80, b"\0"))
        stream.write(struct.pack("<I", len(faces)))
        stream.write(rec.tobytes())


__all__ = ["load_stl", "write_binary_stl"]
