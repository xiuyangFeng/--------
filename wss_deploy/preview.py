"""Display-only preview mesh (v0.15.8): a connected low-poly surface for the thumbnails and the outlet viewer.

Until v0.15.7 the preview kept every k-th triangle of the cleaned STL (``np.linspace`` over the face list): on a
192 k-face upload that is one triangle in eleven, a scatter of disconnected fragments that the thumbnails drew as an
almost invisible haze.  Vertex clustering on a uniform grid keeps the surface connected: vertices in one cell merge
to their mean, collapsed and duplicate triangles are dropped, and the cell grows until at most ``max_faces``
triangles remain.  Model inference always uses the full cleaned STL; nothing here reaches a prediction.
"""
from __future__ import annotations

import numpy as np

PREVIEW_METHOD = "vertex_cluster"
MAX_PREVIEW_FACES = 18000


def _cluster(vertices: np.ndarray, faces: np.ndarray, cell: float):
    key = np.floor((vertices - vertices.min(0)) / cell).astype(np.int64)
    dims = key.max(0) + 1
    flat = key[:, 0] + dims[0] * (key[:, 1] + dims[1] * key[:, 2])
    _, cluster, counts = np.unique(flat, return_inverse=True, return_counts=True)
    cluster = cluster.reshape(-1)
    tri = cluster[faces]
    keep = (tri[:, 0] != tri[:, 1]) & (tri[:, 1] != tri[:, 2]) & (tri[:, 0] != tri[:, 2])
    tri = tri[keep]
    # one triangle per vertex set, whatever its winding; the first occurrence keeps its orientation
    _, first = np.unique(np.sort(tri, axis=1), axis=0, return_index=True)
    return cluster, counts, tri[np.sort(first)]


def display_mesh(vertices, faces, max_faces: int = MAX_PREVIEW_FACES) -> tuple[np.ndarray, np.ndarray, str]:
    """``(vertices, faces, method)`` with at most ``max_faces`` triangles; small meshes are returned unchanged."""
    v = np.asarray(vertices, np.float64)
    f = np.asarray(faces, np.int64).reshape(-1, 3)
    if len(f) <= max_faces:
        used, inverse = np.unique(f.reshape(-1), return_inverse=True)
        return v[used], inverse.reshape(-1, 3), "full"
    corners = v[f]
    area = float(0.5 * np.linalg.norm(np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1).sum())
    # A grid of cell h leaves roughly area / h² triangles on a smooth surface; grow h until the budget holds.
    cell = max(np.sqrt(area / max_faces), 1e-6)
    for _ in range(40):
        cluster, counts, tri = _cluster(v, f, cell)
        if len(tri) <= max_faces:
            break
        cell *= 1.12
    sums = np.zeros((counts.size, 3))
    np.add.at(sums, cluster, v)
    centres = sums / counts[:, None]
    used, inverse = np.unique(tri.reshape(-1), return_inverse=True)
    return centres[used], inverse.reshape(-1, 3), PREVIEW_METHOD


def preview_payload(vertices, faces, max_faces: int = MAX_PREVIEW_FACES) -> dict:
    """The ``stage_a.json`` ``preview`` block (display only)."""
    verts, tris, method = display_mesh(vertices, faces, max_faces)
    return {"vertices": np.asarray(verts, np.float32).round(3).tolist(), "faces": np.asarray(tris, np.int64).tolist(),
            "display_faces": int(len(tris)), "source_faces": int(len(np.asarray(faces).reshape(-1, 3))),
            "method": method, "display_only": True}


def needs_upgrade(preview) -> bool:
    """A v0.15.7-or-older preview that was a triangle subsample (it hit the 18 000-face budget)."""
    return (isinstance(preview, dict) and preview.get("method") is None
            and int(preview.get("display_faces") or len(preview.get("faces") or [])) >= MAX_PREVIEW_FACES)
