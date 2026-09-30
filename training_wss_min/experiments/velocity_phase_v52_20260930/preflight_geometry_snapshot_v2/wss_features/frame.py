"""Anatomical coordinate frame of the aorto-iliac atlas.

Verbatim copy of ``wss_v5.views.wss_min_view.anatomical_frame``: origin at the
aortic flow divider, +Z toward the inlet along the trunk chord, +X toward the
left common iliac.  ``p_aligned = rotation @ (p - origin_mm)``.
"""
from __future__ import annotations

from typing import Any

import numpy as np

FRAME_VERSION = "v5_atlas_frame_v1"
PEAK_STEP = 1162


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1.0e-9:
        raise ValueError("degenerate axis")
    return v / n


def anatomical_frame(atlas_table: np.ndarray, columns: list[str], semantic_of_segment: dict[str, int]) -> dict[str, Any]:
    """Origin at the aortic flow divider, +Z toward the inlet along the trunk chord, +X toward the left common iliac."""
    col = lambda n: atlas_table[:, columns.index(n)]
    seg = col("segment_id").astype(int)
    idx = col("sample_index").astype(int)
    xyz = np.stack([col("x_mm"), col("y_mm"), col("z_mm")], axis=1)
    trunk_rows = np.flatnonzero(seg == 0)
    trunk_rows = trunk_rows[np.argsort(idx[trunk_rows])]
    origin = xyz[trunk_rows[-1]]           # trunk end = flow divider
    inlet_pt = xyz[trunk_rows[0]]
    z_axis = _unit(inlet_pt - origin)      # trunk points to +Z, iliac branches to -Z
    sem = {int(k): int(v) for k, v in semantic_of_segment.items()}
    left = [s for s, v in sem.items() if v == 1]
    right = [s for s, v in sem.items() if v == 2]
    source = "left_minus_right_cia"
    if left and right:
        lateral = xyz[np.isin(seg, left)].mean(axis=0) - xyz[np.isin(seg, right)].mean(axis=0)
    else:
        lateral = np.array([1.0, 0.0, 0.0])
        source = "world_x_fallback"
    x_axis = lateral - (lateral @ z_axis) * z_axis
    if np.linalg.norm(x_axis) < 1.0e-6:
        x_axis = np.array([1.0, 0.0, 0.0]) - z_axis[0] * z_axis
        source = "world_x_fallback"
    x_axis = _unit(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    rotation = np.stack([x_axis, y_axis, z_axis], axis=0)  # rows: new axes in old coords -> p_new = R @ (p - origin)
    world_x_agreement = float(np.sign(x_axis[0])) if abs(x_axis[0]) > 1.0e-6 else 0.0
    return {"origin_mm": origin, "rotation": rotation, "x_source": source, "left_axis_world_x_sign": world_x_agreement,
            "trunk_length_mm": float(np.linalg.norm(inlet_pt - origin))}


__all__ = ["FRAME_VERSION", "PEAK_STEP", "anatomical_frame"]
