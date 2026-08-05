"""Coordinate alignment shared by active volume sidecar build and audit."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def _transform_raw_interior(
    raw_coords: np.ndarray, bundle: dict[str, np.ndarray]
) -> np.ndarray:
    coords_mm = raw_coords * float(bundle["unit_factor"])
    aligned = (coords_mm - bundle["transform_centroid"]) @ bundle["transform_rotation"]
    return aligned / float(bundle["coord_scale"])


def align_raw_interior_to_bundle(
    interior: dict[str, np.ndarray],
    bundle: dict[str, np.ndarray],
    *,
    tolerance: float = 5e-5,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Align raw Fluent rows to the frozen bundle interior coordinates."""
    raw_aligned_norm = _transform_raw_interior(interior["coords"], bundle)
    bundle_int_coords = np.asarray(bundle["int_coords_norm"], dtype=np.float64)
    if len(raw_aligned_norm) == len(bundle_int_coords):
        direct_delta = np.max(np.abs(raw_aligned_norm - bundle_int_coords), axis=1)
        if float(np.max(direct_delta)) <= tolerance:
            return (
                np.arange(len(bundle_int_coords), dtype=np.int64),
                direct_delta,
                "direct_row",
            )

    tree = cKDTree(raw_aligned_norm)
    _, raw_indices = tree.query(bundle_int_coords, k=1, workers=-1)
    raw_indices = np.asarray(raw_indices, dtype=np.int64)
    if len(np.unique(raw_indices)) != len(raw_indices):
        raise ValueError("spatial raw/bundle alignment is not one-to-one")
    coordinate_delta = np.max(
        np.abs(raw_aligned_norm[raw_indices] - bundle_int_coords), axis=1
    )
    if float(np.max(coordinate_delta)) > tolerance:
        raise ValueError(
            "raw/bundle spatial-subset alignment drift: "
            f"max={float(np.max(coordinate_delta)):.3e}"
        )
    return raw_indices, coordinate_delta, "spatial_subset"
