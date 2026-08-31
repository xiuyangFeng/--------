"""Centerline V2 static geometry and rigid-registration contract.

The historical V4 builder joined two upstream representations: the steady
sidecar stored a physical local radius in millimetres, while the transient CSV
stored ``NormRadius = distance_to_centerline / local_radius``.  This module is
the single source of truth for the replacement route.  Both temporal modes are
mapped from the same Centerline V2 path table and therefore expose the same raw
three-channel geometry schema before train-only normalization.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


GEOMETRY_RAW_NAMES = (
    "abscissa_norm",
    "local_radius_mm",
    "curvature_per_mm",
)
GEOMETRY_MODEL_NAMES = (
    "abscissa_norm",
    "local_radius_mm",
    "curvature_signed_log1p",
)
GEOMETRY_AUX_NAMES = (
    "radial_ratio",
    "centerline_distance_mm",
    "path_id",
    "outlet_id",
)
OUTLET_ORDER = ("out-le", "out-li", "out-ri", "out-re")
LEFT_OUTLETS = frozenset(("out-le", "out-li"))
RIGHT_OUTLETS = frozenset(("out-ri", "out-re"))


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def _unit(vector: np.ndarray, *, name: str) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm <= 1.0e-12:
        raise ValueError(f"cannot normalize degenerate {name}")
    return value / norm


def _common_trunk_junction(frame: pd.DataFrame, tolerance_mm: float = 1.0e-6) -> np.ndarray:
    paths = []
    for path_id in sorted(frame["PathId"].unique().tolist()):
        rows = frame.loc[frame["PathId"] == path_id].sort_values("PointIndex")
        paths.append(rows[["X_mm", "Y_mm", "Z_mm"]].to_numpy(dtype=np.float64))
    if len(paths) != 4:
        raise ValueError(f"Centerline V2 requires four root-to-outlet paths, got {len(paths)}")
    common = -1
    for index in range(min(len(path) for path in paths)):
        points = np.vstack([path[index] for path in paths])
        if float(np.max(np.linalg.norm(points - points[0], axis=1))) > tolerance_mm:
            break
        common = index
    if common < 1:
        raise ValueError("Centerline V2 paths do not contain a stable shared trunk")
    return np.mean(np.vstack([path[common] for path in paths]), axis=0)


def _opening_centers_mm(selection: dict[str, Any]) -> dict[str, np.ndarray]:
    scale = float(selection["authoritative_stl_scale_to_mm"])
    shift = np.asarray(
        selection.get("geometry_translation_to_cfd_wall_mm", [0.0, 0.0, 0.0]),
        dtype=np.float64,
    )
    output: dict[str, np.ndarray] = {}
    for row in selection["opening_assignment"]["openings"]:
        center = np.asarray(row["center_raw"], dtype=np.float64) * scale + shift
        if row["role"] == "inlet":
            output["inlet"] = center
        elif row["role"] == "outlet":
            output[str(row["outlet_name"])] = center
    expected = {"inlet", *OUTLET_ORDER}
    if set(output) != expected:
        raise ValueError(f"opening semantic IDs mismatch: expected {sorted(expected)}, got {sorted(output)}")
    return output


def registration_from_openings(
    junction_mm: np.ndarray,
    openings_mm: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Return ``(centroid_mm, R, diagnostics)`` for the V2 anatomical frame.

    Row-vector convention: ``registered_mm = (raw_mm - centroid_mm) @ R``.
    ``+Z`` points from the aorto-iliac junction to the proximal inlet and ``+X``
    points from the right outlet pair to the patient-left outlet pair.
    """

    centroid = np.asarray(junction_mm, dtype=np.float64)
    e_z = _unit(openings_mm["inlet"] - centroid, name="junction-to-inlet axis")
    left = np.mean(np.vstack([openings_mm[name] for name in LEFT_OUTLETS]), axis=0)
    right = np.mean(np.vstack([openings_mm[name] for name in RIGHT_OUTLETS]), axis=0)
    transverse = left - right
    transverse = transverse - float(transverse @ e_z) * e_z
    e_x = _unit(transverse, name="patient-left axis")
    e_y = _unit(np.cross(e_z, e_x), name="anterior-posterior axis")
    e_x = _unit(np.cross(e_y, e_z), name="orthogonalized patient-left axis")
    rotation = np.column_stack([e_x, e_y, e_z])
    determinant = float(np.linalg.det(rotation))
    if determinant < 0.0:
        e_y = -e_y
        rotation = np.column_stack([e_x, e_y, e_z])
        determinant = float(np.linalg.det(rotation))
    orthogonality = float(np.max(np.abs(rotation.T @ rotation - np.eye(3))))
    if abs(determinant - 1.0) > 1.0e-8 or orthogonality > 1.0e-8:
        raise ValueError("Centerline V2 registration is not a proper orthogonal rotation")

    registered_openings = {
        name: (point - centroid) @ rotation for name, point in openings_mm.items()
    }
    left_x = float(np.mean([registered_openings[name][0] for name in LEFT_OUTLETS]))
    right_x = float(np.mean([registered_openings[name][0] for name in RIGHT_OUTLETS]))
    outlet_z = [float(registered_openings[name][2]) for name in OUTLET_ORDER]
    inlet_z = float(registered_openings["inlet"][2])
    if inlet_z <= 0.0 or max(outlet_z) >= 0.0:
        raise ValueError("opening-based registration does not place inlet at +Z and outlets at -Z")
    if left_x <= right_x:
        raise ValueError("opening-based registration does not place patient-left outlets at +X")
    return centroid, rotation, {
        "determinant": determinant,
        "orthogonality_max_abs": orthogonality,
        "inlet_z_mm": inlet_z,
        "outlet_z_max_mm": max(outlet_z),
        "left_minus_right_x_mm": left_x - right_x,
    }


@dataclass(frozen=True)
class CenterlineV2Geometry:
    canonical_id: str
    case_dir: Path
    coords_mm: np.ndarray
    abscissa_norm: np.ndarray
    radius_mm: np.ndarray
    curvature_per_mm: np.ndarray
    path_id: np.ndarray
    outlet_id: np.ndarray
    junction_mm: np.ndarray
    openings_mm: dict[str, np.ndarray]
    centroid_mm: np.ndarray
    rotation: np.ndarray
    registration_diagnostics: dict[str, float]
    provenance: dict[str, Any]

    @classmethod
    def load(cls, root: str | Path, canonical_id: str) -> "CenterlineV2Geometry":
        case_dir = Path(root) / "cases" / canonical_id
        result_path = case_dir / "result.json"
        selection_path = case_dir / "surface_selection.json"
        paths_path = case_dir / "centerline_paths_mm.csv"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if not bool(result.get("hard_pass")):
            raise ValueError(f"{canonical_id}: Centerline V2 hard Gate did not pass")
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        frame = pd.read_csv(paths_path)
        required = {
            "PathId",
            "OutletId",
            "OutletName",
            "PointIndex",
            "X_mm",
            "Y_mm",
            "Z_mm",
            "Abscissas_mm",
            "MaximumInscribedSphereRadius_mm",
            "Curvature_per_mm",
        }
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise KeyError(f"{canonical_id}: Centerline V2 path CSV missing {missing}")
        if set(frame["OutletName"].unique().tolist()) != set(OUTLET_ORDER):
            raise ValueError(f"{canonical_id}: Centerline V2 outlet names drift")
        shift = np.asarray(
            selection.get("geometry_translation_to_cfd_wall_mm", [0.0, 0.0, 0.0]),
            dtype=np.float64,
        )
        coords = frame[["X_mm", "Y_mm", "Z_mm"]].to_numpy(dtype=np.float64) + shift
        path_length = float(frame.groupby("PathId")["Abscissas_mm"].max().max())
        if path_length <= 0.0:
            raise ValueError(f"{canonical_id}: Centerline V2 path length is degenerate")
        abscissa_norm = frame["Abscissas_mm"].to_numpy(dtype=np.float64) / path_length
        radius = frame["MaximumInscribedSphereRadius_mm"].to_numpy(dtype=np.float64)
        curvature = frame["Curvature_per_mm"].to_numpy(dtype=np.float64)
        if not all(np.isfinite(value).all() for value in (coords, abscissa_norm, radius, curvature)):
            raise ValueError(f"{canonical_id}: Centerline V2 geometry contains NaN/Inf")
        if np.any(radius <= 0.0):
            raise ValueError(f"{canonical_id}: Centerline V2 radius is not positive")
        if float(np.max(np.abs(curvature))) >= 10.0:
            raise ValueError(f"{canonical_id}: Centerline V2 curvature exceeds the hard 10/mm bound")
        local_frame = frame.copy()
        local_frame[["X_mm", "Y_mm", "Z_mm"]] = (
            local_frame[["X_mm", "Y_mm", "Z_mm"]].to_numpy(dtype=np.float64) + shift
        )
        junction = _common_trunk_junction(local_frame)
        openings = _opening_centers_mm(selection)
        centroid, rotation, diagnostics = registration_from_openings(junction, openings)
        return cls(
            canonical_id=canonical_id,
            case_dir=case_dir,
            coords_mm=coords,
            abscissa_norm=abscissa_norm,
            radius_mm=radius,
            curvature_per_mm=curvature,
            path_id=frame["PathId"].to_numpy(dtype=np.int32),
            outlet_id=frame["OutletId"].to_numpy(dtype=np.int32),
            junction_mm=junction,
            openings_mm=openings,
            centroid_mm=centroid,
            rotation=rotation,
            registration_diagnostics=diagnostics,
            provenance={
                "result": {"path": str(result_path.resolve()), "sha256": _sha256(result_path)},
                "surface_selection": {
                    "path": str(selection_path.resolve()),
                    "sha256": _sha256(selection_path),
                },
                "paths": {"path": str(paths_path.resolve()), "sha256": _sha256(paths_path)},
                "status": str(result["status"]),
                "manual_review_required": bool(result.get("manual_review_required")),
                "authoritative_stl": selection["authoritative_stl"],
                "authoritative_stl_sha256": selection["authoritative_stl_sha256"],
                "fluent_to_mm_unit_factor": float(selection["fluent_to_mm_unit_factor"]),
                "geometry_translation_to_cfd_wall_mm": shift.tolist(),
            },
        )

    def registered_unscaled_mm(self, points_mm: np.ndarray) -> np.ndarray:
        return (np.asarray(points_mm, dtype=np.float64) - self.centroid_mm) @ self.rotation

    def register_points(self, points_mm: np.ndarray, scale_mm: float) -> np.ndarray:
        scale = float(scale_mm)
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("coordinate scale must be positive")
        return (self.registered_unscaled_mm(points_mm) / scale).astype(np.float32)

    def rotate_vectors(self, vectors: np.ndarray) -> np.ndarray:
        return (np.asarray(vectors, dtype=np.float64) @ self.rotation).astype(np.float32)

    def map_points(
        self,
        points_mm: np.ndarray,
        *,
        workers: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        points = np.asarray(points_mm, dtype=np.float64)
        distance, index = cKDTree(self.coords_mm).query(points, k=1, workers=max(int(workers), 1))
        index = np.asarray(index, dtype=np.int64)
        raw = np.column_stack(
            [
                self.abscissa_norm[index],
                self.radius_mm[index],
                self.curvature_per_mm[index],
            ]
        ).astype(np.float32)
        radial_ratio = np.asarray(distance, dtype=np.float64) / np.maximum(
            self.radius_mm[index], 1.0e-12
        )
        aux = np.column_stack(
            [
                radial_ratio,
                distance,
                self.path_id[index],
                self.outlet_id[index],
            ]
        ).astype(np.float32)
        if not np.isfinite(raw).all() or not np.isfinite(aux).all():
            raise ValueError(f"{self.canonical_id}: mapped geometry contains NaN/Inf")
        return raw, aux


def transform_geometry(raw: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
    """Apply the single V2 model transform: log curvature, then z-score."""

    if tuple(stats.get("raw_names", GEOMETRY_RAW_NAMES)) != GEOMETRY_RAW_NAMES:
        raise ValueError("geometry raw schema mismatch")
    if tuple(stats.get("names", GEOMETRY_MODEL_NAMES)) != GEOMETRY_MODEL_NAMES:
        raise ValueError("geometry model schema mismatch")
    value = np.asarray(raw, dtype=np.float32).copy()
    value[:, 2] = np.sign(value[:, 2]) * np.log1p(np.abs(value[:, 2]))
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    if mean.shape != (3,) or std.shape != (3,) or np.any(std <= 0.0):
        raise ValueError("geometry stats must contain three positive scales")
    output = (value - mean) / std
    if not np.isfinite(output).all():
        raise ValueError("normalized geometry contains NaN/Inf")
    return output.astype(np.float32)
