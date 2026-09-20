"""Read the 81 volume and 81 wall ASCII frames into anatomy-ordered arrays with verified identity."""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from wss_pinn.data.raw_io import read_interior, read_wall, step_file
from wss_pinn.v4.fluent_topology import match_points

from . import contract as C
from .mesh_topology import MeshData


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class VolumeFrames:
    source_row: np.ndarray        # (Nv,) row index in the reference ASCII frame
    cell_id_cas: np.ndarray       # (Nv,) 1-based Fluent cell id
    xyz_m: np.ndarray             # (Nv, 3) exported cell-centre coordinates (m)
    velocity: np.ndarray          # (T, Nv, 3) float32 m/s
    pressure: np.ndarray          # (T, Nv) float32 Pa
    identity: dict[str, Any]
    frame_files: list[dict[str, Any]] = field(default_factory=list)
    frames_reordered: list[int] = field(default_factory=list)
    export_rows_total: int = 0
    zone_counts: dict[str, int] = field(default_factory=dict)
    seconds: float = 0.0


@dataclass
class WallFrames:
    valid: np.ndarray             # (Nw,) exported & matched
    source_row: np.ndarray        # (Nw,) row index in the reference wall frame, -1 when missing
    wss_scalar: np.ndarray        # (T, Nw) float32, NaN where invalid
    wss_vector: np.ndarray        # (T, Nw, 3)
    pressure: np.ndarray          # (T, Nw)
    identity: dict[str, Any]
    export_kind: str
    delimiter: str
    frame_files: list[dict[str, Any]] = field(default_factory=list)
    frames_rematched: list[int] = field(default_factory=list)
    duplicate_rows: int = 0
    rows_outside_anatomy: int = 0
    seconds: float = 0.0


def _header(path: Path) -> str:
    with path.open(encoding="utf-8", errors="replace") as handle:
        return handle.readline()


def read_volume_frames(raw_dir: Path, md: MeshData, *, hash_files: bool = True) -> VolumeFrames:
    started = time.time()
    steps = C.EXPECTED_STEPS
    ref_path = step_file(raw_dir, steps[0], "ascii_in")
    ref = read_interior(ref_path)
    if ref["id_column"] != "cellnumber":
        raise ValueError(f"{raw_dir}: volume export is node-based ({ref['id_column']}); V5 requires cell exports")
    cell_size = np.cbrt(md.volumes_m3[1:])
    cell_index, diag = match_points(
        md.centroids_m[1:], ref["coords"], label=f"{raw_dir.name}:cells", tolerance_m=C.CELL_TOLERANCE_FRACTION * cell_size
    )
    export_zone = md.cell_zone[cell_index + 1]
    anatomy_rows = np.flatnonzero(export_zone == md.anatomy_zone_id)
    zone_counts = {md.mesh.zone_name(int(z)): int(n) for z, n in zip(*np.unique(export_zone, return_counts=True))}
    n_v = len(anatomy_rows)
    velocity = np.empty((len(steps), n_v, 3), dtype=np.float32)
    pressure = np.empty((len(steps), n_v), dtype=np.float32)
    ref_coords = ref["coords"]
    ref_ids = ref["cell_id"]
    files = []
    reordered = []
    ref_tree = None
    for t, step in enumerate(steps):
        path = ref_path if step == steps[0] else step_file(raw_dir, step, "ascii_in")
        frame = ref if step == steps[0] else read_interior(path)
        same = len(frame["coords"]) == len(ref_coords) and np.array_equal(frame["cell_id"], ref_ids) and bool(
            np.max(np.abs(frame["coords"] - ref_coords)) <= C.FRAME_COORD_TOL_M
        )
        if same:
            rows = anatomy_rows
        else:
            if ref_tree is None:
                ref_tree = cKDTree(ref_coords)
            dist, perm = ref_tree.query(frame["coords"], k=1)  # frame row -> reference row
            if len(frame["coords"]) != len(ref_coords) or np.max(dist) > C.FRAME_COORD_TOL_M or len(np.unique(perm)) != len(perm):
                raise ValueError(f"{path}: frame rows cannot be identified with the reference frame")
            inverse = np.empty_like(perm)
            inverse[perm] = np.arange(len(perm))  # reference row -> frame row
            rows = inverse[anatomy_rows]
            reordered.append(int(step))
        velocity[t] = frame["velocity"][rows].astype(np.float32)
        pressure[t] = frame["pressure"][rows].astype(np.float32)
        files.append({"step": int(step), "file": path.name, "bytes": path.stat().st_size, "mtime": path.stat().st_mtime,
                      "sha256": sha256_of(path) if hash_files else None})
    return VolumeFrames(
        source_row=anatomy_rows.astype(np.int64),
        cell_id_cas=(cell_index[anatomy_rows] + 1).astype(np.int64),
        xyz_m=ref_coords[anatomy_rows],
        velocity=velocity,
        pressure=pressure,
        identity=diag,
        frame_files=files,
        frames_reordered=reordered,
        export_rows_total=int(len(ref_coords)),
        zone_counts=zone_counts,
        seconds=time.time() - started,
    )


def _match_wall_rows(coords_m: np.ndarray, md: MeshData, label: str) -> tuple[np.ndarray, dict[str, Any], int]:
    """Return (wall_row_per_export_row, diagnostics, duplicate_rows); -1 for rows outside the anatomy wall."""
    unique_coords, inverse = np.unique(np.asarray(coords_m, dtype=np.float64), axis=0, return_inverse=True)
    duplicates = int(len(coords_m) - len(unique_coords))
    tree = cKDTree(md.wall_node_coords_m)
    dist, _ = tree.query(unique_coords, k=1)
    inside = dist <= C.NODE_TOLERANCE_M
    matched_rows = -np.ones(len(unique_coords), dtype=np.int64)
    if np.any(inside):
        index, diag = match_points(md.wall_node_coords_m, unique_coords[inside], label=label, tolerance_m=C.NODE_TOLERANCE_M)
        matched_rows[inside] = index
    else:
        diag = {"label": label, "query_rows": 0}
    diag["rows_outside_anatomy_wall"] = int(np.sum(~inside))
    return matched_rows[inverse], diag, duplicates


def read_wall_frames(raw_dir: Path, md: MeshData, *, hash_files: bool = True) -> WallFrames:
    started = time.time()
    steps = C.EXPECTED_STEPS
    n_w = len(md.wall_node_ids)
    ref_path = step_file(raw_dir, steps[0], "ascii")
    header = _header(ref_path)
    delimiter = "comma" if "," in header else "whitespace"
    export_kind = "node" if "nodenumber" in header else "face-centre"
    if export_kind != "node":
        raise ValueError(f"{raw_dir}: wall export is face-centred; V5 wall contract is node-based")
    ref = read_wall(ref_path)
    ref_rows, diag, duplicates = _match_wall_rows(ref["coords"], md, f"{raw_dir.name}:wall-nodes")
    wss = np.full((len(steps), n_w), np.nan, dtype=np.float32)
    vec = np.full((len(steps), n_w, 3), np.nan, dtype=np.float32)
    prs = np.full((len(steps), n_w), np.nan, dtype=np.float32)
    files = []
    rematched = []
    outside_total = int(np.sum(ref_rows < 0))
    source_row = -np.ones(n_w, dtype=np.int64)
    keep = ref_rows >= 0
    # first occurrence wins for duplicated coordinates
    first = np.full(n_w, -1, dtype=np.int64)
    for row in np.flatnonzero(keep)[::-1]:
        first[ref_rows[row]] = row
    source_row[:] = first
    for t, step in enumerate(steps):
        path = ref_path if step == steps[0] else step_file(raw_dir, step, "ascii")
        frame = ref if step == steps[0] else read_wall(path)
        same = len(frame["coords"]) == len(ref["coords"]) and bool(np.max(np.abs(frame["coords"] - ref["coords"])) <= C.FRAME_COORD_TOL_M)
        if same:
            rows = ref_rows
        else:
            rows, _, _ = _match_wall_rows(frame["coords"], md, f"{raw_dir.name}:wall-nodes@{step}")
            rematched.append(int(step))
        sel = rows >= 0
        target = rows[sel]
        wss[t, target] = frame["wss"][sel].astype(np.float32)
        vec[t, target] = frame["wss_vector"][sel].astype(np.float32)
        prs[t, target] = frame["pressure"][sel].astype(np.float32)
        files.append({"step": int(step), "file": path.name, "bytes": path.stat().st_size, "mtime": path.stat().st_mtime,
                      "sha256": sha256_of(path) if hash_files else None})
    valid = np.isfinite(wss).all(axis=0)
    return WallFrames(
        valid=valid,
        source_row=source_row,
        wss_scalar=wss,
        wss_vector=vec,
        pressure=prs,
        identity=diag,
        export_kind=export_kind,
        delimiter=delimiter,
        frame_files=files,
        frames_rematched=rematched,
        duplicate_rows=duplicates,
        rows_outside_anatomy=outside_total,
        seconds=time.time() - started,
    )
