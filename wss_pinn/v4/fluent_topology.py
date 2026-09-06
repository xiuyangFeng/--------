"""Fluent legacy ``.cas``/``.cas.gz`` topology reader for the anatomy-only V4 rebuild.

The historical parser (``wss_pinn/tools/build_qs_smooth_v3.py``) only reads
boundary face zones.  The 2026-09-03 rebuild contract needs the full cell-zone
topology so that every training point can be tied to an auditable Fluent
cell zone (``blood`` vs. the ``blood1…blood5`` extension segments):

* cell sections (``3012``/``2012``) give the cell-index range of every cell
  zone and, for mixed zones, the per-cell element type;
* every face section (``3013``/``2013``) gives node connectivity and the two
  adjacent cells ``(c0, c1)``;
* zone declarations (``39``/``45``) give the zone type/name.

From these we derive Fluent-compatible cell centroids, wall faces grouped by
the adjacent cell zone, the five ``blood <-> bloodN`` interfaces and the distal
boundary-condition zone that closes each extension segment.

The Fluent ASCII exports (``ascii_in`` cell tables and ``ascii`` wall tables)
renumber cells/nodes sequentially in export order; their ``cellnumber`` /
``nodenumber`` columns are *not* case-file indices (verified 2026-09-03: neither
the case order nor any zone-block permutation reproduces the export order).
``match_points`` therefore ties exported rows to case cells/nodes by coordinate
matching and refuses ambiguous or non-bijective matches.

Cell centroid convention (verified against Fluent's exported cell coordinates
on a tetra/wedge case): face centre = area-weighted centre of the triangle fan
around the vertex mean; cell estimate = mean of face centres; cell centroid =
volume-weighted centre of the face pyramids.  Tetrahedra match to 1e-7 of the
cell size, wedges to <4e-3, which leaves the nearest-neighbour identity
unambiguous (first/second neighbour ratio < 0.01).
"""

from __future__ import annotations

import gzip
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

HEADER_RE = re.compile(rb"\(\s*(\d+)\s*\(([^)]*)\)")
ZONE_DECL_RE = re.compile(rb"\(\s*(39|45)\s*\(\s*(\d+)\s+(\S+)\s+(\S+)")
HEX_RE = re.compile(rb"^[0-9a-fA-F]+$")
END_MARKER = b"End of Binary Section"

# Fluent face-zone (boundary-condition) type codes.
FACE_TYPE_NAMES = {
    2: "interior",
    3: "wall",
    4: "pressure-inlet",
    5: "pressure-outlet",
    7: "symmetry",
    8: "periodic-shadow",
    9: "pressure-far-field",
    10: "velocity-inlet",
    12: "periodic",
    14: "fan",
    20: "mass-flow-inlet",
    24: "interface",
    31: "parent",
    36: "outflow",
    37: "axis",
    40: "unattached-geometry",  # observed: triangle faces with c0 == c1 == 0
}
INLET_FACE_TYPES = frozenset({4, 10, 20})
OUTLET_FACE_TYPES = frozenset({5, 36})
FLOW_BC_FACE_TYPES = INLET_FACE_TYPES | OUTLET_FACE_TYPES
CELL_ELEMENT_NAMES = {
    0: "mixed",
    1: "triangle",
    2: "tetrahedron",
    3: "quadrilateral",
    4: "hexahedron",
    5: "pyramid",
    6: "wedge",
    7: "polyhedron",
}


@dataclass
class FaceSection:
    zone_id: int
    bc_type: int
    element_type: int
    first_index: int
    last_index: int
    # ``offsets`` has ``count + 1`` entries into ``nodes`` (1-based Fluent node ids).
    offsets: np.ndarray
    nodes: np.ndarray
    c0: np.ndarray
    c1: np.ndarray

    @property
    def count(self) -> int:
        return int(len(self.c0))

    @property
    def node_counts(self) -> np.ndarray:
        return np.diff(self.offsets)

    @property
    def attached(self) -> bool:
        """False for faces that bound no cell at all (c0 == c1 == 0)."""
        return bool(np.any(self.c0 > 0) or np.any(self.c1 > 0))

    def subset_connectivity(self, index: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(offsets, nodes)`` for the faces selected by ``index``."""
        index = np.asarray(index, dtype=np.int64)
        counts = self.node_counts[index]
        starts = self.offsets[:-1][index]
        total = int(counts.sum())
        local = np.arange(total) - np.repeat(np.concatenate([[0], np.cumsum(counts)[:-1]]), counts)
        gather = np.repeat(starts, counts) + local
        offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        return offsets, self.nodes[gather]


@dataclass
class FluentMesh:
    path: Path
    nodes_m: np.ndarray  # (N+1, 3); row 0 unused so Fluent 1-based ids index directly
    cell_zone_ranges: dict[int, tuple[int, int]]
    cell_element_types: dict[int, int]
    zone_names: dict[int, tuple[str, str]]
    face_sections: list[FaceSection]
    cell_count: int
    declared_face_count: int
    cell_types: np.ndarray  # per-cell Fluent element type (index 0 unused)
    warnings: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------ basics
    @property
    def node_count(self) -> int:
        return int(len(self.nodes_m) - 1)

    def zone_name(self, zone_id: int) -> str:
        return self.zone_names.get(int(zone_id), ("?", f"zone-{int(zone_id)}"))[1]

    def zone_type_name(self, zone_id: int) -> str:
        return self.zone_names.get(int(zone_id), ("?", ""))[0]

    def cell_zone_array(self) -> np.ndarray:
        """Return ``zone_id`` per cell index (index 0 unused, value 0)."""
        zone = np.zeros(self.cell_count + 1, dtype=np.int32)
        for zone_id, (first, last) in self.cell_zone_ranges.items():
            zone[first : last + 1] = int(zone_id)
        if np.any(zone[1:] == 0):
            raise ValueError(f"{self.path}: some cells belong to no declared cell zone")
        return zone

    def fluid_zone_by_name(self) -> dict[str, int]:
        return {self.zone_name(zone_id): int(zone_id) for zone_id in self.cell_zone_ranges}

    def face_sections_of_type(self, bc_type: int) -> list[FaceSection]:
        return [section for section in self.face_sections if section.bc_type == int(bc_type)]

    def summary(self) -> dict[str, Any]:
        cell_type_counts = {
            CELL_ELEMENT_NAMES.get(int(value), str(int(value))): int(count)
            for value, count in zip(*np.unique(self.cell_types[1:], return_counts=True))
        }
        return {
            "path": str(self.path),
            "cells": int(self.cell_count),
            "nodes": int(self.node_count),
            "faces": int(sum(section.count for section in self.face_sections)),
            "declared_faces": int(self.declared_face_count),
            "cell_types": cell_type_counts,
            "cell_zones": {
                self.zone_name(zone_id): {
                    "zone_id": int(zone_id),
                    "first": int(first),
                    "last": int(last),
                    "count": int(last - first + 1),
                }
                for zone_id, (first, last) in self.cell_zone_ranges.items()
            },
            "face_zones": [
                {
                    "zone_id": int(section.zone_id),
                    "type_code": int(section.bc_type),
                    "type": FACE_TYPE_NAMES.get(section.bc_type, str(section.bc_type)),
                    "declared_type": self.zone_type_name(section.zone_id),
                    "name": self.zone_name(section.zone_id),
                    "faces": int(section.count),
                    "attached": section.attached,
                }
                for section in self.face_sections
            ],
            "warnings": list(self.warnings),
        }

    # --------------------------------------------------------------- geometry
    def face_geometry(self, section: FaceSection) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(centres_m, area_vectors_m2)``.

        Area vectors follow the right-hand rule on the stored node order.  The
        centre is the area-weighted centre of the triangle fan around the
        vertex mean, which is exact for planar polygons and matches Fluent for
        non-planar quads.
        """

        return _face_geometry(self.nodes_m, section.offsets, section.nodes)

    def cell_centroids_and_volumes(self) -> tuple[np.ndarray, np.ndarray]:
        """Fluent-compatible cell centroids and volumes (index 0 is NaN)."""

        count = self.cell_count
        face_cache = []
        centre_sum = np.zeros((count + 1, 3), dtype=np.float64)
        centre_n = np.zeros(count + 1, dtype=np.float64)
        for section in self.face_sections:
            if not _contributes_to_cells(section):
                continue
            centre, area = self.face_geometry(section)
            face_cache.append((section, centre, area))
            for cells in (section.c0, section.c1):
                valid = cells > 0
                if not np.any(valid):
                    continue
                for axis in range(3):
                    centre_sum[:, axis] += np.bincount(
                        cells[valid], weights=centre[valid, axis], minlength=count + 1
                    )
                centre_n += np.bincount(cells[valid], minlength=count + 1)
        if np.any(centre_n[1:] <= 0.0):
            raise ValueError(f"{self.path}: some cells have no faces")
        estimate = centre_sum / np.maximum(centre_n, 1.0)[:, None]

        volume = np.zeros(count + 1, dtype=np.float64)
        moment = np.zeros((count + 1, 3), dtype=np.float64)
        sign0 = self.c0_sign
        for section, centre, area in face_cache:
            for cells, sign in ((section.c0, sign0), (section.c1, -sign0)):
                valid = np.flatnonzero(cells > 0)
                if not len(valid):
                    continue
                cell = cells[valid]
                offset = centre[valid] - estimate[cell]
                pyramid_volume = sign * np.einsum("ij,ij->i", area[valid], offset) / 3.0
                pyramid_centre = 0.75 * centre[valid] + 0.25 * estimate[cell]
                volume += np.bincount(cell, weights=pyramid_volume, minlength=count + 1)
                for axis in range(3):
                    moment[:, axis] += np.bincount(
                        cell,
                        weights=pyramid_volume * pyramid_centre[:, axis],
                        minlength=count + 1,
                    )
        if np.any(volume[1:] <= 0.0):
            bad = int(np.sum(volume[1:] <= 0.0))
            raise ValueError(f"{self.path}: {bad} cells have non-positive signed volume")
        with np.errstate(invalid="ignore", divide="ignore"):
            centroids = moment / volume[:, None]
        centroids[0] = np.nan
        volume[0] = np.nan
        return centroids, volume

    @property
    def c0_sign(self) -> float:
        """+1 if the right-hand-rule face normal points out of ``c0``, else -1."""
        if not hasattr(self, "_c0_sign_cache"):
            self._c0_sign_cache = self._infer_orientation()
        return self._c0_sign_cache

    def _infer_orientation(self) -> float:
        approx = self._vertex_mean_centroids()
        votes_out = 0
        votes_in = 0
        for section in self.face_sections:
            if section.bc_type == 2 or not section.attached or not np.all(section.c1 == 0):
                continue
            centre, area = self.face_geometry(section)
            dots = np.einsum("ij,ij->i", area, centre - approx[section.c0])
            votes_out += int(np.sum(dots > 0.0))
            votes_in += int(np.sum(dots < 0.0))
        total = votes_out + votes_in
        if total == 0:
            raise ValueError(f"{self.path}: no boundary faces to infer orientation")
        if max(votes_out, votes_in) / total < 0.99:
            raise ValueError(
                f"{self.path}: inconsistent face orientation ({votes_out} out / {votes_in} in)"
            )
        return 1.0 if votes_out >= votes_in else -1.0

    def _vertex_mean_centroids(self) -> np.ndarray:
        count = self.cell_count
        vertex_sum = np.zeros((count + 1, 3), dtype=np.float64)
        vertex_weight = np.zeros(count + 1, dtype=np.float64)
        for section in self.face_sections:
            if not _contributes_to_cells(section):
                continue
            node_sum = _segment_sum(self.nodes_m[section.nodes], section.offsets)
            counts = section.node_counts.astype(np.float64)
            for cells in (section.c0, section.c1):
                valid = cells > 0
                if not np.any(valid):
                    continue
                for axis in range(3):
                    vertex_sum[:, axis] += np.bincount(
                        cells[valid], weights=node_sum[valid, axis], minlength=count + 1
                    )
                vertex_weight += np.bincount(cells[valid], weights=counts[valid], minlength=count + 1)
        return vertex_sum / np.maximum(vertex_weight, 1.0e-300)[:, None]


# ---------------------------------------------------------------------------
# geometry helpers


def _contributes_to_cells(section: FaceSection) -> bool:
    """Faces that bound real cells.  ``parent`` faces (31) duplicate children."""
    return section.bc_type != 31 and section.attached


def _segment_sum(values: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    cumulative = np.concatenate([np.zeros((1, values.shape[1])), np.cumsum(values, axis=0)])
    return cumulative[offsets[1:]] - cumulative[offsets[:-1]]


def _face_geometry(
    nodes_m: np.ndarray, offsets: np.ndarray, node_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    points = nodes_m[node_ids]
    counts = np.diff(offsets)
    vertex_mean = _segment_sum(points, offsets) / counts[:, None]
    face_index = np.repeat(np.arange(len(counts)), counts)
    starts = np.repeat(offsets[:-1], counts)
    local = np.arange(len(node_ids)) - starts
    next_ids = node_ids[starts + (local + 1) % np.repeat(counts, counts)]
    a = points - vertex_mean[face_index]
    b = nodes_m[next_ids] - vertex_mean[face_index]
    cross = 0.5 * np.cross(a, b)
    tri_area = np.linalg.norm(cross, axis=1)
    tri_centre = (points + nodes_m[next_ids] + vertex_mean[face_index]) / 3.0
    area_vector = np.zeros((len(counts), 3), dtype=np.float64)
    weighted = np.zeros((len(counts), 3), dtype=np.float64)
    for axis in range(3):
        area_vector[:, axis] = np.bincount(face_index, weights=cross[:, axis], minlength=len(counts))
        weighted[:, axis] = np.bincount(
            face_index, weights=tri_area * tri_centre[:, axis], minlength=len(counts)
        )
    total_area = np.bincount(face_index, weights=tri_area, minlength=len(counts))
    degenerate = total_area <= 1.0e-300
    centre = weighted / np.maximum(total_area, 1.0e-300)[:, None]
    centre[degenerate] = vertex_mean[degenerate]
    return centre, area_vector


# ---------------------------------------------------------------------------
# parsing


def _read_bytes(path: Path) -> bytes:
    if str(path).endswith(".gz"):
        with gzip.open(path, "rb") as handle:
            return handle.read()
    return path.read_bytes()


def _decode_faces(
    buffer: np.ndarray, count: int, element_type: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Decode ``count`` face records; return ``(offsets, nodes, c0, c1)``.

    Fixed-width sections (element type 2/3/4) store ``k`` node ids followed by
    ``c0 c1``.  Mixed (0) and polygon (5) sections store ``k n1..nk c0 c1`` per
    face, where ``k`` is the node count of that face.
    """

    if element_type in (2, 3, 4):
        width = element_type + 2
        if len(buffer) < count * width:
            raise ValueError("truncated fixed-width Fluent face section")
        records = buffer[: count * width].reshape(count, width)
        nodes = records[:, :element_type].reshape(-1).astype(np.int64)
        offsets = np.arange(count + 1, dtype=np.int64) * element_type
        return (
            offsets,
            nodes,
            records[:, element_type].astype(np.int64),
            records[:, element_type + 1].astype(np.int64),
        )
    if element_type not in (0, 5):
        raise ValueError(f"unsupported Fluent face element type {element_type}")
    values = buffer.tolist()
    limit = len(values)
    counts = np.empty(count, dtype=np.int64)
    c0 = np.empty(count, dtype=np.int64)
    c1 = np.empty(count, dtype=np.int64)
    node_chunks: list[list[int]] = []
    position = 0
    for face in range(count):
        if position >= limit:
            raise ValueError("truncated variable-width Fluent face section")
        node_count = values[position]
        if node_count < 3 or node_count > 256:
            raise ValueError(f"invalid Fluent face node count {node_count}")
        position += 1
        node_chunks.append(values[position : position + node_count])
        position += node_count
        c0[face] = values[position]
        c1[face] = values[position + 1]
        position += 2
        counts[face] = node_count
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    nodes = np.fromiter(
        (node for chunk in node_chunks for node in chunk), dtype=np.int64, count=int(offsets[-1])
    )
    return offsets, nodes, c0, c1


def _binary_end(data: bytes, position: int, index: int) -> int:
    found = data.find(END_MARKER, position)
    if found < 0:
        raise ValueError(f"missing end marker for section {index}")
    return found + len(END_MARKER)


def read_fluent_mesh(path: str | Path, *, keep_interior: bool = True) -> FluentMesh:
    """Parse a legacy binary Fluent case (optionally gzip-compressed)."""

    source = Path(path)
    data = _read_bytes(source)
    node_blocks: dict[tuple[int, int], np.ndarray] = {}
    cell_ranges: dict[int, tuple[int, int]] = {}
    cell_elements: dict[int, int] = {}
    cell_type_blocks: dict[tuple[int, int], np.ndarray] = {}
    zone_names: dict[int, tuple[str, str]] = {}
    sections: list[FaceSection] = []
    warnings: list[str] = []
    declared_cells = 0
    declared_faces = 0
    position = 0
    while True:
        start = data.find(b"(", position)
        if start < 0:
            break
        match = HEADER_RE.match(data, start)
        if match is None or (start > 0 and data[start - 1 : start] not in (b"\n", b"\r")):
            position = start + 1
            continue
        index = int(match.group(1))
        fields = match.group(2).split()
        if index in (39, 45):
            decl = ZONE_DECL_RE.match(data, start)
            if decl:
                zone_names[int(decl.group(2))] = (
                    decl.group(3).decode("utf-8", "replace"),
                    decl.group(4).decode("utf-8", "replace"),
                )
            position = match.end()
            continue
        if not fields or not all(HEX_RE.match(value) for value in fields):
            position = start + 1
            continue
        position = match.end()
        header = [int(value, 16) for value in fields]
        if index in (3010, 2010, 10):
            zone_id, first, last = header[:3]
            if zone_id == 0:
                continue
            if index == 10:
                raise ValueError(f"{source}: ASCII node sections are not supported")
            dimension = header[4] if len(header) > 4 else 3
            count = last - first + 1
            dtype = np.dtype("<f8" if index == 3010 else "<f4")
            payload_start = data.index(b"(", position) + 1
            payload = data[payload_start : payload_start + count * dimension * dtype.itemsize]
            node_blocks[(first, last)] = (
                np.frombuffer(payload, dtype=dtype).reshape(count, dimension).astype(np.float64)
            )
            position = _binary_end(data, payload_start + len(payload), index)
            continue
        if index in (3012, 2012, 12):
            zone_id, first, last = header[:3]
            element_type = header[4] if len(header) > 4 else 0
            if zone_id == 0:
                declared_cells = last
                continue
            cell_ranges[zone_id] = (first, last)
            cell_elements[zone_id] = element_type
            if element_type == 0:
                count = last - first + 1
                if index == 12:
                    raise ValueError(f"{source}: ASCII mixed cell sections are not supported")
                payload_start = data.index(b"(", position) + 1
                cell_type_blocks[(first, last)] = np.frombuffer(
                    data[payload_start : payload_start + count * 4], dtype="<i4"
                ).astype(np.int8)
                position = _binary_end(data, payload_start + count * 4, index)
            continue
        if index in (3013, 2013, 13):
            zone_id, first, last, bc_type = header[:4]
            element_type = header[4] if len(header) > 4 else 0
            if zone_id == 0:
                declared_faces = last
                continue
            if index == 13:
                raise ValueError(f"{source}: ASCII face sections are not supported")
            count = last - first + 1
            payload_start = data.index(b"(", position) + 1
            marker_pos = data.find(END_MARKER, payload_start)
            if marker_pos < 0:
                raise ValueError(f"{source}: unterminated face section {zone_id}")
            payload = data[payload_start:marker_pos]
            buffer = np.frombuffer(payload[: len(payload) // 4 * 4], dtype="<i4")
            position = marker_pos + len(END_MARKER)
            if bc_type == 2 and not keep_interior:
                continue
            offsets, face_nodes, c0, c1 = _decode_faces(buffer, count, element_type)
            sections.append(
                FaceSection(
                    zone_id=zone_id,
                    bc_type=bc_type,
                    element_type=element_type,
                    first_index=first,
                    last_index=last,
                    offsets=offsets,
                    nodes=face_nodes,
                    c0=c0,
                    c1=c1,
                )
            )
            continue
    if not node_blocks:
        raise ValueError(f"{source}: no node coordinates found")
    total_nodes = max(last for (_, last) in node_blocks)
    node_array = np.full((total_nodes + 1, 3), np.nan, dtype=np.float64)
    for (first, last), block in node_blocks.items():
        node_array[first : last + 1] = block
    if np.isnan(node_array[1:]).any():
        raise ValueError(f"{source}: node index ranges leave gaps")
    if not cell_ranges:
        raise ValueError(f"{source}: no cell zones found")
    cell_count = max(last for (_, last) in cell_ranges.values())
    if declared_cells and declared_cells != cell_count:
        warnings.append(f"declared cell count {declared_cells} != max cell index {cell_count}")
    covered = sum(last - first + 1 for first, last in cell_ranges.values())
    if covered != cell_count:
        warnings.append(f"cell zone ranges cover {covered} of {cell_count} cells")
    parsed_faces = sum(section.count for section in sections)
    if declared_faces and keep_interior and parsed_faces != declared_faces:
        warnings.append(f"parsed {parsed_faces} faces but header declares {declared_faces}")
    for section in sections:
        if len(section.nodes) and (section.nodes.min() < 1 or section.nodes.max() > total_nodes):
            raise ValueError(f"{source}: face zone {section.zone_id} references nodes out of range")
        if section.c0.max() > cell_count or section.c1.max() > cell_count:
            raise ValueError(f"{source}: face zone {section.zone_id} references cells out of range")
    cell_types = np.zeros(cell_count + 1, dtype=np.int8)
    for zone_id, (first, last) in cell_ranges.items():
        element_type = cell_elements[zone_id]
        if element_type == 0:
            block = cell_type_blocks.get((first, last))
            if block is None or len(block) != last - first + 1:
                raise ValueError(f"{source}: mixed cell zone {zone_id} has no per-cell type payload")
            cell_types[first : last + 1] = block
        else:
            cell_types[first : last + 1] = element_type
    return FluentMesh(
        path=source,
        nodes_m=node_array,
        cell_zone_ranges=cell_ranges,
        cell_element_types=cell_elements,
        zone_names=zone_names,
        face_sections=sections,
        cell_count=cell_count,
        declared_face_count=declared_faces,
        cell_types=cell_types,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# identity matching


def match_points(
    reference_m: np.ndarray,
    query_m: np.ndarray,
    *,
    label: str,
    tolerance_m: float | np.ndarray,
    max_neighbour_ratio: float = 0.1,
    exact_fraction: float = 1.0e-3,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Match every query point to a distinct reference point.

    ``tolerance_m`` may be a scalar or a per-reference array (e.g. a fraction of
    the local cell size).  Every query must fall within tolerance of its nearest
    reference point and be much closer to it than to the second-nearest one,
    unless the competing reference points are themselves coincident (Fluent
    meshes may carry duplicated nodes on seams); coincident candidates are
    assigned greedily so that the final mapping is a bijection onto the used
    reference rows.
    """

    reference = np.asarray(reference_m, dtype=np.float64)
    query = np.asarray(query_m, dtype=np.float64)
    tree = cKDTree(reference)
    k = min(8, len(reference))
    distance, index = tree.query(query, k=k)
    distance = np.atleast_2d(distance).reshape(len(query), k)
    index = np.atleast_2d(index).reshape(len(query), k)
    nearest = distance[:, 0]
    second = distance[:, 1] if k > 1 else np.full(len(query), np.inf)
    chosen = index[:, 0].astype(np.int64)
    tolerance = np.asarray(tolerance_m, dtype=np.float64)
    local_tolerance = tolerance[chosen] if tolerance.ndim else np.full(len(chosen), float(tolerance))
    ratio = nearest / np.maximum(second, 1.0e-300)
    ambiguous = ratio > max_neighbour_ratio
    coincident_resolved = 0
    if np.any(ambiguous):
        # candidates that are coincident with the nearest one (within tolerance)
        used = set()
        order = np.argsort(nearest)
        for row in order:
            if not ambiguous[row]:
                continue
            candidates = [
                int(index[row, j])
                for j in range(k)
                if distance[row, j] <= local_tolerance[row]
                and np.linalg.norm(reference[index[row, j]] - reference[index[row, 0]]) <= local_tolerance[row]
            ]
            if len(candidates) < 2:
                continue  # genuine ambiguity, reported below
            free = [c for c in candidates if c not in used]
            if not free:
                continue
            chosen[row] = free[0]
            used.add(free[0])
            ambiguous[row] = False
            coincident_resolved += 1
    # an (essentially) exact hit is never ambiguous: the competing entity is a
    # distinct point that merely happens to lie within 10x of a ~1e-3 tolerance
    exact = nearest <= exact_fraction * local_tolerance
    ambiguous &= ~exact
    mutual_resolved = 0
    if np.any(ambiguous):
        # tightly packed small cells: accept a mutual nearest-neighbour pair
        # (the reference's nearest query is this row and vice versa)
        rows = np.flatnonzero(ambiguous)
        reverse_distance, reverse_index = cKDTree(query).query(reference[chosen[rows]], k=1)
        mutual = np.asarray(reverse_index, dtype=np.int64) == rows
        ambiguous[rows[mutual]] = False
        mutual_resolved = int(np.sum(mutual))
    unique = np.unique(chosen)
    diagnostics = {
        "label": label,
        "query_rows": int(len(query)),
        "reference_rows": int(len(reference)),
        "max_match_distance_m": float(np.max(nearest)) if len(nearest) else 0.0,
        "p99_match_distance_m": float(np.percentile(nearest, 99)) if len(nearest) else 0.0,
        "max_match_over_tolerance": float(np.max(nearest / np.maximum(local_tolerance, 1.0e-300)))
        if len(nearest)
        else 0.0,
        "max_first_second_neighbour_ratio": float(np.max(ratio)) if len(ratio) else 0.0,
        "coincident_duplicates_resolved": int(coincident_resolved),
        "mutual_nearest_resolved": int(mutual_resolved),
        "unique_reference_rows": int(len(unique)),
        "bijective": bool(len(unique) == len(chosen)),
        "within_tolerance": bool(np.all(nearest <= local_tolerance)),
        "unambiguous": bool(not np.any(ambiguous)),
    }
    if not diagnostics["within_tolerance"]:
        bad = int(np.sum(nearest > local_tolerance))
        raise ValueError(
            f"{label}: {bad} rows exceed the identity tolerance "
            f"(max distance {diagnostics['max_match_distance_m']:.3e} m)"
        )
    if not diagnostics["unambiguous"]:
        raise ValueError(f"{label}: {int(np.sum(ambiguous))} rows are ambiguous between two reference entities")
    if not diagnostics["bijective"]:
        raise ValueError(f"{label}: exported rows do not map one-to-one onto case entities")
    return chosen, diagnostics


# ---------------------------------------------------------------------------
# anatomy topology


@dataclass
class InterfaceZone:
    anatomy_zone_id: int
    extension_zone_id: int
    extension_zone_name: str
    face_section_zone_ids: list[int]
    face_offsets: np.ndarray
    face_nodes: np.ndarray  # 1-based node ids
    node_ids: np.ndarray  # unique 1-based node ids
    anatomy_cells: np.ndarray  # adjacent blood cell per face
    extension_cells: np.ndarray
    centres_m: np.ndarray
    area_vectors_m2: np.ndarray  # oriented from anatomy into the extension
    distal_bc_zone_id: int
    distal_bc_type: int
    distal_bc_name: str

    @property
    def area_m2(self) -> float:
        return float(np.linalg.norm(self.area_vectors_m2, axis=1).sum())

    @property
    def unit_normal(self) -> np.ndarray:
        total = self.area_vectors_m2.sum(axis=0)
        return total / max(float(np.linalg.norm(total)), 1.0e-300)

    @property
    def planarity(self) -> float:
        """|sum of area vectors| / sum of |area vectors|; 1.0 for a planar cut."""
        return float(np.linalg.norm(self.area_vectors_m2.sum(axis=0)) / max(self.area_m2, 1.0e-300))


def anatomy_topology(mesh: FluentMesh, anatomy_zone_name: str = "blood") -> dict[str, Any]:
    """Derive anatomy wall faces, interfaces and extension roles from topology."""

    fluid = mesh.fluid_zone_by_name()
    if anatomy_zone_name not in fluid:
        raise ValueError(f"{mesh.path}: no fluid zone named {anatomy_zone_name!r}")
    anatomy_zone = fluid[anatomy_zone_name]
    cell_zone = mesh.cell_zone_array()

    wall_by_zone: dict[int, dict[str, Any]] = {}
    anatomy_wall_sections: list[tuple[FaceSection, np.ndarray]] = []
    for section in mesh.face_sections_of_type(3):
        if not section.attached:
            continue
        if not np.all(section.c1 == 0):
            raise ValueError(f"{mesh.path}: wall zone {section.zone_id} has two-sided faces")
        adjacent = cell_zone[section.c0]
        zones, counts = np.unique(adjacent, return_counts=True)
        wall_by_zone[int(section.zone_id)] = {
            "name": mesh.zone_name(section.zone_id),
            "faces": int(section.count),
            "adjacent_cell_zones": {
                mesh.zone_name(int(zone)): int(count) for zone, count in zip(zones, counts)
            },
        }
        mask = adjacent == anatomy_zone
        if np.any(mask):
            anatomy_wall_sections.append((section, mask))

    # interfaces: two-sided faces whose cells live in different zones
    pair_faces: dict[tuple[int, int], list[tuple[FaceSection, np.ndarray]]] = {}
    for section in mesh.face_sections:
        if section.bc_type == 3 or section.bc_type in FLOW_BC_FACE_TYPES or not section.attached:
            continue
        two_sided = (section.c0 > 0) & (section.c1 > 0)
        if not np.any(two_sided):
            continue
        z0 = cell_zone[section.c0]
        z1 = cell_zone[section.c1]
        mixed = two_sided & (z0 != z1)
        if not np.any(mixed):
            continue
        pairs = {tuple(sorted(pair)) for pair in zip(z0[mixed].tolist(), z1[mixed].tolist())}
        for a, b in pairs:
            mask = mixed & (((z0 == a) & (z1 == b)) | ((z0 == b) & (z1 == a)))
            pair_faces.setdefault((int(a), int(b)), []).append((section, mask))

    # distal boundary zones per cell zone
    bc_by_cell_zone: dict[int, list[dict[str, Any]]] = {}
    for section in mesh.face_sections:
        if section.bc_type not in FLOW_BC_FACE_TYPES or not section.attached:
            continue
        zones, counts = np.unique(cell_zone[section.c0], return_counts=True)
        for zone_id, count in zip(zones.tolist(), counts.tolist()):
            bc_by_cell_zone.setdefault(int(zone_id), []).append(
                {
                    "zone_id": int(section.zone_id),
                    "type_code": int(section.bc_type),
                    "type": FACE_TYPE_NAMES.get(section.bc_type, str(section.bc_type)),
                    "name": mesh.zone_name(section.zone_id),
                    "faces": int(count),
                }
            )

    interfaces: list[InterfaceZone] = []
    sign0 = mesh.c0_sign
    for (a, b), parts in sorted(pair_faces.items()):
        if anatomy_zone not in (a, b):
            raise ValueError(
                f"{mesh.path}: extension zones {mesh.zone_name(a)} and {mesh.zone_name(b)} touch"
            )
        extension = b if a == anatomy_zone else a
        offsets_parts, nodes_parts, c_an, c_ex, centre_parts, area_parts, zone_ids = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for section, mask in parts:
            idx = np.flatnonzero(mask)
            centre, area = mesh.face_geometry(section)
            z0 = cell_zone[section.c0[idx]]
            anatomy_side_c0 = z0 == anatomy_zone
            sign = np.where(anatomy_side_c0, 1.0, -1.0) * sign0
            area_parts.append(area[idx] * sign[:, None])
            centre_parts.append(centre[idx])
            c_an.append(np.where(anatomy_side_c0, section.c0[idx], section.c1[idx]))
            c_ex.append(np.where(anatomy_side_c0, section.c1[idx], section.c0[idx]))
            sub_offsets, sub_nodes = section.subset_connectivity(idx)
            offsets_parts.append(np.diff(sub_offsets))
            nodes_parts.append(sub_nodes)
            zone_ids.append(int(section.zone_id))
        counts_all = np.concatenate(offsets_parts)
        face_nodes = np.concatenate(nodes_parts)
        bc = bc_by_cell_zone.get(int(extension), [])
        if len(bc) != 1:
            raise ValueError(
                f"{mesh.path}: extension {mesh.zone_name(extension)} closes on {len(bc)} flow BC zones"
            )
        interfaces.append(
            InterfaceZone(
                anatomy_zone_id=int(anatomy_zone),
                extension_zone_id=int(extension),
                extension_zone_name=mesh.zone_name(extension),
                face_section_zone_ids=zone_ids,
                face_offsets=np.concatenate([[0], np.cumsum(counts_all)]).astype(np.int64),
                face_nodes=face_nodes,
                node_ids=np.unique(face_nodes),
                anatomy_cells=np.concatenate(c_an),
                extension_cells=np.concatenate(c_ex),
                centres_m=np.concatenate(centre_parts),
                area_vectors_m2=np.concatenate(area_parts),
                distal_bc_zone_id=bc[0]["zone_id"],
                distal_bc_type=bc[0]["type_code"],
                distal_bc_name=bc[0]["name"],
            )
        )

    return {
        "anatomy_zone_id": int(anatomy_zone),
        "cell_zone": cell_zone,
        "wall_by_zone": wall_by_zone,
        "anatomy_wall_sections": anatomy_wall_sections,
        "interfaces": interfaces,
        "anatomy_direct_flow_bc": bc_by_cell_zone.get(int(anatomy_zone), []),
        "extension_flow_bc": {
            mesh.zone_name(zone): rows for zone, rows in bc_by_cell_zone.items() if zone != anatomy_zone
        },
    }


def anatomy_wall_faces(mesh: FluentMesh, topology: dict[str, Any]) -> dict[str, np.ndarray]:
    """Concatenate all wall faces adjacent to the anatomy zone (outward normals)."""

    counts_parts, nodes_parts, cells, centres, areas, zone_of_face = [], [], [], [], [], []
    sign0 = mesh.c0_sign
    for section, mask in topology["anatomy_wall_sections"]:
        idx = np.flatnonzero(mask)
        centre, area = mesh.face_geometry(section)
        centres.append(centre[idx])
        areas.append(area[idx] * sign0)
        cells.append(section.c0[idx])
        sub_offsets, sub_nodes = section.subset_connectivity(idx)
        counts_parts.append(np.diff(sub_offsets))
        nodes_parts.append(sub_nodes)
        zone_of_face.append(np.full(len(idx), int(section.zone_id), dtype=np.int64))
    if not cells:
        raise ValueError(f"{mesh.path}: anatomy zone has no wall faces")
    counts_all = np.concatenate(counts_parts)
    face_nodes = np.concatenate(nodes_parts)
    return {
        "face_offsets": np.concatenate([[0], np.cumsum(counts_all)]).astype(np.int64),
        "face_nodes": face_nodes,
        "adjacent_cells": np.concatenate(cells),
        "centres_m": np.concatenate(centres),
        "area_vectors_m2": np.concatenate(areas),
        "face_zone_ids": np.concatenate(zone_of_face),
        "node_ids": np.unique(face_nodes),
    }
