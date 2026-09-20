"""Map arbitrary points onto the signed-off 7-segment centerline atlas (true mm frame).

Deployment-side: only needs the atlas table (VMTK centerline + radius) and the
points themselves.  Produces canonical tube coordinates (s, rho, theta) with a
rotation-minimising frame per segment, branch identity and validity flags.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from . import contract as C

POINT_FEATURES = (
    "segment_id", "semantic_id", "atlas_row", "s_local_mm", "s_from_root_mm", "radius_mm",
    "curvature_per_mm", "dr_ds", "dist_to_junction_mm", "dist_to_endpoint_mm",
    "junction_mask", "endpoint_mask", "end_zone", "r_mm", "rho", "theta_rad", "axial_offset_mm",
    "junction_ambiguous",
)


@dataclass
class Atlas:
    table: np.ndarray
    columns: list[str]
    segments: list[dict[str, Any]]
    semantic_of_segment: dict[int, int]
    frame_n: np.ndarray  # (S, 3) rotation-minimising normal per sample
    frame_b: np.ndarray  # (S, 3) binormal
    tree_rows: np.ndarray  # rows used in the KD-tree (child-duplicate samples excluded)
    tree: cKDTree
    provenance: dict[str, Any]

    def col(self, name: str) -> np.ndarray:
        return self.table[:, self.columns.index(name)]

    @property
    def xyz(self) -> np.ndarray:
        return np.stack([self.col("x_mm"), self.col("y_mm"), self.col("z_mm")], axis=1)

    @property
    def tangent(self) -> np.ndarray:
        t = np.stack([self.col("tangent_x"), self.col("tangent_y"), self.col("tangent_z")], axis=1)
        return t / np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1.0e-12)

    def endpoints(self) -> list[dict[str, Any]]:
        """Root start + leaf ends with outward direction and radius (virtual cap recipe)."""
        seg = self.col("segment_id").astype(int)
        idx = self.col("sample_index").astype(int)
        out = []
        for s in self.segments:
            rows = np.flatnonzero(seg == s["segment_id"])
            rows = rows[np.argsort(idx[rows])]
            if s.get("starts_at_root"):
                r = rows[0]
                out.append({"label": "inlet", "atlas_row": int(r), "center_mm": self.xyz[r], "outward": -self.tangent[r],
                            "radius_mm": float(self.col("radius_mm")[r]), "segment_id": int(s["segment_id"])})
            if s.get("ends_at_leaf"):
                r = rows[-1]
                out.append({"label": s.get("outlet_name", ""), "atlas_row": int(r), "center_mm": self.xyz[r], "outward": self.tangent[r],
                            "radius_mm": float(self.col("radius_mm")[r]), "segment_id": int(s["segment_id"])})
        return out


def _semantics(segments: list[dict[str, Any]]) -> dict[int, int]:
    by_id = {int(s["segment_id"]): s for s in segments}
    children: dict[int, list[int]] = {}
    for s in segments:
        children.setdefault(int(s["parent_id"]), []).append(int(s["segment_id"]))

    def leaves(seg_id: int) -> list[str]:
        s = by_id[seg_id]
        if s.get("ends_at_leaf"):
            return [s.get("outlet_name", "")]
        names = []
        for c in children.get(seg_id, []):
            names += leaves(c)
        return names

    out = {}
    for s in segments:
        sid = int(s["segment_id"])
        if s.get("starts_at_root"):
            out[sid] = 0
        elif s.get("ends_at_leaf"):
            out[sid] = C.OUTLET_TO_SEMANTIC.get(s.get("outlet_name", ""), -1)
        else:
            names = set(leaves(sid))
            out[sid] = 1 if names & {"out-le", "out-li"} else (2 if names & {"out-re", "out-ri"} else -1)
    return out


def _rmf(points: np.ndarray, tangents: np.ndarray, r0: np.ndarray) -> np.ndarray:
    """Rotation-minimising frame by double reflection (Wang et al. 2008)."""
    n = len(points)
    r = np.zeros((n, 3))
    r[0] = r0
    for i in range(n - 1):
        v1 = points[i + 1] - points[i]
        c1 = float(v1 @ v1)
        if c1 <= 1.0e-18:
            r[i + 1] = r[i]
            continue
        rl = r[i] - (2.0 / c1) * (v1 @ r[i]) * v1
        tl = tangents[i] - (2.0 / c1) * (v1 @ tangents[i]) * v1
        v2 = tangents[i + 1] - tl
        c2 = float(v2 @ v2)
        r[i + 1] = rl if c2 <= 1.0e-18 else rl - (2.0 / c2) * (v2 @ rl) * v2
        # re-orthogonalise against the stored tangent
        r[i + 1] -= (r[i + 1] @ tangents[i + 1]) * tangents[i + 1]
        r[i + 1] /= max(float(np.linalg.norm(r[i + 1])), 1.0e-12)
    return r


def _perp_to(t: np.ndarray, hint: np.ndarray) -> np.ndarray:
    v = hint - (hint @ t) * t
    if np.linalg.norm(v) < 1.0e-6:
        alt = np.array([0.0, 1.0, 0.0]) if abs(t[0]) > 0.9 else np.array([1.0, 0.0, 0.0])
        v = alt - (alt @ t) * t
    return v / np.linalg.norm(v)


def load_atlas(npz_path: Path, summary_path: Path) -> Atlas:
    z = np.load(npz_path, allow_pickle=True)
    table = np.asarray(z["table"], dtype=np.float64)
    columns = [str(c) for c in z["columns"]]
    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    segments = summary["segments"]
    semantic = _semantics(segments)
    seg = table[:, columns.index("segment_id")].astype(int)
    idx = table[:, columns.index("sample_index")].astype(int)
    xyz = table[:, [columns.index("x_mm"), columns.index("y_mm"), columns.index("z_mm")]]
    tan = table[:, [columns.index("tangent_x"), columns.index("tangent_y"), columns.index("tangent_z")]]
    tan = tan / np.maximum(np.linalg.norm(tan, axis=1, keepdims=True), 1.0e-12)
    frame_n = np.zeros_like(xyz)
    frame_b = np.zeros_like(xyz)
    by_id = {int(s["segment_id"]): s for s in segments}
    order = sorted(by_id, key=lambda k: (by_id[k]["parent_id"] != -1, by_id[k]["s_offset_mm"]))
    end_normal: dict[int, np.ndarray] = {}
    for sid in order:
        rows = np.flatnonzero(seg == sid)
        rows = rows[np.argsort(idx[rows])]
        parent = int(by_id[sid]["parent_id"])
        hint = end_normal.get(parent, np.array([1.0, 0.0, 0.0]))
        r0 = _perp_to(tan[rows[0]], hint)
        r = _rmf(xyz[rows], tan[rows], r0)
        frame_n[rows] = r
        frame_b[rows] = np.cross(tan[rows], r)
        end_normal[sid] = r[-1]
    dup = table[:, columns.index("is_child_duplicate")] > 0.5
    tree_rows = np.flatnonzero(~dup)
    return Atlas(
        table=table, columns=columns, segments=segments, semantic_of_segment=semantic,
        frame_n=frame_n, frame_b=frame_b, tree_rows=tree_rows, tree=cKDTree(xyz[tree_rows]),
        provenance=summary.get("provenance", {}),
    )


def map_points(points_mm: np.ndarray, atlas: Atlas) -> dict[str, np.ndarray]:
    """Nearest-sample tube coordinates for arbitrary points (mm)."""
    pts = np.asarray(points_mm, dtype=np.float64)
    k = min(6, len(atlas.tree_rows))
    dist, nn = atlas.tree.query(pts, k=k)
    dist = np.atleast_2d(dist).reshape(len(pts), k)
    nn = np.atleast_2d(nn).reshape(len(pts), k)
    rows = atlas.tree_rows[nn]
    seg = atlas.col("segment_id").astype(int)
    row0 = rows[:, 0]
    seg0 = seg[row0]
    # ambiguity: another segment's sample almost as close as the nearest one
    other = seg[rows] != seg0[:, None]
    other_dist = np.where(other, dist, np.inf).min(axis=1)
    ambiguous = other_dist <= 1.2 * dist[:, 0]
    xyz = atlas.xyz
    tan = atlas.tangent
    d = pts - xyz[row0]
    t = tan[row0]
    axial = np.einsum("ij,ij->i", d, t)
    radial_vec = d - axial[:, None] * t
    r = np.linalg.norm(radial_vec, axis=1)
    n = atlas.frame_n[row0]
    b = atlas.frame_b[row0]
    theta = np.arctan2(np.einsum("ij,ij->i", radial_vec, b), np.einsum("ij,ij->i", radial_vec, n))
    radius = atlas.col("radius_mm")[row0]
    out = {
        "segment_id": seg0.astype(np.int16),
        "semantic_id": np.array([atlas.semantic_of_segment.get(int(s), -1) for s in seg0], dtype=np.int16),
        "atlas_row": row0.astype(np.int32),
        "s_local_mm": atlas.col("s_local_mm")[row0].astype(np.float32),
        "s_from_root_mm": atlas.col("s_from_root_mm")[row0].astype(np.float32),
        "radius_mm": radius.astype(np.float32),
        "curvature_per_mm": atlas.col("curvature_per_mm")[row0].astype(np.float32),
        "dr_ds": atlas.col("dr_ds")[row0].astype(np.float32),
        "dist_to_junction_mm": atlas.col("dist_to_junction_mm")[row0].astype(np.float32),
        "dist_to_endpoint_mm": atlas.col("dist_to_endpoint_mm")[row0].astype(np.float32),
        "junction_mask": atlas.col("junction_mask")[row0] > 0.5,
        "endpoint_mask": atlas.col("endpoint_mask")[row0] > 0.5,
        "end_zone": atlas.col("end_zone")[row0].astype(np.int8),
        "r_mm": r.astype(np.float32),
        "rho": (r / np.maximum(radius, 1.0e-6)).astype(np.float32),
        "theta_rad": theta.astype(np.float32),
        "axial_offset_mm": axial.astype(np.float32),
        "junction_ambiguous": ambiguous,
    }
    return out


def alignment_metrics(wall_points_mm: np.ndarray, atlas: Atlas) -> dict[str, float]:
    feats = map_points(wall_points_mm, atlas)
    rel = (feats["r_mm"] - feats["radius_mm"]) / np.maximum(feats["radius_mm"], 1.0e-6)
    return {
        "median_rel_gap": float(np.median(rel)),
        "median_abs_rel_gap": float(np.median(np.abs(rel))),
        "p90_abs_rel_gap": float(np.percentile(np.abs(rel), 90)),
        "junction_ambiguous_fraction": float(np.mean(feats["junction_ambiguous"])),
        "semantic_unassigned_fraction": float(np.mean(feats["semantic_id"] < 0)),
    }


def relabel_outlets(atlas: Atlas, interface_centers_mm: dict[str, np.ndarray], *, tol_mm: float = 3.0) -> tuple[Atlas, dict[str, Any]]:
    """Re-derive the leaf-segment outlet names from the geometry (nearest anatomy interface cut).

    The centerline pipeline assigned outlet names in a normalised frame and
    swaps them in a few cases (internal/external or left/right); the mesh
    interface labels are the ones the solver BCs were bound to.  A leaf is
    relabelled only when its endpoint lies within ``tol_mm`` of exactly one
    interface centre and no two leaves claim the same label.
    """
    outlets = {k: np.asarray(v, dtype=np.float64) for k, v in interface_centers_mm.items() if k != "inlet"}
    ends = [e for e in atlas.endpoints() if e["label"] != "inlet"]
    proposal: dict[int, str] = {}
    distances: dict[int, float] = {}
    unresolved: list[dict[str, Any]] = []
    for end in ends:
        if not outlets:
            break
        names = list(outlets)
        d = np.array([np.linalg.norm(end["center_mm"] - outlets[n]) for n in names])
        order = np.argsort(d)
        nearest, second = names[order[0]], (names[order[1]] if len(order) > 1 else None)
        distances[end["segment_id"]] = float(d[order[0]])
        # accept when the nearest cut is within max(tol, 2R) of the endpoint and clearly closer than the runner-up
        limit = max(tol_mm, 2.0 * float(end["radius_mm"]))
        if d[order[0]] <= limit and (second is None or d[order[1]] >= 2.0 * d[order[0]]):
            proposal[end["segment_id"]] = nearest
        else:
            unresolved.append({"segment_id": end["segment_id"], "atlas_label": end["label"], "nearest": nearest, "distance_mm": float(d[order[0]])})
    counts = {}
    for name in proposal.values():
        counts[name] = counts.get(name, 0) + 1
    conflicts = sorted(name for name, n in counts.items() if n > 1)
    if conflicts:
        unresolved.append({"conflict_labels": conflicts})
        proposal = {sid: name for sid, name in proposal.items() if name not in conflicts}
    segments = [dict(seg) for seg in atlas.segments]
    changed = {}
    for seg in segments:
        sid = int(seg["segment_id"])
        if sid in proposal and seg.get("outlet_name") != proposal[sid]:
            changed[sid] = {"atlas": seg.get("outlet_name"), "geometry": proposal[sid], "distance_mm": distances[sid]}
            seg["outlet_name_atlas"] = seg.get("outlet_name")
            seg["outlet_name"] = proposal[sid]
    new_atlas = replace(atlas, segments=segments, semantic_of_segment=_semantics(segments))
    details = {"changed": changed, "unresolved": unresolved, "endpoint_to_interface_mm": distances,
               "all_leaves_resolved": not unresolved and len(proposal) == len(ends)}
    return new_atlas, details
