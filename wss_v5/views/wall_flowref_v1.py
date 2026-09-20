"""Flow-reference geometry sidecar for the ``wss_min_view_v1`` wall point cloud (wave-1 local features).

The frozen view carries tube coordinates whose circumferential angle ``theta`` is
measured in a rotation-minimising frame (RMF).  That angle has no physical
reference across cases: it does not know where the bend is, nor where the flow
divider of the next bifurcation sits.  This pack adds the *physically referenced*
angles and the along-tree context the wave-1 brainstorm asked for:

F1  bend reference   bend_cos      cos(theta - psi_bend), psi_bend = RMF angle of the centreline
                                   curvature vector (0 where kappa*R < BEND_MIN, i.e. straight)
                     bend_signed   (kappa*R) * cos(theta - psi_bend): signed Dean-flow proxy
                     torsion_rr    tau * R (Frenet torsion, smoothed, clipped)
                     bend_cos_up2  same as bend_cos but with the bend direction taken 2 diameters
                     bend_cos_up5  (resp. 5 diameters) upstream along the tree (RMF-transported)
F2  bifurcation      carina_cos    child segments: cos(theta - phi_sibling); +1 = wall facing the
                                   sibling branch (flow-divider / medial side), -1 = lateral hip
                     bif_plane_cos |cos(theta - psi_plane)|: in-plane (0) vs out-of-plane (1) of the
                                   bifurcation, within 3 diameters of the junction (parent + children)
                     bif_angle_cos t_A . t_B of the governing junction (0 far from any junction)
                     sibling_radius_ratio  R_sibling / R_own at the junction (child points only)
F3  upstream history s_over_d      s_from_root / (2 R_inlet)
                     up_min_r_ratio_2d / up_min_r_ratio_5d / up_max_r_ratio_5d
                                   extreme atlas radius in the upstream window (2D / 5D along the
                                   tree) divided by the local radius
                     up_kappa_5d   max kappa*R in the upstream 5D window
F6  Murray prior     log_q_branch_murray  log flow share of the point's segment: trunk 1, common
                                   iliacs 0.5 each (protocol), terminals by Murray R^3 split
                     log_tau0_murray      log_q - 3 log R_mm  (Poiseuille scale, mu dropped)
P5  population prior atlas_prior_logwss  median ln(WSS_peak) of train138 points in the same
                                   (semantic, s_local fraction, theta) bin; leave-one-out for train cases

Everything comes from the deployment inputs (wall point cloud + signed-off atlas)
except ``atlas_prior_logwss`` which is a population table learned from train138
labels only (test cases read the full train table; train cases exclude themselves).

Rows are the ``valid`` wall nodes in master order == rows of the view ``bundle.npz``
(verified against ``wall_node_id_cas`` for every case).

    python -m wss_v5.views.wall_flowref_v1 --workers 12
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from wss_pinn.utils import ROOT, utc_now

from .. import contract as C

PACK_NAME = "wss_min_flowref_v1"
PACK_VERSION = "v1.4"  # v1.4 (2026-09-16): + log_tau0_murray_cap (Murray on virtual-cap radii = the CFD outlet protocol); v1.1: + murray233 / murray_cap variants; v1.2: + log_q_branch_capfit / log_tau0_capfit (train138-fitted cap-area split rule); v1.3: + log_r_over_rdistal (stenosis term inside Murray's log_tau0)
CAPFIT_RULE_PATH = ROOT / "data_wss_v5" / "views" / "wss_min_flowref_v1" / "flow_split_rule_train138.json"
EXTERNAL_OUTLETS = {"out-le", "out-re"}
INTERNAL_OUTLETS = {"out-li", "out-ri"}
VIEW_ROOT = ROOT / "data_wss_v5" / "views" / "wss_min_view_v1"
OUT_ROOT = ROOT / "data_wss_v5" / "views" / PACK_NAME
SPLIT_PATH = VIEW_ROOT / "split_V5_train138_test34.json"
_CAPFIT_RULE_OVERRIDE: Path | None = None  # 2026-09-15: set by main(--capfit-rule); None = CAPFIT_RULE_PATH
BEND_MIN = 0.05            # kappa*R below this is "straight" (radius of curvature > 20 R): no bend side
SMOOTH_HALF = 5            # tangent smoothing half window in atlas samples (0.5 mm spacing)
TORSION_CLIP = 0.2         # 1/mm (tau*R is additionally clipped to +-2 below)
TORSION_SMOOTH_HALF = 10   # wider window for the second derivative
JUNCTION_ZONE_DIAMETERS = 3.0
JUNCTION_MEAN_MM = 10.0    # child tangent/radius averaged over the first 10 mm
TERMINAL_MEAN_MM = 10.0    # terminal radius averaged over the distal 10 mm
PRIOR_S_BINS = 20
PRIOR_THETA_BINS = 12
PRIOR_FLOOR_PA = 0.05
GEOMETRY_KEYS = (
    "bend_cos", "bend_signed", "torsion_rr", "bend_cos_up2", "bend_cos_up5",
    "carina_cos", "bif_plane_cos", "bif_angle_cos", "sibling_radius_ratio",
    "s_over_d", "up_min_r_ratio_2d", "up_min_r_ratio_5d", "up_max_r_ratio_5d", "up_kappa_5d",
    "log_q_branch_murray", "log_tau0_murray",
    "log_q_branch_murray233", "log_q_branch_murray_cap",
    "log_q_branch_capfit", "log_tau0_capfit",
    "log_r_over_rdistal",
    "log_tau0_murray_cap",
)
PRIOR_KEY = "atlas_prior_logwss"


def _unit_rows(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.clip(n, 1e-12, None)


def _smooth_rows(v: np.ndarray, half: int) -> np.ndarray:
    """Moving average along axis 0 with edge clamping (same length)."""
    if half <= 0 or len(v) < 3:
        return v.copy()
    pad = np.concatenate([np.repeat(v[:1], half, axis=0), v, np.repeat(v[-1:], half, axis=0)], axis=0)
    kernel = np.ones(2 * half + 1) / (2 * half + 1)
    return np.stack([np.convolve(pad[:, i], kernel, mode="valid") for i in range(v.shape[1])], axis=1)


def _derivative(v: np.ndarray, s: np.ndarray) -> np.ndarray:
    """d v / d s along an ordered polyline (central differences, robust to duplicate s)."""
    out = np.zeros_like(v)
    if len(v) < 2:
        return out
    ds = np.gradient(s)
    ds = np.where(np.abs(ds) < 1e-9, 1e-9, ds)
    for i in range(v.shape[1]):
        out[:, i] = np.gradient(v[:, i]) / ds
    return out


def _angle_in_frame(vec: np.ndarray, n: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.arctan2(np.einsum("ij,ij->i", vec, b), np.einsum("ij,ij->i", vec, n))


class _Tree:
    """Ordered per-segment atlas samples plus the root-ward chain of every segment."""

    def __init__(self, table: np.ndarray, columns: list[str], segments: list[dict[str, Any]],
                 frame_n: np.ndarray, frame_b: np.ndarray):
        col = lambda name: table[:, columns.index(name)]
        self.seg = col("segment_id").astype(int)
        self.idx = col("sample_index").astype(int)
        self.s_local = col("s_local_mm")
        self.s_root = col("s_from_root_mm")
        self.xyz = np.stack([col("x_mm"), col("y_mm"), col("z_mm")], axis=1)
        self.tangent = _unit_rows(np.stack([col("tangent_x"), col("tangent_y"), col("tangent_z")], axis=1))
        self.kappa = np.clip(col("curvature_per_mm"), 0.0, None)
        self.radius = col("radius_mm")
        self.frame_n = _unit_rows(frame_n.astype(np.float64))
        self.frame_b = _unit_rows(frame_b.astype(np.float64))
        self.segments = {int(s["segment_id"]): s for s in segments}
        self.rows = {}
        for sid in self.segments:
            rows = np.flatnonzero(self.seg == sid)
            self.rows[sid] = rows[np.argsort(self.idx[rows], kind="stable")]
        self.children = {}
        for sid, s in self.segments.items():
            self.children.setdefault(int(s["parent_id"]), []).append(sid)
        # root-ward chain (root first ... own segment last), duplicates kept (harmless)
        self.chain = {}
        for sid in self.segments:
            path = []
            cur = sid
            while cur != -1:
                path.append(cur)
                cur = int(self.segments[cur]["parent_id"])
            self.chain[sid] = np.concatenate([self.rows[c] for c in reversed(path)])
        self.length = {sid: float(self.s_local[self.rows[sid]].max()) for sid in self.segments}
        # smoothed tangent derivatives per segment -> curvature direction, torsion, RMF angle of kappa
        self.kappa_dir = np.zeros_like(self.xyz)
        self.psi_bend = np.zeros(len(table))
        self.torsion = np.zeros(len(table))
        for sid, rows in self.rows.items():
            t = _smooth_rows(self.tangent[rows], SMOOTH_HALF)
            t = _unit_rows(t)
            s = self.s_local[rows]
            dt = _derivative(t, s)
            ddt = _derivative(_smooth_rows(dt, TORSION_SMOOTH_HALF), s)
            # curvature direction: component of dt/ds normal to the (stored) tangent
            tt = self.tangent[rows]
            dt_perp = dt - np.einsum("ij,ij->i", dt, tt)[:, None] * tt
            mag = np.linalg.norm(dt_perp, axis=1)
            direction = np.where((mag > 1e-5)[:, None], dt_perp / np.clip(mag, 1e-12, None)[:, None], 0.0)
            self.kappa_dir[rows] = direction
            self.psi_bend[rows] = _angle_in_frame(direction, self.frame_n[rows], self.frame_b[rows])
            cross = np.cross(t, dt)
            denom = np.einsum("ij,ij->i", cross, cross)
            tau = np.where(denom > 1e-10, np.einsum("ij,ij->i", cross, ddt) / np.clip(denom, 1e-12, None), 0.0)
            self.torsion[rows] = np.clip(tau, -TORSION_CLIP, TORSION_CLIP)

    def upstream_row(self, row: int, back_mm: float) -> int:
        """Atlas row on the root-ward chain at (s_from_root - back_mm), clamped at the inlet."""
        chain = self.chain[self.seg[row]]
        target = self.s_root[row] - back_mm
        s = self.s_root[chain]
        j = int(np.searchsorted(s, target, side="left"))
        return int(chain[min(max(j, 0), len(chain) - 1)])

    def upstream_window(self, row: int, back_mm: float) -> np.ndarray:
        chain = self.chain[self.seg[row]]
        s = self.s_root[chain]
        lo = int(np.searchsorted(s, self.s_root[row] - back_mm, side="left"))
        hi = int(np.searchsorted(s, self.s_root[row], side="right"))
        return chain[max(lo, 0):max(hi, lo + 1)]


def _junction_geometry(tree: _Tree) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    """Return (as_child, as_parent) junction descriptors keyed by segment id.

    A common iliac is the child of the aortic bifurcation *and* the parent of the
    iliac bifurcation, so the two roles are kept in separate tables.
    Angles are RMF angles at the reference sample so they transport along the segment.
    """
    as_child: dict[int, dict[str, Any]] = {}
    as_parent: dict[int, dict[str, Any]] = {}
    for parent, kids in tree.children.items():
        if parent == -1 or len(kids) != 2:
            continue
        info = {}
        for sid in kids:
            rows = tree.rows[sid]
            head = rows[tree.s_local[rows] <= JUNCTION_MEAN_MM]
            head = head if len(head) >= 2 else rows[: min(len(rows), 4)]
            info[sid] = {"t": _unit_rows(tree.tangent[head].mean(axis=0, keepdims=True))[0],
                         "r": float(np.median(tree.radius[head])), "first": int(rows[0])}
        a, b = kids
        t_a, t_b = info[a]["t"], info[b]["t"]
        plane = np.cross(t_a, t_b)
        plane = plane / max(float(np.linalg.norm(plane)), 1e-9)
        angle_cos = float(np.clip(t_a @ t_b, -1.0, 1.0))
        p_rows = tree.rows[parent]
        p_end = int(p_rows[-1])
        t_p = tree.tangent[p_end]
        plane_p = plane - (plane @ t_p) * t_p
        as_parent[parent] = {
            "psi_plane": float(np.arctan2(plane_p @ tree.frame_b[p_end], plane_p @ tree.frame_n[p_end])),
            "angle_cos": angle_cos,
        }
        for sid, sib in ((a, b), (b, a)):
            t_own, t_sib = info[sid]["t"], info[sib]["t"]
            toward = t_sib - (t_sib @ t_own) * t_own
            toward = toward / max(float(np.linalg.norm(toward)), 1e-9)
            first = info[sid]["first"]
            n0, b0 = tree.frame_n[first], tree.frame_b[first]
            plane_c = plane - (plane @ t_own) * t_own
            as_child[sid] = {
                "phi_sibling": float(np.arctan2(toward @ b0, toward @ n0)),
                "psi_plane": float(np.arctan2(plane_c @ b0, plane_c @ n0)),
                "angle_cos": angle_cos,
                "sibling_ratio": float(info[sib]["r"] / max(info[sid]["r"], 1e-6)),
            }
    return as_child, as_parent


def _segment_distal_radius(tree: _Tree, sid: int) -> float:
    """Median atlas radius over the distal TERMINAL_MEAN_MM of a segment (the Murray split's distal radius recipe)."""
    rows = tree.rows[sid]
    tail = rows[tree.s_local[rows] >= tree.length[sid] - TERMINAL_MEAN_MM]
    tail = tail if len(tail) >= 2 else rows[-min(len(rows), 4):]
    return float(np.median(tree.radius[tail]))


def _murray_shares(tree: _Tree, exponent: float = 3.0, leaf_radius: dict[int, float] | None = None) -> dict[int, float]:
    """Flow share per segment: root 1, root children 0.5 each (protocol), deeper levels by R^exponent.

    ``leaf_radius`` optionally overrides the distal radius of leaf segments (e.g. virtual-cap radii).
    """
    shares = {}
    roots = [sid for sid, s in tree.segments.items() if s.get("starts_at_root") or int(s["parent_id"]) == -1]
    for root in roots:
        shares[root] = 1.0

    def distal_radius(sid: int) -> float:
        if leaf_radius is not None and sid in leaf_radius:
            return float(leaf_radius[sid])
        return _segment_distal_radius(tree, sid)

    def visit(sid: int, depth: int) -> None:
        kids = tree.children.get(sid, [])
        if not kids:
            return
        if depth == 0 and len(kids) == 2:
            for k in kids:
                shares[k] = shares[sid] * 0.5
        else:
            r3 = np.array([distal_radius(k) ** float(exponent) for k in kids])
            for k, w in zip(kids, r3 / max(float(r3.sum()), 1e-12)):
                shares[k] = shares[sid] * float(w)
        for k in kids:
            visit(k, depth + 1)

    for root in roots:
        visit(root, 0)
    return shares


def _capfit_shares(segments: dict[int, dict[str, Any]], children: dict[int, list[int]], cap_by_leaf: dict[int, float],
                   rule: dict[str, float]) -> tuple[dict[int, float], dict[str, Any]]:
    """Flow share per segment with the train138-fitted cap-area rule for external/internal iliac pairs.

    root 1, root children 0.5 each (protocol); a pair of leaf children that are one external and one internal
    outlet with finite cap radii splits by ratio = exp(a * log((r_ext/r_int)^2) + b); any other split falls back
    to Murray with exponent 2a on the cap radii (or distal radii when caps are missing) and is recorded in notes.
    """
    a, b = float(rule["a"]), float(rule["b"])
    shares: dict[int, float] = {}
    notes: dict[str, Any] = {"fallback_nodes": []}
    roots = [sid for sid, s in segments.items() if s.get("starts_at_root") or int(s["parent_id"]) == -1]
    for root in roots:
        shares[root] = 1.0

    def visit(sid: int, depth: int) -> None:
        kids = children.get(sid, [])
        if not kids:
            return
        if depth == 0 and len(kids) == 2:
            for k in kids:
                shares[k] = shares[sid] * 0.5
        else:
            ext = [k for k in kids if segments[k].get("outlet_name", "") in EXTERNAL_OUTLETS]
            inte = [k for k in kids if segments[k].get("outlet_name", "") in INTERNAL_OUTLETS]
            if len(kids) == 2 and len(ext) == 1 and len(inte) == 1 and ext[0] in cap_by_leaf and inte[0] in cap_by_leaf:
                ratio = float(np.exp(a * np.log((cap_by_leaf[ext[0]] / cap_by_leaf[inte[0]]) ** 2) + b))
                shares[ext[0]] = shares[sid] * ratio / (1.0 + ratio)
                shares[inte[0]] = shares[sid] / (1.0 + ratio)
            else:
                notes["fallback_nodes"].append(int(sid))
                w = np.array([float(cap_by_leaf.get(k, np.nan)) ** (2.0 * a) for k in kids])
                if not np.isfinite(w).all() or w.sum() <= 0:
                    w = np.ones(len(kids))
                for k, wk in zip(kids, w / w.sum()):
                    shares[k] = shares[sid] * float(wk)
        for k in kids:
            visit(k, depth + 1)

    for root in roots:
        visit(root, 0)
    return shares, notes


def _load_capfit_rule(path: Path = CAPFIT_RULE_PATH) -> dict[str, float]:
    if not path.is_file():
        raise FileNotFoundError(f"cap-area split rule missing: {path} (run training_wss_min.tools.fit_flow_split_rule)")
    rule = json.loads(path.read_text())
    if not all(np.isfinite(float(rule[k])) for k in ("a", "b")):
        raise ValueError(f"invalid cap-area split rule in {path}")
    return {"a": float(rule["a"]), "b": float(rule["b"])}


def compute_point_features(tree: _Tree, cap_labels: list[str], cap_radius: np.ndarray, atlas_row: np.ndarray,
                           seg_pt: np.ndarray, sem_pt: np.ndarray, s_local_pt: np.ndarray, s_root_pt: np.ndarray,
                           theta_pt: np.ndarray, radius_pt: np.ndarray, dist_junction_pt: np.ndarray) -> tuple[dict[str, Any], dict[str, Any]]:
    """Flow-reference features for arbitrary wall points already projected onto the atlas (deployment path).

    Returns (payload, extra): payload = the ``wall_*`` feature arrays (+ ``wall_prior_bin``), extra = report fields.
    Label-free: the population prior sums are computed by the caller from the peak WSS when available.
    """
    atlas_row = np.asarray(atlas_row, dtype=np.int64)
    seg_pt = np.asarray(seg_pt, dtype=int)
    sem_pt = np.asarray(sem_pt, dtype=int)
    s_local_pt = np.asarray(s_local_pt, dtype=np.float64)
    s_root_pt = np.asarray(s_root_pt, dtype=np.float64)
    theta_pt = np.asarray(theta_pt, dtype=np.float64)
    radius_pt = np.asarray(radius_pt, dtype=np.float64)
    dist_junction_pt = np.asarray(dist_junction_pt, dtype=np.float64)
    n = len(seg_pt)
    radius_pt = np.clip(radius_pt, 1e-3, None)
    diameter_pt = 2.0 * radius_pt

    # ---- F1: bend reference (RMF-transported curvature direction) ----
    kappa_r = tree.kappa[atlas_row] * radius_pt
    bendy = kappa_r >= BEND_MIN
    bend_cos = np.where(bendy, np.cos(theta_pt - tree.psi_bend[atlas_row]), 0.0)
    bend_signed = kappa_r * np.cos(theta_pt - tree.psi_bend[atlas_row])
    torsion_rr = np.clip(tree.torsion[atlas_row] * radius_pt, -2.0, 2.0)
    unique_rows, inverse = np.unique(atlas_row, return_inverse=True)
    lag = {}
    for name, factor in (("up2", 2.0), ("up5", 5.0)):
        lag_rows = np.empty(len(unique_rows), dtype=np.int64)
        for j, row in enumerate(unique_rows):
            lag_rows[j] = tree.upstream_row(int(row), factor * 2.0 * float(tree.radius[row]))
        lag[name] = lag_rows[inverse]
    def lagged_bend(rows):
        kr = tree.kappa[rows] * tree.radius[rows]
        return np.where(kr >= BEND_MIN, np.cos(theta_pt - tree.psi_bend[rows]), 0.0)
    bend_cos_up2 = lagged_bend(lag["up2"])
    bend_cos_up5 = lagged_bend(lag["up5"])

    # ---- F2: bifurcation reference ----
    as_child, as_parent = _junction_geometry(tree)
    carina_cos = np.zeros(n)
    bif_plane_cos = np.zeros(n)
    bif_angle_cos = np.zeros(n)
    sibling_ratio = np.zeros(n)
    near_junction = dist_junction_pt <= JUNCTION_ZONE_DIAMETERS * diameter_pt
    seg_len_pt = np.array([tree.length.get(int(s), 0.0) for s in seg_pt])
    near_start = s_local_pt <= JUNCTION_ZONE_DIAMETERS * diameter_pt          # proximal junction zone
    near_end = (seg_len_pt - s_local_pt) <= JUNCTION_ZONE_DIAMETERS * diameter_pt  # distal junction zone
    for sid, info in as_child.items():
        mask = seg_pt == sid
        if not mask.any():
            continue
        carina_cos[mask] = np.cos(theta_pt[mask] - info["phi_sibling"])
        sibling_ratio[mask] = info["sibling_ratio"]
        bif_angle_cos[mask] = info["angle_cos"]
        zone = mask & near_start
        bif_plane_cos[zone] = np.abs(np.cos(theta_pt[zone] - info["psi_plane"]))
    for sid, info in as_parent.items():
        zone = (seg_pt == sid) & near_end
        if not zone.any():
            continue
        bif_plane_cos[zone] = np.abs(np.cos(theta_pt[zone] - info["psi_plane"]))
        bif_angle_cos[zone] = info["angle_cos"]

    # ---- F3: upstream history along the tree ----
    inlet_label = cap_labels.index("inlet") if "inlet" in cap_labels else None
    root_rows = next(rows for sid, rows in tree.rows.items() if int(tree.segments[sid]["parent_id"]) == -1)
    r_inlet = float(np.median(tree.radius[root_rows[tree.s_local[root_rows] <= JUNCTION_MEAN_MM]]))
    if inlet_label is not None and np.isfinite(cap_radius[inlet_label]) and cap_radius[inlet_label] > 0:
        r_inlet = float(cap_radius[inlet_label])
    s_over_d = s_root_pt / (2.0 * max(r_inlet, 1e-3))
    win_min2 = np.empty(len(unique_rows)); win_min5 = np.empty(len(unique_rows))
    win_max5 = np.empty(len(unique_rows)); win_kap5 = np.empty(len(unique_rows))
    for j, row in enumerate(unique_rows):
        row = int(row)
        d = 2.0 * float(tree.radius[row])
        w2 = tree.upstream_window(row, 2.0 * d)
        w5 = tree.upstream_window(row, 5.0 * d)
        win_min2[j] = tree.radius[w2].min()
        win_min5[j] = tree.radius[w5].min()
        win_max5[j] = tree.radius[w5].max()
        win_kap5[j] = (tree.kappa[w5] * tree.radius[w5]).max()
    up_min_r_ratio_2d = win_min2[inverse] / radius_pt
    up_min_r_ratio_5d = win_min5[inverse] / radius_pt
    up_max_r_ratio_5d = win_max5[inverse] / radius_pt
    up_kappa_5d = win_kap5[inverse]

    # ---- F6: Murray flow-share prior ----
    shares = _murray_shares(tree)
    log_q = np.array([np.log(max(shares.get(int(s), 1.0), 1e-6)) for s in seg_pt])
    log_tau0 = log_q - 3.0 * np.log(radius_pt)
    # variants (v1.1): Murray exponent 2.33; leaf radii from the virtual caps of the geometry program
    shares233 = _murray_shares(tree, exponent=2.33)
    cap_by_leaf = {}
    for sid, segment in tree.segments.items():
        name = segment.get("outlet_name", "")
        if segment.get("ends_at_leaf") and name in cap_labels and np.isfinite(cap_radius[cap_labels.index(name)]) \
                and cap_radius[cap_labels.index(name)] > 0:
            cap_by_leaf[sid] = float(cap_radius[cap_labels.index(name)])
    shares_cap = _murray_shares(tree, exponent=3.0, leaf_radius=cap_by_leaf)
    log_q233 = np.array([np.log(max(shares233.get(int(s), 1.0), 1e-6)) for s in seg_pt])
    log_q_cap = np.array([np.log(max(shares_cap.get(int(s), 1.0), 1e-6)) for s in seg_pt])
    # v1.4: protocol-cap variant of the Poiseuille scale (the CFD RCR protocol splits flow by Murray on outlet-face radii)
    log_tau0_cap = log_q_cap - 3.0 * np.log(radius_pt)
    # v1.2: train138-fitted cap-area split rule (external/internal iliac), root and common iliac stay protocol
    capfit_rule = _load_capfit_rule(_CAPFIT_RULE_OVERRIDE or CAPFIT_RULE_PATH)
    shares_capfit, capfit_notes = _capfit_shares(tree.segments, tree.children, cap_by_leaf, capfit_rule)
    log_q_capfit = np.array([np.log(max(shares_capfit.get(int(s), 1.0), 1e-6)) for s in seg_pt])
    log_tau0_capfit = log_q_capfit - 3.0 * np.log(radius_pt)
    # v1.3: stenosis index = ln(R_local / R_distal(segment)).  Murray's log_tau0 = log_q - 3 ln R carries
    # 3 ln(R_distal / R_local) implicitly through the R_distal^3 split; wave 4 showed that term (not the
    # flow split) is what X5 uses in ILO.  Exposed on its own so a split rule can be swapped without losing it.
    distal_by_segment = {int(sid): _segment_distal_radius(tree, int(sid)) for sid in tree.segments}
    log_r_over_rdistal = np.log(radius_pt) - np.log(np.array([max(distal_by_segment[int(s)], 1e-3) for s in seg_pt]))

    # ---- P5: population prior bins (semantic x s_local fraction x theta) ----
    seg_len = np.array([max(tree.length.get(int(s), 1.0), 1e-6) for s in seg_pt])
    s_bin = np.clip((s_local_pt / seg_len * PRIOR_S_BINS).astype(int), 0, PRIOR_S_BINS - 1)
    th_bin = np.clip(((theta_pt + np.pi) / (2 * np.pi) * PRIOR_THETA_BINS).astype(int), 0, PRIOR_THETA_BINS - 1)
    sem = np.clip(sem_pt, 0, 6)
    prior_bin = (sem * PRIOR_S_BINS + s_bin) * PRIOR_THETA_BINS + th_bin

    payload = {
        "wall_bend_cos": bend_cos, "wall_bend_signed": bend_signed, "wall_torsion_rr": torsion_rr,
        "wall_bend_cos_up2": bend_cos_up2, "wall_bend_cos_up5": bend_cos_up5,
        "wall_carina_cos": carina_cos, "wall_bif_plane_cos": bif_plane_cos,
        "wall_bif_angle_cos": bif_angle_cos, "wall_sibling_radius_ratio": sibling_ratio,
        "wall_s_over_d": s_over_d, "wall_up_min_r_ratio_2d": up_min_r_ratio_2d,
        "wall_up_min_r_ratio_5d": up_min_r_ratio_5d, "wall_up_max_r_ratio_5d": up_max_r_ratio_5d,
        "wall_up_kappa_5d": up_kappa_5d,
        "wall_log_q_branch_murray": log_q, "wall_log_tau0_murray": log_tau0,
        "wall_log_q_branch_murray233": log_q233, "wall_log_q_branch_murray_cap": log_q_cap,
        "wall_log_q_branch_capfit": log_q_capfit, "wall_log_tau0_capfit": log_tau0_capfit,
        "wall_log_r_over_rdistal": log_r_over_rdistal,
        "wall_log_tau0_murray_cap": log_tau0_cap,
        "wall_prior_bin": prior_bin.astype(np.int32),
    }
    extra = {
        "bendy_fraction": float(bendy.mean()),
        "bend_cos_abs_median_bendy": float(np.median(np.abs(bend_cos[bendy]))) if bendy.any() else 0.0,
        "junctions_as_child": {str(k): v for k, v in as_child.items()},
        "junctions_as_parent": {str(k): v for k, v in as_parent.items()},
        "murray_shares": {str(k): v for k, v in shares.items()},
        "murray_shares_exp233": {str(k): v for k, v in shares233.items()},
        "murray_shares_cap": {str(k): v for k, v in shares_cap.items()},
        "cap_radius_by_leaf": {str(k): v for k, v in cap_by_leaf.items()},
        "capfit_shares": {str(k): v for k, v in shares_capfit.items()}, "capfit_rule": capfit_rule,
        "capfit_notes": capfit_notes,
        "distal_radius_by_segment": {str(k): v for k, v in distal_by_segment.items()},
        "log_r_over_rdistal_p10_p50_p90": [float(x) for x in np.percentile(log_r_over_rdistal, [10, 50, 90])],
        "r_inlet_mm": r_inlet,
        "carina_cos_child_fraction": float((carina_cos != 0).mean()),
        "junction_zone_fraction": float(near_junction.mean()),
        "up_min_r_ratio_5d_p10_p50_p90": [float(x) for x in np.percentile(up_min_r_ratio_5d, [10, 50, 90])],
        "torsion_rr_abs_p90": float(np.percentile(np.abs(torsion_rr), 90)),
    }
    return payload, extra


def build_geometry(canonical_id: str, out_root: Path = OUT_ROOT, view_root: Path = VIEW_ROOT) -> dict[str, Any]:
    """Pass 1: geometry-only features + prior bin ids + this case's bin sums (for the LOO prior)."""
    started = time.time()
    bundle_path = Path(view_root) / canonical_id / "bundle.npz"
    with np.load(bundle_path, allow_pickle=True) as bundle:
        bundle_node_id = bundle["wall_node_id_cas"].astype(np.int64)
        steps = bundle["steps"].tolist()
        peak = int(bundle["peak_step"])
        wss_peak = bundle["wall_wss"][steps.index(peak)].astype(np.float64)
    with h5py.File(C.case_dir(canonical_id) / "case.h5", "r") as h5:
        g = h5["geometry"]
        table = g["atlas_table"][()]
        columns = json.loads(g.attrs["atlas_columns"])
        segments = json.loads(g.attrs["atlas_segments"])
        frame_n = g["atlas_frame_n"][()]
        frame_b = g["atlas_frame_b"][()]
        cap_labels = json.loads(g.attrs["cap_labels"])
        cap_radius = g["cap_radius_mm"][()]
        ws = h5["wall_static"]
        keep = ws["valid"][()].astype(bool)
        node_id = ws["node_id_cas"][()][keep].astype(np.int64)
        atlas_row = ws["atlas_row"][()][keep].astype(np.int64)
        seg_pt = ws["segment_id"][()][keep].astype(int)
        sem_pt = ws["semantic_id"][()][keep].astype(int)
        s_local_pt = ws["s_local_mm"][()][keep].astype(np.float64)
        s_root_pt = ws["s_from_root_mm"][()][keep].astype(np.float64)
        theta_pt = ws["theta_rad"][()][keep].astype(np.float64)
        radius_pt = ws["radius_mm"][()][keep].astype(np.float64)
        dist_junction_pt = ws["dist_to_junction_mm"][()][keep].astype(np.float64)
    if not np.array_equal(node_id, bundle_node_id):
        raise ValueError(f"{canonical_id}: sidecar rows do not match the frozen view rows")
    if np.any(atlas_row < 0) or np.any(atlas_row >= len(table)):
        raise ValueError(f"{canonical_id}: invalid atlas projection")
    tree = _Tree(table, columns, segments, frame_n, frame_b)
    n = len(node_id)
    payload, extra = compute_point_features(tree, cap_labels, cap_radius, atlas_row, seg_pt, sem_pt, s_local_pt, s_root_pt,
                                            theta_pt, radius_pt, dist_junction_pt)
    payload = {"wall_node_id_cas": node_id, **payload}
    prior_bin = payload["wall_prior_bin"].astype(np.int64)
    log_wss = np.log(np.clip(wss_peak, PRIOR_FLOOR_PA, None))
    n_bins = 7 * PRIOR_S_BINS * PRIOR_THETA_BINS
    bin_sum = np.bincount(prior_bin, weights=log_wss, minlength=n_bins)
    bin_cnt = np.bincount(prior_bin, minlength=n_bins).astype(np.float64)
    for key, value in list(payload.items()):
        if key in {"wall_node_id_cas", "wall_prior_bin"}:
            continue
        value = np.asarray(value, dtype=np.float64)
        if not np.isfinite(value).all():
            raise ValueError(f"{canonical_id}: non-finite values in {key}")
        payload[key] = value.astype(np.float32)
    out_dir = out_root / canonical_id
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / "features.tmp.npz"
    np.savez(tmp, **payload)
    tmp.replace(out_dir / "features.npz")
    np.savez(out_dir / "prior_bins.npz", bin_sum=bin_sum, bin_cnt=bin_cnt)
    report = {
        "canonical_id": canonical_id, "pack": PACK_NAME, "version": PACK_VERSION,
        "n_points": int(n), "seconds": round(time.time() - started, 2),
        **extra,
    }
    (out_dir / "flowref_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report


def build_prior(cases: list[str], out_root: Path = OUT_ROOT, split_path: Path = SPLIT_PATH) -> dict[str, Any]:
    """Pass 2: leave-one-out population prior for train138, full-train table for everyone else."""
    split = json.loads(Path(split_path).read_text(encoding="utf-8"))
    train = set(split["train_cases"])
    n_bins = 7 * PRIOR_S_BINS * PRIOR_THETA_BINS
    sums = {}
    cnts = {}
    for cid in cases:
        with np.load(out_root / cid / "prior_bins.npz") as z:
            sums[cid] = z["bin_sum"].astype(np.float64)
            cnts[cid] = z["bin_cnt"].astype(np.float64)
    total_sum = sum(sums[c] for c in cases if c in train)
    total_cnt = sum(cnts[c] for c in cases if c in train)
    if len([c for c in cases if c in train]) != len(train):
        raise ValueError(f"prior needs every train case built: have {len([c for c in cases if c in train])} of {len(train)}")

    def table(sum_b: np.ndarray, cnt_b: np.ndarray) -> np.ndarray:
        """Fine-bin mean with hierarchical fallback (semantic x s, semantic, global)."""
        fine = np.where(cnt_b > 0, sum_b / np.clip(cnt_b, 1, None), np.nan)
        s3 = sum_b.reshape(7, PRIOR_S_BINS, PRIOR_THETA_BINS)
        c3 = cnt_b.reshape(7, PRIOR_S_BINS, PRIOR_THETA_BINS)
        mid = np.where(c3.sum(2) > 0, s3.sum(2) / np.clip(c3.sum(2), 1, None), np.nan)
        coarse = np.where(c3.sum((1, 2)) > 0, s3.sum((1, 2)) / np.clip(c3.sum((1, 2)), 1, None), np.nan)
        overall = float(sum_b.sum() / max(cnt_b.sum(), 1.0))
        mid = np.where(np.isfinite(mid), mid, np.where(np.isfinite(coarse), coarse, overall)[:, None])
        fine3 = fine.reshape(7, PRIOR_S_BINS, PRIOR_THETA_BINS)
        fine3 = np.where(np.isfinite(fine3), fine3, mid[:, :, None])
        return fine3.reshape(-1)

    full = table(total_sum, total_cnt)
    stats = {"train_cases": len(train), "bins_total": int(n_bins), "bins_populated_full": int((total_cnt > 0).sum())}
    loo_bins_empty = []
    for cid in cases:
        if cid in train:
            values = table(total_sum - sums[cid], total_cnt - cnts[cid])
            loo_bins_empty.append(int(((total_cnt - cnts[cid]) <= 0).sum()))
        else:
            values = full
        path = out_root / cid / "features.npz"
        with np.load(path) as z:
            payload = {k: z[k] for k in z.files}
        payload[f"wall_{PRIOR_KEY}"] = values[payload["wall_prior_bin"].astype(np.int64)].astype(np.float32)
        payload["wall_prior_is_loo"] = np.bool_(cid in train)
        tmp = out_root / cid / "features.tmp.npz"
        np.savez(tmp, **payload)
        tmp.replace(path)
    stats["loo_empty_bins_max"] = int(max(loo_bins_empty)) if loo_bins_empty else 0
    return stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT_ROOT))
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--cases", nargs="*", default=None)
    ap.add_argument("--skip-prior", action="store_true")
    ap.add_argument("--view-root", default=str(VIEW_ROOT), help="wss_min_view bundles root (2026-09-15: v5.1 refresh)")
    ap.add_argument("--split", default=str(SPLIT_PATH), help="split json for the leave-one-out prior (train partition)")
    ap.add_argument("--capfit-rule", default=str(CAPFIT_RULE_PATH), help="cap-area split rule json (fit_flow_split_rule)")
    args = ap.parse_args(argv)
    global _CAPFIT_RULE_OVERRIDE
    _CAPFIT_RULE_OVERRIDE = Path(args.capfit_rule)
    out_root = Path(args.out)
    view_root = Path(args.view_root)
    split_path = Path(args.split)
    cases = args.cases or sorted(
        str(p.parent.relative_to(view_root)) for p in view_root.glob("*/*/*/bundle.npz")
    )
    started = time.time()
    reports: list[dict[str, Any]] = []
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(build_geometry, cid, out_root, view_root) for cid in cases]
            for future in as_completed(futures):
                report = future.result()
                reports.append(report)
                print(f"[wall_flowref_v1] {len(reports)}/{len(cases)} {report['canonical_id']} "
                      f"n={report['n_points']} bendy={report['bendy_fraction']:.2f}", flush=True)
    else:
        for i, cid in enumerate(cases):
            reports.append(build_geometry(cid, out_root, view_root))
            print(f"[wall_flowref_v1] {i + 1}/{len(cases)} {cid}", flush=True)
    reports.sort(key=lambda r: r["canonical_id"])
    prior_stats = None if args.skip_prior else build_prior(cases, out_root, split_path)
    manifest = {
        "pack": PACK_NAME, "version": PACK_VERSION, "created_at": utc_now(),
        "view_root": str(view_root), "snapshot_root": str(C.SNAPSHOT_ROOT), "capfit_rule_path": str(_CAPFIT_RULE_OVERRIDE),
        "geometry_program_version": C.GEOMETRY_PROGRAM_VERSION,
        "features": list(GEOMETRY_KEYS) + ([PRIOR_KEY] if prior_stats else []),
        "method": ("bend/bifurcation angles measured in the atlas RMF and transported along the centreline; "
                   "upstream windows walk the root-ward segment chain; Murray split uses distal radii^3; "
                   "capfit split (v1.2) = train138-fitted log(Q_ext/Q_int) = a log((r_cap_ext/r_cap_int)^2) + b on the "
                   "geometry-program cap radii, root/common-iliac shares stay protocol; "
                   "v1.3 log_r_over_rdistal = ln(R_local / R_distal(segment)) with the Murray distal-radius recipe "
                   "(median atlas radius over the distal 10 mm of the point's segment); "
                   f"population prior = mean ln(WSS_peak, floor {PRIOR_FLOOR_PA} Pa) per (semantic, s/L x{PRIOR_S_BINS}, "
                   f"theta x{PRIOR_THETA_BINS}) bin over train138, leave-one-out for train cases"),
        "deployment_inputs_only_geometry": True,
        "label_derived_fields": [f"wall_{PRIOR_KEY}"],
        "prior": prior_stats, "split_path": str(split_path),
        "cases": len(reports), "seconds": round(time.time() - started, 1), "reports": reports,
    }
    Path(out_root).mkdir(parents=True, exist_ok=True)
    (Path(out_root) / "flowref_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"[wall_flowref_v1] wrote {len(reports)} cases to {out_root} in {manifest['seconds']}s; prior={prior_stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
