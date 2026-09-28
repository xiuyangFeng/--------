"""Rebuilt case vs library case: the pre-registered comparison (COMPARISON_CRITERIA.md in each trial directory).

WSS = the ``wall-shear`` scalar column (the label definition). Node sets:
  * coincident (tet-prism family: both walls are the STL nodes) -> node-to-node, tolerance 1 um (float32 STL);
  * different (poly family: both walls are remeshed; lost-case rebuild) -> the rebuilt field is sampled at the library
    nodes by barycentric interpolation on the rebuilt wall triangulation (polygons fan-triangulated, rows matched to
    case nodes by coordinates). When the library case file exists, an *interpolation floor* is reported: the library
    field sampled at the rebuilt nodes and back (round trip) against itself.
Outlet flow/pressure come from the UDF's own per-step print (``P_ave_out*``/``Q_ave_out*``, distal boundary faces).

Criteria v1 (QIN_SI_FU, 2026-09-27) used the raw wall-pressure R2; v2 (2026-09-28, AAA/ILO/YANG) replaces it by the
case-mean-centred R2 (the project's pressure label is centred per case) and keeps the mean-pressure check.

    python -m cfd_auto.compare <library case dir> <trial case dir> [--criteria v2] [--ref-log <transcript>] [--no-volume]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

PEAK_STEP = 1162
CYCLE = range(1120, 1281, 2)       # the 81 exported frames
OUTLETS = ("outle", "outli", "outri", "outre")
GATES = {
    "v1": {"P1_wss_r2": (">=", 0.95), "P2_wss_rel_l2": ("<=", 0.15), "P3_mean_rel": ("abs<=", 0.05), "P3_p50_rel": ("abs<=", 0.05),
           "P3_p99_rel": ("abs<=", 0.10), "P4_hotspot_iou": (">=", 0.80), "P5_pressure_r2": (">=", 0.99), "P5_pressure_mean_rel": ("abs<=", 0.02),
           "P6_flow_share_abs_max": ("<=", 0.02), "P6_outlet_pressure_rel_max": ("abs<=", 0.03)},
    "v2": {"P1_wss_r2": (">=", 0.95), "P2_wss_rel_l2": ("<=", 0.15), "P3_mean_rel": ("abs<=", 0.05), "P3_p50_rel": ("abs<=", 0.05),
           "P3_p99_rel": ("abs<=", 0.10), "P4_hotspot_iou": (">=", 0.80), "P5_pressure_centred_r2": (">=", 0.99), "P5_pressure_mean_rel": ("abs<=", 0.02),
           "P6_flow_share_abs_max": ("<=", 0.02), "P6_outlet_pressure_rel_max": ("abs<=", 0.03)},
}


def read_export(path: Path) -> tuple[list[str], np.ndarray]:
    with open(path) as fh:
        header = [h.strip() for h in fh.readline().split(",")]
        sep = "," if "," in fh.readline() else None      # 4 library cases export space-separated
    return header, np.loadtxt(path, skiprows=1, delimiter=sep)


def frame_file(case_dir: Path, sub: str, step: int) -> Path:
    hits = sorted((case_dir / sub).glob(f"*-{step:04d}"))
    if len(hits) != 1:
        raise FileNotFoundError(f"{case_dir}/{sub}: {len(hits)} files for step {step}")
    return hits[0]


def r2(ref: np.ndarray, new: np.ndarray, w: np.ndarray | None = None) -> float:
    w = np.ones_like(ref) if w is None else w
    mu = np.average(ref, weights=w)
    return float(1 - np.sum(w * (new - ref) ** 2) / np.sum(w * (ref - mu) ** 2))


def stl_vertex_areas(stl: Path, xyz_m: np.ndarray) -> np.ndarray | None:
    """Lumped (1/3 triangle) STL vertex areas ordered like ``xyz_m``; None when the nodes are not STL nodes."""
    from cfd_auto.surface import read_stl
    pts, tri = read_stl(stl)
    a = 0.5 * np.linalg.norm(np.cross(pts[tri[:, 1]] - pts[tri[:, 0]], pts[tri[:, 2]] - pts[tri[:, 0]]), axis=1)
    va = np.zeros(len(pts)); np.add.at(va, tri.ravel(), np.repeat(a / 3, 3))
    d, k = cKDTree(pts * 1e-3).query(xyz_m)
    # only when the export nodes ARE the STL vertex set (a subset -- e.g. a remeshed wall that kept some STL vertices --
    # would get areas of the wrong surface); STL is float32: ~6e-8 m rounding at 0.8 m
    return va[k] if d.max() <= 1e-6 and len(np.unique(k)) == len(pts) else None


def _closest_on_triangles(p, a, b, c):
    """Closest points on triangles (a, b, c) to p (all (n, 3)); returns barycentric weights (n, 3) (Ericson 5.1.5)."""
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = np.einsum("ij,ij->i", ab, ap), np.einsum("ij,ij->i", ac, ap)
    bp = p - b; d3, d4 = np.einsum("ij,ij->i", ab, bp), np.einsum("ij,ij->i", ac, bp)
    cp = p - c; d5, d6 = np.einsum("ij,ij->i", ab, cp), np.einsum("ij,ij->i", ac, cp)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
    w = np.zeros((len(p), 3)); done = np.zeros(len(p), bool)
    def put(mask, u, v, t):
        m = mask & ~done; w[m, 0], w[m, 1], w[m, 2] = u[m], v[m], t[m]; done[m] = True
    one, zero = np.ones(len(p)), np.zeros(len(p))
    put((d1 <= 0) & (d2 <= 0), one, zero, zero)
    put((d3 >= 0) & (d4 <= d3), zero, one, zero)
    put((d6 >= 0) & (d5 <= d6), zero, zero, one)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = d1 / (d1 - d3); put((vc <= 0) & (d1 >= 0) & (d3 <= 0), 1 - t, t, zero)
        t = d2 / (d2 - d6); put((vb <= 0) & (d2 >= 0) & (d6 <= 0), 1 - t, zero, t)
        t = (d4 - d3) / ((d4 - d3) + (d5 - d6)); put((va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0), zero, 1 - t, t)
        den = va + vb + vc; v, u = vb / den, vc / den; put(np.ones(len(p), bool), 1 - v - u, v, u)
    return w


class WallSampler:
    """Barycentric sampling of a case's wall export on its own wall triangulation."""

    def __init__(self, case_file: Path, wall_zone: str, export_xyz: np.ndarray):
        from wss_pinn.v4.fluent_topology import read_fluent_mesh
        mesh = read_fluent_mesh(case_file, keep_interior=False)
        tris, areas = [], None
        for sec in mesh.face_sections:
            if mesh.zone_name(sec.zone_id).split(")")[0] != wall_zone or not sec.attached:
                continue
            cnt = np.diff(sec.offsets)
            for k in np.unique(cnt):                        # fan-triangulate polygons with k nodes
                idx = np.flatnonzero(cnt == k)
                rows = sec.nodes[sec.offsets[idx][:, None] + np.arange(k)]
                for j in range(1, k - 1):
                    tris.append(np.c_[rows[:, 0], rows[:, j], rows[:, j + 1]])
        tris = np.vstack(tris)
        used = np.unique(tris)
        d, row = cKDTree(export_xyz).query(mesh.nodes_m[used])
        if d.max() > 1e-8:
            raise ValueError(f"{case_file}: wall nodes do not match the export rows (max {d.max():.2e} m)")
        node_row = np.full(len(mesh.nodes_m), -1); node_row[used] = row
        self.tri_rows = node_row[tris]                        # triangles in export-row indices
        self.xyz = export_xyz
        P = export_xyz[self.tri_rows]
        ta = 0.5 * np.linalg.norm(np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]), axis=1)
        self.row_area = np.zeros(len(export_xyz)); np.add.at(self.row_area, self.tri_rows.ravel(), np.repeat(ta / 3, 3))
        order = np.argsort(self.tri_rows.ravel(), kind="stable")
        self._tri_of = (np.arange(len(tris)).repeat(3)[order], np.searchsorted(self.tri_rows.ravel()[order], np.arange(len(export_xyz) + 1)))
        self._tree = cKDTree(export_xyz)

    def weights(self, query: np.ndarray, k: int = 4) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(rows (n,3), bary (n,3), distance (n,)) of the closest point on the wall for every query point."""
        _, nn = self._tree.query(query, k=k)
        tri_ids, ptr = self._tri_of
        cand = [np.concatenate([tri_ids[ptr[r]:ptr[r + 1]] for r in row]) for row in nn]
        L = max(len(c) for c in cand)
        C = np.array([np.pad(c, (0, L - len(c)), mode="edge") for c in cand])     # (n, L) candidate triangles
        q = np.repeat(query, L, axis=0); T = self.tri_rows[C.ravel()]
        a, b, c = self.xyz[T[:, 0]], self.xyz[T[:, 1]], self.xyz[T[:, 2]]
        w = _closest_on_triangles(q, a, b, c)
        pt = w[:, :1] * a + w[:, 1:2] * b + w[:, 2:] * c
        dist = np.linalg.norm(pt - q, axis=1).reshape(len(query), L)
        best = dist.argmin(1)
        sel = np.arange(len(query)) * L + best
        return T[sel], w[sel], dist[np.arange(len(query)), best]


def sample(values: np.ndarray, rows: np.ndarray, bary: np.ndarray) -> np.ndarray:
    return np.einsum("ij,ij->i", values[rows], bary)


def udf_prints(log: Path) -> dict[int, dict[str, float]]:
    """Per time step: P_ave_out*/Q_ave_out* printed by execute_at_end (last print of each step wins)."""
    out, step = {}, None
    for line in open(log, errors="replace"):
        m = re.search(r"time step = (\d+)", line)
        if m:
            step = int(m.group(1)); continue
        if "P_ave_out" in line and step is not None:
            for key, val in re.findall(r"(P_ave_out\w\w|Q_ave_out\w\w)=([-+0-9.eE]+)", line):
                out.setdefault(step, {})[key] = float(val)
    return out


def solver_log(case_dir: Path) -> Path:
    cands = sorted(case_dir.glob("Global_conditions/Fluent_*.out")) + sorted(case_dir.glob("Fluent_*.out"))
    cands = [c for c in cands if "P_ave_out" in c.read_text(errors="replace")[-200000:]]
    if not cands:
        raise FileNotFoundError(f"{case_dir}: no Fluent transcript with UDF prints")
    return cands[-1]


def _case_file(d: Path) -> Path | None:
    """The case a directory runs: the read-case of its 2.jou when present (trial directories also hold
    reference_settings.cas.gz), else its only .cas.gz."""
    jou = d / "2.jou"
    if jou.exists():
        m = re.search(r"/file/read-case\s+(\S+)", jou.read_text())
        if m and Path(m.group(1)).exists() and "cfd_auto_trial" in m.group(1):
            return Path(m.group(1))
    c = [x for x in sorted(d.glob("*.cas.gz")) if x.name != "reference_settings.cas.gz"]
    return c[0] if len(c) == 1 else None


def compare(ref_dir: Path, new_dir: Path, criteria: str = "v2", ref_log: Path | None = None, ref_case: Path | None = None,
            volume: bool = True, wall_zone: str = "wall", use_ref_case: bool = True) -> dict:
    stl = sorted(ref_dir.glob("*.stl"))[0]
    hdr, ref = read_export(frame_file(ref_dir, "ascii", PEAK_STEP))
    _, new = read_export(frame_file(new_dir, "ascii", PEAK_STEP))
    col = {h: i for i, h in enumerate(hdr)}
    ref_xyz, new_xyz = ref[:, 1:4], new[:, 1:4]
    d, k = cKDTree(new_xyz).query(ref_xyz)
    coincident = bool(d.max() <= 1e-6 and len(np.unique(k)) == len(k) and len(ref) == len(new))
    match = {"ref_nodes": int(len(ref)), "new_nodes": int(len(new)), "coincident": coincident, "nearest_node_distance_mm_p50_p99_max": (np.percentile(d, [50, 99, 100]) * 1e3).round(4).tolist()}
    new_case = _case_file(new_dir)
    if coincident:
        to_ref = lambda vals_new, _k=k: vals_new[_k]
    else:
        ns = WallSampler(new_case, wall_zone, new_xyz)
        rows, bary, dist = ns.weights(ref_xyz)
        match["projection_distance_mm_p50_p99_max"] = (np.percentile(dist, [50, 99, 100]) * 1e3).round(4).tolist()
        to_ref = lambda vals_new, _r=rows, _b=bary: sample(vals_new, _r, _b)
    ref_case = ref_case if ref_case is not None else (_case_file(ref_dir) if use_ref_case else None)
    w = stl_vertex_areas(stl, ref_xyz)
    weights_source = "stl"
    rs = None
    if w is None and ref_case is not None:
        rs = WallSampler(ref_case, wall_zone, ref_xyz); w = rs.row_area; weights_source = "library wall triangulation"
    if w is None:
        w = np.ones(len(ref)); weights_source = "uniform (no library triangulation)"
    match["area_weights"] = weights_source
    ws_r = ref[:, col["wall-shear"]]; ws_n = to_ref(new[:, col["wall-shear"]])
    p_r = ref[:, col["pressure"]]; p_n = to_ref(new[:, col["pressure"]])
    hot_r, hot_n = ws_r >= np.percentile(ws_r, 90), ws_n >= np.percentile(ws_n, 90)
    res = {"criteria": criteria, "match": match, "peak_step": PEAK_STEP,
           "P1_wss_r2": r2(ws_r, ws_n), "P1_wss_r2_area_weighted": r2(ws_r, ws_n, w),
           "P2_wss_rel_l2": float(np.linalg.norm(ws_n - ws_r) / np.linalg.norm(ws_r)),
           "wss_mean_pa": [float(np.average(ws_r, weights=w)), float(np.average(ws_n, weights=w))],
           "P3_mean_rel": float(np.average(ws_n, weights=w) / np.average(ws_r, weights=w) - 1),
           "wss_p50_pa": [float(np.percentile(ws_r, 50)), float(np.percentile(ws_n, 50))],
           "P3_p50_rel": float(np.percentile(ws_n, 50) / np.percentile(ws_r, 50) - 1),
           "wss_p99_pa": [float(np.percentile(ws_r, 99)), float(np.percentile(ws_n, 99))],
           "P3_p99_rel": float(np.percentile(ws_n, 99) / np.percentile(ws_r, 99) - 1),
           "P4_hotspot_iou": float(np.sum(hot_r & hot_n) / np.sum(hot_r | hot_n)),
           "P5_pressure_r2": r2(p_r, p_n), "P5_pressure_centred_r2": r2(p_r - np.average(p_r, weights=w), p_n - np.average(p_n, weights=w), w),
           "P5_pressure_mean_rel": float(np.average(p_n, weights=w) / np.average(p_r, weights=w) - 1),
           "pressure_mean_pa": [float(np.average(p_r, weights=w)), float(np.average(p_n, weights=w))]}
    if not coincident and rs is not None:
        # interpolation floor: library field -> rebuilt nodes -> library nodes
        r2rows, r2bary, _ = rs.weights(new_xyz)
        back = sample(sample(ws_r, r2rows, r2bary), rows, bary)
        res["interpolation_floor"] = {"wss_r2": r2(ws_r, back), "wss_rel_l2": float(np.linalg.norm(back - ws_r) / np.linalg.norm(ws_r))}
    prof_path = new_dir / "reference_profile.json"
    if prof_path.exists():
        oc = np.array([o["centre_mm"] for o in json.loads(prof_path.read_text())["openings"]]) * 1e-3
        dist = cKDTree(oc).query(ref_xyz)[0] * 1e3
        res["by_distance_to_opening_mm"] = {}
        for lo, hi in ((0, 10), (10, 20), (20, 40), (40, 1e9)):
            m = (dist >= lo) & (dist < hi)
            if m.sum() > 10:
                res["by_distance_to_opening_mm"][f"{lo}-{hi if hi < 1e9 else 'inf'}"] = {"nodes": int(m.sum()), "wss_r2": r2(ws_r[m], ws_n[m]),
                                                                                        "wss_mean_rel": float(ws_n[m].mean() / ws_r[m].mean() - 1)}
    ur, un = udf_prints(ref_log or solver_log(ref_dir)), udf_prints(solver_log(new_dir))
    common = sorted(set(ur) & set(un))
    last = [s for s in common if s > common[-1] - 160]
    def share(u):
        q = {o: np.mean([u[s][f"Q_ave_{o}"] for s in last]) for o in OUTLETS}
        tot = sum(q.values()); return {o: float(q[o] / tot) for o in OUTLETS}, float(tot)
    sr, tr = share(ur); sn, tn = share(un)
    res["last_cycle_steps"] = [last[0], last[-1]]
    res["outlet_flow_share_last_cycle"] = {o: [sr[o], sn[o]] for o in OUTLETS}
    res["outlet_total_flow_last_cycle_kg_s"] = [tr, tn]
    res["P6_flow_share_abs_max"] = float(max(abs(sn[o] - sr[o]) for o in OUTLETS))
    pr = {o: ur[PEAK_STEP][f"P_ave_{o}"] for o in OUTLETS}; pn = {o: un[PEAK_STEP][f"P_ave_{o}"] for o in OUTLETS}
    res["outlet_pressure_peak_pa"] = {o: [pr[o], pn[o]] for o in OUTLETS}
    res["P6_outlet_pressure_rel_max"] = float(max((abs(pn[o] / pr[o] - 1) for o in OUTLETS)))
    # cycle: per-frame R2, TAWSS, OSI (frames re-ordered to the peak-frame rows by coordinates)
    cols = [col["wall-shear"], col["x-wall-shear"], col["y-wall-shear"], col["z-wall-shear"]]
    Fr, Fn, tr_ref, tr_new = [], [], cKDTree(ref_xyz), cKDTree(new_xyz)
    for s in CYCLE:
        _, a = read_export(frame_file(ref_dir, "ascii", s)); _, b = read_export(frame_file(new_dir, "ascii", s))
        a = a[np.argsort(tr_ref.query(a[:, 1:4])[1])]; b = b[np.argsort(tr_new.query(b[:, 1:4])[1])]
        Fr.append(a[:, cols]); Fn.append(np.stack([to_ref(b[:, c]) for c in cols], 1))
    Fr, Fn = np.stack(Fr), np.stack(Fn)
    per_frame = [r2(Fr[i, :, 0], Fn[i, :, 0]) for i in range(len(Fr))]
    tawss_r, tawss_n = Fr[:, :, 0].mean(0), Fn[:, :, 0].mean(0)
    def osi(F):
        mv = np.linalg.norm(F[:, :, 1:4].mean(0), axis=1); mm = np.linalg.norm(F[:, :, 1:4], axis=2).mean(0)
        return 0.5 * (1 - mv / np.maximum(mm, 1e-12))
    osi_r, osi_n = osi(Fr), osi(Fn)
    res["secondary"] = {"per_frame_wss_r2_min_median": [float(np.min(per_frame)), float(np.median(per_frame))],
                        "tawss_r2": r2(tawss_r, tawss_n), "tawss_mean_rel": float(np.average(tawss_n, weights=w) / np.average(tawss_r, weights=w) - 1),
                        "osi_r2": r2(osi_r, osi_n), "osi_mae": float(np.mean(np.abs(osi_n - osi_r))), "osi_mean": [float(osi_r.mean()), float(osi_n.mean())]}
    if volume:
        try:
            vh, vr = read_export(frame_file(ref_dir, "ascii_in", PEAK_STEP)); _, vn = read_export(frame_file(new_dir, "ascii_in", PEAK_STEP))
            vc = {h: i for i, h in enumerate(vh)}
            dv, kv = cKDTree(vn[:, 1:4]).query(vr[:, 1:4])
            res["secondary"]["volume_velocity_r2"] = r2(vr[:, vc["velocity-magnitude"]], vn[kv, vc["velocity-magnitude"]])
            res["secondary"]["volume_pressure_centred_r2"] = r2(vr[:, vc["pressure"]] - vr[:, vc["pressure"]].mean(), vn[kv, vc["pressure"]] - vn[kv, vc["pressure"]].mean())
            res["secondary"]["volume_cells"] = [int(len(vr)), int(len(vn))]
            res["secondary"]["volume_match_distance_mm_p50_p99"] = (np.percentile(dv, [50, 99]) * 1e3).round(3).tolist()
        except FileNotFoundError as exc:
            res["secondary"]["volume"] = f"skipped: {exc}"
    verdict = {}
    for key, (op, thr) in GATES[criteria].items():
        v = res[key]
        verdict[key] = {"value": v, "threshold": f"{op} {thr}", "pass": bool(v >= thr if op == ">=" else v <= thr if op == "<=" else abs(v) <= thr)}
    res["gates"] = verdict
    res["all_pass"] = all(g["pass"] for g in verdict.values())
    return res


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("ref_dir", type=Path); ap.add_argument("new_dir", type=Path); ap.add_argument("--out", type=Path)
    ap.add_argument("--criteria", default="v2", choices=sorted(GATES)); ap.add_argument("--ref-log", type=Path); ap.add_argument("--ref-case", type=Path)
    ap.add_argument("--no-volume", action="store_true"); ap.add_argument("--wall-zone", default="wall")
    ap.add_argument("--no-ref-case", action="store_true", help="the library .cas is not this case's mesh (e.g. overwritten)")
    a = ap.parse_args()
    res = compare(a.ref_dir, a.new_dir, a.criteria, a.ref_log, a.ref_case, not a.no_volume, a.wall_zone, not a.no_ref_case)
    (a.out or a.new_dir / "comparison.json").write_text(json.dumps(res, indent=1))
    for k, g in res["gates"].items():
        print(f"{k:28s} {g['value']:+.4f}  {g['threshold']:10s} {'PASS' if g['pass'] else 'FAIL'}")
    print("ALL PASS" if res["all_pass"] else "NOT ALL PASS")


if __name__ == "__main__":
    main()
