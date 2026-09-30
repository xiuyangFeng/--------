#!/usr/bin/env python3
"""Turn postview point clouds (wall ∪ interior) into a sliceable tetrahedral lumen mesh.

Why
---
``interior_pointcloud.vtp`` holds the Fluent cell centres of the ``blood`` zone. A point cloud has
no volume cells, so ParaView/EnSight can only draw dots and *Slice/Clip* show nothing useful.
``anatomy_volume.vtu`` fixes that with the native Fluent polyhedra (VTK_CONVEX_POINT_SET), but those
cells are cut by a per-cell Delaunay inside VTK: slow, non-conforming triangle soup, 580 MB, and it
needs the ``.cas`` file (not available at deployment).

This tool builds a plain linear tetrahedral mesh from the points we already ship:

1. points  = CFD wall nodes (85 k, u = 0 no-slip) ∪ interior cell centres (677 k)
2. tets    = scipy Delaunay of those points (convex hull, ~5 M tets)
3. filter  = keep tets whose centroid is inside the wall STL (signed distance, orientation auto-
             detected from the interior points) → removes the hull tets that bridge concavities and
             the gap between iliac branches; long tets are additionally checked at 10 sample points
4. fields  = every arm × phase array is attached as *point data* (float32), so ParaView's Slice /
             Clip / Contour interpolate linearly → CFD-Post-like smooth fill

Outputs (``<case>/sliceable_tets/`` by default)
  lumen_tets_geometry.vtu   tets + ``is_wall`` only (attach your own arrays)
  velocity_tets.vtu         ``<phase>_speed_cfd``, ``<arm>_<phase>_speed_pred``, vectors, errors, selfmax
  pressure_tets.vtu         ``<phase>_pressure_cfd``, ``<arm>_<phase>_pressure_pred``, errors, selfmax
  ensight/*.case            same two meshes as EnSight Gold (scalar/vector per node)
  plots/preview_slice_<phase>.png   matplotlib slice through the plane used by anatomy_volume
  paraview_slice_demo.py    pvpython / ParaView-Python-Shell script reproducing the slice view
  build_report.json         counts, timings, orientation sign, dropped tets, kept lumen volume

Formal R²/MAE stay on the same-point CSV; this mesh is for display only.

Usage
-----
  /public/newhome/cy/.conda/envs/GNN/bin/python3.10 training_wss_min/tools/build_sliceable_tets.py \
      --case-dir <postview>/AAA__unruputer__SHEN_FANG_JIN [--out ...] [--no-ensight] [--preview-phases peak]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

TOOL_VERSION = "1.0 (2026-09-22)"
PHASE_ORDER = ("accel", "peak", "decel", "trough")
KIND_ARRAY = "point_kind_0_wall_1_interior"

VEL_CFD_ARRAYS = ("speed_cfd", "velocity_cfd_aligned_m_s", "speed_cfd_selfmax")
VEL_PRED_ARRAYS = ("speed_pred", "velocity_pred_aligned_m_s", "err_speed", "abs_err_speed",
                   "velocity_error_aligned_m_s", "speed_pred_selfmax",
                   "speed_selfmax_error_pred_minus_cfd", "speed_selfmax_abs_error")
PRS_CFD_ARRAYS = ("pressure_cfd", "pressure_cfd_selfmax")
PRS_PRED_ARRAYS = ("pressure_pred", "err_pressure", "abs_err_pressure", "pressure_pred_selfmax",
                   "pressure_selfmax_error_pred_minus_cfd", "pressure_selfmax_abs_error")


def log(msg: str) -> None:
    print(f"[sliceable_tets] {msg}", flush=True)


# --------------------------------------------------------------------------- discovery
def discover(case_dir: Path) -> tuple[list[str], list[str], list[str]]:
    vel_arms, prs_arms, phases = [], [], set()
    vel_root, prs_root = case_dir / "velocity", case_dir / "pressure"
    if vel_root.is_dir():
        for arm in sorted(p for p in vel_root.iterdir() if p.is_dir() and p.name != "anatomy_volume"):
            ph = [p.name for p in arm.iterdir() if (p / "interior_pointcloud.vtp").is_file()]
            if ph:
                vel_arms.append(arm.name)
                phases.update(ph)
    if prs_root.is_dir():
        for arm in sorted(p for p in prs_root.iterdir() if p.is_dir()):
            ph = [p.name for p in arm.iterdir() if (p / "full_pointcloud.vtp").is_file()]
            if ph:
                prs_arms.append(arm.name)
                phases.update(ph)
    ordered = [p for p in PHASE_ORDER if p in phases] + sorted(p for p in phases if p not in PHASE_ORDER)
    return vel_arms, prs_arms, ordered


# --------------------------------------------------------------------------- points
def reference_points(case_dir: Path, vel_arms: list[str], prs_arms: list[str], phases: list[str]):
    """Return (P, n_wall, perm_info). Wall points first, then interior, both in file order."""
    import pyvista as pv

    if prs_arms:
        f = case_dir / "pressure" / prs_arms[0] / phases[0] / "full_pointcloud.vtp"
        pd = pv.read(f)
        kind = np.asarray(pd.point_data[KIND_ARRAY]).astype(np.int64)
        order = np.argsort(kind, kind="stable")  # wall (0) first, then interior (1)
        P = np.asarray(pd.points, dtype=np.float64)[order]
        n_wall = int((kind == 0).sum())
        src = str(f)
    else:
        arm = vel_arms[0]
        wall = pv.read(case_dir / "velocity" / arm / phases[0] / "wall_geometry_only.vtp")
        inte = pv.read(case_dir / "velocity" / arm / phases[0] / "interior_pointcloud.vtp")
        P = np.vstack([np.asarray(wall.points, np.float64), np.asarray(inte.points, np.float64)])
        n_wall = int(wall.n_points)
        src = f"{wall} + {inte}"
    return P, n_wall, src


def load_matching(path: Path, ref_xyz: np.ndarray, names: tuple[str, ...], tol_mm: float = 1e-3):
    """Read a .vtp and return {name: array} re-ordered so that its points match ``ref_xyz``."""
    import pyvista as pv
    from scipy.spatial import cKDTree

    pd = pv.read(path)
    xyz = np.asarray(pd.points, np.float64)
    if xyz.shape != ref_xyz.shape:
        raise ValueError(f"{path}: {xyz.shape[0]} points, expected {ref_xyz.shape[0]}")
    if np.allclose(xyz, ref_xyz, atol=tol_mm):
        idx = None
    else:  # same points, different order → match by coordinates
        d, idx = cKDTree(xyz).query(ref_xyz, k=1)
        if d.max() > tol_mm or len(np.unique(idx)) != len(idx):
            raise ValueError(f"{path}: points do not coincide with reference (max {d.max():.4f} mm)")
    out = {}
    for nm in names:
        if nm not in pd.point_data:
            continue
        a = np.asarray(pd.point_data[nm])
        out[nm] = (a if idx is None else a[idx]).astype(np.float32)
    return out


# --------------------------------------------------------------------------- tets
def build_tets(P: np.ndarray, n_wall: int, stl_path: Path, guard_edge_mm: float | None,
               guard_tol_mm: float, report: dict):
    import pyvista as pv
    from scipy.spatial import Delaunay, cKDTree

    stl = pv.read(stl_path)
    t = time.time()
    rng = np.random.default_rng(0)
    samp = rng.choice(np.arange(n_wall, len(P)), min(20000, len(P) - n_wall), replace=False)
    sd = np.asarray(pv.PolyData(P[samp]).compute_implicit_distance(stl).point_data["implicit_distance"])
    frac_neg = float((sd < 0).mean())
    sign = 1.0 if frac_neg > 0.5 else -1.0
    if min(frac_neg, 1 - frac_neg) > 0.05:
        raise RuntimeError(f"STL orientation ambiguous: {frac_neg:.3f} of interior points have sdf<0")
    report["stl"] = dict(path=str(stl_path), n_faces=int(stl.n_cells), open_edges=int(stl.n_open_edges),
                         interior_sample_frac_negative=frac_neg, sign_applied=sign,
                         note="sign=-1 means STL normals point into the lumen; corrected automatically")
    log(f"STL orientation: frac(sdf<0)={frac_neg:.3f} → sign {sign:+.0f}  ({time.time()-t:.1f}s)")

    t = time.time()
    tri = Delaunay(P)
    tets = tri.simplices.astype(np.int64)
    del tri
    log(f"Delaunay: {len(tets)} tets from {len(P)} points ({time.time()-t:.1f}s)")
    X = P[tets]
    cen = X.mean(axis=1)
    v6 = np.einsum("ij,ij->i", np.cross(X[:, 1] - X[:, 0], X[:, 2] - X[:, 0]), X[:, 3] - X[:, 0])
    vol = np.abs(v6) / 6.0
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    emax = np.max([np.linalg.norm(X[:, a] - X[:, b], axis=1) for a, b in pairs], axis=0)

    t = time.time()
    sdf = sign * np.asarray(pv.PolyData(cen).compute_implicit_distance(stl).point_data["implicit_distance"])
    log(f"centroid signed distance ({time.time()-t:.1f}s)")
    keep = (sdf < 0.0) & (vol > 1e-7)
    n_outside, n_degen = int((sdf >= 0).sum()), int((vol <= 1e-7).sum())

    nn = cKDTree(P[n_wall:]).query(P[n_wall:], k=2)[0][:, 1]
    p95 = float(np.percentile(nn, 95))
    if guard_edge_mm is None:
        guard_edge_mm = max(2.0, 2.0 * p95)
    big = np.flatnonzero(keep & (emax > guard_edge_mm))
    n_guard_drop = 0
    if len(big):
        t = time.time()
        Xb = X[big]
        samples = np.concatenate([(Xb[:, a] + Xb[:, b]) / 2 for a, b in pairs]
                                 + [Xb[:, [i for i in range(4) if i != k]].mean(1) for k in range(4)])
        sb = sign * np.asarray(pv.PolyData(samples).compute_implicit_distance(stl).point_data["implicit_distance"])
        sb = sb.reshape(10, len(big)).max(axis=0)
        bad = big[sb > guard_tol_mm]
        keep[bad] = False
        n_guard_drop = int(len(bad))
        log(f"guard: {len(big)} tets with max edge > {guard_edge_mm:.2f} mm checked, {n_guard_drop} dropped ({time.time()-t:.1f}s)")
    used = np.zeros(len(P), bool)
    used[tets[keep].ravel()] = True
    report["tets"] = dict(
        n_delaunay=int(len(tets)), n_kept=int(keep.sum()), n_dropped_outside=n_outside,
        n_dropped_degenerate=n_degen, n_dropped_guard=n_guard_drop,
        guard_edge_mm=float(guard_edge_mm), guard_tol_mm=float(guard_tol_mm),
        interior_nn_spacing_mm=dict(p05=float(np.percentile(nn, 5)), p50=float(np.percentile(nn, 50)), p95=p95),
        kept_volume_mm3=float(vol[keep].sum()), dropped_volume_mm3=float(vol[~keep].sum()),
        kept_max_edge_mm=dict(p50=float(np.percentile(emax[keep], 50)), p99=float(np.percentile(emax[keep], 99)),
                              max=float(emax[keep].max())),
        points_unused=int((~used).sum()), wall_points_unused=int((~used[:n_wall]).sum()),
    )
    log(f"kept {keep.sum()} tets, lumen volume {vol[keep].sum():.0f} mm³, unused points {(~used).sum()}")
    return tets[keep]


def make_grid(P: np.ndarray, tets: np.ndarray, n_wall: int):
    import pyvista as pv

    cells = np.hstack([np.full((len(tets), 1), 4, np.int64), tets]).ravel()
    ug = pv.UnstructuredGrid(cells, np.full(len(tets), pv.CellType.TETRA, np.uint8), P)
    ug.point_data["is_wall"] = (np.arange(len(P)) < n_wall).astype(np.uint8)
    return ug


# --------------------------------------------------------------------------- fields
def attach_velocity(ug, case_dir: Path, arms: list[str], phases: list[str], P: np.ndarray, n_wall: int, report: dict):
    n = len(P)
    ref_int = P[n_wall:]
    cfd_stored: dict[str, dict[str, np.ndarray]] = {}
    per_arm_cfd = False
    for arm in arms:
        for ph in phases:
            f = case_dir / "velocity" / arm / ph / "interior_pointcloud.vtp"
            if not f.is_file():
                continue
            arrs = load_matching(f, ref_int, VEL_CFD_ARRAYS + VEL_PRED_ARRAYS)
            for nm in VEL_CFD_ARRAYS:
                if nm not in arrs:
                    continue
                if ph in cfd_stored and nm in cfd_stored[ph]:
                    if not np.allclose(cfd_stored[ph][nm], arrs[nm], atol=1e-6, equal_nan=True):
                        per_arm_cfd = True
                        ug.point_data[f"{arm}_{ph}_{nm}"] = _pad(arrs[nm], n, n_wall)
                    continue
                cfd_stored.setdefault(ph, {})[nm] = arrs[nm]
                ug.point_data[f"{ph}_{nm}"] = _pad(arrs[nm], n, n_wall)
            for nm in VEL_PRED_ARRAYS:
                if nm in arrs:
                    ug.point_data[f"{arm}_{ph}_{nm}"] = _pad(arrs[nm], n, n_wall)
    report["velocity"] = dict(arms=arms, phases=phases, wall_value="0 (rigid wall, no-slip) for cfd/pred/error",
                              cfd_differs_between_arms=per_arm_cfd, n_arrays=len(ug.point_data))


def attach_pressure(ug, case_dir: Path, arms: list[str], phases: list[str], P: np.ndarray, n_wall: int, report: dict):
    import pyvista as pv

    cfd_stored: dict[str, dict[str, np.ndarray]] = {}
    per_arm_cfd = False
    for arm in arms:
        for ph in phases:
            f = case_dir / "pressure" / arm / ph / "full_pointcloud.vtp"
            if not f.is_file():
                continue
            arrs = load_matching(f, P, PRS_CFD_ARRAYS + PRS_PRED_ARRAYS)
            for nm in PRS_CFD_ARRAYS:
                if nm not in arrs:
                    continue
                if ph in cfd_stored and nm in cfd_stored[ph]:
                    if not np.allclose(cfd_stored[ph][nm], arrs[nm], atol=1e-6, equal_nan=True):
                        per_arm_cfd = True
                        ug.point_data[f"{arm}_{ph}_{nm}"] = arrs[nm]
                    continue
                cfd_stored.setdefault(ph, {})[nm] = arrs[nm]
                ug.point_data[f"{ph}_{nm}"] = arrs[nm]
            for nm in PRS_PRED_ARRAYS:
                if nm in arrs:
                    ug.point_data[f"{arm}_{ph}_{nm}"] = arrs[nm]
    report["pressure"] = dict(arms=arms, phases=phases, wall_value="native wall node values from full_pointcloud.vtp",
                              cfd_differs_between_arms=per_arm_cfd, n_arrays=len(ug.point_data))


def _pad(a: np.ndarray, n: int, n_wall: int) -> np.ndarray:
    """Interior-only array → full-length array with zeros on wall rows."""
    out = np.zeros((n,) + a.shape[1:], np.float32)
    out[n_wall:] = a
    return out


# --------------------------------------------------------------------------- EnSight
def write_ensight(ug, out_dir: Path, name: str, report: dict) -> None:
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    out_dir.mkdir(parents=True, exist_ok=True)
    w = vtk.vtkEnSightWriter()
    w.SetInputData(ug)
    w.SetPath(str(out_dir) + os.sep)
    w.SetBaseName(name)
    w.SetFileName(str(out_dir / name))
    w.SetTimeStep(0)
    w.Write()
    w.WriteCaseFile(1)
    case = out_dir / f"{name}.0.case"
    ok = False
    if case.is_file():
        r = vtk.vtkEnSightGoldBinaryReader()
        r.SetCaseFileName(str(case))
        r.ReadAllVariablesOn()
        r.Update()
        blk = r.GetOutput().GetBlock(0)
        if blk is not None and blk.GetNumberOfCells() == ug.n_cells:
            first = next(k for k in ug.point_data.keys() if k != "is_wall")
            arr = blk.GetPointData().GetArray(first + "_n")
            ok = arr is not None and np.allclose(vtk_to_numpy(arr).ravel(), np.asarray(ug.point_data[first]).ravel(), atol=1e-5)
    report.setdefault("ensight", {})[name] = dict(case=str(case), readback_ok=bool(ok))
    log(f"EnSight {case.name}: readback_ok={ok}")


# --------------------------------------------------------------------------- preview
def slice_plane(case_dir: Path, P: np.ndarray):
    rep = case_dir / "velocity" / "anatomy_volume" / "identity_report.json"
    if rep.is_file():
        sp = json.loads(rep.read_text()).get("slice_preview", {})
        if "origin_mm" in sp and "normal" in sp:
            return np.asarray(sp["origin_mm"], float), np.asarray(sp["normal"], float), "identity_report.slice_preview"
    c = P.mean(axis=0)
    w, v = np.linalg.eigh(np.cov((P - c).T))
    return c, v[:, 0], "PCA smallest-variance axis through centroid"


def preview(vel_ug, prs_ug, origin, normal, phases: list[str], vel_arms: list[str], prs_arms: list[str], plots: Path,
           suptitle: str | None = None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    plots.mkdir(parents=True, exist_ok=True)
    n = normal / np.linalg.norm(normal)
    helper = np.array([0, 0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0, 0])
    u = np.cross(n, helper); u /= np.linalg.norm(u); v = np.cross(n, u)

    def frame(sl):
        Q = np.asarray(sl.points) - origin
        return mtri.Triangulation(Q @ u, Q @ v, sl.faces.reshape(-1, 4)[:, 1:])

    out = []
    for ph in phases:
        panels = []
        if vel_ug is not None:
            sl = vel_ug.slice(normal=n, origin=origin).triangulate()
            T = frame(sl)
            cfd = np.asarray(sl.point_data.get(f"{ph}_speed_cfd", sl.point_data.get(f"{vel_arms[0]}_{ph}_speed_cfd")))
            vmax = float(np.percentile(cfd, 99.5))
            panels.append((T, cfd, f"CFD |u| {ph} (m/s)", "turbo", 0.0, vmax))
            for arm in vel_arms:
                key = f"{arm}_{ph}_speed_pred"
                if key in sl.point_data:
                    panels.append((T, np.asarray(sl.point_data[key]), f"{arm} pred |u| {ph} (m/s)", "turbo", 0.0, vmax))
        if prs_ug is not None:
            sl = prs_ug.slice(normal=n, origin=origin).triangulate()
            T = frame(sl)
            cfd = np.asarray(sl.point_data.get(f"{ph}_pressure_cfd", sl.point_data.get(f"{prs_arms[0]}_{ph}_pressure_cfd")))
            lo, hi = np.percentile(cfd, [0.5, 99.5])
            panels.append((T, cfd, f"CFD p−p̄ {ph} (Pa)", "coolwarm", float(lo), float(hi)))
            for arm in prs_arms:
                key = f"{arm}_{ph}_pressure_pred"
                if key in sl.point_data:
                    panels.append((T, np.asarray(sl.point_data[key]), f"{arm} pred p−p̄ {ph} (Pa)", "coolwarm", float(lo), float(hi)))
        if not panels:
            continue
        cols = len(panels)
        fig, axes = plt.subplots(1, cols, figsize=(4.2 * cols, 8.5), squeeze=False)
        for ax, (T, z, title, cmap, lo, hi) in zip(axes[0], panels):
            tp = ax.tripcolor(T, z, shading="gouraud", cmap=cmap, vmin=lo, vmax=hi)
            ax.set_aspect("equal"); ax.set_title(title, fontsize=10); ax.set_xticks([]); ax.set_yticks([])
            fig.colorbar(tp, ax=ax, shrink=0.55)
        fig.suptitle(suptitle or f"Delaunay lumen tets, plane origin={np.round(origin,1).tolist()} normal={np.round(n,3).tolist()}", fontsize=9)
        fig.tight_layout()
        png = plots / f"preview_slice_{ph}.png"
        fig.savefig(png, dpi=110); plt.close(fig)
        out.append(str(png))
        log(f"preview {png.name}")
    return out


PARAVIEW_DEMO = r'''# Run with:  pvpython paraview_slice_demo.py [velocity_tets.vtu] [ARRAY]
# or paste into ParaView → View → Python Shell (set FILE / ARRAY first).
import sys, os
from paraview.simple import *
HERE = os.path.dirname(os.path.abspath(__file__))
FILE = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "velocity_tets.vtu")
ARRAY = sys.argv[2] if len(sys.argv) > 2 else "__ARRAY__"
ORIGIN, NORMAL = __ORIGIN__, __NORMAL__
vol = XMLUnstructuredGridReader(FileName=[FILE])
view = GetActiveViewOrCreate("RenderView")
shell = ExtractSurface(Input=vol)                       # transparent lumen shell
sd = Show(shell, view); sd.Opacity = 0.15; ColorBy(sd, None); sd.DiffuseColor = [0.8, 0.8, 0.8]
sl = Slice(Input=vol); sl.SliceType = "Plane"; sl.SliceType.Origin = ORIGIN; sl.SliceType.Normal = NORMAL
d = Show(sl, view); ColorBy(d, ("POINTS", ARRAY)); d.SetScalarBarVisibility(view, True)
lut = GetColorTransferFunction(ARRAY); lut.ApplyPreset("Turbo", True)
view.ResetCamera(); Render()
'''


# --------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case-dir", required=True, type=Path, help="postview case dir (contains velocity/, pressure/, aligned_geometry.stl)")
    ap.add_argument("--out", type=Path, default=None, help="output dir (default <case-dir>/sliceable_tets)")
    ap.add_argument("--stl", type=Path, default=None, help="wall STL in the same aligned mm frame (default <case-dir>/aligned_geometry.stl)")
    ap.add_argument("--guard-edge-mm", type=float, default=None, help="tets with a longer edge get the 10-sample inside check (default 2×p95 interior spacing, ≥2 mm)")
    ap.add_argument("--guard-tol-mm", type=float, default=0.05)
    ap.add_argument("--no-ensight", action="store_true")
    ap.add_argument("--preview-phases", default="peak", help="comma list or 'all' or 'none'")
    args = ap.parse_args(argv)

    t0 = time.time()
    case_dir: Path = args.case_dir.resolve()
    out: Path = (args.out or case_dir / "sliceable_tets").resolve()
    stl_path: Path = (args.stl or case_dir / "aligned_geometry.stl").resolve()
    out.mkdir(parents=True, exist_ok=True)
    report: dict = dict(tool="build_sliceable_tets.py", version=TOOL_VERSION, case_dir=str(case_dir), out=str(out),
                        argv=sys.argv[1:], display_only="formal R²/MAE remain on the same-point CSV; tets are for display")

    vel_arms, prs_arms, phases = discover(case_dir)
    if not phases:
        raise SystemExit(f"no velocity/pressure phase folders under {case_dir}")
    log(f"velocity arms {vel_arms}, pressure arms {prs_arms}, phases {phases}")
    P, n_wall, src = reference_points(case_dir, vel_arms, prs_arms, phases)
    report["points"] = dict(source=src, n_total=int(len(P)), n_wall=n_wall, n_interior=int(len(P) - n_wall),
                            bbox_mm=[float(x) for x in np.r_[P.min(0), P.max(0)]])
    log(f"points: {n_wall} wall + {len(P)-n_wall} interior")

    tets = build_tets(P, n_wall, stl_path, args.guard_edge_mm, args.guard_tol_mm, report)

    geo = make_grid(P, tets, n_wall)
    t = time.time(); geo.save(out / "lumen_tets_geometry.vtu"); log(f"wrote lumen_tets_geometry.vtu ({time.time()-t:.1f}s)")

    vel_ug = prs_ug = None
    if vel_arms:
        vel_ug = make_grid(P, tets, n_wall)
        attach_velocity(vel_ug, case_dir, vel_arms, phases, P, n_wall, report)
        t = time.time(); vel_ug.save(out / "velocity_tets.vtu"); log(f"wrote velocity_tets.vtu ({len(vel_ug.point_data)} arrays, {time.time()-t:.1f}s)")
    if prs_arms:
        prs_ug = make_grid(P, tets, n_wall)
        attach_pressure(prs_ug, case_dir, prs_arms, phases, P, n_wall, report)
        t = time.time(); prs_ug.save(out / "pressure_tets.vtu"); log(f"wrote pressure_tets.vtu ({len(prs_ug.point_data)} arrays, {time.time()-t:.1f}s)")

    if not args.no_ensight:
        if vel_ug is not None:
            write_ensight(vel_ug, out / "ensight", "velocity_tets", report)
        if prs_ug is not None:
            write_ensight(prs_ug, out / "ensight", "pressure_tets", report)

    origin, normal, plane_src = slice_plane(case_dir, P)
    report["preview_plane"] = dict(origin_mm=origin.tolist(), normal=normal.tolist(), source=plane_src)
    if args.preview_phases != "none":
        ph = phases if args.preview_phases == "all" else [p for p in args.preview_phases.split(",") if p in phases]
        report["preview_png"] = preview(vel_ug, prs_ug, origin, normal, ph, vel_arms, prs_arms, out / "plots")

    demo_array = f"{vel_arms[0]}_{phases[min(1, len(phases)-1)]}_speed_pred" if vel_arms else f"{prs_arms[0]}_{phases[0]}_pressure_pred"
    (out / "paraview_slice_demo.py").write_text(
        PARAVIEW_DEMO.replace("__ARRAY__", demo_array).replace("__ORIGIN__", str([round(float(x), 3) for x in origin]))
        .replace("__NORMAL__", str([round(float(x), 4) for x in normal])))

    sizes = {p.name: round(p.stat().st_size / 1e6, 1) for p in out.glob("*.vtu")}
    report["file_size_mb"] = sizes
    report["elapsed_s"] = round(time.time() - t0, 1)
    (out / "build_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    log(f"done in {report['elapsed_s']} s → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
