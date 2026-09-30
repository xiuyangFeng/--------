#!/usr/bin/env python3
"""Coarse hexahedral lumen for EnSight slice viewing.

EnSight Standard 2023 draws a part as points once the client cannot hold
every element (this case's 4.7M-tet Gold file shows as
``48730 client / 4774713 server``). A clip of that mesh stays over the
same budget, so the cross-section stays a point cloud.

``wss_deploy`` fills a slice by interpolating the discrete samples onto a
grid inside the wall contour. This tool does the 3D equivalent, coarse
enough that EnSight loads every cell:

  wall nodes (velocity 0) ∪ interior cell centres
  → inverse-distance onto a ~2.5 mm hexahedral lattice
  → drop hexes whose centre is outside ``aligned_geometry.stl``

A cross-section is then a few hundred quads. Clip draws them as a filled
face. Display only; formal R²/MAE stay on the same-point CSV.

Usage
-----
  /public/newhome/cy/.conda/envs/GNN/bin/python3.10 training_wss_min/tools/build_ensight_hexgrid.py \
      --case-dir <postview>/AAA__unruputer__SHEN_FANG_JIN
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_sliceable_tets as bt

TOOL_VERSION = "1.0 (2026-09-23)"
# EnSight Standard 2023 on this mesh reported ~48k client elements and
# drew anything larger as points. Stay under that with margin.
CLIENT_ELEMENT_BUDGET = 40000
HEX_CORNERS = ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
               (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1))


def log(msg: str) -> None:
    print(f"[ensight_hex] {msg}", flush=True)


def collect_source(case_dir: Path):
    """Point coordinates plus every velocity/pressure array, wall rows included."""
    import pyvista as pv

    vel_arms, prs_arms, phases = bt.discover(case_dir)
    if not phases:
        raise SystemExit(f"no velocity/pressure phase folders under {case_dir}")
    P, n_wall, src = bt.reference_points(case_dir, vel_arms, prs_arms, phases)
    cloud = pv.PolyData(P)
    report_v: dict = {}
    report_p: dict = {}
    if vel_arms:
        bt.attach_velocity(cloud, case_dir, vel_arms, phases, P, n_wall, report_v)
    if prs_arms:
        bt.attach_pressure(cloud, case_dir, prs_arms, phases, P, n_wall, report_p)
    arrays = {k: np.asarray(cloud.point_data[k]) for k in cloud.point_data.keys()}
    return P, n_wall, src, vel_arms, prs_arms, phases, arrays, report_v, report_p


def stl_sign(stl, interior: np.ndarray) -> float:
    import pyvista as pv

    rng = np.random.default_rng(0)
    samp = interior[rng.choice(len(interior), min(20000, len(interior)), replace=False)]
    sd = np.asarray(pv.PolyData(samp).compute_implicit_distance(stl).point_data["implicit_distance"])
    frac_neg = float((sd < 0).mean())
    if min(frac_neg, 1.0 - frac_neg) > 0.05:
        raise RuntimeError(f"STL orientation ambiguous: {frac_neg:.3f} of interior points have sdf<0")
    sign = 1.0 if frac_neg > 0.5 else -1.0
    log(f"STL orientation: frac(sdf<0)={frac_neg:.3f} → sign {sign:+.0f}")
    return sign, frac_neg


def hexes_inside(P: np.ndarray, stl, sign: float, pitch: float):
    """Hexes of size ``pitch`` whose centre lies inside the lumen."""
    import pyvista as pv

    bmin = P.min(0) - pitch
    bmax = P.max(0) + pitch
    axes = [np.arange(bmin[d], bmax[d] + 0.5 * pitch, pitch) for d in range(3)]
    centers = [0.5 * (ax[:-1] + ax[1:]) for ax in axes]
    X, Y, Z = np.meshgrid(*centers, indexing="ij")
    xyz = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    t = time.time()
    sdf = sign * np.asarray(pv.PolyData(xyz).compute_implicit_distance(stl).point_data["implicit_distance"])
    inside = (sdf < 0.0).reshape(X.shape)
    log(f"SDF {xyz.shape[0]} voxel centres, {int(inside.sum())} inside ({time.time() - t:.1f}s)")

    node_of: dict[tuple[int, int, int], int] = {}
    points: list[tuple[float, float, float]] = []
    hexes: list[list[int]] = []
    xs, ys, zs = axes
    for i, j, k in zip(*np.nonzero(inside)):
        ids = []
        for di, dj, dk in HEX_CORNERS:
            key = (int(i + di), int(j + dj), int(k + dk))
            slot = node_of.get(key)
            if slot is None:
                slot = len(points)
                node_of[key] = slot
                points.append((float(xs[key[0]]), float(ys[key[1]]), float(zs[key[2]])))
            ids.append(slot)
        hexes.append(ids)
    return np.asarray(points, np.float64), np.asarray(hexes, np.int64), int(inside.sum())


def idw(src: np.ndarray, values: dict[str, np.ndarray], dst: np.ndarray, k: int = 8):
    from scipy.spatial import cKDTree

    t = time.time()
    dist, idx = cKDTree(src).query(dst, k=k, workers=-1)
    dist = np.maximum(dist, 1e-3)
    w = 1.0 / dist.astype(np.float64) ** 2
    w /= w.sum(axis=1, keepdims=True)
    out = {}
    for name, arr in values.items():
        a = np.asarray(arr)
        if a.ndim == 1:
            out[name] = (a[idx] * w).sum(axis=1).astype(np.float32)
        else:
            out[name] = np.einsum("nk,nkc->nc", w, a[idx]).astype(np.float32)
    log(f"IDW k={k} onto {len(dst)} nodes, {len(out)} arrays ({time.time() - t:.1f}s), "
        f"nearest-neighbour mm p50={np.median(dist[:, 0]):.2f} p95={np.percentile(dist[:, 0], 95):.2f}")
    return out, dist[:, 0]


def make_hex_grid(points: np.ndarray, hexes: np.ndarray, arrays: dict[str, np.ndarray]):
    import pyvista as pv

    cells = np.hstack([np.full((len(hexes), 1), 8, np.int64), hexes]).ravel()
    ug = pv.UnstructuredGrid(cells, np.full(len(hexes), pv.CellType.HEXAHEDRON, np.uint8), points)
    for name, arr in arrays.items():
        ug.point_data[name] = arr
    return ug


def exterior_quads(hexes: np.ndarray) -> int:
    faces = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (2, 3, 7, 6), (0, 3, 7, 4), (1, 2, 6, 5)]
    acc: dict[tuple[int, ...], int] = {}
    for h in hexes:
        for f in faces:
            key = tuple(sorted(int(h[i]) for i in f))
            acc[key] = acc.get(key, 0) + 1
    return sum(1 for n in acc.values() if n == 1)


def main(argv=None) -> int:
    import argparse
    import pyvista as pv

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case-dir", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--pitch-mm", type=float, default=2.5)
    ap.add_argument("--stl", type=Path, default=None)
    args = ap.parse_args(argv)

    t0 = time.time()
    case_dir = args.case_dir.resolve()
    out = (args.out or case_dir / "sliceable_tets" / "ensight_hex").resolve()
    stl_path = (args.stl or case_dir / "aligned_geometry.stl").resolve()
    out.mkdir(parents=True, exist_ok=True)

    P, n_wall, src, vel_arms, prs_arms, phases, arrays, report_v, report_p = collect_source(case_dir)
    log(f"source {n_wall} wall + {len(P) - n_wall} interior, {len(arrays)} arrays")
    stl = pv.read(stl_path)
    sign, frac_neg = stl_sign(stl, P[n_wall:])

    pitch = float(args.pitch_mm)
    points, hexes, n_in = hexes_inside(P, stl, sign, pitch)
    if len(hexes) > CLIENT_ELEMENT_BUDGET:
        pitch *= (len(hexes) / 26000) ** (1.0 / 3.0)
        log(f"{len(hexes)} hexes over budget, retry pitch {pitch:.2f} mm")
        points, hexes, n_in = hexes_inside(P, stl, sign, pitch)
    if len(hexes) > CLIENT_ELEMENT_BUDGET:
        raise SystemExit(f"still {len(hexes)} hexes at pitch {pitch:.2f} mm; raise --pitch-mm")
    n_ext = exterior_quads(hexes)
    log(f"hexes {len(hexes)}, nodes {len(points)}, exterior quads {n_ext}, pitch {pitch:.2f} mm")

    node_arrays, nn = idw(P, arrays, points)
    ug = make_hex_grid(points, hexes, node_arrays)
    vtu = out / "lumen_hex.vtu"
    ug.save(vtu)
    log(f"wrote {vtu.name} ({vtu.stat().st_size / 1e6:.1f} MB)")

    report: dict = {}
    bt.write_ensight(ug, out, "lumen_hex", report)

    origin, normal, plane_src = bt.slice_plane(case_dir, P)
    sl = ug.slice(normal=normal, origin=origin).triangulate()
    n_slice = int(sl.n_cells)
    sl_ug = sl.cast_to_unstructured_grid()
    bt.write_ensight(sl_ug, out, "peak_slice", report)
    log(f"example slice triangles {n_slice}")

    bt.preview(ug, ug, origin, normal, ["peak"], vel_arms, prs_arms, out / "plots",
               suptitle=f"{pitch:.1f} mm lumen hex, plane origin={np.round(origin,1).tolist()} normal={np.round(normal,3).tolist()}")

    summary = dict(
        tool="build_ensight_hexgrid.py", version=TOOL_VERSION, case_dir=str(case_dir), out=str(out),
        why="EnSight client draws parts above ~48k elements as points; this grid stays under that",
        display_only="IDW from the postview point cloud, same role as wss_deploy fillSection; formal R²/MAE stay on the same-point CSV",
        pitch_mm=pitch, n_hex=int(len(hexes)), n_nodes=int(len(points)),
        exterior_quads=int(n_ext), example_slice_triangles=n_slice,
        volume_mm3=float(len(hexes) * pitch ** 3),
        fluent_lumen_mm3=369368.0,
        source=dict(path=src, n_wall=n_wall, n_interior=int(len(P) - n_wall), n_arrays=len(arrays)),
        idw=dict(k=8, power=2, nn_mm=dict(p50=float(np.median(nn)), p95=float(np.percentile(nn, 95)))),
        stl=dict(path=str(stl_path), sign_applied=sign, interior_sample_frac_negative=frac_neg),
        velocity=report_v.get("velocity"), pressure=report_p.get("pressure"),
        preview_plane=dict(origin_mm=origin.tolist(), normal=normal.tolist(), source=plane_src),
        ensight=report.get("ensight"),
        open=[
            "EnSight: File → Open ensight_hex/lumen_hex.0.case (speed and pressure together)",
            "Hide nothing; the shell should be a solid surface, not a point cloud",
            "创建 → 剪切 → XYZ, then hide VTK Part and keep the clip",
            "peak_slice.0.case is one already-filled plane if you only want to confirm the face",
        ],
        elapsed_s=round(time.time() - t0, 1),
    )
    (out / "build_report.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    log(f"done in {summary['elapsed_s']} s, slice triangles {n_slice}, volume {summary['volume_mm3']:.0f} mm³")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
