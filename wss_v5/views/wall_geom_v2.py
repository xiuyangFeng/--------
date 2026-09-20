"""Local differential-geometry sidecar for the ``wss_min_view_v1`` wall point cloud.

Adds the features the V6 single-frame matrix needs but the frozen view does not
carry: principal curvatures of the wall surface at two neighbourhood scales and
the angle between the centreline tangent and the wall normal.

Everything here is computable from the **deployment inputs alone** (the wall
point cloud plus the signed-off centreline atlas): curvatures come from a local
quadric fit in the PCA-normal tangent frame, never from the CFD face topology.
The per-node mesh dual area is stored as ``mesh_area_mm2`` for auditing only and
is deliberately *not* registered as a model feature: it is a mesh-refinement
artefact, not patient geometry.

Rows are the ``valid`` wall nodes in master order, i.e. exactly the rows of the
view's ``bundle.npz`` (verified against ``wall_node_id_cas`` for every case).

Sign convention: curvature is positive where the surface is convex along the
outward normal, so a straight tube of radius R gives ``k1 = 1/R``, ``k2 = 0``.

    python -m wss_v5.views.wall_geom_v2 --workers 12
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
from scipy.spatial import cKDTree

from wss_pinn.utils import ROOT, utc_now

from .. import contract as C

PACK_NAME = "wss_min_geom_v2"
PACK_VERSION = "v2.0"
VIEW_ROOT = ROOT / "data_wss_v5" / "views" / "wss_min_view_v1"
OUT_ROOT = ROOT / "data_wss_v5" / "views" / PACK_NAME
FINE_K = 32          # ~1.3 mm neighbourhood at the 0.41 mm median wall spacing
COARSE_K = 128       # ~2.6 mm neighbourhood
CURV_CLIP_PER_MM = 2.0   # |k| beyond 2 /mm (R < 0.5 mm) is a fit artefact, not anatomy
CHUNK = 8192


def _unit_rows(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.clip(n, 1e-12, None)


def _tangent_basis(normals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Any orthonormal pair spanning the tangent plane (principal curvatures are basis independent)."""
    helper = np.where(
        (np.abs(normals[:, 0]) < 0.9)[:, None],
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
    )
    t1 = _unit_rows(np.cross(normals, helper))
    t2 = np.cross(normals, t1)
    return t1, t2


def principal_curvatures(xyz: np.ndarray, normals: np.ndarray, neighbours: np.ndarray) -> dict[str, np.ndarray]:
    """Least-squares Monge patch ``w = a u^2/2 + b uv + c v^2/2 + d u + e v + f`` per point."""
    n_points = xyz.shape[0]
    t1, t2 = _tangent_basis(normals)
    k1 = np.empty(n_points)
    k2 = np.empty(n_points)
    residual = np.empty(n_points)
    radius = np.empty(n_points)
    for start in range(0, n_points, CHUNK):
        stop = min(start + CHUNK, n_points)
        nb = neighbours[start:stop]                       # (m, k) excluding self
        delta = xyz[nb] - xyz[start:stop, None, :]        # (m, k, 3)
        u = np.einsum("mkd,md->mk", delta, t1[start:stop])
        v = np.einsum("mkd,md->mk", delta, t2[start:stop])
        w = np.einsum("mkd,md->mk", delta, normals[start:stop])
        design = np.stack([0.5 * u * u, u * v, 0.5 * v * v, u, v, np.ones_like(u)], axis=-1)
        gram = np.einsum("mki,mkj->mij", design, design)
        rhs = np.einsum("mki,mk->mi", design, w)
        scale = np.trace(gram, axis1=1, axis2=2)[:, None, None] / 6.0
        gram = gram + 1e-9 * np.clip(scale, 1e-12, None) * np.eye(6)
        coef = np.linalg.solve(gram, rhs)                 # (m, 6)
        fit = np.einsum("mki,mi->mk", design, coef)
        residual[start:stop] = np.sqrt(np.mean((fit - w) ** 2, axis=1))
        radius[start:stop] = np.max(np.sqrt(u * u + v * v), axis=1)
        a, b, c, du, dv = coef[:, 0], coef[:, 1], coef[:, 2], coef[:, 3], coef[:, 4]
        first_e, first_f, first_g = 1.0 + du * du, du * dv, 1.0 + dv * dv
        denom = np.sqrt(1.0 + du * du + dv * dv)
        second_l, second_m, second_n = a / denom, b / denom, c / denom
        det_first = np.clip(first_e * first_g - first_f * first_f, 1e-12, None)
        mean_h = (first_e * second_n - 2.0 * first_f * second_m + first_g * second_l) / (2.0 * det_first)
        gauss_k = (second_l * second_n - second_m * second_m) / det_first
        root = np.sqrt(np.clip(mean_h * mean_h - gauss_k, 0.0, None))
        # eigenvalues of the Weingarten map, then flipped so convex-outward is positive
        k1[start:stop] = -(mean_h - root)
        k2[start:stop] = -(mean_h + root)
    clipped = float(np.mean((np.abs(k1) > CURV_CLIP_PER_MM) | (np.abs(k2) > CURV_CLIP_PER_MM)))
    k1 = np.clip(k1, -CURV_CLIP_PER_MM, CURV_CLIP_PER_MM)
    k2 = np.clip(k2, -CURV_CLIP_PER_MM, CURV_CLIP_PER_MM)
    k_max = np.maximum(k1, k2)
    k_min = np.minimum(k1, k2)
    curvedness = np.sqrt(0.5 * (k_max * k_max + k_min * k_min))
    spread = k_max - k_min
    shape_index = np.where(spread > 1e-9, (2.0 / np.pi) * np.arctan((k_max + k_min) / np.clip(spread, 1e-9, None)), 0.0)
    return {
        "k1": k_max, "k2": k_min,
        "mean": 0.5 * (k_max + k_min), "gauss": k_max * k_min,
        "curvedness": curvedness, "shape_index": shape_index,
        "_fit_residual_mm": residual, "_neighbourhood_mm": radius, "_clipped_fraction": clipped,
    }


def build_case(canonical_id: str, out_root: Path = OUT_ROOT, view_root: Path = VIEW_ROOT) -> dict[str, Any]:
    started = time.time()
    bundle_path = Path(view_root) / canonical_id / "bundle.npz"
    with np.load(bundle_path, allow_pickle=True) as bundle:
        bundle_node_id = bundle["wall_node_id_cas"].astype(np.int64)
        atlas_radius_mm = bundle["wall_local_radius"].astype(np.float64)
    with h5py.File(C.case_dir(canonical_id) / "case.h5", "r") as h5:
        wall = h5["wall_static"]
        keep = wall["valid"][()].astype(bool)
        xyz = wall["xyz_mm"][()][keep]
        normals = _unit_rows(wall["normal_out_pca"][()][keep].astype(np.float64))
        node_id = wall["node_id_cas"][()][keep].astype(np.int64)
        atlas_row = wall["atlas_row"][()][keep].astype(np.int64)
        mesh_area_mm2 = wall["area_m2"][()][keep] * (C.LENGTH_M_TO_MM ** 2)
        table = h5["geometry/atlas_table"][()]
        columns = json.loads(h5["geometry"].attrs["atlas_columns"])
    if not np.array_equal(node_id, bundle_node_id):
        raise ValueError(f"{canonical_id}: sidecar rows do not match the frozen view rows")

    tree = cKDTree(xyz)
    max_k = min(COARSE_K, xyz.shape[0] - 1)
    _, neighbours = tree.query(xyz, k=max_k + 1, workers=-1)
    neighbours = neighbours[:, 1:]                                  # drop self
    fine = principal_curvatures(xyz, normals, neighbours[:, : min(FINE_K, max_k)])
    coarse = principal_curvatures(xyz, normals, neighbours)

    tangent = table[:, [columns.index("tangent_x"), columns.index("tangent_y"), columns.index("tangent_z")]]
    tangent = _unit_rows(np.asarray(tangent, dtype=np.float64))[atlas_row]
    tn_dot = np.einsum("nd,nd->n", tangent, normals)

    payload = {
        "wall_node_id_cas": node_id,
        "wall_tn_dot": tn_dot.astype(np.float32),
        "wall_mesh_area_mm2": mesh_area_mm2.astype(np.float32),   # audit only, never a model feature
    }
    for suffix, pack in (("", fine), ("_c", coarse)):
        # key names must equal the training feature names prefixed with wall_ (config.V6_SURFACE_FEATURE_KEYS)
        for name in ("k1", "k2", "mean", "gauss"):
            payload[f"wall_curv_{name}{suffix}"] = pack[name].astype(np.float32)
        payload[f"wall_curvedness{suffix}"] = pack["curvedness"].astype(np.float32)
        payload[f"wall_shape_index{suffix}"] = pack["shape_index"].astype(np.float32)
    out_dir = out_root / canonical_id
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / "features.tmp.npz"
    np.savez(tmp, **payload)
    tmp.replace(out_dir / "features.npz")

    tube = (atlas_radius_mm > 1e-6) & (np.abs(fine["shape_index"]) < 0.95)
    report = {
        "canonical_id": canonical_id, "pack": PACK_NAME, "version": PACK_VERSION,
        "n_points": int(xyz.shape[0]), "seconds": round(time.time() - started, 2),
        "fine_k": int(min(FINE_K, max_k)), "coarse_k": int(max_k),
        "fine_neighbourhood_mm_median": float(np.median(fine["_neighbourhood_mm"])),
        "coarse_neighbourhood_mm_median": float(np.median(coarse["_neighbourhood_mm"])),
        "fine_fit_residual_mm_median": float(np.median(fine["_fit_residual_mm"])),
        "coarse_fit_residual_mm_median": float(np.median(coarse["_fit_residual_mm"])),
        "fine_clipped_fraction": fine["_clipped_fraction"],
        "coarse_clipped_fraction": coarse["_clipped_fraction"],
        # sanity: on a tube k1 ~ 1/R_atlas, so k1 * R_atlas should sit near 1
        "k1_times_atlas_radius_median": float(np.median(fine["k1"][tube] * atlas_radius_mm[tube])),
        "k1_c_times_atlas_radius_median": float(np.median(coarse["k1"][tube] * atlas_radius_mm[tube])),
        "abs_tn_dot_median": float(np.median(np.abs(tn_dot))),
        "abs_tn_dot_p95": float(np.percentile(np.abs(tn_dot), 95)),
        "nonfinite": int(sum(int((~np.isfinite(v)).sum()) for k, v in payload.items() if k != "wall_node_id_cas")),
    }
    (out_dir / "geom_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT_ROOT))
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--cases", nargs="*", default=None)
    ap.add_argument("--view-root", default=str(VIEW_ROOT), help="wss_min_view bundles root (2026-09-15: v5.1 refresh writes views to a new root)")
    args = ap.parse_args(argv)
    out_root = Path(args.out)
    view_root = Path(args.view_root)
    cases = args.cases or sorted(
        str(p.parent.relative_to(view_root)) for p in view_root.glob("*/*/*/bundle.npz")
    )
    started = time.time()
    reports: list[dict[str, Any]] = []
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(build_case, cid, out_root, view_root) for cid in cases]
            for future in as_completed(futures):
                report = future.result()
                reports.append(report)
                print(f"[wall_geom_v2] {len(reports)}/{len(cases)} {report['canonical_id']} "
                      f"n={report['n_points']} k1R={report['k1_times_atlas_radius_median']:.2f} "
                      f"|t.n|={report['abs_tn_dot_median']:.3f}", flush=True)
    else:
        for i, cid in enumerate(cases):
            report = build_case(cid, out_root, view_root)
            reports.append(report)
            print(f"[wall_geom_v2] {i + 1}/{len(cases)} {cid}", flush=True)
    reports.sort(key=lambda r: r["canonical_id"])
    manifest = {
        "pack": PACK_NAME, "version": PACK_VERSION, "created_at": utc_now(),
        "view_root": str(view_root), "snapshot_root": str(C.SNAPSHOT_ROOT),
        "geometry_program_version": C.GEOMETRY_PROGRAM_VERSION,
        "method": ("principal curvatures from a least-squares Monge quadric in the PCA-normal tangent frame; "
                   "positive = convex along the outward normal (tube of radius R -> k1 = 1/R, k2 = 0); "
                   f"fine k={FINE_K}, coarse k={COARSE_K}, |k| clipped at {CURV_CLIP_PER_MM} /mm"),
        "deployment_inputs_only": True,
        "audit_only_fields": ["wall_mesh_area_mm2"],
        "cases": len(reports), "seconds": round(time.time() - started, 1),
        "reports": reports,
    }
    Path(out_root).mkdir(parents=True, exist_ok=True)
    (Path(out_root) / "geom_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"[wall_geom_v2] wrote {len(reports)} cases to {out_root} in {manifest['seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
