"""Multi-density sidecars for density augmentation (deployment robustness, 2026-09-15).

For every case and every density level the valid wall nodes are voxel-decimated (one point per occupied voxel,
nearest the voxel centroid; same recipe as tools/deployment_resample_reeval) and the cloud-derived inputs are
recomputed on the decimated cloud exactly as a deployment would: PCA normals with centreline orientation (k=16) and
the principal-curvature families at fixed k=32/128 plus tn_dot.  Centreline-projected inputs are per-point and are
simply subset at training time.

Output: data_wss_v5/views/wss_min_density_v1/L<pct>/<case>/features.npz with
  rows                (n_sub,)  int64   row indices into the frozen view bundle (valid wall rows)
  wall_node_id_cas    (n_sub,)  int64   node identity of those rows (audit)
  wall_normal_pca_aligned (n_sub, 3) float32   recomputed normals in the aligned frame
  wall_curv_k1, wall_curv_k2, wall_curv_gauss, wall_curvedness, wall_curv_k1_c, wall_curv_k2_c, wall_curv_gauss_c,
  wall_curvedness_c, wall_tn_dot, wall_shape_index, wall_curv_mean, wall_shape_index_c, wall_curv_mean_c (float32)

    python -m training_wss_min.tools.build_density_sidecars --levels 70 50 35 25 --workers 24
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial import cKDTree

from training_wss_min import config as C
from training_wss_min import next_geometry as NG
from training_wss_min.tools.deployment_resample_reeval import atlas_from_h5, voxel_rows
from wss_v5.pointcloud import pca_normals
from wss_v5.views.wall_geom_v2 import COARSE_K, FINE_K, _unit_rows, principal_curvatures

ROOT = C.PROJECT_ROOT
VIEW = ROOT / "data_wss_v5/views/wss_min_view_v1"
OUT_ROOT = ROOT / "data_wss_v5/views/wss_min_density_v1"
PACK_VERSION = "v1.0"
SEED = 20260915


def build_one(cid: str, levels: list[int], out_root: Path, seed: int, view_root: Path = VIEW) -> dict:
    started = time.time()
    bundle_path = Path(view_root) / cid / "bundle.npz"
    with np.load(bundle_path, allow_pickle=True) as z, h5py.File(NG.source_h5_for_bundle(bundle_path), "r") as h:
        ids = z["wall_node_id_cas"].astype(np.int64)
        rotation = z["transform_rotation"].astype(np.float64)
        ref_normals = _unit_rows(z["wall_normal_pca_aligned"].astype(np.float64))
        src = h["wall_static/node_id_cas"][:].astype(np.int64)
        order = np.argsort(src)
        at = np.searchsorted(src[order], ids)
        if np.any(at >= len(order)) or not np.array_equal(src[order][at], ids):
            raise ValueError(f"wall node identity mismatch: {bundle_path}")
        rows_h5 = order[at]
        xyz_all = h["wall_static/xyz_mm"][:][rows_h5].astype(np.float64)
        atlas_row = h["wall_static/atlas_row"][:][rows_h5].astype(np.int64)
        atlas = atlas_from_h5(h)
        tangent_all = atlas.tangent[atlas_row]
    report = {"canonical_id": cid, "n_full": int(len(ids)), "levels": {}}
    for li, level in enumerate(levels):
        rows = voxel_rows(xyz_all, level / 100.0, seed + 7919 * li + (hash(cid) % 1000))
        xyz = xyz_all[rows]
        normals, variation = pca_normals(xyz, atlas)
        normals = _unit_rows(np.asarray(normals, dtype=np.float64))
        tree = cKDTree(xyz)
        max_k = min(COARSE_K, len(xyz) - 1)
        _, nb = tree.query(xyz, k=max_k + 1, workers=1)
        nb = nb[:, 1:]
        fine = principal_curvatures(xyz, normals, nb[:, : min(FINE_K, max_k)])
        coarse = principal_curvatures(xyz, normals, nb)
        payload = {"rows": rows.astype(np.int64), "wall_node_id_cas": ids[rows],
                   "wall_normal_pca_aligned": (normals @ rotation.T).astype(np.float32),
                   "wall_tn_dot": np.einsum("nd,nd->n", tangent_all[rows], normals).astype(np.float32)}
        for suffix, pack in (("", fine), ("_c", coarse)):
            for name in ("k1", "k2", "mean", "gauss"):
                payload[f"wall_curv_{name}{suffix}"] = pack[name].astype(np.float32)
            payload[f"wall_curvedness{suffix}"] = pack["curvedness"].astype(np.float32)
            payload[f"wall_shape_index{suffix}"] = pack["shape_index"].astype(np.float32)
        for key, value in payload.items():
            if not np.isfinite(value).all():
                raise ValueError(f"{cid} L{level}: non-finite {key}")
        out_dir = out_root / f"L{level}" / cid
        out_dir.mkdir(parents=True, exist_ok=True)
        tmp = out_dir / "features.tmp.npz"
        np.savez(tmp, **payload)
        tmp.replace(out_dir / "features.npz")
        report["levels"][str(level)] = {
            "n": int(len(rows)), "fraction": float(len(rows) / len(ids)),
            "normal_cos_vs_full_median": float(np.median(np.abs(np.einsum("nd,nd->n", normals @ rotation.T, ref_normals[rows])))),
            "fine_neighbourhood_mm_median": float(np.median(fine["_neighbourhood_mm"])),
            "coarse_neighbourhood_mm_median": float(np.median(coarse["_neighbourhood_mm"])),
            "surface_variation_median": float(np.median(variation)),
        }
    report["seconds"] = round(time.time() - started, 1)
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", nargs="*", type=int, default=[70, 50, 35, 25])
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--out", default=str(OUT_ROOT))
    ap.add_argument("--cases", nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--view-root", default=str(VIEW), help="wss_min_view root (2026-09-15: v5.1 refresh uses a new root)")
    args = ap.parse_args(argv)
    out_root = Path(args.out)
    view = Path(args.view_root)
    cases = args.cases or sorted(str(p.parent.relative_to(view)) for p in view.glob("*/*/*/bundle.npz"))
    started = time.time()
    reports = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(build_one, cid, args.levels, out_root, args.seed, view) for cid in cases]
        for future in as_completed(futures):
            r = future.result()
            reports.append(r)
            print(f"[density] {len(reports)}/{len(cases)} {r['canonical_id']} n={r['n_full']} "
                  + " ".join(f"L{l}:{v['n']}" for l, v in r["levels"].items()) + f" {r['seconds']}s", flush=True)
    reports.sort(key=lambda r: r["canonical_id"])
    manifest = {"pack": "wss_min_density_v1", "version": PACK_VERSION, "levels": args.levels, "seed": args.seed,
                "view_root": str(view), "decimation": "voxel (one point per occupied voxel, nearest the centroid); voxel size by bisection to the target fraction",
                "recomputed": ["wall_normal_pca_aligned (pca_normals k=16 + centreline orientation)",
                               "principal curvature families k=32/128 (wall_geom_v2.principal_curvatures)", "wall_tn_dot"],
                "deployment_inputs_only": True, "cases": len(reports), "seconds": round(time.time() - started, 1), "reports": reports}
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "density_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"[density] wrote {len(reports)} cases x {len(args.levels)} levels to {out_root} in {manifest['seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
