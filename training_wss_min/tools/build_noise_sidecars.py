"""Precompute boundary-noise views of every wall cloud for training-time augmentation (2026-09-16).

For each case and each sigma (mm): all wall nodes are displaced along the outward PCA normal by a spatially
correlated Gaussian field (correlation length --corr-mm, default 2 mm, the deployment_geometry_sensitivity
recipe), then the deployment-side inputs that depend on the cloud are recomputed on the noisy cloud exactly
as build_density_sidecars does: PCA normals (centreline oriented), the two principal-curvature families and
tn_dot.  Positions are stored in the view's normalised anatomical frame (same transform as bundle.npz).

Output: <out>/S<sigma>/<case>/features.npz with rows (= arange(n)), wall_pos_norm, wall_normal_pca_aligned,
wall_curv_* / wall_curvedness* / wall_shape_index* (both families), wall_tn_dot.  Deployment inputs only.
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
from training_wss_min.tools.deployment_geometry_sensitivity import gaussian_weights
from training_wss_min.tools.deployment_resample_reeval import atlas_from_h5
from wss_v5.pointcloud import pca_normals
from wss_v5.views.wall_geom_v2 import COARSE_K, FINE_K, _unit_rows, principal_curvatures

ROOT = C.PROJECT_ROOT
VIEW = ROOT / "data_wss_v5/views/wss_min_view_v1"
OUT_ROOT = ROOT / "data_wss_v5/views/wss_min_noise_v1"
PACK_VERSION = "v1.0"
SEED = 20260916
CORR_MM = 2.0


def build_one(cid: str, sigmas: list[str], out_root: Path, seed: int, corr_mm: float, view_root: Path = VIEW) -> dict:
    started = time.time()
    bundle_path = Path(view_root) / cid / "bundle.npz"
    with np.load(bundle_path, allow_pickle=True) as z, h5py.File(NG.source_h5_for_bundle(bundle_path), "r") as h:
        ids = z["wall_node_id_cas"].astype(np.int64)
        rotation = z["transform_rotation"].astype(np.float64)
        origin = z["transform_centroid"].astype(np.float64)
        scale = float(z["coord_scale"])
        pos_clean = z["wall_coords_norm"].astype(np.float64)
        ref_normals = _unit_rows(z["wall_normal_pca_aligned"].astype(np.float64))
        src = h["wall_static/node_id_cas"][:].astype(np.int64)
        order = np.argsort(src)
        at = np.searchsorted(src[order], ids)
        if np.any(at >= len(order)) or not np.array_equal(src[order][at], ids):
            raise ValueError(f"wall node identity mismatch: {bundle_path}")
        rows_h5 = order[at]
        xyz_all = h["wall_static/xyz_mm"][:][rows_h5].astype(np.float64)
        n_out = _unit_rows(h["wall_static/normal_out_pca"][:][rows_h5].astype(np.float64))
        atlas_row = h["wall_static/atlas_row"][:][rows_h5].astype(np.int64)
        atlas = atlas_from_h5(h)
        tangent_all = atlas.tangent[atlas_row]
    check = (xyz_all - origin) @ rotation.T / scale
    if not np.allclose(check, pos_clean, atol=2e-4):
        raise ValueError(f"{cid}: bundle transform does not reproduce wall_coords_norm (max |d| {np.abs(check - pos_clean).max():.2e})")
    W = gaussian_weights(xyz_all, corr_mm)
    report = {"canonical_id": cid, "n": int(len(ids)), "sigmas": {}}
    for si, tag in enumerate(sigmas):
        sigma = float(tag)
        rng = np.random.default_rng(seed + 7919 * si + (hash(cid) % 1000))
        field = W @ rng.standard_normal(len(ids))
        field = field / max(float(field.std()), 1e-12) * sigma
        xyz = xyz_all + field[:, None] * n_out
        normals, variation = pca_normals(xyz, atlas)
        normals = _unit_rows(np.asarray(normals, dtype=np.float64))
        tree = cKDTree(xyz)
        max_k = min(COARSE_K, len(xyz) - 1)
        _, nb = tree.query(xyz, k=max_k + 1, workers=1)
        nb = nb[:, 1:]
        fine = principal_curvatures(xyz, normals, nb[:, : min(FINE_K, max_k)])
        coarse = principal_curvatures(xyz, normals, nb)
        payload = {"rows": np.arange(len(ids), dtype=np.int64), "wall_node_id_cas": ids,
                   "wall_pos_norm": ((xyz - origin) @ rotation.T / scale).astype(np.float32),
                   "wall_normal_pca_aligned": (normals @ rotation.T).astype(np.float32),
                   "wall_tn_dot": np.einsum("nd,nd->n", tangent_all, normals).astype(np.float32)}
        for suffix, pack in (("", fine), ("_c", coarse)):
            for name in ("k1", "k2", "mean", "gauss"):
                payload[f"wall_curv_{name}{suffix}"] = pack[name].astype(np.float32)
            payload[f"wall_curvedness{suffix}"] = pack["curvedness"].astype(np.float32)
            payload[f"wall_shape_index{suffix}"] = pack["shape_index"].astype(np.float32)
        for key, value in payload.items():
            if not np.isfinite(value).all():
                raise ValueError(f"{cid} S{tag}: non-finite {key}")
        out_dir = out_root / f"S{tag}" / cid
        out_dir.mkdir(parents=True, exist_ok=True)
        tmp = out_dir / "features.tmp.npz"
        np.savez(tmp, **payload)
        tmp.replace(out_dir / "features.npz")
        report["sigmas"][tag] = {
            "sigma_mm": sigma, "displacement_abs_mean_mm": float(np.abs(field).mean()), "displacement_abs_p95_mm": float(np.quantile(np.abs(field), 0.95)),
            "normal_cos_vs_clean_median": float(np.median(np.abs(np.einsum("nd,nd->n", normals @ rotation.T, ref_normals)))),
            "surface_variation_median": float(np.median(variation)),
        }
    report["seconds"] = round(time.time() - started, 1)
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sigmas", nargs="*", default=["0.1", "0.2", "0.3"], help="sigma tags in mm; directory names are S<tag>")
    ap.add_argument("--corr-mm", type=float, default=CORR_MM)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--out", default=str(OUT_ROOT))
    ap.add_argument("--cases", nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--view-root", default=str(VIEW))
    args = ap.parse_args(argv)
    out_root = Path(args.out)
    view = Path(args.view_root)
    cases = args.cases or sorted(str(p.parent.relative_to(view)) for p in view.glob("*/*/*/bundle.npz"))
    started = time.time()
    reports = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(build_one, cid, args.sigmas, out_root, args.seed, args.corr_mm, view) for cid in cases]
        for future in as_completed(futures):
            r = future.result()
            reports.append(r)
            print(f"[noise] {len(reports)}/{len(cases)} {r['canonical_id']} n={r['n']} "
                  + " ".join(f"S{t}:|d|={v['displacement_abs_mean_mm']:.3f}" for t, v in r["sigmas"].items()) + f" {r['seconds']}s", flush=True)
    reports.sort(key=lambda r: r["canonical_id"])
    manifest = {"pack": "wss_min_noise_v1", "version": PACK_VERSION, "sigmas": args.sigmas, "corr_mm": args.corr_mm, "seed": args.seed,
                "view_root": str(view), "noise": "spatially correlated Gaussian field (row-normalised Gaussian weights, 3 sigma window) along the outward PCA normal, rescaled to std sigma",
                "recomputed": ["wall_pos_norm (bundle transform)", "wall_normal_pca_aligned (pca_normals k=16 + centreline orientation)",
                               "principal curvature families k=32/128 (wall_geom_v2.principal_curvatures)", "wall_tn_dot"],
                "deployment_inputs_only": True, "cases": len(reports), "seconds": round(time.time() - started, 1), "reports": reports}
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "noise_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps({k: v for k, v in manifest.items() if k != "reports"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
