"""Deployment-form re-evaluation of trained WSS models on resampled (coarser) wall clouds (inference only).

For every test case the valid wall nodes are decimated to a target fraction (voxel decimation = one point per
occupied voxel, i.e. a roughly uniform resampled cloud; ``--method random`` thins at random) and every
cloud-derived input is recomputed on the coarse cloud with the same geometry programs a deployment would run:

* PCA normals with centreline orientation (``wss_v5.pointcloud.pca_normals``, k=16) -> nx/ny/nz_aligned and the
  normal columns of the raw wall-geometry tensor;
* principal-curvature families at the same fixed neighbourhood sizes (``wss_v5.views.wall_geom_v2``, k=32 / 128)
  and ``tn_dot``.

Centreline-projected inputs (abscissa, radius, rho/theta, distances, Murray / capfit shares, stenosis index, ...)
are per-point functions of the signed-off centreline and keep their values on the retained points.  The model's own
neighbourhoods (support sampling, directional neighbourhoods, K16 query patch, section tokens) run on the coarse
cloud.  Truth is the CFD WSS at the retained nodes; the reference is the full-cloud prediction of the same
checkpoint restricted to the same nodes.  The ``frozen`` variant keeps the full-cloud feature values on the
retained points, isolating the feature-recompute effect from the encoder / patch density effect.  Fraction 1.0 with
recomputation is a control for the recompute path itself (it should reproduce the saved full-cloud predictions).

    CUDA_VISIBLE_DEVICES=3 python -m training_wss_min.tools.deployment_resample_reeval --out <dir>
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.spatial import cKDTree

from training_wss_min import config as C
from training_wss_min import dataset as D
from training_wss_min import evaluate as E
from training_wss_min import metrics as M
from training_wss_min import next_geometry as NG
from wss_v5.centerline_features import Atlas, _semantics
from wss_v5.pointcloud import pca_normals
from wss_v5.views.wall_geom_v2 import COARSE_K, FINE_K, _unit_rows, principal_curvatures

ROOT = C.PROJECT_ROOT
RUNS = ROOT / "training_wss_min/runs"
DEFAULT_RUNS = [
    RUNS / "wss_local_wave1_20260912/X5_s1234", RUNS / "wss_local_wave1b_20260912/X5_s7",
    RUNS / "wss_local_wave1b_20260912/X5_s2025", RUNS / "wss_local_wave4_20260913/X5_s11",
    RUNS / "wss_local_wave4_20260913/X5_s2026",
]
FINE_MAP = {"curv_k1": "k1", "curv_k2": "k2", "curv_mean": "mean", "curv_gauss": "gauss",
            "curvedness": "curvedness", "shape_index": "shape_index"}


def atlas_from_h5(h) -> Atlas:
    g = h["geometry"]
    table = np.asarray(g["atlas_table"][()], dtype=np.float64)
    columns = json.loads(g.attrs["atlas_columns"])
    segments = json.loads(g.attrs["atlas_segments"])
    xyz = table[:, [columns.index(c) for c in ("x_mm", "y_mm", "z_mm")]]
    if "is_child_duplicate" in columns:
        tree_rows = np.flatnonzero(table[:, columns.index("is_child_duplicate")] < 0.5)
    else:
        tree_rows = np.arange(len(table))
    return Atlas(table=table, columns=columns, segments=segments, semantic_of_segment=_semantics(segments),
                 frame_n=np.asarray(g["atlas_frame_n"][()]), frame_b=np.asarray(g["atlas_frame_b"][()]),
                 tree_rows=tree_rows, tree=cKDTree(xyz[tree_rows]), provenance={})


def wall_source(case: dict) -> dict:
    """Raw-frame wall coordinates, atlas tangents and the aligned-frame rotation for the frozen view rows."""
    bundle_path = Path(case["bundle_path"])
    with np.load(bundle_path, allow_pickle=True) as z, h5py.File(NG.source_h5_for_bundle(bundle_path), "r") as h:
        ids = z["wall_node_id_cas"].astype(np.int64)
        src = h["wall_static/node_id_cas"][:].astype(np.int64)
        order = np.argsort(src)
        at = np.searchsorted(src[order], ids)
        if np.any(at >= len(order)) or not np.array_equal(src[order][at], ids):
            raise ValueError(f"wall node identity mismatch: {bundle_path}")
        rows = order[at]
        xyz = h["wall_static/xyz_mm"][:][rows].astype(np.float64)
        atlas_row = h["wall_static/atlas_row"][:][rows].astype(np.int64)
        atlas = atlas_from_h5(h)
        return {"xyz": xyz, "atlas": atlas, "tangent": atlas.tangent[atlas_row],
                "rotation": z["transform_rotation"].astype(np.float64),
                "normal_aligned_ref": z["wall_normal_pca_aligned"].astype(np.float64)}


def voxel_rows(xyz: np.ndarray, frac: float, seed: int) -> np.ndarray:
    """One point per occupied voxel (the one nearest the voxel centroid); voxel size found by bisection."""
    n = len(xyz)
    target = max(int(round(n * frac)), 64)
    if frac >= 1.0:
        return np.arange(n)
    d, _ = cKDTree(xyz).query(xyz, k=2)
    spacing = float(np.median(d[:, 1]))
    lo, hi = spacing * 0.5, spacing * 40.0
    origin = xyz.min(axis=0) + np.random.default_rng(seed).uniform(0.0, spacing, 3)
    best = None
    for _ in range(40):
        h = 0.5 * (lo + hi)
        keys = np.floor((xyz - origin) / h).astype(np.int64)
        keys -= keys.min(axis=0)
        span = keys.max(axis=0) + 1
        # 1-D voxel code (collision-free: strides are the per-axis spans); groups are identical to np.unique(axis=0),
        # and the retained row per group (nearest the centroid, ties by row order) does not depend on group numbering.
        code = (keys[:, 0] * span[1] + keys[:, 1]) * span[2] + keys[:, 2]
        _, inv, counts = np.unique(code, return_inverse=True, return_counts=True)
        m = len(counts)
        if best is None or abs(m - target) < abs(best[0] - target):
            best = (m, inv.reshape(-1), counts)
        if abs(m - target) <= max(1, int(0.01 * target)):
            break
        if m > target:
            lo = h
        else:
            hi = h
    m, inv, counts = best
    centroid = np.zeros((m, 3))
    np.add.at(centroid, inv, xyz)
    centroid /= counts[:, None]
    dist = np.linalg.norm(xyz - centroid[inv], axis=1)
    order = np.lexsort((dist, inv))
    first = np.concatenate([[True], inv[order][1:] != inv[order][:-1]])
    return np.sort(order[first])


def _voxel_rows_reference(xyz: np.ndarray, frac: float, seed: int) -> np.ndarray:
    """Original implementation (row-wise np.unique); kept as the reference for tests/test_voxel_rows_fast.py."""
    n = len(xyz)
    target = max(int(round(n * frac)), 64)
    if frac >= 1.0:
        return np.arange(n)
    d, _ = cKDTree(xyz).query(xyz, k=2)
    spacing = float(np.median(d[:, 1]))
    lo, hi = spacing * 0.5, spacing * 40.0
    origin = xyz.min(axis=0) + np.random.default_rng(seed).uniform(0.0, spacing, 3)
    best = None
    for _ in range(40):
        h = 0.5 * (lo + hi)
        keys = np.floor((xyz - origin) / h).astype(np.int64)
        _, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
        m = len(counts)
        if best is None or abs(m - target) < abs(best[0] - target):
            best = (m, inv.reshape(-1), counts)
        if abs(m - target) <= max(1, int(0.01 * target)):
            break
        if m > target:
            lo = h
        else:
            hi = h
    m, inv, counts = best
    centroid = np.zeros((m, 3))
    np.add.at(centroid, inv, xyz)
    centroid /= counts[:, None]
    dist = np.linalg.norm(xyz - centroid[inv], axis=1)
    order = np.lexsort((dist, inv))
    first = np.concatenate([[True], inv[order][1:] != inv[order][:-1]])
    return np.sort(order[first])


def random_rows(n: int, frac: float, seed: int) -> np.ndarray:
    if frac >= 1.0:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, size=max(int(round(n * frac)), 64), replace=False))


def recompute_cloud_features(sub: dict, src: dict, rows: np.ndarray, input_features) -> dict:
    """Recompute PCA normals and curvature families on the coarse cloud and write them into the case dict."""
    xyz = src["xyz"][rows]
    normals, variation = pca_normals(xyz, src["atlas"])
    normals = _unit_rows(np.asarray(normals, dtype=np.float64))
    aligned = normals @ src["rotation"].T
    sub["nx_aligned"], sub["ny_aligned"], sub["nz_aligned"] = (aligned[:, i].astype(np.float32) for i in range(3))
    geometry = np.array(sub["local_geometry"], dtype=np.float32, copy=True)
    geometry[:, :3] = aligned.astype(np.float32)
    sub["local_geometry"] = geometry
    tree = cKDTree(xyz)
    max_k = min(COARSE_K, len(xyz) - 1)
    _, neighbours = tree.query(xyz, k=max_k + 1, workers=-1)
    neighbours = neighbours[:, 1:]
    fine = principal_curvatures(xyz, normals, neighbours[:, : min(FINE_K, max_k)])
    coarse = principal_curvatures(xyz, normals, neighbours)
    written = []
    for name in input_features:
        if name in FINE_MAP and name in sub:
            sub[name] = fine[FINE_MAP[name]].astype(np.float32); written.append(name)
        elif name.endswith("_c") and name[:-2] in FINE_MAP and name in sub:
            sub[name] = coarse[FINE_MAP[name[:-2]]].astype(np.float32); written.append(name)
    if "tn_dot" in input_features and "tn_dot" in sub:
        sub["tn_dot"] = np.einsum("nd,nd->n", src["tangent"][rows], normals).astype(np.float32); written.append("tn_dot")
    cos_ref = float(np.median(np.abs(np.einsum("nd,nd->n", aligned, _unit_rows(src["normal_aligned_ref"][rows])))))
    return {"written": written, "normal_cos_vs_full_median": cos_ref, "fine_k": int(min(FINE_K, max_k)), "coarse_k": int(max_k),
            "fine_neighbourhood_mm_median": float(np.median(fine["_neighbourhood_mm"])),
            "coarse_neighbourhood_mm_median": float(np.median(coarse["_neighbourhood_mm"])),
            "surface_variation_median": float(np.median(variation))}


def coarse_case(full: dict, rows: np.ndarray) -> dict:
    sub = D.subset_case(full, rows)
    for key in ("_full_wall_tree", "_full_wall_features", "section", "local_geometry"):
        pass
    sub.pop("_full_wall_tree", None)
    sub.pop("_full_wall_features", None)
    return sub


def tail_metrics(true_list, pred_list) -> dict:
    t = np.concatenate(true_list); p = np.concatenate(pred_list)
    q = np.quantile(t, 0.9); m = t >= q
    ious = []
    for tt, pp in zip(true_list, pred_list):
        a = tt >= np.percentile(tt, 90); b = pp >= np.percentile(pp, 90)
        ious.append(float((a & b).sum() / max((a | b).sum(), 1)))
    return {"high_wss_r2": float(1 - ((t[m] - p[m]) ** 2).sum() / ((t[m] - t[m].mean()) ** 2).sum()),
            "top10_ratio": float(p[m].mean() / t[m].mean()), "p99_ratio": float(np.quantile(p, 0.99) / np.quantile(t, 0.99)),
            "top10_iou_mean": float(np.mean(ious)), "mae_pooled": float(np.mean(np.abs(t - p)))}


def summarise(true_pa, pred_pa, true_n, pred_n) -> dict:
    out = {"pa_r2_cb": float(M.casebalanced_field_metrics(true_pa, pred_pa)["r2"]),
           "norm_r2_cb": float(M.casebalanced_field_metrics(true_n, pred_n)["r2"])}
    pc = [float(1 - ((t - p) ** 2).sum() / ((t - t.mean()) ** 2).sum()) for t, p in zip(true_pa, pred_pa)]
    out.update(case_r2_mean=float(np.mean(pc)), case_r2_p10=float(np.quantile(pc, 0.1)), n_negative=int(sum(x < 0 for x in pc)))
    out.update(tail_metrics(true_pa, pred_pa))
    out["per_case_r2"] = pc
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=[str(p) for p in DEFAULT_RUNS])
    ap.add_argument("--fractions", nargs="*", type=float, default=[1.0, 0.5, 0.25, 0.1])
    ap.add_argument("--method", choices=("voxel", "random"), default="voxel")
    ap.add_argument("--seed", type=int, default=20260913)
    ap.add_argument("--variants", nargs="*", default=["recompute", "frozen"])
    ap.add_argument("--checkpoint", default="best")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit-cases", type=int, default=0)
    args = ap.parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    models = []
    for run in args.runs:
        run = Path(run)
        cfg, feat_stats, model, _ = E.load_model_from_run(run, device, args.checkpoint)
        model.eval()
        models.append({"run": run, "name": run.name, "cfg": cfg, "feat_stats": feat_stats, "model": model,
                       "stats": E.load_wss_stats_for_run(run)})
    cfg0, stats0 = models[0]["cfg"], models[0]["stats"]
    for m in models[1:]:
        if tuple(m["cfg"].data.input_features) != tuple(cfg0.data.input_features) or m["cfg"].data.split_path != cfg0.data.split_path:
            raise ValueError("all runs must share input features and split")
    pairs = D.load_split_cases(cfg0.data.split_path, "test")
    if args.limit_cases:
        pairs = pairs[: args.limit_cases]
    variants = [(f, v) for f in args.fractions for v in args.variants if not (f >= 1.0 and v == "frozen")]
    results = {(f, v): {"true_pa": [], "true_n": [], "pred_n": {m["name"]: [] for m in models}, "ref_n": {m["name"]: [] for m in models},
                        "n_points": [], "recompute": []} for f, v in variants}
    started = time.time()
    for ci, (cohort, name) in enumerate(pairs):
        full = D.load_case(cohort, name, stats0, target="wss", target_normalization=cfg0.data.target_normalization,
                           data_root=cfg0.data.data_root, required_frame_version=cfg0.data.required_frame_version,
                           extra_point_features=C.v6_point_features(cfg0), point_features_root=cfg0.data.point_features_root)
        NG.case_geometry(full)
        if getattr(cfg0.data, "section_tokens", False):
            NG.case_section(full)
        src = wall_source(full)
        n = len(full["pos"])
        saved = {}
        for m in models:
            with np.load(m["run"] / "eval" / f"ckpt_{args.checkpoint}" / "predictions/test" / full["unit_id"] / "predictions.npz") as z:
                ref = np.full(n, np.nan); ref[z["row_index"]] = z["pred_norm"]
                saved[m["name"]] = ref
        for (frac, variant) in variants:
            seed = args.seed + ci
            rows = voxel_rows(src["xyz"], frac, seed) if args.method == "voxel" else random_rows(n, frac, seed)
            sub = coarse_case(full, rows)
            info = {}
            if variant == "recompute":
                info = recompute_cloud_features(sub, src, rows, cfg0.data.input_features)
            store = results[(frac, variant)]
            store["true_pa"].append(np.asarray(sub["y_raw"], dtype=np.float64))
            store["true_n"].append(np.asarray(sub["y_norm"], dtype=np.float64))
            store["n_points"].append(int(len(rows)))
            store["recompute"].append(info)
            for m in models:
                with torch.no_grad():
                    pred = E.predict_case_norm(m["model"], sub, m["cfg"].data.input_features, m["feat_stats"], device, cfg=m["cfg"])
                store["pred_n"][m["name"]].append(np.asarray(pred, dtype=np.float64))
                store["ref_n"][m["name"]].append(saved[m["name"]][rows])
        print(f"[{ci + 1}/{len(pairs)}] {full['unit_id']} n={n} {time.time() - started:.0f}s", flush=True)
    report = {"runs": [str(m["run"]) for m in models], "checkpoint": args.checkpoint, "method": args.method, "fractions": args.fractions,
              "variants": args.variants, "n_cases": len(pairs), "units": [f"{a}/{b}" for a, b in pairs], "rows": []}
    for (frac, variant), store in results.items():
        true_pa, true_n = store["true_pa"], store["true_n"]
        names = list(store["pred_n"])
        entry = {"fraction": frac, "variant": variant, "points_median": float(np.median(store["n_points"])),
                 "points_total": int(np.sum(store["n_points"])), "seeds": {}, "ensemble": None, "reference_ensemble": None}
        if variant == "recompute":
            infos = [i for i in store["recompute"] if i]
            entry["recompute"] = {k: float(np.median([i[k] for i in infos])) for k in
                                  ("normal_cos_vs_full_median", "fine_neighbourhood_mm_median", "coarse_neighbourhood_mm_median", "surface_variation_median")}
        for nm in names:
            pred_n = store["pred_n"][nm]; ref_n = store["ref_n"][nm]
            entry["seeds"][nm] = {"coarse": summarise(true_pa, [D.denormalize_wss(p, stats0) for p in pred_n], true_n, pred_n),
                                  "reference_same_points": summarise(true_pa, [D.denormalize_wss(p, stats0) for p in ref_n], true_n, ref_n)}
        ens = [np.mean([store["pred_n"][nm][i] for nm in names], axis=0) for i in range(len(true_pa))]
        ref = [np.mean([store["ref_n"][nm][i] for nm in names], axis=0) for i in range(len(true_pa))]
        entry["ensemble"] = summarise(true_pa, [D.denormalize_wss(p, stats0) for p in ens], true_n, ens)
        entry["reference_ensemble"] = summarise(true_pa, [D.denormalize_wss(p, stats0) for p in ref], true_n, ref)
        report["rows"].append(entry)
    (out_dir / "deployment_resample_reeval.json").write_text(json.dumps(report, indent=1))
    lines = ["| fraction | variant | points/case (median) | ens Pa R²_cb | ref Pa R²_cb (same points) | Δ | ens norm R²_cb | ref norm | case R² mean | P10 | MAE Pa | high-WSS R² | top10 比 | p99 比 | IoU | seed Pa R²_cb (mean ± sd) | normal cos vs full |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|"]
    for e in report["rows"]:
        s = e["ensemble"]; r = e["reference_ensemble"]
        seeds = [v["coarse"]["pa_r2_cb"] for v in e["seeds"].values()]
        lines.append(f"| {e['fraction']:.2f} | {e['variant']} | {e['points_median']:.0f} | {s['pa_r2_cb']:.4f} | {r['pa_r2_cb']:.4f} | {s['pa_r2_cb'] - r['pa_r2_cb']:+.4f} | "
                     f"{s['norm_r2_cb']:.4f} | {r['norm_r2_cb']:.4f} | {s['case_r2_mean']:.3f} | {s['case_r2_p10']:.3f} | {s['mae_pooled']:.3f} | {s['high_wss_r2']:.3f} | "
                     f"{s['top10_ratio']:.3f} | {s['p99_ratio']:.3f} | {s['top10_iou_mean']:.3f} | {np.mean(seeds):.4f} ± {np.std(seeds):.4f} | "
                     f"{e.get('recompute', {}).get('normal_cos_vs_full_median', float('nan')):.3f} |")
    (out_dir / "deployment_resample_reeval.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
