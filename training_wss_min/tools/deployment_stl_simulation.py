"""Route A: deployment simulation from the original STL surface (inference only).

For each test case the STL surface (same frame and unit as the CFD wall, mm) is resampled at a target spacing with
area-weighted random sampling followed by voxel decimation (one point per voxel, nearest the centroid), and a full
deployment case is built from the resampled cloud + the case's centreline atlas with the same programs the CFD
cases went through:
  * atlas projection (centerline_features.map_points) -> abscissa, radius, curvature, dr/ds, rho/theta, distances,
    end zone, segment/semantic ids;
  * anatomical frame from the atlas (wss_min_view.anatomical_frame), per-case max-abs scale from the cloud;
  * PCA normals with centreline orientation (pointcloud.pca_normals) and virtual caps from the cloud
    (pointcloud.build_oriented_cloud) -> cap radii for Murray / capfit shares;
  * principal-curvature families (wall_geom_v2.principal_curvatures, k=32/128) and tn_dot;
  * flow-reference features (wall_flowref_v1.compute_point_features): bend/bifurcation/upstream, Murray, capfit,
    stenosis index.
Truth = CFD WSS at the nearest valid wall node (peak frame).  Reference = the saved full-cloud prediction of the same
checkpoint mapped to the same points by nearest node, so every delta is read on identical points.

    CUDA_VISIBLE_DEVICES=3 python -m training_wss_min.tools.deployment_stl_simulation --spacings 0.4 0.3 --out <dir>
"""
from __future__ import annotations

import argparse
import glob
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
from training_wss_min.surface import load_stl
from training_wss_min.tools.deployment_resample_reeval import DEFAULT_RUNS, atlas_from_h5, summarise, voxel_rows
from wss_v5.centerline_features import map_points
from wss_v5.pointcloud import build_oriented_cloud, median_spacing, pca_normals
from wss_v5.views.wall_flowref_v1 import _Tree, compute_point_features
from wss_v5.views.wall_geom_v2 import COARSE_K, FINE_K, _unit_rows, principal_curvatures
from wss_v5.views.wss_min_view import anatomical_frame

ROOT = C.PROJECT_ROOT
DATA_NEW = ROOT / "data_new"
FINE_MAP = {"curv_k1": "k1", "curv_k2": "k2", "curv_mean": "mean", "curv_gauss": "gauss", "curvedness": "curvedness", "shape_index": "shape_index"}


_STL_MANIFEST = None


def _stl_manifest():
    """tools/stl_manifest.py loaded by file path (``tools`` is a plain folder at the repository root, not a package on every
    caller's sys.path; in a frozen code copy it is a symlink to the main tree)."""
    global _STL_MANIFEST
    if _STL_MANIFEST is None:
        import importlib.util
        spec = importlib.util.spec_from_file_location("_stl_manifest_for_deployment_tests", ROOT / "tools/stl_manifest.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _STL_MANIFEST = module
    return _STL_MANIFEST


def find_stl_in_folder(unit_id: str) -> Path | None:
    """The first ``*.stl`` of the unit's ``data_new`` folder (the rule ``find_stl`` used until 2026-10-02)."""
    cohort, subset, case = unit_id.split("/")
    candidates = sorted(glob.glob(str(DATA_NEW / cohort / subset / case / "*.stl")))
    return Path(candidates[0]) if candidates else None


def find_stl(unit_id: str) -> Path | None:
    """The surface a deployment-style test of a library unit starts from.

    Units listed in the canonical STL manifest (the 273 real units of v5.2d; ``tools/stl_manifest.py``) get their canonical
    surface: the ``data_new`` STL when the CFD wall lies on it, otherwise the CFD wall exported from ``case.h5`` (24 units
    whose folder holds a pre-processing assembly, a file in another frame or unit, a capped surface, or no STL at all).
    "First ``*.stl`` of the folder" returned those wrong files; it gives the same file as the manifest for 248 units.
    Units outside the manifest (synthetic children) keep the folder rule."""
    module = _stl_manifest()
    if not module.MANIFEST.is_file():
        raise FileNotFoundError(f"canonical STL manifest missing: {module.MANIFEST}")
    stl = module.canonical_stl(unit_id)
    if stl is None:
        return find_stl_in_folder(unit_id)
    if not stl.is_file():
        raise FileNotFoundError(f"{unit_id}: canonical surface listed in the manifest is missing: {stl}")
    return stl


def sample_surface(vertices: np.ndarray, faces: np.ndarray, spacing: float, seed: int) -> np.ndarray:
    """Area-weighted random points (oversampled) then voxel decimation, calibrated so that the median
    nearest-neighbour distance of the result matches ``spacing`` (two refinement passes)."""
    tri = vertices[faces]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    total = float(area.sum())
    rng = np.random.default_rng(seed)
    target = int(round(1.155 * total / (spacing * spacing)))   # hexagonal packing at NN distance = spacing
    n_raw = max(target * 6, 20000)
    pick = rng.choice(len(faces), size=n_raw, p=area / total)
    u, v = rng.random(n_raw), rng.random(n_raw)
    flip = u + v > 1.0
    u[flip], v[flip] = 1.0 - u[flip], 1.0 - v[flip]
    pts = tri[pick, 0] + u[:, None] * (tri[pick, 1] - tri[pick, 0]) + v[:, None] * (tri[pick, 2] - tri[pick, 0])
    out = None
    for _ in range(3):
        rows = voxel_rows(pts, min(target / n_raw, 1.0), seed)
        out = pts[rows]
        actual = float(median_spacing(out))
        if abs(actual - spacing) / spacing < 0.05:
            break
        target = max(int(round(target * (actual / spacing) ** 2)), 500)
    return out


def build_deployment_case(unit_id: str, pts: np.ndarray, h5_path: Path, bundle_path: Path, stats: dict, input_features) -> tuple[dict, dict]:
    with h5py.File(h5_path, "r") as h:
        atlas = atlas_from_h5(h)
        g = h["geometry"]
        table = np.asarray(g["atlas_table"][()], dtype=np.float64)
        columns = json.loads(g.attrs["atlas_columns"])
        segments = json.loads(g.attrs["atlas_segments"])
        semantic_of_segment = json.loads(g.attrs["semantic_of_segment"])
        frame_n = np.asarray(g["atlas_frame_n"][()]); frame_b = np.asarray(g["atlas_frame_b"][()])
    with np.load(bundle_path, allow_pickle=True) as z:
        cfd_xyz = z["wall_coords_raw"].astype(np.float64)
        steps = z["steps"].tolist(); si = steps.index(int(z["peak_step"]))
        cfd_wss = z["wall_wss"][si].astype(np.float64)
        cfd_rotation = z["transform_rotation"].astype(np.float64)
        cfd_scale = float(z["coord_scale"])
    feats = map_points(pts, atlas)
    frame = anatomical_frame(table, columns, semantic_of_segment)
    aligned = (pts - frame["origin_mm"]) @ frame["rotation"].T
    scale = float(np.abs(aligned).max()) or 1.0
    normals, variation = pca_normals(pts, atlas)
    normals = _unit_rows(np.asarray(normals, dtype=np.float64))
    _, info = build_oriented_cloud(pts, atlas, normals_out=normals, calibrate=False)
    cap_labels = [c["label"] for c in info["caps"]]
    cap_radius = np.array([c["radius_mm"] for c in info["caps"]], dtype=np.float64)
    tree = cKDTree(pts)
    max_k = min(COARSE_K, len(pts) - 1)
    _, nb = tree.query(pts, k=max_k + 1, workers=-1); nb = nb[:, 1:]
    fine = principal_curvatures(pts, normals, nb[:, : min(FINE_K, max_k)])
    coarse = principal_curvatures(pts, normals, nb)
    atlas_row = feats["atlas_row"].astype(np.int64)
    tangent = atlas.tangent[atlas_row]
    ftree = _Tree(table, columns, segments, frame_n, frame_b)
    radius_pt = np.clip(feats["radius_mm"].astype(np.float64), 1e-3, None)
    flow, extra = compute_point_features(ftree, cap_labels, cap_radius, atlas_row, feats["segment_id"], feats["semantic_id"],
                                         feats["s_local_mm"], feats["s_from_root_mm"], feats["theta_rad"], radius_pt,
                                         feats["dist_to_junction_mm"])
    dist, nearest = cKDTree(cfd_xyz).query(pts, k=1)
    y_raw = cfd_wss[nearest].astype(np.float32)
    s_max = float(np.max(feats["s_from_root_mm"]))
    n = len(pts)
    case = dict(
        cohort=unit_id.rsplit("/", 1)[0], case=unit_id.rsplit("/", 1)[1], unit_id=unit_id,
        pos=(aligned / scale).astype(np.float32), wall_coords_raw=pts.astype(np.float64),
        y_raw=y_raw, y_norm=D.normalize_wss(y_raw, stats).astype(np.float32), target_normalization="global_stats",
        abscissa_norm=(feats["s_from_root_mm"] / s_max).astype(np.float32), local_radius=radius_pt.astype(np.float32),
        log_local_radius=np.log(radius_pt).astype(np.float32), curvature=feats["curvature_per_mm"].astype(np.float32),
        coord_scale=np.full(n, scale, dtype=np.float32), coord_scale_scalar=scale, radius_gradient=feats["dr_ds"].astype(np.float32),
        rho=feats["rho"].astype(np.float32), theta_sin=np.sin(feats["theta_rad"]).astype(np.float32),
        theta_cos=np.cos(feats["theta_rad"]).astype(np.float32), dr_ds=feats["dr_ds"].astype(np.float32),
        dist_to_junction_mm=feats["dist_to_junction_mm"].astype(np.float32), dist_to_endpoint_mm=feats["dist_to_endpoint_mm"].astype(np.float32),
        end_zone=feats["end_zone"].astype(np.float32), _wall_segment_id=feats["segment_id"].astype(np.int64),
        peak_step=int(steps[si]), bundle_path=str(bundle_path),
    )
    aligned_normals = normals @ frame["rotation"].T
    case["nx_aligned"], case["ny_aligned"], case["nz_aligned"] = (aligned_normals[:, i].astype(np.float32) for i in range(3))
    for name in input_features:
        if name in FINE_MAP:
            case[name] = fine[FINE_MAP[name]].astype(np.float32)
        elif name.endswith("_c") and name[:-2] in FINE_MAP:
            case[name] = coarse[FINE_MAP[name[:-2]]].astype(np.float32)
        elif name == "tn_dot":
            case[name] = np.einsum("nd,nd->n", tangent, normals).astype(np.float32)
        elif f"wall_{name}" in flow:
            case[name] = np.asarray(flow[f"wall_{name}"], dtype=np.float32)
    missing = [f for f in input_features if f not in case and f not in ("x", "y", "z")]
    if missing:
        raise KeyError(f"{unit_id}: could not build deployment features {missing}")
    tangent_aligned = _unit_rows(tangent @ frame["rotation"].T)
    case["local_geometry"] = np.column_stack((aligned_normals, tangent_aligned, radius_pt, np.full(n, scale))).astype(np.float32)
    case["section"] = np.column_stack((feats["segment_id"].astype(np.float64), np.clip(feats["s_local_mm"], 0.0, None))).astype(np.float32)
    diag = {"n_points": n, "spacing_mm": float(median_spacing(pts)), "scale_mm": scale, "cfd_scale_mm": cfd_scale,
            "rotation_max_abs_diff": float(np.abs(frame["rotation"] - cfd_rotation).max()),
            "nearest_cfd_node_mm_median": float(np.median(dist)), "nearest_cfd_node_mm_p95": float(np.quantile(dist, 0.95)),
            "junction_ambiguous_fraction": float(np.mean(feats["junction_ambiguous"])),
            "caps": {c["label"]: {"radius_mm": c["radius_mm"], "atlas_radius_mm": c["atlas_radius_mm"]} for c in info["caps"]},
            "murray_shares": extra["murray_shares"], "capfit_shares": extra["capfit_shares"], "capfit_fallback_nodes": extra["capfit_notes"]["fallback_nodes"],
            "fine_neighbourhood_mm_median": float(np.median(fine["_neighbourhood_mm"])),
            "coarse_neighbourhood_mm_median": float(np.median(coarse["_neighbourhood_mm"])),
            "surface_variation_median": float(np.median(variation))}
    return case, {"nearest": nearest, "diag": diag}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=[str(p) for p in DEFAULT_RUNS])
    ap.add_argument("--spacings", nargs="*", type=float, default=[0.4, 0.3])
    ap.add_argument("--checkpoint", default="best")
    ap.add_argument("--seed", type=int, default=20260915)
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
        models.append({"run": run, "name": run.name, "cfg": cfg, "feat_stats": feat_stats, "model": model, "stats": E.load_wss_stats_for_run(run)})
    cfg0, stats0 = models[0]["cfg"], models[0]["stats"]
    pairs = D.load_split_cases(cfg0.data.split_path, "test")
    if args.limit_cases:
        pairs = pairs[: args.limit_cases]
    results = {s: {"units": [], "true_pa": [], "true_n": [], "pred_n": {m["name"]: [] for m in models}, "ref_n": {m["name"]: [] for m in models}, "diag": []} for s in args.spacings}
    skipped = []
    started = time.time()
    for ci, (cohort, name) in enumerate(pairs):
        unit_id = f"{cohort}/{name}"
        bundle_path = Path(cfg0.data.data_root) / cohort / name / "bundle.npz"
        stl = find_stl(unit_id)
        if stl is None:
            skipped.append({"unit_id": unit_id, "reason": "no STL found"}); continue
        vertices, faces = load_stl(stl)
        with np.load(bundle_path, allow_pickle=True) as z:
            cfd_xyz = z["wall_coords_raw"].astype(np.float64)
        d_frame, _ = cKDTree(cfd_xyz).query(vertices[:: max(1, len(vertices) // 5000)], k=1)
        if np.median(d_frame) > 1.0:
            skipped.append({"unit_id": unit_id, "reason": f"STL frame mismatch (median {np.median(d_frame):.2f} mm)", "stl": str(stl)}); continue
        h5_path = NG.source_h5_for_bundle(bundle_path)
        saved = {}
        for m in models:
            with np.load(m["run"] / "eval" / f"ckpt_{args.checkpoint}" / "predictions/test" / unit_id / "predictions.npz") as z:
                ref = np.full(len(cfd_xyz), np.nan); ref[z["row_index"]] = z["pred_norm"]; saved[m["name"]] = ref
        native = float(median_spacing(cfd_xyz))
        for spacing in args.spacings:
            # spacing 0 = "native": resample at this case's own CFD wall node spacing (isolates resampling from density)
            target_spacing = native if spacing == 0 else spacing
            pts = sample_surface(vertices, faces, target_spacing, args.seed + ci)
            case, aux = build_deployment_case(unit_id, pts, h5_path, bundle_path, stats0, cfg0.data.input_features)
            store = results[spacing]
            store["units"].append(unit_id); store["true_pa"].append(case["y_raw"].astype(np.float64)); store["true_n"].append(case["y_norm"].astype(np.float64))
            store["diag"].append({**aux["diag"], "stl": str(stl), "native_cfd_spacing_mm": native, "target_spacing_mm": target_spacing})
            for m in models:
                with torch.no_grad():
                    pred = E.predict_case_norm(m["model"], case, m["cfg"].data.input_features, m["feat_stats"], device, cfg=m["cfg"])
                store["pred_n"][m["name"]].append(np.asarray(pred, dtype=np.float64))
                store["ref_n"][m["name"]].append(saved[m["name"]][aux["nearest"]])
        print(f"[{ci + 1}/{len(pairs)}] {unit_id} stl={stl.name} n_cfd={len(cfd_xyz)} " + " ".join(f"{s}mm:{results[s]['diag'][-1]['n_points']}" for s in args.spacings) + f" {time.time() - started:.0f}s", flush=True)
    report = {"runs": [str(m["run"]) for m in models], "checkpoint": args.checkpoint, "spacings": args.spacings, "skipped": skipped, "rows": []}
    for spacing, store in results.items():
        if not store["units"]:
            continue
        names = list(store["pred_n"]); true_pa, true_n = store["true_pa"], store["true_n"]
        entry = {"spacing_mm": spacing, "n_cases": len(store["units"]), "units": store["units"], "points_median": float(np.median([d["n_points"] for d in store["diag"]])),
                 "diag_median": {k: float(np.median([d[k] for d in store["diag"]])) for k in ("spacing_mm", "target_spacing_mm", "native_cfd_spacing_mm", "nearest_cfd_node_mm_median", "rotation_max_abs_diff", "fine_neighbourhood_mm_median", "coarse_neighbourhood_mm_median", "junction_ambiguous_fraction")},
                 "scale_ratio_median": float(np.median([d["scale_mm"] / d["cfd_scale_mm"] for d in store["diag"]])), "seeds": {}, "per_case_diag": store["diag"]}
        for nm in names:
            pred_n, ref_n = store["pred_n"][nm], store["ref_n"][nm]
            entry["seeds"][nm] = {"stl": summarise(true_pa, [D.denormalize_wss(p, stats0) for p in pred_n], true_n, pred_n),
                                  "reference_same_points": summarise(true_pa, [D.denormalize_wss(p, stats0) for p in ref_n], true_n, ref_n)}
        ens = [np.mean([store["pred_n"][nm][i] for nm in names], axis=0) for i in range(len(true_pa))]
        ref = [np.mean([store["ref_n"][nm][i] for nm in names], axis=0) for i in range(len(true_pa))]
        entry["ensemble"] = summarise(true_pa, [D.denormalize_wss(p, stats0) for p in ens], true_n, ens)
        entry["reference_ensemble"] = summarise(true_pa, [D.denormalize_wss(p, stats0) for p in ref], true_n, ref)
        report["rows"].append(entry)
    (out_dir / "deployment_stl_simulation.json").write_text(json.dumps(report, indent=1))
    lines = ["| spacing mm | cases | points/case | actual spacing (median) | ens Pa R²_cb (STL) | ref (same points) | Δ | norm (STL / ref) | case mean / P10 | MAE | high-WSS R² | top10 比 | IoU | seed mean ± sd | nearest CFD node mm |",
             "|---|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---|---:|"]
    for e in report["rows"]:
        s, r = e["ensemble"], e["reference_ensemble"]; seeds = [v["stl"]["pa_r2_cb"] for v in e["seeds"].values()]
        label = "native" if e["spacing_mm"] == 0 else f"{e['spacing_mm']}"
        lines.append(f"| {label} | {e['n_cases']} | {e['points_median']:.0f} | {e['diag_median']['spacing_mm']:.3f} | {s['pa_r2_cb']:.4f} | {r['pa_r2_cb']:.4f} | {s['pa_r2_cb'] - r['pa_r2_cb']:+.4f} | {s['norm_r2_cb']:.4f} / {r['norm_r2_cb']:.4f} | "
                     f"{s['case_r2_mean']:.3f} / {s['case_r2_p10']:.3f} | {s['mae_pooled']:.3f} | {s['high_wss_r2']:.3f} | {s['top10_ratio']:.3f} | {s['top10_iou_mean']:.3f} | {np.mean(seeds):.4f} ± {np.std(seeds):.4f} | {e['diag_median']['nearest_cfd_node_mm_median']:.3f} |")
    if skipped:
        lines.append(""); lines.append("skipped: " + "; ".join(f"{s['unit_id']} ({s['reason']})" for s in skipped))
    (out_dir / "deployment_stl_simulation.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
