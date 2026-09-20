"""Geometry-fidelity sensitivity of the deployment form (inference only).

§22.1–22.3 only changed the *point sampling* of the same source surface.  A clinical segmentation differs from
the CFD surface itself: small-scale geometry is smoothed away, the boundary carries spatially correlated noise,
and the iliac cut planes sit at a different level.  This tool perturbs the original STL before the standard
0.5 mm deployment resampling (``deployment_stl_simulation``) and re-runs the full deployment geometry program:

* ``smooth_s<σ>``  : Taubin smoothing (λ = 0.5, μ = −0.53, 20 iterations) with a Gaussian-weighted umbrella
                     operator of physical width σ mm (cohort-independent pass-band; no shrinkage);
* ``noise_<σ>``    : spatially correlated boundary noise: white noise per vertex, Gaussian-smoothed with
                     correlation length 2 mm, rescaled to std σ mm, applied along the vertex normal;
* ``cut_<d>``      : iliac cut planes d mm more proximal.  X5/X5D's Murray share uses the atlas radius over the
                     distal 10 mm of each leaf segment; a shorter cut moves that window to [L−d−10, L−d].  The
                     effect is applied to the two Murray inputs (log_q_branch_murray, log_tau0_murray) on the
                     clean resampled cloud, so this isolates the "cut level → flow prior → WSS" pathway (the
                     wall points themselves stay, truth is unchanged);
* ``clean``        : the unperturbed 0.5 mm resampling (same as §22.1 at 0.5 mm) for reference.

Truth = CFD WSS at the nearest valid wall node of the *original* geometry (the perturbation is an observation
error of the input, not an anatomical change), reference = saved full-cloud prediction mapped by the same
nearest node.  Deltas are read against the reference on identical points, as in §22.

    CUDA_VISIBLE_DEVICES=2 python -m training_wss_min.tools.deployment_geometry_sensitivity --out <dir>
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import scipy.sparse as sp
import torch
from scipy.spatial import cKDTree

from training_wss_min import dataset as D
from training_wss_min import evaluate as E
from training_wss_min import next_geometry as NG
from training_wss_min.surface import load_stl
from training_wss_min.tools.deployment_resample_reeval import DEFAULT_RUNS, summarise
from training_wss_min.tools.deployment_stl_simulation import build_deployment_case, find_stl, sample_surface
from wss_v5.pointcloud import median_spacing
from wss_v5.views.wall_flowref_v1 import TERMINAL_MEAN_MM, _Tree, _murray_shares

MURRAY_KEYS = ("log_q_branch_murray", "log_tau0_murray")
NOISE_CORR_MM = 2.0   # correlation length of the boundary-noise field (a few CT voxels)


# ----------------------------------------------------------------------------- surface perturbations
def gaussian_weights(vertices: np.ndarray, sigma_mm: float) -> sp.csr_matrix:
    """Row-normalised Gaussian weights over vertices within 3 σ, self included (weight 1), so that a vertex with
    no neighbour inside the window is a fixed point of the operator instead of being pulled towards the origin."""
    tree = cKDTree(vertices)
    pairs = tree.query_pairs(r=3.0 * sigma_mm, output_type="ndarray")
    own = np.arange(len(vertices))
    i = np.concatenate([pairs[:, 0], pairs[:, 1], own]); j = np.concatenate([pairs[:, 1], pairs[:, 0], own])
    d2 = ((vertices[i] - vertices[j]) ** 2).sum(axis=1)
    w = np.exp(-0.5 * d2 / (sigma_mm * sigma_mm))
    W = sp.csr_matrix((w, (i, j)), shape=(len(vertices), len(vertices)))
    rs = np.asarray(W.sum(axis=1)).ravel()
    return sp.diags(1.0 / np.maximum(rs, 1e-12)) @ W


def taubin_smooth(vertices: np.ndarray, sigma_mm: float, iterations: int = 20, lam: float = 0.5, mu: float = -0.53) -> np.ndarray:
    W = gaussian_weights(vertices, sigma_mm)
    V = np.array(vertices, dtype=np.float64, copy=True)
    for _ in range(iterations):
        V = V + lam * (W @ V - V)
        V = V + mu * (W @ V - V)
    return V


def vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = vertices[faces]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])   # area-weighted
    vn = np.zeros_like(vertices)
    for k in range(3):
        np.add.at(vn, faces[:, k], fn)
    return vn / np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)


def correlated_noise(vertices: np.ndarray, faces: np.ndarray, sigma_mm: float, corr_mm: float, seed: int) -> np.ndarray:
    W = gaussian_weights(vertices, corr_mm)
    white = np.random.default_rng(seed).standard_normal(len(vertices))
    field = W @ white
    field = field / max(float(field.std()), 1e-12) * sigma_mm
    return vertices + field[:, None] * vertex_normals(vertices, faces)


def apply_cut_shift(case: dict, h5_path: Path, shift_mm: float) -> dict:
    """Move the Murray distal-radius window d mm proximal on every leaf segment and rewrite the two Murray inputs."""
    with h5py.File(h5_path, "r") as h:
        g = h["geometry"]
        table = np.asarray(g["atlas_table"][()], dtype=np.float64)
        columns = json.loads(g.attrs["atlas_columns"]); segments = json.loads(g.attrs["atlas_segments"])
        frame_n = np.asarray(g["atlas_frame_n"][()]); frame_b = np.asarray(g["atlas_frame_b"][()])
    tree = _Tree(table, columns, segments, frame_n, frame_b)
    base = _murray_shares(tree)
    leaf_radius, window = {}, {}
    for sid, seg in tree.segments.items():
        if not seg.get("ends_at_leaf"):
            continue
        rows = tree.rows[sid]; length = tree.length[sid]
        s = tree.s_local[rows]
        tail = rows[(s >= length - shift_mm - TERMINAL_MEAN_MM) & (s <= length - shift_mm)]
        if len(tail) < 2:
            tail = rows[np.argsort(np.abs(s - (length - shift_mm)))[:4]]
        leaf_radius[sid] = float(np.median(tree.radius[tail]))
        window[str(sid)] = {"outlet": seg.get("outlet_name", ""), "length_mm": float(length), "r_distal_base": None,
                            "r_distal_shifted": leaf_radius[sid]}
    shifted = _murray_shares(tree, leaf_radius=leaf_radius)
    seg_pt = np.asarray(case["_wall_segment_id"], dtype=int)
    delta = np.array([np.log(max(shifted.get(int(s), 1.0), 1e-6)) - np.log(max(base.get(int(s), 1.0), 1e-6)) for s in seg_pt], dtype=np.float32)
    for key in MURRAY_KEYS:
        if key not in case:
            raise KeyError(f"cut shift needs {key} in the deployment case")
        case[key] = (np.asarray(case[key], dtype=np.float32) + delta).astype(np.float32)
    return {"share_base": {str(k): v for k, v in base.items()}, "share_shifted": {str(k): v for k, v in shifted.items()},
            "max_abs_dlog_q": float(np.abs(delta).max()), "leaf_windows": window}


def perturb(vertices: np.ndarray, faces: np.ndarray, variant: str, seed: int) -> tuple[np.ndarray, dict]:
    """``a+b`` applies the steps in order (e.g. ``noise_0.2+smooth_s1.0`` = noisy segmentation, then the smoothing a
    deployment pipeline would run before resampling)."""
    if variant == "clean" or variant.startswith("cut_"):
        return vertices, {}
    V = vertices
    for step in variant.split("+"):
        kind, value = step.split("_", 1)
        if kind == "smooth":
            V = taubin_smooth(V, float(value[1:]) if value.startswith("s") else float(value))
        elif kind == "noise":       # noise_<σ>  or  noise_<σ>c<corr_mm>  (default correlation length NOISE_CORR_MM)
            sigma, _, corr = value.partition("c")
            V = correlated_noise(V, faces, float(sigma), float(corr) if corr else NOISE_CORR_MM, seed)
        else:
            raise ValueError(f"unknown variant step {step}")
    disp = np.linalg.norm(V - vertices, axis=1)
    return V, {"vertex_disp_mm_median": float(np.median(disp)), "vertex_disp_mm_p95": float(np.quantile(disp, 0.95)),
               "vertex_disp_mm_max": float(disp.max())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=[str(p) for p in DEFAULT_RUNS])
    ap.add_argument("--variants", nargs="*", default=["clean", "smooth_s1.0", "smooth_s1.5", "smooth_s2.0",
                                                     "noise_0.1", "noise_0.2", "noise_0.4", "cut_5", "cut_10"])
    ap.add_argument("--spacing", type=float, default=0.5)
    ap.add_argument("--checkpoint", default="best")
    ap.add_argument("--seed", type=int, default=20260915, help="same resampling seed as deployment_stl_simulation")
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
    results = {v: {"units": [], "true_pa": [], "true_n": [], "pred_n": {m["name"]: [] for m in models},
                   "ref_n": {m["name"]: [] for m in models}, "diag": []} for v in args.variants}
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
            skipped.append({"unit_id": unit_id, "reason": f"STL frame mismatch (median {np.median(d_frame):.2f} mm)"}); continue
        h5_path = NG.source_h5_for_bundle(bundle_path)
        saved = {}
        for m in models:
            with np.load(m["run"] / "eval" / f"ckpt_{args.checkpoint}" / "predictions/test" / unit_id / "predictions.npz") as z:
                ref = np.full(len(cfd_xyz), np.nan); ref[z["row_index"]] = z["pred_norm"]; saved[m["name"]] = ref
        clean = None   # the clean 0.5 mm case is built once; cut variants only rewrite its Murray inputs
        for variant in args.variants:
            v_pts, vdiag = perturb(vertices, faces, variant, args.seed + 7919 * ci)
            if variant == "clean" or variant.startswith("cut_"):
                if clean is None:
                    pts = sample_surface(vertices, faces, args.spacing, args.seed + ci)
                    clean = build_deployment_case(unit_id, pts, h5_path, bundle_path, stats0, cfg0.data.input_features)
                case, aux = dict(clean[0]), clean[1]
                for key in ("_full_wall_tree", "_full_wall_features", "_fps_pool", "_fps_pool_k", "_fps_pool_size"):
                    case.pop(key, None)        # never reuse feature caches across variants
            else:
                pts = sample_surface(v_pts, faces, args.spacing, args.seed + ci)
                case, aux = build_deployment_case(unit_id, pts, h5_path, bundle_path, stats0, cfg0.data.input_features)
            if variant.startswith("cut_"):
                vdiag = apply_cut_shift(case, h5_path, float(variant.split("_", 1)[1]))
            store = results[variant]
            store["units"].append(unit_id); store["true_pa"].append(case["y_raw"].astype(np.float64)); store["true_n"].append(case["y_norm"].astype(np.float64))
            store["diag"].append({**aux["diag"], **vdiag, "stl": str(stl)})
            for m in models:
                with torch.no_grad():
                    pred = E.predict_case_norm(m["model"], case, m["cfg"].data.input_features, m["feat_stats"], device, cfg=m["cfg"])
                store["pred_n"][m["name"]].append(np.asarray(pred, dtype=np.float64))
                store["ref_n"][m["name"]].append(saved[m["name"]][aux["nearest"]])
        print(f"[{ci + 1}/{len(pairs)}] {unit_id} n_stl={len(vertices)} n_cfd={len(cfd_xyz)} {time.time() - started:.0f}s", flush=True)
    report = {"runs": [str(m["run"]) for m in models], "checkpoint": args.checkpoint, "spacing_mm": args.spacing, "variants": args.variants,
              "skipped": skipped, "rows": []}
    diag_keys = ("spacing_mm", "nearest_cfd_node_mm_median", "nearest_cfd_node_mm_p95", "fine_neighbourhood_mm_median",
                 "surface_variation_median", "vertex_disp_mm_median", "vertex_disp_mm_p95", "max_abs_dlog_q")
    for variant, store in results.items():
        if not store["units"]:
            continue
        names = list(store["pred_n"]); true_pa, true_n = store["true_pa"], store["true_n"]
        entry = {"variant": variant, "n_cases": len(store["units"]), "units": store["units"],
                 "points_median": float(np.median([d["n_points"] for d in store["diag"]])),
                 "diag_median": {k: float(np.median([d[k] for d in store["diag"] if k in d])) for k in diag_keys if any(k in d for d in store["diag"])},
                 "seeds": {}, "per_case_diag": store["diag"]}
        for nm in names:
            pred_n, ref_n = store["pred_n"][nm], store["ref_n"][nm]
            entry["seeds"][nm] = {"perturbed": summarise(true_pa, [D.denormalize_wss(p, stats0) for p in pred_n], true_n, pred_n),
                                  "reference_same_points": summarise(true_pa, [D.denormalize_wss(p, stats0) for p in ref_n], true_n, ref_n)}
        ens = [np.mean([store["pred_n"][nm][i] for nm in names], axis=0) for i in range(len(true_pa))]
        ref = [np.mean([store["ref_n"][nm][i] for nm in names], axis=0) for i in range(len(true_pa))]
        entry["ensemble"] = summarise(true_pa, [D.denormalize_wss(p, stats0) for p in ens], true_n, ens)
        entry["reference_ensemble"] = summarise(true_pa, [D.denormalize_wss(p, stats0) for p in ref], true_n, ref)
        ens_pa = [np.mean([D.denormalize_wss(store["pred_n"][nm][i], stats0) for nm in names], axis=0) for i in range(len(true_pa))]
        ref_pa = [np.mean([D.denormalize_wss(store["ref_n"][nm][i], stats0) for nm in names], axis=0) for i in range(len(true_pa))]
        entry["ensemble_pa_mean"] = summarise(true_pa, ens_pa, true_n, [D.normalize_wss(p, stats0) for p in ens_pa])
        entry["reference_ensemble_pa_mean"] = summarise(true_pa, ref_pa, true_n, [D.normalize_wss(p, stats0) for p in ref_pa])
        report["rows"].append(entry)
    (out_dir / "geometry_sensitivity.json").write_text(json.dumps(report, indent=1))
    lines = ["| variant | cases | points/case | vertex disp mm (median / p95) | nearest CFD node mm (median / p95) | max |Δlog q| | ens Pa R²_cb (log-mean) | ref (same points) | Δ | Δ (Pa-mean ens) | norm (ens / ref) | case mean / P10 | high-WSS R² | top10 比 | IoU | seed Δ (mean ± sd) |",
             "|---|---:|---:|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---|"]
    for e in report["rows"]:
        s, r = e["ensemble"], e["reference_ensemble"]; sp_, rp = e["ensemble_pa_mean"], e["reference_ensemble_pa_mean"]
        dm = e["diag_median"]
        deltas = [v["perturbed"]["pa_r2_cb"] - v["reference_same_points"]["pa_r2_cb"] for v in e["seeds"].values()]
        disp = f"{dm['vertex_disp_mm_median']:.3f} / {dm['vertex_disp_mm_p95']:.3f}" if "vertex_disp_mm_median" in dm else "—"
        dq = f"{dm['max_abs_dlog_q']:.3f}" if "max_abs_dlog_q" in dm else "—"
        lines.append(f"| {e['variant']} | {e['n_cases']} | {e['points_median']:.0f} | {disp} | {dm['nearest_cfd_node_mm_median']:.3f} / {dm['nearest_cfd_node_mm_p95']:.3f} | {dq} | "
                     f"{s['pa_r2_cb']:.4f} | {r['pa_r2_cb']:.4f} | {s['pa_r2_cb'] - r['pa_r2_cb']:+.4f} | {sp_['pa_r2_cb'] - rp['pa_r2_cb']:+.4f} | {s['norm_r2_cb']:.4f} / {r['norm_r2_cb']:.4f} | "
                     f"{s['case_r2_mean']:.3f} / {s['case_r2_p10']:.3f} | {s['high_wss_r2']:.3f} | {s['top10_ratio']:.3f} | {s['top10_iou_mean']:.3f} | {np.mean(deltas):+.4f} ± {np.std(deltas):.4f} |")
    if skipped:
        lines.append(""); lines.append("skipped: " + "; ".join(f"{s['unit_id']} ({s['reason']})" for s in skipped))
    (out_dir / "geometry_sensitivity.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
