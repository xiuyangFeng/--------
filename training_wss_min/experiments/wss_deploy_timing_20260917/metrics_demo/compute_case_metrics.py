"""Metrics catalogue demo: what a deployment report can state WITHOUT any CFD truth.
STL -> resample 0.5 mm -> 27 features -> 5-seed X5D_v51 -> Pa-mean ensemble -> per-case metrics JSON (+ npz for plots).
Atlas is taken from the archived case.h5 (the vessel_geom adapter is not written yet). Population reference = CFD peak-frame
WSS of the 136 training cases (labels, allowed: used only as a reference distribution, never as an input)."""
import json, sys, time
from pathlib import Path
import numpy as np, torch, h5py
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from training_wss_min import config as C, dataset as D, evaluate as E, next_geometry as NG
from training_wss_min.surface import load_stl
from training_wss_min.tools.deployment_stl_simulation import find_stl, sample_surface, build_deployment_case
from training_wss_min.tools.deployment_resample_reeval import atlas_from_h5
from wss_v5.centerline_features import map_points
from wss_v5.pointcloud import median_spacing
from wss_v5.views import wall_flowref_v1 as WF
WF._CAPFIT_RULE_OVERRIDE = Path("/public/newhome/cy/Digital_twin/GNN/data_wss_v5/views_v5_1/wss_min_flowref_v1/flow_split_rule_train136.json")

ROOT = C.PROJECT_ROOT
OUT = Path(__file__).parent
RUNS = [ROOT / "training_wss_min/runs/wss_v51_wave1_20260916" / f"X5D_v51_s{s}" for s in (1234, 7, 2025, 11, 2026)]
NAME = {"root": "主动脉", "out-le": "左髂外", "out-li": "左髂内", "out-re": "右髂外", "out-ri": "右髂内", "left_cia": "左髂总", "right_cia": "右髂总"}
LOW_PA, HIGH_PA, VERY_HIGH_PA = 0.4, 4.0, 7.0

def branch_names(segments):
    out = {}
    for s in segments:
        sid = int(s["segment_id"])
        if s.get("starts_at_root"): out[sid] = NAME["root"]
        elif s.get("ends_at_leaf"): out[sid] = NAME.get(s.get("outlet_name", ""), f"出口{sid}")
        else:
            d = set(s.get("descendant_outlets", []))
            out[sid] = NAME["left_cia"] if d <= {"out-le", "out-li"} else (NAME["right_cia"] if d <= {"out-re", "out-ri"} else f"段{sid}")
    return out

def stats(v):
    v = np.asarray(v, dtype=np.float64)
    return {"n": int(v.size), "mean": float(v.mean()), "median": float(np.median(v)), "p95": float(np.quantile(v, .95)), "p99": float(np.quantile(v, .99)), "max": float(v.max()), "min": float(v.min())}

def population_reference(split_path, data_root):
    ref_path = OUT / "population_reference_train136.json"
    if ref_path.exists(): return json.load(open(ref_path))
    split = json.load(open(split_path)); rows = []
    for uid in split["train_cases"]:
        with np.load(Path(data_root) / uid / "bundle.npz", allow_pickle=True) as z:
            steps = z["steps"].tolist(); si = steps.index(int(z["peak_step"])); w = z["wall_wss"][si].astype(np.float64)
        rows.append({"unit_id": uid, **stats(w), "frac_low": float(np.mean(w < LOW_PA)), "frac_high": float(np.mean(w > HIGH_PA))})
    ref = {"source": "CFD peak-frame wall WSS, train136 (v5.1)", "cases": rows}
    json.dump(ref, open(ref_path, "w"), indent=1, ensure_ascii=False); return ref

def percentile_of(value, arr):
    arr = np.asarray(arr, dtype=np.float64); return float(100.0 * np.mean(arr <= value))

def main(cases):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = []
    for run in RUNS:
        cfg, feat_stats, model, _ = E.load_model_from_run(run, device, "best"); model.eval()
        models.append((cfg, feat_stats, model, E.load_wss_stats_for_run(run)))
    cfg0, stats0 = models[0][0], models[0][3]
    ref = population_reference(cfg0.data.split_path, cfg0.data.data_root)
    ref_p99 = [r["p99"] for r in ref["cases"]]; ref_max = [r["max"] for r in ref["cases"]]; ref_mean = [r["mean"] for r in ref["cases"]]
    for unit_id in cases:
        t_all = time.perf_counter(); T = {}
        cohort, name = unit_id.rsplit("/", 1)
        bundle_path = Path(cfg0.data.data_root) / cohort / name / "bundle.npz"; h5_path = NG.source_h5_for_bundle(bundle_path)
        stl = find_stl(unit_id)
        t = time.perf_counter(); vertices, faces = load_stl(stl); T["load_stl"] = time.perf_counter() - t
        tri = vertices[faces]; total_area = float(0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum())
        t = time.perf_counter(); pts = sample_surface(vertices, faces, 0.5, 20260915); T["resample"] = time.perf_counter() - t
        t = time.perf_counter(); case, aux = build_deployment_case(unit_id, pts, h5_path, bundle_path, stats0, cfg0.data.input_features); T["features"] = time.perf_counter() - t
        with h5py.File(h5_path, "r") as h:
            atlas = atlas_from_h5(h); segments = json.loads(h["geometry"].attrs["atlas_segments"])
        feats = map_points(pts, atlas)
        names = branch_names(segments)
        t = time.perf_counter(); preds = []
        for cfg, feat_stats, model, st in models:
            with torch.no_grad():
                p = E.predict_case_norm(model, case, cfg.data.input_features, feat_stats, device, cfg=cfg)
            preds.append(D.denormalize_wss(np.asarray(p, dtype=np.float64), st))
        if device == "cuda": torch.cuda.synchronize()
        T["infer_5_models"] = time.perf_counter() - t
        P = np.stack(preds); ens = P.mean(0); sd = P.std(0)
        n = len(pts); seg = feats["segment_id"].astype(int); s_root = feats["s_from_root_mm"]; dj = feats["dist_to_junction_mm"]; rad = feats["radius_mm"]
        # --- hotspots
        thr10 = np.quantile(ens, 0.90); thr5 = np.quantile(ens, 0.95); top5 = ens >= thr5
        tree = cKDTree(pts[top5]); pairs = tree.query_pairs(r=2.0 * aux["diag"]["spacing_mm"], output_type="ndarray")
        m = int(top5.sum()); g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(m, m))
        ncomp, lab = connected_components(g, directed=False)
        comp_sizes = np.bincount(lab); big = int(np.sum(comp_sizes >= 20))
        imax = int(np.argmax(ens))
        top10_masks = [P[i] >= np.quantile(P[i], 0.90) for i in range(len(P))]; ens10 = ens >= thr10
        iou_seed_vs_ens = [float(np.sum(mk & ens10) / np.sum(mk | ens10)) for mk in top10_masks]
        # --- per-branch
        branch = {}
        for sid in sorted(set(seg.tolist())):
            mk = seg == sid; v = ens[mk]
            if mk.sum() < 10: continue
            branch[names.get(sid, str(sid))] = {"segment_id": sid, "n_points": int(mk.sum()), "area_mm2_est": float(total_area * mk.mean()),
                "wss_mean_pa": float(v.mean()), "wss_p99_pa": float(np.quantile(v, .99)), "wss_max_pa": float(v.max()),
                "frac_low_lt_0p4": float(np.mean(v < LOW_PA)), "frac_high_gt_4": float(np.mean(v > HIGH_PA)),
                "share_of_top10_hotspot": float(np.sum(mk & ens10) / np.sum(ens10)), "radius_mean_mm": float(rad[mk].mean()), "seed_sd_median_pa": float(np.median(sd[mk]))}
        # --- geometry from atlas
        tab, cols = atlas.table, list(atlas.columns); col = lambda c: tab[:, cols.index(c)]
        geo = {}
        for s in segments:
            sid = int(s["segment_id"]); mk = col("segment_id").astype(int) == sid; r = col("radius_mm")[mk]; xyz = np.stack([col("x_mm")[mk], col("y_mm")[mk], col("z_mm")[mk]], 1)
            if mk.sum() < 3: continue
            chord = float(np.linalg.norm(xyz[-1] - xyz[0])); L = float(s.get("length_mm", 0.0))
            geo[names.get(sid, str(sid))] = {"length_mm": L, "radius_min_mm": float(r.min()), "radius_median_mm": float(np.median(r)), "radius_max_mm": float(r.max()),
                "max_diameter_mm": float(2 * r.max()), "stenosis_index_1_minus_rmin_over_rmedian": float(1 - r.min() / np.median(r)),
                "tortuosity_len_over_chord": float(L / chord) if chord > 0 else None, "curvature_max_per_mm": float(col("curvature_per_mm")[mk].max())}
        d = aux["diag"]
        rep = {"unit_id": unit_id, "stl": str(stl), "device": device, "release": "X5D_v51_5seed_20260916",
          "input_check": {"stl_vertices": int(len(vertices)), "stl_faces": int(len(faces)), "stl_area_mm2": total_area, "bbox_diag_mm": float(np.linalg.norm(vertices.max(0) - vertices.min(0))),
                          "points_after_resample": n, "median_spacing_mm": d["spacing_mm"], "surface_variation_median": d["surface_variation_median"],
                          "junction_ambiguous_fraction": d["junction_ambiguous_fraction"], "caps_mm": d["caps"], "murray_flow_shares": d["murray_shares"]},
          "wss_field_pa": {**stats(ens), "top10_threshold_pa": float(thr10), "top5_threshold_pa": float(thr5),
                          "area_frac_low_lt_0p4": float(np.mean(ens < LOW_PA)), "area_frac_high_gt_4": float(np.mean(ens > HIGH_PA)), "area_frac_gt_7": float(np.mean(ens > VERY_HIGH_PA)),
                          "area_low_mm2": float(total_area * np.mean(ens < LOW_PA)), "area_high_mm2": float(total_area * np.mean(ens > HIGH_PA))},
          "peak_location": {"branch": names.get(int(seg[imax]), str(seg[imax])), "s_from_inlet_mm": float(s_root[imax]), "dist_to_nearest_junction_mm": float(dj[imax]),
                            "local_radius_mm": float(rad[imax]), "xyz_mm": pts[imax].round(2).tolist(), "value_pa": float(ens[imax]),
                            "top5_hotspot_clusters_ge20pts": big, "top5_hotspot_cluster_sizes": sorted(comp_sizes.tolist(), reverse=True)[:5]},
          "uncertainty": {"seed_sd_pa_median": float(np.median(sd)), "seed_sd_pa_p90": float(np.quantile(sd, .9)), "seed_sd_at_peak_pa": float(sd[imax]),
                          "seed_sd_over_mean_median": float(np.median(sd / np.maximum(ens, 1e-3))), "top10_iou_seed_vs_ensemble": iou_seed_vs_ens, "per_seed_p99_pa": [float(np.quantile(p, .99)) for p in P]},
          "population_percentile_train136": {"p99": percentile_of(np.quantile(ens, .99), ref_p99), "max": percentile_of(ens.max(), ref_max), "mean": percentile_of(ens.mean(), ref_mean),
                                             "reference_p99_pa_median": float(np.median(ref_p99)), "reference_p99_pa_p10_p90": [float(np.quantile(ref_p99, .1)), float(np.quantile(ref_p99, .9))]},
          "per_branch": branch, "geometry": geo, "timing_s": {**{k: round(v, 2) for k, v in T.items()}, "total_after_centerline": round(time.perf_counter() - t_all, 2)}}
        tag = unit_id.replace("/", "__")
        json.dump(rep, open(OUT / f"{tag}.metrics.json", "w"), indent=1, ensure_ascii=False)
        np.savez_compressed(OUT / f"{tag}.field.npz", pts=pts.astype(np.float32), wss_pa=ens.astype(np.float32), seed_sd_pa=sd.astype(np.float32), seed_pred_pa=P.astype(np.float32),
                            segment_id=seg.astype(np.int16), s_from_root_mm=s_root.astype(np.float32), theta_rad=feats["theta_rad"].astype(np.float32), radius_mm=rad.astype(np.float32),
                            atlas_table=tab.astype(np.float32), atlas_columns=np.array(cols), branch_names=np.array(json.dumps(names, ensure_ascii=False)), stl_vertices=vertices.astype(np.float32), stl_faces=faces.astype(np.int32))
        print(unit_id, json.dumps({k: rep[k] for k in ("wss_field_pa", "peak_location", "population_percentile_train136", "timing_s")}, ensure_ascii=False)[:600], flush=True)

if __name__ == "__main__":
    main(sys.argv[1:] or ["AAA/ruputer/KANG_YONG"])
