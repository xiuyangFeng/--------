"""Stage timing for the deployment path: STL -> resample 0.5 mm -> features -> 5-model inference -> Pa ensemble.
Reuses the exact functions of training_wss_min.tools.deployment_stl_simulation (inference only, no training)."""
import json, sys, time
from pathlib import Path
import numpy as np, torch
from scipy.spatial import cKDTree
from training_wss_min import config as C, dataset as D, evaluate as E, next_geometry as NG
from training_wss_min.surface import load_stl
from training_wss_min.tools.deployment_stl_simulation import find_stl, sample_surface, build_deployment_case
from wss_v5.pointcloud import median_spacing
from wss_v5.views import wall_flowref_v1 as WF
WF._CAPFIT_RULE_OVERRIDE = Path("/public/newhome/cy/Digital_twin/GNN/data_wss_v5/views_v5_1/wss_min_flowref_v1/flow_split_rule_train136.json")

ROOT = C.PROJECT_ROOT
RUNS = [ROOT / "training_wss_min/runs/wss_v51_wave1_20260916" / f"X5D_v51_s{s}" for s in (1234, 7, 2025, 11, 2026)]
device = "cuda" if torch.cuda.is_available() else "cpu"
cases = sys.argv[1:] or ["AAA/ruputer/KANG_YONG"]
spacing = 0.5
T = {}
t0 = time.perf_counter()
models = []
for run in RUNS:
    cfg, feat_stats, model, _ = E.load_model_from_run(run, device, "best"); model.eval()
    models.append((cfg, feat_stats, model, E.load_wss_stats_for_run(run)))
T["load_5_models_s"] = time.perf_counter() - t0
cfg0, stats0 = models[0][0], models[0][3]
rows = []
for unit_id in cases:
    cohort, name = unit_id.rsplit("/", 1)
    bundle_path = Path(cfg0.data.data_root) / cohort / name / "bundle.npz"
    h5_path = NG.source_h5_for_bundle(bundle_path)
    stl = find_stl(unit_id)
    r = {"unit_id": unit_id, "stl": str(stl), "device": device, "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu"}
    t = time.perf_counter(); vertices, faces = load_stl(stl); r["t_load_stl_s"] = time.perf_counter() - t
    r["stl_vertices"] = int(len(vertices)); r["stl_faces"] = int(len(faces))
    t = time.perf_counter(); pts = sample_surface(vertices, faces, spacing, 20260915); r["t_resample_s"] = time.perf_counter() - t
    r["n_points"] = int(len(pts)); r["median_spacing_mm"] = float(median_spacing(pts))
    t = time.perf_counter(); case, aux = build_deployment_case(unit_id, pts, h5_path, bundle_path, stats0, cfg0.data.input_features); r["t_features_s"] = time.perf_counter() - t
    preds_pa = []; per_model = []
    if device == "cuda": torch.cuda.synchronize()
    t_all = time.perf_counter()
    for cfg, feat_stats, model, stats in models:
        t = time.perf_counter()
        with torch.no_grad():
            pred = E.predict_case_norm(model, case, cfg.data.input_features, feat_stats, device, cfg=cfg)
        if device == "cuda": torch.cuda.synchronize()
        per_model.append(time.perf_counter() - t)
        preds_pa.append(D.denormalize_wss(np.asarray(pred, dtype=np.float64), stats))
    r["t_infer_5_models_s"] = time.perf_counter() - t_all; r["t_infer_per_model_s"] = per_model
    t = time.perf_counter(); ens = np.mean(np.stack(preds_pa), axis=0); r["t_ensemble_s"] = time.perf_counter() - t
    y = case["y_raw"].astype(np.float64)
    ss = float(np.sum((y - ens) ** 2)); st = float(np.sum((y - y.mean()) ** 2))
    r["r2_pa_vs_nearest_cfd_node"] = 1 - ss / st
    r["peak_wss_pred_pa_p99"] = float(np.quantile(ens, 0.99)); r["peak_wss_true_pa_p99"] = float(np.quantile(y, 0.99))
    r["peak_wss_pred_pa_max"] = float(ens.max()); r["peak_wss_true_pa_max"] = float(y.max())
    r["t_total_after_centerline_s"] = r["t_load_stl_s"] + r["t_resample_s"] + r["t_features_s"] + r["t_infer_5_models_s"] + r["t_ensemble_s"]
    rows.append(r); print(json.dumps(r, ensure_ascii=False), flush=True)
out = Path(sys.argv[0]).parent / "deploy_stage_timing.json"
json.dump({"timing_global": T, "cases": rows, "spacing_mm": spacing, "runs": [str(x) for x in RUNS]}, open(out, "w"), indent=1, ensure_ascii=False)
print("wrote", out)
