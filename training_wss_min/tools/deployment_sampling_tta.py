"""Test-time sampling ensemble for the deployment form (inference only).

Deployment resamples the surface at ~0.5 mm before inference (§22.3).  A single random resampling carries
sampling variance (which points were drawn, hence which normals / curvatures / K16 patches the model sees).
Averaging the predictions of several independent resamplings of the *same* surface is free at inference time
and stacks with the seed ensemble.  This tool measures it on the CFD wall nodes of every test case:

  for k in 1..K:  resample the STL at the target spacing with seed k -> full deployment case -> per-model
                  prediction on the resampled points -> interpolate to the CFD nodes (inverse-distance, 3 NN).

Reported on the identical CFD-node point set (so the reference is the standard test34 number):
  single      = resampling 1 only;
  tta_<k>     = mean over the first k resamplings (per model, log space), then the seed ensemble;
  reference   = the saved full-cloud predictions of the same checkpoints.
Both ensemble conventions are given (log-mean as in §22 tables, Pa-mean as the deployment convention), plus the
per-case spread of the single-resampling R² across the K draws (sampling variance).

    CUDA_VISIBLE_DEVICES=2 python -m training_wss_min.tools.deployment_sampling_tta --out <dir> --draws 5
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from training_wss_min import config as C
from training_wss_min import dataset as D
from training_wss_min import evaluate as E
from training_wss_min import next_geometry as NG
from training_wss_min.surface import load_stl
from training_wss_min.tools.deployment_resample_reeval import DEFAULT_RUNS, summarise
from training_wss_min.tools.deployment_stl_simulation import build_deployment_case, find_stl, sample_surface


def to_nodes(pts: np.ndarray, values: np.ndarray, nodes: np.ndarray, k: int = 3) -> np.ndarray:
    d, idx = cKDTree(pts).query(nodes, k=k)
    w = 1.0 / np.maximum(d, 1e-6)
    return (values[idx] * w).sum(axis=1) / w.sum(axis=1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=[str(p) for p in DEFAULT_RUNS])
    ap.add_argument("--spacing", type=float, default=0.5)
    ap.add_argument("--draws", type=int, default=5)
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
    names = [m["name"] for m in models]
    pairs = D.load_split_cases(cfg0.data.split_path, "test")
    if args.limit_cases:
        pairs = pairs[: args.limit_cases]
    units, true_pa, true_n, ref_n = [], [], [], {nm: [] for nm in names}
    draws_n = {nm: [] for nm in names}      # per model: list over cases of [K, n_nodes] arrays
    diag, skipped = [], []
    started = time.time()
    for ci, (cohort, name) in enumerate(pairs):
        unit_id = f"{cohort}/{name}"
        bundle_path = Path(cfg0.data.data_root) / cohort / name / "bundle.npz"
        stl = find_stl(unit_id)
        if stl is None:
            skipped.append({"unit_id": unit_id, "reason": "no STL found"}); continue
        vertices, faces = load_stl(stl)
        full = D.load_case(cohort, name, stats0, target="wss", target_normalization=cfg0.data.target_normalization,
                           data_root=cfg0.data.data_root, required_frame_version=cfg0.data.required_frame_version,
                           extra_point_features=C.v6_point_features(cfg0), point_features_root=cfg0.data.point_features_root)
        nodes = np.asarray(full["wall_coords_raw"], dtype=np.float64)
        d_frame, _ = cKDTree(nodes).query(vertices[:: max(1, len(vertices) // 5000)], k=1)
        if np.median(d_frame) > 1.0:
            skipped.append({"unit_id": unit_id, "reason": f"STL frame mismatch (median {np.median(d_frame):.2f} mm)"}); continue
        h5_path = NG.source_h5_for_bundle(bundle_path)
        n = len(nodes)
        for m in models:
            with np.load(m["run"] / "eval" / f"ckpt_{args.checkpoint}" / "predictions/test" / unit_id / "predictions.npz") as z:
                ref = np.full(n, np.nan); ref[z["row_index"]] = z["pred_norm"]
                if not np.isfinite(ref).all():
                    raise ValueError(f"saved predictions do not cover all wall rows: {unit_id}")
                ref_n[m["name"]].append(ref)
        units.append(unit_id); true_pa.append(np.asarray(full["y_raw"], dtype=np.float64)); true_n.append(np.asarray(full["y_norm"], dtype=np.float64))
        per_model = {nm: [] for nm in names}
        spacings, map_dist = [], []
        for k in range(args.draws):
            pts = sample_surface(vertices, faces, args.spacing, args.seed + ci + 100003 * k)
            case, aux = build_deployment_case(unit_id, pts, h5_path, bundle_path, stats0, cfg0.data.input_features)
            spacings.append(aux["diag"]["spacing_mm"])
            d, _ = cKDTree(pts).query(nodes, k=1); map_dist.append(float(np.median(d)))
            for m in models:
                with torch.no_grad():
                    pred = E.predict_case_norm(m["model"], case, m["cfg"].data.input_features, m["feat_stats"], device, cfg=m["cfg"])
                per_model[m["name"]].append(to_nodes(pts, np.asarray(pred, dtype=np.float64), nodes))
        for nm in names:
            draws_n[nm].append(np.stack(per_model[nm]))
        diag.append({"unit_id": unit_id, "n_nodes": n, "spacing_mm_draws": spacings, "node_to_sample_mm_median": map_dist})
        print(f"[{ci + 1}/{len(pairs)}] {unit_id} n_nodes={n} draws={args.draws} {time.time() - started:.0f}s", flush=True)

    def ens_rows(pred_by_model):
        """pred_by_model: {name: list over cases of [n] arrays} -> (log-mean summary, Pa-mean summary)."""
        log_ens = [np.mean([pred_by_model[nm][i] for nm in names], axis=0) for i in range(len(units))]
        pa_ens = [np.mean([D.denormalize_wss(pred_by_model[nm][i], stats0) for nm in names], axis=0) for i in range(len(units))]
        return (summarise(true_pa, [D.denormalize_wss(p, stats0) for p in log_ens], true_n, log_ens),
                summarise(true_pa, pa_ens, true_n, [D.normalize_wss(p, stats0) for p in pa_ens]))

    report = {"runs": [str(m["run"]) for m in models], "checkpoint": args.checkpoint, "spacing_mm": args.spacing, "draws": args.draws,
              "n_cases": len(units), "units": units, "skipped": skipped, "diag": diag, "rows": []}
    ref_log, ref_pa = ens_rows(ref_n)
    report["rows"].append({"label": "reference_full_cloud", "ensemble": ref_log, "ensemble_pa_mean": ref_pa,
                           "seeds": {nm: summarise(true_pa, [D.denormalize_wss(p, stats0) for p in ref_n[nm]], true_n, ref_n[nm]) for nm in names}})
    for k in range(1, args.draws + 1):
        pred = {nm: [draws_n[nm][i][:k].mean(axis=0) for i in range(len(units))] for nm in names}
        log_s, pa_s = ens_rows(pred)
        row = {"label": "single" if k == 1 else f"tta_{k}", "k": k, "ensemble": log_s, "ensemble_pa_mean": pa_s,
               "seeds": {nm: summarise(true_pa, [D.denormalize_wss(p, stats0) for p in pred[nm]], true_n, pred[nm]) for nm in names}}
        report["rows"].append(row)
    # sampling variance: per case, ensemble R² of each single draw
    per_draw = []
    for k in range(args.draws):
        pred = {nm: [draws_n[nm][i][k] for i in range(len(units))] for nm in names}
        per_draw.append(ens_rows(pred)[0]["per_case_r2"])
    per_draw = np.asarray(per_draw)   # [K, cases]
    report["single_draw_case_r2_sd_median"] = float(np.median(per_draw.std(axis=0)))
    report["single_draw_case_r2_sd_p90"] = float(np.quantile(per_draw.std(axis=0), 0.9))
    report["single_draw_pa_r2_cb_by_draw"] = [float(x) for x in [ens_rows({nm: [draws_n[nm][i][k] for i in range(len(units))] for nm in names})[0]["pa_r2_cb"] for k in range(args.draws)]]
    (out_dir / "sampling_tta.json").write_text(json.dumps(report, indent=1))
    lines = [f"CFD-node evaluation, {len(units)} cases, spacing {args.spacing} mm, {args.draws} draws; single-draw case R² sd median {report['single_draw_case_r2_sd_median']:.4f} / p90 {report['single_draw_case_r2_sd_p90']:.4f}; single-draw ensemble Pa R²_cb by draw: " + ", ".join(f"{x:.4f}" for x in report["single_draw_pa_r2_cb_by_draw"]),
             "", "| form | Pa R²_cb log-mean | Pa R²_cb Pa-mean | norm R²_cb | case mean / P10 | MAE Pa | high-WSS R² | top10 比 | p99 比 | IoU | seed Pa R²_cb mean ± sd |",
             "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---|"]
    for e in report["rows"]:
        s, p = e["ensemble"], e["ensemble_pa_mean"]; seeds = [v["pa_r2_cb"] for v in e["seeds"].values()]
        lines.append(f"| {e['label']} | {s['pa_r2_cb']:.4f} | {p['pa_r2_cb']:.4f} | {p['norm_r2_cb']:.4f} | {p['case_r2_mean']:.3f} / {p['case_r2_p10']:.3f} | {p['mae_pooled']:.3f} | "
                     f"{p['high_wss_r2']:.3f} | {p['top10_ratio']:.3f} | {p['p99_ratio']:.3f} | {p['top10_iou_mean']:.3f} | {np.mean(seeds):.4f} ± {np.std(seeds):.4f} |")
    if skipped:
        lines.append(""); lines.append("skipped: " + "; ".join(f"{s['unit_id']} ({s['reason']})" for s in skipped))
    (out_dir / "sampling_tta.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
