"""Which model neighbourhood carries the density loss? (inference only, frozen features)

§22.3 showed that after density augmentation (X5D) the *feature-recompute* layer of the density loss is almost
gone and what remains is the *model-neighbourhood* layer.  That layer has two point sets the model builds
neighbourhoods on:

* the **support** cloud (encoder SA hierarchy incl. the knn_cover first stage, and the directional local
  branch, both computed on the support points), and
* the **full-wall K16 query patch** (``build_query_patch`` on the case's own point cloud).

This tool decimates the CFD wall nodes exactly like ``deployment_resample_reeval`` (voxel decimation, frozen
full-cloud feature values on the retained points) and then predicts the retained nodes with every combination
of  support ∈ {thin, full} × patch ∈ {thin, full}:

    thin/thin  = the existing "frozen" variant (control: must reproduce §22.3);
    full/thin  = support sampled from the full cloud, K16 patch from the thinned cloud  -> patch effect only;
    thin/full  = support from the thinned cloud, K16 patch from the full cloud          -> support effect only;
    full/full  = both from the full cloud (control: must reproduce the saved full-cloud predictions, Δ = 0).

Query points are always the retained nodes, truth is the CFD WSS there, reference is the saved full-cloud
prediction of the same checkpoint restricted to the same nodes; the support seed is the evaluation seed, so
"full" support is the very same 5000 points the saved predictions used.

    CUDA_VISIBLE_DEVICES=2 python -m training_wss_min.tools.deployment_neighbourhood_decomposition --out <dir>
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from training_wss_min import config as C
from training_wss_min import dataset as D
from training_wss_min import evaluate as E
from training_wss_min import next_geometry as NG
from training_wss_min import surface as S
from training_wss_min.tools.deployment_resample_reeval import (DEFAULT_RUNS, coarse_case, summarise, voxel_rows,
                                                               wall_source)

VARIANTS = ["thin/thin", "full/thin", "thin/full", "full/full"]
# diagnostic patch sources (support thin):  thin_rescaled = thin patch with its mm offsets multiplied by the case's
# empirical K16-radius ratio full/thin (scale back to the full-density radius, features/points stay those of the
# thin cloud) -> tests whether the patch loss is an offset-scale shift;  full_inflated = full-cloud patch with its
# offsets divided by the same ratio -> tests whether inflating the scale alone (with the fine features) hurts.


@torch.no_grad()
def predict_mixed(m: dict, full: dict, sub: dict, rows: np.ndarray, support_source: str, patch_source: str,
                  device: str) -> np.ndarray:
    """Predict every point of ``sub`` (= ``full`` restricted to ``rows``) with the support and the K16 patch taken
    from either the thinned (``sub``) or the full cloud."""
    cfg, feat_stats, model = m["cfg"], m["feat_stats"], m["model"]
    input_features = cfg.data.input_features
    if not cfg.eval.fixed_support:
        raise ValueError("decomposition requires eval.fixed_support=True")
    support_case = full if support_source == "full" else sub
    support_n = int(cfg.data.support_n_points or cfg.data.wall_n_points)
    sampling = cfg.data.support_sampling or cfg.data.sampling
    seed = S.stable_seed(cfg.eval.support_seed, full["unit_id"], "eval_support")
    support_idx = D.sample_support_indices(support_case, cfg.data, seed, n_points=support_n, sampling=sampling,
                                           stream="eval_support")
    support_pos = torch.from_numpy(np.ascontiguousarray(support_case["pos"][support_idx])).to(device)
    support_x = torch.from_numpy(D.build_features(support_case, support_idx, input_features, feat_stats)).to(device)
    support_batch = torch.zeros(len(support_idx), dtype=torch.long, device=device)
    extra = {}
    if cfg.data.local_geometry:
        geometry_s = D.case_geometry(support_case)
        extra["geometry"] = torch.from_numpy(np.ascontiguousarray(geometry_s[support_idx])).to(device)
    if getattr(cfg.data, "section_tokens", False):
        raise NotImplementedError("section tokens are not part of the X5/X5D recipe")
    encoded = model.encode_support(support_pos, support_x, support_batch, unit_ids=[full["unit_id"]], epoch=0,
                                   global_seed=cfg.eval.support_seed, evaluation=True, **extra)
    q_rows = np.arange(len(sub["pos"]))
    chunk = int(cfg.eval.query_chunk_size) or len(q_rows)
    geometry_q = D.case_geometry(sub) if cfg.data.local_geometry else None
    scale = 1.0
    if patch_source in ("thin_rescaled", "full_inflated") and cfg.data.query_patch_nsample:
        # empirical K16 radius ratio full/thin on a fixed subset of query rows (median of the farthest-neighbour distance)
        k = int(cfg.data.query_patch_nsample)
        probe = np.random.default_rng(0).choice(len(q_rows), size=min(2000, len(q_rows)), replace=False)
        r_full = np.linalg.norm(D.build_query_patch(full, rows[probe], input_features, feat_stats, nsample=k)["relative_mm"], axis=-1).max(axis=1)
        r_thin = np.linalg.norm(D.build_query_patch(sub, probe, input_features, feat_stats, nsample=k)["relative_mm"], axis=-1).max(axis=1)
        scale = float(np.median(r_full) / max(np.median(r_thin), 1e-9))
        m.setdefault("_patch_scale", {}).setdefault(len(sub["pos"]) / len(full["pos"]), {})[full["unit_id"]] = scale
    outputs = []
    for start in range(0, len(q_rows), chunk):
        idx = q_rows[start:start + chunk]
        pos = torch.from_numpy(np.ascontiguousarray(sub["pos"][idx])).to(device)
        x = torch.from_numpy(D.build_features(sub, idx, input_features, feat_stats)).to(device)
        batch = torch.zeros(len(idx), dtype=torch.long, device=device)
        dec = {}
        if geometry_q is not None:
            dec["geometry"] = torch.from_numpy(np.ascontiguousarray(geometry_q[idx])).to(device)
        if cfg.data.query_patch_nsample:
            k = int(cfg.data.query_patch_nsample)
            if patch_source in ("full", "full_inflated"):
                patch = D.build_query_patch(full, rows[idx], input_features, feat_stats, nsample=k)
            else:
                patch = D.build_query_patch(sub, idx, input_features, feat_stats, nsample=k)
            if patch_source == "thin_rescaled":      # thin-cloud points/features, offsets shrunk to the full-density radius
                patch = {**patch, "relative_mm": (patch["relative_mm"] * scale).astype(np.float32)}
            elif patch_source == "full_inflated":    # full-cloud points/features, offsets inflated to the thin-density radius
                patch = {**patch, "relative_mm": (patch["relative_mm"] / scale).astype(np.float32)}
            dec["patch"] = {key: torch.as_tensor(v, device=device) for key, v in patch.items()}
        outputs.append(model.decode_query(encoded, pos, x, batch, **dec).detach().float().cpu())
    out = torch.cat(outputs)
    if out.ndim == 2:
        out = out[:, 0]
    return out.numpy()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=[str(p) for p in DEFAULT_RUNS])
    ap.add_argument("--fractions", nargs="*", type=float, default=[0.5, 0.25, 0.1])
    ap.add_argument("--variants", nargs="*", default=VARIANTS)
    ap.add_argument("--seed", type=int, default=20260913, help="same decimation seed as deployment_resample_reeval")
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
    keys = [(f, v) for f in args.fractions for v in args.variants]
    results = {k: {"true_pa": [], "true_n": [], "pred_n": {m["name"]: [] for m in models},
                   "ref_n": {m["name"]: [] for m in models}, "n_points": []} for k in keys}
    started = time.time()
    for ci, (cohort, name) in enumerate(pairs):
        full = D.load_case(cohort, name, stats0, target="wss", target_normalization=cfg0.data.target_normalization,
                           data_root=cfg0.data.data_root, required_frame_version=cfg0.data.required_frame_version,
                           extra_point_features=C.v6_point_features(cfg0), point_features_root=cfg0.data.point_features_root)
        NG.case_geometry(full)
        src = wall_source(full)
        n = len(full["pos"])
        saved = {}
        for m in models:
            with np.load(m["run"] / "eval" / f"ckpt_{args.checkpoint}" / "predictions/test" / full["unit_id"] / "predictions.npz") as z:
                ref = np.full(n, np.nan); ref[z["row_index"]] = z["pred_norm"]
                saved[m["name"]] = ref
        for frac in args.fractions:
            rows = voxel_rows(src["xyz"], frac, args.seed + ci)
            sub = coarse_case(full, rows)          # frozen full-cloud feature values on the retained rows
            for variant in args.variants:
                support_source, patch_source = variant.split("/")
                store = results[(frac, variant)]
                store["true_pa"].append(np.asarray(sub["y_raw"], dtype=np.float64))
                store["true_n"].append(np.asarray(sub["y_norm"], dtype=np.float64))
                store["n_points"].append(int(len(rows)))
                for m in models:
                    pred = predict_mixed(m, full, sub, rows, support_source, patch_source, device)
                    store["pred_n"][m["name"]].append(np.asarray(pred, dtype=np.float64))
                    store["ref_n"][m["name"]].append(saved[m["name"]][rows])
        print(f"[{ci + 1}/{len(pairs)}] {full['unit_id']} n={n} {time.time() - started:.0f}s", flush=True)
    report = {"runs": [str(m["run"]) for m in models], "checkpoint": args.checkpoint, "fractions": args.fractions,
              "variants": args.variants, "features": "frozen", "n_cases": len(pairs),
              "units": [f"{a}/{b}" for a, b in pairs], "rows": []}
    for (frac, variant), store in results.items():
        true_pa, true_n = store["true_pa"], store["true_n"]
        names = list(store["pred_n"])
        entry = {"fraction": frac, "variant": variant, "support": variant.split("/")[0], "patch": variant.split("/")[1],
                 "points_median": float(np.median(store["n_points"])), "seeds": {},
                 "patch_scale_median": float(np.median([v for f, d in models[0].get("_patch_scale", {}).items()
                                                        if abs(f - frac) < 0.05 for v in d.values()] or [1.0]))}
        for nm in names:
            pred_n, ref_n = store["pred_n"][nm], store["ref_n"][nm]
            entry["seeds"][nm] = {"mixed": summarise(true_pa, [D.denormalize_wss(p, stats0) for p in pred_n], true_n, pred_n),
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
    (out_dir / "neighbourhood_decomposition.json").write_text(json.dumps(report, indent=1))
    lines = ["| fraction | support | patch | points/case | ens Pa R²_cb (log-mean) | ref (same points) | Δ | Δ (Pa-mean ens) | norm R²_cb (ens / ref) | case R² mean / P10 | high-WSS R² | top10 比 | IoU | seed Pa R²_cb (mean ± sd) | seed Δ (mean ± sd) |",
             "|---|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---|---|"]
    for e in report["rows"]:
        s, r = e["ensemble"], e["reference_ensemble"]
        sp, rp = e["ensemble_pa_mean"], e["reference_ensemble_pa_mean"]
        seeds = [v["mixed"]["pa_r2_cb"] for v in e["seeds"].values()]
        deltas = [v["mixed"]["pa_r2_cb"] - v["reference_same_points"]["pa_r2_cb"] for v in e["seeds"].values()]
        lines.append(f"| {e['fraction']:.2f} | {e['support']} | {e['patch']} | {e['points_median']:.0f} | {s['pa_r2_cb']:.4f} | {r['pa_r2_cb']:.4f} | "
                     f"{s['pa_r2_cb'] - r['pa_r2_cb']:+.4f} | {sp['pa_r2_cb'] - rp['pa_r2_cb']:+.4f} | {s['norm_r2_cb']:.4f} / {r['norm_r2_cb']:.4f} | "
                     f"{s['case_r2_mean']:.3f} / {s['case_r2_p10']:.3f} | {s['high_wss_r2']:.3f} | {s['top10_ratio']:.3f} | {s['top10_iou_mean']:.3f} | "
                     f"{np.mean(seeds):.4f} ± {np.std(seeds):.4f} | {np.mean(deltas):+.4f} ± {np.std(deltas):.4f} |")
    (out_dir / "neighbourhood_decomposition.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
