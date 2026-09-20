"""Summarize existing X5/X5D test34 caches; no training or model inference.

Section diagnostics reproduce the historical 4 mm axial-bin definition.
Pa-mean and log_z-mean ensembles are reported separately. All geometry is
aligned through the cache row_index, and seed metrics are checked against JSON.
"""
from pathlib import Path
import json
import sys
from datetime import datetime, timezone

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from training_wss_min import metrics as M

RUNS = ROOT / "training_wss_min/runs"
VIEW = ROOT / "data_wss_v5/views/wss_min_view_v1"
SEEDS = (1234, 7, 2025)
MODELS = {
    "X5": [RUNS / ("wss_local_wave1_20260912" if s == 1234 else
                    "wss_local_wave1b_20260912") / f"X5_s{s}" for s in SEEDS],
    "X5D": [RUNS / "wss_local_wave6_20260915" / f"X5D_s{s}" for s in SEEDS],
}
stats = json.loads((VIEW / "wss_global_stats_train138.json").read_text())
mu, sigma = stats["log"]["mean"], stats["log"]["std"]
units = sorted(json.loads((VIEW / "split_V5_train138_test34.json").read_text())["test_cases"])


def load(run, unit):
    with np.load(run / "eval/ckpt_best/predictions/test" / unit / "predictions.npz") as z:
        return {k: z[k].copy() for k in
                ("true_pa", "pred_pa", "true_norm", "pred_norm", "row_index")}


def summarize(truth, predictions):
    pc = [M.r2_score(t, p) for t, p in zip(truth, predictions)]
    iou = []
    for t, p in zip(truth, predictions):
        a, b = t >= np.quantile(t, .9), p >= np.quantile(p, .9)
        iou.append(float(np.sum(a & b) / np.sum(a | b)))
    cal = M.calibration_metrics(np.concatenate(truth), np.concatenate(predictions))
    return {
        "pa_r2_cb": M.casebalanced_field_metrics(truth, predictions)["r2"],
        "pa_mae_cb": M.casebalanced_field_metrics(truth, predictions)["mae"],
        "case_r2_mean": float(np.mean(pc)), "case_r2_p10": float(np.quantile(pc, .1)),
        "top10_ratio_pooled_global_p90": cal["top10_pred_true_ratio"],
        "p99_ratio_pooled": cal["p99_pred_true_ratio"],
        "top10_iou_case_mean": float(np.mean(iou)),
        "per_case_pa_r2": dict(zip(units, pc)),
    }


output = {
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "scope": "Existing ckpt_best prediction caches only; test34 is development-exposed.",
    "protocol": {"seeds": SEEDS, "n_cases": len(units), "checkpoint": "ckpt_best",
                 "selection": "train_loss", "section_bin_mm": 4.0,
                 "section_aggregation": "median of within-case variance shares / R2",
                 "norm_ensemble": "arithmetic mean in log_z; separate from log(Pa mean)",
                 "tail_threshold": "pooled global true p90; IoU uses each case p90"},
    "models": {},
}
cached = {}
for name, runs in MODELS.items():
    data = [[load(run, u) for u in units] for run in runs]
    for seed_data in data:
        for first, item in zip(data[0], seed_data):
            for k in ("row_index", "true_pa", "true_norm"):
                np.testing.assert_array_equal(first[k], item[k])
            assert np.isfinite(item["pred_pa"]).all() and np.isfinite(item["pred_norm"]).all()
    truth = [d["true_pa"].astype(float) for d in data[0]]
    truth_norm = [d["true_norm"].astype(float) for d in data[0]]
    single = []
    for run, seed_data in zip(runs, data):
        expected = json.loads((run / "eval/ckpt_best/metrics.json").read_text())["test"]
        value = M.casebalanced_field_metrics(truth, [d["pred_pa"] for d in seed_data])["r2"]
        np.testing.assert_allclose(value, expected["field_casebalanced"]["r2"], atol=1e-10, rtol=0)
        single.append(value)
    pn = [np.mean([d[i]["pred_norm"] for d in data], axis=0) for i in range(len(units))]
    pp = [np.mean([d[i]["pred_pa"] for d in data], axis=0) for i in range(len(units))]
    plog = [np.maximum(np.exp(p * sigma + mu) - stats["eps"], 0) for p in pn]
    details = []
    for i, u in enumerate(units):
        if name == "X5":
            with np.load(VIEW / u / "bundle.npz", allow_pickle=True) as b:
                idx = data[0][i]["row_index"]
                seg = b["wall_segment_id"][idx].astype(int)
                s = b["wall_s_local_mm"][idx].astype(float)
            _, inv, cnt = np.unique(np.column_stack([seg, np.floor(s / 4).astype(int)]),
                                    axis=0, return_inverse=True, return_counts=True)
            cached[u] = (data[0][i], inv, cnt, seg)
        ref, inv, cnt, seg = cached[u]
        for k in ("row_index", "true_pa", "true_norm"):
            np.testing.assert_array_equal(ref[k], data[0][i][k])
        t, p = truth_norm[i], pn[i]
        tm = (np.bincount(inv, weights=t) / cnt)[inv]
        pm = (np.bincount(inv, weights=p) / cnt)[inv]
        r, rm = t - p, tm - pm
        _, bi, bc = np.unique(seg, return_inverse=True, return_counts=True)
        br = (np.bincount(bi, weights=r) / bc)[bi]
        q = np.quantile(t, np.linspace(0, 1, 11))
        dec = np.clip(np.searchsorted(q, t, side="right") - 1, 0, 9)
        details.append({
            "unit": u,
            "section_mean_share": float(np.var(rm) / np.var(r)),
            "within_section_share": float(np.var(r - rm) / np.var(r)),
            "branch_mean_share": float(np.var(br) / np.var(r)),
            "section_track_r2": M.r2_score(tm, pm),
            "within_section_pattern_r2": M.r2_score(t - tm, p - pm),
            "residual_pred_minus_true_deciles": [float(np.mean((p - t)[dec == k])) for k in range(10)],
        })
    output["models"][name] = {
        "sources": [str(r / "eval/ckpt_best") for r in runs],
        "single_seed_pa_r2_verified": single,
        "pa_mean": summarize(truth, pp), "log_mean": summarize(truth, plog),
        "log_mean_norm_r2_cb": M.casebalanced_field_metrics(truth_norm, pn)["r2"],
        "pa_mean_back_to_norm_r2_cb": M.casebalanced_field_metrics(
            truth_norm, [(np.log(p + stats["eps"]) - mu) / sigma for p in pp])["r2"],
        "residual_medians": {k: float(np.median([d[k] for d in details])) for k in
                             ("section_mean_share", "within_section_share", "branch_mean_share",
                              "section_track_r2", "within_section_pattern_r2")},
        "residual_decile_case_medians": np.median(
            [d["residual_pred_minus_true_deciles"] for d in details], axis=0).tolist(),
        "per_case_residual": details,
    }
    print(name, json.dumps({k: v for k, v in output["models"][name].items()
                           if k in ("single_seed_pa_r2_verified", "residual_medians",
                                    "residual_decile_case_medians", "log_mean_norm_r2_cb",
                                    "pa_mean_back_to_norm_r2_cb")}, ensure_ascii=False))
    print(name, "Pa mean", json.dumps({k: v for k, v in output["models"][name]["pa_mean"].items()
                                      if k != "per_case_pa_r2"}))
dest = Path(__file__).with_name("cached_residual_analysis.json")
dest.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
print(dest)
