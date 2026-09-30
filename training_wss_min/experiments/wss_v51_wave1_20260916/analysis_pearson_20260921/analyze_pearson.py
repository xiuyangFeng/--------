"""Compare peak-frame X5D spatial Pearson r with WSSNet's reported metric.

Reads existing ckpt_best predictions only; no inference or checkpoint selection.
Run with python3 from any directory. Outputs are beside this script.
"""
import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
WAVE1 = ROOT / "training_wss_min/runs/wss_v51_wave1_20260916"
CV3 = ROOT / "training_wss_min/runs/wss_v51_wave2a_20260916"
SEEDS = (1234, 7, 2025, 11, 2026)
ROWS = []
SOURCE_RUNS = []


def field_metrics(truths, predictions, casebalanced):
    weights = np.ones(len(truths)) if casebalanced else np.array([x.size for x in truths])
    weights = weights / weights.sum()
    mt = sum(w * t.mean() for w, t in zip(weights, truths))
    mp = sum(w * p.mean() for w, p in zip(weights, predictions))
    vt = sum(w * np.mean((t - mt) ** 2) for w, t in zip(weights, truths))
    vp = sum(w * np.mean((p - mp) ** 2) for w, p in zip(weights, predictions))
    cov = sum(w * np.mean((t - mt) * (p - mp)) for w, t, p in zip(weights, truths, predictions))
    mse = sum(w * np.mean((t - p) ** 2) for w, t, p in zip(weights, truths, predictions))
    mae = sum(w * np.mean(np.abs(t - p)) for w, t, p in zip(weights, truths, predictions))
    slope = cov / vt
    return dict(pearson_r=float(cov / np.sqrt(vt * vp)),
                predictive_r2=float(1 - mse / vt), mae_pa=float(mae),
                fit_slope=float(slope), fit_intercept_pa=float(mp - slope * mt))


def read_run(run, expected_cases):
    config = json.loads((run / "config.json").read_text())
    assert config["data"]["target"] == "wss"
    assert config["data"]["timesteps"] == "peak"
    assert config["model"]["out_dim"] == 1
    assert "views_v5_1" in config["data"]["data_root"]
    evaluation = run / "eval/ckpt_best"
    base = evaluation / "predictions/test"
    data = {}
    for path in sorted(base.glob("*/*/*/predictions.npz")):
        case = str(path.parent.relative_to(base))
        with np.load(path) as z:
            t, p = z["true_pa"].astype(np.float64), z["pred_pa"].astype(np.float64)
            assert t.ndim == 1 and t.shape == p.shape
            assert np.isfinite(t).all() and np.isfinite(p).all()
            data[case] = (t, p, z["row_index"].copy())
    assert len(data) == expected_cases
    metrics = json.loads((evaluation / "metrics.json").read_text())["test"]
    ts, ps = [x[0] for x in data.values()], [x[1] for x in data.values()]
    for cb, key in [(True, "field_casebalanced"), (False, "field")]:
        actual = field_metrics(ts, ps, cb)
        assert abs(actual["predictive_r2"] - metrics[key]["r2"]) < 1e-8
        if "r2_linear_fit" in metrics[key]:
            assert abs(actual["pearson_r"] ** 2 - metrics[key]["r2_linear_fit"]) < 1e-8
    SOURCE_RUNS.append(dict(run=str(run.relative_to(ROOT)),
                            split=config["data"]["split_path"],
                            selection_rule=config["train"].get("selection_rule"),
                            checkpoint="ckpt_best", partition="test", n_cases=len(data),
                            verified_existing_r2=True))
    return data


def summarize(name, data):
    rs = []
    for case, (t, p, _) in sorted(data.items()):
        assert np.var(t) > 0 and np.var(p) > 0
        r = float(np.corrcoef(t, p)[0, 1])
        r2 = float(1 - np.sum((t - p) ** 2) / np.sum((t - t.mean()) ** 2))
        rs.append(r)
        ROWS.append(dict(model=name, case=case, n_points=t.size, pearson_r=r,
                         predictive_r2=r2))
    ts, ps = [x[0] for x in data.values()], [x[1] for x in data.values()]
    return dict(n_cases=len(rs), n_points=sum(x.size for x in ts),
                case_pearson_mean=float(np.mean(rs)),
                case_pearson_sd=float(np.std(rs, ddof=1)),
                case_pearson_median=float(np.median(rs)),
                case_pearson_p10=float(np.quantile(rs, .1)),
                case_pearson_min=float(np.min(rs)),
                casebalanced_field=field_metrics(ts, ps, True),
                pooled_field=field_metrics(ts, ps, False))


def main():
    result = {}
    base = None
    pred_sum = {}
    for seed in SEEDS:
        name = "X5D_v51_s%d" % seed
        data = read_run(WAVE1 / name, 34)
        result[name] = summarize(name, data)
        if base is None:
            base = data
        assert set(base) == set(data)
        for case, (t, p, indices) in data.items():
            assert np.array_equal(t, base[case][0])
            assert np.array_equal(indices, base[case][2])
            if case not in pred_sum:
                pred_sum[case] = np.zeros_like(p)
            pred_sum[case] += p
    ensemble = {case: (t, pred_sum[case] / len(SEEDS), indices)
                for case, (t, _, indices) in base.items()}
    result["X5D_v51_5seed_pa_mean"] = summarize("X5D_v51_5seed_pa_mean", ensemble)
    prior = json.loads((OUT.parent / "offline/analyze_wave1.json").read_text())
    assert abs(result["X5D_v51_5seed_pa_mean"]["casebalanced_field"]["predictive_r2"]
               - prior["X5D_v51 5-seed (Pa mean)"]["pa_r2_cb"]) < 1e-8
    folds = {}
    for k, n in enumerate((46, 46, 44)):
        name = "X5D_v51_f%d_s1234" % k
        data = read_run(CV3 / name, n)
        assert not (set(folds) & set(data))
        folds.update(data)
        result[name] = summarize(name, data)
    assert len(folds) == 136 and not (set(folds) & set(base))
    result["X5D_v51_cv3_oof136"] = summarize("X5D_v51_cv3_oof136", folds)
    prior = json.loads((OUT.parent.parent / "wss_v51_wave2a_20260916/offline/analyze_cv3.json").read_text())
    assert abs(result["X5D_v51_cv3_oof136"]["casebalanced_field"]["predictive_r2"]
               - prior["pooled"]["pa_r2_cb"]) < 1e-8
    output = dict(protocol=dict(
        data="v5.1", target="peak frame 1162 WSS magnitude in Pa",
        inputs="Saved true_pa/pred_pa/row_index only; no inference or training",
        spatial_weighting="legacy_vertex: equal weight per wall point within each case",
        primary="Pearson r calculated separately per case, arithmetic mean and sample SD (ddof=1)",
        ensemble="Five fixed seeds, arithmetic average of pred_pa at identical row_index",
        comparison="WSSNet reports r=0.92 +/- 0.05 on 6 CFD cases x 72 frames; ours is peak only",
        limitations="test34 development-exposed; cv3 used in method screening; different anatomy/input/phase",
        numpy_version=np.__version__), source_runs=SOURCE_RUNS, results=result)
    (OUT / "summary.json").write_text(json.dumps(output, indent=2) + "\n")
    with (OUT / "per_case_pearson.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(ROWS[0]))
        writer.writeheader()
        writer.writerows(ROWS)
    for name, r in result.items():
        print("%s: n=%d Pearson %.6f +/- %.6f; R2_cb %.6f; cb-r %.6f; pooled-r %.6f" % (
            name, r["n_cases"], r["case_pearson_mean"], r["case_pearson_sd"],
            r["casebalanced_field"]["predictive_r2"], r["casebalanced_field"]["pearson_r"],
            r["pooled_field"]["pearson_r"]))


if __name__ == "__main__":
    main()
