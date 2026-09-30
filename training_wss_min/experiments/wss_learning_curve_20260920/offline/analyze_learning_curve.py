"""Learning-curve readout (read-only): held-out-fold predictions → pooled 136-case R2_cb per (fraction, seed),
per-doubling slopes, jet subset (true p99 > 40 Pa), cohort split.  Handles partial matrices (missing arms are reported).

    python training_wss_min/experiments/wss_learning_curve_20260920/offline/analyze_learning_curve.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/public/newhome/cy/Digital_twin/GNN")
NAME = "wss_learning_curve_20260920"
CONFIGS = ROOT / "training_wss_min/configs" / NAME
EXP = ROOT / "training_wss_min/experiments" / NAME
RUNS = ROOT / "training_wss_min/runs"
OUT = EXP / "offline"
JET_P99 = 40.0
FRACTIONS = (25, 50, 75, 100)
SEEDS = (1234, 7, 2025)
FOLDS = (0, 1, 2)
FAMILIES = ("AG", "AAA", "ILO")


def run_dir(frac: int, fold: int, seed: int) -> Path:
    if frac == 100 and seed == 1234:
        return RUNS / "wss_v51_wave2a_20260916" / f"X5D_v51_f{fold}_s1234"
    return RUNS / NAME / f"LC{frac}_f{fold}_s{seed}"


def load_cases(run: Path) -> dict[str, dict]:
    man = run / "eval/ckpt_best/predictions/test/manifest.json"
    if not man.is_file():
        return {}
    out = {}
    for c in json.loads(man.read_text())["cases"]:
        z = np.load(run / "eval/ckpt_best/predictions/test" / c["file"])
        y = z["true_pa"].astype(np.float64); p = z["pred_pa"].astype(np.float64)
        m = y >= np.percentile(y, 90)
        out[c["unit_id"]] = dict(mse=float(np.mean((y - p) ** 2)), mean=float(y.mean()), var=float(y.var()),
                                 p99=float(np.percentile(y, 99)), n=int(len(y)),
                                 top10_ratio=float(p[m].mean() / y[m].mean()),
                                 high_sse=float(np.sum((y[m] - p[m]) ** 2)), high_sst_part=(y[m], p[m]))
    return out


def r2cb(cases: dict[str, dict], keys=None) -> float | None:
    keys = list(cases) if keys is None else [k for k in keys if k in cases]
    if len(keys) < 2:
        return None
    gm = np.mean([cases[k]["mean"] for k in keys])
    res = np.mean([cases[k]["mse"] for k in keys])
    tot = np.mean([cases[k]["var"] + (cases[k]["mean"] - gm) ** 2 for k in keys])
    return float(1.0 - res / tot)


def high_r2(cases: dict[str, dict], keys=None) -> float | None:
    keys = list(cases) if keys is None else [k for k in keys if k in cases]
    if not keys:
        return None
    y = np.concatenate([cases[k]["high_sst_part"][0] for k in keys]); p = np.concatenate([cases[k]["high_sst_part"][1] for k in keys])
    return float(1.0 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def main() -> None:
    matrix = json.loads((CONFIGS / "matrix.json").read_text())
    subsets = matrix["learning_curve"]["subsets"]
    n_train = {}
    for tag, s in subsets.items():
        n_train[(s["fraction_percent"], s["fold"], s["seed"])] = s["n_train"]
    for fold in FOLDS:  # LC100 s1234 reference
        n_train[(100, fold, 1234)] = n_train[(100, fold, 7)]
    pooled, missing = {}, []
    for frac in FRACTIONS:
        for seed in SEEDS:
            cases = {}
            for fold in FOLDS:
                c = load_cases(run_dir(frac, fold, seed))
                if not c:
                    missing.append(f"LC{frac}_f{fold}_s{seed}")
                cases.update(c)
            if len(cases) != 136:
                pooled[(frac, seed)] = dict(n_cases=len(cases), complete=False)
                continue
            jets = [k for k, v in cases.items() if v["p99"] > JET_P99]
            non = [k for k in cases if k not in jets]
            pooled[(frac, seed)] = dict(
                n_cases=136, complete=True,
                n_train_mean=float(np.mean([n_train[(frac, f, seed)] for f in FOLDS])),
                r2cb=r2cb(cases), r2cb_jet=r2cb(cases, jets), r2cb_nonjet=r2cb(cases, non), n_jet=len(jets),
                high_r2=high_r2(cases), top10_ratio_jet=float(np.mean([cases[k]["top10_ratio"] for k in jets])),
                case_r2_mean=float(np.mean([1 - v["mse"] / v["var"] for v in cases.values()])),
                case_r2_p10=float(np.percentile([1 - v["mse"] / v["var"] for v in cases.values()], 10)),
                cohorts={f: r2cb(cases, [k for k in cases if k.startswith(f + "/")]) for f in FAMILIES},
                per_fold={f: r2cb(load_cases(run_dir(frac, f, seed))) for f in FOLDS})
    lines = [f"# Learning curve readout ({NAME}); jet = true p99 > {JET_P99:.0f} Pa; held-out folds pooled to 136 cases", ""]
    if missing:
        lines.append(f"missing / incomplete arms ({len(missing)}): {', '.join(missing)}"); lines.append("")
    lines.append(f"{'frac':>5} {'seed':>5} {'n_train':>8} {'R2cb':>7} {'jet':>7} {'nonjet':>7} {'highR2':>7} {'top10j':>7} {'caseμ':>6} {'P10':>6}  AG/AAA/ILO   f0/f1/f2")
    for (frac, seed), r in sorted(pooled.items()):
        if not r["complete"]:
            lines.append(f"{frac:>5} {seed:>5}   (incomplete: {r['n_cases']}/136 cases)"); continue
        co = r["cohorts"]; pf = r["per_fold"]
        lines.append(f"{frac:>5} {seed:>5} {r['n_train_mean']:>8.1f} {r['r2cb']:>7.4f} {r['r2cb_jet']:>7.4f} {r['r2cb_nonjet']:>7.4f} {r['high_r2']:>7.3f} {r['top10_ratio_jet']:>7.3f} {r['case_r2_mean']:>6.3f} {r['case_r2_p10']:>6.3f}  "
                     f"{co['AG']:.3f}/{co['AAA']:.3f}/{co['ILO']:.3f}   {pf[0]:.3f}/{pf[1]:.3f}/{pf[2]:.3f}")
    lines.append("")
    lines.append("## per fraction: mean ± sd over available seeds (pooled 136-case R2cb; jet; non-jet)")
    summary = {}
    for frac in FRACTIONS:
        rs = [pooled[(frac, s)] for s in SEEDS if pooled.get((frac, s), {}).get("complete")]
        if not rs:
            continue
        summary[frac] = dict(n_seeds=len(rs), n_train=float(np.mean([r["n_train_mean"] for r in rs])),
                             r2cb=float(np.mean([r["r2cb"] for r in rs])), r2cb_sd=float(np.std([r["r2cb"] for r in rs], ddof=1)) if len(rs) > 1 else None,
                             jet=float(np.mean([r["r2cb_jet"] for r in rs])), nonjet=float(np.mean([r["r2cb_nonjet"] for r in rs])),
                             ilo=float(np.mean([r["cohorts"]["ILO"] for r in rs])), high_r2=float(np.mean([r["high_r2"] for r in rs])))
        s = summary[frac]
        sd = f"± {s['r2cb_sd']:.4f}" if s["r2cb_sd"] is not None else "(1 seed)"
        lines.append(f"  {frac:>3}%  n_train {s['n_train']:5.1f}  R2cb {s['r2cb']:.4f} {sd}  jet {s['jet']:.4f}  non-jet {s['nonjet']:.4f}  ILO {s['ilo']:.4f}  high-WSS {s['high_r2']:.3f}  [{s['n_seeds']} seed]")
    lines.append("")
    lines.append("## slope per doubling of n_train between consecutive fractions (same-seed paired where both exist)")
    slopes = {}
    for lo, hi in zip(FRACTIONS[:-1], FRACTIONS[1:]):
        pairs = [(pooled[(lo, s)], pooled[(hi, s)]) for s in SEEDS
                 if pooled.get((lo, s), {}).get("complete") and pooled.get((hi, s), {}).get("complete")]
        if not pairs:
            continue
        def slope(key):
            vals = [(b[key] - a[key]) / math.log2(b["n_train_mean"] / a["n_train_mean"]) for a, b in pairs]
            return float(np.mean(vals)), vals
        s_all, v_all = slope("r2cb"); s_jet, v_jet = slope("r2cb_jet"); s_non, v_non = slope("r2cb_nonjet")
        slopes[f"{lo}->{hi}"] = dict(n_pairs=len(pairs), all=s_all, all_per_seed=v_all, jet=s_jet, jet_per_seed=v_jet, nonjet=s_non)
        lines.append(f"  {lo:>3}% → {hi:>3}%  ΔR2cb per doubling: all {s_all:+.4f} ({', '.join(f'{v:+.3f}' for v in v_all)})   jet {s_jet:+.4f}   non-jet {s_non:+.4f}   [{len(pairs)} paired seed]")
    lines.append("")
    lines.append("Pre-registered reading: 75→100 slope ≥ +0.02 per doubling → data volume still the bottleneck (B-tier CFD reruns worth it);")
    lines.append("jet slope ≫ overall → prioritise jet cases; flat → switch to model/objective side.  Single readings are not ranked.")
    text = "\n".join(lines) + "\n"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "analyze_learning_curve.txt").write_text(text)
    (OUT / "analyze_learning_curve.json").write_text(json.dumps(
        dict(jet_p99_threshold=JET_P99, missing=missing,
             pooled={f"LC{f}_s{s}": {k: v for k, v in r.items() if k != "per_fold"} | {"per_fold": {str(k): v for k, v in r.get("per_fold", {}).items()}} for (f, s), r in pooled.items()},
             per_fraction=summary, slopes=slopes), ensure_ascii=False, indent=2) + "\n")
    print(text)


if __name__ == "__main__":
    main()
