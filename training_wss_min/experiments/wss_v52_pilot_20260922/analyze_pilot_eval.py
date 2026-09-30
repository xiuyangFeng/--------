"""Pilot independent evaluation readout (read-only): 5 new units × frozen X5D_v51 five seeds → per-case R² per seed,
Pa-mean five-seed ensemble, and comparison with the cv3 out-of-fold expectation of the same cohort.

    python training_wss_min/experiments/wss_v52_pilot_20260922/analyze_pilot_eval.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path("/public/newhome/cy/Digital_twin/GNN")
EXP = ROOT / "training_wss_min/experiments/wss_v52_pilot_20260922"
SEEDS = (1234, 7, 2025, 11, 2026)
CV3 = ROOT / "training_wss_min/experiments/wss_v51_wave1_20260916/analysis_20260917/current_residuals.json"


def load(run_eval: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    man = json.loads((run_eval / "predictions/test/manifest.json").read_text())
    out = {}
    for c in man["cases"]:
        z = np.load(run_eval / "predictions/test" / c["file"])
        out[c["unit_id"]] = (z["true_pa"].astype(np.float64), z["pred_pa"].astype(np.float64))
    return out


def r2(y, p):
    return float(1.0 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def main() -> None:
    per_seed = {s: load(EXP / "pilot_eval" / f"X5D_v51_s{s}" ) for s in SEEDS}
    units = sorted(per_seed[SEEDS[0]])
    ens = {u: np.mean([per_seed[s][u][1] for s in SEEDS], axis=0) for u in units}
    cv = json.loads(CV3.read_text())["cv3_oof"]
    ref = {row["unit"]: row for row in cv["per_case"]}
    coh_r2 = {}
    for fam in ("AG", "AAA", "ILO"):
        vals = [row["r2_pa"] for row in cv["per_case"] if row["unit"].startswith(fam + "/")]
        coh_r2[fam] = (float(np.mean(vals)), float(np.percentile(vals, 10)), float(np.percentile(vals, 90)))
    lines = ["# pilot independent eval: frozen X5D_v51 (train136 stats), 5 seeds, Pa-mean ensemble", "",
             "| unit | n | true p99 Pa | seeds R² (min..max) | ensemble R² | top10 ratio | cohort cv3 folded-out R² mean (p10..p90) | verdict |",
             "|---|---:|---:|---|---:|---:|---|---|"]
    summary = {}
    for u in units:
        y = per_seed[SEEDS[0]][u][0]
        rs = [r2(y, per_seed[s][u][1]) for s in SEEDS]
        e = r2(y, ens[u]); m = y >= np.percentile(y, 90); top = float(ens[u][m].mean() / y[m].mean())
        fam = u.split("/")[0]; mu, p10, p90 = coh_r2[fam]
        verdict = "in band" if e >= p10 else ("below p10 → inspect" if e >= p10 - 0.15 else "far below → suspect data")
        lines.append(f"| {u} | {len(y)} | {np.percentile(y, 99):.1f} | {min(rs):.3f}..{max(rs):.3f} | **{e:.3f}** | {top:.2f} | {mu:.3f} ({p10:.3f}..{p90:.3f}) | {verdict} |")
        summary[u] = dict(n=int(len(y)), true_p99=float(np.percentile(y, 99)), seed_r2=rs, ensemble_r2=e, top10_ratio=top,
                          cohort_ref=dict(mean=mu, p10=p10, p90=p90), verdict=verdict)
    # pooled case-balanced over the 5
    gm = np.mean([per_seed[SEEDS[0]][u][0].mean() for u in units])
    res = np.mean([np.mean((per_seed[SEEDS[0]][u][0] - ens[u]) ** 2) for u in units])
    tot = np.mean([np.mean((per_seed[SEEDS[0]][u][0] - gm) ** 2) for u in units])
    lines += ["", f"5-unit case-balanced R²_cb (ensemble) = {1 - res / tot:.4f}; cv3 folded-out 136-case reference 0.7100 / test34 0.7749"]
    text = "\n".join(lines) + "\n"
    (EXP / "pilot_eval_readout.md").write_text(text)
    (EXP / "pilot_eval_readout.json").write_text(json.dumps(dict(units=summary, r2cb_ensemble=float(1 - res / tot)), indent=1))
    print(text)


if __name__ == "__main__":
    main()
