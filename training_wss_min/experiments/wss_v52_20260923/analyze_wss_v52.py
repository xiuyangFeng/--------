"""v5.2 baseline retraining readout (read-only; tolerant of unfinished arms).

IND : per-arm R2_cb (physical Pa, case-balanced) on the 91 recovered units + five-seed Pa-mean ensemble per recipe,
      per cohort, per unit; reference = frozen X5D_v51 five-seed ensemble on the same 91 (0.749, eval before the
      end-radius fix; the 18 rebuilt units re-evaluated separately).
CV5 : per (recipe, seed) pooled out-of-fold R2_cb over the five folds (every case once), mean +- sd over seeds;
      X5Dcap - X5D paired deltas per (fold, seed) and per IND seed; jet subset (true p99 > 40 Pa) read from per-case rows.

    python training_wss_min/experiments/wss_v52_20260923/analyze_wss_v52.py [--checkpoint best]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path("/public/newhome/cy/Digital_twin/GNN")
RUNS = ROOT / "training_wss_min/runs/wss_v52_20260923"
EXP = ROOT / "training_wss_min/experiments/wss_v52_20260923"
RECIPES = ("X5D", "X5Dcap")
SEEDS_IND = (1234, 7, 2025, 11, 2026)
SEEDS_CV = (1234, 7, 2025)
FOLDS = range(5)
FROZEN_REF = {"r2cb_91": 0.749, "note": "frozen X5D_v51 five-seed ensemble on the 91 recovered units (wave-2/3/4/5 evals, before the end-radius rebuild)"}


def load_preds(run: Path, ckpt: str) -> dict[str, tuple[np.ndarray, np.ndarray]] | None:
    man = run / "eval" / f"ckpt_{ckpt}" / "predictions/test/manifest.json"
    if not man.is_file():
        return None
    payload = json.loads(man.read_text())
    out = {}
    for c in payload["cases"]:
        z = np.load(man.parent / c["file"])
        out[c["unit_id"]] = (z["true_pa"].astype(np.float64), z["pred_pa"].astype(np.float64))
    return out


def r2cb(units: dict[str, tuple[np.ndarray, np.ndarray]]) -> float:
    gm = np.mean([y.mean() for y, _ in units.values()])
    res = np.mean([np.mean((y - p) ** 2) for y, p in units.values()])
    tot = np.mean([np.mean((y - gm) ** 2) for y, _ in units.values()])
    return float(1.0 - res / tot)


def r2(y, p):
    return float(1.0 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--checkpoint", default="best"); a = ap.parse_args()
    lines = [f"# v5.2 retraining readout (checkpoint {a.checkpoint}) — {RUNS.name}", ""]
    out: dict = {"checkpoint": a.checkpoint, "frozen_reference": FROZEN_REF, "IND": {}, "CV5": {}}
    # ---- IND ---------------------------------------------------------------------------------------------------
    lines += ["## IND: train 170 library / test 91 recovered", "", "| recipe | seed | R²_cb (91) | jet subset R²_cb (p99>40) | n |", "|---|---|---:|---:|---:|"]
    ind_preds: dict[str, dict[int, dict]] = {}
    for recipe in RECIPES:
        ind_preds[recipe] = {}
        for seed in SEEDS_IND:
            preds = load_preds(RUNS / f"{recipe}_v52ind_s{seed}", a.checkpoint)
            if preds is None:
                lines.append(f"| {recipe} | {seed} | (pending) | | |"); continue
            ind_preds[recipe][seed] = preds
            jet = {u: v for u, v in preds.items() if np.percentile(v[0], 99) > 40}
            lines.append(f"| {recipe} | {seed} | {r2cb(preds):.4f} | {r2cb(jet) if len(jet) >= 3 else float('nan'):.4f} | {len(preds)} |")
            out["IND"].setdefault(recipe, {})[str(seed)] = {"r2cb": r2cb(preds), "n": len(preds)}
    lines += ["", "| recipe | seeds in ensemble | ensemble R²_cb (91) | AG | AAA | ILO | units below cohort p10 (cv3 band) |", "|---|---|---:|---:|---:|---:|---|"]
    cv3 = json.loads((ROOT / "training_wss_min/experiments/wss_v51_wave1_20260916/analysis_20260917/current_residuals.json").read_text())["cv3_oof"]["per_case"]
    p10 = {fam: float(np.percentile([r["r2_pa"] for r in cv3 if r["unit"].startswith(fam + "/")], 10)) for fam in ("AG", "AAA", "ILO")}
    ens_units: dict[str, dict[str, float]] = {}
    for recipe in RECIPES:
        seeds = sorted(ind_preds[recipe])
        if not seeds:
            continue
        units = sorted(set.intersection(*(set(ind_preds[recipe][s]) for s in seeds)))
        ens = {u: (ind_preds[recipe][seeds[0]][u][0], np.mean([ind_preds[recipe][s][u][1] for s in seeds], axis=0)) for u in units}
        per_unit = {u: r2(*ens[u]) for u in units}; ens_units[recipe] = per_unit
        fam = {f: r2cb({u: v for u, v in ens.items() if u.startswith(f + "/")}) for f in ("AG", "AAA", "ILO") if any(u.startswith(f + "/") for u in ens)}
        low = [f"{u} {per_unit[u]:.2f}" for u in units if per_unit[u] < p10[u.split('/')[0]]]
        lines.append(f"| {recipe} | {seeds} | **{r2cb(ens):.4f}** | {fam.get('AG', float('nan')):.3f} | {fam.get('AAA', float('nan')):.3f} | {fam.get('ILO', float('nan')):.3f} | {len(low)}: {', '.join(low[:8])} |")
        out["IND"].setdefault(recipe, {})["ensemble"] = {"seeds": seeds, "r2cb": r2cb(ens), "by_cohort": fam, "per_unit": per_unit, "below_p10": low}
    if all(r in ens_units for r in RECIPES):
        d = [ens_units["X5Dcap"][u] - ens_units["X5D"][u] for u in ens_units["X5D"] if u in ens_units["X5Dcap"]]
        lines.append(f"\nIND per-unit ensemble R² delta X5Dcap − X5D: mean {np.mean(d):+.4f}, median {np.median(d):+.4f}, units improved {sum(x > 0 for x in d)}/{len(d)}")
    lines.append(f"\nreference: {FROZEN_REF['r2cb_91']} — {FROZEN_REF['note']}")
    # ---- CV5 ---------------------------------------------------------------------------------------------------
    lines += ["", "## CV5: 261 cases, patient-grouped 5-fold (pooled out-of-fold, every case once)", "", "| recipe | seed | folds done | pooled OOF R²_cb | jet subset | per-fold R²_cb |", "|---|---|---:|---:|---:|---|"]
    pooled: dict[str, dict[int, float]] = {}
    fold_r2: dict[str, dict[tuple[int, int], float]] = {}
    for recipe in RECIPES:
        pooled[recipe] = {}; fold_r2[recipe] = {}
        for seed in SEEDS_CV:
            allp: dict = {}; per_fold = []
            for k in FOLDS:
                preds = load_preds(RUNS / f"{recipe}_v52cv_f{k}_s{seed}", a.checkpoint)
                if preds is None:
                    per_fold.append("·"); continue
                allp.update(preds); v = r2cb(preds); fold_r2[recipe][(k, seed)] = v; per_fold.append(f"{v:.3f}")
            done = sum(1 for x in per_fold if x != "·")
            if done == len(FOLDS):
                pooled[recipe][seed] = r2cb(allp)
                jet = {u: v for u, v in allp.items() if np.percentile(v[0], 99) > 40}
                lines.append(f"| {recipe} | {seed} | {done}/5 | **{pooled[recipe][seed]:.4f}** | {r2cb(jet):.4f} | {' / '.join(per_fold)} |")
            else:
                lines.append(f"| {recipe} | {seed} | {done}/5 | (pending) | | {' / '.join(per_fold)} |")
        if pooled[recipe]:
            vals = list(pooled[recipe].values())
            lines.append(f"| {recipe} | mean±sd | | **{np.mean(vals):.4f} ± {np.std(vals, ddof=1) if len(vals) > 1 else 0:.4f}** | | n_seeds={len(vals)} |")
            out["CV5"][recipe] = {"pooled_by_seed": {str(k): v for k, v in pooled[recipe].items()}, "mean": float(np.mean(vals)), "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
                                  "per_fold": {f"f{k}_s{s}": v for (k, s), v in fold_r2[recipe].items()}}
    common = [key for key in fold_r2["X5D"] if key in fold_r2["X5Dcap"]]
    if common:
        d = [fold_r2["X5Dcap"][key] - fold_r2["X5D"][key] for key in common]
        lines.append(f"\nCV5 paired fold deltas X5Dcap − X5D ({len(common)} pairs): mean {np.mean(d):+.4f}, sd {np.std(d, ddof=1) if len(d) > 1 else 0:.4f}, positive {sum(x > 0 for x in d)}/{len(d)}")
        out["CV5"]["paired_delta"] = {"n": len(common), "mean": float(np.mean(d)), "values": {f"f{k}_s{s}": fold_r2["X5Dcap"][(k, s)] - fold_r2["X5D"][(k, s)] for (k, s) in common}}
    lines += ["", "reading rule: IND compares the five-seed ensemble with the frozen 0.749; CV5 compares recipes as paired (fold, seed) deltas and reports mean ± sd over seeds; no single reading is ranked."]
    text = "\n".join(lines) + "\n"
    (EXP / f"readout_ckpt_{a.checkpoint}.md").write_text(text); (EXP / f"readout_ckpt_{a.checkpoint}.json").write_text(json.dumps(out, indent=1))
    print(text)


if __name__ == "__main__":
    main()
