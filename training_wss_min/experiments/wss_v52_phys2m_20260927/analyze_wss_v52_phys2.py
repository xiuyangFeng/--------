"""§36 stage 2 readout (read-only, tolerant of unfinished arms).

CV5 confirmation of T2 (loss_asym_under_weight 2.0): every finished X5Dcap_asym2_v52cv_f{k}_s{seed} arm is paired with
wss_v52_20260923/X5Dcap_v52cv_f{k}_s{seed} on the same held-out fold (R2_cb Pa, jet subset, top-10% tail metrics);
gate: mean paired delta > +0.005 and >= 11/15 positive. Pooled OOF per seed once all 5 folds exist.
Dose controls (wss_v52_phys2n_20260927, IND asym 1.5 / 3.0) are paired with X5Dcap_v52ind same seed when present.

    python training_wss_min/experiments/wss_v52_phys2m_20260927/analyze_wss_v52_phys2.py [--checkpoint best]
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path("/public/newhome/cy/Digital_twin/GNN")
sys.path.insert(0, str(ROOT / "training_wss_min/experiments/wss_v52_20260923")); sys.path.insert(0, str(ROOT / "training_wss_min/experiments/wss_v52_phys_20260926"))
from analyze_wss_v52 import load_preds, r2cb  # noqa: E402
from analyze_wss_v52_phys import tail, hotspot_iou  # noqa: E402

RUNS = ROOT / "training_wss_min/runs"; BASE = RUNS / "wss_v52_20260923"; EXP = ROOT / "training_wss_min/experiments/wss_v52_phys2m_20260927"
SEEDS = (1234, 7, 2025); FOLDS = range(5)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--checkpoint", default="best"); a = ap.parse_args()
    lines = [f"# §36 stage 2 readout (checkpoint {a.checkpoint})", "", "## T2 CV5 confirmation: X5Dcap_asym2 vs X5Dcap, paired per (fold, seed)", "",
             "| fold | seed | R²_cb arm | R²_cb base | Δ | jet Δ | top10 under-frac base→arm | top10 pred/true base→arm | IoU base→arm |", "|---|---|---:|---:|---:|---:|---|---|---|"]
    out = {"checkpoint": a.checkpoint, "cv5": {}, "dose": {}}; deltas = []; jet_d = []; pooled_arm = {s: {} for s in SEEDS}; pooled_base = {s: {} for s in SEEDS}
    for s in SEEDS:
        for k in FOLDS:
            arm = load_preds(RUNS / f"wss_v52_phys2m_20260927/X5Dcap_asym2_v52cv_f{k}_s{s}", a.checkpoint); base = load_preds(BASE / f"X5Dcap_v52cv_f{k}_s{s}", a.checkpoint)
            if arm is None or base is None:
                lines.append(f"| {k} | {s} | (pending) | | | | | | |"); continue
            units = sorted(set(arm) & set(base)); arm = {u: arm[u] for u in units}; base = {u: base[u] for u in units}
            pooled_arm[s].update(arm); pooled_base[s].update(base)
            ra, rb = r2cb(arm), r2cb(base); d = ra - rb; deltas.append(d)
            ja = {u: v for u, v in arm.items() if np.percentile(v[0], 99) > 40}; jb = {u: base[u] for u in ja}
            jd = (r2cb(ja) - r2cb(jb)) if len(ja) >= 3 else float("nan"); jet_d.append(jd)
            ta, tb = tail(arm), tail(base); ia, ib = hotspot_iou(RUNS / f"wss_v52_phys2m_20260927/X5Dcap_asym2_v52cv_f{k}_s{s}", a.checkpoint), hotspot_iou(BASE / f"X5Dcap_v52cv_f{k}_s{s}", a.checkpoint)
            lines.append(f"| {k} | {s} | {ra:.4f} | {rb:.4f} | {d:+.4f} | {jd:+.4f} | {tb[1]:.3f}→{ta[1]:.3f} | {tb[2]:.3f}→{ta[2]:.3f} | {ib:.3f}→{ia:.3f} |")
            out["cv5"][f"f{k}_s{s}"] = dict(r2cb_arm=ra, r2cb_base=rb, delta=d, jet_delta=jd, n=len(units), tail_arm=list(ta), tail_base=list(tb))
    n = len(deltas)
    if n:
        pos = sum(x > 0 for x in deltas); g = n == 15 and np.mean(deltas) > 0.005 and pos >= 11
        lines += ["", f"paired deltas: n={n}/15, mean {np.mean(deltas):+.4f}, sd {np.std(deltas, ddof=1) if n > 1 else 0:.4f}, positive {pos}/{n}; jet Δ mean {np.nanmean(jet_d):+.4f} | gate (mean > +0.005 and ≥ 11/15 positive): {'过' if g else ('不过' if n == 15 else '未齐')}"]
        out["cv5_summary"] = dict(n=n, mean=float(np.mean(deltas)), positive=int(pos), gate=bool(g) if n == 15 else None)
        for s in SEEDS:
            if len([k for k in FOLDS if f"f{k}_s{s}" in out["cv5"]]) == 5:
                lines.append(f"pooled OOF seed {s}: arm {r2cb(pooled_arm[s]):.4f} vs base {r2cb(pooled_base[s]):.4f} ({r2cb(pooled_arm[s]) - r2cb(pooled_base[s]):+.4f})")
                out["cv5"][f"pooled_s{s}"] = dict(arm=r2cb(pooled_arm[s]), base=r2cb(pooled_base[s]))
    lines += ["", "## dose controls (IND, paired with X5Dcap_v52ind same seed)", "", "| arm | seed | R²_cb | Δ | top10 under-frac base→arm |", "|---|---|---:|---:|---|"]
    for armname in ("X5Dcap_asym15", "X5Dcap_asym3"):
        for s in SEEDS:
            arm = next((r for r in (load_preds(RUNS / f"{exp}/{armname}_s{s}", a.checkpoint) for exp in ("wss_v52_phys2na_20260927", "wss_v52_phys2nb_20260927", "wss_v52_phys2n_20260927")) if r is not None), None)  # dose halves ran as phys2na (x3.0) / phys2nb (x1.5) on node04
            base = load_preds(BASE / f"X5Dcap_v52ind_s{s}", a.checkpoint)
            if arm is None or base is None:
                lines.append(f"| {armname} | {s} | (pending) | | |"); continue
            ra, rb = r2cb(arm), r2cb(base); ta, tb = tail(arm), tail(base)
            lines.append(f"| {armname} | {s} | {ra:.4f} | {ra - rb:+.4f} | {tb[1]:.3f}→{ta[1]:.3f} |"); out["dose"][f"{armname}_s{s}"] = dict(r2cb=ra, delta=ra - rb)
    text = "\n".join(lines) + "\n"; (EXP / f"readout_phys2_ckpt_{a.checkpoint}.md").write_text(text); (EXP / f"readout_phys2_ckpt_{a.checkpoint}.json").write_text(json.dumps(out, indent=1)); print(text)


if __name__ == "__main__":
    main()
