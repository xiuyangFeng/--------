"""§36 direction 3 readout (read-only, tolerant of unfinished arms): X5Dcap + synthetic children vs X5Dcap.

CV5 (wss_v52_synm_20260927): each X5Dcap_syn_v52cv_f{k}_s{seed} paired with wss_v52_20260923/X5Dcap_v52cv_f{k}_s{seed}
on the same held-out fold (real cases only). Gate G3.1: 15 paired deltas mean > +0.010 and >= 12/15 positive.
IND (wss_v52_synn_20260927): X5Dcap_syn_v52ind_s{seed} vs X5Dcap_v52ind_s{seed} on the 91 recovered units. Gate G3.2: mean > +0.010.

    python training_wss_min/experiments/wss_v52_synm_20260927/analyze_wss_v52_syn.py [--checkpoint best]
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path("/public/newhome/cy/Digital_twin/GNN")
sys.path.insert(0, str(ROOT / "training_wss_min/experiments/wss_v52_20260923")); sys.path.insert(0, str(ROOT / "training_wss_min/experiments/wss_v52_phys_20260926"))
from analyze_wss_v52 import load_preds, r2cb, r2  # noqa: E402
from analyze_wss_v52_phys import tail, hotspot_iou  # noqa: E402

RUNS = ROOT / "training_wss_min/runs"; BASE = RUNS / "wss_v52_20260923"; EXP = ROOT / "training_wss_min/experiments/wss_v52_synm_20260927"
SEEDS = (1234, 7, 2025); FOLDS = range(5)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--checkpoint", default="best"); a = ap.parse_args()
    lines = [f"# §36 direction 3 readout (checkpoint {a.checkpoint}): X5Dcap + synthetic vs X5Dcap", "", "## CV5 (synm): paired per (fold, seed), held-out real cases only", "",
             "| fold | seed | R²_cb syn | R²_cb base | Δ | jet Δ | top10 under-frac base→syn | IoU base→syn | n held-out |", "|---|---|---:|---:|---:|---:|---|---|---:|"]
    out = {"checkpoint": a.checkpoint, "cv5": {}, "ind": {}}; deltas = []; jet_d = []; pooled_arm = {s: {} for s in SEEDS}; pooled_base = {s: {} for s in SEEDS}
    for s in SEEDS:
        for k in FOLDS:
            arm = load_preds(RUNS / f"wss_v52_synm_20260927/X5Dcap_syn_v52cv_f{k}_s{s}", a.checkpoint); base = load_preds(BASE / f"X5Dcap_v52cv_f{k}_s{s}", a.checkpoint)
            if arm is None or base is None:
                lines.append(f"| {k} | {s} | (pending) | | | | | | |"); continue
            assert not any("~m" in u for u in arm), "synthetic case leaked into a held-out partition"
            units = sorted(set(arm) & set(base)); arm = {u: arm[u] for u in units}; base = {u: base[u] for u in units}
            pooled_arm[s].update(arm); pooled_base[s].update(base)
            ra, rb = r2cb(arm), r2cb(base); d = ra - rb; deltas.append(d)
            ja = {u: v for u, v in arm.items() if np.percentile(v[0], 99) > 40}; jb = {u: base[u] for u in ja}
            jd = (r2cb(ja) - r2cb(jb)) if len(ja) >= 3 else float("nan"); jet_d.append(jd)
            ta, tb = tail(arm), tail(base); ia, ib = hotspot_iou(RUNS / f"wss_v52_synm_20260927/X5Dcap_syn_v52cv_f{k}_s{s}", a.checkpoint), hotspot_iou(BASE / f"X5Dcap_v52cv_f{k}_s{s}", a.checkpoint)
            lines.append(f"| {k} | {s} | {ra:.4f} | {rb:.4f} | {d:+.4f} | {jd:+.4f} | {tb[1]:.3f}→{ta[1]:.3f} | {ib:.3f}→{ia:.3f} | {len(units)} |")
            out["cv5"][f"f{k}_s{s}"] = dict(r2cb_syn=ra, r2cb_base=rb, delta=d, jet_delta=jd, n=len(units))
    n = len(deltas)
    if n:
        pos = sum(x > 0 for x in deltas); g = n == 15 and np.mean(deltas) > 0.010 and pos >= 12
        lines += ["", f"paired deltas: n={n}/15, mean {np.mean(deltas):+.4f}, sd {np.std(deltas, ddof=1) if n > 1 else 0:.4f}, positive {pos}/{n}; jet Δ mean {np.nanmean(jet_d):+.4f} | G3.1 (mean > +0.010 and ≥ 12/15 positive): {'过' if g else ('不过' if n == 15 else '未齐')}"]
        out["cv5_summary"] = dict(n=n, mean=float(np.mean(deltas)), positive=int(pos), gate=bool(g) if n == 15 else None)
        for s in SEEDS:
            if len([k for k in FOLDS if f"f{k}_s{s}" in out["cv5"]]) == 5:
                lines.append(f"pooled OOF seed {s}: syn {r2cb(pooled_arm[s]):.4f} vs base {r2cb(pooled_base[s]):.4f} ({r2cb(pooled_arm[s]) - r2cb(pooled_base[s]):+.4f})")
    lines += ["", "## IND (synn): 170 library + 42 children of library parents -> 91 recovered units (real)", "", "| seed | R²_cb syn | R²_cb base | Δ | jet Δ | top10 under-frac base→syn |", "|---|---:|---:|---:|---:|---|"]
    ind_d = []
    for s in SEEDS:
        arm = load_preds(RUNS / f"wss_v52_synn_20260927/X5Dcap_syn_v52ind_s{s}", a.checkpoint); base = load_preds(BASE / f"X5Dcap_v52ind_s{s}", a.checkpoint)
        if arm is None or base is None:
            lines.append(f"| {s} | (pending) | | | | |"); continue
        units = sorted(set(arm) & set(base)); arm = {u: arm[u] for u in units}; base = {u: base[u] for u in units}
        ra, rb = r2cb(arm), r2cb(base); ind_d.append(ra - rb); ja = {u: v for u, v in arm.items() if np.percentile(v[0], 99) > 40}; jb = {u: base[u] for u in ja}
        ta, tb = tail(arm), tail(base)
        lines.append(f"| {s} | {ra:.4f} | {rb:.4f} | {ra - rb:+.4f} | {r2cb(ja) - r2cb(jb):+.4f} | {tb[1]:.3f}→{ta[1]:.3f} |"); out["ind"][str(s)] = dict(r2cb_syn=ra, r2cb_base=rb, delta=ra - rb)
    if ind_d:
        lines.append(f"\nIND paired Δ: n={len(ind_d)}/3, mean {np.mean(ind_d):+.4f}, positive {sum(x > 0 for x in ind_d)}/{len(ind_d)} | G3.2 (mean > +0.010): {'过' if (len(ind_d) == 3 and np.mean(ind_d) > 0.010) else ('不过' if len(ind_d) == 3 else '未齐')}")
        out["ind_summary"] = dict(n=len(ind_d), mean=float(np.mean(ind_d)))
    text = "\n".join(lines) + "\n"; (EXP / f"readout_syn_ckpt_{a.checkpoint}.md").write_text(text); (EXP / f"readout_syn_ckpt_{a.checkpoint}.json").write_text(json.dumps(out, indent=1)); print(text)


if __name__ == "__main__":
    main()
