"""T2 (X5Dcap + loss_asym_under_weight 2.0) five-seed IND ensemble vs X5Dcap five-seed ensemble on the 91 recovered units.
Seeds 1234/7/2025 from runs/wss_v52_phys_20260926/X5Dcap_asym2_s*, 11/2026 from runs/wss_v52_phys2e_20260928/X5Dcap_asym2_s*.
Deployment-grade readout (Pa-mean ensemble, R2_cb, cohorts, jet subset, top-10% tail, per-unit deltas); not a gate."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path("/public/newhome/cy/Digital_twin/GNN")
sys.path.insert(0, str(ROOT / "training_wss_min/experiments/wss_v52_20260923")); sys.path.insert(0, str(ROOT / "training_wss_min/experiments/wss_v52_phys_20260926"))
from analyze_wss_v52 import load_preds, r2cb, r2  # noqa: E402
from analyze_wss_v52_phys import tail  # noqa: E402
RUNS = ROOT / "training_wss_min/runs"; EXP = ROOT / "training_wss_min/experiments/wss_v52_phys2e_20260928"
T2 = {1234: RUNS / "wss_v52_phys_20260926/X5Dcap_asym2_s1234", 7: RUNS / "wss_v52_phys_20260926/X5Dcap_asym2_s7", 2025: RUNS / "wss_v52_phys_20260926/X5Dcap_asym2_s2025",
      11: RUNS / "wss_v52_phys2e_20260928/X5Dcap_asym2_s11", 2026: RUNS / "wss_v52_phys2e_20260928/X5Dcap_asym2_s2026"}
BASE = {s: RUNS / f"wss_v52_20260923/X5Dcap_v52ind_s{s}" for s in (1234, 7, 2025, 11, 2026)}


def ensemble(runs):
    preds = {s: load_preds(p, "best") for s, p in runs.items()}
    assert all(v is not None for v in preds.values()), [s for s, v in preds.items() if v is None]
    units = sorted(set.intersection(*(set(v) for v in preds.values())))
    return {u: (preds[1234][u][0], np.mean([preds[s][u][1] for s in preds], axis=0)) for u in units}, {s: r2cb(v) for s, v in preds.items()}


def main():
    ens_t2, single_t2 = ensemble(T2); ens_b, single_b = ensemble(BASE)
    units = sorted(set(ens_t2) & set(ens_b)); ens_t2 = {u: ens_t2[u] for u in units}; ens_b = {u: ens_b[u] for u in units}
    lines = ["# T2 five-seed IND ensemble (deployment-grade) vs X5Dcap five-seed ensemble, 91 recovered units", "",
             "| | X5Dcap 5-seed | T2 5-seed | Δ |", "|---|---:|---:|---:|"]
    def row(name, a, b): lines.append(f"| {name} | {a:.4f} | {b:.4f} | {b - a:+.4f} |")
    row("R²_cb (Pa, 91)", r2cb(ens_b), r2cb(ens_t2))
    for fam in ("AG", "AAA", "ILO"):
        sb = {u: v for u, v in ens_b.items() if u.startswith(fam + "/")}; st = {u: ens_t2[u] for u in sb}
        row(f"R²_cb {fam} (n={len(sb)})", r2cb(sb), r2cb(st))
    jb = {u: v for u, v in ens_b.items() if np.percentile(v[0], 99) > 40}; jt = {u: ens_t2[u] for u in jb}
    row(f"jet subset R²_cb (p99>40 Pa, n={len(jb)})", r2cb(jb), r2cb(jt))
    tb, tt = tail(ens_b), tail(ens_t2)
    row("top10 SSE share", tb[0], tt[0]); row("top10 under-estimation fraction", tb[1], tt[1]); row("top10 pred/true mean ratio", tb[2], tt[2])
    pu = {u: r2(*ens_t2[u]) - r2(*ens_b[u]) for u in units}
    lines += ["", f"per-unit ensemble R² delta: mean {np.mean(list(pu.values())):+.4f}, median {np.median(list(pu.values())):+.4f}, improved {sum(v > 0 for v in pu.values())}/{len(pu)}; worst 3: " + ", ".join(f"{u} {pu[u]:+.3f}" for u in sorted(pu, key=pu.get)[:3]) + "; best 3: " + ", ".join(f"{u} {pu[u]:+.3f}" for u in sorted(pu, key=pu.get, reverse=True)[:3]),
              "", "single-seed R²_cb: " + ", ".join(f"s{s} {single_b[s]:.4f}→{single_t2[s]:.4f} ({single_t2[s] - single_b[s]:+.4f})" for s in single_b),
              "", "reference: v5.2 readout X5Dcap five-seed ensemble 0.764; frozen X5D_v51 five-seed on the same 91 units 0.749 (before end-radius rebuild)."]
    text = "\n".join(lines) + "\n"; (EXP / "readout_t2_ensemble.md").write_text(text)
    (EXP / "readout_t2_ensemble.json").write_text(json.dumps(dict(r2cb_base=r2cb(ens_b), r2cb_t2=r2cb(ens_t2), single_base=single_b, single_t2=single_t2, per_unit_delta=pu), indent=1))
    print(text)


if __name__ == "__main__":
    main()
