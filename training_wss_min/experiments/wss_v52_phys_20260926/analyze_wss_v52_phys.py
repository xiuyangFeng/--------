"""§36 screening readout (read-only, tolerant of unfinished arms).

Each arm (P1 womfeat / P2 womres / P3 poisres / T1 tw / T2 asym2 / T3 pin95) is compared with the X5Dcap v5.2 IND
baseline of the SAME seed (runs/wss_v52_20260923/X5Dcap_v52ind_s{seed}) on the 91 recovered units:
  R2_cb (Pa, case-balanced), jet subset R2_cb (true p99 > 40 Pa), top-10%-true-points: SSE share, under-estimation
  fraction, pred/true mean ratio; hotspot top-10 IoU (from metrics.json).
Gates (pre-registered, matrix doc §1/§2): G1.1 mean paired dR2_cb > +0.010 and 3/3 same sign; G1.2 jet dR2_cb >= 0;
G1.3 top-10% under-estimation fraction decreases; G2.1 (direction two) overall R2_cb >= X5Dcap - 0.005.

    python training_wss_min/experiments/wss_v52_phys_20260926/analyze_wss_v52_phys.py [--checkpoint best]
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path("/public/newhome/cy/Digital_twin/GNN")
sys.path.insert(0, str(ROOT / "training_wss_min/experiments/wss_v52_20260923"))
from analyze_wss_v52 import load_preds, r2cb, r2  # noqa: E402

RUNS = ROOT / "training_wss_min/runs/wss_v52_phys_20260926"
BASE = ROOT / "training_wss_min/runs/wss_v52_20260923"
EXP = ROOT / "training_wss_min/experiments/wss_v52_phys_20260926"
ARMS = {"X5Dcap_womfeat": "P1 Womersley 特征", "X5Dcap_womres": "P2 Womersley 残差目标", "X5Dcap_poisres": "P3 泊肃叶残差（消融）",
        "X5Dcap_tw": "T1 目标加权", "X5Dcap_asym2": "T2 低估×2", "X5Dcap_pin95": "T3 pinball 0.95"}
SEEDS = (1234, 7, 2025)


def tail(units):
    """top-10% true points pooled per case then averaged: SSE share, under-estimation fraction, pred/true mean ratio."""
    share, under, ratio = [], [], []
    for y, p in units.values():
        thr = np.percentile(y, 90); m = y >= thr
        sse = (y - p) ** 2
        share.append(sse[m].sum() / max(sse.sum(), 1e-12)); under.append(np.mean(p[m] < y[m])); ratio.append(p[m].mean() / y[m].mean())
    return float(np.mean(share)), float(np.mean(under)), float(np.mean(ratio))


def hotspot_iou(run, ckpt):
    f = run / "eval" / f"ckpt_{ckpt}" / "metrics.json"
    if not f.is_file():
        return float("nan")
    return float(json.loads(f.read_text())["test"]["hotspot"]["top10_iou_casemean"])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--checkpoint", default="best"); a = ap.parse_args()
    base = {s: load_preds(BASE / f"X5Dcap_v52ind_s{s}", a.checkpoint) for s in SEEDS}
    assert all(v is not None for v in base.values()), "baseline predictions missing"
    bstat = {s: dict(r2=r2cb(base[s]), jet=r2cb({u: v for u, v in base[s].items() if np.percentile(v[0], 99) > 40}), tail=tail(base[s]), iou=hotspot_iou(BASE / f"X5Dcap_v52ind_s{s}", a.checkpoint)) for s in SEEDS}
    lines = [f"# §36 screening readout (checkpoint {a.checkpoint}) — IND 170/91, paired with X5Dcap v5.2 same seed", "",
             "| arm | seed | R²_cb | ΔR²_cb | jet R²_cb | Δjet | top10 SSE share | top10 under-frac (base→arm) | top10 pred/true (base→arm) | hotspot IoU (base→arm) |", "|---|---|---:|---:|---:|---:|---:|---|---|---|"]
    out = {"checkpoint": a.checkpoint, "baseline": {str(s): {"r2cb": bstat[s]["r2"], "jet": bstat[s]["jet"], "tail": bstat[s]["tail"], "iou": bstat[s]["iou"]} for s in SEEDS}, "arms": {}}
    for arm, title in ARMS.items():
        rows = {}
        for s in SEEDS:
            preds = load_preds(RUNS / f"{arm}_s{s}", a.checkpoint)
            if preds is None:
                lines.append(f"| {arm} | {s} | (pending) | | | | | | | |"); continue
            jet = {u: v for u, v in preds.items() if np.percentile(v[0], 99) > 40}
            t = tail(preds); b = bstat[s]
            rows[s] = dict(r2=r2cb(preds), d=r2cb(preds) - b["r2"], jet=r2cb(jet), djet=r2cb(jet) - b["jet"], tail=t, iou=hotspot_iou(RUNS / f"{arm}_s{s}", a.checkpoint))
            lines.append(f"| {arm} | {s} | {rows[s]['r2']:.4f} | {rows[s]['d']:+.4f} | {rows[s]['jet']:.4f} | {rows[s]['djet']:+.4f} | {t[0]:.3f} (base {b['tail'][0]:.3f}) | {b['tail'][1]:.3f}→{t[1]:.3f} | {b['tail'][2]:.3f}→{t[2]:.3f} | {b['iou']:.3f}→{rows[s]['iou']:.3f} |")
        if not rows:
            continue
        d = [r["d"] for r in rows.values()]; dj = [r["djet"] for r in rows.values()]
        under_down = [rows[s]["tail"][1] < bstat[s]["tail"][1] for s in rows]
        g11 = len(rows) == 3 and np.mean(d) > 0.010 and (all(x > 0 for x in d) or all(x < 0 for x in d))
        g12 = len(rows) == 3 and np.mean(dj) >= 0
        g13 = len(rows) == 3 and all(under_down)
        g21 = len(rows) == 3 and all(rows[s]["r2"] >= bstat[s]["r2"] - 0.005 for s in rows)
        verdict = f"n={len(rows)}/3 ΔR²_cb mean {np.mean(d):+.4f} (同向 {sum(x > 0 for x in d)}/{len(d)} 正) | Δjet mean {np.mean(dj):+.4f} | 低估比例下降 {sum(under_down)}/{len(rows)} | G1.1 {'过' if g11 else '不过'} · G1.2 {'过' if g12 else '不过'} · G1.3 {'过' if g13 else '不过'} · G2.1 {'过' if g21 else '不过'}"
        lines.append(f"| **{arm}** ({title}) | 汇总 | | | | | | | | {verdict} |")
        # three-seed Pa-mean ensemble vs baseline three-seed ensemble on the same units
        if len(rows) == 3:
            units = sorted(set.intersection(*(set(load_preds(RUNS / f"{arm}_s{s}", a.checkpoint)) for s in SEEDS)))
            ens_a = {u: (base[SEEDS[0]][u][0], np.mean([load_preds(RUNS / f"{arm}_s{s}", a.checkpoint)[u][1] for s in SEEDS], axis=0)) for u in units}
            ens_b = {u: (base[SEEDS[0]][u][0], np.mean([base[s][u][1] for s in SEEDS], axis=0)) for u in units}
            pu = [r2(*ens_a[u]) - r2(*ens_b[u]) for u in units]
            lines.append(f"| | 三 seed 集成 | {r2cb(ens_a):.4f} | {r2cb(ens_a) - r2cb(ens_b):+.4f} (基线集成 {r2cb(ens_b):.4f}) | | | | 逐例 Δ 中位 {np.median(pu):+.4f}，改善 {sum(x > 0 for x in pu)}/{len(pu)} | | |")
            out["arms"][arm] = {"per_seed": {str(s): {k: (v if not isinstance(v, tuple) else list(v)) for k, v in r.items()} for s, r in rows.items()}, "ensemble": {"r2cb": r2cb(ens_a), "baseline_r2cb": r2cb(ens_b), "per_unit_delta_median": float(np.median(pu)), "improved": int(sum(x > 0 for x in pu)), "n": len(pu)}, "gates": {"G1.1": bool(g11), "G1.2": bool(g12), "G1.3": bool(g13), "G2.1": bool(g21)}}
        else:
            out["arms"][arm] = {"per_seed": {str(s): {k: (v if not isinstance(v, tuple) else list(v)) for k, v in r.items()} for s, r in rows.items()}, "gates": None}
    lines += ["", "baseline X5Dcap v5.2 IND per seed: " + ", ".join(f"s{s} R²_cb {bstat[s]['r2']:.4f} / jet {bstat[s]['jet']:.4f} / top10 under {bstat[s]['tail'][1]:.3f}" for s in SEEDS),
              "reading rule: 单 seed 对单 seed 的 95% 带 ±0.045；只看三 seed 配对均值与同向性；jet 子集 = 真值 p99 > 40 Pa 的病例。"]
    text = "\n".join(lines) + "\n"
    (EXP / f"readout_phys_ckpt_{a.checkpoint}.md").write_text(text); (EXP / f"readout_phys_ckpt_{a.checkpoint}.json").write_text(json.dumps(out, indent=1))
    print(text)


if __name__ == "__main__":
    main()
