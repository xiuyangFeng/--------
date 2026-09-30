#!/usr/bin/env python3
"""汇总「峰值预训练 → 时间适配器」v5.1 探针（矩阵 P0.10）：逐臂逐折表 + 预注册门 G0/F1/F2/A。

参照：阶段 2 同折 T0 与 X5D 峰值（stage2_report_best.json，evaluate 口径）、D2 T-null（d2_tnull.json，TimeMetrics 口径）。
用法：python -m training_wss_min.tools.report_time_adapter_probe [--experiment wss_time_adapter_v51_20260923]
输出：experiments/<exp>/report.{md,json}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path("/public/newhome/cy/Digital_twin/GNN/training_wss_min")
ARMS = ["Tnull", "P-lin", "P-mlp", "P-mlp-qx", "C-scalar", "C-raw"]
FOLDS = [0, 1, 2]
KEYS = [("cycle", "cycle_r2cb_pa"), ("trough", "trough_r2cb_pa"), ("peak", "peak_r2cb_pa"), ("tawss", "tawss_r2cb_pa"),
        ("cycle_ln", "cycle_r2cb_ln"), ("pk_med", "peak_time_err_med_frames"), ("pk_p90", "peak_time_err_p90_frames"),
        ("ts_corr", "ts_corr_ln"), ("amp_err", "amp_err_ln_med"), ("min_frame", "min_frame_r2cb_pa")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="wss_time_adapter_v51_20260923")
    args = ap.parse_args()
    exp = ROOT / "experiments" / args.experiment
    runs = ROOT / "runs" / args.experiment
    stage2 = json.loads((ROOT / "experiments/wss_time_ecc_20260918/stage2_report_best.json").read_text())["rows"]
    d2 = json.loads((ROOT / "experiments/wss_time_ecc_20260918/offline/d2_tnull.json").read_text())
    rows, missing = {}, []
    for arm in ARMS:
        for k in FOLDS:
            p = runs / f"{arm}_f{k}_s1234" / "metrics.json"
            if not p.is_file():
                missing.append(f"{arm}_f{k}")
                continue
            r = json.loads(p.read_text())
            ec, tn = r["metrics"]["eval_compatible"], r["metrics"]["tnull_compatible"]
            m = {name: float(ec[key]) for name, key in KEYS}
            m.update(d2_cycle=tn["cycle_r2cb_pa"], d2_trough=tn["trough_r2cb_pa"], d2_tawss=tn["tawss_r2cb"],
                     anchor_dev=r["anchor_check"]["max_abs_peak_ln_vs_frozen_anchor"], fit=r.get("fit", {}),
                     seconds=r["elapsed_seconds"], host=r["execution"]["hostname"], gpu=r["execution"].get("gpu"))
            rows[(arm, k)] = m
    t0 = {k: stage2[f"T0_f{k}"] for k in FOLDS}
    ref_rows = {("T0 (阶段2)", k): {"cycle": t0[k]["cycle_pa"], "trough": t0[k]["trough_pa"], "peak": t0[k]["peak_pa"],
                                    "tawss": t0[k]["tawss"], "cycle_ln": t0[k]["cycle_ln"], "pk_med": t0[k]["pk_med"],
                                    "pk_p90": t0[k]["pk_p90"], "ts_corr": t0[k]["ts_corr"], "amp_err": t0[k]["amp_err"],
                                    "min_frame": float("nan")} for k in FOLDS}
    show = ["cycle", "trough", "peak", "tawss", "cycle_ln", "pk_med", "pk_p90", "ts_corr", "amp_err"]
    lines = [f"缺失：{', '.join(missing) if missing else '无'}", "",
             "| 臂 | 折 | " + " | ".join(show) + " |", "|" + "---|" * (len(show) + 2)]
    means = {}
    for arm in ARMS + ["T0 (阶段2)"]:
        src = ref_rows if arm.startswith("T0") else rows
        got = [src[(arm, k)] for k in FOLDS if (arm, k) in src]
        for k in FOLDS:
            if (arm, k) in src:
                lines.append(f"| {arm} | {k} | " + " | ".join(f"{src[(arm, k)][s]:.4f}" for s in show) + " |")
        if len(got) == 3:
            means[arm] = {s: float(np.mean([g[s] for g in got])) for s in show}
            lines.append(f"| **{arm}** | 均值 | " + " | ".join(f"**{means[arm][s]:.4f}**" for s in show) + " |")
    gates = {}
    if all((a, k) in rows for a in ("Tnull",) for k in FOLDS):
        dev = {k: {"cycle": rows[("Tnull", k)]["d2_cycle"] - d2[f"fold{k}"]["B_scale"]["cycle_r2cb_pa"],
                   "trough": rows[("Tnull", k)]["d2_trough"] - d2[f"fold{k}"]["B_scale"]["trough_r2cb_pa"],
                   "tawss": rows[("Tnull", k)]["d2_tawss"] - d2[f"fold{k}"]["B_scale"]["tawss_r2cb"]} for k in FOLDS}
        max_dev = max(abs(v) for d in dev.values() for v in d.values())
        anchor = max(rows[key]["anchor_dev"] for key in rows)
        gates["G0"] = {"passed": bool(max_dev <= 1e-3 and anchor <= 1e-9), "tnull_vs_d2_max_abs": max_dev,
                       "anchor_max_abs": anchor, "per_fold": dev}
        cache_ref = {}
        for k in FOLDS:
            man = exp / "cache" / f"fold{k}" / "manifest.json"
            if man.is_file():
                cache_ref[k] = json.loads(man.read_text()).get("reference_check", {})
        gates["G0"]["donor_rerun_vs_stored_predictions"] = cache_ref

    def delta(a, b, key):
        return [rows[(a, k)][key] - rows[(b, k)][key] for k in FOLDS]

    if all((a, k) in rows for a in ("P-mlp", "Tnull") for k in FOLDS):
        dc, dt = delta("P-mlp", "Tnull", "cycle"), delta("P-mlp", "Tnull", "trough")
        gates["F1"] = {"passed": bool(np.mean(dc) >= 0.05 and np.mean(dt) >= 0.08 and all(x > 0 for x in dc + dt)),
                       "d_cycle": dc, "d_trough": dt, "mean_d_cycle": float(np.mean(dc)), "mean_d_trough": float(np.mean(dt))}
        d_t0 = [rows[("P-mlp", k)]["cycle"] - t0[k]["cycle_pa"] for k in FOLDS]
        gates["F2"] = {"passed": bool(np.mean(d_t0) >= -0.01), "d_cycle_vs_T0": d_t0, "mean": float(np.mean(d_t0)),
                       "d_trough_vs_T0": [rows[("P-mlp", k)]["trough"] - t0[k]["trough_pa"] for k in FOLDS],
                       "d_peak_vs_T0": [rows[("P-mlp", k)]["peak"] - t0[k]["peak_pa"] for k in FOLDS]}
    if all((a, k) in rows for a in ("P-mlp", "C-raw", "C-scalar") for k in FOLDS):
        dr, ds = delta("P-mlp", "C-raw", "cycle"), delta("P-mlp", "C-scalar", "cycle")
        gates["A"] = {"passed": bool(np.mean(dr) >= 0.02 and np.mean(ds) >= 0.02 and all(x > 0 for x in dr + ds)),
                      "d_cycle_vs_C_raw": dr, "d_cycle_vs_C_scalar": ds,
                      "d_trough_vs_C_raw": delta("P-mlp", "C-raw", "trough"), "d_trough_vs_C_scalar": delta("P-mlp", "C-scalar", "trough")}
    for arm in ("P-lin", "P-mlp-qx", "C-scalar", "C-raw"):
        if all((a, k) in rows for a in (arm, "Tnull") for k in FOLDS):
            gates.setdefault("info", {})[f"{arm} - Tnull"] = {"cycle": delta(arm, "Tnull", "cycle"), "trough": delta(arm, "Tnull", "trough")}
    lines += ["", "门：", "```json", json.dumps(gates, indent=1, ensure_ascii=False, default=float), "```"]
    fit_lines = [f"- {a}_f{k}: {json.dumps({kk: v for kk, v in rows[(a, k)]['fit'].items() if not isinstance(v, list)}, ensure_ascii=False)}"
                 f"，用时 {rows[(a, k)]['seconds']:.0f} s @ {rows[(a, k)]['host']} {rows[(a, k)]['gpu']}" for (a, k) in rows]
    lines += ["", "拟合信息："] + fit_lines
    (exp / "report.md").write_text("\n".join(lines) + "\n")
    (exp / "report.json").write_text(json.dumps({"rows": {f"{a}_f{k}": v for (a, k), v in rows.items()}, "means": means,
                                                 "gates": gates, "missing": missing}, indent=1, ensure_ascii=False, default=float))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
