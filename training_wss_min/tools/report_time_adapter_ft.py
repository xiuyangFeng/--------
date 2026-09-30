#!/usr/bin/env python3
"""汇总导师方案第二步「解冻版时间适配器」（矩阵 P0.11）：TL-warm / TL-random × 三折 + 预注册门 U1–U4。

参照同折同口径：P0.10 探针（Tnull / P-mlp-qx / C-raw，report.json）与阶段 2 T0（stage2_report_best.json）。
用法：python -m training_wss_min.tools.report_time_adapter_ft
输出：experiments/wss_time_adapter_ft_v51_20260923/report.{md,json}
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path("/public/newhome/cy/Digital_twin/GNN/training_wss_min")
NAME = "wss_time_adapter_ft_v51_20260923"
FOLDS = [0, 1, 2]
SHOW = ["cycle", "trough", "peak", "tawss", "cycle_ln", "pk_med", "pk_p90", "ts_corr", "amp_err"]
KEYMAP = {"cycle": "cycle_r2cb_pa", "trough": "trough_r2cb_pa", "peak": "peak_r2cb_pa", "tawss": "tawss_r2cb_pa",
          "cycle_ln": "cycle_r2cb_ln", "pk_med": "peak_time_err_med_frames", "pk_p90": "peak_time_err_p90_frames",
          "ts_corr": "ts_corr_ln", "amp_err": "amp_err_ln_med"}


def main():
    exp, runs = ROOT / "experiments" / NAME, ROOT / "runs" / NAME
    probe = json.loads((ROOT / "experiments/wss_time_adapter_v51_20260923/report.json").read_text())["rows"]
    stage2 = json.loads((ROOT / "experiments/wss_time_ecc_20260918/stage2_report_best.json").read_text())["rows"]
    rows, extra, missing = {}, {}, []
    for arm in ("TL-warm", "TL-random"):
        for k in FOLDS:
            p = runs / f"{arm}_f{k}_s1234" / "metrics.json"
            if not p.is_file():
                missing.append(f"{arm}_f{k}")
                continue
            r = json.loads(p.read_text())
            ec = r["metrics"]["eval_compatible"]
            rows[(arm, k)] = {s: float(ec[KEYMAP[s]]) for s in SHOW}
            extra[(arm, k)] = {"anchor_dev": r["anchor_check"]["max_abs_peak_ln_vs_frozen_anchor"],
                               "init_qx": r["initialization"].get("init_query_x_max_abs_diff_vs_cache"),
                               "init_delta": r["initialization"].get("init_delta_max_abs"),
                               "final_loss_z": r["fit"]["final_loss_z"], "train_min": r["fit"]["train_seconds"] / 60,
                               "host": r["execution"]["hostname"], "gpu": r["execution"].get("gpu")}
    for arm in ("Tnull", "P-mlp-qx", "C-raw"):
        for k in FOLDS:
            rows[(arm, k)] = {s: probe[f"{arm}_f{k}"][s] for s in SHOW}
    for k in FOLDS:
        t = stage2[f"T0_f{k}"]
        rows[("T0", k)] = {"cycle": t["cycle_pa"], "trough": t["trough_pa"], "peak": t["peak_pa"], "tawss": t["tawss"],
                           "cycle_ln": t["cycle_ln"], "pk_med": t["pk_med"], "pk_p90": t["pk_p90"], "ts_corr": t["ts_corr"],
                           "amp_err": t["amp_err"]}
    order = ["TL-warm", "TL-random", "P-mlp-qx", "C-raw", "Tnull", "T0"]
    lines = [f"缺失：{', '.join(missing) if missing else '无'}", "",
             "| 臂 | 折 | " + " | ".join(SHOW) + " |", "|" + "---|" * (len(SHOW) + 2)]
    means = {}
    for arm in order:
        got = [rows[(arm, k)] for k in FOLDS if (arm, k) in rows]
        for k in FOLDS:
            if (arm, k) in rows:
                lines.append(f"| {arm} | {k} | " + " | ".join(f"{rows[(arm, k)][s]:.4f}" for s in SHOW) + " |")
        if len(got) == 3:
            means[arm] = {s: float(np.mean([g[s] for g in got])) for s in SHOW}
            lines.append(f"| **{arm}** | 均值 | " + " | ".join(f"**{means[arm][s]:.4f}**" for s in SHOW) + " |")

    def d(a, b, key="cycle"):
        return [rows[(a, k)][key] - rows[(b, k)][key] for k in FOLDS]

    gates = {}
    if all(("TL-warm", k) in rows for k in FOLDS):
        warm = [extra[("TL-warm", k)] for k in FOLDS]
        allx = [extra[key] for key in extra]
        gates["G0"] = {"passed": bool(max(e["anchor_dev"] for e in allx) <= 1e-9 and max(e["init_qx"] for e in warm) <= 1e-4
                                      and max(e["init_delta"] for e in allx) == 0.0),
                       "anchor_max": max(e["anchor_dev"] for e in allx), "warm_init_qx_max": max(e["init_qx"] for e in warm)}
        dc, dt = d("TL-warm", "Tnull"), d("TL-warm", "Tnull", "trough")
        gates["U1"] = {"passed": bool(np.mean(dc) >= 0.05 and np.mean(dt) >= 0.08 and all(x > 0 for x in dc + dt)),
                       "d_cycle": dc, "d_trough": dt}
        du = d("TL-warm", "P-mlp-qx")
        gates["U2"] = {"passed": bool(np.mean(du) >= 0.02 and all(x > 0 for x in du)), "d_cycle_vs_frozen": du}
        d0, dr = d("TL-warm", "T0"), d("TL-warm", "C-raw")
        gates["U3"] = {"passed": bool(np.mean(d0) >= -0.01 and np.mean(dr) >= -0.01), "d_cycle_vs_T0": d0, "d_cycle_vs_C_raw": dr,
                       "d_trough_vs_T0": d("TL-warm", "T0", "trough"), "d_trough_vs_C_raw": d("TL-warm", "C-raw", "trough"),
                       "d_peak_vs_T0": d("TL-warm", "T0", "peak")}
    if all((a, k) in rows for a in ("TL-warm", "TL-random") for k in FOLDS):
        dw = d("TL-warm", "TL-random")
        gates["U4"] = {"passed": bool(np.mean(dw) >= 0.02 and all(x > 0 for x in dw)), "d_cycle_warm_minus_random": dw,
                       "d_trough_warm_minus_random": d("TL-warm", "TL-random", "trough")}
        gates["info"] = {"TL-random - Tnull": d("TL-random", "Tnull"), "TL-random - C-raw": d("TL-random", "C-raw")}
    lines += ["", "门：", "```json", json.dumps(gates, indent=1, ensure_ascii=False, default=float), "```", "", "运行信息："]
    lines += [f"- {a}_f{k}: 末 epoch loss_z {e['final_loss_z']:.4f}，训练 {e['train_min']:.0f} min，锚偏差 {e['anchor_dev']:.1e}，"
              f"初始 query_x 对缓存 {e['init_qx']:.2e} @ {e['host']} {e['gpu']}" for (a, k), e in extra.items()]
    (exp / "report.md").write_text("\n".join(lines) + "\n")
    (exp / "report.json").write_text(json.dumps({"rows": {f"{a}_f{k}": v for (a, k), v in rows.items()}, "means": means,
                                                 "gates": gates, "runs": {f"{a}_f{k}": v for (a, k), v in extra.items()},
                                                 "missing": missing}, indent=1, ensure_ascii=False, default=float))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
