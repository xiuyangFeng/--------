#!/usr/bin/env python3
"""汇总 P0.12 Temporal Transformer 单 seed cv3 结果。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


KEYS = ("cycle_r2cb_pa", "trough_r2cb_pa", "tawss_r2cb_pa", "cycle_r2cb_ln",
        "peak_time_err_med_frames", "peak_time_err_p90_frames", "ts_corr_ln", "amp_err_ln_med")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-dir", required=True)
    args = ap.parse_args()
    config_dir = Path(args.config_dir).resolve()
    matrix = json.loads((config_dir / "matrix.json").read_text())
    rows, missing = {}, []
    for job in matrix["arms"]:
        p = Path(job["done_marker"])
        if not p.is_file():
            missing.append(job["id"])
            continue
        d = json.loads(p.read_text())
        rows[job["id"]] = {"arm": d["arm"], "fold": d["fold"], "seed": d["seed"],
                            "metrics": d["metrics"]["eval_compatible"],
                            "train_seconds": d.get("train_seconds")}
    by_arm = {}
    for row in rows.values():
        by_arm.setdefault(row["arm"], []).append(row)
    means = {}
    for arm, vals in by_arm.items():
        means[arm] = {k: sum(float(v["metrics"][k]) for v in vals) / len(vals) for k in KEYS}
    paired = {}
    for a, b, label in (("TT-warm", "TT-warm-noattn", "A1_warm_minus_warm_noattn"),
                        ("TT-warm", "TT-raw", "A2_warm_minus_raw"),
                        ("TT-raw", "TT-raw-noattn", "A3_raw_attention_minus_noattn"),
                        ("TT-warm-noattn", "TT-raw-noattn", "A4_warm_representation_minus_raw")):
        av = {r["fold"]: r for r in by_arm.get(a, [])}
        bv = {r["fold"]: r for r in by_arm.get(b, [])}
        common = sorted(set(av) & set(bv))
        paired[label] = {"folds": common,
                         "delta": {k: [float(av[f]["metrics"][k] - bv[f]["metrics"][k]) for f in common] for k in KEYS},
                         "mean": {k: (sum(float(av[f]["metrics"][k] - bv[f]["metrics"][k]) for f in common) / len(common)
                                      if common else None) for k in KEYS}}
    report = {"schema": "time_transformer_report_v1", "experiment": matrix["experiment"],
              "matrix": str(config_dir / "matrix.json"), "missing": missing, "rows": rows,
              "means": means, "paired": paired,
              "note": "cv3, one seed; fold means are descriptive development evidence, not seed uncertainty."}
    exp = Path("/public/newhome/cy/Digital_twin/GNN/training_wss_min/experiments") / matrix["experiment"]
    exp.mkdir(parents=True, exist_ok=True)
    (exp / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    lines = [f"# {matrix['experiment']}", "", "缺失：" + (", ".join(missing) if missing else "无"), "",
             "| 臂 | cycle Pa | trough Pa | TAWSS | cycle ln | peak p90 | ts corr | amp err |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for arm in ("TT-warm", "TT-warm-noattn", "TT-raw", "TT-raw-noattn"):
        m = means.get(arm)
        if not m:
            continue
        lines.append(f"| {arm} | {m['cycle_r2cb_pa']:.4f} | {m['trough_r2cb_pa']:.4f} | {m['tawss_r2cb_pa']:.4f} | "
                     f"{m['cycle_r2cb_ln']:.4f} | {m['peak_time_err_p90_frames']:.2f} | {m['ts_corr_ln']:.4f} | {m['amp_err_ln_med']:.4f} |")
    lines += ["", "配对差（前者减后者，fold0/1/2；单 seed 描述）：", ""]
    for label, p in paired.items():
        lines.append(f"- **{label}** cycle Pa = " + "/".join(f"{x:+.4f}" for x in p["delta"]["cycle_r2cb_pa"]) +
                     f"；mean={p['mean']['cycle_r2cb_pa']:+.4f}；trough mean={p['mean']['trough_r2cb_pa']:+.4f}")
    lines += ["", "解释边界：本矩阵使用静态 V5.1 几何和相位，没有逐帧速度/压力状态；Transformer 只在病例级几何 token 上做时间注意力。结果用于筛选结构，不改变部署。"]
    (exp / "report.md").write_text("\n".join(lines) + "\n")
    print(exp / "report.json")
    print(exp / "report.md")


if __name__ == "__main__":
    main()
