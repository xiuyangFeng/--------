"""Aggregate the volume_time_20260919 matrix over cv3 folds and read the gates.

峰值帧看 `field_casebalanced.r2`（对各自折底座），周期看 `cycle` 块（对 A3 的 T-null）。
每个臂只在自己的留出折上读数，门控用三折均值 + 三折符号一致性。

    python3 -m training_wss_min.experiments.volume_time_20260919.offline.report
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path("/public/newhome/cy/Digital_twin/GNN")
RUNS = ROOT / "training_wss_min/runs/volume_time_20260919"
EXP = ROOT / "training_wss_min/experiments/volume_time_20260919"
OFF = EXP / "offline"
FOLDS = (0, 1, 2)
FAMILIES = {"pressure": ("PF6_v51", ("PT0", "PTB8", "PTB16"), "Pa"),
            "velocity": ("VF6_v51", ("VT0", "VTB4", "VTB8"), "m/s")}
# 周期块里的 peak_r2cb 是固定子采样上的峰值帧，与全云的 field_casebalanced.r2 不同名以免互相覆盖
CYCLE_KEYS = ("cycle_r2cb", "trough_r2cb", "min_frame_r2cb", "tmean_r2cb",
              "ts_corr", "peak_time_err_med_frames")


def metrics(arm: str, fold: int, ckpt: str = "best") -> dict:
    path = RUNS / f"{arm}_f{fold}_s1234/eval/ckpt_{ckpt}/metrics.json"
    return json.loads(path.read_text())["test"]


def fold_mean(values: list[float]) -> float:
    return float(np.mean(values))


def main() -> None:
    a3 = json.loads((OFF / "a3_tnull.json").read_text())
    a2 = json.loads((OFF / "a2_time_basis.json").read_text())
    out: dict = {"generated_at": __import__("time").strftime("%Y-%m-%dT%H:%M:%S%z"),
                 "reading_rule": "each arm on its own held-out fold; three-fold mean + sign agreement",
                 "families": {}}
    for target, (base, arms, unit) in FAMILIES.items():
        peak_base = [metrics(base, f)["field_casebalanced"]["r2"] for f in FOLDS]
        tnull = dict(a3[target]["fold_mean"]["tnull"])
        # A3 的 fold_mean 不含最差帧，从逐折补上（三折均值）
        tnull["min_frame_r2cb"] = fold_mean([a3[target][f"fold{f}"]["arms"]["tnull"]["min_frame_r2cb"]
                                             for f in FOLDS])
        tnull_folds = [a3[target][f"fold{f}"]["arms"]["tnull"]["cycle_r2cb"] for f in FOLDS]
        block = {"unit": unit, "k_star": a2[target]["k_star"],
                 "fold_base": {"arm": base, "peak_r2cb_folds": peak_base, "peak_r2cb": fold_mean(peak_base)},
                 "t_null_cycle": tnull, "t_null_cycle_folds": tnull_folds, "arms": {}}
        for arm in arms:
            rec: dict = {}
            for ckpt in ("best", "last"):
                ms = [metrics(arm, f, ckpt) for f in FOLDS]
                peak = [m["field_casebalanced"]["r2"] for m in ms]
                cyc = {k: [m["cycle"][k] for m in ms] for k in CYCLE_KEYS}
                rec[ckpt] = {
                    "peak_r2cb_folds": peak, "peak_r2cb": fold_mean(peak),
                    "d_peak_vs_base_folds": [p - b for p, b in zip(peak, peak_base)],
                    "d_peak_vs_base": fold_mean(peak) - fold_mean(peak_base),
                    "peak_sign_agreement": int(sum(1 for p, b in zip(peak, peak_base) if p > b)),
                    **{k: fold_mean(v) for k, v in cyc.items()},
                    **{f"{k}_folds": v for k, v in cyc.items()},
                    "d_cycle_vs_tnull": fold_mean(cyc["cycle_r2cb"]) - tnull["cycle_r2cb"],
                    "cycle_sign_agreement": int(sum(1 for v, t in zip(cyc["cycle_r2cb"], tnull_folds) if v > t)),
                    "argmin_frame_folds": [m["cycle"]["argmin_frame"] for m in ms],
                    "peak_r2cb_subsample": fold_mean([m["cycle"]["peak_r2cb"] for m in ms])}
                # 门：峰值帧不劣于折底座（三折均值），周期严格胜 T-null（三折均值 + 三折同号）
                rec[ckpt]["gate_peak"] = bool(rec[ckpt]["d_peak_vs_base"] >= 0.0)
                rec[ckpt]["gate_cycle"] = bool(rec[ckpt]["d_cycle_vs_tnull"] > 0.0
                                               and rec[ckpt]["cycle_sign_agreement"] == 3)
                rec[ckpt]["gate"] = bool(rec[ckpt]["gate_peak"] and rec[ckpt]["gate_cycle"])
            block["arms"][arm] = rec
        out["families"][target] = block
    (EXP / "gate_report.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")

    lines = ["# volume_time_20260919 门控读数", "",
             "每个臂只在自己的留出折上读数；峰值帧对各自折底座，周期对 A3 的 T-null（三折均值）。", ""]
    for target, block in out["families"].items():
        tn = block["t_null_cycle"]
        lines += [f"## {target}（K\\*={block['k_star']}，单位 {block['unit']}）", "",
                  f"折底座 {block['fold_base']['arm']} 峰值帧 R²_cb 三折均值 "
                  f"**{block['fold_base']['peak_r2cb']:.4f}**"
                  f"（{'/'.join(f'{v:.4f}' for v in block['fold_base']['peak_r2cb_folds'])}）；"
                  f"T-null 周期 R²_cb **{tn['cycle_r2cb']:.4f}**、谷底 {tn['trough_r2cb']:.4f}、"
                  f"最差帧 {tn['min_frame_r2cb']:.4f}、时均 {tn['tmean_r2cb']:.4f}。", "",
                  "| 臂 | ckpt | 峰值 R²_cb | Δ vs 折底座 | 折符号 | 周期 R²_cb | Δ vs T-null | 折符号 | 谷底 | 最差帧 | 时均 | ts-corr | 峰时误差(帧) | 门 |",
                  "|---|---|---:|---:|:-:|---:|---:|:-:|---:|---:|---:|---:|---:|:-:|"]
        for arm, rec in block["arms"].items():
            for ckpt in ("best", "last"):
                r = rec[ckpt]
                lines.append(
                    f"| {arm} | {ckpt} | {r['peak_r2cb']:.4f} | {r['d_peak_vs_base']:+.4f} | "
                    f"{r['peak_sign_agreement']}/3 | {r['cycle_r2cb']:.4f} | {r['d_cycle_vs_tnull']:+.4f} | "
                    f"{r['cycle_sign_agreement']}/3 | {r['trough_r2cb']:.4f} | {r['min_frame_r2cb']:.4f} | "
                    f"{r['tmean_r2cb']:.4f} | {r['ts_corr']:.3f} | {r['peak_time_err_med_frames']:.1f} | "
                    f"{'PASS' if r['gate'] else 'NO'} |")
        lines.append("")
    (EXP / "gate_report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
