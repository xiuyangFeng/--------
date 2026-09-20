#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Report for the V6 single-frame WSS matrix against the R4 baseline on test34.

Statistics contract is inherited from the bottleneck-transformer matrix, and the
decision rule itself is imported (``report_bt_matrix.verdict_for``) rather than
re-implemented: the 2026-09-08 review found a copy of that rule drifting between
the report and the workbook, which produced opposite verdicts on one dataset.

* Reference = the mean over every R4 seed on disk (not its best seed).
* Noise band is recomputed here from those runs, per metric, as a band on a
  DIFFERENCE: sd * sqrt(1/n_arm + 1/n_base).
* Every arm in this matrix has ONE seed, so no arm gets a verdict; the tables
  rank candidates for a three-seed confirmation round.
* Decision metric is normalized R2_cb; physical R2_cb is the consistency check.
* Deployment metrics (high-WSS R2, top10 amplitude ratio, p99 ratio, top10 IoU)
  are reported next to it because the matrix targets peak amplitude, not only R2.

    python -m training_wss_min.tools.report_v6_matrix [--checkpoint best]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from training_wss_min.tools.report_bt_matrix import BASE_ARM, dig, f, seeds_of, verdict_for

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "training_wss_min/runs"
MATRIX = ROOT / "training_wss_min/configs/v6_singleframe_20260909/matrix.json"
EXP_DIR = ROOT / "training_wss_min/experiments/v6_singleframe_20260909"
PHYS = ("field_casebalanced", "r2")
NORM = ("normalized", "field_casebalanced", "r2")
DEPLOY = {
    "high_wss_r2": ("regional_field", "high_wss", "r2"),
    "top10_ratio": ("calibration", "top10_pred_true_ratio"),
    "p99_ratio": ("calibration", "p99_pred_true_ratio"),
    "top10_iou": ("hotspot", "top10_iou_casemean"),
    "case_mean": ("aggregate", "r2_casemean"),
    "case_p10": ("aggregate", "r2_casep10"),
    "mae": ("field", "mae"),
    "spearman": ("hotspot", "spearman_all_casemean"),
    "norm_high_wss_r2": ("normalized", "regional_field", "high_wss", "r2"),
    "norm_case_mean": ("normalized", "aggregate", "r2_casemean"),
}


def load_run(run_name: str, checkpoint: str, sub: str | None = None):
    path = RUNS / run_name / "eval" / (sub or f"ckpt_{checkpoint}") / "metrics.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("test", payload)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="best", choices=("best", "last"))
    ap.add_argument("--out", default=str(EXP_DIR))
    args = ap.parse_args(argv)
    matrix = json.loads(MATRIX.read_text(encoding="utf-8"))

    base = seeds_of(BASE_ARM[1], args.checkpoint)
    if not base:
        print("R4 baseline runs not found")
        return 1
    base_phys = np.array([dig(m, PHYS) for m in base.values()], dtype=float)
    base_norm = np.array([dig(m, NORM) for m in base.values()], dtype=float)
    n_base = len(base_phys)
    sd_p = float(base_phys.std(ddof=1))
    sd_n = float(base_norm.std(ddof=1))
    ref = {name: float(np.mean([dig(m, path) for m in base.values()])) for name, path in DEPLOY.items()}

    lines = []
    add = lines.append
    add(f"### 噪声标定与参照（R4 同一配置 {n_base} 个 seed {sorted(base)}，checkpoint={args.checkpoint}）")
    add(f"物理 R²_cb：{' / '.join(f(v) for v in base_phys)}  → 均值 {f(base_phys.mean())}，seed 标准差 {f(sd_p)}")
    add(f"归一化 R²_cb：{' / '.join(f(v) for v in base_norm)}  → 均值 {f(base_norm.mean())}，seed 标准差 {f(sd_n)}")
    add(f"单 seed 对 {n_base} seed 参照之差的 95% 带：物理 ±{f(1.96 * sd_p * math.sqrt(1 + 1 / n_base))}，"
        f"归一化 ±{f(1.96 * sd_n * math.sqrt(1 + 1 / n_base))}")
    add("**本矩阵每臂只有 1 个 seed，按既定统计合同一律不判定**；下表的 Δ 只用来排出补 seed 的优先级。")
    add("判据指标 = 归一化 R²_cb（预登记空间，seed sd 只有物理的 1/4），物理 R²_cb 作一致性检查。\n")

    rows = []
    for arm in matrix["arms"]:
        rows.append((arm, load_run(arm["run_name"], args.checkpoint)))

    add("### 主表（vs R4 全部 seed 的均值）")
    add("| 臂 | 唯一变化 | 物理 R²_cb | Δ物理 | 归一化 R²_cb | Δ归一化 | high-WSS R² | top10 幅值比 | p99 比 | top10 IoU | 判定 |")
    add("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    add(f"| R4 基准（参照，{n_base} seed 均值） | — | {f(base_phys.mean())} | +0.0000 | {f(base_norm.mean())} | "
        f"+0.0000 | {f(ref['high_wss_r2'])} | {f(ref['top10_ratio'], 3)} | {f(ref['p99_ratio'], 3)} | "
        f"{f(ref['top10_iou'], 3)} | 参照 |")
    summary = {}
    for arm, m in rows:
        if m is None:
            add(f"| {arm['id']} {arm['config'][:-5]} | {', '.join(arm['changes'])} |" + " — |" * 8 + " 未完成 |")
            continue
        phys, norm = dig(m, PHYS), dig(m, NORM)
        d_p, d_n = phys - base_phys.mean(), norm - base_norm.mean()
        verdict, band_p, band_n = verdict_for(d_p, d_n, 1, n_base, sd_p, sd_n)
        values = {name: dig(m, path) for name, path in DEPLOY.items()}
        summary[arm["id"]] = {
            "config": arm["config"], "run_name": arm["run_name"], "changes": arm["changes"],
            "hypothesis": arm["hypothesis"], "physical_r2_cb": phys, "normalized_r2_cb": norm,
            "delta_physical": d_p, "delta_normalized": d_n, "verdict": verdict,
            "band_physical": band_p, "band_normalized": band_n,
            "back_transform": m.get("back_transform", "exp(mu)"), **values,
        }
        add("| " + " | ".join([
            f"{arm['id']} {arm['config'][:-5]}", ", ".join(arm["changes"]),
            f(phys), f(d_p), f(norm), f(d_n), f(values["high_wss_r2"]),
            f(values["top10_ratio"], 3), f(values["p99_ratio"], 3), f(values["top10_iou"], 3), verdict,
        ]) + " |")

    add("\n### 部署侧补充（物理空间；Δ 相对同一参照）")
    add("| 臂 | case mean | case P10 | MAE (Pa) | Spearman | Δhigh-WSS | Δtop10 幅值比 | Δp99 比 | ΔIoU | 回变换 |")
    add("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    add(f"| R4 基准（参照） | {f(ref['case_mean'])} | {f(ref['case_p10'])} | {f(ref['mae'], 3)} | "
        f"{f(ref['spearman'], 3)} | +0.0000 | +0.000 | +0.000 | +0.000 | exp(mu) |")
    for arm_id, s in summary.items():
        add("| " + " | ".join([
            f"{arm_id} {s['config'][:-5]}", f(s["case_mean"]), f(s["case_p10"]), f(s["mae"], 3), f(s["spearman"], 3),
            f(s["high_wss_r2"] - ref["high_wss_r2"]), f(s["top10_ratio"] - ref["top10_ratio"], 3),
            f(s["p99_ratio"] - ref["p99_ratio"], 3), f(s["top10_iou"] - ref["top10_iou"], 3), s["back_transform"],
        ]) + " |")

    # ---- per-case gains: does an arm lift the hard cases, and do two arms fix the same cases?
    def per_case(m):
        return {k: v["overall"]["r2"] for k, v in m["per_case"].items()}
    ref_case = {}
    for case in per_case(list(base.values())[0]):
        ref_case[case] = float(np.mean([per_case(m)[case] for m in base.values()]))
    hardest = sorted(ref_case, key=ref_case.get)[:8]
    deltas = {}
    add("\n### 逐例增益（相对同一参照的逐例 R²；最差 8 例 = 参照最低的 8 个病例）")
    add("| 臂 | 逐例 ΔR² 均值 | 变好病例数 | 最差 8 例 ΔR² 均值 |")
    add("| --- | ---: | ---: | ---: |")
    for arm, m in rows:
        if m is None:
            continue
        case_r2 = per_case(m)
        d = np.array([case_r2[c] - ref_case[c] for c in ref_case])
        deltas[arm["id"]] = {c: case_r2[c] - ref_case[c] for c in ref_case}
        dw = np.array([case_r2[c] - ref_case[c] for c in hardest])
        add(f"| {arm['id']} {arm['config'][:-5]} | {f(d.mean(), 4)} | {int((d > 0).sum())}/{len(d)} | {f(dw.mean(), 4)} |")
    ids = list(deltas)
    if len(ids) > 1:
        add("\n逐例增益的相关系数（低 = 两个臂修好的不是同一批病例 → 有互补空间）：")
        add("| | " + " | ".join(ids) + " |")
        add("| --- |" + " ---: |" * len(ids))
        for a in ids:
            cells = [f(float(np.corrcoef([deltas[a][c] for c in ref_case], [deltas[b][c] for c in ref_case])[0, 1]), 2)
                     for b in ids]
            add(f"| {a} | " + " | ".join(cells) + " |")

    add("\n### 归一化空间（判据空间）")
    add("| 臂 | R²_cb | case mean | high-WSS R² |")
    add("| --- | ---: | ---: | ---: |")
    add(f"| R4 基准（参照） | {f(base_norm.mean())} | {f(ref['norm_case_mean'])} | {f(ref['norm_high_wss_r2'])} |")
    for arm_id, s in summary.items():
        add(f"| {arm_id} {s['config'][:-5]} | {f(s['normalized_r2_cb'])} | {f(s['norm_case_mean'])} | "
            f"{f(s['norm_high_wss_r2'])} |")

    text = "\n".join(lines) + "\n"
    print(text)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"matrix_tables_{args.checkpoint}.md").write_text(text, encoding="utf-8")
    (out_dir / f"matrix_summary_{args.checkpoint}.json").write_text(json.dumps({
        "checkpoint": args.checkpoint,
        "reference": {
            "run_stem": BASE_ARM[1], "seeds": sorted(base), "n": n_base,
            "physical_r2_cb_mean": float(base_phys.mean()), "normalized_r2_cb_mean": float(base_norm.mean()),
            "sd_physical": sd_p, "sd_normalized": sd_n, "deployment_metrics": ref,
        },
        "arms": summary,
        "per_case_delta": {k: {c: round(v, 6) for c, v in d.items()} for k, d in deltas.items()},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"written -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
