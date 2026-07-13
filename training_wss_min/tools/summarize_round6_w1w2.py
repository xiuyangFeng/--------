#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""汇总第六轮 W1（A/B/D/E 2×2 因子）与 W2-L1（raw-Huber λ 扫描）结果。

只读 ``runs/<name>/eval/metrics.json``，输出：
- W1：A/B/D/E 三 seed 均值 + 严格归因 B-A / E-B / E-D / D-A / 交互 E-D-B+A（配对到 seed）。
- W2-L1：λ∈{0.1,0.3,1.0} 单 seed vs L0 控制（r6_scale_D_xyzgeom_s1234, λ=0）。

用法：
    python -m training_wss_min.tools.summarize_round6_w1w2
产物：
    training_wss_min/runs/_audits/round6_w1w2_summary/{summary.json,report.md}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from training_wss_min import config as C

OUT_DIR = C.RUNS_ROOT / "_audits" / "round6_w1w2_summary"
SEEDS = (1234, 7, 2025)

W1_GROUPS = {
    "A": "r6_scale_A_xyz_s{seed}",
    "B": "r6_scale_B_xyzscale_s{seed}",
    "D": "r6_scale_D_xyzgeom_s{seed}",
    "E": "r6_scale_E_xyzscalegeom_s{seed}",
}
L1_CONTROL = "r6_scale_D_xyzgeom_s1234"
L1_RUNS = {
    "lam0.1": "r6_l1_rawhuber_lam0p1_s1234",
    "lam0.3": "r6_l1_rawhuber_lam0p3_s1234",
    "lam1.0": "r6_l1_rawhuber_lam1_s1234",
}
# W3 确认阶段：L1 λ=1.0 三 seed（loss-control on D）、W3 组合 E×λ1.0 三 seed。
L1_LAM1_PATTERN = "r6_l1_rawhuber_lam1_s{seed}"
W3_PATTERN = "r6_w3_e_rawhuber_lam1_s{seed}"
E_PATTERN = "r6_scale_E_xyzscalegeom_s{seed}"
D_PATTERN = "r6_scale_D_xyzgeom_s{seed}"
# 主 + 护栏指标键（从 metrics.json 的 val 提取）
METRIC_KEYS = [
    "r2_field", "r2_field_cb", "r2_casemean", "r2_casemed",
    "r2_negative_cases", "mae_field", "top10_ratio", "top10_iou",
    "p99_ratio", "high_wss_mae", "max_ratio",
]


def _load_metrics(run_name: str) -> Optional[Dict[str, float]]:
    p = C.RUNS_ROOT / run_name / "eval" / "metrics.json"
    if not p.is_file():
        return None
    m = json.loads(p.read_text())["val"]
    f = m["field"]
    fb = m.get("field_casebalanced", {})
    a = m["aggregate"]
    cal = m.get("calibration", {})
    hot = m.get("hotspot", {})
    return {
        "r2_field": f.get("r2", float("nan")),
        "r2_field_cb": fb.get("r2", float("nan")),
        "r2_casemean": a.get("r2_casemean", float("nan")),
        "r2_casemed": a.get("r2_casemed", float("nan")),
        "r2_negative_cases": a.get("r2_negative_cases", float("nan")),
        "mae_field": f.get("mae", float("nan")),
        "top10_ratio": cal.get("top10_pred_true_ratio", float("nan")),
        "top10_iou": hot.get("top10_iou", float("nan")),
        "p99_ratio": cal.get("p99_pred_true_ratio", float("nan")),
        "high_wss_mae": hot.get("high_wss_mae", float("nan")),
        "max_ratio": cal.get("max_pred_true_ratio", float("nan")),
    }


def _group_seed_metrics(pattern: str) -> Dict[int, Optional[Dict]]:
    return {s: _load_metrics(pattern.format(seed=s)) for s in SEEDS}


def _mean_std(vals: List[float]) -> str:
    v = [x for x in vals if x is not None and np.isfinite(x)]
    if not v:
        return "NA"
    if len(v) == 1:
        return f"{v[0]:+.3f}"
    return f"{np.mean(v):+.3f}±{np.std(v):.3f}"


def _paired_diff(g1: Dict[int, Optional[Dict]], g2: Dict[int, Optional[Dict]],
                 key: str) -> List[float]:
    """g1 - g2 逐 seed 配对差。"""
    out = []
    for s in SEEDS:
        a = g1.get(s); b = g2.get(s)
        if a and b and np.isfinite(a[key]) and np.isfinite(b[key]):
            out.append(a[key] - b[key])
    return out


def _interaction(groups: Dict[str, Dict[int, Optional[Dict]]], key: str) -> List[float]:
    """交互项 E-D-B+A 逐 seed。"""
    out = []
    for s in SEEDS:
        vals = {g: groups[g].get(s) for g in ("A", "B", "D", "E")}
        if all(v and np.isfinite(v[key]) for v in vals.values()):
            out.append(vals["E"][key] - vals["D"][key] - vals["B"][key] + vals["A"][key])
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    groups = {g: _group_seed_metrics(pat) for g, pat in W1_GROUPS.items()}

    L = []
    L.append("# 第六轮 W1（尺度×几何 2×2 因子）与 W2-L1（raw-Huber）汇总")
    L.append("")
    L.append("> 只读 `runs/<name>/eval/metrics.json`（val, dev1）。`r2_field_cb` = 病例等权全场 R²。")
    L.append("")

    # ---- W1 组均值 ----
    L.append("## W1｜A/B/D/E 三 seed 结果")
    L.append("")
    L.append("| 组 | 输入 | R²_field | R²_field_cb | R²_casemean | R²_casemed | 负例(∑) | top10 | IoU | MAE |")
    L.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    inputs = {"A": "xyz", "B": "xyz+scale", "D": "xyz+geom", "E": "xyz+scale+geom"}
    complete = {}
    for g in ("A", "B", "D", "E"):
        gm = groups[g]
        complete[g] = sum(1 for s in SEEDS if gm[s] is not None)
        def col(k):
            return _mean_std([gm[s][k] if gm[s] else None for s in SEEDS])
        neg = [gm[s]["r2_negative_cases"] if gm[s] else None for s in SEEDS]
        neg_sum = sum(int(x) for x in neg if x is not None and np.isfinite(x))
        L.append(f"| {g} ({complete[g]}/3) | {inputs[g]} | {col('r2_field')} | {col('r2_field_cb')} | "
                 f"{col('r2_casemean')} | {col('r2_casemed')} | {neg_sum} | {col('top10_ratio')} | "
                 f"{col('top10_iou')} | {col('mae_field')} |")
    L.append("")

    # ---- 严格归因 ----
    L.append("## W1｜严格归因（逐 seed 配对差，mean±std）")
    L.append("")
    L.append("| 对比 | 含义 | ΔR²_field | ΔR²_field_cb | ΔR²_casemean |")
    L.append("|---|---|---:|---:|---:|")
    contrasts = [
        ("B", "A", "B−A", "尺度主效应"),
        ("D", "A", "D−A", "几何主效应"),
        ("E", "B", "E−B", "几何｜有尺度"),
        ("E", "D", "E−D", "尺度｜有几何"),
    ]
    for g1, g2, tag, meaning in contrasts:
        row = f"| {tag} | {meaning} |"
        for k in ("r2_field", "r2_field_cb", "r2_casemean"):
            row += f" {_mean_std(_paired_diff(groups[g1], groups[g2], k))} |"
        L.append(row)
    # 交互项
    row = "| E−D−B+A | 尺度×几何交互 |"
    for k in ("r2_field", "r2_field_cb", "r2_casemean"):
        row += f" {_mean_std(_interaction(groups, k))} |"
    L.append(row)
    L.append("")
    L.append("读法：尺度主效应=B−A；在已有几何条件下尺度的增益=E−D。若 E−D 明显小于 B−A，"
             "说明几何已吸收大部分尺度信息（冗余）；交互项>seed 噪声才可宣称尺度×几何协同。")
    L.append("")

    # ---- W2 L1 ----
    L.append("## W2-L1｜raw-Huber λ 扫描（固定 D 输入，单 seed 1234）")
    L.append("")
    ctrl = _load_metrics(L1_CONTROL)
    L.append("| run | λ | R²_field | R²_field_cb | R²_casemean | top10 | p99 | IoU | high_wss_MAE | max_ratio |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    def _row(name, lam, m):
        if m is None:
            return f"| {name} | {lam} | NA | NA | NA | NA | NA | NA | NA | NA |"
        def f(k, d=3):
            return f"{m[k]:+.{d}f}" if np.isfinite(m[k]) else "NA"
        return (f"| {name} | {lam} | {f('r2_field')} | {f('r2_field_cb')} | {f('r2_casemean')} | "
                f"{f('top10_ratio')} | {f('p99_ratio')} | {f('top10_iou')} | "
                f"{m['high_wss_mae']:.3f} | {m['max_ratio']:.2f} |")

    L.append(_row("L0 (D, λ=0)", 0.0, ctrl))
    l1_metrics = {}
    for tag, run in L1_RUNS.items():
        m = _load_metrics(run)
        l1_metrics[tag] = m
        L.append(_row(run.replace("r6_l1_rawhuber_", ""), tag.replace("lam", ""), m))
    L.append("")
    if ctrl:
        L.append("L1 Gate（vs L0）：看 high_wss_MAE / p99 / top10 是否改善且 R²_field/casemean 不明显退化；"
                 "max_ratio 明显>1.5 表示爆峰。只做单 seed 廉价筛选，选中的 λ 才补三 seed。")
    L.append("")

    # ---- W3 确认阶段：L1 λ=1.0 三 seed 与 W3 组合 ----
    l1_lam1 = _group_seed_metrics(L1_LAM1_PATTERN)
    w3 = _group_seed_metrics(W3_PATTERN)
    e_grp = _group_seed_metrics(E_PATTERN)
    d_grp = _group_seed_metrics(D_PATTERN)
    n_l1_lam1 = sum(1 for s in SEEDS if l1_lam1[s] is not None)
    n_w3 = sum(1 for s in SEEDS if w3[s] is not None)
    if n_l1_lam1 or n_w3:
        L.append("## W3｜确认阶段（三 seed）：L1 λ=1.0 与唯一组合 E×raw-Huber")
        L.append("")
        L.append("| 配置 | 输入 | λ | R²_field | R²_field_cb | R²_casemean | R²_casemed | 负例(∑) | top10 | IoU | high_wss_MAE |")
        L.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

        def _grp_row(label, inp, lam, grp):
            def col(k):
                return _mean_std([grp[s][k] if grp[s] else None for s in SEEDS])
            neg = sum(int(grp[s]["r2_negative_cases"]) for s in SEEDS
                      if grp[s] and np.isfinite(grp[s]["r2_negative_cases"]))
            hmae = _mean_std([grp[s]["high_wss_mae"] if grp[s] else None for s in SEEDS])
            nseed = sum(1 for s in SEEDS if grp[s] is not None)
            return (f"| {label} ({nseed}/3) | {inp} | {lam} | {col('r2_field')} | {col('r2_field_cb')} | "
                    f"{col('r2_casemean')} | {col('r2_casemed')} | {neg} | {col('top10_ratio')} | "
                    f"{col('top10_iou')} | {hmae} |")

        L.append(_grp_row("D (loss-control 基线)", "xyz+geom", 0.0, d_grp))
        L.append(_grp_row("D+rawHuber (loss-control)", "xyz+geom", 1.0, l1_lam1))
        L.append(_grp_row("E (input-control)", "xyz+scale+geom", 0.0, e_grp))
        L.append(_grp_row("**W3 = E+rawHuber**", "xyz+scale+geom", 1.0, w3))
        L.append("")
        if n_w3 == 3:
            L.append("### W3 Gate（配对 seed 差，mean±std）")
            L.append("")
            L.append("| 对比 | 含义 | ΔR²_field | ΔR²_field_cb | ΔR²_casemean | Δtop10 | ΔIoU |")
            L.append("|---|---|---:|---:|---:|---:|---:|")
            for g2, tag, meaning in [(e_grp, "W3−E", "加 raw-Huber 的增益(控输入)"),
                                     (l1_lam1, "W3−(D+rH)", "加 scale 的增益(控 loss)")]:
                row = f"| {tag} | {meaning} |"
                for k in ("r2_field", "r2_field_cb", "r2_casemean", "top10_ratio", "top10_iou"):
                    row += f" {_mean_std(_paired_diff(w3, g2, k))} |"
                L.append(row)
            L.append("")
            L.append("W3 判据（§6）：W3 需**同时**优于 input-control(E) 与 loss-control(D+rawHuber)，否则不再扩展交互组合。"
                     "W3−E 反映 raw-Huber 在最佳输入上的增益；W3−(D+rH) 反映 scale 在最佳 loss 上的增益。")
            L.append("")

    (OUT_DIR / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    summary = {
        "w1_groups": {g: {str(s): groups[g][s] for s in SEEDS} for g in groups},
        "w1_complete": complete,
        "w2_l1_control": ctrl,
        "w2_l1_runs": l1_metrics,
        "w3_l1_lam1_3seed": {str(s): l1_lam1[s] for s in SEEDS},
        "w3_combo_3seed": {str(s): w3[s] for s in SEEDS},
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print("== round6 W1/W2/W3 summary ==")
    for g in ("A", "B", "D", "E"):
        print(f"  group {g}: {complete[g]}/3 seeds complete")
    print("  L1 λ-gate:", {t: ("done" if m else "pending") for t, m in l1_metrics.items()})
    print(f"  L1 λ=1.0 confirm: {n_l1_lam1}/3 seeds;  W3 combo: {n_w3}/3 seeds")
    print(f"-> {OUT_DIR/'report.md'}")


if __name__ == "__main__":
    main()
