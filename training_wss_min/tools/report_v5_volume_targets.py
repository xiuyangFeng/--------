"""Markdown summary for the V5 volume-target runs (mixed pressure, velocity |u|) on test34."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "runs" / "v5_rerun_20260906" / "outputs"
RUNS = {
    "压力（壁面∪内部，Pa 相对）": "r5p_pressure_mixed_qad_s1234",
    "速度 |u|（内部，m/s）": "r5v_velocity_qad_s1234",
}


def load(run: str, ck: str):
    p = ROOT / run / "eval" / f"ckpt_{ck}" / "metrics.json"
    if not p.is_file():
        return None
    m = json.loads(p.read_text(encoding="utf-8"))
    return m.get("test", m)


def esc(text: str) -> str:
    """表格单元格里的竖线必须转义。臂名含 |u|（速度幅值），不转义会把该行拆成多列，
    渲染后所有数值右移两列、R² cb 位置显示成 'u'（2026-09-08 用户发现）。"""
    return str(text).replace("|", "\\|")


def f(x, d=4):
    try:
        return f"{float(x):.{d}f}"
    except (TypeError, ValueError):
        return "—"


def row(label, m):
    n = m["normalized"]
    return (f"| {esc(label)} | {f(m['field_casebalanced']['r2'])} | {f(m['field']['r2'])} | {f(m['aggregate']['r2_casemean'])} | "
            f"{f(m['aggregate']['r2_casemed'])} | {f(m['aggregate']['r2_casep10'])} | {m['aggregate']['r2_negative_cases']}/{m['aggregate']['n_cases']} | "
            f"{f(m['field']['rmse'],3)} | {f(m['field']['mae'],3)} | {f(m['field']['nrmse_range'])} | {f(m['field'].get('nmae_range'))} | "
            f"{f(m['regional_field']['high_wss']['r2'])} | {f(m['calibration']['top10_pred_true_ratio'],3)} | {f(m['calibration']['p99_pred_true_ratio'],3)} | "
            f"{f(m['hotspot']['top10_iou_casemean'],3)} | {f(m['hotspot']['spearman_all_casemean'],3)} | {f(n['field'].get('nmae_range'))} | {f(m['field_casebalanced'].get('r2_linear_fit'))} | {f(m['field_casebalanced'].get('linear_fit_slope'),3)} |")


def main(argv=None) -> int:
    hdr = ("| run·ckpt | R² cb | R² pooled | 逐例均值 R² | 逐例中位 R² | 逐例 p10 R² | 负 R² 例 | RMSE | MAE | nRMSE(range) | nMAE(range) | "
           "高值区(top10%真值) R² | top10 幅值比 | p99 比 | top10 IoU | Spearman | 归一化 nMAE | R² fit cb | 斜率 |")
    sep = "| --- |" + " ---: |" * 18
    print("### 物理空间（压力 Pa / 速度幅值 m/s；归一化为线性 z，R² 与物理相同）")
    print(hdr); print(sep)
    loaded = {}
    for label, run in RUNS.items():
        for ck in ("best", "last"):
            m = load(run, ck)
            if m is None:
                print(f"| {label}·{ck} | (missing) |"); continue
            loaded[(label, ck)] = m
            print(row(f"{label}·{ck}", m))
    print()
    for (label, ck), m in loaded.items():
        if "query_groups" in m and ck == "best":
            print(f"### {esc(label)}：壁面 / 内部分组（best）")
            print(hdr); print(sep)
            for g, gm in m["query_groups"].items():
                print(row(f"{g}（n={gm['n_points']}）", gm))
            print()
        if "vector" in m and ck == "best":
            v = m["vector"]
            print(f"### {esc(label)}：向量补充（best）")
            print("| 分量 | R² cb | R² pooled | RMSE (m/s) |"); print("| --- | ---: | ---: | ---: |")
            for c, cm in v["components"].items():
                print(f"| {c} | {f(cm['r2_casebalanced'])} | {f(cm['r2_pooled'])} | {f(cm['rmse'],4)} |")
            print(f"\n方向余弦（|u|>{v['speed_floor_m_s']} m/s，逐例均值）{f(v['direction_cosine_casemean'],3)}；速度加权 {f(v['direction_cosine_speedweighted_casemean'],3)}；"
                  f"分量 z 空间 pooled R² {f(v['component_normalized_r2_pooled'])}；向量 RMSE {f(v['vector_rmse_m_s'],4)} m/s\n")
        if ck == "best":
            print(f"### {esc(label)}：分域（best，病例等权 R² cb）")
            print("| " + " | ".join(m["group_casebalanced"].keys()) + " |")
            print("|" + " ---: |" * len(m["group_casebalanced"]))
            print("| " + " | ".join(f(g["r2"]) for g in m["group_casebalanced"].values()) + " |")
            print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
