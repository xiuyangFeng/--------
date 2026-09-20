"""Workbook-style result tables for the V5 rerun (physical space + normalized space + linear fit).

Reads ``eval/ckpt_<best|last>/metrics.json`` of the new runs and the legacy
checkpoints (full test36 and the 34-case overlap re-evaluation) and prints
markdown tables in the layout of 《WSS_PointNet实验矩阵与结果汇总last.xlsx》.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "training_wss_min/runs"
EXP = ROOT / "training_wss_min/experiments/v5_rerun_20260906"

NEW = {
    "R1 L-SA2+H2+logR": "v5_rerun_20260906/outputs/r1_lsa2_h2_logradius_s1234",
    "R1 L-SA2+H2+logR s7": "v5_rerun_20260906/outputs/r1_lsa2_h2_logradius_s7",
    "R1 L-SA2+H2+logR s2025": "v5_rerun_20260906/outputs/r1_lsa2_h2_logradius_s2025",
    "R2 L-SA2 MSE": "v5_rerun_20260906/outputs/r2_lsa2_mse_s1234",
    "R3 S3-GEOPE MSE": "v5_rerun_20260906/outputs/r3_s3_geope_mse_s1234",
    "R3 S3-GEOPE MSE s7": "v5_rerun_20260906/outputs/r3_s3_geope_mse_s7",
    "R3 S3-GEOPE MSE s2025": "v5_rerun_20260906/outputs/r3_s3_geope_mse_s2025",
    "R4 R1+V5feat": "v5_rerun_20260906/outputs/r4_lsa2_h2_logradius_v5feat_s1234",
    "R4 R1+V5feat s7": "v5_rerun_20260906/outputs/r4_lsa2_h2_logradius_v5feat_s7",
    "R4 R1+V5feat s2025": "v5_rerun_20260906/outputs/r4_lsa2_h2_logradius_v5feat_s2025",
    "R4-abl no tube(rho,theta)": "v5_rerun_20260906/outputs/r4abl_notube_s1234",
    "R4-abl no dist/end_zone": "v5_rerun_20260906/outputs/r4abl_nodist_s1234",
    "R4-abl no dr_ds": "v5_rerun_20260906/outputs/r4abl_nodrds_s1234",
    "R4-abl no PCA normal": "v5_rerun_20260906/outputs/r4abl_nonormal_s1234",
    "R4-tail q95 l0.2": "v5_rerun_20260906/outputs/r4tail_q95_s1234",
    "R4-tail q90 l0.5": "v5_rerun_20260906/outputs/r4tail_q90l05_s1234",
    "R4-NLL head (mu only)": "v5_rerun_20260906/outputs/r4nll_s1234",
}
SEED_GROUPS = {
    "R1 L-SA2+H2+logR": ["R1 L-SA2+H2+logR", "R1 L-SA2+H2+logR s7", "R1 L-SA2+H2+logR s2025"],
    "R3 S3-GEOPE MSE": ["R3 S3-GEOPE MSE", "R3 S3-GEOPE MSE s7", "R3 S3-GEOPE MSE s2025"],
    "R4 R1+V5feat": ["R4 R1+V5feat", "R4 R1+V5feat s7", "R4 R1+V5feat s2025"],
}
LEGACY = {
    "R1 L-SA2+H2+logR": ("pointnetpp_lsa2_h2_logradius/outputs/lsa2_h2_logradius_s1234", "r1"),
    "R2 L-SA2 MSE": ("pointnetpp_regp10_transformer/outputs/localtf_sa2_s1234", "r2"),
    "R3 S3-GEOPE MSE": ("pointnetpp_d2_k64_ilo_structure/outputs/s3_d2k64_pnxr_geope_mixed", "r3"),
    "R3 S3-GEOPE MSE s7": ("pointnetpp_d2_k64_ilo_structure/outputs/s3_d2k64_pnxr_geope_mixed_s7", "r3s7"),
    "R3 S3-GEOPE MSE s2025": ("pointnetpp_d2_k64_ilo_structure/outputs/s3_d2k64_pnxr_geope_mixed_s2025", "r3s2025"),
}


def load(path: Path):
    if not path.is_file():
        return None
    m = json.loads(path.read_text(encoding="utf-8"))
    return m.get("test", m)


def f(x, spec=".4f"):
    if x is None:
        return "—"
    try:
        return format(float(x), spec)
    except (TypeError, ValueError):
        return str(x)


def row_physical(label, m):
    if m is None:
        return f"| {label} | — |" + " — |" * 14
    agg, fld, cb, cal, hot, reg = m["aggregate"], m["field"], m["field_casebalanced"], m["calibration"], m["hotspot"], m["regional_field"]
    return "| " + " | ".join([
        label, f(cb["r2"]), f(fld["r2"]), f(agg["r2_casemean"]), f(agg["r2_casemed"]), f(agg["r2_casep10"]), str(agg.get("r2_negative_cases", "—")),
        f(fld["rmse"], ".3f"), f(fld["mae"], ".3f"), f(100 * fld["nmae_range"], ".2f"), f(100 * fld["nrmse_range"], ".2f"),
        f(reg.get("high_wss", {}).get("r2")), f(cal["top10_pred_true_ratio"]), f(hot["top10_iou_casemean"]), f(cal["p99_pred_true_ratio"]),
    ]) + " |"


def row_normalized(label, m):
    if m is None:
        return f"| {label} | — |" + " — |" * 9
    n = m.get("normalized", m)
    agg, fld, cb, hot, reg = n["aggregate"], n["field"], n["field_casebalanced"], n["hotspot"], n["regional_field"]
    return "| " + " | ".join([
        label, f(cb["r2"]), f(fld["r2"]), f(agg["r2_casemean"]), f(agg["r2_casemed"]), f(agg["r2_casep10"]),
        f(100 * fld["nrmse_range"], ".2f"), f(fld["mae"], ".4f"), f(reg.get("high_wss", {}).get("r2")), f(hot["spearman_all_casemean"]),
    ]) + " |"


def row_fit(label, m):
    if m is None:
        return f"| {label} | — |" + " — |" * 5
    fld, cb = m["field"], m["field_casebalanced"]
    n = m.get("normalized", m)
    return "| " + " | ".join([label, f(cb.get("r2_linear_fit")), f(fld.get("r2_linear_fit")), f(fld.get("linear_fit_slope")), f(fld.get("linear_fit_intercept"), ".3f"),
                              f(n["field_casebalanced"].get("r2_linear_fit")), f(n["field"].get("r2_linear_fit"))]) + " |"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", default=["best", "last"])
    args = parser.parse_args(argv)
    entries = []
    for label, (rd, key) in LEGACY.items():
        entries.append((f"旧·{label}·test36", load(RUNS / rd / "eval/ckpt_best/metrics.json")))
        entries.append((f"旧·{label}·交集34", load(EXP / "legacy_overlap34" / key / "metrics.json")))
    for label, rd in NEW.items():
        for ck in args.checkpoints:
            entries.append((f"新·{label}·test34·{ck}", load(RUNS / rd / f"eval/ckpt_{ck}/metrics.json")))
    print("### 物理空间（Pa）\n")
    print("| 实验 | R² cb | R² raw | case mean | case med | case P10 | 负例 | RMSE pooled | MAE | NMAE pooled % | range-NRMSE pooled % | high-WSS R² | top10 幅值比 | top10 IoU | p99 比 |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for label, m in entries:
        print(row_physical(label, m))
    print("\n### 归一化空间（log_z）\n")
    print("| 实验 | R² cb | R² raw | case mean | case med | case P10 | range-NRMSE % | MAE | high-WSS R² | Spearman |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for label, m in entries:
        print(row_normalized(label, m))
    print("\n### 多 seed 汇总（物理 R² cb / case mean；均值 ± 极差半宽；新=best ckpt，旧=交集34）\n")
    print("| 组 | 旧 3 seed R² cb | 新 3 seed R² cb | 旧 case mean | 新 case mean | 新 top10 IoU | 新 high-WSS R² |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    import statistics as st
    def ms(vals):
        vals = [v for v in vals if v is not None]
        if not vals:
            return "—"
        return f"{st.mean(vals):.4f} ± {(max(vals) - min(vals)) / 2:.4f}" if len(vals) > 1 else f"{vals[0]:.4f}"
    for group, labels in SEED_GROUPS.items():
        new_m = [load(RUNS / NEW[l] / "eval/ckpt_best/metrics.json") for l in labels]
        legacy_keys = [k for k in LEGACY if k == group or k.startswith(group + " s")]
        old_m = [load(EXP / "legacy_overlap34" / LEGACY[k][1] / "metrics.json") for k in legacy_keys]
        print("| " + " | ".join([group,
            ms([m["field_casebalanced"]["r2"] for m in old_m if m]), ms([m["field_casebalanced"]["r2"] for m in new_m if m]),
            ms([m["aggregate"]["r2_casemean"] for m in old_m if m]), ms([m["aggregate"]["r2_casemean"] for m in new_m if m]),
            ms([m["hotspot"]["top10_iou_casemean"] for m in new_m if m]), ms([m["regional_field"]["high_wss"]["r2"] for m in new_m if m])]) + " |")
    print("\n### 线性拟合补充\n")
    print("| 实验 | R² fit cb（物理） | R² fit pooled（物理） | fit a pooled | fit b pooled (Pa) | R² fit cb（归一化） | R² fit pooled（归一化） |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for label, m in entries:
        print(row_fit(label, m))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
