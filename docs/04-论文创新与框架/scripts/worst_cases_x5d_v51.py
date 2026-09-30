#!/usr/bin/env python3
"""只读脚本：从已保存的评估产物取 X5D_v51 底座的逐病例结果，产出"最差病例"取证表。

不训练、不推理。输出到 docs/04-论文创新与框架/导师汇报_最差病例取证_2026-09-21/：
  cv3_percase_X5D_v51.csv        136 例患者分组折外（波 2a 三折，seed 1234，ckpt_best）逐例指标 + SSE 份额
  test34_percase_X5D_v51_5seed.csv  test34 五 seed Pa 均值集成逐例指标（analysis_20260917/current_residuals.json）
  cv3_worst_best_geometry.csv    折外最差 14 + 高误差份额 3 + 最好 4 例的几何/真值特征（母库 bundle + flowref）
  subgroup_and_family_summary.json  临床亚组统计 + 三类失效族计数
在仓库根目录运行：python3 docs/04-论文创新与框架/scripts/worst_cases_x5d_v51.py
"""
import csv, json, statistics as st
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "docs/04-论文创新与框架/导师汇报_最差病例取证_2026-09-21"
OUT.mkdir(parents=True, exist_ok=True)
VIEW = ROOT / "data_wss_v5/views_v5_1/wss_min_view_v1"
FLOW = ROOT / "data_wss_v5/views_v5_1/wss_min_flowref_v1"
CV3 = ROOT / "training_wss_min/runs/wss_v51_wave2a_20260916"
RES = ROOT / "training_wss_min/experiments/wss_v51_wave1_20260916/analysis_20260917/current_residuals.json"


def subgroup(case: str) -> str:
    if case.startswith("ILO"):
        return "ILO/1 发生髂支闭塞" if "-1/" in case else "ILO/0 未发生髂支闭塞"
    return "/".join(case.split("/")[:2])


def cv3_rows():
    rows = []
    for fold in range(3):
        m = json.load(open(CV3 / f"X5D_v51_f{fold}_s1234/eval/ckpt_best/metrics.json"))["test"]["per_case"]
        for cid, r in m.items():
            o, c, d, h = r["overall"], r["calibration"], r["distribution"], r["hotspot"]
            rows.append(dict(fold=fold, case=cid, cohort=cid.split("/")[0], subgroup=subgroup(cid),
                             r2=o["r2"], mae=o["mae"], rmse=o["rmse"], n=o["n"], sse=o["rmse"] ** 2 * o["n"],
                             true_p50=d["true_p50"], true_p99=d["true_p99"], true_max=d["true_dynamic_range"],
                             top10_ratio=c["top10_pred_true_ratio"], p99_ratio=c["p99_pred_true_ratio"],
                             calib_slope=c["calibration_slope"], top10_iou=h["top10_iou"], spearman=h["spearman_all"],
                             r2_high_wss=r["high_wss"]["r2"], r2_stenosis=r.get("stenosis", {}).get("r2"),
                             r2_bifurcation=r.get("bifurcation", {}).get("r2")))
    tot = sum(r["sse"] for r in rows)
    for r in rows:
        r["sse_share"] = r["sse"] / tot
    rows.sort(key=lambda r: r["r2"])
    return rows


def geometry(case: str, r2: float):
    z = np.load(VIEW / case / "bundle.npz", allow_pickle=True)
    fr = json.load(open(FLOW / case / "flowref_report.json"))
    vr = json.load(open(VIEW / case / "view_report.json"))
    r, seg = z["wall_local_radius"], z["wall_segment_id"]
    peak = int(np.where(z["steps"] == int(z["peak_step"]))[0][0])
    w = z["wall_wss"][peak]
    leaves = [int(k) for k in fr["cap_radius_by_leaf"]]
    aorta, leaf = seg == 0, np.isin(seg, leaves)
    top = w >= np.percentile(w, 90)
    rin = fr["r_inlet_mm"]
    rmax = float(np.percentile(r[aorta], 99))
    rmin = float(np.percentile(r[leaf], 1))
    return dict(case=case, r2=r2, r_inlet_mm=rin, r_aorta_p99_mm=rmax, r_leaf_p01_mm=rmin,
                aorta_over_inlet=rmax / rin, leaf_min_over_inlet=rmin / rin, trunk_mm=vr["frame"]["trunk_length_mm"],
                wss_p50=float(np.median(w)), wss_p99=float(np.percentile(w, 99)), wss_max=float(w.max()),
                top10_share_aorta=float((top & aorta).sum() / top.sum()), top10_share_leaf=float((top & leaf).sum() / top.sum()))


def main():
    rows = cv3_rows()
    with open(OUT / "cv3_percase_X5D_v51.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)

    res = json.load(open(RES))
    t34 = sorted(res["test34_base5"]["per_case"], key=lambda r: r["r2_pa"])
    with open(OUT / "test34_percase_X5D_v51_5seed.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(t34[0])); wr.writeheader(); wr.writerows(t34)

    picks = [r["case"] for r in rows[:14]] + ["AAA/ruputer/DING_JUN_FENG", "AAA/ruputer/WANG_KUI_WU", "ILO/YANG_WANG_QI-1/before"] + [r["case"] for r in rows[-4:]]
    r2map = {r["case"]: r["r2"] for r in rows}
    geo = [geometry(c, r2map[c]) for c in picks]
    with open(OUT / "cv3_worst_best_geometry.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(geo[0])); wr.writeheader(); wr.writerows(geo)

    groups = {}
    for r in rows:
        groups.setdefault(r["subgroup"], []).append(r)
    fam_a = [r for r in rows if r["true_p99"] > 35 and r["top10_ratio"] < 0.7]
    fam_b = [r for r in rows if r["true_p99"] < 20 and r["mae"] < 1.5]
    oof = {r["unit"]: r for r in res["cv3_oof"]["per_case"]}
    fam_c = [u for u, r in oof.items() if r["branch_mean_share"] > 0.35]
    summary = {
        "source": {"cv3": str(CV3.relative_to(ROOT)), "residuals": str(RES.relative_to(ROOT))},
        "subgroups": {g: dict(n=len(v), mean_r2=st.mean(x["r2"] for x in v), median_r2=st.median(x["r2"] for x in v),
                              min_r2=min(x["r2"] for x in v), n_below_0p6=sum(x["r2"] < 0.6 for x in v)) for g, v in sorted(groups.items())},
        "families": {
            "A_jet_high_amplitude(p99>35Pa & top10_ratio<0.7)": dict(n=len(fam_a), mean_r2=st.mean(r["r2"] for r in fam_a), sse_share=sum(r["sse_share"] for r in fam_a), cases=[r["case"] for r in fam_a]),
            "B_low_wss_sac(p99<20Pa & MAE<1.5Pa)": dict(n=len(fam_b), mean_r2=st.mean(r["r2"] for r in fam_b), sse_share=sum(r["sse_share"] for r in fam_b)),
            "C_branch_offset(branch_mean_share>0.35)": dict(n=len(fam_c), cases=fam_c, median_branch_share_all=st.median(r["branch_mean_share"] for r in oof.values())),
        },
    }
    json.dump(summary, open(OUT / "subgroup_and_family_summary.json", "w"), ensure_ascii=False, indent=1)
    print("written", OUT)


if __name__ == "__main__":
    main()
