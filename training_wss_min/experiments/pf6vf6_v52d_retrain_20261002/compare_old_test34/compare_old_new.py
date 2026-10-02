"""Old (deployed v5.0 PF6/VF6, train138) vs new (v5.2d) on the same units and the same v5.2d labels (2026-10-02).

Two paired comparisons, metric definitions = training_wss_min/tools/report_pf6vf6_v52d_retrain.py:
  test34    the 34 v5.0 test units (never seen by the old models). Old = the deployed weights re-scored on v5.2d labels
            (job 16973 -> compare_old_test34/<model>/); new = the CV5 out-of-fold predictions of the same units
            (each unit predicted by the fold model that held it out; ~209 training units vs 138).
  recover8  8 units: old = E5 (e5_deployed_v50/*/eval), new = full265 runs and the 15 CV5 fold models (E3).
Writes compare_old_new.{md,json} next to this script.
"""
import json
from pathlib import Path

import numpy as np

import training_wss_min.tools.report_pf6vf6_v52d_retrain as R

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
RUNS = R.RUNS
SEEDS, FOLDS = R.SEEDS, R.FOLDS
KEY = {"PF6": "all", "VF6": "speed"}


def files_of(pred_root):
    f = R.manifest_units(pred_root)
    if f is None:
        raise FileNotFoundError(pred_root)
    return f


def summarize_set(target, members):
    """members: list of {unit: file}; returns per-member summaries, ensemble summary, ensemble per-unit records."""
    units = sorted(set.intersection(*(set(m) for m in members)))
    singles = [R.summarize(target, {u: R.member_record(target, [m[u]]) for u in units}) for m in members]
    ens_recs = {u: R.member_record(target, [m[u] for m in members]) for u in units}
    return singles, R.summarize(target, ens_recs), ens_recs


def paired(target, new_recs, old_recs):
    k = KEY[target]
    d = {u: new_recs[u][k]["r2_unit"] - old_recs[u][k]["r2_unit"] for u in new_recs}
    vals = np.array(list(d.values()))
    return {"n": len(vals), "better": int((vals > 0).sum()), "median_delta_unit_r2": float(np.median(vals)),
            "mean_delta_unit_r2": float(np.mean(vals)), "per_unit": d}


def row(label, s, target):
    if target == "PF6":
        return f"| {label} | {s['n_units']} | {s['r2cb_all']:.4f} | {s['r2cb_wall']:.4f} | {s['r2cb_interior']:.4f} | {s['unit_median_r2']:.4f} | {s['mae_pa']:.1f} |"
    return (f"| {label} | {s['n_units']} | {s['r2cb_speed']:.4f} | {s['unit_median_r2']:.4f} | {s['vector_rmse_m_s']:.4f} | "
            f"{s['direction_cosine_casemean']:.4f} |")


HEAD = {"PF6": "| 模型 | 单元数 | 压力 R²_cb | 壁面 | 内部 | 逐例中位 R² | MAE (Pa) |\n|---|---:|---:|---:|---:|---:|---:|",
        "VF6": "| 模型 | 单元数 | 速率 R²_cb | 逐例中位 R² | 向量 RMSE (m/s) | 方向余弦 |\n|---|---:|---:|---:|---:|---:|"}


def main():
    out, lines = {}, ["# 新旧体场模型精度对比（同一批单元、同一套 v5.2d 标签）", "",
                      "旧 = 已部署 `PF6_VF6_peak_3seed_20260920`（v5.0 数据，train138）；新 = v5.2d 重训（§43）。"
                      "指标定义同 `tools/report_pf6vf6_v52d_retrain.py`。逐例配对差 = 新 − 旧的逐例 R²（三 seed 集成）。", ""]
    split34 = json.loads((HERE / "split_test34_on_v52d.json").read_text())["test_cases"]
    for target in ("PF6", "VF6"):
        name = "压力 PF6" if target == "PF6" else "速度 VF6"
        res = {}
        # ---- test34: new CV5 OOF vs old
        new_members = []
        for s in SEEDS:
            m = {}
            for k in FOLDS:
                f = files_of(R.pred_root(RUNS / f"{target}_v52cv_f{k}_s{s}", "best"))
                m.update({u: p for u, p in f.items() if u in split34})
            new_members.append(m)
        old_members = [files_of(HERE / f"{target}_s{s}" / "predictions/test") for s in SEEDS]
        ns, ne, nrec = summarize_set(target, new_members)
        os_, oe, orec = summarize_set(target, old_members)
        assert ne["n_units"] == oe["n_units"] == 34
        pr = paired(target, nrec, orec)
        k = "r2cb_all" if target == "PF6" else "r2cb_speed"
        lines += [f"## {name}", "", f"### test34（34 例，旧模型没见过；新模型用 CV5 折外预测）", "", HEAD[target]]
        for s, x in zip(SEEDS, os_):
            lines.append(row(f"旧 v5.0 s{s}", x, target))
        lines.append(row("**旧 v5.0 三 seed 集成**", oe, target))
        for s, x in zip(SEEDS, ns):
            lines.append(row(f"新 v5.2d CV5 折外 s{s}", x, target))
        lines.append(row("**新 v5.2d CV5 折外三 seed 集成**", ne, target))
        lines += ["", f"单 seed 均值：旧 {np.mean([x[k] for x in os_]):.4f} → 新 {np.mean([x[k] for x in ns]):.4f}（{np.mean([x[k] for x in ns]) - np.mean([x[k] for x in os_]):+.4f}）；"
                      f"集成：旧 {oe[k]:.4f} → 新 {ne[k]:.4f}（{ne[k] - oe[k]:+.4f}）；逐例 {pr['better']}/{pr['n']} 例改善，逐例 R² 中位差 {pr['median_delta_unit_r2']:+.4f}。", ""]
        coh = "分队列（集成）：" + "；".join(
            f"{c} {oe['cohorts'][c]['n_units']} 例 {oe['cohorts'][c]['r2cb']:.3f} → {ne['cohorts'][c]['r2cb']:.3f}" for c in R.COHORTS if c in ne["cohorts"])
        worst = sorted(pr["per_unit"].items(), key=lambda kv: kv[1])
        lines += [coh, "", "变化最大的单元（逐例 R²，旧 → 新）：" + "；".join(
            f"{u} {orec[u][KEY[target]]['r2_unit']:.3f} → {nrec[u][KEY[target]]['r2_unit']:.3f}" for u, _ in worst[:3] + worst[-3:]), ""]
        res["test34"] = {"old_singles": os_, "old_ens": oe, "new_singles": ns, "new_ens": ne, "paired": pr}
        # ---- recover8: full265 vs E5 (and E3)
        old8 = [files_of(EXP / "e5_deployed_v50" / f"{target}_s{s}" / "eval/predictions/test") for s in SEEDS]
        new8 = [files_of(R.pred_root(RUNS / f"{target}_full265_s{s}", "best")) for s in SEEDS]
        e3 = [files_of(EXP / "recover8_cv5" / f"{target}_v52cv_f{k}_s{s}" / "predictions/test") for s in SEEDS for k in FOLDS]
        os8, oe8, orec8 = summarize_set(target, old8)
        ns8, ne8, nrec8 = summarize_set(target, new8)
        _, ee3, erec3 = summarize_set(target, e3)
        pr8, pr8e3 = paired(target, nrec8, orec8), paired(target, erec3, orec8)
        lines += ["### recover8（8 例，只作描述）", "", HEAD[target], row("**旧 v5.0 三 seed 集成**", oe8, target),
                  row("**新 full265 三 seed 集成**", ne8, target), row("**新 CV5 15 折模型集成**", ee3, target), "",
                  f"单 seed 均值：旧 {np.mean([x[k] for x in os8]):.4f} → 新 full265 {np.mean([x[k] for x in ns8]):.4f}；"
                  f"集成 {oe8[k]:.4f} → {ne8[k]:.4f}（{ne8[k] - oe8[k]:+.4f}），逐例 {pr8['better']}/8 改善；15 折模型集成对旧 {ee3[k] - oe8[k]:+.4f}（{pr8e3['better']}/8）。", ""]
        res["recover8"] = {"old_ens": oe8, "new_full265_ens": ne8, "new_e3_ens": ee3, "paired_full265": pr8, "paired_e3": pr8e3}
        out[target] = res
    lines += ["## 读法", "",
              "- 两个集合都是同一批单元、同一套 v5.2d 标签上的配对比较。",
              "- 新旧差值混着三件事：训练数据量（test34 上新模型每折约 209 例、full265 是 265 例，旧模型 138 例）、v5.0 之后的数据修正（26 例 RCR 面积、91 个回收单元、9 例协议重算、v5.2d 合并），以及旧模型的训练标签里还带着当时的错误。不单独归因。",
              "- test34 是旧模型的开发暴露集（结构选择时看过），对旧模型略偏乐观；新模型这边是折外预测。recover8 只有 8 例。", ""]
    (HERE / "compare_old_new.md").write_text("\n".join(lines) + "\n")
    (HERE / "compare_old_new.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
