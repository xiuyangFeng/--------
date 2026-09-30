"""Summarize existing joint-cycle metrics; never train or run inference."""
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
RUNS = ROOT / "training_wss_min/runs/joint_cycle_v52_20260929"
ARMS = ["Iu", "Ip", "Iw", "J0", "J1c", "J1"]
PHASES = ["peak", "trough", "cycle", "accel", "decel", "plateau"]
TASK_NAMES = {"velocity": "速度向量", "pressure": "相对压力", "wss": "标量 WSS"}


def main():
    rows, sources, paired, histories = [], {}, [], {}
    records = {}
    common_ids = None
    for arm in ARMS:
        run = RUNS / f"{arm}_f0_s1234"
        d = json.loads((run / "metrics.json").read_text())
        per = [json.loads(s) for s in (run / "per_case.jsonl").read_text().splitlines()]
        h = [json.loads(s) for s in (run / "history.jsonl").read_text().splitlines()]
        assert d["status"] == "complete" and d["n_units"] == 55
        assert d["selection"] == "last" and d["phase_count"] == 80
        assert [r["epoch"] for r in h] == list(range(1, 151))
        ids = {r["unit_id"] for r in per}
        assert len(ids) == len(per) == 55
        if common_ids is None:
            common_ids = ids
        assert common_ids == ids
        records[arm] = {r["unit_id"]: r["tasks"] for r in per}
        sources[arm] = str(run / "metrics.json")
        histories[arm] = {}
        for task, data in d["tasks"].items():
            old = sum(r["losses"][task] for r in h[110:130]) / 20
            new = sum(r["losses"][task] for r in h[130:150]) / 20
            histories[arm][task] = {"epochs111_130": old, "epochs131_150": new,
                                    "relative_change": new / old - 1}
            for phase in PHASES:
                row = {"arm": arm, "task": task, "phase": phase,
                       **{k: data[phase][k] for k in ["r2", "mae", "rmse", "relative_l2"]},
                       "direction_cosine": data[phase].get("direction_cosine"),
                       "direction_weight_coverage": data[phase].get("direction_weight_coverage"),
                       "spatial_spearman": data[phase].get("spatial_spearman_unweighted")}
                for key in ["r2", "mae", "rmse", "relative_l2"]:
                    vals = [r["tasks"][task][phase][key] for r in per]
                    assert all(math.isfinite(v) for v in vals)
                    assert math.isclose(sum(vals) / 55, row[key], rel_tol=1e-10, abs_tol=1e-10)
                rows.append(row)
    for task, base in [("velocity", "Iu"), ("pressure", "Ip"), ("wss", "Iw")]:
        for phase in PHASES[:3]:
            a = [records["J1"][i][task][phase] for i in sorted(common_ids)]
            b = [records[base][i][task][phase] for i in sorted(common_ids)]
            paired.append({"comparison": f"J1-{base}", "task": task, "phase": phase,
                           "delta_r2": sum(x["r2"]-y["r2"] for x, y in zip(a,b)) / 55,
                           "relative_mae_change": sum(x["mae"] for x in a) / sum(y["mae"] for y in b) - 1,
                           "r2_improved": sum(x["r2"] > y["r2"] for x,y in zip(a,b)),
                           "mae_improved": sum(x["mae"] < y["mae"] for x,y in zip(a,b)), "n": 55})
    payload = {"source": sources, "aggregation": "equal unit mean; within-unit volume/area weighted",
               "phase_ranges_zero_based_right_exclusive": {
                   "peak": [[17,27]], "trough": [[5,10],[43,58]], "cycle": [[0,80]],
                   "accel": [[10,17]], "decel": [[27,43]], "plateau": [[0,5],[58,80]]},
               "rows": rows, "paired": paired, "training_loss_windows": histories}
    (OUT / "phase_metrics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    with (OUT / "phase_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# 首轮峰值窗、谷底窗与整周期精度核对", "",
             "仅读取既有 metrics/per_case/history；未新增模型推理。seed1234、fold0、55个相同留出单元、last150。",
             "", "峰值窗为0-based帧17–26，谷底为5–9与43–57，整周期0–79。不是单帧21，也不是逐病例自身极值。",
             "整段R²包含该段的时空变异，不是逐帧R²的算术均值。速度MAE为分量MAE（m/s），压力/WSS为Pa。", "",
             "| 任务 | 模型 | 峰值窗 R² / MAE | 谷底窗 R² / MAE | 整周期 R² / MAE |",
             "|---|---|---:|---:|---:|"]
    for task, base in [("velocity", "Iu"), ("pressure", "Ip"), ("wss", "Iw")]:
        for arm in [base, "J0", "J1c", "J1"]:
            cells = []
            for phase in PHASES[:3]:
                r = next(r for r in rows if (r["arm"],r["task"],r["phase"]) == (arm,task,phase))
                cells.append(f'{r["r2"]:.4f} / {r["mae"]:.5f}')
            lines.append(f'| {TASK_NAMES[task]} | {arm} | '+" | ".join(cells)+" |")
    lines += ["", "## J1 相对独立模型的同单元配对变化", "",
              "| 任务 | 阶段 | ΔR² | MAE变化 | R²改善单元 | MAE改善单元 |",
              "|---|---|---:|---:|---:|---:|"]
    for r in paired:
        lines.append(f'| {TASK_NAMES[r["task"]]} | {r["phase"]} | {r["delta_r2"]:+.4f} | {r["relative_mae_change"]:+.2%} | {r["r2_improved"]}/55 | {r["mae_improved"]}/55 |')
    lines += ["", "完整六段R²/MAE/RMSE/relative-L2、速度方向覆盖、WSS空间Spearman、训练末段loss与来源路径见 [JSON](phase_metrics.json) / [CSV](phase_metrics.csv)。",
              "本表的55单元并非55个独立患者；仅单折单seed开发比较，不提供显著性推断。", ""]
    (OUT / "phase_report.md").write_text("\n".join(lines))
    print(f"Verified {len(rows)} task-phase rows across six runs; wrote {OUT}")


if __name__ == "__main__":
    main()
