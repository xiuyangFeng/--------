"""Evidence-qualified MS1–MS6 report using the frozen V6 metric conventions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from training_wss_min.tools.report_v6_followup import (
    METRIC_DEFINITIONS, PATHS, fmt, load_metrics, matrix_finalization,
    run_evidence, source_integrity, values,
)
from training_wss_min.tools.tracker_section import replace_or_append

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "training_wss_min/experiments/v6_multiradius_bt_20260909"
MATRIX = ROOT / "training_wss_min/configs/v6_multiradius_bt_20260909/matrix.json"
RUNS = ROOT / "training_wss_min/runs"
ANCHOR = "v6_followup_20260909/M2_a5_independent_k3_s1234"
TRACKER = ROOT / "docs/02-推进与变更/00-V5设计与历史跟踪/WSS_V5_训练实验跟踪_历史卷_2026-09-06至09-20.md"
LABELS = {
    "MS1": "单半径新SA3桥接与中心条件，无新增全局BT",
    "MS2": "MS1加单路全局BT",
    "MS3": "三半径独立集合编码与融合，无新增全局BT",
    "MS4": "三半径各配独立BT并融合；老师完整方案",
    "MS5": "三支均用中半径；分支数与参数量对照",
    "MS6": "三半径逐token近等参数FFN替代新增BT",
}
PARENTS = {"MS1": "MS0", "MS2": "MS1", "MS3": "MS1", "MS4": "MS3", "MS5": "MS4", "MS6": "MS4"}


def matrix_arms():
    matrix = json.loads(MATRIX.read_text())
    arms = []
    for arm in matrix["arms"]:
        config = json.loads((MATRIX.parent / arm["config"]).read_text())
        if arm.get("run_name", config["name"]) != config["name"]:
            raise ValueError(f"Manifest/config run-name mismatch: {arm['id']}")
        parent = arm.get("baseline", PARENTS[arm["id"]])
        if parent != PARENTS[arm["id"]]:
            raise ValueError(f"Unexpected direct parent for {arm['id']}: {parent}")
        arms.append({**arm, "run_name": config["name"], "hypothesis": arm.get("hypothesis", LABELS[arm["id"]]),
                     "parent": parent, "resolved_config": config})
    if [a["id"] for a in arms] != [f"MS{i}" for i in range(1, 7)]:
        raise ValueError("Report is restricted to the six authorized MS1–MS6 arms")
    return arms


def report(checkpoint):
    if checkpoint not in {"best", "last"}:
        raise ValueError("checkpoint must be best or last")
    arms = matrix_arms()
    base = load_metrics(ANCHOR, checkpoint)
    if base is None:
        raise FileNotFoundError(f"Historical MS0=M2 {checkpoint} evaluation missing")
    bv = values(base)
    state_path, submit_path = EXP / "queue_status.json", EXP / "submission.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    submission = json.loads(submit_path.read_text()) if submit_path.exists() else {}
    integrity = source_integrity(state)
    finalization = matrix_finalization(state, [a["id"] for a in arms], integrity)
    loaded = {a["id"]: run_evidence(a, checkpoint, state, submission, integrity) for a in arms}
    for _, (m, evidence) in loaded.items():
        if m is not None and set(m["per_case"]) != set(base["per_case"]):
            evidence["valid_scientific_result"] = False
            evidence["issues"].append("patient set differs from MS0")
            evidence["status"] = "评估病例不一致，结果不纳入比较"
    summary = {"checkpoint": checkpoint, "reference": {"id": "MS0", "run_name": ANCHOR, "seed": 1234, **bv},
               "source_integrity": integrity, "queue_status": state.get("status"),
               "matrix_finalized": finalization["finalized"], "matrix_finalization": finalization,
               "metric_definitions": METRIC_DEFINITIONS, "expected_count": len(arms),
               "checkpoint_selection": "best按训练总损失选模；last固定epoch399；每臂单seed1234，仅探索性比较",
               "arms": {}}
    lines = [f"# V6 多半径集合与分支 BT（{checkpoint}）", "",
             "仅预测峰值单帧WSS；原25D输入、原A5损失；seed1234、400epoch；参照为同seed历史MS0=M2。",
             "best按训练总损失选模；last固定epoch399。每一张表仅比较同一checkpoint。G/M6及新几何组合继续等待用户验收。", "",
             "| 臂 | 父臂 | 状态 | 物理 R²_cb | Δ vs MS0 | Δ vs 父臂 | log_z R²_cb | case mean | MAE Pa | high-WSS R² | top10 比 | p99 比 | IoU | Spearman |",
             "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    lines.append("| " + " | ".join(["MS0=M2", "—", "已有", fmt(bv["physical_r2_cb"]), "—", "—",
        fmt(bv["normalized_r2_cb"]), fmt(bv["case_mean"]), fmt(bv["mae"], 3), fmt(bv["high_wss_r2"]),
        fmt(bv["top10_ratio"], 3), fmt(bv["p99_ratio"], 3), fmt(bv["top10_iou"], 3), fmt(bv["spearman"], 3)]) + " |")
    for arm in arms:
        aid, parent = arm["id"], arm["parent"]
        m, evidence = loaded[aid]
        pm = base if parent == "MS0" else loaded[parent][0] if loaded[parent][1]["valid_scientific_result"] else None
        entry = {"run_name": arm["run_name"], "config": arm["config"], "hypothesis": arm["hypothesis"],
                 "parent": parent, "evidence": evidence, "queue": state.get("arms", {}).get(aid, {})}
        if m is not None:
            entry["raw_metrics"] = values(m)
            entry["raw_metrics_path"] = str(RUNS / arm["run_name"] / "eval" / f"ckpt_{checkpoint}" / "metrics.json")
        if not evidence["valid_scientific_result"]:
            lines.append(f"| {aid} | {parent} | {evidence['status']} |" + " — |" * 11)
        else:
            v = values(m)
            entry.update(v)
            entry["deltas_vs_ms0"] = {key: v[key] - bv[key] for key in v}
            entry["deltas_vs_parent"] = {key: v[key] - values(pm)[key] for key in v} if pm else None
            entry["delta_physical"] = entry["deltas_vs_ms0"]["physical_r2_cb"]
            entry["delta_normalized"] = entry["deltas_vs_ms0"]["normalized_r2_cb"]
            entry["delta_parent_physical"] = entry["deltas_vs_parent"]["physical_r2_cb"] if pm else None
            entry["delta_parent_normalized"] = entry["deltas_vs_parent"]["normalized_r2_cb"] if pm else None
            entry["per_case_delta"] = {c: m["per_case"][c]["overall"]["r2"] - base["per_case"][c]["overall"]["r2"] for c in base["per_case"]}
            entry["cases_improved"] = sum(d > 0 for d in entry["per_case_delta"].values())
            lines.append("| " + " | ".join([aid, parent, evidence["status"], fmt(v["physical_r2_cb"]), fmt(entry["delta_physical"]),
                fmt(entry["delta_parent_physical"]), fmt(v["normalized_r2_cb"]), fmt(v["case_mean"]), fmt(v["mae"], 3),
                fmt(v["high_wss_r2"]), fmt(v["top10_ratio"], 3), fmt(v["p99_ratio"], 3), fmt(v["top10_iou"], 3), fmt(v["spearman"], 3)]) + " |")
        summary["arms"][aid] = entry
    # The planned factorial comparison and mechanism controls use the same checkpoint.
    qualified = {k: v for k, v in summary["arms"].items() if v["evidence"]["valid_scientific_result"]}
    if {"MS1", "MS2", "MS3", "MS4"} <= set(qualified):
        summary["factorial_interaction"] = {key: (qualified["MS4"][key] - qualified["MS3"][key]) - (qualified["MS2"][key] - qualified["MS1"][key]) for key in PATHS}
    summary["planned_contrasts"] = {}
    for child, parent in (("MS4", "MS2"), ("MS4", "MS3"), ("MS4", "MS5"), ("MS4", "MS6")):
        if child in qualified and parent in qualified:
            summary["planned_contrasts"][f"{child}-{parent}"] = {key: qualified[child][key] - qualified[parent][key] for key in PATHS}
    summary["completed_count"] = len(qualified)
    summary["raw_metrics_count"] = sum("raw_metrics" in v for v in summary["arms"].values())
    if len(qualified) != len(arms):
        finalization['issues'].append(f'{checkpoint}: not all six histories/checkpoints/metrics qualify')
        finalization['finalized'] = False
        summary['matrix_finalized'] = False
    lines += ["", "预定比较与边界：", "",
              "- MS1承接新分组/中心条件的整体变化，不能当作单项条件MLP消融；MS2/3/4形成单/多尺度×有/无新增全局BT的2×2对照。",
              "- MS4同时比较MS0、MS2、MS3、同参数MS5和近容量MS6；MS6保留原SA2局部注意力，只替换新增跨中心BT。",
              "- 参数量、显存、耗时和实际邻居/分支激活诊断见预检与运行产物；单seed与已暴露test34不支持统计显著性或稳定泛化声明。", "",
              f"有效{checkpoint}评估：{len(qualified)}/{len(arms)}；整批完结归档：{'已核验' if finalization['finalized'] else '尚未核验'}。",
              f"源码核验：{integrity['basis']}；训练起止源码{'一致' if integrity['archived_hashes_match'] else '尚未核验或不一致'}。",
              "源文件变化、失败阶段或证据未齐的指标只保留在JSON的raw_metrics，不进入主表比较。", "",
              "指标口径：", ""] + [f"- {k}：{v}。" for k, v in METRIC_DEFINITIONS.items()]
    lines += ["", "工作簿本节只填best；文件名last沿用历史命名。总览AA/AB为log_z pooled MAE/RMSE；教师O为log_z p99比，汇总L为Pa p99比。", ""]
    EXP.mkdir(parents=True, exist_ok=True)
    markdown = "\n".join(lines)
    (EXP / f"matrix_tables_{checkpoint}.md").write_text(markdown)
    (EXP / f"matrix_summary_{checkpoint}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    return markdown, summary


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--update-tracker", action="store_true")
    args = ap.parse_args(argv)
    best, summary = report("best")
    _, last = report("last")
    print(f"MS best={summary['completed_count']}/6 last={last['completed_count']}/6")
    if args.update_tracker:
        heading = "### 14.2 执行与结果（best；由原始指标及队列证据自动回填）"
        section = heading + "\n\n" + best.split("\n", 1)[1]
        TRACKER.write_text(replace_or_append(TRACKER.read_text(), heading, section))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
