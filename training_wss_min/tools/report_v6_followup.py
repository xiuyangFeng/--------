"""Read actual run artifacts and report the 15 single-seed A5 follow-up arms."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from training_wss_min.tools.tracker_section import replace_or_append

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "training_wss_min/experiments/v6_followup_20260909"
MATRIX = ROOT / "training_wss_min/configs/v6_followup_20260909/matrix.json"
RUNS = ROOT / "training_wss_min/runs"
ANCHOR = "v6_singleframe_20260909/A5_r4_localbranch_diffgeom_s1234"
TRACKER = ROOT / "docs/02-推进与变更/WSS_PINN/WSS_V5_训练实验跟踪.md"
PATHS = {
    "physical_r2_cb": ("field_casebalanced", "r2"),
    "normalized_r2_cb": ("normalized", "field_casebalanced", "r2"),
    "case_mean": ("aggregate", "r2_casemean"), "case_p10": ("aggregate", "r2_casep10"),
    "mae": ("field", "mae"), "rmse": ("field", "rmse"),
    "high_wss_r2": ("regional_field", "high_wss", "r2"),
    "top10_ratio": ("calibration", "top10_pred_true_ratio"),
    "p99_ratio": ("calibration", "p99_pred_true_ratio"),
    "top10_iou": ("hotspot", "top10_iou_casemean"),
    "spearman": ("hotspot", "spearman_all_casemean"),
    "linear_fit_r2_cb": ("field_casebalanced", "r2_linear_fit"),
    "linear_fit_slope_cb": ("field_casebalanced", "linear_fit_slope"),
    "linear_fit_intercept_cb": ("field_casebalanced", "linear_fit_intercept"),
}
METRIC_DEFINITIONS = {
    "physical_r2_cb": "Pa空间病例等权的共享均值R²；不是逐病例R²平均，也不是表面积加权",
    "normalized_r2_cb": "log_z空间病例等权R²；仍同时受幅值和空间分布误差影响",
    "case_mean": "逐病例物理R²的算术平均",
    "mae": "全部病例壁面顶点pool后的Pa MAE；病例点数影响权重",
    "high_wss_r2": "每个病例按自身真值q90选区，再将这些区域的顶点pool计算物理R²",
    "top10_ratio": "全部病例顶点pool后，以全局真值q90选区；该区pred均值/true均值",
    "p99_ratio": "全部病例顶点pool后，预测p99/真值p99；两者分位数独立计算，不检验热点位置",
    "top10_iou": "每病例分别按真值和预测各自q90选区求IoU，再对病例等权平均",
    "spearman": "每病例全壁面排名相关再平均；沿用冻结实现的argsort排名，并列值未使用平均秩",
    "linear_fit_cb": "病例等权拟合同一条pred=a*true+b直线；R²_fit=Pearson相关平方，须同时读取a和b；不是校准后的测试准确率",
}


def dig(m, path):
    for p in path:
        m = m[p]
    value = float(m)
    if not math.isfinite(value):
        raise ValueError(f"Nonfinite result at {path}")
    return value


def load_metrics(run_name, checkpoint):
    p = RUNS / run_name / "eval" / f"ckpt_{checkpoint}" / "metrics.json"
    if not p.exists():
        return None
    payload = json.loads(p.read_text())
    m = payload["test"]
    if m["aggregate"]["n_cases"] != 34 or len(m["per_case"]) != 34:
        raise ValueError(f"Incomplete test34 evaluation: {p}")
    return m


def values(m):
    return {k: dig(m, p) for k, p in PATHS.items()}


def fmt(v, n=4):
    return "—" if v is None else f"{v:.{n}f}"


def source_integrity(state):
    """Use the runner's archived hashes after exit; expose later drift separately."""
    start = state.get("source_sha256", {})
    end = state.get("source_sha256_end", {})
    current_drift = []
    for relative, expected in start.items():
        path = ROOT / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            current_drift.append(relative)
    archived = bool(start and end and state.get("ended_at")
                    and state.get("status") in {"complete", "incomplete"})
    archived_changes = sorted(k for k in set(start) | set(end) if start.get(k) != end.get(k)) if start and end else []
    changed = archived_changes if archived else sorted(set(current_drift) | set(archived_changes))
    return {
        "changed": bool(state.get("source_changed")) or bool(changed),
        "changed_files": changed,
        "basis": "archived_start_end" if archived else "live_start" if start else "unavailable",
        "archived": archived, "start_hash_available": bool(start), "end_hash_available": bool(end),
        "archived_hashes_match": bool(start and end and start == end),
        "current_drift": bool(current_drift), "current_drift_files": current_drift,
    }


def matrix_finalization(state, expected_ids, integrity):
    """A usable arm is not proof that the runner finalized the whole matrix."""
    issues = []
    if state.get("status") != "complete":
        issues.append("queue status is not complete")
    if not state.get("ended_at"):
        issues.append("queue end timestamp missing")
    if not integrity["archived_hashes_match"]:
        issues.append("matching start/end source hashes not verified")
    if integrity["changed"]:
        issues.append("source_changed")
    arms = state.get("arms", {})
    if set(arms) != set(expected_ids):
        issues.append("queue arm set differs from matrix")
    for arm_id in expected_ids:
        arm = arms.get(arm_id, {})
        if arm.get("status") != "complete":
            issues.append(f"{arm_id} is not complete")
        for stage in ("train", "eval_best", "eval_last"):
            if arm.get("stages", {}).get(stage, {}).get("returncode") != 0:
                issues.append(f"{arm_id}/{stage} has not completed successfully")
    return {"finalized": not issues, "issues": issues}


def run_evidence(arm, checkpoint, state=None, submission=None, integrity=None):
    state, submission = state or {}, submission or {}
    integrity = integrity if integrity is not None else source_integrity(state)
    queue = state.get("arms", {}).get(arm["id"], {})
    directory = RUNS / arm["run_name"]
    hp = directory / "history.jsonl"
    history, history_issue = [], None
    if hp.exists():
        for line in hp.read_text().splitlines():
            if not line.strip():
                continue
            try:
                history.append(json.loads(line))
            except json.JSONDecodeError:
                # A monitor may read while train.py is appending its final row.
                history_issue = "history contains an incomplete or malformed row"
                break
    trained = history_issue is None and [h.get("epoch") for h in history] == list(range(400))
    metric = load_metrics(arm["run_name"], checkpoint)
    issues = []
    if integrity["changed"]:
        issues.append("source_changed")
    if history_issue:
        issues.append(history_issue)
    stages = queue.get("stages", {})
    eval_stage = f"eval_{checkpoint}"
    if metric is not None:
        if not trained:
            issues.append("400 complete epochs not verified")
        if not (directory / f"ckpt_{checkpoint}.pt").exists():
            issues.append("checkpoint missing")
        if queue.get("run_name") != arm["run_name"]:
            issues.append("queue provenance missing or run name differs")
        for stage in ("train", eval_stage):
            if stages.get(stage, {}).get("returncode") != 0:
                issues.append(f"{stage} has not completed successfully")
    valid = metric is not None and not issues
    qstatus = queue.get("status")
    failed_stage = queue.get("failed_stage")
    if integrity["changed"]:
        status = "源文件变化，结果不纳入比较"
    elif qstatus in {"failed", "interrupted"}:
        status = (f"失败（{failed_stage or '阶段未知'}）" if qstatus == "failed" else "已中断")
        if valid:
            status += f"；{checkpoint}评估可用"
    elif valid:
        status = "已评估"
    elif qstatus == "pending":
        status = "队列待运行"
    elif qstatus == "train":
        status = "训练中"
    elif qstatus in {"eval_best", "eval_last"}:
        status = f"评估中（{qstatus.removeprefix('eval_')}）"
    elif metric is not None:
        status = "原始指标已保存，证据待核验"
    elif trained:
        status = "待评估"
    elif history:
        status = "已有部分训练记录，运行状态未确认"
    elif submission.get("formal_job_id") and not state:
        status = "作业已提交，等待队列启动（未查询调度状态）"
    else:
        status = "未开始"
    return metric, {"history_rows": len(history), "last_epoch": history[-1]["epoch"] if history else None,
                    "training_400_epochs": trained,
                    "valid_scientific_result": valid, "issues": issues,
                    "source_changed": integrity["changed"], "status": status,
                    "queue_status": qstatus, "failed_stage": failed_stage}


def report(checkpoint):
    matrix = json.loads(MATRIX.read_text())
    base = load_metrics(ANCHOR, checkpoint)
    if base is None:
        raise FileNotFoundError("Historical A5 evaluation missing")
    bv = values(base)
    state_path = EXP / "queue_status.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    submission_path = EXP / "submission.json"
    submission = json.loads(submission_path.read_text()) if submission_path.exists() else {}
    integrity = source_integrity(state)
    finalization = matrix_finalization(state, [a["id"] for a in matrix["arms"]], integrity)
    loaded = {a["id"]: run_evidence(a, checkpoint, state, submission, integrity) for a in matrix["arms"]}
    for aid, (m, evidence) in loaded.items():
        if m is not None and set(m["per_case"]) != set(base["per_case"]):
            evidence["valid_scientific_result"] = False
            evidence["issues"].append("patient set differs from anchor")
            evidence["status"] = "评估病例不一致，结果不纳入比较"
    summary = {"checkpoint": checkpoint, "reference": {"run_name": ANCHOR, "seed": 1234, "checkpoint": checkpoint, **bv},
               "source_integrity": integrity, "queue_status": state.get("status"),
               "matrix_finalized": finalization["finalized"], "matrix_finalization": finalization,
               "metric_definitions": METRIC_DEFINITIONS,
               "checkpoint_selection": "best按各臂自身训练总损失最低选择；last固定第400轮（epoch399）；损失变化也会改变best选模准则，须对照last",
               "arms": {}}
    lines = [f"# V6-A5 后续 D/L/M1–M5（{checkpoint}）", "",
             "全部单 seed=1234；仅 WSS；参照为同 seed 的 A5。Δ仅描述本次结果，不作显著性或稳定性结论。",
             "本表只比较同一checkpoint；best按各臂自身训练总损失最低选择，last固定第400轮（epoch399）。L臂的损失变化也会改变best选模准则，须结合last判断。",
             "M2→M3只改query邻居数，编码器FP不变；M3→M4评价整套可学解码方案。D4保留由法向派生的曲率和t·n。", "",
             "| 臂 | 父臂 | 状态 | 物理 R²_cb | Δ vs A5 | Δ vs 父臂 | log_z R²_cb | case mean | MAE Pa | high-WSS R² | top10 比 | p99 比 | IoU | Spearman |",
             "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    lines.append("| " + " | ".join(["B0=A5", "—", "已有", fmt(bv["physical_r2_cb"]), "—", "—", fmt(bv["normalized_r2_cb"]),
        fmt(bv["case_mean"]), fmt(bv["mae"], 3), fmt(bv["high_wss_r2"]), fmt(bv["top10_ratio"], 3), fmt(bv["p99_ratio"], 3), fmt(bv["top10_iou"], 3), fmt(bv["spearman"], 3)]) + " |")
    diagnostics = ["", "病例等权线性校准诊断（物理Pa空间，pred = a × true + b）：", "",
                   "| 臂 | R²_fit_cb | a | b (Pa) |", "| --- | ---: | ---: | ---: |",
                   f"| B0=A5 | {fmt(bv['linear_fit_r2_cb'])} | {fmt(bv['linear_fit_slope_cb'])} | {fmt(bv['linear_fit_intercept_cb'])} |"]
    for a in matrix["arms"]:
        m, evidence = loaded[a["id"]]
        parent = a.get("baseline", a.get("parent", "B0"))
        if parent in {"A5", "B0", "B0=A5"}:
            pm = base
        else:
            pm = loaded[parent][0] if loaded[parent][1]["valid_scientific_result"] else None
        entry = {"run_name": a["run_name"], "config": a["config"], "hypothesis": a["hypothesis"], "parent": parent,
                 "evidence": evidence, "queue": state.get("arms", {}).get(a["id"], {})}
        if m is not None:
            # Raw values remain inspectable even if queue/provenance checks fail.
            entry["raw_metrics"] = values(m)
            entry["raw_metrics_path"] = str(RUNS / a["run_name"] / "eval" / f"ckpt_{checkpoint}" / "metrics.json")
        if not evidence["valid_scientific_result"]:
            lines.append(f"| {a['id']} | {parent} | {evidence['status']} |" + " — |" * 11)
        else:
            v = values(m)
            dp = v["physical_r2_cb"] - dig(pm, PATHS["physical_r2_cb"]) if pm is not None else None
            dn = v["normalized_r2_cb"] - dig(pm, PATHS["normalized_r2_cb"]) if pm is not None else None
            entry.update(**v, delta_physical=v["physical_r2_cb"] - bv["physical_r2_cb"],
                         delta_normalized=v["normalized_r2_cb"] - bv["normalized_r2_cb"], delta_parent_physical=dp, delta_parent_normalized=dn)
            entry["per_case_delta"] = {c: m["per_case"][c]["overall"]["r2"] - base["per_case"][c]["overall"]["r2"] for c in base["per_case"]}
            entry["cases_improved"] = sum(d > 0 for d in entry["per_case_delta"].values())
            lines.append("| " + " | ".join([a["id"], parent, evidence["status"], fmt(v["physical_r2_cb"]), fmt(entry["delta_physical"]),
                fmt(dp), fmt(v["normalized_r2_cb"]), fmt(v["case_mean"]), fmt(v["mae"], 3), fmt(v["high_wss_r2"]),
                fmt(v["top10_ratio"], 3), fmt(v["p99_ratio"], 3), fmt(v["top10_iou"], 3), fmt(v["spearman"], 3)]) + " |")
            diagnostics.append(f"| {a['id']} | {fmt(v['linear_fit_r2_cb'])} | {fmt(v['linear_fit_slope_cb'])} | {fmt(v['linear_fit_intercept_cb'])} |")
        summary["arms"][a["id"]] = entry
    completed = sum(v["evidence"]["valid_scientific_result"] for v in summary["arms"].values())
    summary["completed_count"] = completed
    summary["raw_metrics_count"] = sum("raw_metrics" in v for v in summary["arms"].values())
    summary["expected_count"] = len(matrix["arms"])
    lines += diagnostics
    lines += ["", "口径与归因：", ""] + [f"- {key}：{value}。" for key, value in METRIC_DEFINITIONS.items()]
    lines += ["", "幅值比接近1、a接近1且b接近0，同时Pa误差改善，支持幅值校准改善；IoU与排名相关提高支持空间分布改善。两组可同时改善。high-WSS R²和log_z R²单独不能区分两者。",
              "R²_fit是标签与预测的相关性诊断；不对测试标签拟合校准器后重报主指标。并列排名沿用原评估实现，不能据微小Spearman差异作强结论。",
              "工作簿本节只填best，文件名last为历史命名。总览AA/AB列为log_z空间pooled MAE/RMSE；教师视图O列为log_z p99比，汇总对比L列为物理Pa p99比，二者不可混用。",
              "", f"有效的该checkpoint评估：{completed}/{len(matrix['arms'])}；已保存原始指标：{summary['raw_metrics_count']}。",
              f"整批完结归档：{'已核验' if summary['matrix_finalized'] else '尚未核验'}；要求队列完成、全部train/best/last退出码0，以及真实起止源码哈希一致。",
              f"源码核验依据：{integrity['basis']}；当前工作区相对起始源码{'有变化（单列记录，不追溯否定已归档结果）' if integrity['current_drift'] and integrity['archived'] else '有变化' if integrity['current_drift'] else '无变化'}。", "",
              "源文件变化、失败阶段或证据未齐的指标仅保留在汇总JSON的raw_metrics中，不计入完成数，也不进入主表比较。",
              "原始配置差异见 configs/v6_followup_20260909/matrix.json；运行命令、日志与退出码见 queue_status.json。",
              "test34为已暴露开发集；本轮沿用旧协议作探索性比较。新几何候选尚未取得用户可视化确认，不用于这些实验。", ""]
    EXP.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines)
    (EXP / f"matrix_tables_{checkpoint}.md").write_text(text)
    (EXP / f"matrix_summary_{checkpoint}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    return text, summary


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--update-tracker", action="store_true")
    args = ap.parse_args(argv)
    best, summary = report("best")
    _, last = report("last")
    print(f"best={summary['completed_count']}/15 last={last['completed_count']}/15")
    if args.update_tracker:
        heading = "### 13.3 结果（checkpoint=best；由原始 metrics 自动回填）"
        section = heading + "\n\n" + best.replace("# V6-A5 后续 D/L/M1–M5（best）\n", "")
        TRACKER.write_text(replace_or_append(TRACKER.read_text(), heading, section))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
