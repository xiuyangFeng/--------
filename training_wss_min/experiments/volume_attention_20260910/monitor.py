"""Five-minute cluster monitor and independently checked incremental publication.

All mutable monitoring code lives outside the frozen scientific source set.
The reporting lock is also used by finalize.py; final results take precedence.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from training_wss_min.tools.volume_attention_common import EXP, fingerprints, save_json, sha256, stamp
from training_wss_min.tools import report_volume_attention as R
from training_wss_min.tools.update_volume_attention_xlsx import update_workbook, NUMERIC_FIELDS
from training_wss_min.tools.run_volume_attention_queue import verify_data_inputs
from training_wss_min.tools.tracker_section import replace_or_append
from finalize import HEADING, TRACKER, guarded_write, render


def read_json(path, default=None):
    return json.loads(Path(path).read_text()) if Path(path).exists() else default


def jsonl_records(path):
    path = Path(path)
    if not path.exists():
        return []
    # A writer may be appending the final line while this snapshot is read.
    lines = path.read_text().splitlines()
    result = []
    for i, line in enumerate(lines):
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(lines) - 1:
                raise
    return result


def history(run):
    return jsonl_records(Path(run) / "history.jsonl")


def snapshot(queue):
    arms = {}
    now = time.time()
    for aid, arm in queue["arms"].items():
        run = Path(arm["run_dir"])
        hist = history(run)
        active = arm["status"] not in ("pending", "complete", "failed", "interrupted")
        logfile = Path(arm.get("stages", {}).get(arm["status"], {}).get("log", run / "train.log"))
        events = run / "nonfinite_events.jsonl"
        nonfinite = jsonl_records(events)
        arms[aid] = dict(status=arm["status"], epochs=len(hist),
                         last_train_loss=hist[-1]["train_loss"] if hist else None,
                         log_age_seconds=max(0., round(now-logfile.stat().st_mtime, 1)) if active and logfile.exists() else None,
                         fatal_nonfinite=sum("fatal" in e.get("kind", "") for e in nonfinite),
                         amp_overflow_skips=sum(e.get("kind") == "amp_gradient_overflow_skipped" for e in nonfinite))
    return arms


def scheduler_jobs():
    result = subprocess.run(["/public/slurm/bin/sacct", "-n", "-X", "-j", "14040,14041",
                             "--format=JobIDRaw,State,Elapsed", "--parsable2"],
                            capture_output=True, text=True, check=True, timeout=20)
    jobs = {}
    for line in result.stdout.splitlines():
        parts = line.strip().split("|")
        if len(parts) >= 3 and parts[0] in ("14040", "14041"):
            jobs[parts[0]] = dict(state=parts[1], elapsed=parts[2])
    return jobs


def validate_completed(report, queue, cache):
    """Validate only immutable completed arms, then cache by result/checkpoint hash."""
    import torch
    checked = set()
    for row in report["rows"]:
        aid, task, ck = row["id"], row["task"], row["checkpoint"]
        if not aid.startswith("R5") and queue["arms"][aid]["status"] != "complete":
            continue
        run = Path(row["run_dir"])
        base = run / "eval" / f"ckpt_{ck}"
        paths = [Path(row["metrics_path"]), base / "diagnostics/summary.json",
                 base / "predictions/test/manifest.json", run / f"ckpt_{ck}.pt"]
        if not aid.startswith("R5"):
            paths.append(run / "history.jsonl")
        digest = hashlib.sha256(json.dumps({str(p): sha256(p) for p in paths}, sort_keys=True).encode()).hexdigest()
        key = f"{aid}/{task}/{ck}"
        if cache.get(key, {}).get("digest") != digest or not cache.get(key, {}).get("passed"):
            metric = read_json(row["metrics_path"])["test"]
            validation = R.independently_validate_predictions(run, ck, task, metric)
            if not validation["passed"]:
                raise ValueError(f"Independent incremental metric validation failed: {key}")
            diag = read_json(base / "diagnostics/summary.json")["tasks"][task]
            if set(diag["longwave"]["per_case"]) != set(metric["per_case"]):
                raise ValueError(f"Diagnostic case mismatch: {key}")
            if not aid.startswith("R5"):
                hist = history(run)
                checkpoint = torch.load(run / f"ckpt_{ck}.pt", map_location="cpu", weights_only=False)
                expected = min(hist, key=lambda x: x["train_loss"])["epoch"] if ck == "best" else 399
                if [h["epoch"] for h in hist] != list(range(400)) or checkpoint["epoch"] != expected:
                    raise ValueError(f"Incomplete/incorrect checkpoint history: {key}")
                validation["history"] = dict(epochs=400, checkpoint_epoch=checkpoint["epoch"], passed=True)
            cache[key] = dict(passed=True, digest=digest, checked_at=stamp(), validation=validation)
            save_json(EXP / "monitor_validation.json", cache)
        checked.add(key)
    return checked


def pending_report(report, checked, arms):
    """Never publish half-written/unevaluated checkpoint metrics or final verdicts."""
    result = copy.deepcopy(report)
    result["complete"] = False
    for row in result["rows"]:
        key = f"{row['id']}/{row['task']}/{row['checkpoint']}"
        if key in checked:
            row["status"] = "历史参照·数值已核验" if row["id"].startswith("R5") else "400轮/评估/诊断完成·数值已核验"
            if not row["id"].startswith("R5"):
                row["gate"] = "阶段数值已核验；整批判定待26臂完成"
                parent_key = f"{row['parent']}/{row['task']}/{row['checkpoint']}"
                if parent_key not in checked:
                    row["delta_r2_parent"] = None
        else:
            arm = arms[row["id"]]
            row["status"] = f"{arm['status']} · {arm['epochs']}/400轮"
            for field in NUMERIC_FIELDS:
                row[field] = None
            row["groups"], row["pressure_query_groups"] = {}, {}
            row.pop("longwave_summary", None)
            row["gate"] = "待训练/评估/数值核验"
    result["comparisons"] = [p for p in result["comparisons"]
                             if f"{p['id']}/{p['task']}/{p['checkpoint']}" in checked
                             and f"{p['reference']}/{p['task']}/{p['checkpoint']}" in checked]
    for gate in result["gates"].values():
        gate.update(qualified=None, longwave_improved=None, issues=["整批待核验"],
                    accepted_scientific_result=False)
    for pair in result["comparisons"]:
        pair.update(qualified=None, gate_issues="阶段数值已核验；整批判定待完成")
    result["factorials"] = [v for v in result["factorials"]
                            if all(f"{aid}/{v['task']}/{v['checkpoint']}" in checked for aid in v["arms"])]
    result["incremental_validation"] = sorted(checked)
    return result


def update_documents(report, state):
    done = state["completed_arms"]
    active = "、".join(f"{aid} {a['epochs']}/400（{a['status']}）" for aid, a in state["arms"].items()
                       if a["status"] not in ("complete", "pending")) or "无"
    alert = "；".join(state["alerts"]) or "未见致命异常"
    live = (f"### 后台监控\n\n更新时间：{state['updated_at']}；每5分钟巡检。正式任务14040：已完成 **{done}/26臂**；"
            f"运行中：{active}。{alert}。已通过逐点数值核验的结果滚动回填工作簿三张结果表，未完成项为空；最终14041仍负责整批验收与结论。\n\n"
            "[实时状态](../../../training_wss_min/experiments/volume_attention_20260910/monitor_status.json) · "
            "[阶段验收](../../../training_wss_min/experiments/volume_attention_20260910/monitor_validation.json) · "
            "[工作簿](../../03-汇报材料/WSS_PointNet实验矩阵与结果汇总last.xlsx)\n")
    section = render(report).replace("状态：执行中或尚未通过完整验收，不作最终结论。", f"状态：正式训练已完成{done}/26臂；阶段结果滚动回填，整批结论待完成。")
    section = section.replace("### 压力", live + "\n### 压力", 1)
    original = TRACKER.read_text()
    guarded_write(TRACKER, original, replace_or_append(original, HEADING, section))
    path = EXP / "README.md"
    original = path.read_text()
    lines = original.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if line.startswith("## "):
            break
        if "状态：" in line:
            lines[i] = line.split("状态：", 1)[0] + f"状态：正式14040已完成{done}/26臂；后台每5分钟巡检并滚动回填，最终验收由14041执行。\n"
    status = (f"## 当前执行状态\n\n更新时间：{state['updated_at']}。监控任务{state['job_id']}每5分钟巡检；"
              f"已完成{done}/26臂，运行中：{active}。{alert}。\n\n"
              "已核验阶段成绩见[results.md](results.md)，实时状态见[monitor_status.json](monitor_status.json)，"
              "阶段数值验收见[monitor_validation.json](monitor_validation.json)。工作簿在新结果或执行阶段变化时更新，"
              "整批结束后14041生成最终报告、联合query成本与正式判定。冻结协议不变；中止的14039记录仍在原归档。\n")
    guarded_write(path, original, replace_or_append("".join(lines), "## 当前执行状态", status))


def append_event(event):
    with (EXP / "monitor_events.jsonl").open("a") as stream:
        stream.write(json.dumps(dict(timestamp=stamp(), **event), ensure_ascii=False) + "\n")
    print(stamp(), json.dumps(event, ensure_ascii=False), flush=True)


def tick():
    previous = read_json(EXP / "monitor_status.json", {})
    queue = read_json(EXP / "execution/queue_status.json")
    arms = snapshot(queue)
    state = dict(status="running", updated_at=stamp(), job_id=os.environ.get("SLURM_JOB_ID", "manual"),
                 interval_seconds=300, completed_arms=sum(a["status"] == "complete" for a in arms.values()),
                 arms=arms, queue_status=queue["status"], alerts=[],
                 workbook_updated_at=previous.get("workbook_updated_at"), publication_signature=previous.get("publication_signature"))
    state["scheduler_jobs"] = scheduler_jobs()
    terminal_failures = ("FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY", "PREEMPTED", "BOOT_FAIL", "DEADLINE")
    for job, record in state["scheduler_jobs"].items():
        if record["state"].startswith(terminal_failures):
            state["alerts"].append(f"Slurm {job}: {record['state']}")
    training_state = state["scheduler_jobs"].get("14040", {}).get("state", "unknown")
    if queue["status"] == "running" and training_state == "COMPLETED":
        state["alerts"].append("Slurm训练已结束但队列仍为running；需要核对退出记录")
    for aid, arm in arms.items():
        if arm["status"] in ("failed", "interrupted") or arm["fatal_nonfinite"]:
            state["alerts"].append(f"{aid}: {arm['status']}, fatal_nonfinite={arm['fatal_nonfinite']}")
        if (arm["log_age_seconds"] or 0) > 1800:
            state["alerts"].append(f"{aid}: 当前阶段日志超过30分钟未更新")
    source_ok = fingerprints() == queue["source_sha256"]
    state["source_unchanged"] = source_ok
    if not source_ok:
        state["alerts"].append("冻结源码发生变化；暂停回填")
    with (EXP / ".reporting.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            state["status"] = "finalizer_or_publisher_active"
            save_json(EXP / "monitor_status.json", state)
            return False
        final = read_json(EXP / "completion_status.json", {})
        if final.get("status") == "complete":
            acceptance = read_json(EXP / "xlsx_acceptance.json", {})
            report = read_json(EXP / "report.json", {})
            if not (acceptance.get("complete") and report.get("complete") and acceptance.get("all_historical_values_formulas_merges_preserved")):
                raise ValueError("Finalizer marked complete without matching final workbook/report acceptance")
            state["status"] = "complete"
            state["workbook_updated_at"] = final["ended_at"]
            save_json(EXP / "monitor_status.json", state)
            append_event(dict(kind="complete", completed_arms=state["completed_arms"], workbook_acceptance=str(EXP / "xlsx_acceptance.json")))
            return True
        if final.get("status") == "failed":
            state["alerts"].append("最终汇总失败：" + final.get("error", "unknown"))
            state["status"] = "needs_attention"
        elif queue["status"] != "running":
            state["status"] = "waiting_finalizer"
        if source_ok and queue["status"] == "running" and not final and training_state in ("RUNNING", "COMPLETING"):
            verify_data_inputs()
            raw_report = R.build_report(validate=False)
            cache = read_json(EXP / "monitor_validation.json", {})
            checked = validate_completed(raw_report, queue, cache)
            report = pending_report(raw_report, checked, arms)
            signature = hashlib.sha256(json.dumps({"monitor_source": sha256(__file__), "stages": {k: v['status'] for k, v in arms.items()},
                       "validations": {k: cache[k]['digest'] for k in sorted(checked)}}, sort_keys=True).encode()).hexdigest()
            if signature != previous.get("publication_signature"):
                R.write_outputs(report)
                acceptance = update_workbook(require_complete=False)
                state.update(publication_signature=signature, workbook_updated_at=stamp(),
                             validated_task_checkpoints=len(checked), workbook_sha256=acceptance["after_sha256"])
                append_event(dict(kind="results_or_stage_changed", completed_arms=state["completed_arms"],
                                  validated_task_checkpoints=len(checked), workbook_sha256=acceptance["after_sha256"]))
            update_documents(report, state)
        elif final.get("status") != "running":
            report = read_json(EXP / "report.json", {})
            if report.get("rows"):
                report["complete"] = False
                update_documents(report, state)
        if state["scheduler_jobs"].get("14041", {}).get("state", "").startswith(terminal_failures):
            state["status"] = "needs_attention"
        if state["alerts"] != previous.get("alerts", []):
            append_event(dict(kind="alerts_changed", alerts=state["alerts"]))
        save_json(EXP / "monitor_status.json", state)
        return state["status"] == "needs_attention"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    with (EXP / ".monitor.lock").open("a") as singleton:
        fcntl.flock(singleton, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            start = time.monotonic()
            try:
                done = tick()
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                previous = read_json(EXP / "monitor_status.json", {})
                if previous.get("error") != error:
                    append_event(dict(kind="monitor_error", error=error, traceback=traceback.format_exc()))
                previous.update(status="monitor_error", updated_at=stamp(), error=error)
                save_json(EXP / "monitor_status.json", previous)
                done = False
            if args.once or done:
                return
            time.sleep(max(1, 300 - (time.monotonic() - start)))


if __name__ == "__main__":
    main()
