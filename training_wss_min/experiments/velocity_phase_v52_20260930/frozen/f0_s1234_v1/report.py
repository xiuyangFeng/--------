#!/usr/bin/env python3
"""Report the registered eight-arm velocity screen, never starting inference.

Completion requires source/config/data provenance, all55 unique units, last150,
and exact agreement of aggregate and per-unit metrics. Unavailable diagnostics
stay missing. Baseline Iu/U0 artifacts are read without modification.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
from datetime import datetime,timezone
import json
from pathlib import Path
import statistics
import subprocess
try:
    from . import report_joint_cycle_round2_v52 as previous
except ImportError:
    import report_joint_cycle_round2_v52 as previous
base=previous.base
ROOT=base.ROOT
ARMS=("D1","D2","A0","A1","G00","G01","G10","G11")
BASELINES=("Iu","U0")
TASKS=("velocity",)
ARM_CONTRACT={a:(["velocity"],"I") for a in BASELINES+ARMS}
PRIMARY_SEGMENTS=("peak","trough","decel","cycle")
SEGMENTS=PRIMARY_SEGMENTS+("accel","plateau","early_trough","late_trough","peak_frame21")
read_json,digest,finite,mean,nested=base.read_json,base.digest,base.finite,base.mean,base.nested
recursive_mean,absolute_path,fmt=base.recursive_mean,base.absolute_path,base.fmt
required_fields=previous.required_fields
compare_aggregates=previous.compare_aggregates
training_progress=previous.training_progress
percentile=previous.percentile
robustness=previous.robustness

def scores(tasks):
    return {f"velocity_{stage}_{metric}": nested(tasks, ("velocity", stage, metric))
            for stage in SEGMENTS for metric in ("r2", "mae", "rmse", "direction_cosine")}

def load_arm(cfg_path, cfg, contract):
    arm, out = cfg["arm"], Path(cfg["out_dir"])
    errors, warnings = [], []
    is_new = arm in ARMS
    expected_schema = "velocity_phase_v52_v1" if is_new else ("joint_cycle_round2_v52_v1" if arm == "U0" else "joint_cycle_v52_v1")
    if cfg.get("schema") != expected_schema:
        errors.append("configuration schema mismatch")
    if cfg.get("train", {}).get("epochs") != 150:
        errors.append("configuration must specify 150 epochs")
    if cfg.get("seed") != 1234 or cfg.get("train", {}).get("phases") != 80:
        errors.append("configuration must specify seed1234 and 80 training phases")
    if cfg.get("train", {}).get("selection") != "last" or cfg.get("eval", {}).get("partition") != "test":
        errors.append("configuration must use fixed-last selection and test partition")
    if cfg["data"].get("split_sha256", contract["split_sha256"]) != contract["split_sha256"] or cfg["data"].get("fold", 0) != 0:
        errors.append("configuration split hash/fold mismatch")
    expected_tasks, expected_variant = ARM_CONTRACT[arm]
    if cfg.get("tasks") != expected_tasks or cfg.get("variant") != expected_variant:
        errors.append("arm variant/tasks differ from the registered matrix")
    artifacts = {}
    for name in ("metrics", "provenance", "config", "training_complete", "execution", "failed", "error", "launch_status"):
        path = out / f"{name}.json"
        if path.is_file():
            try:
                artifacts[name] = read_json(path)
            except (ValueError, OSError) as exc:
                errors.append(f"cannot read {path.name}: {exc}")
    metrics, provenance = artifacts.get("metrics", {}), artifacts.get("provenance", {})
    smoke = bool(provenance.get("smoke") or artifacts.get("training_complete", {}).get("smoke")
                 or metrics.get("status") == "smoke_passed" or "smoke" in out.parts)
    if smoke:
        errors.append("smoke output excluded from formal results")
    runtime_cfg = artifacts.get("config")
    if runtime_cfg is not None:
        for field in ("schema", "seed", "arm", "variant", "tasks", "data", "train", "eval", "out_dir"):
            if runtime_cfg.get(field) != cfg.get(field):
                errors.append(f"runtime config differs from registered config: {field}")
        actual_model, expected_model = dict(runtime_cfg.get("model", {})), dict(cfg.get("model", {}))
        actual_model.pop("geometry_config_path", None)
        expected_model.pop("geometry_config_path", None)
        if actual_model != expected_model:
            errors.append("runtime model options differ from registered configuration")
    stats_path = Path(cfg["data"]["stats_path"])
    if stats_path.is_file():
        stats = read_json(stats_path)
        if stats.get("split_sha256") != contract["split_sha256"]:
            errors.append("normalization statistics split hash mismatch")
        if not stats.get("complete_train_partition") or stats.get("train_ids") != contract["train_unit_ids"]:
            errors.append("normalization statistics were not fitted on the complete train partition")
        if provenance and provenance.get("stats_sha256") != digest(stats_path):
            errors.append("run provenance statistics hash mismatch")
    elif provenance or metrics:
        errors.append("normalization statistics file unavailable for provenance verification")
    if provenance:
        for code_path, expected_hash in provenance.get("code", {}).items():
            if not Path(code_path).is_file() or digest(code_path) != expected_hash:
                errors.append(f"provenance code fingerprint mismatch: {code_path}")
        source = Path(provenance.get("config_path", "/nonexistent"))
        if not source.is_file() or provenance.get("config_sha256") != digest(source):
            errors.append("run provenance configuration hash cannot be verified")
        elif runtime_cfg is not None and read_json(source) != runtime_cfg:
            errors.append("run config snapshot differs from its provenance source")
    per_case = {}
    cases_file = out / "per_case.jsonl"
    if cases_file.is_file():
        lines = cases_file.read_text().splitlines()
        for i, line in enumerate(lines):
            try:
                row = json.loads(line)
            except ValueError:
                if i == len(lines) - 1 and not metrics:
                    warnings.append("incomplete final per_case line ignored while evaluation is running")
                    continue
                errors.append(f"invalid per_case JSON at line {i + 1}")
                continue
            unit = row.get("unit_id")
            if not isinstance(unit, str) or unit not in contract["heldout_unit_ids"] or row.get("partition") != "test":
                errors.append(f"per_case line {i + 1} is outside the registered held-out partition")
                continue
            if unit in per_case:
                errors.append(f"duplicate per_case unit: {unit}")
            if sorted(row.get("tasks", {})) != sorted(expected_tasks):
                errors.append(f"per_case line {i + 1} task set mismatch")
            for name, keys in required_fields(is_new or arm == "U0").items():
                if keys[0] not in expected_tasks:
                    continue
                parent = nested(row.get("tasks", {}), keys[:-1])
                value = nested(row.get("tasks", {}), keys)
                if not isinstance(parent, dict) or keys[-1] not in parent or (value is not None and not finite(value)):
                    errors.append(f"per_case line {i + 1} missing/non-finite {name}")
            duration = row.get("cache_load_encode_decode_seconds")
            if not finite(duration) or duration <= 0:
                errors.append(f"per_case line {i + 1} has invalid inference timing")
            per_case[unit] = row
    if metrics:
        if artifacts.get("training_complete", {}).get("epochs") != 150:
            errors.append("completed evaluation requires 150 completed training epochs")
        for key, value in {"arm": arm, "variant": cfg["variant"], "seed": 1234, "phase_count": 80,
                           "partition": "test", "n_units": 55, "selection": "last", "status": "complete"}.items():
            if metrics.get(key) != value:
                errors.append(f"metrics.{key} must be {value!r}")
        if set(per_case) != set(contract["heldout_unit_ids"]):
            errors.append("metrics completion requires all 55 unique held-out per_case records")
        if not provenance or runtime_cfg is None:
            errors.append("completed metrics require provenance and runtime config snapshots")
        if sorted(metrics.get("tasks", {})) != sorted(expected_tasks):
            errors.append("metrics task set mismatch")
        if metrics.get("parameter_count") != provenance.get("parameter_count"):
            errors.append("metrics/provenance parameter count mismatch")
        if not isinstance(metrics.get("parameter_count"), int) or metrics["parameter_count"] <= 0:
            errors.append("invalid model parameter count")
        if nested(metrics, ("inference", "n_units")) != 55:
            errors.append("inference timing must cover the 55 held-out units")
    markers = [artifacts.get(k, {}) for k in ("execution", "launch_status", "failed", "error")]
    failed = any(x.get("status") in ("failed", "failure", "cancelled", "timeout", "out_of_memory")
                 or (x.get("returncode") is not None and x["returncode"] != 0) for x in markers)
    failed |= "failed" in artifacts or "error" in artifacts
    has_progress = bool(artifacts or per_case or (out / "history.jsonl").is_file() or (out / "ckpt_last.pt").is_file())
    status = "failed" if errors or failed else "completed" if metrics else "running" if has_progress else "pending"
    if not metrics and "training_complete" in artifacts:
        warnings.append("training finished, but formal 55-unit evaluation is not complete")
    # An invalid contract must not leak numbers into comparisons. Valid partial
    # records remain useful, including a run that later fails during evaluation.
    usable = not errors and not smoke and bool(provenance) and runtime_cfg is not None
    records = per_case if usable else {}
    task_metrics = recursive_mean([r["tasks"] for r in records.values()])
    if metrics and usable:
        errors.extend(compare_aggregates(task_metrics, metrics.get("tasks", {})))
        if errors:
            status, records, task_metrics = "failed", {}, {}
    times = [r.get("cache_load_encode_decode_seconds") for r in records.values()]
    times = [x for x in times if finite(x)]
    row = {"arm": arm, "status": status, "config_path": str(cfg_path), "out_dir": str(out),
           "errors": errors, "warnings": warnings, "n_evaluated": len(records),
           "formal_complete": status == "completed", "task_metrics": task_metrics,
           "scores": scores(task_metrics), "parameter_count": provenance.get("parameter_count") if not smoke else None,
           "sampled_query_median_s": statistics.median(times) if times else None,
           "metric_source": "per_case equal-data-unit aggregation" if records else None,
           "inference_definition": nested(metrics, ("inference", "definition")),
           "evaluation_query_n": cfg["data"].get("eval_query_n"), "stats_sha256": provenance.get("stats_sha256"),
           "training_complete": "training_complete" in artifacts,
           "sampled_query_p95_s": percentile(times, .95),
           "train_seconds": artifacts.get("training_complete", {}).get("seconds"),
           "peak_cuda_memory_bytes": metrics.get("peak_cuda_memory_bytes"),
           "has_metrics_artifact": "metrics" in artifacts,
           "execution": artifacts.get("execution", artifacts.get("launch_status"))}
    row.update(training_progress(out))
    if row["peak_cuda_memory_bytes"] is None:
        row["peak_cuda_memory_bytes"] = row.get("history_peak_cuda_memory_bytes")
    row["robustness"] = robustness(records)
    return row, records


def flatten(value, prefix=()):
    if isinstance(value, dict):
        result = {}
        for key, child in value.items():
            result.update(flatten(child, prefix + (str(key),)))
        return result
    return {"/".join(prefix): value} if value is None or finite(value) else {}


def paired(left, right, records):
    units = sorted(set(records[left]) & set(records[right]))
    flattened = {a: {u: flatten(records[a][u]["tasks"]) for u in units} for a in (left, right)}
    names = set().union(*(set(v) for rows in flattened.values() for v in rows.values())) if units else set()
    metrics = {}
    for name in sorted(names):
        differences = {}
        for unit in units:
            a, b = (flattened[k][unit].get(name) for k in (left, right))
            if finite(a) and finite(b):
                differences[unit] = a - b
        values = list(differences.values())
        larger_better = name.endswith(("/r2", "/direction_cosine", "/recall", "/precision"))
        metrics[name] = {"mean_delta": mean(values), "median_delta": statistics.median(values) if values else None,
                         "n_pairs": len(values), "per_unit_delta": differences,
                         "n_improved": sum(v > 0 if larger_better else v < 0 for v in values)
                         if name.endswith(("/r2", "/direction_cosine", "/recall", "/precision", "/rmse", "/mae", "/mse", "/relative_l2")) else None}
    return {"contrast": f"{left}-{right}", "n_common_units": len(units), "metrics": metrics}


def scheduler_evidence(experiment_dir):
    audit = {"available": False, "jobs": {}, "stages": {}, "reason": None}
    path = experiment_dir / "submission.json"
    if not path.is_file():
        audit["reason"] = "not submitted"
        return audit
    submission = read_json(path)
    jobs = [str(submission[k]) for k in ("geometry_job", "smoke_job", "batch1_job", "batch2_job", "report_job") if k in submission]
    audit["submission"] = submission
    if not jobs or any(not x.isdigit() for x in jobs):
        audit["reason"] = "no valid numeric job IDs"
        return audit
    try:
        result = subprocess.run(["/public/slurm/bin/sacct", "-j", ",".join(jobs),
            "--format=JobID%64,State,ExitCode", "--parsable2", "--noheader"],
            capture_output=True, text=True, timeout=15, check=True)
        audit["available"] = True
        for line in result.stdout.splitlines():
            parts = line.split("|")
            if len(parts) < 3 or "." in parts[0]:
                continue
            jid, state, code = parts[:3]
            audit["jobs"][jid] = {"job_id": jid, "state": state.split()[0].rstrip("+"), "exit_code": code}
    except (OSError, subprocess.SubprocessError) as exc:
        audit["reason"] = str(exc)
    for label in ("geometry_job", "smoke_job", "batch1_job", "batch2_job", "report_job"):
        audit["stages"][label] = audit["jobs"].get(str(submission.get(label, "")))
    return audit


def markdown(report):
    lines = ["# V5.2速度谷底/减速段：8臂单seed进度与结果", "",
             f"生成时间：{report['generated_at']}。206训练 / 55开发留出单元；fold0、seed1234、150epochs、last。", "",
             "仅汇总已存在正式产物；本报告不启动训练/推理。数据单元等权，单元内部体积加权。"
             "峰值窗17–26、谷底5–9和43–57、减速27–42、周期0–79（0-based）。阶段R²不是逐帧R²均值。", "",
             "## 执行进度", "", "| 臂 | 状态 | 完成epoch | 留出单元 | Slurm |", "|---|---|---:|---:|---|"]
    for arm in ARMS:
        row = report["models"][arm]
        lines.append(f"| {arm} | {row['status']} | {row['last_completed_epoch']}/150 | {row['n_evaluated']}/55 | "
                     f"{(row.get('slurm_evidence') or {}).get('state', '—')} |")
    lines += ["", "## 主要速度指标", "", "| 臂 | 状态/单元 | 峰值R² | 峰值MAE | 谷底R² | 谷底MAE | 减速R² | 减速MAE | 周期R² | 周期MAE |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for arm in BASELINES + ARMS:
        row = report["models"][arm]
        values = [arm, f"{row['status']}/{row['n_evaluated']}"] + [fmt(nested(row["task_metrics"], ("velocity", stage, metric)))
                  for stage in PRIMARY_SEGMENTS for metric in ("r2", "mae")]
        lines.append("| " + " | ".join(values) + " |")
    lines += ["", "MAE为Cartesian三分量平均绝对误差，单位m/s；部分评价仅作进度，不作最终排名。", "",
              "## 方向、幅值与谷底子窗", "",
              "| 臂 | 谷底方向余弦 | 减速方向余弦 | 谷底方向覆盖 | 谷底速率R² | 早谷底R² | 晚谷底R² | 时间增量RMSE |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    paths = [("trough", "direction_cosine"), ("decel", "direction_cosine"),
             ("trough", "direction_weight_coverage"), ("speed", "trough", "r2"),
             ("early_trough", "r2"), ("late_trough", "r2"), ("temporal_difference", "rmse")]
    for arm in BASELINES + ARMS:
        row = report["models"][arm]
        lines.append("| " + " | ".join([arm] + [fmt(nested(row["task_metrics"], ("velocity",) + p)) for p in paths]) + " |")
    lines += ["", "方向只以真值速度≥0.01m/s定义有效区，不能去掉预测近零点；早/晚谷底分列不替换主谷底窗。"
              "时间增量为相邻相位的物理速度差，未除以Δt。基线缺失的附加指标保持—。", "",
              "## 配对比较", "", "| 对比 | 共同单元 | 峰值ΔR² | 谷底ΔR² | 减速ΔR² | 周期ΔR² | 谷底R²改善/有效 |",
              "|---|---:|---:|---:|---:|---:|---|"]
    for comparison in report["paired_deltas"]:
        metrics = comparison["metrics"]
        item = metrics.get("velocity/trough/r2", {})
        vals = [comparison["contrast"], str(comparison["n_common_units"])]
        vals += [fmt(metrics.get(f"velocity/{stage}/r2", {}).get("mean_delta")) for stage in PRIMARY_SEGMENTS]
        vals += [f"{item.get('n_improved', 0)}/{item.get('n_pairs', 0)}"]
        lines.append("| " + " | ".join(vals) + " |")
    lines += ["", "G11必须分别比较G10、G01。2×2交互量仅是单seed效果分解，不是训练随机性统计检验。", "",
              "## 成本", "", "| 臂 | 参数 | 训练秒 | 显存GiB | 采样推理中位秒 | P95秒 |",
              "|---|---:|---:|---:|---:|---:|"]
    for arm in BASELINES + ARMS:
        row = report["models"][arm]
        mem = row.get("peak_cuda_memory_bytes")
        lines.append("| " + " | ".join([arm, fmt(row.get("parameter_count")), fmt(row.get("train_seconds")),
            fmt(mem / 1024 ** 3) if finite(mem) else "—", fmt(row.get("sampled_query_median_s")), fmt(row.get("sampled_query_p95_s"))]) + " |")
    lines += ["", "计时口径以各run的inference.definition为准；几何预处理成本不包含在旧U0计时中。"
              "单次采样池推理计时不能当临床全点云端到端时延。", "", "## 可核验产物与限制", ""]
    for arm in BASELINES + ARMS:
        row = report["models"][arm]
        issues = row["errors"] + row["warnings"] + row.get("history_warnings", [])
        lines.append(f"- {arm}: " + ("；".join(issues) if issues else "当前产物合同核对通过；未完成臂无正式结果。"))
    lines += ["", "report.json保留全部已保存诊断，包括轴向/横向、逆轴向、速率尾部、frame有效性、逐帧和困难单元。"
              "metrics_flat.csv提供全部有限数值/缺失值；没有保存的指标不补造。", "",
              "本轮仍是单折、单seed开发筛选；55单元已参与设计。训练集尾部阈值、方向lambda与几何独立记录SHA；"
              "不混用全量训练数据，不修改并行运行的全量模型。", ""]
    return "\n".join(lines)


def generate(config_dir, experiment_dir):
    matrix = read_json(config_dir / "matrix.json")
    configs = []
    for item in matrix["arms"]:
        path = absolute_path(item["config"], config_dir)
        if digest(path) != item["config_sha256"]:
            raise ValueError(f"matrix config hash mismatch: {path}")
        configs.append((path, read_json(path)))
    if tuple(cfg["arm"] for _, cfg in configs) != ARMS:
        raise ValueError("unexpected eight-arm matrix")
    contract = base.split_contract([c for _, c in configs])
    baselines = [(Path(matrix["baseline_configs"][arm]), read_json(matrix["baseline_configs"][arm])) for arm in BASELINES]
    previous.assert_shared_evaluation([cfg for _, cfg in configs + baselines])
    audit = scheduler_evidence(experiment_dir)
    models, records = {}, {}
    for path, cfg in baselines + configs:
        arm = cfg["arm"]
        row, cases = load_arm(path, cfg, contract)
        if arm in ARMS:
            index = ARMS.index(arm)
            job = audit.get("submission", {}).get("batch1_job" if index < 4 else "batch2_job")
            evidence = audit["jobs"].get(f"{job}_{index}")
            row["slurm_evidence"] = evidence
            if evidence and evidence["state"] in base.SLURM_FAILURES and not row["has_metrics_artifact"]:
                row["status"], row["formal_complete"] = "failed", False
                row["warnings"].append(f"Slurm terminal {evidence['state']} {evidence['exit_code']}")
            elif row["status"] == "running" and not (evidence and evidence["state"] == "RUNNING"):
                row["warnings"].append("progress files exist; current RUNNING not confirmed")
            elif row["status"] == "pending" and evidence:
                row["status"] = "queued" if evidence["state"] == "PENDING" else row["status"]
        models[arm], records[arm] = row, cases
    comparisons = [(item["arm"], control) for item in matrix["arms"] for control in item["primary_controls"]]
    comparisons += [(a, "Iu") for a in ARMS] + [(a, "U0") for a in ARMS if (a, "U0") not in comparisons]
    comparisons = list(dict.fromkeys(comparisons))
    report = {"schema": "velocity_phase_v52_report_v1", "generated_at": datetime.now(timezone.utc).isoformat(),
              "contract": contract, "status_counts": dict(Counter(models[a]["status"] for a in ARMS)),
              "models": models, "scheduler": audit, "paired_deltas": [paired(a, b, records) for a, b in comparisons],
              "missing_diagnostics_policy": "No new inference; unavailable stored metrics remain null",
              "matrix": matrix}
    # Same-unit interaction preserves case pairing, without fabricating seed uncertainty.
    graph_units = sorted(set.intersection(*(set(records[a]) for a in ("G00", "G01", "G10", "G11"))))
    interaction = {}
    for stage in PRIMARY_SEGMENTS:
        for metric in ("r2", "mae", "rmse", "direction_cosine"):
            values = []
            for unit in graph_units:
                v = [nested(records[a][unit]["tasks"], ("velocity", stage, metric)) for a in ("G00", "G01", "G10", "G11")]
                if all(finite(x) for x in v):
                    values.append(v[3] - v[2] - v[1] + v[0])
            interaction[f"{stage}/{metric}"] = {"mean": mean(values), "n_units": len(values)}
    report["graph_interaction_G11_G10_minus_G01_G00"] = interaction
    experiment_dir.mkdir(parents=True, exist_ok=True)
    for name, value in (("report.json", json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"),
                        ("report.md", markdown(report))):
        path = experiment_dir / name
        tmp = path.with_name(path.name + ".tmp"); tmp.write_text(value); tmp.replace(path)
    with (experiment_dir / "metrics_flat.csv").open("w", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(["arm", "status", "n_evaluated", "metric", "value"])
        for arm, row in models.items():
            for name, value in flatten(row["task_metrics"]).items():
                writer.writerow([arm, row["status"], row["n_evaluated"], name, value])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=ROOT / "training_wss_min/configs/velocity_phase_v52_20260930")
    parser.add_argument("--experiment-dir", type=Path, default=ROOT / "training_wss_min/experiments/velocity_phase_v52_20260930")
    args = parser.parse_args()
    report = generate(args.config_dir, args.experiment_dir)
    print(json.dumps({"status_counts": report["status_counts"], "report": str(args.experiment_dir / "report.md")}, ensure_ascii=False))


if __name__ == "__main__":
    main()

