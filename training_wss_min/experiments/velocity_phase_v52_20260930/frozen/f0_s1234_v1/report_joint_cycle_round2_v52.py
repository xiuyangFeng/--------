#!/usr/bin/env python3
"""Audit and report ten single-seed round-two runs and six existing baselines.

Read-only with respect to training, predictions, and Slurm. Output files are
report.json/report.md only. Missing old per-frame/time-increment metrics stay
missing; this reporter never starts inference to fill them.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics

try:
    from . import report_joint_cycle_v52 as base
except ImportError:  # Direct script execution, including frozen submissions.
    import report_joint_cycle_v52 as base

ROOT = base.ROOT
ARMS = ("UT0", "UT1", "WT0", "WT1", "U0", "U1", "W0", "W1", "P1", "J2")
TASKS = base.TASKS
BASELINES = base.ARMS
ARM_CONTRACT = {
    **{a: (["velocity"], "I") for a in ("Iu", "UT0", "UT1", "U0", "U1")},
    **{a: (["wss"], "I") for a in ("Iw", "WT0", "WT1", "W0", "W1")},
    "Ip": (["pressure"], "I"), "P1": (["pressure"], "J1"),
    "J0": (list(TASKS), "J0"), "J1c": (list(TASKS), "J1c"),
    "J1": (list(TASKS), "J1"), "J2": (list(TASKS), "J1"),
}
PRIMARY_SEGMENTS = ("peak", "trough", "cycle")
SEGMENTS = PRIMARY_SEGMENTS + ("accel", "decel", "plateau")
NEW_SEGMENTS = ("peak_frame21", "temporal_difference", "endpoint_difference")
BASIC = ("r2", "mae", "rmse", "relative_l2")
FIELDS = {
    f"{task}_{segment}_{metric}": (task, segment, metric)
    for task in TASKS for segment in PRIMARY_SEGMENTS + NEW_SEGMENTS
    for metric in ("r2", "mae", "rmse")
}
PAIRS = (("UT1", "UT0"), ("WT1", "WT0"), ("U1", "U0"), ("W1", "W0"),
         ("P1", "Ip"), ("J2", "J1")) + tuple(
             (a, "Iu" if a.startswith("U") else "Iw") for a in ARMS[:8]) + (("J2", "I"), ("P1", "J1"))
DEFAULT_BASELINE_REPORT = ROOT / "training_wss_min/experiments/joint_cycle_v52_20260929/report.json"
read_json, digest, finite, mean, nested = base.read_json, base.digest, base.finite, base.mean, base.nested
recursive_mean, absolute_path, fmt = base.recursive_mean, base.absolute_path, base.fmt


def scores(tasks):
    return {name: nested(tasks, path) for name, path in FIELDS.items()}


def required_fields(is_new):
    fields = dict(base.FIELDS)
    fields.update({f"{t}_{s}_{m}": (t, s, m) for t in TASKS for s in SEGMENTS for m in BASIC})
    if is_new:
        fields.update({f"{t}_{s}_{m}": (t, s, m) for t in TASKS for s in NEW_SEGMENTS for m in BASIC})
        fields.update({f"{t}_frame{f:02d}_{m}": (t, "per_frame", f"{f:02d}", m)
                       for t in TASKS for f in range(80) for m in BASIC})
    return fields


def compare_aggregates(expected, actual, prefix="tasks"):
    errors = []
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return [f"aggregate/per_case mismatch: {prefix}"]
        for key, value in expected.items():
            if key not in actual:
                errors.append(f"aggregate/per_case missing: {prefix}.{key}")
            else:
                errors.extend(compare_aggregates(value, actual[key], f"{prefix}.{key}"))
    elif expected is None:
        if actual is not None:
            errors.append(f"aggregate/per_case mismatch: {prefix}")
    elif not finite(actual) or not math.isclose(expected, actual, rel_tol=1e-6, abs_tol=1e-9):
        errors.append(f"aggregate/per_case mismatch: {prefix}")
    return errors


def percentile(values, q):
    values = sorted(x for x in values if finite(x))
    if not values:
        return None
    index = (len(values) - 1) * q
    low, high = math.floor(index), math.ceil(index)
    return values[low] + (values[high] - values[low]) * (index - low)


def training_progress(out):
    history, warnings = [], []
    path = out / "history.jsonl"
    if path.is_file():
        lines = path.read_text().splitlines()
        for index, line in enumerate(lines):
            try:
                history.append(json.loads(line))
            except ValueError:
                warnings.append(f"history line {index + 1} incomplete/invalid")
    epochs = [r.get("epoch") for r in history if isinstance(r.get("epoch"), int)]
    memory = [r.get("peak_cuda_memory_bytes") for r in history if finite(r.get("peak_cuda_memory_bytes"))]
    return {"last_completed_epoch": max(epochs) if epochs else 0,
            "target_epochs": 150, "history_epochs": len(set(epochs)),
            "history_train_seconds": sum(r.get("seconds", 0) for r in history if finite(r.get("seconds"))),
            "history_peak_cuda_memory_bytes": max(memory) if memory else None,
            "history_warnings": warnings}


def robustness(records):
    result = {}
    for task in TASKS:
        if not any(task in row["tasks"] for row in records.values()):
            continue
        result[task] = {}
        for segment in PRIMARY_SEGMENTS:
            values = [nested(row["tasks"], (task, segment, "r2")) for row in records.values()]
            values = [v for v in values if finite(v)]
            result[task][segment] = {"n_valid_r2": len(values), "r2_p10": percentile(values, .1),
                                     "negative_r2_count": sum(v < 0 for v in values)}
    return result


def load_arm(cfg_path, cfg, contract):
    arm, out = cfg["arm"], Path(cfg["out_dir"])
    errors, warnings = [], []
    is_new = arm in ARMS
    expected_schema = "joint_cycle_round2_v52_v1" if is_new else "joint_cycle_v52_v1"
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
            for name, keys in required_fields(is_new).items():
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


def paired_delta(left, right, records):
    units = sorted(set(records[left]) & set(records[right]))
    result = {"contrast": f"{left}-{right}", "n_common_units": len(units), "unit_ids": units, "metrics": {}}
    for name, path in FIELDS.items():
        differences = {}
        for unit in units:
            a, b = (nested(records[x][unit]["tasks"], path) for x in (left, right))
            if finite(a) and finite(b):
                differences[unit] = a - b
        vals = list(differences.values())
        improving = sum(v > 0 if path[-1] == "r2" else v < 0 for v in vals)
        result["metrics"][name] = {"n_pairs": len(vals), "mean_delta": mean(vals),
                                   "median_delta": statistics.median(vals) if vals else None,
                                   "n_improved": improving, "improvement_fraction": improving / len(vals) if vals else None,
                                   "per_unit_delta": differences}
    return result


def assert_shared_evaluation(configs):
    reference = configs[0]
    for cfg in configs[1:]:
        for section, keys in (("data", ("stats_path", "cache_root", "support_n", "eval_query_n", "split_sha256")),
                              ("eval", ("partition", "query_chunk", "angle_speed_floor_m_s"))):
            if any(cfg[section].get(k) != reference[section].get(k) for k in keys):
                raise ValueError(f"cross-arm/baseline {section} sampling/provenance contract mismatch")


def baseline_configs(path, contract):
    baseline_report = read_json(path)
    if baseline_report.get("schema") != "joint_cycle_v52_report_v1":
        raise ValueError("baseline report schema mismatch")
    for key in ("split_sha256", "seed", "fold", "phase_count", "heldout_unit_ids", "train_unit_ids"):
        if baseline_report.get("contract", {}).get(key) != contract[key]:
            raise ValueError(f"baseline report contract mismatch: {key}")
    configs = []
    for arm in BASELINES:
        source = Path(baseline_report["models"][arm]["config_path"])
        cfg = read_json(source)
        if cfg["arm"] != arm:
            raise ValueError("baseline config arm mismatch")
        configs.append((source, cfg))
    return configs


def markdown(report):
    lines = ["# V5.2 第二轮全周期优化：单 seed 实验进度与结果", "",
             f"生成时间：{report['generated_at']}。CV5 fold0；206 训练 / 55 开发留出单元；seed1234；150 epochs，fixed-last。", "",
             "全部指标按数据单元等权平均；单元内按体积/面积加权。峰值窗17–26、谷底窗5–9和43–57、周期0–79（均0-based）。"
             "阶段R²按整段时空变异计算，不是逐帧R²的平均。该开发留出集已参与设计决策，不作盲测或临床验证。", "",
             "## 新实验进度", "", "| 实验 | 状态 | 完成epoch | 留出评价 | Slurm | 状态依据 |",
             "|---|---|---:|---:|---|---|"]
    for arm in ARMS:
        row = report["models"][arm]
        slurm = row.get("slurm_evidence") or {}
        lines.append(f"| {arm} | {row['status']} | {row['last_completed_epoch']}/150 | {row['n_evaluated']}/55 | "
                     f"{slurm.get('state', '—')} | {row.get('status_evidence', '—')} |")
    task_arms = {
        "velocity": ("Iu", "J0", "J1c", "J1", "UT0", "UT1", "U0", "U1", "J2"),
        "pressure": ("Ip", "J0", "J1c", "J1", "P1", "J2"),
        "wss": ("Iw", "J0", "J1c", "J1", "WT0", "WT1", "W0", "W1", "J2"),
    }
    for task, name in (("velocity", "速度向量（分量MAE，m/s）"), ("pressure", "相对压力 p′（Pa）"), ("wss", "标量 WSS（Pa）")):
        lines += ["", f"## {name}", "",
                  "| 模型 | 状态/单元 | 峰值R² ↑ | 峰值MAE ↓ | 谷底R² ↑ | 谷底MAE ↓ | 周期R² ↑ | 周期MAE ↓ |",
                  "|---|---|---:|---:|---:|---:|---:|---:|"]
        for arm in task_arms[task]:
            row = report["models"][arm]
            vals = [arm, f"{row['status']}/{row['n_evaluated']}"]
            vals += [fmt(nested(row["task_metrics"], (task, stage, metric)))
                     for stage in PRIMARY_SEGMENTS for metric in ("r2", "mae")]
            lines.append("| " + " | ".join(vals) + " |")
    lines += ["", "## 固定峰帧与时间变化误差", "",
              "第21帧独立于峰值窗；时间差分是物理值的一阶相邻增量（0→1至78→79），没有除以Δt，不能称为时间导数。"
              "首尾差分比较79−0的预测与真值差异，不是0与80的周期闭合误差，也不要求0与79相等。旧基线未保存这些指标，保持—，没有追加推理。", "",
              "| 模型 | 任务 | 第21帧R² | 第21帧MAE | 第21帧RMSE | 相邻增量RMSE | 相邻增量rel-L2 | 79−0增量RMSE |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for arm in BASELINES + ARMS:
        row = report["models"][arm]
        for task in ARM_CONTRACT[arm][0]:
            vals = [arm, task] + [fmt(nested(row["task_metrics"], (task, s, m))) for s, m in (
                ("peak_frame21", "r2"), ("peak_frame21", "mae"), ("peak_frame21", "rmse"),
                ("temporal_difference", "rmse"), ("temporal_difference", "relative_l2"), ("endpoint_difference", "rmse"))]
            lines.append("| " + " | ".join(vals) + " |")
    lines += ["", "## 同单元配对比较", "",
              "差值为前者−后者；R²越正越好，误差越负越好。只计算可核验记录的交集；缺失、近零能量和零变异导致的null不填0。"
              "未满55单元的数字仅为进度，不用于最终排名。", "",
              "| 对比 | 任务 | 峰值ΔR² | 谷底ΔR² | 周期ΔR² | 周期ΔMAE | 周期R²改善/有效配对 |",
              "|---|---|---:|---:|---:|---:|---|"]
    for pair in report["paired_deltas"]:
        a, b = pair["contrast"].split("-")
        for task in TASKS:
            if task not in ARM_CONTRACT[a][0] or (b != "I" and task not in ARM_CONTRACT[b][0]):
                continue
            vals = [pair["contrast"], task]
            vals += [fmt(pair["metrics"][f"{task}_{s}_r2"]["mean_delta"]) for s in PRIMARY_SEGMENTS]
            vals += [fmt(pair["metrics"][f"{task}_cycle_mae"]["mean_delta"])]
            item = pair["metrics"][f"{task}_cycle_r2"]
            vals.append(f"{item['n_improved']}/{item['n_pairs']}")
            lines.append("| " + " | ".join(vals) + " |")
    lines += ["", "## 计算成本", "",
              "采样推理计时包含缓存读取、张量传输、编码与80相位解码，未含原始几何预处理和导出；不是全点云临床端到端耗时。"
              "独立臂与联合臂输出任务数量不同，比较完整三场方案时应按同单元先相加各专用网络耗时，再求中位/P95。"
              "I为既有Iu+Ip+Iw实际成本之和；本轮没有事后挑选组合后重推理。显存为运行至今累计峰值，完成后包含评价。", "",
              "| 模型 | 任务 | 参数 | 峰值显存GiB | 训练秒 | 采样推理中位秒 | P95秒 |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for arm in BASELINES + ("I",) + ARMS:
        row = report["models"][arm]
        mem = row.get("peak_cuda_memory_bytes")
        vals = [arm, "+".join(ARM_CONTRACT[arm][0]) if arm != "I" else "velocity+pressure+wss",
                fmt(row.get("parameter_count")), fmt(mem / 1024**3) if finite(mem) else "—",
                fmt(row.get("train_seconds")), fmt(row.get("sampled_query_median_s")), fmt(row.get("sampled_query_p95_s"))]
        lines.append("| " + " | ".join(vals) + " |")
    lines += ["", "## 审核与完整指标", "",
              "completed要求正式metrics、完整55个唯一留出单元、150轮训练完成及配置/统计/源码来源可核验；smoke排除。"
              "训练完成不等于评价完成；Slurm失败终态单列，未确认RUNNING的旧进度标为started_unknown。", ""]
    for arm in BASELINES + ARMS:
        row = report["models"][arm]
        issues = row["errors"] + row["warnings"] + row.get("history_warnings", [])
        lines.append(f"- {arm}: {row['status']}；" + ("；".join(issues) if issues else "当前来源和指标契约核对通过"))
    lines += ["", "report.json包含全部六阶段的RMSE/relative-L2、0–79逐帧指标、速度幅值/方向/覆盖率、"
              "WSS Spearman和TAWSS、R² P10及负值单元数、逐单元配对差值。"
              "单seed、单折结果只用于筛选；压力提升不能抵消速度/WSS退化。", ""]
    return "\n".join(lines)


def generate(config_dir, experiment_dir):
    matrix = read_json(config_dir / "matrix.json")
    configs, indices = [], {}
    for index, item in enumerate(matrix["arms"]):
        path = absolute_path(item["config"] if isinstance(item, dict) else item, config_dir)
        if isinstance(item, dict) and item.get("config_sha256") and digest(path) != item["config_sha256"]:
            raise ValueError(f"matrix configuration hash mismatch: {path}")
        cfg = read_json(path)
        configs.append((path, cfg))
        indices[cfg["arm"]] = item.get("array_index", index) if isinstance(item, dict) else index
    if Counter(cfg["arm"] for _, cfg in configs) != Counter(ARMS):
        raise ValueError("matrix must contain exactly ten registered round-two arms")
    if sorted(indices.values()) != list(range(10)):
        raise ValueError("matrix array indices must uniquely cover 0..9")
    contract = base.split_contract([cfg for _, cfg in configs])
    baseline_path = Path(matrix.get("baseline_report", DEFAULT_BASELINE_REPORT))
    previous = baseline_configs(baseline_path, contract)
    assert_shared_evaluation([cfg for _, cfg in configs + previous])
    accounting = base.scheduler_evidence(experiment_dir)
    rows, records = {}, {}
    for path, cfg in previous + configs:
        arm = cfg["arm"]
        rows[arm], records[arm] = load_arm(path, cfg, contract)
        if arm in ARMS:
            base.apply_scheduler_evidence(rows[arm], indices[arm], accounting)
        else:
            rows[arm]["status_evidence"] = "validated_existing_baseline" if rows[arm]["formal_complete"] else "baseline_artifact_status"
    rows["I"], records["I"] = base.combine_independent(rows, records)
    rows["I"]["sampled_query_p95_s"] = percentile([r["cache_load_encode_decode_seconds"] for r in records["I"].values()], .95)
    costs = [rows[a].get("train_seconds") for a in ("Iu", "Ip", "Iw")]
    rows["I"]["train_seconds"] = sum(costs) if all(finite(x) for x in costs) else None
    rows["I"]["scores"] = scores(rows["I"]["task_metrics"])
    result = {"schema": "joint_cycle_round2_v52_report_v1", "generated_at": datetime.now(timezone.utc).isoformat(),
              "config_dir": str(config_dir), "experiment_dir": str(experiment_dir), "contract": contract,
              "baseline_report": {"path": str(baseline_path), "sha256": digest(baseline_path)},
              "scope": "single-seed development holdout screening; no new baseline inference; sampled-query timing",
              "metric_definitions": {"peak": list(range(17, 27)), "trough": list(range(5, 10)) + list(range(43, 58)),
                  "cycle": list(range(80)), "peak_frame21": "fixed 0-based frame21, independent of peak window",
                  "temporal_difference": "physical increments across 0->1 through 78->79; not divided by dt",
                  "endpoint_difference": "physical frame79-frame0 increment error; not a periodic 0/80 closure test",
                  "r2_null": "zero spatial/segment variance", "relative_l2_null": "near-zero truth energy"},
              "models": rows, "status_counts": dict(Counter(rows[a]["status"] for a in ARMS)),
              "scheduler_accounting": accounting,
              "paired_deltas": [paired_delta(a, b, records) for a, b in PAIRS]}
    experiment_dir.mkdir(parents=True, exist_ok=True)
    for name, content in (("report.json", json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"),
                          ("report.md", markdown(result))):
        target = experiment_dir / name
        temp = target.with_suffix(target.suffix + ".tmp")
        temp.write_text(content)
        temp.replace(target)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", required=True, type=Path)
    parser.add_argument("--experiment-dir", required=True, type=Path)
    args = parser.parse_args()
    if not args.config_dir.is_absolute() or not args.experiment_dir.is_absolute():
        parser.error("--config-dir and --experiment-dir must be absolute paths")
    result = generate(args.config_dir, args.experiment_dir)
    print(json.dumps({"report": str(args.experiment_dir / "report.md"), "status_counts": result["status_counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
