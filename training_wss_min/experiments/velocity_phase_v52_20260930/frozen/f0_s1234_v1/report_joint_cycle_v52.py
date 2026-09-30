#!/usr/bin/env python3
"""Report the six registered V5.2 runs without importing torch or evaluating models.

Only formal run directories named by the matrix are read. Smoke outputs are
excluded. A training-complete marker alone never establishes completion.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess


ROOT = Path(os.environ.get("GNN_JOINT_SOURCE_ROOT", "/public/newhome/cy/Digital_twin/GNN"))
DEFAULT_SPLIT = ROOT / "data_wss_v5/views_v5_2_full_20260923/wss_min_view_v1/cv5_v52/fold0.json"
ARMS = ("Iu", "Ip", "Iw", "J0", "J1c", "J1")
TASKS = ("velocity", "pressure", "wss")
FIELDS = {
    "velocity_relative_l2": ("velocity", "cycle", "relative_l2"),
    "velocity_vector_r2": ("velocity", "cycle", "r2"),
    "pressure_r2": ("pressure", "cycle", "r2"),
    "pressure_mae_pa": ("pressure", "cycle", "mae"),
    "wss_r2": ("wss", "cycle", "r2"),
    "wss_mae_pa": ("wss", "cycle", "mae"),
    "wss_trough_spearman": ("wss", "trough", "spatial_spearman_unweighted"),
    "tawss_r2": ("wss", "tawss_80", "r2"),
    "tawss_mae_pa": ("wss", "tawss_80", "mae"),
}
SACCT = Path("/public/slurm/bin/sacct")
SLURM_FAILURES = {"FAILED", "TIMEOUT", "OUT_OF_MEMORY", "CANCELLED", "NODE_FAIL", "BOOT_FAIL", "DEADLINE"}


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def mean(values):
    values = [float(x) for x in values if finite(x)]
    return statistics.fmean(values) if values else None


def nested(value, keys):
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def recursive_mean(rows):
    if not rows:
        return {}
    if isinstance(rows[0], dict):
        keys = sorted(set().union(*(x.keys() for x in rows)))
        return {k: recursive_mean([x[k] for x in rows if k in x]) for k in keys}
    return mean(rows)


def scores(tasks):
    return {name: nested(tasks, path) for name, path in FIELDS.items()}


def absolute_path(value, base):
    p = Path(value)
    return p if p.is_absolute() else base / p


def scheduler_evidence(experiment_dir):
    """Read array accounting once; never modify scheduler or run artifacts."""
    audit = {"source": "sacct", "available": False, "reason": None, "jobs": {}}
    submission = experiment_dir / "submission.json"
    if not submission.is_file():
        audit["reason"] = "submission.json unavailable"
        return audit
    try:
        job = str(read_json(submission).get("array_job", ""))
    except (ValueError, OSError) as exc:
        audit["reason"] = f"cannot read submission: {exc}"
        return audit
    if not job.isdigit():
        audit["reason"] = "numeric array_job unavailable"
        return audit
    audit["array_job"] = job
    if not SACCT.is_file() or not os.access(SACCT, os.X_OK):
        audit["reason"] = "sacct executable unavailable"
        return audit
    command = [str(SACCT), "-j", job, "--format=JobID%64,State,ExitCode", "--parsable2", "--noheader"]
    audit["command"] = command
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=15, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        audit["reason"] = f"sacct query failed: {exc}"
        return audit
    if result.returncode:
        audit["reason"] = f"sacct exit {result.returncode}: {result.stderr.strip()[:500]}"
        return audit
    audit["available"] = True
    for line in result.stdout.splitlines():
        fields = line.strip().split("|")
        if len(fields) < 3:
            continue
        job_id, raw_state, exit_code = (x.strip() for x in fields[:3])
        # Job steps (.batch/.extern) and compact array ranges are not task evidence.
        prefix, separator, index = job_id.partition("_")
        if prefix != job or separator != "_" or not index.isdigit():
            continue
        state = raw_state.split()[0].rstrip("+") if raw_state else ""
        audit["jobs"][job_id] = {"job_id": job_id, "state": state, "raw_state": raw_state, "exit_code": exit_code}
    return audit


def apply_scheduler_evidence(row, index, audit):
    evidence = audit["jobs"].get(f"{audit.get('array_job', '')}_{index}")
    row["slurm_evidence"] = evidence
    if evidence and not row["has_metrics_artifact"] and evidence["state"] in SLURM_FAILURES:
        row["status"] = "failed"
        row["formal_complete"] = False
        row["status_evidence"] = "slurm_terminal_failure"
        row["warnings"].append(f"Slurm {evidence['job_id']} terminal {evidence['raw_state']}, ExitCode={evidence['exit_code']}; no formal metrics")
    elif row["status"] == "running":
        row["status_evidence"] = "slurm_running" if evidence and evidence["state"] == "RUNNING" else "started_unknown"
        if row["status_evidence"] == "started_unknown":
            row["warnings"].append("started_unknown: progress files exist, but no current RUNNING or failure terminal state was confirmed")
    else:
        row["status_evidence"] = "validated_metrics" if row["formal_complete"] else "artifact_status"


def split_contract(configs):
    """Prefer the statistics-bound split; fall back only before stats exist."""
    paths = set()
    for cfg in configs:
        if cfg["data"].get("split_path"):
            paths.add(cfg["data"]["split_path"])
        stats_path = Path(cfg["data"]["stats_path"])
        if stats_path.is_file():
            paths.add(read_json(stats_path)["split_path"])
    if len(paths) > 1:
        raise ValueError("arms reference different statistics-bound split paths")
    path = Path(next(iter(paths))) if paths else DEFAULT_SPLIT
    split = read_json(path)
    heldout, train = split["test_cases"], split["train_cases"]
    if (split.get("fold") != 0 or split.get("n_folds") != 5
            or len(set(heldout)) != 55 or len(heldout) != 55
            or len(set(train)) != 206 or set(train) & set(heldout)):
        raise ValueError("expected CV5 fold0 with disjoint 206 train / 55 held-out data units")
    return {"split_path": str(path), "split_sha256": digest(path), "fold": 0,
            "n_folds": 5, "seed": 1234, "phase_count": 80,
            "heldout_count": 55, "heldout_unit_ids": heldout, "train_unit_ids": train}


def load_arm(cfg_path, cfg, contract):
    arm, out = cfg["arm"], Path(cfg["out_dir"])
    errors, warnings = [], []
    if cfg.get("seed") != 1234 or cfg.get("train", {}).get("phases") != 80:
        errors.append("configuration must specify seed1234 and 80 training phases")
    if cfg.get("train", {}).get("selection") != "last" or cfg.get("eval", {}).get("partition") != "test":
        errors.append("configuration must use fixed-last selection and test partition")
    if cfg["data"].get("split_sha256", contract["split_sha256"]) != contract["split_sha256"] or cfg["data"].get("fold", 0) != 0:
        errors.append("configuration split hash/fold mismatch")
    expected_tasks = {"Iu": ["velocity"], "Ip": ["pressure"], "Iw": ["wss"]}.get(arm, list(TASKS))
    if cfg.get("tasks") != expected_tasks or cfg.get("variant") != ("I" if arm.startswith("I") else arm):
        errors.append("arm variant/tasks differ from the six-run matrix")
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
        for field in ("seed", "arm", "variant", "tasks", "data", "train", "eval"):
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
            for name, keys in FIELDS.items():
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
        for name, value in scores(task_metrics).items():
            actual = scores(metrics.get("tasks", {}))[name]
            if finite(value) and (not finite(actual) or not math.isclose(value, actual, rel_tol=1e-6, abs_tol=1e-9)):
                errors.append(f"aggregate/per_case mismatch: {name}")
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
           "has_metrics_artifact": "metrics" in artifacts,
           "execution": artifacts.get("execution", artifacts.get("launch_status"))}
    return row, records


def combine_independent(rows, records):
    components = [rows[a] for a in ("Iu", "Ip", "Iw")]
    state = [x["status"] for x in components]
    status = "completed" if all(x == "completed" for x in state) else "failed" if "failed" in state else "running" if any(x != "pending" for x in state) else "pending"
    common = sorted(set(records["Iu"]) & set(records["Ip"]) & set(records["Iw"]))
    combined = {}
    for unit in common:
        rs = [records[a][unit] for a in ("Iu", "Ip", "Iw")]
        ts = [r.get("cache_load_encode_decode_seconds") for r in rs]
        combined[unit] = {"unit_id": unit, "tasks": {k: v for r in rs for k, v in r["tasks"].items()},
                          "cache_load_encode_decode_seconds": sum(ts) if all(finite(t) for t in ts) else None}
    metrics = {k: v for r in components for k, v in r["task_metrics"].items()}
    params = [r["parameter_count"] for r in components]
    medians = [r["sampled_query_median_s"] for r in components]
    timings = [x["cache_load_encode_decode_seconds"] for x in combined.values()]
    timings = [t for t in timings if finite(t)]
    return {"arm": "I", "status": status, "formal_complete": status == "completed", "components": ["Iu", "Ip", "Iw"],
            "n_evaluated": len(common), "n_evaluated_per_task": {a: rows[a]["n_evaluated"] for a in ("Iu", "Ip", "Iw")},
            "task_metrics": metrics, "scores": scores(metrics),
            "parameter_count": sum(params) if all(finite(p) for p in params) else None,
            "sampled_query_median_s": statistics.median(timings) if timings else None,
            "sum_of_component_medians_s": sum(medians) if all(finite(t) for t in medians) else None,
            "inference_definition": "median over matched units of summed Iu+Ip+Iw sampled-query seconds; not parallel wall time"}, combined


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
        result["metrics"][name] = {"n_pairs": len(vals), "mean_delta": mean(vals),
                                   "median_delta": statistics.median(vals) if vals else None,
                                   "per_unit_delta": differences}
    return result


def fmt(value):
    return f"{value:.5g}" if finite(value) else "—"


def markdown(report):
    lines = ["# V5.2 联合周期建模：单种子开发筛查", "",
             f"生成时间：{report['generated_at']}。协议：CV5 fold 0，206 训练 / 55 留出数据单元，seed 1234，80 独立相位，fixed-last。", "",
             "结果仅用于开发阶段模型筛查；留出数据单元不等同于独立患者，不作显著性或硬性 Go/No-Go 判定。", "",
             "速度为三分量矢量指标；压力为逐帧参考压力 p′；WSS 为标量模长，TAWSS 为 80 相位均值，未评估 OSI。", "",
             "| 模型 | 状态 | 已评估单元 | u rel-L2 ↓ | u 矢量 R² ↑ | p′ R² ↑ | p′ MAE/Pa ↓ | WSS R² ↑ | WSS MAE/Pa ↓ | 谷期 Spearman ↑ | TAWSS R² ↑ | TAWSS MAE/Pa ↓ | 参数数 | 采样查询中位秒 ↓ |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for arm in ("I", "J0", "J1c", "J1", "Iu", "Ip", "Iw"):
        row = report["models"][arm]
        vals = [arm, row["status"], str(row["n_evaluated"])]
        vals += [fmt(row["scores"].get(k)) for k in FIELDS]
        vals += [str(row["parameter_count"]) if row["parameter_count"] is not None else "—", fmt(row["sampled_query_median_s"])]
        lines.append("| " + " | ".join(vals) + " |")
    lines += ["", "I 保留 Iu/Ip/Iw 各任务分数，参数相加；主要耗时为同一单元三次独立推理耗时之和的中位数。"
              "三臂尚未全部就绪时，各任务分数可来自不同长度的部分结果；共同单元数与各臂行明确列出。", "",
              "计时覆盖缓存读取、张量传输、几何编码和 80 相位解码；查询点来自固定采样池，"
              "不代表完整原始点云端到端临床时延，未包含原始几何预处理/文件导出，也未提供通量守恒指标。", "",
              "## 同单元配对差值", "", "差值为前者减后者；误差下降为负，R²/Spearman 改善为正。仅使用双方可核验记录的交集，空缺不填零。", "",
              "| 对比 | 共同单元 | 指标 | 有效配对数 | 平均差 | 中位差 |", "|---|---:|---|---:|---:|---:|"]
    for contrast in report["paired_deltas"]:
        for name, item in contrast["metrics"].items():
            lines.append(f"| {contrast['contrast']} | {contrast['n_common_units']} | {name} | {item['n_pairs']} | {fmt(item['mean_delta'])} | {fmt(item['median_delta'])} |")
    lines += ["", "## 状态与契约核对", "",
              "completed 必须同时具备正式 metrics、完整 55 单元记录及通过校验的配置/来源；仅训练完成标记不算完成。"
              "无正式 metrics 且 Slurm 记录 FAILED/TIMEOUT/OUT_OF_MEMORY/CANCELLED 等失败终态时显示 failed；"
              "其余未确认调度器运行状态的进度记录注明 started_unknown。smoke 不纳入结果。", ""]
    for arm in ARMS:
        row = report["models"][arm]
        issues = row["errors"] + row["warnings"]
        lines.append(f"- {arm}: {row['status']}；" + ("；".join(issues) if issues else "当前可用文件契约核对通过"))
    lines += ["", "机器可读完整任务/阶段指标、每单元配对差值与来源路径见同目录 `report.json`。", ""]
    return "\n".join(lines)


def generate(config_dir, experiment_dir):
    matrix = read_json(config_dir / "matrix.json")
    configs, array_indices = [], {}
    for index, item in enumerate(matrix["arms"]):
        path = absolute_path(item["config"] if isinstance(item, dict) else item, config_dir)
        cfg = read_json(path)
        configs.append((path, cfg))
        array_indices[cfg["arm"]] = item.get("array_index", index) if isinstance(item, dict) else index
    if Counter(cfg["arm"] for _, cfg in configs) != Counter(ARMS):
        raise ValueError("matrix must contain exactly Iu, Ip, Iw, J0, J1c, J1")
    contract = split_contract([cfg for _, cfg in configs])
    # Same sampled-query protocol and stats must hold across all model arms.
    reference = configs[0][1]
    for _, cfg in configs[1:]:
        for section, keys in (("data", ("stats_path", "cache_root", "support_n", "eval_query_n")),
                              ("eval", ("partition", "query_chunk"))):
            if any(cfg[section].get(k) != reference[section].get(k) for k in keys):
                raise ValueError(f"cross-arm {section} sampling/provenance contract mismatch")
    rows, records = {}, {}
    accounting = scheduler_evidence(experiment_dir)
    for path, cfg in configs:
        rows[cfg["arm"]], records[cfg["arm"]] = load_arm(path, cfg, contract)
        apply_scheduler_evidence(rows[cfg["arm"]], array_indices[cfg["arm"]], accounting)
    rows["I"], records["I"] = combine_independent(rows, records)
    result = {"schema": "joint_cycle_v52_report_v1", "generated_at": datetime.now(timezone.utc).isoformat(),
              "config_dir": str(config_dir), "experiment_dir": str(experiment_dir), "contract": contract,
              "scope": "single-seed internal development; sampled queries; no clinical/full-cloud timing or flux claims",
              "models": rows, "status_counts": dict(Counter(rows[a]["status"] for a in ARMS)),
              "scheduler_accounting": accounting,
              "paired_deltas": [paired_delta(a, b, records) for a, b in (("J0", "I"), ("J1c", "J0"), ("J1", "J1c"))]}
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
