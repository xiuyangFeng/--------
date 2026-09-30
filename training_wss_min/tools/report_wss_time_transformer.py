#!/usr/bin/env python3
"""Strict v2 temporal-attention development report; no inference or selection.

Complete comparisons require every registered run, source fingerprint and queue
acceptance. Code is checked against saved queue evidence, not working sources.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

RUN_SCHEMA = "time_transformer_v51_v2"
ARMS = ("TT-warm", "TT-warm-noattn", "TT-raw", "TT-raw-noattn")
KEYS = ("cycle_r2cb_pa", "trough_r2cb_pa", "tawss_r2cb_pa", "cycle_r2cb_ln",
        "peak_r2cb_pa", "peak_time_err_med_frames", "peak_time_err_p90_frames",
        "ts_corr_ln", "amp_err_ln_med")
PAIRS = ((ARMS[0], ARMS[1], "A1_warm_attention_minus_diagonal"),
         (ARMS[0], ARMS[2], "A2_warm_minus_raw_attention"),
         (ARMS[2], ARMS[3], "A3_raw_attention_minus_diagonal"),
         (ARMS[1], ARMS[3], "A4_warm_minus_raw_diagonal"))


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def is_digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def invalidated(directory):
    return sorted(str(p.resolve()) for p in Path(directory).glob("INVALIDATED*"))


def frozen_digest(mapping, path):
    key = str(Path(path).resolve())
    require(is_digest(mapping.get(key)), f"missing frozen fingerprint: {key}")
    return mapping[key]


def queue_fingerprint_digest(queue):
    mapping = queue.get("fingerprints")
    require(isinstance(mapping, dict) and mapping, "queue fingerprints missing")
    require(all(isinstance(k, str) and is_digest(v) for k, v in mapping.items()), "invalid queue fingerprints")
    digest = hashlib.sha256(json.dumps(mapping, sort_keys=True).encode()).hexdigest()
    if "fingerprint_sha256" in queue:
        require(queue["fingerprint_sha256"] == digest, "queue fingerprint digest differs from saved mapping")
    return digest


def validate_run(job, config_path, queue, record, matrix_hash):
    path = Path(job["done_marker"]).resolve()
    require(not invalidated(path.parent) and not invalidated(path.parent.parent), "run directory is INVALIDATED")
    cfg, run = read_json(config_path), read_json(path)
    require(cfg.get("schema") == RUN_SCHEMA and run.get("schema") == RUN_SCHEMA, "expected formal v2 config/metrics")
    require(run.get("smoke") is False, "smoke or unspecified smoke status cannot enter formal report")
    require(record.get("state") == "done" and record.get("returncode") == 0, "queue not done with returncode=0")
    require(Path(record["done_marker"]).resolve() == path, "queue done-marker mismatch")
    config_hash = sha256(config_path)
    require(run.get("config_sha256") == record.get("config_sha256") == config_hash, "original config SHA256 mismatch")
    actual_value = record.get("actual_config")
    if isinstance(actual_value, dict):
        require(actual_value == cfg, "queue parsed config differs from registered config")
        require(isinstance(record.get("config"), str), "queue dispatched config path missing")
        actual_config = Path(record["config"]).resolve()
    else:
        require(actual_value is None or isinstance(actual_value, str), "invalid queue actual_config type")
        actual_config = Path(record.get("config") or actual_value or config_path).resolve()
        if actual_value is not None:
            require(Path(actual_value).resolve() == actual_config, "queue config path fields disagree")
    require(frozen_digest(queue["fingerprints"], actual_config) == config_hash, "config differs from queued snapshot")
    require(Path(cfg["out_dir"]).resolve() == path.parent, "config output directory mismatch")
    for key in ("name", "arm", "fold", "seed"):
        require(run.get(key) == cfg[key], f"run/config identity mismatch: {key}")
    require(cfg["name"].split("/")[-1] == job["id"], "matrix/run identity mismatch")
    require(cfg["arm"] in ARMS and cfg["train"]["selection"] == "last", "unsupported arm/checkpoint selection")
    require(run.get("partition") == cfg["eval"]["partition"] == "test", "only existing cv3 holdout is registered")
    expected_steps = int(cfg["train"]["epochs"]) * int(cfg["train"]["steps_per_epoch"])
    require(run.get("steps") == expected_steps, "training steps differ from registered budget")

    split_path = Path(cfg["split_path"]).resolve()
    stats_path = Path(cfg["frame_stats_path"]).resolve()
    waveform_path = Path(cfg["waveform_path"]).resolve()
    cache_path = Path(cfg["cache_dir"]).resolve() / "manifest.json"
    split, stats, waveform = read_json(split_path), read_json(stats_path), read_json(waveform_path)
    train_ids, test_ids = split["train_cases"], split["test_cases"]
    require(len(set(test_ids)) == len(test_ids) and not set(train_ids) & set(test_ids), "split overlap/duplicate holdout")
    require(set(stats["train_units"]) == set(train_ids), "frame statistics training-fold mismatch")
    require(run.get("n_cases") == len(test_ids), "held-out case count mismatch")
    for key, source in (("split_sha256", split_path), ("frame_stats_sha256", stats_path),
                        ("waveform_sha256", waveform_path), ("cache_manifest_sha256", cache_path)):
        require(run.get(key) == sha256(source), f"source fingerprint mismatch: {key}")
    steps, pk = stats["frame"]["steps"], int(cfg["peak_index"])
    require(steps == list(range(1120, 1281, 2)), "not the 81 registered physical Fluent steps")
    require(waveform["steps"] == steps and waveform["peak_index"] == pk == 21, "waveform/peak mismatch")
    ev = run.get("evaluation_contract", {})
    require(ev.get("truth_space") == "raw_pa_unclipped", "raw Pa truth contract missing")
    require(ev.get("frame_steps") == steps and ev.get("peak_index") == pk and ev.get("peak_step") == steps[pk], "evaluation physical-step mismatch")
    require(ev.get("target_floor") == stats["floor"] and ev.get("target_eps") == stats["eps"], "floor/epsilon contract mismatch")
    require(ev.get("checkpoint") == "last", "evaluation checkpoint is not last")

    anchor = run.get("anchor_check", {})
    maximum = anchor.get("max_abs_peak_ln_vs_anchor")
    require(finite(maximum) and 0 <= maximum <= 1e-9 and anchor.get("passed") is True, "G0 peak-anchor gate failed")
    require(anchor.get("n_cases") == len(test_ids) and anchor.get("unit") == "ln Pa", "G0 case count/unit mismatch")
    metrics = run["metrics"]["eval_compatible"]
    require(metrics.get("n_frames") == 81 and metrics.get("full_cycle") is True, "incomplete cycle metric")
    for key in KEYS:
        require(finite(metrics.get(key)), f"missing/nonfinite metric: {key}")
    for key in ("frame_r2cb_pa", "frame_r2cb_ln"):
        values = metrics.get(key, {})
        require(set(values) == {str(x) for x in steps}, f"{key} physical steps mismatch")
        require(all(finite(v) for v in values.values()), f"{key} contains nonfinite values")
    require(metrics.get("argmin_frame_step") in steps, "argmin step is not physical")
    require(len(metrics.get("trough_frames_steps", [])) == 20 and set(metrics["trough_frames_steps"]) <= set(steps), "invalid trough frame steps")
    code = run.get("code_sha256", {})
    require(code and "time_transformer.py" in code, "missing frozen code provenance")
    for key, value in code.items():
        require(frozen_digest(queue["fingerprints"], Path(queue["code_package_dir"]) / key) == value, f"run code differs from frozen queue: {key}")
    require(is_digest(run.get("initialization_sha256")), "missing initialization digest")
    require(isinstance(run.get("n_parameters"), int) and run["n_parameters"] > 0, "invalid parameter count")
    for key in ("input_mean", "input_std"):
        values = run.get(key, [])
        require(len(values) == 60 and all(finite(v) for v in values), f"invalid 60D {key}")
    require(all(v > 0 for v in run["input_std"]), "nonpositive input standard deviation")
    context = run.get("context_contract", {})
    hashes = context.get("holdout_index_sha256", {})
    require(set(hashes) == set(test_ids) and all(is_digest(v) for v in hashes.values()), "holdout context-index hashes incomplete")
    require(context.get("count") == cfg["train"]["context_points"] and context.get("seed") == cfg["seed"] and context.get("stream") == "context", "context sampling contract mismatch")
    for key in ("train_seconds", "eval_seconds", "elapsed_seconds"):
        require(finite(run.get(key)) and run[key] >= 0, f"missing/invalid separate timing: {key}")

    history_path = path.parent / "history.jsonl"
    checkpoint_path = path.parent / "ckpt_last.pt"
    history = [json.loads(line) for line in history_path.read_text().splitlines() if line.strip()]
    require(len(history) == cfg["train"]["epochs"], "history does not contain all epochs")
    for index, row in enumerate(history):
        require(row["epoch"] == index and row["step"] == (index + 1) * cfg["train"]["steps_per_epoch"], "history epoch/step sequence mismatch")
        require(all(finite(row.get(k)) for k in ("loss_z", "lr", "elapsed_s")), "history contains nonfinite values")
    require(checkpoint_path.is_file(), "last checkpoint is missing")
    acceptance_path = Path(record.get("acceptance_path", path.parent / "queue_acceptance.json")).resolve()
    acceptance = read_json(acceptance_path)
    require(acceptance.get("schema") == "time_transformer_queue_acceptance_v1" and acceptance.get("passed") is True and acceptance.get("state") == "complete", "per-run queue acceptance not passed")
    require(acceptance.get("config_sha256") == config_hash and acceptance.get("matrix_sha256") == matrix_hash, "per-run acceptance config/matrix hash mismatch")
    require(acceptance.get("code_sha256") == code and acceptance.get("steps") == expected_steps, "per-run acceptance code/steps mismatch")
    require(acceptance.get("fingerprint_sha256") == queue_fingerprint_digest(queue), "per-run acceptance fingerprint mismatch")
    artifacts = {"metrics": path, "history": history_path, "checkpoint": checkpoint_path}
    for key, source in artifacts.items():
        saved = acceptance["artifacts"][key]
        require(Path(saved["path"]).resolve() == source and saved["sha256"] == sha256(source), f"accepted {key} artifact changed")
    return dict(arm=run["arm"], fold=run["fold"], seed=run["seed"], n_cases=run["n_cases"],
                n_steps=run["steps"], n_parameters=run["n_parameters"], metrics={k: metrics[k] for k in KEYS},
                G0=dict(passed=True, max_abs_peak_ln_vs_anchor=maximum),
                timing_seconds={k: run[k] for k in ("train_seconds", "eval_seconds", "elapsed_seconds")},
                sources=dict(metrics=str(path), metrics_sha256=sha256(path), config=str(config_path),
                    actual_config=str(actual_config), config_sha256=config_hash, split=str(split_path),
                    frame_stats=str(stats_path), waveform=str(waveform_path), cache_manifest=str(cache_path),
                    history=str(history_path), checkpoint=str(checkpoint_path), queue_acceptance=str(acceptance_path)),
                evaluation_contract=ev, input_mean=run["input_mean"], input_std=run["input_std"],
                initialization_sha256=run["initialization_sha256"], context_contract=context, code_sha256=code)


def summarize(config_dir):
    config_dir = Path(config_dir).resolve()
    matrix_path = config_dir / "matrix.json"
    matrix = read_json(matrix_path)
    require("output_experiment_dir" in matrix, "matrix must provide output_experiment_dir (old matrices are audit-only)")
    exp = Path(matrix["output_experiment_dir"]).resolve()
    require(not invalidated(exp) and not invalidated(config_dir), "INVALIDATED matrix cannot be summarized")
    jobs = matrix["arms"]
    require(len(jobs) == 12 and len({j["id"] for j in jobs}) == 12, "registered cv3 four-arm matrix must contain 12 unique runs")
    rows, errors, missing = {}, [], []
    queue_path, acceptance_path = exp / "queue_status.json", exp / "queue_acceptance.json"
    queue = read_json(queue_path) if queue_path.is_file() else {}
    acceptance = read_json(acceptance_path) if acceptance_path.is_file() else {}
    if not queue:
        missing.append(str(queue_path))
    if not acceptance:
        missing.append(str(acceptance_path))
    matrix_hash = sha256(matrix_path)
    if queue:
        try:
            # The queue may use a byte-identical config directory in its frozen copy.
            queued_matrix = queue.get("matrix_path") or queue.get("matrix")
            if not queued_matrix:
                candidates = [p for p in queue.get("fingerprints", {}) if Path(p).name == "matrix.json"]
                require(len(candidates) == 1, "queue matrix fingerprint ambiguous/missing")
                queued_matrix = candidates[0]
            require(frozen_digest(queue["fingerprints"], queued_matrix) == matrix_hash, "matrix differs from queued snapshot")
        except (KeyError, ValueError) as error:
            errors.append(str(error))
        if queue.get("aborted"):
            errors.append(f"queue aborted: {queue['aborted']}")
    for job in jobs:
        path = Path(job["done_marker"]).resolve()
        if not path.is_file():
            missing.append(str(path))
            continue
        record = queue.get("jobs", {}).get(job["id"])
        if record is None or record.get("state") in ("pending", "running"):
            missing.append(f"queue completion: {job['id']}")
            continue
        per_acceptance = Path(record.get("acceptance_path", path.parent / "queue_acceptance.json"))
        if not per_acceptance.is_file():
            missing.append(str(per_acceptance.resolve()))
            continue
        try:
            rows[job["id"]] = validate_run(job, (config_dir / job["config"]).resolve(), queue, record, matrix_hash)
        except (OSError, KeyError, ValueError, TypeError) as error:
            errors.append(f"{job['id']}: {error}")
    if acceptance:
        try:
            require(acceptance.get("passed") is True and acceptance.get("state") == "complete", "queue acceptance is not complete/passed")
            require(acceptance.get("matrix_sha256") == matrix_hash, "queue acceptance matrix hash mismatch")
            require(acceptance.get("fingerprint_sha256") == queue_fingerprint_digest(queue), "queue acceptance fingerprint mismatch")
            accepted_jobs = acceptance.get("jobs", {})
            require(set(accepted_jobs) == {j["id"] for j in jobs}, "queue acceptance job coverage mismatch")
            for job_id, row in rows.items():
                rec = accepted_jobs[job_id]
                require(rec.get("passed") is True, f"queue acceptance failed: {job_id}")
                require(rec.get("metrics_sha256") == row["sources"]["metrics_sha256"], f"accepted metrics changed: {job_id}")
                require(rec.get("config_sha256") == row["sources"]["config_sha256"], f"accepted config changed: {job_id}")
                require(acceptance.get("code_sha256") == row["code_sha256"], f"accepted code changed: {job_id}")
        except (KeyError, ValueError, TypeError) as error:
            errors.append(str(error))
    if queue and not queue.get("finished_at"):
        missing.append("queue finished_at")
    by_fold = {}
    for job_id, row in rows.items():
        key = (row["fold"], row["seed"])
        if row["arm"] in by_fold.setdefault(key, {}):
            errors.append(f"duplicate arm {row['arm']} for fold/seed {key}")
        by_fold[key][row["arm"]] = (job_id, row)
    pairing = {}
    for (fold, seed), arms in sorted(by_fold.items()):
        if set(arms) != set(ARMS):
            missing.append(f"four-arm pairing fold={fold} seed={seed}")
            continue
        base, issues = arms[ARMS[0]][1], []
        for arm in ARMS[1:]:
            row = arms[arm][1]
            for key in ("input_mean", "input_std", "initialization_sha256", "context_contract", "n_parameters", "n_steps", "evaluation_contract"):
                if row[key] != base[key]:
                    issues.append(f"{arm}: {key}")
        pairing[f"fold{fold}_s{seed}"] = dict(passed=not issues, mismatches=issues,
            n_parameters=base["n_parameters"], n_steps=base["n_steps"], sources={a: r[0] for a, r in arms.items()})
        if issues:
            errors.append(f"fold{fold}_s{seed} pairing mismatch: {', '.join(issues)}")
    if rows:
        codes = list(rows.values())
        if any(r["code_sha256"] != codes[0]["code_sha256"] for r in codes[1:]):
            errors.append("run code_sha256 differs across matrix")
    if len(rows) == len(jobs) and set(by_fold) != {(f, 1234) for f in (0, 1, 2)}:
        errors.append("registered fold/seed set differs from cv3 folds 0/1/2 and seed 1234")
    status = "invalid" if errors else "incomplete" if missing or len(rows) != len(jobs) else "complete"
    means, paired = {}, {}
    if status == "complete":
        for arm in ARMS:
            values = [r for r in rows.values() if r["arm"] == arm]
            means[arm] = {k: sum(r["metrics"][k] for r in values) / len(values) for k in KEYS}
        for first, second, label in PAIRS:
            differences = []
            for (fold, seed), arms in sorted(by_fold.items()):
                a, b = arms[first][1], arms[second][1]
                differences.append(dict(fold=fold, seed=seed,
                    delta={k: a["metrics"][k] - b["metrics"][k] for k in KEYS},
                    first=a["sources"]["metrics"], second=b["sources"]["metrics"]))
            paired[label] = dict(first=first, second=second, per_fold=differences,
                mean={k: sum(r["delta"][k] for r in differences) / len(differences) for k in KEYS})
    return exp, dict(schema="time_transformer_report_v2", experiment=matrix["experiment"], status=status,
        generated_at=datetime.now(timezone.utc).isoformat(), expected_runs=len(jobs), validated_runs=len(rows),
        missing=sorted(set(missing)), errors=errors, rows=rows, means=means, paired=paired, pairing_checks=pairing,
        G0=dict(passed=True if status == "complete" else None,
            max_abs_peak_ln_vs_anchor=max((r["G0"]["max_abs_peak_ln_vs_anchor"] for r in rows.values()), default=None)),
        sources=dict(matrix=str(matrix_path), matrix_sha256=matrix_hash, queue=str(queue_path), queue_acceptance=str(acceptance_path)),
        references=matrix.get("references", {}),
        note="V5.1 cv3, seed 1234: descriptive development evidence only. No statistical-significance claim or deployment decision. Historical C-raw/T0 are context only, not matched causal controls.")


def markdown(report):
    lines = [f"# {report['experiment']}", "", f"状态：**{report['status']}**；验收 {report['validated_runs']}/{report['expected_runs']} 个正式 run。", "",
             "G0：" + (f"通过；最大峰值锚偏差 {report['G0']['max_abs_peak_ln_vs_anchor']:.3e} ln Pa。" if report['G0']['passed'] is True else "尚未通过完整矩阵验收。")]
    for key, label in (("missing", "缺失/未完成"), ("errors", "验收错误")):
        if report[key]:
            lines += ["", f"{label}：", ""] + [f"- {x}" for x in report[key]]
    if report["status"] == "complete":
        lines += ["", "三折算术均值（Pa R²_cb 为病例等权逐帧均值；ln 与时间形态单列）：", "",
            "| 臂 | cycle Pa | trough Pa | TAWSS Pa | cycle ln | peak Pa | peak p90（帧）| 时序相关 | 幅值误差 ln |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for arm, m in report["means"].items():
            lines.append(f"| {arm} | {m['cycle_r2cb_pa']:.4f} | {m['trough_r2cb_pa']:.4f} | {m['tawss_r2cb_pa']:.4f} | {m['cycle_r2cb_ln']:.4f} | {m['peak_r2cb_pa']:.4f} | {m['peak_time_err_p90_frames']:.2f} | {m['ts_corr_ln']:.4f} | {m['amp_err_ln_med']:.4f} |")
        lines += ["", "同折配对差（前者减后者）：", ""]
        for label, result in report["paired"].items():
            values = "; ".join(f"f{r['fold']}={r['delta']['cycle_r2cb_pa']:+.4f}" for r in result["per_fold"])
            lines.append(f"- {label}: cycle Pa {values}；均值 {result['mean']['cycle_r2cb_pa']:+.4f}；谷底均值 {result['mean']['trough_r2cb_pa']:+.4f}。")
    else:
        lines += ["", "完整矩阵尚未验收，不生成四臂均值或完整配对结论。"]
    if report["rows"]:
        lines += ["", "已验收单 run 的预算与来源（耗时为秒，训练、评估、总耗时分别报告）：", "",
            "| Run | steps | 参数量 | G0 最大差 ln Pa | 训练 | 评估 | 总耗时 | 原始指标 |",
            "|---|---:|---:|---:|---:|---:|---:|---|"]
        for run_id, row in report["rows"].items():
            t = row["timing_seconds"]
            lines.append(f"| {run_id} | {row['n_steps']} | {row['n_parameters']} | {row['G0']['max_abs_peak_ln_vs_anchor']:.2e} | {t['train_seconds']:.1f} | {t['eval_seconds']:.1f} | {t['elapsed_seconds']:.1f} | [metrics]({row['sources']['metrics']}) |")
    lines += ["", "证据入口：", ""]
    for key in ("matrix", "queue", "queue_acceptance"):
        lines.append(f"- [{key}]({report['sources'][key]})")
    lines += ["", report["note"], "", "静态几何与协议相位输入；没有逐帧观测的速度/压力。noattn 是相同 Transformer 的对角屏蔽，比较仅识别本冻结表示和训练预算下的跨帧交互增量。"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", required=True)
    args = parser.parse_args()
    try:
        exp, report = summarize(args.config_dir)
    except (OSError, KeyError, ValueError, TypeError) as error:
        parser.exit(2, f"report refused: {error}\n")
    exp.mkdir(parents=True, exist_ok=True)
    for name, content in (("report.json", json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"),
                          ("report.md", markdown(report))):
        target, temporary = exp / name, exp / (name + ".tmp")
        temporary.write_text(content)
        temporary.replace(target)
        print(target)
    print(f"status={report['status']} validated={report['validated_runs']}/{report['expected_runs']}")
    if report["status"] == "invalid":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
