#!/usr/bin/env python3
"""Audit and summarize the 2026-07-21 PointNet++ SA1-scale matrix.

与 analyze_sa_grouping_single_seed_results.py 同源：冻结 manifest、run 产物与
几何审计为真源；只写分析 JSON/CSV，并以固定标题小节幂等回填三份跟踪文档与 XLSX。
支持部分完成模式：eval 产物缺失的臂记为 pending，重跑本脚本自动刷新全部产物。
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from copy import copy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill


ROOT = Path(__file__).resolve().parents[2]
PREFLIGHT = ROOT / "training_wss_min/preflight"
PREPARED = PREFLIGHT / "pointnetpp_sa1_scale_prepared.json"
SUBMISSION = PREFLIGHT / "pointnetpp_sa1_scale_submission.json"
GEOM = PREFLIGHT / "pointnetpp_sa1_scale_geometry_audit.json"
GPU = PREFLIGHT / "pointnetpp_sa1_scale_gpu_preflight.json"
ANALYSIS_JSON = PREFLIGHT / "pointnetpp_sa1_scale_results_analysis.json"
SUMMARY_CSV = PREFLIGHT / "pointnetpp_sa1_scale_results_summary.csv"
PER_CASE_CSV = PREFLIGHT / "pointnetpp_sa1_scale_per_case_deltas.csv"
DOC_MATRIX = ROOT / "docs/02-推进与变更/WSS最小化_PointNet_baseline实验矩阵与进度跟踪.md"
DOC_LOG = ROOT / "docs/02-推进与变更/WSS最小化_代码修改与实验推进记录.md"
DOC_TRACK = ROOT / "docs/02-推进与变更/WSS最小化_训练实验跟踪.md"
XLSX = ROOT / "docs/03-汇报材料/WSS_PointNet实验矩阵与结果汇总last.xlsx"
ROUND = "2026-07-21｜PointNet++ SA1-scale 矩阵结果（全点/10k/降center×大k/宽度/Stem；single seed=1234；historical test27）"
TRACK_MARK = "## SA1-scale 矩阵（"

GROUPING_RUNS = ROOT / "training_wss_min/runs/pointnetpp_sa_grouping_single_seed/outputs"
GROUPING_CFGS = ROOT / "training_wss_min/configs/pointnetpp_sa_grouping_single_seed_20260720"

CONTROLS = {
    "Q1V_random5000_ball16_anchor": {
        "label": "Q1V random5000 + ball16（复用锚点）", "family": "control",
        "run_dir": ROOT / "training_wss_min/runs/pointnetpp_v4/outputs/ag_aaa_v4_stratified_sa3_random5000_fpscenter_same",
        "config": ROOT / "training_wss_min/configs/pointnetpp_v4/ag_aaa_v4_stratified_sa3_random5000_fpscenter_same.json",
    },
    "q1v_sa1_knn8_cover": {
        "label": "Q1V KNN-8-cover w32（复用）", "family": "control",
        "run_dir": GROUPING_RUNS / "q1v_sa1_knn8_cover",
        "config": GROUPING_CFGS / "q1v_sa1_knn8_cover.json",
    },
    "q1v_sa1_knn10_cover": {
        "label": "Q1V KNN-10-cover w32（复用）", "family": "control",
        "run_dir": GROUPING_RUNS / "q1v_sa1_knn10_cover",
        "config": GROUPING_CFGS / "q1v_sa1_knn10_cover.json",
    },
    "q1v_n16_w64": {
        "label": "Q1V ball16 w64（复用）", "family": "control",
        "run_dir": GROUPING_RUNS / "q1v_n16_w64",
        "config": GROUPING_CFGS / "q1v_n16_w64.json",
    },
}

LABELS = {
    "d1_allpts_fixed500_k64": "D1-fixed 全点 + k64",
    "d1_allpts_fixed500_k128": "D1-fixed 全点 + k128",
    "d1_allpts_fixed500_k256": "D1-fixed 全点 + k256",
    "d1_allpts_prop10pct_k64": "D1-prop 全点比例center + k64（batch2）",
    "d1_allpts_prop10pct_k128": "D1-prop 全点比例center + k128（batch2）",
    "d1_allpts_prop10pct_k256": "D1-prop 全点比例center + k256（batch2）",
    "d2_rand5000_c250_k64": "D2 c250 × k64",
    "d2_rand5000_c250_k128": "D2 c250 × k128",
    "d2_rand5000_c125_k64": "D2 c125 × k64",
    "d2_rand5000_c125_k128": "D2 c125 × k128",
    "d3_rand10000_fixed500_k64": "D3 random10000 + k64",
    "d3_rand10000_fixed500_k128": "D3 random10000 + k128",
    "d3_rand10000_fixed500_k256": "D3 random10000 + k256",
    "d4_rand5000_knn8_cover_w64": "D4 KNN-8-cover w64",
    "d4_rand5000_knn10_cover_w64": "D4 KNN-10-cover w64",
    "d5_rand5000_knn8_cover_w64_stem32_64": "D5-A stem 6→32→64（w64 KNN-8-cover）",
    "bridge_rand5000_fixed500_k64": "bridge random5000 + 500c + k64",
}


def load(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, list):
        return all(finite(v) for v in value)
    return True


def subset(expected: Any, actual: Any) -> bool:
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(k in actual and subset(v, actual[k]) for k, v in expected.items())
    if isinstance(expected, list):
        return isinstance(actual, list) and len(expected) == len(actual) and all(subset(a, b) for a, b in zip(expected, actual))
    return expected == actual


def metric_row(metrics: dict[str, Any]) -> dict[str, Any]:
    t = metrics["test"]
    n = t["normalized"]
    return {
        "physical_r2_cb": t["field_casebalanced"]["r2"],
        "physical_r2": t["field"]["r2"],
        "physical_casemean_r2": t["aggregate"]["r2_casemean"],
        "physical_r2_negative_cases": t["aggregate"]["r2_negative_cases"],
        "physical_mae": t["field_casebalanced"]["mae"],
        "physical_rmse": t["field_casebalanced"]["rmse"],
        "physical_high_wss_r2": t["regional_field"]["high_wss"]["r2"],
        "physical_high_wss_mae": t["regional_field"]["high_wss"]["mae"],
        "physical_spearman": t["hotspot"]["spearman_all_casemean"],
        "physical_top10_iou": t["hotspot"]["top10_iou_casemean"],
        "normalized_r2_cb": n["field_casebalanced"]["r2"],
        "normalized_r2": n["field"]["r2"],
        "normalized_casemean_r2": n["aggregate"]["r2_casemean"],
        "normalized_r2_negative_cases": n["aggregate"]["r2_negative_cases"],
        "normalized_p99_ratio": n["calibration"]["p99_pred_true_ratio"],
        "parameters": t["efficiency"].get("parameters"),
    }


def contract_key_from_config(cfg: dict[str, Any]) -> tuple:
    data, model = cfg["data"], cfg["model"]
    counts = model.get("sa_center_counts", [])
    return (
        data.get("support_sampling") or data.get("sampling"),
        int(data.get("support_n_points") or data.get("wall_n_points") or 0),
        (model.get("sa_grouping") or ["ball"])[0],
        int(model.get("sa_nsample", [16])[0]),
        int(counts[0]) if counts else -1,
        float(model.get("sa_ratios", [0.25])[0]),
        float(model.get("sa_radius", [0.05])[0]),
    )


def geometry_map() -> dict[tuple, dict[str, Any]]:
    data = load(GEOM)
    assert data["status"] == "passed", "geometry audit did not pass"
    result: dict[tuple, dict[str, Any]] = {}
    for row in data["results"]:
        key = tuple(row["contract"][:7])
        result[key] = {
            "coverage_min": row["coverage_min"], "coverage_median": row["coverage_median"],
            "max_pair_overlap": row["max_pair_overlap"],
            "group_size_min": row["group_size_min"], "group_size_max": row["group_size_max"],
            "sampled_audit": row.get("sampled_audit", False),
            "center_mode": row.get("center_mode"),
        }
    return result


def get_history_tail(run: Path) -> dict[str, Any] | None:
    history = run / "history.jsonl"
    if not history.exists():
        return None
    last = None
    for line in history.read_text(encoding="utf-8").splitlines():
        if line.strip():
            last = json.loads(line)
    return last


def audit_csv(path: Path, expected_cases: int) -> tuple[bool, list[str], list[dict[str, str]]]:
    problems: list[str] = []
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != expected_cases:
        problems.append(f"{path.name}: {len(rows)} rows (expected {expected_cases})")
    case_ids = [r.get("case", "") for r in rows]
    if len(set(case_ids)) != expected_cases or any(not c for c in case_ids):
        problems.append(f"{path.name}: duplicate/missing case IDs")
    for r in rows:
        for key, value in r.items():
            if key in ("case", "partition") or value in (None, ""):
                continue
            try:
                if not math.isfinite(float(value)):
                    problems.append(f"{path.name}: non-finite {key}")
                    break
            except ValueError:
                problems.append(f"{path.name}: nonnumeric {key}")
                break
    return not problems, problems, rows


def run_completed(run: Path) -> bool:
    return all((run / "eval" / c / "metrics.json").is_file() for c in ("ckpt_best", "ckpt_last"))


def audit_run(exp_id: str, meta: dict[str, Any], expected_sha: str | None,
              geom: dict[tuple, dict[str, Any]], is_new: bool) -> dict[str, Any]:
    run = Path(meta["run_dir"])
    source_config = Path(meta["config"])
    frozen = load(source_config)
    runtime = load(run / "config.json")
    checks: dict[str, bool] = {}
    problems: list[str] = []
    source_sha = sha256(source_config)
    checks["prepared_sha256_match"] = expected_sha is None or source_sha == expected_sha
    checks["runtime_preserves_frozen_config"] = subset(frozen, runtime)
    checks["seed_1234"] = runtime["train"].get("seed") == 1234
    split_path = Path(runtime["data"].get("split_path", ""))
    split = load(split_path) if split_path.is_file() else {}
    checks["split_106_0_27"] = (
        len(split.get("train_cases", [])) == 106
        and len(split.get("val_cases", [])) == 0
        and len(split.get("test_cases", [])) == 27
        and "train106" in runtime["data"].get("wss_stats_path", "")
    )
    checks["epochs_400"] = runtime["train"].get("epochs") == 400
    history = [json.loads(x) for x in (run / "history.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    checks["history_complete_400"] = len(history) == 400 and [x.get("epoch") for x in history] == list(range(400))
    checks["ckpt_best_and_last"] = (run / "ckpt_best.pt").is_file() and (run / "ckpt_last.pt").is_file()
    checks["selection_rule_train_loss"] = (
        runtime["train"].get("selection_rule") == "train_loss"
        and runtime["train"].get("ckpt_metric") == "train_loss"
    )
    metric_docs: dict[str, Any] = {}
    all_csv_rows: dict[str, list[dict[str, str]]] = {}
    checks["best_last_eval_complete"] = True
    checks["per_case_27_complete"] = True
    checks["no_nan_inf"] = True
    for checkpoint in ("ckpt_best", "ckpt_last"):
        metric_path = run / "eval" / checkpoint / "metrics.json"
        csv_path = run / "eval" / checkpoint / "per_case_metrics.csv"
        if not metric_path.is_file() or not csv_path.is_file():
            checks["best_last_eval_complete"] = False
            problems.append(f"missing {checkpoint} eval artifact")
            continue
        metrics = load(metric_path)
        metric_docs[checkpoint] = metrics
        ok, csv_problems, rows = audit_csv(csv_path, 27)
        all_csv_rows[checkpoint] = rows
        checks["per_case_27_complete"] &= ok and len(metrics.get("test", {}).get("per_case", {})) == 27
        checks["no_nan_inf"] &= finite(metrics) and not csv_problems
        problems.extend(csv_problems)
    g = geom.get(contract_key_from_config(runtime))
    # knn_cover 的硬门只有 coverage=100%；高重叠是记录性局限而非协议失败
    checks["geometry_hard_gate"] = (not is_new) or bool(g and g["coverage_min"] == 1.0)
    for name, good in checks.items():
        if not good and not any(name in x for x in problems):
            problems.append(name)
    tail = get_history_tail(run)
    best = metric_docs.get("ckpt_best")
    last = metric_docs.get("ckpt_last")
    return {
        "experiment_id": exp_id,
        "label": LABELS.get(exp_id, meta.get("label", exp_id)),
        "is_new_run": is_new, "family": meta.get("family", "control"),
        "run_dir": str(run), "config": str(source_config),
        "source_config_sha256": source_sha, "prepared_config_sha256": expected_sha,
        "status": "completed", "checks": checks, "passed": all(checks.values()),
        "problems": problems,
        "config_data": runtime["data"], "config_model": runtime["model"], "config_train": runtime["train"],
        "metrics_best": metric_row(best) if best else {},
        "metrics_last": metric_row(last) if last else {},
        "final_train_loss": tail.get("train_loss") if tail else None,
        "train_sampled_points": tail.get("train_sampled_points") if tail else None,
        "training_seconds": tail.get("elapsed_s") if tail else None,
        "best_last": load(run / "eval/best_last_comparison.json") if (run / "eval/best_last_comparison.json").exists() else None,
        "geometry": g,
        "per_case_best": all_csv_rows.get("ckpt_best", []),
    }


def delta(a: Any, b: Any) -> Any:
    return None if a is None or b is None else a - b


def table_rows(ids: list[str], results: dict[str, dict[str, Any]], comparator) -> list[dict[str, Any]]:
    rows = []
    for exp_id in ids:
        if exp_id not in results or results[exp_id]["status"] != "completed":
            continue
        result = results[exp_id]
        ref_id = comparator(exp_id)
        ref = results[ref_id]
        m, b = result["metrics_best"], ref["metrics_best"]
        rows.append({
            "experiment_id": exp_id, "label": result["label"], "reference_id": ref_id,
            "r2_cb": m["physical_r2_cb"], "delta_r2_cb": delta(m["physical_r2_cb"], b["physical_r2_cb"]),
            "normalized_r2_cb": m["normalized_r2_cb"], "delta_normalized_r2_cb": delta(m["normalized_r2_cb"], b["normalized_r2_cb"]),
            "case_mean_r2": m["physical_casemean_r2"], "delta_case_mean_r2": delta(m["physical_casemean_r2"], b["physical_casemean_r2"]),
            "negative_cases": m["physical_r2_negative_cases"], "mae": m["physical_mae"], "delta_mae": delta(m["physical_mae"], b["physical_mae"]),
            "rmse": m["physical_rmse"], "delta_rmse": delta(m["physical_rmse"], b["physical_rmse"]),
            "spearman": m["physical_spearman"], "delta_spearman": delta(m["physical_spearman"], b["physical_spearman"]),
            "top10_iou": m["physical_top10_iou"], "delta_top10_iou": delta(m["physical_top10_iou"], b["physical_top10_iou"]),
            "high_wss_r2": m["physical_high_wss_r2"], "delta_high_wss_r2": delta(m["physical_high_wss_r2"], b["physical_high_wss_r2"]),
            "p99_ratio": m["normalized_p99_ratio"], "delta_p99_ratio": delta(m["normalized_p99_ratio"], b["normalized_p99_ratio"]),
            "parameters": m["parameters"], "training_seconds": result["training_seconds"],
            "geometry": result["geometry"],
        })
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({k for r in rows for k in r if k != "geometry"}) + ["geometry"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            payload = {k: row.get(k) for k in fields}
            if isinstance(payload.get("geometry"), dict):
                payload["geometry"] = json.dumps(payload["geometry"], ensure_ascii=False, sort_keys=True)
            writer.writerow(payload)


def f(x: Any, digits: int = 4) -> str:
    return "—" if x is None else f"{x:.{digits}f}" if isinstance(x, float) else str(x)


def md_table(rows: list[dict[str, Any]], geometry: bool = False) -> str:
    headers = ["配置", "物理R²_cb（Δ）", "归一化R²_cb（Δ）", "病例均值R²（Δ）", "负例",
               "MAE（Δ）", "Spearman（Δ）", "top10 IoU（Δ）", "high-WSS R²（Δ）", "p99比（Δ）"]
    if geometry:
        headers += ["覆盖率 min", "最大重叠", "组大小"]
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        cells = [
            r["label"], f(r["r2_cb"]) + " (" + f(r["delta_r2_cb"]) + ")",
            f(r["normalized_r2_cb"]) + " (" + f(r["delta_normalized_r2_cb"]) + ")",
            f(r["case_mean_r2"]) + " (" + f(r["delta_case_mean_r2"]) + ")",
            f(r["negative_cases"]), f(r["mae"]) + " (" + f(r["delta_mae"]) + ")",
            f(r["spearman"]) + " (" + f(r["delta_spearman"]) + ")",
            f(r["top10_iou"]) + " (" + f(r["delta_top10_iou"]) + ")",
            f(r["high_wss_r2"]) + " (" + f(r["delta_high_wss_r2"]) + ")",
            f(r["p99_ratio"]) + " (" + f(r["delta_p99_ratio"]) + ")",
        ]
        if geometry:
            g = r["geometry"] or {}
            cells += [f(g.get("coverage_min")), f(g.get("max_pair_overlap")),
                      f(g.get("group_size_min"), 0) + "–" + f(g.get("group_size_max"), 0)]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def build_section(tables: dict[str, list[dict[str, Any]]], results: dict[str, dict[str, Any]],
                  pending: list[str], d5b: dict[str, Any]) -> str:
    done = sum(1 for r in results.values() if r["is_new_run"] and r["status"] == "completed")
    passed = sum(1 for r in results.values() if r["is_new_run"] and r["status"] == "completed" and r["passed"])
    status = f"{done}/17 新 run 完成且 {passed}/{done} 审计通过"
    if pending:
        status += f"；待跑：{', '.join(pending)}（跑完重跑本分析脚本自动刷新本节与 XLSX）"
    verdict = d5b["verdict"]
    return f"""
## {ROUND}

**状态**：{status}。锚点 Q1V random5000+ball16（0.2763）；导师标准 KNN-8-cover w32（0.2589）。协议同 SA1 分组矩阵（106/0/27、seed1234、400ep、train-loss 选模、train106 global log-z、legacy_vertex、物理 R²_cb 主指标）。**test27 为多轮复用的工程比较集，本轮全部是同协议筛查，非独立确认**；D1-prop 族 batch=2（其余 batch=8）存在优化混杂，跨家族比较需声明。审计逐项检查冻结 SHA、seed、split、400 epoch、best/last eval、27 例 CSV、有限数值与 knn_cover 覆盖硬门（全部 100%）。

### 1) 点数标度（固定 500/125/32 center；Δ 相对 Q1V 锚点）

{md_table(tables['points_scaling'])}

### 2) 比例 center vs 固定 center（全点；Δ 相对同 k 的 D1-fixed；batch 2 vs 8 混杂）

{md_table(tables['prop_vs_fixed'], geometry=True)}

### 3) 降 center × 大邻域（random5000；Δ 相对 KNN-8-cover w32）

{md_table(tables['d2_centers'], geometry=True)}

### 4) width=64 与 Stem 变种（Δ 相对各自父配置）

{md_table(tables['width_stem'])}

**D5-B 触发判定**：{verdict}

**判读与下一步**：①点数标度全线未超锚点——全点/10k + 大 k 的 8 个臂全部低于 random5000+ball16（0.2763），bridge 表明 5000 点下 k64 本身就 -0.0324，点数放大到 10k/全点没有补回该损失；"用全部原始 CFD 点"在当前 500/125/32 center 协议下不成立。家族内 k256 一致优于 k64/k128（D3、D1-fixed 同趋势），但都不及小邻域基线。②比例 center 全败：三个 k 全部低于同 k 的固定 center（k64/k128/k256 分别 -0.0328/-0.0173/-0.0448），负例也更多（含 batch2 混杂），不支持"跨队列 center 密度一致"假设，方向关闭。③**降 center × 大邻域是本轮唯一正向家族**：c125×k128=0.2765（+0.0176 vs KNN-8-cover），与 Q1V 锚点打平（+0.0002），high-WSS R²=-0.504 为全轮最好，且趋势单调——同 k 下 center 500→250→125 递增、同 center 下 k64→k128 递增；等预算对角（250×64 vs 125×128）由"更少 center + 更大邻域"一侧胜出。建议下一轮沿此方向延伸（c125×k256、c64×k128/k256）并将 c125×k128 列为 3-seed/独立确认候选。④w64 在 cover 分组下显著回退（KNN-8/10-cover 从 0.2589/0.2425 掉到 0.2183/0.2172），比 ball16 的 w64 效应（-0.016）严重得多；D5-A 瓶颈 Stem 相对标准 w64 Stem +0.0133、high-WSS +0.052，但绝对值仍低于一切 w32 基线，且两者 final train_loss 几乎相同——"容量不足"证据弱，**建议不自动提交 D5-B**，与导师确认后再定。另注意 bridge 的归一化 R²_cb=0.6139 为本轮最高，物理/归一化排名分裂的既有模式延续，主指标仍按预注册的物理 R²_cb。

真源：`training_wss_min/preflight/pointnetpp_sa1_scale_results_analysis.json`、`pointnetpp_sa1_scale_results_summary.csv`、`pointnetpp_sa1_scale_per_case_deltas.csv`。
""".strip() + "\n"


def append_docs(section: str) -> None:
    for doc in (DOC_MATRIX, DOC_LOG):
        text = doc.read_text(encoding="utf-8")
        if "## " + ROUND in text:
            text = text[:text.index("## " + ROUND)].rstrip() + "\n"
        doc.write_text(text.rstrip() + "\n\n" + section, encoding="utf-8")
    # 训练实验跟踪：新轮次置顶，替换既有 SA1-scale 小节（提交时的待跑占位节或
    # 本脚本先前生成的结果节，保证重跑幂等）。
    text = DOC_TRACK.read_text(encoding="utf-8")
    start = text.find("## " + ROUND)
    if start == -1:
        start = text.find(TRACK_MARK)
    if start == -1:
        start = text.find("\n## ") + 1
        end = start
    else:
        nxt = text.find("\n## ", start)
        end = len(text) if nxt == -1 else nxt + 1
    DOC_TRACK.write_text(text[:start] + section + "\n" + text[end:], encoding="utf-8")


def copy_row_style(ws, source_row: int, target_row: int, columns: int) -> None:
    for col in range(1, columns + 1):
        source, target = ws.cell(source_row, col), ws.cell(target_row, col)
        if source.has_style:
            target._style = copy(source._style)
        if source.number_format:
            target.number_format = source.number_format
        target.alignment = copy(source.alignment)
        target.protection = copy(source.protection)


def support_desc(data: dict[str, Any]) -> tuple[str, Any]:
    n = int(data.get("support_n_points") or data.get("wall_n_points") or 0)
    if n <= 0:
        return "全点（9.3k–103.7k）", "全点"
    return str(n), n


def excel_values(r: dict[str, Any], exp_id: str) -> list[Any]:
    cfg = r["config_model"]
    data = r["config_data"]
    m = r["metrics_best"]
    counts = cfg.get("sa_center_counts", [])
    centers = "/".join(str(c) for c in counts) if counts else "比例0.1N/0.25/0.25"
    k = cfg.get("sa_nsample", [16])[0]
    grouping = (cfg.get("sa_grouping") or ["ball"])[0]
    stem = cfg.get("stem_channels") or []
    stem_txt = f"；stem 6→{'→'.join(str(c) for c in stem)}" if stem else ""
    sup_txt, sup_val = support_desc(data)
    raw = load(Path(r["run_dir"]) / "eval/ckpt_best/metrics.json")["test"]
    norm = raw["normalized"]
    return [
        r["label"], "106/0/27（historical test27；single seed=1234）", "PointNet++ SA3",
        f"{sup_txt}→{centers}；nsample={k}；width={cfg['width']}；SA1={grouping}{stem_txt}；batch={r['config_train'].get('batch_cases')}",
        f"{data.get('support_sampling')} / SAME；FPS center", sup_val,
        m["physical_r2_cb"], m["physical_r2"], m["physical_casemean_r2"], raw["aggregate"]["r2_casemed"], raw["aggregate"]["r2_casep10"],
        f"{m['physical_r2_negative_cases']}/27", raw["field"]["rmse"], raw["field"]["mae"], raw["field"]["nrmse_range"], raw["aggregate"]["nrmse_casemean"],
        raw["field"].get("nmae_range"), raw["aggregate"].get("nmae_casemean"), m["physical_high_wss_r2"], raw["calibration"]["top10_pred_true_ratio"],
        m["physical_top10_iou"], m["normalized_r2_cb"], m["normalized_r2"], m["normalized_casemean_r2"], norm["aggregate"]["r2_casemed"], norm["aggregate"]["r2_casep10"],
        norm["field_casebalanced"]["mae"], norm["field_casebalanced"]["rmse"], norm["field"]["nrmse_range"], norm["aggregate"]["nrmse_casemean"],
        norm["field"].get("nmae_range"), norm["aggregate"].get("nmae_casemean"), m["physical_spearman"], raw["hotspot"]["spearman_high_wss_casemean"],
        norm["calibration"]["top10_pred_true_ratio"], norm["calibration"]["p99_pred_true_ratio"], exp_id,
    ]


def update_xlsx(results: dict[str, dict[str, Any]], tables: dict[str, list[dict[str, Any]]], new_ids: list[str]) -> None:
    wb = load_workbook(XLSX, data_only=False)
    for ws in (wb["实验矩阵总览"], wb["教师汇报视图"], wb["汇总对比"]):
        for row in range(ws.max_row, 0, -1):
            if ws.cell(row, 1).value == ROUND:
                ws.delete_rows(row, ws.max_row - row + 1)
                break
    ids = [i for i in new_ids if results[i]["status"] == "completed"]
    overall = wb["实验矩阵总览"]
    start = overall.max_row + 1
    overall.merge_cells(start_row=start, start_column=1, end_row=start, end_column=37)
    cell = overall.cell(start, 1, ROUND)
    cell.font = Font(bold=True, color="FFFFFF")
    cell.fill = PatternFill("solid", fgColor="1F4E78")
    for i, exp_id in enumerate(ids, start + 1):
        copy_row_style(overall, 19, i, 37)
        for col, value in enumerate(excel_values(results[exp_id], exp_id), 1):
            overall.cell(i, col, value)
    teacher = wb["教师汇报视图"]
    start_teacher = teacher.max_row + 1
    teacher.merge_cells(start_row=start_teacher, start_column=1, end_row=start_teacher, end_column=15)
    cell = teacher.cell(start_teacher, 1, ROUND)
    cell.font = Font(bold=True, color="FFFFFF")
    cell.fill = PatternFill("solid", fgColor="1F4E78")
    for i, exp_id in enumerate(ids, start_teacher + 1):
        copy_row_style(teacher, 24, i, 15)
        r, m = results[exp_id], results[exp_id]["metrics_best"]
        raw = load(Path(r["run_dir"]) / "eval/ckpt_best/metrics.json")["test"]
        sup_txt, _ = support_desc(r["config_data"])
        values = ["SA1-scale", r["label"], "PointNet++ SA3", f"{sup_txt} / SAME",
                  m["physical_r2_cb"], m["physical_casemean_r2"], raw["field"]["rmse"],
                  raw["field"]["nrmse_range"], raw["normalized"]["field"]["nrmse_range"],
                  m["physical_high_wss_r2"], m["normalized_r2_cb"],
                  raw["normalized"]["field"]["nrmse_range"],
                  m["physical_spearman"], m["physical_top10_iou"], m["normalized_r2"]]
        for col, value in enumerate(values, 1):
            teacher.cell(i, col, value)
    summary = wb["汇总对比"]
    row = summary.max_row + 1
    summary.merge_cells(start_row=row, start_column=1, end_row=row, end_column=15)
    summary.cell(row, 1, ROUND).font = Font(bold=True, color="FFFFFF")
    summary.cell(row, 1).fill = PatternFill("solid", fgColor="1F4E78")
    row += 1
    for title, rows in tables.items():
        summary.merge_cells(start_row=row, start_column=1, end_row=row, end_column=15)
        summary.cell(row, 1, title + "（Δ 相对 reference_id）").font = Font(bold=True)
        row += 1
        headers = ["配置", "对照", "物理R²_cb", "ΔR²_cb", "归一化R²_cb", "Δ", "病例均值R²", "负例", "MAE", "ΔMAE", "Spearman", "Δ", "top10 IoU", "Δ", "high-WSS R²"]
        for col, value in enumerate(headers, 1):
            summary.cell(row, col, value)
        for col in range(1, 16):
            summary.cell(row, col).font = Font(bold=True)
        row += 1
        for r in rows:
            values = [r["label"], r["reference_id"], r["r2_cb"], r["delta_r2_cb"], r["normalized_r2_cb"], r["delta_normalized_r2_cb"], r["case_mean_r2"], r["negative_cases"], r["mae"], r["delta_mae"], r["spearman"], r["delta_spearman"], r["top10_iou"], r["delta_top10_iou"], r["high_wss_r2"]]
            for col, value in enumerate(values, 1):
                summary.cell(row, col, value)
            row += 1
    wb.save(XLSX)
    check = load_workbook(XLSX, data_only=False)
    for sheet in ("实验矩阵总览", "教师汇报视图", "汇总对比"):
        assert ROUND in [c.value for c in check[sheet]["A"]], sheet


def d5b_trigger(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    a = results.get("d5_rand5000_knn8_cover_w64_stem32_64")
    parent = results.get("d4_rand5000_knn8_cover_w64")
    if not a or not parent or a["status"] != "completed" or parent["status"] != "completed":
        return {"verdict": "D5-A 或父配置未完成，暂不判定。", "triggered": None}
    da = a["metrics_best"]["physical_r2_cb"] - parent["metrics_best"]["physical_r2_cb"]
    tl_a, tl_p = a["final_train_loss"], parent["final_train_loss"]
    triggered = da >= 0
    verdict = (
        f"D5-A 物理 R²_cb={a['metrics_best']['physical_r2_cb']:.4f} vs 父 d4_knn8_cover_w64 "
        f"{parent['metrics_best']['physical_r2_cb']:.4f}（Δ={da:+.4f}）；final train_loss {tl_a:.4f} vs {tl_p:.4f}。"
    )
    if triggered:
        verdict += "满足主指标不劣条件，若判读认为仍有欠拟合余量，可用 prepare/submit --variant-b 提交 D5-B（stem 6→256→512→64）。"
    else:
        verdict += "瓶颈 Stem 未优于标准 Stem，按预注册规则不触发 D5-B。"
    return {"verdict": verdict, "triggered": triggered, "delta_r2_cb": da,
            "train_loss_stem_a": tl_a, "train_loss_parent": tl_p}


def main() -> None:
    prepared = load(PREPARED)
    submission = load(SUBMISSION)
    assert prepared["status"] == "prepared" and submission["status"] == "submitted"
    assert load(GPU)["status"] == "passed"
    geom = geometry_map()
    results: dict[str, dict[str, Any]] = {}
    pending: list[str] = []
    new_ids: list[str] = []
    for item in prepared["configs"]:
        exp_id = item["experiment_id"]
        new_ids.append(exp_id)
        if not run_completed(Path(item["run_dir"])):
            pending.append(exp_id)
            results[exp_id] = {"experiment_id": exp_id, "label": LABELS.get(exp_id, exp_id),
                               "is_new_run": True, "family": item["family"], "status": "pending",
                               "run_dir": item["run_dir"], "passed": False, "problems": ["run not finished"]}
            continue
        results[exp_id] = audit_run(exp_id, item, item["sha256"], geom, is_new=True)
    for exp_id, meta in CONTROLS.items():
        results[exp_id] = audit_run(exp_id, meta, None, geom, is_new=False)
    anchor = "Q1V_random5000_ball16_anchor"
    knn8c = "q1v_sa1_knn8_cover"
    tables = {
        "points_scaling": table_rows(
            [anchor, "bridge_rand5000_fixed500_k64",
             "d3_rand10000_fixed500_k64", "d3_rand10000_fixed500_k128", "d3_rand10000_fixed500_k256",
             "d1_allpts_fixed500_k64", "d1_allpts_fixed500_k128", "d1_allpts_fixed500_k256"],
            results, lambda _: anchor),
        "prop_vs_fixed": table_rows(
            ["d1_allpts_prop10pct_k64", "d1_allpts_prop10pct_k128", "d1_allpts_prop10pct_k256"],
            results, lambda x: x.replace("prop10pct", "fixed500")),
        "d2_centers": table_rows(
            [knn8c, "bridge_rand5000_fixed500_k64",
             "d2_rand5000_c250_k64", "d2_rand5000_c125_k64",
             "d2_rand5000_c250_k128", "d2_rand5000_c125_k128"],
            results, lambda _: knn8c),
        "width_stem": table_rows(
            ["q1v_n16_w64", knn8c, "q1v_sa1_knn10_cover",
             "d4_rand5000_knn8_cover_w64", "d4_rand5000_knn10_cover_w64",
             "d5_rand5000_knn8_cover_w64_stem32_64"],
            results,
            lambda x: {"d4_rand5000_knn8_cover_w64": knn8c,
                       "d4_rand5000_knn10_cover_w64": "q1v_sa1_knn10_cover",
                       "d5_rand5000_knn8_cover_w64_stem32_64": "d4_rand5000_knn8_cover_w64"}.get(x, x)),
    }
    d5b = d5b_trigger(results)
    summary_rows = []
    for r in results.values():
        if r["status"] != "completed":
            summary_rows.append({"experiment_id": r["experiment_id"], "label": r["label"],
                                 "family": r.get("family"), "status": "pending"})
            continue
        row = {k: v for k, v in r.items() if k not in {"config_data", "config_model", "config_train", "per_case_best", "best_last", "checks"}}
        row.update(r["metrics_best"])
        summary_rows.append(row)
    write_csv(SUMMARY_CSV, summary_rows)
    per_case_rows = []
    for table_name, rows in tables.items():
        for r in rows:
            exp, ref = results[r["experiment_id"]], results[r["reference_id"]]
            if exp["experiment_id"] == r["reference_id"]:
                continue
            by_case = {x["case"]: x for x in ref["per_case_best"]}
            for case in exp["per_case_best"]:
                base = by_case[case["case"]]
                per_case_rows.append({
                    "matrix": table_name, "experiment_id": r["experiment_id"], "reference_id": r["reference_id"], "case": case["case"],
                    "physical_r2": float(case["physical_overall_r2"]), "delta_physical_r2": float(case["physical_overall_r2"]) - float(base["physical_overall_r2"]),
                    "normalized_r2": float(case["normalized_overall_r2"]), "delta_normalized_r2": float(case["normalized_overall_r2"]) - float(base["normalized_overall_r2"]),
                    "physical_mae": float(case["physical_overall_mae"]), "delta_physical_mae": float(case["physical_overall_mae"]) - float(base["physical_overall_mae"]),
                    "top10_iou": float(case["physical_hotspot_top10_iou"]), "delta_top10_iou": float(case["physical_hotspot_top10_iou"]) - float(base["physical_hotspot_top10_iou"]),
                    "high_wss_r2": float(case["high_wss_r2"]), "delta_high_wss_r2": float(case["high_wss_r2"]) - float(base["high_wss_r2"]),
                })
    write_csv(PER_CASE_CSV, per_case_rows)
    done = [r for r in results.values() if r["is_new_run"] and r["status"] == "completed"]
    analysis = {
        "schema_version": 1, "generated_at": datetime.now(timezone.utc).isoformat(), "round": ROUND,
        "source_of_truth": {"prepared": str(PREPARED), "submission": str(SUBMISSION), "geometry": str(GEOM), "gpu": str(GPU)},
        "slurm_jobs": submission["jobs"],
        "audit": {"new_runs_total": len(new_ids), "new_runs_completed": len(done),
                  "new_runs_passed": sum(r["passed"] for r in done), "pending": pending},
        "d5b_trigger": d5b,
        "results": {k: {kk: vv for kk, vv in v.items() if kk != "per_case_best"} for k, v in results.items()},
        "tables": tables,
        "limitations": [
            "single seed=1234 only; no 3-seed uncertainty estimate.",
            "historical test27 is a repeated engineering comparison set, not an independent confirmation set.",
            "D1-prop family trains with batch_cases=2 (others 8); cross-family deltas carry an optimization confound.",
            "main metrics use ckpt_best selected by train_loss; best/last retained for sensitivity only.",
        ],
    }
    ANALYSIS_JSON.write_text(json.dumps(analysis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    section = build_section(tables, results, pending, d5b)
    append_docs(section)
    update_xlsx(results, tables, new_ids)
    print(json.dumps({"completed": len(done), "pending": pending,
                      "passed": sum(r["passed"] for r in done),
                      "d5b_triggered": d5b.get("triggered"),
                      "xlsx_rows_added": len([i for i in new_ids if results[i]["status"] == "completed"])},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
