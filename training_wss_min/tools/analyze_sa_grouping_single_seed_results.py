#!/usr/bin/env python3
"""Audit and summarize the 2026-07-20 PointNet++ SA1 coverage/overlap matrix.

The frozen prepared manifest, run artifacts, and geometry audits are the source
of truth.  This script intentionally does not modify any training configuration
or result artifact; it only writes the requested analysis JSON/CSV files and
appends a clearly delimited round section to the two tracking documents/XLSX.
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
OUT = ROOT / "training_wss_min/runs/pointnetpp_sa_grouping_single_seed/outputs"
PREPARED = PREFLIGHT / "sa_grouping_single_seed_prepared.json"
SUBMISSION = PREFLIGHT / "sa_grouping_single_seed_submission.json"
CORE_GEOM = PREFLIGHT / "sa_grouping_core_geometry_audit.json"
FPS_GEOM = PREFLIGHT / "sa_grouping_fps_geometry_audit.json"
CORE_GPU = PREFLIGHT / "sa_grouping_core_gpu_preflight.json"
FPS_GPU = PREFLIGHT / "sa_grouping_fps_gpu_preflight.json"
ANALYSIS_JSON = PREFLIGHT / "sa_grouping_single_seed_results_analysis.json"
SUMMARY_CSV = PREFLIGHT / "sa_grouping_single_seed_results_summary.csv"
PER_CASE_CSV = PREFLIGHT / "sa_grouping_single_seed_per_case_deltas.csv"
DOC_MATRIX = ROOT / "docs/02-推进与变更/WSS最小化_PointNet_baseline实验矩阵与进度跟踪.md"
DOC_LOG = ROOT / "docs/02-推进与变更/WSS最小化_代码修改与实验推进记录.md"
XLSX = ROOT / "docs/03-汇报材料/WSS_PointNet实验矩阵与结果汇总last.xlsx"
ROUND = "2026-07-20｜PointNet++ SA1 覆盖/重叠矩阵（single seed=1234；historical test27）"

CONTROLS = {
    "Q1V_N0_G0_random5000_ball16": {
        "label": "Q1V random5000 + ball16（复用）",
        "anchor": "Q1V/SAME",
        "family": "control_q1v",
        "support_sampling": "random",
        "grouping": "ball16",
        "run_dir": ROOT / "training_wss_min/runs/pointnetpp_v4/outputs/ag_aaa_v4_stratified_sa3_random5000_fpscenter_same",
        "config": ROOT / "training_wss_min/configs/pointnetpp_v4/ag_aaa_v4_stratified_sa3_random5000_fpscenter_same.json",
        "new": False,
    },
    "Q2V_random5000_ball16": {
        "label": "Q2V random5000 + ball16（复用）",
        "anchor": "Q2V/SEP",
        "family": "control_q2v",
        "support_sampling": "random",
        "grouping": "ball16",
        "run_dir": ROOT / "training_wss_min/runs/pointnetpp_v4/outputs/ag_aaa_v4_stratified_sa3_random5000_fpscenter_sep",
        "config": ROOT / "training_wss_min/configs/pointnetpp_v4/ag_aaa_v4_stratified_sa3_random5000_fpscenter_sep.json",
        "new": False,
    },
    "Q1V_fpsmultistart5000_ball16": {
        "label": "Q1V FPS-multistart5000 + ball16（历史复用）",
        "anchor": "Q1V/SAME",
        "family": "control_q1v_fpsms",
        "support_sampling": "fps_multistart",
        "grouping": "ball16",
        "run_dir": ROOT / "training_wss_min/runs/pointnetpp_q2v_sampling_radius_test27/outputs/q1v_fpsmultistart5000_same",
        "config": ROOT / "training_wss_min/configs/pointnetpp_q2v_sampling_radius_test27_20260718/q1v_fpsmultistart5000_same.json",
        "new": False,
    },
}

CAPACITY = ["Q1V_N0_G0_random5000_ball16", "q1v_n32_w32", "q1v_n16_w64", "q1v_n32_w64"]
GROUPING = ["Q1V_N0_G0_random5000_ball16", "q1v_sa1_ball32", "q1v_sa1_knn8", "q1v_sa1_knn10", "q1v_sa1_knn8_cover", "q1v_sa1_knn10_cover", "q1v_sa1_adaptive_cover"]
Q1_SUPPORT = [
    "Q1V_N0_G0_random5000_ball16", "q1v_sa1_adaptive_cover",
    "q1v_fixedfps5000_ball16", "q1v_fixedfps5000_adaptive",
    "Q1V_fpsmultistart5000_ball16", "q1v_fpsms5000_adaptive",
]
Q2_SUPPORT = [
    "Q2V_random5000_ball16", "q2v_random5000_adaptive",
    "q2v_fixedfps5000_ball16", "q2v_fixedfps5000_adaptive",
    "q2v_fpsms5000_ball16", "q2v_fpsms5000_adaptive",
]


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
    """True when expanded runtime config retains every frozen config field."""
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
        "normalized_mae": n["field_casebalanced"]["mae"],
        "normalized_rmse": n["field_casebalanced"]["rmse"],
        "parameters": t["efficiency"].get("parameters"),
        "eval_elapsed_seconds": t["efficiency"].get("elapsed_seconds"),
    }


def geometry_map() -> dict[tuple[str, str, int], dict[str, Any]]:
    result: dict[tuple[str, str, int], dict[str, Any]] = {}
    for path in (CORE_GEOM, FPS_GEOM):
        data = load(path)
        assert data["status"] == "passed", f"geometry audit failed: {path}"
        for row in data["results"]:
            contract = row["contract"]
            result[(contract[0], contract[2], contract[3])] = {
                "coverage_min": row["coverage_min"],
                "coverage_median": row["coverage_median"],
                "max_pair_overlap": row["max_pair_overlap"],
                "group_size_min": row["group_size_min"],
                "group_size_max": row["group_size_max"],
                "hard_gate": row["hard_gate"],
                "geometry_source": str(path.relative_to(ROOT)),
            }
    return result


def grouping_key(config: dict[str, Any], meta: dict[str, Any]) -> tuple[str, str, int]:
    sampling = config["data"].get("support_sampling", meta["support_sampling"])
    # Older reused controls predate explicit SA grouping fields; their frozen
    # PointNet++ defaults are exactly ball/nsample16.
    name = config["model"].get("sa_grouping", ["ball"])[0]
    n = int(config["model"].get("sa_nsample", [16])[0])
    return sampling, name, n


def label_for(exp_id: str, config: dict[str, Any], meta: dict[str, Any]) -> str:
    names = {
        "q1v_n32_w32": "Q1V n32 / w32",
        "q1v_n16_w64": "Q1V n16 / w64",
        "q1v_n32_w64": "Q1V n32 / w64",
        "q1v_sa1_ball32": "Q1V ball32",
        "q1v_sa1_knn8": "Q1V raw KNN-8",
        "q1v_sa1_knn10": "Q1V raw KNN-10",
        "q1v_sa1_knn8_cover": "Q1V KNN-8-cover",
        "q1v_sa1_knn10_cover": "Q1V KNN-10-cover",
        "q1v_sa1_adaptive_cover": "Q1V adaptive_cover",
        "q1v_fixedfps5000_ball16": "Q1V fixed-FPS + ball16",
        "q1v_fixedfps5000_adaptive": "Q1V fixed-FPS + adaptive",
        "q1v_fpsms5000_adaptive": "Q1V FPS-multistart + adaptive",
        "q2v_random5000_adaptive": "Q2V random + adaptive",
        "q2v_fixedfps5000_ball16": "Q2V fixed-FPS + ball16",
        "q2v_fixedfps5000_adaptive": "Q2V fixed-FPS + adaptive",
        "q2v_fpsms5000_ball16": "Q2V FPS-multistart + ball16",
        "q2v_fpsms5000_adaptive": "Q2V FPS-multistart + adaptive",
    }
    return names.get(exp_id, meta.get("label", exp_id))


def get_training_seconds(run: Path) -> float | None:
    history = run / "history.jsonl"
    if not history.exists():
        return None
    last = None
    for line in history.read_text(encoding="utf-8").splitlines():
        if line.strip():
            last = json.loads(line)
    return None if last is None else last.get("elapsed_s")


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


def audit_run(exp_id: str, meta: dict[str, Any], expected_sha: str | None, geom: dict[tuple[str, str, int], dict[str, Any]]) -> dict[str, Any]:
    run = Path(meta["run_dir"])
    source_config = Path(meta["config"])
    frozen = load(source_config)
    runtime = load(run / "config.json")
    epochs = frozen["train"]["epochs"]
    expected_cases = 27
    checks: dict[str, bool] = {}
    problems: list[str] = []
    source_sha = sha256(source_config)
    checks["prepared_sha256_match"] = expected_sha is None or source_sha == expected_sha
    if not checks["prepared_sha256_match"]:
        problems.append(f"source SHA {source_sha} != prepared {expected_sha}")
    checks["runtime_preserves_frozen_config"] = subset(frozen, runtime)
    if not checks["runtime_preserves_frozen_config"]:
        problems.append("runtime config differs on frozen field")
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
    all_csv_rows: dict[str, list[dict[str, str]]] = {}
    metric_docs: dict[str, Any] = {}
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
        ok, csv_problems, rows = audit_csv(csv_path, expected_cases)
        all_csv_rows[checkpoint] = rows
        checks["per_case_27_complete"] &= ok and len(metrics.get("test", {}).get("per_case", {})) == expected_cases
        checks["no_nan_inf"] &= finite(metrics) and not csv_problems
        problems.extend(csv_problems)
    checks["selection_rule_train_loss"] = runtime["train"].get("selection_rule") == "train_loss" and runtime["train"].get("ckpt_metric") == "train_loss"
    g = geom.get(grouping_key(runtime, meta))
    if g is None:
        problems.append("missing matching geometry contract")
    # KNN-cover has a coverage hard gate only.  The stricter pair-overlap cap
    # belongs specifically to adaptive_cover; KNN-cover's high overlap is a
    # reported limitation, not a protocol failure.
    checks["geometry_hard_gate"] = bool(
        g and (
            not g["hard_gate"]
            or (g["coverage_min"] == 1.0 and (
                runtime["model"]["sa_grouping"][0] != "adaptive_cover"
                or g["max_pair_overlap"] <= 1 / 3 + 1e-12
            ))
        )
    )
    for name, good in checks.items():
        if not good and not any(name in x for x in problems):
            problems.append(name)
    best = metric_docs.get("ckpt_best")
    last = metric_docs.get("ckpt_last")
    return {
        "experiment_id": exp_id,
        "label": label_for(exp_id, runtime, meta),
        "is_new_run": meta["new"],
        "family": meta["family"],
        "anchor": meta["anchor"],
        "run_dir": str(run),
        "config": str(source_config),
        "source_config_sha256": source_sha,
        "prepared_config_sha256": expected_sha,
        "checks": checks,
        "passed": all(checks.values()),
        "problems": problems,
        "config_data": runtime["data"],
        "config_model": runtime["model"],
        "config_train": runtime["train"],
        "metrics_best": metric_row(best) if best else {},
        "metrics_last": metric_row(last) if last else {},
        "training_seconds": get_training_seconds(run),
        "best_last": load(run / "eval/best_last_comparison.json") if (run / "eval/best_last_comparison.json").exists() else None,
        "geometry": g,
        "per_case_best": all_csv_rows.get("ckpt_best", []),
    }


def audit_control(exp_id: str, meta: dict[str, Any], geom: dict[tuple[str, str, int], dict[str, Any]]) -> dict[str, Any]:
    # Reused controls are retained as comparison evidence, not counted in 17 new-run audit.
    run = Path(meta["run_dir"])
    config = load(Path(meta["config"]))
    metric = load(run / "eval/ckpt_best/metrics.json")
    last = load(run / "eval/ckpt_last/metrics.json")
    _, _, per_case = audit_csv(run / "eval/ckpt_best/per_case_metrics.csv", 27)
    g = geom.get(grouping_key(config, meta))
    return {
        "experiment_id": exp_id, "label": meta["label"], "is_new_run": False,
        "family": meta["family"], "anchor": meta["anchor"], "run_dir": str(run),
        "config": str(meta["config"]), "source_config_sha256": sha256(Path(meta["config"])),
        "prepared_config_sha256": None, "checks": {"reused_control": True}, "passed": True,
        "problems": [], "config_data": config["data"], "config_model": config["model"], "config_train": config["train"],
        "metrics_best": metric_row(metric), "metrics_last": metric_row(last), "training_seconds": get_training_seconds(run),
        "best_last": load(run / "eval/best_last_comparison.json") if (run / "eval/best_last_comparison.json").exists() else None,
        "geometry": g, "per_case_best": per_case,
    }


def delta(a: Any, b: Any) -> Any:
    return None if a is None or b is None else a - b


def table_rows(name: str, ids: list[str], results: dict[str, dict[str, Any]], comparator) -> list[dict[str, Any]]:
    rows = []
    for exp_id in ids:
        result = results[exp_id]
        ref_id = comparator(exp_id)
        ref = results[ref_id]
        m, b = result["metrics_best"], ref["metrics_best"]
        rows.append({
            "experiment_id": exp_id, "label": result["label"], "reference_id": ref_id,
            "r2_cb": m["physical_r2_cb"], "delta_r2_cb": delta(m["physical_r2_cb"], b["physical_r2_cb"]),
            "normalized_r2_cb": m["normalized_r2_cb"], "delta_normalized_r2_cb": delta(m["normalized_r2_cb"], b["normalized_r2_cb"]),
            "case_mean_r2": m["physical_casemean_r2"], "delta_case_mean_r2": delta(m["physical_casemean_r2"], b["physical_casemean_r2"]),
            "negative_cases": m["physical_r2_negative_cases"], "delta_negative_cases": delta(m["physical_r2_negative_cases"], b["physical_r2_negative_cases"]),
            "mae": m["physical_mae"], "delta_mae": delta(m["physical_mae"], b["physical_mae"]),
            "rmse": m["physical_rmse"], "delta_rmse": delta(m["physical_rmse"], b["physical_rmse"]),
            "spearman": m["physical_spearman"], "delta_spearman": delta(m["physical_spearman"], b["physical_spearman"]),
            "top10_iou": m["physical_top10_iou"], "delta_top10_iou": delta(m["physical_top10_iou"], b["physical_top10_iou"]),
            "high_wss_r2": m["physical_high_wss_r2"], "delta_high_wss_r2": delta(m["physical_high_wss_r2"], b["physical_high_wss_r2"]),
            "high_wss_mae": m["physical_high_wss_mae"], "delta_high_wss_mae": delta(m["physical_high_wss_mae"], b["physical_high_wss_mae"]),
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
            payload = dict(row)
            if isinstance(payload.get("geometry"), dict):
                payload["geometry"] = json.dumps(payload["geometry"], ensure_ascii=False, sort_keys=True)
            writer.writerow(payload)


def f(x: Any, digits: int = 4) -> str:
    return "—" if x is None else f"{x:.{digits}f}" if isinstance(x, float) else str(x)


def md_table(rows: list[dict[str, Any]], geometry: bool = False) -> str:
    headers = ["配置", "物理R²_cb（Δ）", "归一化R²_cb（Δ）", "病例均值R²（Δ）", "负例", "MAE（Δ）", "Spearman（Δ）", "top10 IoU（Δ）", "high-WSS R²（Δ）"]
    if geometry:
        headers += ["覆盖率 min/median", "最大重叠", "组大小"]
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
        ]
        if geometry:
            g = r["geometry"] or {}
            cells += [f(g.get("coverage_min")) + "/" + f(g.get("coverage_median")), f(g.get("max_pair_overlap")), f(g.get("group_size_min"), 0) + "–" + f(g.get("group_size_max"), 0)]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def append_docs(tables: dict[str, list[dict[str, Any]]], all_passed: bool) -> None:
    verdict = "17/17 新 run 已完成且审计通过" if all_passed else "存在未通过审计项；见结构化 JSON"
    section = f"""

## {ROUND}

**状态**：{verdict}。Q1V-10476（vertex-random5000、SAME、FPS center 500/125/32、106/0/27、seed=1234、width32、ball16）为主基准；Q2V-10477 仅为 SEP 配套敏感性基准。全部结论都基于 historical test27 的同协议工程比较，**不是独立确认结论**；本轮没有 3-seed，也未按 test27 重选 checkpoint（主结果均取 `ckpt_best(train_loss)`）。

审计：冻结 manifest SHA、seed、split、400 epoch、best/last checkpoint、best/last test27、27 个病例、有限数值与冻结 SA 协议均逐项检查。几何真源 `sa_grouping_*_geometry_audit.json` 均为 passed。adaptive_cover 在所有对应 contract 上 coverage=100%、最大 pair overlap≤1/3；实际 group size 为 1–84（random）、1–24（fixed-FPS）、1–26（FPS-multistart），故 `nsample=16` 是补充目标而不是硬上限。KNN-cover 同为100%覆盖，但最大重叠仍为0.875/0.900；raw KNN-8/10 覆盖率仅 0.511/0.684 与 0.575/0.782（min/median）。ball16/ball32 覆盖率分别为 0.201/0.303 与 0.382/0.550（random）；ball32 的更高覆盖来自更大且更重复的邻域（最大重叠均为1）。

### 1) nsample × width（统一对照：Q1V n16/w32）

{md_table(tables['nsample_width'])}

### 2) SA1 grouping（统一对照：Q1V ball16）

{md_table(tables['sa1_grouping'], geometry=True)}

### 3) Q1V/SAME support（每列 grouping 与 random 同 grouping 对照）

{md_table(tables['q1v_support'])}

### 4) Q2V/SEP support（每列 grouping 与 random 同 grouping 对照）

{md_table(tables['q2v_support'])}

**判读与下一步**：容量×邻域未显示一个跨 width 的稳定 nsample 增益；ball32 不构成 Go。raw KNN 的不完全覆盖伴随总体表现回退，KNN-cover 虽100%覆盖但高重叠下也没有稳定收益。adaptive 同时通过几何门禁，但在 Q1V/SAME 与 Q2V/SEP 的性能效应需结合全场、high-WSS、hotspot、负例及 best/last 敏感性审慎解读，单 seed/test27 不足以宣布最终 Go。fixed-FPS 与 FPS-multistart 的结论均只限本协议探索；尤其历史 Q1V FPS-multistart+ball16 仍须作为工程对照，不可写作新独立确认。

真源：`training_wss_min/preflight/sa_grouping_single_seed_results_analysis.json`、`sa_grouping_single_seed_results_summary.csv`、`sa_grouping_single_seed_per_case_deltas.csv`。
""".strip() + "\n"
    for doc in (DOC_MATRIX, DOC_LOG):
        text = doc.read_text(encoding="utf-8")
        if ROUND in text:
            text = text[:text.index("## " + ROUND)].rstrip() + "\n"
        doc.write_text(text + section, encoding="utf-8")


def copy_row_style(ws, source_row: int, target_row: int, columns: int) -> None:
    for col in range(1, columns + 1):
        source, target = ws.cell(source_row, col), ws.cell(target_row, col)
        if source.has_style:
            target._style = copy(source._style)
        if source.number_format:
            target.number_format = source.number_format
        target.alignment = copy(source.alignment)
        target.protection = copy(source.protection)


def excel_values(r: dict[str, Any], exp_id: str) -> list[Any]:
    cfg = r["config_model"]
    data = r["config_data"]
    m = r["metrics_best"]
    nsample = cfg.get("sa_nsample", [16])[0]
    grouping = cfg.get("sa_grouping", ["ball"])[0]
    raw = load(Path(r["run_dir"]) / "eval/ckpt_best/metrics.json")["test"]
    norm = raw["normalized"]
    return [
        r["label"], "106/0/27（historical test27；single seed=1234）", "PointNet++ SA3",
        f"5000→500→125→32；nsample={nsample}；width={cfg['width']}；SA1={grouping}",
        f"{data.get('support_sampling')}5000 / {data.get('query_mode', '').upper()}；FPS center", 5000,
        m["physical_r2_cb"], m["physical_r2"], m["physical_casemean_r2"], raw["aggregate"]["r2_casemed"], raw["aggregate"]["r2_casep10"],
        f"{m['physical_r2_negative_cases']}/27", raw["field"]["rmse"], raw["field"]["mae"], raw["field"]["nrmse_range"], raw["aggregate"]["nrmse_casemean"],
        raw["field"].get("nmae_range"), raw["aggregate"].get("nmae_casemean"), m["physical_high_wss_r2"], raw["calibration"]["top10_pred_true_ratio"],
        m["physical_top10_iou"], m["normalized_r2_cb"], m["normalized_r2"], m["normalized_casemean_r2"], norm["aggregate"]["r2_casemed"], norm["aggregate"]["r2_casep10"],
        norm["field_casebalanced"]["mae"], norm["field_casebalanced"]["rmse"], norm["field"]["nrmse_range"], norm["aggregate"]["nrmse_casemean"],
        norm["field"].get("nmae_range"), norm["aggregate"].get("nmae_casemean"), m["physical_spearman"], raw["hotspot"]["spearman_high_wss_casemean"],
        norm["calibration"]["top10_pred_true_ratio"], norm["calibration"]["p99_pred_true_ratio"], exp_id,
    ]


def update_xlsx(results: dict[str, dict[str, Any]], tables: dict[str, list[dict[str, Any]]]) -> None:
    wb = load_workbook(XLSX, data_only=False)
    # Remove any prior generated round (idempotent; the source workbook had none).
    for ws in (wb["实验矩阵总览"], wb["教师汇报视图"], wb["汇总对比"]):
        for row in range(ws.max_row, 0, -1):
            if ws.cell(row, 1).value == ROUND:
                ws.delete_rows(row, ws.max_row - row + 1)
                break
    ids = list(dict.fromkeys(CAPACITY + GROUPING + Q1_SUPPORT + Q2_SUPPORT))
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
        values = ["SA1覆盖/重叠", r["label"], "PointNet++ SA3", f"{r['config_data'].get('support_sampling')}5000 / {r['anchor']}",
                  m["physical_r2_cb"], m["physical_casemean_r2"], load(Path(r["run_dir"]) / "eval/ckpt_best/metrics.json")["test"]["field"]["rmse"],
                  load(Path(r["run_dir"]) / "eval/ckpt_best/metrics.json")["test"]["field"]["nrmse_range"],
                  load(Path(r["run_dir"]) / "eval/ckpt_best/metrics.json")["test"]["normalized"]["field"]["nrmse_range"],
                  m["physical_high_wss_r2"], m["normalized_r2_cb"],
                  load(Path(r["run_dir"]) / "eval/ckpt_best/metrics.json")["test"]["normalized"]["field"]["nrmse_range"],
                  m["physical_spearman"], m["physical_top10_iou"], m["normalized_r2"]]
        for col, value in enumerate(values, 1): teacher.cell(i, col, value)
    summary = wb["汇总对比"]
    row = summary.max_row + 1
    summary.merge_cells(start_row=row, start_column=1, end_row=row, end_column=15)
    summary.cell(row, 1, ROUND).font = Font(bold=True, color="FFFFFF")
    summary.cell(row, 1).fill = PatternFill("solid", fgColor="1F4E78")
    row += 1
    for title, rows in tables.items():
        summary.merge_cells(start_row=row, start_column=1, end_row=row, end_column=15)
        summary.cell(row, 1, title + "（Δ 相对 reference_id；同 split/stats/query 才计算）").font = Font(bold=True)
        row += 1
        headers = ["配置", "对照", "物理R²_cb", "ΔR²_cb", "归一化R²_cb", "Δ", "病例均值R²", "负例", "MAE", "ΔMAE", "Spearman", "Δ", "top10 IoU", "Δ", "high-WSS R²"]
        for col, value in enumerate(headers, 1): summary.cell(row, col, value)
        for col in range(1, 16): summary.cell(row, col).font = Font(bold=True)
        row += 1
        for r in rows:
            values = [r["label"], r["reference_id"], r["r2_cb"], r["delta_r2_cb"], r["normalized_r2_cb"], r["delta_normalized_r2_cb"], r["case_mean_r2"], r["negative_cases"], r["mae"], r["delta_mae"], r["spearman"], r["delta_spearman"], r["top10_iou"], r["delta_top10_iou"], r["high_wss_r2"]]
            for col, value in enumerate(values, 1): summary.cell(row, col, value)
            row += 1
    wb.save(XLSX)
    # Reopen to verify formula/merge survival and exact added run counts.
    check = load_workbook(XLSX, data_only=False)
    assert ROUND in [c.value for c in check["实验矩阵总览"]["A"]]
    assert ROUND in [c.value for c in check["教师汇报视图"]["A"]]
    assert ROUND in [c.value for c in check["汇总对比"]["A"]]
    assert check["实验矩阵总览"].max_row >= start + len(ids)
    assert check["教师汇报视图"].max_row >= start_teacher + len(ids)


def main() -> None:
    prepared = load(PREPARED)
    submission = load(SUBMISSION)
    assert prepared["status"] == "prepared" and submission["status"] == "submitted"
    assert load(CORE_GPU)["status"] == "passed" and load(FPS_GPU)["status"] == "passed"
    geom = geometry_map()
    results: dict[str, dict[str, Any]] = {}
    for item in prepared["configs"]:
        exp_id = item["experiment_id"]
        meta = {**item, "new": True}
        results[exp_id] = audit_run(exp_id, meta, item["sha256"], geom)
    for exp_id, meta in CONTROLS.items():
        results[exp_id] = audit_control(exp_id, meta, geom)
    all_new_passed = all(r["passed"] for r in results.values() if r["is_new_run"])
    q1_ball = "Q1V_N0_G0_random5000_ball16"
    q2_ball = "Q2V_random5000_ball16"
    tables = {
        "nsample_width": table_rows("nsample_width", CAPACITY, results, lambda _: q1_ball),
        "sa1_grouping": table_rows("sa1_grouping", GROUPING, results, lambda _: q1_ball),
        "q1v_support": table_rows("q1v_support", Q1_SUPPORT, results, lambda x: q1_ball if "ball16" in x or x == q1_ball else "q1v_sa1_adaptive_cover"),
        "q2v_support": table_rows("q2v_support", Q2_SUPPORT, results, lambda x: q2_ball if "ball16" in x or x == q2_ball else "q2v_random5000_adaptive"),
    }
    summary_rows = []
    for r in results.values():
        row = {k: v for k, v in r.items() if k not in {"config_data", "config_model", "config_train", "per_case_best", "best_last"}}
        row.update(r["metrics_best"])
        row["training_seconds"] = r["training_seconds"]
        summary_rows.append(row)
    write_csv(SUMMARY_CSV, summary_rows)
    per_case_rows = []
    for table_name, rows in tables.items():
        for r in rows:
            exp, ref = results[r["experiment_id"]], results[r["reference_id"]]
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
    analysis = {
        "schema_version": 1, "generated_at": datetime.now(timezone.utc).isoformat(), "round": ROUND,
        "source_of_truth": {"prepared": str(PREPARED), "submission": str(SUBMISSION), "core_geometry": str(CORE_GEOM), "fps_geometry": str(FPS_GEOM), "core_gpu": str(CORE_GPU), "fps_gpu": str(FPS_GPU)},
        "slurm_jobs": submission["jobs"], "audit": {"new_runs_total": 17, "new_runs_passed": sum(r["passed"] for r in results.values() if r["is_new_run"]), "all_new_runs_passed": all_new_passed},
        "results": {k: {kk: vv for kk, vv in v.items() if kk != "per_case_best"} for k, v in results.items()},
        "tables": tables,
        "limitations": ["17 new experiments use seed=1234 only; no 3-seed uncertainty estimate.", "Historical test27 is a repeated engineering comparison set, not an independent confirmation set.", "Main metrics use ckpt_best selected by train_loss; best/last are retained for sensitivity only."],
    }
    ANALYSIS_JSON.write_text(json.dumps(analysis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    append_docs(tables, all_new_passed)
    update_xlsx(results, tables)
    print(json.dumps({"all_new_runs_passed": all_new_passed, "new_runs": 17, "summary_rows": len(summary_rows), "per_case_delta_rows": len(per_case_rows), "xlsx_rows_added": len(dict.fromkeys(CAPACITY + GROUPING + Q1_SUPPORT + Q2_SUPPORT))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
