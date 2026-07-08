#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""日志与审计。

- setup_logging: 同时输出到控制台和带时间戳的日志文件
- 每病例 report.json：该病例全部处理参数与统计
- 批量审计 CSV + JSON：每个病例 ok/skipped/error 一目了然（供全量跑后核查）
"""

from __future__ import annotations

import csv
import json
import logging
import time
from pathlib import Path
from typing import Dict, List

from . import config as C

LOG_DIR = C.PROJECT_ROOT / "logs"
REPORT_DIR = C.OUT_ROOT / "pipeline_reports"

# 审计列（顺序固定，便于 diff）
AUDIT_FIELDS = [
    "cohort", "case", "status", "reason",
    "n_wall", "n_interior", "n_near_wall", "near_wall_capped",
    "n_steps", "step_min", "step_max", "peak_step",
    "unit_factor", "unit_anomaly", "coord_scale_mm", "rotation_det",
    "origin_kind", "main_axis_mode", "main_axis_source", "main_axis_wall_sep_delta", "roll_source",
    "roll_sign_source", "roll_sign_cos", "roll_sign_reliable",
    "wss_raw_min", "wss_raw_max",
    "wall_delimiter", "interior_delimiter",
    "bundle_mb", "elapsed_s",
]


def setup_logging(stage: str) -> Path:
    """配置根 logger -> 控制台 + logs/wss_min_<stage>_<ts>.log。返回日志文件路径。"""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_path = LOG_DIR / f"wss_min_{stage}_{ts}.log"

    logger = logging.getLogger("wss_min")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    logger.propagate = False
    logger.info("log file: %s", log_path)
    return log_path


def get_logger() -> logging.Logger:
    return logging.getLogger("wss_min")


def write_case_report(out_dir: Path, report: Dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=_json_default)
    )


def write_batch_audit(rows: List[Dict], stage: str) -> Dict:
    """写批量审计 CSV + JSON，返回汇总统计。"""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())

    csv_path = REPORT_DIR / f"{stage}_audit_{ts}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=AUDIT_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in AUDIT_FIELDS})

    summary = {
        "stage": stage,
        "timestamp": ts,
        "total": len(rows),
        "ok": sum(r["status"] == "ok" for r in rows),
        "skipped": sum(r["status"] == "skipped" for r in rows),
        "error": sum(r["status"] == "error" for r in rows),
        "skipped_cases": [f"{r['cohort']}/{r['case']}: {r.get('reason','')}"
                          for r in rows if r["status"] == "skipped"],
        "error_cases": [f"{r['cohort']}/{r['case']}: {r.get('reason','')}"
                        for r in rows if r["status"] == "error"],
        "unit_anomaly_cases": [f"{r['cohort']}/{r['case']}: factor={r.get('unit_factor','')}"
                               for r in rows if r.get("unit_anomaly")],
        "roll_sign_unreliable_cases": [
            f"{r['cohort']}/{r['case']}: roll_source={r.get('roll_source','')} "
            f"sign_source={r.get('roll_sign_source','')} |cos|={r.get('roll_sign_cos','')}"
            for r in rows if r.get("status") == "ok" and r.get("roll_sign_reliable") is False],
        "main_axis_fallback_cases": [
            f"{r['cohort']}/{r['case']}: source={r.get('main_axis_source','')} "
            f"sep_delta={r.get('main_axis_wall_sep_delta','')}"
            for r in rows if r.get("status") == "ok"
            and r.get("main_axis_source") not in ("", "centerline_chord")],
    }
    json_path = REPORT_DIR / f"{stage}_audit_{ts}.json"
    json_path.write_text(json.dumps({"summary": summary, "rows": rows},
                                    indent=2, ensure_ascii=False, default=_json_default))
    # 同时写一份不带时间戳的 latest，便于脚本读取
    (REPORT_DIR / f"{stage}_audit_latest.json").write_text(
        json.dumps({"summary": summary, "rows": rows},
                   indent=2, ensure_ascii=False, default=_json_default))
    return summary


def _json_default(o):
    import numpy as np
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
