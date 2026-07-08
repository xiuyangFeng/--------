#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pipeline_wss_min 一键运行。

阶段:
  1 preprocess    每病例：配准 + 正交化/标准化 + 掩码 + 堆叠时间步 -> bundle.npz
  2 global-stats  全局 WSS 统计（第二遍）
  3 build-samples 稀疏化 + 样本装配（配置驱动，最后一步）

示例:
  # 单病例跑通(冒烟)
  python -m pipeline_wss_min.run --stage preprocess --cohort AG/fast --case CHEN_SHI_MING
  # 全流程
  python -m pipeline_wss_min.run --stage all
  # 只重扫点数（改配置后）
  python -m pipeline_wss_min.run --stage build-samples --wall-n 1500 --sample-name wss_min_peak_w1500
"""

from __future__ import annotations

import argparse
import time
from typing import List

from . import config as C
from . import preprocess, global_stats, build_samples, reporting


def _cohorts(args) -> List[str]:
    if args.cohort:
        return [args.cohort]
    return list(C.COHORTS.values())


def run_preprocess(cfg: C.PipelineConfig, cohorts: List[str], case: str | None):
    """逐病例处理，产出批量审计（ok/skipped/error 分类清晰）。"""
    import traceback
    log = reporting.get_logger()
    rows: List[dict] = []
    for cohort in cohorts:
        cases = [case] if case else C.list_cases(cohort)
        # 全量时也扫一遍缺 raw 目录、但可能没进 list_cases 的目录
        if not case:
            raw_dir = C.RAW_ROOT / cohort
            if raw_dir.is_dir():
                cases = sorted({p.name for p in raw_dir.iterdir() if p.is_dir()})
        for cn in cases:
            try:
                rows.append(preprocess.preprocess_case(cohort, cn, cfg))
            except preprocess.SkipCase as e:
                log.warning("  SKIP %s/%s: %s", cohort, cn, e)
                rows.append({"cohort": cohort, "case": cn, "status": "skipped", "reason": str(e)})
            except Exception as e:  # noqa: BLE001 - 单病例失败不阻断批处理
                log.error("  ERROR %s/%s: %s", cohort, cn, e)
                log.debug(traceback.format_exc())
                rows.append({"cohort": cohort, "case": cn, "status": "error", "reason": str(e)})

    summary = reporting.write_batch_audit(rows, "preprocess")
    log.info("[preprocess] done: ok=%d skipped=%d error=%d / total=%d",
             summary["ok"], summary["skipped"], summary["error"], summary["total"])
    for c in summary["skipped_cases"]:
        log.info("   skipped: %s", c)
    for c in summary["error_cases"]:
        log.info("   error:   %s", c)
    for c in summary.get("unit_anomaly_cases", []):
        log.warning("   单位异常(待复核): %s", c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["preprocess", "global-stats", "build-samples", "all"],
                    default="all")
    ap.add_argument("--cohort", default=None, help="如 AG/fast；默认全部 COHORTS")
    ap.add_argument("--case", default=None, help="单病例名（配合 --cohort）")
    ap.add_argument("--wall-n", type=int, default=None, help="覆盖壁面稀疏点数")
    ap.add_argument("--timesteps", default=None, choices=["peak", "all"], help="覆盖时间步选择")
    ap.add_argument("--sample-name", default=None, help="覆盖样本集名")
    args = ap.parse_args()

    cfg = C.DEFAULT
    if args.wall_n is not None:
        cfg.sample.wall_n_points = args.wall_n
    if args.timesteps is not None:
        cfg.sample.timesteps = args.timesteps
    if args.sample_name is not None:
        cfg.sample.name = args.sample_name

    cohorts = _cohorts(args)
    log_path = reporting.setup_logging(args.stage)
    log = reporting.get_logger()
    log.info("stage=%s cohorts=%s case=%s sample=%s wall_n=%d timesteps=%s",
             args.stage, cohorts, args.case, cfg.sample.name,
             cfg.sample.wall_n_points, cfg.sample.timesteps)
    t0 = time.time()

    if args.stage in ("preprocess", "all"):
        run_preprocess(cfg, cohorts, args.case)
    if args.stage in ("global-stats", "all"):
        global_stats.compute_global_wss_stats(cohorts, cfg)
    if args.stage in ("build-samples", "all"):
        build_samples.build_all(cohorts, cfg)

    log.info("[run] stage=%s elapsed=%.1fs  (log: %s)", args.stage, time.time()-t0, log_path)


if __name__ == "__main__":
    main()
