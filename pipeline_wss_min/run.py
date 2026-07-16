#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pipeline_wss_min 一键运行。

阶段:
  1 preprocess    每病例：配准 + 正交化/标准化 + 掩码 + 堆叠时间步 -> bundle.npz
  2 qa-gate       split included 病例硬 QA（excluded/pending 不参与）
  3 global-stats  全局 WSS 统计（第二遍，默认 peak-only train）
  4 build-samples 稀疏化 + 样本装配（配置驱动，最后一步）

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
from dataclasses import replace
from pathlib import Path
from typing import List

from . import config as C
from . import preprocess, global_stats, build_samples, qa_gate, reporting


def _cohorts(args) -> List[str]:
    if args.cohort:
        return [args.cohort]
    return list(C.COHORTS.values())


def _config_from_args(args) -> C.PipelineConfig:
    """根据命令行覆盖项构造新配置，不修改全局默认配置对象。"""
    sample = replace(
        C.DEFAULT.sample,
        wall_n_points=(args.wall_n if args.wall_n is not None
                       else C.DEFAULT.sample.wall_n_points),
        timesteps=(args.timesteps if args.timesteps is not None
                   else C.DEFAULT.sample.timesteps),
        name=(args.sample_name if args.sample_name is not None
              else C.DEFAULT.sample.name),
    )
    return replace(C.DEFAULT, sample=sample)


def run_preprocess(
    cfg: C.PipelineConfig,
    cohorts: List[str],
    case: str | None,
    split_name: str | None,
    out_root: Path,
):
    """逐病例处理，产出批量审计（ok/skipped/error 分类清晰）。"""
    import traceback
    log = reporting.get_logger()
    rows: List[dict] = []
    for cohort in cohorts:
        if case:
            cases = [case]
        elif split_name:
            cases = C.list_split_cases(cohort, split_name, ("train", "val", "test"))
        else:
            raw_dir = C.RAW_ROOT / cohort
            cases = sorted({p.name for p in raw_dir.iterdir() if p.is_dir()}) if raw_dir.is_dir() else []
        for cn in cases:
            try:
                rows.append(preprocess.preprocess_case(
                    cohort, cn, cfg, out_root=out_root,
                ))
            except preprocess.SkipCase as e:
                log.warning("  SKIP %s/%s: %s", cohort, cn, e)
                rows.append({"cohort": cohort, "case": cn, "status": "skipped", "reason": str(e)})
            except Exception as e:  # noqa: BLE001 - 单病例失败不阻断批处理
                log.error("  ERROR %s/%s: %s", cohort, cn, e)
                log.debug(traceback.format_exc())
                rows.append({"cohort": cohort, "case": cn, "status": "error", "reason": str(e)})

    summary = reporting.write_batch_audit(
        rows, "preprocess", report_dir=out_root / "pipeline_reports",
    )
    log.info("[preprocess] done: ok=%d skipped=%d error=%d / total=%d",
             summary["ok"], summary["skipped"], summary["error"], summary["total"])
    for c in summary["skipped_cases"]:
        log.info("   skipped: %s", c)
    for c in summary["error_cases"]:
        log.info("   error:   %s", c)
    for c in summary.get("unit_anomaly_cases", []):
        log.warning("   单位异常(待复核): %s", c)
    for c in summary.get("unit_extent_mismatch_cases", []):
        log.warning("   壁面/中心线覆盖不一致(待复核): %s", c)
    for c in summary.get("wall_crop_cases", []):
        log.info("   未描主动脉尾巴裁剪: %s", c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["preprocess", "qa-gate", "global-stats", "build-samples", "all"],
                    default="all")
    ap.add_argument("--cohort", default=None, help="如 AG/fast；默认全部 COHORTS")
    ap.add_argument("--case", default=None, help="单病例名（配合 --cohort）")
    ap.add_argument("--split", default=C.DEFAULT_SPLIT_NAME,
                    help="默认按该 split 的 train/val/test included 病例运行")
    ap.add_argument("--all-raw", action="store_true",
                    help="忽略 split，扫描原始 cohort 目录；仅建议数据排查时使用")
    ap.add_argument("--wall-n", type=int, default=None, help="覆盖壁面稀疏点数")
    ap.add_argument("--timesteps", default=None, choices=["peak", "all"], help="覆盖时间步选择")
    ap.add_argument("--stats-timesteps", default="peak", choices=["peak", "all"],
                    help="global-stats 使用的时间步范围；第三轮 clean-data 默认 peak")
    ap.add_argument("--sample-name", default=None, help="覆盖样本集名")
    ap.add_argument(
        "--out-root", default=str(C.OUT_ROOT),
        help="病例与审计产物根目录；v4 staging 应显式指定隔离目录",
    )
    args = ap.parse_args()
    if args.all_raw and args.stage != "preprocess":
        raise SystemExit(
            "--all-raw 只允许用于诊断性 preprocess；正式 qa-gate/global-stats/build-samples/all 必须按 split 运行"
        )

    cfg = _config_from_args(args)
    out_root = Path(args.out_root).expanduser().resolve()
    if args.stage in ("global-stats", "build-samples", "all") and out_root != C.OUT_ROOT.resolve():
        raise SystemExit(
            "非默认 --out-root 当前只支持 preprocess/qa-gate；禁止其它阶段静默读取活动数据根"
        )

    cohorts = _cohorts(args)
    split_name = None if args.all_raw else args.split
    log_path = reporting.setup_logging(args.stage)
    log = reporting.get_logger()
    log.info("stage=%s cohorts=%s case=%s split=%s sample=%s wall_n=%d timesteps=%s",
             args.stage, cohorts, args.case, split_name or "ALL_RAW", cfg.sample.name,
             cfg.sample.wall_n_points, cfg.sample.timesteps)
    t0 = time.time()

    if args.stage in ("preprocess", "all"):
        run_preprocess(cfg, cohorts, args.case, split_name, out_root)
    if args.stage in ("qa-gate", "all"):
        if split_name is None:
            raise SystemExit("qa-gate 只允许按 split 运行，不能配合 --all-raw")
        qa_gate.audit_split(split_name=split_name, out_root=out_root)
    if args.stage in ("global-stats", "all"):
        global_stats.compute_global_wss_stats(
            cohorts, cfg, split_name=split_name, timesteps_scope=args.stats_timesteps
        )
    if args.stage in ("build-samples", "all"):
        build_samples.build_all(cohorts, cfg, split_name=split_name)

    log.info("[run] stage=%s elapsed=%.1fs  (log: %s)", args.stage, time.time()-t0, log_path)


if __name__ == "__main__":
    main()
