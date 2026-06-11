#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3 · TODO-1：`data_new` 三域（AAA + AG + ILO）全流程齐套候选池终审（不依 split，不做训练）。

扫描范围与后续 **V3 全量实验**一致：含 `data_new/AG`（与 `split_AG_v1` 同源数据域）及 AAA/ILO。
`split_AG_v1` 中 `excluded_cases`（如 PENG）须在 `PREPROCESS_DENYLIST` 中同步，终审脚本只读该常量。

对齐 `docs/01-任务/任务A/03-V3路线/V3_后续优化待办.md` · TODO-1 风险：
  - 与「仅 AG + 旧 split_AG_v1」主线历史指标不得混表；
  - `PREPROCESS_DENYLIST` 路径不得纳入候选；
  - 复检异常 CFD、归一化统计落盘、BC / WSS 分布尾部。

示例（repo 根、GNN 环境）::

  python -m training.scripts.run_v3_aaa_ilo_prepool_audit \\
    --data-root data_new \\
    --out-dir outputs/field/diagnostics/v3_data_new_prepool_audit \\
    --write-candidate-list training/splits/cases_data_new_v3_candidate_pool.txt
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from pipeline.audit_inputs import discover_case_dirs
from pipeline.config import BC_DIR, DATA_ROOT, FEATURE_INDICES, GRAPHS_DIR
from pipeline.export_gap_preprocess_queue import PREPROCESS_DENYLIST
from pipeline.export_post_preprocess_queue import is_pipeline_fully_complete
from pipeline.utils.io import load_boundary_conditions
from pipeline.validation import inspect_case_inputs


@dataclass(frozen=True)
class _FeatIdx:
    is_wall: int = FEATURE_INDICES["is_wall"][0]


def _graphs_dir(case_abs: Path) -> Path:
    return case_abs / GRAPHS_DIR


def _summarize_normalization_global(data_root: Path) -> Dict[str, Any]:
    p = data_root / "normalization_params_global.json"
    if not p.is_file():
        return {"path": str(p), "exists": False, "error": "missing_file"}
    try:
        with open(p, encoding="utf-8") as f:
            doc = json.load(f)
    except json.JSONDecodeError as e:
        return {"path": str(p), "exists": True, "error": f"json_decode: {e}"}
    stats = doc.get("statistics")
    keys = sorted(stats.keys()) if isinstance(stats, dict) else []
    bc_fields = []
    if isinstance(stats, dict):
        for k in ["BC_Inlet", "BC_O1", "BC_O2", "BC_O3", "BC_O4"]:
            bc_fields.append(
                {"name": k, "present": k in stats, "has_mean_std": isinstance(stats.get(k), dict)}
            )
        for wss_k in ["wss", "wss_x", "wss_y", "wss_z"]:
            bc_fields.append(
                {
                    "name": wss_k,
                    "present": wss_k in stats,
                    "has_mean_std": isinstance(stats.get(wss_k), dict),
                }
            )
    return {
        "path": str(p.resolve()),
        "exists": True,
        "description": doc.get("description"),
        "n_statistics_keys": len(keys),
        "statistics_key_sample_head": keys[:12],
        "bc_and_wss_field_checks": bc_fields,
    }


def _raw_bc_quick_scan(case_dir: Path, max_frames: int) -> Dict[str, Any]:
    bc_path = case_dir / BC_DIR
    out: Dict[str, Any] = {
        "global_conditions_exists": bc_path.is_dir(),
        "n_bc_timesteps_loaded": 0,
        "inlet_q_p05_abs_sampled": None,
        "inlet_q_p50_abs_sampled": None,
        "inlet_q_min_abs_sampled": None,
        "error": None,
    }
    if not bc_path.is_dir():
        out["error"] = "missing_Global_conditions"
        return out
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            bc_data = load_boundary_conditions(bc_path)
    except Exception as e:
        out["error"] = f"load_boundary_conditions_failed: {e}"
        return out
    steps = sorted(int(s) for s in bc_data.keys())
    out["n_bc_timesteps_loaded"] = len(steps)
    if not steps:
        out["error"] = "empty_bc_data"
        return out
    step_stride = max(1, len(steps) // max_frames)
    picked = steps[::step_stride][:max_frames]
    inlet_vals: List[float] = []
    for st in picked:
        row = bc_data.get(st)
        if not row or len(row) < 5:
            continue
        inlet_vals.append(float(row[0]))
    if not inlet_vals:
        out["error"] = "no_rows_after_sample"
        return out
    arr_i = np.abs(np.asarray(inlet_vals, dtype=np.float64))
    arr_i = arr_i[np.isfinite(arr_i)]
    if arr_i.size == 0:
        out["error"] = "inlet_all_nonfinite_after_sample"
        return out
    out["inlet_q_p05_abs_sampled"] = float(np.percentile(arr_i, 5))
    out["inlet_q_p50_abs_sampled"] = float(np.percentile(arr_i, 50))
    out["inlet_q_min_abs_sampled"] = float(np.min(arr_i))
    return out


def _graph_quick_stats(case_dir: Path, idx: _FeatIdx, max_timesteps: int) -> Optional[Dict[str, Any]]:
    gdir = _graphs_dir(case_dir)
    if not gdir.is_dir():
        return None
    paths = sorted(gdir.glob("*.pt"))
    if not paths:
        return None
    step = max(1, len(paths) // max_timesteps)
    sampled = paths[::step][:max_timesteps]
    wall_wss_chunks: List[np.ndarray] = []
    bc_rows: List[np.ndarray] = []
    for pt_path in sampled:
        try:
            g = torch.load(pt_path, weights_only=False)
        except Exception:
            continue
        x = g.x.numpy()
        is_wall = x[:, idx.is_wall] > 0.5
        if hasattr(g, "y_wss") and g.y_wss is not None and bool(np.any(is_wall)):
            wall_wss_chunks.append(g.y_wss.numpy()[is_wall])
        bc_rows.append(g.global_cond.numpy().flatten())
    stats: Dict[str, Any] = {
        "n_graphs_sampled": len(sampled),
        "n_graphs_total": len(paths),
    }
    labels = ["t_norm", "BC_Inlet", "BC_O1", "BC_O2", "BC_O3", "BC_O4"]
    if bc_rows:
        bc = np.stack(bc_rows, axis=0)
        for i, name in enumerate(labels):
            col = bc[:, i]
            stats[f"bc_{name}_absmax_sampled"] = float(np.max(np.abs(col)))
    if wall_wss_chunks:
        wss = np.concatenate(wall_wss_chunks, axis=0).astype(np.float64)
        # wss_mag from components wss_x,y,z (training labels indexing)
        comp_mag = np.sqrt(np.sum(wss[:, 1:4] ** 2, axis=1))
        stats["wall_wss_mag_absmax_sampled_z"] = float(np.max(comp_mag))
        stats["wall_wss_mag_q99_abs_sampled_z"] = float(np.quantile(np.abs(comp_mag), 0.99))
    return stats


def _flag_candidates(
    raw_bc: Dict[str, Any],
    gstats: Optional[Dict[str, Any]],
    inlet_global_median: float,
    wss_ref_q99: float,
) -> List[str]:
    flags: List[str] = []
    if raw_bc.get("error"):
        flags.append(f"raw_bc:{raw_bc['error']}")
        return flags
    p05 = raw_bc.get("inlet_q_p05_abs_sampled")
    if p05 is not None and p05 < 1e-12:
        flags.append("raw_bc:inlet_p05_near_zero_or_invalid")
    elif (
        p05 is not None
        and inlet_global_median > 1e-18
        and p05 < inlet_global_median * 5e-4
    ):
        flags.append("raw_bc:inlet_p05_far_below_typical_pool")
    if gstats:
        zq = gstats.get("wall_wss_mag_q99_abs_sampled_z")
        mz = gstats.get("wall_wss_mag_absmax_sampled_z")
        if zq is not None and mz is not None and wss_ref_q99 > 0:
            z_thr = max(12.0, wss_ref_q99 * 2.5)
            m_thr = max(30.0, wss_ref_q99 * 4.0)
            if zq > z_thr and mz > m_thr:
                flags.append("graph:wss_mag_tail_heavy_zscore_space")
        bo1 = gstats.get("bc_BC_O1_absmax_sampled")
        if isinstance(bo1, float) and bo1 > 20.0:
            flags.append("graph:bc_o1_large_zscore_space")
    if not flags:
        flags.append("ok_or_unflagged")
    return flags


def main() -> int:
    ap = argparse.ArgumentParser(
        description="data_new（AAA+AG+ILO）V3 prepool completeness + distribution audit"
    )
    ap.add_argument("--data-root", type=Path, default=DATA_ROOT)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument(
        "--write-candidate-list",
        type=Path,
        default=None,
        help="将全流程齐套候选（相对路径）写入此文件",
    )
    ap.add_argument("--max-timesteps", type=int, default=10)
    ap.add_argument("--skip-graph-scan", action="store_true", help="仅做路径/BC/raw 抽检，跳过 .pt（更快）")
    args = ap.parse_args()

    root = args.data_root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    deny = set(PREPROCESS_DENYLIST)
    idx = _FeatIdx()
    leaf_dirs = discover_case_dirs(root, ["AAA", "AG", "ILO"], [])

    complete_rel: List[str] = []
    incomplete_items: List[Dict[str, Any]] = []

    rows_for_flags: List[Tuple[str, Dict[str, Any], Optional[Dict[str, Any]]]] = []

    for d in leaf_dirs:
        rel = d.resolve().relative_to(root).as_posix()
        meta = inspect_case_inputs(d)
        row_base = {
            "case_rel": rel,
            "in_preprocess_denylist": rel in deny,
            "matched_frame_count": int(meta.get("matched_frame_count") or 0),
        }
        if rel in deny:
            incomplete_items.append({**row_base, "reason": "PREPROCESS_DENYLIST"})
            continue
        n_exp = row_base["matched_frame_count"]
        if not is_pipeline_fully_complete(d, n_exp):
            incomplete_items.append(
                {
                    **row_base,
                    "reason": "pipeline_incomplete_vs_matched_frame_count",
                    "has_feature_inputs": bool(meta.get("has_feature_inputs")),
                }
            )
            continue

        complete_rel.append(rel)

        raw_bc = _raw_bc_quick_scan(d, args.max_timesteps)
        gstats = None if args.skip_graph_scan else _graph_quick_stats(d, idx, args.max_timesteps)

        rows_for_flags.append((rel, raw_bc, gstats))

    complete_rel_sorted = sorted(complete_rel)
    inlet_medians = [
        float(r[1]["inlet_q_p50_abs_sampled"])
        for r in rows_for_flags
        if r[1].get("inlet_q_p50_abs_sampled") is not None
    ]
    pool_med = float(np.median(inlet_medians)) if inlet_medians else 0.0

    zq_all = np.asarray(
        [
            float(gst["wall_wss_mag_q99_abs_sampled_z"])
            for _, _, gst in rows_for_flags
            if gst and gst.get("wall_wss_mag_q99_abs_sampled_z") is not None
        ],
        dtype=np.float64,
    )
    wss_ref_q99 = float(np.quantile(zq_all, 0.99)) if zq_all.size > 8 else 12.0

    anomalies: List[Dict[str, Any]] = []
    for rel, raw_bc, gst in rows_for_flags:
        fl = _flag_candidates(raw_bc, gst, pool_med, wss_ref_q99)
        if fl == ["ok_or_unflagged"]:
            continue
        anomalies.append(
            {
                "case_rel": rel,
                "flags": fl,
                "raw_bc_quick": raw_bc,
                "graph_quick": gst,
            }
        )

    severity = sorted(anomalies, key=lambda x: x["case_rel"])

    summary: Dict[str, Any] = {
        "audit_id": "V3.TODO-1.data_new_tri_domain_prepool",
        "scan_groups": ["AAA", "AG", "ILO"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_root": str(root),
        "preprocess_denylist_n": len(deny),
        "leaf_cases_discovered": len(leaf_dirs),
        "n_candidates_pipeline_complete_after_deny": len(complete_rel_sorted),
        "n_pipeline_incomplete_or_deny": len(incomplete_items),
        "expected_hint_from_docs": {
            "note": "叶级总数随三域扫描而变化；终审以前一次 AAA+ILO 清点为参照时约 187 叶 − deny 条目；含 AG 后请以本 JSON 实计为准。"
        },
        "normalization_params_global": _summarize_normalization_global(root),
        "distribution_screening_notes": (
            "图侧统计均在 z-score/固定缩放标签空间（与训练中一致）；"
            "raw_bc 分支读 Global_conditions 物理量。"
        ),
        "raw_inlet_global_median_of_per_case_medians": pool_med,
        "graph_wss_q99_abs_z_pool_p99_used_for_guardrail": wss_ref_q99,
        "flagged_cases_non_ok": severity,
        "denylist_paths": sorted(deny),
    }

    summary_path = out_dir / "prepool_audit_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    inc_path = out_dir / "incomplete_or_denied_cases.json"
    with open(inc_path, "w", encoding="utf-8") as f:
        json.dump({"items": incomplete_items}, f, ensure_ascii=False, indent=2)

    if args.write_candidate_list is not None:
        lst = args.write_candidate_list.resolve()
        lst.parent.mkdir(parents=True, exist_ok=True)
        lst.write_text("\n".join(complete_rel_sorted) + "\n", encoding="utf-8")

    print(f"候选齐套病例: {len(complete_rel_sorted)}")
    print(f"不完备或 deny: {len(incomplete_items)}")
    print(f"复核 flags (非 trivial): {len(severity)}")
    print(f"概要 -> {summary_path}")
    print(f"不齐套清单 -> {inc_path}")
    if args.write_candidate_list:
        print(f"候选清单 -> {args.write_candidate_list.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
