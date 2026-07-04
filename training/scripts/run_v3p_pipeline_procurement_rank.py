#!/usr/bin/env python3
"""V3P 257 池采购优先级：failure_cluster 物理 proxy 排名 + batch 清单。

口径：V3P · split_AG_v1 外 · post5463 · 对照 I6-diag 0.429
性质：CPU 排名 / QA；**非** GPU 训练；**非** blind 257 全量 pipeline

产物（默认 ``outputs/field/f0_decision/``）：
  - v3p_procurement_rank_<date>.json
  - v3p_procurement_batch1_<date>.txt
  - docs/.../平台期-G5与主动选数/采购_batch1_<date>.md

用法::

    python -m training.scripts.run_v3p_pipeline_procurement_rank
    python -m training.scripts.run_v3p_pipeline_procurement_rank --batch-size 6 --exclude AG/slow/ZHAO_XIU_XUAN
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from ..core.splits import SplitSpec
from ..core.utils import ensure_dir
from ._figure_utils import save_json
from .run_v3_f0_decision import DEFAULT_NORM_PARAMS, REPO_ROOT, _load_norm_stats
from .run_v3p_platform_oracle_bundle import _case_wss_stats

try:
    from pipeline.export_gap_preprocess_queue import PREPROCESS_DENYLIST
except ImportError:
    PREPROCESS_DENYLIST = set()

DEFAULT_POOL = REPO_ROOT / "training/splits/cases_data_new_v3_candidate_pool.txt"
DEFAULT_SPLIT = REPO_ROOT / "training/splits/split_AG_v1.json"
WSS_QA_MIN_P95 = 0.05
WSS_QA_MIN_P50 = 0.01

DEFAULT_FAILURE_CLUSTER = [
    "fast/ZHANG_HAO",
    "fast/LIU_JUN_FENG",
    "fast/ZHANG_CHUN",
    "fast/LIU_LI_QUN",
    "fast/LI_ZHI_LIN",
]

DOMAIN_TIER = {
    "AG_fast": 0.0,
    "AG_slow": 0.1,
    "AAA_ruputer": 0.3,
    "AAA_unruputer": 0.4,
    "ILO": 0.5,
}


def _graphs_ready(case: str, data_root: Path) -> bool:
    g = data_root / case / "processed" / "graphs"
    return g.is_dir() and any(g.glob("*.pt"))


def _case_wss_row(
    case: str,
    stats: Mapping[str, Dict[str, float]],
    *,
    max_graphs: int,
) -> Optional[Dict[str, Any]]:
    dom = case.split("/")[0]
    ag_root = REPO_ROOT / "data_new" / "AG"
    data_root = REPO_ROOT / "data_new"
    if dom == "AG":
        rel = "/".join(case.split("/")[1:])
        row = _case_wss_stats(rel, ag_root, stats, max_graphs)
    else:
        row = _case_wss_stats(case, data_root, stats, max_graphs)
    if not row:
        return None
    row["case"] = case
    return row


def _phys_feat(row: Mapping[str, Any]) -> np.ndarray:
    return np.array(
        [
            float(row.get("wss_p95", 0.0) or 0.0),
            float(row.get("wss_p50", 0.0) or 0.0),
            float(row.get("curvature_mean", 0.0) or 0.0),
        ],
        dtype=np.float64,
    )


def _domain_tier(case: str) -> float:
    dom = case.split("/")[0]
    if dom == "AG":
        pace = case.split("/")[1] if len(case.split("/")) > 2 else "slow"
        return DOMAIN_TIER["AG_fast" if pace == "fast" else "AG_slow"]
    if dom == "AAA":
        return DOMAIN_TIER["AAA_ruputer" if "/ruputer/" in case else "AAA_unruputer"]
    if dom == "ILO":
        return DOMAIN_TIER["ILO"]
    return 1.0


def _qa_pass(row: Mapping[str, Any]) -> bool:
    return float(row.get("wss_p95", 0.0) or 0.0) >= WSS_QA_MIN_P95 and float(
        row.get("wss_p50", 0.0) or 0.0
    ) >= WSS_QA_MIN_P50


def rank_procurement_pool(
    *,
    candidate_pool_path: Path,
    base_split: SplitSpec,
    failure_cluster: Sequence[str],
    stats: Mapping[str, Dict[str, float]],
    max_graphs: int,
    exclude_cases: Sequence[str],
) -> Dict[str, Any]:
    data_root = REPO_ROOT / "data_new"
    in_split = set(base_split.train_cases) | set(base_split.val_cases) | set(base_split.test_cases)
    exclude = set(exclude_cases)

    fail_rows = []
    for fc in failure_cluster:
        row = _case_wss_row(f"AG/{fc}", stats, max_graphs=max_graphs)
        if row:
            fail_rows.append(row)
    if not fail_rows:
        raise SystemExit("failure_cluster WSS 统计为空，无法排名")

    fail_feats = np.stack([_phys_feat(r) for r in fail_rows], axis=0)
    centroid = fail_feats.mean(axis=0)
    scale = fail_feats.std(axis=0) + 1e-6

    pool_lines = [
        ln.strip()
        for ln in candidate_pool_path.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    graphs_ready_n = 0
    ranked: List[Dict[str, Any]] = []
    skipped_in_split = 0
    skipped_denylist = 0

    for case in pool_lines:
        if case in PREPROCESS_DENYLIST:
            skipped_denylist += 1
            continue
        if case in exclude:
            continue
        dom = case.split("/")[0]
        if dom == "AG":
            rel = case.replace("AG/", "")
            if rel in in_split:
                skipped_in_split += 1
                continue

        if dom == "AG":
            ready = _graphs_ready("/".join(case.split("/")[1:]), data_root / "AG")
        else:
            ready = _graphs_ready(case, data_root)
        if ready:
            graphs_ready_n += 1

        row = _case_wss_row(case, stats, max_graphs=max_graphs)
        if not row:
            ranked.append(
                {
                    "case": case,
                    "domain": dom,
                    "graphs_ready": ready,
                    "qa_pass": False,
                    "source": "v3-procurement-needs-pipeline",
                    "priority_score": 999.0 + _domain_tier(case),
                    "near_failure_dist": None,
                    "note": "graphs 或 WSS 统计不可用 → 需 pipeline",
                }
            )
            continue

        dist = float(np.linalg.norm((_phys_feat(row) - centroid) / scale))
        tier = _domain_tier(case)
        pace = case.split("/")[1] if dom == "AG" else dom
        ranked.append(
            {
                "case": case,
                "domain": dom,
                "pace": pace,
                "graphs_ready": ready,
                "qa_pass": _qa_pass(row),
                "wss_p95": row.get("wss_p95"),
                "wss_p50": row.get("wss_p50"),
                "curvature_mean": row.get("curvature_mean"),
                "near_failure_dist": dist,
                "domain_tier": tier,
                "priority_score": dist + tier,
                "source": "v3-procurement-ranked",
            }
        )

    ranked.sort(key=lambda r: (r.get("priority_score", 9999), r["case"]))
    qa_pass = [r for r in ranked if r.get("qa_pass")]
    need_pipeline = [r for r in ranked if not r.get("graphs_ready")]

    return {
        "failure_cluster": list(failure_cluster),
        "failure_centroid": centroid.tolist(),
        "failure_scale": scale.tolist(),
        "pool_total": len(pool_lines),
        "outside_ag_split": len(ranked),
        "skipped_in_split": skipped_in_split,
        "skipped_denylist": skipped_denylist,
        "graphs_ready_outside_split": graphs_ready_n,
        "qa_pass_outside_split": len(qa_pass),
        "need_pipeline_count": len(need_pipeline),
        "ranked": ranked,
        "qa_pass_ranked": qa_pass,
        "need_pipeline": need_pipeline,
    }


def _write_batch_md(
    path: Path,
    *,
    date_str: str,
    batch_label: str,
    batch_cases: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> None:
    batch_title = batch_label.replace("_", "-").upper()
    lines = [
        f"# V3P 采购优先级 {batch_title}（{date_str}）",
        "",
        "> 口径：V3P · 257 池 · failure_cluster 物理 proxy · **非** blind pipeline · **不开 J5 GPU**",
        "",
        "## 1. 池状态",
        "",
        f"- 257 池总量：**{summary['pool_total']}**",
        f"- split_AG_v1 外候选：**{summary['outside_ag_split']}**（graphs-ready **{summary['graphs_ready_outside_split']}** · QA 通过 **{summary['qa_pass_outside_split']}**）",
        f"- 需 pipeline（graphs 缺失）：**{summary['need_pipeline_count']}**",
        "",
        f"## 2. {batch_title}（near-domain fast 失败簇 proxy · 排除已采购批次）",
        "",
        "| # | case | domain | near_failure_dist | wss_p95 | graphs | QA |",
        "| ---: | --- | --- | ---: | ---: | --- | --- |",
    ]
    for i, r in enumerate(batch_cases, 1):
        lines.append(
            f"| {i} | `{r['case']}` | {r.get('domain')} | {r.get('near_failure_dist', 0):.3f} | "
            f"{r.get('wss_p95', 0):.3f} | {'✅' if r.get('graphs_ready') else '❌'} | "
            f"{'✅' if r.get('qa_pass') else '❌'} |"
        )
    lines.extend(
        [
            "",
            "## 3. 判读与下一跳",
            "",
            "- 257 池 **graphs 已齐** → 本批为 **候选深审 / 入训评估**，非 blind 257 pipeline。",
            "- 优先 **AAA near-domain**（距 failure 簇 wss_p95/p50/curvature 最近）；ILO 域 tier 靠后。",
            "- 本批 QA 通过后：重跑 Phase1 v5 邻居覆盖；**J5/J6 GPU 已封口**，仅评 phase1_pass 与 split 草案。",
            "- **禁止**：full 257 pipeline · J6 GPU · full V3D 长训。",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="V3P 257 池采购优先级排名")
    ap.add_argument("--candidate-pool", type=Path, default=DEFAULT_POOL)
    ap.add_argument("--base-split", type=Path, default=DEFAULT_SPLIT)
    ap.add_argument("--norm-params", type=Path, default=REPO_ROOT / DEFAULT_NORM_PARAMS)
    ap.add_argument("--max-graphs-per-case", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=6)
    ap.add_argument(
        "--batch-label",
        default="batch1",
        help="输出批次标签（batch1 / batch2）",
    )
    ap.add_argument(
        "--rank-skip",
        type=int,
        default=0,
        help="QA 通过排名跳过前 N 例（Batch-2 用 6）",
    )
    ap.add_argument(
        "--exclude-from-file",
        type=Path,
        default=None,
        help="从 txt 读取额外排除病例（如 Batch-1 清单）",
    )
    ap.add_argument(
        "--exclude",
        nargs="*",
        default=["AG/slow/ZHAO_XIU_XUAN"],
        help="排除病例（默认 J4 已 +1）",
    )
    ap.add_argument("--date", default=date.today().strftime("%Y%m%d"))
    ap.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs/field/f0_decision")
    args = ap.parse_args()

    exclude = list(args.exclude)
    if args.exclude_from_file is not None and args.exclude_from_file.is_file():
        for line in args.exclude_from_file.read_text(encoding="utf-8").splitlines():
            case = line.strip()
            if case:
                exclude.append(case)

    stats = _load_norm_stats(args.norm_params.resolve())
    base_split = SplitSpec.from_json(args.base_split.resolve())
    summary = rank_procurement_pool(
        candidate_pool_path=args.candidate_pool.resolve(),
        base_split=base_split,
        failure_cluster=DEFAULT_FAILURE_CLUSTER,
        stats=stats,
        max_graphs=args.max_graphs_per_case,
        exclude_cases=exclude,
    )

    qa_ranked = summary["qa_pass_ranked"]
    skip = max(0, args.rank_skip)
    batch_cases = qa_ranked[skip : skip + args.batch_size]
    batch_label = args.batch_label
    out_dir = ensure_dir(args.output_dir.resolve())
    rank_path = out_dir / f"v3p_procurement_rank_{batch_label}_{args.date}.json"
    batch_txt = out_dir / f"v3p_procurement_{batch_label}_{args.date}.txt"
    md_path = (
        REPO_ROOT
        / "docs/01-任务/任务A/03-V3路线/01-执行与待办/平台期-G5与主动选数"
        / f"采购_{batch_label}_{args.date}.md"
    )

    payload = {
        "label": "v3p_procurement_rank",
        "date": args.date,
        "batch_label": batch_label,
        "context": "V3P · 257 pool procurement · failure_cluster phys proxy · no blind pipeline",
        "reference_band": "I6-diag 0.429",
        "batch_cases": [r["case"] for r in batch_cases],
        "rank_skip": skip,
        "exclude_cases": exclude,
        **summary,
    }
    save_json(rank_path, payload)
    batch_txt.write_text("\n".join(r["case"] for r in batch_cases) + "\n", encoding="utf-8")
    _write_batch_md(
        md_path,
        date_str=args.date,
        batch_label=batch_label,
        batch_cases=batch_cases,
        summary=summary,
    )

    print(f"Wrote {rank_path}")
    print(f"Wrote {batch_txt}")
    print(f"Wrote {md_path}")
    print(f"{batch_label} ({len(batch_cases)} cases):")
    for r in batch_cases:
        print(f"  {r['case']} dist={r.get('near_failure_dist', 0):.3f}")


if __name__ == "__main__":
    main()
