#!/usr/bin/env python3
"""V3P Phase 1 · TODO-1 主动选数：误差画像 + 失败簇映射 + split 草案。

产物（默认 ``outputs/field/f0_decision/``）：
  - v3p_error_profile_<date>.json
  - v3p_active_selection_v2_<date>.json
  - docs/.../平台期-G5与主动选数/主动选数清单.md（当前）
  - docs/.../平台期-G5与主动选数/_history/主动选数_v2_<date>.md（历史）
  - training/splits/split_AG_active_v1.json（草案）

用法::

    python -m training.scripts.run_v3p_phase1_active_data
    python -m training.scripts.run_v3p_phase1_active_data --top-n 10 --top-n-draft 5
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..core.splits import SplitSpec
from ..core.utils import ensure_dir
from ._figure_utils import load_json, load_manifest, load_prediction_payload, save_json
from .run_v3_f0_decision import (
    DEFAULT_NORM_PARAMS,
    REPO_ROOT,
    _denorm_wss_mag,
    _load_norm_stats,
    _r2_score,
)
from ..core.denylist import denylist_hit
from .run_v3p_platform_oracle_bundle import V3D_DATA_ROOT, _case_wss_stats

DEFAULT_I6_RUN = REPO_ROOT / (
    "outputs/field/field_v3_pointnext_i6diag_localpool_main01_geom_pw_asymw_a"
    "_wall13000_near2000_split_AG_v1_seed1_20260619_174001"
)
DEFAULT_J4_RUN = REPO_ROOT / (
    "outputs/field/field_v3_pointnext_j4v3dlite_localpool_main01_geom_pw_asymw_a"
    "_wall13000_near2000_split_AG_v3lite_v1_seed1_20260630_131820"
)
DEFAULT_J5_RUN = REPO_ROOT / (
    "outputs/field/field_v3_pointnext_j5activedata_localpool_main01_geom_pw_asymw_a"
    "_wall13000_near2000_split_AG_active_v2_j5_seed1_20260701_002549"
)


def _normalize_ag_case(case: str) -> str:
    return case[3:] if case.startswith("AG/") else case


def _case_for_data_new(case: str) -> str:
    """Return a case path that is valid under ``data_root=data_new``.

    Historical AG-only splits store cases as ``fast/NAME`` or ``slow/NAME``
    because they are paired with ``data_root=data_new/AG``. Active splits that
    mix AAA/ILO must be paired with ``data_root=data_new``; in that setting AG
    cases need the explicit ``AG/`` prefix.
    """
    if case.startswith(("AG/", "AAA/", "ILO/")):
        return case
    return f"AG/{case}"


def _case_error_from_manifest(
    manifest_path: Path,
    stats: Mapping[str, Dict[str, float]],
) -> Dict[str, Dict[str, Any]]:
    man = load_manifest(manifest_path)
    by_case: Dict[str, Dict[str, List[float]]] = {}
    for item in man.get("items", []):
        pred_path = Path(str(item.get("prediction_path", "")))
        if not pred_path.is_absolute():
            pred_path = REPO_ROOT / pred_path
        if not pred_path.is_file():
            continue
        case = str(item.get("case_name", pred_path.stem))
        payload = load_prediction_payload(pred_path)
        if "y_wss_true" not in payload or "y_wss_pred" not in payload:
            continue
        gt = _denorm_wss_mag(payload["y_wss_true"].detach().cpu().numpy()[:, 0], stats.get("wss"))
        pred = _denorm_wss_mag(payload["y_wss_pred"].detach().cpu().numpy()[:, 0], stats.get("wss"))
        err = np.abs(gt - pred)
        by_case.setdefault(case, {"gt": [], "pred": [], "err": []})
        by_case[case]["gt"].extend(gt.tolist())
        by_case[case]["pred"].extend(pred.tolist())
        by_case[case]["err"].extend(err.tolist())

    out: Dict[str, Dict[str, Any]] = {}
    for case, d in by_case.items():
        gt = np.asarray(d["gt"], dtype=np.float64)
        pred = np.asarray(d["pred"], dtype=np.float64)
        err = np.asarray(d["err"], dtype=np.float64)
        pace = case.split("/")[0] if "/" in case else "unknown"
        resid = gt - pred
        out[case] = {
            "case": case,
            "pace": pace,
            "n_wall": int(gt.size),
            "wss_r2_wss": float(_r2_score(gt, pred)),
            "mean_abs_err": float(np.mean(err)),
            "p95_abs_err": float(np.percentile(err, 95)),
            "wss_p95_gt": float(np.percentile(gt, 95)),
            "explained_sq_err_frac": float(np.sum(resid ** 2) / (np.sum(resid ** 2) + 1e-12)),
        }
    return out


def _load_run_errors(
    run_dir: Path,
    label: str,
    stats: Mapping[str, Dict[str, float]],
) -> Optional[Dict[str, Any]]:
    manifest = run_dir / "predictions_test_best_wss" / "manifest.json"
    if not manifest.is_file():
        return None
    per_case = _case_error_from_manifest(manifest, stats)
    fast = [v for v in per_case.values() if v["pace"] == "fast"]
    slow = [v for v in per_case.values() if v["pace"] == "slow"]
    return {
        "label": label,
        "run_dir": str(run_dir),
        "n_cases": len(per_case),
        "per_case": sorted(per_case.values(), key=lambda r: -r["mean_abs_err"]),
        "pace_mean_abs_err": {
            "fast": float(np.mean([x["mean_abs_err"] for x in fast])) if fast else None,
            "slow": float(np.mean([x["mean_abs_err"] for x in slow])) if slow else None,
        },
    }


def _phys_feature_vector(row: Mapping[str, Any]) -> np.ndarray:
    return np.array(
        [
            float(row.get("wss_p95", row.get("wss_p95_gt", 0.0)) or 0.0),
            float(row.get("wss_p50", 0.0) or 0.0),
            float(row.get("curvature_mean", 0.0) or 0.0),
        ],
        dtype=np.float64,
    )


def _scan_procurement_batch(
    batch_path: Path,
    stats: Mapping[str, Dict[str, float]],
    *,
    max_graphs_per_case: int,
    source_label: str = "procurement-batch",
) -> List[Dict[str, Any]]:
    """从采购 batch 列表加载 graphs-ready 候选（含 AAA/ILO 全路径）。"""
    if not batch_path.is_file():
        return []
    data_root = REPO_ROOT / "data_new"
    ag_root = data_root / "AG"
    rows: List[Dict[str, Any]] = []
    for line in batch_path.read_text(encoding="utf-8").splitlines():
        case = line.strip()
        if not case:
            continue
        dom = case.split("/")[0]
        if dom == "AG":
            rel = _normalize_ag_case(case)
            row = _case_wss_stats(rel, ag_root, stats, max_graphs_per_case)
            pace = rel.split("/")[0] if "/" in rel else "unknown"
        elif dom in ("AAA", "ILO"):
            row = _case_wss_stats(case, data_root, stats, max_graphs_per_case)
            pace = dom
        else:
            continue
        if not row:
            continue
        row["case_norm"] = case if dom != "AG" else rel
        row["case_full"] = case
        row["source"] = source_label
        row["lite_candidate"] = True
        row["graphs_ready"] = True
        row["qa_pass"] = _qa_pass_row(row)
        row["pace"] = pace
        row["domain"] = dom
        rows.append(row)
    return rows


def _feature_vector(row: Mapping[str, Any]) -> np.ndarray:
    return np.array(
        [
            float(row.get("wss_p95", row.get("wss_p95_gt", 0.0)) or 0.0),
            float(row.get("wss_p50", 0.0) or 0.0),
            float(row.get("curvature_mean", 0.0) or 0.0),
            1.0 if str(row.get("domain", row.get("pace", ""))) == "fast" else 0.0,
        ],
        dtype=np.float64,
    )


DEFAULT_CANDIDATE_POOL = REPO_ROOT / "training/splits/cases_data_new_v3_candidate_pool.txt"
WSS_QA_MIN_P95 = 0.05
WSS_QA_MIN_P50 = 0.01


def _qa_pass_row(row: Mapping[str, Any]) -> bool:
    p95 = float(row.get("wss_p95", 0.0) or 0.0)
    p50 = float(row.get("wss_p50", 0.0) or 0.0)
    return p95 >= WSS_QA_MIN_P95 and p50 >= WSS_QA_MIN_P50


def _graphs_ready(case: str, data_root: Path) -> bool:
    case_dir = data_root / case / "processed" / "graphs"
    return case_dir.is_dir() and any(case_dir.glob("*.pt"))


def _failure_cluster_stats(
    failure_cluster: Sequence[str],
    data_root: Path,
    stats: Mapping[str, Dict[str, float]],
    *,
    max_graphs_per_case: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for case in failure_cluster:
        row = _case_wss_stats(case, data_root, stats, max_graphs_per_case)
        if row and _qa_pass_row(row):
            row["case_norm"] = case
            row["pace"] = case.split("/")[0] if "/" in case else "unknown"
            rows.append(row)
    return rows


def _scan_ag_outside_split(
    base_split: SplitSpec,
    data_root: Path,
    stats: Mapping[str, Dict[str, float]],
    *,
    max_graphs_per_case: int,
    audit: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    in_split = set(base_split.train_cases) | set(base_split.val_cases) | set(base_split.test_cases)
    lite_set = {_normalize_ag_case(str(c)) for c in audit.get("pools", {}).get("V3D-lite-candidates", {}).get("cases", [])}
    pool: List[Dict[str, Any]] = []
    denylist_excluded: List[str] = []
    for pace in ("fast", "slow"):
        pace_dir = data_root / pace
        if not pace_dir.is_dir():
            continue
        for case_dir in sorted(pace_dir.iterdir()):
            if not case_dir.is_dir():
                continue
            case = f"{pace}/{case_dir.name}"
            if denylist_hit(case, data_root):
                denylist_excluded.append(case)
                continue
            if case in in_split:
                continue
            if not _graphs_ready(case, data_root):
                continue
            row = _case_wss_stats(case, data_root, stats, max_graphs_per_case)
            if not row:
                continue
            row["case_norm"] = case
            row["source"] = "AG-outside-split"
            row["lite_candidate"] = case in lite_set
            row["graphs_ready"] = True
            row["qa_pass"] = _qa_pass_row(row)
            pool.append(row)
    return pool, denylist_excluded


def _scan_v3_pipeline_candidates(
    failure_rows: Sequence[Mapping[str, Any]],
    *,
    candidate_pool_path: Path,
    top_n: int,
    base_split: Optional[SplitSpec] = None,
    stats: Optional[Mapping[str, Dict[str, float]]] = None,
    max_graphs_per_case: int = 3,
) -> List[Dict[str, Any]]:
    """257 池 split 外采购候选：graphs-ready 用 WSS 物理 proxy 排名；缺失 graphs 的排后。"""
    if not failure_rows or not candidate_pool_path.is_file():
        return []

    fail_feats = np.stack(
        [
            np.array(
                [
                    float(r.get("wss_p95", 0.0) or 0.0),
                    float(r.get("wss_p50", 0.0) or 0.0),
                    float(r.get("curvature_mean", 0.0) or 0.0),
                ],
                dtype=np.float64,
            )
            for r in failure_rows
        ],
        axis=0,
    )
    centroid = fail_feats.mean(axis=0)
    scale = fail_feats.std(axis=0) + 1e-6
    in_split: set[str] = set()
    if base_split is not None:
        in_split = set(base_split.train_cases) | set(base_split.val_cases) | set(base_split.test_cases)

    ranked: List[Dict[str, Any]] = []
    data_root = REPO_ROOT / "data_new"
    ag_root = data_root / "AG"
    if stats is None:
        stats = _load_norm_stats(REPO_ROOT / DEFAULT_NORM_PARAMS)

    for line in candidate_pool_path.read_text(encoding="utf-8").splitlines():
        case = line.strip()
        if not case:
            continue
        dom = case.split("/")[0]
        if dom == "AG":
            rel = _normalize_ag_case(case)
            if rel in in_split:
                continue
            graphs_path = ag_root / rel / "processed" / "graphs"
            pace = rel.split("/")[0] if "/" in rel else "unknown"
        elif dom in ("AAA", "ILO"):
            graphs_path = data_root / case / "processed" / "graphs"
            pace = dom
        else:
            continue

        ready = graphs_path.is_dir() and any(graphs_path.glob("*.pt"))
        tier = 0.0
        if dom == "AG":
            tier = 0.0 if pace == "fast" else 0.1
        elif dom == "AAA":
            tier = 0.3 if "/ruputer/" in case else 0.4
        elif dom == "ILO":
            tier = 0.5

        if ready:
            rel_for_stats = rel if dom == "AG" else case
            stats_root = ag_root if dom == "AG" else data_root
            row = _case_wss_stats(rel_for_stats, stats_root, stats, max_graphs_per_case)
            if row:
                feat = np.array(
                    [
                        float(row.get("wss_p95", 0)),
                        float(row.get("wss_p50", 0)),
                        float(row.get("curvature_mean", 0) or 0),
                    ],
                    dtype=np.float64,
                )
                dist = float(np.linalg.norm((feat - centroid) / scale))
                ranked.append({
                    "case": case,
                    "domain": dom,
                    "source": "v3-procurement-ranked",
                    "graphs_ready": True,
                    "qa_pass": _qa_pass_row(row),
                    "pace": pace,
                    "wss_p95": row.get("wss_p95"),
                    "wss_p50": row.get("wss_p50"),
                    "near_failure_dist": dist,
                    "priority_score": dist + tier,
                })
                continue

        ranked.append({
            "case": case,
            "domain": dom,
            "source": "v3-pipeline-backlog",
            "graphs_ready": False,
            "qa_pass": None,
            "pace": pace,
            "priority_score": 900.0 + tier,
            "near_failure_dist": None,
        })

    ranked.sort(key=lambda r: (r.get("priority_score", 9999), r["case"]))
    return ranked[:top_n]


def _rank_active_candidates(
    failure_rows: Sequence[Mapping[str, Any]],
    pool: Sequence[Mapping[str, Any]],
    *,
    top_n: int,
    use_phys_neighbor: bool = False,
    neighbor_dist_threshold: float = 3.0,
    phys_neighbor_dist_threshold: float = 8.0,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not failure_rows:
        return [], {"failure_cluster_neighbors": {}, "qa_rejected": []}

    fail_feats = np.stack([_feature_vector(r) for r in failure_rows], axis=0)
    centroid = fail_feats.mean(axis=0)
    scale = fail_feats.std(axis=0) + 1e-6
    phys_fail = np.stack([_phys_feature_vector(r) for r in failure_rows], axis=0)
    phys_centroid = phys_fail.mean(axis=0)
    phys_scale = phys_fail.std(axis=0) + 1e-6

    qa_rejected = [r for r in pool if not r.get("qa_pass")]
    eligible = [r for r in pool if r.get("qa_pass")]

    ranked: List[Dict[str, Any]] = []
    for row in eligible:
        case = str(row.get("case_full") or row.get("case_norm") or row.get("case", ""))
        dom = str(row.get("domain") or case.split("/")[0])
        pace = str(row.get("pace") or (case.split("/")[1] if dom == "AG" and "/" in case else dom))
        feat = _feature_vector({**row, "pace": pace, "domain": dom})
        dist = float(np.linalg.norm((feat - centroid) / scale))
        phys_dist = float(
            np.linalg.norm(
                (_phys_feature_vector({**row, "wss_p95": row.get("wss_p95"), "wss_p50": row.get("wss_p50")}) - phys_centroid)
                / phys_scale
            )
        )
        score = dist
        if pace == "fast":
            score -= 0.4
        if row.get("lite_candidate"):
            score -= 0.3
        if dom == "AAA":
            score -= 0.15
        ranked.append({
            "case": case,
            "domain": dom,
            "source": row.get("source", "AG"),
            "lite_candidate": bool(row.get("lite_candidate")),
            "near_failure_dist": phys_dist if use_phys_neighbor else dist,
            "wss_p95": row.get("wss_p95"),
            "wss_p50": row.get("wss_p50"),
            "curvature_mean": row.get("curvature_mean"),
            "pace": pace,
            "graphs_ready": bool(row.get("graphs_ready")),
            "qa_pass": True,
            "priority_score": score,
        })

    ranked.sort(key=lambda r: r["priority_score"])

    neighbors: Dict[str, Any] = {}
    covered = 0
    threshold = phys_neighbor_dist_threshold if use_phys_neighbor else neighbor_dist_threshold
    for fr in failure_rows:
        fc = str(fr.get("case_norm") or fr.get("case", ""))
        if use_phys_neighbor:
            ffeat = _phys_feature_vector(fr)
            fscale = np.stack([_phys_feature_vector(r) for r in failure_rows], axis=0).std(axis=0) + 1e-6
            best = None
            best_dist = float("inf")
            for cand in ranked:
                cfeat = _phys_feature_vector({**cand, "wss_p95": cand.get("wss_p95"), "wss_p50": cand.get("wss_p50")})
                d = float(np.linalg.norm((cfeat - ffeat) / fscale))
                if d < best_dist:
                    best_dist = d
                    best = cand["case"]
            neighbors[fc] = {"nearest_candidate": best, "dist": best_dist, "metric": "phys_wss"}
            if best and best_dist < threshold:
                covered += 1
        else:
            ffeat = _feature_vector(fr)
            best = None
            best_dist = float("inf")
            for cand in ranked:
                cfeat = _feature_vector({**cand, "wss_p95": cand.get("wss_p95"), "wss_p50": cand.get("wss_p50")})
                d = float(np.linalg.norm((cfeat - ffeat) / scale))
                if d < best_dist:
                    best_dist = d
                    best = cand["case"]
            neighbors[fc] = {"nearest_candidate": best, "dist": best_dist, "metric": "feature_pace"}
            if best and best_dist < threshold:
                covered += 1

    meta = {
        "failure_cluster_neighbors": neighbors,
        "failure_cluster_neighbor_coverage": covered,
        "neighbor_dist_threshold": threshold,
        "neighbor_metric": "phys_wss" if use_phys_neighbor else "feature_pace",
        "qa_rejected": [
            {"case": r.get("case_full") or r.get("case_norm") or r.get("case"), "wss_p95": r.get("wss_p95"), "reason": "wss_qa_fail"}
            for r in qa_rejected
        ],
    }
    return ranked[:top_n], meta


def _build_active_split(
    base_split: SplitSpec,
    ranked: Sequence[Mapping[str, Any]],
    *,
    n_add: int,
    audit_path: Path,
    selection_path: Path,
    split_version: str = "split_AG_active_v1",
) -> Dict[str, Any]:
    to_add = [r["case"] for r in ranked[:n_add]]
    external_added = [c for c in to_add if c.startswith("AAA/") or c.startswith("ILO/")]
    use_data_new_root = bool(external_added)
    if use_data_new_root:
        train = {_case_for_data_new(c) for c in base_split.train_cases}
        val = [_case_for_data_new(c) for c in base_split.val_cases]
        test = [_case_for_data_new(c) for c in base_split.test_cases]
        to_add = [_case_for_data_new(c) for c in to_add]
    else:
        train = set(base_split.train_cases)
        val = list(base_split.val_cases)
        test = list(base_split.test_cases)
    for c in to_add:
        train.add(c)
    out = {
        "split_version": split_version,
        "source": "AG + Phase1 active selection (TODO-1 · QA pass + procurement)",
        "derived_from": {
            "base_split": "training/splits/split_AG_v1.json",
            "audit_json": str(audit_path),
            "selection_json": str(selection_path),
        },
        "train_cases": sorted(train),
        "val_cases": val,
        "test_cases": test,
        "active_added": to_add,
        "external_domain_added": external_added,
        "n_train_base": len(base_split.train_cases),
        "n_train_active": len(train),
        "data_root_note": (
            "含 AAA/ILO，所有 AG 病例已写成 AG/...；需 data_root=data_new"
            if use_data_new_root
            else "AG-only；可配 data_root=data_new/AG"
        ),
    }
    return out


def _write_selection_md(
    path: Path,
    *,
    date_str: str,
    selection: Mapping[str, Any],
    error_report: Mapping[str, Any],
    gate: Mapping[str, Any],
    version_tag: str = "v3",
) -> None:
    lines = [
        f"# V3P 主动选数清单 {version_tag}（{date_str}）",
        "",
        f"> 口径：V3P · post5463 · Phase 1 TODO-1 · **{selection.get('context', '')}**",
        "> 性质：数据采购/入训候选；**非** GPU 训练结论",
        "",
        "## 1. 误差画像摘要",
        "",
        f"- 参考 run：**{error_report.get('primary', {}).get('label', 'I6-diag')}**",
        f"- fast 均值 |err|：{error_report.get('primary', {}).get('pace_mean_abs_err', {}).get('fast')}",
        f"- slow 均值 |err|：{error_report.get('primary', {}).get('pace_mean_abs_err', {}).get('slow')}",
        f"- 候选池总量：**{selection.get('candidate_pool_total', '?')}** · QA 通过 **{selection.get('qa_pass_total', '?')}**",
    ]
    if selection.get("procurement_batch"):
        lines.append(f"- 采购 Batch 并入：**{selection.get('procurement_batch')}**（+{selection.get('procurement_added', 0)} 例）")
    lines.extend([
        "",
        "## 2. Top 可入训候选（graphs-ready · QA 通过）",
        "",
        "| 排名 | case | domain | near_failure_dist | lite | wss_p95 |",
        "| ---: | --- | --- | ---: | --- | ---: |",
    ])
    for i, r in enumerate(selection.get("ranked_candidates", [])[:15], 1):
        lines.append(
            f"| {i} | `{r['case']}` | {r.get('domain', r.get('pace'))} | {r.get('near_failure_dist', 0):.3f} | "
            f"{'Y' if r.get('lite_candidate') else 'N'} | {r.get('wss_p95', 0):.3f} |"
        )
    if selection.get("denylist_excluded"):
        lines.extend(["", "## 2b. PREPROCESS_DENYLIST（不可入训）", ""])
        for c in selection.get("denylist_excluded", []):
            lines.append(f"- `{c}`")
    if selection.get("qa_rejected"):
        lines.extend(["", "## 2c. QA 拒绝（WSS≈0 / 坏数据）", ""])
        for r in selection.get("qa_rejected", []):
            lines.append(f"- `{r.get('case')}` · wss_p95={r.get('wss_p95')}")
    if selection.get("pipeline_backlog"):
        lines.extend([
            "",
            "## 3. 257 池采购排名 Top-10（graphs-ready · failure proxy）",
            "",
        ])
        for r in selection.get("pipeline_backlog", [])[:10]:
            dist = r.get("near_failure_dist")
            dist_s = f"{dist:.3f}" if dist is not None else "—"
            lines.append(f"- `{r.get('case')}` · {r.get('domain')} · dist={dist_s}")
    lines.extend([
        "",
        "## 4. Phase 1 门禁（开 J5 GPU 前）",
        "",
        f"- failure 簇 neighbor 覆盖：**{gate.get('failure_cluster_neighbor_coverage')}** / {gate.get('failure_cluster_total')} "
        f"（阈值 {gate.get('neighbor_dist_threshold')} · {gate.get('neighbor_metric')}）",
        f"- QA 可入训：**{gate.get('qa_pass_total')}** 例（建议 ≥3）",
        f"- explained_sq_err：**{gate.get('explained_sq_err_frac')}**（阈值 ≥ 0.35）",
        f"- **phase1_pass**：{'✅' if gate.get('phase1_pass') else '❌'}",
        "",
        "## 5. 下一跳",
        "",
    ])
    if gate.get("phase1_pass"):
        lines.extend([
            "- **phase1_pass ✅** → 可立项 **J5 seed1 40ep**（`V3P-J5-ActiveData-MixedProbe` · 需 data_root=data_new split）",
            "- **不** full V3D · **不** J4 扩 seed",
        ])
    else:
        lines.extend([
            "- phase1_pass ❌ → **仍不开 J5 GPU**",
            "- 继续 Batch-2 采购或调整 neighbor 阈值 / 入训病例",
            "- **不** full V3D · blind 257 pipeline",
        ])
    lines.append("")
    ensure_dir(path.parent)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="V3P Phase1 TODO-1 主动选数")
    ap.add_argument("--reference-run", type=Path, default=DEFAULT_I6_RUN)
    ap.add_argument("--compare-run", type=Path, default=DEFAULT_J4_RUN)
    ap.add_argument(
        "--audit",
        type=Path,
        default=REPO_ROOT / "outputs/field/f0_decision/v3p_v3d_lite_data_audit_20260630.json",
    )
    ap.add_argument("--base-split", type=Path, default=REPO_ROOT / "training/splits/split_AG_v1.json")
    ap.add_argument("--norm-params", type=Path, default=DEFAULT_NORM_PARAMS)
    ap.add_argument(
        "--candidate-pool",
        type=Path,
        default=DEFAULT_CANDIDATE_POOL,
        help="257 三域候选池（pipeline 采购）",
    )
    ap.add_argument("--max-graphs-per-case", type=int, default=3)
    ap.add_argument("--top-n", type=int, default=15, help="排名输出 Top-N")
    ap.add_argument("--top-n-draft", type=int, default=10, help="split 草案增补病例数")
    ap.add_argument(
        "--procurement-batch",
        type=Path,
        action="append",
        default=None,
        help="采购 batch 病例列表（可多次指定，如 batch1 + batch2）",
    )
    ap.add_argument(
        "--selection-version",
        default="v3",
        help="输出版本标签（v3/v4）",
    )
    ap.add_argument(
        "--use-phys-neighbor",
        action="store_true",
        help="跨域候选时用 WSS 物理特征评 neighbor 覆盖",
    )
    ap.add_argument("--phys-neighbor-threshold", type=float, default=8.0)
    ap.add_argument("--date", default=date.today().strftime("%Y%m%d"))
    ap.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs/field/f0_decision")
    ap.add_argument(
        "--split-output",
        type=Path,
        default=None,
        help="split 草案输出路径（默认随 selection-version 变化）",
    )
    args = ap.parse_args()

    ver = args.selection_version
    if args.split_output is None:
        split_map = {"v4": "split_AG_active_v2.json", "v5": "split_AG_active_v3.json"}
        split_name = split_map.get(ver, "split_AG_active_v1.json")
        args.split_output = REPO_ROOT / "training/splits" / split_name

    stats = _load_norm_stats(args.norm_params.resolve())
    audit = load_json(args.audit.resolve()) if args.audit.is_file() else {}
    base_split = SplitSpec.from_json(args.base_split.resolve())
    failure_cluster = audit.get("ag_failure_cluster") or [
        "fast/ZHANG_HAO",
        "fast/LIU_JUN_FENG",
        "fast/ZHANG_CHUN",
        "fast/LIU_LI_QUN",
        "fast/LI_ZHI_LIN",
    ]

    i6_dir = args.reference_run.resolve()
    j4_dir = args.compare_run.resolve()
    primary = _load_run_errors(i6_dir, "I6-diag", stats)
    if primary is None:
        raise SystemExit(f"I6-diag 缺少 predictions: {i6_dir}/predictions_test_best_wss")
    compare = _load_run_errors(j4_dir, "J4", stats)

    error_by_case = {r["case"]: r for r in primary["per_case"]}
    if compare:
        for r in compare["per_case"]:
            c = r["case"]
            if c in error_by_case:
                error_by_case[c]["j4_mean_abs_err"] = r["mean_abs_err"]
                error_by_case[c]["j4_wss_r2_wss"] = r["wss_r2_wss"]
                error_by_case[c]["delta_err_j4_minus_i6"] = r["mean_abs_err"] - error_by_case[c]["mean_abs_err"]

    ag_root = REPO_ROOT / "data_new" / "AG"
    failure_rows = _failure_cluster_stats(
        failure_cluster, ag_root, stats, max_graphs_per_case=args.max_graphs_per_case,
    )

    pool, denylist_excluded = _scan_ag_outside_split(
        base_split, ag_root, stats,
        max_graphs_per_case=args.max_graphs_per_case, audit=audit,
    )
    procurement_added = 0
    procurement_batches: List[str] = []
    if args.procurement_batch:
        existing = {str(r.get("case_norm") or r.get("case_full") or "") for r in pool}
        for batch_path in args.procurement_batch:
            if batch_path is None or not batch_path.is_file():
                continue
            procurement_batches.append(str(batch_path.resolve()))
            proc_rows = _scan_procurement_batch(
                batch_path.resolve(),
                stats,
                max_graphs_per_case=args.max_graphs_per_case,
                source_label=f"procurement-{batch_path.stem}",
            )
            for pr in proc_rows:
                key = str(pr.get("case_norm") or pr.get("case_full") or "")
                if key and key not in existing:
                    pool.append(pr)
                    procurement_added += 1
                    existing.add(key)

    use_phys = args.use_phys_neighbor or procurement_added > 0
    ranked, rank_meta = _rank_active_candidates(
        failure_rows,
        pool,
        top_n=args.top_n,
        use_phys_neighbor=use_phys,
        phys_neighbor_dist_threshold=args.phys_neighbor_threshold,
    )
    pipeline_backlog = _scan_v3_pipeline_candidates(
        failure_rows,
        candidate_pool_path=args.candidate_pool.resolve(),
        top_n=20,
        base_split=base_split,
        stats=stats,
        max_graphs_per_case=args.max_graphs_per_case,
    )

    total_sq = sum(float(r.get("mean_abs_err", 0)) ** 2 * r.get("n_wall", 1) for r in primary["per_case"])
    top_err_cases = sorted(primary["per_case"], key=lambda r: -r["mean_abs_err"])[: max(3, len(failure_cluster))]
    cluster_sq = sum(float(r.get("mean_abs_err", 0)) ** 2 * r.get("n_wall", 1) for r in top_err_cases)
    explained = cluster_sq / (total_sq + 1e-12)

    qa_pass_total = sum(1 for r in pool if r.get("qa_pass"))
    neighbor_cov = int(rank_meta.get("failure_cluster_neighbor_coverage", 0))

    out_dir = ensure_dir(args.output_dir.resolve())
    err_path = out_dir / f"v3p_error_profile_{args.date}.json"
    sel_path = out_dir / f"v3p_active_selection_{ver}_{args.date}.json"

    error_report = {
        "label": "v3p_error_profile",
        "date": args.date,
        "primary": primary,
        "compare": compare,
        "failure_cluster": failure_cluster,
        "failure_cluster_stats": failure_rows,
        "top_error_cases": top_err_cases[:10],
    }
    save_json(err_path, error_report)

    gate = {
        "failure_cluster_neighbor_coverage": neighbor_cov,
        "failure_cluster_total": len(failure_cluster),
        "candidate_pool_total": len(pool),
        "ag_outside_split_total": len(pool) - procurement_added,
        "qa_pass_total": qa_pass_total,
        "ag_qa_pass_total": qa_pass_total,
        "procurement_added": procurement_added,
        "explained_sq_err_frac": round(explained, 4),
        "explained_threshold": 0.35,
        "neighbor_dist_threshold": rank_meta.get("neighbor_dist_threshold"),
        "neighbor_metric": rank_meta.get("neighbor_metric"),
        "n_add_recommended": min(args.top_n_draft, qa_pass_total),
        "phase1_pass": explained >= 0.35 and qa_pass_total >= 3 and neighbor_cov >= 2,
    }
    selection = {
        "label": f"v3p_active_selection_{ver}",
        "date": args.date,
        "context": f"V3P · Phase1 TODO-1 {ver} (QA + procurement batch + neighbor re-eval)",
        "reference_run": str(i6_dir),
        "compare_run": str(j4_dir) if compare else None,
        "candidate_pool_total": len(pool),
        "qa_pass_total": qa_pass_total,
        "ag_outside_split_total": len(pool) - procurement_added,
        "ag_qa_pass_total": qa_pass_total,
        "procurement_batches": procurement_batches,
        "procurement_batch": procurement_batches[0] if len(procurement_batches) == 1 else None,
        "procurement_added": procurement_added,
        "ranked_candidates": ranked,
        "qa_rejected": rank_meta.get("qa_rejected", []),
        "failure_cluster_neighbors": rank_meta.get("failure_cluster_neighbors", {}),
        "pipeline_backlog": pipeline_backlog,
        "denylist_excluded": denylist_excluded,
        "gate": gate,
        "split_draft": f"{args.split_output} (+{gate['n_add_recommended']})",
    }
    save_json(sel_path, selection)

    split_version_map = {"v4": "split_AG_active_v2", "v5": "split_AG_active_v3"}
    split_doc = _build_active_split(
        base_split, ranked, n_add=gate["n_add_recommended"],
        audit_path=args.audit.resolve(), selection_path=sel_path,
        split_version=split_version_map.get(ver, "split_AG_active_v1"),
    )
    save_json(args.split_output.resolve(), split_doc)

    platform_docs = (
        REPO_ROOT
        / "docs/01-任务/任务A/03-V3路线/01-执行与待办/平台期-G5与主动选数"
    )
    history_md = platform_docs / "_history" / f"主动选数_{ver}_{args.date}.md"
    current_md = platform_docs / "主动选数清单.md"
    _write_selection_md(
        history_md, date_str=args.date, selection=selection,
        error_report=error_report, gate=gate, version_tag=ver,
    )
    _write_selection_md(
        current_md, date_str=args.date, selection=selection,
        error_report=error_report, gate=gate, version_tag=ver,
    )

    print(json.dumps({
        "error_profile": str(err_path),
        "active_selection": str(sel_path),
        "split_draft": str(args.split_output),
        "markdown_history": str(history_md),
        "markdown_current": str(current_md),
        "gate": gate,
        "top5": [r["case"] for r in ranked[:5]],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
