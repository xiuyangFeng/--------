#!/usr/bin/env python3
"""G4-b Phase 0 · 分支级 (branch_id) 2D 壁面展开可行性审计（V3P · CPU · 0 重训）。

对照 G4-a No-Go（5539 反映射 gap 0.16）：按 branch_id 切段后重算 occupancy，
判断分支级展开是否优于全局展开。

用法::

    python -m training.scripts.run_v3_g4b_branch_unwrap_feasibility
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch

from ..core.denylist import filter_case_names
from ..core.splits import SplitSpec
from pipeline.config import NODE_FEATURE_NAMES
from pipeline.wall_unwrap.grid import compute_theta

REPO_ROOT = Path(__file__).resolve().parents[2]

NODE_IDX = {n: i for i, n in enumerate(NODE_FEATURE_NAMES)}
X_TAN = slice(NODE_IDX["Tangent_X"], NODE_IDX["Tangent_Z"] + 1)
BRANCH_IDX = NODE_IDX.get("branch_id")


def _grid_stats(absc: np.ndarray, theta: np.ndarray, *, grid_s: int, n_sectors: int) -> Dict[str, Any]:
    a_lo, a_hi = float(absc.min()), float(absc.max())
    if a_hi - a_lo < 1e-9:
        return {"error": "degenerate_abscissa", "abscissa_unique": int(np.unique(absc).size)}
    s_bin = np.clip(((absc - a_lo) / (a_hi - a_lo) * grid_s).astype(int), 0, grid_s - 1)
    th_bin = np.clip(((theta + math.pi) / (2 * math.pi) * n_sectors).astype(int), 0, n_sectors - 1)
    occupied = np.zeros((grid_s, n_sectors), dtype=bool)
    counts = np.zeros((grid_s, n_sectors), dtype=np.int64)
    for si, ti in zip(s_bin, th_bin):
        occupied[si, ti] = True
        counts[si, ti] += 1
    n_cells = grid_s * n_sectors
    n_occ = int(occupied.sum())
    return {
        "grid_s": grid_s,
        "n_sectors": n_sectors,
        "abscissa_unique": int(np.unique(absc).size),
        "n_wall_points": int(absc.size),
        "cells_occupied": n_occ,
        "cells_total": n_cells,
        "occupancy_ratio": round(n_occ / n_cells, 4),
        "empty_rows": int((counts.sum(axis=1) == 0).sum()),
    }


def _segment_wall_stats(
    xyz: np.ndarray,
    absc: np.ndarray,
    tan: np.ndarray,
    *,
    grid_s: int,
    n_sectors: int,
) -> Optional[Dict[str, Any]]:
    if absc.size < 20:
        return None
    theta = compute_theta(xyz, tan)
    gr = _grid_stats(absc, theta, grid_s=grid_s, n_sectors=n_sectors)
    return gr


def _audit_graph_branch(
    graph_path: Path,
    *,
    grid_s: int,
    sectors: Sequence[int],
) -> Optional[Dict[str, Any]]:
    if BRANCH_IDX is None:
        return None
    data = torch.load(graph_path, map_location="cpu", weights_only=False)
    x = data.x.numpy()
    wall = x[:, NODE_IDX["is_wall"]] > 0.5
    if int(np.sum(wall)) < 40:
        return None

    xyz_all = x[wall, :3].astype(np.float64)
    absc_all = x[wall, NODE_IDX["Abscissa"]].astype(np.float64)
    tan_all = x[wall, X_TAN].astype(np.float64)
    branch_all = x[wall, BRANCH_IDX].astype(np.float64)

    global_grids = {
        str(s): _grid_stats(absc_all, compute_theta(xyz_all, tan_all), grid_s=grid_s, n_sectors=s)
        for s in sectors
    }

    segments: Dict[str, Any] = {}
    for seg_id, label in ((0, "trunk"), (1, "branch")):
        mask = branch_all <= 0.5 if seg_id == 0 else branch_all > 0.5
        if int(mask.sum()) < 20:
            continue
        xyz = xyz_all[mask]
        absc = absc_all[mask]
        tan = tan_all[mask]
        seg_grids = {}
        for s in sectors:
            st = _segment_wall_stats(xyz, absc, tan, grid_s=grid_s, n_sectors=s)
            if st:
                seg_grids[str(s)] = st
        if seg_grids:
            segments[label] = {
                "n_wall": int(mask.sum()),
                "grids": seg_grids,
            }

    occ_global = [
        g["occupancy_ratio"]
        for g in global_grids.values()
        if "occupancy_ratio" in g
    ]
    occ_branch_best: List[float] = []
    for seg in segments.values():
        for g in seg["grids"].values():
            if "occupancy_ratio" in g:
                occ_branch_best.append(g["occupancy_ratio"])
    mean_global = float(np.mean(occ_global)) if occ_global else None
    max_branch = float(max(occ_branch_best)) if occ_branch_best else None
    delta = (max_branch - mean_global) if (max_branch is not None and mean_global is not None) else None

    flags: List[str] = []
    for g in global_grids.values():
        if g.get("error") == "degenerate_abscissa":
            flags.append("global_degenerate_abscissa")
        elif g.get("occupancy_ratio", 1.0) < 0.15:
            flags.append("global_sparse_grid")
    if not segments:
        flags.append("no_branch_segments")

    return {
        "graph": graph_path.name,
        "n_wall": int(wall.sum()),
        "global_grids": global_grids,
        "branch_segments": segments,
        "mean_global_occupancy_n4": mean_global,
        "max_branch_segment_occupancy_n4": max_branch,
        "branch_vs_global_delta_n4": round(delta, 4) if delta is not None else None,
        "flags": sorted(set(flags)),
    }


def _audit_case(
    case_rel: str,
    data_root: Path,
    *,
    grid_s: int,
    sectors: Sequence[int],
    max_graphs: int,
) -> Dict[str, Any]:
    graphs_dir = data_root / case_rel / "processed" / "graphs"
    graph_paths = sorted(graphs_dir.glob("*.pt"))[:max_graphs] if graphs_dir.is_dir() else []
    rows = []
    for gp in graph_paths:
        row = _audit_graph_branch(gp, grid_s=grid_s, sectors=sectors)
        if row:
            rows.append(row)

    deltas = [r["branch_vs_global_delta_n4"] for r in rows if r.get("branch_vs_global_delta_n4") is not None]
    globals_occ = [r["mean_global_occupancy_n4"] for r in rows if r.get("mean_global_occupancy_n4") is not None]
    return {
        "case": case_rel,
        "n_graphs_audited": len(rows),
        "mean_global_occupancy_n4": round(float(np.mean(globals_occ)), 4) if globals_occ else None,
        "mean_branch_delta_n4": round(float(np.mean(deltas)), 4) if deltas else None,
        "n_graphs_branch_improves": sum(1 for d in deltas if d > 0.05),
        "graphs": rows,
    }


def _summarize(cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    deltas = [c["mean_branch_delta_n4"] for c in cases if c.get("mean_branch_delta_n4") is not None]
    global_occ = [c["mean_global_occupancy_n4"] for c in cases if c.get("mean_global_occupancy_n4") is not None]
    n_improve = sum(c.get("n_graphs_branch_improves", 0) for c in cases)
    n_graphs = sum(c.get("n_graphs_audited", 0) for c in cases)
    return {
        "n_cases": len(cases),
        "n_graphs_audited": n_graphs,
        "mean_global_occupancy_n4": round(float(np.mean(global_occ)), 4) if global_occ else None,
        "mean_branch_delta_n4": round(float(np.mean(deltas)), 4) if deltas else None,
        "n_graphs_branch_improves_gt_0p05": n_improve,
        "phase0b_gate": bool(
            global_occ
            and float(np.mean(global_occ)) >= 0.20
            and (not deltas or float(np.mean(deltas)) > 0.0)
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="V3P G4-b 分支级 2D 展开可行性审计")
    ap.add_argument("--split", type=Path, default=REPO_ROOT / "training/splits/split_AG_v1.json")
    ap.add_argument("--data-root", type=Path, default=REPO_ROOT / "data_new/AG")
    ap.add_argument("--grid-s", type=int, default=64)
    ap.add_argument("--sectors", type=int, nargs="+", default=[4])
    ap.add_argument("--max-graphs-per-case", type=int, default=3)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    split = SplitSpec.from_json(args.split.resolve())
    data_root = args.data_root.resolve()
    active = filter_case_names(
        list(split.train_cases) + list(split.val_cases) + list(split.test_cases),
        data_root,
    )

    case_rows = [
        _audit_case(
            case_rel,
            data_root,
            grid_s=args.grid_s,
            sectors=args.sectors,
            max_graphs=args.max_graphs_per_case,
        )
        for case_rel in sorted(active)
    ]

    overall = _summarize(case_rows)
    report: Dict[str, Any] = {
        "label": "V3P-G4b-branch-unwrap-feasibility",
        "date": date.today().isoformat(),
        "motivation": "G4-a 5539 No-Go：test 3D 0.373 · 2D→3D gap 0.16；审计分支切段能否改善 occupancy/反映射前提",
        "split": str(args.split.resolve()),
        "data_root": str(data_root),
        "grid_s": args.grid_s,
        "sectors": list(args.sectors),
        "summary": overall,
        "cases": case_rows,
        "recommendation": (
            "phase0b_gate=true → 立项 G4-b 单 case 过拟合 + Probe；"
            "否则转 G4-c Patch 或 I6/I7"
        ),
    }

    out = args.output or (
        REPO_ROOT / "outputs/field/f0_decision" / f"v3p_g4b_branch_unwrap_feasibility_{date.today().strftime('%Y%m%d')}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(overall, indent=2, ensure_ascii=False))
    print(out)


if __name__ == "__main__":
    main()
