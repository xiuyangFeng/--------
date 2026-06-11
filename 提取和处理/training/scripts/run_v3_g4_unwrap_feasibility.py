#!/usr/bin/env python3
"""G4 Phase 0 · 2D 壁面展开可行性审计（V3P · 0 重训 · CPU）。

对照 [V3P_路径G_G4_2D换轨预研方案.md] Phase 0：逐 case 检查 Abscissa/θ 网格
覆盖率、周向 bin 稀疏度、分叉/退化风险，产出
``outputs/field/f0_decision/v3p_g4_unwrap_feasibility_<date>.json``。

用法::

    python -m training.scripts.run_v3_g4_unwrap_feasibility
    python -m training.scripts.run_v3_g4_unwrap_feasibility --max-graphs-per-case 2
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

REPO_ROOT = Path(__file__).resolve().parents[2]

NODE_IDX = {n: i for i, n in enumerate(NODE_FEATURE_NAMES)}
X_TAN = slice(NODE_IDX["Tangent_X"], NODE_IDX["Tangent_Z"] + 1)


def _load_graph(path: Path) -> torch.Tensor:
    return torch.load(path, map_location="cpu", weights_only=False)


def _grid_stats(
    absc: np.ndarray,
    theta: np.ndarray,
    *,
    grid_s: int,
    n_sectors: int,
) -> Dict[str, Any]:
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
        "abscissa_range": [a_lo, a_hi],
        "n_wall_points": int(absc.size),
        "cells_occupied": n_occ,
        "cells_total": n_cells,
        "occupancy_ratio": round(n_occ / n_cells, 4),
        "min_cell_count": int(counts[counts > 0].min()) if n_occ else 0,
        "max_cell_count": int(counts.max()),
        "empty_rows": int((counts.sum(axis=1) == 0).sum()),
    }


def _audit_graph(
    graph_path: Path,
    *,
    grid_s: int,
    sectors: Sequence[int],
) -> Optional[Dict[str, Any]]:
    data = _load_graph(graph_path)
    x = data.x.numpy()
    wall = x[:, NODE_IDX["is_wall"]] > 0.5
    if int(np.sum(wall)) < 20:
        return None
    xyz = x[wall, :3].astype(np.float64)
    absc = x[wall, NODE_IDX["Abscissa"]].astype(np.float64)
    tan = x[wall, X_TAN].astype(np.float64)

    t_mean = tan.mean(axis=0)
    tn = np.linalg.norm(t_mean)
    t_mean = t_mean / tn if tn > 1e-9 else np.array([0.0, 0.0, 1.0])
    ref = np.array([0.0, 0.0, 1.0]) if abs(t_mean[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = ref - (ref @ t_mean) * t_mean
    e1 = e1 / (np.linalg.norm(e1) + 1e-12)
    e2 = np.cross(t_mean, e1)
    c = xyz - xyz.mean(axis=0, keepdims=True)
    theta = np.arctan2(c @ e2, c @ e1)

    grids = [_grid_stats(absc, theta, grid_s=grid_s, n_sectors=s) for s in sectors]
    return {
        "graph": graph_path.name,
        "n_wall": int(wall.sum()),
        "grids": grids,
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
    graph_rows = []
    for gp in graph_paths:
        row = _audit_graph(gp, grid_s=grid_s, sectors=sectors)
        if row:
            graph_rows.append(row)

    flags: List[str] = []
    if not graph_rows:
        flags.append("no_graphs")
    else:
        for g in graph_rows:
            for gr in g["grids"]:
                if gr.get("error") == "degenerate_abscissa":
                    flags.append("degenerate_abscissa")
                elif gr.get("occupancy_ratio", 1.0) < 0.15:
                    flags.append("sparse_grid")
                elif gr.get("empty_rows", 0) > gr.get("grid_s", 64) * 0.3:
                    flags.append("abscissa_gaps")

    occ_ratios = [
        gr["occupancy_ratio"]
        for g in graph_rows
        for gr in g["grids"]
        if "occupancy_ratio" in gr
    ]
    return {
        "case": case_rel,
        "n_graphs_audited": len(graph_rows),
        "mean_occupancy_ratio": round(float(np.mean(occ_ratios)), 4) if occ_ratios else None,
        "flags": sorted(set(flags)),
        "graphs": graph_rows,
    }


def _summarize(cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(cases)
    n_degen = sum(1 for c in cases if "degenerate_abscissa" in c.get("flags", []))
    n_sparse = sum(1 for c in cases if "sparse_grid" in c.get("flags", []))
    n_gaps = sum(1 for c in cases if "abscissa_gaps" in c.get("flags", []))
    occ = [c["mean_occupancy_ratio"] for c in cases if c.get("mean_occupancy_ratio") is not None]
    return {
        "n_cases": n,
        "n_degenerate_abscissa": n_degen,
        "n_sparse_grid": n_sparse,
        "n_abscissa_gaps": n_gaps,
        "mean_occupancy_ratio": round(float(np.mean(occ)), 4) if occ else None,
        "phase0_gate": n_degen == 0 and (not occ or float(np.mean(occ)) >= 0.20),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="V3P G4 2D 展开可行性审计")
    ap.add_argument("--split", type=Path, default=REPO_ROOT / "training" / "splits" / "split_AG_v1.json")
    ap.add_argument("--data-root", type=Path, default=REPO_ROOT / "data_new" / "AG")
    ap.add_argument("--grid-s", type=int, default=64)
    ap.add_argument("--sectors", type=int, nargs="+", default=[1, 4, 8])
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
            case_rel, data_root,
            grid_s=args.grid_s,
            sectors=args.sectors,
            max_graphs=args.max_graphs_per_case,
        )
        for case_rel in sorted(active)
    ]

    by_split = {
        "train": _summarize([r for r in case_rows if r["case"] in split.train_cases]),
        "val": _summarize([r for r in case_rows if r["case"] in split.val_cases]),
        "test": _summarize([r for r in case_rows if r["case"] in split.test_cases]),
    }
    overall = _summarize(case_rows)

    report: Dict[str, Any] = {
        "label": "V3P-G4-unwrap-feasibility",
        "date": date.today().isoformat(),
        "split": str(args.split.resolve()),
        "data_root": str(data_root),
        "grid_s": args.grid_s,
        "sectors": list(args.sectors),
        "oracle_ref": {
            "pod_2d_source": "outputs/field/f0_decision/v3_f0_oracle_v2.json",
            "pod_2d_20modes_n4_r2": 0.672,
            "g0b_gate": ">0.6 at 20 modes",
            "note": "F0 oracle 用 magnitude WSS；G4 训练须 split-safe 且含矢量分量",
        },
        "summary": overall,
        "by_split": by_split,
        "interpretation": (
            f"Phase 0 {'PASS' if overall['phase0_gate'] else 'REVIEW'}: "
            f"{overall['n_degenerate_abscissa']}/{overall['n_cases']} 例 Abscissa 退化；"
            f"平均网格 occupancy {overall.get('mean_occupancy_ratio', 'n/a')}。"
            " occupancy<0.15 或 abscissa_gaps 提示全局展开困难 → 优先 Geodesic Patch 方案。"
        ),
        "cases": case_rows,
    }

    out = args.output or (
        REPO_ROOT / "outputs" / "field" / "f0_decision" / f"v3p_g4_unwrap_feasibility_{date.today():%Y%m%d}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: report[k] for k in report if k != "cases"}, indent=2, ensure_ascii=False))
    print(out)


if __name__ == "__main__":
    main()
