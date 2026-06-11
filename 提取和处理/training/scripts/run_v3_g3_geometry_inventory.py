#!/usr/bin/env python3
"""G3 预研：盘点 V3P AG 几何资产（STL + merged），供 SSL/预训练可行性评估（0 重训 · CPU）。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..core.denylist import filter_case_names
from ..core.splits import SplitSpec

REPO_ROOT = Path(__file__).resolve().parents[2]


def _case_assets(case_dir: Path) -> dict:
    merged = case_dir / "processed" / "merged"
    graphs = case_dir / "processed" / "graphs"
    stl = list(case_dir.glob("*.stl"))
    n_merged = len(list(merged.glob("*.csv"))) if merged.is_dir() else 0
    n_graphs = len(list(graphs.glob("*.pt"))) if graphs.is_dir() else 0
    return {
        "has_stl": bool(stl),
        "n_merged_csv": n_merged,
        "n_graphs": n_graphs,
        "geometry_ready": bool(stl) and n_merged > 0,
        "graphs_ready": n_graphs > 0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="V3P G3 几何资产盘点")
    ap.add_argument("--split", type=Path, default=REPO_ROOT / "training" / "splits" / "split_AG_v1.json")
    ap.add_argument("--data-root", type=Path, default=REPO_ROOT / "data_new" / "AG")
    ap.add_argument("--output", type=Path, default=REPO_ROOT / "outputs" / "field" / "f0_decision" / "v3p_g3_geometry_inventory.json")
    args = ap.parse_args()

    split = SplitSpec.from_json(args.split.resolve())
    data_root = args.data_root.resolve()
    active = filter_case_names(
        list(split.train_cases) + list(split.val_cases) + list(split.test_cases),
        data_root,
    )

    rows = []
    for case_rel in sorted(active):
        assets = _case_assets(data_root / case_rel)
        rows.append({"case": case_rel, **assets})

    n_geo = sum(1 for r in rows if r["geometry_ready"])
    n_graph = sum(1 for r in rows if r["graphs_ready"])
    report = {
        "label": "V3P-G3-geometry-inventory",
        "split": str(args.split.resolve()),
        "data_root": str(data_root),
        "n_active_cases": len(rows),
        "n_geometry_ready": n_geo,
        "n_graphs_ready": n_graph,
        "interpretation": (
            f"{n_geo}/{len(rows)} 例具备 STL+merged，可用于几何 SSL/预训练预研；"
            "须 split-safe，预训练不得使用 test 标签或 test 归一化统计。"
        ),
        "cases": rows,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: report[k] for k in report if k != "cases"}, indent=2, ensure_ascii=False))
    print(args.output)


if __name__ == "__main__":
    main()
