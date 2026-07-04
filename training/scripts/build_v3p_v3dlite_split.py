#!/usr/bin/env python3
"""从 Idea G oracle JSON 生成 V3P V3D-lite mixed split（AG train + near-domain 增补）。

用法::

    python -m training.scripts.build_v3p_v3dlite_split \\
        --audit outputs/field/f0_decision/v3p_v3d_lite_data_audit_20260630.json \\
        --base-split training/splits/split_AG_v1.json
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Set

from ..core.splits import SplitSpec
from ..core.utils import ensure_dir
from ._figure_utils import load_json, save_json
from .run_v3_f0_decision import REPO_ROOT


def _normalize_ag_case(case: str) -> str:
    """``AG/fast/FOO`` → ``fast/FOO``；已是 ``fast/FOO`` 则不变。"""
    if case.startswith("AG/"):
        return case[3:]
    return case


def build_lite_split(
    audit_path: Path,
    base_split_path: Path,
    *,
    output_path: Path,
) -> Dict[str, Any]:
    audit = load_json(audit_path)
    lite_raw: List[str] = audit.get("pools", {}).get("V3D-lite-candidates", {}).get("cases", [])
    lite_ag = sorted({_normalize_ag_case(c) for c in lite_raw})

    base = SplitSpec.from_json(base_split_path)
    train: Set[str] = set(base.train_cases)
    val = list(base.val_cases)
    test = list(base.test_cases)

    added = [c for c in lite_ag if c not in train and c not in val and c not in test]
    for c in added:
        train.add(c)

    out = {
        "split_version": "split_AG_v3lite_v1",
        "source": "AG + V3D-lite near-domain (from platform oracle Idea G)",
        "derived_from": {
            "base_split": str(base_split_path.relative_to(REPO_ROOT))
            if base_split_path.is_relative_to(REPO_ROOT)
            else str(base_split_path),
            "audit_json": str(audit_path),
            "audit_date": audit.get("context", ""),
        },
        "train_cases": sorted(train),
        "val_cases": val,
        "test_cases": test,
        "lite_candidates_added": added,
        "n_train_base": len(base.train_cases),
        "n_train_lite": len(train),
    }
    save_json(output_path, out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Build V3P V3D-lite mixed split")
    ap.add_argument(
        "--audit",
        type=Path,
        default=REPO_ROOT / "outputs/field/f0_decision/v3p_v3d_lite_data_audit_20260630.json",
    )
    ap.add_argument("--base-split", type=Path, default=REPO_ROOT / "training/splits/split_AG_v1.json")
    ap.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "training/splits/split_AG_v3lite_v1.json",
    )
    args = ap.parse_args()
    out = build_lite_split(args.audit.resolve(), args.base_split.resolve(), output_path=args.output.resolve())
    print(json.dumps({
        "output": str(args.output),
        "n_added": len(out["lite_candidates_added"]),
        "n_train": out["n_train_lite"],
        "added": out["lite_candidates_added"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
