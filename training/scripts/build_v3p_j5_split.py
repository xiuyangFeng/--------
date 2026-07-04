#!/usr/bin/env python3
"""将 split_AG_active_v2 转为 data_new 根路径格式（AG/fast/… + AAA/…）。

用法::

    python -m training.scripts.build_v3p_j5_split
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from ..core.splits import SplitSpec
from ._figure_utils import load_json, save_json
from .run_v3_f0_decision import REPO_ROOT


def _to_data_new_case(case: str) -> str:
    if case.startswith(("AAA/", "AG/", "ILO/")):
        return case
    if case.startswith(("fast/", "slow/")):
        return f"AG/{case}"
    return case


def build_j5_split(
    active_v2_path: Path,
    *,
    output_path: Path,
) -> dict:
    raw = load_json(active_v2_path)
    base = SplitSpec.from_json(active_v2_path)

    def conv(cases: List[str]) -> List[str]:
        return sorted({_to_data_new_case(c) for c in cases})

    out = {
        **raw,
        "split_version": "split_AG_active_v2_j5",
        "source": "AG test + train 增补 6×AAA（Phase1 v4 · data_new 路径）",
        "derived_from": {
            **raw.get("derived_from", {}),
            "active_v2": str(active_v2_path),
        },
        "train_cases": conv(base.train_cases),
        "val_cases": conv(base.val_cases),
        "test_cases": conv(base.test_cases),
        "active_added": conv(raw.get("active_added", [])),
        "external_domain_added": conv(raw.get("external_domain_added", [])),
        "data_root": "data_new",
        "n_train_active": len(conv(base.train_cases)),
    }
    save_json(output_path, out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Build J5 data_new split")
    ap.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT / "training/splits/split_AG_active_v2.json",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "training/splits/split_AG_active_v2_j5.json",
    )
    args = ap.parse_args()
    out = build_j5_split(args.input.resolve(), output_path=args.output.resolve())
    print(json.dumps({
        "output": str(args.output),
        "n_train": len(out["train_cases"]),
        "n_active_added": len(out.get("active_added", [])),
        "active_added": out.get("active_added", []),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
