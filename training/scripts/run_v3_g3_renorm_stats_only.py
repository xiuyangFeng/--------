#!/usr/bin/env python3
"""V3P AG · train-only 重算 normalization_params_global.json（备份 + 保存，不重跑 CSV）。"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import date
from pathlib import Path

from ..core.denylist import filter_case_names
from ..core.splits import SplitSpec
from pipeline.normalize import collect_global_statistics, save_normalization_params

REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser(description="V3P AG train-only 归一化统计重算")
    ap.add_argument("--split", type=Path, default=REPO_ROOT / "training/splits/split_AG_v1.json")
    ap.add_argument("--data-root", type=Path, default=REPO_ROOT / "data_new/AG")
    ap.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "data_new/normalization_params_global.json",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    split = SplitSpec.from_json(args.split.resolve())
    data_root = args.data_root.resolve()
    train_cases = filter_case_names(split.train_cases, data_root)
    case_dirs = [data_root / c for c in train_cases if (data_root / c).is_dir()]

    stats = collect_global_statistics(
        case_dirs,
        "processed/coord_normalized",
        train_cases=train_cases,
        data_root=data_root,
    )
    if not stats:
        raise SystemExit("未能收集统计量")

    out = args.output.resolve()
    report = {
        "label": "V3P-G3-renorm-stats-only",
        "date": str(date.today()),
        "n_train_cases": len(train_cases),
        "output": str(out),
        "dry_run": args.dry_run,
        "note": "仅更新 JSON；完整 CSV/graph 一致化请跑 run_v3p_ag_renorm_regraph.slurm",
    }
    report_path = REPO_ROOT / "outputs/field/f0_decision" / f"v3p_g3_renorm_stats_{date.today().strftime('%Y%m%d')}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.dry_run:
        if out.is_file():
            bak = out.with_suffix(out.suffix + f".bak_{date.today().strftime('%Y%m%d')}")
            shutil.copy2(out, bak)
            report["backup"] = str(bak)
        save_normalization_params(stats, str(out))
        ag_link = data_root / "normalization_params_global.json"
        if ag_link.is_symlink() or ag_link.exists():
            report["ag_link"] = str(ag_link)

    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
