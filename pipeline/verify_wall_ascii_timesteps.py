#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
壁面 ascii 多时间步质检（Fluent 导出后 / pipeline 步骤 1 前）。

统计每个病例壁面目录下各时间步的 pressure 均值，识别「多数帧为 0」的导出失败
（勿与体场 ascii_in 混淆；勿用单帧中间步代表全序列）。

示例:
  # Fluent 完成后验收 4 例
  python -m pipeline.verify_wall_ascii_timesteps \\
    --data-root data_new \\
    --cases ILO/DONG_KE_QIN-0/after ILO/DONG_KE_QIN-0/before \\
          ILO/GUO_AI_JUN-0/after ILO/GUO_AI_JUN-0/before

  # 扫描 split train 中 ILO 全例（较慢）
  python -m pipeline.verify_wall_ascii_timesteps \\
    --data-root data_new --from-split training/splits/split_data_new_v3.json --subset train --domain ILO
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from .config import DATA_ROOT
except ImportError:
    from config import DATA_ROOT

# 壁面压力低于此阈值 (Pa) 视为「近零帧」
P_ZERO_THRESHOLD_PA = 1.0


def resolve_wall_ascii_dir(case_dir: Path) -> Optional[Path]:
    """返回壁面 ascii 目录（精确名 ascii；排除 ascii_in）。"""
    exact = case_dir / "ascii"
    if exact.is_dir():
        return exact
    for child in case_dir.iterdir():
        if child.is_dir() and child.name.strip() == "ascii":
            return child
    return None


def wall_ascii_timestep_stats(ascii_file: Path) -> Tuple[float, float, int]:
    """读取单帧壁面 ascii，返回 (p_mean, wss_max, n_points)。"""
    pres: List[float] = []
    wss_vals: List[float] = []
    with ascii_file.open(encoding="utf-8", errors="replace") as fh:
        next(fh, None)
        for ln in fh:
            parts = [p.strip() for p in ln.split(",")]
            if len(parts) < 6:
                continue
            pres.append(float(parts[4]))
            wss_vals.append(float(parts[5]))
    if not pres:
        return float("nan"), float("nan"), 0
    return float(np.mean(pres)), float(np.max(np.abs(wss_vals))), len(pres)


def scan_case(case_rel: str, data_root: Path) -> Dict:
    case_dir = data_root / case_rel
    wall_dir = resolve_wall_ascii_dir(case_dir)
    if wall_dir is None:
        return {
            "case": case_rel,
            "ok": False,
            "error": "missing_wall_ascii_dir",
        }

    files = sorted(f for f in wall_dir.iterdir() if f.is_file())
    if not files:
        return {
            "case": case_rel,
            "ok": False,
            "error": "no_ascii_files",
            "wall_dir": str(wall_dir),
        }

    zero_frames: List[str] = []
    ok_frames: List[Tuple[str, float, float]] = []
    for f in files:
        pm, wm, n = wall_ascii_timestep_stats(f)
        if pm < P_ZERO_THRESHOLD_PA or not np.isfinite(pm):
            zero_frames.append(f.name)
        else:
            ok_frames.append((f.name, pm, wm))

    n_total = len(files)
    n_zero = len(zero_frames)
    n_ok = len(ok_frames)
    frac_ok = n_ok / n_total if n_total else 0.0

    # 通过：至少 90% 时间步壁面压力有效，且存在有效帧
    passed = n_ok > 0 and frac_ok >= 0.9

    rec: Dict = {
        "case": case_rel,
        "wall_dir": str(wall_dir.relative_to(data_root)),
        "n_timesteps": n_total,
        "n_zero_wall_p": n_zero,
        "n_ok_wall_p": n_ok,
        "frac_ok": round(frac_ok, 4),
        "ok": passed,
    }
    if ok_frames:
        rec["example_ok"] = {
            "file": ok_frames[0][0],
            "p_mean": round(ok_frames[0][1], 2),
            "wss_max": round(ok_frames[0][2], 4),
        }
    if zero_frames:
        rec["example_zero"] = zero_frames[:3]
        if len(zero_frames) > 3:
            rec["example_zero"].append(f"... +{len(zero_frames) - 3} more")
    return rec


def load_cases_from_split(split_path: Path, subset: str, domain: Optional[str]) -> List[str]:
    with split_path.open(encoding="utf-8") as f:
        split = json.load(f)
    key = {"train": "train_cases", "val": "val_cases", "test": "test_cases"}[subset]
    cases = list(split.get(key, []))
    if domain:
        prefix = domain.rstrip("/") + "/"
        cases = [c for c in cases if c.startswith(prefix)]
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description="壁面 ascii 多时间步质检")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--cases", nargs="*", default=None, help="病例相对路径列表")
    parser.add_argument("--from-split", type=str, default=None, help="从 split JSON 读取病例")
    parser.add_argument(
        "--subset",
        choices=["train", "val", "test"],
        default="train",
    )
    parser.add_argument("--domain", type=str, default=None, help="仅扫描某域，如 ILO / AAA / AG")
    parser.add_argument(
        "--min-frac-ok",
        type=float,
        default=0.9,
        help="通过线：有效壁面时间步占比（默认 0.9）",
    )
    parser.add_argument("--json-out", type=str, default=None, help="写入 JSON 报告路径")
    parser.add_argument(
        "--fail-on-bad",
        action="store_true",
        help="存在未通过病例时 exit 1",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root) if args.data_root else DATA_ROOT
    if args.cases:
        cases = args.cases
    elif args.from_split:
        cases = load_cases_from_split(Path(args.from_split), args.subset, args.domain)
    else:
        print("请指定 --cases 或 --from-split")
        sys.exit(2)

    if not cases:
        print("病例列表为空")
        sys.exit(2)

    reports: List[Dict] = []
    n_pass = 0
    for case_rel in cases:
        rec = scan_case(case_rel, data_root)
        # 覆盖通过线
        if rec.get("n_timesteps"):
            rec["ok"] = (
                rec.get("n_ok_wall_p", 0) > 0
                and rec.get("frac_ok", 0) >= args.min_frac_ok
            )
        reports.append(rec)
        status = "✅" if rec.get("ok") else "❌"
        if rec.get("ok"):
            n_pass += 1
        err = rec.get("error", "")
        detail = (
            f"n_ok={rec.get('n_ok_wall_p', '?')}/{rec.get('n_timesteps', '?')} "
            f"frac_ok={rec.get('frac_ok', '?')}"
        )
        print(f"{status} {case_rel}  {detail}" + (f"  ({err})" if err else ""))

    summary = {
        "data_root": str(data_root.resolve()),
        "n_cases": len(reports),
        "n_pass": n_pass,
        "min_frac_ok": args.min_frac_ok,
        "p_zero_threshold_pa": P_ZERO_THRESHOLD_PA,
        "cases": reports,
    }
    print(f"\n合计: {n_pass}/{len(reports)} 通过")

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"已写入: {out}")

    if args.fail_on_bad and n_pass < len(reports):
        sys.exit(1)


if __name__ == "__main__":
    main()
