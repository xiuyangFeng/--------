#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
路线 B（Fluent 内插值）：从 export CSV 生成 Fluent Cloud-of-Points 友好的精简 CSV。

在 Fluent 中：Mesh → Interpolate → Cloud of Points，分别插 wss_cfd / wss_pred / p_cfd / p_pred。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_fluent_cloud(df: pd.DataFrame, value_col: str, out_path: Path, var_name: str) -> None:
    if value_col not in df.columns:
        raise SystemExit(f"列不存在: {value_col}")
    sub = df[["x", "y", "z", value_col]].dropna()
    sub = sub.rename(columns={value_col: var_name})
    sub.to_csv(out_path, index=False, float_format="%.8g")
    print(f"  [{var_name}] {len(sub)} points → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 Fluent 点云插值用 CSV（路线 B）")
    parser.add_argument("--csv", required=True, help="export_for_cfdpost 的 __wall.csv 或 __interior.csv")
    parser.add_argument(
        "--output-dir",
        default="tools/cfdpost_cloud_export/output/route_interp/fluent_cloud",
        help="输出目录",
    )
    parser.add_argument(
        "--fields",
        default="wss_cfd,wss_pred,p_cfd,p_pred",
        help="要导出的标量列",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv).resolve()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    stem = csv_path.stem

    for field in [f.strip() for f in args.fields.split(",") if f.strip()]:
        out = output_dir / f"{stem}__fluent_{field}.csv"
        _write_fluent_cloud(df, field, out, field)


if __name__ == "__main__":
    main()
