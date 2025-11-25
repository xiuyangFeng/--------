#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计点云/病例/ascii_clean 下的非壁面点数量（is_wall == 0）。
默认根目录为“点云”，可在命令行通过 --root 指定。
"""
from pathlib import Path
import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="检查 ascii_clean 中是否存在非壁面点")
    parser.add_argument("--root", default="点云", help="点云根目录，默认当前目录下的“点云”")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        print(f"❌ 根目录不存在: {root}")
        return

    any_non_wall = False
    any_missing = False

    for case_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        ac_dir = case_dir / "ascii_clean"
        if not ac_dir.is_dir():
            continue
        print(f"\n📂 病例 {case_dir.name}")
        for f in sorted(ac_dir.glob("*.csv")):
            try:
                df = pd.read_csv(f)
            except Exception as e:
                print(f"   ⚠️  {f.name} 读取失败: {e}")
                continue

            if "is_wall" not in df.columns:
                print(f"   ⚠️  {f.name} 缺少 is_wall 列，行数 {len(df)}")
                any_missing = True
                continue

            non_wall = int((df["is_wall"] == 0).sum())
            wall = int((df["is_wall"] == 1).sum())
            if non_wall > 0:
                any_non_wall = True
                print(f"   {f.name}: 非壁面 {non_wall}, 壁面 {wall}")
            else:
                print(f"   {f.name}: 全是壁面点 (壁面 {wall})")

    if not any_non_wall and not any_missing:
        print("\n✅ 所有文件均为壁面点且包含 is_wall 列。")
    elif not any_non_wall and any_missing:
        print("\nℹ️ 无非壁面点，但有文件缺少 is_wall 列，请检查。")

if __name__ == "__main__":
    main()
