#!/bin/bash
# 导出文件瘦身：ascii/ 与 ascii_in/ 只保留 1120 号与 1122–1280 偶数号（共 81 帧，V5 母库 EXPECTED_STEPS），其余删除。
# 用法：bash thin_exports.sh <病例目录> ...   （只对 Slurm 已 COMPLETED 的病例执行）
set -u
for d in "$@"; do
  for sub in ascii ascii_in; do
    [ -d "$d/$sub" ] || continue
    n_del=0
    for f in "$d/$sub"/*-[0-9][0-9][0-9][0-9]; do
      [ -e "$f" ] || continue
      n=${f##*-}; n=$((10#$n))
      if [ "$n" -lt 1120 ] || { [ "$n" -gt 1120 ] && [ $((n % 2)) -eq 1 ]; } || [ "$n" -gt 1280 ]; then rm -f "$f"; n_del=$((n_del+1)); fi
    done
    echo "$d/$sub: 删 $n_del，剩 $(ls "$d/$sub" | wc -l)"
  done
done
