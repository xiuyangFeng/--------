#!/bin/bash
# 重跑前准备：把旧导出/旧编译产物挪到 *_old_20260915，避免 Fluent 不覆盖同名文件、避免旧 libudf 干扰。
# 用法：bash prepare_rerun.sh <病例目录> [<病例目录> ...]   或   bash prepare_rerun.sh $(cat rerun_cases.txt)
# 之后在每个病例目录里：export PATH=/public/slurm/bin:$PATH && sbatch fluent.slurm
set -u
STAMP=old_20260915
for d in "$@"; do
  [ -d "$d" ] || { echo "跳过（不是目录）: $d"; continue; }
  [ -f "$d/udf-inlet4.c" ] || [ -f "$d/udf-inlet.c" ] || { echo "跳过（无 UDF）: $d"; continue; }
  for sub in ascii ascii_in libudf Global_conditions; do
    if [ -e "$d/$sub" ]; then
      if [ -e "$d/${sub}_$STAMP" ]; then echo "  已存在 $d/${sub}_$STAMP，不动 $sub"; else mv "$d/$sub" "$d/${sub}_$STAMP" && echo "  $d/$sub -> ${sub}_$STAMP"; fi
    fi
  done
  mkdir -p "$d/ascii" && echo "  已建空 $d/ascii（.cas 里壁面导出 export-2 写到 ascii/<case>-NNNN，目录必须先存在；体场导出 export-1 写到病例目录，fluent.slurm 跑完后搬进 ascii_in/）"
  echo "就绪: $d  （cd 进去 sbatch fluent.slurm）"
done
