#!/bin/bash
# 2026-09-21 重跑前准备（与 RCR 批次的 prepare_rerun.sh 同做法，戳 old_20260921）：
#   1) 把旧 ascii/ ascii_in/ libudf/ Global_conditions/ 挪成 *_old_20260921（Fluent 导出不覆盖同名文件；旧 libudf 会被跳过重编）
#   2) 重建空 ascii/（.cas 的 export-2 直接写 ascii/<case>-NNNN，目录必须先存在）
#   3) fluent.slurm 统一：-w <节点>、--ntasks-per-node=64、-t64（原件备份 fluent.slurm.orig_20260921）
# 用法： bash prepare_rerun_20260921.sh <节点> <病例目录> [<病例目录> ...]
set -u
STAMP=old_20260921
NODE=$1; shift
for d in "$@"; do
  [ -d "$d" ] || { echo "跳过（不是目录）: $d"; continue; }
  [ -f "$d/udf-inlet4.c" ] || [ -f "$d/udf-inlet.c" ] || { echo "跳过（无 UDF）: $d"; continue; }
  for sub in ascii ascii_in libudf Global_conditions; do
    if [ -e "$d/$sub" ]; then
      if [ -e "$d/${sub}_$STAMP" ]; then echo "  已存在 $d/${sub}_$STAMP，不动 $sub"; else mv "$d/$sub" "$d/${sub}_$STAMP" && echo "  $d/$sub -> ${sub}_$STAMP"; fi
    fi
  done
  mkdir -p "$d/ascii"
  s="$d/fluent.slurm"
  [ -f "$s.orig_20260921" ] || cp -p "$s" "$s.orig_20260921"
  sed -i -E "s/^#SBATCH -w .*/#SBATCH -w $NODE/; s/^#SBATCH --ntasks-per-node=.*/#SBATCH --ntasks-per-node=64/; s/fluent 3ddp -g -t[0-9]+/fluent 3ddp -g -t64/" "$s"
  grep -q "^#SBATCH -w " "$s" || sed -i "s/^#SBATCH --nodes=1/#SBATCH --nodes=1\n#SBATCH -w $NODE/" "$s"
  echo "就绪: $d  节点 $(grep '^#SBATCH -w' $s | awk '{print $3}')  核数 $(grep -o 'fluent 3ddp -g -t[0-9]*' $s)"
done
