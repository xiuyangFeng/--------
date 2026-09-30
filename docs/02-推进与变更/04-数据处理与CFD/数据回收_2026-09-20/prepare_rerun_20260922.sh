#!/bin/bash
# 2026-09-22 第三批（RCR 协议重算 13 例）重跑前准备，戳 old_20260922；与 prepare_rerun_20260921.sh 同做法但不指定节点（Slurm 自选空闲节点）：
#   1) 把当前 ascii/ ascii_in/ libudf/ Global_conditions/ 挪成 *_old_20260922（Fluent 导出不覆盖同名文件；旧 libudf 会被跳过重编）
#   2) 重建空 ascii/
#   3) fluent.slurm：去掉 -w、--ntasks-per-node=64、-t64（原件备份 fluent.slurm.orig_20260922）
# 用法： bash prepare_rerun_20260922.sh <病例目录> [<病例目录> ...]
set -u
STAMP=${STAMP:-old_20260922}
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
  [ -f "$s.orig_20260922" ] || cp -p "$s" "$s.orig_20260922"
  sed -i -E "/^#SBATCH -w /d; s/^#SBATCH --ntasks-per-node=.*/#SBATCH --ntasks-per-node=64/; s/fluent 3ddp -g -t[0-9]+/fluent 3ddp -g -t64/" "$s"
  echo "就绪: $d  节点 $(grep -c '^#SBATCH -w' $s | sed 's/^0$/自选/')  核数 $(grep -o 'fluent 3ddp -g -t[0-9]*' $s)  ascii 空: $(ls $d/ascii | wc -l)"
done
