#!/bin/bash
# 第二批 15 例（运行态 RCR 审计修复后）分两波提交：第 1 波 8 例立即；第 2 波 7 例各自 afterany 依赖第 1 波的一例，
# 使同时在跑的不超过 ~8 例（每例体场导出峰值约 170 GB，/public 剩 3.5 TB）。用法：bash submit_batch2.sh
set -u
export PATH=/public/slurm/bin:$PATH
R=/public/newhome/cy/Digital_twin/GNN/data_new
HERE=$(cd "$(dirname "$0")" && pwd)
mapfile -t CASES < "$HERE/rerun_batch2_cases.txt"
LOG="$HERE/rerun_jobs.md"
W1=(); i=0
for c in "${CASES[@]}"; do
  if [ $i -lt 8 ]; then
    J=$(cd "$R/$c" && sbatch --parsable fluent.slurm); W1+=("$J")
    echo "| \`$c\`（第 2 批·波 1） | $J | $(date -Is) | (Slurm 自选) |" >> "$LOG"; echo "$c -> $J"
  else
    dep=${W1[$((i-8))]}
    J=$(cd "$R/$c" && sbatch --parsable --dependency=afterany:$dep fluent.slurm)
    echo "| \`$c\`（第 2 批·波 2，afterany:$dep） | $J | $(date -Is) | (Slurm 自选) |" >> "$LOG"; echo "$c -> $J (afterany:$dep)"
  fi
  i=$((i+1))
done
sleep 30; squeue -u cy -o "%.8i %.12j %.2t %.6M %.4C %R" | grep -v "lc_queue\|cycle"
