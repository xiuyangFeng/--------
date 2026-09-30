#!/bin/bash
# 2026-09-26: submit the synthetic children's Fluent runs in waves (<= MAXPAR concurrently via afterany chains); usage: bash submit_synth_cfd.sh [MAXPAR=10]
set -u
export PATH=/public/slurm/bin:$PATH
ROOT=/public/newhome/cy/Digital_twin/GNN; HERE="$ROOT/docs/02-推进与变更/04-数据处理与CFD/数据回收_2026-09-20/synth_20260926"; R=$ROOT/data_new
MAXPAR=${1:-10}
LIST_FILE=${2:-$HERE/synth_children.txt}; mapfile -t KIDS < "$LIST_FILE"
LOG="$HERE/cfd_jobs.md"; [ -f "$LOG" ] || printf "| 子例 | 作业 | 提交时间 | 依赖 |\n|---|---|---|---|\n" > "$LOG"
declare -a CHAIN; i=0
for c in "${KIDS[@]}"; do
  d="$R/$c"; [ -f "$d/fluent.slurm" ] || { echo "no fluent.slurm: $c"; exit 1; }
  [ -d "$d/libudf" ] && { echo "libudf still present (smoke did not rename?): $c"; exit 1; }
  [ "$(ls "$d/ascii" 2>/dev/null | wc -l)" -eq 0 ] || { echo "ascii/ not empty: $c"; exit 1; }
  slot=$((i % MAXPAR)); dep=""
  if [ $i -ge $MAXPAR ]; then dep="--dependency=afterany:${CHAIN[$slot]}"; fi
  J=$(cd "$d" && sbatch --parsable --exclude=node05 $dep fluent.slurm); CHAIN[$slot]=$J
  echo "| \`$c\` | $J | $(date -Is) | ${dep:-none} |" >> "$LOG"; echo "$c -> $J ${dep}"
  i=$((i+1))
done
sleep 20; squeue -u cy -h -o "%.8i %.10j %.2t %.6M %R" | grep -c Fluent
