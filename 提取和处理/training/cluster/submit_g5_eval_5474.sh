#!/usr/bin/env bash
# G5 eval · 5474 SSL Finetune seed1 病例级 Pa（CPU · node03 · 不占 GPU）
# 用法（仓库根）: bash training/cluster/submit_g5_eval_5474.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

RUN_DIR_REL="outputs/field/field_v3_pointnext_localpool_main01_geom_pw_asymw_a_wall13000_near2000_split_AG_v1_seed1_20260609_124213"
LABEL="5474-V3P-G-58-SSL-Finetune-s1"

if [ ! -d "$RUN_DIR_REL" ]; then
  echo "错误: run 目录不存在: $RUN_DIR_REL"
  exit 1
fi

echo "提交 G5 eval: $LABEL -> $RUN_DIR_REL"
job_id=$(RUN_DIR_REL="$RUN_DIR_REL" \
  EVAL_EXTRA_ARGS='--clinical-pa --force' \
  sbatch --job-name="g5_eval_${LABEL}" \
    training/cluster/run_evaluate_field_run_full_cpu.slurm \
  | awk '{print $NF}')
echo "  Job $job_id"
echo "监控: squeue -u \$USER · tail -f logs/g5_eval_${LABEL}_${job_id}.out"
