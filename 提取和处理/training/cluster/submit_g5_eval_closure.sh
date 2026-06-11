#!/usr/bin/env bash
# G5 eval 闭环：5311 PCv2-BLContext + 5439 post-denylist AsymW-a 基线
# 用法（仓库根）: bash training/cluster/submit_g5_eval_closure.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

RUNS=(
  "outputs/field/field_v3_pointnext_localpool_pcv2_blctx_wall13000_near2000_split_AG_v1_seed1_20260602_145347"
  "outputs/field/field_v3_pointnext_localpool_main01_geom_pw_asymw_a_wall13000_near2000_split_AG_v1_seed1_20260607_115628"
)

LABELS=(
  "5311-V3P-F-PCv2-BLContext"
  "5439-V3P-G-Baseline-AsymW-a"
)

for i in "${!RUNS[@]}"; do
  rel="${RUNS[$i]}"
  label="${LABELS[$i]}"
  if [ ! -d "$rel" ]; then
    echo "跳过 $label: 目录不存在 $rel"
    continue
  fi
  echo "提交 G5 eval: $label -> $rel"
  job_id=$(RUN_DIR_REL="$rel" \
    EVAL_EXTRA_ARGS='--clinical-pa --force' \
    sbatch --job-name="g5_eval_${label}" \
      training/cluster/run_evaluate_field_run_full_cpu.slurm \
    | awk '{print $NF}')
  echo "  Job $job_id"
done

echo "监控: squeue -u \$USER"
