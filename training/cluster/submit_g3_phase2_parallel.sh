#!/usr/bin/env bash
# G3 Phase 2 · 4 路 GPU 并行：SSL pretrain seed1/2 + post5463 无 SSL 基线 seed1/2
# 用法（仓库根）: bash training/cluster/submit_g3_phase2_parallel.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

mkdir -p training/cluster/logs

echo "=== G3 Phase 2 并行提交（4×GPU）==="
echo "工作目录: $ROOT"
echo ""

submit_ssl() {
  local seed=$1
  local jid
  jid=$(sbatch --job-name="g3_ssl_s${seed}" \
    training/cluster/run_pretrain_field_ssl.slurm \
    --cases pretrain \
    --epochs 150 \
    --seed "${seed}" \
    --exp-id V3P-G-58-SSL-Pretrain \
    | awk '{print $NF}')
  echo "SSL Pretrain seed${seed}  -> Job ${jid}"
}

submit_baseline() {
  local seed=$1
  local cfg="training/configs/field/generated/v3_pointcloud/V3P-G-Baseline-AsymW-a-post5463_seed${seed}.json"
  local jid
  jid=$(sbatch --job-name="g3_bl_s${seed}" \
    training/cluster/run_train_field.slurm \
    "${cfg}" \
    | awk '{print $NF}')
  echo "Baseline post5463 seed${seed} -> Job ${jid} (${cfg})"
}

submit_ssl 1
submit_baseline 1
submit_ssl 2
submit_baseline 2

echo ""
echo "监控: squeue -u \$USER"
echo "Phase 3（SSL 微调）需等 SSL pretrain ckpt 就绪后再提交。"
