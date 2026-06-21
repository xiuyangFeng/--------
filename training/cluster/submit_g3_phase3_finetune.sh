#!/usr/bin/env bash
# G3 Phase 3 · 2 路 GPU 并行：SSL 微调 seed1/2 vs 5466/5468 无 SSL 基线 band
# 用法（仓库根）: bash training/cluster/submit_g3_phase3_finetune.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

mkdir -p training/cluster/logs

echo "=== G3 Phase 3 SSL 微调提交（2×GPU）==="
echo "工作目录: $ROOT"
echo ""

submit_finetune() {
  local seed=$1
  local cfg="training/configs/field/generated/v3_pointcloud/V3P-G-58-SSL-Finetune_seed${seed}.json"
  local ssl_ckpt="outputs/field/ssl_v3p_g58_pretrain_split_AG_v1_seed${seed}_20260608_154636/ssl_encoder.pt"
  if [ ! -f "$ssl_ckpt" ]; then
    echo "错误: 缺少 SSL encoder ckpt: $ssl_ckpt"
    exit 1
  fi
  local jid
  jid=$(sbatch --job-name="g3_ft_s${seed}" \
    training/cluster/run_train_field.slurm \
    "${cfg}" \
    | awk '{print $NF}')
  echo "SSL Finetune seed${seed} -> Job ${jid} (${cfg})"
}

submit_finetune 1
submit_finetune 2

echo ""
echo "监控: squeue -u \$USER"
echo "对照 band: 5466/5468 wss 0.433±0.005 · Go: Δ≥+0.03 且 r2_p 不崩"
