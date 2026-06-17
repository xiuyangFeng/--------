#!/usr/bin/env bash
# G3 · post5463 无 SSL 基线 seed3（1×GPU · 配方同 5466/5468）
# 用法（仓库根）: bash training/cluster/submit_g3_baseline_post5463_seed3.sh [GPU_ID]
# 默认 GPU_ID=1（与 5475 并行时避开 GPU 0）

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

GPU_ID="${1:-1}"
CFG="training/configs/field/generated/v3_pointcloud/V3P-G-Baseline-AsymW-a-post5463_seed3.json"

if [ ! -f "$CFG" ]; then
  echo "错误: 配置文件不存在: $CFG"
  exit 1
fi

echo "=== post5463 基线 seed3（GPU ${GPU_ID}）==="
echo "配置: $CFG"
echo ""

job_id=$(CUDA_VISIBLE_DEVICES="${GPU_ID}" sbatch --job-name="g3_bl_s3" \
  --export=ALL,CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  training/cluster/run_train_field.slurm \
  "${CFG}" \
  | awk '{print $NF}')
echo "Baseline post5463 seed3 -> Job ${job_id}"
echo "监控: squeue -u \$USER · tail -f logs/g3_bl_s3_${job_id}.out"
