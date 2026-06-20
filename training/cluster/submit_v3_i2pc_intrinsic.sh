#!/usr/bin/env bash
# V3P 路径 I2-PC · intrinsic target point-cloud probe（seed1）
# 用法（仓库根）:
#   bash training/cluster/submit_v3_i2pc_intrinsic.sh [GPU_ID]

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

GPU_ID="${1:-0}"
CFG="training/configs/field/generated/v3_pointcloud/V3P-I2-PC-Intrinsic_seed1.json"
MANIFEST="data_new/AG/i2pc_intrinsic_manifest.json"

if [ ! -f "$CFG" ]; then
  echo "错误: 配置文件不存在: $CFG"
  exit 1
fi
if [ ! -f "$MANIFEST" ]; then
  echo "错误: I2-PC 图目录尚未准备；请先运行:"
  echo "  /public/newhome/cy/.conda/envs/GNN/bin/python -m training.scripts.prepare_v3_i2pc_intrinsic_graphs --overwrite"
  exit 1
fi

job_id=$(CUDA_VISIBLE_DEVICES="${GPU_ID}" sbatch --job-name="v3p_i2pc_s1" \
  --export=ALL,CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  training/cluster/run_train_field.slurm \
  "${CFG}" \
  | awk '{print $NF}')

echo "=== V3P I2-PC intrinsic seed1 已提交 ==="
echo "Job ${job_id} · GPU ${GPU_ID}"
echo "配置: ${CFG}"
echo "监控: squeue -u \$USER · tail -f logs/v3p_i2pc_s1_${job_id}.out"
