#!/bin/bash
# V3P-K1 BranchFeat 短 GPU probe（post5463 · 40ep · seed1）
set -eo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_DIR"
mkdir -p logs

CFG="training/configs/field/generated/v3_pointcloud/V3P-K1-BranchFeat-MixedProbe-post5463_seed1.json"

echo "=== submit K1 BranchFeat GPU probe $(date) ==="
echo "config: $CFG"

JOB=$(sbatch -p GPU --gres=gpu:1 training/cluster/run_train_field.slurm "$CFG" | awk '{print $4}')
echo "train job: $JOB"
echo "monitor: tail -f logs/run_train_field_${JOB}.out"
