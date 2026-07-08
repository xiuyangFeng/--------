#!/bin/bash
#SBATCH --partition=CPU
#SBATCH -w node03
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=k9_ray_audit
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=4:00:00

# 前沿方向 §15.2 · K9 剖面射线训练目标可行性审计（Gate-0 · 全量 train+test · 0 重训）
set -eo pipefail

module purge
PYTHON_BIN="${PYTHON_BIN:-/public/newhome/cy/.conda/envs/GNN/bin/python}"
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"
mkdir -p logs outputs/field/f0_decision

DATE_TAG="${DATE_TAG:-20260705}"

echo "=== K9 ray-target audit $(date) ==="
"$PYTHON_BIN" -m training.scripts.run_v3p_k9_ray_target_audit \
  --subset both \
  --max-graphs-per-case 3 \
  --max-wall-points 4000 \
  --stability-mode split_half \
  --k-full 16 \
  --k-sub 5 \
  --date-suffix "$DATE_TAG" \
  --verbose

echo ">>> K9 ray-target audit done $(date)"
