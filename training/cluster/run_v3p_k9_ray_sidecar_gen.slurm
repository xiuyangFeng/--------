#!/bin/bash
#SBATCH --partition=CPU
#SBATCH -w node03
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=k9_sidecar_gen
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=12:00:00

# 前沿方向 §15.2 · K9 剖面射线目标 sidecar 生成（Gate-0 Go 5935_v2 后落地 · 全量 73 case 全部图）
set -eo pipefail

module purge
PYTHON_BIN="${PYTHON_BIN:-/public/newhome/cy/.conda/envs/GNN/bin/python}"
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"
mkdir -p logs outputs/field/f0_decision

DATE_TAG="${DATE_TAG:-$(date +%Y%m%d)}"

echo "=== K9 ray-target sidecar gen $(date) ==="
"$PYTHON_BIN" -m training.scripts.run_v3p_k9_ray_sidecar_gen \
  --k-full 16 \
  --workers 8 \
  --date-suffix "$DATE_TAG"

echo ">>> K9 sidecar gen done $(date)"
