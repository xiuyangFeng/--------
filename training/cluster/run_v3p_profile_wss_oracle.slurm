#!/bin/bash
#SBATCH --partition=CPU
#SBATCH -w node03
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=prof_wss_oracle
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=4:00:00

# 前沿方向 §3.2 · 剖面基 WSS Gate-0 oracle（全量 train+test · 0 重训）
set -eo pipefail

module purge
PYTHON_BIN="${PYTHON_BIN:-/public/newhome/cy/.conda/envs/GNN/bin/python}"
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"
mkdir -p logs outputs/field/f0_decision

DATE_TAG="${DATE_TAG:-20260704}"

echo "=== profile WSS oracle $(date) ==="
"$PYTHON_BIN" -m training.scripts.run_v3p_profile_wss_oracle \
  --subset both \
  --max-graphs-per-case 3 \
  --max-wall-points 1500 \
  --k-neighbors 8 \
  --date-suffix "$DATE_TAG" \
  --verbose

echo ">>> profile WSS oracle done $(date)"
