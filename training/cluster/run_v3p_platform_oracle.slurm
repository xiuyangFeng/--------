#!/bin/bash
#SBATCH --partition=CPU
#SBATCH -w node03
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=v3p_plat_oracle
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=4:00:00

# V3P 平台期 Ideas A–G oracle（CPU · node03）
set -eo pipefail

module purge
PYTHON_BIN="${PYTHON_BIN:-/public/newhome/cy/.conda/envs/GNN/bin/python}"

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"
mkdir -p logs outputs/field/f0_decision

export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4

echo "=== V3P platform oracle bundle $(date) ==="
"$PYTHON_BIN" -m training.scripts.run_v3p_platform_oracle_bundle \
    --verbose \
    --max-graphs-per-case 3 \
    --max-wall-sample 2000

echo "=== summarize gate $(date) ==="
"$PYTHON_BIN" -m training.scripts.summarize_v3p_platform_oracle

echo ">>> platform oracle done $(date)"
