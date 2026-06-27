#!/bin/bash
#SBATCH --partition=CPU
#SBATCH -w node03
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=v3p_me_oracle
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=6:00:00

# V3P M-E · O1–O12 零重训 oracle 波次（工程方案 §5+.x · CPU · node03）
# 用法: sbatch training/cluster/run_v3_me_oracle_wave.slurm
# 或:   bash training/cluster/submit_v3_me_oracle_wave.sh

set -eo pipefail

module purge
TRAINING_ENV=${TRAINING_ENV:-GNN}
PYTHON_BIN="${PYTHON_BIN:-/public/newhome/cy/.conda/envs/GNN/bin/python}"

if [ -f "$HOME/.bashrc" ]; then source "$HOME/.bashrc"; fi
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$TRAINING_ENV" 2>/dev/null || true
fi
if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
elif [ -x "$PYTHON_BIN" ]; then
    :
else
    echo "错误: 找不到 GNN python"
    exit 1
fi

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"
mkdir -p "$PROJECT_DIR/logs" "$PROJECT_DIR/outputs/field/f0_decision"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "=== V3P M-E oracle wave $(date) ==="
echo "Job: $SLURM_JOB_ID · Node: $SLURM_NODELIST · Python: $PYTHON_BIN"

"$PYTHON_BIN" -m training.scripts.run_v3_me_oracle_wave \
    --max-graphs-per-case 3 \
    --max-cases 24 \
    --max-wall-per-graph 800 \
    --max-pred-items 300 \
    --grid-s 64 \
    --n-sectors 8 \
    --verbose \
    "$@"

echo ">>> V3P M-E oracle wave 完成 $(date)"
