#!/bin/bash
#SBATCH --partition=CPU
#SBATCH -w node03
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=v3p_g4c_p0
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=4:00:00

# V3P G4-c Geodesic Patch Phase 0 可行性（CPU · node03）
# 用法: sbatch training/cluster/run_v3_g4c_patch_feasibility.slurm

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
fi

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"
mkdir -p "$PROJECT_DIR/logs" "$PROJECT_DIR/outputs/field/f0_decision"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "=== V3P G4-c patch feasibility $(date) ==="
echo "Job: $SLURM_JOB_ID · Node: $SLURM_NODELIST · Python: $PYTHON_BIN"

"$PYTHON_BIN" -m training.scripts.run_v3_g4c_patch_feasibility \
    --max-graphs-per-case 3 \
    --max-seeds-per-graph 40 \
    --verbose \
    "$@"

echo ">>> G4-c patch feasibility 完成 $(date)"
