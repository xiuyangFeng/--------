#!/bin/bash
#SBATCH --partition=CPU
#SBATCH -w node03
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=v3p_q0_diag
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=4:00:00

# V3P 路径 I · Q0 诊断（0 GPU · node03 CPU）
#   ① I-oracle bundle（§15.3 + I1 θ 审计）
#   ② I7 ckpt 特征衰减探针
# 用法: sbatch training/cluster/run_v3_q0_diagnostics.slurm

set -eo pipefail

module purge
TRAINING_ENV=${TRAINING_ENV:-GNN}
if [ -f "$HOME/.bashrc" ]; then source "$HOME/.bashrc"; fi
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
elif [ -f "/public/newapps/anaconda3/etc/profile.d/conda.sh" ]; then
    source "/public/newapps/anaconda3/etc/profile.d/conda.sh"
fi
conda activate "$TRAINING_ENV"

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"
mkdir -p "$PROJECT_DIR/logs" "$PROJECT_DIR/outputs/field/f0_decision"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "=== V3P Q0 diagnostics $(date) ==="
echo "Job: $SLURM_JOB_ID · Node: $SLURM_NODELIST"

python -m training.scripts.run_v3_i_oracle_bundle \
    --max-graphs-per-case 3 \
    --grid-s 64 \
    --n-sectors 4 \
    --n-modes 20

python -m training.scripts.run_v3_i7_ckpt_probe

echo ">>> V3P Q0 diagnostics 完成 $(date)"
