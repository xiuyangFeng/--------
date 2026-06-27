#!/bin/bash
#SBATCH --partition=CPU
#SBATCH -w node03
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=v3p_me_followup
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=2:00:00

# V3P M-E oracle 补跑：O9 全81帧 · O3 POD系数 · O11 val校准
# 依赖 val 预测完成后运行（submit 脚本会加 afterok）
#
# 用法: sbatch training/cluster/run_v3_me_oracle_followup.slurm

set -eo pipefail

module purge
PYTHON_BIN="${PYTHON_BIN:-/public/newhome/cy/.conda/envs/GNN/bin/python}"

if [ -f "$HOME/.bashrc" ]; then source "$HOME/.bashrc"; fi
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${TRAINING_ENV:-GNN}" 2>/dev/null || true
fi
if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
fi

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"
mkdir -p logs outputs/field/f0_decision

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "=== V3P M-E oracle followup $(date) ==="
echo "Job: $SLURM_JOB_ID · Node: $SLURM_NODELIST"

"$PYTHON_BIN" -m training.scripts.run_v3_me_oracle_followup \
  --max-frames-per-case 81 \
  --max-cases-temporal 16 \
  --max-graphs-per-case 3 \
  --max-pred-items 2000 \
  "$@"

echo ">>> followup 完成 $(date)"
