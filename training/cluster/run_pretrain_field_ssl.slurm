#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --job-name=ssl_pretrain
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=48:00:00

# G3 · 几何 SSL 预训练（Phase 1 过拟合 / Phase 2 全量 pretrain）
# 用法:
#   sbatch training/cluster/run_pretrain_field_ssl.slurm --cases pretrain --epochs 150 --seed 1 --exp-id V3P-G-58-SSL-Pretrain

set -euo pipefail

module purge
if [ -f "$HOME/.bashrc" ]; then source "$HOME/.bashrc"; fi
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
elif [ -f "/public/newapps/anaconda3/etc/profile.d/conda.sh" ]; then
    source "/public/newapps/anaconda3/etc/profile.d/conda.sh"
fi
conda activate "${TRAINING_ENV:-GNN}"

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"
mkdir -p logs

python -m training.scripts.pretrain_field_ssl \
    --device auto \
    "$@"

echo ">>> SSL pretrain 完成 $(date)"
