#!/bin/bash
#SBATCH --partition=CPU
#SBATCH -w node03
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --job-name=renorm_regraph
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=12:00:00

# ============================================================================
# 重新归一化 + 重建图资产
# 用途：split 变更后（如移除异常病例），重算归一化统计量并重建全部图
#
# 使用方法（在仓库根目录）:
#   sbatch pipeline/cluster/run_renorm_regraph.slurm
#   TRAIN_SPLIT=training/splits/split_AG_v1.json sbatch pipeline/cluster/run_renorm_regraph.slurm
# ============================================================================

set -euo pipefail

PIPELINE_ENV=${PIPELINE_ENV:-GNN}
TRAIN_SPLIT=${TRAIN_SPLIT:-"training/splits/split_AG_v1.json"}

module purge

if [ -f "$HOME/.bashrc" ]; then
    source "$HOME/.bashrc"
fi
if command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
    source "$CONDA_BASE/etc/profile.d/conda.sh"
elif [ -f "/public/newapps/anaconda3/etc/profile.d/conda.sh" ]; then
    source "/public/newapps/anaconda3/etc/profile.d/conda.sh"
fi
conda activate "$PIPELINE_ENV"

SCRIPT_PATH="${SLURM_JOB_SCRIPT:-$0}"
case "$SCRIPT_PATH" in
    /*) ;;
    *) SCRIPT_PATH="$SLURM_SUBMIT_DIR/$SCRIPT_PATH" ;;
esac
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
PIPELINE_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_DIR="$(dirname "$PIPELINE_DIR")"
cd "$PROJECT_DIR"

mkdir -p logs

echo "=============================================="
echo "重新归一化 + 重建图资产"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Time: $(date)"
echo "Train split: $TRAIN_SPLIT"
echo "Python: $(which python)"
echo "=============================================="

echo ""
echo ">>> 步骤 4/5: 重新归一化（使用 --train-split 排除异常病例）"
python -m pipeline.normalize --train-split "$TRAIN_SPLIT"

echo ""
echo ">>> 步骤 5/5: 重建图资产"
python -m pipeline.convert_to_graph

echo ""
echo "=============================================="
echo "完成时间: $(date)"
echo "退出码: $?"
echo "=============================================="
