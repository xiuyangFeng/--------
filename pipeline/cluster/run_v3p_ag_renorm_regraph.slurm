#!/bin/bash
#SBATCH --partition=CPU
#SBATCH -w node03
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --job-name=v3p_ag_renorm
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=12:00:00

# V3P AG-only · post-denylist train-only 重算 normalization + 重建 graphs
#
# 用法（仓库根）:
#   sbatch pipeline/cluster/run_v3p_ag_renorm_regraph.slurm

set -euo pipefail

PIPELINE_ENV=${PIPELINE_ENV:-GNN}
DATA_ROOT=${DATA_ROOT:-data_new/AG}
TRAIN_SPLIT=${TRAIN_SPLIT:-training/splits/split_AG_v1.json}
V3P_SOURCES=(fast slow)

module purge
if [ -f "$HOME/.bashrc" ]; then source "$HOME/.bashrc"; fi
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
elif [ -f "/public/newapps/anaconda3/etc/profile.d/conda.sh" ]; then
    source "/public/newapps/anaconda3/etc/profile.d/conda.sh"
fi
conda activate "$PIPELINE_ENV"

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"
mkdir -p logs

TRAIN_SPLIT_ABS="${PROJECT_DIR}/${TRAIN_SPLIT}"
echo ">>> V3P AG renorm+regraph Job=$SLURM_JOB_ID split=$TRAIN_SPLIT_ABS"

python -m pipeline.verify_train_stats_dirs \
    --data-root "${DATA_ROOT}" \
    --train-split "${TRAIN_SPLIT_ABS}" \
    --sources "${V3P_SOURCES[@]}"

python -m pipeline.normalize \
    --data-root "${DATA_ROOT}" \
    --train-split "${TRAIN_SPLIT_ABS}" \
    --sources "${V3P_SOURCES[@]}"

python -m pipeline.convert_to_graph \
    --data-root "${DATA_ROOT}" \
    --sources "${V3P_SOURCES[@]}"

echo ">>> 完成 $(date)"
