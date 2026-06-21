#!/bin/bash
#SBATCH --partition=CPU
#SBATCH -w node03
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --job-name=local_v1_renorm
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=12:00:00

# V3P local_v1：local_wss → coord_normalize → normalize → convert_to_graph
# 用法（仓库根目录）:
#   DATA_ROOT=data_new/AG TRAIN_SPLIT=training/splits/split_AG_v1.json \
#     sbatch pipeline/cluster/run_local_v1_renorm_regraph.slurm

set -euo pipefail

PIPELINE_ENV=${PIPELINE_ENV:-GNN}
DATA_ROOT=${DATA_ROOT:-data_new/AG}
TRAIN_SPLIT=${TRAIN_SPLIT:-"training/splits/split_AG_v1.json"}
V3_SOURCES=(fast slow)
GRAPHS_SUBDIR=${GRAPHS_SUBDIR:-"processed/graphs_local_v1"}

module purge
if [ -f "$HOME/.bashrc" ]; then source "$HOME/.bashrc"; fi
if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
elif [ -f "/public/newapps/anaconda3/etc/profile.d/conda.sh" ]; then
  source "/public/newapps/anaconda3/etc/profile.d/conda.sh"
fi
conda activate "$PIPELINE_ENV"

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  PROJECT_DIR="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  SCRIPT_PATH="${SLURM_JOB_SCRIPT:-$0}"
  SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
  PROJECT_DIR="$(cd "$(dirname "$(dirname "$SCRIPT_DIR")")" && pwd)"
fi
cd "$PROJECT_DIR"
mkdir -p logs

TRAIN_SPLIT_ABS="${PROJECT_DIR}/${TRAIN_SPLIT}"
NORMAL_SOURCE=${NORMAL_SOURCE:-stl}
echo "DATA_ROOT=$DATA_ROOT TRAIN_SPLIT=$TRAIN_SPLIT GRAPHS_SUBDIR=$GRAPHS_SUBDIR NORMAL_SOURCE=$NORMAL_SOURCE"

echo ">>> 步骤 2.5: local_wss (STL 表面法向 + 投影, local_v1)"
python -m pipeline.local_wss \
  --data-root "${DATA_ROOT}" \
  --sources "${V3_SOURCES[@]}" \
  --normal-source "${NORMAL_SOURCE}"

echo ">>> 步骤 3: coord_normalize"
python -m pipeline.coord_normalize --data-root "${DATA_ROOT}" --sources "${V3_SOURCES[@]}"

echo ">>> 步骤 4: normalize (train-only)"
python -m pipeline.normalize \
  --data-root "${DATA_ROOT}" \
  --train-split "${TRAIN_SPLIT_ABS}" \
  --sources "${V3_SOURCES[@]}"

echo ">>> 步骤 5: convert_to_graph → ${GRAPHS_SUBDIR}"
python -m pipeline.convert_to_graph \
  --data-root "${DATA_ROOT}" \
  --output-subdir "${GRAPHS_SUBDIR}" \
  --sources "${V3_SOURCES[@]}"

echo "完成: $(date)"
