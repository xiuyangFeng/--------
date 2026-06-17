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
DATA_ROOT=${DATA_ROOT:-data_new}
# V3D 三域默认 v3（train=179，已剔除 SU_KAI_LI + ILO/GUO_AI_JUN-0/after）；AG-only 可覆盖 split_AG_v1.json
TRAIN_SPLIT=${TRAIN_SPLIT:-"training/splits/split_data_new_v3.json"}
V3_SOURCES=(AAA/ruputer AAA/unruputer AG/fast AG/slow ILO)

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

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  PROJECT_DIR="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  SCRIPT_PATH="${SLURM_JOB_SCRIPT:-$0}"
  SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
  PROJECT_DIR="$(cd "$(dirname "$(dirname "$SCRIPT_DIR")")" && pwd)"
fi
cd "$PROJECT_DIR"

mkdir -p logs

echo "=============================================="
echo "重新归一化 + 重建图资产"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Time: $(date)"
echo "DATA_ROOT: $DATA_ROOT"
echo "Train split: $TRAIN_SPLIT"
echo "SOURCES: ${V3_SOURCES[*]}"
echo "Python: $(which python)"
echo "=============================================="

TRAIN_SPLIT_ABS="${PROJECT_DIR}/${TRAIN_SPLIT}"
if [ ! -f "${TRAIN_SPLIT_ABS}" ]; then
  echo "ERROR: TRAIN_SPLIT 不存在: ${TRAIN_SPLIT_ABS}"
  exit 1
fi

echo ""
echo ">>> Gate-0 校验（train-only 目录数）"
python -m pipeline.verify_train_stats_dirs \
  --data-root "${DATA_ROOT}" \
  --train-split "${TRAIN_SPLIT_ABS}" \
  --sources "${V3_SOURCES[@]}"

echo ""
echo ">>> 步骤 4/5: 重新归一化（train-only via --train-split）"
python -m pipeline.normalize \
  --data-root "${DATA_ROOT}" \
  --train-split "${TRAIN_SPLIT_ABS}" \
  --sources "${V3_SOURCES[@]}"

echo ""
echo ">>> 步骤 5/5: 重建图资产"
python -m pipeline.convert_to_graph \
  --data-root "${DATA_ROOT}" \
  --sources "${V3_SOURCES[@]}"

echo ""
echo "=============================================="
echo "完成时间: $(date)"
echo "退出码: $?"
echo "=============================================="
