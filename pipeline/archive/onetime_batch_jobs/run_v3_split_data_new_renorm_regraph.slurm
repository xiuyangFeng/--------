#!/bin/bash
#SBATCH --partition=CPU
#SBATCH -w node03
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --job-name=v3_renorm_regraph
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=24:00:00

# ============================================================================
# V3 · split_data_new_v3：train-only 归一化 + 全量转图
#
# 须在仓库根目录提交（使 logs/ 与相对路径一致）：
#   cd /path/to/GNN && sbatch pipeline/archive/onetime_batch_jobs/run_v3_split_data_new_renorm_regraph.slurm
#
# 可选覆盖：
#   PIPELINE_ENV=GNN DATA_ROOT=data_new TRAIN_SPLIT=training/splits/split_data_new_v3.json sbatch ...
# ============================================================================

set -euo pipefail

export PYTHONUNBUFFERED=1

PIPELINE_ENV="${PIPELINE_ENV:-GNN}"
DATA_ROOT="${DATA_ROOT:-data_new}"
TRAIN_SPLIT="${TRAIN_SPLIT:-training/splits/split_data_new_v3.json}"
# 与 candidate_pool / split 三域一致；勿省略（config 默认仅启用 AG）
V3_SOURCES=(AAA/ruputer AAA/unruputer AG/fast AG/slow ILO)

module purge || true

if [ -f "$HOME/.bashrc" ]; then
  source "$HOME/.bashrc"
fi
if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
elif [ -f "/public/newapps/anaconda3/etc/profile.d/conda.sh" ]; then
  source "/public/newapps/anaconda3/etc/profile.d/conda.sh"
fi
conda activate "${PIPELINE_ENV}"

# 计算节点上 Slurm 会把脚本拷到 spool，$0 不是仓库内路径；必须用提交目录定位工程根。
if [ -z "${SLURM_SUBMIT_DIR:-}" ]; then
  echo "ERROR: 未设置 SLURM_SUBMIT_DIR。请在仓库根目录执行: sbatch pipeline/archive/onetime_batch_jobs/run_v3_split_data_new_renorm_regraph.slurm"
  exit 1
fi
PROJECT_DIR="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
cd "${PROJECT_DIR}"
mkdir -p "${PROJECT_DIR}/logs"

TRAIN_SPLIT_ABS="${PROJECT_DIR}/${TRAIN_SPLIT}"
if [ ! -f "${TRAIN_SPLIT_ABS}" ]; then
  echo "ERROR: TRAIN_SPLIT 不存在: ${TRAIN_SPLIT_ABS}"
  exit 1
fi

echo "=============================================="
echo "V3 split_data_new_v3 · 归一化 + 转图"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Start: $(date -Is)"
echo "PROJECT_DIR: ${PROJECT_DIR}"
echo "DATA_ROOT: ${DATA_ROOT}"
echo "TRAIN_SPLIT: ${TRAIN_SPLIT}"
echo "SOURCES: ${V3_SOURCES[*]}"
echo "PIPELINE_ENV: ${PIPELINE_ENV}"
echo "Python: $(which python)"
echo "=============================================="

echo ""
echo ">>> Step 4/5: normalize (train-only via --train-split)"
python -m pipeline.normalize \
  --data-root "${DATA_ROOT}" \
  --train-split "${TRAIN_SPLIT_ABS}" \
  --sources "${V3_SOURCES[@]}"

echo ""
echo ">>> Step 5/5: convert_to_graph (same sources)"
python -m pipeline.convert_to_graph \
  --data-root "${DATA_ROOT}" \
  --sources "${V3_SOURCES[@]}"

echo ""
echo "=============================================="
echo "Done: $(date -Is)"
echo "Exit: $?"
echo "=============================================="
