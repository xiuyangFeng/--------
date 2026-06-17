#!/usr/bin/bash
# V3 TODO-1：AAA+ILO 全流程候选池终审（不写训练、不写 WSS 对比脚本）
#
#   cd <repo-root>/pipeline/archive/onetime_batch_jobs
#   sbatch run_v3_prepool_audit.slurm
#
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --job-name=v3_prepool_audit
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=04:00:00
# 集群规范：需要 CPU 资源时固定在 node03
#SBATCH -w node03

set -euo pipefail

export PYTHONUNBUFFERED=1
PIPELINE_ENV=${PIPELINE_ENV:-GNN}

module purge || true
if [ -f "$HOME/.bashrc" ]; then
  source "$HOME/.bashrc"
fi
CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${PIPELINE_ENV}"

SCRIPT_PATH="${SLURM_JOB_SCRIPT:-}"
if [ -z "$SCRIPT_PATH" ] && command -v scontrol >/dev/null 2>&1; then
  SCRIPT_PATH="$(scontrol show job "$SLURM_JOB_ID" | tr ' ' '\n' | awk -F= '/^Command=/{print $2; exit}')"
fi
if [ -z "$SCRIPT_PATH" ]; then
  SCRIPT_PATH="$0"
fi
case "$SCRIPT_PATH" in
  /*) ;;
  *) SCRIPT_PATH="$SLURM_SUBMIT_DIR/$SCRIPT_PATH" ;;
esac
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
ARCHIVE_DIR="$(dirname "$SCRIPT_DIR")"
PIPELINE_DIR="$(dirname "$ARCHIVE_DIR")"
PROJECT_DIR="$(dirname "$PIPELINE_DIR")"
cd "${PROJECT_DIR}"

mkdir -p "${SCRIPT_DIR}/logs"

OUT_PARENT="${PROJECT_DIR}/outputs/field/diagnostics"
OUT_DIR="${OUT_PARENT}/v3_data_new_prepool_audit_${SLURM_JOB_ID}"

echo "=============================================="
echo "节点: $(hostname)"
echo "开始: $(date -Is)"
echo "输出目录: ${OUT_DIR}"
echo "=============================================="

python -m training.scripts.run_v3_aaa_ilo_prepool_audit \
  --data-root "${PROJECT_DIR}/data_new" \
  --out-dir "${OUT_DIR}" \
  --write-candidate-list "${PROJECT_DIR}/training/splits/cases_data_new_v3_candidate_pool.txt" \
  --max-timesteps 10

echo "完成: $(date -Is)"
