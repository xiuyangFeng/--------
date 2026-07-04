#!/bin/bash
#SBATCH --partition=CPU
#SBATCH -w node03
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --job-name=v3p_proc_b1
#SBATCH --time=01:00:00

# V3P 257 池采购 Batch-1：failure_cluster proxy 排名 + QA 深审
# 257 池 graphs 已齐 → 不提交 blind pipeline；仅 CPU 排名与文档产物
# 不开 J5 GPU

set -euo pipefail

PIPELINE_ENV=${PIPELINE_ENV:-GNN}
DATE_TAG=${DATE_TAG:-$(date +%Y%m%d)}

module purge
if [ -f "$HOME/.bashrc" ]; then source "$HOME/.bashrc"; fi
if [ -f "/public/newapps/anaconda3/etc/profile.d/conda.sh" ]; then
  source "/public/newapps/anaconda3/etc/profile.d/conda.sh"
fi
conda activate "$PIPELINE_ENV"

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "$REPO_ROOT"

echo "=============================================="
echo "V3P procurement Batch-1 · CPU rank + QA"
echo "Job: $SLURM_JOB_ID · Node: $SLURM_NODELIST · $(date)"
echo "REPO: $REPO_ROOT"
echo "=============================================="

python -m training.scripts.run_v3p_pipeline_procurement_rank \
  --date "$DATE_TAG" \
  --batch-size 6 \
  --exclude AG/slow/ZHAO_XIU_XUAN

echo ""
echo "Batch-1 case list:"
cat "outputs/field/f0_decision/v3p_procurement_batch1_${DATE_TAG}.txt"

echo ""
echo "Done $(date) exit=0"
