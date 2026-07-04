#!/bin/bash
#SBATCH --partition=CPU
#SBATCH -w node03
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --job-name=v3p_proc_b2
#SBATCH --time=01:00:00

# V3P 257 池采购 Batch-2：跳过 Batch-1 后排名 + QA 深审
set -euo pipefail

PIPELINE_ENV=${PIPELINE_ENV:-GNN}
DATE_TAG=${DATE_TAG:-$(date +%Y%m%d)}
BATCH1_FILE=${BATCH1_FILE:-outputs/field/f0_decision/v3p_procurement_batch1_${DATE_TAG}.txt}

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
echo "V3P procurement Batch-2 · CPU rank + QA"
echo "Job: $SLURM_JOB_ID · Node: $SLURM_NODELIST · $(date)"
echo "Exclude Batch-1: $BATCH1_FILE"
echo "=============================================="

python -m training.scripts.run_v3p_pipeline_procurement_rank \
  --date "$DATE_TAG" \
  --batch-label batch2 \
  --batch-size 6 \
  --rank-skip 6 \
  --exclude AG/slow/ZHAO_XIU_XUAN \
  --exclude-from-file "$BATCH1_FILE"

echo ""
echo "Batch-2 case list:"
cat "outputs/field/f0_decision/v3p_procurement_batch2_${DATE_TAG}.txt"

echo ""
echo "Done $(date) exit=0"
