#!/bin/bash
#SBATCH --partition=CPU
#SBATCH -w node03
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=v3p_p1_v4
#SBATCH --time=01:00:00

# V3P Phase1 v4：Batch-1 采购候选并入 + phys neighbor 重评
set -eo pipefail

module purge
PYTHON_BIN="${PYTHON_BIN:-/public/newhome/cy/.conda/envs/GNN/bin/python}"
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"
mkdir -p logs outputs/field/f0_decision training/splits

DATE_TAG="${DATE_TAG:-20260701}"
BATCH_FILE="${BATCH_FILE:-outputs/field/f0_decision/v3p_procurement_batch1_${DATE_TAG}.txt}"
I6_RUN="outputs/field/field_v3_pointnext_i6diag_localpool_main01_geom_pw_asymw_a_wall13000_near2000_split_AG_v1_seed1_20260619_174001"
J4_RUN="outputs/field/field_v3_pointnext_j4v3dlite_localpool_main01_geom_pw_asymw_a_wall13000_near2000_split_AG_v3lite_v1_seed1_20260630_131820"

echo "=== Phase1 v4 · Batch-1 merge $(date) ==="
echo "Batch file: $BATCH_FILE"

"$PYTHON_BIN" -m training.scripts.run_v3p_phase1_active_data \
  --reference-run "$I6_RUN" \
  --compare-run "$J4_RUN" \
  --procurement-batch "$BATCH_FILE" \
  --selection-version v4 \
  --use-phys-neighbor \
  --phys-neighbor-threshold 8.0 \
  --date "$DATE_TAG" \
  --top-n 15 \
  --top-n-draft 6

echo ">>> Phase1 v4 done $(date)"
echo "产物:"
echo "  outputs/field/f0_decision/v3p_active_selection_v4_${DATE_TAG}.json"
echo "  training/splits/split_AG_active_v2.json"
echo "  docs/01-任务/任务A/03-V3路线/01-执行与待办/平台期-G5与主动选数/主动选数清单.md"
