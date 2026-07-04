#!/bin/bash
#SBATCH --partition=CPU
#SBATCH -w node03
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=v3p_p1_v5
#SBATCH --time=01:00:00

# V3P Phase1 v5：Batch-1 + Batch-2 并入 + phys neighbor 重评
set -eo pipefail

module purge
PYTHON_BIN="${PYTHON_BIN:-/public/newhome/cy/.conda/envs/GNN/bin/python}"
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"
mkdir -p logs outputs/field/f0_decision training/splits

DATE_TAG="${DATE_TAG:-20260701}"
BATCH1="${BATCH1:-outputs/field/f0_decision/v3p_procurement_batch1_${DATE_TAG}.txt}"
BATCH2="${BATCH2:-outputs/field/f0_decision/v3p_procurement_batch2_${DATE_TAG}.txt}"
I6_RUN="outputs/field/field_v3_pointnext_i6diag_localpool_main01_geom_pw_asymw_a_wall13000_near2000_split_AG_v1_seed1_20260619_174001"
J5_RUN="outputs/field/field_v3_pointnext_j5activedata_localpool_main01_geom_pw_asymw_a_wall13000_near2000_split_AG_active_v2_j5_seed1_20260701_002549"

echo "=== Phase1 v5 · Batch-1+2 merge $(date) ==="
echo "Batch-1: $BATCH1"
echo "Batch-2: $BATCH2"

"$PYTHON_BIN" -m training.scripts.run_v3p_phase1_active_data \
  --reference-run "$I6_RUN" \
  --compare-run "$J5_RUN" \
  --procurement-batch "$BATCH1" \
  --procurement-batch "$BATCH2" \
  --selection-version v5 \
  --use-phys-neighbor \
  --phys-neighbor-threshold 8.0 \
  --date "$DATE_TAG" \
  --top-n 20 \
  --top-n-draft 12

echo ">>> Phase1 v5 done $(date)"
echo "产物:"
echo "  outputs/field/f0_decision/v3p_active_selection_v5_${DATE_TAG}.json"
echo "  training/splits/split_AG_active_v3.json"
echo "  docs/01-任务/任务A/03-V3路线/01-执行与待办/平台期-G5与主动选数/主动选数清单.md"
