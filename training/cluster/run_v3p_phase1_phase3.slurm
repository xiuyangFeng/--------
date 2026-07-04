#!/bin/bash
#SBATCH --partition=CPU
#SBATCH -w node03
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=v3p_p1_p3
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=6:00:00

# V3P Phase 1（TODO-1 主动选数）+ Phase 3（L3 子集 eval）
set -eo pipefail

module purge
PYTHON_BIN="${PYTHON_BIN:-/public/newhome/cy/.conda/envs/GNN/bin/python}"

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"
mkdir -p logs outputs/field/f0_decision training/splits

export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4

I6_RUN="outputs/field/field_v3_pointnext_i6diag_localpool_main01_geom_pw_asymw_a_wall13000_near2000_split_AG_v1_seed1_20260619_174001"
J4_RUN="outputs/field/field_v3_pointnext_j4v3dlite_localpool_main01_geom_pw_asymw_a_wall13000_near2000_split_AG_v3lite_v1_seed1_20260630_131820"
DATE_TAG="${DATE_TAG:-$(date +%Y%m%d)}"

echo "=== [1/2] Phase 3 · L3 subset eval $(date) ==="
"$PYTHON_BIN" -m training.scripts.run_v3p_l3_subset_eval \
  --run-dir "$I6_RUN" \
  --run-dir "$J4_RUN" \
  --label I6-diag \
  --label J4 \
  --baseline-label I6-diag \
  --date "$DATE_TAG"

echo "=== [2/2] Phase 1 · active selection v2 $(date) ==="
"$PYTHON_BIN" -m training.scripts.run_v3p_phase1_active_data \
  --reference-run "$I6_RUN" \
  --compare-run "$J4_RUN" \
  --date "$DATE_TAG" \
  --top-n 15 \
  --top-n-draft 10

echo ">>> Phase1 + Phase3 done $(date)"
echo "产物:"
echo "  outputs/field/f0_decision/v3p_l3_subset_eval_${DATE_TAG}.json"
echo "  outputs/field/f0_decision/v3p_error_profile_${DATE_TAG}.json"
echo "  outputs/field/f0_decision/v3p_active_selection_v2_${DATE_TAG}.json"
echo "  training/splits/split_AG_active_v1.json"
echo "  docs/01-任务/任务A/03-V3路线/01-执行与待办/平台期-G5与主动选数/_history/主动选数_v2_${DATE_TAG}.md"
