#!/bin/bash
#SBATCH --partition=CPU
#SBATCH -w node03
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=j5_l3
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=3:00:00

# J5 完训后 L3 子集 eval（I6-diag vs J5）
set -eo pipefail

module purge
PYTHON_BIN="${PYTHON_BIN:-/public/newhome/cy/.conda/envs/GNN/bin/python}"
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"
mkdir -p logs outputs/field/f0_decision

DATE_TAG="${DATE_TAG:-20260701}"
I6_RUN="outputs/field/field_v3_pointnext_i6diag_localpool_main01_geom_pw_asymw_a_wall13000_near2000_split_AG_v1_seed1_20260619_174001"
J5_RUN="${J5_RUN:-}"
if [[ -z "$J5_RUN" ]]; then
  J5_RUN=$(ls -dt outputs/field/field_v3_pointnext_j5activedata_*_split_AG_active_v2_j5_seed1_* 2>/dev/null | head -1)
fi
if [[ -z "$J5_RUN" || ! -d "$J5_RUN" ]]; then
  echo "错误: 找不到 J5 run"
  exit 1
fi

echo "=== L3 eval I6 vs J5 $(date) ==="
echo "J5 run: $J5_RUN"

"$PYTHON_BIN" -m training.scripts.run_v3p_l3_subset_eval \
  --run-dir "$I6_RUN" \
  --run-dir "$J5_RUN" \
  --label I6-diag \
  --label J5 \
  --baseline-label I6-diag \
  --predict \
  --date "$DATE_TAG" \
  --output "outputs/field/f0_decision/v3p_l3_subset_eval_j5_${DATE_TAG}.json"

echo ">>> L3 done $(date)"
