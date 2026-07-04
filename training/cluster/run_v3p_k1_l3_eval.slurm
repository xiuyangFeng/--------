#!/bin/bash
#SBATCH --partition=CPU
#SBATCH -w node03
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=k1_l3
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=8:00:00

# K1-lite (5886) 完训后 L3 子集误差分解（I6-diag vs K1）
# 两种 checkpoint 口径都评：best_wss（标准口径）+ best_model（5886 反常更优的 ckpt）
set -eo pipefail

module purge
PYTHON_BIN="${PYTHON_BIN:-/public/newhome/cy/.conda/envs/GNN/bin/python}"
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"
mkdir -p logs outputs/field/f0_decision

DATE_TAG="${DATE_TAG:-20260704}"
I6_RUN="outputs/field/field_v3_pointnext_i6diag_localpool_main01_geom_pw_asymw_a_wall13000_near2000_split_AG_v1_seed1_20260619_174001"
K1_RUN="${K1_RUN:-outputs/field/field_v3_pointnext_k1branchfeat_localpool_main01_geom_pw_asymw_a_wall13000_near2000_split_AG_v1_seed1_20260701_150937}"

if [[ ! -d "$K1_RUN" ]]; then
  echo "错误: 找不到 K1 run: $K1_RUN"
  exit 1
fi

echo "=== L3 eval I6 vs K1 (best_wss) $(date) ==="
"$PYTHON_BIN" -m training.scripts.run_v3p_l3_subset_eval \
  --run-dir "$I6_RUN" \
  --run-dir "$K1_RUN" \
  --label I6-diag \
  --label K1 \
  --baseline-label I6-diag \
  --checkpoint best_wss \
  --predict \
  --date "$DATE_TAG" \
  --output "outputs/field/f0_decision/v3p_l3_subset_eval_k1_bestwss_${DATE_TAG}.json"

echo "=== L3 eval I6 vs K1 (best_model) $(date) ==="
"$PYTHON_BIN" -m training.scripts.run_v3p_l3_subset_eval \
  --run-dir "$I6_RUN" \
  --run-dir "$K1_RUN" \
  --label I6-diag \
  --label K1 \
  --baseline-label I6-diag \
  --checkpoint best_model \
  --predict \
  --date "$DATE_TAG" \
  --output "outputs/field/f0_decision/v3p_l3_subset_eval_k1_bestmodel_${DATE_TAG}.json"

echo ">>> K1 L3 done $(date)"
