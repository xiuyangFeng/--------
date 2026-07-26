#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --job-name=sa1_med_ovlp
#SBATCH --output=training_wss_min/cluster/logs/%x_%j.out
#SBATCH --error=training_wss_min/cluster/logs/%x_%j.err
#SBATCH --time=04:00:00

set -euo pipefail
PROJECT=/public/newhome/cy/Digital_twin/GNN
PY=/public/newhome/cy/.conda/envs/GNN/bin/python
cd "$PROJECT"
mkdir -p training_wss_min/cluster/logs training_wss_min/preflight

CFG_DIR=training_wss_min/configs/pointnetpp_sa1_scale_single_seed_20260721
REF_KNN8=training_wss_min/configs/pointnetpp_sa_grouping_single_seed_20260720/q1v_sa1_knn8_cover.json

CONFIGS=(
  "$REF_KNN8"
  "$CFG_DIR/bridge_rand5000_fixed500_k64.json"
  "$CFG_DIR/d2_rand5000_c250_k64.json"
  "$CFG_DIR/d2_rand5000_c125_k64.json"
  "$CFG_DIR/d2_rand5000_c250_k128.json"
  "$CFG_DIR/d2_rand5000_c125_k128.json"
  "$CFG_DIR/d3_rand10000_fixed500_k64.json"
  "$CFG_DIR/d3_rand10000_fixed500_k128.json"
  "$CFG_DIR/d3_rand10000_fixed500_k256.json"
  "$CFG_DIR/d1_allpts_fixed500_k64.json"
  "$CFG_DIR/d1_allpts_fixed500_k128.json"
  "$CFG_DIR/d1_allpts_fixed500_k256.json"
)

echo "computing nearest-pair median overlap for ${#CONFIGS[@]} configs"
"$PY" -u -m training_wss_min.tools.compute_sa1_scale_nearest_pair_median_overlap \
  --configs "${CONFIGS[@]}" \
  --output training_wss_min/preflight/pointnetpp_sa1_scale_nearest_pair_median_overlap.json \
  --max-cases-large 4
