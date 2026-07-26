#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --job-name=d2_c125_pv
#SBATCH --output=training_wss_min/cluster/logs/%x_%j.out
#SBATCH --error=training_wss_min/cluster/logs/%x_%j.err
#SBATCH --time=02:00:00

set -euo pipefail

PROJECT=/public/newhome/cy/Digital_twin/GNN
PY=/public/newhome/cy/.conda/envs/GNN/bin/python
RUN="$PROJECT/training_wss_min/runs/pointnetpp_sa1_scale_single_seed/outputs/d2_rand5000_c125_k64"
SPLIT="$PROJECT/training/splits/split_AG_AAA_wss_min_v4_stratified_seed1234.json"
OUT="$RUN/postview/ckpt_best/test"
# 归一化空间 R² 中位数病例（test27 · ckpt_best）
CASE="AG/fast/LI_ZHI_LIN"
cd "$PROJECT"

mkdir -p "$OUT" "$RUN/postview"
echo "status=start case=$CASE run=$RUN output=$OUT"
"$PY" -m training_wss_min.tools.export_wss_postview \
  --run-dir "$RUN" \
  --cases "$CASE" \
  --allow-test \
  --checkpoint best \
  --output-dir "$OUT" \
  --device cuda \
  --resume
echo "status=completed case=$CASE output=$OUT"
ls -la "$OUT"/AG__fast__LI_ZHI_LIN__peak_wss/plots/
ls -la "$OUT"/AG__fast__LI_ZHI_LIN__peak_wss/