#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --job-name=j5_predict
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=4:00:00

# J5 test 集 best_wss 推理（L3 前置）
set -eo pipefail

module purge
PYTHON_BIN="${PYTHON_BIN:-/public/newhome/cy/.conda/envs/GNN/bin/python}"
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"
mkdir -p logs

J5_RUN="${J5_RUN:-}"
if [[ -z "$J5_RUN" ]]; then
  J5_RUN=$(ls -dt outputs/field/field_v3_pointnext_j5activedata_*_split_AG_active_v2_j5_seed1_* 2>/dev/null | head -1)
fi
if [[ -z "$J5_RUN" || ! -d "$J5_RUN" ]]; then
  echo "错误: 找不到 J5 run 目录，请设置 J5_RUN"
  exit 1
fi

MANIFEST="${J5_RUN}/predictions_test_best_wss/manifest.json"
if [[ -f "$MANIFEST" ]]; then
  echo ">>> J5 predictions 已存在: $J5_RUN $(date)"
  exit 0
fi

echo "=== J5 predict_field best_wss $(date) ==="
echo "run: $J5_RUN"
"$PYTHON_BIN" -m training.scripts.predict_field \
  --config "${J5_RUN}/config.snapshot.json" \
  --checkpoint "${J5_RUN}/best_wss_model.pt" \
  --subset test \
  --output "${J5_RUN}/predictions_test_best_wss"

echo ">>> J5 predict done $(date)"
