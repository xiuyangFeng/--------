#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --job-name=j4_predict
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=4:00:00

# J4 test 集 best_wss 推理（Phase 3 前置）
set -eo pipefail

module purge
PYTHON_BIN="${PYTHON_BIN:-/public/newhome/cy/.conda/envs/GNN/bin/python}"

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"
mkdir -p logs

J4_RUN="outputs/field/field_v3_pointnext_j4v3dlite_localpool_main01_geom_pw_asymw_a_wall13000_near2000_split_AG_v3lite_v1_seed1_20260630_131820"
MANIFEST="${J4_RUN}/predictions_test_best_wss/manifest.json"

if [[ -f "$MANIFEST" ]]; then
  echo ">>> J4 predictions 已存在，跳过 $(date)"
  exit 0
fi

echo "=== J4 predict_field best_wss $(date) ==="
"$PYTHON_BIN" -m training.scripts.predict_field \
  --config "${J4_RUN}/config.snapshot.json" \
  --checkpoint "${J4_RUN}/best_wss_model.pt" \
  --subset test \
  --output "${J4_RUN}/predictions_test_best_wss"

echo ">>> J4 predict done $(date)"
