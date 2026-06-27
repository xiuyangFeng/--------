#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --job-name=v3p_i6d_valpred
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=4:00:00

# I6-diag best_wss · val 集预测导出（供 M-E O11 校准 oracle）
#
# 用法: sbatch training/cluster/run_v3_me_i6diag_val_predict.slurm

set -eo pipefail

module purge
TRAINING_ENV=${TRAINING_ENV:-GNN}
PYTHON_BIN="${PYTHON_BIN:-/public/newhome/cy/.conda/envs/GNN/bin/python}"

if [ -f "$HOME/.bashrc" ]; then source "$HOME/.bashrc"; fi
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$TRAINING_ENV" 2>/dev/null || true
fi
if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
fi

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"
mkdir -p logs

RUN_DIR="${RUN_DIR:-outputs/field/field_v3_pointnext_i6diag_localpool_main01_geom_pw_asymw_a_wall13000_near2000_split_AG_v1_seed1_20260619_174001}"
CFG="$PROJECT_DIR/$RUN_DIR/config.snapshot.json"
CKPT="$PROJECT_DIR/$RUN_DIR/best_wss_model.pt"
OUT="$PROJECT_DIR/$RUN_DIR/predictions_val_best_wss"

echo "=== I6-diag val predict $(date) ==="
echo "Job: $SLURM_JOB_ID · Node: $SLURM_NODELIST"
echo "Run: $RUN_DIR"
echo "Out: $OUT"

if [ ! -f "$CKPT" ]; then
  echo "错误: 缺少 checkpoint $CKPT"
  exit 1
fi

"$PYTHON_BIN" -m training.scripts.predict_field \
  --config "$CFG" \
  --checkpoint "$CKPT" \
  --subset val \
  --output "$OUT"

echo ">>> val 预测完成 $(date) · $(wc -l < "$OUT/manifest.json" 2>/dev/null || echo 0) lines manifest"
