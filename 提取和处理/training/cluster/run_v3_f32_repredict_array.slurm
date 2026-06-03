#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --job-name=v3_f32_pred
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=8:00:00
#SBATCH --array=0-2%3

# TODO-32：AsymW-a 三 seed best_wss_model test 预测导出（并行 3 GPU）
#
# 提交（必须在仓库根目录）:
#   sbatch --array=0-2%3 training/cluster/run_v3_f32_repredict_array.slurm
# 完成后:
#   python -m training.scripts.run_v3_f32_ensemble

set -eo pipefail

module purge

TRAINING_ENV=${TRAINING_ENV:-GNN}
if [ -f "$HOME/.bashrc" ]; then source "$HOME/.bashrc"; fi
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
elif [ -f "/public/newapps/anaconda3/etc/profile.d/conda.sh" ]; then
    source "/public/newapps/anaconda3/etc/profile.d/conda.sh"
fi
conda activate "$TRAINING_ENV"
PYTHON_BIN="$(command -v python)"

# 用提交目录定位仓库根，避免 SLURM 将脚本复制到 spool 后路径错乱
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
SCRIPT_DIR="$PROJECT_DIR/training/cluster"
cd "$PROJECT_DIR"
mkdir -p "$PROJECT_DIR/logs"

MANIFEST_LIST_FILE="${MANIFEST_LIST_FILE:-$SCRIPT_DIR/manifest_list_asymw_repredict_best_wss.tsv}"
RUN_DIR_REL=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST_LIST_FILE" | tr -d '\r')
RUN_DIR="$PROJECT_DIR/$RUN_DIR_REL"
CFG="$RUN_DIR/config.snapshot.json"
CKPT="$RUN_DIR/best_wss_model.pt"
OUT="$RUN_DIR/predictions_test_best_wss"

echo "=============================================="
echo "TODO-32 predict_field (best_wss / test)"
echo "Array: $SLURM_ARRAY_JOB_ID task: $SLURM_ARRAY_TASK_ID"
echo "节点: ${SLURM_NODELIST:-$(hostname)}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "RUN_DIR: $RUN_DIR"
echo "开始: $(date)"
echo "=============================================="

if [ ! -f "$CFG" ] || [ ! -f "$CKPT" ]; then
    echo "错误: 缺少 config 或 best_wss_model.pt"
    exit 1
fi

if [ -d "$OUT" ]; then
    echo ">>> 清理旧 $OUT（保留目录结构由 predict_field 重写 manifest）"
    rm -rf "$OUT"
fi

"$PYTHON_BIN" -m training.scripts.predict_field \
    --config "$CFG" \
    --checkpoint "$CKPT" \
    --subset test \
    --output "$OUT"

MAN="$OUT/manifest.json"
N_PT=$(find "$OUT" -maxdepth 1 -name '*.pt' | wc -l)
echo "manifest: $MAN  pt_files: $N_PT"
test -f "$MAN"
echo ">>> 完成 task $SLURM_ARRAY_TASK_ID $(date)"
