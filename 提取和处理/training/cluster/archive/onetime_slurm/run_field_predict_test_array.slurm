#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --job-name=field_pred_test
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=8:00:00
# 清单行数 N → SBATCH --array=0-$((N-1))%并发度
#SBATCH --array=0-4%2

# ============================================================================
# 测试集 predict_field 导出（不写任务 A 图件；适合 Probe / 仅需 .pt + manifest）
#
# 清单：每行一个 run 目录（相对仓库根），需含 config.snapshot.json 与 best_model.pt
#
# 提交（仓库根目录）:
#   MANIFEST_LIST_FILE=training/cluster/manifest_list_v3_probe_predict.tsv \
#     sbatch training/cluster/run_field_predict_test_array.slurm
# 自定义并发:  sbatch --array=0-4%3 training/cluster/run_field_predict_test_array.slurm
# ============================================================================

set -euo pipefail

module purge

TRAINING_ENV=${TRAINING_ENV:-GNN}
TRAINING_PYTHON=${TRAINING_PYTHON:-""}

if [ -f "$HOME/.bashrc" ]; then
    source "$HOME/.bashrc"
fi
if command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
    source "$CONDA_BASE/etc/profile.d/conda.sh"
elif [ -f "/public/newapps/anaconda3/etc/profile.d/conda.sh" ]; then
    source "/public/newapps/anaconda3/etc/profile.d/conda.sh"
fi
if [ -n "$TRAINING_PYTHON" ]; then
    PYTHON_BIN="$TRAINING_PYTHON"
else
    conda activate "$TRAINING_ENV"
    PYTHON_BIN="$(command -v python)"
fi

SCRIPT_PATH="${SLURM_JOB_SCRIPT:-}"
if [ -z "$SCRIPT_PATH" ] && command -v scontrol >/dev/null 2>&1; then
    SCRIPT_PATH="$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null | tr ' ' '\n' | awk -F= '/^Command=/{print $2; exit}')"
fi
if [ -z "$SCRIPT_PATH" ]; then
    SCRIPT_PATH="$0"
fi
case "$SCRIPT_PATH" in
    /*) ;;
    *) SCRIPT_PATH="$SLURM_SUBMIT_DIR/$SCRIPT_PATH" ;;
esac

SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
ARCHIVE_DIR="$(dirname "$SCRIPT_DIR")"
CLUSTER_DIR="$(dirname "$ARCHIVE_DIR")"
TRAINING_DIR="$(dirname "$CLUSTER_DIR")"
PROJECT_DIR="$(dirname "$TRAINING_DIR")"
cd "$PROJECT_DIR"

mkdir -p logs

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export NUMEXPR_MAX_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

MANIFEST_LIST_FILE="${MANIFEST_LIST_FILE:-$ARCHIVE_DIR/manifests/manifest_list_v3_probe_predict.tsv}"
if [ ! -f "$MANIFEST_LIST_FILE" ]; then
    echo "错误: 清单不存在: $MANIFEST_LIST_FILE"
    exit 1
fi

RUN_DIR_REL=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST_LIST_FILE" | tr -d '\r')
if [ -z "$RUN_DIR_REL" ]; then
    echo "错误: 数组任务 $SLURM_ARRAY_TASK_ID 在清单中无对应行"
    exit 1
fi

RUN_DIR="$PROJECT_DIR/$RUN_DIR_REL"
CFG="$RUN_DIR/config.snapshot.json"
CKPT="$RUN_DIR/best_model.pt"

echo "=============================================="
echo "predict_field (subset=test)"
echo "=============================================="
echo "Array: $SLURM_ARRAY_JOB_ID  task: $SLURM_ARRAY_TASK_ID"
echo "节点: ${SLURM_NODELIST:-$(hostname)}"
echo "开始: $(date)"
echo "RUN_DIR: $RUN_DIR"
echo "=============================================="

if [ ! -f "$CFG" ] || [ ! -f "$CKPT" ]; then
    echo "错误: 缺少 config 或 checkpoint"
    exit 1
fi

if [ -d "$RUN_DIR/predictions_test" ]; then
    echo ">>> 删除旧 predictions_test"
    rm -rf "$RUN_DIR/predictions_test"
fi

"$PYTHON_BIN" -m training.scripts.predict_field \
    --config "$CFG" \
    --checkpoint "$CKPT" \
    --subset test

MAN="$RUN_DIR/predictions_test/manifest.json"
if [ ! -f "$MAN" ]; then
    echo "错误: 未生成 manifest: $MAN"
    exit 1
fi

export _PRED_MANIFEST="$MAN"
"$PYTHON_BIN" -c 'import json, pathlib, os; m=pathlib.Path(os.environ["_PRED_MANIFEST"]); d=json.load(m.open()); n=len(list(m.parent.glob("*.pt"))); assert d["num_predictions"]==n, (d["num_predictions"], n); print("num_predictions", n)'
unset _PRED_MANIFEST

echo ">>> 完成 task $SLURM_ARRAY_TASK_ID $(date)"
