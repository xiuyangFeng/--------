#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --job-name=wss_err_only
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=04:00:00

# ============================================================================
# 仅补跑 plot_error_analysis --wss（假定各 run 下已有 predictions_test/manifest.json）
# 用于主 predict 阵列未传 --wss 或其它需重跑壁面 WSS 图件的场合。
#
# 用法（仓库根）:
#   MANIFEST_LIST_FILE=training/cluster/manifest_list_v2p_wssp05_06_predict.tsv \\
#     sbatch --array=0-5%4 training/cluster/run_manifest_plot_error_analysis_wss_array.slurm
#
# 可选串联主作业:
#   sbatch --dependency=afterok:<PRED_JOBID> \\
#     --array=0-5%4 MANIFEST_LIST_FILE=... \\
#     training/cluster/run_manifest_plot_error_analysis_wss_array.slurm
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
TRAINING_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_DIR="$(dirname "$TRAINING_DIR")"
cd "$PROJECT_DIR"

mkdir -p "$SCRIPT_DIR/logs"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export NUMEXPR_MAX_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

MANIFEST_LIST_FILE="${MANIFEST_LIST_FILE:-$SCRIPT_DIR/manifest_list_wss_multitask_predict.tsv}"
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
MAN="$RUN_DIR/predictions_test/manifest.json"

echo ">>> plot_error_analysis --wss only"
echo "RUN_DIR=$RUN_DIR"

if [ ! -f "$MAN" ]; then
    echo "错误: 不存在 $MAN ，请先在该 run 上完成 predict_field"
    exit 1
fi

"$PYTHON_BIN" -m training.scripts.plot_error_analysis --manifest "$MAN" --wss

echo "完成 task $SLURM_ARRAY_TASK_ID: $(date)"
