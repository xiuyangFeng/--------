#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --job-name=wss_mt_pred_fig
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=12:00:00
# 使用前显式传入数组范围；默认覆盖 manifest全部行（当前为 15 行 → 0-14）
#SBATCH --array=0-14%4

# ============================================================================
# WSS 多任务（baseline_wss_multitask / A-Opt-05-wss-multi 等）：
# 1) 删除各 run 下既有 predictions_test（清掉不完整导出）
# 2) predict_field（config.snapshot.json + best_model.pt）
# 3) 单 run 任务 A 图件：A3 散点、A4 箱线、误差分析、Fig A5 分区柱（interior 默认）
#
# 清单：training/cluster/manifest_list_wss_multitask_predict.tsv（每行一个 run 目录，相对仓库根）
#
# 提交（在仓库根目录）:
#   cd /path/to/GNN && sbatch training/cluster/run_wss_multitask_predict_figs_array.slurm
#   MANIFEST_LIST_FILE=training/cluster/manifest_list_wss_multitask_predict.tsv sbatch --array=0-14%4 training/cluster/run_wss_multitask_predict_figs_array.slurm
#
# 多模型汇总图（全 outputs/field）未在本作业中重跑；仅需时可另交 run_taskA_interior_figs.slurm。
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
mkdir -p logs

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
CFG="$RUN_DIR/config.snapshot.json"
CKPT="$RUN_DIR/best_model.pt"

echo "=============================================="
echo "WSS multitask predict + figs (array)"
echo "=============================================="
echo "Array Job: $SLURM_ARRAY_JOB_ID  task: $SLURM_ARRAY_TASK_ID"
echo "节点: ${SLURM_NODELIST:-$(hostname)}"
echo "开始: $(date)"
echo "RUN_DIR: $RUN_DIR"
echo "Python: $PYTHON_BIN"
echo "=============================================="

if [ ! -f "$CFG" ] || [ ! -f "$CKPT" ]; then
    echo "错误: 缺少 config或 checkpoint"
    echo "  cfg=$CFG"
    echo "  ckpt=$CKPT"
    exit 1
fi

if [ -d "$RUN_DIR/predictions_test" ]; then
    echo ">>> 删除旧 predictions_test: $RUN_DIR/predictions_test"
    rm -rf "$RUN_DIR/predictions_test"
fi

echo ">>> predict_field"
"$PYTHON_BIN" -m training.scripts.predict_field \
    --config "$CFG" \
    --checkpoint "$CKPT" \
    --subset test

MAN="$RUN_DIR/predictions_test/manifest.json"
if [ ! -f "$MAN" ]; then
    echo "错误: 未生成 manifest: $MAN"
    exit 1
fi

echo ">>> 校验 .pt 数量与 manifest 一致"
export _WSS_MANIFEST="$MAN"
"$PYTHON_BIN" -c 'import json, pathlib, os; m=pathlib.Path(os.environ["_WSS_MANIFEST"]); d=json.load(m.open()); n=len(list(m.parent.glob("*.pt"))); assert d["num_predictions"]==n, (d["num_predictions"], n); print("num_predictions", n)'
unset _WSS_MANIFEST

echo ">>> plot_taskA_scatter"
"$PYTHON_BIN" -m training.scripts.plot_taskA_scatter --manifest "$MAN"

echo ">>> plot_taskA_per_case_boxplot"
"$PYTHON_BIN" -m training.scripts.plot_taskA_per_case_boxplot --manifest "$MAN"

echo ">>> plot_error_analysis (含壁面 WSS 子目录，与 WSSP/多任务口径一致)"
"$PYTHON_BIN" -m training.scripts.plot_error_analysis --manifest "$MAN" --wss

echo ">>> plot_taskA_regional_bar (Fig A5)"
"$PYTHON_BIN" -m training.scripts.plot_taskA_regional_bar --manifest "$MAN"

echo "=============================================="
echo "完成 task $SLURM_ARRAY_TASK_ID: $(date)"
echo "=============================================="
