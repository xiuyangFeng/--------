#!/bin/bash
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=field_regional_a5
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=02:00:00
# 集群规范：CPU 任务指定 node03（若排队过久可临时注释本行）
#SBATCH -w node03

# ============================================================================
# 任务 A：按 predictions_test/manifest.json 重算分区域指标（Fig A5）
# 依赖 GNN 环境（torch / PyG / matplotlib）
#
# 用法:
#   cd /path/to/GNN && sbatch training/cluster/run_regional_a5.slurm
#   TRAINING_ENV=GNN sbatch training/cluster/run_regional_a5.slurm
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
    SCRIPT_PATH="$(scontrol show job "$SLURM_JOB_ID" | tr ' ' '\n' | awk -F= '/^Command=/{print $2; exit}')"
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

echo "=============================================="
echo "Field A5 Regional — per-run + multimodel"
echo "=============================================="
echo "作业ID: ${SLURM_JOB_ID:-local}"
echo "节点: ${SLURM_NODELIST:-$(hostname)}"
echo "开始时间: $(date)"
echo "工作目录: $PROJECT_DIR"
echo "环境: ${TRAINING_ENV:-custom}"
echo "Python: $PYTHON_BIN"
echo "=============================================="

shopt -s nullglob
MANIFESTS=(outputs/field/*/predictions_test/manifest.json)
if [ ${#MANIFESTS[@]} -eq 0 ]; then
    echo "错误: 未找到 outputs/field/*/predictions_test/manifest.json"
    exit 1
fi

for m in "${MANIFESTS[@]}"; do
    echo ""
    echo ">>> plot_taskA_regional_bar --manifest $m"
    "$PYTHON_BIN" -m training.scripts.plot_taskA_regional_bar --manifest "$m"
done

echo ""
echo ">>> plot_taskA_multimodel_regional_bar (所有已在 outputs/field 下具备 regional_eval 的 exp_id)"
"$PYTHON_BIN" -m training.scripts.plot_taskA_multimodel_regional_bar \
    --runs-root outputs/field

echo ""
echo ">>> plot_taskA_multimodel_regional_bar (Transformer vs Transformer+Geom)"
"$PYTHON_BIN" -m training.scripts.plot_taskA_multimodel_regional_bar \
    --runs-root outputs/field \
    --exp-filter A-Base-03 --exp-filter A-Main-01

echo ""
echo "完成: $(date)"
echo "=============================================="
