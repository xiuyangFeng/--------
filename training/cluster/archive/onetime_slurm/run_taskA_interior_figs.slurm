#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --job-name=taskA_interior_figs
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=08:00:00
# 与训练脚本一致：GPU 分区；若需固定节点可取消下行注释
# #SBATCH -w node01

# ============================================================================
# 任务 A：2026-03-26 interior-only 口径 — 在集群上顺序重跑出图产物
# 对应推进记录「主指标口径统一为 interior-only」中的下一步。
#
# 包含：每 run 的 A3/A4/误差分析 → A1 主表 → A3/A4 多模型汇总 → A6 消融
#      → A7 效率基准（需 CUDA，故本作业申请 GPU）
#
# 用法（在 GNN 仓库根目录）:
#   cd /path/to/GNN && sbatch training/cluster/run_taskA_interior_figs.slurm
#   TRAINING_ENV=GNN sbatch training/cluster/run_taskA_interior_figs.slurm
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

mkdir -p "$SCRIPT_DIR/logs"
mkdir -p logs

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export NUMEXPR_MAX_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

echo "=============================================="
echo "Task A interior-only 出图批处理"
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
    echo ">>> plot_taskA_scatter (interior default) --manifest $m"
    "$PYTHON_BIN" -m training.scripts.plot_taskA_scatter --manifest "$m"
    echo ">>> plot_taskA_per_case_boxplot --manifest $m"
    "$PYTHON_BIN" -m training.scripts.plot_taskA_per_case_boxplot --manifest "$m"
    echo ">>> plot_error_analysis --manifest $m"
    "$PYTHON_BIN" -m training.scripts.plot_error_analysis --manifest "$m"
done

echo ""
echo ">>> plot_taskA_main_table (--region interior 默认)"
"$PYTHON_BIN" -m training.scripts.plot_taskA_main_table --runs-root outputs/field

echo ""
echo ">>> plot_taskA_multimodel_scatter vel_mag + p (seed 1, interior)"
"$PYTHON_BIN" -m training.scripts.plot_taskA_multimodel_scatter \
    --runs-root outputs/field --seed-filter 1 --variable vel_mag
"$PYTHON_BIN" -m training.scripts.plot_taskA_multimodel_scatter \
    --runs-root outputs/field --seed-filter 1 --variable p

echo ""
echo ">>> plot_taskA_multimodel_per_case_boxplot (interior CSV)"
"$PYTHON_BIN" -m training.scripts.plot_taskA_multimodel_per_case_boxplot \
    --runs-root outputs/field

echo ""
echo ">>> plot_taskA_ablation_summary (interior / regional_eval 优先)"
"$PYTHON_BIN" -m training.scripts.plot_taskA_ablation_summary \
    --runs-root outputs/field

echo ""
echo ">>> run_efficiency_benchmark (Pareto 等使用 interior.rmse_vel_mag)"
"$PYTHON_BIN" -m training.scripts.run_efficiency_benchmark \
    --output-dir outputs/field/plots/efficiency

echo ""
echo "完成: $(date)"
echo "=============================================="
