#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=wssmin_eval
#SBATCH --output=training_wss_min/cluster/logs/%x_%j.out
#SBATCH --error=training_wss_min/cluster/logs/%x_%j.err
#SBATCH --time=01:00:00

# 用法: sbatch run_eval_only.slurm <config_json_path> [--eval-partitions val]
# 仅对已训练好的 run（ckpt_best.pt 已存在）重跑完整点云评估，不重新训练。

set -euo pipefail

CONFIG="${1:?需要 config json 路径}"
EVAL_PARTS="${2:-val}"

if [ -f "$HOME/.bashrc" ]; then source "$HOME/.bashrc"; fi
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
elif [ -f "/public/newapps/anaconda3/etc/profile.d/conda.sh" ]; then
    source "/public/newapps/anaconda3/etc/profile.d/conda.sh"
else
    echo "错误: 找不到 conda 初始化脚本"; exit 1
fi
conda activate GNN

PROJECT_DIR="/public/newhome/cy/Digital_twin/GNN"
cd "$PROJECT_DIR"
PY="${CONDA_PREFIX}/bin/python"

echo "=============================================="
echo "wss_min eval-only  config=${CONFIG}"
echo "eval_partitions=${EVAL_PARTS}"
echo "开始: $(date)   节点: $(hostname)   GPU: ${CUDA_VISIBLE_DEVICES:-NA}"
echo "=============================================="

"${PY}" -m training_wss_min.evaluate --config "${CONFIG}" --partitions "${EVAL_PARTS}"

echo "完成: $(date)"
