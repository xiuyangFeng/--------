#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=wssmin_postview
#SBATCH --output=training_wss_min/cluster/logs/%x_%j.out
#SBATCH --error=training_wss_min/cluster/logs/%x_%j.err
#SBATCH --time=00:30:00

set -euo pipefail

if [ -f "$HOME/.bashrc" ]; then source "$HOME/.bashrc"; fi
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
elif [ -f "/public/newapps/anaconda3/etc/profile.d/conda.sh" ]; then
    source "/public/newapps/anaconda3/etc/profile.d/conda.sh"
else
    echo "错误: 找不到 conda"; exit 1
fi
conda activate GNN

PROJECT_DIR="/public/newhome/cy/Digital_twin/GNN"
cd "$PROJECT_DIR"
PY="${CONDA_PREFIX}/bin/python"

echo "=============================================="
echo "wss_min postview export"
echo "开始: $(date)  节点: $(hostname)  GPU: ${CUDA_VISIBLE_DEVICES:-NA}"
echo "=============================================="

"${PY}" -m training_wss_min.export_wss_postview \
  --run-dir training_wss_min/runs/r5_a0e_b1_ctrl_s1234 \
  --cases slow/WU_FENG_YAN,fast/RAN_QING_BO \
  --output-dir "docs/03-汇报材料/figures/WSS最小路线_20260712/postview_a0e_ctrl" \
  --device cuda

echo "完成: $(date)"
