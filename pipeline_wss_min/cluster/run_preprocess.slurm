#!/bin/bash
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --job-name=wss_min_pre
#SBATCH --output=pipeline_wss_min/cluster/logs/%x_%j.out
#SBATCH --error=pipeline_wss_min/cluster/logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH -w node03

set -euo pipefail

if [ -f "$HOME/.bashrc" ]; then
    source "$HOME/.bashrc"
fi
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
elif [ -f "/public/newapps/anaconda3/etc/profile.d/conda.sh" ]; then
    source "/public/newapps/anaconda3/etc/profile.d/conda.sh"
else
    echo "错误: 找不到 conda 初始化脚本"
    exit 1
fi
conda activate GNN

PROJECT_DIR="/public/newhome/cy/Digital_twin/GNN"
cd "$PROJECT_DIR"

PY="${CONDA_PREFIX}/bin/python"
STAGE="${1:-preprocess}"

echo "=============================================="
echo "pipeline_wss_min stage=${STAGE}"
echo "开始: $(date)"
echo "节点: $(hostname)"
echo "=============================================="

"${PY}" -m pipeline_wss_min.run --stage "${STAGE}"

echo "完成: $(date)"
