#!/bin/bash
#SBATCH --partition=CPU
#SBATCH -w node03
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=wssmin_cov_dev1
#SBATCH --output=training_wss_min/cluster/logs/%x_%j.out
#SBATCH --error=training_wss_min/cluster/logs/%x_%j.err
#SBATCH --time=02:00:00

# 第四轮点数曲线前：v2_dev1 + fold stats 覆盖审计

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

SPLIT="training/splits/split_AG_wss_min_v2_dev1.json"
STATS="data_wss_min/fold_stats/wss_stats_v2_dev1.json"
OUT="training_wss_min/runs/_audit/coverage_audit_v2_dev1.csv"

echo "coverage audit split=${SPLIT} stats=${STATS}"
echo "开始: $(date)"
"${PY}" -m training_wss_min.tools.coverage_audit \
  --split "${SPLIT}" \
  --stats "${STATS}" \
  --k 1000 1500 2000 3000 4000 6000 \
  --pool-size 8 \
  --epochs 40 \
  --seed 1234 \
  --out "${OUT}"
echo "完成: $(date)"
