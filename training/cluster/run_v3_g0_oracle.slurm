#!/bin/bash
#SBATCH --partition=CPU
#SBATCH -w node03
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=v3p_g0_oracle
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=3:00:00

# 路径 G · S0 前置 oracle（V3P · 纯 CPU · 0 重训）：
#   - G0-a 坐标系负担量化（global vs local 探针 R² + 旋转一致性）
#   - QA  标签/壁面/量纲质检 → denylist 候选
#   - G0-c 压力梯度→WSS 相关 + GBDT 上限
#   - G0-d 解析基线残差方差下降
#   - G0-e noise floor 复核
# 用法: sbatch training/cluster/run_v3_g0_oracle.slurm
# 资源: CPU 分区 node03，不占 GPU

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

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"
mkdir -p "$PROJECT_DIR/logs"

# HistGBDT / OpenMP：单线程避免 node03 上卡死（S0 交接约定）
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

python -m training.scripts.run_v3_g0_oracle \
    --max-graphs-per-case 3 \
    --max-wall-per-graph 800 \
    --qa-max-graphs 2 \
    --oracle-max-cases 12 \
    --oracle-max-graphs 1 \
    --n-rot 8 \
    --output-name v3p_g0_oracle_20260606_v2.json

echo ">>> V3P G0 oracle 完成 $(date)"
