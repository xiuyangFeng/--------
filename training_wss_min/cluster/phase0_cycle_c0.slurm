#!/bin/bash
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --job-name=cycle_c0
#SBATCH --output=training_wss_min/cluster/logs/%x_%j.out
#SBATCH --error=training_wss_min/cluster/logs/%x_%j.err
# 周期积分量矩阵 阶段 0 / C0：从 wall_wss_vec 构建 wss_min_cycle_v1（TAWSS/OSI/RRT/rev_frac）+ 审计 + 训练折统计量
set -euo pipefail
cd /public/newhome/cy/Digital_twin/GNN
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/public/newhome/cy/.conda/envs/GNN/bin/python
date -Is; hostname
$PY -u -m wss_v5.views.wall_cycle_v1 --workers 8
echo "===== done $(date -Is) ====="
