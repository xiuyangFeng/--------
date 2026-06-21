#!/bin/bash
#SBATCH --partition=CPU
#SBATCH -w node03
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=v3_f0_oracle_v2
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=2:00:00

# F0 修正版 oracle（纯 CPU，无 GPU）：
#   - gt_wall_diff_v2  : 反归一化 + 切向投影 + 沿法向采样（修旧 −0.34 假信号）
#   - pressure_gradient: 壁面 ∂p/∂s ↔ WSS（思路 2 压力耦合诊断）
#   - pod_2d           : (Abscissa × θ) 2D 壁面 WSS 场 POD 重建上限
# 用法: sbatch training/cluster/run_v3_f0_oracle_v2.slurm
# 资源: CPU 分区 node03，不占 master GPU，对 5253(GPU) 无影响

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

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

python -m training.scripts.run_v3_f0_decision \
    --oracle-max-cases 12 \
    --oracle-max-graphs 2 \
    --pod-max-graphs 3 \
    --pod2d-sectors 1 4 8 \
    --output-name v3_f0_oracle_v2.json

echo ">>> F0 oracle v2 完成 $(date)"
