#!/bin/bash
#SBATCH --partition=CPU
#SBATCH --nodelist=node03
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=06:00:00
#SBATCH --job-name=voltime_prep
#SBATCH --output=training_wss_min/cluster/logs/%x_%j.out
#SBATCH --error=training_wss_min/cluster/logs/%x_%j.err
# 预注册 volume_time_20260919 矩阵：6 个峰值帧折底座（PF6/VF6 × 3 折）+ 18 个时间臂，
# 顺带在体场点集（wall∪interior）上算逐折训练侧特征统计量（要加载 90 例体场，内存吃紧所以给 160G）
set -euo pipefail
cd /public/newhome/cy/Digital_twin/GNN
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
date -Is; hostname
/public/newhome/cy/.conda/envs/GNN/bin/python -u -m training_wss_min.tools.prepare_volume_time
date -Is
