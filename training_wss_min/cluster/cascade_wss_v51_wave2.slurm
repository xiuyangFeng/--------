#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --job-name=wss_v51_cascade
#SBATCH --output=training_wss_min/cluster/logs/%x_%j.out
#SBATCH --error=training_wss_min/cluster/logs/%x_%j.err
#SBATCH --time=06:00:00
# 波 2a 三折模型 → 阶段一 sidecar（train136 折外预测 + test34 折模型轮转）→ 波 2b（T3n）配置
set -euo pipefail
cd /public/newhome/cy/Digital_twin/GNN
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/public/newhome/cy/.conda/envs/GNN/bin/python
V=data_wss_v5/views_v5_1
echo "== 1 cascade sidecar"; "$PY" -u -m training_wss_min.tools.make_cascade_sidecar --seed 1234 --view-root "$V/wss_min_view_v1" --cv-name cv3_v51 \
  --split-name split_V5_train136_test34.json --stats-name wss_global_stats_train136.json --wave wss_v51_wave2a_20260916 \
  --fold-run-pattern 'X5D_v51_f{k}_s{seed}' --out "$V/wss_min_cascade_v1"
echo "== 2 prepare wave 2b"; "$PY" -m training_wss_min.tools.prepare_wss_v51_wave2b
echo "== done"; date
