#!/bin/bash
#SBATCH --job-name=v5_1post
#SBATCH --partition=CPU
#SBATCH --time=0
#SBATCH --mem=200G
#SBATCH --cpus-per-task=24
#SBATCH --output=outputs/wss_pinn/slurm/v5_1_post_%j.out
#SBATCH --error=outputs/wss_pinn/slurm/v5_1_post_%j.err
# 刷新之后：噪声 sidecar（170 例 × σ 0.1/0.2/0.3）→ 生成 v5.1 波 1 的 14 臂配置 + matrix.json。
set -euo pipefail
cd /public/newhome/cy/Digital_twin/GNN
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
PY=/public/newhome/cy/.conda/envs/GNN/bin/python
V=data_wss_v5/views_v5_1
echo "== 1 noise sidecars"; "$PY" -m training_wss_min.tools.build_noise_sidecars --view-root "$V/wss_min_view_v1" --out "$V/wss_min_noise_v1" --sigmas 0.1 0.2 0.3 --workers 12
echo "== 2 prepare v5.1 wave 1"; "$PY" -m training_wss_min.tools.prepare_wss_v51_wave1
echo "== 3 sanity"; ls training_wss_min/configs/wss_v51_wave1_20260916 | wc -l; "$PY" -c "import json; m=json.load(open('training_wss_min/configs/wss_v51_wave1_20260916/matrix.json')); print(m['expected_training_runs'], 'arms', m['data_v51'])"
date
