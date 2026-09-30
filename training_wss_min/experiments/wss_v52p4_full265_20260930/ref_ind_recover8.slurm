#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --job-name=full265_refind
#SBATCH --output=/public/newhome/cy/Digital_twin/GNN/training_wss_min/experiments/wss_v52p4_full265_20260930/logs/ref_ind_recover8_%j.out
# 2026-09-30: paired workbook reference for the full265 rows — the X5Dcap_asym2 IND (train170) runs of the same seeds
# evaluated on recover8 exactly like the queue evaluated the full265 runs (frozen code copy, GPU, best checkpoint,
# the full265 view root and its test partition = recover8). Read-only on the IND runs; outputs under ref_ind_recover8/.
set -euo pipefail
G=/public/newhome/cy/Digital_twin/GNN; X=$G/training_wss_min/experiments/wss_v52p4_full265_20260930
V=$G/data_wss_v5/views_v5_2p4_full265_20260930
cd /public/newhome/cy/Digital_twin/GNN_v52p_frozen_20260926
export PYTHONPATH=$PWD OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
date -Is; hostname; nvidia-smi --query-gpu=index,name,memory.used --format=csv
sha1sum training_wss_min/evaluate.py training_wss_min/dataset.py "$V/wss_min_view_v1/split_v52p4_full265_train265_test8.json"
for s in 1234 7 2025; do
  /public/newhome/cy/.conda/envs/GNN/bin/python -u -m training_wss_min.evaluate \
    --run-dir "$G/training_wss_min/runs/wss_v52_phys_20260926/X5Dcap_asym2_s$s" \
    --partitions test --allow-test --checkpoint best --no-plots \
    --split-path "$V/wss_min_view_v1/split_v52p4_full265_train265_test8.json" --data-root "$V/wss_min_view_v1" \
    --point-features-root "$V/wss_min_geom_v2" "$V/wss_min_flowref_v1" \
    --output-dir "$X/ref_ind_recover8/X5Dcap_asym2_s$s"
done
date -Is
