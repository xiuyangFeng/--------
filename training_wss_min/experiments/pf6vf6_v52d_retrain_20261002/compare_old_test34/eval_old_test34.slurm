#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --job-name=pf6vf6_old_test34
#SBATCH --output=/public/newhome/cy/Digital_twin/GNN/training_wss_min/experiments/pf6vf6_v52d_retrain_20261002/compare_old_test34/eval_%j.out
#SBATCH --time=02:00:00
# old-vs-new comparison (2026-10-02): deployed v5.0 PF6/VF6 (E5 run dirs, repointed to the v5.2d volume root) on the 34 v5.0 test units, v5.2d labels
X=/public/newhome/cy/Digital_twin/GNN/training_wss_min/experiments/pf6vf6_v52d_retrain_20261002
cd /public/newhome/cy/Digital_twin/GNN_pf6vf6_v52d_frozen_20261002
for m in PF6_s1234 PF6_s7 PF6_s2025 VF6_s1234 VF6_s7 VF6_s2025; do
  o=$X/compare_old_test34/$m; [ -e $o ] && { echo "skip $o"; continue; }
  /public/newhome/cy/.conda/envs/GNN/bin/python -u -m training_wss_min.evaluate --run-dir $X/e5_deployed_v50/$m --partitions test --allow-test \
    --checkpoint best --no-plots --save-predictions --split-path $X/compare_old_test34/split_test34_on_v52d.json --output-dir $o | tail -2
done
