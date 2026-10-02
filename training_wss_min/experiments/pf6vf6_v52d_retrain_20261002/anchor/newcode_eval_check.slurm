#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=pf6vf6_anchor_diag
#SBATCH --output=/public/newhome/cy/Digital_twin/GNN/training_wss_min/experiments/pf6vf6_v52d_retrain_20261002/anchor/newcode_eval_check_%j.out
#SBATCH --time=00:30:00
# diagnosis of preflight 16857: new-code evaluation of the PF6 anchor (twice) to list every field that differs from the old-code metrics
cd /public/newhome/cy/Digital_twin/GNN_pf6vf6_v52d_frozen_20261002
X=/public/newhome/cy/Digital_twin/GNN/training_wss_min/experiments/pf6vf6_v52d_retrain_20261002/anchor/PF6_s1234_v50_on_v52d
for r in 1 2; do /public/newhome/cy/.conda/envs/GNN/bin/python -u -m training_wss_min.evaluate --run-dir $X --partitions test --allow-test --checkpoint best --no-plots --output-dir $X/diag_newcode_${SLURM_JOB_ID}_r$r | tail -2; done
