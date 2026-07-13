#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=wssmin_r5_c1_micro
#SBATCH --output=training_wss_min/runs/_round5/branch_experiments/radius_norm/micro/slurm_%j.out
#SBATCH --error=training_wss_min/runs/_round5/branch_experiments/radius_norm/micro/slurm_%j.err
#SBATCH --time=01:00:00
set -euo pipefail
cd /public/newhome/cy/Digital_twin/GNN
/public/newhome/cy/.conda/envs/GNN/bin/python3.10 -m training_wss_min.experiments.brep_c1_radius_norm --mode micro \
  --protocol training_wss_min/configs/round5/brep_c1_radius_norm_protocol.json
