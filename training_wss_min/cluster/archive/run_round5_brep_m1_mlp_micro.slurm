#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=wssmin_r5_m1_micro
#SBATCH --output=training_wss_min/runs/_round5/branch_experiments/mlp/micro/slurm_%j.out
#SBATCH --error=training_wss_min/runs/_round5/branch_experiments/mlp/micro/slurm_%j.err
#SBATCH --time=01:00:00
set -euo pipefail
cd /public/newhome/cy/Digital_twin/GNN
/public/newhome/cy/.conda/envs/GNN/bin/python3.10 -m training_wss_min.experiments.brep_mlp --mode micro \
  --protocol training_wss_min/configs/round5/brep_m1_mlp_protocol.json
