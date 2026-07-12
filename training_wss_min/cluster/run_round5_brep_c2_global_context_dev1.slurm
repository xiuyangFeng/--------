#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=wssmin_r5_c2_dev1
#SBATCH --output=training_wss_min/runs/_round5/branch_experiments/global_context/dev1/slurm_%j.out
#SBATCH --error=training_wss_min/runs/_round5/branch_experiments/global_context/dev1/slurm_%j.err
#SBATCH --time=01:00:00
set -euo pipefail
cd /public/newhome/cy/Digital_twin/GNN
/public/newhome/cy/.conda/envs/GNN/bin/python3.10 -m training_wss_min.brep_c2_global_context --mode dev \
  --protocol training_wss_min/configs/round5/brep_c2_global_context_protocol.json
