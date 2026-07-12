#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --job-name=wss_r5_a0d_four
#SBATCH --output=training_wss_min/runs/_round5/a0d_fit_chain/simple_overfit/four_shared/slurm_%j.out
#SBATCH --error=training_wss_min/runs/_round5/a0d_fit_chain/simple_overfit/four_shared/slurm_%j.err
#SBATCH --time=00:30:00

set -euo pipefail
cd /public/newhome/cy/Digital_twin/GNN
/public/newhome/cy/.conda/envs/GNN/bin/python3.10 -m training_wss_min.a0d_simple_overfit \
  --mode four \
  --protocol training_wss_min/configs/round5/a0d_simple_overfit_protocol.json
